"""Attention backend utilities for Z-Image."""

# Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_dispatch.py
from enum import Enum
import functools
import inspect
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from .import_utils import (
    is_flash_attn_3_available,
    is_flash_attn_available,
    is_torch_version,
)

_CAN_USE_FLASH_ATTN_2 = is_flash_attn_available()
_CAN_USE_FLASH_ATTN_3 = is_flash_attn_3_available()
_TORCH_VERSION_CHECK = is_torch_version(
    ">=", "2.5.0"
)  # have enable_gqa func call in SPDA

if not _TORCH_VERSION_CHECK:
    raise RuntimeError("PyTorch version must be >= 2.5.0 to use this backend.")
else:
    print("PyTorch version is >= 2.5.0, check pass.")

if _CAN_USE_FLASH_ATTN_2:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
else:
    flash_attn_func = None
    flash_attn_varlen_func = None

if _CAN_USE_FLASH_ATTN_3:
    from flash_attn_interface import (
        flash_attn_func as flash_attn_3_func,
        flash_attn_varlen_func as flash_attn_3_varlen_func,
    )

    _flash_attn_3_sig = inspect.signature(flash_attn_3_func)
    _FLASH_ATTN_3_SUPPORTS_RETURN_PROBS = (
        "return_attn_probs" in _flash_attn_3_sig.parameters
    )
else:
    flash_attn_3_func = None
    flash_attn_3_varlen_func = None
    _FLASH_ATTN_3_SUPPORTS_RETURN_PROBS = False


class AttentionBackend(str, Enum):
    """Supported attention backends."""

    # Flash Attention
    FLASH = "flash"
    FLASH_VARLEN = "flash_varlen"
    FLASH_3 = "_flash_3"
    FLASH_VARLEN_3 = "_flash_varlen_3"
    # PyTorch Native Backends
    NATIVE = "native"
    NATIVE_FLASH = "_native_flash"
    NATIVE_MATH = "_native_math"

    @classmethod
    def print_available_backends(cls):
        available_backends = [backend.value for backend in cls.__members__.values()]
        print(f"Available attention backends list: {available_backends}")


# Registry for attention implementations
_ATTENTION_BACKENDS: Dict[str, Callable] = {}
_ATTENTION_CONSTRAINTS: Dict[str, List[Callable]] = {}


def register_backend(name: str, constraints: Optional[List[Callable]] = None):
    def decorator(func):
        _ATTENTION_BACKENDS[name] = func
        _ATTENTION_CONSTRAINTS[name] = constraints or []
        return func

    return decorator


# --- Checks ---
def _check_device_cuda(query: torch.Tensor, **kwargs) -> None:
    if query.device.type != "cuda":
        raise ValueError("Query must be on a CUDA device.")


def _check_qkv_dtype_bf16_or_fp16(query: torch.Tensor, **kwargs) -> None:
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Query must be either bfloat16 or float16.")


def _process_mask(attn_mask: Optional[torch.Tensor], dtype: torch.dtype):
    if attn_mask is None:
        return None

    if attn_mask.ndim == 2:
        attn_mask = attn_mask[:, None, None, :]

    # Convert bool mask to float additive mask
    if attn_mask.dtype == torch.bool:
        # NOTE: We skip checking for all-True mask (torch.all) to avoid graph breaks in torch.compile
        new_mask = torch.zeros_like(attn_mask, dtype=dtype)
        new_mask.masked_fill_(~attn_mask, float("-inf"))
        return new_mask

    return attn_mask


def _normalize_attn_mask(
    attn_mask: torch.Tensor, batch_size: int, seq_len_k: int
) -> torch.Tensor:
    """Normalize an attention mask to shape [batch_size, seq_len_k] (bool)."""
    if attn_mask.dtype != torch.bool:
        # Try to convert float mask back to bool if possible, or assume it's float mask
        # For varlen flash attn, we strictly need bool mask indicating valid tokens
        if torch.is_floating_point(attn_mask):
            return attn_mask > -1  # Assuming -inf is masked
        # raise ValueError(f"Attention mask must be of type bool, got {attn_mask.dtype}.")

    if attn_mask.ndim == 1:
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, seq_len_k)
    elif attn_mask.ndim == 2:
        if attn_mask.size(0) not in [1, batch_size]:
            attn_mask = attn_mask.expand(batch_size, seq_len_k)
    elif attn_mask.ndim == 3:
        attn_mask = attn_mask.any(dim=1)
        attn_mask = attn_mask.expand(batch_size, seq_len_k)
    elif attn_mask.ndim == 4:
        attn_mask = attn_mask.expand(batch_size, -1, -1, seq_len_k)
        attn_mask = attn_mask.any(dim=(1, 2))

    if attn_mask.shape != (batch_size, seq_len_k):
        # Fallback reshape
        return attn_mask.view(batch_size, seq_len_k)

    return attn_mask


@functools.lru_cache(maxsize=128)
def _prepare_for_flash_attn_varlen_without_mask(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    device: Optional[torch.device] = None,
):
    # Optimized to avoid Inductor "pointless_cumsum_replacement" crash and remove graph breaks
    seqlens_q = torch.full((batch_size,), seq_len_q, dtype=torch.int32, device=device)
    seqlens_k = torch.full((batch_size,), seq_len_kv, dtype=torch.int32, device=device)

    cu_seqlens_q = (
        torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seq_len_q
    )
    cu_seqlens_k = (
        torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seq_len_kv
    )

    return (seqlens_q, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (seq_len_q, seq_len_kv)


def _prepare_for_flash_attn_varlen_with_mask(
    batch_size: int,
    seq_len_q: int,
    attn_mask: torch.Tensor,
    device: Optional[torch.device] = None,
):
    seqlens_q = torch.full((batch_size,), seq_len_q, dtype=torch.int32, device=device)
    seqlens_k = attn_mask.sum(dim=1, dtype=torch.int32)
    # Use arange for Q to avoid Inductor crash
    cu_seqlens_q = (
        torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seq_len_q
    )

    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_k[1:] = torch.cumsum(seqlens_k, dim=0)

    max_seqlen_q = seq_len_q
    max_seqlen_k = attn_mask.shape[
        1
    ]  # not max().item(), static shape to avoid graph break

    return (
        (seqlens_q, seqlens_k),
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_q, max_seqlen_k),
    )


def _prepare_for_flash_attn_varlen(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    attn_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> None:
    if attn_mask is None:
        return _prepare_for_flash_attn_varlen_without_mask(
            batch_size, seq_len_q, seq_len_kv, device
        )
    return _prepare_for_flash_attn_varlen_with_mask(
        batch_size, seq_len_q, attn_mask, device
    )


@register_backend(
    AttentionBackend.FLASH,
    constraints=[_check_device_cuda, _check_qkv_dtype_bf16_or_fp16],
)
def _flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if not _CAN_USE_FLASH_ATTN_2:
        raise RuntimeError(
            f"Flash Attention backend '{AttentionBackend.FLASH}' is not usable because of missing package."
        )

    out = flash_attn_func(
        q=query,
        k=key,
        v=value,
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=is_causal,
    )
    return out


@register_backend(
    AttentionBackend.FLASH_VARLEN,
    constraints=[_check_device_cuda, _check_qkv_dtype_bf16_or_fp16],
)
def _flash_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if not _CAN_USE_FLASH_ATTN_2:
        raise RuntimeError(
            f"Backend '{AttentionBackend.FLASH_VARLEN}' requires flash-attn."
        )

    batch_size, seq_len_q, _, _ = query.shape
    _, seq_len_kv, _, _ = key.shape

    if attn_mask is not None:
        attn_mask = _normalize_attn_mask(attn_mask, batch_size, seq_len_kv)

    (_, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = (
        _prepare_for_flash_attn_varlen(
            batch_size, seq_len_q, seq_len_kv, attn_mask=attn_mask, device=query.device
        )
    )

    query_packed = query.flatten(0, 1)

    if attn_mask is not None:
        key_valid = []
        value_valid = []
        for b in range(batch_size):
            valid_len = seqlens_k[b]
            key_valid.append(key[b, :valid_len])
            value_valid.append(value[b, :valid_len])
        key_packed = torch.cat(key_valid, dim=0)
        value_packed = torch.cat(value_valid, dim=0)
    else:
        key_packed = key.flatten(0, 1)
        value_packed = value.flatten(0, 1)

    out = flash_attn_varlen_func(
        q=query_packed,
        k=key_packed,
        v=value_packed,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=is_causal,
    )
    out = out.unflatten(0, (batch_size, -1))
    return out


@register_backend(
    AttentionBackend.FLASH_3,
    constraints=[_check_device_cuda, _check_qkv_dtype_bf16_or_fp16],
)
def _flash_attention_3(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,  # Unused in simple FA3 func
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if not _CAN_USE_FLASH_ATTN_3:
        raise RuntimeError(
            f"Backend '{AttentionBackend.FLASH_3}' requires Flash Attention 3 beta."
        )

    kwargs = {
        "q": query,
        "k": key,
        "v": value,
        "softmax_scale": scale,
        "causal": is_causal,
    }

    if _FLASH_ATTN_3_SUPPORTS_RETURN_PROBS:
        kwargs["return_attn_probs"] = False

    out = flash_attn_3_func(**kwargs)

    if isinstance(out, tuple):
        out = out[0]

    return out


@register_backend(
    AttentionBackend.FLASH_VARLEN_3,
    constraints=[_check_device_cuda, _check_qkv_dtype_bf16_or_fp16],
)
def _flash_varlen_attention_3(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if not _CAN_USE_FLASH_ATTN_3:
        raise RuntimeError(
            f"Backend '{AttentionBackend.FLASH_VARLEN_3}' requires Flash Attention 3 beta."
        )

    batch_size, seq_len_q, _, _ = query.shape
    _, seq_len_kv, _, _ = key.shape

    if attn_mask is not None:
        attn_mask = _normalize_attn_mask(attn_mask, batch_size, seq_len_kv)

    (_, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = (
        _prepare_for_flash_attn_varlen(
            batch_size, seq_len_q, seq_len_kv, attn_mask=attn_mask, device=query.device
        )
    )

    query_packed = query.flatten(0, 1)

    if attn_mask is not None:
        key_valid = []
        value_valid = []
        for b in range(batch_size):
            valid_len = seqlens_k[b]
            key_valid.append(key[b, :valid_len])
            value_valid.append(value[b, :valid_len])
        key_packed = torch.cat(key_valid, dim=0)
        value_packed = torch.cat(value_valid, dim=0)
    else:
        key_packed = key.flatten(0, 1)
        value_packed = value.flatten(0, 1)

    kwargs = {
        "q": query_packed,
        "k": key_packed,
        "v": value_packed,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "softmax_scale": scale,
        "causal": is_causal,
    }

    supports_return_probs = (
        "return_attn_probs" in inspect.signature(flash_attn_3_varlen_func).parameters
    )

    if supports_return_probs:
        kwargs["return_attn_probs"] = False

    out = flash_attn_3_varlen_func(**kwargs)

    if isinstance(out, tuple):
        out = out[0]

    out = out.unflatten(0, (batch_size, -1))
    return out


def _native_attention_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    backend_kernel=None,
) -> torch.Tensor:

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    attn_mask = _process_mask(attn_mask, query.dtype)

    if backend_kernel is not None:
        with torch.nn.attention.sdpa_kernel(backend_kernel):
            out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
    else:
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    return out.transpose(1, 2).contiguous()


@register_backend(AttentionBackend.NATIVE_FLASH)
def _native_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    return _native_attention_wrapper(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        backend_kernel=torch.nn.attention.SDPBackend.FLASH_ATTENTION,
    )


@register_backend(AttentionBackend.NATIVE_MATH)
def _math_attention(*args, **kwargs):
    return _native_attention_wrapper(
        *args, **kwargs, backend_kernel=torch.nn.attention.SDPBackend.MATH
    )


@register_backend(AttentionBackend.NATIVE)
def _native_attention(*args, **kwargs):
    return _native_attention_wrapper(*args, **kwargs, backend_kernel=None)


def dispatch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    backend: Union[str, AttentionBackend, None] = None,
) -> torch.Tensor:

    if isinstance(backend, AttentionBackend):
        backend = backend.value
    elif backend is None:
        backend = AttentionBackend.NATIVE
    else:
        backend = str(backend)

    # Explicit dispatch to avoid dynamo guard issues on global dict
    if backend == AttentionBackend.FLASH:
        return _flash_attention(
            query, key, value, attn_mask, dropout_p, is_causal, scale
        )
    elif backend == AttentionBackend.FLASH_VARLEN:
        return _flash_varlen_attention(
            query, key, value, attn_mask, dropout_p, is_causal, scale
        )
    elif backend == AttentionBackend.FLASH_3:
        return _flash_attention_3(
            query, key, value, attn_mask, dropout_p, is_causal, scale
        )
    elif backend == AttentionBackend.FLASH_VARLEN_3:
        return _flash_varlen_attention_3(
            query, key, value, attn_mask, dropout_p, is_causal, scale
        )
    elif backend == AttentionBackend.NATIVE_FLASH:
        return _native_flash_attention(
            query, key, value, attn_mask, dropout_p, is_causal, scale
        )
    elif backend == AttentionBackend.NATIVE_MATH:
        return _math_attention(
            query, key, value, attn_mask, dropout_p, is_causal, scale
        )
    else:
        return _native_attention(
            query, key, value, attn_mask, dropout_p, is_causal, scale
        )


def set_attention_backend(backend: Union[str, AttentionBackend, None]):
    try:
        from zimage.transformer import ZImageAttention

        if backend is not None:
            backend = str(backend)
        ZImageAttention._attention_backend = backend
    except ImportError:
        pass
