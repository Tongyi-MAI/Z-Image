"""Z-Image PyTorch Native Inference."""

import os
import time
import warnings
from pathlib import Path

import torch
import numpy as np

warnings.filterwarnings("ignore")
from utils import (
    AttentionBackend,
    ensure_model_weights,
    load_from_local_dir,
    set_attention_backend,
)
from utils.config_loader import ConfigConflictError, RuntimeConfig, load_runtime_config
from zimage import generate


def _banner(msg: str) -> None:
    print(f"\n========== {msg} ==========")


def _print_config_summary(
    cfg: RuntimeConfig,
    *,
    attn_backend: str,
    seed: int,
    prompt_src: str,
    compile_flag: bool,
):
    _banner("Runtime configuration")
    print(f"prompt_source : {prompt_src}")
    print(f"output_path   : {cfg.output_path}")
    print(f"height x width: {cfg.height} x {cfg.width}")
    print(f"steps         : {cfg.num_inference_steps}")
    print(f"guidance      : {cfg.guidance_scale}")
    print(f"attention     : {attn_backend}")
    print(f"compile       : {compile_flag}")
    print(f"seed          : {seed}")


def _resolve_compile(flag: bool) -> bool:
    if not flag:
        return False
    try:
        import triton  # type: ignore

        _ = triton
        return True
    except Exception:
        _banner("Triton not available; disabling torch.compile")
        return False


def main():
    cfg: RuntimeConfig = load_runtime_config()
    model_path = ensure_model_weights(
        "ckpts/Z-Image-Turbo", verify=False
    )  # True to verify with md5
    dtype = torch.bfloat16
    compile = _resolve_compile(bool(cfg.compile))
    output_path: Path = cfg.output_path
    height = cfg.height
    width = cfg.width
    num_inference_steps = cfg.num_inference_steps
    guidance_scale = cfg.guidance_scale
    seed = cfg.seed if cfg.seed is not None else np.random.randint(0, 10000)

    # Pick attention backend (env wins via config loader; otherwise flash on CUDA, math elsewhere)
    attn_backend = cfg.attention_backend
    if attn_backend is None:
        attn_backend = "_native_flash" if torch.cuda.is_available() else "_native_math"

    # Resolve prompt
    if cfg.prompt_text:
        prompt = cfg.prompt_text
        prompt_src = "single_prompt (inline)"
    else:
        prompt_path = cfg.prompt_file
        if prompt_path is None:
            raise ConfigConflictError(
                "No prompt source provided; set 'single_prompt' or 'prompt_file'"
            )
        with prompt_path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError(f"Prompt file is empty: {prompt_path}")
        prompt = lines[0]
        prompt_src = f"prompt_file ({prompt_path})"

    _print_config_summary(
        cfg,
        attn_backend=attn_backend,
        seed=seed,
        prompt_src=prompt_src,
        compile_flag=compile,
    )

    # Device selection priority: cuda -> tpu -> mps -> cpu
    if torch.cuda.is_available():
        device = "cuda"
        _banner("Chosen device: cuda")
    else:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
            _banner("Chosen device: tpu")
        except (ImportError, RuntimeError):
            if torch.backends.mps.is_available():
                device = "mps"
                _banner("Chosen device: mps")
            else:
                device = "cpu"
                _banner("Chosen device: cpu")
    # Load models
    components = load_from_local_dir(
        model_path, device=device, dtype=dtype, compile=compile
    )
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    _banner(f"Attention backend: {attn_backend}")

    # Gen an image; if a kernel is unavailable, fall back to math backend automatically
    try:
        start_time = time.time()
        images = generate(
            prompt=prompt,
            **components,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device).manual_seed(seed),
        )
        end_time = time.time()
    except RuntimeError as e:
        if "No available kernel" in str(e) or "Triton" in str(e):
            fallback_backend = "_native_math"
            print(
                f"Attention backend {attn_backend} failed ({e}); retrying with {fallback_backend}."
            )
            set_attention_backend(fallback_backend)
            start_time = time.time()
            images = generate(
                prompt=prompt,
                **components,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device).manual_seed(seed),
            )
            end_time = time.time()
        else:
            raise

    _banner(f"Image saved to {output_path} | Time: {end_time - start_time:.2f} s")
    images[0].save(output_path)

    ### !! For best speed performance, recommend to use `_flash_3` backend and set `compile=True`
    ### This would give you sub-second generation speed on Hopper GPU (H100/H200/H800) after warm-up


if __name__ == "__main__":
    main()
