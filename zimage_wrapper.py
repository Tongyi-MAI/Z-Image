"""Wrapper around Z-Image Diffusers pipelines for serverless inference."""

import gc
import os
from typing import Dict, Optional, Tuple

import torch
from diffusers import ZImageImg2ImgPipeline, ZImageInpaintPipeline, ZImagePipeline
from PIL import Image

from src.utils import ensure_model_weights


PipelineCacheKey = Tuple[str, str, str, str]

_pipeline_cache: Dict[PipelineCacheKey, object] = {}
_device_cache: Optional[str] = None


def _select_device() -> str:
    """Select the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _select_dtype(device: str) -> torch.dtype:
    """Select an inference dtype that is broadly safe for the current device."""
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def _resolve_model_path(model_path: Optional[str]) -> str:
    """Resolve the local model path, downloading weights when needed."""
    if model_path is None:
        model_path = os.environ.get("ZIMAGE_MODEL_PATH", "ckpts/Z-Image-Turbo")
    return ensure_model_weights(model_path, verify=False)


def _clear_pipeline_cache() -> None:
    """Release cached pipelines before switching modes to avoid VRAM buildup."""
    global _pipeline_cache

    for pipe in _pipeline_cache.values():
        try:
            pipe.to("cpu")
        except Exception:
            pass

    _pipeline_cache = {}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _configure_pipeline_memory(pipe, mode: str, device: str) -> None:
    """Apply memory-saving features to pipelines when useful."""
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    if device == "cuda" and mode in {"img2img", "inpaint"}:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)


def _load_pipeline(mode: str, model_path: Optional[str] = None):
    """Load and cache the requested Diffusers pipeline."""
    global _device_cache

    pipeline_classes = {
        "text2img": ZImagePipeline,
        "img2img": ZImageImg2ImgPipeline,
        "inpaint": ZImageInpaintPipeline,
    }
    if mode not in pipeline_classes:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of: text2img, img2img, inpaint.")

    resolved_model_path = _resolve_model_path(model_path)
    device = _select_device()
    dtype = _select_dtype(device)
    _device_cache = device

    cache_key = (mode, resolved_model_path, device, str(dtype))
    if cache_key in _pipeline_cache:
        return _pipeline_cache[cache_key], device

    if _pipeline_cache:
        _clear_pipeline_cache()

    pipeline_class = pipeline_classes[mode]
    pipe = pipeline_class.from_pretrained(
        resolved_model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        use_safetensors=True,
    )
    _configure_pipeline_memory(pipe, mode, device)

    _pipeline_cache[cache_key] = pipe
    return pipe, device


def _build_generator(device: str, seed: Optional[int]) -> Optional[torch.Generator]:
    """Create a deterministic generator when a seed is provided."""
    if seed is None:
        return None

    generator_device = "cuda" if device == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def _prepare_image(image: Image.Image, width: Optional[int], height: Optional[int]) -> Image.Image:
    """Resize an input image to the requested dimensions when provided."""
    prepared = image.convert("RGB")
    if width is not None and height is not None and prepared.size != (width, height):
        prepared = prepared.resize((width, height), Image.Resampling.LANCZOS)
    return prepared


def _prepare_mask(mask_image: Image.Image, width: Optional[int], height: Optional[int]) -> Image.Image:
    """Resize a mask to match the requested dimensions."""
    prepared = mask_image.convert("L")
    if width is not None and height is not None and prepared.size != (width, height):
        prepared = prepared.resize((width, height), Image.Resampling.NEAREST)
    return prepared


def run_zimage(
    prompt: str,
    width: Optional[int] = 1024,
    height: Optional[int] = 1024,
    steps: int = 8,
    seed: Optional[int] = None,
    model_path: Optional[str] = None,
    mode: str = "text2img",
    image: Optional[Image.Image] = None,
    mask_image: Optional[Image.Image] = None,
    negative_prompt: Optional[str] = None,
    guidance_scale: float = 0.0,
    strength: float = 0.6,
) -> Image.Image:
    """
    Run a Z-Image Diffusers pipeline.

    Args:
        prompt: Text prompt describing the desired output
        width: Output width for text-to-image, optional for image-conditioned modes
        height: Output height for text-to-image, optional for image-conditioned modes
        steps: Number of inference steps
        seed: Random seed for reproducibility
        model_path: Local model path (defaults to ZIMAGE_MODEL_PATH or ckpts/Z-Image-Turbo)
        mode: One of text2img, img2img, inpaint
        image: Input image for img2img and inpaint
        mask_image: Grayscale mask for inpaint
        negative_prompt: Optional negative prompt
        guidance_scale: CFG value; Turbo models should generally stay at 0.0
        strength: Edit strength for img2img and inpaint

    Returns:
        First generated PIL image from the requested pipeline.
    """
    pipe, device = _load_pipeline(mode, model_path)
    generator = _build_generator(device, seed)

    kwargs = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
    }
    if generator is not None:
        kwargs["generator"] = generator
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    if mode == "text2img":
        kwargs["width"] = width
        kwargs["height"] = height
    elif mode == "img2img":
        if image is None:
            raise ValueError("`image` is required for img2img mode.")
        kwargs["image"] = _prepare_image(image, width, height)
        kwargs["strength"] = strength
        if width is not None:
            kwargs["width"] = width
        if height is not None:
            kwargs["height"] = height
    elif mode == "inpaint":
        if image is None:
            raise ValueError("`image` is required for inpaint mode.")
        if mask_image is None:
            raise ValueError("`mask_image` is required for inpaint mode.")
        kwargs["image"] = _prepare_image(image, width, height)
        kwargs["mask_image"] = _prepare_mask(mask_image, width, height)
        kwargs["strength"] = strength
        if width is not None:
            kwargs["width"] = width
        if height is not None:
            kwargs["height"] = height

    result = pipe(**kwargs)
    return result.images[0]
