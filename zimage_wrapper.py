"""Minimal wrapper for Z-Image inference."""

import os
from typing import Optional

import torch
from PIL import Image

from src.utils import ensure_model_weights, load_from_local_dir
from src.zimage import generate


# Global cache for loaded model components
_components_cache = None
_device_cache = None


def _select_device() -> str:
    """Select the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    except (ImportError, RuntimeError):
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def _load_components(model_path: Optional[str] = None):
    """Load model components once and cache them."""
    global _components_cache, _device_cache
    
    if _components_cache is not None:
        return _components_cache, _device_cache
    
    # Default model path
    if model_path is None:
        model_path = os.environ.get("ZIMAGE_MODEL_PATH", "ckpts/Z-Image-Turbo")
    
    # Ensure model weights exist
    model_path = ensure_model_weights(model_path, verify=False)
    
    # Select device
    device = _select_device()
    _device_cache = device
    
    # Load components
    dtype = torch.bfloat16
    _components_cache = load_from_local_dir(
        model_path,
        device=device,
        dtype=dtype,
        compile=False,
    )
    
    return _components_cache, device


def run_zimage(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    seed: Optional[int] = None,
    model_path: Optional[str] = None,
) -> Image.Image:
    """
    Generate an image from a text prompt using Z-Image.
    
    Args:
        prompt: Text prompt describing the image to generate
        width: Image width in pixels (default: 1024)
        height: Image height in pixels (default: 1024)
        steps: Number of inference steps (default: 8)
        seed: Random seed for reproducibility (default: None)
        model_path: Path to model directory (default: from env or "ckpts/Z-Image-Turbo")
    
    Returns:
        PIL.Image: Generated image
    """
    # Load components (cached after first call)
    components, device = _load_components(model_path)
    
    # Create generator if seed is provided
    generator = None
    if seed is not None:
        generator = torch.Generator(device).manual_seed(seed)
    
    # Generate image
    images = generate(
        prompt=prompt,
        **components,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=0.0,  # Turbo models use 0.0
        generator=generator,
        output_type="pil",
    )
    
    # Return first image
    return images[0]

