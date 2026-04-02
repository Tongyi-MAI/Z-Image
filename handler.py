"""RunPod Serverless Handler for Z-Image API."""

import base64
import gc
import io
import os
import sys
from typing import Any, Dict, Optional

# Mitigate CUDA memory fragmentation (PyTorch recommendation).
# PYTORCH_CUDA_ALLOC_CONF is deprecated in newer PyTorch versions; keep both for compatibility.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import runpod
    print("OK: runpod imported successfully")
except ImportError:
    print("ERROR: runpod package not found. Install with: pip install runpod")
    sys.exit(1)

try:
    from zimage_wrapper import run_zimage
    print("OK: zimage_wrapper imported successfully")
except ImportError as e:
    print(f"ERROR: Cannot import zimage_wrapper: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


ASPECT_PRESETS = {
    "16:9": (1024, 576),
    "9:16": (576, 1024),
}


def _decode_base64_image(image_data: Optional[str]):
    """Decode a base64 string or data URI into a PIL image."""
    if not image_data:
        return None

    if "," in image_data and image_data.split(",", 1)[0].startswith("data:"):
        image_data = image_data.split(",", 1)[1]

    decoded = base64.b64decode(image_data)
    from PIL import Image

    image = Image.open(io.BytesIO(decoded))
    image.load()
    return image


def _infer_mode(input_data: Dict[str, Any]) -> str:
    """Infer generation mode while preserving text-to-image backwards compatibility."""
    mode = input_data.get("mode")
    if mode:
        return str(mode).strip().lower()
    if input_data.get("mask_image"):
        return "inpaint"
    if input_data.get("image"):
        return "img2img"
    return "text2img"


def _normalize_aspect_ratio(value: Optional[str]) -> Optional[str]:
    """Normalize common aspect ratio spellings to supported presets."""
    if value is None:
        return None

    cleaned = str(value).strip().replace(" ", "")
    aliases = {
        "16:9": "16:9",
        "16/9": "16:9",
        "landscape": "16:9",
        "9:16": "9:16",
        "9/16": "9:16",
        "portrait": "9:16",
    }
    return aliases.get(cleaned)


def _resolve_aspect_ratio(input_data: Dict[str, Any], image=None) -> str:
    """Resolve the target aspect ratio to one of the supported high-quality presets."""
    explicit_aspect = _normalize_aspect_ratio(input_data.get("aspect_ratio"))
    if explicit_aspect:
        return explicit_aspect

    width = input_data.get("width")
    height = input_data.get("height")
    if width and height:
        if width > height:
            return "16:9"
        if height > width:
            return "9:16"

    if image is not None:
        if image.width > image.height:
            return "16:9"
        if image.height > image.width:
            return "9:16"

    return "16:9"


def _resolve_dimensions(input_data: Dict[str, Any], image=None) -> tuple[str, int, int]:
    """Resolve the request dimensions to supported presets only."""
    width = input_data.get("width")
    height = input_data.get("height")

    if width is not None or height is not None:
        if (width, height) == ASPECT_PRESETS["16:9"]:
            return "16:9", width, height
        if (width, height) == ASPECT_PRESETS["9:16"]:
            return "9:16", width, height
        raise ValueError(
            "Only high-quality presets are supported: 1024x576 (16:9) or 576x1024 (9:16)."
        )

    aspect_ratio = _resolve_aspect_ratio(input_data, image=image)
    if aspect_ratio not in ASPECT_PRESETS:
        raise ValueError("Only 16:9 and 9:16 aspect ratios are supported.")

    width, height = ASPECT_PRESETS[aspect_ratio]
    return aspect_ratio, width, height


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless handler function.
    
    Args:
        event: Event dictionary containing:
            - input: Dictionary with prompt, mode, image, mask_image, width, height, steps, seed
    
    Returns:
        Dictionary with:
            - image: Base64 encoded PNG image
            - or error: Error message
    """
    try:
        # Extract input parameters
        input_data = event.get("input", {})
        
        prompt = input_data.get("prompt")
        if not prompt:
            return {"error": "Prompt is required"}
        
        mode = _infer_mode(input_data)
        steps = input_data.get("steps", 9)
        seed = input_data.get("seed")
        negative_prompt = input_data.get("negative_prompt")
        guidance_scale = input_data.get("guidance_scale", 0.0)
        strength = input_data.get("strength", 0.6)
        image = _decode_base64_image(input_data.get("image"))
        mask_image = _decode_base64_image(input_data.get("mask_image"))

        if mode == "text2img":
            aspect_ratio, width, height = _resolve_dimensions(input_data)
        elif mode == "img2img":
            if image is None:
                return {"error": "`image` is required for img2img mode"}
            aspect_ratio, width, height = _resolve_dimensions(input_data, image=image)
        elif mode == "inpaint":
            if image is None:
                return {"error": "`image` is required for inpaint mode"}
            if mask_image is None:
                return {"error": "`mask_image` is required for inpaint mode"}
            aspect_ratio, width, height = _resolve_dimensions(input_data, image=image)
        else:
            return {"error": f"Unsupported mode '{mode}'. Expected text2img, img2img, or inpaint."}

        def _run(w: int, h: int):
            return run_zimage(
                prompt=prompt,
                mode=mode,
                width=w,
                height=h,
                steps=steps,
                seed=seed,
                image=image,
                mask_image=mask_image,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                strength=strength,
            )

        image = _run(width, height)
        
        # Convert PIL Image to base64
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        image_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
        
        return {
            "image": image_base64,
            "format": "png",
            "width": image.width,
            "height": image.height,
            "mode": mode,
            "aspect_ratio": aspect_ratio,
        }
    
    except Exception as e:
        import traceback

        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
    finally:
        # Best-effort cleanup between jobs to reduce VRAM fragmentation.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()


# Start RunPod serverless handler
if __name__ == "__main__":
    try:
        print("Starting RunPod serverless handler...")
        print("Handler function loaded successfully")
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
