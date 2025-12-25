"""RunPod Serverless Handler for Z-Image API."""

import base64
import io
from typing import Dict, Any

import runpod
from zimage_wrapper import run_zimage


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless handler function.
    
    Args:
        event: Event dictionary containing:
            - input: Dictionary with prompt, width, height, steps, seed
    
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
        
        width = input_data.get("width", 1024)
        height = input_data.get("height", 1024)
        steps = input_data.get("steps", 8)
        seed = input_data.get("seed")
        
        # Generate image
        image = run_zimage(
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
        )
        
        # Convert PIL Image to base64
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        image_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
        
        return {
            "image": image_base64,
            "format": "png",
            "width": width,
            "height": height,
        }
    
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }


# Start RunPod serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
