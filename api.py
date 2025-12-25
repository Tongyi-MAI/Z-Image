"""FastAPI wrapper for Z-Image inference."""

import io
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from zimage_wrapper import run_zimage


app = FastAPI(title="Z-Image API", version="1.0.0")


class GenerateRequest(BaseModel):
    """Request model for image generation."""
    
    prompt: str = Field(..., description="Text prompt describing the image to generate")
    width: int = Field(default=1024, ge=256, le=2048, description="Image width in pixels")
    height: int = Field(default=1024, ge=256, le=2048, description="Image height in pixels")
    steps: int = Field(default=8, ge=1, le=50, description="Number of inference steps")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


@app.post("/generate")
async def generate_image(request: GenerateRequest) -> Response:
    """
    Generate an image from a text prompt.
    
    Returns PNG image as bytes.
    """
    try:
        # Generate image using the wrapper
        image = run_zimage(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            steps=request.steps,
            seed=request.seed,
        )
        
        # Convert PIL Image to PNG bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        return Response(
            content=img_bytes.getvalue(),
            media_type="image/png",
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

