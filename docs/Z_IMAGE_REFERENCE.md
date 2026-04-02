# Z-Image Reference

This file captures the Z-Image capabilities and pipeline behavior we want to treat as the working reference.

Sources:
- User-provided documentation excerpt
- Official model pages referenced in the excerpt

## Model Overview

Z-Image is a 6B image generation model family.

Currently available:

| Model | Hugging Face |
| --- | --- |
| Z-Image-Turbo | `https://huggingface.co/Tongyi-MAI/Z-Image-Turbo` |

## Z-Image-Turbo

Z-Image-Turbo is a distilled version of Z-Image that:

- uses 8 NFEs
- offers sub-second inference latency on H800 GPUs
- fits within 16 GB VRAM consumer devices
- is strong at photorealistic generation
- supports bilingual text rendering
- has strong instruction following

## Text-to-Image

Use `ZImagePipeline`.

Official example:

```python
import torch
from diffusers import ZImagePipeline

pipe = ZImagePipeline.from_pretrained("Z-a-o/Z-Image-Turbo", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "..."
image = pipe(
    prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("zimage.png")
```

Important documented parameters:

- `height`: defaults to `1024`
- `width`: defaults to `1024`
- `num_inference_steps`: defaults to `50`, but official Turbo examples use `9`
- `guidance_scale`: defaults to `5.0`, but official Turbo examples use `0.0`
- more denoising steps usually improve quality but cost more time
- higher guidance scale usually follows text more strongly but can reduce quality

## Image-to-Image

Use `ZImageImg2ImgPipeline`.

Documented meaning:

- transforms an existing image based on a text prompt
- the input `image` is the starting point
- `strength` controls how much the reference image is changed
- when `strength == 1.0`, the input image is almost ignored
- if `height` and `width` are not supplied, the pipeline uses the input image size

Official example:

```python
import torch
from diffusers import ZImageImg2ImgPipeline
from diffusers.utils import load_image

pipe = ZImageImg2ImgPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16)
pipe.to("cuda")

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
init_image = load_image(url).resize((1024, 1024))

prompt = "A fantasy landscape with mountains and a river, detailed, vibrant colors"
image = pipe(
    prompt,
    image=init_image,
    strength=0.6,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("zimage_img2img.png")
```

Important documented parameters:

- `image`: starting reference image
- `strength`: defaults to `0.6`
- `height`: defaults to `1024`, or uses input image height if omitted
- `width`: defaults to `1024`, or uses input image width if omitted
- `num_inference_steps`: defaults to `50`
- `guidance_scale`: defaults to `5.0`

## Inpainting

Use `ZImageInpaintPipeline`.

Documented meaning:

- inpaints specific regions based on a text prompt and a mask
- `mask_image` defines where to edit
- white pixels in the mask are edited
- black pixels are preserved
- `strength` controls how strongly the masked region is transformed
- when `strength == 1.0`, the masked region is heavily regenerated

Official example:

```python
import torch
import numpy as np
from PIL import Image
from diffusers import ZImageInpaintPipeline
from diffusers.utils import load_image

pipe = ZImageInpaintPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16)
pipe.to("cuda")

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
init_image = load_image(url).resize((1024, 1024))

mask = np.zeros((1024, 1024), dtype=np.uint8)
mask[256:768, 256:768] = 255
mask_image = Image.fromarray(mask)

prompt = "A beautiful lake with mountains in the background"
image = pipe(
    prompt,
    image=init_image,
    mask_image=mask_image,
    strength=1.0,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("zimage_inpaint.png")
```

Important documented parameters:

- `image`: source image
- `mask_image`: white means edit, black means preserve
- `strength`: defaults to `1.0`
- `height`: defaults to `1024`, or uses input image height if omitted
- `width`: defaults to `1024`, or uses input image width if omitted
- `num_inference_steps`: defaults to `50`
- `guidance_scale`: defaults to `5.0`

## Practical Reference Notes

These are the main behavior rules we should keep in mind when implementing around Z-Image:

- `text2img` and `img2img` are different pipelines
- `img2img` uses the reference image as the starting point for a new generation
- `inpaint` edits only the masked region
- official Turbo examples consistently use:
  - `1024x1024`
  - `num_inference_steps=9`
  - `guidance_scale=0.0`
- if our implementation changes image size or other defaults for memory reasons, that is an implementation compromise and not the documented default behavior

## Implementation Guardrails

When debugging or extending the server:

- do not confuse `img2img` with `inpaint`
- do not assume low-quality output means Turbo itself is low quality
- first compare our runtime settings against the official documented settings
- if we lower resolution or alter defaults to fit VRAM, document that clearly as a runtime workaround
