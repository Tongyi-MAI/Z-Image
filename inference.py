"""Z-Image PyTorch Native Inference."""

import time
import warnings

import torch

warnings.filterwarnings("ignore")
from utils import load_from_local_dir, set_attention_backend
from zimage import generate


# Before starting, put `ckpts/Z-Image-Turbo` in `ckpts` folder after download from `Tongyi-MAI/Z-Image-Turbo`
def main():
    model_path = "ckpts/Z-Image-Turbo"
    dtype = torch.bfloat16
    compile = False  # default False for compatibility
    output_path = "example.png"
    height = 1024
    width = 1024
    num_inference_steps = 8
    guidance_scale = 0.0
    seed = 42
    prompt = (
        "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. "
        "Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. "
        "Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, "
        "silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
    )

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("Chosen device: ", device)
    
    # Load models
    components = load_from_local_dir(model_path, device=device, dtype=dtype, compile=compile)
    set_attention_backend("_native_flash")  # default is "native", this one is a torch native impl ops for your convient
    # may also try `_flash_3`(FA3) or `flash`(FA2), but you may install that env from its official repository

    # Gen an image
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
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    images[0].save(output_path)

    ### !! For best speed performance, recommend to use `_flash_3` backend and set `compile=True`
    ### This would give you sub-second generation speed on Hopper GPU (H100/H200/H800) after warm-up


if __name__ == "__main__":
    main()
