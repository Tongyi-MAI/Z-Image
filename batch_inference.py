"""Batch prompt inference for Z-Image."""

import os
from pathlib import Path
import time

import torch

from inference import ensure_model_weights
from utils import AttentionBackend, load_from_local_dir, set_attention_backend
from zimage import generate


def read_prompts(path: str) -> list[str]:
    """Read prompts from a text file (one per line, empty lines skipped)."""

    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with prompt_path.open("r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_path}")
    return prompts


PROMPTS = read_prompts(os.environ.get("PROMPTS_FILE", "prompts/prompt1.txt"))


def slugify(text: str, max_len: int = 60) -> str:
    """Create a filesystem-safe slug from the prompt."""

    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    slug = "-".join(part for part in slug.split("-") if part)
    return slug[:max_len].rstrip("-") or "prompt"


def select_device() -> str:
    """Choose the best available device without repeating detection logic."""

    if torch.cuda.is_available():
        print("Chosen device: cuda")
        return "cuda"
    try:
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        print("Chosen device: tpu")
        return device
    except (ImportError, RuntimeError):
        if torch.backends.mps.is_available():
            print("Chosen device: mps")
            return "mps"
        print("Chosen device: cpu")
        return "cpu"


def main():
    model_path = ensure_model_weights("ckpts/Z-Image-Turbo")
    dtype = torch.bfloat16
    compile = False
    height = 1024
    width = 1024
    num_inference_steps = 8
    guidance_scale = 0.0
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    device = select_device()

    # Avoid "No available kernel" errors on CPU/MPS by defaulting to math backend there.
    attn_backend = os.environ.get("ZIMAGE_ATTENTION")
    if attn_backend is None:
        attn_backend = "_native_math" if device == "cpu" else "native"

    components = load_from_local_dir(model_path, device=device, dtype=dtype, compile=compile)
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    print(f"Chosen attention backend: {attn_backend}")

    for idx, prompt in enumerate(PROMPTS, start=1):
        output_path = output_dir / f"prompt-{idx:02d}-{slugify(prompt)}.png"
        seed = 42 + idx - 1
        generator = torch.Generator(device).manual_seed(seed)

        start_time = time.time()
        images = generate(
            prompt=prompt,
            **components,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        elapsed = time.time() - start_time
        images[0].save(output_path)
        print(f"[{idx}/{len(PROMPTS)}] Saved {output_path} in {elapsed:.2f} seconds")

    print("Done.")


if __name__ == "__main__":
    main()
