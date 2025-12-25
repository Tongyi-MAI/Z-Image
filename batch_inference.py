"""Batch prompt inference for Z-Image."""

import os
from pathlib import Path
import time
import numpy
import torch

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


def _print_config_summary(cfg: RuntimeConfig, *, attn_backend: str, prompt_path: Path):
    _banner("Runtime configuration")
    print(f"prompt_file   : {prompt_path}")
    print(f"output_dir    : {cfg.output_dir}")
    print(f"height x width: {cfg.height} x {cfg.width}")
    print(f"steps         : {cfg.num_inference_steps}")
    print(f"guidance      : {cfg.guidance_scale}")
    print(f"attention     : {attn_backend}")
    print(f"compile       : {cfg.compile}")
    print(f"seed          : {cfg.seed if cfg.seed is not None else 'auto/random'}")


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


def read_prompts(path: Path) -> list[str]:
    """Read prompts from a text file (one per line, empty lines skipped)."""

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def slugify(text: str, max_len: int = 60) -> str:
    """Create a filesystem-safe slug from the prompt."""

    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    slug = "-".join(part for part in slug.split("-") if part)
    return slug[:max_len].rstrip("-") or "prompt"


def select_device() -> str:
    """Choose the best available device without repeating detection logic."""

    if torch.cuda.is_available():
        _banner("Chosen device: cuda")
        return "cuda"
    try:
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        _banner("Chosen device: tpu")
        return device
    except (ImportError, RuntimeError):
        if torch.backends.mps.is_available():
            _banner("Chosen device: mps")
            return "mps"
        _banner("Chosen device: cpu")
        return "cpu"


def main():
    cfg: RuntimeConfig = load_runtime_config()

    model_path = ensure_model_weights("ckpts/Z-Image-Turbo")
    dtype = torch.bfloat16
    compile = _resolve_compile(bool(cfg.compile))
    height = cfg.height
    width = cfg.width
    num_inference_steps = cfg.num_inference_steps
    guidance_scale = cfg.guidance_scale
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()

    # Avoid "No available kernel" errors on CPU/MPS by defaulting to math backend there.
    attn_backend = cfg.attention_backend
    if attn_backend is None:
        attn_backend = "_native_math" if device == "cpu" else "native"

    components = load_from_local_dir(
        model_path, device=device, dtype=dtype, compile=compile
    )
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    _banner(f"Attention backend: {attn_backend}")

    prompt_path = cfg.prompt_file
    if prompt_path is None:
        raise ConfigConflictError(
            "No prompt file specified; set prompts.file in config"
        )
    _print_config_summary(cfg, attn_backend=attn_backend, prompt_path=prompt_path)
    prompts = read_prompts(prompt_path)

    for idx, prompt in enumerate(prompts, start=1):
        output_path = output_dir / f"prompt-{idx:02d}-{slugify(prompt)}.png"
        seed = (
            cfg.seed + idx - 1
            if cfg.seed is not None
            else numpy.random.randint(0, 10000)
        )
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
        print(f"→ [{idx}/{len(prompts)}] Saved {output_path} in {elapsed:.2f} seconds")

    _banner("Batch complete")


if __name__ == "__main__":
    main()
