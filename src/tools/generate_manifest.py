#!/usr/bin/env python3
"""Generate manifest file with MD5 checksums for model weights.

Usage:
    python -m tools.generate_manifest ckpts/Z-Image-Turbo
    python -m tools.generate_manifest ckpts/Z-Image-Turbo --no-checksums  # Only list files
"""

import argparse
import hashlib
from pathlib import Path
from typing import List


def compute_md5(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def get_essential_files(model_dir: Path) -> List[Path]:
    """Get list of essential model files."""
    essential_patterns = [
        "model_index.json",
        "transformer/config.json",
        "transformer/*.safetensors*",
        "vae/config.json",
        "vae/*.safetensors",
        "text_encoder/config.json",
        "text_encoder/*.safetensors*",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "scheduler/scheduler_config.json",
    ]

    files = []
    for pattern in essential_patterns:
        if "*" in pattern:
            files.extend(model_dir.glob(pattern))
        else:
            file_path = model_dir / pattern
            if file_path.exists():
                files.append(file_path)

    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Generate manifest file for model weights"
    )
    parser.add_argument("model_dir", type=str, help="Path to model directory")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output manifest file path (default: auto-detect to config/manifests/)",
    )
    parser.add_argument(
        "--no-checksums",
        action="store_true",
        help="Only list files without computing checksums",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print progress")

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        return 1

    # Determine output path
    if args.output:
        output_file = Path(args.output)
    else:
        # Auto-detect: save to config/manifests/{model-name}.txt
        model_name = model_dir.name.lower()  # e.g., "Z-Image-Turbo" -> "z-image-turbo"
        script_dir = Path(__file__).parent
        config_dir = script_dir.parent / "config" / "manifests"
        config_dir.mkdir(parents=True, exist_ok=True)
        output_file = config_dir / f"{model_name}.txt"

    # Get essential files
    files = get_essential_files(model_dir)

    if not files:
        print(f"Warning: No essential files found in {model_dir}")
        return 1

    print(f"Found {len(files)} essential files")

    # Generate manifest
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Z-Image Model Manifest\n")
        if args.no_checksums:
            f.write("# Format: <filepath>\n")
        else:
            f.write("# Format: <md5hash>  <filepath>\n")
        f.write("# Generated automatically - DO NOT edit manually\n\n")

        for file_path in files:
            rel_path = file_path.relative_to(model_dir)

            if args.no_checksums:
                f.write(f"{rel_path}\n")
                if args.verbose:
                    print(f"  {rel_path}")
            else:
                if args.verbose:
                    print(f"Computing MD5 for {rel_path}...", end=" ", flush=True)

                try:
                    md5_hash = compute_md5(file_path)
                    f.write(f"{md5_hash}  {rel_path}\n")
                    if args.verbose:
                        print(f"✓ {md5_hash}")
                except Exception as e:
                    print(f"✗ Error: {e}")
                    continue

    print(f"\n✓ Manifest saved to: {output_file}")
    print(f"  Total files: {len(files)}")
    if not args.no_checksums:
        print(f"  With MD5 checksums for integrity verification")

    return 0


if __name__ == "__main__":
    exit(main())
