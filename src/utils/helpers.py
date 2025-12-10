"""Helper utilities for Z-Image."""

import hashlib
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from loguru import logger
import torch

from config import BYTES_PER_GB


def format_bytes(size: float) -> str:
    """
    Format bytes to GB string.

    Args:
        size: Size in bytes

    Returns:
        Formatted string in GB
    """
    n = size / BYTES_PER_GB
    return f"{n:.2f} GB"


def print_memory_stats(stage: str) -> None:
    """
    Print CUDA memory statistics.

    Args:
        stage: Description of current stage
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping memory stats")
        return

    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    current_allocated = torch.cuda.memory_allocated()
    current_reserved = torch.cuda.memory_reserved()

    logger.info(f"[{stage}] Memory Stats:")
    logger.info(f"  Current Allocated: {format_bytes(current_allocated)}")
    logger.info(f"  Current Reserved:  {format_bytes(current_reserved)}")
    logger.info(f"  Peak Allocated:    {format_bytes(allocated)}")
    logger.info(f"  Peak Reserved:     {format_bytes(reserved)}")


def compute_file_md5(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def load_manifest(manifest_file: Path) -> Dict[str, Optional[str]]:
    """Load manifest file. Returns dict mapping file paths to MD5 hashes (or None)."""
    manifest = {}
    if not manifest_file.exists():
        return manifest

    with open(manifest_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) == 1:
                # Only file path, no checksum
                file_path = parts[0]
                manifest[file_path] = None
            elif len(parts) == 2:
                # File path with checksum
                if len(parts[0]) == 32 and all(
                    c in "0123456789abcdef" for c in parts[0].lower()
                ):
                    md5_hash, file_path = parts
                else:
                    file_path, md5_hash = parts
                manifest[file_path] = md5_hash
            else:
                logger.warning(f"Invalid manifest format at line {line_num}: {line}")
                continue

    return manifest


def verify_file_integrity(
    base_dir: Path, manifest: Dict[str, Optional[str]], verify_checksums: bool = True
) -> Tuple[bool, List[str], List[str]]:
    """
    Verify file integrity using a manifest.

    Args:
        base_dir: Base directory for relative file paths
        manifest: Dictionary of relative paths to MD5 hashes (None if no hash provided)
        verify_checksums: If True, verify MD5 checksums when available; if False, only check existence

    Returns:
        Tuple of (all_valid: bool, missing_files: List[str], corrupted_files: List[str])
    """
    missing = []
    corrupted = []

    for rel_path, expected_md5 in manifest.items():
        file_path = base_dir / rel_path

        if not file_path.exists():
            missing.append(rel_path)
            continue

        # Only verify checksum if requested AND hash is available
        if verify_checksums and expected_md5 is not None:
            try:
                actual_md5 = compute_file_md5(file_path)
                if actual_md5 != expected_md5:
                    corrupted.append(rel_path)
                    logger.debug(
                        f"Checksum mismatch for {rel_path}: expected {expected_md5}, got {actual_md5}"
                    )
            except Exception as e:
                logger.error(f"Failed to compute checksum for {rel_path}: {e}")
                corrupted.append(rel_path)

    all_valid = len(missing) == 0 and len(corrupted) == 0
    return all_valid, missing, corrupted


def ensure_model_weights(
    model_path: str,
    repo_id: str = "Tongyi-MAI/Z-Image-Turbo",
    verify: bool = False,
    manifest_name: Optional[str] = None,
) -> Path:
    """
    Ensure model weights exist and optionally verify integrity.

    Args:
        model_path: Path to model directory
        repo_id: HuggingFace repo ID for download
        verify: If True, verify MD5 checksums; if False, only check existence
        manifest_name: Manifest file name in src/config/manifests/ (auto-detect if None)

    Returns:
        Path to validated model directory
    """
    from huggingface_hub import snapshot_download

    target_dir = Path(model_path)

    # Determine manifest path
    if manifest_name:
        # Explicitly specified manifest from config/manifests/
        manifest_path = (
            Path(__file__).parent.parent / "config" / "manifests" / manifest_name
        )
    else:
        # Auto-detect
        model_name = target_dir.name.lower()  # e.g., "Z-Image-Turbo" -> "z-image-turbo"
        config_manifest = (
            Path(__file__).parent.parent / "config" / "manifests" / f"{model_name}.txt"
        )

        if config_manifest.exists():
            manifest_path = config_manifest
        else:
            # Fallback
            manifest_path = target_dir / "manifest.txt"

    manifest = load_manifest(manifest_path)

    if not manifest:
        logger.warning(f"Manifest file not found: {manifest_path}")
        logger.warning("Skipping file verification (assuming model exists)")
        if target_dir.exists():
            logger.info(f"✓ Model directory exists: {target_dir}")
            return target_dir
        else:
            logger.warning(f"Model directory not found: {target_dir}")
            missing_files = ["entire model directory"]
            corrupted_files = []
    else:
        # Count files with checksums
        files_with_checksums = sum(1 for v in manifest.values() if v is not None)

        if verify and files_with_checksums == 0:
            logger.info(
                f"Verify requested but no checksums in manifest, only checking existence"
            )
        elif verify and files_with_checksums > 0:
            logger.info(
                f"Verifying {files_with_checksums} file(s) with MD5 checksums..."
            )

        # Verify files
        all_valid, missing_files, corrupted_files = verify_file_integrity(
            target_dir, manifest, verify_checksums=verify
        )

        if all_valid:
            if verify and files_with_checksums > 0:
                logger.success(
                    f"✓ All files verified with MD5 checksums in {target_dir}"
                )
            else:
                logger.info(
                    f"✓ All {len(manifest)} required files exist in {target_dir}"
                )
            return target_dir

    # Report missing and corrupted files
    if missing_files:
        logger.warning(f"Missing {len(missing_files)} file(s):")
        for f in missing_files[:10]:
            logger.warning(f"  - {f}")
        if len(missing_files) > 10:
            logger.warning(f"  ... and {len(missing_files) - 10} more")

    if corrupted_files:
        logger.error(f"Corrupted {len(corrupted_files)} file(s) (checksum mismatch):")
        for f in corrupted_files[:10]:
            logger.error(f"  - {f}")
        if len(corrupted_files) > 10:
            logger.error(f"  ... and {len(corrupted_files) - 10} more")

    # Download model weights
    logger.info(f"\nAttempting to download from {repo_id}...")
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        logger.success("✓ Download completed")
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        logger.info(
            f"\nIf you are offline, please manually download from:\n"
            f"  https://huggingface.co/{repo_id}\n"
            f"and place in: {target_dir.absolute()}"
        )
        raise RuntimeError(f"Failed to download model weights: {e}")

    # Verify after download
    if manifest:
        all_valid, missing_after, corrupted_after = verify_file_integrity(
            target_dir, manifest, verify_checksums=verify
        )

        if not all_valid:
            error_msg = []
            if missing_after:
                error_msg.append(f"Still missing {len(missing_after)} file(s)")
            if corrupted_after:
                error_msg.append(f"Still corrupted {len(corrupted_after)} file(s)")

            raise FileNotFoundError(
                f"After download: {', '.join(error_msg)}\n"
                f"Please verify the download or manually place files in:\n"
                f"  {target_dir.absolute()}"
            )

    logger.success("✓ All model weights validated successfully")
    return target_dir
