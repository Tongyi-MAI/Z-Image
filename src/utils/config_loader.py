"""Runtime configuration loader for inference scripts.

- Reads a TOML file (default: config/runtime.toml, override with ZIMAGE_CONFIG).
- Supports the sentinel string "auto" to request built-in defaults.
- Detects conflicts between environment variables and config values.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, cast

from config import inference as default_cfg

AUTO = "auto"


class ConfigConflictError(ValueError):
    """Raised when the same parameter is set to conflicting values."""


@dataclass
class RuntimeConfig:
    prompt_file: Optional[Path]
    prompt_text: Optional[str]
    output_dir: Path
    output_path: Path
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    attention_backend: Optional[str]
    compile: bool
    seed: Optional[int]


def _coerce_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        lowered = val.lower().strip()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Cannot parse boolean value from {val!r}")


def _coerce_int(val: Any) -> int:
    if isinstance(val, bool):
        raise ValueError("Boolean is not valid for int parameter")
    return int(val)


def _coerce_float(val: Any) -> float:
    if isinstance(val, bool):
        raise ValueError("Boolean is not valid for float parameter")
    return float(val)


def _coerce_str(val: Any) -> str:
    return str(val)


def _resolve(
    name: str,
    env_key: Optional[str],
    cfg_value: Any,
    default_value: Any,
    coerce: Callable[[Any], Any],
):
    env_raw = os.environ.get(env_key) if env_key else None
    env_set = env_raw is not None
    env_val = coerce(env_raw) if env_set else None

    cfg_set = cfg_value not in (None, AUTO)
    cfg_val = coerce(cfg_value) if cfg_set else None

    if env_set and cfg_set and env_val != cfg_val:
        raise ConfigConflictError(
            f"Conflict for '{name}': env {env_key}={env_raw!r} vs config {cfg_val!r}"
        )

    if env_set:
        return env_val
    if cfg_set:
        return cfg_val
    return default_value


def load_runtime_config(config_path: Optional[str | Path] = None) -> RuntimeConfig:
    cfg_path = Path(
        config_path or os.environ.get("ZIMAGE_CONFIG", "config/runtime.toml")
    )
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("rb") as f:
        raw = tomllib.load(f)

    prompts_cfg = raw.get("prompts", {})
    output_cfg = raw.get("output", {})
    infer_cfg = raw.get("inference", {})

    prompt_file = _resolve(
        "prompt_file",
        env_key="PROMPTS_FILE",
        cfg_value=prompts_cfg.get("file"),
        default_value=None,
        coerce=lambda v: Path(_coerce_str(v)),
    )
    prompt_text = _resolve(
        "prompt_text",
        env_key="PROMPT_TEXT",
        cfg_value=prompts_cfg.get("single_prompt"),
        default_value=None,
        coerce=_coerce_str,
    )

    if prompt_file and prompt_text:
        raise ConfigConflictError(
            "Specify either 'prompt_file' or 'prompt_text', not both"
        )

    output_dir = _resolve(
        "output_dir",
        env_key="OUTPUT_DIR",
        cfg_value=output_cfg.get("output_dir"),
        default_value=Path("outputs"),
        coerce=lambda v: Path(_coerce_str(v)),
    )
    output_path = _resolve(
        "output_path",
        env_key="OUTPUT_PATH",
        cfg_value=output_cfg.get("output_path"),
        default_value=Path("example.png"),
        coerce=lambda v: Path(_coerce_str(v)),
    )

    height = _resolve(
        "height",
        env_key="HEIGHT",
        cfg_value=infer_cfg.get("height"),
        default_value=default_cfg.DEFAULT_HEIGHT,
        coerce=_coerce_int,
    )
    width = _resolve(
        "width",
        env_key="WIDTH",
        cfg_value=infer_cfg.get("width"),
        default_value=default_cfg.DEFAULT_WIDTH,
        coerce=_coerce_int,
    )
    num_inference_steps = _resolve(
        "num_inference_steps",
        env_key="NUM_INFERENCE_STEPS",
        cfg_value=infer_cfg.get("num_inference_steps"),
        default_value=default_cfg.DEFAULT_INFERENCE_STEPS,
        coerce=_coerce_int,
    )
    guidance_scale = _resolve(
        "guidance_scale",
        env_key="GUIDANCE_SCALE",
        cfg_value=infer_cfg.get("guidance_scale"),
        default_value=default_cfg.DEFAULT_GUIDANCE_SCALE,
        coerce=_coerce_float,
    )
    attention_backend = _resolve(
        "attention_backend",
        env_key="ZIMAGE_ATTENTION",
        cfg_value=infer_cfg.get("attention_backend"),
        default_value=None,
        coerce=_coerce_str,
    )
    compile_flag = _resolve(
        "compile",
        env_key="ZIMAGE_COMPILE",
        cfg_value=infer_cfg.get("compile"),
        default_value=False,
        coerce=_coerce_bool,
    )
    seed = _resolve(
        "seed",
        env_key="SEED",
        cfg_value=infer_cfg.get("seed"),
        default_value=None,
        coerce=_coerce_int,
    )

    return RuntimeConfig(
        prompt_file=cast(Optional[Path], prompt_file),
        prompt_text=prompt_text,
        output_dir=cast(Path, output_dir),
        output_path=cast(Path, output_path),
        height=cast(int, height),
        width=cast(int, width),
        num_inference_steps=cast(int, num_inference_steps),
        guidance_scale=cast(float, guidance_scale),
        attention_backend=attention_backend,
        compile=cast(bool, compile_flag),
        seed=cast(Optional[int], seed),
    )
