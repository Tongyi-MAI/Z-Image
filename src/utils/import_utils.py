import importlib.util

import torch


def is_flash_attn_available():
    try:
        return importlib.util.find_spec("flash_attn") is not None
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
        return False


def is_flash_attn_3_available():
    try:
        return importlib.util.find_spec("flash_attn_interface") is not None
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
        return False


def is_torch_version(operator: str, version: str):
    from packaging import version as pversion

    torch_version = pversion.parse(torch.__version__)
    target_version = pversion.parse(version)

    # print(f"torch_version: {torch_version}, target: torch{operator}{target_version}")
    if operator == ">":
        return torch_version > target_version
    elif operator == ">=":
        return torch_version >= target_version
    elif operator == "==":
        return torch_version == target_version
    elif operator == "<=":
        return torch_version <= target_version
    elif operator == "<":
        return torch_version < target_version
    return False
