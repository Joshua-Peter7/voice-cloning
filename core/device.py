"""
core/device.py
──────────────
GPU / device resolution helpers shared across all components.
"""

from __future__ import annotations

from typing import Literal

import torch


DeviceStr = Literal["cuda", "cpu", "mps"]


def resolve_device(preference: str = "auto") -> DeviceStr:
    """
    Resolve the best available compute device.

    preference: "auto" | "cuda" | "cpu" | "mps"
    """
    if preference != "auto":
        return preference  # type: ignore[return-value]

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str, preferred: str = "float16"):
    """Return torch dtype appropriate for the device."""
    if device == "cpu" or preferred == "float32":
        return torch.float32
    return torch.float16


def gpu_info() -> dict:
    """Return basic GPU diagnostics."""
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "vram_gb": round(props.total_memory / 1e9, 1),
        "vram_used_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
        "cuda_version": torch.version.cuda,
    }
