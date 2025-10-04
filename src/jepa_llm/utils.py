"""Shared utility helpers for the JEPA fine-tuning CLI."""

from __future__ import annotations

import torch


def is_primary_process() -> bool:
    """Return True when running on the primary process (GPU 0 or CPU)."""
    if not torch.cuda.is_available():
        return True
    try:
        return torch.cuda.current_device() == 0
    except torch.cuda.CudaError:
        return True


__all__ = ["is_primary_process"]
