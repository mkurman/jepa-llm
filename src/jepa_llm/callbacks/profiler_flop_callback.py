"""Callback for collecting FLOP statistics during training."""

from __future__ import annotations

from typing import Any

import logging

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import TrainerCallback

from ..utils import is_primary_process


logger = logging.getLogger(__name__)


class ProfilerFLOPCallback(TrainerCallback):
    """Profile FLOPs during the initial training steps."""

    def __init__(self, profile_steps: int = 10) -> None:
        self.profile_steps = profile_steps
        self.total_flops = 0

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.global_step < self.profile_steps:
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
            )
            self.profiler.__enter__()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.global_step < self.profile_steps:
            self.profiler.__exit__(None, None, None)
            events = self.profiler.key_averages()
            step_flops = sum(event.flops for event in events if event.flops > 0)
            self.total_flops += step_flops

            if is_primary_process() and state.global_step == 1:
                logger.info(
                    "Step %d: FLOPs: %,d", state.global_step, int(step_flops)
                )


__all__ = ["ProfilerFLOPCallback"]
