"""Exponential moving averages for deterministic model checkpoints."""

from __future__ import annotations

from collections.abc import Mapping

import torch


class ExponentialMovingAverage:
    """Track a same-device EMA over every model state tensor."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must lie in [0, 1)")
        self.decay = float(decay)
        self.num_updates = 0
        self._state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        current = model.state_dict()
        if current.keys() != self._state.keys():
            raise ValueError("EMA model state keys changed during training")
        for name, value in current.items():
            target = self._state[name]
            value = value.detach()
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError(f"EMA tensor contract changed: {name}")
            if torch.is_floating_point(target) or torch.is_complex(target):
                target.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                target.copy_(value)
        self.num_updates += 1

    def model_state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._state)

    def load_state_dict(
        self,
        state: Mapping[str, torch.Tensor],
        num_updates: int,
    ) -> None:
        if state.keys() != self._state.keys():
            raise ValueError("EMA checkpoint state keys do not match the model")
        if num_updates < 0:
            raise ValueError("EMA update count must be non-negative")
        for name, value in state.items():
            target = self._state[name]
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError(f"EMA checkpoint tensor contract changed: {name}")
            target.copy_(value.to(device=target.device))
        self.num_updates = int(num_updates)
