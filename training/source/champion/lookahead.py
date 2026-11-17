"""Lookahead slow-weight synchronization for model training."""

from __future__ import annotations

from collections.abc import Mapping

import torch


class LookaheadSynchronizer:
    """Synchronize fast model parameters with Lookahead slow weights."""

    def __init__(
        self,
        model: torch.nn.Module,
        sync_steps: int = 5,
        alpha: float = 0.5,
    ) -> None:
        if sync_steps <= 0:
            raise ValueError("Lookahead sync steps must be positive")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("Lookahead alpha must lie in (0, 1]")
        self.sync_steps = int(sync_steps)
        self.alpha = float(alpha)
        self.num_updates = 0
        self.num_syncs = 0
        self._slow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> bool:
        """Record one optimizer step and synchronize on every k-th update."""
        current = dict(model.named_parameters())
        if current.keys() != self._slow.keys():
            raise ValueError("Lookahead model parameter keys changed during training")
        self.num_updates += 1
        if self.num_updates % self.sync_steps != 0:
            return False
        for name, parameter in current.items():
            slow = self._slow[name]
            value = parameter.detach()
            if slow.shape != value.shape or slow.dtype != value.dtype:
                raise ValueError(f"Lookahead tensor contract changed: {name}")
            slow.add_(value - slow, alpha=self.alpha)
            parameter.copy_(slow)
        self.num_syncs += 1
        return True

    def slow_parameter_state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._slow)

    def slow_model_state_dict(
        self,
        model: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Return a full model state with slow parameters and current buffers."""
        state = dict(model.state_dict())
        if not self._slow.keys() <= state.keys():
            raise ValueError("Lookahead parameters do not match the model state")
        state.update(self._slow)
        return state

    def load_state_dict(
        self,
        slow_parameters: Mapping[str, torch.Tensor],
        num_updates: int,
        num_syncs: int,
    ) -> None:
        if slow_parameters.keys() != self._slow.keys():
            raise ValueError("Lookahead checkpoint keys do not match the model")
        if num_updates < 0 or num_syncs < 0:
            raise ValueError("Lookahead counters must be non-negative")
        if num_syncs > num_updates // self.sync_steps:
            raise ValueError("Lookahead synchronization count is inconsistent")
        for name, value in slow_parameters.items():
            target = self._slow[name]
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError(f"Lookahead checkpoint tensor contract changed: {name}")
            target.copy_(value.to(device=target.device))
        self.num_updates = int(num_updates)
        self.num_syncs = int(num_syncs)
