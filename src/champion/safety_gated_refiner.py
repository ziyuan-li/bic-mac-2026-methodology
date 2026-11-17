"""CT-safety-weighted hard gate for selective PET-safe teacher transfer."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .model import DirectMuUNet


class MuSafetyGatedRefiner(DirectMuUNet):
    """Use a straight-through hard gate during training and hard inference."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 16,
        correction_scale: float = 0.01,
        mu_scale: float = 0.15,
        gate_prior: float = 0.45,
    ) -> None:
        if correction_scale <= 0.0:
            raise ValueError("correction_scale must be positive")
        if mu_scale <= 0.0:
            raise ValueError("mu_scale must be positive")
        if not 0.0 < gate_prior < 0.5:
            raise ValueError("gate_prior must be strictly between zero and 0.5")
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            output_channels=1,
        )
        self.output = nn.Conv3d(base_channels, 2, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[1] = math.log(gate_prior / (1.0 - gate_prior))
        self.correction_scale = float(correction_scale)
        self.mu_scale = float(mu_scale)
        self.gate_prior = float(gate_prior)

    def forward_components(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base_mu = values[:, :1] * self.mu_scale
        raw = super().forward(values)
        residual_unit = torch.tanh(raw[:, :1])
        gate_logits = raw[:, 1:2]
        gate_probability = torch.sigmoid(gate_logits)
        hard_gate = (gate_probability >= 0.5).to(residual_unit.dtype)
        if self.training:
            applied_gate = hard_gate.detach() + gate_probability - gate_probability.detach()
        else:
            applied_gate = hard_gate
        prediction = (
            base_mu + self.correction_scale * residual_unit * applied_gate
        )
        return prediction, residual_unit, gate_logits, hard_gate

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        prediction, _, _, _ = self.forward_components(values)
        return prediction


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask.to(values.dtype)
    return (values * active).sum() / active.sum().clamp_min(1.0)


def _weighted_masked_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    active = mask.to(values.dtype) * weights.to(values.dtype)
    return (values * active).sum() / active.sum().clamp_min(1.0)


def safety_gated_refiner_loss(
    prediction: torch.Tensor,
    residual_unit: torch.Tensor,
    gate_logits: torch.Tensor,
    base_prediction: torch.Tensor,
    teacher: torch.Tensor,
    target: torch.Tensor,
    prediction_mask: torch.Tensor,
    eval_mask: torch.Tensor,
    *,
    correction_scale: float,
    oracle_margin: float,
    advantage_importance: float = 4.0,
    residual_weight: float = 1.0,
    gate_weight: float = 0.5,
    eval_endpoint_weight: float = 1.0,
    prediction_endpoint_weight: float = 0.25,
    safety_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prioritize consequential CT decisions and penalize harm versus the base."""
    tensors = (
        prediction,
        residual_unit,
        gate_logits,
        base_prediction,
        teacher,
        target,
        prediction_mask,
        eval_mask,
    )
    if any(values.shape != prediction.shape for values in tensors[1:]):
        raise ValueError("safety-gated loss tensors must share one shape")
    if correction_scale <= 0.0 or oracle_margin < 0.0:
        raise ValueError("scale must be positive and margin non-negative")
    weights = (
        advantage_importance,
        residual_weight,
        gate_weight,
        eval_endpoint_weight,
        prediction_endpoint_weight,
        safety_weight,
    )
    if min(weights) < 0.0:
        raise ValueError("safety-gated loss weights must be non-negative")
    if sum(weights[1:]) <= 0.0:
        raise ValueError("at least one safety-gated loss weight must be positive")

    frozen_base = base_prediction.detach()
    frozen_teacher = teacher.detach()
    frozen_target = target.detach()
    base_error = torch.abs(frozen_base - frozen_target)
    teacher_error = torch.abs(frozen_teacher - frozen_target)
    advantage = base_error - teacher_error - oracle_margin
    teacher_better = advantage > 0.0
    gate_target = teacher_better.to(prediction.dtype)
    endpoint = torch.where(teacher_better, frozen_teacher, frozen_base)
    residual_target = torch.clamp(
        (frozen_teacher - frozen_base) / correction_scale,
        -1.0,
        1.0,
    )
    importance = 1.0 + advantage_importance * torch.clamp(
        torch.abs(advantage) / correction_scale,
        0.0,
        1.0,
    )

    residual_loss = _masked_mean(
        F.smooth_l1_loss(
            residual_unit,
            residual_target,
            reduction="none",
            beta=0.1,
        ),
        prediction_mask,
    )
    gate_values = F.binary_cross_entropy_with_logits(
        gate_logits,
        gate_target,
        reduction="none",
    )
    eval_gate_loss = _weighted_masked_mean(
        gate_values,
        importance,
        eval_mask,
    )
    prediction_gate_loss = _weighted_masked_mean(
        gate_values,
        importance,
        prediction_mask,
    )
    gate_loss = eval_gate_loss + 0.25 * prediction_gate_loss
    eval_endpoint_mu_l1 = _masked_mean(
        torch.abs(prediction - endpoint),
        eval_mask,
    )
    prediction_endpoint_mu_l1 = _masked_mean(
        torch.abs(prediction - endpoint),
        prediction_mask,
    )
    normalized_eval_endpoint = eval_endpoint_mu_l1 / correction_scale
    normalized_prediction_endpoint = (
        prediction_endpoint_mu_l1 / correction_scale
    )
    safety_hinge = _masked_mean(
        F.relu(torch.abs(prediction - frozen_target) - base_error),
        eval_mask,
    )
    normalized_safety_hinge = safety_hinge / correction_scale
    total = (
        residual_weight * residual_loss
        + gate_weight * gate_loss
        + eval_endpoint_weight * normalized_eval_endpoint
        + prediction_endpoint_weight * normalized_prediction_endpoint
        + safety_weight * normalized_safety_hinge
    )

    with torch.no_grad():
        predicted_gate = torch.sigmoid(gate_logits) >= 0.5
        eval_gate_accuracy = _weighted_masked_mean(
            (predicted_gate == teacher_better).to(prediction.dtype),
            importance,
            eval_mask,
        )
        eval_mu_l1 = _masked_mean(
            torch.abs(prediction - frozen_target),
            eval_mask,
        )
        prediction_mu_l1 = _masked_mean(
            torch.abs(prediction - frozen_target),
            prediction_mask,
        )
        selected_fraction = _masked_mean(
            predicted_gate.to(prediction.dtype),
            eval_mask,
        )
    return total, {
        "loss": total.detach(),
        "residual_loss": residual_loss.detach(),
        "eval_gate_loss": eval_gate_loss.detach(),
        "prediction_gate_loss": prediction_gate_loss.detach(),
        "normalized_eval_endpoint_loss": normalized_eval_endpoint.detach(),
        "normalized_prediction_endpoint_loss": (
            normalized_prediction_endpoint.detach()
        ),
        "normalized_safety_hinge": normalized_safety_hinge.detach(),
        "eval_mu_l1": eval_mu_l1.detach(),
        "prediction_mu_l1": prediction_mu_l1.detach(),
        "weighted_eval_gate_accuracy": eval_gate_accuracy.detach(),
        "selected_eval_fraction": selected_fraction.detach(),
    }
