"""Two-objective minimum-norm gradient combination utilities."""

from __future__ import annotations

import math
from typing import Any

import torch


def finite_metric_tensor(
    value: float | int | bool,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Materialize gradient diagnostics without inheriting autocast fp16."""
    metric = reference.new_tensor(float(value), dtype=torch.float32)
    if not bool(torch.isfinite(metric).item()):
        raise RuntimeError(f"gradient diagnostic is non-finite: {value!r}")
    return metric


def primary_priority_projection_metrics(
    primary_norm_sq: float,
    constraint_norm_sq: float,
    dot_product: float,
) -> dict[str, float | bool]:
    """Project a conflicting primary gradient onto the constraint half-space."""
    values = (primary_norm_sq, constraint_norm_sq, dot_product)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("gradient geometry must be finite")
    if primary_norm_sq < 0.0 or constraint_norm_sq < 0.0:
        raise ValueError("squared gradient norms must be non-negative")

    tolerance = 1e-20 * max(primary_norm_sq, constraint_norm_sq, 1.0)
    projection_applied = constraint_norm_sq > tolerance and dot_product < 0.0
    coefficient = (
        -dot_product / constraint_norm_sq if projection_applied else 0.0
    )
    primary_dot_direction = primary_norm_sq + coefficient * dot_product
    direction_norm_sq = (
        primary_norm_sq
        + 2.0 * coefficient * dot_product
        + coefficient**2 * constraint_norm_sq
    )
    constraint_dot_direction = (
        0.0
        if projection_applied
        else dot_product
    )
    primary_dot_direction = max(0.0, primary_dot_direction)
    direction_norm_sq = max(0.0, direction_norm_sq)
    primary_retention = (
        primary_dot_direction / primary_norm_sq
        if primary_norm_sq > tolerance
        else 0.0
    )
    cosine = 0.0
    if primary_norm_sq > tolerance and constraint_norm_sq > tolerance:
        cosine = dot_product / math.sqrt(primary_norm_sq * constraint_norm_sq)
        cosine = min(1.0, max(-1.0, cosine))
    return {
        "primary_norm_sq": primary_norm_sq,
        "constraint_norm_sq": constraint_norm_sq,
        "dot_product": dot_product,
        "cosine_similarity": cosine,
        "projection_applied": projection_applied,
        "projection_coefficient": coefficient,
        "primary_dot_direction": primary_dot_direction,
        "constraint_dot_direction": constraint_dot_direction,
        "direction_norm_sq": direction_norm_sq,
        "primary_dot_retention": primary_retention,
        "primary_descent": primary_dot_direction > tolerance,
        "constraint_nonincrease": constraint_dot_direction >= 0.0,
    }


def two_objective_min_norm_metrics(
    primary_norm_sq: float,
    constraint_norm_sq: float,
    dot_product: float,
) -> dict[str, float | bool]:
    """Return the minimum-norm convex combination of two gradients."""
    values = (primary_norm_sq, constraint_norm_sq, dot_product)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("gradient geometry must be finite")
    if primary_norm_sq < 0.0 or constraint_norm_sq < 0.0:
        raise ValueError("squared gradient norms must be non-negative")

    tolerance = 1e-20 * max(primary_norm_sq, constraint_norm_sq, 1.0)
    if constraint_norm_sq <= tolerance:
        alpha_primary = 1.0
    elif primary_norm_sq <= tolerance:
        alpha_primary = 0.0
    else:
        denominator = primary_norm_sq + constraint_norm_sq - 2.0 * dot_product
        if denominator <= tolerance:
            alpha_primary = 0.5
        else:
            alpha_primary = min(
                1.0,
                max(0.0, (constraint_norm_sq - dot_product) / denominator),
            )
    alpha_constraint = 1.0 - alpha_primary
    primary_dot_direction = (
        alpha_primary * primary_norm_sq + alpha_constraint * dot_product
    )
    constraint_dot_direction = (
        alpha_primary * dot_product + alpha_constraint * constraint_norm_sq
    )
    direction_norm_sq = (
        alpha_primary**2 * primary_norm_sq
        + alpha_constraint**2 * constraint_norm_sq
        + 2.0 * alpha_primary * alpha_constraint * dot_product
    )
    common_descent = (
        primary_dot_direction > tolerance
        and constraint_dot_direction > tolerance
        and direction_norm_sq > tolerance
    )
    cosine = 0.0
    if primary_norm_sq > tolerance and constraint_norm_sq > tolerance:
        cosine = dot_product / math.sqrt(primary_norm_sq * constraint_norm_sq)
        cosine = min(1.0, max(-1.0, cosine))
    return {
        "primary_norm_sq": primary_norm_sq,
        "constraint_norm_sq": constraint_norm_sq,
        "dot_product": dot_product,
        "cosine_similarity": cosine,
        "alpha_primary": alpha_primary,
        "alpha_constraint": alpha_constraint,
        "primary_dot_direction": primary_dot_direction,
        "constraint_dot_direction": constraint_dot_direction,
        "direction_norm_sq": max(0.0, direction_norm_sq),
        "common_descent": common_descent,
    }


def accumulate_two_objective_gradients(
    primary_loss: torch.Tensor,
    constraint_loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    accumulation_steps: int,
) -> dict[str, float | bool | int]:
    """Accumulate one minimum-norm common direction into ``parameter.grad``."""
    if accumulation_steps <= 0:
        raise ValueError("accumulation steps must be positive")
    primary_gradients = torch.autograd.grad(
        primary_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    constraint_gradients = torch.autograd.grad(
        constraint_loss,
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    device = primary_loss.device
    primary_norm_sq = torch.zeros((), device=device, dtype=torch.float64)
    constraint_norm_sq = torch.zeros((), device=device, dtype=torch.float64)
    dot_product = torch.zeros((), device=device, dtype=torch.float64)
    nonfinite = 0
    joint_tensors = 0
    gradient_values = 0
    for parameter, primary, constraint in zip(
        parameters,
        primary_gradients,
        constraint_gradients,
        strict=True,
    ):
        if primary is None and constraint is None:
            continue
        gradient_values += parameter.numel()
        if primary is not None:
            nonfinite += int((~torch.isfinite(primary)).sum().item())
            primary_norm_sq += primary.detach().float().square().sum().double()
        if constraint is not None:
            nonfinite += int((~torch.isfinite(constraint)).sum().item())
            constraint_norm_sq += constraint.detach().float().square().sum().double()
        if primary is not None and constraint is not None:
            joint_tensors += 1
            dot_product += (
                primary.detach().float() * constraint.detach().float()
            ).sum().double()
    if nonfinite:
        raise RuntimeError(f"MGDA produced {nonfinite} non-finite gradient values")
    geometry = two_objective_min_norm_metrics(
        float(primary_norm_sq.item()),
        float(constraint_norm_sq.item()),
        float(dot_product.item()),
    )
    alpha_primary = float(geometry["alpha_primary"])
    alpha_constraint = float(geometry["alpha_constraint"])
    for parameter, primary, constraint in zip(
        parameters,
        primary_gradients,
        constraint_gradients,
        strict=True,
    ):
        if primary is None and constraint is None:
            continue
        combined = torch.zeros_like(parameter)
        if primary is not None:
            combined.add_(primary, alpha=alpha_primary)
        if constraint is not None:
            combined.add_(constraint, alpha=alpha_constraint)
        combined.div_(accumulation_steps)
        if parameter.grad is None:
            parameter.grad = combined
        else:
            parameter.grad.add_(combined)
    return {
        **geometry,
        "gradient_nonfinite_value_count": nonfinite,
        "joint_parameter_tensor_count": joint_tensors,
        "gradient_parameter_value_count": gradient_values,
    }


def accumulate_primary_priority_gradients(
    primary_loss: torch.Tensor,
    constraint_loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    accumulation_steps: int,
) -> dict[str, float | bool | int]:
    """Accumulate the primary gradient with only conflicting components removed."""
    if accumulation_steps <= 0:
        raise ValueError("accumulation steps must be positive")
    primary_gradients = torch.autograd.grad(
        primary_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    constraint_gradients = torch.autograd.grad(
        constraint_loss,
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    device = primary_loss.device
    primary_norm_sq = torch.zeros((), device=device, dtype=torch.float64)
    constraint_norm_sq = torch.zeros((), device=device, dtype=torch.float64)
    dot_product = torch.zeros((), device=device, dtype=torch.float64)
    nonfinite = 0
    joint_tensors = 0
    gradient_values = 0
    for parameter, primary, constraint in zip(
        parameters,
        primary_gradients,
        constraint_gradients,
        strict=True,
    ):
        if primary is None and constraint is None:
            continue
        gradient_values += parameter.numel()
        if primary is not None:
            nonfinite += int((~torch.isfinite(primary)).sum().item())
            primary_norm_sq += primary.detach().float().square().sum().double()
        if constraint is not None:
            nonfinite += int((~torch.isfinite(constraint)).sum().item())
            constraint_norm_sq += constraint.detach().float().square().sum().double()
        if primary is not None and constraint is not None:
            joint_tensors += 1
            dot_product += (
                primary.detach().float() * constraint.detach().float()
            ).sum().double()
    if nonfinite:
        raise RuntimeError(
            f"primary-priority projection produced {nonfinite} non-finite gradients"
        )
    geometry = primary_priority_projection_metrics(
        float(primary_norm_sq.item()),
        float(constraint_norm_sq.item()),
        float(dot_product.item()),
    )
    coefficient = float(geometry["projection_coefficient"])
    for parameter, primary, constraint in zip(
        parameters,
        primary_gradients,
        constraint_gradients,
        strict=True,
    ):
        if primary is None and constraint is None:
            continue
        combined = torch.zeros_like(parameter)
        if primary is not None:
            combined.add_(primary)
        if constraint is not None and coefficient > 0.0:
            combined.add_(constraint, alpha=coefficient)
        combined.div_(accumulation_steps)
        if parameter.grad is None:
            parameter.grad = combined
        else:
            parameter.grad.add_(combined)
    return {
        **geometry,
        "gradient_nonfinite_value_count": nonfinite,
        "joint_parameter_tensor_count": joint_tensors,
        "gradient_parameter_value_count": gradient_values,
    }
