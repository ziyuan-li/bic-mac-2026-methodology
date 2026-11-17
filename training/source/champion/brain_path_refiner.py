"""Losses for a full-in-plane residual refiner targeting brain ACF tails."""

from __future__ import annotations

import math

import torch
from torch.nn import functional

from .losses import masked_l1


THRESHOLDS = (0.05, 0.10, 0.15)
DEFAULT_RADON_ANGLES_DEG = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)


def _normalized_exceedance(
    relative: torch.Tensor,
    temperature: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    losses = []
    hard_losses = []
    for threshold in THRESHOLDS:
        zero_offset = functional.softplus(
            relative.new_tensor(-threshold / temperature)
        )
        smooth_excess = temperature * (
            functional.softplus((relative - threshold) / temperature)
            - zero_offset
        )
        losses.append((smooth_excess / threshold).mean())
        hard_losses.append(
            (relative >= threshold).to(relative.dtype).mean()
        )
    return losses, hard_losses


def _tail_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("tail fraction must be in (0, 1]")
    flattened = values.reshape(-1)
    count = max(1, int(math.ceil(fraction * flattened.numel())))
    return torch.topk(flattened, count, sorted=False).values.mean()


def brain_ray_cvar_exceedance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    prediction_mask: torch.Tensor,
    brain_mask: torch.Tensor,
    temperature: float,
    tail_fraction: float,
    spacing_mm: tuple[float, float] = (1.5234375, 1.5234375),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penalize the worst brain-intersecting ACF rays across fixed thresholds."""
    arrays = (prediction, target, prediction_mask, brain_mask)
    if any(array.ndim != 5 for array in arrays):
        raise ValueError("brain-path tensors must have B,C,X,Y,Z dimensions")
    if any(array.shape != prediction.shape for array in arrays):
        raise ValueError("brain-path tensors must have identical shapes")
    if prediction.shape[1] != 1:
        raise ValueError("brain-path tensors must have one channel")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail fraction must be in (0, 1]")
    if len(spacing_mm) != 2 or any(value <= 0.0 for value in spacing_mm):
        raise ValueError("in-plane spacing must contain two positive values")

    active = prediction_mask > 0.5
    brain = brain_mask > 0.5
    errors = (prediction.float() - target.float()) * active
    cvar_losses = []
    hard_losses = []
    acf_losses = []
    path_losses = []
    for axis, axis_spacing_mm in zip((2, 3), spacing_mm):
        valid = torch.any(brain, dim=axis) & torch.any(active, dim=axis)
        if not torch.any(valid):
            continue
        paths = torch.sum(errors, dim=axis) * (float(axis_spacing_mm) / 10.0)
        selected_paths = paths[valid]
        relative = torch.abs(
            torch.expm1(torch.clamp(selected_paths, min=-2.0, max=2.0))
        )
        for threshold in THRESHOLDS:
            zero_offset = functional.softplus(
                relative.new_tensor(-threshold / temperature)
            )
            smooth_excess = temperature * (
                functional.softplus((relative - threshold) / temperature)
                - zero_offset
            )
            normalized = smooth_excess / threshold
            cvar_losses.append(_tail_mean(normalized, tail_fraction))
            hard_losses.append(
                (relative >= threshold).to(relative.dtype).mean()
            )
        acf_losses.append(relative.mean())
        path_losses.append(torch.abs(selected_paths).mean())
    if not cvar_losses:
        raise ValueError("brain mask has no valid in-plane rays")
    return (
        torch.stack(cvar_losses).mean(),
        torch.stack(hard_losses).mean(),
        torch.stack(acf_losses).mean(),
        torch.stack(path_losses).mean(),
    )


def brain_ray_threshold_surrogate(
    prediction: torch.Tensor,
    target: torch.Tensor,
    prediction_mask: torch.Tensor,
    brain_mask: torch.Tensor,
    temperature: float,
    spacing_mm: tuple[float, float] = (1.5234375, 1.5234375),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    arrays = (prediction, target, prediction_mask, brain_mask)
    if any(array.ndim != 5 for array in arrays):
        raise ValueError("brain-path tensors must have B,C,X,Y,Z dimensions")
    if any(array.shape != prediction.shape for array in arrays):
        raise ValueError("brain-path tensors must have identical shapes")
    if prediction.shape[1] != 1:
        raise ValueError("brain-path tensors must have one channel")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if len(spacing_mm) != 2 or any(value <= 0.0 for value in spacing_mm):
        raise ValueError("in-plane spacing must contain two positive values")

    active = prediction_mask > 0.5
    brain = brain_mask > 0.5
    errors = (prediction.float() - target.float()) * active
    surrogate_losses = []
    hard_losses = []
    acf_losses = []
    path_losses = []
    for axis, axis_spacing_mm in zip((2, 3), spacing_mm):
        valid = torch.any(brain, dim=axis) & torch.any(active, dim=axis)
        if not torch.any(valid):
            continue
        paths = torch.sum(errors, dim=axis) * (float(axis_spacing_mm) / 10.0)
        selected_paths = paths[valid]
        relative = torch.abs(
            torch.expm1(torch.clamp(selected_paths, min=-2.0, max=2.0))
        )
        surrogate_losses.extend(
            torch.sigmoid((relative - threshold) / temperature).mean()
            for threshold in THRESHOLDS
        )
        hard_losses.extend(
            (relative >= threshold).to(relative.dtype).mean()
            for threshold in THRESHOLDS
        )
        acf_losses.append(relative.mean())
        path_losses.append(torch.abs(selected_paths).mean())
    if not surrogate_losses:
        raise ValueError("brain mask has no valid in-plane rays")
    return (
        torch.stack(surrogate_losses).mean(),
        torch.stack(hard_losses).mean(),
        torch.stack(acf_losses).mean(),
        torch.stack(path_losses).mean(),
    )


def brain_ray_normalized_exceedance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    prediction_mask: torch.Tensor,
    brain_mask: torch.Tensor,
    temperature: float,
    spacing_mm: tuple[float, float] = (1.5234375, 1.5234375),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penalize threshold exceedance without saturating on severe ray errors."""
    arrays = (prediction, target, prediction_mask, brain_mask)
    if any(array.ndim != 5 for array in arrays):
        raise ValueError("brain-path tensors must have B,C,X,Y,Z dimensions")
    if any(array.shape != prediction.shape for array in arrays):
        raise ValueError("brain-path tensors must have identical shapes")
    if prediction.shape[1] != 1:
        raise ValueError("brain-path tensors must have one channel")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if len(spacing_mm) != 2 or any(value <= 0.0 for value in spacing_mm):
        raise ValueError("in-plane spacing must contain two positive values")

    active = prediction_mask > 0.5
    brain = brain_mask > 0.5
    errors = (prediction.float() - target.float()) * active
    exceedance_losses = []
    hard_losses = []
    acf_losses = []
    path_losses = []
    for axis, axis_spacing_mm in zip((2, 3), spacing_mm):
        valid = torch.any(brain, dim=axis) & torch.any(active, dim=axis)
        if not torch.any(valid):
            continue
        paths = torch.sum(errors, dim=axis) * (float(axis_spacing_mm) / 10.0)
        selected_paths = paths[valid]
        relative = torch.abs(
            torch.expm1(torch.clamp(selected_paths, min=-2.0, max=2.0))
        )
        losses, hard = _normalized_exceedance(relative, temperature)
        exceedance_losses.extend(losses)
        hard_losses.extend(hard)
        acf_losses.append(relative.mean())
        path_losses.append(torch.abs(selected_paths).mean())
    if not exceedance_losses:
        raise ValueError("brain mask has no valid in-plane rays")
    return (
        torch.stack(exceedance_losses).mean(),
        torch.stack(hard_losses).mean(),
        torch.stack(acf_losses).mean(),
        torch.stack(path_losses).mean(),
    )


def brain_radon_path_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    prediction_mask: torch.Tensor,
    brain_mask: torch.Tensor,
    angles_deg: tuple[float, ...] = DEFAULT_RADON_ANGLES_DEG,
    spacing_mm: tuple[float, float] = (1.5234375, 1.5234375),
) -> tuple[torch.Tensor, ...]:
    """Return signed transaxial path errors for each requested Radon angle."""
    arrays = (prediction, target, prediction_mask, brain_mask)
    if any(array.ndim != 5 for array in arrays):
        raise ValueError("brain-path tensors must have B,C,X,Y,Z dimensions")
    if any(array.shape != prediction.shape for array in arrays):
        raise ValueError("brain-path tensors must have identical shapes")
    if prediction.shape[1] != 1:
        raise ValueError("brain-path tensors must have one channel")
    if len(spacing_mm) != 2 or any(value <= 0.0 for value in spacing_mm):
        raise ValueError("in-plane spacing must contain two positive values")
    if not angles_deg or any(not math.isfinite(value) for value in angles_deg):
        raise ValueError("at least one finite Radon angle is required")
    normalized_angles = tuple(float(value) % 180.0 for value in angles_deg)
    if len(set(normalized_angles)) != len(normalized_angles):
        raise ValueError("Radon angles must be unique modulo 180 degrees")
    if not math.isclose(spacing_mm[0], spacing_mm[1], rel_tol=1e-4):
        raise ValueError("multi-angle Radon paths require isotropic in-plane spacing")

    active = (prediction_mask > 0.5).to(torch.float32)
    brain = (brain_mask > 0.5).to(torch.float32)
    errors = (prediction.float() - target.float()) * active
    batch, _, size_x, size_y, size_z = prediction.shape

    def as_planes(values: torch.Tensor) -> torch.Tensor:
        return (
            values.permute(0, 4, 1, 2, 3)
            .reshape(batch * size_z, 1, size_x, size_y)
        )

    error_planes = as_planes(errors)
    active_planes = as_planes(active)
    brain_planes = as_planes(brain)
    selected_path_errors = []
    path_spacing_cm = float(spacing_mm[0]) / 10.0

    for angle_deg in normalized_angles:
        angle = math.radians(angle_deg)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        transform = error_planes.new_tensor(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0))
        ).unsqueeze(0).expand(error_planes.shape[0], -1, -1)
        grid = functional.affine_grid(
            transform,
            error_planes.shape,
            align_corners=False,
        )
        rotated_errors = functional.grid_sample(
            error_planes,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        rotated_active = functional.grid_sample(
            active_planes,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        rotated_brain = functional.grid_sample(
            brain_planes,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        valid = (
            torch.sum(rotated_brain, dim=2) > 0.5
        ) & (
            torch.sum(rotated_active, dim=2) > 0.5
        )
        if not torch.any(valid):
            continue
        paths = torch.sum(rotated_errors, dim=2) * path_spacing_cm
        selected_path_errors.append(paths[valid])

    if not selected_path_errors:
        raise ValueError("brain mask has no valid multi-angle rays")
    return tuple(selected_path_errors)


def brain_radon_normalized_exceedance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    prediction_mask: torch.Tensor,
    brain_mask: torch.Tensor,
    temperature: float,
    angles_deg: tuple[float, ...] = DEFAULT_RADON_ANGLES_DEG,
    spacing_mm: tuple[float, float] = (1.5234375, 1.5234375),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate transaxial PET LOR errors with multi-angle Radon paths."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    path_errors = brain_radon_path_errors(
        prediction,
        target,
        prediction_mask,
        brain_mask,
        angles_deg,
        spacing_mm,
    )
    exceedance_losses = []
    hard_losses = []
    acf_losses = []
    path_losses = []
    for selected_paths in path_errors:
        relative = torch.abs(
            torch.expm1(torch.clamp(selected_paths, min=-2.0, max=2.0))
        )
        losses, hard = _normalized_exceedance(relative, temperature)
        exceedance_losses.extend(losses)
        hard_losses.extend(hard)
        acf_losses.append(relative.mean())
        path_losses.append(torch.abs(selected_paths).mean())
    return (
        torch.stack(exceedance_losses).mean(),
        torch.stack(hard_losses).mean(),
        torch.stack(acf_losses).mean(),
        torch.stack(path_losses).mean(),
    )


def brain_path_refiner_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    base: torch.Tensor,
    eval_mask: torch.Tensor,
    prediction_mask: torch.Tensor,
    brain_mask: torch.Tensor,
    outlier_weight: float,
    brain_mu_weight: float,
    residual_weight: float,
    temperature: float,
    threshold_objective: str = "probability",
    acf_weight: float = 0.0,
    ray_angles_deg: tuple[float, ...] = DEFAULT_RADON_ANGLES_DEG,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if min(outlier_weight, brain_mu_weight, residual_weight, acf_weight) < 0.0:
        raise ValueError("loss weights must be non-negative")
    if brain_mu_weight > 1.0:
        raise ValueError("brain mu weight must not exceed one")
    if threshold_objective not in (
        "probability",
        "normalized-exceedance",
        "multi-angle-normalized-exceedance",
    ):
        raise ValueError(f"unknown threshold objective: {threshold_objective}")
    eval_l1 = masked_l1(prediction, target, eval_mask)
    responsibility_l1 = masked_l1(prediction, target, prediction_mask)
    brain_active = eval_mask.to(prediction.dtype) * (brain_mask > 0.5).to(
        prediction.dtype
    )
    brain_l1 = masked_l1(prediction, target, brain_active)
    residual_l1 = masked_l1(prediction, base, prediction_mask)
    if threshold_objective == "probability":
        objective_function = brain_ray_threshold_surrogate
    elif threshold_objective == "normalized-exceedance":
        objective_function = brain_ray_normalized_exceedance
    else:
        objective_function = brain_radon_normalized_exceedance
    objective_kwargs = (
        {"angles_deg": ray_angles_deg}
        if threshold_objective == "multi-angle-normalized-exceedance"
        else {}
    )
    threshold_loss, hard_outlier, acf_relative, path_abs = objective_function(
        prediction,
        target,
        prediction_mask,
        brain_mask,
        temperature,
        **objective_kwargs,
    )
    range_penalty = torch.relu(-prediction).mean() + torch.relu(
        prediction - 0.30
    ).mean()
    eval_weight = 0.80 - brain_mu_weight
    total = (
        eval_weight * eval_l1
        + 0.20 * responsibility_l1
        + brain_mu_weight * brain_l1
        + outlier_weight * threshold_loss
        + acf_weight * acf_relative
        + residual_weight * residual_l1
        + 0.01 * range_penalty
    )
    return total, {
        "loss": total.detach(),
        "eval_mu_l1": eval_l1.detach(),
        "prediction_mu_l1": responsibility_l1.detach(),
        "brain_mu_l1": brain_l1.detach(),
        "residual_mu_l1": residual_l1.detach(),
        "brain_ray_outlier_surrogate": threshold_loss.detach(),
        "brain_ray_threshold_loss": threshold_loss.detach(),
        "brain_ray_acf_loss": acf_relative.detach(),
        "brain_ray_outlier_hard": hard_outlier.detach(),
        "brain_ray_acf_relative_error": acf_relative.detach(),
        "brain_ray_path_abs_error": path_abs.detach(),
        "range_penalty": range_penalty.detach(),
    }
