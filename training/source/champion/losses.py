"""Losses aligned with the BIC-MAC CT and reconstruction masks."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    difference = torch.abs(prediction.float() - target.float())
    weights = mask.float()
    denominator = torch.clamp(weights.sum(), min=1.0)
    return (difference * weights).sum() / denominator


def multiscale_target_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scales: tuple[int, ...] = (2, 4, 8),
) -> torch.Tensor:
    """Penalize low-frequency target error at fixed average-pooled scales."""
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    if prediction.ndim != 5 or prediction.shape[1] != 1:
        raise ValueError("multiscale tensors must have shape (batch, 1, x, y, z)")
    if not scales or any(scale <= 1 for scale in scales):
        raise ValueError("multiscale factors must all be greater than one")
    if tuple(sorted(set(scales))) != scales:
        raise ValueError("multiscale factors must be unique and increasing")

    active = mask.to(dtype=prediction.dtype)
    errors = (prediction - target) * active
    losses = []
    for scale in scales:
        if any(size < scale for size in prediction.shape[2:]):
            raise ValueError(
                f"scale {scale} exceeds spatial shape {tuple(prediction.shape[2:])}"
            )
        pooled_error = functional.avg_pool3d(
            errors.float(), kernel_size=scale, stride=scale
        )
        pooled_active = functional.avg_pool3d(
            active.float(), kernel_size=scale, stride=scale
        )
        valid = pooled_active > 0.0
        normalized_error = pooled_error / pooled_active.clamp_min(1.0e-6)
        losses.append(
            normalized_error[valid].abs().mean()
            if valid.any()
            else prediction.new_zeros((), dtype=torch.float32)
        )
    return torch.stack(losses).mean()


def axial_slice_signed_bias_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize signed mean mu error independently on each axial slice."""
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    if prediction.ndim != 5 or prediction.shape[1] != 1:
        raise ValueError("axial tensors must have shape (batch, 1, x, y, z)")
    active = mask.to(dtype=prediction.dtype)
    counts = active.sum(dim=(2, 3))
    valid = counts > 0.0
    signed_means = ((prediction - target) * active).sum(dim=(2, 3)) / counts.clamp_min(1.0)
    if not valid.any():
        return prediction.new_zeros((), dtype=torch.float32)
    return signed_means.float()[valid].abs().mean()


def axial_slice_positive_bias_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize only positive mean mu error on each axial slice."""
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    if prediction.ndim != 5 or prediction.shape[1] != 1:
        raise ValueError("axial tensors must have shape (batch, 1, x, y, z)")
    active = mask.to(dtype=prediction.dtype)
    counts = active.sum(dim=(2, 3))
    valid = counts > 0.0
    signed_means = ((prediction - target) * active).sum(dim=(2, 3)) / counts.clamp_min(1.0)
    if not valid.any():
        return prediction.new_zeros((), dtype=torch.float32)
    return torch.relu(signed_means.float()[valid]).mean()


def air_lung_axial_positive_bias_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    upper_mu: float = 0.0672,
) -> torch.Tensor:
    """Penalize the air/lung contribution to positive axial slice bias."""
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    if prediction.ndim != 5 or prediction.shape[1] != 1:
        raise ValueError("axial tensors must have shape (batch, 1, x, y, z)")
    if upper_mu <= 0.0:
        raise ValueError("air/lung upper mu threshold must be positive")
    full_active = mask.to(dtype=prediction.dtype)
    air_lung_active = full_active * (target < upper_mu).to(
        dtype=prediction.dtype
    )
    counts = full_active.sum(dim=(2, 3))
    valid = counts > 0.0
    signed_means = ((prediction - target) * air_lung_active).sum(
        dim=(2, 3)
    ) / counts.clamp_min(1.0)
    if not valid.any():
        return prediction.new_zeros((), dtype=torch.float32)
    return torch.relu(signed_means.float()[valid]).mean()


def masked_gradient_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize local mu-error changes without penalizing a global offset."""
    error = prediction - target
    active = mask.to(dtype=prediction.dtype)
    losses = []
    for axis in (2, 3, 4):
        lower = [slice(None)] * error.ndim
        upper = [slice(None)] * error.ndim
        lower[axis] = slice(None, -1)
        upper[axis] = slice(1, None)
        pair_mask = active[tuple(lower)] * active[tuple(upper)]
        difference = error[tuple(upper)] - error[tuple(lower)]
        losses.append(
            (torch.abs(difference) * pair_mask).sum()
            / torch.clamp(pair_mask.sum(), min=1.0)
        )
    return torch.stack(losses).mean()


def _gaussian_kernel1d(
    sigma_voxels: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    radius = max(1, int(round(3.0 * sigma_voxels)))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (coordinates / sigma_voxels).square())
    return (kernel / kernel.sum()).to(dtype=dtype)


def smooth_mu_4mm(values: torch.Tensor) -> torch.Tensor:
    """Approximate the official 4 mm FWHM blur on the fixed CT grid."""
    # Cache volumes use 1.523 x 1.523 x 2.0 mm spacing.
    sigma_mm = 4.0 / 2.355
    result = values
    for axis, spacing in enumerate((1.5234375, 1.5234375, 2.0)):
        kernel = _gaussian_kernel1d(sigma_mm / spacing, values.device, values.dtype)
        shape = [1, 1, 1, 1, 1]
        shape[axis + 2] = kernel.numel()
        weight = kernel.reshape(shape)
        padding = [0, 0, 0]
        padding[axis] = kernel.numel() // 2
        result = functional.conv3d(result, weight, padding=tuple(padding))
    return result


def projection_path_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize signed in-plane line-integral errors as an ACF surrogate."""
    active = mask.to(dtype=prediction.dtype)
    errors = (prediction - target) * active
    losses = []
    for axis in (2, 3):
        counts = active.sum(dim=axis)
        valid = counts > 0
        mean_path_error = torch.abs(errors.sum(dim=axis)) / torch.clamp(counts, min=1.0)
        losses.append(mean_path_error[valid].mean() if valid.any() else prediction.new_zeros(()))
    return torch.stack(losses).mean()


def acf_path_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Approximate relative ACF error from physical in-plane line integrals."""
    active = mask.to(dtype=prediction.dtype)
    # Mu is cm^-1 while the cache spacing is stored in millimetres.
    errors = ((prediction - target) * active).float()
    losses = []
    for axis, spacing_mm in ((2, 1.5234375), (3, 1.5234375)):
        counts = active.sum(dim=axis)
        valid = counts > 0
        line_integral_error = errors.sum(dim=axis) * (spacing_mm / 10.0)
        relative_acf_error = torch.abs(
            torch.expm1(torch.clamp(line_integral_error, min=-2.0, max=2.0))
        )
        losses.append(
            relative_acf_error[valid].mean()
            if valid.any()
            else prediction.new_zeros((), dtype=torch.float32)
        )
    return torch.stack(losses).mean()


def acf_noninferiority_hinge_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    base: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize rays whose target-relative ACF error exceeds a frozen base."""
    if not (
        prediction.shape == target.shape == base.shape == mask.shape
        and prediction.ndim == 5
        and prediction.shape[1] == 1
    ):
        raise ValueError(
            "prediction, target, base, and mask must share shape (batch, 1, x, y, z)"
        )

    active = mask.float()
    candidate_error = (prediction.float() - target.float()) * active
    base_error = (base.detach().float() - target.float()) * active
    hinge_losses = []
    candidate_losses = []
    base_losses = []
    violation_fractions = []
    for axis, spacing_mm in ((2, 1.5234375), (3, 1.5234375)):
        valid = active.sum(dim=axis) > 0
        if not valid.any():
            zero = prediction.new_zeros((), dtype=torch.float32)
            hinge_losses.append(zero)
            candidate_losses.append(zero)
            base_losses.append(zero)
            violation_fractions.append(zero)
            continue
        scale = spacing_mm / 10.0
        candidate_line_error = candidate_error.sum(dim=axis) * scale
        base_line_error = base_error.sum(dim=axis) * scale
        candidate_relative = torch.abs(
            torch.expm1(torch.clamp(candidate_line_error, min=-2.0, max=2.0))
        )
        base_relative = torch.abs(
            torch.expm1(torch.clamp(base_line_error, min=-2.0, max=2.0))
        )
        excess = torch.relu(candidate_relative - base_relative)
        hinge_losses.append(excess[valid].mean())
        candidate_losses.append(candidate_relative[valid].mean())
        base_losses.append(base_relative[valid].mean())
        violation_fractions.append(
            (candidate_relative[valid] > base_relative[valid]).float().mean()
        )

    hinge = torch.stack(hinge_losses).mean()
    return hinge, {
        "acf_noninferiority_hinge": hinge.detach(),
        "acf_noninferiority_candidate_error": torch.stack(candidate_losses)
        .mean()
        .detach(),
        "acf_noninferiority_base_error": torch.stack(base_losses).mean().detach(),
        "acf_noninferiority_violation_fraction": torch.stack(violation_fractions)
        .mean()
        .detach(),
    }


def base_physics_preservation_loss(
    prediction: torch.Tensor,
    base: torch.Tensor,
    prediction_mask: torch.Tensor,
    smooth_weight: float,
    projection_weight: float,
    acf_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize PET-relevant departures from a reconstruction-safe base map."""
    if prediction.shape != base.shape:
        raise ValueError(
            "prediction and base shapes differ: "
            f"{tuple(prediction.shape)} != {tuple(base.shape)}"
        )
    if min(smooth_weight, projection_weight, acf_weight) < 0.0:
        raise ValueError("base preservation weights must be non-negative")

    base = base.detach()
    smooth_l1 = prediction.new_zeros(())
    if smooth_weight > 0.0:
        smooth_l1 = masked_l1(
            smooth_mu_4mm(prediction),
            smooth_mu_4mm(base),
            prediction_mask,
        )
    path_l1 = prediction.new_zeros(())
    if projection_weight > 0.0:
        path_l1 = projection_path_loss(prediction, base, prediction_mask)
    acf_error = prediction.new_zeros((), dtype=torch.float32)
    if acf_weight > 0.0:
        acf_error = acf_path_loss(prediction, base, prediction_mask)

    total = (
        smooth_weight * smooth_l1
        + projection_weight * path_l1
        + acf_weight * acf_error
    )
    return total, {
        "base_preservation_loss": total.detach(),
        "base_smooth_mu_l1": smooth_l1.detach(),
        "base_projection_path_l1": path_l1.detach(),
        "base_acf_path_relative_error": acf_error.detach(),
    }


def tissue_signed_bias_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eval_mask: torch.Tensor,
) -> torch.Tensor:
    """Give each CT-equivalent tissue equal weight in signed calibration."""
    ranges = (
        (0.0, 0.0672),
        (0.0672, 0.0912),
        (0.0912, 0.1134),
        (0.1134, float("inf")),
    )
    errors = prediction - target
    losses = []
    for lower, upper in ranges:
        active = eval_mask * ((target >= lower) & (target < upper)).to(eval_mask.dtype)
        count = active.sum()
        if count > 0:
            losses.append(torch.abs((errors * active).sum() / count))
    return torch.stack(losses).mean() if losses else prediction.new_zeros(())


def direct_mu_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eval_mask: torch.Tensor,
    prediction_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    eval_l1 = masked_l1(prediction, target, eval_mask)
    responsibility_l1 = masked_l1(prediction, target, prediction_mask)
    range_penalty = torch.relu(-prediction).mean() + torch.relu(prediction - 0.30).mean()
    total = 0.75 * eval_l1 + 0.25 * responsibility_l1 + 0.01 * range_penalty
    return total, {
        "loss": total.detach(),
        "eval_mu_l1": eval_l1.detach(),
        "prediction_mu_l1": responsibility_l1.detach(),
        "range_penalty": range_penalty.detach(),
    }


def eval_only_mu_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eval_mask: torch.Tensor,
    prediction_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optimize the official CT evaluation mask without PET responsibility loss."""
    eval_l1 = masked_l1(prediction, target, eval_mask)
    responsibility_l1 = masked_l1(prediction, target, prediction_mask)
    range_penalty = torch.relu(-prediction).mean() + torch.relu(prediction - 0.30).mean()
    total = eval_l1 + 0.01 * range_penalty
    return total, {
        "loss": total.detach(),
        "eval_mu_l1": eval_l1.detach(),
        "prediction_mu_l1": responsibility_l1.detach(),
        "range_penalty": range_penalty.detach(),
    }


def combine_teacher_distillation_loss(
    supervised_loss: torch.Tensor,
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    prediction_mask: torch.Tensor,
    teacher_weight: float,
    target: torch.Tensor | None = None,
    reliability_scale: float = 0.0,
    base_prediction: torch.Tensor | None = None,
    improvement_only: bool = False,
    oracle_endpoint: bool = False,
    oracle_margin: float = 0.0,
    oracle_selected_weight: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend supervised loss with OOF-teacher agreement inside responsibility."""
    if not 0.0 <= teacher_weight <= 1.0:
        raise ValueError("teacher weight must be between zero and one")
    if reliability_scale < 0.0:
        raise ValueError("teacher reliability scale must be non-negative")
    if oracle_margin < 0.0:
        raise ValueError("teacher oracle margin must be non-negative")
    if oracle_endpoint and improvement_only:
        raise ValueError(
            "oracle-endpoint and improvement-only distillation are mutually exclusive"
        )
    if oracle_margin > 0.0 and not oracle_endpoint:
        raise ValueError("teacher oracle margin requires oracle-endpoint distillation")
    if oracle_endpoint and reliability_scale > 0.0:
        raise ValueError(
            "teacher reliability weighting is not defined for oracle endpoints"
        )
    if oracle_selected_weight is not None and not (
        oracle_endpoint and 0.0 < oracle_selected_weight < 1.0
    ):
        raise ValueError(
            "oracle selected weight must be in (0, 1) and requires an "
            "oracle endpoint"
        )
    if prediction.shape != teacher.shape:
        raise ValueError(
            f"prediction and teacher shapes differ: "
            f"{tuple(prediction.shape)} != {tuple(teacher.shape)}"
        )
    teacher_mask = prediction_mask
    teacher_better: torch.Tensor | None = None
    distillation_target = teacher
    if oracle_endpoint or improvement_only:
        if target is None:
            raise ValueError(
                "target is required when selective distillation is enabled"
            )
        if base_prediction is None:
            raise ValueError(
                "base prediction is required when selective distillation is enabled"
            )
        if base_prediction.shape != teacher.shape:
            raise ValueError(
                "base prediction and teacher shapes differ: "
                f"{tuple(base_prediction.shape)} != {tuple(teacher.shape)}"
            )
        teacher_error = torch.abs(teacher.detach() - target.detach())
        base_error = torch.abs(base_prediction.detach() - target.detach())
        teacher_better = teacher_error + oracle_margin < base_error
        if oracle_endpoint:
            distillation_target = torch.where(
                teacher_better,
                teacher.detach(),
                base_prediction.detach(),
            )
        else:
            teacher_mask = teacher_mask.to(teacher_error.dtype) * (
                teacher_better
            ).to(teacher_error.dtype)
    if reliability_scale > 0.0:
        if target is None:
            raise ValueError(
                "target is required when teacher reliability weighting is enabled"
            )
        if target.shape != teacher.shape:
            raise ValueError(
                f"target and teacher shapes differ: "
                f"{tuple(target.shape)} != {tuple(teacher.shape)}"
            )
        reliability = torch.exp(
            -torch.abs(teacher.detach() - target.detach()) / reliability_scale
        )
        teacher_mask = teacher_mask.to(reliability.dtype) * reliability
    if oracle_selected_weight is None:
        teacher_l1 = masked_l1(prediction, distillation_target, teacher_mask)
    else:
        if teacher_better is None or base_prediction is None:
            raise RuntimeError("balanced oracle endpoint has no selection state")
        active = teacher_mask.to(prediction.dtype)
        selected = active * teacher_better.to(prediction.dtype)
        retained = active * (~teacher_better).to(prediction.dtype)
        selected_l1 = masked_l1(prediction, teacher.detach(), selected)
        retained_l1 = masked_l1(
            prediction,
            base_prediction.detach(),
            retained,
        )
        selected_present = (selected.sum() > 0).to(prediction.dtype)
        retained_present = (retained.sum() > 0).to(prediction.dtype)
        selected_weight = float(oracle_selected_weight)
        normalizer = (
            selected_weight * selected_present
            + (1.0 - selected_weight) * retained_present
        ).clamp_min(torch.finfo(prediction.dtype).eps)
        teacher_l1 = (
            selected_weight * selected_present * selected_l1
            + (1.0 - selected_weight) * retained_present * retained_l1
        ) / normalizer
    total = (1.0 - teacher_weight) * supervised_loss + teacher_weight * teacher_l1
    return total, teacher_l1


def tissue_aware_mu_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    eval_mask: torch.Tensor,
    prediction_mask: torch.Tensor,
    tissue_aux_weight: float,
    extreme_weight: float,
    smooth_weight: float = 0.0,
    projection_weight: float = 0.0,
    acf_weight: float = 0.0,
    tissue_bias_weight: float = 0.0,
    gradient_weight: float = 0.0,
    gradient_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if output.shape[1] not in (1, 5):
        raise ValueError(f"expected one or five output channels, got {output.shape[1]}")
    prediction = output[:, :1]
    eval_l1 = masked_l1(prediction, target, eval_mask)
    responsibility_l1 = masked_l1(prediction, target, prediction_mask)

    extreme = ((target < 0.0672) | (target >= 0.1134)).to(prediction.dtype)
    regression_weights = eval_mask.to(prediction.dtype) * (1.0 + (extreme_weight - 1.0) * extreme)
    weighted_l1 = (
        torch.abs(prediction - target) * regression_weights
    ).sum() / torch.clamp(regression_weights.sum(), min=1.0)

    tissue_ce = prediction.new_zeros(())
    if output.shape[1] == 5:
        tissue_logits = output[:, 1:]
        tissue_target = torch.zeros_like(target[:, 0], dtype=torch.long)
        tissue_target[target[:, 0] >= 0.0672] = 1
        tissue_target[target[:, 0] >= 0.0912] = 2
        tissue_target[target[:, 0] >= 0.1134] = 3
        class_weights = torch.tensor((2.0, 1.0, 0.75, 2.0), device=output.device)
        voxel_ce = functional.cross_entropy(
            tissue_logits.float(), tissue_target, weight=class_weights, reduction="none"
        )
        mask = eval_mask[:, 0].to(voxel_ce.dtype)
        tissue_ce = (voxel_ce * mask).sum() / torch.clamp(mask.sum(), min=1.0)

    range_penalty = torch.relu(-prediction).mean() + torch.relu(prediction - 0.30).mean()
    smooth_l1 = prediction.new_zeros(())
    if smooth_weight > 0:
        smooth_l1 = masked_l1(
            smooth_mu_4mm(prediction),
            smooth_mu_4mm(target),
            eval_mask,
        )
    path_l1 = prediction.new_zeros(())
    if projection_weight > 0:
        path_l1 = projection_path_loss(prediction, target, prediction_mask)
    acf_path_error = prediction.new_zeros((), dtype=torch.float32)
    if acf_weight > 0:
        acf_path_error = acf_path_loss(prediction, target, prediction_mask)
    tissue_bias = prediction.new_zeros(())
    if tissue_bias_weight > 0:
        tissue_bias = tissue_signed_bias_loss(prediction, target, eval_mask)
    gradient_l1 = prediction.new_zeros(())
    if gradient_weight > 0:
        active_gradient_mask = eval_mask if gradient_mask is None else gradient_mask
        if active_gradient_mask.shape != prediction.shape:
            raise ValueError("gradient mask must have the prediction shape")
        gradient_l1 = masked_gradient_l1(
            prediction,
            target,
            active_gradient_mask,
        )
    total = (
        0.55 * eval_l1
        + 0.20 * responsibility_l1
        + 0.25 * weighted_l1
        + tissue_aux_weight * tissue_ce
        + smooth_weight * smooth_l1
        + projection_weight * path_l1
        + acf_weight * acf_path_error
        + tissue_bias_weight * tissue_bias
        + gradient_weight * gradient_l1
        + 0.01 * range_penalty
    )
    return total, {
        "loss": total.detach(),
        "eval_mu_l1": eval_l1.detach(),
        "prediction_mu_l1": responsibility_l1.detach(),
        "weighted_mu_l1": weighted_l1.detach(),
        "tissue_ce": tissue_ce.detach(),
        "smooth_mu_l1": smooth_l1.detach(),
        "projection_path_l1": path_l1.detach(),
        "acf_path_relative_error": acf_path_error.detach(),
        "tissue_signed_bias": tissue_bias.detach(),
        "gradient_mu_l1": gradient_l1.detach(),
        "range_penalty": range_penalty.detach(),
    }
