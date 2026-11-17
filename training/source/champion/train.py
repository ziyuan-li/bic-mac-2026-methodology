"""Train a NAC-PET-to-mu model on one CV fold or all available subjects."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler

from .brain_path_refiner import brain_ray_cvar_exceedance
from .data import MuPatchDataset, resolve_training_subjects
from .ema import ExponentialMovingAverage
from .head_output_adapter import (
    HeadOutputAdapterProjectionConditionedMuUNet,
    LocalContextHeadOutputAdapterProjectionConditionedMuUNet,
    freeze_for_head_output_adapter,
)
from .losses import (
    acf_noninferiority_hinge_loss,
    acf_path_loss,
    air_lung_axial_positive_bias_loss,
    axial_slice_positive_bias_loss,
    axial_slice_signed_bias_loss,
    base_physics_preservation_loss,
    combine_teacher_distillation_loss,
    direct_mu_loss,
    eval_only_mu_loss,
    masked_gradient_l1,
    masked_l1,
    multiscale_target_l1,
    tissue_aware_mu_loss,
)
from .lookahead import LookaheadSynchronizer
from .model import (
    DirectMuUNet,
    FullResolutionMRIProjectionConditionedMuUNet,
    MRIProjectionConditionedMuUNet,
    ProjectionConditionedMuUNet,
)
from .mgda import (
    accumulate_primary_priority_gradients,
    accumulate_two_objective_gradients,
    finite_metric_tensor,
)
from .support_gated_slice_refiner import acf_tail_path_loss


def distributed_mean(value: torch.Tensor, world_size: int) -> float:
    value = value.detach().float()
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size
    return float(value.cpu())


def physics_preservation_reference(
    inputs: torch.Tensor,
    teacher: torch.Tensor | None,
    base_mu_scale: float,
    preserve_teacher_physics: bool,
) -> torch.Tensor:
    """Select the frozen map whose PET physics the prediction must preserve."""
    if preserve_teacher_physics:
        if teacher is None:
            raise ValueError(
                "teacher is required when teacher physics preservation is enabled"
            )
        return teacher
    if base_mu_scale <= 0.0:
        raise ValueError("base_mu_scale must be positive")
    return inputs[:, :1] * base_mu_scale


def validate_physics_preservation_settings(
    teacher_weight: float,
    smooth_weight: float,
    projection_weight: float,
    acf_weight: float,
    preserve_teacher_physics: bool,
    has_online_physics_teacher: bool = False,
) -> None:
    weights = (smooth_weight, projection_weight, acf_weight)
    if min(weights) < 0.0:
        raise ValueError("base preservation weights must be non-negative")
    if (
        preserve_teacher_physics
        and teacher_weight <= 0.0
        and not has_online_physics_teacher
    ):
        raise ValueError(
            "--preserve-teacher-physics requires a positive --teacher-weight "
            "or an online teacher"
        )
    if preserve_teacher_physics and not any(weight > 0.0 for weight in weights):
        raise ValueError(
            "--preserve-teacher-physics requires at least one positive "
            "base preservation weight"
        )


def validate_teacher_source_compatibility(
    *,
    has_precomputed_teacher: bool,
    has_physics_teacher: bool,
    online_teacher_weight: float,
    allow_precomputed_teacher_with_physics_reference: bool,
) -> None:
    """Validate whether distillation and frozen physics references may coexist."""
    if online_teacher_weight > 0.0 and not has_physics_teacher:
        raise ValueError(
            "--online-teacher-weight requires --physics-teacher-checkpoint"
        )
    if online_teacher_weight > 0.0 and has_precomputed_teacher:
        raise ValueError(
            "online and precomputed distillation teachers are mutually exclusive"
        )
    dual_reference = has_precomputed_teacher and has_physics_teacher
    if dual_reference and not allow_precomputed_teacher_with_physics_reference:
        raise ValueError(
            "--physics-teacher-checkpoint and distillation teacher require "
            "--allow-precomputed-teacher-with-physics-reference"
        )
    if allow_precomputed_teacher_with_physics_reference and not dual_reference:
        raise ValueError(
            "--allow-precomputed-teacher-with-physics-reference requires both "
            "a precomputed distillation teacher and --physics-teacher-checkpoint"
        )


def frozen_physics_teacher_prediction(
    teacher: torch.nn.Module | None,
    inputs: torch.Tensor,
) -> torch.Tensor | None:
    """Run an optional online teacher without retaining its activation graph."""
    if teacher is None:
        return None
    with torch.no_grad():
        return teacher(inputs)[:, :1]


def online_teacher_inputs(
    batch: dict[str, Any],
    student_inputs: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Use an optional clean view for an online invariance teacher."""
    clean = batch.get("clean_input")
    if clean is None:
        return student_inputs
    if not isinstance(clean, torch.Tensor):
        raise TypeError("clean_input must be a tensor")
    clean = clean.to(device, non_blocking=True)
    if clean.shape != student_inputs.shape:
        raise ValueError("clean and perturbed inputs must have identical shapes")
    return clean


def online_teacher_distillation_loss(
    supervised_loss: torch.Tensor,
    prediction: torch.Tensor,
    teacher: torch.Tensor | None,
    prediction_mask: torch.Tensor,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchor a continuation to a frozen checkpoint evaluated on the same patch."""
    if not 0.0 < weight <= 1.0:
        raise ValueError("online teacher weight must be in (0, 1]")
    if teacher is None:
        raise ValueError("online teacher prediction is required")
    return combine_teacher_distillation_loss(
        supervised_loss=supervised_loss,
        prediction=prediction,
        teacher=teacher,
        prediction_mask=prediction_mask,
        teacher_weight=weight,
    )


def add_target_acf_regularization(
    loss: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    prediction_mask: torch.Tensor,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add the historical target-relative ACF objective after source anchoring."""
    if weight < 0.0:
        raise ValueError("target ACF weight must be non-negative")
    target_acf = acf_path_loss(prediction, target, prediction_mask)
    return loss + weight * target_acf, target_acf


def add_whole_body_acf_tail_regularization(
    loss: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    prediction_mask: torch.Tensor,
    weight: float,
    tail_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize the worst fixed fraction of target-relative whole-body rays."""
    if weight < 0.0:
        raise ValueError("whole-body ACF tail weight must be non-negative")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("whole-body ACF tail fraction must be in (0, 1]")
    tail_acf = acf_tail_path_loss(
        prediction,
        target,
        prediction_mask,
        tail_fraction,
    )
    return loss + weight * tail_acf, tail_acf


def add_acf_noninferiority_hinge_regularization(
    loss: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    base_prediction: torch.Tensor,
    prediction_mask: torch.Tensor,
    weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Add a strict target-relative ACF noninferiority hinge to a frozen base."""
    if weight < 0.0:
        raise ValueError("ACF noninferiority hinge weight must be non-negative")
    hinge, metrics = acf_noninferiority_hinge_loss(
        prediction,
        target,
        base_prediction,
        prediction_mask,
    )
    return loss + weight * hinge, metrics


def add_responsibility_boost_loss(
    loss: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    prediction_mask: torch.Tensor,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add an explicit CT-target penalty over the PET responsibility mask."""
    if weight < 0.0:
        raise ValueError("responsibility boost weight must be non-negative")
    responsibility_l1 = masked_l1(prediction, target, prediction_mask)
    return loss + weight * responsibility_l1, responsibility_l1


def add_partitioned_gradient_loss(
    loss: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    eval_mask: torch.Tensor,
    prediction_mask: torch.Tensor,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add a separately normalized gradient term outside the CT evaluation mask."""
    if weight < 0.0:
        raise ValueError("partitioned gradient weight must be non-negative")
    if not (
        prediction.shape
        == target.shape
        == eval_mask.shape
        == prediction_mask.shape
    ):
        raise ValueError("partitioned gradient tensors must have matching shapes")
    responsibility_complement = prediction_mask.to(prediction.dtype) * (
        eval_mask <= 0
    ).to(prediction.dtype)
    complement_gradient_l1 = masked_gradient_l1(
        prediction,
        target,
        responsibility_complement,
    )
    return loss + weight * complement_gradient_l1, complement_gradient_l1


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    world_size: int,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    accumulation_steps: int,
    tissue_aux_weight: float,
    extreme_weight: float,
    smooth_weight: float,
    projection_weight: float,
    acf_weight: float,
    tissue_bias_weight: float,
    gradient_weight: float,
    eval_only: bool = False,
    teacher_weight: float = 0.0,
    teacher_mask: str = "prediction",
    teacher_reliability_scale: float = 0.0,
    base_preservation_smooth_weight: float = 0.0,
    base_preservation_projection_weight: float = 0.0,
    base_preservation_acf_weight: float = 0.0,
    base_mu_scale: float = 0.15,
    teacher_improvement_only: bool = False,
    preserve_teacher_physics: bool = False,
    teacher_oracle_endpoint: bool = False,
    teacher_oracle_margin: float = 0.0,
    brain_mu_weight: float = 0.0,
    brain_ray_cvar_weight: float = 0.0,
    brain_ray_cvar_fraction: float = 0.25,
    brain_ray_temperature: float = 0.01,
    responsibility_boost_weight: float = 0.0,
    physics_teacher: torch.nn.Module | None = None,
    multiscale_target_weight: float = 0.0,
    multiscale_target_scales: tuple[int, ...] = (2, 4, 8),
    axial_slice_bias_weight: float = 0.0,
    axial_slice_positive_bias_weight: float = 0.0,
    air_lung_axial_positive_bias_weight: float = 0.0,
    online_teacher_weight: float = 0.0,
    target_acf_regularization_weight: float = 0.0,
    whole_body_acf_tail_weight: float = 0.0,
    whole_body_acf_tail_fraction: float = 0.1,
    acf_noninferiority_hinge_weight: float = 0.0,
    mgda_acf_noninferiority: bool = False,
    project_conflicting_acf_noninferiority: bool = False,
    teacher_oracle_selected_weight: float | None = None,
    optimizer_step_callback: Callable[[], None] | None = None,
    gradient_mask_mode: str = "eval",
) -> dict[str, float]:
    if gradient_mask_mode not in ("eval", "prediction", "partitioned"):
        raise ValueError(
            "gradient mask mode must be eval, prediction, or partitioned"
        )
    training = optimizer is not None
    model.train(training)
    if physics_teacher is not None:
        physics_teacher.eval()
    totals: dict[str, float] = defaultdict(float)
    batch_count = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, batch in enumerate(loader):
            inputs = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            eval_mask = batch["eval_mask"].to(device, non_blocking=True)
            prediction_mask = batch["prediction_mask"].to(device, non_blocking=True)
            teacher = (
                batch["teacher"].to(device, non_blocking=True)
                if teacher_weight > 0.0
                else None
            )
            cached_base_prediction = batch.get("base_prediction")
            cached_base_prediction_tensor = (
                cached_base_prediction.to(device, non_blocking=True)
                if isinstance(cached_base_prediction, torch.Tensor)
                else None
            )
            selective_base_prediction = (
                cached_base_prediction_tensor
                if cached_base_prediction_tensor is not None
                else inputs[:, :1] * base_mu_scale
            )
            brain_mask = (
                batch["brain_mask"].to(device, non_blocking=True)
                if brain_mu_weight > 0.0 or brain_ray_cvar_weight > 0.0
                else None
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(inputs)
                teacher_inputs = online_teacher_inputs(batch, inputs, device)
                online_physics_teacher = frozen_physics_teacher_prediction(
                    physics_teacher,
                    teacher_inputs,
                )
                if eval_only:
                    loss, metrics = eval_only_mu_loss(
                        output,
                        target,
                        eval_mask,
                        prediction_mask,
                    )
                elif (
                    tissue_aux_weight > 0
                    or extreme_weight > 1
                    or smooth_weight > 0
                    or projection_weight > 0
                    or acf_weight > 0
                    or tissue_bias_weight > 0
                    or gradient_weight > 0
                ):
                    loss, metrics = tissue_aware_mu_loss(
                        output,
                        target,
                        eval_mask,
                        prediction_mask,
                        tissue_aux_weight,
                        extreme_weight,
                        smooth_weight,
                        projection_weight,
                        acf_weight,
                        tissue_bias_weight,
                        gradient_weight,
                        gradient_mask=(
                            prediction_mask
                            if gradient_mask_mode == "prediction"
                            else eval_mask
                        ),
                    )
                else:
                    loss, metrics = direct_mu_loss(output, target, eval_mask, prediction_mask)
                if gradient_weight > 0.0 and gradient_mask_mode == "partitioned":
                    loss, responsibility_gradient_l1 = add_partitioned_gradient_loss(
                        loss,
                        output[:, :1],
                        target,
                        eval_mask,
                        prediction_mask,
                        gradient_weight,
                    )
                    metrics = {
                        **metrics,
                        "responsibility_gradient_mu_l1": (
                            responsibility_gradient_l1.detach()
                        ),
                        "loss": loss.detach(),
                    }
                if responsibility_boost_weight > 0.0:
                    loss, responsibility_boost_l1 = add_responsibility_boost_loss(
                        loss,
                        output[:, :1],
                        target,
                        prediction_mask,
                        responsibility_boost_weight,
                    )
                    metrics = {
                        **metrics,
                        "responsibility_boost_l1": responsibility_boost_l1.detach(),
                        "loss": loss.detach(),
                    }
                if teacher is not None:
                    supervised_loss = loss
                    distillation_mask = (
                        eval_mask
                        if teacher_mask == "eval"
                        else prediction_mask
                    )
                    loss, teacher_l1 = combine_teacher_distillation_loss(
                        supervised_loss=supervised_loss,
                        prediction=output[:, :1],
                        teacher=teacher,
                        prediction_mask=distillation_mask,
                        teacher_weight=teacher_weight,
                        target=target,
                        reliability_scale=teacher_reliability_scale,
                        base_prediction=(
                            selective_base_prediction
                            if (
                                teacher_improvement_only
                                or teacher_oracle_endpoint
                            )
                            else None
                        ),
                        improvement_only=teacher_improvement_only,
                        oracle_endpoint=teacher_oracle_endpoint,
                        oracle_margin=teacher_oracle_margin,
                        oracle_selected_weight=teacher_oracle_selected_weight,
                    )
                    metrics = {
                        **metrics,
                        "loss": loss.detach(),
                        "supervised_loss": supervised_loss.detach(),
                        "teacher_mu_l1": teacher_l1.detach(),
                    }
                    if teacher_oracle_endpoint:
                        with torch.no_grad():
                            teacher_better = (
                                torch.abs(teacher - target)
                                + teacher_oracle_margin
                                < torch.abs(selective_base_prediction - target)
                            )
                            active = distillation_mask.to(output.dtype)
                            selected = teacher_better.to(output.dtype) * active
                            metrics["teacher_oracle_fraction"] = (
                                selected.sum() / active.sum().clamp_min(1.0)
                            ).detach()
                if online_teacher_weight > 0.0:
                    supervised_loss = loss
                    loss, online_teacher_l1 = online_teacher_distillation_loss(
                        supervised_loss,
                        output[:, :1],
                        online_physics_teacher,
                        prediction_mask,
                        online_teacher_weight,
                    )
                    metrics = {
                        **metrics,
                        "loss": loss.detach(),
                        "supervised_loss": supervised_loss.detach(),
                        "online_teacher_mu_l1": online_teacher_l1.detach(),
                    }
                if target_acf_regularization_weight > 0.0:
                    loss, target_acf = add_target_acf_regularization(
                        loss,
                        output[:, :1],
                        target,
                        prediction_mask,
                        target_acf_regularization_weight,
                    )
                    metrics = {
                        **metrics,
                        "target_acf_path_relative_error": target_acf.detach(),
                        "loss": loss.detach(),
                    }
                if whole_body_acf_tail_weight > 0.0:
                    loss, whole_body_acf_tail = (
                        add_whole_body_acf_tail_regularization(
                            loss,
                            output[:, :1],
                            target,
                            prediction_mask,
                            whole_body_acf_tail_weight,
                            whole_body_acf_tail_fraction,
                        )
                    )
                    metrics = {
                        **metrics,
                        "whole_body_acf_tail_relative_error": (
                            whole_body_acf_tail.detach()
                        ),
                        "loss": loss.detach(),
                    }
                if acf_noninferiority_hinge_weight > 0.0:
                    if cached_base_prediction_tensor is None:
                        raise ValueError(
                            "ACF noninferiority hinge requires a cached base prediction"
                        )
                    loss, noninferiority_metrics = (
                        add_acf_noninferiority_hinge_regularization(
                            loss,
                            output[:, :1],
                            target,
                            cached_base_prediction_tensor,
                            prediction_mask,
                            acf_noninferiority_hinge_weight,
                        )
                    )
                    metrics = {
                        **metrics,
                        **noninferiority_metrics,
                        "loss": loss.detach(),
                    }
                gradient_constraint = None
                if (
                    mgda_acf_noninferiority
                    or project_conflicting_acf_noninferiority
                ):
                    if cached_base_prediction_tensor is None:
                        raise ValueError(
                            "gradient-constrained ACF noninferiority requires a "
                            "cached base prediction"
                        )
                    gradient_constraint, noninferiority_metrics = (
                        acf_noninferiority_hinge_loss(
                            output[:, :1],
                            target,
                            cached_base_prediction_tensor,
                            prediction_mask,
                        )
                    )
                    metrics = {
                        **metrics,
                        **noninferiority_metrics,
                    }
                if any(
                    weight > 0.0
                    for weight in (
                        base_preservation_smooth_weight,
                        base_preservation_projection_weight,
                        base_preservation_acf_weight,
                    )
                ):
                    preservation_reference = physics_preservation_reference(
                        inputs,
                        (
                            online_physics_teacher
                            if online_physics_teacher is not None
                            else teacher
                        ),
                        base_mu_scale,
                        preserve_teacher_physics,
                    )
                    preservation_loss, preservation_metrics = (
                        base_physics_preservation_loss(
                            output[:, :1],
                            preservation_reference,
                            prediction_mask,
                            base_preservation_smooth_weight,
                            base_preservation_projection_weight,
                            base_preservation_acf_weight,
                        )
                    )
                    if preserve_teacher_physics:
                        preservation_metrics = {
                            key.replace("base_", "teacher_", 1): value
                            for key, value in preservation_metrics.items()
                        }
                    loss = loss + preservation_loss
                    metrics = {
                        **metrics,
                        **preservation_metrics,
                        "loss": loss.detach(),
                    }
                if multiscale_target_weight > 0.0:
                    multiscale_l1 = multiscale_target_l1(
                        output[:, :1],
                        target,
                        eval_mask,
                        multiscale_target_scales,
                    )
                    loss = loss + multiscale_target_weight * multiscale_l1
                    metrics = {
                        **metrics,
                        "multiscale_target_l1": multiscale_l1.detach(),
                        "loss": loss.detach(),
                    }
                if axial_slice_bias_weight > 0.0:
                    axial_bias = axial_slice_signed_bias_loss(
                        output[:, :1],
                        target,
                        prediction_mask,
                    )
                    loss = loss + axial_slice_bias_weight * axial_bias
                    metrics = {
                        **metrics,
                        "axial_slice_signed_bias": axial_bias.detach(),
                        "loss": loss.detach(),
                    }
                if axial_slice_positive_bias_weight > 0.0:
                    axial_positive_bias = axial_slice_positive_bias_loss(
                        output[:, :1],
                        target,
                        prediction_mask,
                    )
                    loss = loss + axial_slice_positive_bias_weight * axial_positive_bias
                    metrics = {
                        **metrics,
                        "axial_slice_positive_bias": axial_positive_bias.detach(),
                        "loss": loss.detach(),
                    }
                if air_lung_axial_positive_bias_weight > 0.0:
                    air_lung_positive_bias = air_lung_axial_positive_bias_loss(
                        output[:, :1],
                        target,
                        prediction_mask,
                    )
                    loss = (
                        loss
                        + air_lung_axial_positive_bias_weight
                        * air_lung_positive_bias
                    )
                    metrics = {
                        **metrics,
                        "air_lung_axial_positive_bias": air_lung_positive_bias.detach(),
                        "loss": loss.detach(),
                    }
                if brain_mask is not None:
                    prediction = output[:, :1]
                    brain_active = eval_mask.to(prediction.dtype) * (
                        brain_mask > 0.5
                    ).to(prediction.dtype)
                    brain_l1 = prediction.new_zeros(())
                    brain_cvar = prediction.new_zeros((), dtype=torch.float32)
                    brain_hard = prediction.new_zeros((), dtype=torch.float32)
                    brain_acf = prediction.new_zeros((), dtype=torch.float32)
                    brain_path = prediction.new_zeros((), dtype=torch.float32)
                    if torch.any(brain_active > 0):
                        brain_l1 = masked_l1(prediction, target, brain_active)
                        if brain_ray_cvar_weight > 0.0:
                            brain_cvar, brain_hard, brain_acf, brain_path = (
                                brain_ray_cvar_exceedance(
                                    prediction,
                                    target,
                                    prediction_mask,
                                    brain_mask,
                                    brain_ray_temperature,
                                    brain_ray_cvar_fraction,
                                )
                            )
                    loss = (
                        loss
                        + brain_mu_weight * brain_l1
                        + brain_ray_cvar_weight * brain_cvar
                    )
                    metrics = {
                        **metrics,
                        "brain_mu_l1": brain_l1.detach(),
                        "brain_ray_cvar_exceedance": brain_cvar.detach(),
                        "brain_ray_outlier_hard": brain_hard.detach(),
                        "brain_ray_acf_relative_error": brain_acf.detach(),
                        "brain_ray_path_abs_error": brain_path.detach(),
                        "loss": loss.detach(),
                    }
                scaled_loss = loss / accumulation_steps
            if training:
                if (
                    mgda_acf_noninferiority
                    or project_conflicting_acf_noninferiority
                ):
                    if world_size != 1:
                        raise ValueError(
                            "gradient-constrained ACF training requires one process"
                        )
                    assert gradient_constraint is not None
                    parameters = [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ]
                    if mgda_acf_noninferiority:
                        geometry = accumulate_two_objective_gradients(
                            loss,
                            gradient_constraint,
                            parameters,
                            accumulation_steps,
                        )
                        prefix = "mgda"
                    else:
                        geometry = accumulate_primary_priority_gradients(
                            loss,
                            gradient_constraint,
                            parameters,
                            accumulation_steps,
                        )
                        prefix = "acf_projection"
                    metrics = {
                        **metrics,
                        **{
                            f"{prefix}_{key}": finite_metric_tensor(value, output)
                            for key, value in geometry.items()
                        },
                        "loss": loss.detach(),
                    }
                    ready = (batch_index + 1) % accumulation_steps == 0
                    if ready or batch_index + 1 == len(loader):
                        torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
                        optimizer.step()
                        if optimizer_step_callback is not None:
                            optimizer_step_callback()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    assert scaler is not None
                    scaler.scale(scaled_loss).backward()
                    if (
                        (batch_index + 1) % accumulation_steps == 0
                        or batch_index + 1 == len(loader)
                    ):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=5.0
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        if optimizer_step_callback is not None:
                            optimizer_step_callback()
                        optimizer.zero_grad(set_to_none=True)
            for key, value in metrics.items():
                totals[key] += float(value.float().cpu())
            batch_count += 1
    metrics = {
        key: distributed_mean(torch.tensor(total / max(1, batch_count), device=device), world_size)
        for key, total in totals.items()
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    split = parser.add_mutually_exclusive_group(required=True)
    split.add_argument("--fold", type=int)
    split.add_argument(
        "--train-all",
        action="store_true",
        help=(
            "Train on every assigned subject. The same subjects form a "
            "monitoring-only validation loader; use a pre-registered final epoch."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(160, 160, 160))
    parser.add_argument(
        "--center-transverse-on-prediction-bbox",
        action="store_true",
        help=(
            "Fix the first two patch coordinates at the prediction-bbox "
            "center and require the patch to cover that full transverse bbox."
        ),
    )
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--train-samples-per-subject", type=int, default=4)
    parser.add_argument("--val-samples-per-subject", type=int, default=2)
    parser.add_argument(
        "--subject-sampling-weights",
        type=Path,
        help=(
            "JSON report with folds.<fold-or-all>.weights. Enables deterministic "
            "weighted sampling with replacement for the training loader only."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--grad-scaler-init-scale",
        type=float,
        default=65536.0,
        help="Initial CUDA AMP loss scale; the historical default is 65536.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="Save an EMA checkpoint updated after every optimizer step; zero disables it.",
    )
    parser.add_argument(
        "--lookahead-sync-steps",
        type=int,
        default=0,
        help="Synchronize Lookahead slow weights every k optimizer steps; zero disables it.",
    )
    parser.add_argument(
        "--lookahead-alpha",
        type=float,
        default=0.5,
        help="Lookahead slow-weight interpolation coefficient.",
    )
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--use-topogram", action="store_true")
    parser.add_argument("--projected-topogram", action="store_true")
    parser.add_argument("--use-mri", action="store_true")
    parser.add_argument(
        "--mri-translation-probability",
        type=float,
        default=0.0,
        help="Probability of a shared integer translation of both MRI phases.",
    )
    parser.add_argument(
        "--mri-translation-max-voxels",
        type=int,
        nargs=3,
        default=(0, 0, 0),
        metavar=("X", "Y", "Z"),
        help="Maximum absolute training-only MRI translation in voxels.",
    )
    parser.add_argument(
        "--nac-absolute-gain-log-std",
        type=float,
        default=0.0,
        help=(
            "Apply a zero-median log-normal gain to only the absolute NAC "
            "channel during training. The relative NAC and topogram channels "
            "remain unchanged."
        ),
    )
    parser.add_argument(
        "--clean-online-teacher-input",
        action="store_true",
        help=(
            "Evaluate the frozen online teacher on the clean input while the "
            "student receives the augmented view."
        ),
    )
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--head-output-adapter", action="store_true")
    parser.add_argument("--head-adapter-local-context", action="store_true")
    parser.add_argument("--head-adapter-hidden-channels", type=int, default=8)
    parser.add_argument(
        "--freeze-head-output-adapter-only",
        action="store_true",
    )
    parser.add_argument("--head-adapter-transition-start", type=int, default=370)
    parser.add_argument("--head-adapter-transition-end", type=int, default=390)
    parser.add_argument("--head-adapter-volume-depth", type=int, default=531)
    parser.add_argument("--head-adapter-correction-scale", type=float, default=0.03)
    parser.add_argument("--full-resolution-mri-adapter", action="store_true")
    parser.add_argument(
        "--freeze-full-resolution-mri-adapter-only",
        action="store_true",
    )
    parser.add_argument("--coordinate-mode", choices=("none", "z", "xyz"), default="none")
    parser.add_argument(
        "--axial-training-region",
        type=int,
        nargs=2,
        metavar=("Z_START", "Z_END"),
        help=(
            "Restrict patch centers and supervised masks to the half-open "
            "axial interval [Z_START, Z_END). Intended for a specialist that "
            "will only be used in the same pre-registered interval."
        ),
    )
    parser.add_argument("--tissue-aux-weight", type=float, default=0.0)
    parser.add_argument("--extreme-weight", type=float, default=1.0)
    parser.add_argument("--smooth-weight", type=float, default=0.0)
    parser.add_argument("--projection-weight", type=float, default=0.0)
    parser.add_argument("--acf-weight", type=float, default=0.0)
    parser.add_argument("--tissue-bias-weight", type=float, default=0.0)
    parser.add_argument("--gradient-weight", type=float, default=0.0)
    parser.add_argument(
        "--gradient-mask",
        choices=("eval", "prediction", "partitioned"),
        default="eval",
        help="Support and normalization used by the local gradient-error term.",
    )
    parser.add_argument("--responsibility-boost-weight", type=float, default=0.0)
    parser.add_argument("--multiscale-target-weight", type=float, default=0.0)
    parser.add_argument("--axial-slice-bias-weight", type=float, default=0.0)
    parser.add_argument("--axial-slice-positive-bias-weight", type=float, default=0.0)
    parser.add_argument("--air-lung-axial-positive-bias-weight", type=float, default=0.0)
    parser.add_argument(
        "--multiscale-target-scales",
        type=int,
        nargs="+",
        default=(2, 4, 8),
    )
    parser.add_argument("--brain-mu-weight", type=float, default=0.0)
    parser.add_argument("--brain-ray-cvar-weight", type=float, default=0.0)
    parser.add_argument("--brain-ray-cvar-fraction", type=float, default=0.25)
    parser.add_argument("--brain-ray-temperature", type=float, default=0.01)
    parser.add_argument("--brain-center-probability", type=float, default=0.0)
    parser.add_argument(
        "--responsibility-center-probability",
        type=float,
        default=0.0,
        help=(
            "Conditional probability that a non-brain patch is centered in "
            "prediction_mask minus eval_mask."
        ),
    )
    parser.add_argument("--teacher-prediction-dir", type=Path)
    parser.add_argument("--base-prediction-dir", type=Path)
    parser.add_argument("--physics-teacher-checkpoint", type=Path)
    parser.add_argument(
        "--allow-precomputed-teacher-with-physics-reference",
        action="store_true",
        help=(
            "Use a precomputed distillation target while a separate frozen "
            "checkpoint supplies only physics-preservation references."
        ),
    )
    parser.add_argument("--online-teacher-weight", type=float, default=0.0)
    parser.add_argument(
        "--target-acf-regularization-weight", type=float, default=0.0
    )
    parser.add_argument("--whole-body-acf-tail-weight", type=float, default=0.0)
    parser.add_argument("--whole-body-acf-tail-fraction", type=float, default=0.1)
    parser.add_argument(
        "--acf-noninferiority-hinge-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--mgda-acf-noninferiority",
        action="store_true",
        help=(
            "Combine the primary objective and strict base-relative ACF hinge "
            "with a per-batch minimum-norm gradient direction."
        ),
    )
    parser.add_argument(
        "--project-conflicting-acf-noninferiority",
        action="store_true",
        help=(
            "Keep the primary gradient when it is ACF-nonincreasing and apply "
            "the minimum half-space projection only when it conflicts."
        ),
    )
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument(
        "--teacher-mask",
        choices=("prediction", "eval"),
        default="prediction",
    )
    parser.add_argument(
        "--teacher-reliability-scale",
        type=float,
        default=0.0,
        help=(
            "Exponentially downweight teacher voxels by teacher-to-target "
            "absolute error; zero disables reliability weighting."
        ),
    )
    parser.add_argument(
        "--teacher-improvement-only",
        action="store_true",
        help="Distill only where the teacher has lower target error than the base.",
    )
    parser.add_argument(
        "--teacher-oracle-endpoint",
        action="store_true",
        help=(
            "Use the teacher where it clears the oracle margin and the base "
            "prediction everywhere else as the full distillation endpoint."
        ),
    )
    parser.add_argument("--teacher-oracle-margin", type=float, default=0.0)
    parser.add_argument("--teacher-oracle-selected-weight", type=float)
    parser.add_argument("--base-preservation-smooth-weight", type=float, default=0.0)
    parser.add_argument("--base-preservation-projection-weight", type=float, default=0.0)
    parser.add_argument("--base-preservation-acf-weight", type=float, default=0.0)
    parser.add_argument(
        "--preserve-teacher-physics",
        action="store_true",
        help="Apply preservation losses against the frozen teacher instead of the input base.",
    )
    return parser.parse_args()


def load_subject_sampling_weights(
    path: Path,
    section: str,
    subjects: list[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sections = payload.get("folds")
    if not isinstance(sections, dict) or section not in sections:
        raise ValueError(f"sampling-weight section is missing: {section}")
    raw = sections[section].get("weights")
    if not isinstance(raw, dict):
        raise ValueError("sampling weights must be a subject mapping")
    weights = {str(subject): float(value) for subject, value in raw.items()}
    if set(weights) != set(subjects):
        missing = sorted(set(subjects) - set(weights))
        extra = sorted(set(weights) - set(subjects))
        raise ValueError(
            f"sampling-weight subjects differ: missing={missing}, extra={extra}"
        )
    values = np.asarray([weights[subject] for subject in subjects], dtype=np.float64)
    if not np.isfinite(values).all() or np.min(values) <= 0.0:
        raise ValueError("sampling weights must be finite and positive")
    return weights, {
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "section": section,
        "subject_count": len(subjects),
        "minimum": float(np.min(values)),
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
        "weights": weights,
        "replacement": True,
    }


def load_expanded_model(model: torch.nn.Module, checkpoint_path: Path) -> None:
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]
    target = model.state_dict()
    loaded = 0
    expanded = []
    for key, source_value in source.items():
        if key not in target:
            continue
        target_value = target[key]
        if source_value.shape == target_value.shape:
            target_value.copy_(source_value)
            loaded += 1
            continue
        if key in ("stem.body.0.weight", "stem.skip.weight") and source_value.shape[1] < target_value.shape[1]:
            target_value.zero_()
            target_value[:, : source_value.shape[1]].copy_(source_value)
            expanded.append(key)
            continue
        if key in ("output.weight", "output.bias") and source_value.shape[0] < target_value.shape[0]:
            target_value[: source_value.shape[0]].copy_(source_value)
            expanded.append(key)
            continue
        raise ValueError(
            f"cannot expand checkpoint tensor {key}: {tuple(source_value.shape)} -> {tuple(target_value.shape)}"
        )
    model.load_state_dict(target)
    print(
        f"Initialized model from {checkpoint_path}: {loaded} exact tensors, "
        f"expanded {', '.join(expanded)}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.resume is not None and args.initialize_from is not None:
        raise ValueError("--resume and --initialize-from are mutually exclusive")
    if args.nac_absolute_gain_log_std < 0.0:
        raise ValueError("--nac-absolute-gain-log-std must be non-negative")
    if not math.isfinite(args.grad_scaler_init_scale) or args.grad_scaler_init_scale <= 0.0:
        raise ValueError("--grad-scaler-init-scale must be finite and positive")
    if args.clean_online_teacher_input and (
        args.nac_absolute_gain_log_std <= 0.0
        or args.physics_teacher_checkpoint is None
        or args.online_teacher_weight <= 0.0
    ):
        raise ValueError(
            "--clean-online-teacher-input requires a positive NAC gain, "
            "an online teacher checkpoint, and positive online teacher weight"
        )
    if args.tissue_aux_weight < 0:
        raise ValueError("--tissue-aux-weight must be non-negative")
    if args.extreme_weight < 1:
        raise ValueError("--extreme-weight must be at least 1")
    if min(
        args.smooth_weight,
        args.projection_weight,
        args.acf_weight,
        args.tissue_bias_weight,
        args.gradient_weight,
        args.responsibility_boost_weight,
        args.multiscale_target_weight,
        args.axial_slice_bias_weight,
        args.axial_slice_positive_bias_weight,
        args.air_lung_axial_positive_bias_weight,
        args.brain_mu_weight,
        args.brain_ray_cvar_weight,
        args.target_acf_regularization_weight,
        args.whole_body_acf_tail_weight,
        args.acf_noninferiority_hinge_weight,
    ) < 0:
        raise ValueError("surrogate loss weights must be non-negative")
    multiscale_target_scales = tuple(args.multiscale_target_scales)
    if (
        not multiscale_target_scales
        or any(scale <= 1 for scale in multiscale_target_scales)
        or tuple(sorted(set(multiscale_target_scales))) != multiscale_target_scales
    ):
        raise ValueError(
            "--multiscale-target-scales must be unique, increasing integers above one"
        )
    if not 0.0 < args.brain_ray_cvar_fraction <= 1.0:
        raise ValueError("--brain-ray-cvar-fraction must be in (0, 1]")
    if not 0.0 < args.whole_body_acf_tail_fraction <= 1.0:
        raise ValueError("--whole-body-acf-tail-fraction must be in (0, 1]")
    if args.brain_ray_temperature <= 0.0:
        raise ValueError("--brain-ray-temperature must be positive")
    if not 0.0 <= args.brain_center_probability <= 1.0:
        raise ValueError("--brain-center-probability must be between zero and one")
    if not 0.0 <= args.responsibility_center_probability <= 1.0:
        raise ValueError(
            "--responsibility-center-probability must be between zero and one"
        )
    brain_objective = args.brain_mu_weight > 0.0 or args.brain_ray_cvar_weight > 0.0
    if brain_objective != (args.brain_center_probability > 0.0):
        raise ValueError(
            "brain losses and --brain-center-probability must be enabled together"
        )
    if not 0.0 <= args.teacher_weight <= 1.0:
        raise ValueError("--teacher-weight must be between zero and one")
    if not 0.0 <= args.online_teacher_weight <= 1.0:
        raise ValueError("--online-teacher-weight must be between zero and one")
    if args.teacher_reliability_scale < 0.0:
        raise ValueError("--teacher-reliability-scale must be non-negative")
    if (args.teacher_prediction_dir is None) != (args.teacher_weight == 0.0):
        raise ValueError(
            "--teacher-prediction-dir and a positive --teacher-weight "
            "must be provided together"
        )
    if args.teacher_reliability_scale > 0.0 and args.teacher_weight == 0.0:
        raise ValueError(
            "--teacher-reliability-scale requires a positive --teacher-weight"
        )
    if args.teacher_improvement_only and args.teacher_weight == 0.0:
        raise ValueError(
            "--teacher-improvement-only requires a positive --teacher-weight"
        )
    if args.teacher_oracle_margin < 0.0:
        raise ValueError("--teacher-oracle-margin must be non-negative")
    if args.teacher_oracle_endpoint and args.teacher_weight == 0.0:
        raise ValueError(
            "--teacher-oracle-endpoint requires a positive --teacher-weight"
        )
    if args.teacher_oracle_endpoint and args.teacher_improvement_only:
        raise ValueError(
            "--teacher-oracle-endpoint and --teacher-improvement-only are "
            "mutually exclusive"
        )
    if args.teacher_oracle_margin > 0.0 and not args.teacher_oracle_endpoint:
        raise ValueError(
            "--teacher-oracle-margin requires --teacher-oracle-endpoint"
        )
    if args.teacher_oracle_endpoint and args.teacher_reliability_scale > 0.0:
        raise ValueError(
            "--teacher-reliability-scale is not supported with "
            "--teacher-oracle-endpoint"
        )
    if args.teacher_oracle_selected_weight is not None and not (
        args.teacher_oracle_endpoint
        and 0.0 < args.teacher_oracle_selected_weight < 1.0
    ):
        raise ValueError(
            "--teacher-oracle-selected-weight must be in (0, 1) and requires "
            "--teacher-oracle-endpoint"
        )
    selective_teacher = args.teacher_improvement_only or args.teacher_oracle_endpoint
    requires_base_prediction = (
        selective_teacher
        or args.acf_noninferiority_hinge_weight > 0.0
        or args.mgda_acf_noninferiority
        or args.project_conflicting_acf_noninferiority
    )
    if requires_base_prediction != (args.base_prediction_dir is not None):
        raise ValueError(
            "selective distillation or ACF noninferiority hinge and "
            "--base-prediction-dir must be provided together"
        )
    if sum(
        int(value)
        for value in (
            args.acf_noninferiority_hinge_weight > 0.0,
            args.mgda_acf_noninferiority,
            args.project_conflicting_acf_noninferiority,
        )
    ) > 1:
        raise ValueError(
            "scalar, MGDA, and primary-priority ACF noninferiority modes are "
            "mutually exclusive"
        )
    validate_teacher_source_compatibility(
        has_precomputed_teacher=args.teacher_prediction_dir is not None,
        has_physics_teacher=args.physics_teacher_checkpoint is not None,
        online_teacher_weight=args.online_teacher_weight,
        allow_precomputed_teacher_with_physics_reference=(
            args.allow_precomputed_teacher_with_physics_reference
        ),
    )
    if (
        args.physics_teacher_checkpoint is not None
        and not args.preserve_teacher_physics
    ):
        raise ValueError(
            "--physics-teacher-checkpoint requires --preserve-teacher-physics"
        )
    validate_physics_preservation_settings(
        args.teacher_weight,
        args.base_preservation_smooth_weight,
        args.base_preservation_projection_weight,
        args.base_preservation_acf_weight,
        args.preserve_teacher_physics,
        has_online_physics_teacher=args.physics_teacher_checkpoint is not None,
    )
    if args.use_topogram and args.projected_topogram:
        raise ValueError("choose either --use-topogram or --projected-topogram")
    if args.use_mri and not args.projected_topogram:
        raise ValueError("--use-mri currently requires --projected-topogram")
    if not 0.0 <= args.mri_translation_probability <= 1.0:
        raise ValueError(
            "--mri-translation-probability must be between zero and one"
        )
    if len(args.mri_translation_max_voxels) != 3 or min(
        args.mri_translation_max_voxels
    ) < 0:
        raise ValueError(
            "--mri-translation-max-voxels must contain three nonnegative values"
        )
    if args.mri_translation_probability > 0.0 and not args.use_mri:
        raise ValueError("MRI translation augmentation requires --use-mri")
    if args.full_resolution_mri_adapter and not args.use_mri:
        raise ValueError("--full-resolution-mri-adapter requires --use-mri")
    if args.freeze_backbone and (not args.use_mri or args.initialize_from is None):
        raise ValueError("--freeze-backbone requires --use-mri and --initialize-from")
    if args.freeze_backbone and args.freeze_full_resolution_mri_adapter_only:
        raise ValueError("choose only one parameter-freezing mode")
    if args.head_output_adapter != args.freeze_head_output_adapter_only:
        raise ValueError(
            "--head-output-adapter and --freeze-head-output-adapter-only "
            "must be enabled together"
        )
    if args.head_adapter_local_context and not args.head_output_adapter:
        raise ValueError(
            "--head-adapter-local-context requires --head-output-adapter"
        )
    if args.head_adapter_hidden_channels <= 0:
        raise ValueError("--head-adapter-hidden-channels must be positive")
    if args.head_output_adapter and (
        not args.projected_topogram
        or args.use_mri
        or args.coordinate_mode != "z"
        or args.initialize_from is None
    ):
        raise ValueError(
            "head output adaptation requires projected topogram, global z, "
            "no MRI branch, and an inherited checkpoint"
        )
    if args.head_output_adapter and args.physics_teacher_checkpoint is not None:
        raise ValueError("head output adaptation does not accept an online teacher")
    if sum(
        int(value)
        for value in (
            args.freeze_backbone,
            args.freeze_full_resolution_mri_adapter_only,
            args.freeze_head_output_adapter_only,
        )
    ) > 1:
        raise ValueError("choose only one parameter-freezing mode")
    if args.freeze_full_resolution_mri_adapter_only and (
        not args.full_resolution_mri_adapter or args.initialize_from is None
    ):
        raise ValueError(
            "--freeze-full-resolution-mri-adapter-only requires "
            "--full-resolution-mri-adapter and --initialize-from"
        )
    if args.axial_training_region is not None:
        start, end = args.axial_training_region
        if not 0 <= start < end:
            raise ValueError(
                "--axial-training-region must satisfy 0 <= Z_START < Z_END"
            )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if (
        args.mgda_acf_noninferiority
        or args.project_conflicting_acf_noninferiority
    ) and world_size != 1:
        raise ValueError("gradient-constrained ACF training requires WORLD_SIZE=1")
    if args.subject_sampling_weights is not None and world_size != 1:
        raise ValueError("subject-weighted sampling requires WORLD_SIZE=1")
    if world_size > 1:
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    train_subjects, val_subjects = resolve_training_subjects(
        args.folds,
        fold=args.fold,
        train_all=args.train_all,
    )
    subject_sampling_weights = None
    subject_sampling_audit = None
    if args.subject_sampling_weights is not None:
        sampling_section = "all" if args.train_all else str(args.fold)
        subject_sampling_weights, subject_sampling_audit = (
            load_subject_sampling_weights(
                args.subject_sampling_weights,
                sampling_section,
                train_subjects,
            )
        )
        subject_sampling_audit["sampler_seed"] = args.seed
    train_dataset = MuPatchDataset(
        args.cache_dir,
        train_subjects,
        args.patch_size,
        args.train_samples_per_subject,
        augment=True,
        use_topogram=args.use_topogram or args.projected_topogram,
        use_mri=args.use_mri,
        coordinate_mode=args.coordinate_mode,
        teacher_prediction_dir=args.teacher_prediction_dir,
        base_prediction_dir=args.base_prediction_dir,
        axial_training_region=args.axial_training_region,
        brain_mask_filename="brain_mask.npy" if brain_objective else None,
        brain_center_probability=args.brain_center_probability,
        responsibility_center_probability=args.responsibility_center_probability,
        center_transverse_on_prediction_bbox=(
            args.center_transverse_on_prediction_bbox
        ),
        nac_absolute_gain_log_std=args.nac_absolute_gain_log_std,
        mri_translation_probability=args.mri_translation_probability,
        mri_translation_max_voxels=args.mri_translation_max_voxels,
        return_clean_input=args.clean_online_teacher_input,
    )
    val_dataset = MuPatchDataset(
        args.cache_dir,
        val_subjects,
        args.patch_size,
        args.val_samples_per_subject,
        augment=False,
        use_topogram=args.use_topogram or args.projected_topogram,
        use_mri=args.use_mri,
        coordinate_mode=args.coordinate_mode,
        teacher_prediction_dir=args.teacher_prediction_dir,
        base_prediction_dir=args.base_prediction_dir,
        axial_training_region=args.axial_training_region,
        brain_mask_filename="brain_mask.npy" if brain_objective else None,
        brain_center_probability=1.0 if brain_objective else 0.0,
        responsibility_center_probability=0.0,
        center_transverse_on_prediction_bbox=(
            args.center_transverse_on_prediction_bbox
        ),
    )
    distributed_train_sampler = (
        DistributedSampler(train_dataset, shuffle=True, seed=args.seed)
        if world_size > 1
        else None
    )
    if subject_sampling_weights is not None:
        sample_weights = [
            subject_sampling_weights[
                train_subjects[index % len(train_subjects)]
            ]
            for index in range(len(train_dataset))
        ]
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(args.seed)
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=sampler_generator,
        )
    else:
        train_sampler = distributed_train_sampler
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=train_sampler, shuffle=train_sampler is None, **loader_kwargs)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, shuffle=False, **loader_kwargs)

    output_channels = 5 if args.tissue_aux_weight > 0 else 1
    if args.projected_topogram:
        coordinate_channels = {"none": 0, "z": 1, "xyz": 3}[args.coordinate_mode]
        if args.head_output_adapter:
            model_class = (
                LocalContextHeadOutputAdapterProjectionConditionedMuUNet
                if args.head_adapter_local_context
                else HeadOutputAdapterProjectionConditionedMuUNet
            )
            model_kwargs = {}
            if args.head_adapter_local_context:
                model_kwargs["hidden_channels"] = args.head_adapter_hidden_channels
            model = model_class(
                base_channels=args.base_channels,
                output_channels=output_channels,
                transition_start_slice=args.head_adapter_transition_start,
                transition_end_slice=args.head_adapter_transition_end,
                volume_depth=args.head_adapter_volume_depth,
                correction_scale=args.head_adapter_correction_scale,
                **model_kwargs,
            ).to(device)
        elif args.full_resolution_mri_adapter:
            model_class = FullResolutionMRIProjectionConditionedMuUNet
            model = model_class(
                base_channels=args.base_channels,
                output_channels=output_channels,
                coordinate_channels=coordinate_channels,
            ).to(device)
        else:
            model_class = (
                MRIProjectionConditionedMuUNet
                if args.use_mri
                else ProjectionConditionedMuUNet
            )
            model = model_class(
                base_channels=args.base_channels,
                output_channels=output_channels,
                coordinate_channels=coordinate_channels,
            ).to(device)
    else:
        coordinate_channels = {"none": 0, "z": 1, "xyz": 3}[args.coordinate_mode]
        input_channels = (3 if args.use_topogram else 2) + coordinate_channels
        model = DirectMuUNet(
            in_channels=input_channels,
            base_channels=args.base_channels,
            output_channels=output_channels,
        ).to(device)
    if args.initialize_from is not None:
        load_expanded_model(model, args.initialize_from)
    physics_teacher: torch.nn.Module | None = None
    if args.physics_teacher_checkpoint is not None:
        if args.projected_topogram:
            teacher_class = (
                MRIProjectionConditionedMuUNet
                if args.use_mri
                else ProjectionConditionedMuUNet
            )
            physics_teacher = teacher_class(
                base_channels=args.base_channels,
                output_channels=output_channels,
                coordinate_channels=coordinate_channels,
            ).to(device)
        else:
            physics_teacher = DirectMuUNet(
                in_channels=input_channels,
                base_channels=args.base_channels,
                output_channels=output_channels,
            ).to(device)
        load_expanded_model(physics_teacher, args.physics_teacher_checkpoint)
        physics_teacher.eval()
        for parameter in physics_teacher.parameters():
            parameter.requires_grad_(False)
    if args.freeze_head_output_adapter_only:
        trainable_count = freeze_for_head_output_adapter(model)
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        print(
            f"Frozen inherited network; training {trainable_count:,} "
            "head output adapter parameters",
            flush=True,
        )
    elif args.freeze_full_resolution_mri_adapter_only:
        adapter_prefixes = ("mri_fullres_stem.", "mri_fullres_adapter.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(adapter_prefixes))
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        print(
            "Frozen inherited network; training "
            f"{sum(value.numel() for value in trainable_parameters):,} "
            "full-resolution MRI adapter parameters",
            flush=True,
        )
    elif args.freeze_backbone:
        mri_prefixes = ("mri_stem.", "mri_down.", "mri_adapters.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(mri_prefixes))
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        print(
            f"Frozen backbone; training {sum(value.numel() for value in trainable_parameters):,} "
            "MRI-branch parameters",
            flush=True,
        )
    else:
        trainable_parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=args.grad_scaler_init_scale,
    )

    start_epoch = 0
    best_val = float("inf")
    best_objective = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(
            checkpoint.get("best_val", checkpoint.get("val_metrics", {}).get("eval_mu_l1", float("inf")))
        )
        best_objective = float(
            checkpoint.get("best_objective", checkpoint.get("val_metrics", {}).get("loss", float("inf")))
        )
        if start_epoch >= args.epochs:
            raise ValueError(
                f"checkpoint already reached epoch {start_epoch}; --epochs={args.epochs} must be larger"
            )
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must lie in [0, 1)")
    if args.lookahead_sync_steps < 0:
        raise ValueError("--lookahead-sync-steps must be non-negative")
    if not 0.0 < args.lookahead_alpha <= 1.0:
        raise ValueError("--lookahead-alpha must lie in (0, 1]")
    ema = (
        ExponentialMovingAverage(model, args.ema_decay)
        if args.ema_decay > 0.0
        else None
    )
    if ema is not None and args.resume is not None:
        if "ema_model" not in checkpoint or "ema_updates" not in checkpoint:
            raise ValueError("resuming EMA training requires EMA state in the checkpoint")
        ema.load_state_dict(checkpoint["ema_model"], int(checkpoint["ema_updates"]))
    lookahead = (
        LookaheadSynchronizer(
            model,
            sync_steps=args.lookahead_sync_steps,
            alpha=args.lookahead_alpha,
        )
        if args.lookahead_sync_steps > 0
        else None
    )
    if lookahead is not None and args.resume is not None:
        required = (
            "lookahead_slow_parameters",
            "lookahead_updates",
            "lookahead_syncs",
        )
        if any(name not in checkpoint for name in required):
            raise ValueError("resuming Lookahead training requires slow-weight state")
        lookahead.load_state_dict(
            checkpoint["lookahead_slow_parameters"],
            int(checkpoint["lookahead_updates"]),
            int(checkpoint["lookahead_syncs"]),
        )
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    def update_optimizer_averages() -> None:
        raw = model.module if isinstance(model, DistributedDataParallel) else model
        if lookahead is not None:
            lookahead.update(raw)
        if ema is not None:
            ema.update(raw)

    output_dir = args.output_dir.resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        config = vars(args).copy()
        config.update(
            world_size=world_size,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            subject_sampling=subject_sampling_audit,
        )
        for key, value in list(config.items()):
            if isinstance(value, Path):
                config[key] = str(value)
        (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for epoch in range(start_epoch, args.epochs):
        if distributed_train_sampler is not None:
            distributed_train_sampler.set_epoch(epoch)
        torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            world_size,
            optimizer,
            scaler,
            args.accumulation_steps,
            args.tissue_aux_weight,
            args.extreme_weight,
            args.smooth_weight,
            args.projection_weight,
            args.acf_weight,
            args.tissue_bias_weight,
            args.gradient_weight,
            False,
            args.teacher_weight,
            args.teacher_mask,
            args.teacher_reliability_scale,
            teacher_improvement_only=args.teacher_improvement_only,
            teacher_oracle_endpoint=args.teacher_oracle_endpoint,
            teacher_oracle_margin=args.teacher_oracle_margin,
            base_preservation_smooth_weight=args.base_preservation_smooth_weight,
            base_preservation_projection_weight=args.base_preservation_projection_weight,
            base_preservation_acf_weight=args.base_preservation_acf_weight,
            preserve_teacher_physics=args.preserve_teacher_physics,
            brain_mu_weight=args.brain_mu_weight,
            brain_ray_cvar_weight=args.brain_ray_cvar_weight,
            brain_ray_cvar_fraction=args.brain_ray_cvar_fraction,
            brain_ray_temperature=args.brain_ray_temperature,
            responsibility_boost_weight=args.responsibility_boost_weight,
            physics_teacher=physics_teacher,
            multiscale_target_weight=args.multiscale_target_weight,
            multiscale_target_scales=multiscale_target_scales,
            axial_slice_bias_weight=args.axial_slice_bias_weight,
            axial_slice_positive_bias_weight=args.axial_slice_positive_bias_weight,
            air_lung_axial_positive_bias_weight=args.air_lung_axial_positive_bias_weight,
            online_teacher_weight=args.online_teacher_weight,
            target_acf_regularization_weight=args.target_acf_regularization_weight,
            whole_body_acf_tail_weight=args.whole_body_acf_tail_weight,
            whole_body_acf_tail_fraction=args.whole_body_acf_tail_fraction,
            acf_noninferiority_hinge_weight=(
                args.acf_noninferiority_hinge_weight
            ),
            mgda_acf_noninferiority=args.mgda_acf_noninferiority,
            project_conflicting_acf_noninferiority=(
                args.project_conflicting_acf_noninferiority
            ),
            teacher_oracle_selected_weight=args.teacher_oracle_selected_weight,
            optimizer_step_callback=(
                update_optimizer_averages
                if ema is not None or lookahead is not None
                else None
            ),
            gradient_mask_mode=args.gradient_mask,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            world_size,
            None,
            None,
            1,
            args.tissue_aux_weight,
            args.extreme_weight,
            args.smooth_weight,
            args.projection_weight,
            args.acf_weight,
            args.tissue_bias_weight,
            args.gradient_weight,
            False,
            args.teacher_weight,
            args.teacher_mask,
            args.teacher_reliability_scale,
            teacher_improvement_only=args.teacher_improvement_only,
            teacher_oracle_endpoint=args.teacher_oracle_endpoint,
            teacher_oracle_margin=args.teacher_oracle_margin,
            base_preservation_smooth_weight=args.base_preservation_smooth_weight,
            base_preservation_projection_weight=args.base_preservation_projection_weight,
            base_preservation_acf_weight=args.base_preservation_acf_weight,
            preserve_teacher_physics=args.preserve_teacher_physics,
            brain_mu_weight=args.brain_mu_weight,
            brain_ray_cvar_weight=args.brain_ray_cvar_weight,
            brain_ray_cvar_fraction=args.brain_ray_cvar_fraction,
            brain_ray_temperature=args.brain_ray_temperature,
            responsibility_boost_weight=args.responsibility_boost_weight,
            physics_teacher=physics_teacher,
            multiscale_target_weight=args.multiscale_target_weight,
            multiscale_target_scales=multiscale_target_scales,
            axial_slice_bias_weight=args.axial_slice_bias_weight,
            axial_slice_positive_bias_weight=args.axial_slice_positive_bias_weight,
            air_lung_axial_positive_bias_weight=args.air_lung_axial_positive_bias_weight,
            online_teacher_weight=args.online_teacher_weight,
            target_acf_regularization_weight=args.target_acf_regularization_weight,
            whole_body_acf_tail_weight=args.whole_body_acf_tail_weight,
            whole_body_acf_tail_fraction=args.whole_body_acf_tail_fraction,
            acf_noninferiority_hinge_weight=(
                args.acf_noninferiority_hinge_weight
            ),
            mgda_acf_noninferiority=args.mgda_acf_noninferiority,
            project_conflicting_acf_noninferiority=(
                args.project_conflicting_acf_noninferiority
            ),
            teacher_oracle_selected_weight=args.teacher_oracle_selected_weight,
            gradient_mask_mode=args.gradient_mask,
        )
        epoch_seconds = time.perf_counter() - epoch_started
        peak_memory = torch.tensor(torch.cuda.max_memory_allocated(device) / 1024**3, device=device)
        if world_size > 1:
            dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
        scheduler.step()
        if rank == 0:
            record = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "epoch_seconds": epoch_seconds,
                "peak_allocated_gib": float(peak_memory.cpu()),
                "train": train_metrics,
                "val": val_metrics,
            }
            if ema is not None:
                record["ema_updates"] = ema.num_updates
            if lookahead is not None:
                record["lookahead_updates"] = lookahead.num_updates
                record["lookahead_syncs"] = lookahead.num_syncs
            print(json.dumps(record, sort_keys=True), flush=True)
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            checkpoint = {
                "epoch": epoch,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "val_metrics": val_metrics,
                "best_val": min(best_val, val_metrics["eval_mu_l1"]),
                "best_objective": min(best_objective, val_metrics["loss"]),
            }
            if lookahead is not None:
                checkpoint.update(
                    lookahead_slow_parameters=(
                        lookahead.slow_parameter_state_dict()
                    ),
                    lookahead_updates=lookahead.num_updates,
                    lookahead_syncs=lookahead.num_syncs,
                    lookahead_sync_steps=lookahead.sync_steps,
                    lookahead_alpha=lookahead.alpha,
                )
            torch.save(checkpoint, output_dir / "last.pt")
            if ema is not None:
                ema_checkpoint = {
                    **checkpoint,
                    "model": ema.model_state_dict(),
                    "ema_model": ema.model_state_dict(),
                    "ema_decay": ema.decay,
                    "ema_updates": ema.num_updates,
                }
                torch.save(ema_checkpoint, output_dir / "ema-last.pt")
            if lookahead is not None:
                lookahead_checkpoint = {
                    **checkpoint,
                    "model": lookahead.slow_model_state_dict(raw_model),
                }
                torch.save(
                    lookahead_checkpoint,
                    output_dir / "lookahead-last.pt",
                )
            if val_metrics["eval_mu_l1"] < best_val:
                best_val = val_metrics["eval_mu_l1"]
                torch.save(checkpoint, output_dir / "best.pt")
            if val_metrics["loss"] < best_objective:
                best_objective = val_metrics["loss"]
                torch.save(checkpoint, output_dir / "best-objective.pt")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
