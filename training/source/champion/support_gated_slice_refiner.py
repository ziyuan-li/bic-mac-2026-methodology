"""Full-plane residual refiner with a jointly learned conservative support gate."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from .losses import (
    acf_path_loss,
    direct_mu_loss,
    masked_l1,
    projection_path_loss,
)
from .path_neutral_refiner import (
    DownStage2d,
    PathNeutralSliceDataset,
    ResidualBlock2d,
    UpStage2d,
    build_slice_input,
)


class SupportGatedSliceRefiner(nn.Module):
    """Predict a bounded CT residual and the support on which it may act."""

    def __init__(
        self,
        in_channels: int,
        context_radius: int,
        base_channels: int = 16,
        correction_scale: float = 0.005,
        mu_scale: float = 0.15,
    ) -> None:
        super().__init__()
        if min(in_channels, base_channels) <= 0 or context_radius < 0:
            raise ValueError("channel counts must be positive and radius non-negative")
        if min(correction_scale, mu_scale) <= 0.0:
            raise ValueError("correction and mu scales must be positive")
        channels = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 12,
        )
        self.stem = ResidualBlock2d(in_channels, channels[0])
        self.down1 = DownStage2d(channels[0], channels[1])
        self.down2 = DownStage2d(channels[1], channels[2])
        self.down3 = DownStage2d(channels[2], channels[3])
        self.down4 = DownStage2d(channels[3], channels[4])
        self.up3 = UpStage2d(channels[4], channels[3], channels[3])
        self.up2 = UpStage2d(channels[3], channels[2], channels[2])
        self.up1 = UpStage2d(channels[2], channels[1], channels[1])
        self.up0 = UpStage2d(channels[1], channels[0], channels[0])
        self.output = nn.Conv2d(channels[0], 2, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.context_radius = int(context_radius)
        self.correction_scale = float(correction_scale)
        self.mu_scale = float(mu_scale)

    def raw_outputs(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        skip0 = self.stem(values)
        skip1 = self.down1(skip0)
        skip2 = self.down2(skip1)
        skip3 = self.down3(skip2)
        bottleneck = self.down4(skip3)
        features = self.up3(bottleneck, skip3)
        features = self.up2(features, skip2)
        features = self.up1(features, skip1)
        features = self.up0(features, skip0)
        raw_correction, support_logits = self.output(features).float().split(1, dim=1)
        base = (
            values[:, self.context_radius : self.context_radius + 1].float()
            * self.mu_scale
        )
        bounded = self.correction_scale * torch.tanh(raw_correction)
        return base, bounded, support_logits

    def forward(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base, bounded, support_logits = self.raw_outputs(values)
        correction = bounded * torch.sigmoid(support_logits)
        return base + correction, support_logits, correction

    def infer(
        self,
        values: torch.Tensor,
        support_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 0.0 < support_threshold < 1.0:
            raise ValueError("support threshold must lie strictly between zero and one")
        base, bounded, support_logits = self.raw_outputs(values)
        support = torch.sigmoid(support_logits) >= support_threshold
        correction = bounded * support.to(dtype=bounded.dtype)
        return torch.clamp_min(base + correction, 0.0), support_logits, correction


class SupportGatedSliceDataset(PathNeutralSliceDataset):
    """Sample full axial slices with random z and only left-right reflection."""

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        subject_index, sample = divmod(index, self.samples_per_subject)
        subject = self.subjects[subject_index]
        arrays, metadata = self._load(subject)
        seed = (
            int(torch.randint(0, 2**31 - 1, (1,)).item())
            if self.augment
            else 202608439 + index
        )
        rng = np.random.default_rng(seed)
        z = self._select_z(metadata, sample, rng)
        inputs, body_mask = build_slice_input(
            arrays,
            metadata,
            z,
            self.context_radius,
            self.mu_scale,
            self.body_threshold,
        )
        target = np.asarray(arrays["mu"][:, :, z], dtype=np.float32)[None]
        eval_mask = np.asarray(arrays["eval_mask"][:, :, z], dtype=np.float32)[None]
        prediction_mask = np.asarray(
            arrays["prediction_mask"][:, :, z], dtype=np.float32
        )[None]
        if self.augment and rng.random() < 0.5:
            inputs = inputs[:, ::-1, :]
            body_mask = body_mask[:, ::-1, :]
            target = target[:, ::-1, :]
            eval_mask = eval_mask[:, ::-1, :]
            prediction_mask = prediction_mask[:, ::-1, :]
        return {
            "input": torch.from_numpy(np.ascontiguousarray(inputs)),
            "body_mask": torch.from_numpy(np.ascontiguousarray(body_mask)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "eval_mask": torch.from_numpy(np.ascontiguousarray(eval_mask)),
            "prediction_mask": torch.from_numpy(np.ascontiguousarray(prediction_mask)),
        }


def support_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    truth = target.to(dtype=logits.dtype)
    bce = functional.binary_cross_entropy_with_logits(logits, truth)
    probability = torch.sigmoid(logits)
    intersection = (probability * truth).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice_loss, bce, dice_loss


def acf_tail_path_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    tail_fraction: float,
) -> torch.Tensor:
    """Penalize the worst fixed fraction of in-plane relative ACF errors."""
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail fraction must lie in (0, 1]")
    active = mask.to(dtype=prediction.dtype)
    errors = ((prediction - target) * active).float()
    losses = []
    for axis in (2, 3):
        counts = active.sum(dim=axis)
        valid = counts > 0
        line_integral = errors.sum(dim=axis) * 0.15234375
        relative = torch.abs(torch.expm1(torch.clamp(line_integral, -2.0, 2.0)))
        selected = relative[valid]
        if selected.numel() == 0:
            losses.append(prediction.new_zeros((), dtype=torch.float32))
            continue
        count = max(1, int(np.ceil(tail_fraction * selected.numel())))
        losses.append(torch.topk(selected, count, sorted=False).values.mean())
    return torch.stack(losses).mean()


def support_classification_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly between zero and one")
    prediction = torch.sigmoid(logits) >= threshold
    truth = target > 0.5
    true_positive = (prediction & truth).sum().float()
    predicted_positive = prediction.sum().float()
    target_positive = truth.sum().float()
    precision = true_positive / torch.clamp(predicted_positive, min=1.0)
    recall = true_positive / torch.clamp(target_positive, min=1.0)
    dice = 2.0 * true_positive / torch.clamp(
        predicted_positive + target_positive,
        min=1.0,
    )
    return {
        "support_precision": precision,
        "support_recall": recall,
        "support_dice": dice,
    }


def support_gated_objective(
    prediction: torch.Tensor,
    support_logits: torch.Tensor,
    correction: torch.Tensor,
    target: torch.Tensor,
    eval_mask: torch.Tensor,
    prediction_mask: torch.Tensor,
    projection_weight: float,
    acf_weight: float,
    tail_weight: float,
    segmentation_weight: float,
    outside_weight: float,
    tail_fraction: float,
    support_threshold: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = (
        projection_weight,
        acf_weight,
        tail_weight,
        segmentation_weight,
        outside_weight,
    )
    if min(weights) < 0.0:
        raise ValueError("objective weights must be non-negative")
    direct, metrics = direct_mu_loss(
        prediction,
        target,
        eval_mask,
        prediction_mask,
    )
    projection = projection_path_loss(prediction, target, prediction_mask)
    acf = acf_path_loss(prediction, target, prediction_mask)
    tail = acf_tail_path_loss(
        prediction,
        target,
        prediction_mask,
        tail_fraction,
    )
    segmentation, bce, dice_loss = support_segmentation_loss(
        support_logits,
        prediction_mask,
    )
    outside = masked_l1(
        correction,
        torch.zeros_like(correction),
        1.0 - prediction_mask,
    )
    total = (
        direct
        + projection_weight * projection
        + acf_weight * acf
        + tail_weight * tail
        + segmentation_weight * segmentation
        + outside_weight * outside
    )
    support_metrics = support_classification_metrics(
        support_logits,
        prediction_mask,
        support_threshold,
    )
    return total, {
        **metrics,
        **{key: value.detach() for key, value in support_metrics.items()},
        "loss": total.detach(),
        "projection_path_l1": projection.detach(),
        "acf_path_relative_error": acf.detach(),
        "acf_tail_relative_error": tail.detach(),
        "support_loss": segmentation.detach(),
        "support_bce": bce.detach(),
        "support_dice_loss": dice_loss.detach(),
        "outside_correction_l1": outside.detach(),
        "correction_l1": torch.mean(torch.abs(correction)).detach(),
    }
