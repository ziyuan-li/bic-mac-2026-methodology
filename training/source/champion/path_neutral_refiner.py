"""Full-plane 2.5D residual refiner with attenuation-path-neutral corrections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .model import group_count


def path_neutralize(
    correction: torch.Tensor,
    body_mask: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    """Project a 2D correction toward zero row and column attenuation sums."""
    if correction.ndim != 4 or correction.shape[1] != 1:
        raise ValueError("correction must have shape (batch, 1, height, width)")
    if correction.shape != body_mask.shape:
        raise ValueError("correction and body_mask shapes differ")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    active = body_mask.to(dtype=torch.float32)
    result = correction.float() * active
    for _ in range(iterations):
        height_counts = active.sum(dim=2, keepdim=True)
        height_means = result.sum(dim=2, keepdim=True) / torch.clamp(
            height_counts, min=1.0
        )
        result = (result - active * height_means) * active

        width_counts = active.sum(dim=3, keepdim=True)
        width_means = result.sum(dim=3, keepdim=True) / torch.clamp(
            width_counts, min=1.0
        )
        result = (result - active * width_means) * active
    return result


class ResidualBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(values) + self.skip(values))


class DownStage2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.stage = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            ResidualBlock2d(out_channels, out_channels),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.stage(values)


class UpStage2d(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ResidualBlock2d(out_channels + skip_channels, out_channels)

    def forward(self, values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        values = self.up(values)
        if values.shape[2:] != skip.shape[2:]:
            raise ValueError(
                f"decoder shape {values.shape[2:]} does not match skip shape {skip.shape[2:]}"
            )
        return self.block(torch.cat((values, skip), dim=1))


class PathNeutralSliceRefiner(nn.Module):
    """Predict a bounded residual and project it into the in-plane path null space."""

    def __init__(
        self,
        in_channels: int,
        context_radius: int,
        base_channels: int = 24,
        correction_scale: float = 0.01,
        mu_scale: float = 0.15,
        neutral_iterations: int = 8,
    ) -> None:
        super().__init__()
        if context_radius < 0:
            raise ValueError("context_radius must be non-negative")
        if correction_scale <= 0.0 or mu_scale <= 0.0:
            raise ValueError("correction_scale and mu_scale must be positive")
        if neutral_iterations <= 0:
            raise ValueError("neutral_iterations must be positive")
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
        self.output = nn.Conv2d(channels[0], 1, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.context_radius = int(context_radius)
        self.correction_scale = float(correction_scale)
        self.mu_scale = float(mu_scale)
        self.neutral_iterations = int(neutral_iterations)

    def forward(self, values: torch.Tensor, body_mask: torch.Tensor) -> torch.Tensor:
        skip0 = self.stem(values)
        skip1 = self.down1(skip0)
        skip2 = self.down2(skip1)
        skip3 = self.down3(skip2)
        bottleneck = self.down4(skip3)
        features = self.up3(bottleneck, skip3)
        features = self.up2(features, skip2)
        features = self.up1(features, skip1)
        features = self.up0(features, skip0)
        raw = self.correction_scale * torch.tanh(self.output(features))
        correction = path_neutralize(raw, body_mask, self.neutral_iterations)
        base = values[:, self.context_radius : self.context_radius + 1].float()
        return base * self.mu_scale + correction


def context_indices(depth: int, z: int, radius: int) -> list[int]:
    if not 0 <= z < depth:
        raise ValueError(f"slice {z} is outside depth {depth}")
    return [min(max(z + offset, 0), depth - 1) for offset in range(-radius, radius + 1)]


def spatial_channels(height: int, width: int, z: int, depth: int) -> list[np.ndarray]:
    x = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    y = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    z_value = np.float32(2.0 * z / max(1, depth - 1) - 1.0)
    return [
        np.broadcast_to(x, (height, width)),
        np.broadcast_to(y, (height, width)),
        np.full((height, width), z_value, dtype=np.float32),
    ]


def build_slice_input(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
    z: int,
    context_radius: int,
    mu_scale: float,
    body_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    base = arrays["base_mu"]
    depth = base.shape[2]
    indices = context_indices(depth, z, context_radius)
    relative_scale = max(float(metadata["nac_log_p99"]), 1e-6)
    channels = [
        np.asarray(base[:, :, index], dtype=np.float32) / mu_scale
        for index in indices
    ]
    channels.extend(
        np.clip(
            np.asarray(arrays["nac_log"][:, :, index], dtype=np.float32) / 7.0,
            0.0,
            2.0,
        )
        for index in indices
    )
    channels.extend(
        np.clip(
            np.asarray(arrays["nac_log"][:, :, index], dtype=np.float32)
            / relative_scale,
            0.0,
            2.0,
        )
        for index in indices
    )
    channels.extend(
        np.asarray(arrays[name][:, :, z], dtype=np.float32)
        for name in ("mri_in", "mri_out")
    )
    channels.extend(spatial_channels(base.shape[0], base.shape[1], z, depth))
    body_mask = np.asarray(base[:, :, z] > body_threshold, dtype=np.float32)[None]
    return np.ascontiguousarray(np.stack(channels, axis=0)), body_mask


def load_slice_arrays(
    cache_dir: Path,
    base_prediction_dir: Path,
    subject: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    subject_dir = cache_dir / "train" / subject
    arrays = {
        "base_mu": np.load(base_prediction_dir / subject / "mu.npy", mmap_mode="r"),
        "nac_log": np.load(subject_dir / "nac_log.npy", mmap_mode="r"),
        "mri_in": np.load(subject_dir / "mri_in.npy", mmap_mode="r"),
        "mri_out": np.load(subject_dir / "mri_out.npy", mmap_mode="r"),
        "mu": np.load(subject_dir / "mu.npy", mmap_mode="r"),
        "eval_mask": np.load(subject_dir / "eval_mask.npy", mmap_mode="r"),
        "prediction_mask": np.load(subject_dir / "prediction_mask.npy", mmap_mode="r"),
    }
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"{subject}: path-neutral input shapes differ: {sorted(shapes)}")
    metadata = json.loads((subject_dir / "metadata.json").read_text(encoding="utf-8"))
    return arrays, metadata


class PathNeutralSliceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        cache_dir: Path,
        base_prediction_dir: Path,
        subjects: Sequence[str],
        samples_per_subject: int,
        context_radius: int,
        mu_scale: float,
        body_threshold: float,
        augment: bool,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.base_prediction_dir = Path(base_prediction_dir)
        self.subjects = tuple(subjects)
        self.samples_per_subject = int(samples_per_subject)
        self.context_radius = int(context_radius)
        self.mu_scale = float(mu_scale)
        self.body_threshold = float(body_threshold)
        self.augment = bool(augment)
        self._arrays: dict[str, dict[str, np.ndarray]] = {}
        self._metadata: dict[str, dict[str, object]] = {}
        if self.samples_per_subject <= 0:
            raise ValueError("samples_per_subject must be positive")
        if self.context_radius < 0:
            raise ValueError("context_radius must be non-negative")
        if self.mu_scale <= 0.0 or self.body_threshold < 0.0:
            raise ValueError("mu_scale must be positive and body_threshold non-negative")

    def __len__(self) -> int:
        return len(self.subjects) * self.samples_per_subject

    def _load(self, subject: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        if subject not in self._arrays:
            arrays, metadata = load_slice_arrays(
                self.cache_dir,
                self.base_prediction_dir,
                subject,
            )
            self._arrays[subject] = arrays
            self._metadata[subject] = metadata
        return self._arrays[subject], self._metadata[subject]

    def _select_z(
        self,
        metadata: dict[str, object],
        sample: int,
        rng: np.random.Generator,
    ) -> int:
        bbox_name = "eval_bbox" if rng.random() < 0.8 else "prediction_bbox"
        bbox = metadata[bbox_name]
        if bbox is None:
            raise ValueError(f"missing {bbox_name}")
        lower, upper = (int(value) for value in bbox[2])
        if self.augment:
            return int(rng.integers(lower, upper))
        fraction = (sample + 0.5) / self.samples_per_subject
        return min(upper - 1, lower + int(fraction * (upper - lower)))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        subject_index, sample = divmod(index, self.samples_per_subject)
        subject = self.subjects[subject_index]
        arrays, metadata = self._load(subject)
        seed = (
            int(torch.randint(0, 2**31 - 1, (1,)).item())
            if self.augment
            else 202607137 + index
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

        if self.augment:
            if rng.random() < 0.5:
                inputs = inputs[:, ::-1, :]
                body_mask = body_mask[:, ::-1, :]
                target = target[:, ::-1, :]
                eval_mask = eval_mask[:, ::-1, :]
                prediction_mask = prediction_mask[:, ::-1, :]
            if rng.random() < 0.5:
                inputs = inputs[:, :, ::-1]
                body_mask = body_mask[:, :, ::-1]
                target = target[:, :, ::-1]
                eval_mask = eval_mask[:, :, ::-1]
                prediction_mask = prediction_mask[:, :, ::-1]

        return {
            "input": torch.from_numpy(np.ascontiguousarray(inputs)),
            "body_mask": torch.from_numpy(np.ascontiguousarray(body_mask)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "eval_mask": torch.from_numpy(np.ascontiguousarray(eval_mask)),
            "prediction_mask": torch.from_numpy(np.ascontiguousarray(prediction_mask)),
        }
