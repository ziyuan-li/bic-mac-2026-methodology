"""Patch data and model for refining cached out-of-fold mu-map predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .data import MuPatchDataset, spatial_coordinate_channels
from .model import DirectMuUNet


def extract_translated_patch(
    values: np.ndarray,
    slices: Sequence[slice],
    shift: Sequence[int],
) -> np.ndarray:
    """Extract a patch after a zero-filled whole-volume integer translation."""
    if values.ndim != len(slices) or values.ndim != len(shift):
        raise ValueError("values, slices and shift must have the same dimensionality")
    output_shape: list[int] = []
    source: list[slice] = []
    destination: list[slice] = []
    for size, selection, offset_value in zip(values.shape, slices, shift):
        if selection.step not in (None, 1):
            raise ValueError("translated patch slices must have unit stride")
        start = 0 if selection.start is None else int(selection.start)
        stop = int(size) if selection.stop is None else int(selection.stop)
        if start < 0 or stop > size or stop <= start:
            raise ValueError("translated patch slice is outside the source volume")
        offset = int(offset_value)
        patch_size = stop - start
        source_start = max(0, start - offset)
        source_stop = min(int(size), stop - offset)
        output_shape.append(patch_size)
        if source_stop <= source_start:
            source.append(slice(0, 0))
            destination.append(slice(0, 0))
            continue
        destination_start = source_start + offset - start
        destination_stop = destination_start + source_stop - source_start
        source.append(slice(source_start, source_stop))
        destination.append(slice(destination_start, destination_stop))
    output = np.zeros(tuple(output_shape), dtype=values.dtype)
    if all(item.stop > item.start for item in source):
        output[tuple(destination)] = values[tuple(source)]
    return output


class MuResidualRefiner(DirectMuUNet):
    """Bounded residual U-Net whose zero initialization reproduces the base mu-map."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 12,
        correction_scale: float = 0.03,
        mu_scale: float = 0.15,
    ) -> None:
        if correction_scale <= 0.0:
            raise ValueError("correction_scale must be positive")
        if mu_scale <= 0.0:
            raise ValueError("mu_scale must be positive")
        super().__init__(in_channels=in_channels, base_channels=base_channels, output_channels=1)
        self.correction_scale = float(correction_scale)
        self.mu_scale = float(mu_scale)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        base_mu = values[:, :1] * self.mu_scale
        correction = self.correction_scale * torch.tanh(super().forward(values))
        return base_mu + correction


class MuRefinementPatchDataset(MuPatchDataset):
    def __init__(
        self,
        cache_dir: Path,
        base_prediction_dir: Path,
        subjects: Sequence[str],
        patch_size: Sequence[int],
        samples_per_subject: int,
        augment: bool,
        eval_center_probability: float = 0.8,
        use_mri: bool = False,
        coordinate_mode: str = "xyz",
        mu_scale: float = 0.15,
        teacher_prediction_dir: Path | None = None,
        brain_mask_filename: str | None = None,
        mri_translation_probability: float = 0.0,
        mri_translation_max_voxels: Sequence[int] = (0, 0, 0),
    ) -> None:
        super().__init__(
            cache_dir=cache_dir,
            subjects=subjects,
            patch_size=patch_size,
            samples_per_subject=samples_per_subject,
            augment=augment,
            eval_center_probability=eval_center_probability,
            use_mri=use_mri,
            coordinate_mode=coordinate_mode,
            teacher_prediction_dir=teacher_prediction_dir,
        )
        self.base_prediction_dir = Path(base_prediction_dir)
        self.mu_scale = float(mu_scale)
        self.brain_mask_filename = brain_mask_filename
        self.mri_translation_probability = float(mri_translation_probability)
        self.mri_translation_max_voxels = tuple(
            int(value) for value in mri_translation_max_voxels
        )
        if self.mu_scale <= 0.0:
            raise ValueError("mu_scale must be positive")
        if not 0.0 <= self.mri_translation_probability <= 1.0:
            raise ValueError("MRI translation probability must be between zero and one")
        if len(self.mri_translation_max_voxels) != 3 or min(
            self.mri_translation_max_voxels
        ) < 0:
            raise ValueError("MRI translation maximum must contain three nonnegative values")
        if self.mri_translation_probability > 0.0 and not self.use_mri:
            raise ValueError("MRI translation augmentation requires MRI inputs")

    def _get_arrays(self, subject: str) -> dict[str, np.ndarray]:
        arrays = super()._get_arrays(subject)
        if "base_mu" not in arrays:
            base_path = self.base_prediction_dir / subject / "mu.npy"
            arrays["base_mu"] = np.load(base_path, mmap_mode="r")
            if arrays["base_mu"].shape != arrays["mu"].shape:
                raise ValueError(
                    f"{subject}: base prediction shape {arrays['base_mu'].shape} "
                    f"does not match target {arrays['mu'].shape}"
                )
        if self.brain_mask_filename is not None and "brain_mask" not in arrays:
            brain_path = self.cache_dir / "train" / subject / self.brain_mask_filename
            arrays["brain_mask"] = np.load(brain_path, mmap_mode="r")
            if arrays["brain_mask"].shape != arrays["mu"].shape:
                raise ValueError(
                    f"{subject}: brain mask shape {arrays['brain_mask'].shape} "
                    f"does not match target {arrays['mu'].shape}"
                )
        return arrays

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        subject = self.subjects[index % len(self.subjects)]
        arrays = self._get_arrays(subject)
        seed = index + 20260711 if not self.augment else int(
            torch.randint(0, 2**31 - 1, (1,)).item()
        )
        rng = np.random.default_rng(seed)
        center = self._choose_center(subject, arrays, rng)
        slices = self._slices(center, arrays["mu"].shape)
        flip_x = self.augment and rng.random() < 0.5

        base_mu = np.asarray(arrays["base_mu"][slices], dtype=np.float32)
        nac_log = np.asarray(arrays["nac_log"][slices], dtype=np.float32)
        relative_scale = max(float(self._metadata[subject]["nac_log_p99"]), 1e-6)
        inputs = [
            base_mu / self.mu_scale,
            np.clip(nac_log / 7.0, 0.0, 2.0),
            np.clip(nac_log / relative_scale, 0.0, 2.0),
        ]
        if self.use_mri:
            mri_shift = (0, 0, 0)
            if self.augment and rng.random() < self.mri_translation_probability:
                mri_shift = tuple(
                    int(rng.integers(-maximum, maximum + 1)) if maximum else 0
                    for maximum in self.mri_translation_max_voxels
                )
            inputs.extend(
                np.asarray(
                    extract_translated_patch(arrays[name], slices, mri_shift),
                    dtype=np.float32,
                )
                for name in ("mri_in", "mri_out")
            )
        inputs_array = np.stack(inputs, axis=0)
        target = np.asarray(arrays["mu"][slices], dtype=np.float32)[None]
        teacher = (
            np.asarray(arrays["teacher_mu"][slices], dtype=np.float32)[None]
            if "teacher_mu" in arrays
            else None
        )
        prediction_mask = np.asarray(
            arrays["prediction_mask"][slices],
            dtype=np.float32,
        )[None]
        eval_mask = np.asarray(
            arrays["eval_mask"][slices],
            dtype=np.float32,
        )[None]
        brain_mask = (
            np.asarray(arrays["brain_mask"][slices], dtype=np.float32)[None]
            if "brain_mask" in arrays
            else None
        )

        if flip_x:
            inputs_array = np.flip(inputs_array, axis=1)
            target = np.flip(target, axis=1)
            if teacher is not None:
                teacher = np.flip(teacher, axis=1)
            prediction_mask = np.flip(prediction_mask, axis=1)
            eval_mask = np.flip(eval_mask, axis=1)
            if brain_mask is not None:
                brain_mask = np.flip(brain_mask, axis=1)

        coordinates = spatial_coordinate_channels(
            arrays["mu"].shape,
            slices,
            self.coordinate_mode,
        )
        if coordinates.shape[0]:
            inputs_array = np.concatenate((inputs_array, coordinates), axis=0)

        result: dict[str, torch.Tensor | str] = {
            "input": torch.from_numpy(np.ascontiguousarray(inputs_array)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "prediction_mask": torch.from_numpy(
                np.ascontiguousarray(prediction_mask)
            ),
            "eval_mask": torch.from_numpy(np.ascontiguousarray(eval_mask)),
            "subject": subject,
        }
        if teacher is not None:
            result["teacher"] = torch.from_numpy(np.ascontiguousarray(teacher))
        if brain_mask is not None:
            result["brain_mask"] = torch.from_numpy(
                np.ascontiguousarray(brain_mask)
            )
        return result


class BrainPathRefinementPatchDataset(MuRefinementPatchDataset):
    """Sample full in-plane patches centered on labelled brain slices."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs["brain_mask_filename"] = "brain_mask.npy"
        super().__init__(*args, **kwargs)

    def _choose_center(
        self,
        subject: str,
        arrays: dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> tuple[int, int, int]:
        brain = arrays["brain_mask"]
        if tuple(self.patch_size[:2]) != tuple(brain.shape[:2]):
            raise ValueError(
                f"{subject}: brain-path patches must cover the full in-plane "
                f"shape {brain.shape[:2]}, got {self.patch_size[:2]}"
            )
        brain_slices = np.flatnonzero(np.any(brain, axis=(0, 1)))
        if brain_slices.size == 0:
            raise ValueError(f"{subject}: brain mask is empty")
        z = int(brain_slices[int(rng.integers(brain_slices.size))])
        return brain.shape[0] // 2, brain.shape[1] // 2, z


def build_refiner_input(
    subject_dir: Path,
    base_prediction_dir: Path,
    use_mri: bool,
    coordinate_mode: str,
    mu_scale: float,
) -> torch.Tensor:
    metadata = json.loads((subject_dir / "metadata.json").read_text(encoding="utf-8"))
    base_mu = np.asarray(
        np.load(base_prediction_dir / subject_dir.name / "mu.npy", mmap_mode="r"),
        dtype=np.float32,
    )
    nac_log = np.asarray(np.load(subject_dir / "nac_log.npy", mmap_mode="r"), dtype=np.float32)
    if base_mu.shape != nac_log.shape:
        raise ValueError(f"{subject_dir.name}: base prediction and NAC shapes differ")
    channels = [
        base_mu / mu_scale,
        np.clip(nac_log / 7.0, 0.0, 2.0),
        np.clip(nac_log / max(float(metadata["nac_log_p99"]), 1e-6), 0.0, 2.0),
    ]
    if use_mri:
        channels.extend(
            np.asarray(np.load(subject_dir / name, mmap_mode="r"), dtype=np.float32)
            for name in ("mri_in.npy", "mri_out.npy")
        )
    full_slices = tuple(slice(0, size) for size in nac_log.shape)
    channels.extend(spatial_coordinate_channels(nac_log.shape, full_slices, coordinate_mode))
    return torch.from_numpy(np.ascontiguousarray(np.stack(channels, axis=0)[None]))
