"""Memory-mapped patch dataset for direct mu-map training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def spatial_coordinate_channels(
    shape: Sequence[int],
    slices: Sequence[slice],
    mode: str,
) -> np.ndarray:
    """Build global coordinates for a patch without allocating full-volume grids."""
    if mode not in ("none", "z", "xyz"):
        raise ValueError(f"unsupported coordinate mode: {mode}")
    if mode == "none":
        patch_shape = tuple(int(part.stop) - int(part.start) for part in slices)
        return np.empty((0, *patch_shape), dtype=np.float32)

    selected_axes = (2,) if mode == "z" else (0, 1, 2)
    patch_shape = tuple(int(part.stop) - int(part.start) for part in slices)
    channels = []
    for axis in selected_axes:
        full_size = int(shape[axis])
        positions = np.arange(
            int(slices[axis].start),
            int(slices[axis].stop),
            dtype=np.float32,
        )
        positions = 2.0 * positions / max(1, full_size - 1) - 1.0
        reshape = [1, 1, 1]
        reshape[axis] = positions.size
        channels.append(np.broadcast_to(positions.reshape(reshape), patch_shape))
    return np.ascontiguousarray(np.stack(channels, axis=0))


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


class MuPatchDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        cache_dir: Path,
        subjects: Sequence[str],
        patch_size: Sequence[int],
        samples_per_subject: int,
        augment: bool,
        eval_center_probability: float = 0.8,
        use_topogram: bool = False,
        use_mri: bool = False,
        coordinate_mode: str = "none",
        teacher_prediction_dir: Path | None = None,
        base_prediction_dir: Path | None = None,
        axial_training_region: Sequence[int] | None = None,
        brain_mask_filename: str | None = None,
        brain_center_probability: float = 0.0,
        responsibility_center_probability: float = 0.0,
        center_transverse_on_prediction_bbox: bool = False,
        nac_absolute_gain_log_std: float = 0.0,
        mri_translation_probability: float = 0.0,
        mri_translation_max_voxels: Sequence[int] = (0, 0, 0),
        return_clean_input: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.subjects = tuple(subjects)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.samples_per_subject = int(samples_per_subject)
        self.augment = augment
        self.eval_center_probability = float(eval_center_probability)
        self.use_topogram = use_topogram
        self.use_mri = use_mri
        self.teacher_prediction_dir = (
            Path(teacher_prediction_dir)
            if teacher_prediction_dir is not None
            else None
        )
        self.base_prediction_dir = (
            Path(base_prediction_dir)
            if base_prediction_dir is not None
            else None
        )
        self.axial_training_region = (
            tuple(int(value) for value in axial_training_region)
            if axial_training_region is not None
            else None
        )
        self.brain_mask_filename = brain_mask_filename
        self.brain_center_probability = float(brain_center_probability)
        self.responsibility_center_probability = float(
            responsibility_center_probability
        )
        self.center_transverse_on_prediction_bbox = bool(
            center_transverse_on_prediction_bbox
        )
        self.nac_absolute_gain_log_std = float(nac_absolute_gain_log_std)
        self.mri_translation_probability = float(mri_translation_probability)
        self.mri_translation_max_voxels = tuple(
            int(value) for value in mri_translation_max_voxels
        )
        self.return_clean_input = bool(return_clean_input)
        if coordinate_mode not in ("none", "z", "xyz"):
            raise ValueError(f"unsupported coordinate mode: {coordinate_mode}")
        self.coordinate_mode = coordinate_mode
        self._arrays: dict[str, dict[str, np.ndarray]] = {}
        self._metadata = {
            subject: json.loads((self.cache_dir / "train" / subject / "metadata.json").read_text(encoding="utf-8"))
            for subject in self.subjects
        }
        if any(value <= 0 for value in self.patch_size):
            raise ValueError("patch_size must be positive")
        if self.samples_per_subject <= 0:
            raise ValueError("samples_per_subject must be positive")
        if self.nac_absolute_gain_log_std < 0.0:
            raise ValueError("nac_absolute_gain_log_std must be non-negative")
        if self.nac_absolute_gain_log_std > 0.0 and not self.augment:
            raise ValueError(
                "NAC absolute gain perturbation is only valid for augmented data"
            )
        if not 0.0 <= self.mri_translation_probability <= 1.0:
            raise ValueError("MRI translation probability must be between zero and one")
        if len(self.mri_translation_max_voxels) != 3 or min(
            self.mri_translation_max_voxels
        ) < 0:
            raise ValueError(
                "MRI translation maximum must contain three nonnegative values"
            )
        if self.mri_translation_probability > 0.0 and not self.use_mri:
            raise ValueError("MRI translation augmentation requires MRI inputs")
        if self.mri_translation_probability > 0.0 and not self.augment:
            raise ValueError(
                "MRI translation augmentation is only valid for augmented data"
            )
        if not 0.0 <= self.brain_center_probability <= 1.0:
            raise ValueError("brain_center_probability must be between zero and one")
        if not 0.0 <= self.responsibility_center_probability <= 1.0:
            raise ValueError(
                "responsibility_center_probability must be between zero and one"
            )
        if self.brain_center_probability > 0.0 and self.brain_mask_filename is None:
            raise ValueError(
                "brain_center_probability requires brain_mask_filename"
            )
        if self.axial_training_region is not None:
            if len(self.axial_training_region) != 2:
                raise ValueError("axial_training_region must contain start and end")
            start, end = self.axial_training_region
            if not 0 <= start < end:
                raise ValueError(
                    "axial_training_region must satisfy 0 <= start < end"
                )

    def __len__(self) -> int:
        return len(self.subjects) * self.samples_per_subject

    def _get_arrays(self, subject: str) -> dict[str, np.ndarray]:
        if subject not in self._arrays:
            subject_dir = self.cache_dir / "train" / subject
            self._arrays[subject] = {
                "nac_log": np.load(subject_dir / "nac_log.npy", mmap_mode="r"),
                "mu": np.load(subject_dir / "mu.npy", mmap_mode="r"),
                "prediction_mask": np.load(subject_dir / "prediction_mask.npy", mmap_mode="r"),
                "eval_mask": np.load(subject_dir / "eval_mask.npy", mmap_mode="r"),
            }
            if self.use_topogram:
                self._arrays[subject]["topogram"] = np.load(
                    subject_dir / "topogram.npy", mmap_mode="r"
                )
            if self.use_mri:
                self._arrays[subject]["mri_in"] = np.load(
                    subject_dir / "mri_in.npy", mmap_mode="r"
                )
                self._arrays[subject]["mri_out"] = np.load(
                    subject_dir / "mri_out.npy", mmap_mode="r"
                )
            if self.brain_mask_filename is not None:
                brain_mask = np.load(
                    subject_dir / self.brain_mask_filename,
                    mmap_mode="r",
                )
                if brain_mask.shape != self._arrays[subject]["mu"].shape:
                    raise ValueError(
                        f"{subject}: brain mask shape {brain_mask.shape} "
                        f"does not match target {self._arrays[subject]['mu'].shape}"
                    )
                self._arrays[subject]["brain_mask"] = brain_mask
            if self.teacher_prediction_dir is not None:
                teacher = np.load(
                    self.teacher_prediction_dir / subject / "mu.npy",
                    mmap_mode="r",
                )
                if teacher.shape != self._arrays[subject]["mu"].shape:
                    raise ValueError(
                        f"{subject}: teacher prediction shape {teacher.shape} "
                        f"does not match target {self._arrays[subject]['mu'].shape}"
                    )
                self._arrays[subject]["teacher_mu"] = teacher
            if self.base_prediction_dir is not None:
                base_prediction = np.load(
                    self.base_prediction_dir / subject / "mu.npy",
                    mmap_mode="r",
                )
                if base_prediction.shape != self._arrays[subject]["mu"].shape:
                    raise ValueError(
                        f"{subject}: base prediction shape {base_prediction.shape} "
                        f"does not match target {self._arrays[subject]['mu'].shape}"
                    )
                self._arrays[subject]["base_prediction_mu"] = base_prediction
            if (
                self.axial_training_region is not None
                and self.axial_training_region[1]
                > self._arrays[subject]["mu"].shape[2]
            ):
                raise ValueError(
                    f"{subject}: axial training region {self.axial_training_region} "
                    f"exceeds depth {self._arrays[subject]['mu'].shape[2]}"
                )
        return self._arrays[subject]

    def _choose_from_mask(
        self,
        subject: str,
        mask_name: str,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[int, int, int]:
        if self.axial_training_region is None:
            coordinates = np.nonzero(mask)
        else:
            start, end = self.axial_training_region
            restricted = np.nonzero(mask[:, :, start:end])
            coordinates = (
                restricted[0],
                restricted[1],
                restricted[2] + start,
            )
        if coordinates[0].size == 0:
            raise ValueError(f"{subject}: {mask_name} has no selectable voxels")
        selected = int(rng.integers(coordinates[0].size))
        return tuple(int(axis[selected]) for axis in coordinates)

    def _choose_center(self, subject: str, arrays: dict[str, np.ndarray], rng: np.random.Generator) -> tuple[int, int, int]:
        if (
            self.brain_center_probability > 0.0
            and rng.random() < self.brain_center_probability
        ):
            return self._choose_from_mask(
                subject,
                "brain_mask",
                arrays["brain_mask"],
                rng,
            )
        if (
            self.responsibility_center_probability > 0.0
            and rng.random() < self.responsibility_center_probability
        ):
            responsibility = np.asarray(arrays["prediction_mask"], dtype=bool) & ~np.asarray(
                arrays["eval_mask"], dtype=bool
            )
            return self._choose_from_mask(
                subject,
                "responsibility_mask",
                responsibility,
                rng,
            )
        use_eval = rng.random() < self.eval_center_probability
        mask_name = "eval_mask" if use_eval else "prediction_mask"
        mask = arrays[mask_name]
        bbox = self._metadata[subject]["eval_bbox" if use_eval else "prediction_bbox"]
        if bbox is None:
            raise ValueError(f"{subject}: {mask_name} is empty")
        bbox = [list(axis) for axis in bbox]
        if self.axial_training_region is not None:
            start, end = self.axial_training_region
            bbox[2][0] = max(int(bbox[2][0]), start)
            bbox[2][1] = min(int(bbox[2][1]), end)
            if bbox[2][0] >= bbox[2][1]:
                raise ValueError(
                    f"{subject}: {mask_name} has no support in axial training "
                    f"region {self.axial_training_region}"
                )
        for _ in range(256):
            center = tuple(int(rng.integers(lower, upper)) for lower, upper in bbox)
            if mask[center] != 0:
                return center
        if self.axial_training_region is None:
            coordinates = np.nonzero(mask)
        else:
            start, end = self.axial_training_region
            restricted = np.nonzero(mask[:, :, start:end])
            coordinates = (
                restricted[0],
                restricted[1],
                restricted[2] + start,
            )
        if coordinates[0].size == 0:
            raise ValueError(
                f"{subject}: {mask_name} has no selectable training voxels"
            )
        selected = int(rng.integers(coordinates[0].size))
        return tuple(int(axis[selected]) for axis in coordinates)

    def _slices(self, center: Sequence[int], shape: Sequence[int]) -> tuple[slice, slice, slice]:
        slices = []
        for coordinate, size, full_size in zip(center, self.patch_size, shape):
            if size > full_size:
                raise ValueError(f"patch dimension {size} exceeds volume dimension {full_size}")
            start = max(0, min(int(coordinate) - size // 2, full_size - size))
            slices.append(slice(start, start + size))
        return tuple(slices)  # type: ignore[return-value]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        subject = self.subjects[index % len(self.subjects)]
        arrays = self._get_arrays(subject)
        # Keep validation patches fixed across epochs so their metric is comparable.
        seed = index + 20260711 if not self.augment else int(torch.randint(0, 2**31 - 1, (1,)).item())
        rng = np.random.default_rng(seed)
        center = self._choose_center(subject, arrays, rng)
        if self.center_transverse_on_prediction_bbox:
            prediction_bbox = self._metadata[subject]["prediction_bbox"]
            for axis in (0, 1):
                width = int(prediction_bbox[axis][1]) - int(
                    prediction_bbox[axis][0]
                )
                if width > self.patch_size[axis]:
                    raise ValueError(
                        f"{subject}: transverse patch axis {axis} size "
                        f"{self.patch_size[axis]} does not cover prediction "
                        f"bbox width {width}"
                    )
            center = (
                (int(prediction_bbox[0][0]) + int(prediction_bbox[0][1])) // 2,
                (int(prediction_bbox[1][0]) + int(prediction_bbox[1][1])) // 2,
                center[2],
            )
        slices = self._slices(center, arrays["mu"].shape)
        if self.center_transverse_on_prediction_bbox:
            for axis in (0, 1):
                lower, upper = (
                    int(value)
                    for value in self._metadata[subject]["prediction_bbox"][axis]
                )
                if slices[axis].start > lower or slices[axis].stop < upper:
                    raise RuntimeError(
                        f"{subject}: centered transverse patch does not cover "
                        f"prediction bbox on axis {axis}"
                    )

        nac_log = np.asarray(arrays["nac_log"][slices], dtype=np.float32)
        absolute = np.clip(nac_log / 7.0, 0.0, 2.0)
        relative_scale = max(float(self._metadata[subject]["nac_log_p99"]), 1e-6)
        relative = np.clip(nac_log / relative_scale, 0.0, 2.0)
        inputs = np.stack((absolute, relative), axis=0)
        if self.use_topogram:
            topogram_2d = np.asarray(
                arrays["topogram"][slices[0], slices[2]], dtype=np.float32
            )
            topogram = np.broadcast_to(
                topogram_2d[:, None, :], tuple(self.patch_size)
            )
            inputs = np.concatenate((inputs, topogram[None]), axis=0)
        if self.use_mri:
            mri_shift = (0, 0, 0)
            if self.mri_translation_probability > 0.0:
                # Keep the established center/flip RNG stream unchanged so this
                # augmentation can alter only the paired MRI channels.
                mri_rng = np.random.default_rng(seed ^ 0x4D524954)
                if mri_rng.random() < self.mri_translation_probability:
                    mri_shift = tuple(
                        int(mri_rng.integers(-maximum, maximum + 1))
                        if maximum
                        else 0
                        for maximum in self.mri_translation_max_voxels
                    )
            mri = np.stack(
                tuple(
                    np.asarray(
                        extract_translated_patch(arrays[name], slices, mri_shift),
                        dtype=np.float32,
                    )
                    for name in ("mri_in", "mri_out")
                ),
                axis=0,
            )
            inputs = np.concatenate((inputs, mri), axis=0)
        clean_input = inputs.copy() if self.return_clean_input else None
        if self.nac_absolute_gain_log_std > 0.0:
            gain = float(
                np.exp(rng.normal(0.0, self.nac_absolute_gain_log_std))
            )
            inputs[0] = np.clip(inputs[0] * gain, 0.0, 2.0)
        target = np.asarray(arrays["mu"][slices], dtype=np.float32)[None]
        teacher = (
            np.asarray(arrays["teacher_mu"][slices], dtype=np.float32)[None]
            if "teacher_mu" in arrays
            else None
        )
        base_prediction = (
            np.asarray(
                arrays["base_prediction_mu"][slices], dtype=np.float32
            )[None]
            if "base_prediction_mu" in arrays
            else None
        )
        prediction_mask = np.asarray(arrays["prediction_mask"][slices], dtype=np.float32)[None]
        eval_mask = np.asarray(arrays["eval_mask"][slices], dtype=np.float32)[None]
        brain_mask = (
            np.asarray(arrays["brain_mask"][slices], dtype=np.float32)[None]
            if "brain_mask" in arrays
            else None
        )
        if self.axial_training_region is not None:
            start, end = self.axial_training_region
            global_z = np.arange(slices[2].start, slices[2].stop)
            axial_mask = ((global_z >= start) & (global_z < end))[None, None, None, :]
            prediction_mask = prediction_mask * axial_mask
            eval_mask = eval_mask * axial_mask

        if self.augment and rng.random() < 0.5:
            inputs = np.flip(inputs, axis=1)
            if clean_input is not None:
                clean_input = np.flip(clean_input, axis=1)
            target = np.flip(target, axis=1)
            if teacher is not None:
                teacher = np.flip(teacher, axis=1)
            if base_prediction is not None:
                base_prediction = np.flip(base_prediction, axis=1)
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
            inputs = np.concatenate((inputs, coordinates), axis=0)
            if clean_input is not None:
                clean_input = np.concatenate((clean_input, coordinates), axis=0)

        result: dict[str, torch.Tensor | str] = {
            "input": torch.from_numpy(np.ascontiguousarray(inputs)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "prediction_mask": torch.from_numpy(np.ascontiguousarray(prediction_mask)),
            "eval_mask": torch.from_numpy(np.ascontiguousarray(eval_mask)),
            "subject": subject,
        }
        if teacher is not None:
            result["teacher"] = torch.from_numpy(np.ascontiguousarray(teacher))
        if base_prediction is not None:
            result["base_prediction"] = torch.from_numpy(
                np.ascontiguousarray(base_prediction)
            )
        if brain_mask is not None:
            result["brain_mask"] = torch.from_numpy(
                np.ascontiguousarray(brain_mask)
            )
        if clean_input is not None:
            result["clean_input"] = torch.from_numpy(
                np.ascontiguousarray(clean_input)
            )
        return result


def load_fold_subjects(folds_path: Path, fold: int) -> tuple[list[str], list[str]]:
    payload: dict[str, Any] = json.loads(Path(folds_path).read_text(encoding="utf-8"))
    assignment = {subject: int(value) for subject, value in payload["assignment"].items()}
    validation = sorted(subject for subject, assigned_fold in assignment.items() if assigned_fold == fold)
    training = sorted(subject for subject, assigned_fold in assignment.items() if assigned_fold != fold)
    if not validation or not training:
        raise ValueError(f"fold {fold} has an empty training or validation split")
    return training, validation


def resolve_training_subjects(
    folds_path: Path,
    *,
    fold: int | None,
    train_all: bool,
) -> tuple[list[str], list[str]]:
    """Resolve a CV split or all-subject training with monitoring-only validation."""
    if train_all:
        if fold is not None:
            raise ValueError("fold and train_all are mutually exclusive")
        payload: dict[str, Any] = json.loads(
            Path(folds_path).read_text(encoding="utf-8")
        )
        subjects = sorted(str(subject) for subject in payload["assignment"])
        if not subjects:
            raise ValueError("fold assignment has no subjects")
        return subjects, subjects
    if fold is None:
        raise ValueError("fold is required unless train_all is enabled")
    return load_fold_subjects(folds_path, fold)
