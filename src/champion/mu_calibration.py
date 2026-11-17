"""Evaluate small tissue-aware calibration offsets on cached OOF mu-maps."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from champion.tissue_blend import TISSUE_NAMES, TISSUE_THRESHOLDS, head_weight_profile


@dataclass(frozen=True)
class MuCalibrationCandidate:
    name: str
    air_offset: float
    fat_offset: float
    soft_offset: float
    bone_offset: float
    head_fraction: float
    transition_start: int
    transition_end: int
    lower_transition_start: int = 0
    lower_transition_end: int = 0

    @property
    def body_offsets(self) -> np.ndarray:
        return np.asarray(
            (self.air_offset, self.fat_offset, self.soft_offset, self.bone_offset),
            dtype=np.float32,
        )

    @classmethod
    def parse(cls, value: str) -> "MuCalibrationCandidate":
        fields = value.split(":")
        if len(fields) not in (8, 10):
            raise argparse.ArgumentTypeError(
                "candidate must be "
                "NAME:AIR_OFFSET:FAT_OFFSET:SOFT_OFFSET:BONE_OFFSET:"
                "HEAD_FRACTION:UPPER_START:UPPER_END or "
                "NAME:AIR_OFFSET:FAT_OFFSET:SOFT_OFFSET:BONE_OFFSET:"
                "HEAD_FRACTION:LOWER_START:LOWER_END:UPPER_START:UPPER_END"
            )
        if len(fields) == 8:
            name, air, fat, soft, bone, head, start, end = fields
            lower_start, lower_end = "0", "0"
        else:
            (
                name,
                air,
                fat,
                soft,
                bone,
                head,
                lower_start,
                lower_end,
                start,
                end,
            ) = fields
        valid_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        if not name or any(character not in valid_characters for character in name):
            raise argparse.ArgumentTypeError(f"invalid candidate name: {name!r}")
        candidate = cls(
            name=name,
            air_offset=float(air),
            fat_offset=float(fat),
            soft_offset=float(soft),
            bone_offset=float(bone),
            head_fraction=float(head),
            transition_start=int(start),
            transition_end=int(end),
            lower_transition_start=int(lower_start),
            lower_transition_end=int(lower_end),
        )
        if any(abs(offset) > 0.005 for offset in candidate.body_offsets):
            raise argparse.ArgumentTypeError("offset magnitudes must not exceed 0.005")
        if not 0.0 <= candidate.head_fraction <= 1.0:
            raise argparse.ArgumentTypeError("head fraction must be between zero and one")
        if candidate.transition_start > candidate.transition_end:
            raise argparse.ArgumentTypeError(
                "transition start must not exceed transition end"
            )
        if candidate.lower_transition_start > candidate.lower_transition_end:
            raise argparse.ArgumentTypeError(
                "lower transition start must not exceed lower transition end"
            )
        return candidate


def calibration_offset(
    anchor: np.ndarray,
    candidate: MuCalibrationCandidate,
    start_slice: int,
    total_slices: int,
) -> np.ndarray:
    """Build an inference-safe tissue offset with smooth axial tapers."""
    tissue_indices = np.digitize(anchor, TISSUE_THRESHOLDS)
    body_offset = candidate.body_offsets[tissue_indices]
    end_slice = start_slice + anchor.shape[2]
    head_weight = head_weight_profile(
        total_slices,
        candidate.transition_start,
        candidate.transition_end,
    )[start_slice:end_slice][None, None, :]
    lower_weight = head_weight_profile(
        total_slices,
        candidate.lower_transition_start,
        candidate.lower_transition_end,
    )[start_slice:end_slice][None, None, :]
    upper_multiplier = 1.0 - head_weight * (1.0 - candidate.head_fraction)
    multiplier = lower_weight * upper_multiplier
    return body_offset * multiplier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--folds-path", type=Path, required=True)
    parser.add_argument("--prediction-folds", type=Path, nargs=5, required=True)
    parser.add_argument(
        "--candidate", type=MuCalibrationCandidate.parse, action="append", required=True
    )
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument("--materialize")
    parser.add_argument("--materialize-dir", type=Path)
    parser.add_argument("--chunk-slices", type=int, default=16)
    return parser.parse_args()


def _atomic_memmap(path: Path, shape: Sequence[int]) -> tuple[Path, np.memmap]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    return temporary, np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float16,
        shape=tuple(shape),
    )


def main() -> None:
    args = parse_args()
    if args.chunk_slices <= 0:
        raise ValueError("--chunk-slices must be positive")
    names = [candidate.name for candidate in args.candidate]
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")
    selected_index = None
    if args.materialize is not None:
        if args.materialize not in names:
            raise ValueError(f"unknown materialization candidate: {args.materialize}")
        if args.materialize_dir is None:
            raise ValueError("--materialize-dir is required with --materialize")
        selected_index = names.index(args.materialize)

    folds = json.loads(args.folds_path.read_text(encoding="utf-8"))["assignment"]
    subjects = args.subjects or sorted(folds)
    unknown = sorted(set(subjects) - set(folds))
    if unknown:
        raise ValueError(f"subjects missing from fold assignment: {unknown}")

    rows: list[dict[str, object]] = []
    for subject in subjects:
        fold = int(folds[subject])
        cache = args.cache_dir / "train" / subject
        target = np.load(cache / "mu.npy", mmap_mode="r")
        eval_mask = np.load(cache / "eval_mask.npy", mmap_mode="r")
        prediction_mask = np.load(cache / "prediction_mask.npy", mmap_mode="r")
        prediction = np.load(
            args.prediction_folds[fold] / subject / "mu.npy", mmap_mode="r"
        )
        if any(
            array.shape != target.shape
            for array in (eval_mask, prediction_mask, prediction)
        ):
            raise ValueError(f"{subject}: cached array shapes do not match")

        count = len(args.candidate)
        eval_abs_sums = np.zeros(count, dtype=np.float64)
        eval_signed_sums = np.zeros(count, dtype=np.float64)
        prediction_abs_sums = np.zeros(count, dtype=np.float64)
        projection_sums = np.zeros(count, dtype=np.float64)
        acf_sums = np.zeros(count, dtype=np.float64)
        correction_path_sums = np.zeros(count, dtype=np.float64)
        tissue_abs_sums = np.zeros((count, 4), dtype=np.float64)
        tissue_signed_sums = np.zeros((count, 4), dtype=np.float64)
        tissue_counts = np.zeros(4, dtype=np.int64)
        eval_count = 0
        prediction_count = 0
        path_count = 0

        output_path = None
        output = None
        temporary = None
        if selected_index is not None:
            output_path = args.materialize_dir / subject / "mu.npy"
            temporary, output = _atomic_memmap(output_path, target.shape)

        for start in range(0, target.shape[2], args.chunk_slices):
            end = min(target.shape[2], start + args.chunk_slices)
            slices = np.s_[:, :, start:end]
            target_chunk = np.asarray(target[slices], dtype=np.float32)
            anchor = np.asarray(prediction[slices], dtype=np.float32)
            active_eval = np.asarray(eval_mask[slices], dtype=bool)
            active_prediction = np.asarray(prediction_mask[slices], dtype=bool)
            tissue_indices = np.digitize(anchor, TISSUE_THRESHOLDS)

            eval_count += int(active_eval.sum())
            prediction_count += int(active_prediction.sum())
            for tissue_index in range(4):
                tissue_counts[tissue_index] += int(
                    np.count_nonzero(active_eval & (tissue_indices == tissue_index))
                )
            path_metadata = []
            for axis in (0, 1):
                counts = active_prediction.sum(axis=axis)
                valid = counts > 0
                path_metadata.append((axis, counts, valid))
                path_count += int(valid.sum())

            for index, candidate in enumerate(args.candidate):
                offset = calibration_offset(anchor, candidate, start, target.shape[2])
                calibrated = anchor + offset
                np.maximum(calibrated, 0.0, out=calibrated)
                actual_offset = calibrated - anchor
                error = calibrated - target_chunk
                eval_errors = error[active_eval]
                eval_abs_sums[index] += np.abs(eval_errors).sum(dtype=np.float64)
                eval_signed_sums[index] += eval_errors.sum(dtype=np.float64)
                prediction_abs_sums[index] += np.abs(error[active_prediction]).sum(
                    dtype=np.float64
                )
                for tissue_index in range(4):
                    active_tissue = active_eval & (tissue_indices == tissue_index)
                    values = error[active_tissue]
                    tissue_abs_sums[index, tissue_index] += np.abs(values).sum(
                        dtype=np.float64
                    )
                    tissue_signed_sums[index, tissue_index] += values.sum(
                        dtype=np.float64
                    )

                active_error = error * active_prediction
                active_offset = actual_offset * active_prediction
                for axis, counts, valid in path_metadata:
                    signed_paths = active_error.sum(axis=axis, dtype=np.float64)
                    normalized = np.abs(signed_paths) / np.maximum(counts, 1)
                    projection_sums[index] += normalized[valid].sum(dtype=np.float64)
                    line_integral = np.clip(signed_paths * 0.15234375, -2.0, 2.0)
                    acf_sums[index] += np.abs(np.expm1(line_integral[valid])).sum(
                        dtype=np.float64
                    )
                    correction_paths = active_offset.sum(axis=axis, dtype=np.float64)
                    correction_path_sums[index] += np.abs(correction_paths[valid]).sum(
                        dtype=np.float64
                    )
                if index == selected_index:
                    output[slices] = calibrated.astype(np.float16)

        if output is not None and output_path is not None and temporary is not None:
            output.flush()
            del output
            os.replace(temporary, output_path)

        scores = {}
        for index, candidate in enumerate(args.candidate):
            scores[candidate.name] = {
                **candidate.__dict__,
                "eval_mu_mae": float(eval_abs_sums[index] / eval_count),
                "eval_mu_bias": float(eval_signed_sums[index] / eval_count),
                "prediction_mu_mae": float(
                    prediction_abs_sums[index] / prediction_count
                ),
                "projection_path_l1": float(projection_sums[index] / path_count),
                "acf_path_relative_error": float(acf_sums[index] / path_count),
                "correction_path_l1": float(
                    correction_path_sums[index] / path_count
                ),
                "tissues": {
                    tissue_name: {
                        "voxel_count": int(tissue_counts[tissue_index]),
                        "mae": float(
                            tissue_abs_sums[index, tissue_index]
                            / max(1, tissue_counts[tissue_index])
                        ),
                        "bias": float(
                            tissue_signed_sums[index, tissue_index]
                            / max(1, tissue_counts[tissue_index])
                        ),
                    }
                    for tissue_index, tissue_name in enumerate(TISSUE_NAMES)
                },
            }
        rows.append({"subject": subject, "fold": fold, "scores": scores})
        print(f"[{len(rows)}/{len(subjects)}] {subject}", flush=True)

    aggregate = []
    metrics = (
        "eval_mu_mae",
        "eval_mu_bias",
        "prediction_mu_mae",
        "projection_path_l1",
        "acf_path_relative_error",
        "correction_path_l1",
    )
    for candidate in args.candidate:
        result = {**candidate.__dict__}
        for metric in metrics:
            result[metric] = float(
                np.mean([row["scores"][candidate.name][metric] for row in rows])
            )
        result["tissues"] = {}
        for tissue_name in TISSUE_NAMES:
            total_count = sum(
                row["scores"][candidate.name]["tissues"][tissue_name]["voxel_count"]
                for row in rows
            )
            result["tissues"][tissue_name] = {
                metric: float(
                    sum(
                        row["scores"][candidate.name]["tissues"][tissue_name][metric]
                        * row["scores"][candidate.name]["tissues"][tissue_name][
                            "voxel_count"
                        ]
                        for row in rows
                    )
                    / max(1, total_count)
                )
                for metric in ("mae", "bias")
            }
        aggregate.append(result)

    payload = {
        "subjects": subjects,
        "subject_count": len(subjects),
        "prediction_folds": [str(path) for path in args.prediction_folds],
        "materialized": args.materialize,
        "aggregate": aggregate,
        "per_subject": rows,
    }
    args.analysis_output.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
