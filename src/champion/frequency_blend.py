"""Evaluate frequency-separated axial blends of cached OOF mu-map predictions."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from champion.spatial_blend import BlendCandidate, axial_alpha_profile


@dataclass(frozen=True)
class FrequencyBlendCandidate:
    name: str
    low_body_alpha: float
    high_body_alpha: float
    low_head_alpha: float
    high_head_alpha: float
    transition_start: int
    transition_end: int

    @classmethod
    def parse(cls, value: str) -> "FrequencyBlendCandidate":
        fields = value.split(":")
        if len(fields) != 7:
            raise argparse.ArgumentTypeError(
                "candidate must be "
                "NAME:LOW_BODY:HIGH_BODY:LOW_HEAD:HIGH_HEAD:START_SLICE:END_SLICE"
            )
        name, low_body, high_body, low_head, high_head, start, end = fields
        valid_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        if not name or any(character not in valid_characters for character in name):
            raise argparse.ArgumentTypeError(f"invalid candidate name: {name!r}")
        candidate = cls(
            name=name,
            low_body_alpha=float(low_body),
            high_body_alpha=float(high_body),
            low_head_alpha=float(low_head),
            high_head_alpha=float(high_head),
            transition_start=int(start),
            transition_end=int(end),
        )
        alphas = (
            candidate.low_body_alpha,
            candidate.high_body_alpha,
            candidate.low_head_alpha,
            candidate.high_head_alpha,
        )
        if any(not 0.0 <= alpha <= 2.0 for alpha in alphas):
            raise argparse.ArgumentTypeError("all alphas must be between zero and two")
        if candidate.transition_start > candidate.transition_end:
            raise argparse.ArgumentTypeError("transition start must not exceed transition end")
        return candidate


def frequency_blend(
    first: np.ndarray,
    difference: np.ndarray,
    smoothed_difference: np.ndarray,
    low_alpha: np.ndarray | float,
    high_alpha: np.ndarray | float,
) -> np.ndarray:
    """Apply low/high-frequency weights to the B-minus-A correction."""
    return (
        first
        + high_alpha * difference
        + (low_alpha - high_alpha) * smoothed_difference
    )


def _profiles(
    length: int, candidate: FrequencyBlendCandidate
) -> tuple[np.ndarray, np.ndarray]:
    low = BlendCandidate(
        candidate.name,
        candidate.low_body_alpha,
        candidate.low_head_alpha,
        candidate.transition_start,
        candidate.transition_end,
    )
    high = BlendCandidate(
        candidate.name,
        candidate.high_body_alpha,
        candidate.high_head_alpha,
        candidate.transition_start,
        candidate.transition_end,
    )
    return axial_alpha_profile(length, low), axial_alpha_profile(length, high)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--folds-path", type=Path, required=True)
    parser.add_argument(
        "--base-prediction-folds",
        type=Path,
        nargs=5,
        help="Optional fold predictions to receive the B-minus-A correction",
    )
    parser.add_argument("--prediction-a-folds", type=Path, nargs=5, required=True)
    parser.add_argument("--prediction-b-folds", type=Path, nargs=5, required=True)
    parser.add_argument(
        "--candidate", type=FrequencyBlendCandidate.parse, action="append", required=True
    )
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--fwhm-mm", type=float, default=4.0)
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
    if args.fwhm_mm <= 0.0:
        raise ValueError("--fwhm-mm must be positive")
    if args.chunk_slices <= 0:
        raise ValueError("--chunk-slices must be positive")

    candidates = args.candidate
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")
    selected_index = None
    if args.materialize is not None:
        if args.materialize not in names:
            raise ValueError(f"unknown materialization candidate: {args.materialize}")
        if args.materialize_dir is None:
            raise ValueError("--materialize-dir is required with --materialize")
        selected_index = names.index(args.materialize)

    # Import lazily so formula-only unit tests do not require the training environment.
    from scipy.ndimage import gaussian_filter

    folds = json.loads(args.folds_path.read_text(encoding="utf-8"))["assignment"]
    subjects = args.subjects or sorted(folds)
    unknown = sorted(set(subjects) - set(folds))
    if unknown:
        raise ValueError(f"subjects missing from fold assignment: {unknown}")

    rows: list[dict[str, object]] = []
    total_eval_sums = np.zeros(len(candidates), dtype=np.float64)
    total_prediction_sums = np.zeros(len(candidates), dtype=np.float64)
    total_eval_count = 0
    total_prediction_count = 0

    for subject in subjects:
        fold = int(folds[subject])
        cache = args.cache_dir / "train" / subject
        target = np.load(cache / "mu.npy", mmap_mode="r")
        eval_mask = np.load(cache / "eval_mask.npy", mmap_mode="r")
        prediction_mask = np.load(cache / "prediction_mask.npy", mmap_mode="r")
        metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        prediction_a = np.load(
            args.prediction_a_folds[fold] / subject / "mu.npy", mmap_mode="r"
        )
        prediction_b = np.load(
            args.prediction_b_folds[fold] / subject / "mu.npy", mmap_mode="r"
        )
        base_prediction = (
            np.load(
                args.base_prediction_folds[fold] / subject / "mu.npy",
                mmap_mode="r",
            )
            if args.base_prediction_folds is not None
            else prediction_a
        )
        arrays = (eval_mask, prediction_mask, base_prediction, prediction_a, prediction_b)
        if any(array.shape != target.shape for array in arrays):
            raise ValueError(f"{subject}: cached array shapes do not match")

        difference = np.empty(target.shape, dtype=np.float32)
        np.subtract(prediction_b, prediction_a, out=difference, casting="unsafe")
        zooms = [float(value) for value in metadata["zooms"]]
        sigma = [args.fwhm_mm / (2.355 * zoom) for zoom in zooms]
        smoothed_difference = gaussian_filter(difference, sigma=sigma)
        profiles = [_profiles(target.shape[2], candidate) for candidate in candidates]

        eval_sums = np.zeros(len(candidates), dtype=np.float64)
        eval_signed_sums = np.zeros(len(candidates), dtype=np.float64)
        prediction_sums = np.zeros(len(candidates), dtype=np.float64)
        projection_sums = np.zeros(len(candidates), dtype=np.float64)
        acf_sums = np.zeros(len(candidates), dtype=np.float64)
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
            first = np.asarray(base_prediction[slices], dtype=np.float32)
            correction = difference[slices]
            low_correction = smoothed_difference[slices]
            active_eval = np.asarray(eval_mask[slices], dtype=bool)
            active_prediction = np.asarray(prediction_mask[slices], dtype=bool)
            path_metadata = []
            for axis in (0, 1):
                counts = active_prediction.sum(axis=axis)
                valid = counts > 0
                path_metadata.append((axis, counts, valid))
                path_count += int(valid.sum())

            for index, (low_profile, high_profile) in enumerate(profiles):
                low_alpha = low_profile[None, None, start:end]
                high_alpha = high_profile[None, None, start:end]
                blended = frequency_blend(
                    first,
                    correction,
                    low_correction,
                    low_alpha,
                    high_alpha,
                )
                np.maximum(blended, 0.0, out=blended)
                errors = blended - target_chunk
                eval_errors = errors[active_eval]
                eval_sums[index] += np.abs(eval_errors).sum(dtype=np.float64)
                eval_signed_sums[index] += eval_errors.sum(dtype=np.float64)
                prediction_sums[index] += np.abs(errors[active_prediction]).sum(
                    dtype=np.float64
                )
                active_errors = errors * active_prediction
                for axis, counts, valid in path_metadata:
                    signed_paths = active_errors.sum(axis=axis, dtype=np.float64)
                    normalized = np.abs(signed_paths) / np.maximum(counts, 1)
                    projection_sums[index] += normalized[valid].sum(dtype=np.float64)
                    line_integral = np.clip(signed_paths * 0.15234375, -2.0, 2.0)
                    acf_sums[index] += np.abs(np.expm1(line_integral[valid])).sum(
                        dtype=np.float64
                    )
                if index == selected_index:
                    output[slices] = blended.astype(np.float16)

            eval_count += int(active_eval.sum())
            prediction_count += int(active_prediction.sum())

        if output is not None and output_path is not None and temporary is not None:
            output.flush()
            del output
            os.replace(temporary, output_path)

        total_eval_sums += eval_sums
        total_prediction_sums += prediction_sums
        total_eval_count += eval_count
        total_prediction_count += prediction_count
        subject_scores = {
            candidate.name: {
                "eval_mu_mae": float(eval_sums[index] / eval_count),
                "eval_mu_bias": float(eval_signed_sums[index] / eval_count),
                "prediction_mu_mae": float(prediction_sums[index] / prediction_count),
                "projection_path_l1": float(projection_sums[index] / path_count),
                "acf_path_relative_error": float(acf_sums[index] / path_count),
            }
            for index, candidate in enumerate(candidates)
        }
        rows.append({"subject": subject, "fold": fold, "scores": subject_scores})
        print(f"[{len(rows)}/{len(subjects)}] {subject}", flush=True)

        del difference, smoothed_difference

    aggregate = []
    for index, candidate in enumerate(candidates):
        eval_values = [float(row["scores"][candidate.name]["eval_mu_mae"]) for row in rows]
        prediction_values = [
            float(row["scores"][candidate.name]["prediction_mu_mae"]) for row in rows
        ]
        eval_bias_values = [
            float(row["scores"][candidate.name]["eval_mu_bias"]) for row in rows
        ]
        projection_values = [
            float(row["scores"][candidate.name]["projection_path_l1"]) for row in rows
        ]
        acf_values = [
            float(row["scores"][candidate.name]["acf_path_relative_error"]) for row in rows
        ]
        aggregate.append(
            {
                **candidate.__dict__,
                "eval_mu_mae": float(np.mean(eval_values)),
                "eval_mu_bias": float(np.mean(eval_bias_values)),
                "prediction_mu_mae": float(np.mean(prediction_values)),
                "projection_path_l1": float(np.mean(projection_values)),
                "acf_path_relative_error": float(np.mean(acf_values)),
                "voxel_weighted_eval_mu_mae": float(total_eval_sums[index] / total_eval_count),
                "voxel_weighted_prediction_mu_mae": float(
                    total_prediction_sums[index] / total_prediction_count
                ),
            }
        )

    payload = {
        "subjects": subjects,
        "subject_count": len(subjects),
        "fwhm_mm": args.fwhm_mm,
        "base_prediction_folds": (
            [str(path) for path in args.base_prediction_folds]
            if args.base_prediction_folds is not None
            else None
        ),
        "prediction_a_folds": [str(path) for path in args.prediction_a_folds],
        "prediction_b_folds": [str(path) for path in args.prediction_b_folds],
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
