"""Evaluate projection-neutral residual corrections for cached OOF mu-maps."""

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
class ProjectionBlendCandidate:
    name: str
    body_alpha: float
    head_alpha: float
    projection_strength: float
    mask_threshold: float
    transition_start: int
    transition_end: int

    @classmethod
    def parse(cls, value: str) -> "ProjectionBlendCandidate":
        fields = value.split(":")
        if len(fields) != 7:
            raise argparse.ArgumentTypeError(
                "candidate must be "
                "NAME:BODY_ALPHA:HEAD_ALPHA:PROJECTION_STRENGTH:"
                "MASK_THRESHOLD:START_SLICE:END_SLICE"
            )
        name, body, head, strength, threshold, start, end = fields
        valid_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        if not name or any(character not in valid_characters for character in name):
            raise argparse.ArgumentTypeError(f"invalid candidate name: {name!r}")
        candidate = cls(
            name=name,
            body_alpha=float(body),
            head_alpha=float(head),
            projection_strength=float(strength),
            mask_threshold=float(threshold),
            transition_start=int(start),
            transition_end=int(end),
        )
        if not 0.0 <= candidate.body_alpha <= 2.0:
            raise argparse.ArgumentTypeError("body alpha must be between zero and two")
        if not 0.0 <= candidate.head_alpha <= 2.0:
            raise argparse.ArgumentTypeError("head alpha must be between zero and two")
        if not 0.0 <= candidate.projection_strength <= 1.0:
            raise argparse.ArgumentTypeError(
                "projection strength must be between zero and one"
            )
        if not 0.0 <= candidate.mask_threshold <= 0.2:
            raise argparse.ArgumentTypeError("mask threshold must be between zero and 0.2")
        if candidate.transition_start > candidate.transition_end:
            raise argparse.ArgumentTypeError(
                "transition start must not exceed transition end"
            )
        return candidate


def neutralize_axial_projections(
    correction: np.ndarray,
    mask: np.ndarray,
    iterations: int = 4,
) -> np.ndarray:
    """Remove row and column means from each masked axial correction slice."""
    if correction.shape != mask.shape:
        raise ValueError("correction and mask shapes must match")
    if correction.ndim != 3:
        raise ValueError("correction and mask must be three-dimensional")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    active = np.asarray(mask, dtype=bool)
    result = np.where(active, correction, 0.0).astype(np.float32, copy=False)
    for _ in range(iterations):
        for axis in (0, 1):
            counts = active.sum(axis=axis, keepdims=True, dtype=np.int32)
            sums = result.sum(axis=axis, keepdims=True, dtype=np.float64)
            means = np.divide(
                sums,
                counts,
                out=np.zeros_like(sums, dtype=np.float64),
                where=counts > 0,
            ).astype(np.float32)
            result -= means
            result *= active
    return result


def projection_blend(
    base: np.ndarray,
    correction: np.ndarray,
    neutral_correction: np.ndarray,
    alpha: np.ndarray | float,
    projection_strength: float,
) -> np.ndarray:
    """Apply a residual correction with optional projection neutralization."""
    adjusted = correction + projection_strength * (neutral_correction - correction)
    return base + alpha * adjusted


def _profile(length: int, candidate: ProjectionBlendCandidate) -> np.ndarray:
    return axial_alpha_profile(
        length,
        BlendCandidate(
            candidate.name,
            candidate.body_alpha,
            candidate.head_alpha,
            candidate.transition_start,
            candidate.transition_end,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--folds-path", type=Path, required=True)
    parser.add_argument("--base-prediction-folds", type=Path, nargs=5, required=True)
    parser.add_argument("--prediction-a-folds", type=Path, nargs=5, required=True)
    parser.add_argument("--prediction-b-folds", type=Path, nargs=5, required=True)
    parser.add_argument(
        "--candidate", type=ProjectionBlendCandidate.parse, action="append", required=True
    )
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--correction-mode", choices=("full", "highpass"), required=True)
    parser.add_argument("--fwhm-mm", type=float, default=8.0)
    parser.add_argument("--projection-iterations", type=int, default=4)
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
    if args.correction_mode == "highpass" and args.fwhm_mm <= 0.0:
        raise ValueError("--fwhm-mm must be positive for highpass corrections")
    if args.projection_iterations <= 0:
        raise ValueError("--projection-iterations must be positive")
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

    folds = json.loads(args.folds_path.read_text(encoding="utf-8"))["assignment"]
    subjects = args.subjects or sorted(folds)
    unknown = sorted(set(subjects) - set(folds))
    if unknown:
        raise ValueError(f"subjects missing from fold assignment: {unknown}")

    gaussian_filter = None
    if args.correction_mode == "highpass":
        from scipy.ndimage import gaussian_filter as scipy_gaussian_filter

        gaussian_filter = scipy_gaussian_filter

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
        base_prediction = np.load(
            args.base_prediction_folds[fold] / subject / "mu.npy", mmap_mode="r"
        )
        arrays = (eval_mask, prediction_mask, prediction_a, prediction_b, base_prediction)
        if any(array.shape != target.shape for array in arrays):
            raise ValueError(f"{subject}: cached array shapes do not match")

        difference = np.empty(target.shape, dtype=np.float32)
        np.subtract(prediction_b, prediction_a, out=difference, casting="unsafe")
        if args.correction_mode == "highpass":
            zooms = [float(value) for value in metadata["zooms"]]
            sigma = [args.fwhm_mm / (2.355 * zoom) for zoom in zooms]
            smoothed = gaussian_filter(difference, sigma=sigma)
            difference -= smoothed
            del smoothed

        profiles = [_profile(target.shape[2], candidate) for candidate in candidates]
        eval_sums = np.zeros(len(candidates), dtype=np.float64)
        eval_signed_sums = np.zeros(len(candidates), dtype=np.float64)
        prediction_sums = np.zeros(len(candidates), dtype=np.float64)
        projection_sums = np.zeros(len(candidates), dtype=np.float64)
        acf_sums = np.zeros(len(candidates), dtype=np.float64)
        correction_projection_sums = np.zeros(len(candidates), dtype=np.float64)
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
            base = np.asarray(base_prediction[slices], dtype=np.float32)
            correction = difference[slices]
            active_eval = np.asarray(eval_mask[slices], dtype=bool)
            active_prediction = np.asarray(prediction_mask[slices], dtype=bool)
            neutral_by_threshold: dict[float, np.ndarray] = {}
            path_metadata = []
            for axis in (0, 1):
                counts = active_prediction.sum(axis=axis)
                valid = counts > 0
                path_metadata.append((axis, counts, valid))
                path_count += int(valid.sum())

            for index, (candidate, profile) in enumerate(zip(candidates, profiles)):
                if candidate.projection_strength > 0.0:
                    neutral = neutral_by_threshold.get(candidate.mask_threshold)
                    if neutral is None:
                        inference_mask = base > candidate.mask_threshold
                        neutral = neutralize_axial_projections(
                            correction,
                            inference_mask,
                            iterations=args.projection_iterations,
                        )
                        neutral_by_threshold[candidate.mask_threshold] = neutral
                else:
                    neutral = correction
                alpha = profile[None, None, start:end]
                blended = projection_blend(
                    base,
                    correction,
                    neutral,
                    alpha,
                    candidate.projection_strength,
                )
                np.maximum(blended, 0.0, out=blended)
                adjusted_correction = blended - base
                errors = blended - target_chunk
                eval_errors = errors[active_eval]
                eval_sums[index] += np.abs(eval_errors).sum(dtype=np.float64)
                eval_signed_sums[index] += eval_errors.sum(dtype=np.float64)
                prediction_sums[index] += np.abs(errors[active_prediction]).sum(
                    dtype=np.float64
                )
                active_errors = errors * active_prediction
                active_correction = adjusted_correction * active_prediction
                for axis, counts, valid in path_metadata:
                    signed_paths = active_errors.sum(axis=axis, dtype=np.float64)
                    normalized = np.abs(signed_paths) / np.maximum(counts, 1)
                    projection_sums[index] += normalized[valid].sum(dtype=np.float64)
                    line_integral = np.clip(signed_paths * 0.15234375, -2.0, 2.0)
                    acf_sums[index] += np.abs(np.expm1(line_integral[valid])).sum(
                        dtype=np.float64
                    )
                    correction_paths = active_correction.sum(axis=axis, dtype=np.float64)
                    correction_projection_sums[index] += (
                        np.abs(correction_paths[valid]).sum(dtype=np.float64)
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
                "correction_path_l1": float(
                    correction_projection_sums[index] / path_count
                ),
            }
            for index, candidate in enumerate(candidates)
        }
        rows.append({"subject": subject, "fold": fold, "scores": subject_scores})
        print(f"[{len(rows)}/{len(subjects)}] {subject}", flush=True)
        del difference

    aggregate = []
    metric_names = (
        "eval_mu_mae",
        "eval_mu_bias",
        "prediction_mu_mae",
        "projection_path_l1",
        "acf_path_relative_error",
        "correction_path_l1",
    )
    for index, candidate in enumerate(candidates):
        row = {**candidate.__dict__}
        for metric in metric_names:
            row[metric] = float(
                np.mean(
                    [float(item["scores"][candidate.name][metric]) for item in rows]
                )
            )
        row["voxel_weighted_eval_mu_mae"] = float(
            total_eval_sums[index] / total_eval_count
        )
        row["voxel_weighted_prediction_mu_mae"] = float(
            total_prediction_sums[index] / total_prediction_count
        )
        aggregate.append(row)

    payload = {
        "subjects": subjects,
        "subject_count": len(subjects),
        "correction_mode": args.correction_mode,
        "fwhm_mm": args.fwhm_mm if args.correction_mode == "highpass" else None,
        "projection_iterations": args.projection_iterations,
        "base_prediction_folds": [str(path) for path in args.base_prediction_folds],
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
