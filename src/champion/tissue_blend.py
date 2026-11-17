"""Evaluate tissue-gated axial residual blends of cached OOF mu-map predictions."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


TISSUE_THRESHOLDS = np.asarray((0.0672, 0.0912, 0.1134), dtype=np.float32)
TISSUE_NAMES = ("air_lung", "fat", "soft_tissue", "bone")
MAX_TISSUE_ALPHA = 4.0


@dataclass(frozen=True)
class TissueBlendCandidate:
    name: str
    air_alpha: float
    fat_alpha: float
    soft_alpha: float
    bone_alpha: float
    head_alpha: float
    transition_start: int
    transition_end: int
    support_low: float | None = None
    support_high: float | None = None

    @property
    def body_alphas(self) -> np.ndarray:
        return np.asarray(
            (self.air_alpha, self.fat_alpha, self.soft_alpha, self.bone_alpha),
            dtype=np.float32,
        )

    @classmethod
    def parse(cls, value: str) -> "TissueBlendCandidate":
        fields = value.split(":")
        if len(fields) not in (8, 10):
            raise argparse.ArgumentTypeError(
                "candidate must be "
                "NAME:AIR:FAT:SOFT:BONE:HEAD:START_SLICE:END_SLICE"
                "[:SUPPORT_LOW:SUPPORT_HIGH]"
            )
        name, air, fat, soft, bone, head, start, end = fields[:8]
        support_low, support_high = (
            (float(fields[8]), float(fields[9]))
            if len(fields) == 10
            else (None, None)
        )
        valid_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        if not name or any(character not in valid_characters for character in name):
            raise argparse.ArgumentTypeError(f"invalid candidate name: {name!r}")
        candidate = cls(
            name=name,
            air_alpha=float(air),
            fat_alpha=float(fat),
            soft_alpha=float(soft),
            bone_alpha=float(bone),
            head_alpha=float(head),
            transition_start=int(start),
            transition_end=int(end),
            support_low=support_low,
            support_high=support_high,
        )
        if any(
            not np.isfinite(alpha) or alpha < 0.0 or alpha > MAX_TISSUE_ALPHA
            for alpha in (*candidate.body_alphas, candidate.head_alpha)
        ):
            raise argparse.ArgumentTypeError(
                f"all alpha values must be between zero and {MAX_TISSUE_ALPHA:g}"
            )
        if candidate.transition_start > candidate.transition_end:
            raise argparse.ArgumentTypeError("transition start must not exceed transition end")
        if candidate.support_low is not None and (
            not np.isfinite(candidate.support_low)
            or not np.isfinite(candidate.support_high)
            or candidate.support_low < 0.0
            or candidate.support_high <= candidate.support_low
        ):
            raise argparse.ArgumentTypeError(
                "support thresholds must be finite and satisfy "
                "0 <= SUPPORT_LOW < SUPPORT_HIGH"
            )
        return candidate


def head_weight_profile(length: int, transition_start: int, transition_end: int) -> np.ndarray:
    """Return a smooth body-to-head interpolation weight in [0, 1]."""
    if length <= 0:
        raise ValueError("length must be positive")
    start = max(0, min(transition_start, length))
    end = max(0, min(transition_end, length))
    profile = np.zeros(length, dtype=np.float32)
    if end <= start:
        profile[start:] = 1.0
        return profile
    coordinate = np.arange(start, end, dtype=np.float32)
    coordinate = (coordinate - start) / float(end - start)
    profile[start:end] = coordinate * coordinate * (3.0 - 2.0 * coordinate)
    profile[end:] = 1.0
    return profile


def support_weight_profile(
    anchor: np.ndarray,
    support_low: float,
    support_high: float,
) -> np.ndarray:
    """Return a smooth inference-only foreground support weight in [0, 1]."""
    if (
        not np.isfinite(support_low)
        or not np.isfinite(support_high)
        or support_low < 0.0
        or support_high <= support_low
    ):
        raise ValueError(
            "support thresholds must be finite and satisfy "
            "0 <= support_low < support_high"
        )
    weight = np.asarray(anchor, dtype=np.float32) - np.float32(support_low)
    weight /= np.float32(support_high - support_low)
    np.clip(weight, 0.0, 1.0, out=weight)
    return weight * weight * (3.0 - 2.0 * weight)


def alpha_chunk(
    anchor: np.ndarray,
    candidate: TissueBlendCandidate,
    start_slice: int,
    total_slices: int,
) -> np.ndarray:
    """Build inference-available alpha weights from an anchor mu-map."""
    tissue_indices = np.digitize(anchor, TISSUE_THRESHOLDS)
    body_alpha = candidate.body_alphas[tissue_indices]
    end_slice = start_slice + anchor.shape[2]
    head_weight = head_weight_profile(
        total_slices,
        candidate.transition_start,
        candidate.transition_end,
    )[start_slice:end_slice][None, None, :]
    alpha = body_alpha + head_weight * (candidate.head_alpha - body_alpha)
    if candidate.support_low is not None:
        alpha *= support_weight_profile(
            anchor,
            candidate.support_low,
            candidate.support_high,
        )
    return alpha


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
        "--candidate", type=TissueBlendCandidate.parse, action="append", required=True
    )
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument("--materialize")
    parser.add_argument("--materialize-dir", type=Path)
    parser.add_argument(
        "--residual-fwhm-mm",
        type=float,
        help="Optionally low-pass filter the B-minus-A correction",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Compute only eval-mask mu-map metrics for faster candidate screening",
    )
    parser.add_argument("--chunk-slices", type=int, default=16)
    return parser.parse_args()


def _atomic_memmap(path: Path, shape: Sequence[int]) -> tuple[Path, np.memmap]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    return temporary, np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16, shape=tuple(shape)
    )


def main() -> None:
    args = parse_args()
    if args.chunk_slices <= 0:
        raise ValueError("--chunk-slices must be positive")
    if args.residual_fwhm_mm is not None and args.residual_fwhm_mm <= 0.0:
        raise ValueError("--residual-fwhm-mm must be positive")
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

    results = []
    for subject in subjects:
        fold = int(folds[subject])
        cache = args.cache_dir / "train" / subject
        target = np.load(cache / "mu.npy", mmap_mode="r")
        eval_mask = np.load(cache / "eval_mask.npy", mmap_mode="r")
        prediction_mask = (
            None
            if args.eval_only
            else np.load(cache / "prediction_mask.npy", mmap_mode="r")
        )
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
        arrays = [eval_mask, base_prediction, prediction_a, prediction_b]
        if prediction_mask is not None:
            arrays.append(prediction_mask)
        if any(array.shape != target.shape for array in arrays):
            raise ValueError(f"{subject}: cached array shapes do not match")

        filtered_residual = None
        if args.residual_fwhm_mm is not None:
            from scipy.ndimage import gaussian_filter

            metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
            residual = np.empty(target.shape, dtype=np.float32)
            np.subtract(prediction_b, prediction_a, out=residual, casting="unsafe")
            sigma = [
                args.residual_fwhm_mm / (2.355 * float(zoom))
                for zoom in metadata["zooms"]
            ]
            filtered_residual = gaussian_filter(residual, sigma=sigma)
            del residual

        candidate_count = len(args.candidate)
        eval_abs_sums = np.zeros(candidate_count, dtype=np.float64)
        eval_signed_sums = np.zeros(candidate_count, dtype=np.float64)
        prediction_abs_sums = np.zeros(candidate_count, dtype=np.float64)
        projection_sums = np.zeros(candidate_count, dtype=np.float64)
        acf_sums = np.zeros(candidate_count, dtype=np.float64)
        tissue_abs_sums = np.zeros((candidate_count, 4), dtype=np.float64)
        tissue_signed_sums = np.zeros((candidate_count, 4), dtype=np.float64)
        tissue_counts = np.zeros(4, dtype=np.int64)
        eval_count = 0
        prediction_count = 0
        path_count = 0

        output = None
        output_path = None
        temporary = None
        if selected_index is not None:
            output_path = args.materialize_dir / subject / "mu.npy"
            temporary, output = _atomic_memmap(output_path, target.shape)

        for start in range(0, target.shape[2], args.chunk_slices):
            end = min(target.shape[2], start + args.chunk_slices)
            slices = np.s_[:, :, start:end]
            target_chunk = np.asarray(target[slices], dtype=np.float32)
            anchor = np.asarray(base_prediction[slices], dtype=np.float32)
            residual_chunk = (
                filtered_residual[slices]
                if filtered_residual is not None
                else np.asarray(prediction_b[slices], dtype=np.float32)
                - np.asarray(prediction_a[slices], dtype=np.float32)
            )
            active_eval = np.asarray(eval_mask[slices], dtype=bool)
            tissue_indices = np.digitize(anchor, TISSUE_THRESHOLDS)

            eval_count += int(active_eval.sum())
            if args.eval_only:
                eval_anchor = anchor[active_eval]
                eval_residual = residual_chunk[active_eval]
                eval_target = target_chunk[active_eval]
                eval_tissues = tissue_indices[active_eval]
                eval_z_indices = np.nonzero(active_eval)[2]
            path_metadata = []
            if not args.eval_only:
                assert prediction_mask is not None
                active_prediction = np.asarray(prediction_mask[slices], dtype=bool)
                prediction_count += int(active_prediction.sum())
                for tissue_index in range(4):
                    tissue_counts[tissue_index] += int(
                        np.count_nonzero(active_eval & (tissue_indices == tissue_index))
                    )

                for axis in (0, 1):
                    counts = active_prediction.sum(axis=axis)
                    valid = counts > 0
                    path_metadata.append((axis, counts, valid))
                    path_count += int(valid.sum())

            for index, candidate in enumerate(args.candidate):
                if args.eval_only:
                    body_alpha = candidate.body_alphas[eval_tissues]
                    head_profile = head_weight_profile(
                        target.shape[2],
                        candidate.transition_start,
                        candidate.transition_end,
                    )[start:end]
                    head_weight = head_profile[eval_z_indices]
                    eval_alpha = body_alpha + head_weight * (
                        candidate.head_alpha - body_alpha
                    )
                    if candidate.support_low is not None:
                        eval_alpha *= support_weight_profile(
                            eval_anchor,
                            candidate.support_low,
                            candidate.support_high,
                        )
                    blended_eval = eval_anchor + eval_alpha * eval_residual
                    np.maximum(blended_eval, 0.0, out=blended_eval)
                    eval_errors = blended_eval - eval_target
                    eval_abs_sums[index] += np.abs(eval_errors).sum(
                        dtype=np.float64
                    )
                    eval_signed_sums[index] += eval_errors.sum(dtype=np.float64)

                    if index == selected_index and output is not None:
                        alpha = alpha_chunk(
                            anchor, candidate, start, target.shape[2]
                        )
                        blended = anchor + alpha * residual_chunk
                        np.maximum(blended, 0.0, out=blended)
                        output[slices] = blended.astype(np.float16)
                    continue

                alpha = alpha_chunk(anchor, candidate, start, target.shape[2])
                blended = anchor + alpha * residual_chunk
                np.maximum(blended, 0.0, out=blended)
                error = blended - target_chunk

                eval_errors = error[active_eval]
                eval_abs_sums[index] += np.abs(eval_errors).sum(dtype=np.float64)
                eval_signed_sums[index] += eval_errors.sum(dtype=np.float64)
                prediction_abs_sums[index] += np.abs(
                    error[active_prediction]
                ).sum(dtype=np.float64)
                for tissue_index in range(4):
                    active_tissue = active_eval & (
                        tissue_indices == tissue_index
                    )
                    values = error[active_tissue]
                    tissue_abs_sums[index, tissue_index] += np.abs(values).sum(
                        dtype=np.float64
                    )
                    tissue_signed_sums[index, tissue_index] += values.sum(
                        dtype=np.float64
                    )

                active_error = error * active_prediction
                for axis, counts, valid in path_metadata:
                    signed_paths = active_error.sum(axis=axis, dtype=np.float64)
                    normalized = np.abs(signed_paths) / np.maximum(counts, 1)
                    projection_sums[index] += normalized[valid].sum(
                        dtype=np.float64
                    )
                    line_integral = np.clip(
                        signed_paths * 0.15234375, -2.0, 2.0
                    )
                    acf_sums[index] += np.abs(
                        np.expm1(line_integral[valid])
                    ).sum(dtype=np.float64)

                if index == selected_index and output is not None:
                    output[slices] = blended.astype(np.float16)

        if output is not None and output_path is not None and temporary is not None:
            output.flush()
            del output
            os.replace(temporary, output_path)

        subject_results = []
        for index, candidate in enumerate(args.candidate):
            row = {
                "name": candidate.name,
                "air_alpha": candidate.air_alpha,
                "fat_alpha": candidate.fat_alpha,
                "soft_alpha": candidate.soft_alpha,
                "bone_alpha": candidate.bone_alpha,
                "head_alpha": candidate.head_alpha,
                "transition_start": candidate.transition_start,
                "transition_end": candidate.transition_end,
                "support_low": candidate.support_low,
                "support_high": candidate.support_high,
                "eval_mu_mae": float(eval_abs_sums[index] / eval_count),
                "eval_mu_bias": float(eval_signed_sums[index] / eval_count),
            }
            if not args.eval_only:
                row.update(
                    {
                        "prediction_mu_mae": float(
                            prediction_abs_sums[index] / prediction_count
                        ),
                        "projection_path_l1": float(
                            projection_sums[index] / path_count
                        ),
                        "acf_path_relative_error": float(
                            acf_sums[index] / path_count
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
                )
            subject_results.append(row)
        results.append({"subject": subject, "fold": fold, "candidates": subject_results})
        print(f"[{len(results)}/{len(subjects)}] {subject}")

        del filtered_residual

    payload = {
        "subjects": subjects,
        "subject_count": len(subjects),
        "eval_only": args.eval_only,
        "residual_fwhm_mm": args.residual_fwhm_mm,
        "base_prediction_folds": (
            [str(path) for path in args.base_prediction_folds]
            if args.base_prediction_folds is not None
            else None
        ),
        "prediction_a_folds": [str(path) for path in args.prediction_a_folds],
        "prediction_b_folds": [str(path) for path in args.prediction_b_folds],
        "materialized": args.materialize,
        "per_subject": results,
    }
    args.analysis_output.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"subjects": subjects, "materialized": args.materialize}, indent=2))


if __name__ == "__main__":
    main()
