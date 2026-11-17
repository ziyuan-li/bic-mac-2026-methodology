"""Evaluate and materialize axial blends of cached OOF mu-map predictions."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BlendCandidate:
    name: str
    body_alpha: float
    head_alpha: float
    transition_start: int
    transition_end: int

    @classmethod
    def parse(cls, value: str) -> "BlendCandidate":
        fields = value.split(":")
        if len(fields) != 5:
            raise argparse.ArgumentTypeError(
                "candidate must be NAME:BODY_ALPHA:HEAD_ALPHA:START_SLICE:END_SLICE"
            )
        name, body, head, start, end = fields
        valid_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        if not name or any(character not in valid_characters for character in name):
            raise argparse.ArgumentTypeError(f"invalid candidate name: {name!r}")
        candidate = cls(name, float(body), float(head), int(start), int(end))
        if not 0.0 <= candidate.body_alpha <= 2.0:
            raise argparse.ArgumentTypeError("body alpha must be between zero and two")
        if not 0.0 <= candidate.head_alpha <= 2.0:
            raise argparse.ArgumentTypeError("head alpha must be between zero and two")
        if candidate.transition_start > candidate.transition_end:
            raise argparse.ArgumentTypeError("transition start must not exceed transition end")
        return candidate


def axial_alpha_profile(length: int, candidate: BlendCandidate) -> np.ndarray:
    """Return the E08 blend weight, with low-z body and high-z head weights."""
    if length <= 0:
        raise ValueError("length must be positive")
    start = max(0, min(candidate.transition_start, length))
    end = max(0, min(candidate.transition_end, length))
    profile = np.full(length, candidate.body_alpha, dtype=np.float32)
    if end <= start:
        profile[start:] = candidate.head_alpha
        return profile

    # Smoothstep avoids a visible attenuation discontinuity at the neck.
    transition = np.arange(start, end, dtype=np.float32)
    transition = (transition - start) / float(end - start)
    transition = transition * transition * (3.0 - 2.0 * transition)
    profile[start:end] = candidate.body_alpha + transition * (
        candidate.head_alpha - candidate.body_alpha
    )
    profile[end:] = candidate.head_alpha
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--folds-path", type=Path, required=True)
    parser.add_argument("--prediction-a-folds", type=Path, nargs=5, required=True)
    parser.add_argument("--prediction-b-folds", type=Path, nargs=5, required=True)
    parser.add_argument("--candidate", type=BlendCandidate.parse, action="append", required=True)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument("--materialize")
    parser.add_argument("--materialize-dir", type=Path)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Compute only eval-mask metrics while still materializing the selected blend",
    )
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
        arrays = [eval_mask, prediction_a, prediction_b]
        if prediction_mask is not None:
            arrays.append(prediction_mask)
        if any(array.shape != target.shape for array in arrays):
            raise ValueError(f"{subject}: cached array shapes do not match")

        profiles = [
            axial_alpha_profile(target.shape[2], candidate)
            for candidate in candidates
        ]
        eval_sums = np.zeros(len(candidates), dtype=np.float64)
        responsibility_sums = np.zeros(len(candidates), dtype=np.float64)
        eval_count = 0
        responsibility_count = 0

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
            first = np.asarray(prediction_a[slices], dtype=np.float32)
            difference = np.asarray(prediction_b[slices], dtype=np.float32) - first
            active_eval = np.asarray(eval_mask[slices], dtype=bool)
            eval_first = first[active_eval]
            eval_difference = difference[active_eval]
            eval_target = target_chunk[active_eval]
            eval_z_indices = np.nonzero(active_eval)[2]
            if not args.eval_only:
                assert prediction_mask is not None
                active_prediction = np.asarray(prediction_mask[slices], dtype=bool)
                prediction_first = first[active_prediction]
                prediction_difference = difference[active_prediction]
                prediction_target = target_chunk[active_prediction]
                prediction_z_indices = np.nonzero(active_prediction)[2]

            for index, profile in enumerate(profiles):
                eval_alpha = profile[start:end][eval_z_indices]
                blended_eval = eval_first + eval_alpha * eval_difference
                np.maximum(blended_eval, 0.0, out=blended_eval)
                eval_sums[index] += np.abs(
                    blended_eval - eval_target
                ).sum(dtype=np.float64)

                if not args.eval_only:
                    prediction_alpha = profile[start:end][prediction_z_indices]
                    blended_prediction = (
                        prediction_first
                        + prediction_alpha * prediction_difference
                    )
                    np.maximum(
                        blended_prediction, 0.0, out=blended_prediction
                    )
                    responsibility_sums[index] += np.abs(
                        blended_prediction - prediction_target
                    ).sum(dtype=np.float64)

                if index == selected_index:
                    full_alpha = profile[None, None, start:end]
                    blended = first + full_alpha * difference
                    np.maximum(blended, 0.0, out=blended)
                    output[slices] = blended.astype(np.float16)

            eval_count += int(active_eval.sum())
            if not args.eval_only:
                responsibility_count += int(active_prediction.sum())

        if output is not None and output_path is not None and temporary is not None:
            output.flush()
            del output
            os.replace(temporary, output_path)

        total_eval_sums += eval_sums
        total_eval_count += eval_count
        if not args.eval_only:
            total_prediction_sums += responsibility_sums
            total_prediction_count += responsibility_count
        subject_scores = {}
        for index, candidate in enumerate(candidates):
            score = {
                "eval_mu_mae": float(eval_sums[index] / eval_count),
            }
            if not args.eval_only:
                score["prediction_mu_mae"] = float(
                    responsibility_sums[index] / responsibility_count
                )
            subject_scores[candidate.name] = score
        rows.append({"subject": subject, "fold": fold, "scores": subject_scores})
        print(f"[{len(rows)}/{len(subjects)}] {subject}")

    aggregate = []
    for index, candidate in enumerate(candidates):
        eval_values = [
            float(row["scores"][candidate.name]["eval_mu_mae"])
            for row in rows
        ]
        row = {
            "name": candidate.name,
            "body_alpha": candidate.body_alpha,
            "head_alpha": candidate.head_alpha,
            "transition_start": candidate.transition_start,
            "transition_end": candidate.transition_end,
            "eval_mu_mae": float(np.mean(eval_values)),
            "voxel_weighted_eval_mu_mae": float(
                total_eval_sums[index] / total_eval_count
            ),
        }
        if not args.eval_only:
            prediction_values = [
                float(subject["scores"][candidate.name]["prediction_mu_mae"])
                for subject in rows
            ]
            row.update(
                {
                    "prediction_mu_mae": float(np.mean(prediction_values)),
                    "voxel_weighted_prediction_mu_mae": float(
                        total_prediction_sums[index] / total_prediction_count
                    ),
                }
            )
        aggregate.append(row)

    payload = {
        "subjects": subjects,
        "subject_count": len(subjects),
        "eval_only": args.eval_only,
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
