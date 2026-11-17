"""Average validation folds and apply a frequency-separated residual blend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .frequency_blend import FrequencyBlendCandidate, _profiles, frequency_blend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-prediction-folds", type=Path, nargs="+")
    parser.add_argument("--prediction-a-folds", type=Path, nargs="+", required=True)
    parser.add_argument("--prediction-b-folds", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--candidate", type=FrequencyBlendCandidate.parse, required=True
    )
    parser.add_argument("--fwhm-mm", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--chunk-slices", type=int, default=16)
    return parser.parse_args()


def _fold_mean(arrays: list[np.ndarray], slices: object) -> np.ndarray:
    result = np.zeros(np.asarray(arrays[0][slices]).shape, dtype=np.float32)
    for array in arrays:
        result += np.asarray(array[slices], dtype=np.float32)
    result /= len(arrays)
    return result


def main() -> None:
    args = parse_args()
    if len(args.prediction_a_folds) != len(args.prediction_b_folds):
        raise ValueError("prediction A and B must have the same number of folds")
    if args.base_prediction_folds is not None and len(
        args.base_prediction_folds
    ) != len(args.prediction_a_folds):
        raise ValueError(
            "base prediction and residual predictions must have the same number of folds"
        )
    if not args.prediction_a_folds:
        raise ValueError("at least one prediction fold is required")
    if args.fwhm_mm <= 0.0:
        raise ValueError("--fwhm-mm must be positive")
    if args.chunk_slices <= 0:
        raise ValueError("--chunk-slices must be positive")

    from scipy.ndimage import gaussian_filter

    subjects = args.subjects or sorted(
        path.name for path in args.prediction_a_folds[0].glob("sub-*") if path.is_dir()
    )
    rows: list[dict[str, object]] = []
    for subject in subjects:
        first_folds = [
            np.load(path / subject / "mu.npy", mmap_mode="r")
            for path in args.prediction_a_folds
        ]
        second_folds = [
            np.load(path / subject / "mu.npy", mmap_mode="r")
            for path in args.prediction_b_folds
        ]
        base_folds = (
            [
                np.load(path / subject / "mu.npy", mmap_mode="r")
                for path in args.base_prediction_folds
            ]
            if args.base_prediction_folds is not None
            else first_folds
        )
        shape = first_folds[0].shape
        if any(
            array.shape != shape for array in first_folds + second_folds + base_folds
        ):
            raise ValueError(f"{subject}: fold prediction shapes differ")

        metadata_path = args.cache_dir / "val" / subject / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        zooms = [float(value) for value in metadata["zooms"]]
        if len(zooms) != len(shape):
            raise ValueError(f"{subject}: metadata zoom count differs from image rank")
        sigma = [args.fwhm_mm / (2.355 * zoom) for zoom in zooms]

        difference = np.empty(shape, dtype=np.float32)
        for start in range(0, shape[2], args.chunk_slices):
            end = min(shape[2], start + args.chunk_slices)
            slices = np.s_[:, :, start:end]
            difference[slices] = _fold_mean(second_folds, slices) - _fold_mean(
                first_folds, slices
            )
        smoothed_difference = gaussian_filter(difference, sigma=sigma)

        output_path = args.output_dir / subject / "mu.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()
        output = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float16, shape=shape
        )
        low_profile, high_profile = _profiles(shape[2], args.candidate)
        minimum = float("inf")
        maximum = float("-inf")
        finite = True
        for start in range(0, shape[2], args.chunk_slices):
            end = min(shape[2], start + args.chunk_slices)
            slices = np.s_[:, :, start:end]
            first = _fold_mean(base_folds, slices)
            blended = frequency_blend(
                first,
                difference[slices],
                smoothed_difference[slices],
                low_profile[None, None, start:end],
                high_profile[None, None, start:end],
            )
            np.maximum(blended, 0.0, out=blended)
            output[slices] = blended.astype(np.float16)
            minimum = min(minimum, float(np.min(blended)))
            maximum = max(maximum, float(np.max(blended)))
            finite = finite and bool(np.all(np.isfinite(blended)))

        output.flush()
        del output
        os.replace(temporary, output_path)
        rows.append(
            {
                "subject": subject,
                "minimum": minimum,
                "maximum": maximum,
                "finite": finite,
            }
        )
        print(f"[{len(rows)}/{len(subjects)}] {subject}", flush=True)

    payload = {
        "candidate": args.candidate.__dict__,
        "fwhm_mm": args.fwhm_mm,
        "base_prediction_folds": (
            [str(path) for path in args.base_prediction_folds]
            if args.base_prediction_folds is not None
            else None
        ),
        "prediction_a_folds": [str(path) for path in args.prediction_a_folds],
        "prediction_b_folds": [str(path) for path in args.prediction_b_folds],
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not all(bool(row["finite"]) for row in rows):
        raise SystemExit("non-finite blended validation prediction")


if __name__ == "__main__":
    main()
