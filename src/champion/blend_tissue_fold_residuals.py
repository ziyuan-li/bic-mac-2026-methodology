"""Apply a tissue-gated mean fold residual to validation base predictions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .blend_frequency_fold_predictions import _fold_mean
from .tissue_blend import TissueBlendCandidate, alpha_chunk


def smooth_residual(
    residual: np.ndarray,
    *,
    residual_fwhm_mm: float,
    zooms: list[float] | tuple[float, ...],
) -> np.ndarray:
    """Apply the configured physical smoothing, or preserve a raw residual."""
    if residual_fwhm_mm < 0.0:
        raise ValueError("residual_fwhm_mm must be non-negative")
    if len(zooms) != residual.ndim or any(float(zoom) <= 0.0 for zoom in zooms):
        raise ValueError("zooms must be positive and match the residual dimensions")
    if residual_fwhm_mm == 0.0:
        return residual

    from scipy.ndimage import gaussian_filter

    sigma = [
        residual_fwhm_mm / (2.355 * float(zoom))
        for zoom in zooms
    ]
    return gaussian_filter(residual, sigma=sigma)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--base-prediction-dir", type=Path, required=True)
    parser.add_argument("--prediction-a-folds", type=Path, nargs="+", required=True)
    parser.add_argument("--prediction-b-folds", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=TissueBlendCandidate.parse, required=True)
    parser.add_argument(
        "--residual-fwhm-mm",
        type=float,
        default=4.0,
        help="Residual smoothing FWHM in mm; use 0 to preserve the raw residual",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--chunk-slices", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.prediction_a_folds) != len(args.prediction_b_folds):
        raise ValueError("prediction A and B must have the same number of folds")
    if not args.prediction_a_folds:
        raise ValueError("at least one residual fold is required")
    if args.residual_fwhm_mm < 0.0:
        raise ValueError("--residual-fwhm-mm must be non-negative")
    if args.chunk_slices <= 0:
        raise ValueError("--chunk-slices must be positive")

    subjects = args.subjects or sorted(
        path.name for path in args.base_prediction_dir.glob("sub-*") if path.is_dir()
    )
    rows: list[dict[str, object]] = []
    for subject in subjects:
        base = np.load(
            args.base_prediction_dir / subject / "mu.npy",
            mmap_mode="r",
        )
        prediction_a = [
            np.load(path / subject / "mu.npy", mmap_mode="r")
            for path in args.prediction_a_folds
        ]
        prediction_b = [
            np.load(path / subject / "mu.npy", mmap_mode="r")
            for path in args.prediction_b_folds
        ]
        shape = base.shape
        if any(array.shape != shape for array in prediction_a + prediction_b):
            raise ValueError(f"{subject}: fold prediction shapes differ")

        metadata = json.loads(
            (args.cache_dir / args.split / subject / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        difference = np.empty(shape, dtype=np.float32)
        for start in range(0, shape[2], args.chunk_slices):
            end = min(shape[2], start + args.chunk_slices)
            slices = np.s_[:, :, start:end]
            difference[slices] = _fold_mean(prediction_b, slices) - _fold_mean(
                prediction_a, slices
            )
        filtered_residual = smooth_residual(
            difference,
            residual_fwhm_mm=args.residual_fwhm_mm,
            zooms=metadata["zooms"],
        )
        if filtered_residual is not difference:
            del difference

        output_path = args.output_dir / subject / "mu.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()
        output = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float16,
            shape=shape,
        )

        minimum = float("inf")
        maximum = float("-inf")
        finite = True
        for start in range(0, shape[2], args.chunk_slices):
            end = min(shape[2], start + args.chunk_slices)
            slices = np.s_[:, :, start:end]
            base_chunk = np.asarray(base[slices], dtype=np.float32)
            alpha = alpha_chunk(base_chunk, args.candidate, start, shape[2])
            blended = base_chunk + alpha * filtered_residual[slices]
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
        "residual_fwhm_mm": args.residual_fwhm_mm,
        "base_prediction_dir": str(args.base_prediction_dir),
        "prediction_a_folds": [str(path) for path in args.prediction_a_folds],
        "prediction_b_folds": [str(path) for path in args.prediction_b_folds],
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not all(bool(row["finite"]) for row in rows):
        raise SystemExit("non-finite blended validation prediction")


if __name__ == "__main__":
    main()
