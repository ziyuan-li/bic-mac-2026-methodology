"""Apply the selected projection blend and tissue calibration without labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from champion.mu_calibration import MuCalibrationCandidate, calibration_offset
from champion.projection_blend import neutralize_axial_projections, projection_blend
from champion.spatial_blend import BlendCandidate, axial_alpha_profile


def transform_chunk(
    base: np.ndarray,
    alternative: np.ndarray,
    alpha: np.ndarray,
    projection_strength: float,
    mask_threshold: float,
    projection_iterations: int,
    calibration: MuCalibrationCandidate,
    start_slice: int,
    total_slices: int,
) -> np.ndarray:
    """Apply the inference-only E28b transformation to one axial chunk."""
    if base.shape != alternative.shape:
        raise ValueError("base and alternative shapes must match")
    correction = alternative - base
    neutral = neutralize_axial_projections(
        correction,
        base > mask_threshold,
        iterations=projection_iterations,
    )
    blended = projection_blend(
        base,
        correction,
        neutral,
        alpha,
        projection_strength,
    )
    np.maximum(blended, 0.0, out=blended)
    # Match the two-stage OOF path, where the projection blend is cached first.
    blended = blended.astype(np.float16).astype(np.float32)
    blended += calibration_offset(
        blended,
        calibration,
        start_slice,
        total_slices,
    )
    np.maximum(blended, 0.0, out=blended)
    return blended


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--alternative-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--body-alpha", type=float, default=0.5)
    parser.add_argument("--head-alpha", type=float, default=0.0)
    parser.add_argument("--projection-strength", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.01)
    parser.add_argument("--projection-iterations", type=int, default=4)
    parser.add_argument("--transition-start", type=int, default=370)
    parser.add_argument("--transition-end", type=int, default=390)
    parser.add_argument(
        "--calibration",
        type=MuCalibrationCandidate.parse,
        default=MuCalibrationCandidate.parse(
            "sf-n250-100:0:-0.0001:-0.00025:0:0:370:390"
        ),
    )
    parser.add_argument("--chunk-slices", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.body_alpha <= 2.0:
        raise ValueError("--body-alpha must be between zero and two")
    if not 0.0 <= args.head_alpha <= 2.0:
        raise ValueError("--head-alpha must be between zero and two")
    if not 0.0 <= args.projection_strength <= 1.0:
        raise ValueError("--projection-strength must be between zero and one")
    if args.mask_threshold < 0.0:
        raise ValueError("--mask-threshold must be non-negative")
    if args.projection_iterations <= 0:
        raise ValueError("--projection-iterations must be positive")
    if args.chunk_slices <= 0:
        raise ValueError("--chunk-slices must be positive")

    rows = []
    for index, subject in enumerate(args.subjects, start=1):
        base_path = args.base_dir / subject / "mu.npy"
        alternative_path = args.alternative_dir / subject / "mu.npy"
        base = np.load(base_path, mmap_mode="r")
        alternative = np.load(alternative_path, mmap_mode="r")
        if base.shape != alternative.shape:
            raise ValueError(f"{subject}: prediction shapes do not match")

        output_path = args.output_dir / subject / "mu.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()
        output = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float16,
            shape=base.shape,
        )
        profile = axial_alpha_profile(
            base.shape[2],
            BlendCandidate(
                "selected",
                args.body_alpha,
                args.head_alpha,
                args.transition_start,
                args.transition_end,
            ),
        )
        minimum = float("inf")
        maximum = float("-inf")
        for start in range(0, base.shape[2], args.chunk_slices):
            end = min(base.shape[2], start + args.chunk_slices)
            slices = np.s_[:, :, start:end]
            transformed = transform_chunk(
                np.asarray(base[slices], dtype=np.float32),
                np.asarray(alternative[slices], dtype=np.float32),
                profile[None, None, start:end],
                args.projection_strength,
                args.mask_threshold,
                args.projection_iterations,
                args.calibration,
                start,
                base.shape[2],
            )
            output[slices] = transformed.astype(np.float16)
            minimum = min(minimum, float(np.min(transformed)))
            maximum = max(maximum, float(np.max(transformed)))
        output.flush()
        del output
        os.replace(temporary, output_path)
        rows.append(
            {
                "subject": subject,
                "shape": list(base.shape),
                "minimum": minimum,
                "maximum": maximum,
                "output": str(output_path),
            }
        )
        print(f"[{index}/{len(args.subjects)}] {subject}", flush=True)

    summary = {
        "subjects": args.subjects,
        "base_dir": str(args.base_dir),
        "alternative_dir": str(args.alternative_dir),
        "body_alpha": args.body_alpha,
        "head_alpha": args.head_alpha,
        "projection_strength": args.projection_strength,
        "mask_threshold": args.mask_threshold,
        "projection_iterations": args.projection_iterations,
        "transition_start": args.transition_start,
        "transition_end": args.transition_end,
        "calibration": args.calibration.__dict__,
        "outputs": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
