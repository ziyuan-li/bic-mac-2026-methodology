"""Add robustly normalized Dixon MRI volumes to an existing training cache."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from tqdm import tqdm

from champion.prepare_cache import atomic_save_array, atomic_save_json


MRI_CACHE_FILES = ("mri_in.npy", "mri_out.npy", "mri_metadata.json")


def normalize_mri_phase(values: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(values, 0.0, out=values)
    positive = values[values > 0.0]
    scale = float(np.percentile(positive, 99.0)) if positive.size else 1.0
    scale = max(scale, 1e-6)
    values /= scale
    np.clip(values, 0.0, 2.0, out=values)
    return values, scale


def prepare_subject(
    dataset_dir_value: str,
    cache_dir_value: str,
    split: str,
    subject: str,
    overwrite: bool,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir_value)
    cache_dir = Path(cache_dir_value)
    feature_dir = dataset_dir / split / subject / "features"
    output_dir = cache_dir / split / subject
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and all((output_dir / name).is_file() for name in MRI_CACHE_FILES):
        return json.loads((output_dir / "mri_metadata.json").read_text(encoding="utf-8"))

    nac_image = nib.load(feature_dir / "nacpet.nii.gz")
    phase_images = {
        "in": nib.load(feature_dir / "mri_combined_in_phase.nii.gz"),
        "out": nib.load(feature_dir / "mri_combined_out_phase.nii.gz"),
    }
    for phase, image in phase_images.items():
        if image.shape != nac_image.shape:
            raise ValueError(
                f"{subject}: MRI {phase} shape {image.shape} != NAC shape {nac_image.shape}"
            )
        if not np.allclose(image.affine, nac_image.affine, rtol=0.0, atol=1e-5):
            raise ValueError(f"{subject}: MRI {phase} affine does not match NAC")

    normalized_in, scale_in = normalize_mri_phase(
        phase_images["in"].get_fdata(dtype=np.float32)
    )
    normalized_out, scale_out = normalize_mri_phase(
        phase_images["out"].get_fdata(dtype=np.float32)
    )
    atomic_save_array(output_dir / "mri_in.npy", normalized_in.astype(np.float16))
    atomic_save_array(output_dir / "mri_out.npy", normalized_out.astype(np.float16))
    metadata = {
        "subject": subject,
        "shape": list(nac_image.shape),
        "in_phase_p99": scale_in,
        "out_phase_p99": scale_out,
        "in_phase_nonzero_fraction": float(np.count_nonzero(normalized_in) / normalized_in.size),
        "out_phase_nonzero_fraction": float(np.count_nonzero(normalized_out) / normalized_out.size),
    }
    atomic_save_json(output_dir / "mri_metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least one")
    dataset_dir = args.dataset_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    subjects = sorted(
        path.name for path in (dataset_dir / args.split).glob("sub-*") if path.is_dir()
    )
    if not subjects:
        raise FileNotFoundError(f"no train subjects under {dataset_dir}")

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                prepare_subject,
                str(dataset_dir),
                str(cache_dir),
                args.split,
                subject,
                args.overwrite,
            ): subject
            for subject in subjects
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Preparing MRI cache"
        ):
            rows.append(future.result())
    rows.sort(key=lambda row: str(row["subject"]))
    atomic_save_json(
        cache_dir / f"mri-manifest-{args.split}.json",
        {"split": args.split, "subject_count": len(rows), "subjects": rows},
    )
    print(f"Prepared MRI cache for {len(rows)} subjects under {cache_dir}")


if __name__ == "__main__":
    main()
