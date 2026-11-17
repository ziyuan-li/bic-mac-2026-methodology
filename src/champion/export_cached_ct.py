"""Convert cached mu-map predictions to challenge-compatible HU NIfTI files."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


MU_BREAKPOINT = 5.10e-5 * 1047.0 + 4.71e-2


def mu_to_hu(mu: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(mu, dtype=np.float32), 0.0)
    hu = np.where(
        values < MU_BREAKPOINT,
        values / 9.6e-5 - 1000.0,
        (values - 4.71e-2) / 5.10e-5 - 1000.0,
    )
    return np.clip(hu, -1024.0, 3071.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--subjects", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = args.subjects or sorted(path.name for path in args.prediction_dir.glob("sub-*"))
    for subject in subjects:
        prediction_path = args.prediction_dir / subject / "mu.npy"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        reference_path = args.dataset_dir / args.split / subject / "features" / "nacpet.nii.gz"
        reference = nib.load(reference_path)
        mu = np.load(prediction_path, mmap_mode="r")
        if mu.shape != reference.shape:
            raise ValueError(f"{subject}: prediction shape {mu.shape} != reference {reference.shape}")
        hu = mu_to_hu(mu)
        output_path = args.output_dir / subject / "ct.nii.gz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        header = reference.header.copy()
        header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(hu, reference.affine, header), output_path)
        print(f"{subject}: {output_path} HU=[{float(hu.min()):.1f}, {float(hu.max()):.1f}]")


if __name__ == "__main__":
    main()
