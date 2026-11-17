"""Prepare memory-mapped arrays for fast BIC-MAC patch training."""

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


CACHE_FILES = ("nac_log.npy", "mu.npy", "prediction_mask.npy", "eval_mask.npy", "metadata.json")


def hu_to_mu(hu: np.ndarray) -> np.ndarray:
    hu1000 = hu + 1000.0
    mu = np.where(hu1000 < 1047.0, 9.6e-5 * hu1000, 5.10e-5 * hu1000 + 4.71e-2)
    return np.clip(mu, 0.0, None)


def bounding_box(mask: np.ndarray) -> list[list[int]] | None:
    coordinates = np.nonzero(mask)
    if not coordinates[0].size:
        return None
    return [
        [int(np.min(axis)), int(np.max(axis)) + 1]
        for axis in coordinates
    ]


def atomic_save_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def atomic_save_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_subject(dataset_dir_value: str, cache_dir_value: str, subject: str, overwrite: bool) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir_value)
    cache_dir = Path(cache_dir_value)
    source_dir = dataset_dir / "train" / subject
    output_dir = cache_dir / "train" / subject
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and all((output_dir / name).is_file() for name in CACHE_FILES):
        return json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

    nac_image = nib.load(str(source_dir / "features/nacpet.nii.gz"))
    nac = nac_image.get_fdata(dtype=np.float32)
    np.maximum(nac, 0.0, out=nac)
    nac_log = np.log1p(nac)
    del nac
    positive = nac_log[nac_log > 0]
    nac_log_p99 = float(np.percentile(positive, 99)) if positive.size else 1.0
    nac_log_p999 = float(np.percentile(positive, 99.9)) if positive.size else 1.0

    ct = nib.load(str(source_dir / "ct-label/ct.nii.gz")).get_fdata(dtype=np.float32)
    mu = hu_to_mu(ct)
    del ct
    prediction_mask = np.asanyarray(
        nib.load(str(source_dir / "ct-label/prediction_mask.nii.gz")).dataobj
    ).astype(bool, copy=False)
    body_mask = np.asanyarray(nib.load(str(source_dir / "ct-label/body_seg.nii.gz")).dataobj) > 0
    organ = np.asanyarray(nib.load(str(source_dir / "ct-label/organ_seg.nii.gz")).dataobj)
    liver_z = np.where(organ == 5)[2]
    if not liver_z.size:
        raise ValueError(f"{subject}: liver label 5 is empty")
    superior_liver_slice = int(np.max(liver_z))
    slice_thickness_mm = float(nac_image.header.get_zooms()[2])
    exclusion_slices = int(round(40.0 / slice_thickness_mm))
    eval_mask = body_mask.copy()
    z_min = max(0, superior_liver_slice - exclusion_slices)
    z_max = min(eval_mask.shape[2], superior_liver_slice + exclusion_slices)
    eval_mask[:, :, z_min:z_max] = False

    atomic_save_array(output_dir / "nac_log.npy", nac_log.astype(np.float16))
    atomic_save_array(output_dir / "mu.npy", mu.astype(np.float16))
    atomic_save_array(output_dir / "prediction_mask.npy", prediction_mask.astype(np.uint8))
    atomic_save_array(output_dir / "eval_mask.npy", eval_mask.astype(np.uint8))

    metadata = {
        "subject": subject,
        "shape": list(nac_image.shape),
        "zooms": [float(value) for value in nac_image.header.get_zooms()[:3]],
        "nac_log_p99": nac_log_p99,
        "nac_log_p999": nac_log_p999,
        "prediction_voxels": int(np.count_nonzero(prediction_mask)),
        "eval_voxels": int(np.count_nonzero(eval_mask)),
        "prediction_bbox": bounding_box(prediction_mask),
        "eval_bbox": bounding_box(eval_mask),
        "superior_liver_slice": superior_liver_slice,
        "exclusion_slices": exclusion_slices,
    }
    atomic_save_json(output_dir / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    subjects = sorted(path.name for path in (dataset_dir / "train").glob("sub-*") if path.is_dir())
    if not subjects:
        raise FileNotFoundError(f"No train subjects found under {dataset_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(prepare_subject, str(dataset_dir), str(cache_dir), subject, args.overwrite): subject
            for subject in subjects
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preparing train cache"):
            results.append(future.result())
    results.sort(key=lambda row: row["subject"])
    manifest = {
        "dataset_dir": str(dataset_dir),
        "cache_dir": str(cache_dir),
        "subject_count": len(results),
        "subjects": results,
    }
    atomic_save_json(cache_dir / "manifest.json", manifest)
    print(f"Prepared {len(results)} subjects under {cache_dir}")


if __name__ == "__main__":
    main()

