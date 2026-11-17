"""Run exact E77 inference with label-free critical-path optimizations."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

from .apply_projection_calibration import transform_chunk
from .container_inference import (
    configure_inference_runtime,
    float16_roundtrip,
    load_refiner_model,
    prepare_input,
    save_ct,
    sliding_predict,
    update_refiner_base_channel,
)
from .data import spatial_coordinate_channels
from .e77_six_family_inference_v2 import (
    E15,
    E21,
    E24,
    E28_CALIBRATION,
    E35,
    E43,
    E46_CALIBRATION,
    HISTORICAL_CHUNK_SLICES,
    _validate_arrays,
    frequency_stage,
    refiner_tissue_stage,
    spatial_stage,
    stored_nonnegative,
    tissue_residual_stage,
)
from .evaluate_cached import load_model
from .export_cached_ct import mu_to_hu
from .mu_calibration import MuCalibrationCandidate, calibration_offset
from .prepare_mri_cache import normalize_mri_phase
from .projection_blend import projection_blend
from .spatial_blend import BlendCandidate, axial_alpha_profile


DEFAULT_PROJECTION_WORKERS = 4


def prepare_refiner_static_input(
    features_dir: Path,
    base_input: torch.Tensor,
    reference: nib.Nifti1Image,
) -> torch.Tensor:
    """Reuse exact NAC channels while preparing MRI/coordinate channels once."""
    if base_input.ndim != 5 or tuple(base_input.shape[:2]) != (1, 3):
        raise ValueError("E77 base input must have shape [1, 3, X, Y, Z]")
    if tuple(base_input.shape[2:]) != tuple(reference.shape):
        raise ValueError("E77 base input and reference grids differ")
    if base_input.device.type != "cpu":
        raise ValueError("E77 reusable base input must remain on CPU")

    shape = tuple(int(value) for value in reference.shape)
    values = np.empty((1, 8, *shape), dtype=np.float32)
    values[0, 0].fill(0.0)
    base_values = base_input.numpy()
    values[0, 1] = base_values[0, 0]
    values[0, 2] = base_values[0, 1]

    for channel, phase in enumerate(("in", "out"), start=3):
        path = features_dir / f"mri_combined_{phase}_phase.nii.gz"
        image = nib.load(path)
        if image.shape != reference.shape:
            raise ValueError(
                f"{path}: MRI grid {image.shape} does not match NAC-PET "
                f"{reference.shape}"
            )
        if not np.allclose(
            image.affine,
            reference.affine,
            rtol=0.0,
            atol=1e-5,
        ):
            raise ValueError(f"{path}: MRI affine does not match NAC-PET")
        normalized, _ = normalize_mri_phase(image.get_fdata(dtype=np.float32))
        values[0, channel] = float16_roundtrip(normalized)

    full_slices = tuple(slice(0, size) for size in shape)
    coordinates = spatial_coordinate_channels(shape, full_slices, "xyz")
    for channel, coordinate in enumerate(coordinates, start=5):
        values[0, channel] = coordinate
    if not np.isfinite(values).all():
        raise ValueError("prepared E77 refiner static input contains non-finite values")
    return torch.from_numpy(values)


def _timed_refiner_static_input(
    features_dir: Path,
    base_input: torch.Tensor,
    reference: nib.Nifti1Image,
) -> tuple[torch.Tensor, float]:
    started = time.perf_counter()
    values = prepare_refiner_static_input(features_dir, base_input, reference)
    return values, time.perf_counter() - started


def projection_calibration_stage_parallel(
    base: np.ndarray,
    alternative: np.ndarray,
    *,
    body_alpha: float,
    head_alpha: float,
    projection_strength: float,
    mask_threshold: float,
    projection_iterations: int,
    transition_start: int,
    transition_end: int,
    calibration: MuCalibrationCandidate,
    chunk_slices: int = HISTORICAL_CHUNK_SLICES,
    workers: int = DEFAULT_PROJECTION_WORKERS,
) -> np.ndarray:
    """Evaluate independent historical projection chunks in a fixed thread pool."""
    shape = _validate_arrays(base, alternative)
    if chunk_slices <= 0:
        raise ValueError("chunk_slices must be positive")
    if workers <= 0:
        raise ValueError("projection workers must be positive")
    profile = axial_alpha_profile(
        shape[2],
        BlendCandidate(
            "stage",
            body_alpha,
            head_alpha,
            transition_start,
            transition_end,
        ),
    )
    stored_base = np.asarray(base, dtype=np.float16)
    stored_alternative = np.asarray(alternative, dtype=np.float16)
    starts = list(range(0, shape[2], chunk_slices))

    def transform(start: int) -> tuple[int, int, np.ndarray]:
        end = min(start + chunk_slices, shape[2])
        slices = np.s_[:, :, start:end]
        transformed = transform_chunk(
            np.asarray(stored_base[slices], dtype=np.float32),
            np.asarray(stored_alternative[slices], dtype=np.float32),
            profile[None, None, start:end],
            projection_strength,
            mask_threshold,
            projection_iterations,
            calibration,
            start,
            shape[2],
        )
        return start, end, transformed.astype(np.float16)

    if workers == 1 or len(starts) == 1:
        chunks = map(transform, starts)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=min(workers, len(starts)))
        chunks = executor.map(transform, starts)
    output = np.empty(shape, dtype=np.float16)
    try:
        for start, end, transformed in chunks:
            output[:, :, start:end] = transformed
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return output


def zero_projection_calibration_stage(
    base: np.ndarray,
    *,
    calibration: MuCalibrationCandidate,
    chunk_slices: int = HISTORICAL_CHUNK_SLICES,
) -> np.ndarray:
    """Match E46 exactly while skipping its algebraically inactive projection sums."""
    shape = _validate_arrays(base)
    if chunk_slices <= 0:
        raise ValueError("chunk_slices must be positive")
    profile = axial_alpha_profile(
        shape[2],
        BlendCandidate("stage", 0.0, 0.0, 370, 390),
    )
    stored_base = np.asarray(base, dtype=np.float16)
    output = np.empty(shape, dtype=np.float16)
    for start in range(0, shape[2], chunk_slices):
        end = min(start + chunk_slices, shape[2])
        slices = np.s_[:, :, start:end]
        anchor = np.asarray(stored_base[slices], dtype=np.float32)
        correction = anchor - anchor
        neutral = np.zeros_like(correction)
        transformed = projection_blend(
            anchor,
            correction,
            neutral,
            profile[None, None, start:end],
            0.0,
        )
        np.maximum(transformed, 0.0, out=transformed)
        transformed = transformed.astype(np.float16).astype(np.float32)
        transformed += calibration_offset(
            transformed,
            calibration,
            start,
            shape[2],
        )
        np.maximum(transformed, 0.0, out=transformed)
        output[slices] = transformed.astype(np.float16)
    return output


def compose_base_to_e35(
    *,
    e06: np.ndarray,
    e08: np.ndarray,
    e13: np.ndarray,
    e24: np.ndarray,
    zooms: tuple[float, float, float],
    projection_workers: int = DEFAULT_PROJECTION_WORKERS,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    _validate_arrays(e06, e08, e13, e24)
    e15 = frequency_stage(
        e08,
        e08,
        e13,
        zooms=zooms,
        candidate=E15,
        fwhm_mm=4.0,
    )
    e21 = tissue_residual_stage(
        e15,
        e08,
        e06,
        zooms=zooms,
        candidate=E21,
        residual_fwhm_mm=4.0,
    )
    e24_head = spatial_stage(e24, e15, E24)
    e28b = projection_calibration_stage_parallel(
        e21,
        e24_head,
        body_alpha=0.5,
        head_alpha=0.0,
        projection_strength=0.5,
        mask_threshold=0.01,
        projection_iterations=4,
        transition_start=370,
        transition_end=390,
        calibration=E28_CALIBRATION,
        workers=projection_workers,
    )
    e35 = frequency_stage(
        e28b,
        e21,
        e24_head,
        zooms=zooms,
        candidate=E35,
        fwhm_mm=16.0,
    )
    return e35, {
        "e15": e15,
        "e21": e21,
        "e24_head": e24_head,
        "e28b": e28b,
        "e35": e35,
    }


def compose_refiners_to_e77(
    *,
    e35: np.ndarray,
    refiner_outputs: list[np.ndarray],
    chunk_slices: int = HISTORICAL_CHUNK_SLICES,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    e43 = refiner_tissue_stage(
        e35,
        refiner_outputs,
        E43,
        chunk_slices=chunk_slices,
    )
    e46 = zero_projection_calibration_stage(
        e43,
        calibration=E46_CALIBRATION,
        chunk_slices=chunk_slices,
    )
    stored_e43 = np.asarray(e43, dtype=np.float16)
    stored_e46 = np.asarray(e46, dtype=np.float16)
    e77 = np.empty(stored_e43.shape, dtype=np.float16)
    for start in range(0, stored_e43.shape[2], chunk_slices):
        end = min(start + chunk_slices, stored_e43.shape[2])
        slices = np.s_[:, :, start:end]
        values = (
            2.0 * np.asarray(stored_e46[slices], dtype=np.float32)
            - np.asarray(stored_e43[slices], dtype=np.float32)
        )
        np.maximum(values, 0.0, out=values)
        e77[slices] = values.astype(np.float16)
    return e77, {"e43": e43, "e46": e46, "e77": e77}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-ct", type=Path, required=True)
    parser.add_argument("--output-mu", type=Path)
    parser.add_argument("--e06-checkpoint", type=Path, required=True)
    parser.add_argument("--e08-checkpoint", type=Path, required=True)
    parser.add_argument("--e13-checkpoint", type=Path, required=True)
    parser.add_argument("--e24-checkpoint", type=Path, required=True)
    parser.add_argument("--e40-refiner-checkpoint", type=Path, required=True)
    parser.add_argument("--e43-refiner-checkpoint", type=Path, required=True)
    parser.add_argument("--roi-size", type=int, nargs=3, default=(192, 192, 192))
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument(
        "--projection-workers",
        type=int,
        default=DEFAULT_PROJECTION_WORKERS,
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if not 0.0 <= args.overlap < 1.0:
        raise ValueError("--overlap must be in [0, 1)")
    if any(size <= 0 or size % 16 != 0 for size in args.roi_size):
        raise ValueError("--roi-size entries must be positive multiples of 16")
    if args.sw_batch_size <= 0:
        raise ValueError("--sw-batch-size must be positive")
    if args.projection_workers <= 0:
        raise ValueError("--projection-workers must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    runtime_policy = configure_inference_runtime(device)
    base_paths = {
        "e06": args.e06_checkpoint,
        "e08": args.e08_checkpoint,
        "e13": args.e13_checkpoint,
        "e24": args.e24_checkpoint,
    }
    base_outputs: dict[str, np.ndarray] = {}
    network_seconds: dict[str, float] = {}
    input_values = None
    reference = None
    static_future = None
    prepare_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as static_executor:
        for name, checkpoint_path in base_paths.items():
            model, use_topogram, use_mri, coordinate_mode = load_model(
                checkpoint_path,
                device,
            )
            if not use_topogram or use_mri or coordinate_mode != "none":
                raise ValueError(f"{name}: unexpected E77 base input signature")
            if input_values is None:
                input_values, reference = prepare_input(
                    args.features_dir,
                    use_topogram=True,
                    use_mri=False,
                    coordinate_mode="none",
                )
                static_future = static_executor.submit(
                    _timed_refiner_static_input,
                    args.features_dir,
                    input_values,
                    reference,
                )
            inference_started = time.perf_counter()
            base_outputs[name] = stored_nonnegative(
                sliding_predict(
                    input_values,
                    model,
                    roi_size=tuple(args.roi_size),
                    overlap=args.overlap,
                    sw_batch_size=args.sw_batch_size,
                    device=device,
                )
            )
            network_seconds[name] = time.perf_counter() - inference_started
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if reference is None or input_values is None or static_future is None:
            raise RuntimeError("no E77 input was prepared")
        prepare_seconds = time.perf_counter() - prepare_started - sum(
            network_seconds.values()
        )
        zooms = tuple(float(value) for value in reference.header.get_zooms()[:3])
        transform_started = time.perf_counter()
        e35, base_stages = compose_base_to_e35(
            e06=base_outputs["e06"],
            e08=base_outputs["e08"],
            e13=base_outputs["e13"],
            e24=base_outputs["e24"],
            zooms=zooms,
            projection_workers=args.projection_workers,
        )
        base_transform_seconds = time.perf_counter() - transform_started
        stage_names = list(base_stages)
        del base_outputs, base_stages

        refiner_wait_started = time.perf_counter()
        refiner_input, refiner_static_seconds = static_future.result()
        update_refiner_base_channel(refiner_input, e35, mu_scale=0.15)
        refiner_prepare_wait_seconds = time.perf_counter() - refiner_wait_started
        del input_values

    refiner_outputs = []
    refiner_paths = {
        "e40_refiner_seed1": args.e40_refiner_checkpoint,
        "e43_refiner_seed2": args.e43_refiner_checkpoint,
    }
    for name, checkpoint_path in refiner_paths.items():
        config_path = checkpoint_path.parent / "config.json"
        model, config = load_refiner_model(checkpoint_path, config_path, device)
        signature = (
            bool(config["use_mri"]),
            str(config["coordinate_mode"]),
            float(config["mu_scale"]),
        )
        if signature != (True, "xyz", 0.15):
            raise ValueError(f"{name}: unexpected E77 refiner input signature")
        inference_started = time.perf_counter()
        refiner_outputs.append(
            stored_nonnegative(
                sliding_predict(
                    refiner_input,
                    model,
                    roi_size=tuple(args.roi_size),
                    overlap=args.overlap,
                    sw_batch_size=args.sw_batch_size,
                    device=device,
                )
            )
        )
        network_seconds[name] = time.perf_counter() - inference_started
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del refiner_input

    transform_started = time.perf_counter()
    e77, final_stages = compose_refiners_to_e77(
        e35=e35,
        refiner_outputs=refiner_outputs,
    )
    final_transform_seconds = time.perf_counter() - transform_started
    stage_names.extend(final_stages)
    del refiner_outputs, final_stages
    output_started = time.perf_counter()
    if args.output_mu is not None:
        args.output_mu.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_mu.with_name(args.output_mu.name + ".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, e77.astype(np.float16), allow_pickle=False)
        temporary.replace(args.output_mu)
    save_ct(args.output_ct, mu_to_hu(e77), reference)
    output_seconds = time.perf_counter() - output_started
    total_seconds = time.perf_counter() - started
    payload: dict[str, Any] = {
        "program": "E77 six-family full-75 refit exact critical-path v3",
        "base_checkpoints": {name: str(path) for name, path in base_paths.items()},
        "refiner_checkpoints": {
            name: str(path) for name, path in refiner_paths.items()
        },
        "network_count": len(network_seconds),
        "projection_workers": args.projection_workers,
        "runtime_policy": runtime_policy,
        "shape": list(e77.shape),
        "zooms": list(zooms),
        "finite": bool(np.isfinite(e77).all()),
        "minimum_mu": float(e77.min()),
        "maximum_mu": float(e77.max()),
        "storage_precision": "float16-roundtrip at every historical stage",
        "stage_names": stage_names,
        "seconds": {
            "prepare_base_input": prepare_seconds,
            "refiner_static_background": refiner_static_seconds,
            "refiner_prepare_wait_and_base_update": refiner_prepare_wait_seconds,
            "networks": network_seconds,
            "network_total": float(sum(network_seconds.values())),
            "base_transforms": base_transform_seconds,
            "final_transforms": final_transform_seconds,
            "output": output_seconds,
            "total": total_seconds,
        },
        "output_ct": str(args.output_ct),
        "output_mu": str(args.output_mu) if args.output_mu is not None else None,
    }
    if not payload["finite"]:
        raise SystemExit("E77 six-family output is non-finite")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
