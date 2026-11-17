"""Run E77 with the exact historical float16 and axial-chunk semantics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .apply_projection_calibration import transform_chunk
from .blend_frequency_fold_predictions import _fold_mean
from .blend_tissue_fold_residuals import smooth_residual
from .container_inference import (
    configure_inference_runtime,
    load_refiner_model,
    prepare_input,
    prepare_refiner_input,
    save_ct,
    sliding_predict,
)
from .evaluate_cached import load_model
from .export_cached_ct import mu_to_hu
from .frequency_blend import FrequencyBlendCandidate, _profiles, frequency_blend
from .mu_calibration import MuCalibrationCandidate
from .spatial_blend import BlendCandidate, axial_alpha_profile
from .tissue_blend import TissueBlendCandidate, alpha_chunk


E15 = FrequencyBlendCandidate.parse("low180-high050:1.8:0.5:1:1:370:390")
E21 = TissueBlendCandidate.parse("f275-s250-b075:0:0.275:0.25:0.75:0:370:390")
E24 = BlendCandidate.parse("head-safe100:0:1:370:390")
E28_CALIBRATION = MuCalibrationCandidate.parse(
    "sf-n250-100:0:-0.0001:-0.00025:0:0:370:390"
)
E35 = FrequencyBlendCandidate.parse("f16-h075:0:0.75:0:0:370:390")
E43 = TissueBlendCandidate.parse("dual-seed:1.5:0.1:0.1:0:0:370:390")
E46_CALIBRATION = MuCalibrationCandidate.parse(
    "torso15:0:-0.0000375:-0.000105:0:0:80:140:370:390"
)
HISTORICAL_CHUNK_SLICES = 16


def _validate_arrays(*arrays: np.ndarray) -> tuple[int, int, int]:
    if not arrays:
        raise ValueError("at least one array is required")
    shape = tuple(np.asarray(arrays[0]).shape)
    if len(shape) != 3:
        raise ValueError("E77 arrays must be three-dimensional")
    for array in arrays:
        values = np.asarray(array)
        if values.shape != shape:
            raise ValueError("E77 component shapes differ")
        if not np.isfinite(values).all():
            raise ValueError("E77 component contains non-finite values")
    return shape


def stored_nonnegative(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    np.maximum(output, 0.0, out=output)
    return output.astype(np.float16)


def frequency_stage(
    base: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    *,
    zooms: tuple[float, float, float],
    candidate: FrequencyBlendCandidate,
    fwhm_mm: float,
    chunk_slices: int = HISTORICAL_CHUNK_SLICES,
) -> np.ndarray:
    shape = _validate_arrays(base, prediction_a, prediction_b)
    if len(zooms) != 3 or any(float(zoom) <= 0.0 for zoom in zooms):
        raise ValueError("zooms must contain three positive values")
    if fwhm_mm <= 0.0:
        raise ValueError("frequency FWHM must be positive")
    if chunk_slices <= 0:
        raise ValueError("chunk_slices must be positive")

    # Historical materializers read float16 files. NumPy therefore performs
    # the subtraction with float16 input loops even though the destination is
    # float32. Restoring only the values as float32 is not bitwise equivalent.
    stored_base = np.asarray(base, dtype=np.float16)
    stored_a = np.asarray(prediction_a, dtype=np.float16)
    stored_b = np.asarray(prediction_b, dtype=np.float16)
    difference = np.empty(shape, dtype=np.float32)
    np.subtract(
        stored_b,
        stored_a,
        out=difference,
        casting="unsafe",
    )
    sigma = [fwhm_mm / (2.355 * float(zoom)) for zoom in zooms]
    smoothed = gaussian_filter(difference, sigma=sigma)
    low, high = _profiles(shape[2], candidate)
    output = np.empty(shape, dtype=np.float16)
    for start in range(0, shape[2], chunk_slices):
        end = min(start + chunk_slices, shape[2])
        slices = np.s_[:, :, start:end]
        blended = frequency_blend(
            np.asarray(stored_base[slices], dtype=np.float32),
            difference[slices],
            smoothed[slices],
            low[None, None, start:end],
            high[None, None, start:end],
        )
        np.maximum(blended, 0.0, out=blended)
        output[slices] = blended.astype(np.float16)
    return output


def tissue_residual_stage(
    base: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    *,
    zooms: tuple[float, float, float],
    candidate: TissueBlendCandidate,
    residual_fwhm_mm: float,
    fold_repetitions: int = 5,
    chunk_slices: int = HISTORICAL_CHUNK_SLICES,
) -> np.ndarray:
    shape = _validate_arrays(base, prediction_a, prediction_b)
    if fold_repetitions <= 0:
        raise ValueError("fold_repetitions must be positive")
    if chunk_slices <= 0:
        raise ValueError("chunk_slices must be positive")
    stored_base = np.asarray(base, dtype=np.float16)
    stored_a = np.asarray(prediction_a, dtype=np.float16)
    stored_b = np.asarray(prediction_b, dtype=np.float16)
    prediction_a_folds = [stored_a] * fold_repetitions
    prediction_b_folds = [stored_b] * fold_repetitions
    residual = np.empty(shape, dtype=np.float32)
    for start in range(0, shape[2], chunk_slices):
        end = min(start + chunk_slices, shape[2])
        slices = np.s_[:, :, start:end]
        residual[slices] = _fold_mean(
            prediction_b_folds,
            slices,
        ) - _fold_mean(prediction_a_folds, slices)
    filtered = smooth_residual(
        residual,
        residual_fwhm_mm=residual_fwhm_mm,
        zooms=zooms,
    )
    output = np.empty(shape, dtype=np.float16)
    for start in range(0, shape[2], chunk_slices):
        end = min(start + chunk_slices, shape[2])
        slices = np.s_[:, :, start:end]
        anchor = np.asarray(stored_base[slices], dtype=np.float32)
        alpha = alpha_chunk(anchor, candidate, start, shape[2])
        blended = anchor + alpha * filtered[slices]
        np.maximum(blended, 0.0, out=blended)
        output[slices] = blended.astype(np.float16)
    return output


def spatial_stage(
    prediction_a: np.ndarray,
    reference: np.ndarray,
    candidate: BlendCandidate,
    chunk_slices: int = HISTORICAL_CHUNK_SLICES,
) -> np.ndarray:
    shape = _validate_arrays(prediction_a, reference)
    if chunk_slices <= 0:
        raise ValueError("chunk_slices must be positive")
    profile = axial_alpha_profile(shape[2], candidate)[None, None, :]
    stored_first = np.asarray(prediction_a, dtype=np.float16)
    stored_reference = np.asarray(reference, dtype=np.float16)
    output = np.empty(shape, dtype=np.float16)
    for start in range(0, shape[2], chunk_slices):
        end = min(start + chunk_slices, shape[2])
        slices = np.s_[:, :, start:end]
        first = _fold_mean([stored_first], slices)
        target = np.asarray(stored_reference[slices], dtype=np.float32)
        blended = first + profile[:, :, start:end] * (target - first)
        np.maximum(blended, 0.0, out=blended)
        output[slices] = blended.astype(np.float16)
    return output


def projection_calibration_stage(
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
) -> np.ndarray:
    shape = _validate_arrays(base, alternative)
    if chunk_slices <= 0:
        raise ValueError("chunk_slices must be positive")
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
    output = np.empty(shape, dtype=np.float16)
    for start in range(0, shape[2], chunk_slices):
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
        output[slices] = transformed.astype(np.float16)
    return output


def refiner_tissue_stage(
    base: np.ndarray,
    refiners: list[np.ndarray],
    candidate: TissueBlendCandidate,
    chunk_slices: int = HISTORICAL_CHUNK_SLICES,
) -> np.ndarray:
    if not refiners:
        raise ValueError("at least one refiner output is required")
    shape = _validate_arrays(base, *refiners)
    if chunk_slices <= 0:
        raise ValueError("chunk_slices must be positive")
    stored_base = np.asarray(base, dtype=np.float16)
    stored_refiners = [np.asarray(refined, dtype=np.float16) for refined in refiners]
    output = np.empty(shape, dtype=np.float16)
    for start in range(0, shape[2], chunk_slices):
        end = min(start + chunk_slices, shape[2])
        slices = np.s_[:, :, start:end]
        anchor = np.asarray(stored_base[slices], dtype=np.float32)
        refined_mean = _fold_mean(stored_refiners, slices)
        alpha = alpha_chunk(anchor, candidate, start, shape[2])
        blended = anchor + alpha * (refined_mean - anchor)
        np.maximum(blended, 0.0, out=blended)
        output[slices] = blended.astype(np.float16)
    return output


def compose_base_to_e35(
    *,
    e06: np.ndarray,
    e08: np.ndarray,
    e13: np.ndarray,
    e24: np.ndarray,
    zooms: tuple[float, float, float],
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
    e28b = projection_calibration_stage(
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
    e46 = projection_calibration_stage(
        e43,
        e43,
        body_alpha=0.0,
        head_alpha=0.0,
        projection_strength=0.0,
        mask_threshold=0.01,
        projection_iterations=1,
        transition_start=370,
        transition_end=390,
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
    prepare_started = time.perf_counter()
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
    if reference is None:
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
    )
    base_transform_seconds = time.perf_counter() - transform_started
    stage_names = list(base_stages)
    del base_outputs, base_stages, input_values

    refiner_outputs = []
    refiner_prepare_started = time.perf_counter()
    refiner_input = prepare_refiner_input(
        args.features_dir,
        e35,
        use_mri=True,
        coordinate_mode="xyz",
        mu_scale=0.15,
    )
    refiner_prepare_seconds = time.perf_counter() - refiner_prepare_started
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
        "program": "E77 six-family full-75 refit",
        "base_checkpoints": {name: str(path) for name, path in base_paths.items()},
        "refiner_checkpoints": {
            name: str(path) for name, path in refiner_paths.items()
        },
        "network_count": len(network_seconds),
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
            "networks": network_seconds,
            "network_total": float(sum(network_seconds.values())),
            "base_transforms": base_transform_seconds,
            "prepare_refiner_input": refiner_prepare_seconds,
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
