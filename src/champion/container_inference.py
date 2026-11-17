"""Run audited champion checkpoints directly from challenge feature NIfTIs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from .data import spatial_coordinate_channels
from .evaluate_cached import load_model
from .export_cached_ct import mu_to_hu
from .mu_calibration import MuCalibrationCandidate, calibration_offset
from .prepare_mri_cache import normalize_mri_phase
from .refiner import MuResidualRefiner
from .safety_gated_refiner import MuSafetyGatedRefiner


def configure_inference_runtime(device: torch.device) -> dict[str, Any]:
    """Match cached validation inference and avoid A40-only TF32 drift."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    return {
        "autocast": "float16" if device.type == "cuda" else "disabled",
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def float16_roundtrip(values: np.ndarray) -> np.ndarray:
    """Return the float32 values represented by the stored float16 artifact."""
    return np.asarray(np.asarray(values, dtype=np.float16), dtype=np.float32)


def prepare_input(
    features_dir: Path,
    *,
    use_topogram: bool,
    use_mri: bool = False,
    coordinate_mode: str,
) -> tuple[torch.Tensor, nib.Nifti1Image]:
    nac_path = features_dir / "nacpet.nii.gz"
    reference = nib.load(nac_path)
    nac = reference.get_fdata(dtype=np.float32)
    if nac.ndim != 3 or not np.isfinite(nac).all():
        raise ValueError(f"{nac_path}: expected a finite 3D volume")
    np.maximum(nac, 0.0, out=nac)
    nac_log = np.log1p(nac)
    positive = nac_log[nac_log > 0.0]
    p99 = float(np.percentile(positive, 99.0)) if positive.size else 1.0
    nac_log = float16_roundtrip(nac_log)
    channels = [
        np.clip(nac_log / 7.0, 0.0, 2.0),
        np.clip(nac_log / max(p99, 1e-6), 0.0, 2.0),
    ]

    if use_topogram:
        topogram_path = features_dir / "topogram.nii.gz"
        topogram = nib.load(topogram_path).get_fdata(dtype=np.float32)
        if (
            topogram.ndim != 3
            or topogram.shape[1] != 1
            or (topogram.shape[0], topogram.shape[2])
            != (nac.shape[0], nac.shape[2])
        ):
            raise ValueError(
                f"{topogram_path}: topogram grid {topogram.shape} "
                f"does not match NAC-PET {nac.shape}"
            )
        normalized = float16_roundtrip(
            np.clip(topogram[:, 0, :] / 250.0, -1.0, 1.0)
        )
        channels.append(np.broadcast_to(normalized[:, None, :], nac.shape))

    if use_mri:
        for phase in ("in", "out"):
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
            normalized, _ = normalize_mri_phase(
                image.get_fdata(dtype=np.float32)
            )
            channels.append(float16_roundtrip(normalized))

    full_slices = tuple(slice(0, size) for size in nac.shape)
    coordinates = spatial_coordinate_channels(
        nac.shape,
        full_slices,
        coordinate_mode,
    )
    channels.extend(coordinates)
    values = np.ascontiguousarray(np.stack(channels, axis=0)[None])
    if not np.isfinite(values).all():
        raise ValueError("prepared model input contains non-finite values")
    return torch.from_numpy(values), reference


def prepare_refiner_input(
    features_dir: Path,
    base_mu: np.ndarray,
    *,
    use_mri: bool,
    coordinate_mode: str,
    mu_scale: float,
) -> torch.Tensor:
    """Build the deployment-time equivalent of MuRefinementPatchDataset inputs."""
    if mu_scale <= 0.0:
        raise ValueError("refiner mu_scale must be positive")
    nac_path = features_dir / "nacpet.nii.gz"
    reference = nib.load(nac_path)
    nac = reference.get_fdata(dtype=np.float32)
    base = np.asarray(base_mu, dtype=np.float32)
    if nac.ndim != 3 or not np.isfinite(nac).all():
        raise ValueError(f"{nac_path}: expected a finite 3D volume")
    if base.shape != nac.shape or not np.isfinite(base).all():
        raise ValueError("base mu-map must be finite and match NAC-PET shape")
    np.maximum(nac, 0.0, out=nac)
    nac_log = np.log1p(nac)
    positive = nac_log[nac_log > 0.0]
    p99 = float(np.percentile(positive, 99.0)) if positive.size else 1.0
    nac_log = float16_roundtrip(nac_log)
    channels = [
        float16_roundtrip(base) / mu_scale,
        np.clip(nac_log / 7.0, 0.0, 2.0),
        np.clip(nac_log / max(p99, 1e-6), 0.0, 2.0),
    ]

    if use_mri:
        for phase in ("in", "out"):
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
            normalized, _ = normalize_mri_phase(
                image.get_fdata(dtype=np.float32)
            )
            channels.append(float16_roundtrip(normalized))

    full_slices = tuple(slice(0, size) for size in nac.shape)
    channels.extend(
        spatial_coordinate_channels(
            nac.shape,
            full_slices,
            coordinate_mode,
        )
    )
    values = np.ascontiguousarray(np.stack(channels, axis=0)[None])
    if not np.isfinite(values).all():
        raise ValueError("prepared refiner input contains non-finite values")
    return torch.from_numpy(values)


def update_refiner_base_channel(
    values: torch.Tensor,
    base_mu: np.ndarray,
    *,
    mu_scale: float,
) -> torch.Tensor:
    """Reuse static NAC/MRI/coordinate channels for another paired base model."""
    if mu_scale <= 0.0:
        raise ValueError("refiner mu_scale must be positive")
    if values.ndim != 5 or values.shape[0] != 1 or values.shape[1] < 1:
        raise ValueError("refiner input must have shape [1, C, X, Y, Z]")
    base = np.asarray(base_mu, dtype=np.float32)
    if base.shape != tuple(values.shape[2:]) or not np.isfinite(base).all():
        raise ValueError("base mu-map must be finite and match refiner input shape")
    scaled = np.ascontiguousarray(float16_roundtrip(base) / mu_scale)
    values[0, 0].copy_(torch.from_numpy(scaled))
    return values


def load_refiner_model(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("refiner config must contain a JSON object")
    architecture = str(config.get("architecture", "MuResidualRefiner"))
    common = {
        "in_channels": int(config["input_channels"]),
        "base_channels": int(config["base_channels"]),
        "correction_scale": float(config["correction_scale"]),
        "mu_scale": float(config["mu_scale"]),
    }
    if architecture == "MuResidualRefiner":
        model = MuResidualRefiner(**common)
    elif architecture == "MuSafetyGatedRefiner":
        model = MuSafetyGatedRefiner(
            **common,
            gate_prior=float(config["gate_prior"]),
        )
    else:
        raise ValueError(f"unsupported refiner architecture: {architecture}")
    model = model.to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def sliding_predict(
    values: torch.Tensor,
    model: torch.nn.Module,
    *,
    roi_size: tuple[int, int, int],
    overlap: float,
    sw_batch_size: int,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        prediction = sliding_window_inference(
            values,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode="gaussian",
            sw_device=device,
            device=torch.device("cpu"),
            progress=False,
        )
    result = np.asarray(prediction[0, 0].float().numpy(), dtype=np.float32)
    np.maximum(result, 0.0, out=result)
    if not np.isfinite(result).all():
        raise ValueError("model prediction contains non-finite values")
    return result


def average_refiner_outputs(outputs: list[np.ndarray]) -> np.ndarray:
    """Average independently predicted refined mu-maps without stacking them."""
    if not outputs:
        raise ValueError("at least one refiner output is required")
    first = np.asarray(outputs[0], dtype=np.float32)
    if not np.isfinite(first).all():
        raise ValueError("refiner output contains non-finite values")
    total = first.copy()
    for output in outputs[1:]:
        values = np.asarray(output, dtype=np.float32)
        if values.shape != total.shape:
            raise ValueError("refiner outputs must share one shape")
        if not np.isfinite(values).all():
            raise ValueError("refiner output contains non-finite values")
        np.add(total, values, out=total)
    total /= float(len(outputs))
    np.maximum(total, 0.0, out=total)
    return total


def validate_model_layout(
    checkpoints: list[Path],
    refiner_checkpoints: list[Path],
    refiner_configs: list[Path],
    *,
    pair_refiners: bool,
) -> None:
    """Validate single-model, base-ensemble, and paired-chain layouts."""
    if not checkpoints:
        raise ValueError("at least one --checkpoint is required")
    if len(refiner_checkpoints) != len(refiner_configs):
        raise ValueError(
            "each --refiner-checkpoint requires one --refiner-config"
        )
    if pair_refiners:
        if not refiner_checkpoints:
            raise ValueError(
                "--pair-refiners requires one refiner for every base checkpoint"
            )
        if len(checkpoints) != len(refiner_checkpoints):
            raise ValueError(
                "paired base checkpoint and refiner counts must match"
            )
    elif len(checkpoints) > 1 and refiner_checkpoints:
        raise ValueError(
            "multiple base checkpoints with refiners require --pair-refiners"
        )


def apply_mu_calibration(
    mu: np.ndarray,
    candidate: MuCalibrationCandidate | None,
    *,
    chunk_slices: int = 32,
) -> np.ndarray:
    values = np.asarray(mu, dtype=np.float32).copy()
    np.maximum(values, 0.0, out=values)
    if candidate is None:
        return values
    if chunk_slices <= 0:
        raise ValueError("chunk_slices must be positive")
    for start in range(0, values.shape[2], chunk_slices):
        stop = min(values.shape[2], start + chunk_slices)
        slices = np.s_[:, :, start:stop]
        anchor = values[slices].copy()
        values[slices] = anchor + calibration_offset(
            anchor,
            candidate,
            start_slice=start,
            total_slices=values.shape[2],
        )
    np.maximum(values, 0.0, out=values)
    if not np.isfinite(values).all():
        raise ValueError("calibrated mu-map contains non-finite values")
    return values


def save_ct(
    output_path: Path,
    hu: np.ndarray,
    reference: nib.Nifti1Image,
) -> None:
    if hu.shape != reference.shape:
        raise ValueError(f"output shape {hu.shape} != reference {reference.shape}")
    if not np.isfinite(hu).all():
        raise ValueError("output CT contains non-finite values")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.nii.gz")
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(
        nib.Nifti1Image(
            np.asarray(hu, dtype=np.float32),
            reference.affine,
            header,
        ),
        temporary,
    )
    os.replace(temporary, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("/data/features"))
    parser.add_argument(
        "--output-ct",
        type=Path,
        default=Path("/data/output/ct.nii.gz"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Repeat to average independent base models.",
    )
    parser.add_argument("--roi-size", type=int, nargs=3, default=(192, 192, 192))
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration", type=MuCalibrationCandidate.parse)
    parser.add_argument(
        "--refiner-checkpoint",
        type=Path,
        action="append",
        help="Repeat with --refiner-config to average independent refiners.",
    )
    parser.add_argument("--refiner-config", type=Path, action="append")
    parser.add_argument(
        "--pair-refiners",
        action="store_true",
        help=(
            "Pair each repeated base checkpoint with the corresponding "
            "refiner before averaging chain outputs."
        ),
    )
    parser.add_argument("--refiner-roi-size", type=int, nargs=3)
    parser.add_argument("--refiner-overlap", type=float)
    parser.add_argument("--refiner-sw-batch-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if not 0.0 <= args.overlap < 1.0:
        raise ValueError("--overlap must be in [0, 1)")
    if any(size <= 0 or size % 16 != 0 for size in args.roi_size):
        raise ValueError("--roi-size entries must be positive multiples of 16")
    checkpoints = args.checkpoint or []
    refiner_checkpoints = args.refiner_checkpoint or []
    refiner_configs = args.refiner_config or []
    validate_model_layout(
        checkpoints,
        refiner_checkpoints,
        refiner_configs,
        pair_refiners=args.pair_refiners,
    )
    refiner_roi_size = tuple(args.refiner_roi_size or args.roi_size)
    refiner_overlap = (
        args.overlap
        if args.refiner_overlap is None
        else args.refiner_overlap
    )
    refiner_sw_batch_size = (
        args.sw_batch_size
        if args.refiner_sw_batch_size is None
        else args.refiner_sw_batch_size
    )
    if any(size <= 0 or size % 16 != 0 for size in refiner_roi_size):
        raise ValueError(
            "--refiner-roi-size entries must be positive multiples of 16"
        )
    if not 0.0 <= refiner_overlap < 1.0:
        raise ValueError("--refiner-overlap must be in [0, 1)")
    if refiner_sw_batch_size <= 0:
        raise ValueError("--refiner-sw-batch-size must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    runtime_policy = configure_inference_runtime(device)
    reference: nib.Nifti1Image | None = None
    input_cache: dict[tuple[bool, bool, str], torch.Tensor] = {}
    base_outputs: list[np.ndarray] = []
    paired_outputs: list[np.ndarray] = []
    paired_refiner_input_cache: dict[
        tuple[bool, str, float, int],
        torch.Tensor,
    ] = {}
    prepare_seconds = 0.0
    base_inference_seconds = 0.0
    refiner_prepare_seconds = 0.0
    refiner_inference_seconds = 0.0

    for index, checkpoint_path in enumerate(checkpoints):
        preparing = time.perf_counter()
        model, use_topogram, use_mri, coordinate_mode = load_model(
            checkpoint_path,
            device,
        )
        input_signature = (use_topogram, use_mri, coordinate_mode)
        if input_signature not in input_cache:
            input_values, candidate_reference = prepare_input(
                args.features_dir,
                use_topogram=use_topogram,
                use_mri=use_mri,
                coordinate_mode=coordinate_mode,
            )
            input_cache[input_signature] = input_values
            if reference is None:
                reference = candidate_reference
            elif (
                candidate_reference.shape != reference.shape
                or not np.allclose(
                    candidate_reference.affine,
                    reference.affine,
                    rtol=0.0,
                    atol=1e-5,
                )
            ):
                raise ValueError(
                    "base ensemble checkpoints produced incompatible input grids"
                )
        input_values = input_cache[input_signature]
        prepared_base = time.perf_counter()
        prepare_seconds += prepared_base - preparing
        base_mu = sliding_predict(
            input_values,
            model,
            roi_size=tuple(args.roi_size),
            overlap=args.overlap,
            sw_batch_size=args.sw_batch_size,
            device=device,
        )
        base_mu = float16_roundtrip(base_mu)
        inferred_base = time.perf_counter()
        base_inference_seconds += inferred_base - prepared_base
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if not args.pair_refiners:
            base_outputs.append(base_mu)
            continue

        preparing_refiner = time.perf_counter()
        refiner_model, refiner_config = load_refiner_model(
            refiner_checkpoints[index],
            refiner_configs[index],
            device,
        )
        expected_channels = int(refiner_config["input_channels"])
        input_signature = (
            bool(refiner_config["use_mri"]),
            str(refiner_config["coordinate_mode"]),
            float(refiner_config["mu_scale"]),
            expected_channels,
        )
        if input_signature not in paired_refiner_input_cache:
            paired_refiner_input_cache[input_signature] = (
                prepare_refiner_input(
                    args.features_dir,
                    base_mu,
                    use_mri=input_signature[0],
                    coordinate_mode=input_signature[1],
                    mu_scale=input_signature[2],
                )
            )
        else:
            update_refiner_base_channel(
                paired_refiner_input_cache[input_signature],
                base_mu,
                mu_scale=input_signature[2],
            )
        refiner_values = paired_refiner_input_cache[input_signature]
        if refiner_values.shape[1] != expected_channels:
            raise ValueError(
                "prepared refiner channels "
                f"{refiner_values.shape[1]} != configured {expected_channels}"
            )
        prepared_refiner = time.perf_counter()
        refiner_prepare_seconds += prepared_refiner - preparing_refiner
        paired_outputs.append(
            float16_roundtrip(
                sliding_predict(
                    refiner_values,
                    refiner_model,
                    roi_size=refiner_roi_size,
                    overlap=refiner_overlap,
                    sw_batch_size=refiner_sw_batch_size,
                    device=device,
                )
            )
        )
        inferred_refiner = time.perf_counter()
        refiner_inference_seconds += inferred_refiner - prepared_refiner
        del refiner_model
        del refiner_values
        del base_mu
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del paired_refiner_input_cache

    if reference is None:
        raise RuntimeError("no model input was prepared")
    if args.pair_refiners:
        mu = average_refiner_outputs(paired_outputs)
    else:
        mu = average_refiner_outputs(base_outputs)

    if refiner_checkpoints and not args.pair_refiners:
        base_mu = mu
        refiner_outputs = []
        refiner_input_cache: dict[
            tuple[bool, str, float, int],
            torch.Tensor,
        ] = {}
        for checkpoint_path, config_path in zip(
            refiner_checkpoints,
            refiner_configs,
        ):
            preparing_refiner = time.perf_counter()
            refiner_model, refiner_config = load_refiner_model(
                checkpoint_path,
                config_path,
                device,
            )
            input_signature = (
                bool(refiner_config["use_mri"]),
                str(refiner_config["coordinate_mode"]),
                float(refiner_config["mu_scale"]),
                int(refiner_config["input_channels"]),
            )
            if input_signature not in refiner_input_cache:
                refiner_input_cache[input_signature] = prepare_refiner_input(
                    args.features_dir,
                    base_mu,
                    use_mri=input_signature[0],
                    coordinate_mode=input_signature[1],
                    mu_scale=input_signature[2],
                )
            refiner_values = refiner_input_cache[input_signature]
            expected_channels = int(refiner_config["input_channels"])
            if refiner_values.shape[1] != expected_channels:
                raise ValueError(
                    "prepared refiner channels "
                    f"{refiner_values.shape[1]} != configured "
                    f"{expected_channels}"
                )
            prepared_refiner = time.perf_counter()
            refiner_outputs.append(
                float16_roundtrip(
                    sliding_predict(
                        refiner_values,
                        refiner_model,
                        roi_size=refiner_roi_size,
                        overlap=refiner_overlap,
                        sw_batch_size=refiner_sw_batch_size,
                        device=device,
                    )
                )
            )
            inferred_refiner = time.perf_counter()
            refiner_prepare_seconds += prepared_refiner - preparing_refiner
            refiner_inference_seconds += inferred_refiner - prepared_refiner
            del refiner_model
            del refiner_values
            if device.type == "cuda":
                torch.cuda.empty_cache()
        mu = average_refiner_outputs(refiner_outputs)
        del refiner_input_cache

    mu = float16_roundtrip(mu)
    mu = apply_mu_calibration(mu, args.calibration)
    mu = float16_roundtrip(mu)
    hu = mu_to_hu(mu)
    save_ct(args.output_ct, hu, reference)
    finished = time.perf_counter()

    result: dict[str, Any] = {
        "checkpoint": (
            str(checkpoints[0]) if len(checkpoints) == 1 else None
        ),
        "checkpoints": [str(path) for path in checkpoints],
        "base_count": len(checkpoints),
        "pair_refiners": bool(args.pair_refiners),
        "storage_precision": "float16-roundtrip",
        "runtime_policy": runtime_policy,
        "refiner_checkpoint": (
            str(refiner_checkpoints[0])
            if len(refiner_checkpoints) == 1
            else None
        ),
        "refiner_checkpoints": [
            str(path) for path in refiner_checkpoints
        ],
        "refiner_count": len(refiner_checkpoints),
        "output": str(args.output_ct),
        "shape": list(hu.shape),
        "minimum_hu": float(np.min(hu)),
        "maximum_hu": float(np.max(hu)),
        "finite": bool(np.isfinite(hu).all()),
        "calibration": (
            args.calibration.__dict__
            if args.calibration is not None
            else None
        ),
        "seconds": {
            "prepare": prepare_seconds,
            "base_inference": base_inference_seconds,
            "refiner_prepare": refiner_prepare_seconds,
            "refiner_inference": refiner_inference_seconds,
            "inference": (
                base_inference_seconds
                + refiner_inference_seconds
            ),
            "postprocess_and_save": (
                finished
                - started
                - prepare_seconds
                - base_inference_seconds
                - refiner_prepare_seconds
                - refiner_inference_seconds
            ),
            "total": finished - started,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
