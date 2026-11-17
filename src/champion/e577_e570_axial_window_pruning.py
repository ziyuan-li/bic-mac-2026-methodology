"""Run E570 with exact, aligned axial-window pruning for its head branch."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .container_inference import (
    configure_inference_runtime,
    load_refiner_model,
    prepare_input,
    save_ct,
    sliding_predict,
    update_refiner_base_channel,
)
from .e566_e13_fold_ensemble_inference import E13_FOLD_COUNT, mean_e13_outputs
from .e569_fixed_head_hybrid import fixed_head_hybrid
from .e77_six_family_inference_v2 import stored_nonnegative
from .e77_six_family_inference_v3 import (
    DEFAULT_PROJECTION_WORKERS,
    _timed_refiner_static_input,
    compose_refiners_to_e77,
)
from .e77_six_family_inference_v4 import _timed_base_prefix, compose_base_suffix
from .evaluate_cached import load_model
from .export_cached_ct import mu_to_hu


PROGRAM = "E577 exact E570 axial-window pruning"
BACKGROUND_WORKERS = 3
NETWORK_COUNT = 13
FOLD_CROP_START = 144
HEAD_REFINER_CROP_START = 288
HEAD_EXACT_START = 370
MINIMUM_CPU_CONTEXT_SLICES = 64


def axial_window_starts(length: int, roi: int, overlap: float) -> list[int]:
    """Match MONAI dense-patch starts for one unpadded spatial dimension."""

    if length < roi:
        raise ValueError("axial length must be at least the ROI length")
    if roi <= 0 or not 0.0 <= overlap < 1.0:
        raise ValueError("invalid ROI or overlap")
    interval = max(int(roi * (1.0 - overlap)), 1)
    count = int(math.ceil(float(length - roi) / float(interval))) + 1
    return [min(index * interval, length - roi) for index in range(count)]


def validate_axial_pruning_schedule(
    *,
    length: int,
    roi: int,
    overlap: float,
    fold_crop_start: int = FOLD_CROP_START,
    head_crop_start: int = HEAD_REFINER_CROP_START,
    exact_start: int = HEAD_EXACT_START,
) -> dict[str, object]:
    """Prove both crops preserve the original windows needed by the final CT."""

    if not 0 <= fold_crop_start < head_crop_start < exact_start < length:
        raise ValueError("invalid frozen axial crop order")
    if head_crop_start - fold_crop_start < MINIMUM_CPU_CONTEXT_SLICES:
        raise ValueError("insufficient CPU transform context between crops")
    if length - fold_crop_start < roi or length - head_crop_start < roi:
        raise ValueError("cropped axial extent is smaller than the ROI")

    full = axial_window_starts(length, roi, overlap)
    interval = max(int(roi * (1.0 - overlap)), 1)
    discarded_overlap = roi - interval
    effective_cpu_context = (
        head_crop_start - fold_crop_start - discarded_overlap
    )

    def shifted(crop_start: int) -> list[int]:
        return [
            crop_start + start
            for start in axial_window_starts(length - crop_start, roi, overlap)
        ]

    fold = shifted(fold_crop_start)
    head = shifted(head_crop_start)
    expected_fold = [start for start in full if start >= fold_crop_start]
    expected_head = [start for start in full if start >= head_crop_start]
    contributing = [
        start for start in full if start <= exact_start < start + roi
    ]
    gates = {
        "fold_crop_is_original_window_start": fold_crop_start in full,
        "head_crop_is_original_window_start": head_crop_start in full,
        "fold_crop_schedule_is_exact_suffix": fold == expected_fold,
        "head_crop_schedule_is_exact_suffix": head == expected_head,
        "all_final_head_contributors_retained": all(
            start in head for start in contributing
        ),
        "no_discarded_window_contributes_at_exact_start": all(
            start + roi <= exact_start
            for start in full
            if start < head_crop_start
        ),
        "cpu_context_guard_passes": (
            effective_cpu_context >= MINIMUM_CPU_CONTEXT_SLICES
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"axial pruning schedule is not exact: {gates}")
    return {
        "full_starts": full,
        "fold_global_starts": fold,
        "head_global_starts": head,
        "final_head_contributors": contributing,
        "scan_interval": interval,
        "discarded_window_overlap_slices": discarded_overlap,
        "effective_cpu_context_slices": effective_cpu_context,
        "gates": gates,
    }


def merge_cropped_prediction(
    fallback: np.ndarray,
    cropped: np.ndarray,
    crop_start: int,
) -> np.ndarray:
    """Place an exact cropped suffix into a finite full-volume fallback."""

    fallback_values = np.asarray(fallback)
    cropped_values = np.asarray(cropped)
    if fallback_values.ndim != 3 or cropped_values.ndim != 3:
        raise ValueError("predictions must be three-dimensional")
    if crop_start < 0 or crop_start > fallback_values.shape[2]:
        raise ValueError("crop start is outside the full prediction")
    expected = (
        fallback_values.shape[0],
        fallback_values.shape[1],
        fallback_values.shape[2] - crop_start,
    )
    if cropped_values.shape != expected:
        raise ValueError("cropped prediction shape does not match its suffix")
    if not np.isfinite(fallback_values).all() or not np.isfinite(cropped_values).all():
        raise ValueError("predictions must be finite")
    output = np.asarray(fallback_values, dtype=np.float16).copy()
    output[:, :, crop_start:] = np.asarray(cropped_values, dtype=np.float16)
    return output


def place_cropped_prediction(
    cropped: np.ndarray,
    full_shape: Sequence[int],
    crop_start: int,
) -> np.ndarray:
    """Place a cropped suffix in a finite audit array with an unused zero prefix."""

    shape = tuple(int(value) for value in full_shape)
    if len(shape) != 3:
        raise ValueError("full shape must be three-dimensional")
    zeros = np.zeros(shape, dtype=np.float16)
    return merge_cropped_prediction(zeros, cropped, crop_start)


def _validate_args(args: argparse.Namespace) -> torch.device:
    if len(args.e13_fold_checkpoints) != E13_FOLD_COUNT:
        raise ValueError("--e13-fold-checkpoints requires exactly five paths")
    if not 0.0 <= args.overlap < 1.0:
        raise ValueError("--overlap must be in [0, 1)")
    if any(size <= 0 or size % 16 != 0 for size in args.roi_size):
        raise ValueError("--roi-size entries must be positive multiples of 16")
    if args.sw_batch_size != 1:
        raise ValueError("E577 freezes --sw-batch-size at one")
    if args.projection_workers <= 0:
        raise ValueError("projection workers must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    return device


def _save_mu(path: Path | None, values: np.ndarray) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype=np.float16), allow_pickle=False)
    os.replace(temporary, path)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _timed_suffix(
    *,
    e15: np.ndarray,
    e21: np.ndarray,
    e24: np.ndarray,
    zooms: tuple[float, float, float],
    projection_workers: int,
) -> tuple[np.ndarray, list[str], float]:
    started = time.perf_counter()
    e35, stages = compose_base_suffix(
        e15=e15,
        e21=e21,
        e24=e24,
        zooms=zooms,
        projection_workers=projection_workers,
    )
    return e35, stages, time.perf_counter() - started


def _timed_final(
    e35: np.ndarray, refiners: list[np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    started = time.perf_counter()
    output, stages = compose_refiners_to_e77(e35=e35, refiner_outputs=refiners)
    return output, stages, time.perf_counter() - started


def _predict(
    values: torch.Tensor,
    model: torch.nn.Module,
    *,
    roi_size: tuple[int, int, int],
    overlap: float,
    device: torch.device,
) -> np.ndarray:
    return stored_nonnegative(
        sliding_predict(
            values,
            model,
            roi_size=roi_size,
            overlap=overlap,
            sw_batch_size=1,
            device=device,
        )
    )


def execute(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = _validate_args(args)
    runtime_policy = configure_inference_runtime(device)
    jobs = [
        ("e06", args.e06_checkpoint),
        ("e08", args.e08_checkpoint),
        ("e13_full75", args.e13_full75_checkpoint),
        *[
            (f"e13_fold{fold}", checkpoint)
            for fold, checkpoint in enumerate(args.e13_fold_checkpoints)
        ],
        ("e24", args.e24_checkpoint),
    ]
    outputs: dict[str, np.ndarray] = {}
    fold_outputs: list[np.ndarray] = []
    network_seconds: dict[str, float] = {}
    input_values = None
    reference = None
    static_future: Future[tuple[torch.Tensor, float]] | None = None
    body_prefix_future: Future[tuple[np.ndarray, np.ndarray, float]] | None = None
    head_prefix_future: Future[tuple[np.ndarray, np.ndarray, float]] | None = None
    zooms: tuple[float, float, float] | None = None
    schedule: dict[str, object] | None = None
    prepare_started = time.perf_counter()
    roi_size = tuple(args.roi_size)

    with ThreadPoolExecutor(max_workers=BACKGROUND_WORKERS) as background:
        for name, checkpoint_path in jobs:
            model, use_topogram, use_mri, coordinate_mode = load_model(
                checkpoint_path, device
            )
            if not use_topogram or use_mri or coordinate_mode != "none":
                raise ValueError(f"{name}: unexpected E577 base input signature")
            if input_values is None:
                input_values, reference = prepare_input(
                    args.features_dir,
                    use_topogram=True,
                    use_mri=False,
                    coordinate_mode="none",
                )
                schedule = validate_axial_pruning_schedule(
                    length=int(input_values.shape[-1]),
                    roi=roi_size[-1],
                    overlap=args.overlap,
                )
                zooms = tuple(
                    float(value) for value in reference.header.get_zooms()[:3]
                )
                static_future = background.submit(
                    _timed_refiner_static_input,
                    args.features_dir,
                    input_values,
                    reference,
                )
            inference_started = time.perf_counter()
            if name.startswith("e13_fold"):
                if "e13_full75" not in outputs:
                    raise RuntimeError("E577 fold crop requires the full75 fallback")
                cropped = _predict(
                    input_values[..., FOLD_CROP_START:],
                    model,
                    roi_size=roi_size,
                    overlap=args.overlap,
                    device=device,
                )
                prediction = merge_cropped_prediction(
                    outputs["e13_full75"], cropped, FOLD_CROP_START
                )
                del cropped
            else:
                prediction = _predict(
                    input_values,
                    model,
                    roi_size=roi_size,
                    overlap=args.overlap,
                    device=device,
                )
            network_seconds[name] = time.perf_counter() - inference_started
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if name == "e13_full75":
                outputs[name] = prediction
                if zooms is None:
                    raise RuntimeError("E577 zooms unavailable")
                body_prefix_future = background.submit(
                    _timed_base_prefix,
                    e06=outputs["e06"],
                    e08=outputs["e08"],
                    e13=outputs[name],
                    zooms=zooms,
                )
            elif name.startswith("e13_fold"):
                fold_outputs.append(prediction)
                if len(fold_outputs) == E13_FOLD_COUNT:
                    outputs["e13_fold_mean"] = mean_e13_outputs(fold_outputs)
                    del fold_outputs
                    if zooms is None:
                        raise RuntimeError("E577 zooms unavailable")
                    head_prefix_future = background.submit(
                        _timed_base_prefix,
                        e06=outputs["e06"],
                        e08=outputs["e08"],
                        e13=outputs["e13_fold_mean"],
                        zooms=zooms,
                    )
            else:
                outputs[name] = prediction

        if any(
            value is None
            for value in (
                input_values,
                reference,
                static_future,
                body_prefix_future,
                head_prefix_future,
                zooms,
                schedule,
            )
        ):
            raise RuntimeError("E577 failed to prepare its frozen branch inputs")
        base_loop_non_network_seconds = time.perf_counter() - prepare_started - sum(
            network_seconds.values()
        )
        prefix_wait_started = time.perf_counter()
        body_e15, body_e21, body_prefix_work = body_prefix_future.result()
        head_e15, head_e21, head_prefix_work = head_prefix_future.result()
        prefix_wait_seconds = time.perf_counter() - prefix_wait_started

        suffix_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as branch_executor:
            body_suffix_future = branch_executor.submit(
                _timed_suffix,
                e15=body_e15,
                e21=body_e21,
                e24=outputs["e24"],
                zooms=zooms,
                projection_workers=args.projection_workers,
            )
            head_suffix_future = branch_executor.submit(
                _timed_suffix,
                e15=head_e15,
                e21=head_e21,
                e24=outputs["e24"],
                zooms=zooms,
                projection_workers=args.projection_workers,
            )
            body_e35, body_suffix_stages, body_suffix_work = body_suffix_future.result()
            head_e35, head_suffix_stages, head_suffix_work = head_suffix_future.result()
        suffix_wall_seconds = time.perf_counter() - suffix_started
        del outputs, body_e15, body_e21, head_e15, head_e21

        static_wait_started = time.perf_counter()
        refiner_input, refiner_static_seconds = static_future.result()
        refiner_static_wait_seconds = time.perf_counter() - static_wait_started
        del input_values

    body_refiners: list[np.ndarray] = []
    head_refiners: list[np.ndarray] = []
    refiner_paths = {
        "e40_refiner_seed1": args.e40_refiner_checkpoint,
        "e43_refiner_seed2": args.e43_refiner_checkpoint,
    }
    refiner_update_seconds = 0.0
    for name, checkpoint_path in refiner_paths.items():
        config_path = checkpoint_path.parent / "config.json"
        model, config = load_refiner_model(checkpoint_path, config_path, device)
        signature = (
            bool(config["use_mri"]),
            str(config["coordinate_mode"]),
            float(config["mu_scale"]),
        )
        if signature != (True, "xyz", 0.15):
            raise ValueError(f"{name}: unexpected E577 refiner input signature")

        update_started = time.perf_counter()
        update_refiner_base_channel(refiner_input, body_e35, mu_scale=0.15)
        refiner_update_seconds += time.perf_counter() - update_started
        inference_started = time.perf_counter()
        body_refiners.append(
            _predict(
                refiner_input,
                model,
                roi_size=roi_size,
                overlap=args.overlap,
                device=device,
            )
        )
        network_seconds[f"{name}_body"] = time.perf_counter() - inference_started

        update_started = time.perf_counter()
        update_refiner_base_channel(refiner_input, head_e35, mu_scale=0.15)
        refiner_update_seconds += time.perf_counter() - update_started
        inference_started = time.perf_counter()
        cropped_head = _predict(
            refiner_input[..., HEAD_REFINER_CROP_START:],
            model,
            roi_size=roi_size,
            overlap=args.overlap,
            device=device,
        )
        head_refiners.append(
            place_cropped_prediction(
                cropped_head, head_e35.shape, HEAD_REFINER_CROP_START
            )
        )
        del cropped_head
        network_seconds[f"{name}_head"] = time.perf_counter() - inference_started
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del refiner_input

    final_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as branch_executor:
        body_final_future = branch_executor.submit(
            _timed_final, body_e35, body_refiners
        )
        head_final_future = branch_executor.submit(
            _timed_final, head_e35, head_refiners
        )
        body_final, body_final_stages, body_final_work = body_final_future.result()
        head_final, head_final_stages, head_final_work = head_final_future.result()
    final_wall_seconds = time.perf_counter() - final_started
    del body_e35, head_e35, body_refiners, head_refiners

    hybrid_started = time.perf_counter()
    candidate = fixed_head_hybrid(body_final, head_final)
    hybrid_seconds = time.perf_counter() - hybrid_started

    output_started = time.perf_counter()
    _save_mu(args.output_mu, candidate)
    _save_mu(args.output_body_mu, body_final)
    _save_mu(args.output_head_mu, head_final)
    save_ct(args.output_ct, mu_to_hu(candidate), reference)
    output_seconds = time.perf_counter() - output_started
    total_seconds = time.perf_counter() - started

    payload: dict[str, Any] = {
        "schema": "bic-mac-e577-e570-axial-window-pruning/v1",
        "program": PROGRAM,
        "network_count": len(network_seconds),
        "candidate_ct_only": True,
        "pet_output_or_postprocessing": False,
        "runtime_only_optimization": True,
        "auxiliary_branch_mu_outputs_for_audit": (
            args.output_body_mu is not None or args.output_head_mu is not None
        ),
        "base_checkpoints": {
            "e06": str(args.e06_checkpoint),
            "e08": str(args.e08_checkpoint),
            "e13_full75": str(args.e13_full75_checkpoint),
            "e13_folds": [str(path) for path in args.e13_fold_checkpoints],
            "e24": str(args.e24_checkpoint),
        },
        "refiner_checkpoints": {
            name: str(path) for name, path in refiner_paths.items()
        },
        "runtime_policy": runtime_policy,
        "projection_workers_per_branch": args.projection_workers,
        "background_workers": BACKGROUND_WORKERS,
        "shape": list(candidate.shape),
        "zooms": list(zooms),
        "finite": bool(
            np.isfinite(candidate).all()
            and np.isfinite(body_final).all()
            and np.isfinite(head_final).all()
        ),
        "minimum_mu": float(candidate.min()),
        "maximum_mu": float(candidate.max()),
        "transition": [370, 390],
        "axial_pruning": {
            "fold_crop_start": FOLD_CROP_START,
            "head_refiner_crop_start": HEAD_REFINER_CROP_START,
            "head_exact_start": HEAD_EXACT_START,
            "minimum_cpu_context_slices": MINIMUM_CPU_CONTEXT_SLICES,
            "schedule": schedule,
        },
        "stage_names": {
            "body": ["e13_full75", "e15", "e21", *body_suffix_stages, *body_final_stages],
            "head": [
                "e13_fold_mean_aligned_suffix",
                "e15",
                "e21",
                *head_suffix_stages,
                *head_final_stages,
            ],
            "final": "E253_body_to_E566_head_fixed_smoothstep",
        },
        "seconds": {
            "base_loop_non_network": base_loop_non_network_seconds,
            "body_prefix_work": body_prefix_work,
            "head_prefix_work": head_prefix_work,
            "prefix_wait": prefix_wait_seconds,
            "body_suffix_work": body_suffix_work,
            "head_suffix_work": head_suffix_work,
            "suffix_wall": suffix_wall_seconds,
            "refiner_static_work": refiner_static_seconds,
            "refiner_static_wait": refiner_static_wait_seconds,
            "refiner_base_updates": refiner_update_seconds,
            "body_final_work": body_final_work,
            "head_final_work": head_final_work,
            "final_wall": final_wall_seconds,
            "fixed_hybrid": hybrid_seconds,
            "networks": network_seconds,
            "network_total": float(sum(network_seconds.values())),
            "output": output_seconds,
            "total": total_seconds,
        },
        "output_ct": str(args.output_ct),
        "output_mu": str(args.output_mu) if args.output_mu is not None else None,
        "output_body_mu": (
            str(args.output_body_mu) if args.output_body_mu is not None else None
        ),
        "output_head_mu": (
            str(args.output_head_mu) if args.output_head_mu is not None else None
        ),
    }
    if payload["network_count"] != NETWORK_COUNT:
        raise SystemExit("E577 did not execute exactly 13 network forwards")
    if not payload["finite"]:
        raise SystemExit("E577 output is non-finite")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-ct", type=Path, required=True)
    parser.add_argument("--output-mu", type=Path)
    parser.add_argument("--output-body-mu", type=Path)
    parser.add_argument("--output-head-mu", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--e06-checkpoint", type=Path, required=True)
    parser.add_argument("--e08-checkpoint", type=Path, required=True)
    parser.add_argument("--e13-full75-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--e13-fold-checkpoints", type=Path, nargs="+", required=True
    )
    parser.add_argument("--e24-checkpoint", type=Path, required=True)
    parser.add_argument("--e40-refiner-checkpoint", type=Path, required=True)
    parser.add_argument("--e43-refiner-checkpoint", type=Path, required=True)
    parser.add_argument("--roi-size", type=int, nargs=3, default=(192, 192, 192))
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument(
        "--projection-workers", type=int, default=DEFAULT_PROJECTION_WORKERS
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = execute(args)
    _write_report(args.report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
