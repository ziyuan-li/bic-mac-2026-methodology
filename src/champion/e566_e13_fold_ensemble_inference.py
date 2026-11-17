"""Run the exact deployable E566 graph with a five-fold E13 ensemble."""

from __future__ import annotations

import argparse
import json
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
from .e77_six_family_inference_v3 import (
    DEFAULT_PROJECTION_WORKERS,
    _timed_refiner_static_input,
    compose_refiners_to_e77,
)
from .e77_six_family_inference_v4 import (
    _timed_base_prefix,
    compose_base_suffix,
)
from .evaluate_cached import load_model
from .export_cached_ct import mu_to_hu
from .e77_six_family_inference_v2 import stored_nonnegative


PROGRAM = "E566 deployable five-fold E13 ACF ensemble"
BACKGROUND_WORKERS = 1
E13_FOLD_COUNT = 5


def mean_e13_outputs(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Match historical ordered float32 averaging and float16 storage."""

    arrays = tuple(arrays)
    if len(arrays) != E13_FOLD_COUNT:
        raise ValueError(f"E566 requires exactly {E13_FOLD_COUNT} E13 outputs")
    shape = tuple(int(value) for value in np.asarray(arrays[0]).shape)
    if len(shape) != 3 or any(np.asarray(array).shape != shape for array in arrays):
        raise ValueError("E13 output shapes differ or are not three-dimensional")
    total = np.zeros(shape, dtype=np.float32)
    for array in arrays:
        values = np.asarray(array, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("E13 output contains non-finite values")
        total += values
    total /= float(E13_FOLD_COUNT)
    return total.astype(np.float16)


def _validate_args(args: argparse.Namespace) -> torch.device:
    if len(args.e13_checkpoints) != E13_FOLD_COUNT:
        raise ValueError("--e13-checkpoints requires exactly five paths")
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
    return device


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = _validate_args(args)
    runtime_policy = configure_inference_runtime(device)
    base_jobs = [
        ("e06", args.e06_checkpoint),
        ("e08", args.e08_checkpoint),
        *[
            (f"e13_fold{fold}", checkpoint)
            for fold, checkpoint in enumerate(args.e13_checkpoints)
        ],
        ("e24", args.e24_checkpoint),
    ]
    base_outputs: dict[str, np.ndarray] = {}
    e13_outputs: list[np.ndarray] = []
    network_seconds: dict[str, float] = {}
    input_values = None
    reference = None
    static_future: Future[tuple[torch.Tensor, float]] | None = None
    prefix_future: Future[tuple[np.ndarray, np.ndarray, float]] | None = None
    prepare_started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=BACKGROUND_WORKERS) as background_executor:
        for name, checkpoint_path in base_jobs:
            model, use_topogram, use_mri, coordinate_mode = load_model(
                checkpoint_path, device
            )
            if not use_topogram or use_mri or coordinate_mode != "none":
                raise ValueError(f"{name}: unexpected E566 base input signature")
            if input_values is None:
                input_values, reference = prepare_input(
                    args.features_dir,
                    use_topogram=True,
                    use_mri=False,
                    coordinate_mode="none",
                )
                static_future = background_executor.submit(
                    _timed_refiner_static_input,
                    args.features_dir,
                    input_values,
                    reference,
                )
            inference_started = time.perf_counter()
            prediction = stored_nonnegative(
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

            if name.startswith("e13_fold"):
                e13_outputs.append(prediction)
                if len(e13_outputs) == E13_FOLD_COUNT:
                    if reference is None:
                        raise RuntimeError("E566 reference unavailable before E13 mean")
                    base_outputs["e13"] = mean_e13_outputs(e13_outputs)
                    del e13_outputs
                    zooms = tuple(
                        float(value) for value in reference.header.get_zooms()[:3]
                    )
                    prefix_future = background_executor.submit(
                        _timed_base_prefix,
                        e06=base_outputs["e06"],
                        e08=base_outputs["e08"],
                        e13=base_outputs["e13"],
                        zooms=zooms,
                    )
            else:
                base_outputs[name] = prediction

        if (
            reference is None
            or input_values is None
            or static_future is None
            or prefix_future is None
        ):
            raise RuntimeError("E566 input or overlap task was not prepared")
        base_loop_non_network_seconds = time.perf_counter() - prepare_started - sum(
            network_seconds.values()
        )
        prefix_wait_started = time.perf_counter()
        e15, e21, prefix_work_seconds = prefix_future.result()
        prefix_wait_seconds = time.perf_counter() - prefix_wait_started
        suffix_started = time.perf_counter()
        e35, suffix_stages = compose_base_suffix(
            e15=e15,
            e21=e21,
            e24=base_outputs["e24"],
            zooms=zooms,
            projection_workers=args.projection_workers,
        )
        suffix_seconds = time.perf_counter() - suffix_started
        stage_names = ["e13_fold_mean", "e15", "e21", *suffix_stages]
        del base_outputs, e15, e21, suffix_stages

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
            raise ValueError(f"{name}: unexpected E566 refiner input signature")
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
    e566, final_stages = compose_refiners_to_e77(
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
            np.save(handle, e566.astype(np.float16), allow_pickle=False)
        os.replace(temporary, args.output_mu)
    save_ct(args.output_ct, mu_to_hu(e566), reference)
    output_seconds = time.perf_counter() - output_started
    total_seconds = time.perf_counter() - started

    payload: dict[str, Any] = {
        "schema": "bic-mac-e566-exact-inference/v1",
        "program": PROGRAM,
        "base_checkpoints": {
            "e06": str(args.e06_checkpoint),
            "e08": str(args.e08_checkpoint),
            "e13_folds": [str(path) for path in args.e13_checkpoints],
            "e24": str(args.e24_checkpoint),
        },
        "refiner_checkpoints": {
            name: str(path) for name, path in refiner_paths.items()
        },
        "network_count": len(network_seconds),
        "candidate_ct_only": True,
        "pet_output_or_postprocessing": False,
        "background_workers": BACKGROUND_WORKERS,
        "projection_workers": args.projection_workers,
        "runtime_policy": runtime_policy,
        "shape": list(e566.shape),
        "zooms": list(zooms),
        "finite": bool(np.isfinite(e566).all()),
        "minimum_mu": float(e566.min()),
        "maximum_mu": float(e566.max()),
        "storage_precision": "float16-roundtrip at every historical stage",
        "stage_names": stage_names,
        "overlap": {
            "cpu_prefix": ["e15", "e21"],
            "concurrent_gpu_network": "e24",
            "join_before": "e24_head",
        },
        "seconds": {
            "base_loop_non_network": base_loop_non_network_seconds,
            "base_prefix_work": prefix_work_seconds,
            "base_prefix_wait": prefix_wait_seconds,
            "base_suffix": suffix_seconds,
            "refiner_static_background": refiner_static_seconds,
            "refiner_prepare_wait_and_base_update": refiner_prepare_wait_seconds,
            "networks": network_seconds,
            "network_total": float(sum(network_seconds.values())),
            "final_transforms": final_transform_seconds,
            "output": output_seconds,
            "total": total_seconds,
        },
        "output_ct": str(args.output_ct),
        "output_mu": str(args.output_mu) if args.output_mu is not None else None,
    }
    if not payload["finite"]:
        raise SystemExit("E566 output is non-finite")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-ct", type=Path, required=True)
    parser.add_argument("--output-mu", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--e06-checkpoint", type=Path, required=True)
    parser.add_argument("--e08-checkpoint", type=Path, required=True)
    parser.add_argument("--e13-checkpoints", type=Path, nargs="+", required=True)
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
