"""Launch the exact E569/E577 final-container graph."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .e577_e570_axial_window_pruning import main as inference_main


DEPLOYMENT_PATH = Path("/app/model/deployment.json")
RUNTIME_REPORT_PATH = Path("/tmp/e577-runtime-report.json")
SCHEMA = "bicmac-e569-e577-deployment/v1"
BASE_LAYOUT = (
    ("e06_projected_topogram", "e06", "--e06-checkpoint"),
    ("e08_pet_surrogate", "e08", "--e08-checkpoint"),
    ("e13_acf_full75", "e13-full75", "--e13-full75-checkpoint"),
    ("e24_rank_ct", "e24", "--e24-checkpoint"),
    ("e40_refiner_seed1", "e40", "--e40-refiner-checkpoint"),
    ("e43_refiner_seed2", "e43", "--e43-refiner-checkpoint"),
)
E13_FOLD_LAYOUT = tuple(
    (f"e13_acf_fold{fold}", f"e13-fold{fold}") for fold in range(5)
)


def expected_deployment() -> dict[str, Any]:
    checkpoints = {
        family: f"/app/model/{short_name}/last.pt"
        for family, short_name, _ in BASE_LAYOUT
    }
    checkpoints.update(
        {
            family: f"/app/model/{short_name}/best-objective.pt"
            for family, short_name in E13_FOLD_LAYOUT
        }
    )
    return {
        "schema": SCHEMA,
        "candidate": "E569/E577",
        "inference_module": "champion.e577_e570_axial_window_pruning",
        "features_dir": "/data/features",
        "output_ct": "/data/output/ct.nii.gz",
        "runtime_report": str(RUNTIME_REPORT_PATH),
        "output_files": ["ct.nii.gz"],
        "device": "cuda:0",
        "network_forward_count": 13,
        "unique_checkpoint_count": 11,
        "roi_size": [192, 192, 192],
        "overlap": 0.25,
        "sw_batch_size": 1,
        "projection_workers": 4,
        "background_workers": 3,
        "transition": [370, 390],
        "fold_crop_start": 144,
        "head_refiner_crop_start": 288,
        "head_exact_start": 370,
        "minimum_cpu_context_slices": 64,
        "checkpoints": checkpoints,
    }


def validate_deployment(config: dict[str, Any]) -> dict[str, Any]:
    expected = expected_deployment()
    unknown = sorted(set(config) - set(expected))
    missing = sorted(set(expected) - set(config))
    if unknown or missing:
        raise ValueError(
            f"E578 deployment keys differ: missing={missing}, unknown={unknown}"
        )
    mismatches = sorted(
        key for key, value in expected.items() if config.get(key) != value
    )
    if mismatches:
        raise ValueError(f"E578 deployment values differ: {mismatches}")
    return expected


def build_inference_argv(config: dict[str, Any]) -> list[str]:
    deployment = validate_deployment(config)
    checkpoints = deployment["checkpoints"]
    argv = [
        "champion.e577_e570_axial_window_pruning",
        "--features-dir",
        deployment["features_dir"],
        "--output-ct",
        deployment["output_ct"],
        "--report",
        deployment["runtime_report"],
    ]
    for family, _, flag in BASE_LAYOUT:
        if family == "e13_acf_full75":
            argv.extend((flag, checkpoints[family]))
    argv.extend(
        (
            "--e13-fold-checkpoints",
            *(checkpoints[family] for family, _ in E13_FOLD_LAYOUT),
        )
    )
    for family, _, flag in BASE_LAYOUT:
        if family != "e13_acf_full75":
            argv.extend((flag, checkpoints[family]))
    argv.extend(
        (
            "--roi-size",
            *(str(value) for value in deployment["roi_size"]),
            "--overlap",
            str(deployment["overlap"]),
            "--sw-batch-size",
            str(deployment["sw_batch_size"]),
            "--projection-workers",
            str(deployment["projection_workers"]),
            "--device",
            deployment["device"],
        )
    )
    forbidden = {"--output-mu", "--output-body-mu", "--output-head-mu"}
    if forbidden & set(argv):
        raise RuntimeError("final-container inference may not emit mu sidecars")
    return argv


def validate_runtime_report(payload: dict[str, Any]) -> None:
    pruning = payload.get("axial_pruning", {})
    schedule = pruning.get("schedule", {})
    schedule_gates = schedule.get("gates", {})
    if not (
        payload.get("schema") == "bic-mac-e577-e570-axial-window-pruning/v1"
        and payload.get("network_count") == 13
        and payload.get("candidate_ct_only") is True
        and payload.get("pet_output_or_postprocessing") is False
        and payload.get("runtime_only_optimization") is True
        and payload.get("auxiliary_branch_mu_outputs_for_audit") is False
        and payload.get("output_mu") is None
        and payload.get("output_body_mu") is None
        and payload.get("output_head_mu") is None
        and payload.get("transition") == [370, 390]
        and pruning.get("fold_crop_start") == 144
        and pruning.get("head_refiner_crop_start") == 288
        and pruning.get("head_exact_start") == 370
        and pruning.get("minimum_cpu_context_slices") == 64
        and schedule.get("effective_cpu_context_slices") == 96
        and bool(schedule_gates)
        and all(value is True for value in schedule_gates.values())
        and payload.get("finite") is True
    ):
        raise RuntimeError("E578 final-container runtime report violates its contract")


def main() -> None:
    config = json.loads(DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("E578 deployment manifest must contain an object")
    if RUNTIME_REPORT_PATH.exists():
        RUNTIME_REPORT_PATH.unlink()
    sys.argv = build_inference_argv(config)
    inference_main()

    runtime = json.loads(RUNTIME_REPORT_PATH.read_text(encoding="utf-8"))
    if not isinstance(runtime, dict):
        raise RuntimeError("E578 runtime report must contain an object")
    validate_runtime_report(runtime)


if __name__ == "__main__":
    main()
