"""Run the final Team TUM BIC-MAC inference graph from a portable model directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REQUIRED_FEATURES = (
    "nacpet.nii.gz",
    "topogram.nii.gz",
    "mri_combined_in_phase.nii.gz",
    "mri_combined_out_phase.nii.gz",
)

CHECKPOINTS = {
    "e06_checkpoint": "e06/last.pt",
    "e08_checkpoint": "e08/last.pt",
    "e13_full75_checkpoint": "e13-full75/last.pt",
    "e24_checkpoint": "e24/last.pt",
    "e40_refiner_checkpoint": "e40/last.pt",
    "e43_refiner_checkpoint": "e43/last.pt",
}
E13_FOLDS = tuple(f"e13-fold{fold}/best-objective.pt" for fold in range(5))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def validate_model_dir(model_dir: Path, *, verify_hashes: bool) -> None:
    closure_path = model_dir / "runtime-closure.json"
    closure = _read_json(closure_path)
    rows = closure.get("models")
    if (
        closure.get("schema") != "bicmac-e580-runtime-closure/v1"
        or closure.get("checkpoint_count") != 11
        or closure.get("config_count") != 11
        or not isinstance(rows, list)
        or len(rows) != 11
    ):
        raise RuntimeError(f"invalid runtime closure: {closure_path}")

    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("runtime closure contains a malformed model row")
        for kind in ("checkpoint", "config"):
            relative = Path(str(row[kind]))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe {kind} path: {relative}")
            path = model_dir / relative
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(path)
            expected_bytes = int(row[f"{kind}_bytes"])
            if path.stat().st_size != expected_bytes:
                raise RuntimeError(f"size mismatch: {path}")
            if verify_hashes and _sha256(path) != row[f"{kind}_sha256"]:
                raise RuntimeError(f"SHA-256 mismatch: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-ct", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--projection-workers", type=int, default=4)
    parser.add_argument(
        "--verify-hashes",
        action="store_true",
        help="hash all 1.61 GB of model files before inference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [
        name for name in REQUIRED_FEATURES if not (args.features_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing feature files: {', '.join(missing)}")
    validate_model_dir(args.model_dir, verify_hashes=args.verify_hashes)

    from .e577_e570_axial_window_pruning import execute

    inference_args = argparse.Namespace(
        features_dir=args.features_dir,
        output_ct=args.output_ct,
        output_mu=None,
        output_body_mu=None,
        output_head_mu=None,
        report=args.report,
        e13_fold_checkpoints=[args.model_dir / path for path in E13_FOLDS],
        roi_size=(192, 192, 192),
        overlap=0.25,
        sw_batch_size=1,
        projection_workers=args.projection_workers,
        device=args.device,
        **{name: args.model_dir / path for name, path in CHECKPOINTS.items()},
    )
    payload = execute(inference_args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.report)


if __name__ == "__main__":
    main()
