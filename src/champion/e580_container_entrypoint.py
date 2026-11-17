"""Validate the complete E580 model closure, then launch unchanged E578 inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .e578_container_entrypoint import (
    DEPLOYMENT_PATH,
    main as e578_main,
    validate_deployment,
)


MODEL_ROOT = Path("/app/model")
CLOSURE_PATH = MODEL_ROOT / "runtime-closure.json"
CLOSURE_SCHEMA = "bicmac-e580-runtime-closure/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_runtime_closure(
    deployment: dict[str, Any],
    *,
    model_root: Path = MODEL_ROOT,
    closure_path: Path = CLOSURE_PATH,
) -> dict[str, Any]:
    expected = validate_deployment(deployment)
    closure = _read_object(closure_path)
    rows = closure.get("models")
    if not (
        closure.get("schema") == CLOSURE_SCHEMA
        and closure.get("candidate") == "E569/E577 packaged by E580"
        and closure.get("scientific_candidate_changed") is False
        and closure.get("runtime_graph_changed") is False
        and closure.get("checkpoint_count") == 11
        and closure.get("config_count") == 11
        and isinstance(rows, list)
        and len(rows) == 11
    ):
        raise RuntimeError("E580 runtime closure header differs")

    observed_families = [row.get("family") for row in rows if isinstance(row, dict)]
    if len(observed_families) != len(rows) or len(set(observed_families)) != len(rows):
        raise RuntimeError("E580 runtime closure families are malformed")
    if set(observed_families) != set(expected["checkpoints"]):
        raise RuntimeError("E580 runtime closure family set differs")

    for row in rows:
        family = row["family"]
        checkpoint = Path(expected["checkpoints"][family])
        try:
            checkpoint_relative = checkpoint.relative_to(MODEL_ROOT)
        except ValueError as error:
            raise RuntimeError(f"unsafe checkpoint path for {family}") from error
        config_relative = checkpoint_relative.with_name("config.json")
        if row.get("checkpoint") != checkpoint_relative.as_posix():
            raise RuntimeError(f"E580 checkpoint path differs for {family}")
        if row.get("config") != config_relative.as_posix():
            raise RuntimeError(f"E580 config path differs for {family}")
        checkpoint_path = model_root / checkpoint_relative
        config_path = model_root / config_relative
        if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
            raise RuntimeError(f"E580 checkpoint is missing or unsafe: {family}")
        if checkpoint_path.stat().st_size != row.get("checkpoint_bytes"):
            raise RuntimeError(f"E580 checkpoint size differs: {family}")
        if not config_path.is_file() or config_path.is_symlink():
            raise RuntimeError(f"E580 config is missing or unsafe: {family}")
        if config_path.stat().st_size != row.get("config_bytes"):
            raise RuntimeError(f"E580 config size differs: {family}")
        if _sha256(config_path) != row.get("config_sha256"):
            raise RuntimeError(f"E580 config hash differs: {family}")
        _read_object(config_path)
    return closure


def main() -> None:
    deployment = _read_object(DEPLOYMENT_PATH)
    validate_runtime_closure(deployment)
    e578_main()


if __name__ == "__main__":
    main()
