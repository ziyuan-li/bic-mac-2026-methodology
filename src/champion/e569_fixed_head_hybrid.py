"""Materialize the fixed E253-body/E566-head CT hybrid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .tissue_blend import head_weight_profile


TRANSITION_START = 370
TRANSITION_END = 390
CHUNK_SLICES = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_head_hybrid(
    body: np.ndarray,
    head: np.ndarray,
    *,
    transition_start: int = TRANSITION_START,
    transition_end: int = TRANSITION_END,
    chunk_slices: int = CHUNK_SLICES,
) -> np.ndarray:
    """Keep body exactly, keep head exactly, and smooth only the fixed neck band."""

    if body.ndim != 3 or head.shape != body.shape:
        raise ValueError("body and head predictions must share one 3D shape")
    if chunk_slices <= 0:
        raise ValueError("chunk_slices must be positive")
    if transition_start < 0 or transition_end < transition_start:
        raise ValueError("invalid transition interval")
    if not np.isfinite(body).all() or not np.isfinite(head).all():
        raise ValueError("body and head predictions must be finite")

    length = body.shape[2]
    start = min(transition_start, length)
    end = min(transition_end, length)
    output = np.empty(body.shape, dtype=np.float16)
    output[:, :, :start] = np.asarray(body[:, :, :start], dtype=np.float16)
    if end < length:
        output[:, :, end:] = np.asarray(head[:, :, end:], dtype=np.float16)
    if start < end:
        profile = head_weight_profile(length, transition_start, transition_end)
        for chunk_start in range(start, end, chunk_slices):
            chunk_end = min(chunk_start + chunk_slices, end)
            slices = np.s_[:, :, chunk_start:chunk_end]
            body_values = np.asarray(body[slices], dtype=np.float32)
            head_values = np.asarray(head[slices], dtype=np.float32)
            weights = profile[chunk_start:chunk_end][None, None, :]
            values = body_values + weights * (head_values - body_values)
            np.maximum(values, 0.0, out=values)
            output[slices] = values.astype(np.float16)
    return output


def materialize(
    *,
    body_dir: Path,
    head_dir: Path,
    output_dir: Path,
    subjects: Sequence[str],
) -> dict[str, Any]:
    rows = {}
    for subject in subjects:
        body_path = body_dir / subject / "mu.npy"
        head_path = head_dir / subject / "mu.npy"
        body = np.load(body_path, mmap_mode="r")
        head = np.load(head_path, mmap_mode="r")
        candidate = fixed_head_hybrid(body, head)
        subject_dir = output_dir / subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        output_path = subject_dir / "mu.npy"
        temporary = output_path.with_name(output_path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, candidate, allow_pickle=False)
        os.replace(temporary, output_path)
        rows[subject] = {
            "body_sha256": _sha256(body_path),
            "head_sha256": _sha256(head_path),
            "output_sha256": _sha256(output_path),
            "shape": list(candidate.shape),
            "finite": bool(np.isfinite(candidate).all()),
            "body_bitwise_exact_below_370": bool(
                np.array_equal(candidate[:, :, :TRANSITION_START], body[:, :, :TRANSITION_START])
            ),
            "head_bitwise_exact_from_390": bool(
                np.array_equal(candidate[:, :, TRANSITION_END:], head[:, :, TRANSITION_END:])
            ),
        }
        print(f"[{len(rows)}/{len(subjects)}] {subject}", flush=True)
    gates = {
        "subject_count_matches": len(rows) == len(subjects),
        "all_finite": all(row["finite"] for row in rows.values()),
        "body_bitwise_exact_every_subject": all(
            row["body_bitwise_exact_below_370"] for row in rows.values()
        ),
        "head_bitwise_exact_every_subject": all(
            row["head_bitwise_exact_from_390"] for row in rows.values()
        ),
    }
    return {
        "schema": "bic-mac-e569-fixed-head-hybrid/v1",
        "transition": [TRANSITION_START, TRANSITION_END],
        "subjects": list(subjects),
        "body_dir": str(body_dir),
        "head_dir": str(head_dir),
        "output_dir": str(output_dir),
        "per_subject": rows,
        "gates": gates,
        "passes_all_gates": all(gates.values()),
        "candidate_ct_only": True,
        "pet_output_or_postprocessing": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-dir", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = materialize(
        body_dir=args.body_dir,
        head_dir=args.head_dir,
        output_dir=args.output_dir,
        subjects=args.subjects,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.report)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
