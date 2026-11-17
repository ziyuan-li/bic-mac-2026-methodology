"""Run full-volume cached inference and exact mu-mask evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from .data import load_fold_subjects, spatial_coordinate_channels
from .model import (
    DirectMuUNet,
    FullResolutionMRIProjectionConditionedMuUNet,
    MRIProjectionConditionedMuUNet,
    ProjectionConditionedMuUNet,
)


class BlendedModel(torch.nn.Module):
    def __init__(
        self,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        alpha_b: float,
        model_a_uses_topogram: bool,
        model_b_uses_topogram: bool,
    ) -> None:
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.alpha_b = alpha_b
        self.model_a_uses_topogram = model_a_uses_topogram
        self.model_b_uses_topogram = model_b_uses_topogram

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        input_a = values if self.model_a_uses_topogram else values[:, :2]
        input_b = values if self.model_b_uses_topogram else values[:, :2]
        return (1.0 - self.alpha_b) * self.model_a(input_a) + self.alpha_b * self.model_b(input_b)


def build_input(
    subject_dir: Path,
    use_topogram: bool,
    use_mri: bool = False,
    coordinate_mode: str = "none",
) -> torch.Tensor:
    metadata = json.loads((subject_dir / "metadata.json").read_text(encoding="utf-8"))
    nac_log = np.asarray(np.load(subject_dir / "nac_log.npy", mmap_mode="r"), dtype=np.float32)
    absolute = np.clip(nac_log / 7.0, 0.0, 2.0)
    relative = np.clip(nac_log / max(float(metadata["nac_log_p99"]), 1e-6), 0.0, 2.0)
    channels = [absolute, relative]
    if use_topogram:
        topogram_2d = np.asarray(
            np.load(subject_dir / "topogram.npy", mmap_mode="r"), dtype=np.float32
        )
        channels.append(np.broadcast_to(topogram_2d[:, None, :], nac_log.shape))
    if use_mri:
        channels.extend(
            np.asarray(np.load(subject_dir / name, mmap_mode="r"), dtype=np.float32)
            for name in ("mri_in.npy", "mri_out.npy")
        )
    full_slices = tuple(slice(0, size) for size in nac_log.shape)
    coordinates = spatial_coordinate_channels(nac_log.shape, full_slices, coordinate_mode)
    channels.extend(coordinates)
    return torch.from_numpy(np.ascontiguousarray(np.stack(channels, axis=0)[None]))


def evaluate_subject(
    model: torch.nn.Module,
    cache_dir: Path,
    subject: str,
    device: torch.device,
    roi_size: tuple[int, int, int],
    overlap: float,
    sw_batch_size: int,
    prediction_dir: Path | None,
    use_topogram: bool,
    use_mri: bool,
    coordinate_mode: str,
) -> dict[str, Any]:
    subject_dir = cache_dir / "train" / subject
    inputs = build_input(subject_dir, use_topogram, use_mri, coordinate_mode)
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = sliding_window_inference(
            inputs,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode="gaussian",
            sw_device=device,
            device=torch.device("cpu"),
            progress=False,
        )
    inference_seconds = time.perf_counter() - started
    prediction_mu = np.asarray(prediction[0, 0].float().numpy(), dtype=np.float32)
    np.maximum(prediction_mu, 0.0, out=prediction_mu)
    target = np.asarray(np.load(subject_dir / "mu.npy", mmap_mode="r"), dtype=np.float32)
    eval_mask = np.asarray(np.load(subject_dir / "eval_mask.npy", mmap_mode="r"), dtype=bool)
    responsibility_mask = np.asarray(
        np.load(subject_dir / "prediction_mask.npy", mmap_mode="r"), dtype=bool
    )
    result = {
        "subject": subject,
        "eval_mu_mae": float(np.mean(np.abs(prediction_mu[eval_mask] - target[eval_mask]))),
        "prediction_mu_mae": float(
            np.mean(np.abs(prediction_mu[responsibility_mask] - target[responsibility_mask]))
        ),
        "prediction_mu_min": float(np.min(prediction_mu)),
        "prediction_mu_max": float(np.max(prediction_mu)),
        "inference_seconds": inference_seconds,
    }
    if prediction_dir is not None:
        output_dir = prediction_dir / subject
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary = output_dir / "mu.npy.tmp"
        with temporary.open("wb") as handle:
            np.save(handle, prediction_mu.astype(np.float16), allow_pickle=False)
        temporary.replace(output_dir / "mu.npy")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path)
    parser.add_argument("--blend-alpha", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi-size", type=int, nargs=3, default=(192, 192, 192))
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, bool, bool, str]:
    config_path = checkpoint_path.resolve().parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    projected_topogram = bool(config.get("projected_topogram", False))
    use_topogram = bool(config.get("use_topogram", False)) or projected_topogram
    use_mri = bool(config.get("use_mri", False))
    tissue_aux_weight = float(config.get("tissue_aux_weight", 0.0))
    coordinate_mode = str(config.get("coordinate_mode", "none"))
    coordinate_channels = {"none": 0, "z": 1, "xyz": 3}[coordinate_mode]
    if projected_topogram:
        if bool(config.get("full_resolution_mri_adapter", False)):
            model_class = FullResolutionMRIProjectionConditionedMuUNet
        else:
            model_class = (
                MRIProjectionConditionedMuUNet
                if use_mri
                else ProjectionConditionedMuUNet
            )
        model: torch.nn.Module = model_class(
            base_channels=int(config["base_channels"]),
            output_channels=5 if tissue_aux_weight > 0 else 1,
            coordinate_channels=coordinate_channels,
        ).to(device)
    else:
        model = DirectMuUNet(
            in_channels=(3 if use_topogram else 2) + coordinate_channels,
            base_channels=int(config["base_channels"]),
            output_channels=5 if tissue_aux_weight > 0 else 1,
        ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, use_topogram, use_mri, coordinate_mode


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _, subjects = load_fold_subjects(args.folds, args.fold)
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]

    device = torch.device("cuda", 0)
    model, use_topogram, use_mri, coordinate_mode = load_model(args.checkpoint, device)
    checkpoints = [str(args.checkpoint.resolve())]
    if args.checkpoint_b is not None:
        if not 0.0 <= args.blend_alpha <= 1.0:
            raise ValueError("--blend-alpha must be between zero and one")
        model_b, model_b_uses_topogram, model_b_uses_mri, model_b_coordinate_mode = load_model(
            args.checkpoint_b, device
        )
        if use_mri or model_b_uses_mri or coordinate_mode != "none" or model_b_coordinate_mode != "none":
            raise ValueError("checkpoint blending with MRI or coordinate inputs is not supported")
        model = BlendedModel(
            model,
            model_b,
            args.blend_alpha,
            use_topogram,
            model_b_uses_topogram,
        ).to(device).eval()
        use_topogram = use_topogram or model_b_uses_topogram
        checkpoints.append(str(args.checkpoint_b.resolve()))

    prediction_dir = output_dir / "predictions" if args.save_predictions else None
    rows = []
    for subject in tqdm(subjects, desc="Full-volume inference"):
        rows.append(
            evaluate_subject(
                model,
                args.cache_dir,
                subject,
                device,
                tuple(args.roi_size),
                args.overlap,
                args.sw_batch_size,
                prediction_dir,
                use_topogram,
                use_mri,
                coordinate_mode,
            )
        )
    with (output_dir / "subjects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoints": checkpoints,
        "blend_alpha_b": args.blend_alpha if args.checkpoint_b is not None else None,
        "fold": args.fold,
        "subject_count": len(rows),
        "roi_size": list(args.roi_size),
        "overlap": args.overlap,
        "eval_mu_mae": float(np.mean([row["eval_mu_mae"] for row in rows])),
        "prediction_mu_mae": float(np.mean([row["prediction_mu_mae"] for row in rows])),
        "mean_inference_seconds": float(np.mean([row["inference_seconds"] for row in rows])),
        "max_inference_seconds": float(np.max([row["inference_seconds"] for row in rows])),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
