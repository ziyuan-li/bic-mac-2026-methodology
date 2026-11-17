"""Train a bounded refiner on one CV fold or all out-of-fold predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import resolve_training_subjects
from .refiner import MuRefinementPatchDataset, MuResidualRefiner
from .refiner_full_model_initialization import (
    configure_inserted_input_channel_only_training,
    initialize_refiner_full_model_with_inserted_channel,
)
from .train import run_epoch


def initialize_refiner_trunk(
    model: MuResidualRefiner,
    checkpoint_path: Path,
    preserve_target_input_channels: Sequence[int] = (),
) -> dict[str, object]:
    """Load a source trunk, optionally retaining target-semantic input columns."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    source = checkpoint.get("model")
    if not isinstance(source, dict):
        raise ValueError("trunk initialization checkpoint has no model state")

    target = model.state_dict()
    skipped = {"output.weight", "output.bias"}
    input_weights = {"stem.body.0.weight", "stem.skip.weight"}
    preserved = tuple(int(value) for value in preserve_target_input_channels)
    if len(set(preserved)) != len(preserved):
        raise ValueError("preserved target input channels must be unique")
    if tuple(sorted(preserved)) != preserved:
        raise ValueError("preserved target input channels must be sorted")
    input_channel_count = int(target["stem.body.0.weight"].shape[1])
    if any(value < 0 or value >= input_channel_count for value in preserved):
        raise ValueError(
            "preserved target input channel is outside the model input range"
        )
    transferable = {}
    for name, target_value in target.items():
        if name in skipped:
            continue
        if name not in source:
            raise ValueError(f"trunk initialization is missing parameter: {name}")
        source_value = source[name]
        if source_value.shape != target_value.shape:
            raise ValueError(
                f"trunk initialization shape mismatch for {name}: "
                f"{tuple(source_value.shape)} != {tuple(target_value.shape)}"
            )
        if preserved and name in input_weights:
            source_value = source_value.clone()
            source_value[:, preserved] = target_value[:, preserved]
        transferable[name] = source_value

    result = model.load_state_dict(transferable, strict=False)
    if set(result.missing_keys) != skipped or result.unexpected_keys:
        raise ValueError(
            "unexpected trunk initialization state: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    with torch.no_grad():
        model.output.weight[:1].zero_()
        model.output.bias[:1].zero_()

    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "sha256": digest,
        "loaded_parameter_count": len(transferable),
        "skipped_parameters": sorted(skipped),
        "preserved_target_input_channels": list(preserved),
        "transferred_source_input_channels": [
            value for value in range(input_channel_count) if value not in preserved
        ],
        "semantically_merged_input_parameters": (
            sorted(input_weights) if preserved else []
        ),
        "output_reset_to_zero": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-prediction-dir", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    split = parser.add_mutually_exclusive_group(required=True)
    split.add_argument("--fold", type=int)
    split.add_argument(
        "--train-all",
        action="store_true",
        help=(
            "Train on every assigned subject. The same subjects form a "
            "monitoring-only validation loader; use a pre-registered final epoch."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--scheduler-epochs", type=int)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(192, 192, 192))
    parser.add_argument("--base-channels", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--train-samples-per-subject", type=int, default=8)
    parser.add_argument("--val-samples-per-subject", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--correction-scale", type=float, default=0.03)
    parser.add_argument("--mu-scale", type=float, default=0.15)
    parser.add_argument("--use-topogram", action="store_true")
    parser.add_argument("--use-mri", action="store_true")
    parser.add_argument("--mri-translation-probability", type=float, default=0.0)
    parser.add_argument(
        "--mri-translation-max-voxels",
        type=int,
        nargs=3,
        default=(0, 0, 0),
    )
    parser.add_argument("--mri-gamma-probability", type=float, default=0.0)
    parser.add_argument(
        "--mri-gamma-range",
        type=float,
        nargs=2,
        default=(1.0, 1.0),
    )
    parser.add_argument("--coordinate-mode", choices=("none", "z", "xyz"), default="xyz")
    parser.add_argument("--smooth-weight", type=float, default=0.10)
    parser.add_argument("--projection-weight", type=float, default=0.05)
    parser.add_argument("--acf-weight", type=float, default=0.02)
    parser.add_argument("--tissue-bias-weight", type=float, default=0.02)
    parser.add_argument("--tissue-aux-weight", type=float, default=0.0)
    parser.add_argument("--base-preservation-smooth-weight", type=float, default=0.0)
    parser.add_argument("--base-preservation-projection-weight", type=float, default=0.0)
    parser.add_argument("--base-preservation-acf-weight", type=float, default=0.0)
    parser.add_argument(
        "--preserve-teacher-physics",
        action="store_true",
        help=(
            "Apply the smooth, projection, and ACF preservation penalties "
            "against the frozen teacher instead of the deployable base map."
        ),
    )
    parser.add_argument("--eval-only-loss", action="store_true")
    parser.add_argument("--teacher-prediction-dir", type=Path)
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument(
        "--teacher-mask",
        choices=("prediction", "eval"),
        default="prediction",
    )
    parser.add_argument(
        "--teacher-reliability-scale",
        type=float,
        default=0.0,
        help=(
            "Exponentially downweight teacher voxels by teacher-to-target "
            "absolute error; zero disables reliability weighting."
        ),
    )
    parser.add_argument(
        "--teacher-improvement-only",
        action="store_true",
        help=(
            "Distill only where the frozen teacher is closer to the CT target "
            "than the frozen base prediction."
        ),
    )
    parser.add_argument(
        "--teacher-oracle-endpoint",
        action="store_true",
        help=(
            "Distill the teacher where it beats the base by the frozen margin "
            "and otherwise explicitly reproduce the base endpoint."
        ),
    )
    parser.add_argument(
        "--teacher-oracle-margin",
        type=float,
        default=0.0,
        help="Required teacher error advantage for selecting the teacher endpoint.",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--initialize-trunk-from",
        type=Path,
        help=(
            "Load every shape-compatible source parameter except output.weight "
            "and output.bias, which remain zero initialized."
        ),
    )
    parser.add_argument(
        "--preserve-target-input-channels",
        type=int,
        nargs="*",
        default=(),
        help=(
            "For source trunk initialization, retain these sorted input-channel "
            "columns from the seeded target stem instead of loading them from "
            "the source checkpoint."
        ),
    )
    parser.add_argument(
        "--initialize-full-model-from",
        type=Path,
        help=(
            "Load every source parameter while expanding the stem for one "
            "exact-zero inserted input channel."
        ),
    )
    parser.add_argument(
        "--insert-zero-input-channel",
        type=int,
        help="Target input-channel index inserted during full-model initialization.",
    )
    parser.add_argument(
        "--train-inserted-input-channel-only",
        action="store_true",
        help=(
            "Freeze every inherited parameter and optimize only the inserted "
            "input column in both stem weights. Requires zero weight decay."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.smooth_weight,
        args.projection_weight,
        args.acf_weight,
        args.tissue_bias_weight,
        args.tissue_aux_weight,
        args.base_preservation_smooth_weight,
        args.base_preservation_projection_weight,
        args.base_preservation_acf_weight,
    ) < 0.0:
        raise ValueError("loss weights must be non-negative")
    if not 0.0 <= args.teacher_weight <= 1.0:
        raise ValueError("--teacher-weight must be between zero and one")
    if args.teacher_reliability_scale < 0.0:
        raise ValueError("--teacher-reliability-scale must be non-negative")
    if (args.teacher_prediction_dir is None) != (args.teacher_weight == 0.0):
        raise ValueError(
            "--teacher-prediction-dir and a positive --teacher-weight "
            "must be provided together"
        )
    if args.teacher_reliability_scale > 0.0 and args.teacher_weight == 0.0:
        raise ValueError(
            "--teacher-reliability-scale requires a positive --teacher-weight"
        )
    if args.teacher_improvement_only and args.teacher_weight == 0.0:
        raise ValueError(
            "--teacher-improvement-only requires a positive --teacher-weight"
        )
    if args.teacher_oracle_margin < 0.0:
        raise ValueError("--teacher-oracle-margin must be non-negative")
    if args.teacher_oracle_endpoint and args.teacher_weight == 0.0:
        raise ValueError(
            "--teacher-oracle-endpoint requires a positive --teacher-weight"
        )
    if args.teacher_oracle_endpoint and args.teacher_improvement_only:
        raise ValueError(
            "--teacher-oracle-endpoint and --teacher-improvement-only are "
            "mutually exclusive"
        )
    if args.teacher_oracle_margin > 0.0 and not args.teacher_oracle_endpoint:
        raise ValueError(
            "--teacher-oracle-margin requires --teacher-oracle-endpoint"
        )
    if args.teacher_oracle_endpoint and args.teacher_reliability_scale > 0.0:
        raise ValueError(
            "--teacher-reliability-scale is not supported with "
            "--teacher-oracle-endpoint"
        )
    if args.preserve_teacher_physics and args.teacher_weight == 0.0:
        raise ValueError(
            "--preserve-teacher-physics requires a positive --teacher-weight"
        )
    if args.preserve_teacher_physics and not any(
        weight > 0.0
        for weight in (
            args.base_preservation_smooth_weight,
            args.base_preservation_projection_weight,
            args.base_preservation_acf_weight,
        )
    ):
        raise ValueError(
            "--preserve-teacher-physics requires a positive preservation weight"
        )
    if (
        args.preserve_target_input_channels
        and args.initialize_trunk_from is None
    ):
        raise ValueError(
            "--preserve-target-input-channels requires --initialize-trunk-from"
        )
    if args.initialize_trunk_from is not None and args.initialize_full_model_from is not None:
        raise ValueError(
            "--initialize-trunk-from and --initialize-full-model-from are mutually exclusive"
        )
    if (args.initialize_full_model_from is None) != (
        args.insert_zero_input_channel is None
    ):
        raise ValueError(
            "--initialize-full-model-from and --insert-zero-input-channel are required together"
        )
    if (
        args.train_inserted_input_channel_only
        and args.initialize_full_model_from is None
    ):
        raise ValueError(
            "--train-inserted-input-channel-only requires full-model initialization"
        )
    if args.train_inserted_input_channel_only and args.weight_decay != 0.0:
        raise ValueError(
            "--train-inserted-input-channel-only requires --weight-decay 0"
        )
    if not 0.0 <= args.mri_translation_probability <= 1.0:
        raise ValueError("--mri-translation-probability must be between zero and one")
    if min(args.mri_translation_max_voxels) < 0:
        raise ValueError("--mri-translation-max-voxels must be nonnegative")
    if args.mri_translation_probability > 0.0 and not args.use_mri:
        raise ValueError("MRI translation augmentation requires --use-mri")
    if not 0.0 <= args.mri_gamma_probability <= 1.0:
        raise ValueError("--mri-gamma-probability must be between zero and one")
    if (
        len(args.mri_gamma_range) != 2
        or min(args.mri_gamma_range) <= 0.0
        or args.mri_gamma_range[0] > args.mri_gamma_range[1]
    ):
        raise ValueError("--mri-gamma-range must contain two ordered positives")
    if args.mri_gamma_probability > 0.0 and not args.use_mri:
        raise ValueError("MRI gamma augmentation requires --use-mri")
    scheduler_epochs = args.scheduler_epochs or args.epochs
    if scheduler_epochs < args.epochs:
        raise ValueError("--scheduler-epochs must be at least --epochs")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    train_subjects, val_subjects = resolve_training_subjects(
        args.folds,
        fold=args.fold,
        train_all=args.train_all,
    )
    dataset_kwargs = {
        "cache_dir": args.cache_dir,
        "base_prediction_dir": args.base_prediction_dir,
        "patch_size": args.patch_size,
        "use_topogram": args.use_topogram,
        "use_mri": args.use_mri,
        "coordinate_mode": args.coordinate_mode,
        "mu_scale": args.mu_scale,
        "teacher_prediction_dir": args.teacher_prediction_dir,
        "mri_translation_probability": args.mri_translation_probability,
        "mri_translation_max_voxels": args.mri_translation_max_voxels,
        "mri_gamma_probability": args.mri_gamma_probability,
        "mri_gamma_range": args.mri_gamma_range,
    }
    train_dataset = MuRefinementPatchDataset(
        subjects=train_subjects,
        samples_per_subject=args.train_samples_per_subject,
        augment=True,
        **dataset_kwargs,
    )
    val_dataset = MuRefinementPatchDataset(
        subjects=val_subjects,
        samples_per_subject=args.val_samples_per_subject,
        augment=False,
        **dataset_kwargs,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    coordinate_channels = {"none": 0, "z": 1, "xyz": 3}[args.coordinate_mode]
    input_channels = (
        3
        + (1 if args.use_topogram else 0)
        + (2 if args.use_mri else 0)
        + coordinate_channels
    )
    model = MuResidualRefiner(
        in_channels=input_channels,
        base_channels=args.base_channels,
        correction_scale=args.correction_scale,
        mu_scale=args.mu_scale,
        output_channels=5 if args.tissue_aux_weight > 0.0 else 1,
    )
    trunk_initialization = (
        initialize_refiner_trunk(
            model,
            args.initialize_trunk_from,
            args.preserve_target_input_channels,
        )
        if args.initialize_trunk_from is not None
        else None
    )
    full_model_initialization = (
        initialize_refiner_full_model_with_inserted_channel(
            model,
            args.initialize_full_model_from,
            args.insert_zero_input_channel,
        )
        if args.initialize_full_model_from is not None
        else None
    )
    model = model.to(device)
    inserted_input_channel_training = (
        configure_inserted_input_channel_only_training(
            model,
            args.insert_zero_input_channel,
        )
        if args.train_inserted_input_channel_only
        else None
    )
    optimizer_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not optimizer_parameters:
        raise ValueError("training configuration has no trainable parameters")
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_epochs)
    scaler = torch.amp.GradScaler("cuda")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        input_channels=input_channels,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        trunk_initialization=trunk_initialization,
        full_model_initialization=full_model_initialization,
        inserted_input_channel_training=inserted_input_channel_training,
    )
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value.resolve())
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    best_val = float("inf")
    best_objective = float("inf")
    for epoch in range(args.epochs):
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            1,
            optimizer,
            scaler,
            args.accumulation_steps,
            args.tissue_aux_weight,
            1.0,
            args.smooth_weight,
            args.projection_weight,
            args.acf_weight,
            args.tissue_bias_weight,
            0.0,
            args.eval_only_loss,
            args.teacher_weight,
            args.teacher_mask,
            args.teacher_reliability_scale,
            args.base_preservation_smooth_weight,
            args.base_preservation_projection_weight,
            args.base_preservation_acf_weight,
            args.mu_scale,
            args.teacher_improvement_only,
            args.preserve_teacher_physics,
            args.teacher_oracle_endpoint,
            args.teacher_oracle_margin,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            1,
            None,
            None,
            1,
            args.tissue_aux_weight,
            1.0,
            args.smooth_weight,
            args.projection_weight,
            args.acf_weight,
            args.tissue_bias_weight,
            0.0,
            args.eval_only_loss,
            args.teacher_weight,
            args.teacher_mask,
            args.teacher_reliability_scale,
            args.base_preservation_smooth_weight,
            args.base_preservation_projection_weight,
            args.base_preservation_acf_weight,
            args.mu_scale,
            args.teacher_improvement_only,
            args.preserve_teacher_physics,
            args.teacher_oracle_endpoint,
            args.teacher_oracle_margin,
        )
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "epoch_seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "train": train_metrics,
            "val": val_metrics,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        scheduler.step()
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "val_metrics": val_metrics,
            "best_val": min(best_val, val_metrics["eval_mu_l1"]),
            "best_objective": min(best_objective, val_metrics["loss"]),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["eval_mu_l1"] < best_val:
            best_val = val_metrics["eval_mu_l1"]
            torch.save(checkpoint, output_dir / "best.pt")
        if val_metrics["loss"] < best_objective:
            best_objective = val_metrics["loss"]
            torch.save(checkpoint, output_dir / "best-objective.pt")


if __name__ == "__main__":
    main()
