"""Function-preserving refiner initialization across one inserted input channel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import torch

from .refiner import MuResidualRefiner


INPUT_WEIGHT_NAMES = {"stem.body.0.weight", "stem.skip.weight"}
FROZEN_FORWARD_PROBE_SEED = 20260899
FROZEN_FORWARD_ATOL = 1e-6


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize_refiner_full_model_with_inserted_channel(
    model: MuResidualRefiner,
    checkpoint_path: Path,
    inserted_input_channel: int,
) -> dict[str, Any]:
    """Load a full refiner while inserting an exact-zero input-weight column."""
    checkpoint_path = Path(checkpoint_path).resolve()
    config_path = checkpoint_path.parent / "config.json"
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("source checkpoint and adjacent config.json are required")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = checkpoint.get("model")
    if not isinstance(source_state, dict):
        raise ValueError("full-model initialization checkpoint has no model state")
    source_config = json.loads(config_path.read_text(encoding="utf-8"))

    target_state = model.state_dict()
    first_target = target_state["stem.body.0.weight"]
    first_source = source_state.get("stem.body.0.weight")
    if not isinstance(first_source, torch.Tensor):
        raise ValueError("source checkpoint has no stem input weight")
    source_input_channels = int(first_source.shape[1])
    target_input_channels = int(first_target.shape[1])
    inserted_input_channel = int(inserted_input_channel)
    if target_input_channels != source_input_channels + 1:
        raise ValueError(
            "target refiner must have exactly one more input channel than source: "
            f"{target_input_channels} != {source_input_channels} + 1"
        )
    if not 0 <= inserted_input_channel < target_input_channels:
        raise ValueError("inserted input channel is outside the target input range")

    transformed: dict[str, torch.Tensor] = {}
    for name, target_value in target_state.items():
        source_value = source_state.get(name)
        if not isinstance(source_value, torch.Tensor):
            raise ValueError(f"source checkpoint is missing parameter: {name}")
        if name in INPUT_WEIGHT_NAMES:
            expected_shape = list(target_value.shape)
            expected_shape[1] -= 1
            if list(source_value.shape) != expected_shape:
                raise ValueError(
                    f"source input weight shape mismatch for {name}: "
                    f"{tuple(source_value.shape)} != {tuple(expected_shape)}"
                )
            expanded = torch.zeros_like(target_value)
            expanded[:, :inserted_input_channel].copy_(
                source_value[:, :inserted_input_channel]
            )
            expanded[:, inserted_input_channel + 1 :].copy_(
                source_value[:, inserted_input_channel:]
            )
            transformed[name] = expanded
        else:
            if source_value.shape != target_value.shape:
                raise ValueError(
                    f"source parameter shape mismatch for {name}: "
                    f"{tuple(source_value.shape)} != {tuple(target_value.shape)}"
                )
            transformed[name] = source_value.clone()

    result = model.load_state_dict(transformed, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            "unexpected full-model initialization state: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )

    parameter_transform_exact = True
    for name, source_value in source_state.items():
        target_value = model.state_dict()[name]
        if name in INPUT_WEIGHT_NAMES:
            parameter_transform_exact &= bool(
                torch.equal(
                    target_value[:, :inserted_input_channel],
                    source_value[:, :inserted_input_channel],
                )
                and torch.count_nonzero(
                    target_value[:, inserted_input_channel]
                ).item()
                == 0
                and torch.equal(
                    target_value[:, inserted_input_channel + 1 :],
                    source_value[:, inserted_input_channel:],
                )
            )
        else:
            parameter_transform_exact &= bool(torch.equal(target_value, source_value))
    if not parameter_transform_exact:
        raise ValueError("expanded model is not an exact source-parameter transform")

    source_output_channels = int(source_state["output.weight"].shape[0])
    source_model = MuResidualRefiner(
        in_channels=source_input_channels,
        base_channels=int(source_config["base_channels"]),
        correction_scale=float(source_config["correction_scale"]),
        mu_scale=float(source_config["mu_scale"]),
        output_channels=source_output_channels,
    ).eval()
    source_model.load_state_dict(source_state)
    target_was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(FROZEN_FORWARD_PROBE_SEED)
    source_probe = torch.rand(
        (1, source_input_channels, 16, 16, 16),
        generator=generator,
    )
    inserted_probe = torch.rand((1, 1, 16, 16, 16), generator=generator)
    target_probe = torch.cat(
        (
            source_probe[:, :inserted_input_channel],
            inserted_probe,
            source_probe[:, inserted_input_channel:],
        ),
        dim=1,
    )
    with torch.no_grad():
        source_prediction = source_model(source_probe)
        target_prediction = model(target_probe)
    forward_max_abs_error = float(
        (source_prediction - target_prediction).abs().max()
    )
    if target_was_training:
        model.train()
    forward_equivalent = forward_max_abs_error <= FROZEN_FORWARD_ATOL
    if not forward_equivalent:
        raise ValueError(
            "expanded model exceeds the frozen forward tolerance: "
            f"{forward_max_abs_error} > {FROZEN_FORWARD_ATOL}"
        )

    return {
        "schema": "bicmac-refiner-full-model-inserted-channel/v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "source_input_channels": source_input_channels,
        "target_input_channels": target_input_channels,
        "inserted_input_channel": inserted_input_channel,
        "inserted_input_weight_value": 0.0,
        "copied_output_head": True,
        "parameter_transform_exact": parameter_transform_exact,
        "forward_probe_seed": FROZEN_FORWARD_PROBE_SEED,
        "forward_probe_shape": list(source_probe.shape),
        "forward_max_abs_error": forward_max_abs_error,
        "forward_absolute_tolerance": FROZEN_FORWARD_ATOL,
        "forward_equivalent": forward_equivalent,
    }


def configure_inserted_input_channel_only_training(
    model: MuResidualRefiner,
    inserted_input_channel: int,
) -> dict[str, Any]:
    """Freeze a refiner except for one inserted column in each stem weight."""
    inserted_input_channel = int(inserted_input_channel)
    named_parameters = dict(model.named_parameters())
    missing = INPUT_WEIGHT_NAMES.difference(named_parameters)
    if missing:
        raise ValueError(f"model is missing stem input parameters: {sorted(missing)}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trainable_element_count = 0
    trainable_tensor_count = 0
    for name in sorted(INPUT_WEIGHT_NAMES):
        parameter = named_parameters[name]
        if not 0 <= inserted_input_channel < parameter.shape[1]:
            raise ValueError(
                f"inserted input channel {inserted_input_channel} is outside {name}"
            )
        parameter.requires_grad_(True)

        def mask_gradient(
            gradient: torch.Tensor,
            channel: int = inserted_input_channel,
        ) -> torch.Tensor:
            masked = torch.zeros_like(gradient)
            masked[:, channel : channel + 1].copy_(
                gradient[:, channel : channel + 1]
            )
            return masked

        parameter.register_hook(mask_gradient)
        trainable_element_count += int(parameter[:, inserted_input_channel].numel())
        trainable_tensor_count += 1

    return {
        "schema": "bicmac-inserted-input-channel-only-training/v1",
        "inserted_input_channel": inserted_input_channel,
        "optimizer_parameter_names": sorted(INPUT_WEIGHT_NAMES),
        "optimizer_parameter_tensor_count": trainable_tensor_count,
        "effective_trainable_element_count": trainable_element_count,
        "frozen_parameter_tensor_count": sum(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "gradient_masked_to_inserted_channel": True,
        "requires_zero_weight_decay": True,
    }


def audit_inserted_input_channel_only_training(
    target_state: Mapping[str, torch.Tensor],
    source_checkpoint_path: Path,
    inserted_input_channel: int,
) -> dict[str, Any]:
    """Prove that only the two inserted stem columns changed after training."""
    source_checkpoint_path = Path(source_checkpoint_path).resolve()
    checkpoint = torch.load(
        source_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    source_state = checkpoint.get("model")
    if not isinstance(source_state, dict):
        raise ValueError("source checkpoint has no model state")
    if set(target_state) != set(source_state):
        raise ValueError("source and target model state keys differ")

    inserted_input_channel = int(inserted_input_channel)
    frozen_mismatch_names: list[str] = []
    nonfinite_names: list[str] = []
    inserted_nonzero_element_count = 0
    inserted_element_count = 0
    inserted_maximum_absolute_value = 0.0
    for name, target_value in target_state.items():
        source_value = source_state[name]
        target_value = target_value.detach().cpu()
        source_value = source_value.detach().cpu()
        if torch.is_floating_point(target_value) and not torch.isfinite(
            target_value
        ).all():
            nonfinite_names.append(name)
        if name in INPUT_WEIGHT_NAMES:
            if target_value.shape[1] != source_value.shape[1] + 1:
                raise ValueError(f"unexpected inserted-channel shape for {name}")
            if not 0 <= inserted_input_channel < target_value.shape[1]:
                raise ValueError("inserted input channel is outside target state")
            inherited_exact = bool(
                torch.equal(
                    target_value[:, :inserted_input_channel],
                    source_value[:, :inserted_input_channel],
                )
                and torch.equal(
                    target_value[:, inserted_input_channel + 1 :],
                    source_value[:, inserted_input_channel:],
                )
            )
            if not inherited_exact:
                frozen_mismatch_names.append(name)
            inserted = target_value[:, inserted_input_channel]
            inserted_nonzero_element_count += int(torch.count_nonzero(inserted))
            inserted_element_count += int(inserted.numel())
            inserted_maximum_absolute_value = max(
                inserted_maximum_absolute_value,
                float(inserted.abs().max()),
            )
        elif not torch.equal(target_value, source_value):
            frozen_mismatch_names.append(name)

    frozen_parameter_integrity_exact = not frozen_mismatch_names
    inserted_columns_changed = inserted_nonzero_element_count > 0
    return {
        "schema": "bicmac-inserted-input-channel-only-integrity/v1",
        "source_checkpoint": str(source_checkpoint_path),
        "source_checkpoint_sha256": _sha256(source_checkpoint_path),
        "inserted_input_channel": inserted_input_channel,
        "frozen_parameter_integrity_exact": frozen_parameter_integrity_exact,
        "only_inserted_columns_changed": bool(
            frozen_parameter_integrity_exact and inserted_columns_changed
        ),
        "inserted_columns_changed": inserted_columns_changed,
        "inserted_element_count": inserted_element_count,
        "inserted_nonzero_element_count": inserted_nonzero_element_count,
        "inserted_maximum_absolute_value": inserted_maximum_absolute_value,
        "frozen_mismatch_names": frozen_mismatch_names,
        "nonfinite_tensor_count": len(nonfinite_names),
        "nonfinite_names": nonfinite_names,
    }
