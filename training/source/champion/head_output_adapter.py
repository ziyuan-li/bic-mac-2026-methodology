"""A tiny axial head adapter over a frozen projected-topogram model."""

from __future__ import annotations

import torch
from torch.nn import functional

from .model import ProjectionConditionedMuUNet


def normalized_axial_coordinate(index: int, volume_depth: int) -> float:
    if volume_depth <= 1:
        raise ValueError("volume depth must exceed one")
    if not 0 <= index < volume_depth:
        raise ValueError("axial index must lie inside the volume")
    return 2.0 * float(index) / float(volume_depth - 1) - 1.0


def smooth_head_gate(
    axial_coordinate: torch.Tensor,
    transition_start: float,
    transition_end: float,
) -> torch.Tensor:
    """Return a smooth body-to-head gate while preserving exact body zeros."""
    if transition_end <= transition_start:
        raise ValueError("head transition end must exceed its start")
    progress = (axial_coordinate - transition_start) / (
        transition_end - transition_start
    )
    progress = torch.clamp(progress, 0.0, 1.0)
    return progress.square() * (3.0 - 2.0 * progress)


class HeadOutputAdapterProjectionConditionedMuUNet(
    ProjectionConditionedMuUNet
):
    """Keep the inherited model exact outside a gated 1x1x1 head correction."""

    def __init__(
        self,
        base_channels: int = 24,
        output_channels: int = 5,
        *,
        transition_start_slice: int = 370,
        transition_end_slice: int = 390,
        volume_depth: int = 531,
        correction_scale: float = 0.03,
    ) -> None:
        if not 0 <= transition_start_slice < transition_end_slice < volume_depth:
            raise ValueError(
                "head transition must satisfy 0 <= start < end < volume depth"
            )
        if correction_scale <= 0.0:
            raise ValueError("head adapter correction scale must be positive")
        # The inherited trunk intentionally remains a three-channel E13 model.
        super().__init__(
            base_channels=base_channels,
            output_channels=output_channels,
            coordinate_channels=0,
        )
        self.head_adapter = torch.nn.Conv3d(
            base_channels,
            1,
            kernel_size=1,
        )
        torch.nn.init.zeros_(self.head_adapter.weight)
        torch.nn.init.zeros_(self.head_adapter.bias)
        self.correction_scale = float(correction_scale)
        self.transition_start_slice = int(transition_start_slice)
        self.transition_end_slice = int(transition_end_slice)
        self.volume_depth = int(volume_depth)
        self.head_transition_start_coordinate = normalized_axial_coordinate(
            transition_start_slice,
            volume_depth,
        )
        self.head_transition_end_coordinate = normalized_axial_coordinate(
            transition_end_slice,
            volume_depth,
        )

    def _adapter_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.head_adapter(features)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 5 or values.shape[1] != 4:
            raise ValueError(
                "head output adapter expects NAC pair, topogram, and global z"
            )
        axial_coordinate = values[:, 3:4]
        inherited = values[:, :3]

        projection = self.projection_stem(inherited[:, 2:3, :, 0, :])
        trunk = inherited[:, :2]
        skip0 = self._condition(
            self.stem(trunk),
            self.projection_adapters[0](projection),
        )
        projection = self.projection_down[0](projection)
        skip1 = self._condition(
            self.down1(skip0),
            self.projection_adapters[1](projection),
        )
        projection = self.projection_down[1](projection)
        skip2 = self._condition(
            self.down2(skip1),
            self.projection_adapters[2](projection),
        )
        projection = self.projection_down[2](projection)
        skip3 = self._condition(
            self.down3(skip2),
            self.projection_adapters[3](projection),
        )
        projection = self.projection_down[3](projection)
        bottleneck = self._condition(
            self.down4(skip3),
            self.projection_adapters[4](projection),
        )
        features = self.up3(bottleneck, skip3)
        features = self.up2(features, skip2)
        features = self.up1(features, skip1)
        features = self.up0(features, skip0)

        inherited_output = self.output(features)
        gate = smooth_head_gate(
            axial_coordinate,
            self.head_transition_start_coordinate,
            self.head_transition_end_coordinate,
        ).to(dtype=inherited_output.dtype)
        correction = self.correction_scale * torch.tanh(
            self._adapter_logits(features)
        )
        inherited_mu = inherited_output[:, :1]
        adapted_mu = torch.where(
            gate > 0,
            inherited_mu + gate * correction,
            inherited_mu,
        )
        if inherited_output.shape[1] == 1:
            return adapted_mu
        return torch.cat((adapted_mu, inherited_output[:, 1:]), dim=1)


class LocalContextHeadOutputAdapterProjectionConditionedMuUNet(
    HeadOutputAdapterProjectionConditionedMuUNet
):
    """Add a nonlinear local-context head while keeping the inherited trunk frozen."""

    def __init__(
        self,
        base_channels: int = 24,
        output_channels: int = 5,
        *,
        hidden_channels: int = 8,
        transition_start_slice: int = 370,
        transition_end_slice: int = 390,
        volume_depth: int = 531,
        correction_scale: float = 0.03,
    ) -> None:
        if hidden_channels <= 0:
            raise ValueError("local-context hidden channels must be positive")
        super().__init__(
            base_channels=base_channels,
            output_channels=output_channels,
            transition_start_slice=transition_start_slice,
            transition_end_slice=transition_end_slice,
            volume_depth=volume_depth,
            correction_scale=correction_scale,
        )
        self.head_adapter = torch.nn.Sequential(
            torch.nn.Conv3d(2 * base_channels, hidden_channels, kernel_size=1),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv3d(hidden_channels, 1, kernel_size=1),
        )
        torch.nn.init.zeros_(self.head_adapter[-1].weight)
        torch.nn.init.zeros_(self.head_adapter[-1].bias)
        self.head_adapter_hidden_channels = int(hidden_channels)

    def _adapter_logits(self, features: torch.Tensor) -> torch.Tensor:
        local_mean = functional.avg_pool3d(
            features,
            kernel_size=(3, 3, 1),
            stride=1,
            padding=(1, 1, 0),
            count_include_pad=False,
        )
        return self.head_adapter(torch.cat((features, local_mean), dim=1))


def freeze_for_head_output_adapter(model: torch.nn.Module) -> int:
    """Freeze every parameter except the small output adapter."""
    trainable = 0
    for name, parameter in model.named_parameters():
        enabled = name.startswith("head_adapter.")
        parameter.requires_grad_(enabled)
        if enabled:
            trainable += parameter.numel()
    if trainable <= 0:
        raise ValueError("model has no head output adapter parameters")
    return trainable
