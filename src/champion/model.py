"""Efficient residual 3D U-Net for direct attenuation-map regression."""

from __future__ import annotations

import torch
from torch import nn


def group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(values) + self.skip(values))


class DownStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.stage = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            ResidualBlock(out_channels, out_channels),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.stage(values)


class UpStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        values = self.up(values)
        if values.shape[2:] != skip.shape[2:]:
            raise ValueError(f"decoder shape {values.shape[2:]} does not match skip shape {skip.shape[2:]}")
        return self.block(torch.cat((values, skip), dim=1))


class DirectMuUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 24,
        output_channels: int = 1,
    ) -> None:
        super().__init__()
        if output_channels not in (1, 5):
            raise ValueError("output_channels must be 1 (mu) or 5 (mu + four tissue logits)")
        channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 12)
        self.stem = ResidualBlock(in_channels, channels[0])
        self.down1 = DownStage(channels[0], channels[1])
        self.down2 = DownStage(channels[1], channels[2])
        self.down3 = DownStage(channels[2], channels[3])
        self.down4 = DownStage(channels[3], channels[4])
        self.up3 = UpStage(channels[4], channels[3], channels[3])
        self.up2 = UpStage(channels[3], channels[2], channels[2])
        self.up1 = UpStage(channels[2], channels[1], channels[1])
        self.up0 = UpStage(channels[1], channels[0], channels[0])
        self.output = nn.Conv3d(channels[0], output_channels, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[0] = 0.095

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        skip0 = self.stem(values)
        skip1 = self.down1(skip0)
        skip2 = self.down2(skip1)
        skip3 = self.down3(skip2)
        bottleneck = self.down4(skip3)
        values = self.up3(bottleneck, skip3)
        values = self.up2(values, skip2)
        values = self.up1(values, skip1)
        values = self.up0(values, skip0)
        return self.output(values)


class ProjectionConditionedMuUNet(DirectMuUNet):
    """Direct mu U-Net with zero-initialized multi-scale 2D topogram conditioning."""

    def __init__(
        self,
        base_channels: int = 24,
        output_channels: int = 5,
        coordinate_channels: int = 0,
    ) -> None:
        if coordinate_channels not in (0, 1, 3):
            raise ValueError("coordinate_channels must be zero, one, or three")
        super().__init__(
            in_channels=2 + coordinate_channels,
            base_channels=base_channels,
            output_channels=output_channels,
        )
        self.coordinate_channels = coordinate_channels
        channels = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 12,
        )
        projection_channels = (
            max(8, base_channels // 2),
            base_channels,
            base_channels * 2,
            base_channels * 3,
            base_channels * 4,
        )
        self.projection_stem = nn.Sequential(
            nn.Conv2d(1, projection_channels[0], 3, padding=1, bias=False),
            nn.GroupNorm(group_count(projection_channels[0]), projection_channels[0]),
            nn.SiLU(inplace=True),
        )
        self.projection_down = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(group_count(out_channels), out_channels),
                nn.SiLU(inplace=True),
            )
            for in_channels, out_channels in zip(projection_channels, projection_channels[1:])
        )
        self.projection_adapters = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, 1)
            for in_channels, out_channels in zip(projection_channels, channels)
        )
        for adapter in self.projection_adapters:
            nn.init.zeros_(adapter.weight)
            nn.init.zeros_(adapter.bias)

    @staticmethod
    def _condition(values: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        return values + projection.unsqueeze(3)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        expected_channels = 3 + self.coordinate_channels
        if values.shape[1] != expected_channels:
            raise ValueError(
                f"expected NAC pair, topogram, and {self.coordinate_channels} coordinate "
                f"channels; got {values.shape[1]} channels"
            )
        projection = self.projection_stem(values[:, 2:3, :, 0, :])
        trunk = (
            torch.cat((values[:, :2], values[:, 3:]), dim=1)
            if self.coordinate_channels
            else values[:, :2]
        )
        skip0 = self._condition(self.stem(trunk), self.projection_adapters[0](projection))
        projection = self.projection_down[0](projection)
        skip1 = self._condition(self.down1(skip0), self.projection_adapters[1](projection))
        projection = self.projection_down[1](projection)
        skip2 = self._condition(self.down2(skip1), self.projection_adapters[2](projection))
        projection = self.projection_down[2](projection)
        skip3 = self._condition(self.down3(skip2), self.projection_adapters[3](projection))
        projection = self.projection_down[3](projection)
        bottleneck = self._condition(self.down4(skip3), self.projection_adapters[4](projection))
        values = self.up3(bottleneck, skip3)
        values = self.up2(values, skip2)
        values = self.up1(values, skip1)
        values = self.up0(values, skip0)
        return self.output(values)


class MRIProjectionConditionedMuUNet(ProjectionConditionedMuUNet):
    """Projection-conditioned U-Net with coarse multi-scale Dixon MRI context."""

    def __init__(
        self,
        base_channels: int = 24,
        output_channels: int = 5,
        coordinate_channels: int = 0,
    ) -> None:
        super().__init__(
            base_channels=base_channels,
            output_channels=output_channels,
            coordinate_channels=coordinate_channels,
        )
        channels = (
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 12,
        )
        mri_channels = (
            max(8, base_channels // 2),
            base_channels,
            base_channels * 2,
            base_channels * 3,
        )
        self.mri_stem = nn.Sequential(
            nn.Conv3d(2, mri_channels[0], 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(group_count(mri_channels[0]), mri_channels[0]),
            nn.SiLU(inplace=True),
        )
        self.mri_down = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(group_count(out_channels), out_channels),
                nn.SiLU(inplace=True),
            )
            for in_channels, out_channels in zip(mri_channels, mri_channels[1:])
        )
        self.mri_adapters = nn.ModuleList(
            nn.Conv3d(in_channels, out_channels, 1)
            for in_channels, out_channels in zip(mri_channels, channels)
        )
        for adapter in self.mri_adapters:
            nn.init.zeros_(adapter.weight)
            nn.init.zeros_(adapter.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        expected_channels = 5 + self.coordinate_channels
        if values.shape[1] != expected_channels:
            raise ValueError(
                f"expected NAC pair, topogram, MRI pair, and {self.coordinate_channels} "
                f"coordinate channels; got {values.shape[1]} channels"
            )

        projection = self.projection_stem(values[:, 2:3, :, 0, :])
        mri = self.mri_stem(values[:, 3:5])
        trunk = (
            torch.cat((values[:, :2], values[:, 5:]), dim=1)
            if self.coordinate_channels
            else values[:, :2]
        )
        skip0 = self._condition(self.stem(trunk), self.projection_adapters[0](projection))

        projection = self.projection_down[0](projection)
        skip1 = (
            self._condition(self.down1(skip0), self.projection_adapters[1](projection))
            + self.mri_adapters[0](mri)
        )
        projection = self.projection_down[1](projection)
        mri = self.mri_down[0](mri)
        skip2 = (
            self._condition(self.down2(skip1), self.projection_adapters[2](projection))
            + self.mri_adapters[1](mri)
        )
        projection = self.projection_down[2](projection)
        mri = self.mri_down[1](mri)
        skip3 = (
            self._condition(self.down3(skip2), self.projection_adapters[3](projection))
            + self.mri_adapters[2](mri)
        )
        projection = self.projection_down[3](projection)
        mri = self.mri_down[2](mri)
        bottleneck = (
            self._condition(self.down4(skip3), self.projection_adapters[4](projection))
            + self.mri_adapters[3](mri)
        )
        values = self.up3(bottleneck, skip3)
        values = self.up2(values, skip2)
        values = self.up1(values, skip1)
        values = self.up0(values, skip0)
        return self.output(values)


class FullResolutionMRIProjectionConditionedMuUNet(MRIProjectionConditionedMuUNet):
    """MRI-conditioned U-Net with a zero-initialized full-resolution adapter."""

    def __init__(
        self,
        base_channels: int = 24,
        output_channels: int = 5,
        coordinate_channels: int = 0,
    ) -> None:
        super().__init__(
            base_channels=base_channels,
            output_channels=output_channels,
            coordinate_channels=coordinate_channels,
        )
        adapter_channels = max(4, base_channels // 4)
        self.mri_fullres_stem = nn.Sequential(
            nn.Conv3d(2, adapter_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(adapter_channels), adapter_channels),
            nn.SiLU(inplace=True),
        )
        self.mri_fullres_adapter = nn.Conv3d(
            adapter_channels,
            base_channels,
            kernel_size=1,
            bias=False,
        )
        nn.init.zeros_(self.mri_fullres_adapter.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        expected_channels = 5 + self.coordinate_channels
        if values.shape[1] != expected_channels:
            raise ValueError(
                f"expected NAC pair, topogram, MRI pair, and {self.coordinate_channels} "
                f"coordinate channels; got {values.shape[1]} channels"
            )

        projection = self.projection_stem(values[:, 2:3, :, 0, :])
        mri_values = values[:, 3:5]
        mri = self.mri_stem(mri_values)
        fullres_mri = self.mri_fullres_adapter(self.mri_fullres_stem(mri_values))
        trunk = (
            torch.cat((values[:, :2], values[:, 5:]), dim=1)
            if self.coordinate_channels
            else values[:, :2]
        )
        skip0 = (
            self._condition(self.stem(trunk), self.projection_adapters[0](projection))
            + fullres_mri
        )

        projection = self.projection_down[0](projection)
        skip1 = (
            self._condition(self.down1(skip0), self.projection_adapters[1](projection))
            + self.mri_adapters[0](mri)
        )
        projection = self.projection_down[1](projection)
        mri = self.mri_down[0](mri)
        skip2 = (
            self._condition(self.down2(skip1), self.projection_adapters[2](projection))
            + self.mri_adapters[1](mri)
        )
        projection = self.projection_down[2](projection)
        mri = self.mri_down[1](mri)
        skip3 = (
            self._condition(self.down3(skip2), self.projection_adapters[3](projection))
            + self.mri_adapters[2](mri)
        )
        projection = self.projection_down[3](projection)
        mri = self.mri_down[2](mri)
        bottleneck = (
            self._condition(self.down4(skip3), self.projection_adapters[4](projection))
            + self.mri_adapters[3](mri)
        )
        values = self.up3(bottleneck, skip3)
        values = self.up2(values, skip2)
        values = self.up1(values, skip1)
        values = self.up0(values, skip0)
        return self.output(values)
