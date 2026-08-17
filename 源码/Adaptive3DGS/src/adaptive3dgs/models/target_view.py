"""First explicit target-view-conditioned RGB-D model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .occlusion_hidden import ConvBlock


@dataclass(frozen=True)
class TargetViewOutput:
    rgb: torch.Tensor
    depth_z: torch.Tensor
    occlusion_support_logits: torch.Tensor
    occlusion_support_probability: torch.Tensor
    outside_support_logits: torch.Tensor
    outside_support_probability: torch.Tensor
    geometry_confidence: torch.Tensor
    appearance_confidence: torch.Tensor


class TargetViewUNet(nn.Module):
    """Target-space U-Net conditioned by geometric warp and explicit target camera rays."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        if base_channels < 8:
            raise ValueError("base_channels must be at least 8")
        # warped RGB(3), normalized warped depth(1), coverage(1), ray in source(3), origin in source(3)
        self.enc1 = ConvBlock(11, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        self.dec4 = ConvBlock(base_channels * 24, base_channels * 8)
        self.dec3 = ConvBlock(base_channels * 12, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 3, base_channels)
        self.rgb_head = nn.Conv2d(base_channels, 3, 1)
        self.depth_head = nn.Conv2d(base_channels, 1, 1)
        self.occlusion_support_head = nn.Conv2d(base_channels, 1, 1)
        self.outside_support_head = nn.Conv2d(base_channels, 1, 1)
        self.geometry_confidence_head = nn.Conv2d(base_channels, 1, 1)
        self.appearance_confidence_head = nn.Conv2d(base_channels, 1, 1)

    def forward(
        self,
        warped_rgb: torch.Tensor,
        warped_depth_z: torch.Tensor,
        warp_valid: torch.Tensor,
        target_rays_in_source: torch.Tensor,
        target_origin_in_source: torch.Tensor,
        source_depth_scale: torch.Tensor,
    ) -> TargetViewOutput:
        batch, _, height, width = warped_rgb.shape
        expected_one = (batch, 1, height, width)
        expected_three = (batch, 3, height, width)
        if warped_rgb.shape != expected_three:
            raise ValueError("warped_rgb must have shape [B,3,H,W]")
        if warped_depth_z.shape != expected_one or warp_valid.shape != expected_one:
            raise ValueError("warped depth and valid mask must have shape [B,1,H,W]")
        if target_rays_in_source.shape != expected_three or target_origin_in_source.shape != expected_three:
            raise ValueError("target ray and origin conditioning must have shape [B,3,H,W]")
        if source_depth_scale.shape != (batch, 1, 1, 1):
            raise ValueError("source_depth_scale must have shape [B,1,1,1]")
        if not torch.isfinite(source_depth_scale).all() or not (source_depth_scale > 0).all():
            raise ValueError("source_depth_scale must be finite and positive")
        normalized_depth = torch.where(
            warp_valid > 0.5,
            torch.log(warped_depth_z.clamp_min(1e-6) / source_depth_scale),
            torch.zeros_like(warped_depth_z),
        )
        value = torch.cat(
            (warped_rgb, normalized_depth, warp_valid, target_rays_in_source, target_origin_in_source), dim=1
        )
        enc1 = self.enc1(value)
        enc2 = self.enc2(F.avg_pool2d(enc1, 2))
        enc3 = self.enc3(F.avg_pool2d(enc2, 2))
        enc4 = self.enc4(F.avg_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.avg_pool2d(enc4, 2))
        dec4 = self.dec4(torch.cat((F.interpolate(bottleneck, size=enc4.shape[-2:], mode="bilinear", align_corners=False), enc4), dim=1))
        dec3 = self.dec3(torch.cat((F.interpolate(dec4, size=enc3.shape[-2:], mode="bilinear", align_corners=False), enc3), dim=1))
        dec2 = self.dec2(torch.cat((F.interpolate(dec3, size=enc2.shape[-2:], mode="bilinear", align_corners=False), enc2), dim=1))
        features = self.dec1(torch.cat((F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False), enc1), dim=1))
        depth_z = source_depth_scale * (F.softplus(self.depth_head(features)) + 1e-4)
        occlusion_logits = self.occlusion_support_head(features)
        outside_logits = self.outside_support_head(features)
        return TargetViewOutput(
            rgb=torch.sigmoid(self.rgb_head(features)),
            depth_z=depth_z,
            occlusion_support_logits=occlusion_logits,
            occlusion_support_probability=torch.sigmoid(occlusion_logits),
            outside_support_logits=outside_logits,
            outside_support_probability=torch.sigmoid(outside_logits),
            geometry_confidence=torch.sigmoid(self.geometry_confidence_head(features)),
            appearance_confidence=torch.sigmoid(self.appearance_confidence_head(features)),
        )


@dataclass(frozen=True)
class TargetViewLoss:
    total: torch.Tensor
    rgb: torch.Tensor
    log_depth: torch.Tensor
    occlusion_support: torch.Tensor
    outside_support: torch.Tensor
    geometry_confidence: torch.Tensor
    appearance_confidence: torch.Tensor


def _balanced_region_mean(value: torch.Tensor, masks: tuple[torch.Tensor, ...]) -> torch.Tensor:
    terms = [value[mask].mean() for mask in masks if mask.any()]
    if not terms:
        raise ValueError("at least one supervised region must be nonempty")
    return torch.stack(terms).mean()


def _balanced_binary_loss(logits: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    if torch.any(positive & negative) or not positive.any() or not negative.any():
        raise ValueError("support supervision requires disjoint nonempty positive and negative masks")
    return 0.5 * (F.softplus(-logits[positive]).mean() + F.softplus(logits[negative]).mean())


def compute_target_view_loss(
    output: TargetViewOutput,
    target_rgb: torch.Tensor,
    target_depth_z: torch.Tensor,
    observed_mask: torch.Tensor,
    occlusion_mask: torch.Tensor,
    outside_mask: torch.Tensor,
    *,
    depth_weight: float = 1.0,
    support_weight: float = 0.5,
    confidence_weight: float = 0.05,
) -> TargetViewLoss:
    masks = tuple(mask.bool() for mask in (observed_mask, occlusion_mask, outside_mask))
    if any(mask.shape != output.depth_z.shape for mask in masks):
        raise ValueError("region masks must match output depth")
    if any(torch.any(first & second) for index, first in enumerate(masks) for second in masks[index + 1 :]):
        raise ValueError("region masks must be pairwise disjoint")
    classified = masks[0] | masks[1] | masks[2]
    valid = classified & torch.isfinite(target_depth_z) & (target_depth_z > 0)
    regions = tuple(mask & valid for mask in masks)
    if target_rgb.shape != output.rgb.shape or target_depth_z.shape != output.depth_z.shape:
        raise ValueError("target RGB-D must match model output")
    rgb_error = torch.abs(output.rgb - target_rgb).mean(dim=1, keepdim=True)
    depth_error = torch.abs(torch.log(output.depth_z.clamp_min(1e-6)) - torch.log(target_depth_z.clamp_min(1e-6)))
    rgb_loss = _balanced_region_mean(rgb_error, regions)
    depth_loss = _balanced_region_mean(depth_error, regions)
    occlusion_support = _balanced_binary_loss(output.occlusion_support_logits, regions[1], regions[0] | regions[2])
    outside_support = _balanced_binary_loss(output.outside_support_logits, regions[2], regions[0] | regions[1])
    geometry_correct = (depth_error.detach() < torch.log(torch.tensor(1.10, device=depth_error.device))).float()
    appearance_correct = (rgb_error.detach() < (20.0 / 255.0)).float()
    geometry_confidence = F.binary_cross_entropy(output.geometry_confidence[valid], geometry_correct[valid])
    appearance_confidence = F.binary_cross_entropy(output.appearance_confidence[valid], appearance_correct[valid])
    total = (
        rgb_loss
        + depth_weight * depth_loss
        + support_weight * (occlusion_support + outside_support)
        + confidence_weight * (geometry_confidence + appearance_confidence)
    )
    return TargetViewLoss(
        total,
        rgb_loss,
        depth_loss,
        occlusion_support,
        outside_support,
        geometry_confidence,
        appearance_confidence,
    )
