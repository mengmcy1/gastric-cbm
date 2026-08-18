"""Source-feature-conditioned target-view RGB-D model for Stage 1.9."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .occlusion_hidden import ConvBlock
from .target_view import TargetViewOutput


class SourceFeatureTargetViewNet(nn.Module):
    """Encode the complete source image before target-space projection."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        if base_channels < 8:
            raise ValueError("base_channels must be at least 8")
        self.source1 = ConvBlock(3, base_channels)
        self.source2 = ConvBlock(base_channels, base_channels * 2)
        self.source3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.source_projection = nn.Conv2d(base_channels * 7, base_channels, 1)
        self.global_projection = nn.Sequential(nn.Linear(base_channels * 4, base_channels), nn.SiLU())
        self.enc1 = ConvBlock(11 + 2 * base_channels, base_channels)
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
        source_rgb: torch.Tensor,
        target_to_source_grid: torch.Tensor,
        warped_rgb: torch.Tensor,
        warped_depth_z: torch.Tensor,
        warp_valid: torch.Tensor,
        target_rays_in_source: torch.Tensor,
        target_origin_in_source: torch.Tensor,
        source_depth_scale: torch.Tensor,
    ) -> TargetViewOutput:
        batch, _, height, width = warped_rgb.shape
        if source_rgb.shape != (batch, 3, height, width):
            raise ValueError("source_rgb and warped_rgb must share [B,3,H,W]")
        if target_to_source_grid.shape != (batch, height, width, 2):
            raise ValueError("target_to_source_grid must have shape [B,H,W,2]")
        one = (batch, 1, height, width)
        three = (batch, 3, height, width)
        if warped_depth_z.shape != one or warp_valid.shape != one:
            raise ValueError("warped depth and valid mask must have shape [B,1,H,W]")
        if target_rays_in_source.shape != three or target_origin_in_source.shape != three:
            raise ValueError("target ray and origin conditioning must have shape [B,3,H,W]")
        if source_depth_scale.shape != (batch, 1, 1, 1):
            raise ValueError("source_depth_scale must have shape [B,1,1,1]")
        if not torch.isfinite(target_to_source_grid).all() or not torch.isfinite(source_depth_scale).all():
            raise ValueError("grid and depth scale must be finite")
        if not (source_depth_scale > 0).all():
            raise ValueError("source_depth_scale must be positive")

        source1 = self.source1(source_rgb)
        source2 = self.source2(F.avg_pool2d(source1, 2))
        source3 = self.source3(F.avg_pool2d(source2, 2))
        source_pyramid = self.source_projection(torch.cat((
            source1,
            F.interpolate(source2, size=(height, width), mode="bilinear", align_corners=False),
            F.interpolate(source3, size=(height, width), mode="bilinear", align_corners=False),
        ), dim=1))
        warped_features = F.grid_sample(
            source_pyramid, target_to_source_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ) * warp_valid
        global_context = self.global_projection(F.adaptive_avg_pool2d(source3, 1).flatten(1))
        global_context = global_context[:, :, None, None].expand(-1, -1, height, width)
        normalized_depth = torch.where(
            warp_valid > 0.5,
            torch.log(warped_depth_z.clamp_min(1e-6) / source_depth_scale),
            torch.zeros_like(warped_depth_z),
        )
        value = torch.cat((warped_rgb, normalized_depth, warp_valid, target_rays_in_source,
                           target_origin_in_source, warped_features, global_context), dim=1)
        enc1 = self.enc1(value)
        enc2 = self.enc2(F.avg_pool2d(enc1, 2))
        enc3 = self.enc3(F.avg_pool2d(enc2, 2))
        enc4 = self.enc4(F.avg_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.avg_pool2d(enc4, 2))
        dec4 = self.dec4(torch.cat((F.interpolate(bottleneck, size=enc4.shape[-2:], mode="bilinear", align_corners=False), enc4), dim=1))
        dec3 = self.dec3(torch.cat((F.interpolate(dec4, size=enc3.shape[-2:], mode="bilinear", align_corners=False), enc3), dim=1))
        dec2 = self.dec2(torch.cat((F.interpolate(dec3, size=enc2.shape[-2:], mode="bilinear", align_corners=False), enc2), dim=1))
        features = self.dec1(torch.cat((F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False), enc1), dim=1))
        depth = source_depth_scale * (F.softplus(self.depth_head(features)) + 1e-4)
        occ = self.occlusion_support_head(features)
        outside = self.outside_support_head(features)
        return TargetViewOutput(
            rgb=torch.sigmoid(self.rgb_head(features)), depth_z=depth,
            occlusion_support_logits=occ, occlusion_support_probability=torch.sigmoid(occ),
            outside_support_logits=outside, outside_support_probability=torch.sigmoid(outside),
            geometry_confidence=torch.sigmoid(self.geometry_confidence_head(features)),
            appearance_confidence=torch.sigmoid(self.appearance_confidence_head(features)),
        )
