"""First trainable occlusion-hidden geometry model and positive-only losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


@dataclass(frozen=True)
class HiddenGeometryOutput:
    delta_ratio: torch.Tensor
    hidden_depth_z: torch.Tensor
    support_logits: torch.Tensor
    support_probability: torch.Tensor
    geometry_uncertainty: torch.Tensor
    geometry_confidence: torch.Tensor


class OcclusionHiddenUNet(nn.Module):
    """Small U-Net with independent ratio, support, and uncertainty heads.

    The positive relative increment makes the deployed hidden depth strictly
    farther than the replaceable base-depth prediction while allowing training
    labels to remain invariant to monocular metric-scale error.
    """

    def __init__(self, base_channels: int = 16, minimum_delta_ratio: float = 1e-4) -> None:
        super().__init__()
        if base_channels < 8 or minimum_delta_ratio <= 0:
            raise ValueError("base_channels must be >=8 and minimum_delta_ratio must be positive")
        self.minimum_delta_ratio = float(minimum_delta_ratio)
        self.enc1 = ConvBlock(6, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        self.dec4 = ConvBlock(base_channels * 24, base_channels * 8)
        self.dec3 = ConvBlock(base_channels * 12, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 3, base_channels)
        self.delta_head = nn.Conv2d(base_channels, 1, 1)
        self.support_head = nn.Conv2d(base_channels + 1, 1, 1)
        self.uncertainty_head = nn.Conv2d(base_channels + 1, 1, 1)

    def forward(
        self,
        rgb: torch.Tensor,
        base_depth_z: torch.Tensor,
        camera_rays_xy: torch.Tensor | None = None,
    ) -> HiddenGeometryOutput:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [B,3,H,W]")
        if base_depth_z.shape != (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]):
            raise ValueError("base_depth_z must have shape [B,1,H,W]")
        if not torch.isfinite(base_depth_z).all() or not (base_depth_z > 0).all():
            raise ValueError("base_depth_z must be finite and strictly positive")
        log_depth = torch.log(base_depth_z)
        # Mean/MAD avoids CUDA's nondeterministic indexed median implementation.
        center = log_depth.mean(dim=(2, 3), keepdim=True)
        scale = (log_depth - center).abs().mean(dim=(2, 3), keepdim=True)
        normalized_log_depth = (log_depth - center) / scale.clamp_min(0.1)
        if camera_rays_xy is None:
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, rgb.shape[2], device=rgb.device, dtype=rgb.dtype),
                torch.linspace(-1, 1, rgb.shape[3], device=rgb.device, dtype=rgb.dtype),
                indexing="ij",
            )
            camera_rays_xy = torch.stack((xx, yy), dim=0)[None].expand(rgb.shape[0], -1, -1, -1)
        if camera_rays_xy.shape != (rgb.shape[0], 2, rgb.shape[2], rgb.shape[3]):
            raise ValueError("camera_rays_xy must have shape [B,2,H,W]")
        value = torch.cat((rgb, normalized_log_depth, camera_rays_xy), dim=1)
        enc1 = self.enc1(value)
        enc2 = self.enc2(F.avg_pool2d(enc1, 2))
        enc3 = self.enc3(F.avg_pool2d(enc2, 2))
        enc4 = self.enc4(F.avg_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.avg_pool2d(enc4, 2))
        up4 = F.interpolate(bottleneck, size=enc4.shape[-2:], mode="bilinear", align_corners=False)
        dec4 = self.dec4(torch.cat((up4, enc4), dim=1))
        up3 = F.interpolate(dec4, size=enc3.shape[-2:], mode="bilinear", align_corners=False)
        dec3 = self.dec3(torch.cat((up3, enc3), dim=1))
        up2 = F.interpolate(dec3, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        up1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        features = self.dec1(torch.cat((up1, enc1), dim=1))
        delta_ratio = F.softplus(self.delta_head(features)) + self.minimum_delta_ratio
        geometry_condition = torch.log(delta_ratio).clamp(-8, 8)
        conditioned_features = torch.cat((features, geometry_condition), dim=1)
        support_logits = self.support_head(conditioned_features)
        support_probability = torch.sigmoid(support_logits)
        uncertainty = F.softplus(self.uncertainty_head(conditioned_features)) + 1e-4
        confidence = torch.exp(-torch.sqrt(uncertainty))
        return HiddenGeometryOutput(
            delta_ratio=delta_ratio,
            hidden_depth_z=base_depth_z * (1.0 + delta_ratio),
            support_logits=support_logits,
            support_probability=support_probability,
            geometry_uncertainty=uncertainty,
            geometry_confidence=confidence,
        )


def positive_unlabeled_logistic_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    class_prior: float,
) -> torch.Tensor:
    """Non-negative PU risk; unknown pixels are an unlabeled mixture, not negatives."""
    if not 0 < class_prior < 1:
        raise ValueError("class_prior must be in (0,1)")
    positive = positive_mask.bool()
    if positive.shape != logits.shape or not positive.any():
        raise ValueError("positive_mask must match logits and contain positive evidence")
    positive_logits = logits[positive]
    positive_risk = class_prior * F.softplus(-positive_logits).mean()
    negative_risk = F.softplus(logits).mean() - class_prior * F.softplus(positive_logits).mean()
    return positive_risk + torch.clamp_min(negative_risk, 0.0)


def positive_verified_negative_logistic_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    verified_negative_mask: torch.Tensor,
) -> torch.Tensor:
    """Balanced support loss using only evidence-backed positive/negative pixels."""
    positive = positive_mask.bool()
    negative = verified_negative_mask.bool()
    if positive.shape != logits.shape or negative.shape != logits.shape:
        raise ValueError("support masks must match logits")
    if torch.any(positive & negative):
        raise ValueError("positive and verified-negative masks must be disjoint")
    if not positive.any():
        raise ValueError("support loss requires positive evidence")
    positive_loss = F.softplus(-logits[positive]).mean()
    if not negative.any():
        # No unknown pixel is promoted to a negative merely to fill a batch.
        return positive_loss
    return 0.5 * (positive_loss + F.softplus(logits[negative]).mean())


@dataclass(frozen=True)
class HiddenGeometryLoss:
    total: torch.Tensor
    relative_depth: torch.Tensor
    support_pu: torch.Tensor
    uncertainty_nll: torch.Tensor


def compute_hidden_geometry_loss(
    output: HiddenGeometryOutput,
    source_truth_depth_z_label_only: torch.Tensor,
    hidden_truth_depth_z: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    class_prior: float,
    support_weight: float = 0.2,
    uncertainty_weight: float = 0.05,
) -> HiddenGeometryLoss:
    """Compute scale-invariant positive geometry and statistical PU support loss."""
    positive = positive_mask.bool()
    expected = output.delta_ratio.shape
    if source_truth_depth_z_label_only.shape != expected or hidden_truth_depth_z.shape != expected:
        raise ValueError("truth tensors must match output shape")
    if positive.shape != expected or not positive.any():
        raise ValueError("positive_mask must match output and contain positives")
    source = source_truth_depth_z_label_only[positive]
    hidden = hidden_truth_depth_z[positive]
    if not torch.isfinite(source).all() or not torch.isfinite(hidden).all():
        raise ValueError("positive truth depths must be finite")
    target_ratio = hidden / source - 1.0
    if not (target_ratio > 0).all():
        raise ValueError("hidden positive truth must be strictly behind source truth")
    # Direct log-ratio regression gives small-but-valid occlusion increments the
    # same multiplicative importance as large ones. log1p would underweight the
    # protocol's 1% boundary and can pass MAE while failing relative error.
    predicted_log_ratio = torch.log(output.delta_ratio[positive])
    target_log_ratio = torch.log(target_ratio)
    residual = predicted_log_ratio - target_log_ratio
    relative_depth = F.smooth_l1_loss(predicted_log_ratio, target_log_ratio)
    variance = output.geometry_uncertainty[positive]
    uncertainty_nll = (0.5 * residual.square() / variance + 0.5 * torch.log(variance)).mean()
    support_pu = positive_unlabeled_logistic_loss(output.support_logits, positive, class_prior)
    total = relative_depth + support_weight * support_pu + uncertainty_weight * uncertainty_nll
    return HiddenGeometryLoss(total, relative_depth, support_pu, uncertainty_nll)
