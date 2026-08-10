"""Gaussian Decoder -- per-point Sharp-style decoder with NDC output."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from src.model.encoder.depth.infinidepth.sampling_utils import SAMPLE_KIND_TRIANGLE_VERTEX
from src.utils.color_space import sRGB2linearRGB
from src.utils.gaussians import Gaussians3D

@dataclass
class GaussianDecoderCfg:
    delta_factor_xy: float
    delta_factor_z: float
    delta_factor_scale: float
    delta_factor_rotation: float
    delta_factor_color: float
    delta_factor_opacity: float

    scale_min: float
    scale_max: float
    init_opacity: float

    opacity_min: float
    opacity_max: float
    rgb_min: float
    rgb_max: float

    depth_normalization_min: float
    depth_normalization_max: float
    base_scale_multiplier: float


def _inverse_softplus(x: Tensor) -> Tensor:
    return torch.where(x > 20.0, x, torch.log(torch.expm1(x.clamp(min=1e-6))))


def _inverse_sigmoid(x: Tensor) -> Tensor:
    x = x.clamp(min=1e-6, max=1.0 - 1e-6)
    return torch.log(x / (1.0 - x))


def _scale_activation_constants(scale_max: float, scale_min: float) -> tuple[float, float]:
    constant_a = (scale_max - scale_min) / (1.0 - scale_min) / (scale_max - 1.0)
    constant_b = _inverse_sigmoid(
        torch.tensor((1.0 - scale_min) / (scale_max - scale_min))
    ).item()
    return constant_a, constant_b


class GaussianBaseValues(NamedTuple):
    mean_x_ndc: Tensor
    mean_y_ndc: Tensor
    mean_inverse_z_ndc: Tensor
    scales: Tensor
    quaternions: Tensor
    colors: Tensor
    opacities: Tensor


class PointwiseInitializerOutput(NamedTuple):
    gaussian_base_values: GaussianBaseValues
    global_scale: Tensor


def _rescale_depth(
    depth: Float[Tensor, "b v n"],
    depth_min: float,
    depth_max: float,
) -> tuple[Float[Tensor, "b v n"], Float[Tensor, "b v"]]:
    """Rescale sparse metric depth using SHARP's minimum-depth normalization.

    Args:
        depth: Metric depth with shape [B, V, N].
        depth_min: Target minimum depth after normalization.
        depth_max: Maximum normalized depth after scaling.

    Returns:
        A tuple of normalized depths with shape [B, V, N] and per-view depth
        factors with shape [B, V].
    """
    current_depth_min = depth.min(dim=-1).values
    depth_factor = depth_min / (current_depth_min + 1e-6)
    depth = (depth * depth_factor[..., None]).clamp(max=depth_max)
    return depth, depth_factor


class PointwiseInitializer(nn.Module):
    def __init__(self, cfg: GaussianDecoderCfg) -> None:
        super().__init__()
        self.cfg = cfg
        if float(cfg.base_scale_multiplier) <= 0.0:
            raise ValueError("base_scale_multiplier must be positive.")

    def forward(
        self,
        coords_yx_ndc: Float[Tensor, "b v n 2"],
        depths: Float[Tensor, "b v n 1"],
        rgb: Float[Tensor, "b v n 3"],
        intrinsics: Float[Tensor, "b v 3 3"],
        sample_kind: Tensor,
        sample_responsibility_area_metric: Float[Tensor, "b v n"],
        image_shape: tuple[int, int],
    ) -> PointwiseInitializerOutput:
        """Build sparse Gaussian base values in normalized depth space.

        Args:
            coords_yx_ndc: Sparse query coordinates in YX NDC order with shape [B, V, N, 2].
            depths: Metric sparse depths with shape [B, V, N, 1].
            rgb: Sampled RGB values with shape [B, V, N, 3].
            intrinsics: Pixel-space intrinsics with shape [B, V, 3, 3].
            sample_kind: Sampling source ids with shape [B, V, N].
            sample_responsibility_area_metric: Per-sample metric area represented by
                each sample with shape [B, V, N].
            image_shape: Image shape as (height, width).

        Returns:
            Base Gaussian values and a per-view global scale with shape [B, V].
        """
        depth = depths.squeeze(-1)
        depth, depth_factor = _rescale_depth(
            depth,
            depth_min=self.cfg.depth_normalization_min,
            depth_max=self.cfg.depth_normalization_max,
        )
        global_scale = 1.0 / depth_factor

        y_ndc = coords_yx_ndc[..., 0]
        x_ndc = coords_yx_ndc[..., 1]
        inv_z = 1.0 / depth.clamp(min=1e-3)
        image_height, image_width = image_shape
        pixel_scale_factor = 2.0 / math.sqrt(float(image_height * image_width))
        base_scale_vertex = depth * pixel_scale_factor

        fx = intrinsics[..., 0, 0]
        fy = intrinsics[..., 1, 1]
        view_metric_to_internal = torch.sqrt(
            (2.0 * fx / float(image_width)) * (2.0 * fy / float(image_height))
        )
        base_scale_extra = (
            torch.sqrt(sample_responsibility_area_metric.clamp(min=1e-12))
            * depth_factor[..., None]
            * view_metric_to_internal[..., None]
            * float(self.cfg.base_scale_multiplier)
        )
        base_scale = torch.where(
            sample_kind == SAMPLE_KIND_TRIANGLE_VERTEX,
            base_scale_vertex,
            base_scale_extra,
        )

        b, v, n = depth.shape
        quaternions = torch.zeros(b, v, n, 4, device=depth.device, dtype=depth.dtype)
        quaternions[..., 0] = 1.0

        colors = rgb.clamp(self.cfg.rgb_min, self.cfg.rgb_max)
        opacities = torch.full(
            (b, v, n),
            self.cfg.init_opacity,
            device=depth.device,
            dtype=depth.dtype,
        )

        return PointwiseInitializerOutput(
            gaussian_base_values=GaussianBaseValues(
                mean_x_ndc=x_ndc,
                mean_y_ndc=y_ndc,
                mean_inverse_z_ndc=inv_z,
                scales=base_scale,
                quaternions=quaternions,
                colors=colors,
                opacities=opacities,
            ),
            global_scale=global_scale,
        )


class PointwiseGaussianComposer(nn.Module):
    def __init__(self, cfg: GaussianDecoderCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self._scale_const_a, self._scale_const_b = _scale_activation_constants(
            cfg.scale_max,
            cfg.scale_min,
        )

    def _mean_activation(
        self,
        base: GaussianBaseValues,
        delta: Float[Tensor, "b v n 14"],
    ) -> Float[Tensor, "b v n 3"]:
        xx = base.mean_x_ndc + self.cfg.delta_factor_xy * delta[..., 0]
        yy = base.mean_y_ndc + self.cfg.delta_factor_xy * delta[..., 1]
        inverse_zz = self._predicted_inverse_depth(base, delta[..., 2])
        zz = 1.0 / (inverse_zz + 1e-3)
        return torch.stack([zz * xx, zz * yy, zz], dim=-1)

    def _predicted_inverse_depth(
        self,
        base: GaussianBaseValues,
        delta_z: Float[Tensor, "b v n"],
    ) -> Float[Tensor, "b v n"]:
        """Predict inverse depth before the rendering-specific epsilon is applied."""
        return F.softplus(
            _inverse_softplus(base.mean_inverse_z_ndc) + self.cfg.delta_factor_z * delta_z
        )

    def _scale_activation(
        self,
        base_scales: Float[Tensor, "b v n"],
        delta_scale: Float[Tensor, "b v n 3"],
    ) -> Float[Tensor, "b v n 3"]:
        factor = (self.cfg.scale_max - self.cfg.scale_min) * torch.sigmoid(
            self._scale_const_a * self.cfg.delta_factor_scale * delta_scale + self._scale_const_b
        ) + self.cfg.scale_min
        return base_scales.unsqueeze(-1) * factor

    def _quaternion_activation(
        self,
        base_quaternions: Float[Tensor, "b v n 4"],
        delta_rot: Float[Tensor, "b v n 4"],
    ) -> Float[Tensor, "b v n 4"]:
        # Will be normalized at unproject_gaussians(rotation_matrices_from_quaternions)
        return base_quaternions + self.cfg.delta_factor_rotation * delta_rot

    def _color_activation(
        self,
        base_colors: Float[Tensor, "b v n 3"],
        delta_color: Float[Tensor, "b v n 3"],
    ) -> Float[Tensor, "b v n 3"]:
        base_colors = base_colors.clamp(self.cfg.rgb_min, self.cfg.rgb_max)
        colors = torch.sigmoid(
            _inverse_sigmoid(base_colors)
            + self.cfg.delta_factor_color * delta_color
        )
        return sRGB2linearRGB(colors)

    def _opacity_activation(
        self,
        base_opacities: Float[Tensor, "b v n"],
        delta_opacity: Float[Tensor, "b v n"],
    ) -> Float[Tensor, "b v n"]:
        base_opacities = base_opacities.clamp(self.cfg.opacity_min, self.cfg.opacity_max)
        return torch.sigmoid(
            _inverse_sigmoid(base_opacities)
            + self.cfg.delta_factor_opacity * delta_opacity
        )

    def forward(
        self,
        base: GaussianBaseValues,
        delta: Float[Tensor, "b v n 14"],
        global_scale: Float[Tensor, "b v"],
    ) -> tuple[
        Float[Tensor, "b v n 3"],
        Float[Tensor, "b v n 3"],
        Float[Tensor, "b v n 4"],
        Float[Tensor, "b v n 3"],
        Float[Tensor, "b v n"],
    ]:
        mean_vectors = self._mean_activation(base, delta)
        predicted_depth_for_scale = 1.0 / self._predicted_inverse_depth(
            base,
            delta[..., 2],
        ).clamp(min=1e-6)
        base_scales = base.scales * base.mean_inverse_z_ndc * predicted_depth_for_scale

        singular_values = self._scale_activation(base_scales, delta[..., 3:6])
        quaternions = self._quaternion_activation(base.quaternions, delta[..., 6:10])
        colors = self._color_activation(base.colors, delta[..., 10:13])
        opacities = self._opacity_activation(base.opacities, delta[..., 13])
        mean_vectors = mean_vectors * global_scale[..., None, None]
        singular_values = singular_values * global_scale[..., None, None]
        return mean_vectors, singular_values, quaternions, colors, opacities


class GaussianDecoder(nn.Module):
    def __init__(self, cfg: GaussianDecoderCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.initializer = PointwiseInitializer(cfg)
        self.composer = PointwiseGaussianComposer(cfg)

    def forward(
        self,
        delta: Float[Tensor, "b v n 14"],
        coords_yx_ndc: Float[Tensor, "b v n 2"],
        depths: Float[Tensor, "b v n 1"],
        rgb: Float[Tensor, "b v n 3"],
        intrinsics: Float[Tensor, "b v 3 3"],
        sample_kind: Tensor,
        sample_responsibility_area_metric: Float[Tensor, "b v n"],
        image_shape: tuple[int, int],
    ) -> Gaussians3D:
        initializer_output = self.initializer(
            coords_yx_ndc,
            depths,
            rgb,
            intrinsics,
            sample_kind,
            sample_responsibility_area_metric,
            image_shape,
        )
        mean_vectors, singular_values, quaternions, colors, opacities = self.composer(
            initializer_output.gaussian_base_values,
            delta,
            initializer_output.global_scale,
        )

        return Gaussians3D(
            mean_vectors=rearrange(mean_vectors, "b v n c -> b (v n) c"),
            singular_values=rearrange(singular_values, "b v n c -> b (v n) c"),
            quaternions=rearrange(quaternions, "b v n c -> b (v n) c"),
            colors=rearrange(colors, "b v n c -> b (v n) c"),
            opacities=rearrange(opacities, "b v n -> b (v n)"),
        )
