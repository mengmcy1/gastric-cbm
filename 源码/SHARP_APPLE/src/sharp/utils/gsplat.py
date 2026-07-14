"""包含基于 gsplat 的 3D Gaussian renderer。

=对应论文 Gaussian renderer，但属于公开推理实现:=

论文中训练使用 in-house differentiable renderer；公开代码为了方便复现，
使用开源 gsplat 的 rasterization 接口。输入是 Gaussians3D、相机外参、
相机内参和输出分辨率，输出 RGB 图、深度图和 alpha 图。

这部分只负责"已有高斯 → 图像"，不参与从单张 RGB 预测高斯的网络前向。

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import gsplat
import torch
from torch import nn

from sharp.utils import color_space as cs_utils
from sharp.utils import io, vis
from sharp.utils.gaussians import BackgroundColor, Gaussians3D


class RenderingOutputs(NamedTuple):
    """3D 高斯渲染器输出。"""

    color: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor


def write_renderings(rendering: RenderingOutputs, output_folder: Path, filename: str):
    """将渲染得到的 color/depth/alpha 保存为图片。"""
    batch_size = len(rendering.color)
    if batch_size != 1:
        raise RuntimeError("We only support saving rendering of batch size = 1")

    def _save_image_tensor(tensor: torch.Tensor, suffix: str):
        np_array = tensor.permute(1, 2, 0).numpy()
        io.save_image(np_array, (output_folder / filename).with_suffix(suffix))

    color = (rendering.color[0].cpu() * 255.0).to(dtype=torch.uint8)
    colorized_depth = vis.colorize_depth(rendering.depth[0], val_max=100.0)
    colorized_alpha = vis.colorize_alpha(rendering.alpha[0])

    _save_image_tensor(color, ".color.png")
    _save_image_tensor(colorized_depth, ".depth.png")
    _save_image_tensor(colorized_alpha, ".alpha.png")


class GSplatRenderer(nn.Module):
    """使用 gsplat 将 3D 高斯渲染成图像的模块。"""

    color_space: cs_utils.ColorSpace
    background_color: BackgroundColor

    def __init__(
        self,
        color_space: cs_utils.ColorSpace = "sRGB",
        background_color: BackgroundColor = "black",
        low_pass_filter_eps: float = 0.0,
    ) -> None:
        """初始化 gsplat 渲染器。

        Args:
            color_space: 渲染输出使用的色彩空间。
            background_color: 背景颜色。
            low_pass_filter_eps: 2D 低通滤波 epsilon，用于缓解过小 splat 的数值问题。
        """
        super().__init__()
        self.color_space = color_space
        self.background_color = background_color
        self.low_pass_filter_eps = low_pass_filter_eps

    def forward(
        self,
        gaussians: Gaussians3D,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> RenderingOutputs:
        """从 3D 高斯和相机参数渲染图像。

        Args:
            gaussians: 要渲染的 3D 高斯。
            extrinsics: 目标相机外参，OpenCV 坐标约定。
            intrinsics: 目标相机内参，OpenCV 坐标约定。
            image_width: 输出图像宽度。
            image_height: 输出图像高度。
        """
        batch_size = len(gaussians.mean_vectors)
        outputs_list: list[RenderingOutputs] = []

        for ib in range(batch_size):
            colors, alphas, meta = gsplat.rendering.rasterization(
                means=gaussians.mean_vectors[ib],
                quats=gaussians.quaternions[ib],
                scales=gaussians.singular_values[ib],
                opacities=gaussians.opacities[ib],
                colors=gaussians.colors[ib],
                viewmats=extrinsics[ib : ib + 1],
                Ks=intrinsics[ib : ib + 1, :3, :3],
                width=image_width,
                height=image_height,
                render_mode="RGB+D",
                rasterize_mode="classic",
                absgrad=False,
                packed=False,
                eps2d=self.low_pass_filter_eps,
            )

            rendered_color = colors[..., 0:3].permute([0, 3, 1, 2])
            rendered_depth_unnormalized = colors[..., 3:4].permute([0, 3, 1, 2])
            rendered_alpha = alphas.permute([0, 3, 1, 2])

            # Compose with background color.
            rendered_color = self.compose_with_background(
                rendered_color, rendered_alpha, self.background_color
            )

            # Colorspace conversion.
            if self.color_space == "sRGB":
                pass
            elif self.color_space == "linearRGB":
                rendered_color = cs_utils.linearRGB2sRGB(rendered_color)
            else:
                ValueError("Unsupported ColorSpace type.")

            # splats: (B, N, 10)
            cov2d = self._conics_to_covars2d(meta["conics"])
            # Set the cov2d of invisible splats to 1 to avoid nan in condition number calculation..
            splats_visible_mask = meta["depths"] > 1e-2
            cov2d[~splats_visible_mask][..., 0, 0] = 1
            cov2d[~splats_visible_mask][..., 1, 1] = 1
            cov2d[~splats_visible_mask][..., 0, 1] = 0

            # Normalize the depth by alpha.
            rendered_depth = rendered_depth_unnormalized / torch.clip(rendered_alpha, min=1e-8)

            outputs = RenderingOutputs(
                color=rendered_color,
                depth=rendered_depth,
                alpha=rendered_alpha,
            )
            outputs_list.append(outputs)

        return RenderingOutputs(
            color=torch.cat([item.color for item in outputs_list], dim=0).contiguous(),
            depth=torch.cat([item.depth for item in outputs_list], dim=0).contiguous(),
            alpha=torch.cat([item.alpha for item in outputs_list], dim=0).contiguous(),
        )

    @staticmethod
    def compose_with_background(
        rendered_rgb: torch.Tensor,
        rendered_alpha: torch.Tensor,
        background_color: BackgroundColor,
    ) -> torch.Tensor:
        """Compose rendered RGB with background color."""
        if background_color == "black":
            return rendered_rgb
        elif background_color == "white":
            return rendered_rgb + (1.0 - rendered_alpha)
        elif background_color == "random_color":
            return (
                rendered_rgb
                + (1.0 - rendered_alpha)
                * torch.rand(3, dtype=rendered_rgb.dtype, device=rendered_rgb.device)[
                    None, :, None, None
                ]
            )
        elif background_color == "random_pixel":
            return rendered_rgb + (1.0 - rendered_alpha) * torch.rand_like(rendered_rgb)
        else:
            raise ValueError("Unsupported BackgroundColor type.")

    @staticmethod
    def _conics_to_covars2d(conics: torch.Tensor, eps=1e-8) -> torch.Tensor:
        """Convert conics to covariance matrices."""
        a = conics[..., 0]
        b = conics[..., 1]
        c = conics[..., 2]
        # Reconstruct determinant.
        det = 1 / (a * c - b**2 + eps)
        det = det.clamp(min=eps)
        # Reconstruct covars2d.
        covars2d = torch.zeros(*conics.shape[:-1], 2, 2, device=conics.device)
        covars2d[..., 1, 1] = a * det
        covars2d[..., 0, 0] = c * det
        covars2d[..., 0, 1] = -b * det
        covars2d[..., 1, 0] = -b * det
        covars2d = torch.nan_to_num(covars2d, nan=0.0, posinf=0.0, neginf=0.0)
        return covars2d
