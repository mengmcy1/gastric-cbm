"""包含 Sliding Pyramid Network(SPN，滑动金字塔网络)架构。

=对应论文 Section 3.2 (Feature Encoder / Depth Pro Backbone):=

SHARP 最核心的特征编码器。它的任务是: 从 1536×1536 的输入图像中，
提取 5 张不同分辨率的特征图，供后续的深度解码器和高斯解码器使用。

SPN 的核心思想：
  1. 塔 3 层图像金字塔 (1536→768→384)
  2. 两层用滑动窗口切块，所有块拼成一个 batch 一起跑 ViT
  3. 块编码器(Patch Encoder) + 图像编码器(Image Encoder) 两个 ViT 分工
  4. split/merge 操作处理块边界重叠(消除接缝)

=两个 ViT 的分工(之前讲过的):=
  块编码器(patch_encoder, 冻结):
    处理高分辨率切块(1536→25块, 768→9块, 384→1块)，
    提取锐利的局部细节(边缘、纹理、细小结构)。
    冻结原因: DINOv2 的场景理解能力完全通用，不需要为视图合成改动。

  图像编码器(image_encoder, 解冻):
    处理 384×384 的整图缩小版，
    提供全局场景上下文(深度排序、整体布局、前景背景关系)。
    解冻原因: 全局理解需要适配视图合成任务的具体需求。

=金字塔层级:=
  原始图 1536  → 切 25 块(5×5, overlap 0.25) → ViT → merge → 96×96 特征
  半分辨率 768 → 切 9 块(3×3, overlap 0.5)  → ViT → merge → 48×48 特征
  1/4 分辨率 384 → 1 块(不切)               → ViT+L →     → 24×24 特征

=split/merge 中的 padding 机制:=
  切块时有重叠(overlap)，拼回时裁掉重叠部分的边界像素，
  这消除了"块与块之间的编码不连续"问题(接缝效应)。

参考: Bochkovskii et al. - "Depth pro: Sharp monocular metric depth in less
      than a second." (ICLR 2024)

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F

from sharp.utils.training import checkpoint_wrapper

from .base_encoder import BaseEncoder
from .vit_encoder import TimmViT

# torch.fx.wrap 用于将函数标记为符号追踪(symbolic tracing)中的叶节点，
# 确保它们不被追踪而是被视为原子操作。
# 简单说: 符号追踪难以处理原生 Python 函数和条件分支，
# 所以把 split/merge 这些纯 Python 操作标记为"不可追踪的原子操作"。
non_traceable_ops = ("len", "int")
for op in non_traceable_ops:
    torch.fx.wrap(op)


class SlidingPyramidNetwork(BaseEncoder):
    """滑动金字塔网络(SPN) —— 对应论文 Section 3.2 中 Depth Pro 骨干。

    一个旨在从 Vision Transformer 创建多分辨率编码的编码器。
    框架流程:
        1. 创建图像金字塔(3层: 1536/768/384)
        2. 在每层金字塔上用滑动窗口生成重叠块
        3. 通过 ViT 骨干产生批量编码
        4. 合并为多分辨率编码

    参考: Bochkovskii et al. - "Depth pro: Sharp monocular metric depth
          in less than a second." (ICLR 2024)
    """

    def __init__(
        self,
        dims_encoder: Iterable[int],
        patch_encoder: TimmViT,
        image_encoder: TimmViT,
        use_patch_overlap: bool = True,
    ):
        """初始化滑动金字塔网络。

        Args:
            dims_encoder: 编码器各层的输出维度(5 个值)。
            patch_encoder: 用于金字塔高分辨率部分的 ViT 骨干(块编码器)。
            image_encoder: 用于金字塔低分辨率部分的 ViT 骨干(图像编码器)。
            use_patch_overlap: 是否在 SPN 中的块之间有重叠(默认为 True)。
        """
        super().__init__()

        self.dim_in = patch_encoder.dim_in

        self.dims_encoder = list(dims_encoder)
        self.patch_encoder = patch_encoder
        self.image_encoder = image_encoder

        # 两个 ViT 的嵌入维度一致(DINOv2 ViT-L/16: 1024)
        base_embed_dim = patch_encoder.embed_dim      # 1024
        lowres_embed_dim = image_encoder.embed_dim    # 1024
        self.patch_size = patch_encoder.internal_resolution()  # 384

        self.grad_checkpointing = False
        self.use_patch_overlap = use_patch_overlap

        # 检索在 create_monodepth_encoder 中注册的中间特征 id
        self.patch_intermediate_features_ids = (
            patch_encoder.intermediate_features_ids
        )
        if (
            not isinstance(self.patch_intermediate_features_ids, list)
            or not len(self.patch_intermediate_features_ids) == 4
        ):
            raise ValueError(
                "块编码器中间特征 id 必须是包含 4 项的列表。"
            )

        self.image_intermediate_features_ids = (
            image_encoder.intermediate_features_ids
        )

        # ---- 上采样和融合模块 ----
        # 这些模块将 ViT 输出的低分辨率特征图(24×24 tokens)
        # 逐层上采样到目标分辨率，并融合"路径A(块编码器)"和
        # "路径B(图像编码器)"的多尺度特征。
        #
        # 上采样块的结构: 1×1投影(降维/升维) + 转置卷积上采样
        def _create_project_upsample_block(
            dim_in: int,
            dim_out: int,
            upsample_layers: int,
            dim_intermediate=None,
        ) -> nn.Module:
            """创建"投影+上采样"块。

            Args:
                dim_in: 输入维度(ViT 嵌入维度, 1024)。
                dim_out: 输出维度(编码器各层的目标维度)。
                upsample_layers: 上采样层数(每层 2× 上采样)。
                dim_intermediate: 中间维度。
            """
            if dim_intermediate is None:
                dim_intermediate = dim_out

            # 投影: 1×1 卷积改变通道数
            blocks = [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=dim_intermediate,
                    kernel_size=1, stride=1, padding=0, bias=False,
                )
            ]

            # 上采样: 转置卷积，每层 2× 放大
            blocks += [
                nn.ConvTranspose2d(
                    in_channels=(
                        dim_intermediate if i == 0 else dim_out
                    ),
                    out_channels=dim_out,
                    kernel_size=2, stride=2, padding=0, bias=False,
                )
                for i in range(upsample_layers)
            ]

            return nn.Sequential(*blocks)

        # ------ 路径A(块编码器)的上采样模块 ------
        # 处理从 24×24 ViT token 上采样到不同分辨率

        # latent0: 中间层特征，上采样 3 次(24→48→96→192)
        self.upsample_latent0 = _create_project_upsample_block(
            dim_in=base_embed_dim,        # 1024
            dim_out=self.dims_encoder[0],  # 256
            upsample_layers=3,
            dim_intermediate=self.dims_encoder[1],
        )
        # latent1: 中间层特征，上采样 2 次(24→48→96)
        self.upsample_latent1 = _create_project_upsample_block(
            dim_in=base_embed_dim,
            dim_out=self.dims_encoder[1],  # 256
            upsample_layers=2,
        )

        # x0 特征(96×96): 上采样 1 次(96→192)
        self.upsample0 = _create_project_upsample_block(
            dim_in=base_embed_dim,
            dim_out=self.dims_encoder[2],  # 256
            upsample_layers=1,
        )
        # x1 特征(48×48): 上采样 1 次(48→96)
        self.upsample1 = _create_project_upsample_block(
            dim_in=base_embed_dim,
            dim_out=self.dims_encoder[3],  # 256
            upsample_layers=1,
        )
        # x2 特征(24×24): 上采样 1 次(24→48)
        self.upsample2 = _create_project_upsample_block(
            dim_in=base_embed_dim,
            dim_out=self.dims_encoder[4],  # 256
            upsample_layers=1,
        )

        # ------ 路径B(图像编码器)的上采样模块 ------
        # 图像编码器输出的 24×24 特征 → 转置卷积上采样 → 48×48
        self.upsample_lowres = nn.ConvTranspose2d(
            in_channels=lowres_embed_dim,       # 1024
            out_channels=self.dims_encoder[4],  # 256
            kernel_size=2, stride=2, padding=0, bias=True,
        )

        # ------ 融合模块 ------
        # 将路径A(patch_encoder)和路径B(image_encoder)的特征拼接后融合
        # 输入: [patch_x2(256ch) + image_lowres(256ch)] = 512ch
        # 1×1 卷积融合为 256ch
        self.fuse_lowres = nn.Conv2d(
            in_channels=(
                self.dims_encoder[4] + self.dims_encoder[4]
            ),  # 512
            out_channels=self.dims_encoder[4],  # 256
            kernel_size=1, stride=1, padding=0, bias=True,
        )

    def internal_resolution(self) -> int:
        """返回 SPN 网络的完整图像尺寸(384×4=1536)。"""
        return self.patch_size * 4

    @torch.jit.ignore
    def set_grad_checkpointing(self, is_enabled=True):
        """启用梯度检查点(同时设置两个 ViT 编码器)。"""
        self.grad_checkpointing = is_enabled
        self.patch_encoder.set_grad_checkpointing(is_enabled)
        self.image_encoder.set_grad_checkpointing(is_enabled)

    @torch.jit.ignore
    def set_requires_grad_(self, patch_encoder: bool, image_encoder: bool):
        """设置各组件是否参与梯度计算 —— 对应论文 Section 3.2 冻结策略。

        这是 SHARP 训练时选择性微调的核心逻辑:
        - patch_encoder=False: 冻结(保留 DINOv2 通用局部特征)
        - image_encoder=True:  解冻(让全局理解适配视图合成)
        - 两个 ViT 的 head(分类头)永远冻结(预测深度时用不到)
        """
        # 两个 ViT 骨干: 各自独立控制
        self.patch_encoder.requires_grad_(patch_encoder)
        self.image_encoder.requires_grad_(image_encoder)

        # 永久冻结未使用的 TimmViT head(DINOv2 的分类头)
        # 以排除它干扰可训练参数的计算
        self.patch_encoder.head.requires_grad_(False)
        self.image_encoder.head.requires_grad_(False)

        # 以下上采样器仅影响块编码器的特征图
        self.upsample_latent0.requires_grad_(patch_encoder)
        self.upsample_latent1.requires_grad_(patch_encoder)
        self.upsample0.requires_grad_(patch_encoder)
        self.upsample1.requires_grad_(patch_encoder)
        self.upsample2.requires_grad_(patch_encoder)

        # 此上采样器仅影响图像编码器的特征图
        self.upsample_lowres.requires_grad_(image_encoder)

        # 此融合器同时影响图像和块编码器
        self.fuse_lowres.requires_grad_(image_encoder or patch_encoder)

    def _create_pyramid(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """创建 3 层图像金字塔。

        输入:  [B, 3, 1536, 1536]
        输出:  x0=1536(原图), x1=768(半), x2=384(1/4)
        """
        x0 = x  # 原分辨率，默认为 1536

        # 中等分辨率，默认为 768
        x1 = F.interpolate(
            x, size=None, scale_factor=0.5,
            mode="bilinear", align_corners=False,
        )

        # 低分辨率，默认为 384(对应骨干分辨率)
        x2 = F.interpolate(
            x, size=None, scale_factor=0.25,
            mode="bilinear", align_corners=False,
        )

        return x0, x1, x2

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """以多分辨率编码输入 —— SPN 的前向传播。

        Args:
            x: [B, 3, 1536, 1536]，已归一化到 [-1,1]。

        Returns:
            5 张多分辨率特征图列表:
            [latent0, latent1, x0, x1, fused_lowres]
        """
        batch_size = x.shape[0]

        # ---- 步骤 0: 创建 3 层图像金字塔 ----
        x0, x1, x2 = self._create_pyramid(x)

        # ---- 步骤 1: 滑动窗口切块 ----
        if self.use_patch_overlap:
            # 最高分辨率(1536): 5×5 = 25 块，384×384，重叠 0.25
            x0_patches = split(
                x0, overlap_ratio=0.25, patch_size=self.patch_size,
            )
            # 中等分辨率(768): 3×3 = 9 块，384×384，重叠 0.5
            x1_patches = split(
                x1, overlap_ratio=0.5, patch_size=self.patch_size,
            )
            # 最低分辨率(384): 1×1 = 1 块，384×384
            x2_patches = x2
            padding = 3
        else:
            # 无重叠版本: 4×4 + 2×2 + 1×1 = 21 块
            # (速度更快但可能有接缝)
            x0_patches = split(
                x0, overlap_ratio=0.0, patch_size=self.patch_size,
            )
            x1_patches = split(
                x1, overlap_ratio=0.0, patch_size=self.patch_size,
            )
            x2_patches = x2
            padding = 0
        x0_tile_size = x0_patches.shape[0]

        # ---- 步骤 2: 批量 ViT 编码 ----
        # 将所有滑动窗口块拼接为一个大批次:
        #   有重叠: 25 + 9 + 1 = 35 块
        #   无重叠: 16 + 4 + 1 = 21 块
        x_pyramid_patches = torch.cat(
            (x0_patches, x1_patches, x2_patches), dim=0,
        )

        # 运行 ViT 模型并获得大批次的编码结果
        # ViT 将每块 384×384 编码为 24×24 tokens
        #
        # 关于为何不使用 forward hooks 来检索中间特征:
        # forward hooks 更简洁，但与符号追踪(symbolic tracing)不兼容。
        # 因为子模块的属性在追踪过程中可能丢失，hook 在
        # 图变换时可能无法保留，导致意外行为。
        # 因此改为在 forward 返回中一并输出中间特征。
        x_pyramid_encodings, patch_intermediate_features = (
            self.patch_encoder(x_pyramid_patches)
        )

        # ---- 步骤 3: 合并(merge) ----
        # 将 ViT token 序列按网格拼回特征图，裁掉重叠边界

        # 路径 A: 块编码器的高分辨率中间特征
        # latent0: 浅层中间特征 → 合并 → 上采样
        x_latent0_encodings = self.patch_encoder.reshape_feature(
            patch_intermediate_features[
                self.patch_intermediate_features_ids[0]
            ]
        )
        x_latent0_features = merge(
            x_latent0_encodings[: batch_size * x0_tile_size],
            batch_size=batch_size,
            padding=padding,
        )

        # latent1: 另一层中间特征 → 合并 → 上采样
        x_latent1_encodings = self.patch_encoder.reshape_feature(
            patch_intermediate_features[
                self.patch_intermediate_features_ids[1]
            ]
        )
        x_latent1_features = merge(
            x_latent1_encodings[: batch_size * x0_tile_size],
            batch_size=batch_size,
            padding=padding,
        )

        # ---- 步骤 4: 按金字塔层拆回 ViT 输出 ----
        # 将 35(或 21)批次的输出拆回: 25→9→1(按金字塔层级)
        x0_encodings, x1_encodings, x2_encodings = torch.split(
            x_pyramid_encodings,
            [len(x0_patches), len(x1_patches), len(x2_patches)],
            dim=0,
        )

        # 合并各层的 ViT tokens 为特征图:
        #   25 块(5×5) × 24×24 tokens → merge → 96×96 特征图
        #   (5×24 - 3×4(裁边) = 120 - 24 = 96)
        x0_features = merge(
            x0_encodings, batch_size=batch_size, padding=padding,
        )

        #   9 块(3×3) × 24×24 tokens → merge → 48×48 特征图
        #   (3×24 - 3×2×2(裁边) = 72 - 24 = 48)
        x1_features = merge(
            x1_encodings, batch_size=batch_size,
            padding=2 * padding,
        )

        #   1 块(1×1) × 24×24 tokens → 24×24 特征图(不merge)
        x2_features = x2_encodings

        # ---- 步骤 5: 路径 B - 图像编码器 ----
        # 将 384×384 整图面给图像编码器(第二个 ViT)
        x_lowres_features, image_intermediate_features = (
            self.image_encoder(x2_patches)
        )

        # ---- 步骤 6: 上采样所有特征图到目标分辨率 ----
        # 路径 A(块编码器)特征:
        x_latent0_features = checkpoint_wrapper(
            self, self.upsample_latent0, x_latent0_features,
        )
        x_latent1_features = checkpoint_wrapper(
            self, self.upsample_latent1, x_latent1_features,
        )
        x0_features = checkpoint_wrapper(
            self, self.upsample0, x0_features,
        )
        x1_features = checkpoint_wrapper(
            self, self.upsample1, x1_features,
        )
        x2_features = checkpoint_wrapper(
            self, self.upsample2, x2_features,
        )

        # 路径 B(图像编码器)特征:
        x_lowres_features = checkpoint_wrapper(
            self, self.upsample_lowres, x_lowres_features,
        )

        # ---- 步骤 7: 融合路径 A 和路径 B ----
        # 将块编码器的 x2 特征(局部细节)与图像编码器的
        # lowres 特征(全局上下文)拼接后融合
        # 这是"局部锐利 + 全局一致"的关键一步
        x_lowres_features = checkpoint_wrapper(
            self,
            self.fuse_lowres,
            torch.cat((x2_features, x_lowres_features), dim=1),
        )

        # ---- 输出: 5 张多分辨率特征图 ----
        # 这些特征图供后续的 DPT 深度解码器使用
        # 分辨率从细到粗: latent0 > latent1 > x0 > x1 > fused
        output = [
            x_latent0_features,   # 最细粒度(路径A中间特征)
            x_latent1_features,   # 次细粒度(路径A中间特征)
            x0_features,          # 较细(原图切块的编码)
            x1_features,          # 中等(半分辨率切块的编码)
            x_lowres_features,    # 最粗(路径A+B融合，含全局上下文)
        ]

        return output


# ==========================================================================
# split/merge 辅助函数
# ==========================================================================
# torch.fx.wrap 只能应用于函数，不能应用于方法。
# 因此 split 和 merge 被转换为全局函数，以在符号追踪中被标记为原子操作。

@torch.fx.wrap
def split(
    image: torch.Tensor,
    overlap_ratio: float = 0.25,
    patch_size: int = 384,
) -> torch.Tensor:
    """使用滑动窗口将输入图像切成小块。

    这是 SPN 的"切块"操作:
    在图像上用步长 = patch_size × (1 - overlap_ratio) 的滑动窗口
    逐一切出 patch_size × patch_size 的块。

    Args:
        image: 输入图像 [B, C, H, W]。
        overlap_ratio: 相邻块之间的重叠比例(默认 0.25)。
        patch_size: 每块的尺寸(默认 384，对应 ViT 的分辨率)。

    Returns:
        拼接后的块 [B*N_patches, C, patch_size, patch_size]。

    示例: 1536×1536 的图，patch_size=384, overlap_ratio=0.25
          stride = 384×0.75 = 288
          steps = ceil((1536-384)/288)+1 = 5
          所以切成 5×5 = 25 块，相邻块有 96pixel 重叠
    """
    # 步长: patch_size 减去重叠部分(overlap)
    patch_stride = int(patch_size * (1 - overlap_ratio))

    image_size = image.shape[-1]
    # 需要多少步才能覆盖整张图
    steps = int(
        math.ceil((image_size - patch_size) / patch_stride)
    ) + 1

    # 逐行逐列切块
    x_patch_list = []
    for j in range(steps):
        j0 = j * patch_stride
        j1 = j0 + patch_size

        for i in range(steps):
            i0 = i * patch_stride
            i1 = i0 + patch_size
            x_patch_list.append(image[..., j0:j1, i0:i1])

    # 拼接所有块为一个大 batch
    return torch.cat(x_patch_list, dim=0)


@torch.fx.wrap
def merge(
    image_patches: torch.Tensor,
    batch_size: int,
    padding: int = 3,
) -> torch.Tensor:
    """将切块后的 ViT 编码拼回为一张大特征图。

    这是 split 的逆操作:
    将 N×N 个小块(token网格)按原始位置拼回，并裁掉重叠部分。

    Args:
        image_patches: ViT 编码后的 token [B*N², D, h, w]。
        batch_size: 原始 batch 大小。
        padding: 每块要裁掉的边距(因切块时有重叠)。

    Returns:
        合并后的特征图 [B, D, H', W']。

    裁边逻辑:
      左/上边缘: 不裁(没有左边/上边的邻块)
      内部块: 裁掉 left/right/top/bottom 各 padding 列/行
      右/底边缘: 不裁(没有右边/下边的邻块)

    这样可以消除块与块之间因"独立编码"造成的接缝不连续。
    """
    # 每轴的块数(假设正方形切块)
    steps = int(math.sqrt(image_patches.shape[0] // batch_size))

    idx = 0

    output_list = []
    for j in range(steps):
        output_row_list = []
        for i in range(steps):
            output = image_patches[
                batch_size * idx : batch_size * (idx + 1)
            ]

            # 裁掉重叠边界(内部块才需要裁):
            #   第一行/列: 不裁 top/left(没有邻接块)
            #   最后一行/列: 不裁 bottom/right(没有邻接块)
            #   内部块: 裁 top/left 各 padding(这些像素在邻块中已出现)
            if padding != 0:
                if j != 0:        # 非最顶行 → 裁上边界
                    output = output[..., padding:, :]
                if i != 0:        # 非最左列 → 裁左边界
                    output = output[..., :, padding:]
                if j != steps - 1: # 非最底行 → 裁下边界
                    output = output[..., :-padding, :]
                if i != steps - 1: # 非最右列 → 裁右边界
                    output = output[..., :, :-padding]

            output_row_list.append(output)
            idx += 1

        # 横向拼接 → 一行
        output_row = torch.cat(output_row_list, dim=-1)
        output_list.append(output_row)

    # 纵向拼接 → 完整特征图
    output = torch.cat(output_list, dim=-2)
    return output
