"""包含 DPT(Dense Prediction Transformer，稠密预测 Transformer)架构的实现。

=对应论文 Section 3.2 (Gaussian Decoder):=

GaussianDensePredictionTransformer 是论文中"高斯解码器"的核心实现。
它接收来自编码器的多尺度特征(源码中为 5 张特征图)和来自初始化器的
RGB+两层归一化视差特征，
通过 MultiresConvDecoder(DPT 改)解码为统一特征图，再经两个独立的
特征头(texture_head 和 geometry_head)生成两路特征。真正输出 ΔG 的是
heads.py 中的 DirectPredictionHead:
  - geometry_prediction_head: 位置(3ch)
  - texture_prediction_head: 尺度(3ch) + 朝向(4ch) + 颜色(3ch) + 不透明度(1ch)
合计 14 通道 × 2 层 × 768×768 ≈ 1650 万个数值。

这是一个"从零训练(from scratch)"的模块(约 7.8M 参数)，
因为没有现成的"预测高斯修正量"预训练模型可用。

参考: Vision Transformers for Dense Prediction, https://arxiv.org/abs/2103.13413

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from sharp.models.blocks import (
    FeatureFusionBlock2d,
    NormLayerName,
    residual_block_2d,
)
from sharp.models.decoders import BaseDecoder, MultiresConvDecoder
from sharp.models.params import DPTImageEncoderType, GaussianDecoderParams


def create_gaussian_decoder(
    params: GaussianDecoderParams, dims_depth_features: list[int],
) -> "GaussianDensePredictionTransformer":
    """根据 GaussianDecoderParams 创建高斯解码器。—— 对应论文 Section 3.2。

    架构: DPT 适配版
    子模块:
      1. MultiresConvDecoder: 5 层多分辨率卷积解码器
      2. SkipConvBackbone: 处理 RGB + 两层归一化视差输入，提取浅层特征
      3. FeatureFusionBlock2d: 融合浅层特征(Skip)和深层特征(Decoder)
      4. texture_head + geometry_head: 双分支特征头，供 DirectPredictionHead 输出修正量
    """
    decoder = MultiresConvDecoder(
        dims_depth_features,
        params.dims_decoder,
        grad_checkpointing=params.grad_checkpointing,
        upsampling_mode=params.upsampling_mode,
    )

    return GaussianDensePredictionTransformer(
        decoder=decoder,
        dim_in=params.dim_in,
        dim_out=params.dim_out,
        stride_out=params.stride,
        norm_type=params.norm_type,
        norm_num_groups=params.norm_num_groups,
        use_depth_input=params.use_depth_input,
        grad_checkpointing=params.grad_checkpointing,
        image_encoder_type=params.image_encoder_type,
        image_encoder_params=params,
    )


# -- 辅助函数 -------------------------------------------------------------------

def _create_project_upsample_block(
    dim_in: int,
    dim_out: int,
    upsample_layers: int,
    dim_intermediate: int | None = None,
) -> nn.Module:
    """创建"投影 + 上采样"块。

    步骤: 1×1 卷积投影(降维/升维) → 多个转置卷积(2×上采样)
    用于将解码器输出的特征图匹配到所需的输出分辨率。
    """
    if dim_intermediate is None:
        dim_intermediate = dim_out
    # 第一步: 1×1 投影
    blocks = [
        nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_intermediate,
            kernel_size=1, stride=1, padding=0, bias=False,
        )
    ]
    # 后续: 转置卷积上采样(每层 2×)
    blocks += [
        nn.ConvTranspose2d(
            in_channels=dim_intermediate if i == 0 else dim_out,
            out_channels=dim_out,
            kernel_size=2, stride=2, padding=0, bias=False,
        )
        for i in range(upsample_layers)
    ]
    return nn.Sequential(*blocks)


class ImageFeatures(NamedTuple):
    """从解码器中提取的图像特征。

    分成两个分支:
    - texture_features: 用于预测颜色和不透明度修正量
    - geometry_features: 用于预测位置、尺度和朝向修正量
    """

    texture_features: torch.Tensor
    geometry_features: torch.Tensor


class SkipConvBackbone(nn.Module):
    """跳跃卷积骨干 —— 处理 RGB + 两层归一化视差输入产生浅层特征。

    类似 ResNet 的"跳跃连接"思路: 用一个或几个卷积层将
    输入特征映射到解码器空间，产出与解码器深度特征同尺寸的
    浅层"skip"特征，供后续融合(FeatureFusionBlock2d)。

    这个模块只是简单的卷积(不是 Transformer)，因为输入特征
    已经经过初始化器的预处理，只需要轻量编码即可。
    """

    def __init__(
        self, dim_in: int, dim_out: int, kernel_size: int, stride_out: int,
    ):
        super().__init__()
        self.stride_out = stride_out
        if stride_out == 1 and kernel_size != 1:
            raise ValueError(
                "stride_out=1 时仅支持 kernel_size=1。"
            )
        padding: int = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            dim_in, dim_out, kernel_size=kernel_size,
            stride=stride_out, padding=padding,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        encodings: list[torch.Tensor] | None = None,
    ) -> ImageFeatures:
        """将 SkipConv 应用于输入特征并返回纹理和几何特征。

        texture_features == geometry_features (同一个卷积输出)，
        后续 FeatureFusionBlock2d 再做特征融合和分叉。
        """
        output = self.conv(input_features)
        return ImageFeatures(
            texture_features=output,
            geometry_features=output,
        )

    @property
    def stride(self) -> int:
        """有效下采样倍率。"""
        return self.stride_out


class GaussianDensePredictionTransformer(nn.Module):
    """用于高斯预测的 DPT(稠密预测 Transformer) —— 对应论文 Section 3.2。

    这是论文中"高斯解码器"的核心模块。它的作用:
    1. 从单目深度编码器接收 5 张多分辨率特征图
    2. 用 MultiresConvDecoder 解码 → 深层特征
    3. 用 SkipConvBackbone 从 RGB + 两层归一化视差提取浅层特征
    4. 融合(FeatureFusionBlock2d) → 统一特征图
    5. 双头(texture + geometry) → 最终修正量 ΔG
    """

    norm_type: NormLayerName

    def __init__(
        self,
        decoder: BaseDecoder,
        dim_in: int,
        dim_out: int,
        stride_out: int,
        image_encoder_params: GaussianDecoderParams,
        image_encoder_type: DPTImageEncoderType = "skip_conv",
        norm_type: NormLayerName = "group_norm",
        norm_num_groups: int = 8,
        use_depth_input: bool = True,
        grad_checkpointing: bool = False,
    ):
        """初始化高斯 DPT。

        Args:
            decoder: 用于解码特征的多分辨率卷积解码器。
            dim_in: 输入维度(RGB 3 + 视差 1 + 深度 1 = 5 若用深度)。
            dim_out: 最终输出维度(默认为 32)。
            stride_out: 输出特征图的下采样倍率(默认 2)。
            image_encoder_params: 图像编码器的骨干参数(控制 SkipConv)。
            image_encoder_type: 图像编码器类型("skip_conv" 或 "skip_conv_kernel2")。
            norm_type: 归一化层类型("group_norm" 等)。
            norm_num_groups: GroupNorm 的分组数(默认为 8)。
            use_depth_input: 是否包含深度作为输入(默认 True)。
            grad_checkpointing: 是否启用梯度检查点。
        """
        super().__init__()

        self.decoder = decoder
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.stride_out = stride_out
        self.norm_type = norm_type
        self.norm_num_groups = norm_num_groups
        self.use_depth_input = use_depth_input
        self.grad_checkpointing = grad_checkpointing
        self.image_encoder_type = image_encoder_type

        # ---- 1. SkipConv 图像编码器: 处理 RGB + 两层归一化视差输入 ----
        # 将输入特征升维到解码器空间，分辨率匹配解码器输出
        dim_in = self.dim_in if use_depth_input else self.dim_in - 1
        image_encoder_params.dim_in = dim_in
        image_encoder_params.dim_out = decoder.dim_out
        self.image_encoder = self._create_image_encoder(
            image_encoder_params, stride_out
        )

        # ---- 2. 特征融合块: 融合浅层输入特征和深层(MultiresDecoder)特征 ----
        self.fusion = FeatureFusionBlock2d(decoder.dim_out)

        # ---- 3. 上采样层(若 stride_out=1 则上采样 1 次) ----
        if stride_out == 1:
            self.upsample = _create_project_upsample_block(
                decoder.dim_out, decoder.dim_out, upsample_layers=1,
            )
        elif stride_out == 2:
            self.upsample = nn.Identity()  # 不需要上采样
        else:
            raise ValueError(
                "DPT 骨干仅支持 stride 为 1 或 2。"
            )

        # ---- 4. 双头结构 ----
        # 纹理头: 颜色(3) + 不透明度(1) = 4 通道修正量
        self.texture_head = self._create_head(
            dim_decoder=decoder.dim_out, dim_out=self.dim_out,
        )
        # 几何头: 位置(3) + 尺度(3) + 朝向(4) = 10 通道修正量
        self.geometry_head = self._create_head(
            dim_decoder=decoder.dim_out, dim_out=self.dim_out,
        )

    def _create_head(self, dim_decoder: int, dim_out: int) -> nn.Module:
        """创建预测头(纹理头/几何头)。

        每个头由: 残差块×2 + ReLU + 1×1 卷积 + ReLU 组成。
        残差块帮助梯度流动，1×1 卷积做通道维度的投影。
        """
        return nn.Sequential(
            # 第一个残差块(维度不变)
            residual_block_2d(
                dim_in=dim_decoder,
                dim_out=dim_decoder,
                dim_hidden=dim_decoder // 2,
                norm_type=self.norm_type,
                norm_num_groups=self.norm_num_groups,
            ),
            # 第二个残差块(维度不变)
            residual_block_2d(
                dim_in=dim_decoder,
                dim_hidden=dim_decoder // 2,
                dim_out=dim_decoder,
                norm_type=self.norm_type,
                norm_num_groups=self.norm_num_groups,
            ),
            nn.ReLU(),
            # 1×1 卷积压缩到输出通道(如 32)
            nn.Conv2d(dim_decoder, dim_out, kernel_size=1, stride=1),
            nn.ReLU(),
        )

    def _create_image_encoder(
        self, image_encoder_params: GaussianDecoderParams, stride_out: int,
    ) -> nn.Module:
        """根据参数创建浅层输入编码器。

        skip_conv: 1×1 或 3×3 卷积
        skip_conv_kernel2: kernel_size = stride_out 的卷积
        """
        if self.image_encoder_type == "skip_conv":
            # stride_out=2 时用 3×3 卷积(stride=2)，否则用 1×1
            return SkipConvBackbone(
                image_encoder_params.dim_in,
                image_encoder_params.dim_out,
                kernel_size=3 if stride_out != 1 else 1,
                stride_out=stride_out,
            )
        elif self.image_encoder_type == "skip_conv_kernel2":
            return SkipConvBackbone(
                image_encoder_params.dim_in,
                image_encoder_params.dim_out,
                kernel_size=stride_out,
                stride_out=stride_out,
            )
        else:
            raise ValueError(
                f"不支持的图像编码器类型: {self.image_encoder_type}"
            )

    def forward(
        self, input_features: torch.Tensor, encodings: list[torch.Tensor],
    ) -> ImageFeatures:
        """运行单目深度模型并融合特征，预测高斯修正量。

        Args:
            input_features: RGB + 两层归一化视差拼接输入 [B, 5, H, W]。
            encodings: 来自单目深度编码器的多尺度特征 f₁..f₄。

        Returns:
            ImageFeatures: texture_features(纹理修正量) + geometry_features(几何修正量)。
            每个的形状为 [B, dim_out, 2L, H', W']。
        """
        # 步骤 A: 解码深层特征
        # MultiresConvDecoder: 接收 f₁..f₄，逐层上采样并融合 → 统一特征图
        features = self.decoder(encodings).contiguous()
        features = self.upsample(features)

        # 步骤 B: 处理 RGB + 两层归一化视差输入，提取浅层特征
        # SkipConv 卷积编码 → ImageFeatures(texture, geometry)
        if self.use_depth_input:
            skip_features = self.image_encoder(
                input_features
            ).texture_features
        else:
            # 不使用深度: 仅用 RGB 3 通道
            skip_features = self.image_encoder(
                input_features[:, :3].contiguous()
            )

        # 步骤 C: 融合浅层(Skip)与深层(Decoder)特征
        # FeatureFusionBlock2d: 融合残差块 + 注意力(可选)
        features = self.fusion(features, skip_features)

        # 步骤 D: 双头分别预测
        # texture_head → 纹理修正量(颜色Δ + 不透明度Δ)
        texture_features = self.texture_head(features)
        # geometry_head → 位置相关特征；最终只预测位置Δ(见 heads.py)
        geometry_features = self.geometry_head(features)

        return ImageFeatures(
            texture_features=texture_features,
            geometry_features=geometry_features,
        )

    @property
    def stride(self) -> int:
        """GaussianDensePredictionTransformer 的内部分步长。"""
        return self.stride_out
