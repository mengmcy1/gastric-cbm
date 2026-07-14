"""包含 DPT(Dense Prediction Transformer，稠密预测 Transformer)架构的实现。

=对应论文 Section 3.2 (Feature Encoder / Depth Decoder):=

这个模块组装了 SHARP 的深度预测流水线:
  1. SPN(Sliding Pyramid Network): 两个 ViT 的编码器
  2. MultiresConvDecoder: DPT 解码器
  3. 输出头: 解码器特征 → 单通道或多通道视差(两层深度)

冻结/解冻策略(论文 Section 3.2 "选择性微调"):
  create_monodepth_dpt() 中:
    ① 先整体冻结(requires_grad_(False))
    ② 按需解冻: image_encoder(全局理解), decoder(深度预测), head(两层输出)
    ③ 永远冻结: patch_encoder(局部特征), norm_layers(预训练统计量)

MonodepthWithEncodingAdaptor: 包装器，从编码器抽取多尺度特征供高斯解码器使用。

参考: Vision Transformers for Dense Prediction, https://arxiv.org/abs/2103.13413

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import copy
from typing import NamedTuple, Tuple

import torch
import torch.nn as nn

from sharp.models import normalizers
from sharp.models.decoders import MultiresConvDecoder, create_monodepth_decoder
from sharp.models.encoders import (
    SlidingPyramidNetwork,
    create_monodepth_encoder,
)
from sharp.utils import module_surgery

from .params import MonodepthAdaptorParams, MonodepthParams

DimsDecoder = Tuple[int, int, int, int, int]


class MonodepthDensePredictionTransformer(nn.Module):
    """单目深度的 DPT(稠密预测 Transformer) —— 对应论文 Section 3.2。

    将 SPN 编码器(两个 ViT)和 DPT 解码器组装为完整的深度预测模型。
    """

    def __init__(
        self,
        encoder: SlidingPyramidNetwork,
        decoder: MultiresConvDecoder,
        last_dims: tuple[int, int],
    ):
        """初始化 DPT(单目深度预测)。

        Args:
            encoder: SPN 骨干(两个 ViT 的金字塔结构)。
            decoder: 多分辨率卷积解码器。
            last_dims: 最后一个卷积块的(中间维度, 最终输出维度)。
        """
        super().__init__()

        # 输入归一化: [0,1] → [-1,1](ViT 期望输入范围)
        self.normalizer = normalizers.AffineRangeNormalizer(
            input_range=(0, 1), output_range=(-1, 1),
        )
        self.encoder = encoder
        self.decoder = decoder

        # 输出头: 解码器特征 → 视差图
        dim_decoder = decoder.dim_out
        self.head = nn.Sequential(
            nn.Conv2d(
                dim_decoder, dim_decoder // 2,
                kernel_size=3, stride=1, padding=1,
            ),
            # 转置卷积上采样(2×)，恢复到编码器输入分辨率
            nn.ConvTranspose2d(
                in_channels=dim_decoder // 2,
                out_channels=dim_decoder // 2,
                kernel_size=2, stride=2, padding=0, bias=True,
            ),
            nn.Conv2d(
                dim_decoder // 2, last_dims[0],
                kernel_size=3, stride=1, padding=1,
            ),
            nn.ReLU(True),
            # 最终 1×1 卷积: 中间维度 → 输出维度(1 或 n 通道视差)
            nn.Conv2d(
                last_dims[0], last_dims[1],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ReLU(),
        )

        # 最终卷积层偏置初始化为 0(视差从 0 开始)
        self.head[4].bias.data.fill_(0)

        self.grad_checkpointing = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, is_enabled=True):
        """启用梯度检查点(训练时用显存换计算时间)。"""
        self.grad_checkpointing = is_enabled
        self.encoder.set_grad_checkpointing(self.grad_checkpointing)
        self.decoder.set_grad_checkpointing(self.grad_checkpointing)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """前向传播: 输入图像 → 编码 → 解码 → 视差图。

        Args:
            image: [B, 3, 1536, 1536]，归一化到 [0,1]。

        Returns:
            disparity: [B, num_layers, 1536, 1536]，视差(1/深度)。
        """
        # 归一化 [0,1] → [-1,1](ViT 输入)
        encodings = self.encoder(self.normalizer(image))

        # 解码: 5 张多分辨率特征图 → 融合 → 统一特征图
        num_encoder_features = len(self.encoder.dims_encoder)
        features = self.decoder(encodings[:num_encoder_features])

        # 输出头: 特征 → 视差
        disparity = self.head(features)
        return disparity

    def internal_resolution(self) -> int:
        """网络的内部图像尺寸(1536)。"""
        return self.encoder.internal_resolution()


def create_monodepth_dpt(
    params: MonodepthParams | None = None,
) -> MonodepthDensePredictionTransformer:
    """创建 DepthDensePredictionTransformer 模型 —— 对应论文 Section 3.2。

    包含 SPN 编码器 + DPT 解码器 + 冻结/解冻逻辑。
    这是 SHARP 推理和训练时都调用的工厂函数。

    Args:
        params: 单目深度网络参数(控制冻结/解冻、编码器配置等)。

    Returns:
        配置好的单目深度 DPT 模型。
    """
    if params is None:
        params = MonodepthParams()

    # 1. 创建 SPN 编码器(两个 ViT 的金字塔结构)
    encoder: SlidingPyramidNetwork = create_monodepth_encoder(
        params.patch_encoder_preset,
        params.image_encoder_preset,
        use_patch_overlap=params.use_patch_overlap,
        last_encoder=params.dims_decoder[0],
    )

    # 2. 创建 DPT 多分辨率卷积解码器
    decoder: MultiresConvDecoder = create_monodepth_decoder(
        params.patch_encoder_preset, params.dims_decoder,
    )

    # 3. 组装
    monodepth_model = MonodepthDensePredictionTransformer(
        encoder=encoder, decoder=decoder, last_dims=(32, 1),
    )

    # ---- 4. 冻结/解冻控制(论文 Section 3.2 的关键设计) ----
    # 默认不训练单目深度模型(保留预训练权重)。
    # 但允许选择性地解冻网络的某些部分以适配视图合成任务。
    monodepth_model.requires_grad_(False)                # 先整体冻结

    # 编码器: 两路独立设置(见 spn_encoder.py set_requires_grad_)
    monodepth_model.encoder.set_requires_grad_(
        patch_encoder=params.unfreeze_patch_encoder,      # 默认 False(冻)
        image_encoder=params.unfreeze_image_encoder,      # 训练时 True(解冻)
    )
    # 解码器和输出头: 独立控制
    monodepth_model.decoder.requires_grad_(params.unfreeze_decoder)
    monodepth_model.head.requires_grad_(params.unfreeze_head)

    # 归一化层: 永远冻结(保留 DINOv2 预训练的统计量)
    if not params.unfreeze_norm_layers:
        module_surgery.freeze_norm_layer(monodepth_model)

    # 梯度检查点(可选)
    monodepth_model.set_grad_checkpointing(params.grad_checkpointing)

    return monodepth_model


class MonodepthOutput(NamedTuple):
    """单目深度模型的输出。

    包含:
      disparity: 视差图 [B, num_layers, H, W]
      encoder_features: 多尺度编码器特征 f₁..f₄
      decoder_features: 解码器输出的中间特征(可选)
      output_features: 打包的特征列表(供高斯解码器用)
      intermediate_features: 编码器的中间层特征(可选，用于知识蒸馏)
    """

    disparity: torch.Tensor
    encoder_features: list[torch.Tensor]
    decoder_features: torch.Tensor
    output_features: list[torch.Tensor]            # 给高斯解码器用
    intermediate_features: list[torch.Tensor] = []  # 给知识蒸馏用


class MonodepthWithEncodingAdaptor(nn.Module):
    """带特征输出的单目深度适配器 —— 对应论文 Section 3.1 图3 中连接部分。

    将 MonodepthDensePredictionTransformer 包装一层，
    控制哪些特征需要输出给后续的高斯解码器。
    支持多通道深度输出(如 2 层深度)和深度排序。
    """

    def __init__(
        self,
        monodepth_predictor: MonodepthDensePredictionTransformer,
        return_encoder_features: bool,
        return_decoder_features: bool,
        num_monodepth_layers: int,
        sorting_monodepth: bool,
    ):
        """初始化单目深度特征适配器。

        Args:
            monodepth_predictor: 底层单目深度模型。
            return_encoder_features: 是否返回编码器特征(给高斯解码器)。
            return_decoder_features: 是否返回解码器特征。
            num_monodepth_layers: 单目深度模型预测的深度层数(默认为 2)。
            sorting_monodepth: 是否对两层深度排序(前景/背景)。
        """
        super().__init__()
        self.monodepth_predictor = monodepth_predictor
        self.return_encoder_features = return_encoder_features
        self.return_decoder_features = return_decoder_features
        self.num_monodepth_layers = num_monodepth_layers
        self.sorting_monodepth = sorting_monodepth

    def forward(self, image: torch.Tensor) -> MonodepthOutput:
        """处理图像并返回视差和特征图。

        Args:
            image: [B, 3, 1536, 1536]，范围 [0,1]。

        Returns:
            MonodepthOutput: 视差 + 编码器特征(可选) + 解码器特征(可选)。
        """
        # 归一化 [0,1] → [-1,1]
        inputs = self.monodepth_predictor.normalizer(image)

        # SPN 编码
        encoder_output = self.monodepth_predictor.encoder(inputs)

        # 分离: 前 5 张是编码器特征(给 DPT)，其余是中间层特征
        num_encoder_features = len(
            self.monodepth_predictor.encoder.dims_encoder
        )
        encoder_features = encoder_output[:num_encoder_features]
        intermediate_features = encoder_output[num_encoder_features:]

        # DPT 解码
        decoder_features = self.monodepth_predictor.decoder(encoder_features)

        # 输出头 → 视差
        disparity = self.monodepth_predictor.head(decoder_features)

        # 若为两层深度且启用排序，按视差大小排序
        # (大视差=近处 为第1层，小视差=远处 为第2层)
        if self.num_monodepth_layers == 2 and self.sorting_monodepth:
            first_layer_disparity = disparity.max(dim=1, keepdims=True).values
            second_layer_disparity = disparity.min(dim=1, keepdims=True).values
            disparity = torch.cat(
                [first_layer_disparity, second_layer_disparity], dim=1,
            )

        # 打包给高斯解码器用的特征
        output_features = []
        if self.return_encoder_features:
            output_features.extend(encoder_features)
        if self.return_decoder_features:
            output_features.append(decoder_features)

        return MonodepthOutput(
            disparity=disparity,
            encoder_features=encoder_features,
            decoder_features=decoder_features,
            output_features=output_features,
            intermediate_features=intermediate_features,
        )

    def get_feature_dims(self) -> list[int]:
        """返回输出特征图的维度列表。"""
        dims = []
        if self.return_encoder_features:
            dims.extend(self.monodepth_predictor.encoder.dims_encoder)
        if self.return_decoder_features:
            dims.append(self.monodepth_predictor.decoder.dim_out)
        return dims

    def internal_resolution(self) -> int:
        """返回网络的内部图像尺寸(1536)。"""
        return self.monodepth_predictor.internal_resolution()

    def replicate_head(self, num_repeat: int):
        """复制最后的卷积层(head[4])以支持多通道深度输出。

        当需要从单通道扩展到多通道(如2层深度)时，
        不是重新创建整个头，而是复制现有卷积的权重和偏置。
        这保留了预训练的深度预测能力。
        """
        conv_last = copy.deepcopy(
            self.monodepth_predictor.head[4]
        )
        self.monodepth_predictor.head[4].out_channels = num_repeat
        self.monodepth_predictor.head[4].weight = nn.Parameter(
            conv_last.weight.repeat(num_repeat, 1, 1, 1),
        )
        self.monodepth_predictor.head[4].bias = nn.Parameter(
            conv_last.bias.repeat(num_repeat),
        )


def create_monodepth_adaptor(
    monodepth_predictor: MonodepthDensePredictionTransformer,
    params: MonodepthAdaptorParams,
    num_monodepth_layers: int,
    sorting_monodepth: bool,
) -> MonodepthWithEncodingAdaptor:
    """创建返回视差和特征的适配器 —— 推理时调用。

    推理配置: return_encoder_features=True, return_decoder_features=False。
    即只输出编码器多尺度特征(供高斯解码器用)，不输出解码器特征。
    """
    adaptor = MonodepthWithEncodingAdaptor(
        monodepth_predictor=monodepth_predictor,
        return_encoder_features=params.encoder_features,
        return_decoder_features=params.decoder_features,
        num_monodepth_layers=num_monodepth_layers,
        sorting_monodepth=sorting_monodepth,
    )
    return adaptor
