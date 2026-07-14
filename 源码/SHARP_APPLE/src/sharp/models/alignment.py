"""包含不同类型的深度对齐模块。

=对应论文 Section 3.1 (Depth Adjustment):=

深度对齐模块(LearnedAlignment)是一个小型 U-Net(~2M 参数)，
训练时接收预测深度和真值深度，输出一张"局部缩放图 S"，
将单目深度对齐到真值深度: D̄ = S(D̂, D) ⊙ D̂
推理时替换为恒等函数(缩放图=全1)。

灵感来源: 条件变分自编码器(C-VAE)。
传统 C-VAE 用后验模型从真值深度中提取隐变量 z 来消除歧义；
SHARP 简化为: 把 z 解释为"缩放图 S"，并将 C-VAE 中常见的 KL 散度
替换为任务特定的正则化项，让 S 编码"最少必要信息"来纠正深度歧义。

推理时逻辑在 predictor.py 的 DepthAlignment 包装类中实现(见该文件)。

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from sharp.models.decoders import UNetDecoder
from sharp.models.encoders import UNetEncoder
from sharp.utils import math as math_utils

from .params import AlignmentParams


def create_alignment(
    params: AlignmentParams, depth_decoder_dim: int | None = None,
) -> nn.Module | None:
    """创建深度对齐模块。—— 对应论文 Section 3.1 (Depth Adjustment)。

    当前公开代码始终创建 LearnedAlignment；若 depth_decoder_dim 未提供会直接报错。
    推理时虽然模块仍会被构造并加载权重，但 predictor.forward(depth=None)
    会走恒等分支，因此不会真正调用 LearnedAlignment.forward()。
    """
    if depth_decoder_dim is None:
        raise ValueError("LearnedAlignment 需要 depth_decoder_dim。")
    alignment = LearnedAlignment(
        depth_decoder_features=params.depth_decoder_features,
        depth_decoder_dim=depth_decoder_dim,
        steps=params.steps,
        stride=params.stride,
        base_width=params.base_width,
        activation_type=params.activation_type,
    )

    if params.frozen:
        # 冻结对齐模块(使其不参与训练)，可用于推理或训练后期
        alignment.requires_grad_(False)

    return alignment


class LearnedAlignment(nn.Module):
    """使用 U-Net 实现的可学习深度对齐 —— 对应论文 Section 3.1。

    设计思路(论文中描述的 C-VAE 简化):
    1. 输入: 预测逆深度(1/D̂) 和 真值逆深度(1/D)
    2. 编码: 小 U-Net 压缩为隐空间后解码
    3. 输出: 一张"缩放图 S"(shape 同输入)，该图编码了"预测深度需要
       在每一点乘多少倍才能匹配真值深度"
    4. 激活: S = exp(output) 或 S = sigmoid(output)，保证缩放图为正

    初始化技巧:
    输出层权重初始化为 0，偏置初始化为 activation⁻¹(1.0)。
    这保证训练开始时 S = 1(恒等)，对齐从"不改动"开始慢慢学习。
    """

    def __init__(
        self,
        steps: int = 4,                     # U-Net 的下采样/上采样步数
        stride: int = 8,                     # 对齐模块的有效下采样倍率
        base_width: int = 16,                # U-Net 基础通道宽度
        depth_decoder_features: bool = False, # 是否同时输入深度解码器的中间特征
        depth_decoder_dim: int = 256,        # 深度解码器特征的维度
        activation_type: math_utils.ActivationType = "exp",  # 输出激活函数类型
    ) -> None:
        """初始化可学习深度对齐模块。

        Args:
            steps: U-Net 中的步数。
            stride: 对齐模块的有效下采样倍率(必须是 2 的幂)。
            base_width: U-Net 的基础宽度(每步翻倍)。
            depth_decoder_features: 是否使用深度解码器的中间特征。
            depth_decoder_dim: 深度解码器特征的维度。
            activation_type: 对齐输出的激活类型(默认 "exp")。
        """
        super().__init__()
        self.activation = math_utils.create_activation_pair(activation_type)

        # 巧妙的初始化: 偏置设为 f⁻¹(1.0)，使训练初期输出≈1(恒等)
        bias_value = self.activation.inverse(torch.tensor(1.0))

        self.depth_decoder_features = depth_decoder_features
        if depth_decoder_features:
            # 输入: 预测逆深度(1ch) + 真值逆深度(1ch) + 解码器特征(depth_decoder_dim)
            dim_in = 2 + depth_decoder_dim
        else:
            # 默认: 仅预测逆深度 + 真值逆深度(2 通道)
            dim_in = 2

        def is_power_of_two(n: int) -> bool:
            """检查一个数是否是 2 的幂。"""
            if n <= 0:
                return False
            return (n & (n - 1)) == 0

        if not is_power_of_two(stride):
            raise ValueError(f"步长 {stride} 必须是 2 的幂。")

        # 编码器的步数 = steps，解码器的步数 = steps - log₂(stride)
        # 因为 stride 已经提供了部分下采样
        steps_decoder = steps - int(math.log2(stride))
        if steps_decoder < 1:
            raise ValueError(
                f"解码器步数 {steps_decoder} 必须 ≥ 1。"
            )

        # U-Net 的各层宽度: base_width → 2×base_width → 4×base_width → ...
        # 最大不超过 1024
        widths = [min(base_width << i, 1024) for i in range(steps + 1)]

        # 编码器: 输入 dim_in 通道，逐层下采样
        self.encoder = UNetEncoder(
            dim_in=dim_in, width=widths, steps=steps, norm_num_groups=4,
        )
        # 解码器: 输出 width[0] 通道，逐层上采样
        self.decoder = UNetDecoder(
            dim_out=widths[0], width=widths,
            steps=steps_decoder, norm_num_groups=4,
        )

        # 输出卷积: width[0] → 1(单通道缩放图)
        self.conv_out = nn.Conv2d(widths[0], 1, 1, bias=True)

        # 关键初始化: 权重为 0，偏置使输出 = activation⁻¹(1.0)
        # 这样训练从"恒等映射"开始，网络只需学习必要的偏离
        nn.init.zeros_(self.conv_out.weight)
        nn.init.constant_(self.conv_out.bias, bias_value)

    def forward(
        self,
        tensor_src: torch.Tensor,   # 预测深度 D̂
        tensor_tgt: torch.Tensor,   # 真值深度 D
        depth_decoder_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """计算对齐图 —— 预测深度需要乘多少才能匹配真值深度。

        输入在逆深度空间操作(1/深度)，因为逆深度与视差成正比，
        U-Net 在逆深度空间中更容易学到缩放关系。

        Args:
            tensor_src: 预测的深度图(度量单位)，shape [B, C, H, W]。
            tensor_tgt: 真值深度图(度量单位)，shape [B, C, H, W]。
            depth_decoder_features: 可选的深度解码器中间特征。

        Returns:
            对齐图 S，shape [B, 1, H, W]。
            预测深度 × S = 对齐后的深度。
            推理时该模块不会被调用(在 predictor.py 中走恒等分支)。
        """
        # 深度 → 逆深度(1/深度)，因为逆深度数值范围更适合 U-Net 处理
        # 深度通常 ≥ 1.0 且可能跨度很大 → 逆深度在 (0, 1] 范围内
        tensor_src = 1.0 / tensor_src.clamp(min=1e-4)
        tensor_tgt = 1.0 / tensor_tgt.clamp(min=1e-4)

        # 将预测和真值逆深度拼接为 2 通道输入
        tensor_input = torch.cat([tensor_src, tensor_tgt], dim=1)

        # 若使用深度解码器特征，上采样后拼接
        if self.depth_decoder_features:
            height, width = tensor_src.shape[-2:]
            upsampled_encodings = F.interpolate(
                depth_decoder_features,
                size=(height, width),
                mode="bilinear",
            )
            tensor_input = torch.cat(
                [tensor_input, upsampled_encodings], dim=1
            )

        # U-Net 前向: 编码 → 解码 → 1×1 卷积 → 激活
        features = self.encoder(tensor_input)
        output = self.conv_out(self.decoder(features))

        # 应用激活函数得到缩放图 S
        # 若使用 exp 激活: S = exp(output)，S > 0(保证深度符号不变)
        # 初始 output ≈ 0 → S ≈ exp(0) = 1(恒等)
        alignment_map_lowres = self.activation.forward(output)

        # 若输出分辨率与输入不同，上采样对齐
        if alignment_map_lowres.shape[-2:] != tensor_src.shape[-2:]:
            alignment_map = F.interpolate(
                alignment_map_lowres,
                size=tensor_src.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return alignment_map
