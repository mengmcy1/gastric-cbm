"""定义仅使用 RGB 图像的高斯预测器。

=对应论文 Section 3.1 (Method - Overview):=
整个 SHARP 方法的核心编排模块。
RGBGaussianPredictor.forward() 实现了论文图3中的完整推理流程:
  输入图 → ①Depth Pro 编码器(特征) → ②深度解码器(两层深度)
  → ③深度对齐(训练时用真值纠偏，推理时跳过)
  → ④高斯初始化器(不可学习，几何规则) → ⑤高斯解码器(DPT，预测修正量 ΔG)
  → ⑥高斯合成器(base + η·Δ → 最终高斯)
注意: 渲染器不在 RGBGaussianPredictor.forward() 内部调用，而是在 cli/render.py 或
predict.py 的 --render 分支中单独执行。
详见 forward() 方法内的流程图注释。

DepthAlignment 类负责包装对齐逻辑，推理时如果 depth=None 则走恒等分支(乘1)。

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from sharp.models.monodepth import MonodepthWithEncodingAdaptor
from sharp.utils.gaussians import Gaussians3D

from .composer import GaussianComposer

LOGGER = logging.getLogger(__name__)


class DepthAlignment(nn.Module):
    """深度对齐模块 —— 对应论文 Section 3.1 (Depth Adjustment)。

    包装 scale_map_estimator，使得包含条件逻辑的对齐操作可以从
    符号追踪(symbolic tracing)中排除，保证预测器的计算图是静态的。

    推理时(无真值深度)自动走恒等分支，与论文描述的"将深度对齐模块替换为恒等函数"完全一致。
    """

    def __init__(self, scale_map_estimator: nn.Module | None):
        """初始化深度对齐包装器。

        Args:
            scale_map_estimator: 将单目深度对齐到真值深度的模块（训练时有，推理时可为 None）。
        """
        super().__init__()
        self.scale_map_estimator = scale_map_estimator

    def forward(
        self,
        monodepth: torch.Tensor,
        depth: torch.Tensor,
        depth_decoder_features: torch.Tensor | None = None,
    ):
        """（可选）用局部缩放图将单目深度对齐到真值深度。

        Args:
            monodepth: 单目深度模型预测的深度图。
            depth: 用于对齐的真值深度图（推理时为 None）。
            depth_decoder_features: （可选）深度解码器的中间特征。
        """
        if depth is not None and self.scale_map_estimator is not None:
            # 训练时：有真值深度，用 U-Net 算局部缩放图，校正单目深度
            depth_alignment_map = self.scale_map_estimator(
                monodepth[:, 0:1], depth, depth_decoder_features
            )
            monodepth = depth_alignment_map * monodepth
        else:
            # 推理时（depth=None）或无对齐模块：缩放图全为 1 = 恒等映射
            # 某些损失函数依赖对齐图的存在，因此创建一个"假"的全 1 对齐图
            depth_alignment_map = torch.ones_like(monodepth)
        return monodepth, depth_alignment_map


class RGBGaussianPredictor(nn.Module):
    """从图像预测 3D 高斯 —— 对应论文 Section 3.1，SHARP 的核心模型。

    将四个可学习模块(编码器、深度解码器、高斯解码器、深度对齐)
    和两个不可学习但可微模块(初始化器、合成器)串联为完整的端到端流水线。
    """

    feature_model: nn.Module

    def __init__(
        self,
        init_model: nn.Module,
        monodepth_model: MonodepthWithEncodingAdaptor,
        feature_model: nn.Module,
        prediction_head: nn.Module,
        gaussian_composer: GaussianComposer,
        scale_map_estimator: nn.Module | None,
    ) -> None:
        """初始化 RGBGaussianPredictor。

        Args:
            init_model: 将图像和深度映射为基础高斯值的初始化器(对应论文 3.1 图3 中 "Gaussian Initializer")。
            monodepth_model: 带中间特征的单目深度模型(对应论文 Depth Pro 骨干 + DPT 深度解码器)。
            feature_model: 从 RGB + 两层归一化视差特征预测高斯修正量的核心网络(对应论文 3.2 中 "Gaussian Decoder" 的编码器部分)。
            prediction_head: 将特征解码为 ΔG 的预测头(对应论文 3.2 中 "Gaussian Decoder" 的预测头)。
            gaussian_composer: 将基础值(base)和修正量(Δ)合成为最终高斯的模块(对应论文公式 3.2)。
            scale_map_estimator: 将单目深度对齐到真值深度的模块(对应论文 3.1 中 "Depth Adjustment")。

        Note:
        ----
            当 monodepth_model 可训练时，使用局部深度对齐可能导致单目深度模型
            丧失预测形状的能力。建议在这种情况下停用对应标志。
        """
        super().__init__()
        self.init_model = init_model
        self.feature_model = feature_model
        self.monodepth_model = monodepth_model
        self.prediction_head = prediction_head
        self.gaussian_composer = gaussian_composer
        self.depth_alignment = DepthAlignment(scale_map_estimator)

    def forward(
        self,
        image: torch.Tensor,
        disparity_factor: torch.Tensor,
        depth: torch.Tensor | None = None,
    ) -> Gaussians3D:
        """预测 3D 高斯 —— SHARP 模型的主前向传播。

        Args:
            image: 输入图像（已 resize 到 1536×1536）。
            disparity_factor: 将深度转换为视差的因子(焦距/图宽)。
            depth: 用于对齐预测深度的真值深度图（仅在训练时传入，推理时为 None）。

        Returns:
            预测的 3D 高斯(Gaussians3D)，包含位置/尺度/朝向/颜色/不透明度。

        Note:
        ----
            训练时建议传入真值深度图以对齐预测深度。
            推理时建议 depth=None，使用模型的单目深度输出即可。

        完整数据流（对应论文图3）：

                输入图 I  (1536×1536×3)
                     │
            ┌────────▼────────┐
            │ ① monodepth_model │  Depth Pro 骨干: SPN 编码器 → 特征 f₁..f₄
            │   → disparity     │  DPT 解码器 → 两层视差图
            └────────┬────────┘
                     │ disparity_factor / disparity = monodepth(度量深度)
                     ▼
            ┌────────▼────────┐
            │ ② depth_alignment│  训练时: U-Net 算缩放图 S，乘到深度上
            │   (推理时恒等)    │  推理时: 缩放图=全1，深度不变
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │ ③ init_model     │  RGB + depth → 基础高斯 G₀ (NDC坐标、固定尺度/朝向/颜色/不透明度)
            │   高斯初始化器    │  + feature_input (RGB+视差拼接，供高斯解码器用)
            │   不可学习        │
            └────────┬────────┘
            init_output: .gaussian_base_values, .feature_input, .global_scale
                     │
            ┌────────▼────────┐
            │ ④ feature_model  │  SkipConv 处理 RGB + 两层归一化视差 → 浅层特征
            │   + decoder      │  MultiresConvDecoder 解码 f₁..f₄ → 深层特征
            │                  │  FeatureFusionBlock2d 融合 → 统一特征图
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │ ⑤ prediction_head│  geometry_features → 位置修正量(3通道)
            │   (双分支特征)    │  texture_features → 尺度/朝向/颜色/不透明度修正量(11通道)
            │                  │  合并 → ΔG (14通道 × 2层 × 768×768)
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │ ⑥ gaussian_      │  论文公式 3.2:
            │   composer       │  G_attr = γ_attr( γ⁻¹_attr(G₀,attr) + η_attr · ΔG_attr )
            │   高斯合成器      │  在各属性的"激活空间"中加法，再映射回物理空间
            │                  │  flatten → 1,179,648 个 3D 高斯 (NDC 空间)
            └────────┬────────┘
                     │
                     ▼
               Gaussians3D (NDC 空间，后续由 unproject_gaussians 转到世界坐标)
        """
        # 步骤①: 估计深度 + 提取多尺度特征
        # 对应论文 3.2 中 Depth Pro 骨干(特征编码器) + DPT 深度解码器
        monodepth_output = self.monodepth_model(image)
        monodepth_disparity = monodepth_output.disparity

        # 视差 → 度量深度: depth = disparity_factor / disparity
        # disparity_factor = 焦距(像素) / 图宽(像素)，用于还原物理尺度
        disparity_factor = disparity_factor[:, None, None, None]
        monodepth = disparity_factor / monodepth_disparity.clamp(min=1e-4, max=1e4)

        # 步骤②: （可选）将对齐预测深度到真值深度
        # 对应论文 3.1 中 "Depth adjustment" 模块
        # 训练时: depth != None → 走 U-Net 对齐
        # 推理时: depth == None → 走恒等(对齐图全为 1)
        #
        # 深度对齐包装为独立子模块 DepthAlignment，以方便符号追踪(symbolic tracing)。
        # 这样包含条件逻辑的对齐子模块可以在追踪时被排除，使预测器的计算图保持静态。
        monodepth, _ = self.depth_alignment(
            monodepth,
            depth,
            monodepth_output.decoder_features,
        )

        # 步骤③: RGB + depth → 基础高斯 G₀
        # 对应论文 3.1 中 "Gaussian initializer"
        # MultiLayerInitializer: 在归一化(NDC)空间中，用纯几何规则计算高斯的初值
        # 位置: 反投影(不用相机内参，增强泛化)
        # 尺度: s = s₀ · depth (近小远大)
        # 颜色: 取降采样输入图的 RGB
        # 不透明度: 固定 0.5
        # 朝向: 单位四元数 [1,0,0,0]
        init_output = self.init_model(image, monodepth)

        # 步骤④: 特征提取 + 多分辨率解码
        # 对应论文 3.2 中 "Gaussian decoder" 的解码器部分
        # feature_input: concat(image, normalized_disparity) → 浅层编码(SkipConv)
        # encodings: f₁..f₄ 来自 SPN 编码器 → MultiresConvDecoder → 深层编码
        # FeatureFusionBlock2d: 融合浅层+深层 → 统一特征图
        image_features = self.feature_model(
            init_output.feature_input, encodings=monodepth_output.output_features
        )

        # 步骤⑤: 预测头 → ΔG (14通道修正量)
        # 对应论文 3.2 中 "Gaussian decoder" 的预测头部分
        # 注意这里的通道分配由 heads.py 决定:
        # geometry_prediction_head: 3(位置) 通道修正
        # texture_prediction_head: 3(尺度)+4(朝向)+3(颜色)+1(不透明度) = 11 通道修正
        # 合计 14 通道 × 2 层 × 768×768 ≈ 16.5M 个数值
        delta_values = self.prediction_head(image_features)

        # 步骤⑥: base + Δ → 最终高斯
        # 对应论文公式 3.2: G = γ( γ⁻¹(base) + η · Δ )
        # GaussianComposer 在各属性的"激活空间"中做加法：
        #   - 位置: NDC 坐标 x/y 直接加；z 在 softplus 逆空间加(保证正深度)
        #   - 尺度: sigmoid 控制相对初始尺度的缩放范围 [min_scale, max_scale]
        #   - 朝向: 四元数直接加
        #   - 颜色: sigmoid 空间加 + 转 linearRGB(正确 alpha 混合)
        #   - 不透明度: sigmoid 空间加
        # flatten: [B,C,L,H,W] → [B, N=2×768×768, C] (展平成约 120 万高斯)
        # global_scale: 撤销 init_model 内部的深度数值归一化；
        #               此时仍是相机归一坐标，predict_image() 后续会反投影到相机/世界坐标。
        gaussians = self.gaussian_composer(
            delta=delta_values,
            base_values=init_output.gaussian_base_values,
            global_scale=init_output.global_scale,
        )
        return gaussians

    def internal_resolution(self) -> int:
        """网络的内部处理分辨率（输入图缩放到的尺寸）。"""
        return self.monodepth_model.internal_resolution()

    @property
    def output_resolution(self) -> int:
        """高斯输出的分辨率（内部分辨率的一半 = 768）。"""
        return self.internal_resolution() // 2
