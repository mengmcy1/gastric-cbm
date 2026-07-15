"""包含直接预测高斯属性修正量 ΔG 的输出头。

=对应论文 Section 3.2 (Gaussian Decoder):=

Gaussian decoder 先生成两路特征:
  - geometry_features: 主要用于预测位置修正量
  - texture_features: 用于预测尺度、朝向、颜色和不透明度修正量

本文件的 DirectPredictionHead 才是真正把特征变成 ΔG 张量的地方。
输出通道顺序与 GaussianComposer 保持一致:
  [0:3]   位置 mean delta
  [3:6]   尺度 scale delta
  [6:10]  朝向 quaternion delta
  [10:13] 颜色 color delta
  [13]    不透明度 opacity delta

注意: 这里的"geometry/texture"是源码中的特征分支命名，不等同于论文里
所有几何属性都走 geometry 分支。源码实际让 position 单独走 geometry 分支，
其余 11 个属性走 texture 分支。

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import torch
from torch import nn

from .gaussian_decoder import ImageFeatures


class DirectPredictionHead(nn.Module):
    """用 1×1 卷积将特征解码为高斯属性修正量 ΔG。"""

    def __init__(self, feature_dim: int, num_layers: int) -> None:
        """初始化直接预测头。

        Args:
            feature_dim: 输入特征通道数。
            num_layers: 要预测的高斯层数，SHARP 默认是 2。
        """
        super().__init__()
        self.num_layers = num_layers

        # 14 = 3位置 + 3尺度 + 4四元数 + 3颜色 + 1不透明度。
        # 但源码把通道拆成:
        #   geometry_prediction_head: 只预测位置 delta，3 × num_layers
        #   texture_prediction_head: 预测其余 11 个属性，(14-3) × num_layers
        self.geometry_prediction_head = nn.Conv2d(feature_dim, 3 * num_layers, 1)
        self.geometry_prediction_head.weight.data.zero_()
        assert self.geometry_prediction_head.bias is not None
        self.geometry_prediction_head.bias.data.zero_()

        self.texture_prediction_head = nn.Conv2d(feature_dim, (14 - 3) * num_layers, 1)
        self.texture_prediction_head.weight.data.zero_()
        assert self.texture_prediction_head.bias is not None
        self.texture_prediction_head.bias.data.zero_()

    def forward(self, image_features: ImageFeatures) -> torch.Tensor:
        """预测 3D 高斯的属性修正量 ΔG。

        Args:
            image_features: Gaussian decoder 输出的两路特征。

        Returns:
            delta_values: 形状 [B, 14, num_layers, H, W]。
        """
        delta_values_geometry = self.geometry_prediction_head(image_features.geometry_features)
        delta_values_texture = self.texture_prediction_head(image_features.texture_features)
        delta_values_geometry = delta_values_geometry.unflatten(1, (3, self.num_layers))
        delta_values_texture = delta_values_texture.unflatten(1, (14 - 3, self.num_layers))
        delta_values = torch.cat([delta_values_geometry, delta_values_texture], dim=1)
        return delta_values
