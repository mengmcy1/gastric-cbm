"""包含从 RGB 图像和预测深度初始化高斯所需的模块。

=对应论文 Section 3.1 (Gaussian Initializer):=

MultiLayerInitializer 按照纯几何规则(不可学习、不参与训练)，
从 RGB 图和深度图计算出基础高斯 G₀:
  - 位置(mean): 用 NDC 坐标 + 逆深度反投影(故意不用相机内参，增强泛化)
  - 尺度(scale): s = s₀ × depth(深度成正比，近小远大)
  - 颜色(color): 默认所有层都取降采样输入图的 RGB
  - 朝向(quaternion): 单位四元数 [1,0,0,0](正对相机)
  - 不透明度(opacity): 固定 0.5

所有属性都在归一化空间中定义，后续高斯解码器在这些初值基础上预测修正量 ΔG。

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from .params import ColorInitOption, DepthInitOption, InitializerParams


def create_initializer(params: InitializerParams) -> nn.Module:
    """根据参数创建初始化器。"""
    return MultiLayerInitializer(
        num_layers=params.num_layers,
        stride=params.stride,
        base_depth=params.base_depth,
        scale_factor=params.scale_factor,
        disparity_factor=params.disparity_factor,
        color_option=params.color_option,
        first_layer_depth_option=params.first_layer_depth_option,
        rest_layer_depth_option=params.rest_layer_depth_option,
        normalize_depth=params.normalize_depth,
        feature_input_stop_grad=params.feature_input_stop_grad,
    )


class GaussianBaseValues(NamedTuple):
    """高斯预测器的基础值 —— 对应论文 Equation 3.2 中的 G₀。

    在 NDC(归一化设备坐标)空间中表示高斯初值:
    - x, y ∈ [-1, 1](归一化图像平面坐标)
    - z = 逆深度(正值，近大远小)

    不使用相机内参——NDC 空间使高斯与图像视场角无关，增强泛化能力。
    """

    mean_x_ndc: torch.Tensor        # NDC 空间中的 x 坐标，shape [B, 1, L, H', W']
    mean_y_ndc: torch.Tensor        # NDC 空间中的 y 坐标，shape [B, 1, L, H', W']
    mean_inverse_z_ndc: torch.Tensor  # NDC 空间中的逆深度(1/z)，shape [B, 1, L, H', W']

    scales: torch.Tensor            # 初始尺度(s₀ × depth)，shape [B, 1, L, H', W']
    quaternions: torch.Tensor       # 初始朝向(全为单位四元数)，shape [B, 4, 1, H', W']
    colors: torch.Tensor            # 初始颜色(取自降采样图像)，shape [B, 3, L, H', W']
    opacities: torch.Tensor         # 初始不透明度(固定 0.5)，shape [B, 1, L, H', W']


class InitializerOutput(NamedTuple):
    """初始化器的输出——包含基础高斯值和后续高斯解码器用的特征输入。"""

    gaussian_base_values: GaussianBaseValues  # 基础高斯值 G₀

    feature_input: torch.Tensor  # 输入到高斯解码器的特征(RGB+视差拼接)

    global_scale: torch.Tensor | None = None  # 全局缩放因子(来自深度归一化的还原因子)


class MultiLayerInitializer(nn.Module):
    """用多层表示初始化高斯 —— 对应论文 Section 3.1 (Gaussian Initializer)。

    输出的张量形状为:
        batch_size × dim × num_layers × height' × width'
    其中 dim 表示属性的维度(位置=3/尺度=3/朝向=4/颜色=3/不透明度=1)。

    每层的深度可以独立设置(前景层用 surface_min 取最近点，
    遮挡层用 base_depth 或 surface_min 取相应的深度)。

    为什么初始化器不可学习:
    深度反投影已经给出了"合理的"3D位置估计，让网络学"修正量"
    比学"从零预测"容易得多——这是一个"残差学习"的设计。
    """

    def __init__(
        self,
        num_layers: int,
        stride: int,
        base_depth: float,
        scale_factor: float,
        disparity_factor: float,
        color_option: ColorInitOption = "first_layer",
        first_layer_depth_option: DepthInitOption = "surface_min",
        rest_layer_depth_option: DepthInitOption = "surface_min",
        normalize_depth: bool = True,
        feature_input_stop_grad: bool = True,
    ) -> None:
        """初始化多层高斯初始化器。

        Args:
            stride: 输出特征图的下采样倍率(默认2，1536→768)。
            base_depth: 第一层的深度值(前景层之后)。若不使用前景层则为所有层的基深。
            scale_factor: 高斯的初始尺度乘以此因子。
            disparity_factor: 将逆深度转换为视差(disparity)的因子。
            num_layers: 高斯层数(论文设为2:前景层+遮挡层)。
            color_option: 多层高斯的颜色初始化选项。
            first_layer_depth_option: 第一层高斯的深度初始化方式(默认 "surface_min")。
            rest_layer_depth_option: 其余层高斯的深度初始化方式(默认 "surface_min")。
            normalize_depth: 是否将深度归一化到 [depth_min, depth_max] 范围。
            feature_input_stop_grad: 是否阻断特征输入的反向传播梯度。
                True 时: 初始化器的输出不给梯度，防止高斯解码器"依赖初始化而不是真正学 3D 结构"。
        """
        super().__init__()
        self.num_layers = num_layers
        self.stride = stride
        self.base_depth = base_depth
        self.scale_factor = scale_factor
        self.disparity_factor = disparity_factor
        self.color_option = color_option
        self.first_layer_depth_option = first_layer_depth_option
        self.rest_layer_depth_option = rest_layer_depth_option
        self.normalize_depth = normalize_depth
        self.feature_input_stop_grad = feature_input_stop_grad

    def prepare_feature_input(
        self, image: torch.Tensor, depth: torch.Tensor
    ) -> torch.Tensor:
        """准备输入高斯解码器的特征。

        将 RGB 图和归一化视差拼接为特征输入。
        SHARP 默认是两层深度，因此通道数为 3 + 2 = 5。
        若 feature_input_stop_grad=True，则阻断图像和深度的梯度，
        迫使高斯解码器不依赖初始化器的输出，而是依赖编码器的中间特征来学习 3D 结构。
        """
        if self.feature_input_stop_grad:
            # 阻断梯度：init 不可学习 → 高斯解码器必须从编码器特征中学
            image = image.detach()
            depth = depth.detach()

        # 深度 → 视差(1/深度)，然后与 RGB 拼接
        normalized_disparity = self.disparity_factor / depth
        features_in = torch.cat([image, normalized_disparity], dim=1)

        # 归一化到 [-1, 1]（常见的网络输入预处理）
        features_in = 2.0 * features_in - 1.0
        return features_in

    def forward(
        self, image: torch.Tensor, depth: torch.Tensor
    ) -> InitializerOutput:
        """从 RGB 图和深度图构建基础高斯值并准备特征输入。

        Args:
            image: 输入图像(已 resize 到 1536×1536)。
            depth: 来自深度网络的深度图(度量单位，可能为多层)。

        Returns:
            InitializerOutput: 包含 base_values, feature_input, global_scale。
        """
        image = image.contiguous()
        depth = depth.contiguous()
        device = depth.device
        batch_size, _, image_height, image_width = depth.shape

        # 输出分辨率 = 输入分辨率 / stride(1536 / 2 = 768)
        base_height, base_width = (
            image_height // self.stride,
            image_width // self.stride,
        )

        # ---- 步骤A: 深度归一化(使训练数值稳定) ----
        # 将深度缩放到 [1.0, 100.0] 范围
        global_scale: torch.Tensor | None = None
        if self.normalize_depth:
            depth, depth_factor = _rescale_depth(depth)
            global_scale = 1.0 / depth_factor  # 后续用于撤销深度数值归一化

        # ---- 步骤B: 创建多层的视差(逆深度)层 ----
        # 各层的深度可以不同: 第一层用 surface_min(最近表面)，
        # 其余层用表面深度或固定深度平面

        # 第一层视差: 取对应深度
        if self.first_layer_depth_option == "surface_min":
            # surface_min: 取最小池化(视差取最大池化) = 选最近表面
            first_disparity = _create_surface_layer(depth[:, 0:1], "min")
        elif self.first_layer_depth_option == "surface_max":
            # surface_max: 取最大池化(视差取最小池化) = 选最远表面
            first_disparity = _create_surface_layer(depth[:, 0:1], "max")
        elif self.first_layer_depth_option in ("base_depth", "linear_disparity"):
            # 用固定或等距的深度平面(1/base_depth 到 0)
            first_disparity = _create_disparity_layers()
        else:
            raise ValueError(
                f"未知的深度初始化选项: {self.first_layer_depth_option}."
            )

        if self.num_layers == 1:
            # 单层: 只有前景层
            disparity = first_disparity
        else:
            # 多层(默认2层): 可见表面层 + 其余潜在遮挡/视角相关层
            # 其余层使用深度图的余下层或特定初始化方式
            following_depth = (
                depth if depth.shape[1] == 1 else depth[:, 1:]
            )
            if self.rest_layer_depth_option == "surface_min":
                following_disparity = _create_surface_layer(following_depth, "min")
            elif self.rest_layer_depth_option == "surface_max":
                following_disparity = _create_surface_layer(following_depth, "max")
            elif self.rest_layer_depth_option == "base_depth":
                following_disparity = torch.cat(
                    [
                        _create_disparity_layers()
                        for _ in range(self.num_layers - 1)
                    ],
                    dim=2,
                )
            elif self.rest_layer_depth_option == "linear_disparity":
                following_disparity = _create_disparity_layers(self.num_layers - 1)
            else:
                raise ValueError(
                    f"未知的深度初始化选项: {self.rest_layer_depth_option}."
                )

            # 拼接第一层和其余层的视差
            disparity = torch.cat([first_disparity, following_disparity], dim=2)

        # ---- 步骤C: 计算基础高斯值 ----

        # C1. 位置: NDC 坐标网格
        # x ∈ [-1,1], y ∈ [-1,1]，网格等距分布
        base_x_ndc, base_y_ndc = _create_base_xy(depth, self.stride, self.num_layers)

        # C2. 尺度: s = s₀ × depth(近小远大)
        # disparity_scale_factor 将视差的数值映射到合适的尺度范围
        disparity_scale_factor = (
            2 * self.scale_factor * self.stride / float(image_width)
        )
        base_scales = _create_base_scale(disparity, disparity_scale_factor)

        # C3. 朝向: 单位四元数 [1,0,0,0]——所有高斯初始朝向均为正对相机
        base_quaternions = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        base_quaternions = base_quaternions[None, :, None, None, None]

        # C4. 不透明度: 固定值
        # 初始化为 min(1/num_layers, 0.5) 确保初始通过率约为:
        #     1/e ≈ (1 - 1/num_layers)^num_layers
        # 即各层合起来的整体透明度与层数无关
        base_opacities = torch.tensor(
            [min(1.0 / self.num_layers, 0.5)], device=device
        )

        # C5. 颜色: 从降采样输入图取 RGB
        # 形状 [B, 3, num_layers, H', W']，默认初始化为灰色 0.5
        base_colors = torch.empty(
            batch_size, 3, self.num_layers, base_height, base_width,
            device=device,
        ).fill_(0.5)

        if self.color_option == "none":
            pass  # 保持灰色
        elif self.color_option == "first_layer":
            # 仅第一层取图像颜色(avg_pool 降采样)
            base_colors[:, :, 0] = torch.nn.functional.avg_pool2d(
                image, self.stride, self.stride
            )
        elif self.color_option == "all_layers":
            # 所有层都取图像颜色
            temp = torch.nn.functional.avg_pool2d(
                image, self.stride, self.stride
            )
            base_colors = temp[:, :, None, :, :].repeat(
                1, 1, self.num_layers, 1, 1
            )
        else:
            raise ValueError(
                f"未知的颜色初始化选项: {self.color_option}."
            )

        # ---- 步骤D: 准备高斯解码器的输入特征 ----
        features_in = self.prepare_feature_input(image, depth)

        # 打包基础高斯值
        base_gaussians = GaussianBaseValues(
            mean_x_ndc=base_x_ndc,
            mean_y_ndc=base_y_ndc,
            mean_inverse_z_ndc=disparity,
            scales=base_scales,
            quaternions=base_quaternions,
            colors=base_colors,
            opacities=base_opacities,
        )

        return InitializerOutput(
            gaussian_base_values=base_gaussians,
            feature_input=features_in,
            global_scale=global_scale,
        )


# -- 辅助函数: 在 NDC 空间中创建基础坐标 ------------------------------------------

def _create_base_xy(
    depth: torch.Tensor, stride: int, num_layers: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """在 NDC 空间中为高斯创建基础 x 和 y 坐标。

    坐标从 -1 到 1，均匀覆盖整个图像平面。
    形状: [B, 1, num_layers, H', W']
    """
    device = depth.device
    batch_size, _, image_height, image_width = depth.shape
    xx = torch.arange(0.5 * stride, image_width, stride, device=device)
    yy = torch.arange(0.5 * stride, image_height, stride, device=device)
    # 归一化到 NDC [-1, 1]
    xx = 2 * xx / image_width - 1.0
    yy = 2 * yy / image_height - 1.0

    xx, yy = torch.meshgrid(xx, yy, indexing="xy")
    base_x_ndc = xx[None, None, None].repeat(
        batch_size, 1, num_layers, 1, 1
    )
    base_y_ndc = yy[None, None, None].repeat(
        batch_size, 1, num_layers, 1, 1
    )

    return base_x_ndc, base_y_ndc


def _create_base_scale(
    disparity: torch.Tensor, disparity_scale_factor: float,
) -> torch.Tensor:
    """为高斯创建基础尺度: s = s₀ × depth。

    尺度与深度成正比——远处的高斯更大(覆盖更多像素)，
    近处的高斯更小(覆盖更少像素)。这和 3DGS 的"近小远大"一致。

    用逆视差(=深度)缩放，因为视差=1/深度。
    """
    inverse_disparity = torch.ones_like(disparity) / disparity
    base_scales = inverse_disparity * disparity_scale_factor
    return base_scales


def _rescale_depth(
    depth: torch.Tensor, depth_min: float = 1.0, depth_max: float = 1e2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将深度图张量缩放到稳定数值范围。

    操作: 找出 batch 中每张图的最小深度，将所有深度按比例缩放到 depth_min。
    例如: 最小深度 0.5m → factor = 1.0/0.5 = 2.0，所有深度 ×2.0。
    最后 clamp 到 depth_max。
    """
    current_depth_min = depth.flatten(depth.ndim - 3).min(dim=-1).values
    depth_factor = depth_min / (current_depth_min + 1e-6)
    depth = (depth * depth_factor[..., None, None, None]).clamp(max=depth_max)
    return depth, depth_factor


# 根据层数的关键词索引: num_layers → 视差列表
_DISPARITY_LAYERS_CACHE: dict[int, torch.Tensor] = {}


def _create_disparity_layers(num_layers: int = 1) -> torch.Tensor:
    """创建多层视差(深度/逆深度平面)。

    返回形状 [B, 1, num_layers, H', W']。
    第一层视差 = 1/base_depth (最近的固定平面)，
    后续层视差线性递减到 0(无穷远)。
    """
    if num_layers not in _DISPARITY_LAYERS_CACHE:
        disparity = torch.linspace(1.0, 0.0, num_layers + 1)[:-1]
        _DISPARITY_LAYERS_CACHE[num_layers] = disparity
    return _DISPARITY_LAYERS_CACHE[num_layers][None, None, :, None, None]


def _create_surface_layer(
    depth: torch.Tensor, depth_pooling_mode: str,
) -> torch.Tensor:
    """创建"表面层"的视差。

    根据深度池化模式:
    - "min": 取最小深度(视差最大化，选最近点)
    - "max": 取最大深度(视差最小化，选最远点)

    这用于从单目深度图中选出"最前表面"的深度，作为前景高斯的初始深度。
    """
    disparity = 1.0 / depth
    if depth_pooling_mode == "min":
        # min pooling on depth = max pooling on disparity(选最近点的视差)
        disparity = torch.max_pool2d(disparity, stride, stride)
    elif depth_pooling_mode == "max":
        # max pooling on depth(选远点): 视差取负
        disparity = -torch.max_pool2d(-disparity, stride, stride)
    else:
        raise ValueError(f"无效的深度池化模式 {depth_pooling_mode}.")

    return disparity[:, :, None, :, :]
