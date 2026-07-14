"""定义将基础高斯(base)和修正量(delta)合成为最终高斯的模块。

=对应论文 Section 3.1，公式 3.2:=

GaussianComposer 实现了论文的核心合成公式:
  G_attr = γ_attr( γ⁻¹_attr(G₀,attr) + η_attr · ΔG_attr )

即：不是直接加，而是先将基础值映射到"激活空间"(逆激活函数 γ⁻¹)，
在该空间中加修正量，再从激活空间映射回物理空间(激活函数 γ)。
这样保证各种属性的物理约束(如尺度非负、颜色在 [0,1] 等)，
同时让网络在不受约束的空间中自由学习修正量。

五个属性各自的激活/合成方式:
  位置(mean):     NDC 坐标，x/y 直接加；z 在 softplus⁻¹ 空间加(保证正深度)
  尺度(scale):     在 sigmoid⁻¹ 空间加(保证缩放因子在 [min,max] 区间内)
  朝向(quaternion):直接加四元数
  颜色(color):     在 sigmoid/exp 空间加 + 转 linearRGB
  不透明度(opacity):在 sigmoid 空间加(保证在 (0,1) 区间)

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from sharp.models.initializer import GaussianBaseValues
from sharp.utils import math as math_utils
from sharp.utils.color_space import ColorSpace, sRGB2linearRGB
from sharp.utils.gaussians import Gaussians3D

from .params import DeltaFactor


# -- 尺度激活函数的辅助常量计算 ----------------------------------------------------
def _get_scale_activation_constant(max_scale: float, min_scale: float) -> tuple[float, float]:
    """计算尺度激活函数的两个常量 a 和 b。

    设计目标：当修正量 Δ=0 时，缩放因子 = 1（即不改变初始尺度），且梯度也为 1。
    这样训练初期网络输出 0 时，尺度就是初始化的几何估计，让训练更稳定。

    这本质上是将 sigmoid 的输出范围从 (0,1) 缩放到 (min_scale, max_scale)。
    """
    # 为保证 Δ=0 时 scale_factor=1 且梯度=1，需要这样设计 a 和 b
    constant_a = (max_scale - min_scale) / (1 - min_scale) / (max_scale - 1)
    constant_b = math_utils.inverse_sigmoid(
        torch.tensor((1.0 - min_scale) / (max_scale - min_scale))
    ).item()
    return constant_a, constant_b


class GaussianComposer(nn.Module):
    """将基础高斯值和预测修正量合成为最终 3D 高斯 —— 对应论文公式 3.2。

    不在原始物理空间中直接做加法，而是在各属性的"激活空间"中操作：
      物理值 → 逆激活(γ⁻¹) → 加修正量(η·Δ) → 激活(γ) → 物理值

    这样网络可以在无约束空间中自由输出修正量，而合成后的物理值天然满足约束。
    """

    color_activation_type: math_utils.ActivationType
    opacity_activation_type: math_utils.ActivationType

    def __init__(
        self,
        delta_factor: DeltaFactor,
        min_scale: float,
        max_scale: float,
        color_activation_type: math_utils.ActivationType,
        opacity_activation_type: math_utils.ActivationType,
        color_space: ColorSpace,
        base_scale_on_predicted_mean: bool,
        scale_factor: int = 1,
    ) -> None:
        """初始化高斯合成器。

        Args:
            delta_factor: 修正量的缩放因子(η)，按属性分别控制修正幅度。
                xy/z 通常很小(0.001)，让网络在小范围内微调位置。
            min_scale: 尺度缩放因子的最小值（相对于初始尺度）。
            max_scale: 尺度缩放因子的最大值（相对于初始尺度）。
                默认 [0,10]，即高斯可以缩小到 0 倍或膨胀到初始尺度的 10 倍。
            color_activation_type: 颜色使用的激活函数类型('sigmoid'/'exp'/'softplus')。
            opacity_activation_type: 不透明度使用的激活函数类型。
            color_space: 训练使用的色彩空间('linearRGB' 或 'sRGB')。
                linearRGB 用于正确的 alpha 混合；若是 sRGB 则在激活后转为 linear。
            scale_factor: 将修正量上采样的倍率（若修正量与基础值分辨率不同时使用）。
            base_scale_on_predicted_mean: 是否使用"预测均值"来更新基础尺度。
                开启时: 尺度会随预测的 z 偏移量调整 (z_new/z_old * base_scale)。
        """
        super().__init__()
        self.delta_factor = delta_factor
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.color_activation_type = color_activation_type
        self.opacity_activation_type = opacity_activation_type
        self.color_space = color_space
        self.scale_factor = scale_factor
        self.base_scale_on_predicted_mean = base_scale_on_predicted_mean

    def upsample_delta_value(self, delta: torch.Tensor, scale_factor: int = 1):
        """上采样修正量张量，使其与基础高斯值的分辨率匹配。

        当修正量的分辨率低于基础高斯值时（如 stride 不同），需要上采样对齐。
        将 [B, C, L, H, W] 的倒数两维放大 scale_factor 倍。
        """
        (
            batch_size,
            num_channels,
            num_layers,
            image_height,
            image_width,
        ) = delta.shape
        new_height = image_height * scale_factor
        new_width = image_width * scale_factor
        upsampled_delta = F.interpolate(
            delta.view(
                batch_size, num_channels * num_layers, image_height, image_width
            ),
            scale_factor=scale_factor,
        ).view(batch_size, num_channels, num_layers, new_height, new_width)
        return upsampled_delta

    def forward(
        self,
        delta: torch.Tensor,
        base_values: GaussianBaseValues,
        global_scale: torch.Tensor | None = None,
        flatten_output: bool = True,
    ) -> Gaussians3D:
        """将预测的修正量(ΔG)与基础高斯值(G₀)合成 —— 论文公式 3.2。

        Args:
            delta: 高斯解码器预测的修正量 ΔG。
                形状 [B, 14, 2, H', W']，14个通道:
                [0:3]=位置Δ, [3:6]=尺度Δ, [6:10]=朝向Δ, [10:13]=颜色Δ, [13]=不透明度Δ
            base_values: 高斯初始化器计算的基础值 G₀ (GaussianBaseValues)。
            global_scale: 全局缩放因子（来自深度归一化的还原因子）。
            flatten_output: 是否展平输出，将网格高斯变为点列表。

        Returns:
            合成的 3D 高斯(Gaussians3D)。
            若 flatten_output=True: 每个属性形状 [B, N, C]（N=2×768×768≈120万）。
            包含: mean_vectors(位置), singular_values(尺度对角线上值),
                  quaternions(朝向四元数), colors(RGB), opacities(不透明度)。
        """
        # 若修正量和基础值的分辨率不同(如 triplane head)，先上采样
        scale_factor = self.scale_factor
        actual_scale_factor = base_values.mean_x_ndc.shape[-1] // delta.shape[-1]
        if scale_factor != 1 and actual_scale_factor != 1:
            delta = self.upsample_delta_value(delta, scale_factor)

        # ---- ① 位置合成 -------------------------------------------------------
        # x,y 在 NDC 坐标中直接加偏移；z 在 softplus⁻¹(逆 softplus)空间中加，
        # 再经 softplus 回到深度空间，保证深度始终为正
        mean_vectors = self._forward_mean(base_values, delta)

        # ---- ② 尺度合成 -------------------------------------------------------
        # 若开启 base_scale_on_predicted_mean，基础尺度随 z 偏移量等比调整
        # (z_预测 / z_初始) * base_scale，保持"深度越远、覆盖面积越大"的尺度关系
        base_scales = (
            (base_values.scales * base_values.mean_inverse_z_ndc * mean_vectors[:, 2:3, ...])
            if self.base_scale_on_predicted_mean
            else base_values.scales
        )
        # 在对数-逆sigmoid空间中加修正量，经 sigmoid 缩放后乘到基础尺度上
        singular_values = self._scale_activation(
            base_scales,
            delta[:, 3:6],
            self.min_scale,
            self.max_scale,
        )

        # ---- ③ 朝向合成：四元数直接加 ------------------------------------------
        # 不需要激活空间变换——四元数本身无约束(渲染时归一化即可)
        quaternions = self._quaternion_activation(base_values.quaternions, delta[:, 6:10])

        # ---- ④ 颜色合成 -------------------------------------------------------
        # 在 sigmoid/exp 逆空间中加修正量，再映射回 [0,1] 颜色空间
        # 若目标色彩空间是 linearRGB，则额外做 sRGB→linear 转换
        colors = self._color_activation(base_values.colors, delta[:, 10:13])

        # ---- ⑤ 不透明度合成 ---------------------------------------------------
        # 在 sigmoid 逆空间中加修正量，保证结果在 (0,1) 区间
        opacities = self._opacity_activation(base_values.opacities, delta[:, 13])

        # ---- ⑥ 展平 ----------------------------------------------------------
        # [B, C, L, H, W] → [B, N=L*H*W, C]，即将网格高斯变为点列表
        # N = 2 × 768 × 768 = 1,179,648 个 3D 高斯
        if flatten_output:
            mean_vectors = mean_vectors.permute(0, 2, 3, 4, 1).flatten(1, 3)
            singular_values = singular_values.permute(0, 2, 3, 4, 1).flatten(1, 3)
            quaternions = quaternions.permute(0, 2, 3, 4, 1).flatten(1, 3)
            colors = colors.permute(0, 2, 3, 4, 1).flatten(1, 3)
            opacities = opacities.flatten(1, 3)

        # ---- ⑦ 撤销深度归一化的全局缩放 -----------------------------------------
        # 这一步恢复 init_model 内部为数值稳定而缩放过的深度/尺度。
        # 注意: 这里仍是相机归一坐标；predict_image() 之后还会用相机内参反投影到相机/世界坐标。
        if global_scale is not None:
            mean_vectors = global_scale[:, None, None] * mean_vectors
            singular_values = global_scale[:, None, None] * singular_values

        return Gaussians3D(
            mean_vectors=mean_vectors,
            singular_values=singular_values,
            quaternions=quaternions,
            colors=colors,
            opacities=opacities,
        )

    # -- 各属性的激活函数 -----------------------------------------------------------

    def _forward_mean(
        self, base_values: GaussianBaseValues, delta: torch.Tensor
    ) -> torch.Tensor:
        """位置(均值向量)的激活函数。

        输入的 base 第三维是逆深度 1/z；x,y 直接加偏移；
        逆深度在 softplus⁻¹ 空间中加偏移，再取倒数得到正深度 z。
        softplus 保证逆深度始终为正，从而深度也为正。
        """
        # 每个维度的修正因子 η
        delta_factor = torch.tensor(
            [self.delta_factor.xy, self.delta_factor.xy, self.delta_factor.z],
            device=delta.device,
        )[None, :, None, None, None]

        dtype = base_values.mean_x_ndc.dtype
        device = base_values.mean_x_ndc.device
        target_shape = (1, 3, 1, 1, 1)

        # 用掩码从分量组装 NDC 坐标向量
        mean_x_mask = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device).reshape(target_shape)
        mean_y_mask = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).reshape(target_shape)
        mean_z_mask = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device).reshape(target_shape)

        # 组装基础 NDC 坐标向量 [x, y, 1/z]
        mean_vectors_ndc = (
            base_values.mean_x_ndc.repeat(target_shape) * mean_x_mask
            + base_values.mean_y_ndc.repeat(target_shape) * mean_y_mask
            + base_values.mean_inverse_z_ndc.repeat(target_shape) * mean_z_mask
        )

        mean_vectors = self._mean_activation(mean_vectors_ndc, delta_factor * delta[:, :3])
        return mean_vectors

    def _mean_activation(
        self, base: torch.Tensor, learned_delta: torch.Tensor
    ) -> torch.Tensor:
        """均值激活函数的核心实现。

        x,y: 直接加 Δ
        z: 逆深度 a，在 softplus⁻¹ 空间中加 Δb，再经 softplus 变回深度
            inverse_zz = softplus( softplus⁻¹(a) + b )
            zz = 1 / inverse_zz

        这样做的好处：
        - softplus 输出始终 > 0，保证深度为正
        - 在逆深度空间操作更稳定（近处敏感、远处不敏感，和立体视觉一致）
        """
        # x,y: 直接加法
        xx = base[:, 0:1] + learned_delta[:, 0:1]
        yy = base[:, 1:2] + learned_delta[:, 1:2]

        # z(逆深度): softplus⁻¹ 空间中加法，再经 softplus 还原
        a = base[:, 2:3]
        b = learned_delta[:, 2:3]

        # 原公式: inverse_zz = softplus( softplus⁻¹(a) + b )
        inverse_zz = F.softplus(math_utils.inverse_softplus(a) + b)
        # 转回深度: zz = 1/(逆深度)，加 epsilon 防除零
        zz = 1.0 / (inverse_zz + 1e-3)

        # 组合: mean = [zz·xx, zz·yy, zz]
        # 注意 zz 乘回 x,y，是因为 NDC 中 x,y 是用归一化坐标表示的，
        # 而 3D 位置需要将 NDC 坐标"缩放"到对应深度
        mean_vectors = torch.cat([zz * xx, zz * yy, zz], dim=1)
        return mean_vectors

    def _scale_activation(
        self,
        base: torch.Tensor,
        learned_delta: torch.Tensor,
        min_scale: float,
        max_scale: float,
    ) -> torch.Tensor:
        """尺度的激活函数。

        基础尺度 × 缩放因子，缩放因子通过 sigmoid 约束在 [min_scale, max_scale] 内:
            scale_factor = (max - min) · sigmoid(a·η·Δ + b) + min
            output = base × scale_factor

        Δ=0 时 scale_factor=1(不变)，使训练初期稳定。
        """
        constant_a, constant_b = _get_scale_activation_constant(max_scale, min_scale)
        scale_factor = (max_scale - min_scale) * torch.sigmoid(
            constant_a * self.delta_factor.scale * learned_delta + constant_b
        ) + min_scale
        return base * scale_factor

    def _quaternion_activation(
        self, base: torch.Tensor, learned_delta: torch.Tensor
    ) -> torch.Tensor:
        """朝向(四元数)的激活函数：直接加法。

        不对四元数做归一化——渲染时会自行处理。
        约束宽松是因为四元数本身可以表示任意旋转，不存在物理范围限制。
        """
        return base + self.delta_factor.quaternion * learned_delta

    def _color_activation(
        self, base: torch.Tensor, learned_delta: torch.Tensor
    ) -> torch.Tensor:
        """颜色的激活函数。

        在激活空间(sigmoid/exp/softplus 的逆)中加法，再经激活函数回到 [0,1]。
        若目标是 linearRGB，则额外做 sRGB→linear 转换。
        """
        # 对需要有限域的激活函数，钳制基础值到有效范围
        if self.color_activation_type == "sigmoid":
            base = torch.clamp(base, min=0.01, max=0.99)
        elif self.color_activation_type in ("exp", "softplus"):
            base = torch.clamp(base, min=0.01)

        activation = math_utils.create_activation_pair(self.color_activation_type)
        colors: torch.Tensor = activation.forward(
            activation.inverse(base) + self.delta_factor.color * learned_delta
        )
        # linearRGB 色彩空间需要做 gamma 反变换，用于正确的 alpha 混合
        if self.color_space == "linearRGB":
            colors = sRGB2linearRGB(colors)
        return colors

    def _opacity_activation(
        self, base: torch.Tensor, learned_delta: torch.Tensor
    ) -> torch.Tensor:
        """不透明度的激活函数。

        在 sigmoid⁻¹ 空间中加法，保证输出在 (0,1) 区间。
        Δ=0 时不透明度 = 初始值 0.5，训练中网络可增大或减小。
        """
        activation = math_utils.create_activation_pair(self.opacity_activation_type)
        return activation.forward(
            activation.inverse(base) + self.delta_factor.opacity * learned_delta
        )
