"""包含骨干网络的参数定义。

=对应论文 Section 3.1, 3.2:=
这里定义了 SHARP 所有可配置参数，各 dataclass 按模块组织：
  PredictorParams    → 顶层参数，包含所有子模块参数
  MonodepthParams    → 深度骨干网络(Depth Pro SPN + DPT 解码器)
  GaussianDecoderParams → 高斯解码器(DPT 改，预测 ΔG)
  InitializerParams  → 高斯初始化器(不可学习的几何规则)
  AlignmentParams    → 深度对齐模块(训练时用的小 U-Net)
  DeltaFactor        → 各属性修正量的学习率缩放因子η

默认值对应 ICLR 2026 论文中描述的配置：2层高斯、768x768网格、
DINOv2 ViT-L/16 骨干、1536x1536 内部处理分辨率。

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import dataclasses
from typing import Literal

import sharp.utils.math as math_utils
from sharp.models.blocks import NormLayerName, UpsamplingMode
from sharp.models.presets import ViTPreset
from sharp.utils.color_space import ColorSpace

# -- 类型别名 --------------------------------------------------------------------
# 深度解码器的五层特征维度 (从粗到细)
# 默认 [256, 256, 256, 256, 256] — 对应 DPT 解码器的 5 个分辨率级别
DimsDecoder = tuple[int, int, int, int, int]

# DPT 图像编码器类型
# "skip_conv": 普通跳跃卷积
# "skip_conv_kernel2": kernel_size=2 的跳跃卷积(用于 stride=2 时)
DPTImageEncoderType = Literal["skip_conv", "skip_conv_kernel2"]


# -- 初始化选项枚举 ----------------------------------------------------------------

ColorInitOption = Literal[
    "none",          # 初始化为灰色(0.5, 0.5, 0.5)
    "first_layer",   # 第一层取输入图的降采样 RGB，其余层为灰色
    "all_layers",    # 所有层都取输入图的降采样 RGB
]

DepthInitOption = Literal[
    "surface_min",    # 用深度图的最小池化值，选最近表面
    "surface_max",    # 用深度图的最大池化值，选最远表面
    "base_depth",     # 用 base_depth(默认10米)作为固定深度平面
    "linear_disparity", # 从 base_depth 到无穷远的均匀视差分布
]


# -- 深度对齐参数 ----------------------------------------------------------------

@dataclasses.dataclass
class AlignmentParams:
    """深度对齐模块的参数 —— 对应论文 Section 3.1 (Depth Adjustment)。

    推理时该模块被替换为恒等函数，不影响推理性能。
    """

    kernel_size: int = 16
    stride: int = 1
    frozen: bool = False                # 是否冻结对齐模块（使其不参与训练）

    # 以下参数仅用于 LearnedAlignment (基于 U-Net 的局部缩放图估计)
    steps: int = 4                      # U-Net 的下采样/上采样步数
    activation_type: math_utils.ActivationType = "exp"
    depth_decoder_features: bool = False  # 是否输入深度解码器的中间特征
    base_width: int = 16                 # U-Net 基础通道宽度


# -- 修正量缩放因子 ----------------------------------------------------------------

@dataclasses.dataclass
class DeltaFactor:
    """各属性修正量(Δ)的缩放因子 η —— 对应论文公式 3.2 中的 η_attr。

    这些因子在"激活空间"中乘法应用，起到"选择性降低学习率"的效果。
    η 越小，修正量的有效学习率越低，网络需要更谨慎地调整该属性。

    为什么 xy/z 的因子(0.001)远小于其他属性(0.1~1.0):
    位置的基础值来自深度反投影，已经相当准确，只需微调；
    颜色和尺度的初始值较粗糙，需要更大的修正幅度。
    """

    xy: float = 0.001          # 位置 x,y 的修正因子(NDC 空间)
    z: float = 0.001           # 位置 z(深度) 的修正因子
    color: float = 0.1         # 颜色的修正因子(linearRGB 推荐 0.1，sRGB 推荐 1.0)
    opacity: float = 1.0       # 不透明度的修正因子
    scale: float = 1.0         # 尺度的修正因子
    quaternion: float = 1.0    # 朝向(四元数)的修正因子


# -- 初始化器参数 ----------------------------------------------------------------

@dataclasses.dataclass
class InitializerParams:
    """高斯初始化器的参数 —— 对应论文 Section 3.1 (Gaussian Initializer)。

    控制如何从 RGB 图和深度图算出基础高斯 G₀。
    这是一个不可学习的模块(纯几何/规则)，所以参数都是超参数而非可训练权重。
    """

    # ---- 通用参数 ----
    scale_factor: float = 1.0            # 高斯的初始尺度乘以此因子
    disparity_factor: float = 1.0        # 逆深度 → 视差的转换因子
    stride: int = 2                      # 输出的下采样倍率(输入1536→输出768)

    # ---- 仅用于 MultiLayerInitializer 的参数 ----
    num_layers: int = 2                  # 高斯层数(论文用 2:可见表面层 + 潜在遮挡/视角相关层)
    first_layer_depth_option: DepthInitOption = "surface_min"  # 第一层深度初始化方式
    rest_layer_depth_option: DepthInitOption = "surface_min"   # 其余层深度初始化方式
    color_option: ColorInitOption = "all_layers"   # 颜色初始化方式
    base_depth: float = 10.0             # 固定深度平面的深度值(米)
    feature_input_stop_grad: bool = False  # 是否阻断特征输入的反向传播梯度
    normalize_depth: bool = True          # 是否将深度归一化到 [1.0, 100.0]

    # ---- 仅用于 InpaintingInitializer 的参数(当前未使用) ----
    output_inpainted_layer_only: bool = False
    set_uninpainted_opacity_to_zero: bool = False
    concat_inpainting_mask: bool = False


# -- 单目深度网络参数 -------------------------------------------------------------

@dataclasses.dataclass
class MonodepthParams:
    """单目深度网络的参数 —— 对应论文 Section 3.2 中 Feature Encoder + Depth Decoder。

    控制 Depth Pro 骨干网络(基于 DINOv2 ViT-L/16 + SPN 编码器 + DPT 解码器)。
    冻结/解冻设置实现论文描述的"选择性微调"策略。
    """

    # DINOv2 预训练模型的配置(块编码器和图像编码器都用 DINOv2 ViT-L/16)
    patch_encoder_preset: ViTPreset = "dinov2l16_384"     # 块编码器：384×384 分辨率的 ViT-L/16
    image_encoder_preset: ViTPreset = "dinov2l16_384"     # 图像编码器：384×384 分辨率的 ViT-L/16

    checkpoint_uri: str | None = None       # 预留字段；公开推理代码通过整体 SHARP checkpoint 加载权重

    # -- 训练时冻结/解冻控制(对应论文 3.2 的选择性微调) --
    # 推理时这些全为 False(不训练)，训练时的推荐设置:
    #   patch_encoder: False(冻结) — 保留 DINOv2 的通用局部特征
    #   image_encoder: True(解冻)  — 让全局理解适配视图合成任务
    #   decoder: True(解冻)        — 让深度解码器适配视图合成目标
    #   head: True(解冻)           — 适配两层深度输出
    #   norm_layers: False(冻结)   — 保留预训练归一化层的统计量
    unfreeze_patch_encoder: bool = False    # 是否训练块编码器(默认冻结)
    unfreeze_image_encoder: bool = False    # 是否训练图像编码器(默认冻结)
    unfreeze_decoder: bool = False          # 是否训练深度解码器(默认冻结)
    unfreeze_head: bool = False             # 是否训练深度输出头(默认冻结)
    unfreeze_norm_layers: bool = False      # 是否解冻所有归一化层(默认冻结)

    grad_checkpointing: bool = False        # 是否启用梯度检查点(省显存、慢训练)
    use_patch_overlap: bool = True          # SPN 中切块是否重叠(默认有重叠)

    # DPT 解码器各层的特征通道数(5层，从粗到细)
    dims_decoder: DimsDecoder = (256, 256, 256, 256, 256)


# -- 深度网络特征适配器参数 -------------------------------------------------------

@dataclasses.dataclass
class MonodepthAdaptorParams:
    """单目深度网络特征适配器的参数。

    控制从深度网络往外输出哪些特征供后续模块(如高斯解码器)使用。
    """

    encoder_features: bool = True     # 是否输出编码器的多尺度特征 f₁..f₄
    decoder_features: bool = False    # 是否输出解码器的中间特征


# -- 高斯解码器参数 -------------------------------------------------------------

@dataclasses.dataclass
class GaussianDecoderParams:
    """高斯解码器的参数 —— 对应论文 Section 3.2 (Gaussian Decoder)。

    控制从编码器特征 + RGB/两层归一化视差输入预测高斯修正量 ΔG 的网络结构。
    使用 MultiresConvDecoder(DPT 改) + 双头(texture/geometry)架构。
    """

    dim_in: int = 5                    # 输入通道数(RGB 3 + 两层归一化视差 2)
    dim_out: int = 32                  # 中间(解码器)输出通道数

    norm_type: NormLayerName = "group_norm"  # 归一化层类型
    norm_num_groups: int = 8                 # GroupNorm 的分组数

    stride: int = 2                    # 输出的下采样倍率(相对输入)
    patch_encoder_preset: ViTPreset = "dinov2l16_384"
    image_encoder_preset: ViTPreset = "dinov2l16_384"

    dims_decoder: DimsDecoder = (128, 128, 128, 128, 128)  # 解码器各层特征维度(较小，因输出仅14通道)

    use_depth_input: bool = True       # 是否在输入中包含深度(默认包含)
    grad_checkpointing: bool = False   # 是否启用梯度检查点

    upsampling_mode: UpsamplingMode = "transposed_conv"  # 解码器上采样方式

    image_encoder_type: DPTImageEncoderType = "skip_conv_kernel2"  # 特征编码器类型


# -- 预测器顶层参数 --------------------------------------------------------------

@dataclasses.dataclass
class PredictorParams:
    """SHARP 预测器的顶层参数 —— 包含所有子模块的参数。

    它组合了 5 个子模块的参数和 1 个合成器的参数,
    构成 SHARP 模型的完整配置。训练和推理都从这里统一读取。
    对应论文 Section 3.1 图3 中的所有模型组件。
    """

    # ---- 子模块参数 ----
    initializer: InitializerParams = dataclasses.field(
        default_factory=InitializerParams
    )  # 高斯初始化器(不可学习)
    monodepth: MonodepthParams = dataclasses.field(
        default_factory=MonodepthParams
    )  # 深度骨干(SPN + DPT)
    monodepth_adaptor: MonodepthAdaptorParams = dataclasses.field(
        default_factory=MonodepthAdaptorParams
    )  # 特征适配器(控制哪些特征输出)
    gaussian_decoder: GaussianDecoderParams = dataclasses.field(
        default_factory=GaussianDecoderParams
    )  # 高斯解码器(预测 ΔG)
    depth_alignment: AlignmentParams = dataclasses.field(
        default_factory=AlignmentParams
    )  # 深度对齐(仅训练时使用)

    # ---- 合成器参数 ----
    delta_factor: DeltaFactor = dataclasses.field(
        default_factory=DeltaFactor
    )  # 各属性修正量的缩放因子 η

    max_scale: float = 10.0           # 高斯尺度缩放因子的最大值(相对初始尺度)
    min_scale: float = 0.0            # 高斯尺度缩放因子的最小值(相对初始尺度)

    norm_type: NormLayerName = "group_norm"    # 归一化层类型
    norm_num_groups: int = 8                   # GroupNorm 分组数

    use_predicted_mean: bool = False   # 是否用预测均值而非基础均值来采样 triplane 特征

    # 激活函数类型
    color_activation_type: math_utils.ActivationType = "sigmoid"     # 颜色: sigmoid(输出 [0,1])
    opacity_activation_type: math_utils.ActivationType = "sigmoid"   # 不透明度: sigmoid(输出 [0,1])

    color_space: ColorSpace = "linearRGB"   # 渲染器的色彩空间(linearRGB 用于正确 alpha 混合)

    low_pass_filter_eps: float = 1e-2       # 低通滤波的小值，防止退化的 splat

    num_monodepth_layers: int = 2           # 单目深度模型输出的深度层数(论文设为 2)
    sorting_monodepth: bool = False         # 是否对两层深度排序(前景/背景)

    base_scale_on_predicted_mean: bool = True  # 是否随预测 z 偏移调整基础尺度
