"""包含 3D 高斯的基本数据结构和功能。

=核心数据结构与操作:=

Gaussians3D: 3D 高斯的五种属性(位置、尺度、朝向、颜色、不透明度)
SceneMetaData: 场景元数据(焦距、分辨率、色彩空间)

主要的函数:
  save_ply():     将 Gaussians3D 保存为 .ply 文件(标准 3DGS 格式)
  load_ply():     从 .ply 文件加载 Gaussians3D
  unproject_gaussians(): 将高斯从 NDC 空间变换到度量世界空间
  apply_transform():    对高斯应用仿射变换(位置+协方差矩阵)
  compose/decompose_covariance_matrices(): 四元数/尺度 ↔ 3×3 协方差矩阵互转
  convert_spherical_harmonics_to_rgb / convert_rgb_to_spherical_harmonics:
    DC 球谐(0阶) ↔ RGB 颜色互转

=.ply 文件格式说明:=
SHARP 输出的 .ply 文件遵循 3DGS 标准格式:
  vertex: x,y,z (位置) + f_dc_0,1,2 (球谐DC颜色) + opacity
        + scale_0,1,2 (尺度logits) + rot_0,1,2,3 (四元数)
  extrinsic: 外参矩阵(4×4)
  intrinsic: 内参矩阵(3×3)
  image_size: 原始图像尺寸(w,h)
  frame: 帧数和每帧粒子数
  disparity: 视差的十分位数(用于归一化)
  color_space: 色彩空间索引
  version: 版本号

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from sharp.utils import color_space as cs_utils
from sharp.utils import linalg

LOGGER = logging.getLogger(__name__)

BackgroundColor = Literal["black", "white", "random_color", "random_pixel"]


class Gaussians3D(NamedTuple):
    """表示一组 3D 高斯(椭球体的集合)。

    每个属性都是形状 [B, N, C] 的张量:
      mean_vectors:   [B, N, 3] — 中心位置(xyz)
      singular_values:[B, N, 3] — 尺度(sx, sy, sz, 对角线上元素)
      quaternions:    [B, N, 4] — 朝向(四元数)
      colors:         [B, N, 3] — 颜色(RGB, linearRGB 空间)
      opacities:      [B, N, 1] — 不透明度(0~1)

    渲染时: 将椭球按深度排序 → 逐个 splat 到像素 → alpha 混合 → 最终图像
    """

    mean_vectors: torch.Tensor
    singular_values: torch.Tensor
    quaternions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor

    def to(self, device: torch.device) -> "Gaussians3D":
        """将高斯移到指定设备。"""
        return Gaussians3D(
            mean_vectors=self.mean_vectors.to(device),
            singular_values=self.singular_values.to(device),
            quaternions=self.quaternions.to(device),
            colors=self.colors.to(device),
            opacities=self.opacities.to(device),
        )


class SceneMetaData(NamedTuple):
    """高斯场景的元数据。"""

    focal_length_px: float         # 焦距(像素)
    resolution_px: tuple[int, int]  # 图像分辨率(宽, 高)
    color_space: cs_utils.ColorSpace  # 色彩空间("linearRGB" 或 "sRGB")


# -- NDC ↔ 世界空间变换 ----------------------------------------------------------

def get_unprojection_matrix(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
) -> torch.Tensor:
    """计算将高斯从 NDC 空间反投影到欧几里得(世界)空间的变换矩阵。

    变换链: NDC ← 像素坐标 ← 相机坐标 ← 世界坐标
    反投影: NDC → 世界 = inv(NDC_matrix @ intrinsics @ extrinsics)

    NDC 矩阵将像素坐标映射到 [-1,1] 的归一化空间:
      x_ndc = 2 * u / width - 1
      y_ndc = 2 * v / height - 1
      z_ndc = z (不变)

    Args:
        extrinsics: 相机外参(4×4)，世界→相机。
        intrinsics: 相机内参(4×4)，相机→像素。
        image_shape: 输入图像的 (宽, 高)。

    Returns:
        4×4 矩阵，将高斯从 NDC 空间转换到世界空间。
    """
    device = intrinsics.device
    image_width, image_height = image_shape

    # NDC 矩阵: 将 OpenCV 像素坐标 (0..width, 0..height) 映射到
    # NDC 坐标 (-1..1, -1..1)，左上角为 (-1, -1)，右下角为 (1, 1)
    #
    # 注意到 ndc_matrix @ intrinsics 通常仅做:
    #   x 轴: 缩放 2*focal_length/image_width
    #   y 轴: 缩放 2*focal_length/image_height
    ndc_matrix = torch.tensor(
        [
            [2.0 / image_width, 0.0, -1.0, 0.0],
            [0.0, 2.0 / image_height, -1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
    )
    return torch.linalg.inv(ndc_matrix @ intrinsics @ extrinsics)


def unproject_gaussians(
    gaussians_ndc: Gaussians3D,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
) -> Gaussians3D:
    """将高斯从 NDC 空间反投影到世界坐标。

    这是预测后处理的关键步骤 —— 对应论文 Section 3.1 最后一步：
    网络在归一化空间中预测高斯，然后用相机内参和外参将其
    "反投影"到有物理尺度的世界坐标。

    Args:
        gaussians_ndc: NDC 空间中的高斯。
        extrinsics: 相机外参矩阵(推理时通常为单位矩阵)。
        intrinsics: 相机内参矩阵。
        image_shape: 图像(宽, 高)。

    Returns:
        世界坐标中的高斯(度量单位，米)。
    """
    unprojection_matrix = get_unprojection_matrix(
        extrinsics, intrinsics, image_shape,
    )
    gaussians = apply_transform(gaussians_ndc, unprojection_matrix[:3])
    return gaussians


def apply_transform(
    gaussians: Gaussians3D, transform: torch.Tensor,
) -> Gaussians3D:
    """对 3D 高斯应用仿射变换(旋转+缩放+平移)。

    变换同时作用于位置(3D 点)和协方差矩阵(旋转+缩放):
      - 新位置 = R @ old_position + t
      - 新协方差 = R @ old_covariance @ R^T

    注意: 此操作不可微。

    Args:
        gaussians: 要变换的高斯。
        transform: 仿射变换，形状 3×4(前三列 = 线性变换 R，第四列 = 平移 t)。

    Returns:
        变换后的高斯。
    """
    transform_linear = transform[..., :3, :3]   # 3×3 线性部分
    transform_offset = transform[..., :3, 3]    # 3×1 平移部分

    # 位置: 线性变换 + 平移
    mean_vectors = gaussians.mean_vectors @ transform_linear.T + transform_offset

    # 协方差矩阵: R @ cov @ R^T
    covariance_matrices = compose_covariance_matrices(
        gaussians.quaternions, gaussians.singular_values,
    )
    covariance_matrices = (
        transform_linear @ covariance_matrices
        @ transform_linear.transpose(-1, -2)
    )

    # 分解回四元数 + 尺度
    quaternions, singular_values = decompose_covariance_matrices(
        covariance_matrices,
    )

    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=gaussians.colors,     # 颜色不受仿射变换影响
        opacities=gaussians.opacities,
    )


# -- 协方差矩阵 ↔ 四元数+尺度 互转 -----------------------------------------------

def decompose_covariance_matrices(
    covariance_matrices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将 3×3 协方差矩阵分解为四元数(朝向)和奇异值(尺度)。

    用 SVD 分解: cov = U @ Σ² @ U^T
    - 旋转 R = U(由四元数表示)
    - 尺度 = sqrt(diag(Σ))(对角线上)

    如果 SVD 给出反射矩阵(det < 0)，则翻转最后一列修正为旋转矩阵。
    这一步在 float64 下计算，确保数值稳定性。

    注意: 此操作不可微。
    """
    device = covariance_matrices.device
    dtype = covariance_matrices.dtype

    # 转为 fp64 避免数值误差(协方差矩阵的条件数可能很大)
    covariance_matrices = (
        covariance_matrices.detach().cpu().to(torch.float64)
    )
    # SVD: cov = U @ Σ² @ V^T，其中 U 是旋转，Σ 是尺度对角阵
    rotations, singular_values_2, _ = torch.linalg.svd(
        covariance_matrices,
    )

    # 修正: 如果 U 是反射(det < 0)而不是旋转，翻转最后一列
    batch_idx, gaussian_idx = torch.where(
        torch.linalg.det(rotations) < 0
    )
    num_reflections = len(gaussian_idx)
    if num_reflections > 0:
        LOGGER.warning(
            "从 SVD 中获得了 %d 个反射矩阵。正在将其翻转为旋转矩阵。",
            num_reflections,
        )
        # 翻转最后一列，将反射转为旋转(det 从 -1 变为 +1)
        rotations[batch_idx, gaussian_idx, :, -1] *= -1

    # 旋转矩阵 → 四元数
    quaternions = linalg.quaternions_from_rotation_matrices(rotations)
    quaternions = quaternions.to(dtype=dtype, device=device)

    # 奇异值平方根 = 尺度
    singular_values = singular_values_2.sqrt().to(
        dtype=dtype, device=device,
    )
    return quaternions, singular_values


def compose_covariance_matrices(
    quaternions: torch.Tensor, singular_values: torch.Tensor,
) -> torch.Tensor:
    """将四元数(朝向)和奇异值(尺度)组合为 3×3 协方差矩阵。

    cov = R @ Σ² @ R^T
    其中 R 来自四元数，Σ 来自尺度对角阵。
    """
    device = quaternions.device
    rotations = linalg.rotation_matrices_from_quaternions(quaternions)
    # Σ² = diag(sx², sy², sz²)
    diagonal_matrix = torch.eye(3, device=device) * singular_values[
        ..., :, None
    ]
    return rotations @ diagonal_matrix.square() @ rotations.transpose(-1, -2)


# -- 球谐函数 ↔ RGB 颜色转换 ---------------------------------------------------

def convert_spherical_harmonics_to_rgb(sh0: torch.Tensor) -> torch.Tensor:
    """将 0 阶球谐系数(DC 分量)转换为 RGB 颜色。

    参照: https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    DC 分量: Y₀₀ = sqrt(1/(4π))
    颜色: c = sh0 * coeff + 0.5
    """
    coeff_degree0 = np.sqrt(1.0 / (4.0 * np.pi))
    return sh0 * coeff_degree0 + 0.5


def convert_rgb_to_spherical_harmonics(rgb: torch.Tensor) -> torch.Tensor:
    """将 RGB 颜色转换为 0 阶球谐系数(DC 分量)。

    SHARP 不使用高阶球谐函数(论文说这会大幅增加输出体积)，
    只存最基础的 DC 分量(即恒定颜色，不随观察方向变化)。
    """
    coeff_degree0 = np.sqrt(1.0 / (4.0 * np.pi))
    return (rgb - 0.5) / coeff_degree0


# -- PLY 文件读写 ---------------------------------------------------------------

def load_ply(path: Path) -> tuple[Gaussians3D, SceneMetaData]:
    """从 .ply 文件加载 3D 高斯和场景元数据。

    解析 SHARP 标准的 .ply 文件，读取顶点属性(位置、颜色、尺度、旋转、不透明度)
    和附加元数据(内参、外参、色彩空间、图像尺寸、视差范围)。
    """
    plydata = PlyData.read(path)

    vertices = next(
        filter(lambda x: x.name == "vertex", plydata.elements)
    )

    # 验证所需属性都存在
    properties = ["x", "y", "z"]
    properties.extend([f"f_dc_{i}" for i in range(3)])
    properties.extend([f"scale_{i}" for i in range(3)])
    properties.extend([f"rot_{i}" for i in range(3)])

    for prop in properties:
        if prop not in vertices:
            raise KeyError(
                f".ply 文件不兼容: 顶点元素中缺少属性 {prop}。"
            )

    # 解析位置
    mean_vectors = np.stack(
        (
            np.asarray(vertices["x"]),
            np.asarray(vertices["y"]),
            np.asarray(vertices["z"]),
        ),
        axis=1,
    )

    # 解析尺度(存储为 logits)
    scale_logits = np.stack(
        (
            np.asarray(vertices["scale_0"]),
            np.asarray(vertices["scale_1"]),
            np.asarray(vertices["scale_2"]),
        ),
        axis=1,
    )

    # 解析朝向(四元数)
    quaternions = np.stack(
        (
            np.asarray(vertices["rot_0"]),
            np.asarray(vertices["rot_1"]),
            np.asarray(vertices["rot_2"]),
            np.asarray(vertices["rot_3"]),
        ),
        axis=1,
    )

    # 解析球谐DC系数 → RGB
    spherical_harmonics_deg0 = np.stack(
        (
            np.asarray(vertices["f_dc_0"]),
            np.asarray(vertices["f_dc_1"]),
            np.asarray(vertices["f_dc_2"]),
        ),
        axis=1,
    )
    colors = convert_spherical_harmonics_to_rgb(spherical_harmonics_deg0)

    # 解析不透明度(存储为 logits)
    opacity_logits = np.asarray(vertices["opacity"])[..., None]

    # 解析附加元数据
    supplement_elements = [
        element
        for element in plydata.elements
        if element.name != "vertex"
    ]
    supplement_data: dict[str, Any] = {}
    supplement_keys = [
        "extrinsic", "intrinsic", "color_space", "image_size",
    ]

    for element in supplement_elements:
        for key in supplement_keys:
            if key not in supplement_data and key in element:
                supplement_data[key] = np.asarray(element[key])

    # 解析内参和图像尺寸
    if "intrinsic" in supplement_data:
        intrinsics_data = supplement_data["intrinsic"]

        # 兼容旧格式: image_size 包含在 intrinsic 元素中
        if "image_size" not in supplement_data:
            if len(intrinsics_data) != 4:
                raise ValueError(
                    "期望旧格式内参长度为 4(含图像尺寸)，"
                    f"但实际收到 {len(intrinsics_data)}。"
                )
            focal_length_px = (
                intrinsics_data[0], intrinsics_data[1],
            )
            width = int(intrinsics_data[2])
            height = int(intrinsics_data[3])
        else:
            if len(intrinsics_data) != 9:
                raise ValueError(
                    "期望 9 个元素的内参，"
                    f"但实际收到 {len(intrinsics_data)} 个。"
                )
            intrinsics_matrix = intrinsics_data.reshape((3, 3))
            focal_length_px = (
                intrinsics_matrix[0, 0], intrinsics_matrix[1, 1],
            )
            image_size_data = supplement_data["image_size"]
            width = image_size_data[0]
            height = image_size_data[1]
    else:
        # 默认 VGA 分辨率
        focal_length_px = (512, 512)
        width = 640
        height = 480

    # 解析外参
    extrinsics_data = supplement_data.get(
        "extrinsic", np.eye(4).flatten(),
    )
    extrinsics_matrix = np.eye(4)

    # 兼容旧格式: 外参存 12 个元素
    if len(extrinsics_data) == 12:
        extrinsics_matrix[:3] = extrinsics_data.reshape((3, 4))
        extrinsics_matrix[:3, :3] = (
            extrinsics_matrix[:3, :3].copy().T
        )
    elif len(extrinsics_data) == 16:
        extrinsics_matrix[:] = extrinsics_data.reshape((4, 4))
    else:
        raise ValueError(
            f"无法识别的外参矩阵形状 {len(extrinsics_data)}"
        )

    # 解析色彩空间
    color_space_index = supplement_data.get("color_space", 1)
    color_space = cs_utils.decode_color_space(color_space_index)
    colors = torch.from_numpy(colors).view(1, -1, 3).float()

    # 如果是 sRGB，转换为 linearRGB 以正确 alpha 混合
    if color_space == "sRGB":
        colors = (
            cs_utils.sRGB2linearRGB(colors.flatten(0, 1))
            .view(1, -1, 3)
        )
        color_space = "linearRGB"

    mean_vectors = torch.from_numpy(mean_vectors).view(1, -1, 3).float()
    quaternions = torch.from_numpy(quaternions).view(1, -1, 4).float()
    singular_values = (
        torch.exp(torch.from_numpy(scale_logits).view(1, -1, 3))
    ).float()
    opacities = (
        torch.sigmoid(torch.from_numpy(opacity_logits).view(1, -1))
    ).float()

    gaussians = Gaussians3D(
        mean_vectors=mean_vectors,
        quaternions=quaternions,
        singular_values=singular_values,
        opacities=opacities,
        colors=colors,
    )
    metadata = SceneMetaData(
        focal_length_px[0], (width, height), color_space,
    )
    return gaussians, metadata


@torch.no_grad()
def save_ply(
    gaussians: Gaussians3D,
    f_px: float,
    image_shape: tuple[int, int],
    path: Path,
) -> PlyData:
    """将预测的 Gaussians3D 保存为 .ply 文件。

    保存前的关键处理:
      1. 属性转换: 原始值 → .ply 存储格式
         - 位置: 直接存 xyz
         - 颜色: linearRGB → sRGB → 0阶球谐DC系数
         - 尺度: 取 log 存为 logits(exp 可还原)
         - 不透明度: sigmoid⁻¹ 存为 logits(sigmoid 可还原)
         - 朝向: 直接存四元数
      2. 附加元数据: 内参、外参、图像尺寸、视差范围、色彩空间

    注意(sRGB/linearRGB 转换):
      SHARP 预测的高斯颜色是 linearRGB(用于正确 alpha 混合)。
      但 public renderers(如 SuperSplat)通常按 sRGB 渲染。
      为使兼容，保存时将颜色转换为 sRGB → 球谐系数。
      SHARP 自带的 renderer 会自行处理转换。

    Args:
        gaussians: 要保存的 3D 高斯(世界/度量空间)。
        f_px: 焦距(像素)。
        image_shape: 原始图像的(高, 宽)。
        path: 输出 .ply 路径。

    Returns:
        PlyData 对象。
    """

    def _inverse_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
        """sigmoid 的逆函数: logit(x) = log(x/(1-x))。"""
        return torch.log(tensor / (1.0 - tensor))

    xyz = gaussians.mean_vectors.flatten(0, 1)
    scale_logits = torch.log(gaussians.singular_values).flatten(0, 1)
    quaternions = gaussians.quaternions.flatten(0, 1)

    # SHARP 以 sRGB 图像为输入，预测 linearRGB 的高斯输出。
    # SHARP 渲染器可混合 linearRGB 高斯并转换回 sRGB 以获得最佳显示质量。
    #
    # 但公共渲染器没有 linear→sRGB 转换。
    # 如果它们直接渲染 linearRGB 高斯，输出将偏暗(缺少 Gamma 校正)。
    #
    # 为兼容公共渲染器，导出时强制将 linearRGB 转为 sRGB，再转为球谐系数。
    # - SHARP 渲染器仍会正确处理转换。
    # - 公共渲染器将 sRGB 视作 linearRGB 也能基本正常工作。
    #   (尽管最佳效果仍需应用完整的 sRGB→linear→blend→sRGB 管线)
    colors = convert_rgb_to_spherical_harmonics(
        cs_utils.linearRGB2sRGB(gaussians.colors.flatten(0, 1)),
    )
    color_space_index = cs_utils.encode_color_space("sRGB")

    # 存不透明度 logits
    opacity_logits = (
        _inverse_sigmoid(gaussians.opacities)
        .flatten(0, 1)
        .unsqueeze(-1)
    )

    # 拼接所有属性: xyz(3) + sh_dc(3) + opacity(1) + scale(3) + rot(4) = 14
    attributes = torch.cat(
        (xyz, colors, opacity_logits, scale_logits, quaternions),
        dim=1,
    )

    # 定义 PLY 顶点数据类型(全部 float32)
    dtype_full = [
        (attribute, "f4")
        for attribute in ["x", "y", "z"]
        + [f"f_dc_{i}" for i in range(3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    ]

    num_gaussians = len(xyz)
    elements = np.empty(num_gaussians, dtype=dtype_full)
    elements[:] = list(map(tuple, attributes.detach().cpu().numpy()))
    vertex_elements = PlyElement.describe(elements, "vertex")

    # ---- 附加元数据 ----
    image_height, image_width = image_shape

    # 图像尺寸
    dtype_image_size = [("image_size", "u4")]
    image_size_array = np.empty(2, dtype=dtype_image_size)
    image_size_array[:] = np.array([image_width, image_height])
    image_size_element = PlyElement.describe(
        image_size_array, "image_size",
    )

    # 相机内参(3×3 矩阵展平为 9 个元素)
    dtype_intrinsic = [("intrinsic", "f4")]
    intrinsic_array = np.empty(9, dtype=dtype_intrinsic)
    intrinsic = np.array(
        [
            f_px, 0, image_width * 0.5,
            0, f_px, image_height * 0.5,
            0, 0, 1,
        ]
    )
    intrinsic_array[:] = intrinsic.flatten()
    intrinsic_element = PlyElement.describe(
        intrinsic_array, "intrinsic",
    )

    # 相机外参(默认单位矩阵，4×4 展平)
    dtype_extrinsic = [("extrinsic", "f4")]
    extrinsic_array = np.empty(16, dtype=dtype_extrinsic)
    extrinsic_array[:] = np.eye(4).flatten()
    extrinsic_element = PlyElement.describe(
        extrinsic_array, "extrinsic",
    )

    # 帧数和每帧粒子数(兼容多帧格式)
    dtype_frames = [("frame", "i4")]
    frame_array = np.empty(2, dtype=dtype_frames)
    frame_array[:] = np.array([1, num_gaussians], dtype=np.int32)
    frame_element = PlyElement.describe(frame_array, "frame")

    # 视差范围(十分位数)，用于渲染时的归一化
    dtype_disparity = [("disparity", "f4")]
    disparity_array = np.empty(2, dtype=dtype_disparity)

    disparity = 1.0 / gaussians.mean_vectors[0, ..., -1]
    quantiles = (
        torch.quantile(
            disparity,
            q=torch.tensor([0.1, 0.9], device=disparity.device),
        )
        .float()
        .cpu()
        .numpy()
    )
    disparity_array[:] = quantiles
    disparity_element = PlyElement.describe(
        disparity_array, "disparity",
    )

    # 色彩空间
    dtype_color_space = [("color_space", "u1")]
    color_space_array = np.empty(1, dtype=dtype_color_space)
    color_space_array[:] = np.array([color_space_index]).flatten()
    color_space_element = PlyElement.describe(
        color_space_array, "color_space",
    )

    # 版本号
    dtype_version = [("version", "u1")]
    version_array = np.empty(3, dtype=dtype_version)
    version_array[:] = np.array([1, 5, 0], dtype=np.uint8).flatten()
    version_element = PlyElement.describe(version_array, "version")

    plydata = PlyData(
        [
            vertex_elements,
            extrinsic_element,
            intrinsic_element,
            image_size_element,
            frame_element,
            disparity_element,
            color_space_element,
            version_element,
        ]
    )

    plydata.write(path)
    return plydata
