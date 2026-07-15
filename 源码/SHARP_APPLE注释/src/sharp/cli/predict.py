"""包含 `sharp predict` CLI 命令的实现。

=sharp predict 命令的功能:=
这是 SHARP 推理的入口命令，对应你跑过的:
  sharp predict -i <输入图> -o <输出目录> --no-render --device cuda

执行流程:
  1. 扫描输入目录，收集所有支持的图像文件
  2. 下载/加载模型权重(sharp_2572gikvuh.pt，约 2.62GB)
  3. 创建 RGBGaussianPredictor 并加载权重
  4. 逐张图像: 预处理 → 推理 → 后处理 → 存 ply

核心函数 predict_image() 实现了预处理和后处理逻辑:
  - 预处理: numpy → tensor, resize 1536×1536, 算 disparity_factor
  - 推理: predictor.forward()(NDC 空间的高斯)
  - 后处理: unproject_gaussians(NDC → 度量世界空间, 论文3.1最后一步)
  - 保存: save_ply(存为标准 3DGS .ply 文件)

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from sharp.models import (
    PredictorParams,
    RGBGaussianPredictor,
    create_predictor,
)
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    SceneMetaData,
    save_ply,
    unproject_gaussians,
)

from .render import render_gaussians

LOGGER = logging.getLogger(__name__)

# 默认模型权重下载地址(Apple CDN)
DEFAULT_MODEL_URL = (
    "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
)


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    help="指向单个图像或包含图像的文件夹的路径。",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="保存预测结果(高斯.ply 和渲染视频)的路径。",
    required=True,
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help=".pt 权重文件的路径。若未提供，则自动下载默认模型。",
    required=False,
)
@click.option(
    "--render/--no-render",
    "with_rendering",
    is_flag=True,
    default=False,
    help="是否渲染验证画面(需要 gsplat + CUDA toolkit 编译 CUDA 核)。",
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="运行设备: 'cpu', 'mps', 'cuda'(默认自动选择)",
)
@click.option(
    "-v", "--verbose", is_flag=True, help="启用调试日志。",
)
def predict_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    with_rendering: bool,
    device: str,
    verbose: bool,
):
    """从输入图像预测 3D 高斯(3D Gaussian Splatting 表达)。

    这是 SHARP 推理的命令行入口。
    输入: 普通 RGB 照片(jpg/png 等)
    输出: .ply 文件(3DGS 高斯点云，可以用 SuperSplat 等查看器打开)
    可选: 渲染的验证视频(.mp4，需要 CUDA)
    """
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    # -- 步骤1: 收集所有输入图像 --
    extensions = io.get_supported_image_extensions()
    image_paths = []
    if input_path.is_file():
        if input_path.suffix in extensions:
            image_paths = [input_path]
    else:
        for ext in extensions:
            image_paths.extend(list(input_path.glob(f"**/*{ext}")))

    if len(image_paths) == 0:
        LOGGER.info("未找到有效图像。输入路径: %s。", input_path)
        return

    LOGGER.info("处理 %d 个有效图像文件。", len(image_paths))

    # -- 步骤2: 选择设备 --
    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    LOGGER.info("使用设备: %s", device)

    if with_rendering and device != "cuda":
        LOGGER.warning(
            "gsplat 渲染仅支持 CUDA。渲染已禁用。",
        )
        with_rendering = False

    # -- 步骤3: 加载或下载权重 --
    if checkpoint_path is None:
        LOGGER.info(
            "未提供权重文件。正在从 %s 下载默认模型。", DEFAULT_MODEL_URL,
        )
        state_dict = torch.hub.load_state_dict_from_url(
            DEFAULT_MODEL_URL, progress=True,
        )
    else:
        LOGGER.info("正在从 %s 加载权重。", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True)

    # -- 步骤4: 创建预测器并加载权重 --
    # create_predictor(PredictorParams()): 使用默认参数创建完整模型
    # 包含: SPN编码器 + DPT深度解码器 + 高斯解码器 + 初始化器 + 合成器
    # 总参数量: ~702M(可训练 ~340M，但推理时全部冻结)
    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()   # 设为推理模式(禁用 dropout/batchnorm 等)
    gaussian_predictor.to(device)

    output_path.mkdir(exist_ok=True, parents=True)

    # -- 步骤5: 逐张图像处理 --
    for image_path in image_paths:
        LOGGER.info("正在处理: %s", image_path)

        # 5a: 加载图像，获取焦距
        # io.load_rgb 返回: RGB 数组(H,W,3)，alpha 通道(H,W)，焦距(像素)
        image, _, f_px = io.load_rgb(image_path)
        height, width = image.shape[:2]

        # 5b: 构造相机内参矩阵(用于后续 NDC→世界 反投影)
        intrinsics = torch.tensor(
            [
                [f_px, 0, (width - 1) / 2.0, 0],
                [0, f_px, (height - 1) / 2.0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=device,
            dtype=torch.float32,
        )

        # 5c: 核心推理: 图像 → 3D 高斯(NDC 空间)
        gaussians = predict_image(
            gaussian_predictor, image, f_px, torch.device(device),
        )

        # 5d: 保存为 .ply 文件
        LOGGER.info("正在保存 3DGS 到 %s", output_path)
        save_ply(
            gaussians, f_px, (height, width),
            output_path / f"{image_path.stem}.ply",
        )

        # 5e: 可选: 渲染验证轨迹
        if with_rendering:
            output_video_path = (
                output_path / image_path.stem
            ).with_suffix(".mp4")
            LOGGER.info(
                "正在渲染轨迹到 %s", output_video_path,
            )
            metadata = SceneMetaData(
                intrinsics[0, 0].item(), (width, height), "linearRGB",
            )
            render_gaussians(gaussians, metadata, output_video_path)


@torch.no_grad()  # 推理时不需要梯度计算
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
) -> Gaussians3D:
    """从单张图像预测 3D 高斯 —— 包含完整的预处理、推理、后处理。

    Args:
        predictor: 已加载权重的 RGBGaussianPredictor 模型。
        image: 输入图像(H×W×3, uint8, NumPy 数组)。
        f_px: 从 EXIF 中读取的焦距(像素)，若无则默认 30mm 等效。
        device: 运行设备。

    Returns:
        Gaussians3D: 度量空间中的 3D 高斯(世界坐标)，约 120 万个。

    处理流程(对应论文 Section 3.1):
        预处理: resize 1536×1536, 转 tensor, 算 disparity_factor
        推理:   predictor.forward() → NDC 空间高斯
        后处理: unproject_gaussians(NDC → 世界/度量空间)
    """
    # SHARP 内部处理分辨率: 1536×1536
    internal_shape = (1536, 1536)

    LOGGER.info("执行预处理。")

    # -- 预处理 --------------------------------------------------------------
    # numpy(H,W,C) → torch.float → permute(C,H,W) → 归一化 [0,1]
    image_pt = (
        torch.from_numpy(image.copy())
        .float()
        .to(device)
        .permute(2, 0, 1)
        / 255.0
    )
    _, height, width = image_pt.shape

    # disparity_factor: 用于"度量深度 → 归一化视差"的转换因子
    # disparity = f_px / (width × depth)
    # 这是将物理深度映射到网络内部归一化空间的关键参数
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    # 将输入图像缩放到内部处理分辨率 1536×1536
    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    # -- 推理 ----------------------------------------------------------------
    LOGGER.info("执行推理。")
    # predictor.forward() 返回 NDC 空间中的高斯
    # NDC(归一化设备坐标): x∈[-1,1], y∈[-1,1], z=逆深度
    # 不使用相机内参，生成的高斯在所有视场角下泛化更好
    gaussians_ndc = predictor(image_resized_pt, disparity_factor)

    # -- 后处理: NDC → 度量世界空间 ---------------------------------------
    LOGGER.info("执行后处理。")

    # 根据原始图像尺寸构造内参矩阵
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    # 对内参进行缩放，以匹配网络内部处理分辨率(1536)
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # 将高斯从 NDC 空间反投影到度量(世界)空间
    # unproject_gaussians:
    #   构造 NDC→世界 变换矩阵(含内参和外参)
    #   应用仿射变换到所有高斯(位置+协方差矩阵)
    #   外参设为单位矩阵(不做旋转平移)
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device),
        intrinsics_resized, internal_shape,
    )

    return gaussians
