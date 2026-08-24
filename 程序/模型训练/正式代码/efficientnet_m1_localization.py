#!/usr/bin/env python3
"""EfficientNet-B0 M1：加载M0并用癌图bbox进行CenterNet式辅助定位。"""

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.models import efficientnet_b0
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as tvf


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from train_utils import (  # noqa: E402
    AUGMENTATION_CONFIG,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RandomDownsampleUpsample,
    RandomJPEGCompression,
    compute_metrics,
    file_sha256,
    format_metrics,
    git_snapshot,
    json_ready,
    patient_prediction_frame,
    seed_everything,
    seed_worker,
    select_screening_threshold,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT / "数据整理记录/图像裁剪"
    / "胃早癌概念提取训练集0804_预处理_v1"
    / "08_M1辅助定位清单_20260810"
    / "m1_balanced_keep_primary_1to1p3_split_seed42.csv"
)
DEFAULT_M0_ROOT = (
    PROJECT_ROOT / "结果/M0平衡_0804/正式验证集筛选"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/M1辅助定位_0804"
GRID_SIZE = 14
IMAGE_SIZE = 224
LABEL_SMOOTHING = 0.10


# -----------------------------------------------------------------------------
# 配置与冻结输入：只读取已验收的Keep清单，并沿用对应种子的M0权重。
# -----------------------------------------------------------------------------
def parse_args():
    """定义M1输入、两阶段训练、损失权重和输出控制参数。

    Returns:
        argparse.Namespace: 命令行参数；路径参数已转换为``Path``，数值参数
        保持``int``或``float``，开关参数为``bool``。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--m0-checkpoint", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--joint-epochs", type=int, default=20)
    parser.add_argument("--warmup-lr", type=float, default=1e-3)
    parser.add_argument("--joint-lr", type=float, default=1e-4)
    parser.add_argument(
        "--warmup-only",
        action="store_true",
        help="只训练冻结M0 Encoder/分类头的定位头，不进入joint阶段。",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-loc", type=float, default=1.0)
    parser.add_argument("--lambda-size", type=float, default=0.1)
    parser.add_argument("--lambda-offset", type=float, default=1.0)
    parser.add_argument("--classification-noninferiority", type=float, default=0.005)
    parser.add_argument("--crop-min-bbox-retention", type=float, default=0.80)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=12,
        help="joint阶段val患者AUC连续未创新高的容忍epoch数。",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-units", type=int, default=3)
    parser.add_argument(
        "--evaluate-test", action="store_true",
        help="显式解锁内部test；M1模型选择阶段不要使用。",
    )
    return parser.parse_args()


def default_m0_checkpoint(seed):
    """按随机种子定位对应的来源内平衡M0 EfficientNet最佳权重。

    Args:
        seed (int): M0正式实验随机种子，当前应为42、202或503。

    Returns:
        Path: 对应M0 ``efficientnet_b0_debiased_best.pth``路径。
    """
    return (
        DEFAULT_M0_ROOT / f"m0_balanced_keep_efficientnet_b0_seed{seed}"
        / "efficientnet_b0_debiased_best.pth"
    )


def load_manifest(path, image_root, debug, debug_units, seed):
    """读取并校验冻结M1清单，调试时按split和标签抽取患者子集。

    Args:
        path (Path): 含固定``train/val/test``、标签和bbox字段的M1 CSV。
        image_root (Path): ``image_relpath``所依附的项目图像根目录。
        debug (bool): 是否按患者抽取小样本，而不是使用完整清单。
        debug_units (int): debug时每个split、每个标签最多抽取的患者数。
        seed (int): debug患者抽样的随机种子。

    Returns:
        pandas.DataFrame: 通过路径、标签、split和定位监督校验的图片级清单。

    Raises:
        ValueError: 字段、标签、split、患者归属或bbox监督不符合冻结口径。
        FileNotFoundError: 清单引用的处理图不存在。
    """
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"patient_id": str})
    required = {
        "image_relpath", "patient_id", "label", "split", "branch",
        "localization_supervision", "bbox_valid", "bbox_x1_norm",
        "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"M1 manifest缺少字段: {sorted(missing)}")
    if set(frame.split.unique()) != {"train", "val", "test"}:
        raise ValueError("M1 manifest必须包含train/val/test")
    if not frame.branch.eq("keep").all():
        raise ValueError("M1正式入口只接受Keep清单")
    if frame.groupby("patient_id").split.nunique().gt(1).any():
        raise ValueError("患者跨split")
    if frame.groupby("patient_id").label.nunique().gt(1).any():
        raise ValueError("患者跨标签")
    positive = frame.label.eq(1)
    if not frame.loc[positive, "localization_supervision"].eq(1).all():
        raise ValueError("癌图定位监督不完整")
    if not frame.loc[positive, "bbox_valid"].astype(str).str.lower().isin(
        {"true", "1"}
    ).all():
        raise ValueError("癌图存在无效bbox")
    if frame.loc[~positive, "localization_supervision"].ne(0).any():
        raise ValueError("M1非癌图不得计算定位损失")
    missing_paths = [
        value for value in frame.image_relpath.astype(str)
        if not (image_root / value).is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(f"缺失{len(missing_paths)}张图: {missing_paths[0]}")
    if debug:
        selected = []
        for split, split_frame in frame.groupby("split", sort=False):
            patients = split_frame[["patient_id", "label"]].drop_duplicates()
            sampled = pd.concat([
                group.sample(min(debug_units, len(group)), random_state=seed)
                for _, group in patients.groupby("label", sort=True)
            ])
            chosen = split_frame.loc[split_frame.patient_id.isin(sampled.patient_id)]
            selected.append(chosen)
            print(f"[DEBUG] {split}: {chosen.patient_id.nunique()}人/{len(chosen)}张")
        frame = pd.concat(selected, ignore_index=True)
    return frame.reset_index(drop=True)


# -----------------------------------------------------------------------------
# 数据与变换：所有几何增强必须同步更新bbox，非癌图不伪造定位框。
# -----------------------------------------------------------------------------
class BBoxAwareTransform:
    """对图像和癌灶框同步执行A0增强，并保证框主体不被随机裁掉。

    Args:
        training (bool): ``True``启用随机增强，``False``只做确定性resize。
        min_bbox_retention (float): 随机裁剪后至少保留的原bbox面积比例，范围
        ``(0, 1]``。

    Outputs:
        调用实例返回``(image_tensor, bbox_tensor)``；图像形状为``[3,224,224]``，
        bbox为归一化``XYXY``、形状``[4]``。非癌图bbox返回全零占位，但由
        ``valid_box=False``屏蔽定位监督。
    """

    def __init__(self, training, min_bbox_retention=0.80):
        """初始化训练/评估变换及最小病灶框保留比例。

        Args:
            training (bool): 是否构建随机训练增强流程。
            min_bbox_retention (float): 癌图候选裁剪最小bbox面积保留率。

        Returns:
            None: 变换组件保存为实例状态。
        """
        self.training = training
        self.min_bbox_retention = min_bbox_retention
        self.downsample = RandomDownsampleUpsample(
            tuple(AUGMENTATION_CONFIG["downsample_scale"])
        )
        self.jpeg = RandomJPEGCompression(tuple(AUGMENTATION_CONFIG["jpeg_quality"]))
        self.blur = v2.GaussianBlur(
            kernel_size=5,
            sigma=tuple(AUGMENTATION_CONFIG["gaussian_blur_sigma"]),
        )
        self.color_jitter = v2.ColorJitter(
            brightness=AUGMENTATION_CONFIG["color_jitter_brightness"],
            contrast=AUGMENTATION_CONFIG["color_jitter_contrast"],
            saturation=AUGMENTATION_CONFIG["color_jitter_saturation"],
        )
        self.to_image = v2.ToImage()
        self.to_float = v2.ToDtype(torch.float32, scale=True)
        self.normalize = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    @staticmethod
    def make_boxes(box_norm, canvas_size):
        """把归一化XYXY坐标转换为带画布信息的torchvision框对象。

        Args:
            box_norm (tuple[float, float, float, float]): ``[0,1]``范围的
            ``(x1,y1,x2,y2)``。
            canvas_size (tuple[int, int]): 画布``(height,width)``，单位为像素。

        Returns:
            tv_tensors.BoundingBoxes: 形状``[1,4]``、像素坐标制的XYXY框。
        """
        height, width = canvas_size
        x1, y1, x2, y2 = box_norm
        return tv_tensors.BoundingBoxes(
            [[x1 * width, y1 * height, x2 * width, y2 * height]],
            format="XYXY",
            canvas_size=canvas_size,
        )

    def sample_bbox_aware_crop(self, image, boxes):
        """采样同时保留病灶中心和足够框面积的随机裁剪窗口。

        Args:
            image (torch.Tensor): 形状``[C,H,W]``的未标准化图像张量。
            boxes (tv_tensors.BoundingBoxes | None): 癌图像素框；非癌图为``None``。

        Returns:
            tuple[int, int, int, int]: ``(top,left,height,width)``裁剪窗口；
            20次均不满足约束时返回完整画布。
        """
        fallback = (0, 0, image.shape[-2], image.shape[-1])
        for _ in range(20):
            top, left, height, width = RandomResizedCrop.get_params(
                image,
                scale=tuple(AUGMENTATION_CONFIG["random_resized_crop_scale"]),
                ratio=tuple(AUGMENTATION_CONFIG["random_resized_crop_ratio"]),
            )
            if boxes is None:
                return top, left, height, width
            x1, y1, x2, y2 = boxes[0].tolist()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            inter_width = max(0.0, min(x2, left + width) - max(x1, left))
            inter_height = max(0.0, min(y2, top + height) - max(y1, top))
            retention = inter_width * inter_height / max((x2 - x1) * (y2 - y1), 1e-8)
            center_inside = (
                left <= center_x <= left + width
                and top <= center_y <= top + height
            )
            if center_inside and retention >= self.min_bbox_retention:
                return top, left, height, width
        return fallback

    def __call__(self, image, box_norm=None):
        """同步变换单张图及可选bbox，返回网络输入和最终框。

        Args:
            image (PIL.Image.Image): RGB胃镜图像。
            box_norm (tuple[float, float, float, float] | None): 原图归一化
            XYXY癌灶框；非癌图传``None``。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ImageNet标准化图像``[3,224,224]``
            和归一化XYXY框``[4]``；无框时第二项为全零张量。

        Raises:
            RuntimeError: 几何增强后bbox退化为零面积。
        """
        if self.training:
            image = image.resize((256, 256), Image.Resampling.BILINEAR)
            if random.random() < AUGMENTATION_CONFIG["downsample_probability"]:
                image = self.downsample(image)
            if random.random() < AUGMENTATION_CONFIG["jpeg_probability"]:
                image = self.jpeg(image)
            image = self.to_image(image)
            boxes = self.make_boxes(box_norm, (256, 256)) if box_norm is not None else None
            if random.random() < AUGMENTATION_CONFIG["gaussian_blur_probability"]:
                image = self.blur(image)
            top, left, height, width = self.sample_bbox_aware_crop(image, boxes)
            image = tvf.resized_crop(
                image, top, left, height, width, [IMAGE_SIZE, IMAGE_SIZE],
                interpolation=InterpolationMode.BILINEAR, antialias=True,
            )
            if boxes is not None:
                boxes = tvf.resized_crop(
                    boxes, top, left, height, width, [IMAGE_SIZE, IMAGE_SIZE]
                )
            if random.random() < AUGMENTATION_CONFIG["horizontal_flip_probability"]:
                image = tvf.horizontal_flip(image)
                if boxes is not None:
                    boxes = tvf.horizontal_flip(boxes)
            angle = random.uniform(
                -AUGMENTATION_CONFIG["rotation_degrees"],
                AUGMENTATION_CONFIG["rotation_degrees"],
            )
            image = tvf.rotate(
                image, angle, interpolation=InterpolationMode.BILINEAR, fill=0
            )
            if boxes is not None:
                boxes = tvf.rotate(boxes, angle)
                boxes = tvf.clamp_bounding_boxes(boxes)
            image = self.color_jitter(image)
        else:
            image = self.to_image(image)
            boxes = self.make_boxes(
                box_norm, (image.shape[-2], image.shape[-1])
            ) if box_norm is not None else None
            image = tvf.resize(
                image, [IMAGE_SIZE, IMAGE_SIZE],
                interpolation=InterpolationMode.BILINEAR, antialias=True,
            )
            if boxes is not None:
                boxes = tvf.resize(boxes, [IMAGE_SIZE, IMAGE_SIZE])

        image = self.normalize(self.to_float(image))
        if boxes is None:
            return image, torch.zeros(4, dtype=torch.float32)
        box = boxes[0].to(torch.float32)
        box[[0, 2]] /= IMAGE_SIZE
        box[[1, 3]] /= IMAGE_SIZE
        box = box.clamp(0.0, 1.0)
        if box[2] <= box[0] or box[3] <= box[1]:
            raise RuntimeError(f"增强后bbox退化: {box.tolist()}")
        return image, box


class M1Dataset(Dataset):
    """从冻结清单读取Keep图，并为癌图返回同步增强后的定位监督。

    Args:
        frame (pandas.DataFrame): 单个split的图片级M1清单。
        image_root (Path): 清单相对图像路径的根目录。
        training (bool): 是否启用随机bbox感知增强。

    Outputs:
        每个样本为``dict[str, torch.Tensor]``，包含``image [3,224,224]``、
        标量``label``、``bbox [4]``、标量``valid_box``和稳定``row_index``。
    """

    def __init__(self, frame, image_root, training):
        """绑定当前split、项目图像根和训练或评估变换。

        Args:
            frame (pandas.DataFrame): 当前split的图片记录。
            image_root (Path): 处理图根目录。
            training (bool): ``True``用于train，``False``用于val/test。

        Returns:
            None: 清单副本和变换对象保存为实例状态。
        """
        self.df = frame.reset_index(drop=True)
        self.image_root = image_root
        self.transform = BBoxAwareTransform(training=training)

    def __len__(self):
        """返回当前split的图片数量。

        Returns:
            int: ``self.df``中的图片记录数。
        """
        return len(self.df)

    def __getitem__(self, index):
        """读取并变换指定图片，返回分类和定位训练字段。

        Args:
            index (int): 当前Dataset中的零基图片索引。

        Returns:
            dict[str, torch.Tensor]: 图像、标签、bbox、有效框掩码和行号。
        """
        row = self.df.iloc[index]
        valid = bool(int(row.localization_supervision))
        box = None
        if valid:
            box = (
                float(row.bbox_x1_norm), float(row.bbox_y1_norm),
                float(row.bbox_x2_norm), float(row.bbox_y2_norm),
            )
        with Image.open(self.image_root / row.image_relpath) as source:
            image = source.convert("RGB")
        image, transformed_box = self.transform(image, box)
        return {
            "image": image,
            "label": torch.tensor(int(row.label), dtype=torch.long),
            "bbox": transformed_box,
            "valid_box": torch.tensor(valid, dtype=torch.bool),
            "row_index": torch.tensor(index, dtype=torch.long),
        }


# -----------------------------------------------------------------------------
# 模型结构：分类仍使用完整图，定位头仅作为features[5]上的辅助监督。
# -----------------------------------------------------------------------------
class LocalizationHead(nn.Module):
    """从14x14中层特征预测中心热图、归一化宽高和中心偏移。

    Args:
        in_channels (int): EfficientNet ``features[5]``输入通道数，默认112。
        hidden_channels (int): 定位头共享卷积输出通道数，默认64。

    Outputs:
        ``forward``返回三个批量张量：``heatmap_logits [B,1,14,14]``、
        ``size [B,2,14,14]``和``offset [B,2,14,14]``。
    """

    def __init__(self, in_channels=112, hidden_channels=64):
        """创建共享卷积和三个CenterNet式输出分支。

        Args:
            in_channels (int): 输入特征通道数。
            hidden_channels (int): 共享卷积的隐层通道数。

        Returns:
            None: 卷积层注册为PyTorch子模块。
        """
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.heatmap = nn.Conv2d(hidden_channels, 1, 1)
        self.size = nn.Conv2d(hidden_channels, 2, 1)
        self.offset = nn.Conv2d(hidden_channels, 2, 1)
        nn.init.constant_(self.heatmap.bias, -2.19)

    def forward(self, feature):
        """计算未归一化热图以及限制在0到1的size/offset预测。

        Args:
            feature (torch.Tensor): 形状``[B,112,14,14]``的中层特征。

        Returns:
            dict[str, torch.Tensor]: 热图logits、归一化宽高和网格内偏移。
        """
        hidden = self.shared(feature)
        return {
            "heatmap_logits": self.heatmap(hidden),
            "size": torch.sigmoid(self.size(hidden)),
            "offset": torch.sigmoid(self.offset(hidden)),
        }


class EfficientNetM1(nn.Module):
    """共享EfficientNet编码器，同时输出全局分类和癌图辅助定位结果。

    Args:
        无。构造后由调用方把同种子M0权重加载到``backbone``。

    Outputs:
        ``forward``返回定位头三项，并增加``logits [B,2]``分类输出。
    """

    def __init__(self):
        """创建可加载M0权重的骨干、分类头和随机初始化定位头。

        Returns:
            None: EfficientNet-B0和定位头注册为子模块。
        """
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)
        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features, 2
        )
        self.localization_head = LocalizationHead()

    def forward(self, image):
        """在features[5]分流定位特征，并继续完成全局分类前向。

        Args:
            image (torch.Tensor): ImageNet标准化输入，形状``[B,3,224,224]``。

        Returns:
            dict[str, torch.Tensor]: ``logits [B,2]``及三项14x14定位输出。
        """
        feature = image
        localization_feature = None
        for index, block in enumerate(self.backbone.features):
            feature = block(feature)
            if index == 5:
                localization_feature = feature
        pooled = self.backbone.avgpool(feature)
        logits = self.backbone.classifier(torch.flatten(pooled, 1))
        outputs = self.localization_head(localization_feature)
        outputs["logits"] = logits
        return outputs


# -----------------------------------------------------------------------------
# 监督目标与损失：size/offset只在真实中心回归，M1非癌图屏蔽定位损失。
# -----------------------------------------------------------------------------
def gaussian_radius(height, width, minimum_overlap=0.7):
    """依据框在14x14网格上的大小计算CenterNet高斯半径。

    Args:
        height (float): bbox高度，单位为14x14特征网格格数。
        width (float): bbox宽度，单位为14x14特征网格格数。
        minimum_overlap (float): 期望高斯区域与目标框的最小重叠约束。

    Returns:
        int: 非负整数高斯半径，单位为特征网格格数。
    """
    a1, b1 = 1.0, height + width
    c1 = width * height * (1 - minimum_overlap) / (1 + minimum_overlap)
    r1 = (b1 + math.sqrt(max(0.0, b1 * b1 - 4 * a1 * c1))) / 2
    a2, b2 = 4.0, 2 * (height + width)
    c2 = (1 - minimum_overlap) * width * height
    r2 = (b2 + math.sqrt(max(0.0, b2 * b2 - 4 * a2 * c2))) / 2
    a3, b3 = 4 * minimum_overlap, -2 * minimum_overlap * (height + width)
    c3 = (minimum_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(max(0.0, b3 * b3 - 4 * a3 * c3))) / (2 * a3)
    return max(0, int(min(r1, r2, r3)))


def draw_gaussian(heatmap, center_x, center_y, radius):
    """在目标热图指定中心原位写入二维高斯，重叠处保留最大响应。

    Args:
        heatmap (torch.Tensor): 单张目标热图，形状``[14,14]``。
        center_x (int): 中心所在网格列索引。
        center_y (int): 中心所在网格行索引。
        radius (int): 高斯半径，单位为网格格数。

    Returns:
        None: 直接原位修改``heatmap``。
    """
    diameter = 2 * radius + 1
    sigma = max(diameter / 6, 1e-6)
    coordinates = torch.arange(diameter, device=heatmap.device) - radius
    gaussian = torch.exp(
        -(coordinates[:, None] ** 2 + coordinates[None, :] ** 2) / (2 * sigma * sigma)
    )
    left = min(center_x, radius)
    right = min(GRID_SIZE - center_x - 1, radius)
    top = min(center_y, radius)
    bottom = min(GRID_SIZE - center_y - 1, radius)
    target = heatmap[
        center_y - top:center_y + bottom + 1,
        center_x - left:center_x + right + 1,
    ]
    source = gaussian[
        radius - top:radius + bottom + 1,
        radius - left:radius + right + 1,
    ]
    torch.maximum(target, source, out=target)


def build_targets(boxes, valid_box):
    """由增强后bbox生成heatmap、中心索引、size和offset监督目标。

    Args:
        boxes (torch.Tensor): 批量归一化XYXY框，形状``[B,4]``。
        valid_box (torch.Tensor): 是否计算定位监督的布尔掩码，形状``[B]``。

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 依次为
        ``heatmap [B,1,14,14]``、展平中心索引``[B]``、归一化宽高``[B,2]``
        和中心网格内偏移``[B,2]``。
    """
    batch_size = len(boxes)
    device = boxes.device
    heatmap = torch.zeros((batch_size, 1, GRID_SIZE, GRID_SIZE), device=device)
    indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    size_target = torch.zeros((batch_size, 2), device=device)
    offset_target = torch.zeros((batch_size, 2), device=device)
    for index in torch.where(valid_box)[0].tolist():
        x1, y1, x2, y2 = boxes[index]
        center_x = ((x1 + x2) / 2 * GRID_SIZE).clamp(0, GRID_SIZE - 1e-4)
        center_y = ((y1 + y2) / 2 * GRID_SIZE).clamp(0, GRID_SIZE - 1e-4)
        cell_x = int(torch.floor(center_x).item())
        cell_y = int(torch.floor(center_y).item())
        box_width = (x2 - x1).clamp(min=1e-6)
        box_height = (y2 - y1).clamp(min=1e-6)
        radius = gaussian_radius(
            float(box_height * GRID_SIZE), float(box_width * GRID_SIZE)
        )
        draw_gaussian(heatmap[index, 0], cell_x, cell_y, radius)
        indices[index] = cell_y * GRID_SIZE + cell_x
        size_target[index] = torch.stack((box_width, box_height))
        offset_target[index] = torch.stack((center_x - cell_x, center_y - cell_y))
    return heatmap, indices, size_target, offset_target


def centernet_focal_loss(logits, target):
    """计算CenterNet改进focal loss，降低高斯中心周围负样本惩罚。

    Args:
        logits (torch.Tensor): 有效癌图热图logits，形状``[N,1,14,14]``。
        target (torch.Tensor): 同形状高斯目标，数值范围``[0,1]``。

    Returns:
        torch.Tensor: 标量热图损失，按真实中心数归一化。
    """
    prediction = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    positive = target.eq(1).float()
    negative = target.lt(1).float()
    negative_weight = (1 - target).pow(4)
    positive_loss = torch.log(prediction) * (1 - prediction).pow(2) * positive
    negative_loss = (
        torch.log(1 - prediction) * prediction.pow(2) * negative_weight * negative
    )
    positive_count = positive.sum().clamp(min=1.0)
    return -(positive_loss.sum() + negative_loss.sum()) / positive_count


def gather_at_indices(feature, indices):
    """从每张图的二维输出图中提取指定中心网格位置的通道值。

    Args:
        feature (torch.Tensor): 形状``[B,C,H,W]``的预测图。
        indices (torch.Tensor): 每张图展平空间索引，形状``[B]``。

    Returns:
        torch.Tensor: 中心位置通道值，形状``[B,C]``。
    """
    flattened = feature.flatten(2).transpose(1, 2)
    gather_index = indices[:, None, None].expand(-1, 1, flattened.shape[-1])
    return flattened.gather(1, gather_index).squeeze(1)


def compute_losses(outputs, labels, boxes, valid_box, criterion, args):
    """组合全图分类损失与仅对有效癌框生效的M1定位损失。

    Args:
        outputs (dict[str, torch.Tensor]): 模型分类和三项定位输出。
        labels (torch.Tensor): 0/1分类标签，形状``[B]``。
        boxes (torch.Tensor): 归一化XYXY框，形状``[B,4]``。
        valid_box (torch.Tensor): 有效癌框布尔掩码，形状``[B]``。
        criterion (torch.nn.Module): 所有图片使用的加权交叉熵损失。
        args (argparse.Namespace): ``lambda_loc/size/offset``等权重配置。

    Returns:
        dict[str, torch.Tensor]: 标量``total``、``classification``、
        ``localization``、``heatmap``、``size``和``offset``损失。
    """
    classification = criterion(outputs["logits"], labels)
    heatmap_target, indices, size_target, offset_target = build_targets(boxes, valid_box)
    if valid_box.any():
        heatmap = centernet_focal_loss(
            outputs["heatmap_logits"][valid_box], heatmap_target[valid_box]
        )
        predicted_size = gather_at_indices(outputs["size"], indices)[valid_box]
        predicted_offset = gather_at_indices(outputs["offset"], indices)[valid_box]
        size = nnf.smooth_l1_loss(predicted_size, size_target[valid_box])
        offset = nnf.smooth_l1_loss(predicted_offset, offset_target[valid_box])
        localization = heatmap + args.lambda_size * size + args.lambda_offset * offset
    else:
        zero = outputs["heatmap_logits"].sum() * 0
        heatmap = size = offset = localization = zero
    total = classification + args.lambda_loc * localization
    return {
        "total": total,
        "classification": classification,
        "localization": localization,
        "heatmap": heatmap,
        "size": size,
        "offset": offset,
    }


def decode_boxes(outputs):
    """从热图峰值及对应size/offset解码归一化预测框和定位置信度。

    Args:
        outputs (dict[str, torch.Tensor]): 模型热图、size和offset批量输出。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``[0,1]``范围的XYXY预测框``[B,4]``
        和热图最大sigmoid响应``[B]``；M1阶段后者仅作定位强度参考。
    """
    heatmap = torch.sigmoid(outputs["heatmap_logits"])
    confidence, indices = heatmap.flatten(1).max(dim=1)
    cell_y = torch.div(indices, GRID_SIZE, rounding_mode="floor")
    cell_x = indices % GRID_SIZE
    size = gather_at_indices(outputs["size"], indices)
    offset = gather_at_indices(outputs["offset"], indices)
    center_x = (cell_x.float() + offset[:, 0]) / GRID_SIZE
    center_y = (cell_y.float() + offset[:, 1]) / GRID_SIZE
    x1 = center_x - size[:, 0] / 2
    y1 = center_y - size[:, 1] / 2
    x2 = center_x + size[:, 0] / 2
    y2 = center_y + size[:, 1] / 2
    return torch.stack((x1, y1, x2, y2), dim=1).clamp(0, 1), confidence


def box_iou_and_coverage(predicted, target):
    """批量计算预测框IoU及预测框对真实病灶框的覆盖比例。

    Args:
        predicted (torch.Tensor): 归一化预测XYXY框，形状``[N,4]``。
        target (torch.Tensor): 同坐标系真实XYXY框，形状``[N,4]``。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 每图IoU和``intersection/GT area``
        覆盖率，二者形状均为``[N]``。
    """
    top_left = torch.maximum(predicted[:, :2], target[:, :2])
    bottom_right = torch.minimum(predicted[:, 2:], target[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
    pred_area = (predicted[:, 2:] - predicted[:, :2]).clamp(min=0).prod(dim=1)
    target_area = (target[:, 2:] - target[:, :2]).clamp(min=0).prod(dim=1)
    union = pred_area + target_area - intersection
    return intersection / union.clamp(min=1e-8), intersection / target_area.clamp(min=1e-8)


# -----------------------------------------------------------------------------
# 定位评估与质检：自动指标和叠加图分开保存，热图不解释为病灶分割。
# -----------------------------------------------------------------------------
def localization_metrics(frame):
    """在有效癌框上汇总中心命中、距离、IoU、覆盖率及固定中心基线。

    Args:
        frame (pandas.DataFrame): 含``valid_box``、GT框和预测框列的图片级表。

    Returns:
        dict[str, int | float]: 有效框数及图片级定位汇总指标；无有效框时
        返回空字典。
    """
    valid = frame.loc[frame.valid_box].copy()
    if valid.empty:
        return {}
    target = torch.tensor(valid[["gt_x1", "gt_y1", "gt_x2", "gt_y2"]].to_numpy())
    predicted = torch.tensor(valid[["pred_x1", "pred_y1", "pred_x2", "pred_y2"]].to_numpy())
    iou, coverage = box_iou_and_coverage(predicted, target)
    pred_center = (predicted[:, :2] + predicted[:, 2:]) / 2
    target_center = (target[:, :2] + target[:, 2:]) / 2
    center_distance = torch.linalg.vector_norm(pred_center - target_center, dim=1)
    center_hit = (
        (pred_center[:, 0] >= target[:, 0])
        & (pred_center[:, 0] <= target[:, 2])
        & (pred_center[:, 1] >= target[:, 1])
        & (pred_center[:, 1] <= target[:, 3])
    )
    fixed_center = torch.full_like(pred_center, 0.5)
    fixed_hit = (
        (fixed_center[:, 0] >= target[:, 0])
        & (fixed_center[:, 0] <= target[:, 2])
        & (fixed_center[:, 1] >= target[:, 1])
        & (fixed_center[:, 1] <= target[:, 3])
    )
    return {
        "count": int(len(valid)),
        "center_hit_rate": float(center_hit.float().mean()),
        "fixed_center_hit_rate": float(fixed_hit.float().mean()),
        "mean_center_distance": float(center_distance.mean()),
        "mean_iou": float(iou.mean()),
        "median_iou": float(iou.median()),
        "iou_ge_0.3": float(iou.ge(0.3).float().mean()),
        "iou_ge_0.5": float(iou.ge(0.5).float().mean()),
        "mean_lesion_coverage": float(coverage.mean()),
    }


def tensor_to_pil(image):
    """反转ImageNet标准化，将训练张量恢复为可视化PIL图像。

    Args:
        image (torch.Tensor): 标准化RGB张量，形状``[3,H,W]``。

    Returns:
        PIL.Image.Image: 数值裁到``[0,1]``后转换的RGB图像。
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype)[:, None, None]
    restored = (image.cpu() * std + mean).clamp(0, 1)
    return v2.ToPILImage()(restored)


def draw_normalized_box(draw, box, color, width=3):
    """把归一化XYXY框按224画布绘制为指定颜色矩形。

    Args:
        draw (PIL.ImageDraw.ImageDraw): 目标图像的绘图上下文。
        box (Sequence[float]): ``[0,1]``范围XYXY坐标。
        color (str | tuple[int, int, int]): PIL支持的边框颜色。
        width (int): 矩形边线像素宽度。

    Returns:
        None: 直接修改绘图上下文对应图像。
    """
    x1, y1, x2, y2 = [float(value) * IMAGE_SIZE for value in box]
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)


def save_contact_sheet(images, labels, path, columns=4):
    """将若干质检图和短标签排版为固定列数的联系表。

    Args:
        images (list[PIL.Image.Image]): 已统一为224像素方形的质检图。
        labels (list[str]): 与图片一一对应的底部短标签。
        path (Path): JPEG输出路径。
        columns (int): 联系表列数。

    Returns:
        None: 图片列表为空时不写文件，否则保存JPEG联系表。
    """
    if not images:
        return
    tile_height = IMAGE_SIZE + 24
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * IMAGE_SIZE, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels)):
        left = index % columns * IMAGE_SIZE
        top = index // columns * tile_height
        sheet.paste(image, (left, top))
        draw.text((left + 4, top + IMAGE_SIZE + 4), label, fill="black")
    sheet.save(path, quality=95)


def export_train_transform_qc(dataset, output, count=12):
    """导出增强后真值框质检图，供正式训练前核对坐标同步。

    Args:
        dataset (M1Dataset): 启用训练增强的train Dataset。
        output (Path): 当前运行输出目录。
        count (int): 最多展示的图片数，默认癌图约占三分之二。

    Returns:
        None: 写出``train_transform_bbox_qc.jpg``。
    """
    cancer_indices = dataset.df.index[dataset.df.localization_supervision.eq(1)].tolist()
    control_indices = dataset.df.index[dataset.df.localization_supervision.eq(0)].tolist()
    chosen = cancer_indices[: max(1, count * 2 // 3)] + control_indices[: max(1, count // 3)]
    images = []
    labels = []
    for index in chosen[:count]:
        sample = dataset[index]
        image = tensor_to_pil(sample["image"])
        if sample["valid_box"]:
            draw_normalized_box(ImageDraw.Draw(image), sample["bbox"], "lime")
        images.append(image)
        labels.append(
            f"label={int(sample['label'])} bbox={int(sample['valid_box'])}"
        )
    save_contact_sheet(images, labels, output / "train_transform_bbox_qc.jpg")


def export_prediction_qc(predictions, image_root, output, count=20):
    """导出验证癌图的真值框与预测框叠加联系表。

    Args:
        predictions (pandas.DataFrame): 含图像路径、GT框、预测框和置信度的表。
        image_root (Path): 处理图根目录。
        output (Path): 当前运行输出目录。
        count (int): 最多展示的有效癌框图片数。

    Returns:
        None: 写出``val_gt_vs_prediction_qc.jpg``；绿色为GT，红色为预测。
    """
    chosen = predictions.loc[predictions.valid_box].head(count)
    images = []
    labels = []
    for row in chosen.itertuples(index=False):
        with Image.open(image_root / row.image_relpath) as source:
            image = source.convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
            )
        draw = ImageDraw.Draw(image)
        draw_normalized_box(
            draw, (row.gt_x1, row.gt_y1, row.gt_x2, row.gt_y2), "lime"
        )
        draw_normalized_box(
            draw, (row.pred_x1, row.pred_y1, row.pred_x2, row.pred_y2), "red"
        )
        images.append(image)
        labels.append(f"GT=green Pred=red conf={row.localization_confidence:.3f}")
    save_contact_sheet(images, labels, output / "val_gt_vs_prediction_qc.jpg")


# -----------------------------------------------------------------------------
# 训练编排：先预热新增定位头，再有限解冻M0后层联合微调。
# -----------------------------------------------------------------------------
def set_warmup_trainable(model):
    """M1预热阶段冻结M0骨干和分类头，只训练随机定位头。

    Args:
        model (EfficientNetM1): 待设置``requires_grad``的M1模型。

    Returns:
        None: 原位修改模型参数的可训练状态。
    """
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.localization_head.parameters():
        parameter.requires_grad = True


def set_joint_trainable(model):
    """联合阶段解冻features[5:]、分类头和定位头，保持前层冻结。

    Args:
        model (EfficientNetM1): 已完成定位头预热的M1模型。

    Returns:
        None: 原位修改模型参数的可训练状态。
    """
    for parameter in model.parameters():
        parameter.requires_grad = False
    for index in range(5, len(model.backbone.features)):
        for parameter in model.backbone.features[index].parameters():
            parameter.requires_grad = True
    for parameter in model.backbone.classifier.parameters():
        parameter.requires_grad = True
    for parameter in model.localization_head.parameters():
        parameter.requires_grad = True


def set_train_mode(model, stage):
    """设置阶段训练模式，并固定冻结骨干部分的BatchNorm统计。

    Args:
        model (EfficientNetM1): 当前训练模型。
        stage (str): ``warmup``或``joint``；前者冻结整个backbone统计，后者
        固定``features[:5]``统计。

    Returns:
        None: 原位切换模块train/eval状态。
    """
    model.train()
    if stage == "warmup":
        model.backbone.eval()
    else:
        for index in range(5):
            model.backbone.features[index].eval()


def build_loaders(frame, image_root, args):
    """基于同一冻结患者划分构建train/val/test DataLoader。

    Args:
        frame (pandas.DataFrame): 已校验、包含三个split的M1图片清单。
        image_root (Path): 处理图根目录。
        args (argparse.Namespace): batch size、worker数和随机种子配置。

    Returns:
        dict[str, DataLoader]: 键为``train/val/test``；仅train启用随机增强和shuffle。
    """
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = M1Dataset(
            frame.loc[frame.split.eq(split)].copy(), image_root, split == "train"
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=generator,
            persistent_workers=args.num_workers > 0,
        )
    return loaders


def train_epoch(model, loader, optimizer, criterion, device, args, stage):
    """完成一个训练epoch并返回分类、定位及各子损失的样本均值。

    Args:
        model (EfficientNetM1): 当前阶段模型。
        loader (DataLoader): train数据加载器。
        optimizer (torch.optim.Optimizer): 当前阶段优化器。
        criterion (torch.nn.Module): 加权且带label smoothing的分类损失。
        device (torch.device): ``cuda``或``cpu``计算设备。
        args (argparse.Namespace): M1损失权重等配置。
        stage (str): ``warmup``或``joint``，决定模块训练模式。

    Returns:
        dict[str, float]: 按图片数加权的total、分类、定位及三个定位子损失均值。
    """
    set_train_mode(model, stage)
    totals = {key: 0.0 for key in [
        "total", "classification", "localization", "heatmap", "size", "offset"
    ]}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        boxes = batch["bbox"].to(device, non_blocking=True)
        valid_box = batch["valid_box"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        losses = compute_losses(outputs, labels, boxes, valid_box, criterion, args)
        losses["total"].backward()
        optimizer.step()
        for key in totals:
            totals[key] += float(losses[key].detach()) * len(images)
    return {key: value / len(loader.dataset) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device, args):
    """无梯度推理一个split，输出分类预测、患者聚合和定位指标。

    Args:
        model (EfficientNetM1): 待评估模型。
        loader (DataLoader): val或显式解锁后的test加载器，必须不shuffle。
        criterion (torch.nn.Module): 与训练一致的分类损失。
        device (torch.device): 推理设备。
        args (argparse.Namespace): 定位损失权重配置。

    Returns:
        dict[str, object]: 标量loss、图片预测表、患者聚合表、图片/患者分类
        指标及癌图定位指标。
    """
    model.eval()
    rows = []
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        boxes = batch["bbox"].to(device, non_blocking=True)
        valid_box = batch["valid_box"].to(device, non_blocking=True)
        outputs = model(images)
        losses = compute_losses(outputs, labels, boxes, valid_box, criterion, args)
        predicted_boxes, confidence = decode_boxes(outputs)
        probabilities = torch.softmax(outputs["logits"], dim=1)[:, 1]
        for local_index, row_index in enumerate(batch["row_index"].tolist()):
            rows.append({
                "row_index": row_index,
                "label": int(labels[local_index]),
                "cancer_probability": float(probabilities[local_index]),
                "valid_box": bool(valid_box[local_index]),
                "localization_confidence": float(confidence[local_index]),
                "gt_x1": float(boxes[local_index, 0]),
                "gt_y1": float(boxes[local_index, 1]),
                "gt_x2": float(boxes[local_index, 2]),
                "gt_y2": float(boxes[local_index, 3]),
                "pred_x1": float(predicted_boxes[local_index, 0]),
                "pred_y1": float(predicted_boxes[local_index, 1]),
                "pred_x2": float(predicted_boxes[local_index, 2]),
                "pred_y2": float(predicted_boxes[local_index, 3]),
            })
        total_loss += float(losses["total"]) * len(images)
    predictions = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
    metadata = loader.dataset.df.copy().reset_index(drop=True)
    predictions = pd.concat([
        metadata,
        predictions.drop(columns=["row_index", "label"]),
    ], axis=1)
    image_metrics = compute_metrics(predictions.label, predictions.cancer_probability)
    patient_predictions = patient_prediction_frame(predictions)
    patient_metrics = compute_metrics(
        patient_predictions.label, patient_predictions.cancer_probability
    )
    return {
        "loss": total_loss / len(loader.dataset),
        "predictions": predictions,
        "patient_predictions": patient_predictions,
        "image_metrics": image_metrics,
        "patient_metrics": patient_metrics,
        "localization_metrics": localization_metrics(predictions),
    }


def save_checkpoint(path, model, args, epoch_record, m0_checkpoint):
    """保存完整M1参数、训练配置、M0来源及对应选择epoch记录。

    Args:
        path (Path): ``.pth``输出路径。
        model (EfficientNetM1): 需要完整保存的当前模型。
        args (argparse.Namespace): 本次运行参数。
        epoch_record (dict[str, object]): 该checkpoint对应的val epoch记录。
        m0_checkpoint (Path): 初始化所用M0权重路径。

    Returns:
        None: 使用``torch.save``写出checkpoint。
    """
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": json_ready({
            **serialized_args,
            "m0_checkpoint": str(m0_checkpoint.resolve()),
            "selection_record": epoch_record,
            "architecture": "EfficientNet-B0 features[5] CenterNet-like M1",
        }),
    }, path)


def prepare_run_dir(args):
    """创建独立运行目录；已有目录一律拒绝覆盖。

    Args:
        args (argparse.Namespace): 输出根、运行名、seed和debug配置。

    Returns:
        Path: 已创建且可写的本次运行目录。

    Raises:
        FileExistsError: 目录已存在但未显式允许覆盖。
    """
    name = args.run_name or f"m1_balanced_keep_efficientnet_b0_seed{args.seed}"
    if args.debug:
        name += "_debug"
    output = args.output_root / name
    if output.exists():
        raise FileExistsError(f"输出已存在，请更换运行名: {output}")
    output.mkdir(parents=True)
    return output


def main():
    """编排M0加载、定位头预热、联合训练、模型选择、质检和结果保存。

    Args:
        无。全部输入来自``parse_args``解析的命令行参数。

    Returns:
        None: 运行产物写入独立目录，并在标准输出打印进度和最终路径。
    """
    args = parse_args()
    if args.debug:
        args.warmup_epochs = min(args.warmup_epochs, 1)
        args.joint_epochs = min(args.joint_epochs, 1)
        args.num_workers = 0
    if not 0 < args.crop_min_bbox_retention <= 1:
        raise ValueError("crop-min-bbox-retention必须在(0,1]内")
    seed_everything(args.seed)
    m0_checkpoint = args.m0_checkpoint or default_m0_checkpoint(args.seed)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not m0_checkpoint.is_file():
        raise FileNotFoundError(m0_checkpoint)
    output = prepare_run_dir(args)
    frame = load_manifest(
        args.manifest, args.image_root, args.debug, args.debug_units, args.seed
    )
    loaders = build_loaders(frame, args.image_root, args)
    export_train_transform_qc(loaders["train"].dataset, output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}; M0权重: {m0_checkpoint}")
    for split, loader in loaders.items():
        table = loader.dataset.df
        print(
            f"{split}: {table.patient_id.nunique()}人/{len(table)}张; "
            f"标签={table.label.value_counts().sort_index().to_dict()}; "
            f"定位监督={int(table.localization_supervision.sum())}张"
        )

    m0_payload = torch.load(m0_checkpoint, map_location="cpu", weights_only=False)
    m0_auc = float(m0_payload["config"]["best_val_patient_auc"])
    model = EfficientNetM1()
    model.backbone.load_state_dict(m0_payload["model_state_dict"], strict=True)
    model.to(device)
    counts = loaders["train"].dataset.df.label.value_counts().reindex([0, 1])
    weights = counts.sum() / (2 * counts.to_numpy(dtype=float))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=LABEL_SMOOTHING,
    )
    print(f"M0 val患者AUC={m0_auc:.4f}; M1分类非劣下限={m0_auc-args.classification_noninferiority:.4f}")

    history = []
    best_class = {"score": -np.inf, "state": None, "record": None}
    best_loc = {"score": (-np.inf, -np.inf), "state": None, "record": None}
    best_joint = {"score": (-np.inf, -np.inf), "state": None, "record": None}
    best_joint_stage_class = {"score": -np.inf, "state": None, "record": None}
    best_joint_stage_eligible = {
        "score": (-np.inf, -np.inf), "state": None, "record": None
    }
    best_warmup_product = {
        "score": (-np.inf, -np.inf, -np.inf),
        "state": None,
        "record": None,
    }
    global_epoch = 0
    best_joint_stage_auc = -np.inf
    joint_auc_no_improve = 0
    joint_eligible_epoch_count = 0
    warmup_eligible_epoch_count = 0

    stages = [("warmup", args.warmup_epochs, args.warmup_lr, set_warmup_trainable)]
    if not args.warmup_only:
        stages.append(("joint", args.joint_epochs, args.joint_lr, set_joint_trainable))
    for stage, epochs, learning_rate, trainable_function in stages:
        trainable_function(model)
        optimizer = optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        ) if stage == "joint" else None
        for stage_epoch in range(1, epochs + 1):
            global_epoch += 1
            started = time.time()
            train_losses = train_epoch(
                model, loaders["train"], optimizer, criterion, device, args, stage
            )
            validation = evaluate(model, loaders["val"], criterion, device, args)
            loc = validation["localization_metrics"]
            record = {
                "epoch": global_epoch,
                "stage": stage,
                "stage_epoch": stage_epoch,
                **{f"train_{key}": value for key, value in train_losses.items()},
                "val_loss": validation["loss"],
                **{f"val_image_{key}": value for key, value in validation["image_metrics"].items()},
                **{f"val_patient_{key}": value for key, value in validation["patient_metrics"].items()},
                **{f"val_loc_{key}": value for key, value in loc.items()},
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - started,
            }
            history.append(record)
            patient_auc = validation["patient_metrics"]["AUC"]
            loc_score = (loc.get("center_hit_rate", -np.inf), loc.get("mean_iou", -np.inf))
            if patient_auc > best_class["score"]:
                best_class = {"score": patient_auc, "state": deepcopy(model.state_dict()), "record": record}
            if loc_score > best_loc["score"]:
                best_loc = {"score": loc_score, "state": deepcopy(model.state_dict()), "record": record}
            joint_eligible = patient_auc >= m0_auc - args.classification_noninferiority
            if joint_eligible and loc_score > best_joint["score"]:
                best_joint = {"score": loc_score, "state": deepcopy(model.state_dict()), "record": record}
            if stage == "warmup":
                center_hit = loc.get("center_hit_rate", -np.inf)
                fixed_center_hit = loc.get("fixed_center_hit_rate", -np.inf)
                mean_iou = loc.get("mean_iou", -np.inf)
                mean_center_distance = loc.get("mean_center_distance", np.inf)
                warmup_eligible = (
                    center_hit >= fixed_center_hit + 0.10 and mean_iou >= 0.40
                )
                if warmup_eligible:
                    warmup_eligible_epoch_count += 1
                    # ROI产品优先框重合质量；平手时再比中心距离和命中率。
                    warmup_score = (mean_iou, -mean_center_distance, center_hit)
                    if warmup_score > best_warmup_product["score"]:
                        best_warmup_product = {
                            "score": warmup_score,
                            "state": deepcopy(model.state_dict()),
                            "record": record,
                        }
            if stage == "joint":
                if patient_auc > best_joint_stage_class["score"]:
                    best_joint_stage_class = {
                        "score": patient_auc,
                        "state": deepcopy(model.state_dict()),
                        "record": record,
                    }
                if joint_eligible:
                    joint_eligible_epoch_count += 1
                    if loc_score > best_joint_stage_eligible["score"]:
                        best_joint_stage_eligible = {
                            "score": loc_score,
                            "state": deepcopy(model.state_dict()),
                            "record": record,
                        }
                # 非劣门槛只决定checkpoint资格；早停独立监测分类AUC是否仍在恢复。
                if patient_auc > best_joint_stage_auc + 1e-12:
                    best_joint_stage_auc = patient_auc
                    joint_auc_no_improve = 0
                else:
                    joint_auc_no_improve += 1
            print(
                f"{stage} {stage_epoch:02d}/{epochs} | "
                f"Train={train_losses['total']:.4f} "
                f"(cls={train_losses['classification']:.4f}, loc={train_losses['localization']:.4f}) | "
                f"Val patient AUC={patient_auc:.4f} | "
                f"center-hit={loc.get('center_hit_rate', float('nan')):.4f} | "
                f"IoU={loc.get('mean_iou', float('nan')):.4f} | {record['elapsed_seconds']:.0f}s"
            )
            if scheduler is not None:
                scheduler.step()
            if (
                stage == "joint" and args.early_stop_patience
                and joint_auc_no_improve >= args.early_stop_patience
            ):
                print(
                    f"joint early stop: val患者AUC连续 "
                    f"{args.early_stop_patience} epoch未创新高；"
                    f"joint最佳AUC={best_joint_stage_auc:.4f}"
                )
                break

    if best_warmup_product["state"] is None:
        raise RuntimeError(
            "warmup无epoch通过产品门槛: "
            "center-hit >= fixed-center-hit + 0.10 且 mean IoU >= 0.40"
        )
    model.load_state_dict(best_warmup_product["state"])
    save_checkpoint(
        output / "m1_best_warmup_localization.pth",
        model,
        args,
        best_warmup_product["record"],
        m0_checkpoint,
    )
    print(
        "Warmup-product summary: "
        f"epoch={best_warmup_product['record']['stage_epoch']} | "
        f"center-hit={best_warmup_product['record']['val_loc_center_hit_rate']:.4f} | "
        f"IoU={best_warmup_product['record']['val_loc_mean_iou']:.4f} | "
        f"eligible epochs={warmup_eligible_epoch_count}"
    )

    if args.warmup_only:
        checkpoint_records = {"warmup_localization": best_warmup_product}
        selected_checkpoint = best_warmup_product
        delivered_stage = "warmup"
    else:
        if best_joint["state"] is None:
            print("警告: 无epoch通过M0分类非劣门槛，joint回退到分类最佳")
            best_joint = deepcopy(best_class)
        checkpoint_records = {
            "classification": best_class,
            "localization": best_loc,
            "joint": best_joint,
        }
        for name, item in checkpoint_records.items():
            model.load_state_dict(item["state"])
            save_checkpoint(
                output / f"m1_best_{name}.pth", model, args, item["record"], m0_checkpoint
            )
        # 独立保留真正joint阶段的证据，避免全局选择回选warm-up后掩盖结果。
        model.load_state_dict(best_joint_stage_class["state"])
        save_checkpoint(
            output / "m1_best_joint_stage_auc.pth",
            model,
            args,
            best_joint_stage_class["record"],
            m0_checkpoint,
        )
        if best_joint_stage_eligible["state"] is not None:
            model.load_state_dict(best_joint_stage_eligible["state"])
            save_checkpoint(
                output / "m1_best_joint_stage_eligible.pth",
                model,
                args,
                best_joint_stage_eligible["record"],
                m0_checkpoint,
            )
        else:
            print("注意: joint阶段无epoch通过分类非劣门槛。")
        delivered_stage = best_joint["record"]["stage"]
        selected_checkpoint = best_joint
        print(
            "Joint-stage summary: "
            f"best AUC={best_joint_stage_class['score']:.4f} | "
            f"floor={m0_auc-args.classification_noninferiority:.4f} | "
            f"eligible epochs={joint_eligible_epoch_count} | "
            f"delivered stage={delivered_stage}"
        )

    model.load_state_dict(selected_checkpoint["state"])
    validation = evaluate(model, loaders["val"], criterion, device, args)
    val_patients = validation["patient_predictions"]
    threshold, threshold_scan = select_screening_threshold(
        val_patients.label, val_patients.cancer_probability, minimum_sensitivity=0.90
    )
    val_image_metrics = compute_metrics(
        validation["predictions"].label,
        validation["predictions"].cancer_probability,
        threshold,
    )
    val_patient_metrics = compute_metrics(
        val_patients.label, val_patients.cancer_probability, threshold
    )
    selection_label = "Warmup product" if args.warmup_only else "Joint"
    print(f"{selection_label} val image: {format_metrics(val_image_metrics)}")
    print(f"{selection_label} val patient: {format_metrics(val_patient_metrics)}")
    print(f"{selection_label} val localization: {validation['localization_metrics']}")

    pd.DataFrame(history).to_csv(
        output / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    frame.to_csv(output / "frozen_split_snapshot.csv", index=False, encoding="utf-8-sig")
    validation["predictions"].to_csv(
        output / "val_image_predictions.csv", index=False, encoding="utf-8-sig"
    )
    export_prediction_qc(validation["predictions"], args.image_root, output)
    val_patients.to_csv(
        output / "val_patient_predictions.csv", index=False, encoding="utf-8-sig"
    )
    threshold_scan.to_csv(
        output / "val_patient_threshold_scan.csv", index=False, encoding="utf-8-sig"
    )
    test_result = None
    if args.evaluate_test:
        test_result = evaluate(model, loaders["test"], criterion, device, args)
        test_result["predictions"].to_csv(
            output / "test_image_predictions.csv", index=False, encoding="utf-8-sig"
        )
        test_result["patient_predictions"].to_csv(
            output / "test_patient_predictions.csv", index=False, encoding="utf-8-sig"
        )
    else:
        print("内部test保持锁定；未生成test预测。")

    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    warmup_product_selection = {
        "record": best_warmup_product["record"],
        "eligible_epoch_count": warmup_eligible_epoch_count,
        "center_hit_margin_over_fixed": 0.10,
        "minimum_mean_iou": 0.40,
        "ranking": ["mean_iou desc", "mean_center_distance asc", "center_hit_rate desc"],
    }
    config = {
        **serialized_args,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "m0_checkpoint": str(m0_checkpoint.resolve()),
        "m0_checkpoint_sha256": file_sha256(m0_checkpoint),
        "m0_val_patient_auc": m0_auc,
        "architecture": "EfficientNet-B0 features[5] CenterNet-like M1",
        "grid_size": GRID_SIZE,
        "classification_noninferiority_floor": m0_auc - args.classification_noninferiority,
        "loss": {
            "classification": "weighted CrossEntropy(label_smoothing=0.1)",
            "heatmap": "CenterNet focal on valid cancer boxes only",
            "size": "SmoothL1 normalized width/height at GT center",
            "offset": "SmoothL1 fractional center offset at GT center",
        },
        "selection": {
            key: value["record"] for key, value in checkpoint_records.items()
        },
        "warmup_product_selection": warmup_product_selection,
        "selected_val_image_metrics": val_image_metrics,
        "selected_val_patient_metrics": val_patient_metrics,
        "selected_val_localization_metrics": validation["localization_metrics"],
        "val_threshold": threshold,
        "test_evaluated": args.evaluate_test,
        **git_snapshot(),
    }
    if not args.warmup_only:
        config.update({
            "joint_stage_selection": {
                "highest_auc": best_joint_stage_class["record"],
                "eligible_localization": best_joint_stage_eligible["record"],
                "eligible_epoch_count": joint_eligible_epoch_count,
                "passed_noninferiority": joint_eligible_epoch_count > 0,
                "delivered_checkpoint_stage": delivered_stage,
                "delivered_checkpoint_is_joint_stage": delivered_stage == "joint",
            },
            "early_stopping": {
                "stage": "joint",
                "metric": "val_patient_auc",
                "mode": "max",
                "patience": args.early_stop_patience,
                "best_joint_stage_auc": best_joint_stage_auc,
                "note": "noninferiority gates joint checkpoint eligibility only",
            },
            "joint_val_image_metrics": val_image_metrics,
            "joint_val_patient_metrics": val_patient_metrics,
            "joint_val_localization_metrics": validation["localization_metrics"],
        })
    (output / "config.json").write_text(
        json.dumps(json_ready(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), output / "source_entry.py")
    print(f"输出目录: {output}")


if __name__ == "__main__":
    main()
