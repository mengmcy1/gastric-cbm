#!/usr/bin/env python3
"""Train the preregistered MG2 full-image student with MAGE-like distillation.

MG2 distills the frozen MG1b attention-pooling teacher (grayscale ROI input)
into an EfficientNet-B0 student that reads the full-color 224x224 WLI image.
The student reuses the MG1b architecture exactly: ``features[8] -> 1x1 Conv ->
spatial softmax over 7x7 -> attention-weighted sum -> dropout + Linear``, with
no GAP bypass. One entry point runs three arms that share architecture,
ImageNet initialization, data order, sampler, augmentation and budget, and
differ only in the loss:

- ``--arm A``: CE only (label_smoothing=0.1);
- ``--arm B``: ``alpha*CE + (1-alpha)*tau^2*KL(teacher_soft||student_soft)``
  on all samples, ``alpha=0.25``, ``tau=4``;
- ``--arm C``: B + ``beta * mean_cancer KL(backfilled_teacher_attention ||
  student_attention)``. Formal arm C reads ``beta`` only from the frozen
  calibration JSON (``--beta-calibration-json``) after enforcing its SHA/seed/
  batch-size binding; manual ``--beta`` is accepted only in debug mode.

Coordinate backfill is the core geometry step: the teacher 7x7 attention is
defined on the crop rectangle; each teacher cell's mass is distributed onto
the student full-image 7x7 grid in proportion to the rectangle-intersection
area with the actual (post-flip) ``base_crop_box``, then renormalized to sum
1; mass outside the crop is zero. Training applies a synchronized horizontal
flip (p=0.5) to the full image together with ``lesion_bbox`` and
``base_crop_box`` before cropping the teacher ROI, so the backfill always uses
the flip state that was actually applied. Teacher logits/attention for both
flip states of every train/val image come from the SHA-bound cache built by
``build_mage_mg2_teacher_cache.py``; arm A never reads the cache.

Only manifest split=train and split=val are read; test/internal test/external
never enter any stage and the config records all three flags as false.
Checkpoint selection: highest val patient AUC, then highest val image AUC,
then lowest val loss. Debug mode (``--debug``) subsamples patients per class
and writes only into the dedicated debug directory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import functional as TF

from build_mage_teacher_roi_manifest import PROJECT_ROOT, file_sha256
from train_mage_mg1_teacher import (
    DEFAULT_MANIFEST,
    DEFAULT_V3_AUDIT,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    git_snapshot,
    load_manifest,
    patient_class_balanced_weights,
    patient_mean,
    seed_everything,
    sensitivity_threshold_metrics,
    stable_uniform,
)
from train_mage_mg1b_attention_teacher import (
    GRID_SIZE,
    LABEL_SMOOTHING,
    AttentionPoolingTeacher,
    cell_overlap_map,
    set_stage,
    set_train_mode,
)


MG2_ROOT = PROJECT_ROOT / "结果/MAGE/MG2全图学生蒸馏_20260818"
DEFAULT_OUTPUT = MG2_ROOT / "正式验证集筛选"
DEFAULT_DEBUG_OUTPUT = MG2_ROOT / "debug"
DEFAULT_TEACHER_CHECKPOINT = PROJECT_ROOT / (
    "结果/MAGE/MG1b注意力池化教师_20260818/正式验证集筛选/"
    "mg1b_attention_efficientnet_b0_seed42/mg1b_best_teacher.pth"
)
FROZEN_TEACHER_SHA256 = (
    "f2cd3b13cbbc0e8b4df64e9090ec50343314137ba71a4c07d9fae9faa8af48f9"
)
DEFAULT_TEACHER_CACHE = MG2_ROOT / "teacher_cache_mg1b_v3.pt"
DEFAULT_DEBUG_TEACHER_CACHE = DEFAULT_DEBUG_OUTPUT / "teacher_cache_debug.pt"
CACHE_FORMAT = "mage_mg2_teacher_cache_v1"

ALPHA = 0.25
TAU = 4.0
DEFAULT_BETA = 1.0
BETA_CALIBRATION_STAGE = "MG2_beta_calibration"
BETA_CALIBRATION_RULE = (
    "beta = 0.25 * median(L_nonspatial_batch) / median(L_attention_batch)"
)
FROZEN_BETA_CALIBRATION_SHA256 = (
    "47179b6228122096c00e6f0e59ab5c45663b3f163a506c77aa8811320803ac7f"
)
NORMALIZED_AIB_MAX_BBOX_AREA = 0.99
STRATUM_MIN_IMAGES = 15

# P6光度增强参数，逐项对齐M0-F正式配置；几何增强全部关闭，仅保留同步水平翻转。
P6_PHOTOMETRIC = {
    "color_jitter_brightness": 0.15,
    "color_jitter_contrast": 0.15,
    "color_jitter_saturation": 0.10,
    "downsample_probability": 0.5,
    "downsample_scale": [0.50, 0.85],
    "jpeg_probability": 0.5,
    "jpeg_quality": [60, 95],
    "gaussian_blur_probability": 0.3,
    "gaussian_blur_sigma": [0.1, 1.2],
    "gaussian_blur_kernel_size": 5,
}


def parse_args() -> argparse.Namespace:
    """Parse the frozen MG2 arm selection, optimization and output parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B", "C"), default=None,
                        help="损失臂：A仅CE；B加logit KD；C再加癌侧attention KD")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--v3-audit", type=Path, default=DEFAULT_V3_AUDIT)
    parser.add_argument("--teacher-checkpoint", type=Path,
                        default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--teacher-cache", type=Path, default=None,
                        help="B/C必读的教师缓存；缺省按正式/debug目录解析")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="缺省为正式目录；--debug时缺省为独立debug目录")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--overwrite", action="store_true",
                        help="仅当目标目录不含任何正式产物时允许复用")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stage-a-epochs", type=int, default=10)
    parser.add_argument("--stage-b-epochs", type=int, default=20)
    parser.add_argument("--stage-a-lr", type=float, default=1e-3)
    parser.add_argument("--stage-b-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--beta", type=float, default=None,
                        help="仅debug模式允许的手工占位beta；正式C组禁止手工指定")
    parser.add_argument("--beta-calibration-json", type=Path, default=None,
                        help="正式C组必选：冻结beta校准JSON，beta只能由此读取")
    parser.add_argument("--calibrate-beta", type=Path, default=None,
                        help="只做beta校准：完整校准epoch前向，写出JSON后退出")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-patients-per-class", type=int, default=3)
    return parser.parse_args()


def seed_worker(worker_id: int) -> None:
    """Seed numpy and python ``random`` in each DataLoader worker.

    参数:
        worker_id (int): PyTorch分配的worker编号；实际种子取自
            ``torch.initial_seed()``，保证三组臂的worker随机序列一致。
    返回: 无。
    """
    del worker_id
    import random

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def flip_box_horizontal(box: np.ndarray) -> np.ndarray:
    """Mirror one normalized ``[x1,y1,x2,y2]`` box under a horizontal flip.

    参数:
        box (np.ndarray): shape ``(4,)``，归一化到 ``[0,1]`` 的矩形。
    返回:
        np.ndarray: shape ``(4,)``，水平翻转后的矩形，x分量取 ``1-x`` 并交换。
    """
    box = np.asarray(box, dtype=np.float64)
    return np.array([1.0 - box[2], box[1], 1.0 - box[0], box[3]])


def backfill_attention_to_full(
    teacher_attention: np.ndarray,
    crop_box: np.ndarray,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """Distribute crop-defined teacher attention onto the full-image grid.

    每个教师cell视为crop矩形内的均匀面积密度，按教师cell矩形与学生完整图
    cell矩形的相交面积占教师cell面积的比例分摊质量；crop外目标质量为0；
    最后归一化为总和1。坐标全部为完整图归一化 ``[0,1]``。

    参数:
        teacher_attention (np.ndarray): shape ``(grid_size, grid_size)``，
            已经过空间softmax、总和为1的教师注意力（行=y，列=x）。
        crop_box (np.ndarray): shape ``(4,)``，本次实际（翻转后）的
            ``[x1,y1,x2,y2]`` crop矩形，完整图归一化坐标。
        grid_size (int): 双方网格边长，冻结为7。
    返回:
        np.ndarray: shape ``(grid_size, grid_size)`` float32，总和为1的
            学生侧目标分布。
    """
    attention = np.asarray(teacher_attention, dtype=np.float64)
    if attention.shape != (grid_size, grid_size):
        raise ValueError(f"教师注意力shape应为{(grid_size, grid_size)}: {attention.shape}")
    x1, y1, x2, y2 = map(float, crop_box)
    for name, value in zip(("x1", "y1", "x2", "y2"), (x1, y1, x2, y2)):
        if value < -1e-6 or value > 1.0 + 1e-6:
            raise ValueError(f"crop_box.{name}超出[0,1]: {value}")
    x1, y1 = max(x1, 0.0), max(y1, 0.0)
    x2, y2 = min(x2, 1.0), min(y2, 1.0)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"crop_box无面积: {crop_box}")
    cell = 1.0 / grid_size
    cell_w = (x2 - x1) / grid_size
    cell_h = (y2 - y1) / grid_size
    teacher_cell_area = cell_w * cell_h
    target = np.zeros((grid_size, grid_size), dtype=np.float64)
    for ty in range(grid_size):
        ry1, ry2 = y1 + ty * cell_h, y1 + (ty + 1) * cell_h
        for tx in range(grid_size):
            mass = attention[ty, tx]
            if mass <= 0:
                continue
            rx1, rx2 = x1 + tx * cell_w, x1 + (tx + 1) * cell_w
            for sy in range(grid_size):
                oy = max(0.0, min(ry2, (sy + 1) * cell) - max(ry1, sy * cell))
                if oy <= 0:
                    continue
                for sx in range(grid_size):
                    ox = max(0.0, min(rx2, (sx + 1) * cell) - max(rx1, sx * cell))
                    if ox <= 0:
                        continue
                    target[sy, sx] += mass * (ox * oy) / teacher_cell_area
    total = float(target.sum())
    if total <= 0:
        raise ValueError("回填后目标质量为0，crop与网格异常")
    return (target / total).astype(np.float32)


def build_student_photometric():
    """Build the student-only P6 photometric augmentation (PIL in, PIL out).

    参数: 无。
    返回:
        torchvision.transforms.Compose: 依次执行RandomDownsampleUpsample
            (p=0.5, scale[0.5,0.85])、RandomJPEGCompression(p=0.5,
            quality[60,95])、GaussianBlur(kernel=5, p=0.3, sigma[0.1,1.2])、
            ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10)，
            逐项对齐M0-F正式配置；不含任何几何操作。
    """
    model_training_dir = PROJECT_ROOT / "程序/模型训练/正式代码"
    if str(model_training_dir) not in sys.path:
        sys.path.append(str(model_training_dir))
    from torchvision import transforms
    from train_utils import RandomDownsampleUpsample, RandomJPEGCompression

    return transforms.Compose([
        transforms.RandomApply(
            [RandomDownsampleUpsample(tuple(P6_PHOTOMETRIC["downsample_scale"]))],
            p=P6_PHOTOMETRIC["downsample_probability"],
        ),
        transforms.RandomApply(
            [RandomJPEGCompression(tuple(P6_PHOTOMETRIC["jpeg_quality"]))],
            p=P6_PHOTOMETRIC["jpeg_probability"],
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(
                kernel_size=P6_PHOTOMETRIC["gaussian_blur_kernel_size"],
                sigma=tuple(P6_PHOTOMETRIC["gaussian_blur_sigma"]),
            )],
            p=P6_PHOTOMETRIC["gaussian_blur_probability"],
        ),
        transforms.ColorJitter(
            brightness=P6_PHOTOMETRIC["color_jitter_brightness"],
            contrast=P6_PHOTOMETRIC["color_jitter_contrast"],
            saturation=P6_PHOTOMETRIC["color_jitter_saturation"],
        ),
    ])


def teacher_roi_tensor(image: Image.Image, crop_box: np.ndarray) -> torch.Tensor:
    """Crop the (possibly flipped) full image to the v3 ROI as a luma tensor.

    参数:
        image (PIL.Image): RGB完整图，尺寸任意。
        crop_box (np.ndarray): shape ``(4,)``，本次实际（翻转后）crop矩形，
            完整图归一化 ``[0,1]`` 坐标。
    返回:
        torch.Tensor: shape ``[3,224,224]``，luma复制三通道、ImageNet归一化，
            与MG1b教师训练输入口径完全一致（矩形直接resize，不正方形化）。
    """
    width, height = image.size
    crop = np.asarray(crop_box, dtype=np.float64)
    pixels = (crop * np.array([width, height, width, height])).round().astype(int)
    pixels[[0, 2]] = np.clip(pixels[[0, 2]], 0, width)
    pixels[[1, 3]] = np.clip(pixels[[1, 3]], 0, height)
    roi = image.crop(tuple(pixels)).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    ).convert("L").convert("RGB")
    tensor = TF.pil_to_tensor(roi).float().div(255.0)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def load_frozen_teacher(
    checkpoint_path: Path, device: torch.device
) -> tuple[AttentionPoolingTeacher, str]:
    """Load the frozen MG1b teacher after verifying its SHA256 binding.

    参数:
        checkpoint_path (Path): 教师checkpoint路径，只读，禁止修改。
        device (torch.device): 前向设备。
    返回:
        tuple: ``(model, sha256)``；model为eval()且requires_grad=False的
            AttentionPoolingTeacher，sha256为实际文件SHA（必须等于冻结值）。
    """
    sha = file_sha256(checkpoint_path)
    if sha != FROZEN_TEACHER_SHA256:
        raise ValueError(
            f"教师checkpoint SHA256不匹配: 实际{sha}，冻结值{FROZEN_TEACHER_SHA256}"
        )
    # 冻结本地资产且SHA已核验；metrics含numpy标量，不能用weights_only=True。
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "efficientnet_b0_attention_pooling_7x7":
        raise ValueError("教师checkpoint架构标记不是MG1b attention-pooling")
    if not checkpoint.get("eligible", False):
        raise ValueError("教师checkpoint未通过MG1b正式门槛，禁止作为MG2教师")
    model = AttentionPoolingTeacher(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, sha


def validate_teacher_cache(
    cache: dict,
    frame: pd.DataFrame,
    manifest_sha256: str,
    debug: bool,
) -> None:
    """Verify SHA binding and per-image alignment of one teacher cache.

    参数:
        cache (dict): torch.load读出的缓存对象，必须含format/SHA/order/entries。
        frame (pd.DataFrame): 当前train+val清单（debug时为debug子集），
            必须含sha256列；行顺序视为逐图对齐基准。
        manifest_sha256 (str): 当前manifest文件SHA256。
        debug (bool): 当前运行是否debug模式，必须与缓存构建时一致。
    返回: 无；任何绑定或对齐失败抛出ValueError。
    """
    if cache.get("format") != CACHE_FORMAT:
        raise ValueError(f"教师缓存格式标记不匹配: {cache.get('format')}")
    if cache.get("teacher_checkpoint_sha256") != FROZEN_TEACHER_SHA256:
        raise ValueError("教师缓存绑定的教师checkpoint SHA与冻结值不一致")
    if cache.get("manifest_sha256") != manifest_sha256:
        raise ValueError("教师缓存绑定的manifest SHA与当前清单不一致")
    if bool(cache.get("debug")) != bool(debug):
        raise ValueError("教师缓存debug标记与当前运行模式不一致")
    if cache.get("grid_size") != GRID_SIZE or cache.get("image_size") != IMAGE_SIZE:
        raise ValueError("教师缓存网格或输入尺寸与MG2协议不一致")
    entries = cache.get("entries")
    order = cache.get("order")
    if not isinstance(entries, dict) or not isinstance(order, list):
        raise ValueError("教师缓存缺少entries或order")
    frame_shas = [str(value) for value in frame.sha256]
    if order != frame_shas:
        raise ValueError("教师缓存逐图顺序与当前manifest行序不一致")
    if set(entries.keys()) != set(frame_shas):
        raise ValueError("教师缓存图片集合与当前manifest不一致")
    for sha in frame_shas:
        entry = entries[sha]
        for state in ("flip0", "flip1"):
            payload = entry.get(state)
            if payload is None:
                raise ValueError(f"教师缓存缺少{sha[:12]}的{state}条目")
            logits = torch.as_tensor(payload["logits"], dtype=torch.float32)
            attention = torch.as_tensor(payload["attention"], dtype=torch.float32)
            if logits.shape != (2,) or attention.shape != (GRID_SIZE * GRID_SIZE,):
                raise ValueError(f"教师缓存{sha[:12]}的{state}张量shape异常")
            if not torch.isfinite(logits).all() or not torch.isfinite(attention).all():
                raise ValueError(f"教师缓存{sha[:12]}的{state}含非有限值")
            if abs(float(attention.sum()) - 1.0) > 1e-4:
                raise ValueError(f"教师缓存{sha[:12]}的{state}注意力总和不为1")


def load_teacher_cache(
    cache_path: Path, frame: pd.DataFrame, manifest_sha256: str, debug: bool
) -> dict:
    """Load one teacher cache from disk and run the full binding validation.

    参数:
        cache_path (Path): 缓存 ``.pt`` 路径；不存在即失败（快速失败原则）。
        frame (pd.DataFrame): 当前train+val清单，见validate_teacher_cache。
        manifest_sha256 (str): 当前manifest文件SHA256。
        debug (bool): 当前是否debug模式。
    返回:
        dict: 校验通过的缓存对象。
    """
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"教师缓存不存在: {cache_path}；请先运行build_mage_mg2_teacher_cache.py"
        )
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    validate_teacher_cache(cache, frame, manifest_sha256, debug)
    return cache


class MG2StudentDataset(Dataset):
    """Serve full-color student inputs plus optional cached teacher targets.

    训练态：以 ``stable_uniform(sha256, epoch, seed, "flip")`` 决定同步水平
    翻转，先翻转完整图并同步变换 ``lesion_bbox`` 与 ``base_crop_box``；学生
    分支将完整图直接resize到224x224后施加P6光度增强；教师分支不做任何光度
    增强，只按翻转状态从缓存取logits与7x7 attention，并用翻转后crop_box回填。
    验证态：无翻转、无光度增强。坐标全部为完整图归一化 ``[0,1]``。

    构造参数:
        frame (pd.DataFrame): 单一split的v3清单行（train或val）。
        training (bool): 是否训练态（翻转+光度增强）。
        seed (int): 冻结seed42，用于翻转哈希。
        cache (dict | None): B/C组校验过的教师缓存；A组与纯学生评估传None。
        force_grayscale (bool): True时学生输入转为luma三通道（颜色扰动稳定
            性描述性副本），默认False。
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        training: bool,
        seed: int,
        cache: dict | None,
        force_grayscale: bool = False,
    ):
        self.frame = frame.reset_index(drop=True)
        self.training = training
        self.seed = seed
        self.cache = cache
        self.force_grayscale = force_grayscale
        self.epoch = 0
        self.photometric = build_student_photometric() if training else None

    def __len__(self) -> int:
        return len(self.frame)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by deterministic synchronized horizontal flips."""
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        """Return one sample; see class docstring for branch semantics.

        返回dict的关键键：
            image (Tensor [3,224,224]) 学生输入；label (long)；row_index (int)；
            sha256 (str)；teacher_logits (Tensor [2]，无缓存时为0)；
            target (Tensor [49]，癌图且有缓存时为回填分布，否则为0)；
            has_target (bool，仅癌图True)；bbox / crop_box (Tensor [4]，
            完整图归一化坐标，已按本次翻转同步)；overlap_bbox / overlap_crop
            (Tensor [7,7]，AiB与crop内外质量用)；lesion_area_fraction (float)。
        """
        row = self.frame.iloc[index]
        with Image.open(PROJECT_ROOT / row.image_relpath) as handle:
            image = handle.convert("RGB")
        sha = str(row.sha256)
        is_cancer = int(row.label) == 1
        bbox = np.array([
            row.bbox_x1_norm, row.bbox_y1_norm, row.bbox_x2_norm, row.bbox_y2_norm
        ], dtype=np.float64) if is_cancer else np.zeros(4, dtype=np.float64)
        crop = np.array([
            row.base_crop_x1, row.base_crop_y1, row.base_crop_x2, row.base_crop_y2
        ], dtype=np.float64)

        flipped = False
        if self.training:
            flipped = stable_uniform(sha, self.epoch, self.seed, "flip") < 0.5
        if flipped:
            image = TF.hflip(image)
            bbox = flip_box_horizontal(bbox)
            crop = flip_box_horizontal(crop)

        student = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        if self.training:
            student = self.photometric(student)
        if self.force_grayscale:
            student = student.convert("L").convert("RGB")
        tensor = TF.normalize(
            TF.pil_to_tensor(student).float().div(255.0), IMAGENET_MEAN, IMAGENET_STD
        )

        teacher_logits = torch.zeros(2, dtype=torch.float32)
        target = torch.zeros(GRID_SIZE * GRID_SIZE, dtype=torch.float32)
        if self.cache is not None:
            state = "flip1" if flipped else "flip0"
            entry = self.cache["entries"][sha][state]
            teacher_logits = torch.as_tensor(entry["logits"], dtype=torch.float32)
            if is_cancer:
                cached_attention = torch.as_tensor(
                    entry["attention"], dtype=torch.float32
                ).reshape(GRID_SIZE, GRID_SIZE).numpy()
                target = torch.from_numpy(
                    backfill_attention_to_full(cached_attention, crop).reshape(-1)
                )

        lesion_area = float(
            (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        ) if is_cancer else 0.0
        return {
            "image": tensor,
            "label": torch.tensor(int(row.label), dtype=torch.long),
            "row_index": index,
            "sha256": sha,
            "has_target": torch.tensor(is_cancer, dtype=torch.bool),
            "teacher_logits": teacher_logits,
            "target": target,
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "crop_box": torch.tensor(crop, dtype=torch.float32),
            "overlap_bbox": torch.tensor(
                cell_overlap_map(bbox) if is_cancer else np.zeros(
                    (GRID_SIZE, GRID_SIZE), dtype=np.float32
                )
            ),
            "overlap_crop": torch.tensor(cell_overlap_map(crop)),
            "lesion_area_fraction": torch.tensor(lesion_area, dtype=torch.float32),
        }


def compute_mg2_losses(
    arm: str,
    logits: torch.Tensor,
    attention: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    has_target: torch.Tensor,
    beta: float,
    label_smoothing: float,
) -> dict[str, torch.Tensor]:
    """Compute the frozen per-arm MG2 loss and its raw components.

    参数:
        arm (str): ``"A"``/``"B"``/``"C"``，唯一改变损失组合方式的开关。
        logits (torch.Tensor): 学生分类logits，shape ``[B,2]``。
        attention (torch.Tensor): 学生空间softmax注意力，shape ``[B,1,7,7]``。
        labels (torch.Tensor): 图像级标签，shape ``[B]``，取值0/1。
        teacher_logits (torch.Tensor): 缓存教师logits，shape ``[B,2]``；
            A组内容为0且不使用。
        targets (torch.Tensor): 回填后的教师注意力目标，shape ``[B,49]``；
            非癌样本为0且被 ``has_target`` 掩码排除。
        has_target (torch.Tensor): bool ``[B]``，仅癌图True。
        beta (float): 癌侧attention KD权重（占位或冻结值）。
        label_smoothing (float): CE标签平滑，训练0.1、评估0.0。
    返回:
        dict: ``total`` 及 ``ce``/``kd_logit``/``kd_attention``/加权项，
            均为标量Tensor；A组KD项为与计算图相连的0。
    """
    ce = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    zero = attention.sum() * 0.0
    kd_logit = zero
    weighted_logit = zero
    if arm in ("B", "C"):
        teacher_soft = F.softmax(teacher_logits / TAU, dim=1)
        student_log_soft = F.log_softmax(logits / TAU, dim=1)
        kd_logit = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean")
        weighted_logit = (1.0 - ALPHA) * (TAU**2) * kd_logit
    kd_attention = zero
    weighted_attention = zero
    if arm == "C" and bool(has_target.any()):
        target = targets[has_target]
        prediction = attention[has_target].flatten(1).clamp_min(1e-8)
        positive = target > 0
        terms = torch.where(
            positive, target * (target.clamp_min(1e-8).log() - prediction.log()),
            torch.zeros_like(target),
        )
        kd_attention = terms.sum(dim=1).mean()
        weighted_attention = beta * kd_attention
    total = ce if arm == "A" else ALPHA * ce + weighted_logit + weighted_attention
    return {
        "total": total,
        "ce": ce,
        "kd_logit": kd_logit,
        "kd_attention": kd_attention,
        "weighted_ce": ce if arm == "A" else ALPHA * ce,
        "weighted_logit": weighted_logit,
        "weighted_attention": weighted_attention,
    }


def lesion_fraction_tercile_bounds(train: pd.DataFrame) -> tuple[float, float]:
    """Freeze lesion-area-fraction tertiles from train cancer images only.

    参数:
        train (pd.DataFrame): split=train清单行，使用全图归一化
            ``bbox_x1_norm..bbox_y2_norm``。
    返回:
        tuple[float, float]: train癌图 ``lesion_area_fraction`` 的1/3与2/3
            分位数；此后冻结应用于val分层，不得用val重估。
    """
    cancer = train.loc[train.label.eq(1)]
    areas = (
        (cancer.bbox_x2_norm - cancer.bbox_x1_norm)
        * (cancer.bbox_y2_norm - cancer.bbox_y1_norm)
    ).to_numpy(dtype=np.float64)
    lower, upper = np.quantile(areas, [1.0 / 3.0, 2.0 / 3.0])
    return float(lower), float(upper)


def lesion_fraction_group(area: float, bounds: tuple[float, float]) -> str:
    """Assign one full-image lesion area fraction to frozen train tertiles."""
    if area <= bounds[0]:
        return "small"
    if area <= bounds[1]:
        return "medium"
    return "large"


def summarize_student_spatial(
    predictions: pd.DataFrame, bounds: tuple[float, float]
) -> dict:
    """Summarize full-image AiB/normalized AiB/PGA and crop inside/outside mass.

    参数:
        predictions (pd.DataFrame): val图像级预测，癌图须含aib/pga/
            lesion_area_fraction，所有图须含crop_inside_mass/crop_outside_mass。
        bounds (tuple[float, float]): train癌图冻结的病灶面积分位边界。
    返回:
        dict: 总体与small/medium/large分层的空间指标；``bbox_area>=0.99``
            的癌图不参与normalized AiB均值，其排除数单独记录；样本不足15张
            的分层只报告、不作任何门槛判定。
    """
    cancer = predictions.loc[predictions.label.eq(1)].copy()
    cancer["lesion_size_group"] = cancer.lesion_area_fraction.map(
        lambda value: lesion_fraction_group(float(value), bounds)
    )
    strata = {}
    for name in ("small", "medium", "large"):
        group = cancer.loc[cancer.lesion_size_group.eq(name)]
        count = int(len(group))
        strata[name] = {
            "images": count,
            "patients": int(group.patient_id.nunique()),
            "mean_aib": float(group.aib.mean()) if count else None,
            "mean_normalized_aib": (
                float(group.normalized_aib.mean())
                if int(group.normalized_aib.notna().sum()) else None
            ),
            "pga": float(group.pga.mean()) if count else None,
            "hard_gate_assessed": count >= STRATUM_MIN_IMAGES,
        }
    return {
        "cancer_images": int(len(cancer)),
        "normalized_aib_evaluable_images": int(cancer.normalized_aib.notna().sum()),
        "normalized_aib_excluded_bbox_area_ge_0_99": int(cancer.normalized_aib.isna().sum()),
        "mean_aib": float(cancer.aib.mean()),
        "mean_normalized_aib": float(cancer.normalized_aib.mean()),
        "pga": float(cancer.pga.mean()),
        "lesion_size_strata": strata,
        "crop_inside_mass_mean_all": float(predictions.crop_inside_mass.mean()),
        "crop_outside_mass_mean_all": float(predictions.crop_outside_mass.mean()),
        "crop_inside_mass_mean_cancer": float(cancer.crop_inside_mass.mean()),
        "crop_outside_mass_mean_cancer": float(cancer.crop_outside_mass.mean()),
    }


def subgroup_auc_report(
    predictions: pd.DataFrame, fields: tuple[str, ...] = ("source", "size_group")
) -> list[dict]:
    """Report per-subgroup image/patient AUC for manifest metadata fields.

    参数:
        predictions (pd.DataFrame): val图像级预测，含label与cancer_probability。
        fields (tuple[str, ...]): manifest中存在的来源/尺寸字段；缺失字段跳过。
    返回:
        list[dict]: 每个字段每个取值的样本数与图像/患者AUC；单一类别组
            的AUC记为None，只报告、不报错。
    """
    rows = []
    for field in fields:
        if field not in predictions.columns:
            continue
        values = predictions[field].fillna("__MISSING__").astype(str)
        for value, group in predictions.assign(_group=values).groupby("_group"):
            patient = patient_mean(group, "cancer_probability")
            rows.append({
                "field": field,
                "value": value,
                "images": int(len(group)),
                "patients": int(group.patient_id.nunique()),
                "image_auc": (
                    float(roc_auc_score(group.label, group.cancer_probability))
                    if group.label.nunique() == 2 else None
                ),
                "patient_auc": (
                    float(roc_auc_score(patient.label, patient.probability))
                    if patient.label.nunique() == 2 else None
                ),
            })
    return rows


def evaluate(
    model: AttentionPoolingTeacher,
    dataset: MG2StudentDataset,
    loader: DataLoader,
    device: torch.device,
    bounds: tuple[float, float],
) -> tuple[pd.DataFrame, dict]:
    """Evaluate the student on val and return predictions plus frozen metrics.

    参数:
        model (AttentionPoolingTeacher): 学生模型（与教师同架构）。
        dataset (MG2StudentDataset): val数据集（无翻转、无光度增强）。
        loader (DataLoader): 顺序不打乱的val加载器。
        device (torch.device): 前向设备。
        bounds (tuple[float, float]): train冻结病灶大小分位边界。
    返回:
        tuple: ``(predictions, metrics)``；predictions为图像级DataFrame，
            含概率、49个attention权重、AiB/normalized AiB/PGA/crop内外质量；
            metrics含val_loss(无平滑CE)、图像/患者AUC、阈值指标与空间汇总。
    """
    model.eval()
    rows = []
    loss_sum = 0.0
    seen = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits, attention = model(images)
            loss_sum += float(F.cross_entropy(logits, labels)) * len(labels)
            seen += len(labels)
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            attention_np = attention[:, 0].cpu().numpy().astype(np.float64)
            overlap_bbox = batch["overlap_bbox"].numpy().astype(np.float64)
            overlap_crop = batch["overlap_crop"].numpy().astype(np.float64)
            bbox_batch = batch["bbox"].numpy()
            has_target = batch["has_target"].tolist()
            for batch_index, row_index in enumerate(batch["row_index"].tolist()):
                source = dataset.frame.iloc[row_index]
                mass = attention_np[batch_index]
                crop_inside = float((mass * overlap_crop[batch_index]).sum())
                record = {
                    "row_index": row_index,
                    "relative_path": source.relative_path,
                    "patient_id": source.patient_id,
                    "label": int(source.label),
                    "split": source.split,
                    "cancer_probability": float(probabilities[batch_index]),
                    "crop_inside_mass": crop_inside,
                    "crop_outside_mass": float(1.0 - crop_inside),
                    "aib": np.nan,
                    "normalized_aib": np.nan,
                    "pga": np.nan,
                    "lesion_area_fraction": np.nan,
                }
                # 来源/尺寸等manifest元数据随预测落盘，供分层AUC报告。
                for field in ("source", "size_group", "style_group"):
                    if field in dataset.frame.columns:
                        record[field] = source[field]
                if has_target[batch_index]:
                    exact = np.array([
                        source.bbox_x1_norm, source.bbox_y1_norm,
                        source.bbox_x2_norm, source.bbox_y2_norm,
                    ], dtype=np.float64)
                    exact_area = float((exact[2] - exact[0]) * (exact[3] - exact[1]))
                    aib = float((mass * overlap_bbox[batch_index]).sum())
                    peak = int(mass.argmax())
                    peak_y, peak_x = divmod(peak, GRID_SIZE)
                    center_x = (peak_x + 0.5) / GRID_SIZE
                    center_y = (peak_y + 0.5) / GRID_SIZE
                    record.update({
                        "aib": aib,
                        "normalized_aib": (
                            (aib - exact_area) / (1.0 - exact_area)
                            if exact_area < NORMALIZED_AIB_MAX_BBOX_AREA else np.nan
                        ),
                        "pga": float(
                            float(bbox_batch[batch_index][0]) <= center_x
                            <= float(bbox_batch[batch_index][2])
                            and float(bbox_batch[batch_index][1]) <= center_y
                            <= float(bbox_batch[batch_index][3])
                        ),
                        "lesion_area_fraction": exact_area,
                    })
                for grid_y in range(GRID_SIZE):
                    for grid_x in range(GRID_SIZE):
                        record[f"attention_{grid_y}_{grid_x}"] = float(
                            attention_np[batch_index, grid_y, grid_x]
                        )
                rows.append(record)
    predictions = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
    patients = patient_mean(predictions, "cancer_probability")
    metrics = {
        "val_loss": loss_sum / seen,
        "val_image_auc": float(
            roc_auc_score(predictions.label, predictions.cancer_probability)
        ),
        "val_patient_auc": float(roc_auc_score(patients.label, patients.probability)),
        "image_threshold_metrics": sensitivity_threshold_metrics(
            predictions, "cancer_probability"
        ),
        "patient_threshold_metrics": sensitivity_threshold_metrics(
            patients, "probability"
        ),
        "spatial": summarize_student_spatial(predictions, bounds),
        "subgroup_auc": subgroup_auc_report(predictions),
    }
    return predictions, metrics


def evaluate_color_stability(
    model: AttentionPoolingTeacher,
    frame: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.Series, dict]:
    """Run the descriptive luma-grayscale val copy and compare probabilities.

    参数:
        model (AttentionPoolingTeacher): 已加载最佳权重的学生。
        frame (pd.DataFrame): val清单行。
        batch_size (int): 前向batch大小。
        num_workers (int): DataLoader worker数。
        device (torch.device): 前向设备。
        seed (int): 冻结seed（本路径无随机增强，仅保持接口一致）。
    返回:
        tuple: ``(gray_probabilities, summary)``；前者按row_index对齐的
            灰度副本癌概率Series，后者含mean/max绝对概率变化与灰度副本
            图像/患者AUC，仅作描述性报告。
    """
    dataset = MG2StudentDataset(frame, False, seed, None, force_grayscale=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
    )
    model.eval()
    gray = {}
    with torch.inference_mode():
        for batch in loader:
            logits, _ = model(batch["image"].to(device, non_blocking=True))
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for row_index, probability in zip(
                batch["row_index"].tolist(), probabilities
            ):
                gray[row_index] = float(probability)
    series = pd.Series(gray).sort_index()
    gray_frame = pd.DataFrame({
        "label": frame.label.to_numpy(),
        "patient_id": frame.patient_id.to_numpy(),
        "gray_probability": series.to_numpy(),
    })
    patients = patient_mean(
        gray_frame.rename(columns={"gray_probability": "cancer_probability"}),
        "cancer_probability",
    )
    summary = {
        "gray_val_image_auc": float(
            roc_auc_score(gray_frame.label, gray_frame.gray_probability)
        ),
        "gray_val_patient_auc": float(roc_auc_score(patients.label, patients.probability)),
    }
    return series, summary


def train_epoch(
    arm: str,
    model: AttentionPoolingTeacher,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    stage: str,
    beta: float,
) -> dict:
    """Train one epoch of one stage and return mean loss components.

    参数:
        arm (str): ``"A"``/``"B"``/``"C"``。
        model (AttentionPoolingTeacher): 学生模型。
        loader (DataLoader): 患者/类别平衡有放回采样的train加载器。
        optimizer (optim.Optimizer): 当前阶段的AdamW。
        device (torch.device): 训练设备。
        stage (str): ``"A"``冻结backbone；``"B"``解冻features[5:]与两个head。
        beta (float): arm C的attention KD权重。
    返回:
        dict: total/ce/kd_logit/kd_attention及加权项的样本均值。
    """
    set_train_mode(model, stage)
    names = ("total", "ce", "kd_logit", "kd_attention",
             "weighted_ce", "weighted_logit", "weighted_attention")
    totals = {name: 0.0 for name in names}
    seen = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        teacher_logits = batch["teacher_logits"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        has_target = batch["has_target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, attention = model(images)
        losses = compute_mg2_losses(
            arm, logits, attention, labels, teacher_logits, targets, has_target,
            beta, LABEL_SMOOTHING,
        )
        losses["total"].backward()
        optimizer.step()
        batch_size = len(labels)
        for name in names:
            totals[name] += float(losses[name].detach()) * batch_size
        seen += batch_size
    return {name: value / seen for name, value in totals.items()}


def selection_key(metrics: dict) -> tuple:
    """Return the frozen MG2 checkpoint ordering tuple.

    参数:
        metrics (dict): evaluate返回的val指标。
    返回:
        tuple: ``(val患者AUC, val图像AUC, -val_loss)``，越大越优。
    """
    return (
        metrics["val_patient_auc"], metrics["val_image_auc"], -metrics["val_loss"]
    )


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Refuse to overwrite existing formal products; creation happens at save.

    参数:
        output_dir (Path): 本arm输出目录。
        overwrite (bool): 显式``--overwrite``；仅当目录不含正式产物时生效。
    返回: 无；目录已存在且含config/checkpoint/预测CSV等正式产物、或非空且
        未指定overwrite时，抛出FileExistsError；检查通过后不在此处创建目录，
        避免训练中途失败留下残缺目录。
    """
    formal_markers = (
        "config.json", "training_history.csv",
        "val_image_predictions.csv", "val_patient_predictions.csv",
    )
    if output_dir.exists():
        has_formal = any((output_dir / name).exists() for name in formal_markers)
        has_formal = has_formal or any(output_dir.glob("*.pth"))
        if has_formal:
            raise FileExistsError(f"目标目录含正式产物，任何情况下拒绝覆盖: {output_dir}")
        if not overwrite and any(output_dir.iterdir()):
            raise FileExistsError(
                f"输出目录已存在且非空，拒绝覆盖（或显式--overwrite）: {output_dir}"
            )


def expected_calibration_batches(train_images: int, batch_size: int) -> int:
    """Return the frozen batch count of one complete calibration epoch.

    参数:
        train_images (int): 当前train图片数（正式2350，debug为子集）。
        batch_size (int): 冻结batch size（正式32）。
    返回:
        int: 一个完整有放回采样epoch的批次数，``ceil(train_images/batch_size)``
            （正式口径74批、2350次抽样）。
    """
    return math.ceil(train_images / batch_size)


def verify_calibration_epoch(
    batch_records: list[dict], train_images: int, batch_size: int
) -> None:
    """Verify the calibration covered exactly one complete sampling epoch.

    参数:
        batch_records (list[dict]): 逐批记录，每条含 ``sha256`` 样本列表。
        train_images (int): 当前train图片数（=有放回抽样总次数上限）。
        batch_size (int): 冻结batch size。
    返回: 无；批次数不足或超出一个完整epoch、或抽样总次数不等于
        ``len(train)`` 时抛出ValueError。
    """
    expected = expected_calibration_batches(train_images, batch_size)
    if len(batch_records) != expected:
        raise ValueError(
            f"校准epoch不完整: {len(batch_records)}批，期望恰好{expected}批"
        )
    draws = sum(len(record["sha256"]) for record in batch_records)
    if draws != train_images:
        raise ValueError(
            f"校准抽样次数异常: {draws}次，期望等于train图片数{train_images}"
        )


def calibration_sampling_audit(
    batch_records: list[dict], train: pd.DataFrame
) -> dict:
    """Build the with-replacement sampling audit fields for the calibration JSON.

    参数:
        batch_records (list[dict]): 逐批样本SHA记录（已经过epoch完整性校验）。
        train (pd.DataFrame): train清单，提供sha256→label/patient_id映射。
    返回:
        dict: 抽样总次数、唯一图片数、重复数统计（重复抽样次数、单图最大
            被抽次数、未被抽中图片数）、标签分布（按抽样次数计）、患者数
            分布（唯一患者数及每患者抽样次数min/max/mean），并显式注明
            有放回抽样不等于每张图各出现一次。
    """
    lookup = train.set_index(train.sha256.astype(str))
    shas = [sha for record in batch_records for sha in record["sha256"]]
    image_counts = Counter(shas)
    patient_counts = Counter(str(lookup.loc[sha].patient_id) for sha in shas)
    label_counts = Counter(int(lookup.loc[sha].label) for sha in shas)
    patient_draws = list(patient_counts.values())
    return {
        "total_draws": len(shas),
        "unique_images": len(image_counts),
        "duplicate_draws": len(shas) - len(image_counts),
        "max_draws_per_image": max(image_counts.values()),
        "images_not_drawn": int(len(train) - len(image_counts)),
        "label_distribution_draws": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "unique_patients": len(patient_counts),
        "patient_draws_min": int(min(patient_draws)),
        "patient_draws_max": int(max(patient_draws)),
        "patient_draws_mean": float(np.mean(patient_draws)),
        "sampling_note": (
            "有放回抽样不等于每张图各出现一次；本epoch允许重复也可能漏图，"
            "与正式训练采样分布完全一致"
        ),
    }


def calibrate_beta(args: argparse.Namespace) -> None:
    """Freeze beta from one complete calibration epoch before any val metric.

    用冻结教师缓存、seed42、固定epoch=1翻转状态和患者/类别平衡有放回
    采样器，运行一个完整校准epoch（正式口径74批、2350次有放回抽样）；
    学生为ImageNet初始化（eval模式前向），逐批记录完整非空间目标
    ``alpha*CE + (1-alpha)*tau^2*KL_logit`` 与癌侧attention KL，按
    ``beta = 0.25*median(nonspatial)/median(attention)`` 冻结。JSON保存
    每批SHA、抽样审计字段、校准设备与全部SHA绑定；自身SHA256写入同名
    ``.sha256`` sidecar。设备遵循 ``--device``，与正式训练保持一致。
    不读取val/test。

    参数:
        args (argparse.Namespace): 命令行参数；必须给出--calibrate-beta输出
            路径，B/C共用的教师缓存必须已存在。
    返回: 无；结果写入JSON文件及其 ``.sha256`` sidecar。
    """
    seed_everything(args.seed)
    manifest_sha = file_sha256(args.manifest.resolve())
    frame, _ = load_manifest(
        args.manifest.resolve(), args.v3_audit.resolve(), args.debug,
        args.debug_patients_per_class, args.seed,
    )
    train = frame.loc[frame.split.eq("train")].reset_index(drop=True)
    cache_path = resolve_teacher_cache_path(args)
    # 缓存逐图对齐基准是完整train+val清单，不能用train子集校验。
    cache = load_teacher_cache(cache_path, frame, manifest_sha, args.debug)
    dataset = MG2StudentDataset(train, True, args.seed, cache)
    dataset.set_epoch(1)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        patient_class_balanced_weights(train), len(train), replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=False,
        worker_init_fn=seed_worker, generator=generator,
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了--device cuda，但当前PyTorch无法使用CUDA")
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )
    model = AttentionPoolingTeacher(pretrained=True).to(device)
    model.eval()
    batches = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            teacher_logits = batch["teacher_logits"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            has_target = batch["has_target"].to(device, non_blocking=True)
            logits, attention = model(images)
            losses = compute_mg2_losses(
                "C", logits, attention, labels, teacher_logits,
                targets, has_target, 1.0, LABEL_SMOOTHING,
            )
            batches.append({
                "batch_index": batch_index,
                "sha256": list(batch["sha256"]),
                "nonspatial": float(losses["weighted_ce"] + losses["weighted_logit"]),
                "attention": (
                    float(losses["kd_attention"])
                    if bool(batch["has_target"].any()) else None
                ),
            })
    verify_calibration_epoch(batches, len(train), args.batch_size)
    nonspatial = np.array([item["nonspatial"] for item in batches], dtype=np.float64)
    attention_values = np.array(
        [item["attention"] for item in batches if item["attention"] is not None],
        dtype=np.float64,
    )
    beta = 0.25 * float(np.median(nonspatial)) / float(np.median(attention_values))
    payload = {
        "stage": BETA_CALIBRATION_STAGE,
        "rule": BETA_CALIBRATION_RULE,
        "seed": args.seed,
        "fixed_epoch": 1,
        "calibration_epoch": "one_complete_sampling_epoch",
        "expected_batches": expected_calibration_batches(len(train), args.batch_size),
        "batches": len(batches),
        "batch_size": args.batch_size,
        "device": str(device),
        "debug": args.debug,
        "median_nonspatial": float(np.median(nonspatial)),
        "median_attention": float(np.median(attention_values)),
        "batches_without_cancer": int(len(batches) - len(attention_values)),
        "beta": beta,
        "alpha": ALPHA,
        "tau": TAU,
        "manifest_sha256": manifest_sha,
        "teacher_checkpoint_sha256": FROZEN_TEACHER_SHA256,
        "teacher_cache": str(cache_path.resolve()),
        "teacher_cache_sha256": file_sha256(cache_path.resolve()),
        "sampling_audit": calibration_sampling_audit(batches, train),
        "batch_records": batches,
        "test_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
        **git_snapshot(),
    }
    output = args.calibrate_beta.resolve()
    if output.exists():
        raise FileExistsError(f"beta校准输出已存在，拒绝覆盖: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    print(
        f"beta校准完成: beta={beta:.6f}; 设备={device}; "
        f"{len(batches)}批/{payload['sampling_audit']['total_draws']}次抽样 "
        f"(唯一图{payload['sampling_audit']['unique_images']}张) -> {output}"
    )
    print(f"校准JSON SHA256: {file_sha256(output)} (sidecar: {sidecar})")


def verify_frozen_calibration(payload: dict, self_sha256: str) -> None:
    """Enforce the frozen-formal binding of the beta calibration JSON.

    参数:
        payload (dict): 已通过基础字段校验的校准payload。
        self_sha256 (str): 校准JSON文件的实际SHA256。
    返回: 无；以下任一不符即抛ValueError：文件SHA逐字符等于冻结值
        ``47179b62...``（重生成/sidecar替换场景同样拒绝）；重算
        ``beta == 0.25*median_nonspatial/median_attention``（容差1e-12）；
        ``batches==74``；``sampling_audit.total_draws==2350``；
        ``alpha==0.25``；``tau==4``；JSON内三项数据锁定标记为false。
        冻结JSON（2026-08-18）已确认含全部上述字段，故不做缺字段回退。
    """
    if self_sha256 != FROZEN_BETA_CALIBRATION_SHA256:
        raise ValueError(
            f"beta校准JSON SHA与冻结值不一致: 实际{self_sha256}，"
            f"冻结值{FROZEN_BETA_CALIBRATION_SHA256}"
        )
    recomputed = 0.25 * float(payload["median_nonspatial"]) / float(
        payload["median_attention"]
    )
    if not math.isclose(float(payload["beta"]), recomputed, abs_tol=1e-12):
        raise ValueError(
            f"beta重算核验失败: JSON记录{payload['beta']}，"
            f"0.25*median_nonspatial/median_attention={recomputed}"
        )
    if int(payload.get("batches", -1)) != 74:
        raise ValueError("正式校准JSON的batches必须为74（一个完整校准epoch）")
    audit = payload.get("sampling_audit") or {}
    if int(audit.get("total_draws", -1)) != 2350:
        raise ValueError("正式校准JSON的sampling_audit.total_draws必须为2350")
    if not math.isclose(float(payload.get("alpha", float("nan"))), ALPHA, abs_tol=1e-12):
        raise ValueError("正式校准JSON的alpha必须为0.25")
    if not math.isclose(float(payload.get("tau", float("nan"))), TAU, abs_tol=1e-12):
        raise ValueError("正式校准JSON的tau必须为4")
    for flag in ("test_evaluated", "internal_test_evaluated", "external_evaluated"):
        if payload.get(flag):
            raise ValueError(f"正式校准JSON的数据锁定标记{flag}必须为false")


def load_beta_calibration(
    calibration_path: Path,
    manifest_sha256: str,
    cache_path: Path,
    seed: int,
    batch_size: int,
    debug: bool,
) -> dict:
    """Load the frozen beta calibration JSON and enforce the SHA binding.

    参数:
        calibration_path (Path): 校准JSON路径；缺失即快速失败。
        manifest_sha256 (str): 当前运行实际manifest SHA256。
        cache_path (Path): 当前运行实际教师缓存路径，将重算SHA比对。
        seed (int): 当前运行seed。
        batch_size (int): 当前运行batch size。
        debug (bool): 当前是否debug模式，必须与校准时一致；debug模式跳过
            冻结SHA/beta重算等正式绑定（显式警告并记录于payload）。
    返回:
        dict: 校验通过的校准payload，附加 ``_self_path``/``_self_sha256``/
            ``_frozen_sha_verified``；规则标识、各SHA、seed、batch size或
            debug标记任一不符即抛ValueError。
    """
    if not calibration_path.is_file():
        raise FileNotFoundError(f"beta校准JSON不存在: {calibration_path}")
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if payload.get("stage") != BETA_CALIBRATION_STAGE:
        raise ValueError("beta校准JSON的stage标识不符")
    if payload.get("rule") != BETA_CALIBRATION_RULE:
        raise ValueError("beta校准JSON的规则标识不符")
    if payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("beta校准JSON的manifest SHA与当前运行不一致")
    if payload.get("teacher_checkpoint_sha256") != FROZEN_TEACHER_SHA256:
        raise ValueError("beta校准JSON的教师checkpoint SHA与冻结值不一致")
    expected_cache_sha = file_sha256(cache_path)
    if payload.get("teacher_cache_sha256") != expected_cache_sha:
        raise ValueError("beta校准JSON的教师缓存SHA与当前缓存不一致")
    if Path(payload.get("teacher_cache", "")).resolve() != cache_path.resolve():
        raise ValueError("beta校准JSON的教师缓存路径与当前运行不一致")
    if int(payload.get("seed", -1)) != seed:
        raise ValueError("beta校准JSON的seed与当前运行不一致")
    if int(payload.get("batch_size", -1)) != batch_size:
        raise ValueError("beta校准JSON的batch size与当前运行不一致")
    if bool(payload.get("debug")) != bool(debug):
        raise ValueError("beta校准JSON的debug标记与当前运行模式不一致")
    if not math.isfinite(float(payload.get("beta", float("nan")))):
        raise ValueError("beta校准JSON缺少有限beta值")
    payload["_self_path"] = str(calibration_path.resolve())
    payload["_self_sha256"] = file_sha256(calibration_path.resolve())
    if debug:
        print(
            "警告: debug模式跳过beta校准JSON的冻结SHA与beta重算核验，"
            "该绑定不构成正式依据。"
        )
        payload["_frozen_sha_verified"] = False
    else:
        verify_frozen_calibration(payload, payload["_self_sha256"])
        payload["_frozen_sha_verified"] = True
    return payload


def resolve_beta_binding(
    args: argparse.Namespace, manifest_sha256: str, cache_path: Path | None
) -> tuple[float, dict | None]:
    """Resolve the only allowed beta source for each arm (frozen defense).

    参数:
        args (argparse.Namespace): 命令行参数；``--beta``仅debug占位合法。
        manifest_sha256 (str): 当前manifest SHA256。
        cache_path (Path | None): B/C组已解析的教师缓存路径。
    返回:
        tuple: ``(beta, calibration_payload)``；A/B组payload为None。
            A/B组传入校准文件、C组正式缺校准文件、C组正式手工指定
            ``--beta``、或JSON任一SHA/seed/batch size绑定不符，均快速失败。
    """
    if args.arm in ("A", "B"):
        if args.beta_calibration_json is not None:
            raise ValueError("A/B组不涉及attention项，拒绝接受--beta-calibration-json")
        return (
            args.beta if args.beta is not None else DEFAULT_BETA
        ), None
    if args.beta is not None and not args.debug:
        raise ValueError(
            "C组正式训练禁止手工--beta；beta只能来自--beta-calibration-json"
        )
    if args.beta_calibration_json is None:
        if args.debug:
            return (
                args.beta if args.beta is not None else DEFAULT_BETA
            ), None
        raise ValueError("C组正式训练必须提供--beta-calibration-json")
    payload = load_beta_calibration(
        args.beta_calibration_json, manifest_sha256, cache_path,
        args.seed, args.batch_size, args.debug,
    )
    return float(payload["beta"]), payload


def resolve_teacher_cache_path(args: argparse.Namespace) -> Path:
    """Resolve the teacher cache path with explicit debug/formal separation."""
    if args.teacher_cache is not None:
        return args.teacher_cache.resolve()
    return (DEFAULT_DEBUG_TEACHER_CACHE if args.debug else DEFAULT_TEACHER_CACHE).resolve()


def main() -> None:
    """Run one formal MG2 arm or the pre-val beta calibration step."""
    args = parse_args()
    if args.calibrate_beta is not None:
        if args.debug:
            args.num_workers = 0
        calibrate_beta(args)
        return
    if args.arm is None:
        raise ValueError("训练模式必须显式指定 --arm A|B|C")
    if args.seed != 42 and not args.debug:
        raise ValueError("MG2预注册只允许正式seed42")
    seed_everything(args.seed)
    manifest_sha = file_sha256(args.manifest.resolve())
    frame, _ = load_manifest(
        args.manifest.resolve(), args.v3_audit.resolve(), args.debug,
        args.debug_patients_per_class, args.seed,
    )
    if not set(frame.split.unique()).issubset({"train", "val"}):
        raise ValueError("MG2只允许train/val；manifest含其他split时必须显式排除")
    train = frame.loc[frame.split.eq("train")].reset_index(drop=True)
    val = frame.loc[frame.split.eq("val")].reset_index(drop=True)
    bounds = lesion_fraction_tercile_bounds(train)

    cache = None
    cache_path = None
    if args.arm in ("B", "C"):
        cache_path = resolve_teacher_cache_path(args)
        cache = load_teacher_cache(cache_path, frame, manifest_sha, args.debug)
    beta, beta_calibration = resolve_beta_binding(args, manifest_sha, cache_path)

    run_name = args.run_name or f"mg2_arm{args.arm.lower()}_efficientnet_b0_seed{args.seed}"
    if args.debug:
        run_name += "_debug"
        args.stage_a_epochs = min(args.stage_a_epochs, 1)
        args.stage_b_epochs = min(args.stage_b_epochs, 1)
        args.num_workers = 0
    output_root = args.output_root
    if output_root is None:
        output_root = DEFAULT_DEBUG_OUTPUT if args.debug else DEFAULT_OUTPUT
    output_dir = output_root.resolve() / run_name
    prepare_output_dir(output_dir, args.overwrite)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了--device cuda，但当前PyTorch无法使用CUDA")
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )
    train_dataset = MG2StudentDataset(train, True, args.seed, cache)
    val_dataset = MG2StudentDataset(val, False, args.seed, cache)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        patient_class_balanced_weights(train), len(train), replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker, generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    # 三组臂在同一seed下构造，ImageNet backbone相同、两个head初始化相同。
    model = AttentionPoolingTeacher(pretrained=True).to(device)
    print(
        f"设备={device}; arm={args.arm}; seed={args.seed}; "
        f"train={len(train)}张/{train.patient_id.nunique()}人; "
        f"val={len(val)}张/{val.patient_id.nunique()}人; "
        f"病灶面积三分位={bounds}; beta={beta}"
    )

    history = []
    best = {"key": None, "state": None, "stage": None, "epoch": None, "metrics": None}
    best_stage_a_state = None
    for stage, epochs, learning_rate in (
        ("A", args.stage_a_epochs, args.stage_a_lr),
        ("B", args.stage_b_epochs, args.stage_b_lr),
    ):
        if stage == "B" and best_stage_a_state is not None:
            model.load_state_dict(best_stage_a_state, strict=True)
        set_stage(model, stage)
        optimizer = optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate, weight_decay=args.weight_decay,
        )
        no_improve = 0
        stage_best_key = None
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_dataset.set_epoch(epoch)
            train_losses = train_epoch(
                args.arm, model, train_loader, optimizer, device, stage, beta
            )
            _, metrics = evaluate(model, val_dataset, val_loader, device, bounds)
            key = selection_key(metrics)
            if best["key"] is None or key > best["key"]:
                best = {
                    "key": key, "state": deepcopy(model.state_dict()),
                    "stage": stage, "epoch": epoch, "metrics": metrics,
                }
            if stage_best_key is None or key > stage_best_key:
                stage_best_key = key
                no_improve = 0
                if stage == "A":
                    best_stage_a_state = deepcopy(model.state_dict())
            else:
                no_improve += 1
            history.append({
                "stage": stage, "epoch": epoch,
                "elapsed_seconds": time.time() - start,
                **{f"train_{name}": value for name, value in train_losses.items()},
                "val_loss": metrics["val_loss"],
                "val_image_auc": metrics["val_image_auc"],
                "val_patient_auc": metrics["val_patient_auc"],
                "val_mean_aib": metrics["spatial"]["mean_aib"],
                "val_mean_normalized_aib": metrics["spatial"]["mean_normalized_aib"],
                "val_pga": metrics["spatial"]["pga"],
                "val_crop_inside_mass": metrics["spatial"]["crop_inside_mass_mean_all"],
            })
            print(
                f"stage{stage} {epoch:02d}/{epochs} | train={train_losses['total']:.4f} "
                f"(ce={train_losses['ce']:.4f}, kd_logit={train_losses['kd_logit']:.4f}, "
                f"kd_att={train_losses['kd_attention']:.4f}) | "
                f"val_loss={metrics['val_loss']:.4f} "
                f"image AUC={metrics['val_image_auc']:.4f} "
                f"patient AUC={metrics['val_patient_auc']:.4f} | "
                f"nAiB={metrics['spatial']['mean_normalized_aib']:.4f} "
                f"PGA={metrics['spatial']['pga']:.4f}"
            )
            if stage == "B" and no_improve >= args.patience:
                print(f"stageB early stop: 连续{args.patience}轮无冻结排序改善")
                break

    model.load_state_dict(best["state"], strict=True)
    final_predictions, final_metrics = evaluate(model, val_dataset, val_loader, device, bounds)
    if abs(final_metrics["val_patient_auc"] - best["metrics"]["val_patient_auc"]) > 1e-12:
        raise RuntimeError("MG2最佳checkpoint复算不一致")
    gray_series, gray_summary = evaluate_color_stability(
        model, val, args.batch_size, args.num_workers, device, args.seed
    )
    final_predictions["cancer_probability_grayscale"] = (
        final_predictions.row_index.map(gray_series)
    )
    delta = (
        final_predictions.cancer_probability
        - final_predictions.cancer_probability_grayscale
    ).abs()
    final_metrics["color_stability"] = {
        **gray_summary,
        "mean_abs_probability_delta": float(delta.mean()),
        "max_abs_probability_delta": float(delta.max()),
        "descriptive_only": True,
    }
    patient_predictions = patient_mean(final_predictions, "cancer_probability")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"mg2_arm{args.arm.lower()}_best_student.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": "efficientnet_b0_attention_pooling_7x7",
        "arm": args.arm,
        "seed": args.seed,
        "best_stage": best["stage"],
        "best_epoch": best["epoch"],
        "metrics": final_metrics,
        "manifest_sha256": manifest_sha,
        "teacher_checkpoint_sha256": FROZEN_TEACHER_SHA256,
    }, checkpoint_path)
    final_predictions.to_csv(
        output_dir / "val_image_predictions.csv", index=False, encoding="utf-8-sig"
    )
    patient_predictions.to_csv(
        output_dir / "val_patient_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    config = {
        "stage": "MG2_full_image_student_distillation",
        "arm": args.arm,
        "arm_loss": {
            "A": "CE only",
            "B": "alpha*CE + (1-alpha)*tau^2*KL(teacher_soft||student_soft)",
            "C": "B + beta*mean_cancer KL(backfilled_teacher_attention||student_attention)",
        }[args.arm],
        "debug": args.debug,
        "seed": args.seed,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha,
        "v3_audit": str(args.v3_audit.resolve()),
        "v3_audit_sha256": file_sha256(args.v3_audit.resolve()),
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": FROZEN_TEACHER_SHA256,
        "teacher_cache": str(cache_path) if cache_path else None,
        "teacher_cache_sha256": (
            file_sha256(cache_path) if cache_path else None
        ),
        "train_images": int(len(train)),
        "train_patients": int(train.patient_id.nunique()),
        "val_images": int(len(val)),
        "val_patients": int(val.patient_id.nunique()),
        "lesion_fraction_tercile_bounds_from_train": list(bounds),
        "input_protocol": {
            "student": "full RGB image resized directly to 224x224; no RandomResizedCrop/rotation",
            "teacher": "v3 crop ROI resized to 224x224 luma 3ch; no photometric augmentation",
            "shared_geometry": "synchronized horizontal flip p=0.5 only",
            "student_photometric_P6": P6_PHOTOMETRIC,
            "bbox_jitter": False,
        },
        "architecture": {
            "backbone": "efficientnet_b0",
            "feature_map": [1280, GRID_SIZE, GRID_SIZE],
            "attention": "1x1 conv then 49-position spatial softmax",
            "pooling": "attention-weighted sum only; no GAP bypass",
            "init": "ImageNet",
        },
        "training": {
            "device": str(device),
            "batch_size": args.batch_size,
            "stage_a_epochs": args.stage_a_epochs,
            "stage_b_epochs": args.stage_b_epochs,
            "stage_a_lr": args.stage_a_lr,
            "stage_b_lr": args.stage_b_lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "optimizer": "AdamW",
            "label_smoothing": LABEL_SMOOTHING,
            "alpha": ALPHA,
            "tau": TAU,
            "beta": beta,
            "beta_source": (
                "calibration_json" if beta_calibration is not None
                else ("placeholder_debug_only" if args.debug else "placeholder")
            ),
            "beta_frozen": beta_calibration is not None,
            "beta_calibration_json": (
                beta_calibration["_self_path"] if beta_calibration else None
            ),
            "beta_calibration_json_sha256": (
                beta_calibration["_self_sha256"] if beta_calibration else None
            ),
            "beta_calibration_frozen_sha256_verified": (
                beta_calibration["_frozen_sha_verified"] if beta_calibration else None
            ),
            "sampler": "patient_and_class_balanced_with_replacement",
            "selection": "val patient AUC, then image AUC, then lower val loss",
        },
        "best_stage": best["stage"],
        "best_epoch": best["epoch"],
        "metrics": final_metrics,
        "normalized_aib_excluded_bbox_area_ge_0_99": final_metrics["spatial"][
            "normalized_aib_excluded_bbox_area_ge_0_99"
        ],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "test_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
        **git_snapshot(),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"MG2 arm{args.arm}完成: best=stage{best['stage']} epoch{best['epoch']} "
        f"patient AUC={final_metrics['val_patient_auc']:.4f} "
        f"image AUC={final_metrics['val_image_auc']:.4f} "
        f"nAiB={final_metrics['spatial']['mean_normalized_aib']:.4f} "
        f"PGA={final_metrics['spatial']['pga']:.4f}"
    )
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
