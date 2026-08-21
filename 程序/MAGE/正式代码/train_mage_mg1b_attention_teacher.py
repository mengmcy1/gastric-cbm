#!/usr/bin/env python3
"""训练预注册的MG1b框监督注意力汇聚教师模型。

输入为冻结的独立轴动态ROI v3灰度裁图。EfficientNet-B0负责提取视觉特征，
注意力头同时接受癌图病灶框监督并作为分类前唯一的特征汇聚路径；非癌图只接受
分类监督。脚本只读取训练和验证队列，满足全部冻结门槛的权重才可进入MG2。
"""

from __future__ import annotations

import argparse
import json
import math
import time
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
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
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
    seed_worker,
    sensitivity_threshold_metrics,
    stable_uniform,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "结果/MAGE/MG1b注意力池化教师_20260818/正式验证集筛选"
GRID_SIZE = 7
LABEL_SMOOTHING = 0.10
ATTENTION_WEIGHT = 0.25
PATIENT_AUC_FLOOR = 0.8472
IMAGE_AUC_FLOOR = 0.8042
NORMALIZED_AIB_FLOOR = 0.30
PGA_FLOOR = 0.80
STRATUM_PGA_FLOOR = 0.70
STRATUM_MIN_IMAGES = 15
NORMALIZED_AIB_MAX_BBOX_AREA = 0.99


def parse_args() -> argparse.Namespace:
    """解析冻结的MG1b优化参数和输出参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--v3-audit", type=Path, default=DEFAULT_V3_AUDIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="mg1b_attention_efficientnet_b0_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stage-a-epochs", type=int, default=5)
    parser.add_argument("--stage-b-epochs", type=int, default=20)
    parser.add_argument("--stage-a-lr", type=float, default=1e-3)
    parser.add_argument("--stage-b-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-patients-per-class", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def bbox_in_crop(row: pd.Series) -> np.ndarray:
    """将原图归一化病灶框换算为v3裁图内的相对坐标。"""
    crop = np.array([
        row.base_crop_x1, row.base_crop_y1, row.base_crop_x2, row.base_crop_y2
    ], dtype=np.float64)
    width = crop[2] - crop[0]
    height = crop[3] - crop[1]
    if width <= 0 or height <= 0:
        raise ValueError(f"无效v3 crop: {row.relative_path}")
    bbox = np.array([
        (row.bbox_x1_norm - crop[0]) / width,
        (row.bbox_y1_norm - crop[1]) / height,
        (row.bbox_x2_norm - crop[0]) / width,
        (row.bbox_y2_norm - crop[1]) / height,
    ], dtype=np.float64)
    bbox = np.clip(bbox, 0.0, 1.0)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"癌图bbox映射到crop后无面积: {row.relative_path}")
    return bbox


def cell_overlap_map(bbox: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    """计算病灶框覆盖每个网格单元的面积比例。"""
    overlap = np.zeros((grid_size, grid_size), dtype=np.float32)
    x1, y1, x2, y2 = map(float, bbox)
    cell = 1.0 / grid_size
    for row in range(grid_size):
        cy1, cy2 = row * cell, (row + 1) * cell
        for column in range(grid_size):
            cx1, cx2 = column * cell, (column + 1) * cell
            width = max(0.0, min(x2, cx2) - max(x1, cx1))
            height = max(0.0, min(y2, cy2) - max(y1, cy1))
            overlap[row, column] = width * height / (cell * cell)
    return overlap


def target_distribution(overlap: np.ndarray) -> np.ndarray:
    """将网格重叠面积归一化为病灶框空间监督分布。"""
    total = float(overlap.sum())
    if total <= 0:
        raise ValueError("bbox与7x7网格没有相交面积")
    return overlap / total


def lesion_tercile_bounds(train: pd.DataFrame) -> tuple[float, float]:
    """只使用训练癌图冻结病灶面积三分位边界。"""
    areas = []
    for _, row in train.loc[train.label.eq(1)].iterrows():
        bbox = bbox_in_crop(row)
        areas.append(float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
    lower, upper = np.quantile(np.asarray(areas), [1.0 / 3.0, 2.0 / 3.0])
    return float(lower), float(upper)


def lesion_group(area: float, bounds: tuple[float, float]) -> str:
    """按冻结边界为裁图内病灶面积分组。"""
    if area <= bounds[0]:
        return "small"
    if area <= bounds[1]:
        return "medium"
    return "large"


class MG1bDataset(Dataset):
    """读取灰度ROI，并同步准备分类标签和病灶框空间监督。"""

    def __init__(self, frame: pd.DataFrame, training: bool, seed: int):
        """保存数据清单，并初始化可复现的数据增强状态。"""
        self.frame = frame.reset_index(drop=True)
        self.training = training
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        """返回当前数据划分的样本数。"""
        return len(self.frame)

    def set_epoch(self, epoch: int) -> None:
        """设置确定性水平翻转所使用的轮次。"""
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        """读取一张ROI，并同步生成分类标签和病灶空间监督。"""
        row = self.frame.iloc[index]
        with Image.open(PROJECT_ROOT / row.image_relpath) as handle:
            image = handle.convert("RGB")
        width, height = image.size
        crop = np.array([
            row.base_crop_x1, row.base_crop_y1, row.base_crop_x2, row.base_crop_y2
        ], dtype=np.float64)
        pixels = (crop * np.array([width, height, width, height])).round().astype(int)
        pixels[[0, 2]] = np.clip(pixels[[0, 2]], 0, width)
        pixels[[1, 3]] = np.clip(pixels[[1, 3]], 0, height)
        roi = image.crop(tuple(pixels)).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        ).convert("L").convert("RGB")
        tensor = TF.pil_to_tensor(roi).float().div(255.0)

        has_target = int(row.label) == 1
        bbox = bbox_in_crop(row) if has_target else np.zeros(4, dtype=np.float64)
        epoch = self.epoch if self.training else -1
        if self.training and stable_uniform(row.sha256, epoch, self.seed, "flip") < 0.5:
            tensor = torch.flip(tensor, dims=(2,))
            if has_target:
                bbox = np.array([1.0 - bbox[2], bbox[1], 1.0 - bbox[0], bbox[3]])

        overlap = cell_overlap_map(bbox) if has_target else np.zeros(
            (GRID_SIZE, GRID_SIZE), dtype=np.float32
        )
        target = target_distribution(overlap) if has_target else overlap.copy()
        area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) if has_target else 0.0
        return {
            "image": TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD),
            "label": torch.tensor(int(row.label), dtype=torch.long),
            "row_index": index,
            "has_target": torch.tensor(has_target, dtype=torch.bool),
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "overlap": torch.tensor(overlap, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "lesion_area": torch.tensor(area, dtype=torch.float32),
        }


class AttentionPoolingTeacher(nn.Module):
    """使用EfficientNet、监督注意力头和分类头完成教师前向计算。"""

    def __init__(self, pretrained: bool):
        """建立EfficientNet特征提取器、注意力头和二分类头。"""
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = efficientnet_b0(weights=weights)
        channels = base.classifier[1].in_features
        self.features = base.features
        self.attention_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(channels, 2))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """按“特征提取 -> 注意力汇聚 -> 分类”完成一次前向计算。"""
        feature = self.features(images)
        attention = torch.softmax(self.attention_head(feature).flatten(1), dim=1)
        attention = attention.view(-1, 1, GRID_SIZE, GRID_SIZE)
        pooled = (feature * attention).sum(dim=(2, 3))
        return self.classifier(pooled), attention


def set_stage(model: AttentionPoolingTeacher, stage: str) -> None:
    """先冻结全部参数，再按预注册阶段启用指定模块。"""
    for parameter in model.parameters():
        parameter.requires_grad = False
    if stage == "A":
        modules = (model.attention_head, model.classifier)
    elif stage == "B":
        modules = (*model.features[5:], model.attention_head, model.classifier)
    else:
        raise ValueError(f"未知MG1b stage: {stage}")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def set_train_mode(model: AttentionPoolingTeacher, stage: str) -> None:
    """保持冻结批归一化层不变，并启用当前阶段模块和两个头。"""
    model.eval()
    if stage == "B":
        for block in model.features[5:]:
            block.train()
    model.attention_head.train()
    model.classifier.train()


def compute_losses(
    logits: torch.Tensor,
    attention: torch.Tensor,
    labels: torch.Tensor,
    targets: torch.Tensor,
    has_target: torch.Tensor,
    label_smoothing: float,
) -> dict[str, torch.Tensor]:
    """计算分类交叉熵和仅用于癌图的病灶框分布KL损失。"""
    classification = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    if bool(has_target.any()):
        target = targets[has_target].flatten(1)
        prediction = attention[has_target].flatten(1).clamp_min(1e-8)
        positive = target > 0
        terms = torch.where(
            positive, target * (target.clamp_min(1e-8).log() - prediction.log()),
            torch.zeros_like(target),
        )
        spatial = terms.sum(dim=1).mean()
    else:
        spatial = attention.sum() * 0.0
    weighted_spatial = ATTENTION_WEIGHT * spatial
    return {
        "total": classification + weighted_spatial,
        "classification": classification,
        "spatial_kl": spatial,
        "weighted_spatial": weighted_spatial,
    }


def spatial_batch_metrics(
    attention: torch.Tensor,
    overlap: torch.Tensor,
    bbox: torch.Tensor,
    has_target: torch.Tensor,
) -> list[dict]:
    """计算癌图AiB、归一化AiB和注意力峰值命中指标。"""
    output = []
    for index in torch.where(has_target)[0].tolist():
        mass = attention[index, 0]
        aib = float((mass * overlap[index]).sum())
        peak = int(mass.argmax())
        peak_y, peak_x = divmod(peak, GRID_SIZE)
        center_x = (peak_x + 0.5) / GRID_SIZE
        center_y = (peak_y + 0.5) / GRID_SIZE
        box = bbox[index]
        pga = float(
            float(box[0]) <= center_x <= float(box[2])
            and float(box[1]) <= center_y <= float(box[3])
        )
        output.append({
            "batch_index": index, "aib": aib, "pga": pga,
        })
    return output


def summarize_spatial(rows: pd.DataFrame, bounds: tuple[float, float]) -> dict:
    """汇总整体及按冻结病灶尺寸分层的空间对齐结果。"""
    cancer = rows.loc[rows.label.eq(1)].copy()
    cancer["lesion_size_group"] = cancer.lesion_area.map(lambda value: lesion_group(value, bounds))
    strata = {}
    group_pass = True
    for name in ("small", "medium", "large"):
        group = cancer.loc[cancer.lesion_size_group.eq(name)]
        count = int(len(group))
        pga = float(group.pga.mean()) if count else None
        assessed = count >= STRATUM_MIN_IMAGES
        passed = (pga >= STRATUM_PGA_FLOOR) if assessed else None
        if assessed:
            group_pass = group_pass and bool(passed)
        strata[name] = {
            "images": count,
            "patients": int(group.patient_id.nunique()),
            "mean_aib": float(group.aib.mean()) if count else None,
            "mean_normalized_aib": float(group.normalized_aib.mean()) if count else None,
            "pga": pga,
            "hard_gate_assessed": assessed,
            "passed_pga_gate": passed,
        }
    return {
        "cancer_images": int(len(cancer)),
        "normalized_aib_evaluable_images": int(cancer.normalized_aib.notna().sum()),
        "normalized_aib_excluded_bbox_area_ge_0_99": int(cancer.normalized_aib.isna().sum()),
        "mean_aib": float(cancer.aib.mean()),
        "mean_normalized_aib": float(cancer.normalized_aib.mean()),
        "pga": float(cancer.pga.mean()),
        "lesion_size_strata": strata,
        "passed_stratum_pga_gates": group_pass,
    }


def evaluate(
    model: AttentionPoolingTeacher,
    dataset: MG1bDataset,
    loader: DataLoader,
    device: torch.device,
    bounds: tuple[float, float],
) -> tuple[pd.DataFrame, dict]:
    """评估模型并返回图像预测、分类指标和空间指标。"""
    model.eval()
    rows = []
    totals = {name: 0.0 for name in ("total", "classification", "spatial_kl", "weighted_spatial")}
    seen = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            has_target = batch["has_target"].to(device, non_blocking=True)
            overlap = batch["overlap"].to(device, non_blocking=True)
            bbox = batch["bbox"].to(device, non_blocking=True)
            logits, attention = model(images)
            losses = compute_losses(logits, attention, labels, targets, has_target, 0.0)
            batch_size = len(labels)
            for name, value in losses.items():
                totals[name] += float(value) * batch_size
            seen += batch_size
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            attention_values = attention[:, 0].cpu().numpy()
            spatial = {item["batch_index"]: item for item in spatial_batch_metrics(
                attention, overlap, bbox, has_target
            )}
            for batch_index, (row_index, probability) in enumerate(zip(
                batch["row_index"].tolist(), probabilities
            )):
                source = dataset.frame.iloc[row_index]
                item = spatial.get(batch_index, {})
                if int(source.label) == 1:
                    exact_bbox = bbox_in_crop(source)
                    exact_area = float(
                        (exact_bbox[2] - exact_bbox[0]) * (exact_bbox[3] - exact_bbox[1])
                    )
                    normalized_aib = (
                        (item["aib"] - exact_area) / (1.0 - exact_area)
                        if exact_area < NORMALIZED_AIB_MAX_BBOX_AREA else np.nan
                    )
                else:
                    exact_area = np.nan
                    normalized_aib = np.nan
                record = {
                    "row_index": row_index,
                    "relative_path": source.relative_path,
                    "patient_id": source.patient_id,
                    "label": int(source.label),
                    "split": source.split,
                    "cancer_probability": float(probability),
                    "aib": item.get("aib", np.nan),
                    "normalized_aib": normalized_aib,
                    "pga": item.get("pga", np.nan),
                    "lesion_area": exact_area,
                }
                for grid_y in range(GRID_SIZE):
                    for grid_x in range(GRID_SIZE):
                        record[f"attention_{grid_y}_{grid_x}"] = float(
                            attention_values[batch_index, grid_y, grid_x]
                        )
                rows.append(record)
    predictions = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
    patients = patient_mean(predictions, "cancer_probability")
    spatial = summarize_spatial(predictions, bounds)
    metrics = {
        "val_total_loss": totals["total"] / seen,
        "val_classification_loss": totals["classification"] / seen,
        "val_spatial_kl": totals["spatial_kl"] / seen,
        "val_weighted_spatial": totals["weighted_spatial"] / seen,
        "val_image_auc": float(roc_auc_score(predictions.label, predictions.cancer_probability)),
        "val_patient_auc": float(roc_auc_score(patients.label, patients.probability)),
        "image_threshold_metrics": sensitivity_threshold_metrics(predictions, "cancer_probability"),
        "patient_threshold_metrics": sensitivity_threshold_metrics(patients, "probability"),
        "spatial": spatial,
    }
    return predictions, metrics


def eligibility(metrics: dict) -> dict:
    """对一个验证轮次应用全部冻结的MG1b产物门槛。"""
    gates = {
        "patient_auc": metrics["val_patient_auc"] >= PATIENT_AUC_FLOOR,
        "image_auc": metrics["val_image_auc"] >= IMAGE_AUC_FLOOR,
        "normalized_aib": metrics["spatial"]["mean_normalized_aib"] >= NORMALIZED_AIB_FLOOR,
        "pga": metrics["spatial"]["pga"] >= PGA_FLOOR,
        "lesion_size_pga": metrics["spatial"]["passed_stratum_pga_gates"],
    }
    return {"eligible": bool(all(gates.values())), "gates": gates}


def selection_key(metrics: dict, eligible: bool) -> tuple:
    """优先选择合格产物，再按冻结的指标优先级排序。"""
    return (
        int(eligible), metrics["val_patient_auc"], metrics["val_image_auc"],
        metrics["spatial"]["mean_normalized_aib"], metrics["spatial"]["pga"],
        -metrics["val_total_loss"],
    )


def train_epoch(
    model: AttentionPoolingTeacher,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    stage: str,
) -> dict:
    """训练一个轮次，并按原始尺度报告各项损失。"""
    set_train_mode(model, stage)
    totals = {name: 0.0 for name in ("total", "classification", "spatial_kl", "weighted_spatial")}
    seen = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        has_target = batch["has_target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, attention = model(images)
        losses = compute_losses(
            logits, attention, labels, targets, has_target, LABEL_SMOOTHING
        )
        losses["total"].backward()
        optimizer.step()
        batch_size = len(labels)
        for name, value in losses.items():
            totals[name] += float(value.detach()) * batch_size
        seen += batch_size
    return {name: value / seen for name, value in totals.items()}


def run_self_test() -> None:
    """自测重叠几何、翻转同步、损失掩码和无旁路模型输出。"""
    bbox = np.array([0.2, 0.3, 0.8, 0.9])
    overlap = cell_overlap_map(bbox)
    target = target_distribution(overlap)
    assert np.isclose(target.sum(), 1.0)
    assert np.isclose(overlap.sum() / GRID_SIZE**2, 0.6 * 0.6, atol=1e-6)
    flipped = np.array([1.0 - bbox[2], bbox[1], 1.0 - bbox[0], bbox[3]])
    assert np.allclose(cell_overlap_map(flipped), np.fliplr(overlap))

    logits = torch.zeros(2, 2, requires_grad=True)
    attention = torch.full((2, 1, GRID_SIZE, GRID_SIZE), 1.0 / GRID_SIZE**2, requires_grad=True)
    targets = torch.stack([
        torch.tensor(target), torch.zeros(GRID_SIZE, GRID_SIZE),
    ])
    losses = compute_losses(
        logits, attention, torch.tensor([1, 0]), targets,
        torch.tensor([True, False]), LABEL_SMOOTHING,
    )
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    model = AttentionPoolingTeacher(pretrained=False)
    output, maps = model(torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert output.shape == (2, 2) and maps.shape == (2, 1, GRID_SIZE, GRID_SIZE)
    assert torch.allclose(maps.sum(dim=(1, 2, 3)), torch.ones(2), atol=1e-6)
    print("MG1b self-test通过: bbox网格、同步翻转、癌图KL掩码与注意力唯一池化路径有效")


def main() -> None:
    """训练、筛选并导出一组正式MG1b注意力教师实验。

    调度顺序:
        ``load_manifest`` -> ``MG1bDataset/DataLoader`` ->
        ``AttentionPoolingTeacher`` -> stage A/B ``train_epoch`` ->
        ``evaluate`` -> ``eligibility/selection_key`` -> checkpoint复算与导出。
    输入来自CLI指定的v3 train/val清单和审计JSON；输出为最佳
    checkpoint、逐图/逐患者val预测、训练历史和config。
    """
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.seed != 42 and not args.debug:
        raise ValueError("MG1b预注册只允许正式seed42")
    # 1) 冻结数据边界：只读train/val v3 ROI，并用train癌图冻结病灶分层。
    seed_everything(args.seed)
    frame, _ = load_manifest(
        args.manifest.resolve(), args.v3_audit.resolve(), args.debug,
        args.debug_patients_per_class, args.seed,
    )
    train = frame.loc[frame.split.eq("train")].reset_index(drop=True)
    val = frame.loc[frame.split.eq("val")].reset_index(drop=True)
    bounds = lesion_tercile_bounds(train)

    run_name = args.run_name + ("_debug" if args.debug else "")
    if args.debug:
        args.stage_a_epochs = min(args.stage_a_epochs, 1)
        args.stage_b_epochs = min(args.stage_b_epochs, 1)
        args.num_workers = 0
    output_dir = args.output_root.resolve() / run_name
    if output_dir.exists():
        raise FileExistsError(f"MG1b输出已存在，拒绝覆盖: {output_dir}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了--device cuda，但当前PyTorch无法使用CUDA")
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )
    # 2) Dataset准备教师输入与空间监督；采样器平衡患者与类别。
    train_dataset = MG1bDataset(train, True, args.seed)
    val_dataset = MG1bDataset(val, False, args.seed)
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
    model = AttentionPoolingTeacher(pretrained=True).to(device)
    print(
        f"设备={device}; seed={args.seed}; train={len(train)}张/{train.patient_id.nunique()}人; "
        f"val={len(val)}张/{val.patient_id.nunique()}人; 病灶三分位={bounds}"
    )

    # 3) stage A只训attention/classifier；stage B从A阶段最佳状态解冻features[5:]。
    history = []
    best = {"key": None, "state": None, "stage": None, "epoch": None, "metrics": None,
            "eligible": False, "gates": None}
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
            train_losses = train_epoch(model, train_loader, optimizer, device, stage)
            _, metrics = evaluate(model, val_dataset, val_loader, device, bounds)
            gate = eligibility(metrics)
            key = selection_key(metrics, gate["eligible"])
            if best["key"] is None or key > best["key"]:
                best = {
                    "key": key, "state": deepcopy(model.state_dict()), "stage": stage,
                    "epoch": epoch, "metrics": metrics, **gate,
                }
            if stage_best_key is None or key > stage_best_key:
                stage_best_key = key
                no_improve = 0
                if stage == "A":
                    best_stage_a_state = deepcopy(model.state_dict())
            else:
                no_improve += 1
            history.append({
                "stage": stage, "epoch": epoch, "elapsed_seconds": time.time() - start,
                "eligible": gate["eligible"], **gate["gates"],
                **{f"train_{name}": value for name, value in train_losses.items()},
                "val_total_loss": metrics["val_total_loss"],
                "val_classification_loss": metrics["val_classification_loss"],
                "val_spatial_kl": metrics["val_spatial_kl"],
                "val_weighted_spatial": metrics["val_weighted_spatial"],
                "val_image_auc": metrics["val_image_auc"],
                "val_patient_auc": metrics["val_patient_auc"],
                "val_mean_aib": metrics["spatial"]["mean_aib"],
                "val_mean_normalized_aib": metrics["spatial"]["mean_normalized_aib"],
                "val_pga": metrics["spatial"]["pga"],
            })
            print(
                f"stage{stage} {epoch:02d}/{epochs} | train={train_losses['total']:.4f} "
                f"(cls={train_losses['classification']:.4f}, kl={train_losses['spatial_kl']:.4f}, "
                f"weighted={train_losses['weighted_spatial']:.4f}) | "
                f"image AUC={metrics['val_image_auc']:.4f} patient AUC={metrics['val_patient_auc']:.4f} | "
                f"nAiB={metrics['spatial']['mean_normalized_aib']:.4f} "
                f"PGA={metrics['spatial']['pga']:.4f} eligible={gate['eligible']}"
            )
            if stage == "B" and no_improve >= args.patience:
                print(f"stageB early stop: 连续{args.patience}轮无预注册排序改善")
                break

    # 4) 重载冻结排序选出的最佳状态，重算一次val防止内存记录与权重不一致。
    model.load_state_dict(best["state"], strict=True)
    final_predictions, final_metrics = evaluate(model, val_dataset, val_loader, device, bounds)
    final_gate = eligibility(final_metrics)
    if final_gate["eligible"] != best["eligible"]:
        raise RuntimeError("MG1b最佳checkpoint合格状态复算不一致")
    patient_predictions = patient_mean(final_predictions, "cancer_probability")

    # 5) 只有训练和最佳状态复算成功后才创建正式产物目录。
    output_dir.mkdir(parents=True)
    checkpoint_name = "mg1b_best_teacher.pth" if best["eligible"] else "mg1b_best_diagnostic_ineligible.pth"
    checkpoint_path = output_dir / checkpoint_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": "efficientnet_b0_attention_pooling_7x7",
        "seed": args.seed,
        "best_stage": best["stage"],
        "best_epoch": best["epoch"],
        "eligible": best["eligible"],
        "metrics": final_metrics,
        "manifest_sha256": file_sha256(args.manifest.resolve()),
    }, checkpoint_path)
    final_predictions.to_csv(output_dir / "val_image_predictions.csv", index=False, encoding="utf-8-sig")
    patient_predictions.to_csv(output_dir / "val_patient_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    config = {
        "stage": "MG1b_bbox_supervised_attention_pooling_teacher",
        "debug": args.debug,
        "passed_mg1b": best["eligible"],
        "decision": "proceed_to_MG2" if best["eligible"] else "stop_before_MG2",
        "seed": args.seed,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest.resolve()),
        "v3_audit": str(args.v3_audit.resolve()),
        "v3_audit_sha256": file_sha256(args.v3_audit.resolve()),
        "train_images": int(len(train)),
        "train_patients": int(train.patient_id.nunique()),
        "val_images": int(len(val)),
        "val_patients": int(val.patient_id.nunique()),
        "lesion_tercile_bounds_from_train": list(bounds),
        "input_protocol": {
            "roi": "independent_axis_dynamic_margin_v3",
            "resize": [IMAGE_SIZE, IMAGE_SIZE],
            "color": "luma copied to 3 channels",
            "square_conversion": False,
            "letterbox": False,
            "bbox_jitter": False,
            "train_horizontal_flip": 0.5,
        },
        "architecture": {
            "backbone": "efficientnet_b0",
            "feature_map": [1280, GRID_SIZE, GRID_SIZE],
            "attention": "1x1 conv then 49-position spatial softmax",
            "pooling": "attention-weighted sum only; no GAP bypass",
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
            "attention_weight": ATTENTION_WEIGHT,
            "sampler": "patient_and_class_balanced_with_replacement",
            "selection": "eligible first, then patient AUC, image AUC, normalized AiB, PGA, lower loss",
        },
        "frozen_gates": {
            "patient_auc_floor": PATIENT_AUC_FLOOR,
            "image_auc_floor": IMAGE_AUC_FLOOR,
            "mean_normalized_aib_floor": NORMALIZED_AIB_FLOOR,
            "pga_floor": PGA_FLOOR,
            "stratum_pga_floor": STRATUM_PGA_FLOOR,
            "stratum_min_images": STRATUM_MIN_IMAGES,
            "normalized_aib_max_bbox_area_exclusive": NORMALIZED_AIB_MAX_BBOX_AREA,
        },
        "best_stage": best["stage"],
        "best_epoch": best["epoch"],
        "eligibility": final_gate,
        "metrics": final_metrics,
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
        f"MG1b完成: best=stage{best['stage']} epoch{best['epoch']} "
        f"patient AUC={final_metrics['val_patient_auc']:.4f} "
        f"nAiB={final_metrics['spatial']['mean_normalized_aib']:.4f} "
        f"PGA={final_metrics['spatial']['pga']:.4f} eligible={best['eligible']}"
    )
    print(f"输出目录: {output_dir}")
    if not best["eligible"]:
        raise RuntimeError("MG1b失败: 没有checkpoint通过全部预注册门槛，已保存ineligible诊断产物")


if __name__ == "__main__":
    main()
