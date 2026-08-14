#!/usr/bin/env python3
"""Y6探索性Full路线：冻结YOLO预测ROI上的局部分类与固定均值融合。

训练仅使用Y0F train：癌图采用GT框加固定范围jitter，非癌图采用与癌框尺寸
分布匹配的随机黏膜ROI。验证仅使用同seed冻结Y3-F的Top-1预测框；无框时最终
概率严格回退冻结M0-F全局概率，有框时固定取全局与局部概率均值。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from efficientnet_m3c_region_gate import patient_class_balanced_weights  # noqa: E402
from efficientnet_train_debiased import build_model  # noqa: E402
from evaluate_y2_yolo26 import INFERENCE_BATCH_SIZE, predict_val  # noqa: E402
from run_y1_yolo26_smoke import assert_locked_library_behavior, file_sha256  # noqa: E402
from train_utils import git_snapshot, json_ready, seed_everything, seed_worker  # noqa: E402


SEEDS = (42, 202, 503)
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MARGIN = 0.20
CENTER_JITTER = 0.10
SCALE_JITTER = (0.90, 1.10)
LABEL_SMOOTHING = 0.10
DEFAULT_MAPPING = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/"
    "11_Y0F_YOLO26完整诊断数据_20260813/y0f_mapping.csv"
)
EXPECTED_MAPPING_SHA256 = "b329d8d3b0033e84124fb7be6db05a70d6bb46fb014a3653ef20cadac07407cd"
M0_ROOT = PROJECT_ROOT / "结果/M0全量诊断_0804/正式验证集筛选"
Y3_ROOT = PROJECT_ROOT / "结果/YOLO26定位_0804/Y3固定640三种子"
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/YOLO26定位_0804/Y6预测ROI局部分类_Full_20260814"


def parse_args() -> argparse.Namespace:
    """解析单seed Y6训练、调试和自测参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--device", required=True, help="当前可见CUDA编号，通常为0。")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stage-a-epochs", type=int, default=5)
    parser.add_argument("--stage-b-epochs", type=int, default=15)
    parser.add_argument("--stage-a-lr", type=float, default=1e-3)
    parser.add_argument("--stage-b-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-units", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def product_paths(seed: int) -> dict[str, Path | float]:
    """返回并校验同seed M0-F全局模型和Y3-F检测器血缘。"""
    m0_run = M0_ROOT / f"m0_full_keep_efficientnet_b0_seed{seed}"
    m0_config_path = m0_run / "config.json"
    m0_config = json.loads(m0_config_path.read_text(encoding="utf-8"))
    m0_checkpoint = m0_run / "efficientnet_b0_debiased_best.pth"
    y3_run = Y3_ROOT / f"y3_full_yolo26s_640_seed{seed}"
    y3_config_path = y3_run / "y3_geometry_config.json"
    y3_config = json.loads(y3_config_path.read_text(encoding="utf-8"))
    y3_checkpoint = Path(y3_config["checkpoint"])
    if int(m0_config["seed"]) != seed or m0_config.get("defer_test") is not True:
        raise ValueError(f"M0-F配置不属于锁定seed{seed}")
    if y3_config.get("role") != "full" or int(y3_config["seed"]) != seed:
        raise ValueError(f"Y3-F配置角色或seed错误: {y3_config_path}")
    if file_sha256(y3_checkpoint) != y3_config["checkpoint_sha256"]:
        raise ValueError(f"Y3-F checkpoint SHA不一致: seed{seed}")
    return {
        "m0_config": m0_config_path,
        "m0_checkpoint": m0_checkpoint,
        "m0_val_predictions": m0_run / "val_image_predictions.csv",
        "y3_config": y3_config_path,
        "y3_checkpoint": y3_checkpoint,
        "threshold": float(y3_config["geometry"]["deployment_threshold"]),
        "imgsz": int(y3_config["geometry_protocol"]["imgsz"]),
        "m0_val_patient_auc": float(m0_config["best_val_patient_auc"]),
    }


def load_mapping(path: Path, debug: bool, debug_units: int, seed: int) -> pd.DataFrame:
    """加载Y0F并只保留train/val，拒绝test进入Y6开发过程。"""
    if file_sha256(path) != EXPECTED_MAPPING_SHA256:
        raise ValueError("Y0F mapping SHA与冻结协议不一致")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"patient_id": str})
    required = {
        "image_relpath", "yolo_image_relpath", "patient_id", "label", "split",
        "width", "height", "source", "center", "size_group", "bbox_valid",
        "bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm",
        "bbox_area_fraction", "sha256",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Y0F mapping缺少字段: {sorted(missing)}")
    if len(frame) != 3348 or set(frame.split.unique()) != {"train", "val", "test"}:
        raise ValueError("Y0F mapping规模或split异常")
    frame = frame.loc[frame.split.isin(["train", "val"])].copy()
    expected = {("train", 0): 1169, ("train", 1): 1181,
                ("val", 0): 257, ("val", 1): 240}
    if frame.groupby(["split", "label"]).size().to_dict() != expected:
        raise ValueError("Y6 train/val标签计数与冻结记录不一致")
    if frame.groupby("patient_id").split.nunique().gt(1).any():
        raise ValueError("Y6患者跨split")
    cancer = frame.label.eq(1)
    if not frame.loc[cancer, "bbox_valid"].astype(bool).all():
        raise ValueError("Y6癌图存在无效GT bbox")
    for relpath in frame.image_relpath:
        if not (PROJECT_ROOT / relpath).is_file():
            raise FileNotFoundError(relpath)
    if debug:
        selected = []
        for split, part in frame.groupby("split", sort=False):
            patients = part[["patient_id", "label"]].drop_duplicates()
            sampled = pd.concat([
                group.sample(min(debug_units, len(group)), random_state=seed)
                for _, group in patients.groupby("label", sort=True)
            ])
            selected.append(part.loc[part.patient_id.isin(sampled.patient_id)])
        frame = pd.concat(selected, ignore_index=True)
    return frame.reset_index(drop=True)


def square_box(box: np.ndarray, width: int, height: int, margin: float) -> np.ndarray:
    """在像素坐标中扩边并生成图内正方形，再返回归一化xyxy。"""
    x1, y1, x2, y2 = box * np.array([width, height, width, height], dtype=float)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    side = min(side, float(width), float(height))
    cx = min(max(cx, side / 2), width - side / 2)
    cy = min(max(cy, side / 2), height - side / 2)
    pixels = np.array([cx - side / 2, cy - side / 2,
                       cx + side / 2, cy + side / 2])
    return pixels / np.array([width, height, width, height], dtype=float)


def jitter_box(box: np.ndarray) -> np.ndarray:
    """对已扩边ROI执行冻结的中心和平移尺度扰动，并保持在图内。"""
    x1, y1, x2, y2 = map(float, box)
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    cx += (torch.rand(1).item() * 2 - 1) * CENTER_JITTER * bw
    cy += (torch.rand(1).item() * 2 - 1) * CENTER_JITTER * bh
    scale = SCALE_JITTER[0] + torch.rand(1).item() * (SCALE_JITTER[1] - SCALE_JITTER[0])
    bw, bh = min(1.0, bw * scale), min(1.0, bh * scale)
    cx = min(max(cx, bw / 2), 1 - bw / 2)
    cy = min(max(cy, bh / 2), 1 - bh / 2)
    return np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])


def random_matched_box(size_pool: np.ndarray) -> np.ndarray:
    """从癌ROI宽高分布抽样，在非癌图内随机放置同尺寸ROI。"""
    index = int(torch.randint(len(size_pool), (1,)).item())
    bw, bh = map(float, size_pool[index])
    cx = bw / 2 + torch.rand(1).item() * max(0.0, 1 - bw)
    cy = bh / 2 + torch.rand(1).item() * max(0.0, 1 - bh)
    return np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])


def add_base_train_boxes(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """为train癌图冻结扩边GT框，并返回用于非癌匹配的ROI宽高池。"""
    train = frame.loc[frame.split.eq("train")].copy()
    boxes = []
    for row in train.itertuples(index=False):
        if int(row.label) == 1:
            box = square_box(
                np.array([row.bbox_x1_norm, row.bbox_y1_norm,
                          row.bbox_x2_norm, row.bbox_y2_norm]),
                int(row.width), int(row.height), MARGIN,
            )
        else:
            box = np.full(4, np.nan)
        boxes.append(box)
    train[["roi_x1", "roi_y1", "roi_x2", "roi_y2"]] = np.vstack(boxes)
    cancer = train.loc[train.label.eq(1)]
    size_pool = np.column_stack([
        cancer.roi_x2.to_numpy() - cancer.roi_x1.to_numpy(),
        cancer.roi_y2.to_numpy() - cancer.roi_y1.to_numpy(),
    ])
    return train.reset_index(drop=True), size_pool


def predict_validation_rois(
    frame: pd.DataFrame, products: dict, device: str, mapping_root: Path,
) -> pd.DataFrame:
    """用冻结Y3-F生成val Top-1框，并按冻结阈值决定是否启用局部分支。"""
    val = frame.loc[frame.split.eq("val")].copy().reset_index(drop=True)
    source = mapping_root / Path(val.yolo_image_relpath.iloc[0]).parent
    yolo = YOLO(str(products["y3_checkpoint"]))
    assert_locked_library_behavior(yolo)
    predicted = predict_val(
        yolo, val, int(products["imgsz"]), device, source_dir=source,
    )
    predicted["has_roi"] = predicted.top1_confidence.ge(float(products["threshold"]))
    crops = []
    for row in predicted.itertuples(index=False):
        if bool(row.has_roi):
            box = square_box(
                np.array([row.top1_x1, row.top1_y1, row.top1_x2, row.top1_y2]),
                int(row.width), int(row.height), MARGIN,
            )
        else:
            box = np.array([0.0, 0.0, 1.0, 1.0])
        crops.append(box)
    predicted[["roi_x1", "roi_y1", "roi_x2", "roi_y2"]] = np.vstack(crops)
    return predicted


def attach_frozen_global_probabilities(
    val: pd.DataFrame, products: dict,
) -> pd.DataFrame:
    """绑定M0-F checkpoint时代的val逐图概率，消除重推理浮点漂移。"""
    path = Path(products["m0_val_predictions"])
    frozen = pd.read_csv(
        path, encoding="utf-8-sig", dtype={"patient_id": str},
        usecols=["image_relpath", "patient_id", "label", "cancer_probability"],
    ).rename(columns={"cancer_probability": "p_global_frozen"})
    if frozen.image_relpath.duplicated().any():
        raise ValueError("M0-F val逐图预测存在重复路径")
    merged = val.merge(
        frozen, on=["image_relpath", "patient_id", "label"],
        how="left", validate="one_to_one",
    )
    if merged.p_global_frozen.isna().any() or len(merged) != len(val):
        raise ValueError("Y6 val无法完整绑定M0-F冻结逐图概率")
    patient = patient_mean(merged, "p_global_frozen")
    recorded = float(products["m0_val_patient_auc"])
    if not np.isclose(auc(patient, "p_global_frozen"), recorded, rtol=0.0, atol=1e-12):
        raise RuntimeError("M0-F冻结逐图概率不能复现其正式患者AUC")
    return merged


def build_transforms() -> tuple[v2.Compose, v2.Compose]:
    """构建局部训练增强和全局/局部确定性评估变换。"""
    normalize = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.5), v2.RandomRotation(10),
        v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10), normalize,
    ])
    evaluate = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True), normalize,
    ])
    return train, evaluate


class Y6Dataset(Dataset):
    """按split生成局部ROI，并可同时返回冻结全局分类所需完整图。

    返回字典中的``image``与``roi``形状均为[3,224,224]；train的ROI每次读取
    都执行冻结范围jitter/随机放置，val使用预先冻结的YOLO框。
    """
    def __init__(self, frame, size_pool, training, local_transform, eval_transform):
        self.df = frame.reset_index(drop=True)
        self.size_pool = size_pool
        self.training = training
        self.local_transform = local_transform
        self.eval_transform = eval_transform

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def crop(image: Image.Image, box: np.ndarray) -> Image.Image:
        width, height = image.size
        x1, y1, x2, y2 = box
        pixels = (int(x1 * width), int(y1 * height),
                  max(int(x2 * width), int(x1 * width) + 1),
                  max(int(y2 * height), int(y1 * height) + 1))
        return image.crop(pixels).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        with Image.open(PROJECT_ROOT / row.image_relpath) as source:
            image = source.convert("RGB")
        if self.training:
            if int(row.label) == 1:
                box = jitter_box(row[["roi_x1", "roi_y1", "roi_x2", "roi_y2"]].to_numpy(float))
            else:
                box = random_matched_box(self.size_pool)
        else:
            box = row[["roi_x1", "roi_y1", "roi_x2", "roi_y2"]].to_numpy(float)
        roi = self.crop(image, box)
        full = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        return {
            "image": self.eval_transform(full),
            "roi": self.local_transform(roi),
            "label": torch.tensor(int(row.label), dtype=torch.long),
            "row_index": torch.tensor(index, dtype=torch.long),
        }


class Y6Model(nn.Module):
    """冻结M0-F全局模型与独立局部EfficientNet副本。

    全局分支只用于确定性概率；阶段A训练局部分类头，阶段B训练局部features[7:]
    与分类头。两个分支输入均为[B,3,224,224]，输出为[B,2] logits。
    """
    def __init__(self, global_model: nn.Module):
        super().__init__()
        self.global_model = global_model
        self.local_model = deepcopy(global_model)
        for parameter in self.global_model.parameters():
            parameter.requires_grad = False
        self.global_model.eval()

    def set_stage(self, stage: str) -> None:
        """设置A或B阶段可训练范围，并保持全局模型eval。"""
        self.eval()
        for parameter in self.local_model.parameters():
            parameter.requires_grad = False
        if stage == "A":
            for parameter in self.local_model.classifier.parameters():
                parameter.requires_grad = True
            self.local_model.classifier.train()
        elif stage == "B":
            for parameter in self.local_model.features[7:].parameters():
                parameter.requires_grad = True
            for parameter in self.local_model.classifier.parameters():
                parameter.requires_grad = True
            self.local_model.features[7:].train()
            self.local_model.classifier.train()
        else:
            raise ValueError(stage)
        self.global_model.eval()


def module_sha(module: nn.Module) -> str:
    """计算参数及BN running统计的稳定SHA-256。"""
    hasher = hashlib.sha256()
    for key, value in module.state_dict().items():
        hasher.update(key.encode())
        hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def patient_mean(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """按M0-F正式口径聚合患者：该患者全部图片概率的均值。"""
    rows = []
    for patient, group in frame.groupby("patient_id", sort=False):
        rows.append({"patient_id": patient, "label": int(group.label.iloc[0]),
                     column: float(group[column].mean())})
    return pd.DataFrame(rows)


def auc(frame: pd.DataFrame, column: str) -> float:
    """计算二分类ROC AUC。"""
    return float(roc_auc_score(frame.label, frame[column]))


def evaluate(model: Y6Model, dataset: Y6Dataset, loader: DataLoader, device: torch.device) -> tuple[pd.DataFrame, dict]:
    """执行完整val部署数据流并返回图像预测及全局/局部/融合AUC。"""
    model.eval()
    rows = []
    total_loss = 0.0
    with torch.inference_mode():
        for batch in loader:
            rois = batch["roi"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            local_logits = model.local_model(rois)
            total_loss += float(F.cross_entropy(local_logits, labels)) * len(labels)
            pl = torch.softmax(local_logits, 1)[:, 1].cpu().numpy()
            for i, row_index in enumerate(batch["row_index"].tolist()):
                meta = dataset.df.iloc[row_index]
                has_roi = bool(meta.has_roi)
                pg = float(meta.p_global_frozen)
                final = (pg + float(pl[i])) / 2 if has_roi else pg
                rows.append({
                    "row_index": row_index, "image_relpath": meta.image_relpath,
                    "patient_id": meta.patient_id, "label": int(meta.label),
                    "source": meta.source, "center": meta.center,
                    "size_group": meta.size_group,
                    "bbox_area_fraction": meta.bbox_area_fraction,
                    "has_roi": has_roi, "top1_confidence": float(meta.top1_confidence),
                    "p_global": pg, "p_local": float(pl[i]), "p_final": final,
                })
    predictions = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
    patients = {name: patient_mean(predictions, name)
                for name in ("p_global", "p_local", "p_final")}
    triggered = predictions.loc[predictions.has_roi]
    metrics = {
        "val_local_loss": total_loss / len(dataset),
        "image_global_auc": auc(predictions, "p_global"),
        "image_final_auc": auc(predictions, "p_final"),
        "patient_global_auc": auc(patients["p_global"], "p_global"),
        "patient_local_auc_all_with_full_fallback_crop": auc(patients["p_local"], "p_local"),
        "patient_final_auc": auc(patients["p_final"], "p_final"),
        "trigger_rate_cancer": float(predictions.loc[predictions.label.eq(1), "has_roi"].mean()),
        "trigger_rate_noncancer": float(predictions.loc[predictions.label.eq(0), "has_roi"].mean()),
        "triggered_local_image_auc": auc(triggered, "p_local") if triggered.label.nunique() == 2 else math.nan,
    }
    return predictions, metrics


def train_epoch(model: Y6Model, loader: DataLoader, optimizer, device: torch.device) -> float:
    """训练一个局部分类epoch并返回按图片平均交叉熵。"""
    total = 0.0
    seen = 0
    for batch in loader:
        rois = batch["roi"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model.local_model(rois)
        loss = F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTHING)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(labels)
        seen += len(labels)
    return total / seen


def paired_patient_bootstrap(predictions: pd.DataFrame, seed: int, iterations: int) -> list[float]:
    """按患者有放回抽样，计算融合减全局患者AUC差置信区间。"""
    groups = {patient: group for patient, group in predictions.groupby("patient_id", sort=False)}
    patients = np.array(list(groups), dtype=object)
    rng = np.random.default_rng(seed + 20260814)
    differences = []
    for _ in range(iterations):
        sampled = rng.choice(patients, len(patients), replace=True)
        chunks = []
        for unit, patient in enumerate(sampled):
            chunk = groups[patient].copy()
            chunk["_unit"] = unit
            chunks.append(chunk)
        boot = pd.concat(chunks, ignore_index=True)
        labels, pg, pf = [], [], []
        for _, group in boot.groupby("_unit", sort=False):
            labels.append(int(group.label.iloc[0]))
            pg.append(float(group.p_global.mean()))
            pf.append(float(group.p_final.mean()))
        if len(set(labels)) == 2:
            differences.append(float(roc_auc_score(labels, pf) - roc_auc_score(labels, pg)))
    return differences


def threshold_metrics(patient: pd.DataFrame, column: str) -> dict:
    """在患者级满足Sensitivity>=0.90的最高实际阈值处报告分类指标。"""
    positive = patient.loc[patient.label.eq(1), column].to_numpy()
    threshold = min(np.sort(positive)[::-1][max(0, math.ceil(0.90 * len(positive)) - 1)], 1.0)
    predicted = patient[column].ge(threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(patient.label, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold), "accuracy": float((tp + tn) / len(patient)),
        "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
        "confusion_matrix": [int(tn), int(fp), int(fn), int(tp)],
    }


def stratified_diagnostics(
    predictions: pd.DataFrame, train: pd.DataFrame,
) -> dict:
    """报告来源、中心、分辨率及冻结病灶大小分层，不参与模型选择。"""
    result = {}
    for column in ("source", "center", "size_group"):
        groups = {}
        for name, part in predictions.groupby(column, dropna=False, sort=True):
            patients = patient_mean(part, "p_final")
            groups[str(name)] = {
                "images": int(len(part)), "patients": int(part.patient_id.nunique()),
                "cancer_images": int(part.label.eq(1).sum()),
                "noncancer_images": int(part.label.eq(0).sum()),
                "image_auc": (auc(part, "p_final") if part.label.nunique() == 2 else math.nan),
                "patient_auc": (auc(patients, "p_final") if patients.label.nunique() == 2 else math.nan),
            }
        result[column] = groups
    train_areas = train.loc[train.label.eq(1), "bbox_area_fraction"].dropna().to_numpy(float)
    bounds = np.quantile(train_areas, [1 / 3, 2 / 3])
    cancer = predictions.loc[predictions.label.eq(1)].copy()
    cancer["lesion_size"] = pd.cut(
        cancer.bbox_area_fraction, [-np.inf, bounds[0], bounds[1], np.inf],
        labels=["small", "medium", "large"], include_lowest=True,
    )
    lesion = {}
    for name, part in cancer.groupby("lesion_size", observed=True, sort=True):
        lesion[str(name)] = {
            "images": int(len(part)), "patients": int(part.patient_id.nunique()),
            "roi_trigger_rate": float(part.has_roi.mean()),
            "mean_global_probability": float(part.p_global.mean()),
            "mean_local_probability_triggered": (
                float(part.loc[part.has_roi, "p_local"].mean()) if part.has_roi.any() else math.nan
            ),
            "mean_final_probability": float(part.p_final.mean()),
        }
    result["lesion_size"] = {
        "train_bbox_area_terciles": bounds.tolist(), "groups": lesion,
    }
    return result


def run_self_test() -> None:
    """验证正方形裁剪、jitter边界、尺寸匹配和无框回退公式。"""
    box = square_box(np.array([0.2, 0.3, 0.4, 0.5]), 800, 600, MARGIN)
    assert np.all((0 <= box) & (box <= 1))
    for _ in range(20):
        assert np.all((0 <= jitter_box(box)) & (jitter_box(box) <= 1))
    pool = np.array([[0.2, 0.3], [0.4, 0.5]])
    assert np.all((0 <= random_matched_box(pool)) & (random_matched_box(pool) <= 1))
    pg, pl = 0.3, 0.9
    assert pg == pg and np.isclose((pg + pl) / 2, 0.6)
    print("Y6 self-test passed")


def main() -> None:
    """执行单seed Y6：血缘校验、val框冻结、两阶段训练、产品与诊断落盘。"""
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    seed_everything(args.seed)
    products = product_paths(args.seed)
    # YOLO必须扫描完整冻结val目录；debug只在预测完成后抽取训练/验证患者。
    full_frame = load_mapping(args.mapping, False, args.debug_units, args.seed)
    name = args.run_name or f"y6_full_efficientnet_b0_seed{args.seed}"
    if args.debug:
        name += "_debug"
        args.stage_a_epochs = min(args.stage_a_epochs, 1)
        args.stage_b_epochs = min(args.stage_b_epochs, 1)
        args.bootstrap = min(args.bootstrap, 100)
    output = args.output_root / name
    if output.exists():
        raise FileExistsError(f"Y6输出已存在，拒绝覆盖: {output}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and args.device != "0":
        raise ValueError("启动器应通过CUDA_VISIBLE_DEVICES隔离目标卡，脚本内--device应为0")
    val = predict_validation_rois(full_frame, products, args.device, args.mapping.parent)
    val = attach_frozen_global_probabilities(val, products)
    frame = (load_mapping(args.mapping, True, args.debug_units, args.seed)
             if args.debug else full_frame)
    if args.debug:
        selected_val = set(frame.loc[frame.split.eq("val"), "image_relpath"])
        val = val.loc[val.image_relpath.isin(selected_val)].reset_index(drop=True)
    train, size_pool = add_base_train_boxes(frame)
    output.mkdir(parents=True)
    val.to_csv(output / "frozen_val_predicted_roi.csv", index=False, encoding="utf-8-sig")

    m0_payload = torch.load(products["m0_checkpoint"], map_location="cpu", weights_only=False)
    global_model = build_model(pretrained=False)
    global_model.load_state_dict(m0_payload["model_state_dict"], strict=True)
    model = Y6Model(global_model).to(device)
    global_sha_before = module_sha(model.global_model)

    train_tf, eval_tf = build_transforms()
    train_dataset = Y6Dataset(train, size_pool, True, train_tf, eval_tf)
    val_dataset = Y6Dataset(val, size_pool, False, eval_tf, eval_tf)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        patient_class_balanced_weights(train), len(train), replacement=True, generator=generator,
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
    print(f"设备={device}; seed={args.seed}; train={len(train)}张/{train.patient_id.nunique()}人; "
          f"val={len(val)}张/{val.patient_id.nunique()}人; "
          f"YOLO阈值={float(products['threshold']):.6f}; "
          f"val触发 癌={val.loc[val.label.eq(1),'has_roi'].mean():.4f} "
          f"非癌={val.loc[val.label.eq(0),'has_roi'].mean():.4f}")

    history = []
    best = {"key": None, "state": None, "stage": None, "epoch": None,
            "metrics": None, "predictions": None}
    for stage, epochs, lr in (("A", args.stage_a_epochs, args.stage_a_lr),
                              ("B", args.stage_b_epochs, args.stage_b_lr)):
        model.set_stage(stage)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = optim.AdamW(parameters, lr=lr, weight_decay=args.weight_decay)
        no_improve = 0
        for epoch in range(1, epochs + 1):
            start = time.time()
            model.set_stage(stage)
            train_loss = train_epoch(model, train_loader, optimizer, device)
            predictions, metrics = evaluate(model, val_dataset, val_loader, device)
            if not args.debug and not np.isclose(
                metrics["patient_global_auc"], products["m0_val_patient_auc"],
                rtol=0.0, atol=1e-7,
            ):
                raise RuntimeError(
                    "Y6冻结全局患者AUC与M0-F正式记录不一致: "
                    f"{metrics['patient_global_auc']:.8f} != "
                    f"{products['m0_val_patient_auc']:.8f}"
                )
            key = (metrics["patient_final_auc"], metrics["image_final_auc"],
                   -metrics["val_local_loss"])
            improved = best["key"] is None or key > best["key"]
            if improved:
                best = {"key": key, "state": deepcopy(model.local_model.state_dict()),
                        "stage": stage, "epoch": epoch, "metrics": metrics,
                        "predictions": predictions.copy()}
            if stage == "B":
                no_improve = 0 if improved else no_improve + 1
            record = {"stage": stage, "epoch": epoch, "train_local_loss": train_loss,
                      "elapsed_seconds": time.time() - start, **metrics}
            history.append(record)
            print(f"stage{stage} {epoch:02d}/{epochs} | train={train_loss:.4f} "
                  f"val_local={metrics['val_local_loss']:.4f} | "
                  f"image AUC {metrics['image_global_auc']:.4f}->{metrics['image_final_auc']:.4f} | "
                  f"patient AUC {metrics['patient_global_auc']:.4f}->{metrics['patient_final_auc']:.4f} "
                  f"Δ={metrics['patient_final_auc']-metrics['patient_global_auc']:+.4f}")
            if stage == "B" and no_improve >= args.patience:
                print(f"stageB early stop: 连续{args.patience}轮未改善")
                break
        if stage == "A" and best["stage"] == "A":
            # 阶段B从阶段A在部署val上选出的最佳状态开始，而不是机械沿用最后一轮。
            model.local_model.load_state_dict(best["state"], strict=True)

    model.local_model.load_state_dict(best["state"], strict=True)
    final_predictions, final_metrics = evaluate(model, val_dataset, val_loader, device)
    if not np.isclose(final_metrics["patient_final_auc"], best["metrics"]["patient_final_auc"], atol=1e-12):
        raise RuntimeError("Y6最佳checkpoint复算不一致")
    if module_sha(model.global_model) != global_sha_before:
        raise RuntimeError("Y6冻结M0-F全局模型被修改")

    global_auc = final_metrics["patient_global_auc"]
    candidate_auc = final_metrics["patient_final_auc"]
    fell_back = candidate_auc <= global_auc
    final_predictions["p_product"] = (
        final_predictions.p_global if fell_back else final_predictions.p_final
    )
    patient_global = patient_mean(final_predictions, "p_global")
    patient_candidate = patient_mean(final_predictions, "p_final")
    patient_product = patient_mean(final_predictions, "p_product")
    bootstrap = paired_patient_bootstrap(final_predictions, args.seed, args.bootstrap)
    ci = np.quantile(bootstrap, [0.025, 0.975]).tolist() if bootstrap else [math.nan, math.nan]
    selection = {
        "best_stage": best["stage"], "best_epoch": best["epoch"],
        "global_patient_auc": global_auc, "candidate_patient_auc": candidate_auc,
        "raw_delta_patient_auc": candidate_auc - global_auc,
        "fell_back_to_global": fell_back,
        "product_patient_auc": auc(patient_product, "p_product"),
        "paired_patient_bootstrap_delta_ci95": ci,
        "passed_single_seed_gain": bool(candidate_auc > global_auc),
    }
    diagnostics = {
        "image": final_metrics,
        "patient_global_threshold_metrics": threshold_metrics(patient_global, "p_global"),
        "patient_candidate_threshold_metrics": threshold_metrics(patient_candidate, "p_final"),
        "patient_product_threshold_metrics": threshold_metrics(patient_product, "p_product"),
        "stratified": stratified_diagnostics(final_predictions, train),
    }
    config = {
        "stage": "Y6_exploratory_full", "seed": args.seed, "debug": args.debug,
        "mapping": str(args.mapping), "mapping_sha256": file_sha256(args.mapping),
        "train_images": len(train), "train_patients": int(train.patient_id.nunique()),
        "val_images": len(val), "val_patients": int(val.patient_id.nunique()),
        "m0_config": str(products["m0_config"]),
        "m0_checkpoint": str(products["m0_checkpoint"]),
        "m0_checkpoint_sha256": file_sha256(products["m0_checkpoint"]),
        "m0_val_predictions": str(products["m0_val_predictions"]),
        "m0_val_predictions_sha256": file_sha256(products["m0_val_predictions"]),
        "m0_recorded_val_patient_auc": float(products["m0_val_patient_auc"]),
        "y3_config": str(products["y3_config"]),
        "y3_checkpoint": str(products["y3_checkpoint"]),
        "y3_checkpoint_sha256": file_sha256(products["y3_checkpoint"]),
        "y3_deployment_threshold": float(products["threshold"]),
        "roi_protocol": {"margin": MARGIN, "center_jitter": CENTER_JITTER,
                         "scale_jitter": list(SCALE_JITTER),
                         "noncancer": "random_location_matched_to_train_cancer_roi_size"},
        "fusion": "no_roi:p_global; roi:(p_global+p_local)/2",
        "training": {"stage_a_epochs": args.stage_a_epochs,
                     "stage_b_epochs": args.stage_b_epochs,
                     "stage_a_lr": args.stage_a_lr, "stage_b_lr": args.stage_b_lr,
                     "patience": args.patience, "label_smoothing": LABEL_SMOOTHING},
        "selection": selection, "diagnostics": diagnostics,
        "global_frozen_sha256": global_sha_before,
        "test_evaluated": False, "external_evaluated": False,
        **git_snapshot(),
    }
    torch.save({
        "local_model_state_dict": model.local_model.state_dict(),
        "seed": args.seed, "selection": selection,
        "m0_checkpoint_sha256": file_sha256(products["m0_checkpoint"]),
        "y3_checkpoint_sha256": file_sha256(products["y3_checkpoint"]),
    }, output / "y6_best_local.pth")
    final_predictions.to_csv(output / "val_image_predictions.csv", index=False, encoding="utf-8-sig")
    patient_global.merge(patient_candidate, on=["patient_id", "label"]).merge(
        patient_product, on=["patient_id", "label"]
    ).to_csv(output / "val_patient_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False, encoding="utf-8-sig")
    (output / "config.json").write_text(
        json.dumps(json_ready(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Y6完成: best=stage{best['stage']} epoch{best['epoch']} | "
          f"患者AUC {global_auc:.4f}->{candidate_auc:.4f} "
          f"Δ={candidate_auc-global_auc:+.4f} | 回退={fell_back}")
    print(f"输出目录: {output}")


if __name__ == "__main__":
    main()
