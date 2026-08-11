#!/usr/bin/env python3
"""M3c-B：冻结EfficientNet-M1，只训练预测框上的独立区域存在性门控头。"""

import argparse
import hashlib
import json
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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.ops import roi_align


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from efficientnet_m1_localization import (  # noqa: E402
    GRID_SIZE,
    M1Dataset,
    EfficientNetM1,
    decode_boxes,
)
from train_utils import (  # noqa: E402
    file_sha256,
    git_snapshot,
    json_ready,
    seed_everything,
    seed_worker,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT / "数据整理记录/图像裁剪"
    / "胃早癌概念提取训练集0804_预处理_v1"
    / "08_M1辅助定位清单_20260810"
    / "m1_balanced_keep_primary_1to1p3_split_seed42.csv"
)
M1_ROOT = PROJECT_ROOT / "结果/M1辅助定位_0804/正式验证集筛选"
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/M3c区域门控_0804/正式验证集筛选"
SEEDS = (42, 202, 503)


# -----------------------------------------------------------------------------
# 参数与冻结输入：正式入口只建立train/val Dataset，test只保留在清单中但从不读取图像。
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--m1-checkpoint", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--fixed-recall", type=float, default=0.90)
    parser.add_argument("--min-fp-drop", type=float, default=0.10)
    parser.add_argument("--min-group-cancer", type=int, default=15)
    parser.add_argument("--max-group-detection-loss", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--qc-count", type=int, default=20)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-units", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def default_m1_checkpoint(seed):
    return (
        M1_ROOT
        / f"m1_balanced_keep_efficientnet_b0_seed{seed}_warmup_product"
        / "m1_best_warmup_localization.pth"
    )


def load_train_val_manifest(path, image_root, debug, debug_units, seed):
    """校验冻结清单后仅返回train/val；绝不为test建立Dataset或读取test图像。"""
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"patient_id": str})
    required = {
        "image_relpath", "patient_id", "label", "split", "branch", "size_group",
        "localization_supervision", "bbox_x1_norm", "bbox_y1_norm",
        "bbox_x2_norm", "bbox_y2_norm",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"M3c-B清单缺少字段: {sorted(missing)}")
    if set(frame.split.unique()) != {"train", "val", "test"}:
        raise ValueError("冻结清单必须包含train/val/test")
    if frame.groupby("patient_id").split.nunique().gt(1).any():
        raise ValueError("患者跨split")
    if frame.groupby("patient_id").label.nunique().gt(1).any():
        raise ValueError("患者跨标签")
    if not frame.branch.eq("keep").all():
        raise ValueError("M3c-B只接受Keep清单")
    frame = frame.loc[frame.split.isin(["train", "val"])].copy()
    positive = frame.label.eq(1)
    if not frame.loc[positive, "localization_supervision"].eq(1).all():
        raise ValueError("train/val癌图bbox监督不完整")
    if frame.loc[~positive, "localization_supervision"].ne(0).any():
        raise ValueError("非癌图不得伪造bbox")
    missing_paths = [
        value for value in frame.image_relpath.astype(str)
        if not (image_root / value).is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(f"train/val缺失{len(missing_paths)}张图: {missing_paths[0]}")
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


def patient_class_balanced_weights(frame):
    """让两标签总权重相同，且同标签内每位患者总权重相同。"""
    patient_sizes = frame.groupby("patient_id").size()
    patient_labels = frame[["patient_id", "label"]].drop_duplicates()
    class_patient_counts = patient_labels.label.value_counts()
    return torch.tensor([
        1.0 / (
            float(class_patient_counts.loc[row.label])
            * float(patient_sizes.loc[row.patient_id])
        )
        for row in frame.itertuples(index=False)
    ], dtype=torch.double)


# -----------------------------------------------------------------------------
# 模型：M1完整冻结；ROIAlign后的112维区域特征只进入一个两层小门控头。
# -----------------------------------------------------------------------------
class RegionGateHead(nn.Module):
    """把ROI特征、定位峰值和框宽高映射为单个区域存在性logit。"""

    def __init__(self, feature_channels=112, hidden=64, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_channels + 3, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, roi_feature, localization_confidence, boxes):
        pooled = roi_feature.mean(dim=(2, 3))
        width_height = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
        gate_input = torch.cat(
            (pooled, localization_confidence[:, None], width_height), dim=1
        )
        return self.network(gate_input).squeeze(1)


class FrozenM1RegionGate(nn.Module):
    """冻结M1产生特征和Top-1框，仅区域门控头参与梯度更新。"""

    def __init__(self, m1):
        super().__init__()
        self.m1 = m1
        self.gate = RegionGateHead()
        for parameter in self.m1.parameters():
            parameter.requires_grad = False
        self.m1.eval()

    def train(self, mode=True):
        self.training = mode
        self.m1.eval()
        self.gate.train(mode)
        return self

    def forward(self, images):
        with torch.no_grad():
            feature = images
            localization_feature = None
            for index, block in enumerate(self.m1.backbone.features):
                feature = block(feature)
                if index == 5:
                    localization_feature = feature
            pooled = self.m1.backbone.avgpool(feature)
            classification_logits = self.m1.backbone.classifier(torch.flatten(pooled, 1))
            location_outputs = self.m1.localization_head(localization_feature)
            boxes, confidence = decode_boxes(location_outputs)
            batch_indices = torch.arange(
                len(images), device=images.device, dtype=boxes.dtype
            )[:, None]
            feature_boxes = boxes * GRID_SIZE
            rois = torch.cat((batch_indices, feature_boxes), dim=1)
            roi_feature = roi_align(
                localization_feature, rois, output_size=(3, 3),
                spatial_scale=1.0, sampling_ratio=2, aligned=True,
            )
        gate_logits = self.gate(
            roi_feature.detach(), confidence.detach(), boxes.detach()
        )
        return {
            "gate_logits": gate_logits,
            "boxes": boxes,
            "localization_confidence": confidence,
            "cancer_probability": torch.softmax(classification_logits, dim=1)[:, 1],
        }


def frozen_m1_snapshot(model):
    """对冻结M1参数和BN running统计生成SHA-256，防止训练路径意外污染。"""
    parameter_hash = hashlib.sha256()
    bn_hash = hashlib.sha256()
    for parameter in model.m1.parameters():
        parameter_hash.update(parameter.detach().cpu().numpy().tobytes())
    for module in model.m1.modules():
        if isinstance(module, nn.BatchNorm2d) and module.running_mean is not None:
            bn_hash.update(module.running_mean.detach().cpu().numpy().tobytes())
            bn_hash.update(module.running_var.detach().cpu().numpy().tobytes())
    return {"parameters": parameter_hash.hexdigest(), "bn_running": bn_hash.hexdigest()}


# -----------------------------------------------------------------------------
# 损失：癌/非癌各自均值后等权；癌图画错位置的ROI不硬标为正例。
# -----------------------------------------------------------------------------
def center_inside_target(predicted, target):
    center = (predicted[:, :2] + predicted[:, 2:]) / 2
    return (
        (center[:, 0] >= target[:, 0]) & (center[:, 0] <= target[:, 2])
        & (center[:, 1] >= target[:, 1]) & (center[:, 1] <= target[:, 3])
    )


def balanced_gate_loss(logits, labels, boxes, targets, valid_box):
    """返回正负等权BCE及正例资格掩码；无某类时保持可微0。"""
    positive_eligible = labels.eq(1) & valid_box & center_inside_target(boxes, targets)
    negative = labels.eq(0)
    zero = logits.sum() * 0
    positive_loss = nnf.softplus(-logits[positive_eligible]).mean() if positive_eligible.any() else zero
    negative_loss = nnf.softplus(logits[negative]).mean() if negative.any() else zero
    total = 0.5 * positive_loss + 0.5 * negative_loss
    return {
        "total": total,
        "positive": positive_loss,
        "negative": negative_loss,
        "positive_eligible": positive_eligible,
        "negative_mask": negative,
    }


def train_epoch(model, loader, optimizer, device):
    """训练一个epoch并按实际正/负监督图数汇总损失。"""
    model.train(True)
    total_weighted = 0.0
    pos_sum = neg_sum = 0.0
    pos_count = neg_count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        targets = batch["bbox"].to(device, non_blocking=True)
        valid_box = batch["valid_box"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        losses = balanced_gate_loss(
            outputs["gate_logits"], labels, outputs["boxes"], targets, valid_box
        )
        losses["total"].backward()
        optimizer.step()
        batch_size = len(images)
        total_weighted += float(losses["total"].detach()) * batch_size
        current_pos = int(losses["positive_eligible"].sum())
        current_neg = int(losses["negative_mask"].sum())
        if current_pos:
            pos_sum += float(losses["positive"].detach()) * current_pos
            pos_count += current_pos
        if current_neg:
            neg_sum += float(losses["negative"].detach()) * current_neg
            neg_count += current_neg
    return {
        "total": total_weighted / len(loader.dataset),
        "positive": pos_sum / max(pos_count, 1),
        "negative": neg_sum / max(neg_count, 1),
        "positive_eligible": pos_count,
        "negative_supervised": neg_count,
    }


# -----------------------------------------------------------------------------
# 评估与门槛：val锁定90%癌召回操作点，按FP下降和相对M1分组计数选择产品。
# -----------------------------------------------------------------------------
def lock_recall_threshold(values, recall=0.90):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("锁阈值至少需要一张癌图")
    for threshold in np.sort(np.unique(values))[::-1]:
        if float(np.mean(values >= threshold)) >= recall:
            return float(threshold)
    return float(values.min())


def rate_metrics(labels, decisions):
    labels = np.asarray(labels, dtype=int)
    decisions = np.asarray(decisions, dtype=bool)
    cancer = labels == 1
    noncancer = ~cancer
    return {
        "cancer_count": int(cancer.sum()),
        "noncancer_count": int(noncancer.sum()),
        "cancer_detected": int((cancer & decisions).sum()),
        "noncancer_false_positive": int((noncancer & decisions).sum()),
        "cancer_recall": float(decisions[cancer].mean()),
        "noncancer_fp": float(decisions[noncancer].mean()),
    }


def resolution_group(frame):
    return frame.size_group.map({
        "large_1025_1600": "large",
        "medium_641_1024": "non_large",
        "small_le640": "non_large",
    })


def group_gate_details(frame, gate_column, baseline_column, minimum_count, max_loss):
    """按每组相对M1最多多漏1张执行离散计数门槛。"""
    cancer = frame.loc[frame.label.eq(1)].copy()
    cancer["resolution_gate_group"] = resolution_group(cancer)
    details = {}
    for name, group in cancer.dropna(subset=["resolution_gate_group"]).groupby(
        "resolution_gate_group"
    ):
        if len(group) < minimum_count:
            continue
        baseline_detected = int(group[baseline_column].sum())
        current_detected = int(group[gate_column].sum())
        details[name] = {
            "n_cancer": int(len(group)),
            "m1_detected": baseline_detected,
            "gate_detected": current_detected,
            "maximum_additional_misses": max_loss,
            "passed": current_detected >= baseline_detected - max_loss,
        }
    return details


def patient_metrics(frame, decision_column):
    patient = frame.groupby("patient_id", as_index=False).agg(
        label=("label", "first"), decision=(decision_column, "max")
    )
    return rate_metrics(patient.label, patient.decision), patient


def patient_bootstrap(patient, iterations, seed):
    rng = np.random.default_rng(seed)
    groups = {
        label: patient.loc[patient.label.eq(label)].reset_index(drop=True)
        for label in (0, 1)
    }
    recalls, fps = np.empty(iterations), np.empty(iterations)
    for index in range(iterations):
        cancer = groups[1].iloc[rng.integers(0, len(groups[1]), len(groups[1]))]
        noncancer = groups[0].iloc[rng.integers(0, len(groups[0]), len(groups[0]))]
        recalls[index] = cancer.decision.mean()
        fps[index] = noncancer.decision.mean()
    return {
        "cancer_recall_ci95": np.quantile(recalls, [0.025, 0.975]).tolist(),
        "noncancer_fp_ci95": np.quantile(fps, [0.025, 0.975]).tolist(),
    }


def evaluate(model, loader, device, args, baseline=None):
    """提取冻结框和区域门控概率；baseline缺失时同时构建M1门控基线。"""
    model.train(False)
    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            targets = batch["bbox"].to(device, non_blocking=True)
            outputs = model(images)
            probabilities = torch.sigmoid(outputs["gate_logits"])
            for local, row_index in enumerate(batch["row_index"].tolist()):
                rows.append({
                    "row_index": row_index,
                    "label": int(labels[local]),
                    "gate_probability": float(probabilities[local]),
                    "localization_confidence": float(outputs["localization_confidence"][local]),
                    "cancer_probability": float(outputs["cancer_probability"][local]),
                    "gt_x1": float(targets[local, 0]), "gt_y1": float(targets[local, 1]),
                    "gt_x2": float(targets[local, 2]), "gt_y2": float(targets[local, 3]),
                    "pred_x1": float(outputs["boxes"][local, 0]),
                    "pred_y1": float(outputs["boxes"][local, 1]),
                    "pred_x2": float(outputs["boxes"][local, 2]),
                    "pred_y2": float(outputs["boxes"][local, 3]),
                })
    predictions = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
    metadata = loader.dataset.df.reset_index(drop=True)
    predictions = pd.concat(
        [metadata, predictions.drop(columns=["row_index", "label"])], axis=1
    )
    if baseline is None:
        loc_threshold = lock_recall_threshold(
            predictions.loc[predictions.label.eq(1), "localization_confidence"],
            args.fixed_recall,
        )
        predictions["m1_display"] = predictions.localization_confidence >= loc_threshold
        baseline_metrics = rate_metrics(predictions.label, predictions.m1_display)
        baseline = {
            "localization_threshold": loc_threshold,
            "image_metrics": baseline_metrics,
        }
    else:
        predictions["m1_display"] = (
            predictions.localization_confidence >= baseline["localization_threshold"]
        )
    gate_threshold = lock_recall_threshold(
        predictions.loc[predictions.label.eq(1), "gate_probability"], args.fixed_recall
    )
    predictions["gate_display"] = predictions.gate_probability >= gate_threshold
    metrics = rate_metrics(predictions.label, predictions.gate_display)
    fp_drop = baseline["image_metrics"]["noncancer_fp"] - metrics["noncancer_fp"]
    groups = group_gate_details(
        predictions, "gate_display", "m1_display",
        args.min_group_cancer, args.max_group_detection_loss,
    )
    group_passed = bool(groups) and all(detail["passed"] for detail in groups.values())
    auc = float(roc_auc_score(predictions.label, predictions.gate_probability))
    patient_gate_metrics, patient = patient_metrics(predictions, "gate_display")
    patient_baseline_metrics, _ = patient_metrics(predictions, "m1_display")
    eligible = fp_drop >= args.min_fp_drop and group_passed
    return {
        "predictions": predictions,
        "patient": patient,
        "baseline": baseline,
        "summary": {
            **metrics,
            "gate_threshold": gate_threshold,
            "fp_drop": fp_drop,
            "gate_auc": auc,
            "group_details": groups,
            "group_gate_passed": group_passed,
            "patient_cancer_recall": patient_gate_metrics["cancer_recall"],
            "patient_noncancer_fp": patient_gate_metrics["noncancer_fp"],
            "patient_baseline_noncancer_fp": patient_baseline_metrics["noncancer_fp"],
            "eligible": eligible,
        },
    }


# -----------------------------------------------------------------------------
# QC、保存与主流程：失败同样落盘history/config，test/external状态恒为False。
# -----------------------------------------------------------------------------
def draw_box(draw, row, color="red"):
    box = tuple(
        int(round(value * 223))
        for value in (row.pred_x1, row.pred_y1, row.pred_x2, row.pred_y2)
    )
    draw.rectangle(box, outline=color, width=3)


def save_qc(frame, image_root, output, count):
    categories = {
        "gated_cancer_qc.jpg": frame.loc[frame.label.eq(1) & frame.gate_display]
            .sort_values("gate_probability").head(count),
        "rejected_cancer_qc.jpg": frame.loc[frame.label.eq(1) & ~frame.gate_display]
            .sort_values("gate_probability", ascending=False).head(count),
        "gated_noncancer_qc.jpg": frame.loc[frame.label.eq(0) & frame.gate_display]
            .sort_values("gate_probability", ascending=False).head(count),
    }
    for filename, selected in categories.items():
        if selected.empty:
            continue
        columns, tile_size, header = 4, 224, 24
        canvas = Image.new(
            "RGB", (columns * tile_size, int(np.ceil(len(selected) / columns)) * (tile_size + header)),
            "white",
        )
        for index, row in enumerate(selected.itertuples(index=False)):
            with Image.open(image_root / row.image_relpath) as source:
                image = source.convert("RGB").resize((tile_size, tile_size), Image.Resampling.BILINEAR)
            draw_box(ImageDraw.Draw(image), row)
            tile = Image.new("RGB", (tile_size, tile_size + header), "white")
            tile.paste(image, (0, header))
            ImageDraw.Draw(tile).text(
                (4, 5), f"y={row.label} gate={int(row.gate_display)} q={row.gate_probability:.3f}",
                fill="black",
            )
            canvas.paste(
                tile,
                ((index % columns) * tile_size, (index // columns) * (tile_size + header)),
            )
        canvas.save(output / filename, quality=95)


def prepare_output(args):
    name = args.run_name or f"m3c_balanced_keep_efficientnet_b0_seed{args.seed}"
    if args.debug:
        name += "_debug"
    output = args.output_root / name
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"输出已存在: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    return output


def save_failure_artifacts(output, args, frame, history, config):
    pd.DataFrame(history).to_csv(
        output / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    frame.to_csv(output / "frozen_split_snapshot.csv", index=False, encoding="utf-8-sig")
    (output / "config.json").write_text(
        json.dumps(json_ready(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), output / "source_entry.py")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.seed not in SEEDS:
        raise ValueError(f"seed只能来自{SEEDS}")
    if args.debug:
        args.epochs = min(args.epochs, 2)
        args.num_workers = 0
        args.min_fp_drop = -1.0
        args.min_group_cancer = 0
    seed_everything(args.seed)
    checkpoint = args.m1_checkpoint or default_m1_checkpoint(args.seed)
    if not checkpoint.is_file() or not args.manifest.is_file():
        raise FileNotFoundError(checkpoint if not checkpoint.is_file() else args.manifest)
    output = prepare_output(args)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    recorded_manifest = payload.get("config", {}).get("manifest")
    if recorded_manifest and Path(recorded_manifest).resolve() != args.manifest.resolve():
        raise ValueError("M1 checkpoint记录的manifest与M3c-B输入不一致")
    frame = load_train_val_manifest(
        args.manifest, args.image_root, args.debug, args.debug_units, args.seed
    )
    datasets = {
        split: M1Dataset(frame.loc[frame.split.eq(split)].copy(), args.image_root, split == "train")
        for split in ("train", "val")
    }
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        patient_class_balanced_weights(datasets["train"].df),
        num_samples=len(datasets["train"]), replacement=True, generator=generator,
    )
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker, persistent_workers=args.num_workers > 0,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker, persistent_workers=args.num_workers > 0,
        ),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m1 = EfficientNetM1()
    m1.load_state_dict(payload["model_state_dict"], strict=True)
    model = FrozenM1RegionGate(m1).to(device)
    freeze_before = frozen_m1_snapshot(model)
    baseline_eval = evaluate(model, loaders["val"], device, args)
    baseline = baseline_eval["baseline"]
    baseline_predictions = baseline_eval["predictions"].copy()
    print(
        f"设备: {device}; seed={args.seed}; train={len(datasets['train'])}张; "
        f"val={len(datasets['val'])}张; M1 FP={baseline['image_metrics']['noncancer_fp']:.4f}"
    )

    optimizer = optim.AdamW(
        model.gate.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history, best = [], {"key": None, "state": None, "record": None}
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics = train_epoch(model, loaders["train"], optimizer, device)
        val = evaluate(model, loaders["val"], device, args, baseline)
        summary = val["summary"]
        record = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in summary.items()},
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"epoch {epoch:02d}/{args.epochs} | loss={train_metrics['total']:.4f} "
            f"(pos={train_metrics['positive']:.4f}, neg={train_metrics['negative']:.4f}) | "
            f"正例ROI={train_metrics['positive_eligible']} | AUC={summary['gate_auc']:.4f} | "
            f"癌召回={summary['cancer_recall']:.4f} | FP={summary['noncancer_fp']:.4f} "
            f"(下降={summary['fp_drop']:.4f}) | 分组={summary['group_gate_passed']} | "
            f"eligible={summary['eligible']}"
        )
        key = (
            summary["noncancer_fp"],
            summary["patient_noncancer_fp"],
            -summary["gate_auc"],
        )
        improved = False
        if summary["eligible"] and (best["key"] is None or key < best["key"]):
            best = {
                "key": key,
                "state": deepcopy(model.gate.state_dict()),
                "record": record,
            }
            improved = True
        if epoch > args.minimum_epochs:
            no_improve = 0 if improved else no_improve + 1
            if no_improve >= args.early_stop_patience:
                print(f"early stop: 第{epoch}轮，连续{no_improve}轮无合格改善")
                break
        elif epoch == args.minimum_epochs:
            no_improve = 0

    freeze_after = frozen_m1_snapshot(model)
    freeze_ok = freeze_before == freeze_after
    final_current = evaluate(model, loaders["val"], device, args, baseline)["predictions"]
    classification_diff = float(np.max(np.abs(
        baseline_predictions.cancer_probability - final_current.cancer_probability
    )))
    box_columns = ["pred_x1", "pred_y1", "pred_x2", "pred_y2"]
    box_diff = float(np.max(np.abs(
        baseline_predictions[box_columns].to_numpy()
        - final_current[box_columns].to_numpy()
    )))
    if not freeze_ok or classification_diff > 1e-6 or box_diff > 1e-6:
        raise RuntimeError(
            f"冻结校验失败: checksum={freeze_ok}, cls_diff={classification_diff:.2e}, "
            f"box_diff={box_diff:.2e}"
        )
    print(
        f"冻结校验通过: 分类概率差={classification_diff:.2e}, 框坐标差={box_diff:.2e}"
    )

    base_config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "m1_checkpoint": str(checkpoint.resolve()),
        "m1_checkpoint_sha256": file_sha256(checkpoint),
        "baseline": baseline,
        "freeze_before": freeze_before,
        "freeze_after": freeze_after,
        "freeze_ok": freeze_ok,
        "classification_max_abs_diff": classification_diff,
        "box_coordinate_max_abs_diff": box_diff,
        "test_evaluated": False,
        "external_evaluated": False,
        **git_snapshot(),
    }
    if best["state"] is None:
        failure_config = {
            **base_config,
            "status": "failed_no_eligible_epoch",
            "m3c_selection": {"passed_seed_success": False, "record": None},
        }
        save_failure_artifacts(output, args, frame, history, failure_config)
        raise RuntimeError("M3c-B失败: 没有epoch同时通过FP下降和分辨率计数门槛")

    model.gate.load_state_dict(best["state"])
    final = evaluate(model, loaders["val"], device, args, baseline)
    final_predictions = final["predictions"]
    final_summary = final["summary"]
    patient_metrics_final, patient = patient_metrics(final_predictions, "gate_display")
    bootstrap = patient_bootstrap(patient, args.bootstrap, args.seed)
    torch.save({
        "gate_state_dict": model.gate.state_dict(),
        "config": json_ready({
            "seed": args.seed,
            "architecture": "Frozen M1 features[5] ROIAlign + independent region gate",
            "selection_record": best["record"],
            "m1_checkpoint": str(checkpoint.resolve()),
        }),
    }, output / "m3c_gate_best.pth")
    pd.DataFrame(history).to_csv(
        output / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    frame.to_csv(output / "frozen_split_snapshot.csv", index=False, encoding="utf-8-sig")
    final_predictions.to_csv(
        output / "val_gate_predictions.csv", index=False, encoding="utf-8-sig"
    )
    patient.to_csv(output / "val_patient_gate_predictions.csv", index=False, encoding="utf-8-sig")
    save_qc(final_predictions, args.image_root, output, args.qc_count)
    config = {
        **base_config,
        "status": "success",
        "m3c_selection": {
            "passed_seed_success": True,
            "record": best["record"],
            "final_summary": final_summary,
            "patient_metrics": patient_metrics_final,
            "patient_bootstrap": bootstrap,
        },
    }
    save_failure_artifacts(output, args, frame, history, config)
    print(f"输出目录: {output}")


def self_test():
    logits = torch.tensor([1.0, -1.0, 0.5, -0.5], requires_grad=True)
    labels = torch.tensor([1, 1, 0, 0])
    boxes = torch.tensor([
        [0.2, 0.2, 0.6, 0.6], [0.0, 0.0, 0.1, 0.1],
        [0.1, 0.1, 0.3, 0.3], [0.4, 0.4, 0.8, 0.8],
    ])
    targets = torch.tensor([
        [0.1, 0.1, 0.7, 0.7], [0.7, 0.7, 0.9, 0.9],
        [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
    ])
    losses = balanced_gate_loss(logits, labels, boxes, targets, labels.eq(1))
    if losses["positive_eligible"].tolist() != [True, False, False, False]:
        raise AssertionError("癌图ROI正例资格判定错误")
    losses["total"].backward()
    if not torch.isfinite(logits.grad).all():
        raise AssertionError("门控loss梯度非有限")
    values = np.arange(10, dtype=float)
    if np.mean(values >= lock_recall_threshold(values, 0.90)) != 0.90:
        raise AssertionError("阈值锁定错误")
    print("M3c-B自测通过: ROI正例资格、平衡BCE和90%召回阈值正确")


if __name__ == "__main__":
    main()
