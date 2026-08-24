#!/usr/bin/env python3
"""EfficientNet-B0 M3：在冻结M0特征上训练“癌相关可疑区域定位器”，抑制非癌假阳性。

M3从M1 warmup-only产品（m1_best_warmup_localization.pth）初始化，Encoder与全局分类头
全程冻结，只训练features[5]后的14x14定位头。癌图热图目标为病灶中心高斯，非癌图热图
目标为全零；正负热图损失独立归一化，避免batch癌非癌比例漂移损失尺度。

分类损失不参与梯度；val分类概率最大绝对差 <= 1e-6 作为冻结正确性的程序化校验。
模型选择只用val：合格门槛为 IoU/center-hit/固定M1阈值癌召回三条，主指标为固定M1阈值
下的非癌FP率。无任何合格epoch时判定失败并回退M1，禁止强行保存M3产品。

正式参数见`数据去偏重训练_参数与报告`与`实验进度与结果讨论.md`的M3预注册协议。
"""

import argparse
import hashlib
import json
import math
import os
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
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from train_utils import (  # noqa: E402
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
# 复用M1已冻结验证的组件：数据变换、数据集、模型、监督目标、解码、定位指标与QC。
from efficientnet_m1_localization import (  # noqa: E402
    GRID_SIZE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    BBoxAwareTransform,
    M1Dataset,
    LocalizationHead,
    EfficientNetM1,
    build_targets,
    gather_at_indices,
    decode_boxes,
    box_iou_and_coverage,
    tensor_to_pil,
    draw_normalized_box,
    save_contact_sheet,
    export_train_transform_qc,
    load_manifest as load_m1_manifest,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "结果/M3定位抑制_0804/正式验证集筛选"
# M1 warmup-only产品目录（`_warmup_product`后缀，六组已统一提取）。
M1_PRODUCT_ROOT = PROJECT_ROOT / "结果/M1辅助定位_0804/正式验证集筛选"
# 按数据角色自动选择对应M1冻结清单，避免入口误用错清单。
DEFAULT_MANIFESTS = {
    "balanced": (
        PROJECT_ROOT / "数据整理记录/图像裁剪"
        / "胃早癌概念提取训练集0804_预处理_v1"
        / "08_M1辅助定位清单_20260810"
        / "m1_balanced_keep_primary_1to1p3_split_seed42.csv"
    ),
    "full": (
        PROJECT_ROOT / "数据整理记录/图像裁剪"
        / "胃早癌概念提取训练集0804_预处理_v1"
        / "09_M1全量诊断清单_20260810"
        / "m1_full_keep_split_seed42.csv"
    ),
}


# -----------------------------------------------------------------------------
# 配置与冻结输入：只读取已验收的M1清单和同角色同种子的warmup-only产品。
# -----------------------------------------------------------------------------
def parse_args():
    """定义M3输入、损失权重、ramp、早停与选择规则的全部参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="M1清单；缺省按--role自动选择。",
    )
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--m1-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--role", choices=["balanced", "full"], default="balanced",
        help="数据角色，用于定位默认M1产品路径与默认清单；显式传对应参数时忽略。",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-size", type=float, default=0.1)
    parser.add_argument("--lambda-offset", type=float, default=1.0)
    parser.add_argument(
        "--w-neg-max", type=float, default=1.0,
        help="非癌热图损失ramp后的最高权重；1.0复现原M3。",
    )
    parser.add_argument(
        "--neg-topk", type=int, default=0,
        help="仅惩罚非癌热图最高k个响应；0表示原M3的14x14全网格。",
    )
    parser.add_argument("--w-neg-ramp-epochs", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--fixed-recall", type=float, default=0.90)
    parser.add_argument("--iou-tolerance", type=float, default=0.02)
    parser.add_argument("--cenhit-tolerance", type=float, default=0.05)
    parser.add_argument("--cancer-recall-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--max-cancer-detection-loss", type=int, default=-1,
        help="固定阈值下相对M1最多允许少检的癌图数；-1沿用召回率容差。",
    )
    parser.add_argument(
        "--resolution-gate-min-cancer", type=int, default=0,
        help="启用分辨率癌召回门槛时每组最少癌图数；0表示禁用。",
    )
    parser.add_argument(
        "--resolution-gate-max-detection-loss", type=int, default=1,
        help="每个有效分辨率组相对M1最多允许少检的癌图数。",
    )
    parser.add_argument("--fp-drop", type=float, default=0.10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-units", type=int, default=3)
    # 故意不提供--evaluate-test：M3模型选择阶段永不评估internal test，
    # config中test_evaluated恒为False，避免误解锁入口。
    parser.add_argument("--self-test", action="store_true",
                        help="运行阈值浮点回归测试后退出。")
    return parser.parse_args()


def default_m1_checkpoint(role, seed):
    """按数据角色与随机种子定位M1 warmup-only产品路径。"""
    prefix = "m1_balanced_keep" if role == "balanced" else "m1_full_keep"
    return (
        M1_PRODUCT_ROOT / f"{prefix}_efficientnet_b0_seed{seed}_warmup_product"
        / "m1_best_warmup_localization.pth"
    )


# -----------------------------------------------------------------------------
# 模型结构与冻结：Encoder/分类头全程冻结，仅定位头可训练；BN保持eval统计。
# -----------------------------------------------------------------------------
def set_m3_trainable(model):
    """冻结全部参数，仅放开定位头。"""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.localization_head.parameters():
        parameter.requires_grad = True


def set_m3_train_mode(model):
    """定位头处于训练模式；骨干与分类头保持eval（BN用running统计，避免漂移）。"""
    model.train()
    model.backbone.eval()


def freeze_snapshot(model):
    """对冻结部分生成参数与BN running统计的SHA-256指纹，供训练前后比对。"""
    state = {}
    for name, module in [
        ("encoder", model.backbone.features),
        ("classifier", model.backbone.classifier),
    ]:
        hasher = hashlib.sha256()
        for tensor in module.parameters():
            hasher.update(tensor.detach().cpu().numpy().tobytes())
        state[f"{name}_params_sha"] = hasher.hexdigest()
    bn_hasher = hashlib.sha256()
    # 只校验冻结骨干（backbone）的BN running统计；定位头BN可训练，不纳入。
    for module in model.backbone.modules():
        if isinstance(module, nn.BatchNorm2d):
            if module.running_mean is not None:
                bn_hasher.update(module.running_mean.detach().cpu().numpy().tobytes())
                bn_hasher.update(module.running_var.detach().cpu().numpy().tobytes())
    state["bn_running_sha"] = bn_hasher.hexdigest()
    return state


# -----------------------------------------------------------------------------
# 目标与损失：正负热图独立归一化，均按“每图空间总损失”取平均，尺度一致。
# -----------------------------------------------------------------------------
def focal_loss_per_image(logits, target):
    """每张图14x14空间总和的CenterNet focal loss；返回[B]。

    参数:
        logits (Tensor[B,1,14,14]): 未归一化热图logits。
        target (Tensor[B,1,14,14]): 目标热图（癌图为中心高斯）。
    返回:
        Tensor[B]: 每张图的focal空间总和（正例+负例分量）。
    """
    prediction = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    positive = target.eq(1).float()
    negative = target.lt(1).float()
    negative_weight = (1 - target).pow(4)
    positive_loss = torch.log(prediction) * (1 - prediction).pow(2) * positive
    negative_loss = (
        torch.log(1 - prediction) * prediction.pow(2) * negative_weight * negative
    )
    return -(positive_loss + negative_loss).flatten(1).sum(dim=1)


def neg_response_loss_per_image(logits, topk=0):
    """非癌负响应：全网格或最高k个位置求和 `-log(1-p)*p^2`；返回[B]。

    非癌目标为全零，(1-target)^4=1，故不再乘负样本权重。与正例同为“每图空间总损失”。
    topk=0严格复现原M3；topk>0只保留响应最高的位置，但仍使用求和而非均值，
    以延续“每图空间总损失”的约定。top-k会同时改变作用区域与原始损失量级。
    """
    prediction = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    prediction = prediction.flatten(1)
    if topk > 0:
        prediction = prediction.topk(k=min(topk, prediction.shape[1]), dim=1).values
    return -(torch.log(1 - prediction) * prediction.pow(2)).sum(dim=1)


def compute_m3_losses(outputs, labels, boxes, valid_box, w_neg, args):
    """组合M3总损失：癌图pos/size/offset，非癌图neg；缺失类别返回可微的0。

    参数:
        outputs (dict): 模型输出，含heatmap_logits[B,1,14,14]、size/offset[B,2]。
        labels (Tensor[B]): 图像标签，1=癌。
        boxes (Tensor[B,4]): 增强后归一化bbox。
        valid_box (Tensor[B]): 有效癌框掩码。
        w_neg (float): 当前epoch非癌热图权重。
        args: 含lambda_size/lambda_offset。
    返回:
        dict: pos/neg/size/offset/total。
    """
    heatmap_target, indices, size_target, offset_target = build_targets(boxes, valid_box)
    pos_mask = labels.eq(1)
    neg_mask = ~pos_mask
    if pos_mask.any():
        heatmap_pos = focal_loss_per_image(
            outputs["heatmap_logits"][pos_mask], heatmap_target[pos_mask]
        ).mean()
        if valid_box[pos_mask].any():
            pred_size = gather_at_indices(outputs["size"], indices)[pos_mask]
            pred_offset = gather_at_indices(outputs["offset"], indices)[pos_mask]
            size = nnf.smooth_l1_loss(
                pred_size[valid_box[pos_mask]], size_target[pos_mask][valid_box[pos_mask]]
            )
            offset = nnf.smooth_l1_loss(
                pred_offset[valid_box[pos_mask]],
                offset_target[pos_mask][valid_box[pos_mask]],
            )
        else:
            size = offset = outputs["size"].sum() * 0
    else:
        heatmap_pos = size = offset = outputs["heatmap_logits"].sum() * 0
    if neg_mask.any():
        heatmap_neg = neg_response_loss_per_image(
            outputs["heatmap_logits"][neg_mask], topk=args.neg_topk
        ).mean()
    else:
        heatmap_neg = outputs["heatmap_logits"].sum() * 0
    heatmap_neg_weighted = w_neg * heatmap_neg
    total = (
        heatmap_pos
        + heatmap_neg_weighted
        + args.lambda_size * size
        + args.lambda_offset * offset
    )
    return {
        "total": total,
        "heatmap_pos": heatmap_pos,
        "heatmap_neg_raw": heatmap_neg,
        "heatmap_neg_weighted": heatmap_neg_weighted,
        "size": size,
        "offset": offset,
    }


# -----------------------------------------------------------------------------
# 阈值与指标：固定M1阈值可执行定义；双阈值报告；三门槛合格判定。
# -----------------------------------------------------------------------------
def lock_recall_threshold(confidences, recall=0.90):
    """在候选置信度中选满足sensitivity>=recall的最高阈值；比较始终用>=。

    遍历唯一候选阈值（降序），返回第一个使 `mean(values >= t) >= recall` 的阈值。
    直接用实际召回率比较，避免 `(1-recall)*n` 的浮点取整误差（如10例目标0.90时
    `(1-0.9)*10` 会得到0.9999...导致floor后选错档），也正确处理重复分数。
    """
    values = np.asarray(confidences, dtype=float)
    if len(values) == 0:
        return 1.0
    for threshold in np.sort(np.unique(values))[::-1]:
        if float(np.mean(values >= threshold)) >= recall:
            return float(threshold)
    return float(values.min())


def lock_recall_threshold_self_test():
    """浮点回归测试：10/240个不同分数在目标召回0.90时都必须恰好得到0.90。"""
    for count in [10, 240]:
        values = np.sort(np.random.default_rng(0).random(count))
        threshold = lock_recall_threshold(values, recall=0.90)
        achieved = float(np.mean(values >= threshold))
        if abs(achieved - 0.90) > 1e-9:
            raise AssertionError(
                f"count={count}: 目标召回0.90，实际{achieved:.6f}，阈值={threshold:.6f}"
            )
    return True


def m3b_loss_self_test():
    """确认全网格分支向后兼容，top-k分支仅累加最高响应位置且数值有限。"""
    logits = torch.linspace(-3, 3, steps=2 * GRID_SIZE * GRID_SIZE).reshape(
        2, 1, GRID_SIZE, GRID_SIZE
    )
    prediction = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6).flatten(1)
    point_loss = -(torch.log(1 - prediction) * prediction.pow(2))
    full = neg_response_loss_per_image(logits, topk=0)
    top5 = neg_response_loss_per_image(logits, topk=5)
    expected_full = point_loss.sum(dim=1)
    expected_top5 = point_loss.gather(
        1, prediction.topk(k=5, dim=1).indices
    ).sum(dim=1)
    if not torch.allclose(full, expected_full):
        raise AssertionError("neg_topk=0未严格复现全网格负损失")
    if not torch.allclose(top5, expected_top5):
        raise AssertionError("neg_topk=5未严格累加最高5个响应位置")
    if not torch.isfinite(top5).all():
        raise AssertionError("top-k负损失出现非有限数")
    return True


def detection_rates_at_threshold(confidence, label, threshold):
    """在给定置信阈值下计算癌图检测召回率与非癌图FP率（图像级）。"""
    predicted = confidence >= threshold
    cancer = label.eq(1)
    recall = float(predicted[cancer].float().mean()) if cancer.any() else 1.0
    fp_rate = float(predicted[~cancer].float().mean()) if (~cancer).any() else 0.0
    return recall, fp_rate


def resolution_gate_counts(predictions, threshold, min_cancer):
    """按冻结的两档原始分辨率统计癌图检出数；小样本组不进入门槛。

    M1清单已有size_group。为避免balanced val的medium=4、small=11分别判定，
    预注册将二者合并为non_large=15，large_1025_1600保留为large=72。
    """
    if min_cancer <= 0 or "size_group" not in predictions.columns:
        return {}
    cancer = predictions.loc[predictions.label.eq(1)].copy()
    cancer["resolution_gate_group"] = cancer.size_group.map({
        "large_1025_1600": "large",
        "medium_641_1024": "non_large",
        "small_le640": "non_large",
    })
    result = {}
    for group_name, group in cancer.dropna(
        subset=["resolution_gate_group"]
    ).groupby("resolution_gate_group"):
        if len(group) < min_cancer:
            continue
        result[group_name] = {
            "n_cancer": int(len(group)),
            "detected": int((group.localization_confidence >= threshold).sum()),
        }
    return result


def resolution_gate_self_test():
    """确认medium+small按预注册合并，且最小样本门槛使用合并后的癌图数。"""
    frame = pd.DataFrame({
        "label": [1] * 30 + [0] * 2,
        "size_group": (
            ["large_1025_1600"] * 15
            + ["medium_641_1024"] * 4
            + ["small_le640"] * 11
            + ["large_1025_1600"] * 2
        ),
        "localization_confidence": [0.3] * 28 + [0.1] * 4,
    })
    groups = resolution_gate_counts(frame, threshold=0.2, min_cancer=15)
    if set(groups) != {"large", "non_large"}:
        raise AssertionError(f"分辨率门槛分组错误: {groups}")
    if groups["large"]["n_cancer"] != 15 or groups["non_large"]["n_cancer"] != 15:
        raise AssertionError(f"分辨率门槛癌图计数错误: {groups}")
    return True


# -----------------------------------------------------------------------------
# 评估与可视化：QC联系表、分层诊断、患者级bootstrap置信区间。
# -----------------------------------------------------------------------------
def export_prediction_qc(predictions, image_root, output, count=20):
    """导出癌图真值框vs预测框与定位置信度联系表。"""
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


def export_noncancer_high_response_qc(predictions, image_root, output, count=20):
    """导出非癌高响应（假阳性候选）分歧病例联系表。"""
    noncancer = predictions.loc[predictions.label.eq(0)]
    chosen = noncancer.sort_values("localization_confidence", ascending=False).head(count)
    images = []
    labels = []
    for row in chosen.itertuples(index=False):
        with Image.open(image_root / row.image_relpath) as source:
            image = source.convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
            )
        draw = ImageDraw.Draw(image)
        draw_normalized_box(
            draw, (row.pred_x1, row.pred_y1, row.pred_x2, row.pred_y2), "red"
        )
        images.append(image)
        labels.append(f"FP conf={row.localization_confidence:.3f}")
    save_contact_sheet(images, labels, output / "noncancer_high_response_qc.jpg")


def stratified_diagnostics(predictions, output, threshold,
                           lesion_tercile_bounds=None):
    """按来源/画幅/风格/PIP与病灶尺寸分桶输出癌召回与非癌FP，并保存CSV。

    参数:
        predictions: 含localization_confidence/label及分组列的val预测表。
        output: 输出目录（写stratified_diagnostics.csv）。
        threshold: 固定M1阈值；与主指标一致，避免0.5阈值口径混乱。
        lesion_tercile_bounds: train癌图bbox_area_fraction的[0.33,0.66]分位，
            用于对val病灶分桶；缺省时退回val自身分位（仅诊断，不参与选择）。
    """
    rows = []
    # 来源中心与画面子组（含PIP列，与声明一致）。
    group_columns = [
        column for column in ["source", "center", "year", "size_group",
                              "aspect_group", "frame_profile", "style_group",
                              "pip_present_original"]
        if column in predictions.columns
    ]
    for column in group_columns:
        for key, group in predictions.groupby(column):
            conf = torch.tensor(group.localization_confidence.to_numpy())
            lab = torch.tensor(group.label.to_numpy())
            recall, fp = detection_rates_at_threshold(conf, lab, threshold)
            rows.append({
                "grouping": column, "group": key,
                "n_images": len(group), "n_cancer": int(lab.eq(1).sum()),
                "cancer_recall_fixed": recall, "noncancer_fp_fixed": fp,
            })
    # 病灶尺寸分桶（按bbox面积分数三分位；边界优先来自train癌图）。
    valid = predictions.loc[predictions.valid_box].copy()
    if len(valid) and "bbox_area_fraction" in valid.columns:
        if lesion_tercile_bounds is None:
            lesion_tercile_bounds = valid.bbox_area_fraction.quantile([0.33, 0.66]).tolist()
        quantiles = lesion_tercile_bounds
        buckets = [("small", -np.inf, quantiles[0]),
                   ("medium", quantiles[0], quantiles[1]),
                   ("large", quantiles[1], np.inf)]
        for name, lo, hi in buckets:
            group = valid.loc[
                (valid.bbox_area_fraction > lo) & (valid.bbox_area_fraction <= hi)
            ]
            if len(group):
                iou, _ = box_iou_and_coverage(
                    torch.tensor(group[["pred_x1", "pred_y1", "pred_x2", "pred_y2"]].to_numpy()),
                    torch.tensor(group[["gt_x1", "gt_y1", "gt_x2", "gt_y2"]].to_numpy()),
                )
                rows.append({
                    "grouping": "lesion_size", "group": name,
                    "n_images": len(group), "n_cancer": len(group),
                    "cancer_recall_fixed": float(
                        (torch.tensor(group.localization_confidence.to_numpy()) >= threshold).float().mean()
                    ),
                    "noncancer_fp_fixed": float("nan"),
                    "mean_iou": float(iou.mean()),
                })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(output / "stratified_diagnostics.csv", index=False, encoding="utf-8-sig")
    return frame


def patient_bootstrap_rates(predictions, threshold, resamples=2000, seed=0):
    """患者整簇bootstrap癌召回与非癌FP的95%CI。

    以patient_id为抽样单元，有放回抽取后按抽中顺序拼接患者全部图片，
    保留重复抽中患者的重复权重（np.isin会折叠重复，不能使用）。结果仅用于报告。
    """
    rng = np.random.default_rng(seed)
    patients = predictions.patient_id.unique()
    recall_ci, fp_ci = [], []
    conf = predictions.localization_confidence.to_numpy()
    lab = predictions.label.to_numpy()
    pid = predictions.patient_id.to_numpy()
    patient_rows = {patient: np.flatnonzero(pid == patient) for patient in patients}
    for _ in range(resamples):
        sample_patients = rng.choice(patients, size=len(patients), replace=True)
        sampled_rows = np.concatenate([patient_rows[patient] for patient in sample_patients])
        sample_conf, sample_lab = conf[sampled_rows], lab[sampled_rows]
        predicted = sample_conf >= threshold
        cancer = sample_lab == 1
        if cancer.any():
            recall_ci.append(float(predicted[cancer].mean()))
        if (~cancer).any():
            fp_ci.append(float(predicted[~cancer].mean()))
    return {
        "cancer_recall_ci": [float(np.percentile(recall_ci, 2.5)),
                             float(np.percentile(recall_ci, 97.5))],
        "noncancer_fp_ci": [float(np.percentile(fp_ci, 2.5)),
                            float(np.percentile(fp_ci, 97.5))],
    }


# -----------------------------------------------------------------------------
# 训练与输出：ramp早停、三门槛选择、冻结校验、产品保存与config记录。
# -----------------------------------------------------------------------------
def train_epoch_m3(model, loader, optimizer, device, args, w_neg):
    """完成一个M3训练epoch，返回各损失分量的样本均值。"""
    set_m3_train_mode(model)
    totals = {
        key: 0.0 for key in [
            "total", "heatmap_pos", "heatmap_neg_raw", "heatmap_neg_weighted",
            "size", "offset",
        ]
    }
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        boxes = batch["bbox"].to(device, non_blocking=True)
        valid_box = batch["valid_box"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        losses = compute_m3_losses(outputs, labels, boxes, valid_box, w_neg, args)
        losses["total"].backward()
        optimizer.step()
        for key in totals:
            totals[key] += float(losses[key].detach()) * len(images)
    return {key: value / len(loader.dataset) for key, value in totals.items()}


def save_m3_checkpoint(path, model, args, epoch_record, m1_checkpoint):
    """保存M3模型、冻结参数与选择记录。"""
    serialized = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": json_ready({
            **serialized,
            "m1_checkpoint": str(m1_checkpoint.resolve()),
            "selection_record": epoch_record,
            "architecture": "EfficientNet-B0 frozen + M3 suppression loc head",
        }),
    }, path)


def prepare_run_dir(args):
    """创建独立运行目录；已有目录一律拒绝覆盖。"""
    name = args.run_name or f"m3_{args.role}_keep_efficientnet_b0_seed{args.seed}"
    if args.debug:
        name += "_debug"
    output = args.output_root / name
    if output.exists():
        raise FileExistsError(f"输出已存在，请更换运行名: {output}")
    output.mkdir(parents=True)
    return output


def main():
    """编排M1产品加载、冻结校验、ramp训练、三门槛选择、结果保存与分层诊断。"""
    args = parse_args()
    if args.self_test:
        lock_recall_threshold_self_test()
        m3b_loss_self_test()
        resolution_gate_self_test()
        print(
            "自测通过: 阈值10/240例目标0.90精确命中；"
            "负损失全网格向后兼容且top-5求和正确；分辨率合并门槛正确"
        )
        return
    if args.debug:
        args.epochs = min(args.epochs, 2)
        args.num_workers = 0
    if not 0 < args.fixed_recall <= 1:
        raise ValueError("fixed-recall必须在(0,1]内")
    if args.w_neg_max < 0:
        raise ValueError("w-neg-max不能小于0")
    if args.neg_topk < 0 or args.neg_topk > GRID_SIZE * GRID_SIZE:
        raise ValueError(f"neg-topk必须在0到{GRID_SIZE * GRID_SIZE}之间")
    if args.w_neg_ramp_epochs < 1:
        raise ValueError("w-neg-ramp-epochs必须至少为1")
    if args.max_cancer_detection_loss < -1:
        raise ValueError("max-cancer-detection-loss只能为-1或非负整数")
    if args.resolution_gate_min_cancer < 0:
        raise ValueError("resolution-gate-min-cancer不能小于0")
    if args.resolution_gate_max_detection_loss < 0:
        raise ValueError("resolution-gate-max-detection-loss不能小于0")
    seed_everything(args.seed)

    m1_checkpoint = args.m1_checkpoint or default_m1_checkpoint(args.role, args.seed)
    manifest = args.manifest or DEFAULT_MANIFESTS[args.role]
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not m1_checkpoint.is_file():
        raise FileNotFoundError(m1_checkpoint)
    output = prepare_run_dir(args)

    # 提前加载M1产品，校验其记录的manifest与输入一致，避免清单错配。
    m1_payload = torch.load(m1_checkpoint, map_location="cpu", weights_only=False)
    recorded_manifest = m1_payload.get("config", {}).get("manifest")
    if recorded_manifest and Path(str(recorded_manifest)).resolve() != manifest.resolve():
        raise ValueError(
            f"M1产品记录的manifest与输入不一致: {recorded_manifest} != {manifest}"
        )

    frame = load_m1_manifest(
        manifest, args.image_root, args.debug, args.debug_units, args.seed
    )
    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = M1Dataset(
            frame.loc[frame.split.eq(split)].copy(), args.image_root, split == "train"
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            persistent_workers=args.num_workers > 0,
        )
    export_train_transform_qc(loaders["train"].dataset, output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}; M1产品: {m1_checkpoint}")
    for split, loader in loaders.items():
        table = loader.dataset.df
        print(
            f"{split}: {table.patient_id.nunique()}人/{len(table)}张; "
            f"标签={table.label.value_counts().sort_index().to_dict()}"
        )

    model = EfficientNetM1()
    model.load_state_dict(m1_payload["model_state_dict"], strict=True)
    model.to(device)
    set_m3_trainable(model)
    set_m3_train_mode(model)
    freeze_before = freeze_snapshot(model)

    # 基线：M1产品在val上评估，并锁定固定M1阈值。
    baseline_eval = evaluate_m3_with_tolerances(
        model, loaders["val"], device, args
    )
    val_predictions = baseline_eval["predictions"]
    cancer_conf = val_predictions.loc[
        val_predictions.label.eq(1), "localization_confidence"
    ].to_numpy()
    fixed_threshold = lock_recall_threshold(cancer_conf, recall=args.fixed_recall)
    baseline = baseline_eval["baseline"]
    print(
        f"固定M1阈值={fixed_threshold:.4f} | 基线: IoU={baseline['mean_iou']:.4f} "
        f"cenhit={baseline['center_hit']:.4f} 癌召回={baseline['cancer_recall']:.4f} "
        f"非癌FP={baseline['fp_fixed']:.4f}"
    )

    # 分类不变性：M1产品在val上的分类概率。
    model_cls = EfficientNetM1()
    model_cls.load_state_dict(m1_payload["model_state_dict"], strict=True)
    model_cls.to(device).eval()
    m1_probs = []
    with torch.no_grad():
        for batch in loaders["val"]:
            images = batch["image"].to(device)
            logits = model_cls(images)["logits"]
            m1_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    m1_probs = torch.cat(m1_probs)

    history = []
    best_epoch = {"key": None, "state": None, "record": None}
    no_improve = 0
    optimizer = optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    for epoch in range(1, args.epochs + 1):
        ramp_fraction = min(
            1.0,
            max(0.0, (epoch - 1) / max(1, args.w_neg_ramp_epochs - 1)),
        )
        w_neg = args.w_neg_max * ramp_fraction
        started = time.time()
        train_losses = train_epoch_m3(
            model, loaders["train"], optimizer, device, args, w_neg
        )
        epoch_eval = evaluate_m3_with_tolerances(
            model, loaders["val"], device, args, fixed_threshold, baseline
        )
        summary = epoch_eval["summary"]
        record = {
            "epoch": epoch,
            "w_neg": w_neg,
            **{f"train_{key}": value for key, value in train_losses.items()},
            **{f"val_{key}": value for key, value in summary.items()},
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"epoch {epoch:02d}/{args.epochs} | w_neg={w_neg:.2f} | "
            f"train_total={train_losses['total']:.4f} | "
            f"pos={train_losses['heatmap_pos']:.4f} | "
            f"neg_raw={train_losses['heatmap_neg_raw']:.4f} | "
            f"neg_weighted={train_losses['heatmap_neg_weighted']:.4f} | "
            f"size={train_losses['size']:.4f} | offset={train_losses['offset']:.4f} | "
            f"IoU={summary['mean_iou']:.4f} | cenhit={summary['center_hit']:.4f} | "
            f"癌召回={summary['cancer_recall_fixed']:.4f} | "
            f"非癌FP={summary['fp_fixed']:.4f} | "
            f"癌计数门槛={summary['cancer_count_gate_passed']} | "
            f"分辨率门槛={summary['resolution_gate_passed']} | "
            f"eligible={summary['eligible']}"
        )
        # 完整排序键：固定阈值FP低→自身阈值FPR低→IoU高→非癌平均响应低。
        # 平手（键相等）也视为一次改善并重置patience，避免离散FP率导致提前停。
        selection_key = (
            summary["fp_fixed"],
            summary["fpr_at_self_sens90"],
            -summary["mean_iou"],
            summary["noncancer_max_response_mean"],
        )
        improved = False
        if summary["eligible"]:
            if best_epoch["state"] is None:
                best_epoch = {
                    "key": selection_key,
                    "state": deepcopy(model.state_dict()),
                    "record": record,
                }
                improved = True
            elif selection_key < best_epoch["key"]:
                best_epoch = {
                    "key": selection_key,
                    "state": deepcopy(model.state_dict()),
                    "record": record,
                }
                improved = True
            elif selection_key == best_epoch["key"]:
                improved = True  # 平手：重置patience但不替换产品。
        if epoch > args.w_neg_ramp_epochs:
            # ramp结束后才开始累计无改善；从第6轮起等待patience轮。
            no_improve = 0 if improved else no_improve + 1
            if no_improve >= args.early_stop_patience:
                print(f"early stop: 第{epoch}轮，连续{no_improve}轮无合格改善")
                break
        elif epoch == args.w_neg_ramp_epochs:
            no_improve = 0  # ramp结束清零，保证第6轮起按patience重计。

    # 冻结校验：参数与BN统计应完全不变，分类概率最大绝对差应<=1e-6。
    freeze_after = freeze_snapshot(model)
    freeze_ok = (
        freeze_before["encoder_params_sha"] == freeze_after["encoder_params_sha"]
        and freeze_before["classifier_params_sha"] == freeze_after["classifier_params_sha"]
        and freeze_before["bn_running_sha"] == freeze_after["bn_running_sha"]
    )
    model.eval()
    m3_probs = []
    with torch.no_grad():
        for batch in loaders["val"]:
            images = batch["image"].to(device)
            logits = model(images)["logits"]
            m3_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    m3_probs = torch.cat(m3_probs)
    prob_max_abs_diff = float((m1_probs - m3_probs).abs().max().item())
    if not freeze_ok or prob_max_abs_diff > 1e-6:
        raise RuntimeError(
            f"冻结校验失败: encoder一致={freeze_before['encoder_params_sha'] == freeze_after['encoder_params_sha']}, "
            f"classifier一致={freeze_before['classifier_params_sha'] == freeze_after['classifier_params_sha']}, "
            f"BN一致={freeze_before['bn_running_sha'] == freeze_after['bn_running_sha']}, "
            f"分类概率最大绝对差={prob_max_abs_diff:.2e} (>1e-6)"
        )
    print(f"冻结校验通过: 分类概率最大绝对差={prob_max_abs_diff:.2e}")

    if best_epoch["state"] is None:
        raise RuntimeError(
            "M3失败: 没有任何epoch同时通过几何、癌检出计数和分辨率合格门槛。"
            "按协议回退M1产品，不保存M3产品。"
        )

    model.load_state_dict(best_epoch["state"])
    save_m3_checkpoint(
        output / "m3_best_localization.pth", model, args, best_epoch["record"], m1_checkpoint
    )
    pd.DataFrame(history).to_csv(
        output / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    frame.to_csv(output / "frozen_split_snapshot.csv", index=False, encoding="utf-8-sig")
    final_eval = evaluate_m3_with_tolerances(
        model, loaders["val"], device, args, fixed_threshold, baseline
    )
    final_pred = final_eval["predictions"]
    final_pred.to_csv(output / "val_image_predictions.csv", index=False, encoding="utf-8-sig")
    final_eval["patient_predictions"].to_csv(
        output / "val_patient_predictions.csv", index=False, encoding="utf-8-sig"
    )
    export_prediction_qc(final_pred, args.image_root, output)
    export_noncancer_high_response_qc(final_pred, args.image_root, output)
    # 病灶三分位边界由train癌图确定，避免用val自身分位造成不一致。
    train_cancer_area = frame.loc[
        (frame.split.eq("train")) & (frame.label.eq(1)), "bbox_area_fraction"
    ]
    lesion_tercile_bounds = (
        train_cancer_area.quantile([0.33, 0.66]).tolist()
        if len(train_cancer_area) else None
    )
    stratified_diagnostics(final_pred, output, fixed_threshold, lesion_tercile_bounds)
    bootstrap = patient_bootstrap_rates(final_pred, fixed_threshold)

    # 成功门控：三门槛 + 非癌FP下降，均按预注册口径判定并写入config。
    fp_drop = baseline["fp_fixed"] - best_epoch["record"]["val_fp_fixed"]
    aux_base = baseline.get("noncancer_max_response_mean", 0.0)
    aux_drop_abs = aux_base - best_epoch["record"]["val_noncancer_max_response_mean"]
    aux_drop_rel = aux_drop_abs / max(aux_base, 1e-8)
    passed_fp_drop = fp_drop >= args.fp_drop
    aux_passed = (aux_drop_rel >= 0.25) and (aux_drop_abs >= 0.02)

    serialized = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config = {
        **serialized,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": file_sha256(manifest),
        "m1_checkpoint": str(m1_checkpoint.resolve()),
        "m1_checkpoint_sha256": file_sha256(m1_checkpoint),
        "fixed_threshold": fixed_threshold,
        "baseline": baseline,
        "freeze_ok": freeze_ok,
        "classification_max_abs_diff": prob_max_abs_diff,
        "m3_selection": {
            "record": best_epoch["record"],
            "fp_fixed": best_epoch["record"]["val_fp_fixed"],
            "fp_drop_vs_baseline": fp_drop,
            "passed_three_gates": True,
            "passed_all_eligibility_gates": True,
            "cancer_count_gate_passed": best_epoch["record"][
                "val_cancer_count_gate_passed"
            ],
            "resolution_gate_passed": best_epoch["record"][
                "val_resolution_gate_passed"
            ],
            "resolution_gate_details": best_epoch["record"][
                "val_resolution_gate_details"
            ],
            "passed_fp_drop": passed_fp_drop,
            "passed_seed_success": passed_fp_drop,
            "auxiliary_noncancer_drop_rel": aux_drop_rel,
            "auxiliary_noncancer_drop_abs": aux_drop_abs,
            "auxiliary_passed": aux_passed,
        },
        "lesion_tercile_bounds": lesion_tercile_bounds,
        "patient_bootstrap_ci": bootstrap,
        "test_evaluated": False,
        **git_snapshot(),
    }
    (output / "config.json").write_text(
        json.dumps(json_ready(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), output / "source_entry.py")
    print(f"输出目录: {output}")


def evaluate_m3_with_tolerances(model, loader, device, args, fixed_threshold=None,
                                baseline=None):
    """封装evaluate_m3：固定阈值缺失时先基于本split癌图锁定，并同时构建基线。

    M3的合格门槛与M1产品基线在main中首次调用时计算；后续epoch复用同一阈值与基线。
    """
    if fixed_threshold is None or baseline is None:
        # 首次调用（基线）：先粗评估拿癌图置信度，锁定阈值并计算基线指标。
        temp = evaluate_m3_core(model, loader, device, threshold=0.5)
        conf = temp["predictions"].localization_confidence.to_numpy()
        lab = temp["predictions"].label.to_numpy()
        fixed_threshold = lock_recall_threshold(conf[lab == 1], recall=args.fixed_recall)
        baseline = compute_baseline(temp["predictions"], fixed_threshold)
        baseline["mean_iou"] = temp["mean_iou"]
        baseline["center_hit"] = temp["center_hit"]
        baseline["resolution_gate_groups"] = resolution_gate_counts(
            temp["predictions"], fixed_threshold, args.resolution_gate_min_cancer
        )
        if args.resolution_gate_min_cancer > 0 and not baseline["resolution_gate_groups"]:
            raise ValueError(
                "已启用分辨率门槛，但val没有任何达到最小癌图数的可识别size_group"
            )
        return {"predictions": temp["predictions"],
                "patient_predictions": temp["patient_predictions"],
                "baseline": baseline,
                "summary": temp["summary"]}
    result = evaluate_m3_core(model, loader, device, threshold=fixed_threshold)
    summary = summarize_against_baseline(
        result, fixed_threshold, baseline, args
    )
    return {"predictions": result["predictions"],
            "patient_predictions": result["patient_predictions"],
            "summary": summary,
            "baseline": baseline}


def evaluate_m3_core(model, loader, device, threshold):
    """基础评估：图像级预测、患者聚合、癌定位IoU/center-hit。"""
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            boxes = batch["bbox"].to(device, non_blocking=True)
            valid_box = batch["valid_box"].to(device, non_blocking=True)
            outputs = model(images)
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
    predictions = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
    metadata = loader.dataset.df.copy().reset_index(drop=True)
    predictions = pd.concat([
        metadata, predictions.drop(columns=["row_index", "label"])
    ], axis=1)
    patient_predictions = patient_prediction_frame(predictions)
    valid = predictions.loc[predictions.valid_box]
    mean_iou = center_hit = 0.0
    if len(valid):
        iou, _ = box_iou_and_coverage(
            torch.tensor(valid[["pred_x1", "pred_y1", "pred_x2", "pred_y2"]].to_numpy()),
            torch.tensor(valid[["gt_x1", "gt_y1", "gt_x2", "gt_y2"]].to_numpy()),
        )
        mean_iou = float(iou.mean())
        center_hit = float(np.mean(
            ((valid[["pred_x1", "pred_x2"]].to_numpy().mean(axis=1)) >= valid["gt_x1"].to_numpy())
            & ((valid[["pred_x1", "pred_x2"]].to_numpy().mean(axis=1)) <= valid["gt_x2"].to_numpy())
            & ((valid[["pred_y1", "pred_y2"]].to_numpy().mean(axis=1)) >= valid["gt_y1"].to_numpy())
            & ((valid[["pred_y1", "pred_y2"]].to_numpy().mean(axis=1)) <= valid["gt_y2"].to_numpy())
        ))
    recall, fp = detection_rates_at_threshold(
        torch.tensor(predictions.localization_confidence.to_numpy()),
        torch.tensor(predictions.label.to_numpy()),
        threshold,
    )
    summary = {
        "mean_iou": mean_iou,
        "center_hit": center_hit,
        "cancer_recall_fixed": recall,
        "fp_fixed": fp,
    }
    return {"predictions": predictions,
            "patient_predictions": patient_predictions,
            "mean_iou": mean_iou,
            "center_hit": center_hit,
            "summary": summary}


def compute_baseline(predictions, fixed_threshold):
    """由M1产品val预测计算基线：IoU/center-hit/癌召回/非癌FP/非癌平均响应。

    mean_iou与center_hit由调用方覆盖（需要有效癌框的定位统计）。
    """
    conf = torch.tensor(predictions.localization_confidence.to_numpy())
    lab = torch.tensor(predictions.label.to_numpy())
    recall, fp = detection_rates_at_threshold(conf, lab, fixed_threshold)
    noncancer = predictions.loc[predictions.label.eq(0)]
    return {
        "mean_iou": 0.0,
        "center_hit": 0.0,
        "cancer_recall": recall,
        "cancer_count": int(lab.eq(1).sum()),
        "cancer_detected": int(
            ((conf >= fixed_threshold) & lab.eq(1)).sum().item()
        ),
        "fp_fixed": fp,
        "noncancer_max_response_mean": (
            float(noncancer.localization_confidence.mean()) if len(noncancer) else 0.0
        ),
    }


def summarize_against_baseline(result, fixed_threshold, baseline, args):
    """按几何、癌检出计数和分辨率门槛判定epoch，并输出主/次指标。"""
    mean_iou = result["mean_iou"]
    center_hit = result["center_hit"]
    recall, fp = detection_rates_at_threshold(
        torch.tensor(result["predictions"].localization_confidence.to_numpy()),
        torch.tensor(result["predictions"].label.to_numpy()),
        fixed_threshold,
    )
    self_threshold = lock_recall_threshold(
        result["predictions"].loc[
            result["predictions"].label.eq(1), "localization_confidence"
        ].to_numpy(),
        recall=0.90,
    )
    _, fp_self = detection_rates_at_threshold(
        torch.tensor(result["predictions"].localization_confidence.to_numpy()),
        torch.tensor(result["predictions"].label.to_numpy()),
        self_threshold,
    )
    noncancer = result["predictions"].loc[result["predictions"].label.eq(0)]
    noncancer_max_mean = (
        float(noncancer.localization_confidence.mean()) if len(noncancer) else 0.0
    )
    cancer = result["predictions"].loc[result["predictions"].label.eq(1)]
    cancer_detected = int(
        (cancer.localization_confidence >= fixed_threshold).sum()
    )
    if args.max_cancer_detection_loss >= 0:
        cancer_count_gate_passed = (
            cancer_detected
            >= baseline["cancer_detected"] - args.max_cancer_detection_loss
        )
    else:
        cancer_count_gate_passed = (
            recall >= baseline["cancer_recall"] - args.cancer_recall_tolerance
        )

    current_resolution = resolution_gate_counts(
        result["predictions"], fixed_threshold, args.resolution_gate_min_cancer
    )
    resolution_gate_details = {}
    for group_name, base_group in baseline.get("resolution_gate_groups", {}).items():
        current_group = current_resolution.get(group_name, {"detected": 0})
        passed = (
            current_group["detected"]
            >= base_group["detected"] - args.resolution_gate_max_detection_loss
        )
        resolution_gate_details[group_name] = {
            "n_cancer": base_group["n_cancer"],
            "baseline_detected": base_group["detected"],
            "current_detected": current_group["detected"],
            "max_allowed_loss": args.resolution_gate_max_detection_loss,
            "passed": passed,
        }
    resolution_gate_passed = all(
        detail["passed"] for detail in resolution_gate_details.values()
    )
    eligible = (
        mean_iou >= baseline["mean_iou"] - args.iou_tolerance
        and center_hit >= baseline["center_hit"] - args.cenhit_tolerance
        and cancer_count_gate_passed
        and resolution_gate_passed
    )
    return {
        "mean_iou": mean_iou,
        "center_hit": center_hit,
        "cancer_recall_fixed": recall,
        "fp_fixed": fp,
        "fpr_at_self_sens90": fp_self,
        "self_threshold": self_threshold,
        "noncancer_max_response_mean": noncancer_max_mean,
        "cancer_detected": cancer_detected,
        "cancer_count_gate_passed": cancer_count_gate_passed,
        "resolution_gate_passed": resolution_gate_passed,
        "resolution_gate_details": resolution_gate_details,
        "eligible": eligible,
    }


if __name__ == "__main__":
    main()
