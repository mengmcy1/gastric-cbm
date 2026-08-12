#!/usr/bin/env python3
"""EfficientNet-B0 M4：冻结M1全局/定位基座，用医生真值ROI或M1预测ROI训练局部+融合分支。

按M4修订预注册（2026-08-11）：癌图使用医生真值ROI、非癌图使用M1预测ROI，只训练
独立局部Encoder副本、局部分类头和融合分类头；M1全局分支全程冻结，q_region/hard-easy
不输入融合头。检验高质量病灶局部信息能否在三个seed上稳定提高患者级分类AUC。

输入：冻结ROI清单（`build_m4_roi_manifest.py`产物，含三套坐标/hard-easy/门控阈值）。
正式参数见`实验进度与结果讨论.md`的M4修订预注册。
"""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from efficientnet_m1_localization import (  # noqa: E402
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    EfficientNetM1,
)
from efficientnet_m3c_region_gate import (  # noqa: E402
    patient_class_balanced_weights,
    lock_recall_threshold,
)
from train_utils import (  # noqa: E402
    file_sha256,
    git_snapshot,
    json_ready,
    seed_everything,
    seed_worker,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "结果/M4真值ROI融合_0804/正式验证集筛选"
ROI_ROOT = PROJECT_ROOT / "结果/M4真值ROI融合_0804/冻结ROI清单"
M1_ROOT = PROJECT_ROOT / "结果/M1辅助定位_0804/正式验证集筛选"
SEEDS = (42, 202, 503)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-manifest", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--m1-checkpoint", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stage-a-epochs", type=int, default=5)
    parser.add_argument("--stage-b-epochs", type=int, default=20)
    parser.add_argument("--stage-a-lr", type=float, default=1e-3)
    parser.add_argument("--stage-b-lr", type=float, default=1e-4)
    parser.add_argument("--stage-b-backbone-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-units", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="运行bootstrap聚合与辅助指标回归自测后退出。")
    return parser.parse_args()


def default_roi_manifest(seed):
    return ROI_ROOT / f"m4_roi_manifest_seed{seed}.csv"


def default_m1_checkpoint(seed):
    return (
        M1_ROOT / f"m1_balanced_keep_efficientnet_b0_seed{seed}_warmup_product"
        / "m1_best_warmup_localization.pth"
    )


def load_roi_manifest(path, image_root, debug, debug_units, seed):
    """校验并加载冻结ROI清单，仅保留train/val；debug时按split+标签抽样患者。

    参数:
        path: m4_roi_manifest CSV路径。
        image_root: 项目根，用于校验图像文件存在。
        debug/debug_units/seed: 调试抽样参数。
    返回:
        pd.DataFrame: train/val行，含roi_crop_bbox三套坐标与noncancer_stratum。
    校验: 只含train/val、无患者跨split/跨标签、ROI裁剪框在界且图像存在。
    """
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"patient_id": str})
    required = {
        "image_relpath", "patient_id", "label", "split", "roi_source",
        "roi_crop_bbox_x1", "roi_crop_bbox_y1", "roi_crop_bbox_x2",
        "roi_crop_bbox_y2", "width", "height", "q_region", "noncancer_stratum",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ROI清单缺少字段: {sorted(missing)}")
    if set(frame.split.unique()) != {"train", "val"}:
        raise ValueError("ROI清单应只含train/val")
    if frame.groupby("patient_id").split.nunique().gt(1).any():
        raise ValueError("患者跨split")
    if frame.groupby("patient_id").label.nunique().gt(1).any():
        raise ValueError("患者跨标签")
    for row in frame.itertuples(index=False):
        crop = (row.roi_crop_bbox_x1, row.roi_crop_bbox_y1,
                row.roi_crop_bbox_x2, row.roi_crop_bbox_y2)
        if not (0 <= crop[0] < crop[2] <= 1 and 0 <= crop[1] < crop[3] <= 1):
            raise ValueError(f"ROI裁剪框越界: {row.image_relpath}")
        if not (image_root / row.image_relpath).is_file():
            raise FileNotFoundError(row.image_relpath)
    if debug:
        selected = []
        for split, split_frame in frame.groupby("split", sort=False):
            patients = split_frame[["patient_id", "label"]].drop_duplicates()
            sampled = pd.concat([
                group.sample(min(debug_units, len(group)), random_state=seed)
                for _, group in patients.groupby("label", sort=True)
            ])
            selected.append(split_frame.loc[split_frame.patient_id.isin(sampled.patient_id)])
        frame = pd.concat(selected, ignore_index=True)
    return frame.reset_index(drop=True)


def build_transforms():
    """构建全局图与局部ROI的训练/评估变换。

    参数: 无。
    返回:
        tuple: 全局确定性eval变换、局部ROI训练增强、局部ROI确定性eval变换。
    """
    normalize = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    eval_transform = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True), normalize,
    ])
    roi_train_transform = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        normalize,
    ])
    roi_eval_transform = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True), normalize,
    ])
    return eval_transform, roi_train_transform, roi_eval_transform


class M4Dataset(Dataset):
    """读取冻结ROI清单：完整图走全局分支(eval变换)，ROI裁图走局部分支。

    参数:
        frame: 单个split的ROI清单行；image_root: 图像根目录；
        training: 是否训练模式（决定局部ROI是否使用增强）；
        eval_transform/roi_train_transform/roi_eval_transform: 全局与局部变换。
    __getitem__返回:
        image[3,224,224]、roi[3,224,224]、label(long)、roi_label(long，癌ROI=1)、
        row_index(long)；roi按roi_crop_bbox从正式处理图裁剪并resize到224。
    """

    def __init__(self, frame, image_root, training,
                 eval_transform, roi_train_transform, roi_eval_transform):
        self.df = frame.reset_index(drop=True)
        self.image_root = image_root
        self.training = training
        self.eval_transform = eval_transform
        self.roi_transform = (
            roi_train_transform if training else roi_eval_transform
        )

    def _load_roi(self, row):
        with Image.open(self.image_root / row.image_relpath) as source:
            image = source.convert("RGB")
        width, height = image.size
        left = int(row.roi_crop_bbox_x1 * width)
        top = int(row.roi_crop_bbox_y1 * height)
        right = int(row.roi_crop_bbox_x2 * width)
        bottom = int(row.roi_crop_bbox_y2 * height)
        roi = image.crop((left, top, right, bottom)).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        )
        return roi

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(self.image_root / row.image_relpath) as source:
            full = source.convert("RGB")
        full = self.eval_transform(full)
        roi = self.roi_transform(self._load_roi(row))
        roi_label = 1 if int(row.label) == 1 else 0
        return {
            "image": full,
            "roi": roi,
            "label": torch.tensor(int(row.label), dtype=torch.long),
            "roi_label": torch.tensor(roi_label, dtype=torch.long),
            "row_index": torch.tensor(index, dtype=torch.long),
        }


class M4Model(nn.Module):
    """冻结M1全局分支 + 独立局部Encoder副本 + 局部分类头 + 融合分类头。

    参数:
        m1_backbone: 冻结M1的backbone（efficientnet_b0实例），其features/avgpool/
            classifier被全局分支直接复用并全程eval；局部Encoder为features+avgpool的
            独立deepcopy，不共享参数对象。
    forward:
        输入: image[B,3,224,224]（全局分支，冻结特征）、roi[B,3,224,224]（局部ROI）。
        返回: global_logits[B,2]、local_logits[B,2]、fusion_logits[B,2]、
              global_feature[B,1280]、local_feature[B,1280]。
    """

    def __init__(self, m1_backbone):
        super().__init__()
        # 全局分支：直接复用冻结M1的features/avgpool/classifier。
        self.global_features = m1_backbone.features
        self.global_avgpool = m1_backbone.avgpool
        self.global_classifier = m1_backbone.classifier
        for parameter in self.global_features.parameters():
            parameter.requires_grad = False
        for parameter in self.global_classifier.parameters():
            parameter.requires_grad = False
        # 局部Encoder：独立副本（不与全局共享参数对象）。
        self.local_features = deepcopy(m1_backbone.features)
        self.local_avgpool = deepcopy(m1_backbone.avgpool)
        # 局部分类头。
        self.local_head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(1280, 2),
        )
        # 融合分类头。
        self.fusion_head = nn.Sequential(
            nn.Linear(2560, 512), nn.SiLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(512, 2),
        )

    def global_forward(self, image):
        with torch.no_grad():
            feature = self.global_avgpool(self.global_features(image))
            feature = torch.flatten(feature, 1)
            logits = self.global_classifier(feature)
        return feature, logits

    def forward(self, image, roi):
        global_feat, global_logits = self.global_forward(image)
        local_feat = self.local_avgpool(self.local_features(roi))
        local_feat = torch.flatten(local_feat, 1)
        local_logits = self.local_head(local_feat)
        fusion_logits = self.fusion_head(
            torch.cat([global_feat, local_feat], dim=1)
        )
        return {
            "global_logits": global_logits,
            "local_logits": local_logits,
            "fusion_logits": fusion_logits,
            "global_feature": global_feat,
            "local_feature": local_feat,
        }


def set_eval_mode(model):
    """将M4模型整体置于eval，并确保全局与局部features均保持eval（BN用running统计）。"""
    model.eval()
    model.global_features.eval()
    model.local_features.eval()


def set_m4_train_mode(model, stage):
    """阶段专属训练模式：冻结层的BN running统计必须保持eval。

    阶段A：局部Encoder全部eval（无梯度且BN不更新），只训局部头+融合头；
    阶段B：local_features全部eval，仅features[7:]置train并更新；
    global_features与global_classifier始终eval。

    参数:
        model: M4Model实例。
        stage: "A"或"B"，决定局部Encoder的解冻范围。
    返回: 无；原地修改模型模式与requires_grad。
    """
    model.train()
    model.global_features.eval()
    for parameter in model.global_features.parameters():
        parameter.requires_grad = False
    for parameter in model.global_classifier.parameters():
        parameter.requires_grad = False
    if stage == "A":
        model.local_features.eval()
        for parameter in model.local_features.parameters():
            parameter.requires_grad = False
    elif stage == "B":
        for index in range(len(model.local_features)):
            module = model.local_features[index]
            if index < 7:
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False
            else:
                module.train()
                for parameter in module.parameters():
                    parameter.requires_grad = True
    model.local_head.train()
    model.fusion_head.train()


def frozen_m1_snapshot(model):
    """对冻结M1全局分支（features+classifier）生成参数与BN running统计的SHA-256指纹。

    参数:
        model: M4Model实例。
    返回:
        dict: global_params_sha、bn_running_sha，用于训练前后比对证明全局分支未变。
    """
    state = {}
    hasher = hashlib.sha256()
    for parameter in model.global_features.parameters():
        hasher.update(parameter.detach().cpu().numpy().tobytes())
    for parameter in model.global_classifier.parameters():
        hasher.update(parameter.detach().cpu().numpy().tobytes())
    state["global_params_sha"] = hasher.hexdigest()
    bn_hasher = hashlib.sha256()
    for module in model.global_features.modules():
        if isinstance(module, nn.BatchNorm2d) and module.running_mean is not None:
            bn_hasher.update(module.running_mean.detach().cpu().numpy().tobytes())
            bn_hasher.update(module.running_var.detach().cpu().numpy().tobytes())
    state["bn_running_sha"] = bn_hasher.hexdigest()
    return state


def compute_losses(outputs, labels, roi_labels, args):
    """M4总损失：融合CE + beta*局部CE。

    参数:
        outputs: 模型输出（含fusion_logits[B,2]、local_logits[B,2]）。
        labels: 图像标签[B]；roi_labels: ROI标签[B]（癌ROI=1、非癌ROI=0）。
        args: 含beta（默认0.5）。
    返回:
        dict: total/fusion/local三个标量损失。
    梯度路径: L_final经融合头更新融合头与未冻结局部Encoder（不经过局部分类头）；
    L_local更新局部分类头与未冻结局部Encoder。
    """
    classification = nn.functional.cross_entropy(outputs["fusion_logits"], labels)
    local = nn.functional.cross_entropy(outputs["local_logits"], roi_labels)
    total = classification + args.beta * local
    return {
        "total": total, "fusion": classification, "local": local,
    }


def patient_top2_mean(predictions, probability_column, group_column="patient_id"):
    """冻结的top2_mean患者聚合：每患者（或bootstrap抽样单元）取图片概率Top-2的均值。

    参数:
        predictions: 含patient_id/label与概率列的图像级表。
        probability_column: 用于聚合的概率列名。
        group_column: 聚合单元列（默认patient_id；bootstrap用_bunit）。
    返回:
        DataFrame: 列[label, patient_probability]，每单元一行（label取单元首行）。
    """
    def agg(group):
        values = np.sort(group[probability_column].to_numpy())[::-1]
        return float(values[:2].mean())
    result = predictions.groupby(group_column, as_index=False).apply(
        lambda group: pd.Series({
            "label": int(group.label.iloc[0]),
            "patient_probability": agg(group),
        }), include_groups=False
    )
    return result


def evaluate(model, loader, device, args):
    """在给定split上无梯度评估，输出图像级预测与AUC指标。

    参数:
        model: M4Model实例；loader: 该split的DataLoader；device: cuda/cpu；
        args: 当前参数（用于兼容签名，暂未直接使用）。
    返回:
        dict: predictions（含global/fusion/local概率与ROI元数据合并）、
              metrics（image/patient的fusion与global AUC、val_fusion_loss）。
    说明: 患者聚合使用top2_mean；全局分支恒eval。
    """
    set_eval_mode(model)
    rows = []
    total_fusion_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            rois = batch["roi"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            outputs = model(images, rois)
            global_probs = torch.softmax(outputs["global_logits"], dim=1)[:, 1]
            fusion_probs = torch.softmax(outputs["fusion_logits"], dim=1)[:, 1]
            local_probs = torch.softmax(outputs["local_logits"], dim=1)[:, 1]
            total_fusion_loss += float(
                nn.functional.cross_entropy(
                    outputs["fusion_logits"], labels
                ).item()
            ) * len(images)
            for local_index, row_index in enumerate(batch["row_index"].tolist()):
                rows.append({
                    "row_index": row_index,
                    "label": int(labels[local_index]),
                    "global_probability": float(global_probs[local_index]),
                    "fusion_probability": float(fusion_probs[local_index]),
                    "local_probability": float(local_probs[local_index]),
                })
    predictions = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
    metadata = loader.dataset.df.copy().reset_index(drop=True)
    predictions = pd.concat([
        metadata, predictions.drop(columns=["row_index", "label"])
    ], axis=1)
    val_fusion_loss = total_fusion_loss / len(loader.dataset)

    def image_auc(sub):
        if sub.label.nunique() < 2:
            return 0.5
        return float(roc_auc_score(sub.label, sub.fusion_probability))

    def patient_auc(sub, column):
        patients = patient_top2_mean(sub, column)
        if patients.label.nunique() < 2:
            return 0.5
        return float(roc_auc_score(patients.label, patients.patient_probability))

    metrics = {
        "image_fusion_auc": image_auc(predictions),
        "patient_fusion_auc": patient_auc(predictions, "fusion_probability"),
        "patient_global_auc": patient_auc(predictions, "global_probability"),
        "image_global_auc": float(roc_auc_score(
            predictions.label, predictions.global_probability
        )) if predictions.label.nunique() >= 2 else 0.5,
        "val_fusion_loss": val_fusion_loss,
    }
    return {"predictions": predictions, "metrics": metrics}


def patient_paired_bootstrap(predictions, iterations, seed):
    """患者整簇配对bootstrap，保留有放回重复患者的重复权重。

    为每次抽样实例生成独立ID（_bunit）并按该ID聚合top2_mean，避免重复抽中的
    同一患者被重新合并、重复权重被消除。

    参数:
        predictions: val图像级表，含患者ID、标签及全局/融合概率。
        iterations: bootstrap重采样次数。
        seed: 随机数种子。
    返回:
        dict: 配对ΔAUC的95% CI、均值与有效迭代数。
    """
    rng = np.random.default_rng(seed)
    patients = predictions.patient_id.unique()
    deltas = []
    for iteration in range(iterations):
        sample = rng.choice(patients, size=len(patients), replace=True)
        chunks = []
        for occurrence, patient in enumerate(sample):
            chunk = predictions.loc[predictions.patient_id.eq(patient)].copy()
            chunk["_bunit"] = f"b{iteration}_{occurrence}"
            chunks.append(chunk)
        sample_frame = pd.concat(chunks, ignore_index=True)
        glob = patient_top2_mean(sample_frame, "global_probability", "_bunit")
        fus = patient_top2_mean(sample_frame, "fusion_probability", "_bunit")
        if glob.label.nunique() < 2:
            continue
        delta = (float(roc_auc_score(fus.label, fus.patient_probability))
                 - float(roc_auc_score(glob.label, glob.patient_probability)))
        deltas.append(delta)
    deltas = np.asarray(deltas)
    return {
        "delta_ci95": [float(np.percentile(deltas, 2.5)),
                       float(np.percentile(deltas, 97.5))],
        "delta_mean": float(deltas.mean()),
        "iterations_used": len(deltas),
    }


def metadata_audit(frame):
    """三档元数据审计：A纯几何 / B q_region-only / C几何+q_region。

    参数:
        frame: 含train/val的冻结ROI清单（含split、roi_base_bbox、q_region、label）。
    返回:
        dict: 各档AUC；用train元数据拟合LogisticRegression，在val上评估，避免同集
        拟合评估的乐观偏差。标签退化时明确返回0.5，其他错误直接抛出不伪装。
    """
    train = frame.loc[frame.split.eq("train")]
    val = frame.loc[frame.split.eq("val")]

    def geometry_matrix(sub):
        geo = sub[[
            "roi_base_bbox_x1", "roi_base_bbox_y1", "roi_base_bbox_x2",
            "roi_base_bbox_y2",
        ]].copy()
        geo["width"] = geo.roi_base_bbox_x2 - geo.roi_base_bbox_x1
        geo["height"] = geo.roi_base_bbox_y2 - geo.roi_base_bbox_y1
        geo["area"] = geo.width * geo.height
        geo["aspect"] = geo.width / geo.height.clip(lower=1e-8)
        geo["center_x"] = (geo.roi_base_bbox_x1 + geo.roi_base_bbox_x2) / 2
        geo["center_y"] = (geo.roi_base_bbox_y1 + geo.roi_base_bbox_y2) / 2
        return geo.to_numpy(), sub.label.to_numpy()

    geo_tr, y_tr = geometry_matrix(train)
    geo_va, y_va = geometry_matrix(val)
    q_tr = train.q_region.to_numpy().reshape(-1, 1)
    q_va = val.q_region.to_numpy().reshape(-1, 1)

    def eval_auc(X_tr, y_tr, X_va, y_va):
        # 标签退化已提前判断返回0.5；拟合/评估的其他错误应直接暴露，不伪装成正常AUC。
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            return 0.5
        model = LogisticRegression(max_iter=500)
        model.fit(X_tr, y_tr)
        return float(roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]))

    return {
        "A_pure_geometry_auc": eval_auc(geo_tr, y_tr, geo_va, y_va),
        "B_qregion_only_auc": eval_auc(q_tr, y_tr, q_va, y_va),
        "C_geometry_plus_qregion_auc": eval_auc(
            np.hstack([geo_tr, q_tr]), y_tr, np.hstack([geo_va, q_va]), y_va
        ),
    }


def compute_auxiliary_metrics(predictions, args, roi_config):
    """协议要求的辅助指标：局部头/hard-easy/临床阈值/分层/一致率/新增对错患者。

    参数:
        predictions: val图像级预测（含global/fusion/local概率与ROI元数据）。
        args: 当前参数（用于兼容，未直接使用）。
        roi_config: 冻结ROI config，提供病灶大小三分位边界（lesion_tercile_bounds）。
    返回:
        dict: local_head_auc、local_auc_cancer_vs_{hard,easy}、local_acc_{hard,easy}、
              patient_sens/spec_{sens90,default_0_5}、patient_threshold_sens90、
              stratified(来源/中心/分辨率)、lesion_size(小中大病灶)、
              agreement_default_0_5与方向计数、new_correct/new_error_patients。
    """
    result = {}
    # 局部头整体AUC：以roi_label（癌ROI=1、非癌ROI=0）为真值。
    local = predictions.loc[predictions.split.eq("val")].copy()
    local["roi_label"] = (local.label == 1).astype(int)
    if local.roi_label.nunique() >= 2:
        result["local_head_auc"] = float(roc_auc_score(
            local.roi_label, local.local_probability
        ))
    else:
        result["local_head_auc"] = 0.5
    # 癌ROI vs hard/easy 的局部AUC（局部分支对这两类难度的区分力）。
    for name, mask in [("cancer_vs_hard",
                        local.label.eq(1) | local.noncancer_stratum.eq("hard")),
                       ("cancer_vs_easy",
                        local.label.eq(1) | local.noncancer_stratum.eq("easy"))]:
        sub = local.loc[mask]
        if sub.roi_label.nunique() >= 2:
            result[f"local_auc_{name}"] = float(roc_auc_score(
                sub.roi_label, sub.local_probability
            ))
        else:
            result[f"local_auc_{name}"] = 0.5
    # hard/easy准确率：局部头阈值0.5下对roi_label的判对比例。
    for name in ["hard", "easy"]:
        sub = local.loc[local.noncancer_stratum.eq(name)]
        if len(sub):
            result[f"local_acc_{name}"] = float(
                ((sub.local_probability >= 0.5) == sub.roi_label).mean()
            )
    # 临床阈值：主口径=val患者Sensitivity>=0.90的最高阈值（与M0一致）；
    # 同时报告0.5作为默认参考。
    patients = patient_top2_mean(local, "fusion_probability")
    if len(patients) and patients.label.nunique() >= 2:
        clinical_threshold = lock_recall_threshold(
            patients.loc[patients.label.eq(1), "patient_probability"].to_numpy(),
            recall=0.90,
        )
        for threshold, label in [(clinical_threshold, "sens90"),
                                 (0.5, "default_0_5")]:
            pred = (patients.patient_probability >= threshold).astype(int)
            tp = int(((pred == 1) & (patients.label == 1)).sum())
            fp = int(((pred == 1) & (patients.label == 0)).sum())
            fn = int(((pred == 0) & (patients.label == 1)).sum())
            tn = int(((pred == 0) & (patients.label == 0)).sum())
            result[f"patient_sens_{label}"] = float(tp / max(tp + fn, 1))
            result[f"patient_spec_{label}"] = float(tn / max(tn + fp, 1))
        result["patient_threshold_sens90"] = clinical_threshold
    # 来源/中心/分辨率分层：图像级与患者级融合AUC，并报告每层图片/患者数。
    result["stratified"] = {}
    for group_col in [column for column in ["source", "center", "size_group"]
                      if column in local.columns]:
        for key, group in local.groupby(group_col):
            if group.label.nunique() < 2:
                continue
            group_patients = patient_top2_mean(group, "fusion_probability")
            result["stratified"][f"{group_col}:{key}"] = {
                "n_images": int(len(group)),
                "n_patients": int(group_patients.patient_id.nunique()),
                "image_auc": float(roc_auc_score(
                    group.label, group.fusion_probability
                )),
                "patient_auc": float(roc_auc_score(
                    group_patients.label, group_patients.patient_probability
                )),
            }
    # 病灶大小分层：只用val癌图，按train冻结的三分位分small/medium/large。
    # 各组只有癌标签，不适合二分类AUC；报告图片/患者数、召回率与全局/融合平均概率及变化。
    result["lesion_size"] = {}
    lesion_bounds = roi_config.get("lesion_tercile_bounds")
    cancer_val = local.loc[local.label.eq(1)].copy()
    if lesion_bounds and len(cancer_val):
        buckets = [("small", -np.inf, lesion_bounds[0]),
                   ("medium", lesion_bounds[0], lesion_bounds[1]),
                   ("large", lesion_bounds[1], np.inf)]
        for name, lo, hi in buckets:
            group = cancer_val.loc[
                (cancer_val.bbox_area_fraction > lo)
                & (cancer_val.bbox_area_fraction <= hi)
            ]
            if len(group):
                recall = float(
                    (group.fusion_probability >= 0.5).mean()
                )
                result["lesion_size"][name] = {
                    "n_images": int(len(group)),
                    "n_patients": int(group.patient_id.nunique()),
                    "recall_default_0_5": recall,
                    "mean_global_probability": float(group.global_probability.mean()),
                    "mean_fusion_probability": float(group.fusion_probability.mean()),
                    "prob_change_fusion_minus_global": float(
                        group.fusion_probability.mean()
                        - group.global_probability.mean()
                    ),
                }
    # 新增正确/新增错误患者：融合 vs 全局，覆盖癌与非癌（merge后label带后缀，二者一致）。
    patient_global = patient_top2_mean(local, "global_probability")
    patient_fusion = patient_top2_mean(local, "fusion_probability")
    merged = patient_global.merge(
        patient_fusion, on="patient_id", suffixes=("_global", "_fusion")
    )
    global_pred = merged.patient_probability_global >= 0.5
    fusion_pred = merged.patient_probability_fusion >= 0.5
    global_correct = global_pred == merged.label_global
    fusion_correct = fusion_pred == merged.label_fusion
    result["new_correct_patients"] = int(
        ((~global_correct) & fusion_correct).sum()
    )
    result["new_error_patients"] = int(
        (global_correct & (~fusion_correct)).sum()
    )
    # 全局 vs 融合预测一致率（患者级，0.5阈值）及方向计数。
    result["agreement_default_0_5"] = float(
        (global_pred == fusion_pred).mean()
    ) if len(merged) else 0.0
    result["global_correct_to_fusion_wrong"] = int(
        (global_correct & (~fusion_correct)).sum()
    )
    result["global_wrong_to_fusion_correct"] = int(
        ((~global_correct) & fusion_correct).sum()
    )
    return result


def prepare_output(args):
    """创建独立输出目录；除非显式--overwrite，否则拒绝覆盖已有结果。

    参数: args（含run_name/seed/debug/output_root）。
    返回: 输出目录Path；debug追加_debug后缀。
    """
    name = args.run_name or f"m4_balanced_keep_efficientnet_b0_seed{args.seed}"
    if args.debug:
        name += "_debug"
    output = args.output_root / name
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def self_test():
    """回归自测：patient_top2_mean按_bunit聚合必须保留有放回重复权重。"""
    frame = pd.DataFrame({
        "patient_id": ["A", "A", "B", "B", "B"],
        "label": [1, 1, 0, 0, 0],
        "fusion_probability": [0.9, 0.8, 0.2, 0.1, 0.3],
        "global_probability": [0.7, 0.6, 0.4, 0.3, 0.2],
    })
    # A抽2次、B抽1次：_bunit应产生3个独立单元，重复患者不被重新合并。
    chunks = []
    for occurrence, patient in enumerate(["A", "A", "B"]):
        chunk = frame.loc[frame.patient_id.eq(patient)].copy()
        chunk["_bunit"] = f"t_{occurrence}"
        chunks.append(chunk)
    sample = pd.concat(chunks, ignore_index=True)
    result = patient_top2_mean(sample, "fusion_probability", "_bunit")
    if len(result) != 3:
        raise AssertionError(f"bootstrap单元数应为3（A两次+B一次），实际{len(result)}")
    if result.patient_probability.nunique() < 2:
        raise AssertionError("重复患者应保留独立权重，不能合并成单点")
    return True


def main():
    """编排M4训练：产物链校验、两阶段训练、选择、冻结校验、审计与保存。

    流程: 校验ROI清单与M1 checkpoint（SHA/计数/debug）→ 构建平衡采样DataLoader →
    阶段A(只训两头)→阶段B(解冻features[7:])→按(患者AUC,图像AUC,-val融合损失)选产品 →
    冻结校验全局分支 → 输出m4_best.pth、预测、元数据审计、bootstrap与辅助指标。
    无返回值；test/external恒False。
    """
    args = parse_args()
    if args.self_test:
        self_test()
        print("M4融合自测通过: _bunit聚合保留有放回重复患者权重")
        return
    if args.debug:
        args.stage_a_epochs = min(args.stage_a_epochs, 1)
        args.stage_b_epochs = min(args.stage_b_epochs, 1)
        args.num_workers = 0
    seed_everything(args.seed)
    roi_manifest = args.roi_manifest or default_roi_manifest(args.seed)
    m1_checkpoint = args.m1_checkpoint or default_m1_checkpoint(args.seed)
    if not Path(roi_manifest).is_file():
        raise FileNotFoundError(f"先运行build_m4_roi_manifest.py生成: {roi_manifest}")
    if not Path(m1_checkpoint).is_file():
        raise FileNotFoundError(m1_checkpoint)

    frame = load_roi_manifest(
        roi_manifest, args.image_root, args.debug, args.debug_units, args.seed
    )
    # 产物链校验：读取与ROI清单同名的config，核验seed/debug/M1 SHA/行数计数。
    roi_config_path = Path(
        str(roi_manifest).replace("m4_roi_manifest", "m4_roi_config")
        .replace(".csv", ".json")
    )
    if not roi_config_path.is_file():
        raise FileNotFoundError(roi_config_path)
    roi_config = json.loads(roi_config_path.read_text(encoding="utf-8"))
    if int(roi_config.get("seed", -1)) != args.seed:
        raise ValueError(
            f"ROI config seed不匹配: {roi_config.get('seed')} != {args.seed}"
        )
    if roi_config.get("debug") and not args.debug:
        raise ValueError("ROI清单是debug产物，禁止用于正式训练")
    if file_sha256(m1_checkpoint) != roi_config.get("m1_checkpoint_sha256"):
        raise ValueError("M1 checkpoint与ROI清单记录的SHA不一致")
    csv_rows = len(pd.read_csv(roi_manifest, encoding="utf-8-sig"))
    expected_rows = int(roi_config["n_train"]) + int(roi_config["n_val"])
    if csv_rows != expected_rows:
        raise ValueError(
            f"ROI CSV行数{csv_rows} != config计数"
            f"{roi_config['n_train']}+{roi_config['n_val']}={expected_rows}"
        )
    # ROI CSV自身SHA校验（防止清单被篡改或替换）。
    if file_sha256(roi_manifest) != roi_config.get("roi_csv_sha256"):
        raise ValueError("ROI CSV与config记录的SHA不一致")
    # 所有输入校验通过后才创建输出目录，避免预检失败留下空目录阻塞下次运行。
    output = prepare_output(args)

    eval_tf, roi_train_tf, roi_eval_tf = build_transforms()
    datasets = {
        split: M4Dataset(
            frame.loc[frame.split.eq(split)].copy(), args.image_root,
            split == "train", eval_tf, roi_train_tf, roi_eval_tf,
        )
        for split in ("train", "val")
    }
    # 患者/标签平衡采样（沿用M3c-B口径），避免多图患者与多数标签获得更高权重。
    generator = torch.Generator().manual_seed(args.seed)
    train_sampler = WeightedRandomSampler(
        patient_class_balanced_weights(datasets["train"].df),
        num_samples=len(datasets["train"]), replacement=True, generator=generator,
    )
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=args.batch_size, sampler=train_sampler,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
        ),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m1_payload = torch.load(m1_checkpoint, map_location="cpu", weights_only=False)
    m1 = EfficientNetM1()
    m1.load_state_dict(m1_payload["model_state_dict"], strict=True)
    model = M4Model(m1.backbone).to(device)
    freeze_before = frozen_m1_snapshot(model)
    print(f"设备: {device}; seed={args.seed}; "
          f"train={len(datasets['train'])}张; val={len(datasets['val'])}张")

    # 基线：冻结全局分类（只用eval变换），M4融合提升以此为参照。
    baseline = evaluate(model, loaders["val"], device, args)
    print(f"冻结全局患者AUC={baseline['metrics']['patient_global_auc']:.4f} "
          f"图像AUC={baseline['metrics']['image_global_auc']:.4f}")

    best = {"key": None, "state": None, "record": None}
    history = []
    no_improve = 0
    stage_b_best_auc = -np.inf
    for stage, epochs, lr, backbone_lr in [
        ("A", args.stage_a_epochs, args.stage_a_lr, 0.0),
        ("B", args.stage_b_epochs, args.stage_b_lr, args.stage_b_backbone_lr),
    ]:
        head_params = list(model.local_head.parameters()) + list(
            model.fusion_head.parameters()
        )
        backbone_params = (
            [parameter for index in range(7, len(model.local_features))
             for parameter in model.local_features[index].parameters()]
            if stage == "B" else []
        )
        groups = [{"params": head_params, "lr": lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        optimizer = optim.AdamW(groups, weight_decay=args.weight_decay)
        for epoch in range(1, epochs + 1):
            started = time.time()
            set_m4_train_mode(model, stage)
            totals = {"total": 0.0, "fusion": 0.0, "local": 0.0}
            for batch in loaders["train"]:
                images = batch["image"].to(device, non_blocking=True)
                rois = batch["roi"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                roi_labels = batch["roi_label"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images, rois)
                losses = compute_losses(outputs, labels, roi_labels, args)
                losses["total"].backward()
                optimizer.step()
                for key in totals:
                    totals[key] += float(losses[key].detach()) * len(images)
            val = evaluate(model, loaders["val"], device, args)
            record = {
                "epoch": epoch, "stage": stage,
                **{f"train_{key}": value / len(datasets["train"])
                   for key, value in totals.items()},
                **{f"val_{key}": value for key, value in val["metrics"].items()},
                "elapsed_seconds": time.time() - started,
            }
            history.append(record)
            patient_auc = val["metrics"]["patient_fusion_auc"]
            # 平手键：患者AUC -> 图像AUC -> val融合损失更低。
            key = (patient_auc, val["metrics"]["image_fusion_auc"],
                   -val["metrics"]["val_fusion_loss"])
            if best["key"] is None or key > best["key"]:
                best = {"key": key, "state": deepcopy(model.state_dict()),
                        "record": record}
            # 阶段B独立patience：以阶段B自身最佳患者AUC计，不被阶段A结果压住。
            if stage == "B":
                if patient_auc > stage_b_best_auc:
                    stage_b_best_auc = patient_auc
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= args.early_stop_patience:
                    print(f"early stop: 第{epoch}轮，阶段B连续{no_improve}轮未超自身最佳")
                    break
            print(
                f"stage{stage} {epoch:02d}/{epochs} | "
                f"train_total={record['train_total']:.4f} | "
                f"val患者AUC={patient_auc:.4f} | 图像AUC={val['metrics']['image_fusion_auc']:.4f} "
                f"| 全局患者AUC={val['metrics']['patient_global_auc']:.4f}"
            )

    # 冻结校验：全局分支参数/BN不变。
    freeze_after = frozen_m1_snapshot(model)
    freeze_ok = (
        freeze_before["global_params_sha"] == freeze_after["global_params_sha"]
        and freeze_before["bn_running_sha"] == freeze_after["bn_running_sha"]
    )
    if not freeze_ok:
        raise RuntimeError("冻结校验失败：M1全局分支被意外修改")

    model.load_state_dict(best["state"])
    final = evaluate(model, loaders["val"], device, args)
    final_pred = final["predictions"]
    patient_final = patient_top2_mean(final_pred, "fusion_probability")
    patient_global = patient_top2_mean(final_pred, "global_probability")
    delta_auc = (
        float(roc_auc_score(patient_final.label, patient_final.patient_probability))
        - float(roc_auc_score(patient_global.label, patient_global.patient_probability))
    ) if patient_final.label.nunique() >= 2 else 0.0
    bootstrap = patient_paired_bootstrap(final_pred, args.bootstrap, args.seed)
    audit = metadata_audit(frame)
    auxiliary = compute_auxiliary_metrics(final_pred, args, roi_config)

    config = {
        **{key: str(value) if isinstance(value, Path) else value
           for key, value in vars(args).items()},
        "roi_manifest": str(Path(roi_manifest).resolve()),
        "roi_manifest_sha256": file_sha256(roi_manifest),
        "roi_config": roi_config,
        "m1_checkpoint": str(Path(m1_checkpoint).resolve()),
        "m1_checkpoint_sha256": file_sha256(m1_checkpoint),
        "freeze_ok": freeze_ok,
        "m4_selection": {
            "record": best["record"],
            "patient_fusion_auc": float(roc_auc_score(
                patient_final.label, patient_final.patient_probability)),
            "patient_global_auc": float(roc_auc_score(
                patient_global.label, patient_global.patient_probability)),
            "delta_patient_auc": delta_auc,
            "metadata_audit": audit,
            "patient_paired_bootstrap": bootstrap,
            "auxiliary_metrics": auxiliary,
        },
        "test_evaluated": False,
        "external_evaluated": False,
        **git_snapshot(),
    }
    torch.save({
        "model_state_dict": best["state"],
        "config": json_ready(config),
    }, output / "m4_best.pth")
    pd.DataFrame(history).to_csv(
        output / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    frame.to_csv(output / "frozen_split_snapshot.csv", index=False, encoding="utf-8-sig")
    final_pred.to_csv(output / "val_image_predictions.csv", index=False, encoding="utf-8-sig")
    patient_final.to_csv(output / "val_patient_predictions.csv", index=False, encoding="utf-8-sig")
    (output / "config.json").write_text(
        json.dumps(json_ready(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), output / "source_entry.py")
    print(
        f"冻结全局患者AUC={config['m4_selection']['patient_global_auc']:.4f} -> "
        f"M4融合患者AUC={config['m4_selection']['patient_fusion_auc']:.4f} | "
        f"Δ={delta_auc:.4f}"
    )
    print(f"元数据审计: {audit}")
    print(f"输出目录: {output}")


if __name__ == "__main__":
    main()
