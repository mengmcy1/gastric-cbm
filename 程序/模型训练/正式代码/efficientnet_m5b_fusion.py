#!/usr/bin/env python3
"""EfficientNet-B0 M5b：残差锚定融合——全局稳定锚点 + 有界局部修正。

按M5b预注册（2026-08-12）：
- M5诊断：局部头有真实互补信息（患者AUC 0.83-0.86），但冷启动concat-MLP融合头破坏
  排序（seed202/503融合<全局，而简单均值>全局）。M5b改为残差锚定。
- m_final = m_global + α×Δm；残差头最后一层init 0 → 训练起点=全局基线；
  α∈[0,1]决定局部修正幅度，`--alpha-mode gate`=α=gate_pass（M5b-A），
  `gate_local`=α=gate_pass×p_local（M5b-B，推荐主方案）。
- 残差输入固定[m_global, m_local, similarity]，第一版排除roi_area/q_region/框几何。
- 阶段L训局部头（CE roi_label，ignore=2），按val局部头AUC选后冻结；
  阶段R只训残差MLP，L=BCE_with_logits(m_final,label)+0.1×mean((α×Δm)²)；
  产品保护：最终产品=argmax(全局基线, 候选融合)，Δ≥0，回退全局记fell_back。
- 复用M5冻结ROI清单（batch32，已验收）；不读internal test/external。

正式参数见`实验进度与结果讨论.md`的M5b预注册。
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
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
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

DEFAULT_OUTPUT = PROJECT_ROOT / "结果/M5b残差融合_0804/正式验证集筛选"
ROI_ROOT = PROJECT_ROOT / "结果/M5预测ROI融合_0804/冻结ROI清单"
M1_ROOT = PROJECT_ROOT / "结果/M1辅助定位_0804/正式验证集筛选"
M4_PRODUCT_ROOT = PROJECT_ROOT / "结果/M4真值ROI融合_0804/正式验证集筛选"
SEEDS = (42, 202, 503)

IGNORE = 2              # roi_label=2 表示不确定，阶段L局部CE被mask
LAMBDA_REG = 0.1        # 冻结：L_M5b = L_final + λ_reg * mean((α*Δm)^2)
M4_MEAN_DELTA = 0.0334  # M4正式成功平均Δ，用于存活率诊断
STAGE_L_EPOCHS = 10
STAGE_R_EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-manifest", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--m1-checkpoint", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha-mode", default="gate_local",
                        choices=["gate", "gate_local"],
                        help="M5b-A=gate(α=gate_pass)；M5b-B=gate_local(α=gate_pass*p_local)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stage-l-epochs", type=int, default=STAGE_L_EPOCHS)
    parser.add_argument("--stage-r-epochs", type=int, default=STAGE_R_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--lambda-reg", type=float, default=LAMBDA_REG)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-units", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="运行残差init-0与BCE一致性自测后退出。")
    return parser.parse_args()


def default_roi_manifest(seed):
    return ROI_ROOT / f"m5_roi_manifest_seed{seed}.csv"


def default_m1_checkpoint(seed):
    return (
        M1_ROOT / f"m1_balanced_keep_efficientnet_b0_seed{seed}_warmup_product"
        / "m1_best_warmup_localization.pth"
    )


def load_roi_manifest(path, image_root, debug, debug_units, seed):
    """校验并加载M5冻结ROI清单（与M5相同），仅保留train/val。

    参数:
        path: m5_roi_manifest CSV路径。
        image_root: 项目根，用于校验图像文件存在。
        debug/debug_units/seed: 调试抽样参数。
    返回:
        pd.DataFrame: train/val行，含roi_crop_bbox、roi_label(0/1/2)、gate_pass。
    """
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"patient_id": str})
    required = {
        "image_relpath", "patient_id", "label", "split", "roi_source",
        "roi_crop_bbox_x1", "roi_crop_bbox_y1", "roi_crop_bbox_x2",
        "roi_crop_bbox_y2", "width", "height", "q_region", "noncancer_stratum",
        "roi_label", "gate_pass",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ROI清单缺少字段: {sorted(missing)}")
    if set(frame.split.unique()) != {"train", "val"}:
        raise ValueError("ROI清单应只含train/val")
    if not set(frame.roi_label.unique()).issubset({0, 1, 2}):
        raise ValueError(f"roi_label取值非法: {sorted(frame.roi_label.unique())}")
    if not set(frame.gate_pass.unique()).issubset({0, 1}):
        raise ValueError(f"gate_pass取值非法: {sorted(frame.gate_pass.unique())}")
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
    """构建全局评估、局部训练增强和局部确定性评估三套变换。"""
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


class M5bDataset(Dataset):
    """读取M5冻结ROI清单：完整图走全局评估变换，ROI使用调用方指定变换。

    返回: image/roi[3,224,224]、label、roi_label(0/1/2)、gate_pass、row_index。
    """
    def __init__(self, frame, image_root, eval_transform, roi_transform):
        self.df = frame.reset_index(drop=True)
        self.image_root = image_root
        self.eval_transform = eval_transform
        self.roi_transform = roi_transform

    def _load_roi(self, row):
        with Image.open(self.image_root / row.image_relpath) as source:
            image = source.convert("RGB")
        width, height = image.size
        left = int(row.roi_crop_bbox_x1 * width)
        top = int(row.roi_crop_bbox_y1 * height)
        right = int(row.roi_crop_bbox_x2 * width)
        bottom = int(row.roi_crop_bbox_y2 * height)
        return image.crop((left, top, right, bottom)).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(self.image_root / row.image_relpath) as source:
            full = source.convert("RGB")
        return {
            "image": self.eval_transform(full),
            "roi": self.roi_transform(self._load_roi(row)),
            "label": torch.tensor(int(row.label), dtype=torch.long),
            "roi_label": torch.tensor(int(row.roi_label), dtype=torch.long),
            "gate_pass": torch.tensor(int(row.gate_pass), dtype=torch.long),
            "row_index": torch.tensor(index, dtype=torch.long),
        }


class M5bModel(nn.Module):
    """残差锚定模型：冻结全局分支 + 冻结局部Encoder副本 + 阶段L局部头 + 阶段R残差头。

    参数:
        m1_backbone: 冻结M1的backbone（efficientnet_b0实例）。
        alpha_mode: "gate"(A) 或 "gate_local"(B)。
    forward:
        输入: image[B,3,224,224]、roi[B,3,224,224]、gate_pass[B]。
        返回: m_global/m_local(logit差)、p_local、similarity、alpha、dm、m_final、p_final。
    残差头最后一层init 0 → 训练起点 m_final=m_global。
    """
    def __init__(self, m1_backbone, alpha_mode):
        super().__init__()
        self.alpha_mode = alpha_mode
        self.global_features = m1_backbone.features
        self.global_avgpool = m1_backbone.avgpool
        self.global_classifier = m1_backbone.classifier
        for parameter in self.global_features.parameters():
            parameter.requires_grad = False
        for parameter in self.global_classifier.parameters():
            parameter.requires_grad = False
        self.local_features = deepcopy(m1_backbone.features)
        self.local_avgpool = deepcopy(m1_backbone.avgpool)
        for parameter in self.local_features.parameters():
            parameter.requires_grad = False
        for parameter in self.local_avgpool.parameters():
            parameter.requires_grad = False
        self.local_head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(1280, 2),
        )
        self.residual_mlp = nn.Sequential(
            nn.Linear(3, 16), nn.SiLU(inplace=True), nn.Linear(16, 1),
        )
        # 最后一层init 0 → Δm=0，m_final=m_global。
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

    def set_stage_l(self):
        """阶段L：只训局部头；全局分支与局部Encoder恒eval。"""
        self.eval()
        self.global_features.eval()
        self.local_features.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.local_head.parameters():
            parameter.requires_grad = True
        self.local_head.train()

    def set_stage_r(self):
        """阶段R：只训残差头；其余全部eval且无梯度。"""
        self.eval()
        self.global_features.eval()
        self.local_features.eval()
        self.local_head.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.residual_mlp.parameters():
            parameter.requires_grad = True
        self.residual_mlp.train()

    def local_logits(self, roi):
        f_local = torch.flatten(self.local_avgpool(self.local_features(roi)), 1)
        return f_local, self.local_head(f_local)

    def forward(self, image, roi, gate_pass):
        with torch.no_grad():
            f_global = torch.flatten(
                self.global_avgpool(self.global_features(image)), 1
            )
            g_logits = self.global_classifier(f_global)
        m_global = g_logits[:, 1] - g_logits[:, 0]
        f_local, l_logits = self.local_logits(roi)
        m_local = l_logits[:, 1] - l_logits[:, 0]
        p_local = torch.sigmoid(m_local)
        similarity = F.cosine_similarity(f_global, f_local, dim=1)
        if self.alpha_mode == "gate":
            alpha = gate_pass.float()
        else:  # gate_local
            alpha = gate_pass.float() * p_local.detach()
        features = torch.stack([m_global, m_local.detach(), similarity.detach()], dim=1)
        dm = torch.tanh(self.residual_mlp(features).squeeze(1))
        m_final = m_global + alpha * dm
        return {
            "m_global": m_global, "m_local": m_local, "p_local": p_local,
            "similarity": similarity, "alpha": alpha, "dm": dm,
            "m_final": m_final, "p_final": torch.sigmoid(m_final),
        }


def frozen_m1_snapshot(model):
    """对冻结的全局分支+局部Encoder生成参数与BN running统计的SHA-256指纹。

    说明: M1定位头与M3c-B门控不在M5b训练图中，由ROI产物链SHA保证来源。
    """
    state = {}
    hasher = hashlib.sha256()
    for module in (model.global_features, model.global_classifier,
                   model.local_features, model.local_avgpool):
        for parameter in module.parameters():
            hasher.update(parameter.detach().cpu().numpy().tobytes())
    state["params_sha"] = hasher.hexdigest()
    bn_hasher = hashlib.sha256()
    for module in model.global_features.modules():
        if isinstance(module, nn.BatchNorm2d) and module.running_mean is not None:
            bn_hasher.update(module.running_mean.detach().cpu().numpy().tobytes())
            bn_hasher.update(module.running_var.detach().cpu().numpy().tobytes())
    for module in model.local_features.modules():
        if isinstance(module, nn.BatchNorm2d) and module.running_mean is not None:
            bn_hasher.update(module.running_mean.detach().cpu().numpy().tobytes())
            bn_hasher.update(module.running_var.detach().cpu().numpy().tobytes())
    state["bn_running_sha"] = bn_hasher.hexdigest()
    return state


def patient_top2_mean(predictions, probability_column, group_column="patient_id"):
    """冻结的top2_mean患者聚合（与M0一致）。"""
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


def patient_auc_of(frame, column):
    """按top2_mean聚合并计算患者级AUC；标签退化返回0.5。"""
    patients = patient_top2_mean(frame, column)
    if patients.label.nunique() < 2:
        return 0.5
    return float(roc_auc_score(patients.label, patients.patient_probability))


def image_auc_of(series, labels):
    if labels.nunique() < 2:
        return 0.5
    return float(roc_auc_score(labels, series))


def evaluate_stage_l(model, loader, device):
    """阶段L评估：val局部头图像级AUC（p_local vs roi_label，ignore不参与）。"""
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            rois = batch["roi"].to(device, non_blocking=True)
            _, l_logits = model.local_logits(rois)
            p_local = torch.softmax(l_logits, dim=1)[:, 1]
            for i, row_index in enumerate(batch["row_index"].tolist()):
                rows.append({"row_index": row_index,
                             "roi_label": int(batch["roi_label"][i]),
                             "p_local": float(p_local[i])})
    preds = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
    known = preds.loc[preds.roi_label.ne(IGNORE)]
    if known.roi_label.nunique() < 2:
        return 0.5
    return float(roc_auc_score(known.roi_label, known.p_local))


def precompute_features(model, datasets, device, batch_size, num_workers):
    """阶段R特征预计算：冻结特征一次性前向，输出train/val两表。

    每行含 patient_id/label/gate_pass/m_global/m_local/p_local/similarity/alpha。
    """
    model.set_stage_r()
    model.eval()
    all_rows = []
    with torch.no_grad():
        for split in ("train", "val"):
            loader = DataLoader(
                datasets[split], batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                worker_init_fn=seed_worker,
            )
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                rois = batch["roi"].to(device, non_blocking=True)
                gate_pass = batch["gate_pass"].to(device)
                out = model(images, rois, gate_pass)
                for i, row_index in enumerate(batch["row_index"].tolist()):
                    meta = datasets[split].df.iloc[row_index]
                    all_rows.append({
                        "split": split,
                        "row_index": row_index,
                        "patient_id": meta["patient_id"],
                        "label": int(meta["label"]),
                        "roi_label": int(meta["roi_label"]),
                        "gate_pass": int(meta["gate_pass"]),
                        "noncancer_stratum": meta["noncancer_stratum"],
                        "bbox_area_fraction": float(meta["bbox_area_fraction"]),
                        "q_region": float(meta["q_region"]),
                        "m_global": float(out["m_global"][i]),
                        "m_local": float(out["m_local"][i]),
                        "p_local": float(out["p_local"][i]),
                        "similarity": float(out["similarity"][i]),
                        "alpha": float(out["alpha"][i]),
                    })
    table = pd.DataFrame(all_rows)
    return {
        "train": table.loc[table.split.eq("train")].reset_index(drop=True),
        "val": table.loc[table.split.eq("val")].reset_index(drop=True),
    }


def train_residual(train_table, val_table, args, device):
    """阶段R：只训残差MLP，返回最佳epoch记录、产品选择与全过程history。

    输入特征与α均来自冻结预计算；Δm=MLP([m_global,m_local,similarity])；
    m_final = m_global + α*Δm；L = BCE(m_final,label) + λ_reg*mean((α*Δm)^2)。
    产品保护：最终产品 = argmax(全局基线患者AUC, 最佳真实epoch患者AUC)。
    """
    feature_cols = ["m_global", "m_local", "similarity"]
    X_train = torch.tensor(train_table[feature_cols].to_numpy(), dtype=torch.float32)
    alpha_train = torch.tensor(train_table["alpha"].to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(train_table["label"].to_numpy(), dtype=torch.float32)
    X_val = torch.tensor(val_table[feature_cols].to_numpy(), dtype=torch.float32, device=device)
    alpha_val = torch.tensor(val_table["alpha"].to_numpy(), dtype=torch.float32, device=device)
    m_global_val = torch.tensor(val_table["m_global"].to_numpy(), dtype=torch.float32, device=device)
    y_val = torch.tensor(val_table["label"].to_numpy(), dtype=torch.float32, device=device)

    # 阶段R继续使用患者/标签平衡采样，避免多图患者和多数标签主导残差方向。
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        patient_class_balanced_weights(train_table),
        num_samples=len(train_table), replacement=True, generator=generator,
    )
    train_loader = DataLoader(
        TensorDataset(X_train, alpha_train, y_train),
        batch_size=args.batch_size, sampler=sampler, num_workers=0,
    )

    # 全局基线：m_final=m_global → val患者AUC。
    val_global_p = torch.sigmoid(m_global_val).cpu().numpy()
    val_frame = val_table.copy()
    val_frame["p_global"] = val_global_p
    global_patient_auc = patient_auc_of(val_frame, "p_global")

    model = nn.Sequential(
        nn.Linear(3, 16), nn.SiLU(inplace=True), nn.Linear(16, 1),
    ).to(device)
    nn.init.zeros_(model[-1].weight)
    nn.init.zeros_(model[-1].bias)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lambda_reg = args.lambda_reg

    history = []
    best = {"key": None, "state": None, "epoch": None, "patient_auc": None}
    for epoch in range(1, args.stage_r_epochs + 1):
        model.train()
        totals = {"final": 0.0, "reg": 0.0, "total": 0.0,
                  "abs_dm": 0.0, "abs_correction": 0.0, "alpha": 0.0}
        seen = 0
        for features, alpha, labels in train_loader:
            features = features.to(device)
            alpha = alpha.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            dm = torch.tanh(model(features).squeeze(1))
            correction = alpha * dm
            m_final = features[:, 0] + correction
            loss_final = F.binary_cross_entropy_with_logits(m_final, labels)
            loss_reg = (correction ** 2).mean()
            loss = loss_final + lambda_reg * loss_reg
            loss.backward()
            optimizer.step()
            batch_n = len(labels)
            seen += batch_n
            totals["final"] += float(loss_final.detach()) * batch_n
            totals["reg"] += float(loss_reg.detach()) * batch_n
            totals["total"] += float(loss.detach()) * batch_n
            totals["abs_dm"] += float(dm.abs().mean().detach()) * batch_n
            totals["abs_correction"] += float(correction.abs().mean().detach()) * batch_n
            totals["alpha"] += float(alpha.mean().detach()) * batch_n

        model.eval()
        with torch.no_grad():
            dm_val = torch.tanh(model(X_val).squeeze(1))
            m_final_val = m_global_val + alpha_val * dm_val
            p_val = torch.sigmoid(m_final_val).cpu().numpy()
            val_loss_final = F.binary_cross_entropy_with_logits(
                m_final_val, y_val
            )
            vf = val_table.copy()
            vf["p_final"] = p_val
            patient_auc = patient_auc_of(vf, "p_final")
            image_auc = image_auc_of(vf["p_final"], vf.label)
        history.append({
            "epoch": epoch,
            "train_loss_final": totals["final"] / seen,
            "train_loss_reg_raw": totals["reg"] / seen,
            "train_loss_reg_weighted": lambda_reg * totals["reg"] / seen,
            "train_loss_total": totals["total"] / seen,
            "train_mean_abs_dm": totals["abs_dm"] / seen,
            "train_mean_abs_correction": totals["abs_correction"] / seen,
            "train_mean_alpha": totals["alpha"] / seen,
            "val_loss_final": float(val_loss_final.detach()),
            "val_patient_auc": patient_auc,
            "val_image_auc": image_auc,
        })
        print(
            f"stageR {epoch:02d}/{args.stage_r_epochs} | "
            f"final={totals['final']/seen:.4f} | "
            f"reg={totals['reg']/seen:.4f} | "
            f"{lambda_reg:g}reg={lambda_reg*totals['reg']/seen:.4f} | "
            f"|dm|={totals['abs_dm']/seen:.4f} | "
            f"|correction|={totals['abs_correction']/seen:.4f} | "
            f"alpha={totals['alpha']/seen:.4f} | "
            f"val患者AUC={patient_auc:.4f} | val图像AUC={image_auc:.4f}"
        )
        key = (patient_auc, image_auc, -float(val_loss_final.detach()))
        if best["key"] is None or key > best["key"]:
            best = {"key": key, "state": deepcopy(model.state_dict()),
                    "epoch": epoch, "patient_auc": patient_auc}

    # 产品保护：全局基线 vs 最佳真实融合epoch。
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        dm_val = torch.tanh(model(X_val).squeeze(1))
        m_final_val = m_global_val + alpha_val * dm_val
        best_p = torch.sigmoid(m_final_val).cpu().numpy()
    vf = val_table.copy()
    vf["p_final"] = best_p
    best_fusion_patient_auc = patient_auc_of(vf, "p_final")
    fell_back = best_fusion_patient_auc <= global_patient_auc
    product_patient_auc = (global_patient_auc if fell_back
                           else best_fusion_patient_auc)
    delta = product_patient_auc - global_patient_auc
    raw_delta = best_fusion_patient_auc - global_patient_auc
    return {
        "model": model,
        "best_epoch": best["epoch"],
        "global_patient_auc": global_patient_auc,
        "best_fusion_patient_auc": best_fusion_patient_auc,
        "fell_back_to_global": fell_back,
        "product_patient_auc": product_patient_auc,
        "delta_patient_auc": delta,
        "raw_delta_patient_auc": raw_delta,
        "candidate_probabilities": best_p,
        "history": history,
    }


def metadata_audit(frame):
    """四档元数据审计（与M5相同）：A预测框几何/A_crop裁图几何/B q_region/C几何+q_region。"""
    train = frame.loc[frame.split.eq("train")]
    val = frame.loc[frame.split.eq("val")]

    def geometry_matrix(sub, prefix):
        geo = sub[[f"{prefix}_x1", f"{prefix}_y1", f"{prefix}_x2", f"{prefix}_y2"]].copy()
        geo.columns = ["x1", "y1", "x2", "y2"]
        geo["width"] = geo.x2 - geo.x1
        geo["height"] = geo.y2 - geo.y1
        geo["area"] = geo.width * geo.height
        geo["aspect"] = geo.width / geo.height.clip(lower=1e-8)
        geo["center_x"] = (geo.x1 + geo.x2) / 2
        geo["center_y"] = (geo.y1 + geo.y2) / 2
        return geo.to_numpy(), sub.label.to_numpy()

    base_tr, y_tr = geometry_matrix(train, "roi_base_bbox")
    base_va, y_va = geometry_matrix(val, "roi_base_bbox")
    crop_tr, _ = geometry_matrix(train, "roi_crop_bbox")
    crop_va, _ = geometry_matrix(val, "roi_crop_bbox")
    q_tr = train.q_region.to_numpy().reshape(-1, 1)
    q_va = val.q_region.to_numpy().reshape(-1, 1)

    def eval_auc(X_tr, y_tr, X_va, y_va):
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            return 0.5
        model = LogisticRegression(max_iter=500)
        model.fit(X_tr, y_tr)
        return float(roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]))

    return {
        "A_pure_geometry_auc": eval_auc(base_tr, y_tr, base_va, y_va),
        "A_crop_geometry_auc": eval_auc(crop_tr, y_tr, crop_va, y_va),
        "B_qregion_only_auc": eval_auc(q_tr, y_tr, q_va, y_va),
        "C_geometry_plus_qregion_auc": eval_auc(
            np.hstack([base_tr, q_tr]), y_tr, np.hstack([base_va, q_va]), y_va
        ),
    }


def compute_attribution(val_table, product_p, global_p):
    """α通道归因审计：各信号单独图像/患者级AUC + 残差工作量 + 排序变化。

    参数:
        val_table: 阶段R val预计算表（含p_local/similarity/alpha/label/patient_id）。
        product_p: 最终产品在val上的p_final（按行序）。
        global_p: 全局p_final（sigmoid(m_global)）。
    返回: dict。
    """
    vf = val_table.copy()
    vf["p_final"] = product_p
    vf["p_global"] = global_p
    result = {"signals": {}}
    for name, column in [("gate_pass", "gate_pass"), ("p_local", "p_local"),
                         ("alpha", "alpha"), ("similarity", "similarity"),
                         ("p_candidate", "p_candidate"),
                         ("p_final", "p_final")]:
        result["signals"][name] = {
            "image_auc": image_auc_of(vf[column], vf.label),
            "patient_auc": patient_auc_of(vf, column),
        }
    # 残差工作量分布（dm/correction由调用方按最佳残差模型填入vf）。
    result["alpha_dist"] = _quantiles(vf.alpha)
    result["dm_dist"] = _quantiles(vf.dm)
    result["correction_dist"] = _quantiles(vf.correction)
    result["mean_abs_alpha"] = float(vf.alpha.abs().mean())
    result["mean_abs_dm"] = float(vf.dm.abs().mean())
    result["mean_abs_correction"] = float((vf.alpha * vf.dm).abs().mean())
    # 排序变化与对错翻转（患者级，0.5阈值）。
    p_glob = patient_top2_mean(vf, "p_global")
    p_fin = patient_top2_mean(vf, "p_final")
    merged = p_glob.merge(p_fin, on="patient_id", suffixes=("_g", "_f"))
    g_pred = merged.patient_probability_g >= 0.5
    f_pred = merged.patient_probability_f >= 0.5
    g_correct = g_pred == merged.label_g
    f_correct = f_pred == merged.label_f
    result["patient_probability_changes"] = int(
        (~np.isclose(
            merged.patient_probability_g, merged.patient_probability_f,
            rtol=0.0, atol=1e-12,
        )).sum()
    )
    rank_global = merged.patient_probability_g.rank(method="average")
    rank_final = merged.patient_probability_f.rank(method="average")
    result["patient_rank_position_changes"] = int(
        (~np.isclose(rank_global, rank_final, rtol=0.0, atol=1e-12)).sum()
    )
    result["global_correct_to_fusion_wrong"] = int(
        (g_correct & (~f_correct)).sum()
    )
    result["global_wrong_to_fusion_correct"] = int(
        ((~g_correct) & f_correct).sum()
    )
    return result


def _quantiles(series):
    vals = series.to_numpy()
    q = [float(v) for v in np.percentile(vals, [0, 25, 50, 75, 100])]
    return {"min": q[0], "p25": q[1], "median": q[2], "p75": q[3], "max": q[4],
            "mean": float(vals.mean()), "std": float(vals.std())}


def patient_paired_bootstrap(
    predictions, target_column, iterations, seed, baseline_column="p_global",
):
    """先固定患者级top2_mean，再对患者行做有放回配对bootstrap。

    每位患者的图像集合与top2_mean在重采样中不会改变；预聚合与旧版逐轮复制图像行
    数学等价，同时保留患者首次出现顺序，确保同seed抽样序列不变。
    """
    patient_order = predictions.patient_id.drop_duplicates().tolist()
    baseline = patient_top2_mean(
        predictions, baseline_column
    ).set_index("patient_id").reindex(patient_order)
    target = patient_top2_mean(
        predictions, target_column
    ).set_index("patient_id").reindex(patient_order)
    if not np.array_equal(
        baseline.label.to_numpy(dtype=int), target.label.to_numpy(dtype=int),
    ):
        raise ValueError("bootstrap的基线与目标患者标签不一致")

    labels = baseline.label.to_numpy(dtype=int)
    baseline_scores = baseline.patient_probability.to_numpy(dtype=float)
    target_scores = target.patient_probability.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    n_patients = len(patient_order)
    deltas = []
    for _ in range(iterations):
        indices = rng.choice(n_patients, size=n_patients, replace=True)
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        deltas.append(float(
            roc_auc_score(sampled_labels, target_scores[indices])
            - roc_auc_score(sampled_labels, baseline_scores[indices])
        ))
    deltas = np.asarray(deltas, dtype=float)
    return {
        "delta_ci95": [float(np.percentile(deltas, 2.5)),
                       float(np.percentile(deltas, 97.5))],
        "delta_mean": float(deltas.mean()),
        "iterations_used": int(len(deltas)),
        "preaggregated_by_patient": True,
    }


def compute_auxiliary_metrics(vf, roi_config):
    """辅助指标：局部头/hard-easy/病灶大小/一致率/新增对错患者（患者级对全局）。

    参数:
        vf: val表（含p_local/p_global/p_final/roi_label/gate_pass/bbox_area_fraction）。
        roi_config: 冻结ROI config，提供train冻结的病灶大小三分位。
    """
    result = {}
    local = vf.copy()
    known = local.loc[local.roi_label.ne(IGNORE)]
    result["local_head_auc"] = (image_auc_of(known.p_local, known.roi_label)
                                if known.roi_label.nunique() >= 2 else 0.5)
    cancer_roi = local.loc[local.label.eq(1) & local.roi_label.eq(1)]
    for name, stratum in [("cancer_vs_hard", "hard"), ("cancer_vs_easy", "easy")]:
        non = local.loc[local.noncancer_stratum.eq(stratum)]
        sub = pd.concat([cancer_roi[["roi_label", "p_local"]],
                         non[["roi_label", "p_local"]]])
        result[f"local_auc_{name}"] = (image_auc_of(sub.p_local, sub.roi_label)
                                       if sub.roi_label.nunique() >= 2 else 0.5)
    for name in ["hard", "easy"]:
        sub = local.loc[local.noncancer_stratum.eq(name)]
        result[f"local_acc_{name}"] = float(
            ((sub.p_local >= 0.5) == sub.roi_label).mean()
        ) if len(sub) else 0.0
    # 病灶大小（只用val癌图，train冻结三分位来自roi_config）。
    result["lesion_size"] = {}
    bounds = roi_config.get("lesion_tercile_bounds")
    cancer_val = local.loc[local.label.eq(1)]
    if bounds and len(cancer_val):
        for name, lo, hi in [("small", -np.inf, bounds[0]),
                             ("medium", bounds[0], bounds[1]),
                             ("large", bounds[1], np.inf)]:
            group = cancer_val.loc[(cancer_val.bbox_area_fraction > lo)
                                   & (cancer_val.bbox_area_fraction <= hi)]
            if len(group):
                result["lesion_size"][name] = {
                    "n_images": int(len(group)),
                    "n_patients": int(group.patient_id.nunique()),
                    "mean_p_global": float(group.p_global.mean()),
                    "mean_p_final": float(group.p_final.mean()),
                    "prob_change_fusion_minus_global": float(
                        group.p_final.mean() - group.p_global.mean()
                    ),
                }
    # 患者级对全局的一致率与新增对错。
    p_glob = patient_top2_mean(local, "p_global")
    p_fin = patient_top2_mean(local, "p_final")
    merged = p_glob.merge(p_fin, on="patient_id", suffixes=("_g", "_f"))
    g_pred = merged.patient_probability_g >= 0.5
    f_pred = merged.patient_probability_f >= 0.5
    g_correct = g_pred == merged.label_g
    f_correct = f_pred == merged.label_f
    result["new_correct_patients"] = int(((~g_correct) & f_correct).sum())
    result["new_error_patients"] = int((g_correct & (~f_correct)).sum())
    result["agreement_default_0_5"] = float((g_pred == f_pred).mean())
    # 临床阈值（产品p_final）。
    patients = patient_top2_mean(local, "p_final")
    if len(patients) and patients.label.nunique() >= 2:
        threshold = lock_recall_threshold(
            patients.loc[patients.label.eq(1), "patient_probability"].to_numpy(),
            recall=0.90,
        )
        pred = (patients.patient_probability >= threshold).astype(int)
        tp = int(((pred == 1) & (patients.label == 1)).sum())
        fp = int(((pred == 1) & (patients.label == 0)).sum())
        fn = int(((pred == 0) & (patients.label == 1)).sum())
        tn = int(((pred == 0) & (patients.label == 0)).sum())
        result["patient_threshold_sens90"] = threshold
        result["patient_sens_sens90"] = float(tp / max(tp + fn, 1))
        result["patient_spec_sens90"] = float(tn / max(tn + fp, 1))
    return result


def survival_diagnostics(patient_final, seed, m5b_delta, iterations):
    """诊断：相对M4的存活率（ΔM5b/ΔM4）+ M4真值框配对损失（AUC_M5b−AUC_M4）。"""
    m4_dir = M4_PRODUCT_ROOT / f"m4_balanced_keep_efficientnet_b0_seed{seed}"
    m4_patient_csv = m4_dir / "val_patient_predictions.csv"
    m4_cfg_path = m4_dir / "config.json"
    if not m4_patient_csv.is_file() or not m4_cfg_path.is_file():
        return {"available": False}
    m4_patients = pd.read_csv(m4_patient_csv, encoding="utf-8-sig",
                              dtype={"patient_id": str})
    m4_cfg = json.loads(m4_cfg_path.read_text(encoding="utf-8"))
    m4_delta = float(m4_cfg["m4_selection"]["delta_patient_auc"])
    merged = patient_final.merge(
        m4_patients[["patient_id", "label", "patient_probability"]],
        on="patient_id", suffixes=("_m5b", "_m4"),
    )
    if "label_m5b" in merged.columns:
        if not np.array_equal(
            merged.label_m5b.to_numpy(dtype=int),
            merged.label_m4.to_numpy(dtype=int),
        ):
            raise ValueError("M4与M5b同名患者的标签不一致")
        merged["label"] = merged.label_m5b.astype(int)
    if len(merged) != len(patient_final) or len(merged) != len(m4_patients):
        return {
            "available": False,
            "reason": "M4与M5b的val患者集合不完全一致",
            "aligned_patients": int(len(merged)),
            "m5b_patients": int(len(patient_final)),
            "m4_patients": int(len(m4_patients)),
        }
    if len(merged) == 0 or merged.label.nunique() < 2:
        return {"available": True, "aligned_patients": int(len(merged))}
    labels = merged.label.to_numpy(dtype=int)
    m5b_scores = merged.patient_probability_m5b.to_numpy(dtype=float)
    m4_scores = merged.patient_probability_m4.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    n_patients = len(merged)
    deltas = []
    for _ in range(iterations):
        indices = rng.choice(n_patients, size=n_patients, replace=True)
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        deltas.append(float(
            roc_auc_score(sampled_labels, m5b_scores[indices])
            - roc_auc_score(sampled_labels, m4_scores[indices])
        ))
    deltas = np.asarray(deltas, dtype=float)
    return {
        "available": True,
        "aligned_patients": int(len(merged)),
        "m4_delta_patient_auc": m4_delta,
        "m5b_delta_patient_auc": m5b_delta,
        "delta_ratio_m5b_over_m4": (m5b_delta / m4_delta) if m4_delta > 0 else None,
        "auc_m5b_minus_m4_ci95": [
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)),
        ],
        "iterations_used": int(len(deltas)),
    }


def prepare_output(args):
    name = args.run_name or (
        f"m5b_{args.alpha_mode}_balanced_keep_efficientnet_b0_seed{args.seed}"
    )
    if args.debug:
        name += "_debug"
    output = args.output_root / name
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def self_test():
    """回归自测：残差初值、二分类损失等价性及患者预聚合bootstrap。"""
    # init-0：Δm=0 → m_final = m_global。
    mlp = nn.Sequential(nn.Linear(3, 4), nn.SiLU(inplace=True), nn.Linear(4, 1))
    nn.init.zeros_(mlp[-1].weight)
    nn.init.zeros_(mlp[-1].bias)
    x = torch.randn(5, 3)
    with torch.no_grad():
        if mlp(x).abs().max().item() > 1e-6:
            raise AssertionError("残差头init-0后Δm应全为0")
    # BCE_with_logits(m, label) == 2类CE([-m/2,+m/2], label)。
    m = torch.tensor([1.5, -0.7, 0.0, 2.0], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    bce = F.binary_cross_entropy_with_logits(m, labels)
    logits = torch.stack([-m / 2, m / 2], dim=1)
    ce = F.cross_entropy(logits, labels.long())
    if abs(float(bce.detach()) - float(ce.detach())) > 1e-5:
        raise AssertionError("BCE(logit差)与2类CE不等价")
    # 同一患者多图先聚合；目标等于基线时bootstrap差异必须解析为精确0。
    frame = pd.DataFrame({
        "patient_id": ["A", "A", "B", "B", "C", "D"],
        "label": [1, 1, 0, 0, 1, 0],
        "p_global": [0.9, 0.7, 0.3, 0.1, 0.8, 0.2],
    })
    frame["p_target"] = frame.p_global
    bootstrap = patient_paired_bootstrap(
        frame, target_column="p_target", iterations=50, seed=42,
    )
    if bootstrap["delta_ci95"] != [0.0, 0.0] \
            or bootstrap["delta_mean"] != 0.0:
        raise AssertionError("患者预聚合bootstrap未保持相同预测的零差异")
    return True


def main():
    """编排M5b：阶段L训局部头→冻结→阶段R预计算+训残差头→产品保护选择→审计保存。"""
    args = parse_args()
    if args.self_test:
        self_test()
        print("M5b自测通过: 残差init-0 + BCE等价性 + 患者预聚合bootstrap")
        return
    if args.debug:
        args.stage_l_epochs = min(args.stage_l_epochs, 1)
        args.stage_r_epochs = min(args.stage_r_epochs, 2)
        args.num_workers = 0
    if args.batch_size != 32:
        raise ValueError("M5b正式协议冻结batch_size=32，不允许运行时改写")
    if not np.isclose(args.lambda_reg, LAMBDA_REG, rtol=0.0, atol=1e-12):
        raise ValueError(f"M5b正式协议冻结lambda_reg={LAMBDA_REG}，不允许运行时改写")
    seed_everything(args.seed)
    roi_manifest = args.roi_manifest or default_roi_manifest(args.seed)
    m1_checkpoint = args.m1_checkpoint or default_m1_checkpoint(args.seed)
    if not Path(roi_manifest).is_file():
        raise FileNotFoundError(f"先运行build_m5_roi_manifest.py生成: {roi_manifest}")
    if not Path(m1_checkpoint).is_file():
        raise FileNotFoundError(m1_checkpoint)

    frame = load_roi_manifest(
        roi_manifest, args.image_root, args.debug, args.debug_units, args.seed
    )
    # 产物链校验（与M5相同）：seed/debug/M1 SHA/行数/ROI CSV SHA。
    roi_config_path = Path(
        str(roi_manifest).replace("m5_roi_manifest", "m5_roi_config")
        .replace(".csv", ".json")
    )
    if not roi_config_path.is_file():
        raise FileNotFoundError(roi_config_path)
    roi_config = json.loads(roi_config_path.read_text(encoding="utf-8"))
    if int(roi_config.get("seed", -1)) != args.seed:
        raise ValueError(f"ROI config seed不匹配: {roi_config.get('seed')} != {args.seed}")
    if roi_config.get("debug") and not args.debug:
        raise ValueError("ROI清单是debug产物，禁止用于正式训练")
    if file_sha256(m1_checkpoint) != roi_config.get("m1_checkpoint_sha256"):
        raise ValueError("M1 checkpoint与ROI清单记录的SHA不一致")
    if file_sha256(roi_manifest) != roi_config.get("roi_csv_sha256"):
        raise ValueError("ROI CSV与config记录的SHA不一致")
    expected_rows = int(roi_config["n_train"]) + int(roi_config["n_val"])
    if not args.debug and len(frame) != expected_rows:
        raise ValueError(f"ROI清单行数{len(frame)} != config计数{expected_rows}")
    if int(roi_config.get("inference_batch_size", -1)) != 32 \
            or int(roi_config.get("reference_m3c_batch_size", -1)) != 32:
        raise ValueError("M5b只接受已验收的batch32 M5 ROI清单")
    output = prepare_output(args)

    eval_tf, roi_train_tf, roi_eval_tf = build_transforms()
    eval_datasets = {
        split: M5bDataset(
            frame.loc[frame.split.eq(split)].copy(), args.image_root,
            eval_tf, roi_eval_tf,
        )
        for split in ("train", "val")
    }
    local_train_dataset = M5bDataset(
        frame.loc[frame.split.eq("train")].copy(), args.image_root,
        eval_tf, roi_train_tf,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m1_payload = torch.load(m1_checkpoint, map_location="cpu", weights_only=False)
    m1 = EfficientNetM1()
    m1.load_state_dict(m1_payload["model_state_dict"], strict=True)
    model = M5bModel(m1.backbone, args.alpha_mode).to(device)
    freeze_before = frozen_m1_snapshot(model)
    print(f"设备: {device}; seed={args.seed} alpha_mode={args.alpha_mode}; "
          f"train={len(eval_datasets['train'])}张; val={len(eval_datasets['val'])}张")

    # ---------- 阶段L：训局部头，按val局部头AUC选择后冻结 ----------
    generator = torch.Generator().manual_seed(args.seed)
    train_sampler = WeightedRandomSampler(
        patient_class_balanced_weights(local_train_dataset.df),
        num_samples=len(local_train_dataset), replacement=True, generator=generator,
    )
    train_loader = DataLoader(local_train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available(),
                              worker_init_fn=seed_worker)
    val_loader = DataLoader(eval_datasets["val"], batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=torch.cuda.is_available(),
                            worker_init_fn=seed_worker)

    best_local = {"auc": -1.0, "state": None, "epoch": None}
    stage_l_history = []
    model.set_stage_l()
    optimizer_l = optim.AdamW(model.local_head.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    for epoch in range(1, args.stage_l_epochs + 1):
        model.local_head.train()
        total = 0.0
        for batch in train_loader:
            rois = batch["roi"].to(device, non_blocking=True)
            roi_labels = batch["roi_label"].to(device, non_blocking=True)
            optimizer_l.zero_grad(set_to_none=True)
            _, l_logits = model.local_logits(rois)
            loss = F.cross_entropy(l_logits, roi_labels, ignore_index=IGNORE)
            loss.backward()
            optimizer_l.step()
            total += float(loss.detach()) * len(rois)
        val_local_auc = evaluate_stage_l(model, val_loader, device)
        if val_local_auc > best_local["auc"]:
            best_local = {"auc": val_local_auc, "state": deepcopy(model.local_head.state_dict()),
                          "epoch": epoch}
        stage_l_history.append({
            "epoch": epoch,
            "train_local_loss": total / len(local_train_dataset),
            "val_local_head_auc": val_local_auc,
        })
        print(f"stageL {epoch:02d}/{args.stage_l_epochs} | train_local_loss={total/len(local_train_dataset):.4f} "
              f"| val局部头AUC={val_local_auc:.4f}")
    model.local_head.load_state_dict(best_local["state"])
    model.set_stage_r()
    print(f"阶段L完成: 最佳局部头 epoch={best_local['epoch']} val局部头AUC={best_local['auc']:.4f}")

    # ---------- 阶段R：预计算冻结特征，只训残差MLP ----------
    tables = precompute_features(
        model, eval_datasets, device, args.batch_size, args.num_workers
    )
    train_table, val_table = tables["train"], tables["val"]
    stage_r = train_residual(train_table, val_table, args, device)
    print(f"阶段R完成: best epoch={stage_r['best_epoch']} "
          f"全局患者AUC={stage_r['global_patient_auc']:.4f} "
          f"最佳融合患者AUC={stage_r['best_fusion_patient_auc']:.4f} "
          f"产品患者AUC={stage_r['product_patient_auc']:.4f} "
          f"Δ={stage_r['delta_patient_auc']:.4f} rawΔ={stage_r['raw_delta_patient_auc']:.4f} "
          f"回退={stage_r['fell_back_to_global']}")

    # 冻结校验：全局分支+局部Encoder参数/BN不变。
    freeze_after = frozen_m1_snapshot(model)
    freeze_ok = (freeze_before["params_sha"] == freeze_after["params_sha"]
                 and freeze_before["bn_running_sha"] == freeze_after["bn_running_sha"])
    if not freeze_ok:
        raise RuntimeError("冻结校验失败：全局分支或局部Encoder被意外修改")

    # 最终产品评估表：val上合并全局/融合概率（用最佳残差模型）。
    vf = val_table.copy()
    with torch.no_grad():
        x = torch.tensor(vf[["m_global", "m_local", "similarity"]].to_numpy(),
                         dtype=torch.float32, device=device)
        dm = torch.tanh(stage_r["model"](x).squeeze(1)).cpu().numpy()
    m_global_np = vf.m_global.to_numpy()
    alpha_np = vf.alpha.to_numpy()
    vf["dm"] = dm
    vf["correction"] = alpha_np * dm
    vf["p_candidate"] = 1.0 / (1.0 + np.exp(-(m_global_np + alpha_np * dm)))
    vf["p_global"] = 1.0 / (1.0 + np.exp(-m_global_np))
    vf["p_final"] = (
        vf.p_global if stage_r["fell_back_to_global"] else vf.p_candidate
    )
    candidate_auc_check = patient_auc_of(vf, "p_candidate")
    product_auc_check = patient_auc_of(vf, "p_final")
    if not np.isclose(
        candidate_auc_check, stage_r["best_fusion_patient_auc"],
        rtol=0.0, atol=1e-12,
    ):
        raise RuntimeError("候选融合预测与selection记录的患者AUC不一致")
    if not np.isclose(
        product_auc_check, stage_r["product_patient_auc"],
        rtol=0.0, atol=1e-12,
    ):
        raise RuntimeError("正式产品预测与selection记录的患者AUC不一致")

    # 候选融合始终归档；若触发回退，正式产品将残差最后一层清零，严格输出全局结果。
    model.residual_mlp.load_state_dict(stage_r["model"].state_dict())
    candidate_model_state = deepcopy(model.state_dict())
    if stage_r["fell_back_to_global"]:
        nn.init.zeros_(model.residual_mlp[-1].weight)
        nn.init.zeros_(model.residual_mlp[-1].bias)
    product_model_state = deepcopy(model.state_dict())
    model.eval()

    patient_final = patient_top2_mean(vf, "p_final")
    patient_candidate = patient_top2_mean(vf, "p_candidate")
    candidate_bootstrap = patient_paired_bootstrap(
        vf, target_column="p_candidate",
        iterations=args.bootstrap, seed=args.seed,
    )
    if stage_r["fell_back_to_global"]:
        bootstrap = {
            "delta_ci95": [0.0, 0.0],
            "delta_mean": 0.0,
            "iterations_used": args.bootstrap,
            "computed_analytically": True,
            "reason": "product_predictions_equal_global",
        }
    else:
        bootstrap = deepcopy(candidate_bootstrap)
        bootstrap["reused_from_candidate"] = True
    audit = metadata_audit(frame)
    attribution = compute_attribution(vf, vf.p_final, vf.p_global)
    auxiliary = compute_auxiliary_metrics(vf, roi_config)
    survival = survival_diagnostics(
        patient_final, args.seed, stage_r["delta_patient_auc"], args.bootstrap
    )

    config = {
        **{key: str(value) if isinstance(value, Path) else value
           for key, value in vars(args).items()},
        "roi_manifest": str(Path(roi_manifest).resolve()),
        "roi_manifest_sha256": file_sha256(roi_manifest),
        "m1_checkpoint": str(Path(m1_checkpoint).resolve()),
        "m1_checkpoint_sha256": file_sha256(m1_checkpoint),
        "freeze_ok": freeze_ok,
        "stage_l": {"best_epoch": best_local["epoch"],
                    "val_local_head_auc": best_local["auc"]},
        "m5b_selection": {
            "alpha_mode": args.alpha_mode,
            "best_stage_r_epoch": stage_r["best_epoch"],
            "global_patient_auc": stage_r["global_patient_auc"],
            "best_fusion_patient_auc": stage_r["best_fusion_patient_auc"],
            "product_patient_auc": stage_r["product_patient_auc"],
            "fell_back_to_global": stage_r["fell_back_to_global"],
            "delta_patient_auc": stage_r["delta_patient_auc"],
            "raw_delta_patient_auc": stage_r["raw_delta_patient_auc"],
            "patient_paired_bootstrap": bootstrap,
            "candidate_patient_paired_bootstrap": candidate_bootstrap,
            "metadata_audit": audit,
            "attribution": attribution,
            "auxiliary_metrics": auxiliary,
            "survival_diagnostics": survival,
        },
        "test_evaluated": False,
        "external_evaluated": False,
        **git_snapshot(),
    }
    torch.save({
        "model_state_dict": product_model_state,
        "config": json_ready(config),
    }, output / "m5b_best.pth")
    torch.save({
        "model_state_dict": candidate_model_state,
        "config": json_ready(config),
        "artifact_role": "best_fusion_candidate_before_product_fallback",
    }, output / "m5b_best_fusion_candidate.pth")
    pd.DataFrame(stage_r["history"]).to_csv(
        output / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(stage_l_history).to_csv(
        output / "stage_l_history.csv", index=False, encoding="utf-8-sig"
    )
    vf.to_csv(output / "val_image_predictions.csv", index=False, encoding="utf-8-sig")
    patient_final.to_csv(output / "val_patient_predictions.csv", index=False,
                         encoding="utf-8-sig")
    patient_candidate.to_csv(
        output / "val_patient_candidate_predictions.csv",
        index=False, encoding="utf-8-sig",
    )
    (output / "config.json").write_text(
        json.dumps(json_ready(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), output / "source_entry.py")
    print(f"全局患者AUC={stage_r['global_patient_auc']:.4f} -> "
          f"M5b产品患者AUC={stage_r['product_patient_auc']:.4f} | "
          f"Δ={stage_r['delta_patient_auc']:.4f} (raw融合Δ={stage_r['raw_delta_patient_auc']:.4f})")
    print(f"回退全局={stage_r['fell_back_to_global']} | 元数据审计: {audit}")
    print(f"归因: {attribution['signals']}")
    if survival.get("available"):
        print(f"存活率: ratio={survival.get('delta_ratio_m5b_over_m4')} "
              f"AUC_M5b-M4 CI={survival.get('auc_m5b_minus_m4_ci95')}")
    print(f"输出目录: {output}")


if __name__ == "__main__":
    main()
