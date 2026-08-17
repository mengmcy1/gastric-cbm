#!/usr/bin/env python3
"""在冻结M0-F全局或Y6局部EfficientNet-B0特征上训练SAE。

全局分支读取完整胃镜图；局部分支读取冻结Y3-F清单中实际触发Y6的预测ROI。脚本只接受
train/val，提取``features[8]``的7x7特征图及1280维GAP向量，训练Linear-ReLU SAE，
复用对应冻结分类头评价重构保真度，并生成Feature统计、剪枝结果和空间响应概览。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.metrics import accuracy_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from torchvision.transforms import v2


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
TRAIN_CODE = PROJECT_ROOT / "程序/模型训练/正式代码"
sys.path.insert(0, str(TRAIN_CODE))
sys.path.insert(0, str(SCRIPT_DIR))

from efficientnet_train_debiased import build_model  # noqa: E402
from run_y1_yolo26_smoke import file_sha256  # noqa: E402
from train_utils import git_snapshot, json_ready  # noqa: E402
from sae_discovery import (  # noqa: E402
    ACTIVE_EPS,
    SparseAutoencoder,
    build_feature_summary,
    patient_class_weights,
    reconstruct_masked_features,
    run_feature_pruning,
)


SEEDS = (42, 202, 503)
INPUT_DIM = 1280
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MAPPING = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/"
    "11_Y0F_YOLO26完整诊断数据_20260813/y0f_mapping.csv"
)
MAPPING_SHA256 = "b329d8d3b0033e84124fb7be6db05a70d6bb46fb014a3653ef20cadac07407cd"
LOCAL_MANIFEST_ROOT = PROJECT_ROOT / "数据整理记录/SAE_EfficientNet_0804/冻结局部ROI清单"
M0_ROOT = PROJECT_ROOT / "结果/M0全量诊断_0804/正式验证集筛选"
Y6_ROOT = PROJECT_ROOT / "结果/YOLO26定位_0804/Y6预测ROI局部分类_Full_20260814"
OUTPUT_ROOT = PROJECT_ROOT / "结果/SAE/EfficientNet全局局部_0804"


def parse_args() -> argparse.Namespace:
    """解析分支、字典宽度、训练和debug参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("global", "local"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument(
        "--activation-mode", choices=("relu_l1", "topk"), default="relu_l1",
        help="relu_l1使用L1软稀疏；topk为每个样本保留固定数量的最大非负激活。",
    )
    parser.add_argument("--top-k", type=int, help="topk模式下每张图保留的Feature数。")
    parser.add_argument("--lambda-l1", type=float, default=5e-4)
    parser.add_argument(
        "--margin-loss-weight", type=float, default=0.0,
        help="冻结分类头标准化margin重构损失权重；0保持既有SAE行为。",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--sae-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overview-features", type=int, default=20)
    parser.add_argument("--top-images", type=int, default=6)
    parser.add_argument("--pruning-min-active-patients", type=int, default=5)
    parser.add_argument("--pruning-ce-tolerance", type=float, default=0.01)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--feature-cache-from", type=Path,
        help="复用同分支/seed/checkpoint/manifest正式特征缓存，仍在新目录保存独立副本。",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-patients-per-class", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """固定SAE、采样和DataLoader随机状态。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def module_sha(module: nn.Module) -> str:
    """计算参数与BN缓冲区的稳定SHA-256。"""
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class TopKSparseAutoencoder(SparseAutoencoder):
    """使用逐样本Top-K激活的过完备SAE，输入/输出维度仍为1280。"""

    def __init__(self, input_dim: int, hidden_dim: int, feature_center: torch.Tensor, top_k: int):
        super().__init__(input_dim, hidden_dim, feature_center)
        self.top_k = top_k

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """返回[B,H]非负激活，每行恰有不超过top_k个非零值。"""
        dense = torch.relu(self.encoder(features - self.decoder_bias))
        values, indices = torch.topk(dense, self.top_k, dim=1, sorted=False)
        return torch.zeros_like(dense).scatter(1, indices, values)


def product_paths(branch: str, seed: int) -> dict[str, Path]:
    """返回同seed冻结分类器、配置和可选局部ROI清单。"""
    m0_run = M0_ROOT / f"m0_full_keep_efficientnet_b0_seed{seed}"
    if branch == "global":
        return {
            "checkpoint": m0_run / "efficientnet_b0_debiased_best.pth",
            "config": m0_run / "config.json",
            "manifest": MAPPING,
        }
    y6_run = Y6_ROOT / f"y6_full_efficientnet_b0_seed{seed}"
    return {
        "checkpoint": y6_run / "y6_best_local.pth",
        "config": y6_run / "config.json",
        "manifest": LOCAL_MANIFEST_ROOT / f"efficientnet_local_roi_seed{seed}.csv",
        "manifest_config": LOCAL_MANIFEST_ROOT / f"efficientnet_local_roi_seed{seed}.json",
    }


def validate_products(branch: str, seed: int, products: dict[str, Path]) -> dict:
    """核验冻结配置属于当前seed和分支，并返回来源配置。"""
    for path in products.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(products["config"].read_text(encoding="utf-8"))
    if int(config["seed"]) != seed:
        raise ValueError("冻结分类器配置seed不一致")
    if branch == "global" and config.get("model") != "EfficientNet-B0":
        raise ValueError("全局SAE来源不是EfficientNet-B0")
    if branch == "local" and config.get("stage") != "Y6_exploratory_full":
        raise ValueError("局部SAE来源不是冻结Y6 Full局部分类器")
    return config


def select_debug_patients(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    """按split和标签选择少量完整患者，保持患者不跨集合。"""
    selected: list[str] = []
    for _, split_frame in frame.groupby("split", sort=False):
        patients = split_frame[["patient_id", "label"]].drop_duplicates()
        for _, group in patients.groupby("label", sort=True):
            selected.extend(
                group.sample(min(count, len(group)), random_state=seed).patient_id.tolist()
            )
    return frame.loc[frame.patient_id.isin(selected)].reset_index(drop=True)


def load_dataframe(args: argparse.Namespace, products: dict[str, Path]) -> pd.DataFrame:
    """加载train/val并落实全局完整图或局部触发ROI口径。"""
    if args.branch == "global":
        if file_sha256(products["manifest"]) != MAPPING_SHA256:
            raise ValueError("Y0F mapping SHA不一致")
        frame = pd.read_csv(products["manifest"], encoding="utf-8-sig", dtype={"patient_id": str})
    else:
        manifest_config = json.loads(products["manifest_config"].read_text(encoding="utf-8"))
        if int(manifest_config["seed"]) != args.seed:
            raise ValueError("局部ROI清单seed不一致")
        if file_sha256(products["manifest"]) != manifest_config["output_csv_sha256"]:
            raise ValueError("局部ROI清单SHA不一致")
        frame = pd.read_csv(
            products["manifest"], encoding="utf-8-sig", dtype={"patient_id": str},
            float_precision="round_trip",
        )
        frame = frame.loc[frame.has_roi.astype(bool)].copy()
    frame = frame.loc[frame.split.isin(["train", "val"])].reset_index(drop=True)
    required = {"image_relpath", "patient_id", "label", "split", "source", "center"}
    if args.branch == "local":
        required.update({"roi_x1", "roi_y1", "roi_x2", "roi_y2", "top1_confidence"})
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"SAE清单缺少字段: {sorted(missing)}")
    if frame.groupby("patient_id").split.nunique().gt(1).any():
        raise ValueError("SAE清单存在患者跨split")
    if frame.groupby("patient_id").label.nunique().gt(1).any():
        raise ValueError("SAE清单存在患者标签冲突")
    for split in ("train", "val"):
        if set(frame.loc[frame.split.eq(split), "label"]) != {0, 1}:
            raise ValueError(f"{split}未同时包含两类")
    frame["domain"] = np.where(frame.center.eq("武大省人民"), "省人民", "外院")
    frame["hospital"] = frame.center.astype(str)
    if args.debug:
        frame = select_debug_patients(frame, args.debug_patients_per_class, args.seed)
    return frame


def load_model(branch: str, products: dict[str, Path], device: torch.device) -> nn.Module:
    """恢复冻结M0-F或Y6局部EfficientNet，并拒绝参数训练。"""
    payload = torch.load(products["checkpoint"], map_location="cpu", weights_only=False)
    state = payload["model_state_dict"] if branch == "global" else payload["local_model_state_dict"]
    model = build_model(pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


class ActivationCapture:
    """保存EfficientNet-B0 ``features[8]``输出[B,1280,7,7]。"""

    def __init__(self, layer: nn.Module):
        self.output: torch.Tensor | None = None
        self.handle = layer.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output) -> None:
        self.output = output

    def close(self) -> None:
        self.handle.remove()


EVAL_TRANSFORM = v2.Compose([
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class FeatureDataset(Dataset):
    """读取完整图或冻结预测ROI，返回[3,224,224]张量和清单行号。"""

    def __init__(self, frame: pd.DataFrame, branch: str):
        self.frame = frame.reset_index(drop=True)
        self.branch = branch

    def __len__(self) -> int:
        return len(self.frame)

    def load_pil(self, index: int) -> Image.Image:
        """按分支返回送入对应Encoder的确定性RGB图像。"""
        row = self.frame.iloc[index]
        with Image.open(PROJECT_ROOT / row.image_relpath) as source:
            image = source.convert("RGB")
        if self.branch == "local":
            width, height = image.size
            values = row[["roi_x1", "roi_y1", "roi_x2", "roi_y2"]].to_numpy(float)
            x1, y1, x2, y2 = values * np.array([width, height, width, height])
            image = image.crop((int(x1), int(y1), max(int(x2), int(x1) + 1), max(int(y2), int(y1) + 1)))
        return image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return EVAL_TRANSFORM(self.load_pil(index)), index


@torch.no_grad()
def extract_features(
    model: nn.Module, capture: ActivationCapture, frame: pd.DataFrame,
    branch: str, args: argparse.Namespace, device: torch.device, cache: Path,
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """逐split提取GAP特征、冻结头概率和对应元数据。"""
    features: dict[str, np.ndarray] = {}
    metadata: dict[str, pd.DataFrame] = {}
    for split in ("train", "val"):
        split_frame = frame.loc[frame.split.eq(split)].reset_index(drop=True)
        loader = DataLoader(
            FeatureDataset(split_frame, branch), batch_size=args.image_batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda",
        )
        feature_batches, probability_batches = [], []
        for batch_index, (images, _) in enumerate(loader, 1):
            logits = model(images.to(device, non_blocking=True))
            if capture.output is None or capture.output.shape[1:] != (INPUT_DIM, 7, 7):
                raise RuntimeError("EfficientNet捕获层形状不是[1280,7,7]")
            feature_batches.append(capture.output.mean((2, 3)).cpu().numpy())
            probability_batches.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            print(f"提取 {branch}/{split}: [{batch_index}/{len(loader)}]")
        array = np.concatenate(feature_batches).astype(np.float32)
        split_frame["cancer_probability"] = np.concatenate(probability_batches).astype(np.float32)
        np.save(cache / f"{split}_gap_features.npy", array)
        split_frame.to_csv(cache / f"{split}_metadata.csv", index=False, encoding="utf-8-sig")
        features[split], metadata[split] = array, split_frame
        print(f"{branch}/{split}: {len(split_frame)}张，特征={array.shape}")
    return features, metadata


def reuse_feature_cache(
    source: Path, branch: str, seed: int, products: dict[str, Path], target: Path,
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """校验A0血缘后复用GAP特征，避免A1每个超参数组合重复跑Encoder。"""
    source_run = source.parent
    config_path = source_run / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "branch": branch,
        "seed": seed,
        "checkpoint_sha256": file_sha256(products["checkpoint"]),
        "manifest_sha256": file_sha256(products["manifest"]),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"复用特征缓存的{key}与当前任务不一致")
    features, metadata = {}, {}
    for split in ("train", "val"):
        feature_path = source / f"{split}_gap_features.npy"
        metadata_path = source / f"{split}_metadata.csv"
        features[split] = np.load(feature_path).astype(np.float32, copy=False)
        metadata[split] = pd.read_csv(
            metadata_path, encoding="utf-8-sig", dtype={"patient_id": str},
            float_precision="round_trip",
        )
        shutil.copy2(feature_path, target / feature_path.name)
        shutil.copy2(metadata_path, target / metadata_path.name)
    print(f"复用冻结特征缓存: {source}")
    return features, metadata


def patient_mean(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """按患者全部可用图像概率均值聚合。"""
    return frame.groupby("patient_id", as_index=False).agg(
        label=("label", "first"), **{column: (column, "mean")},
    )


def lock_sensitivity_threshold(labels: np.ndarray, probabilities: np.ndarray, target: float = 0.90) -> float:
    """返回正类召回不低于目标值的最高实际候选阈值。"""
    positives = probabilities[labels == 1]
    for threshold in np.sort(np.unique(positives))[::-1]:
        if float(np.mean(positives >= threshold)) >= target:
            return float(threshold)
    raise RuntimeError("无法锁定Sensitivity阈值")


def annotate_confusion(metadata: dict[str, pd.DataFrame]) -> tuple[float, float]:
    """用val分别冻结图像/患者阈值，并给train/val标记TP/TN/FP/FN。"""
    val = metadata["val"]
    image_threshold = lock_sensitivity_threshold(
        val.label.to_numpy(), val.cancer_probability.to_numpy(),
    )
    val_patient = patient_mean(val, "cancer_probability")
    patient_threshold = lock_sensitivity_threshold(
        val_patient.label.to_numpy(), val_patient.cancer_probability.to_numpy(),
    )
    for frame in metadata.values():
        predicted = frame.cancer_probability.ge(image_threshold).astype(int)
        frame["pred_label"] = predicted
        frame["confusion_type"] = np.select(
            [
                frame.label.eq(1) & predicted.eq(1), frame.label.eq(0) & predicted.eq(0),
                frame.label.eq(0) & predicted.eq(1), frame.label.eq(1) & predicted.eq(0),
            ], ["TP", "TN", "FP", "FN"], default="",
        )
    return image_threshold, patient_threshold


def make_balanced_loader(
    features: np.ndarray, metadata: pd.DataFrame, batch_size: int, seed: int,
) -> DataLoader:
    """构造类别平衡且每位患者总采样权重相等的训练Loader。"""
    weights = patient_class_weights(metadata)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), len(features), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(TensorDataset(torch.from_numpy(features)), batch_size=batch_size, sampler=sampler)


@torch.no_grad()
def evaluate_loss(
    sae: SparseAutoencoder, features: np.ndarray, metadata: pd.DataFrame,
    target_lambda: float, batch_size: int, device: torch.device,
    classifier_margin: torch.Tensor, margin_std: float, margin_loss_weight: float,
) -> np.ndarray:
    """按患者/类别平衡计算总损失、MSE、L1、L0和标准化margin误差。"""
    weights = patient_class_weights(metadata)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features), torch.from_numpy(weights)),
        batch_size=batch_size, shuffle=False,
    )
    total = np.zeros(5, dtype=np.float64)
    weight_sum = 0.0
    sae.eval()
    for batch, sample_weights in loader:
        batch = batch.to(device)
        reconstructed, hidden = sae(batch)
        mse = (reconstructed - batch).pow(2).mean(dim=1)
        l1 = hidden.abs().sum(dim=1)
        l0 = hidden.gt(ACTIVE_EPS).sum(dim=1).float()
        margin = ((reconstructed - batch) @ classifier_margin / margin_std).pow(2)
        base = mse if isinstance(sae, TopKSparseAutoencoder) else mse + target_lambda * l1
        values = torch.stack([
            base + margin_loss_weight * margin, mse, l1, l0, margin,
        ], dim=1).cpu().numpy()
        current_weights = sample_weights.numpy()
        total += (values * current_weights[:, None]).sum(axis=0)
        weight_sum += current_weights.sum()
    return total / weight_sum


def train_sae(
    train_features: np.ndarray, train_metadata: pd.DataFrame,
    val_features: np.ndarray, val_metadata: pd.DataFrame,
    args: argparse.Namespace, device: torch.device, model_dir: Path,
    classifier_margin: np.ndarray,
) -> tuple[SparseAutoencoder, dict]:
    """患者平衡训练SAE，并按val完整重构目标选择checkpoint。"""
    loader = make_balanced_loader(train_features, train_metadata, args.sae_batch_size, args.seed)
    center = np.average(
        train_features, axis=0, weights=patient_class_weights(train_metadata),
    ).astype(np.float32)
    if args.activation_mode == "topk":
        sae = TopKSparseAutoencoder(
            INPUT_DIM, args.hidden_dim, torch.from_numpy(center).to(device), args.top_k,
        ).to(device)
    else:
        sae = SparseAutoencoder(INPUT_DIM, args.hidden_dim, torch.from_numpy(center).to(device)).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.learning_rate)
    balance_weights = patient_class_weights(train_metadata)
    train_margins = train_features @ classifier_margin
    margin_mean = float(np.average(train_margins, weights=balance_weights))
    margin_std = float(np.sqrt(np.average(
        np.square(train_margins - margin_mean), weights=balance_weights,
    )))
    if margin_std <= 1e-8:
        raise RuntimeError("冻结分类margin在train上没有有效方差")
    margin_vector = torch.from_numpy(classifier_margin.astype(np.float32)).to(device)
    warmup_epochs = max(1, int(math.ceil(args.epochs * args.warmup_fraction)))
    history, best_loss, stale = [], math.inf, 0
    best_path = model_dir / "sae_best.pth"

    for epoch in range(1, args.epochs + 1):
        scale = min(1.0, epoch / warmup_epochs)
        current_lambda = args.lambda_l1 * scale
        current_margin_weight = args.margin_loss_weight * scale
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * scale
        sae.train()
        totals = np.zeros(5, dtype=np.float64)
        sample_count = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, hidden = sae(batch)
            mse = (reconstructed - batch).pow(2).mean()
            l1 = hidden.abs().sum(dim=1).mean()
            margin = ((reconstructed - batch) @ margin_vector / margin_std).pow(2).mean()
            base = mse if args.activation_mode == "topk" else mse + current_lambda * l1
            loss = base + current_margin_weight * margin
            loss.backward()
            with torch.no_grad():
                weight, gradient = sae.decoder_weight, sae.decoder_weight.grad
                projection = (gradient * weight).sum(1, keepdim=True)
                gradient.sub_(projection / weight.square().sum(1, keepdim=True).clamp_min(1e-12) * weight)
            optimizer.step()
            sae.normalize_decoder()
            count = len(batch)
            totals += np.array([
                loss.item(), mse.item(), l1.item(),
                hidden.gt(ACTIVE_EPS).sum(1).float().mean().item(), margin.item(),
            ]) * count
            sample_count += count
        train_values = totals / sample_count
        val_values = evaluate_loss(
            sae, val_features, val_metadata, args.lambda_l1, args.sae_batch_size, device,
            margin_vector, margin_std, args.margin_loss_weight,
        )
        row = {
            "epoch": epoch, "warmup_scale": scale, "learning_rate": args.learning_rate * scale,
            "lambda_l1": current_lambda, "train_total": train_values[0], "train_mse": train_values[1],
            "train_l1": train_values[2], "train_l0": train_values[3],
            "margin_loss_weight": current_margin_weight, "train_margin_mse": train_values[4],
            "val_total": val_values[0], "val_mse": val_values[1],
            "val_l1": val_values[2], "val_l0": val_values[3], "val_margin_mse": val_values[4],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} warmup={scale:.2f} "
            f"train MSE={train_values[1]:.5f} lambdaL1={current_lambda * train_values[2]:.5f} "
            f"margin={current_margin_weight * train_values[4]:.5f} L0={train_values[3]:.1f} | "
            f"val MSE={val_values[1]:.5f} lambdaL1={args.lambda_l1 * val_values[2]:.5f} "
            f"margin={args.margin_loss_weight * val_values[4]:.5f} L0={val_values[3]:.1f}"
        )
        if epoch < warmup_epochs:
            continue
        if val_values[0] < best_loss:
            best_loss, stale = float(val_values[0]), 0
            torch.save({
                "sae_state_dict": sae.state_dict(), "input_dim": INPUT_DIM,
                "hidden_dim": args.hidden_dim, "lambda_l1": args.lambda_l1,
                "activation_mode": args.activation_mode, "top_k": args.top_k,
                "margin_loss_weight": args.margin_loss_weight, "margin_train_std": margin_std,
                "epoch": epoch, "val_total_loss": best_loss,
            }, best_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop: {args.patience}轮无val改善")
                break
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    sae.load_state_dict(checkpoint["sae_state_dict"], strict=True)
    return sae, checkpoint


@torch.no_grad()
def project(sae: SparseAutoencoder, features: np.ndarray, batch_size: int, device: torch.device):
    """以固定顺序返回重构向量和非负SAE激活。"""
    reconstructed, activations = [], []
    loader = DataLoader(TensorDataset(torch.from_numpy(features)), batch_size=batch_size, shuffle=False)
    sae.eval()
    for (batch,) in loader:
        current_reconstruction, current_activation = sae(batch.to(device))
        reconstructed.append(current_reconstruction.cpu().numpy())
        activations.append(current_activation.cpu().numpy())
    return np.concatenate(reconstructed).astype(np.float32), np.concatenate(activations).astype(np.float32)


def ncc90(activations: np.ndarray) -> float:
    """返回每图覆盖90%总激活质量所需Feature数的均值。"""
    sorted_values = np.sort(activations, axis=1)[:, ::-1]
    totals = sorted_values.sum(1)
    cumulative = np.cumsum(sorted_values, axis=1)
    counts = np.where(totals > 0, (cumulative < 0.9 * totals[:, None]).sum(1) + 1, 0)
    return float(counts.mean())


def duplicate_decoder_rate(weights: np.ndarray, threshold: float = 0.95) -> float:
    """统计至少与另一decoder方向绝对余弦超过阈值的Feature比例。"""
    normalized = weights / np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-12)
    duplicated = np.zeros(len(weights), dtype=bool)
    block = 512
    for start in range(0, len(weights), block):
        similarities = np.abs(normalized[start:start + block] @ normalized.T)
        rows = np.arange(start, min(start + block, len(weights)))
        similarities[np.arange(len(rows)), rows] = 0
        duplicated[start:start + len(rows)] = similarities.max(1) >= threshold
    return float(duplicated.mean())


def classification_metrics(
    features: np.ndarray, reconstructed: np.ndarray, metadata: pd.DataFrame,
    weight: np.ndarray, bias: np.ndarray, patient_threshold: float,
) -> dict:
    """报告向量重构和冻结分类头在图像/患者层面的保真度。"""
    original_logits = features @ weight.T + bias
    reconstructed_logits = reconstructed @ weight.T + bias
    original_prob = torch.softmax(torch.from_numpy(original_logits), dim=1)[:, 1].numpy()
    reconstructed_prob = torch.softmax(torch.from_numpy(reconstructed_logits), dim=1)[:, 1].numpy()
    labels = metadata.label.to_numpy(int)
    frame = metadata[["patient_id", "label"]].copy()
    frame["original"] = original_prob
    frame["reconstructed"] = reconstructed_prob
    patients = frame.groupby("patient_id", as_index=False).agg(
        label=("label", "first"), original=("original", "mean"), reconstructed=("reconstructed", "mean"),
    )
    cosine = np.sum(features * reconstructed, axis=1) / (
        np.linalg.norm(features, axis=1) * np.linalg.norm(reconstructed, axis=1) + 1e-12
    )
    weights_balanced = patient_class_weights(metadata).astype(np.float64)
    weights_balanced /= weights_balanced.sum()
    original_ce = -np.log(torch.softmax(torch.from_numpy(original_logits), dim=1).numpy()[np.arange(len(labels)), labels] + 1e-12)
    reconstructed_ce = -np.log(torch.softmax(torch.from_numpy(reconstructed_logits), dim=1).numpy()[np.arange(len(labels)), labels] + 1e-12)
    zero_logits = np.broadcast_to(bias, original_logits.shape).copy()
    zero_ce = -np.log(torch.softmax(torch.from_numpy(zero_logits), dim=1).numpy()[np.arange(len(labels)), labels] + 1e-12)
    original_weighted_ce = float(np.sum(original_ce * weights_balanced))
    reconstructed_weighted_ce = float(np.sum(reconstructed_ce * weights_balanced))
    zero_weighted_ce = float(np.sum(zero_ce * weights_balanced))
    return {
        "mse": float(np.square(features - reconstructed).mean()),
        "mean_cosine": float(cosine.mean()),
        "image_original_auc": float(roc_auc_score(labels, original_prob)),
        "image_reconstructed_auc": float(roc_auc_score(labels, reconstructed_prob)),
        "image_prediction_agreement": float(np.mean(original_logits.argmax(1) == reconstructed_logits.argmax(1))),
        "image_cancer_probability_mae": float(np.mean(np.abs(original_prob - reconstructed_prob))),
        "patient_original_auc": float(roc_auc_score(patients.label, patients.original)),
        "patient_reconstructed_auc": float(roc_auc_score(patients.label, patients.reconstructed)),
        "patient_prediction_agreement_at_locked_threshold": float(np.mean(
            patients.original.ge(patient_threshold) == patients.reconstructed.ge(patient_threshold)
        )),
        "patient_cancer_probability_mae": float(np.mean(np.abs(patients.original - patients.reconstructed))),
        "original_cross_entropy": original_weighted_ce,
        "reconstructed_cross_entropy": reconstructed_weighted_ce,
        "recovered_cross_entropy": float(1 - (reconstructed_weighted_ce - original_weighted_ce) / (zero_weighted_ce - original_weighted_ce + 1e-12)),
    }


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """加载环境中的中文字体。"""
    return ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", size=size)


def heatmap_overlay(image: Image.Image, response: np.ndarray) -> Image.Image:
    """将7x7非负Feature空间响应以jet颜色覆盖在分支实际输入图上。"""
    response = np.maximum(response, 0)
    response /= response.max() + 1e-12
    heat = Image.fromarray((response * 255).astype(np.uint8)).resize(image.size, Image.Resampling.BICUBIC)
    values = np.asarray(heat, dtype=np.float32) / 255
    # 轻量jet映射，避免为概览额外依赖OpenCV。
    red = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    color = np.stack([red, green, blue], axis=-1) * 255
    original = np.asarray(image, dtype=np.float32)
    alpha = (0.65 * values)[..., None]
    return Image.fromarray(np.clip(original * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8))


@torch.no_grad()
def make_overviews(
    model: nn.Module, capture: ActivationCapture, sae: SparseAutoencoder,
    activations: np.ndarray, metadata: pd.DataFrame, summary: pd.DataFrame,
    branch: str, args: argparse.Namespace, device: torch.device, output: Path,
) -> None:
    """为高分类贡献Feature保存跨患者原输入/空间响应并排图。"""
    candidates = summary.loc[
        summary.kept_after_pruning & summary.active_patient_count.ge(2)
    ].nlargest(args.overview_features, "mean_abs_margin_contribution")
    dataset = FeatureDataset(metadata, branch)
    title_font, text_font = load_font(20), load_font(14)
    records = []
    for feature in candidates.itertuples(index=False):
        feature_id = int(feature.feature_id)
        order = np.argsort(activations[:, feature_id])[::-1]
        selected, used = [], set()
        for index in order:
            patient = metadata.iloc[index].patient_id
            if activations[index, feature_id] <= ACTIVE_EPS:
                break
            if patient not in used:
                selected.append(index)
                used.add(patient)
            if len(selected) == args.top_images:
                break
        canvas = Image.new("RGB", (640, 65 + 285 * len(selected)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"Feature {feature_id} 癌方向={feature.cancer_margin_direction:.4f}", fill="black", font=title_font)
        direction = sae.encoder.weight[feature_id].detach()
        for row_number, index in enumerate(selected):
            input_image = dataset.load_pil(index)
            model(EVAL_TRANSFORM(input_image).unsqueeze(0).to(device))
            response = torch.relu((capture.output[0] * direction[:, None, None]).sum(0)).cpu().numpy()
            overlay = heatmap_overlay(input_image, response)
            y = 65 + row_number * 285
            canvas.paste(ImageOps.fit(input_image, (320, 240)), (0, y))
            canvas.paste(ImageOps.fit(overlay, (320, 240)), (320, y))
            item = metadata.iloc[index]
            draw.text((8, y + 245), f"{item.patient_id} label={item.label} {item.source} 激活={activations[index, feature_id]:.4f}", fill="black", font=text_font)
            records.append({"feature_id": feature_id, "rank": row_number + 1, "image_relpath": item.image_relpath, "patient_id": item.patient_id, "label": int(item.label), "activation": float(activations[index, feature_id])})
        canvas.save(output / f"feature_{feature_id:04d}.png")
    pd.DataFrame(records).to_csv(output / "top_examples.csv", index=False, encoding="utf-8-sig")


def run_self_test() -> None:
    """验证L1/Top-K SAE的shape、固定激活数和NCC90。"""
    center = torch.zeros(8)
    sae = SparseAutoencoder(8, 4, center)
    reconstructed, hidden = sae(torch.randn(3, 8))
    assert reconstructed.shape == (3, 8) and hidden.shape == (3, 4)
    topk_sae = TopKSparseAutoencoder(8, 6, center, top_k=2)
    reconstructed, hidden = topk_sae(torch.randn(3, 8))
    assert reconstructed.shape == (3, 8) and hidden.shape == (3, 6)
    assert bool(hidden.gt(0).sum(1).le(2).all())
    assert ncc90(np.array([[4.0, 3.0, 2.0, 1.0]], dtype=np.float32)) == 3.0
    print("EfficientNet SAE self-test passed")


def main() -> None:
    """执行血缘校验、特征提取、SAE训练、剪枝、统计与空间概览。"""
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.activation_mode == "topk":
        if args.top_k is None or not 0 < args.top_k <= args.hidden_dim:
            raise ValueError("topk模式要求0 < top_k <= hidden_dim")
        if args.lambda_l1 != 0:
            raise ValueError("topk模式固定lambda_l1=0，避免同时改变两种稀疏机制")
    elif args.top_k is not None:
        raise ValueError("top_k只用于topk模式")
    if args.margin_loss_weight < 0:
        raise ValueError("margin_loss_weight不能为负数")
    seed_everything(args.seed)
    products = product_paths(args.branch, args.seed)
    source_config = validate_products(args.branch, args.seed, products)
    if not args.debug and args.hidden_dim not in {512, 1280, 2560, 5120, 10240}:
        raise ValueError("正式SAE hidden_dim不在冻结S0-S4网格中")
    output = args.output_root / args.experiment
    if output.exists():
        raise FileExistsError(f"SAE输出已存在，拒绝覆盖: {output}")
    cache, model_dir, pruning_dir, overview_dir = (
        output / "特征缓存", output / "SAE模型", output / "feature筛选", output / "feature概览",
    )
    for directory in (cache, model_dir, pruning_dir, overview_dir):
        directory.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_dataframe(args, products)
    model = load_model(args.branch, products, device)
    frozen_sha_before = module_sha(model)
    capture = ActivationCapture(model.features[8])
    if args.feature_cache_from:
        split_features, split_metadata = reuse_feature_cache(
            args.feature_cache_from.resolve(), args.branch, args.seed, products, cache,
        )
    else:
        split_features, split_metadata = extract_features(
            model, capture, frame, args.branch, args, device, cache,
        )
    image_threshold, patient_threshold = annotate_confusion(split_metadata)
    for split, current in split_metadata.items():
        current.to_csv(cache / f"{split}_metadata.csv", index=False, encoding="utf-8-sig")

    classifier = model.classifier[1]
    classifier_weight = classifier.weight.detach().cpu().numpy()
    classifier_bias = classifier.bias.detach().cpu().numpy()
    classifier_margin = classifier_weight[1] - classifier_weight[0]
    sae, best = train_sae(
        split_features["train"], split_metadata["train"],
        split_features["val"], split_metadata["val"], args, device, model_dir,
        classifier_margin,
    )
    activations, metrics = {}, {"best_epoch": int(best["epoch"]), "splits": {}}
    for split in ("train", "val"):
        reconstructed, current_activations = project(
            sae, split_features[split], args.sae_batch_size, device,
        )
        activations[split] = current_activations
        np.savez_compressed(cache / f"{split}_sae_projection.npz", reconstructed=reconstructed, activations=current_activations)
        current_metrics = classification_metrics(
            split_features[split], reconstructed, split_metadata[split],
            classifier_weight, classifier_bias, patient_threshold,
        )
        current_metrics.update({
            "mean_l0": float((current_activations > ACTIVE_EPS).sum(1).mean()),
            "mean_ncc90": ncc90(current_activations),
            "dead_feature_count": int(((current_activations > ACTIVE_EPS).sum(0) == 0).sum()),
        })
        metrics["splits"][split] = current_metrics

    kept, active_counts, curve, pruning = run_feature_pruning(
        sae, activations["train"], split_metadata["train"],
        split_features["val"], activations["val"], split_metadata["val"],
        classifier_weight, classifier_bias, args, device,
    )
    kept_mask = np.zeros(args.hidden_dim, dtype=bool)
    kept_mask[kept] = True
    pruning["decoder_duplicate_feature_rate_abs_cosine_ge_0p95"] = duplicate_decoder_rate(
        sae.decoder_weight.detach().cpu().numpy(), 0.95,
    )
    pruning["dead_feature_rate_train"] = float(
        metrics["splits"]["train"]["dead_feature_count"] / args.hidden_dim
    )
    pruning["splits"] = {}
    for split in ("train", "val"):
        pruned_reconstructed = reconstruct_masked_features(
            sae, activations[split], kept_mask, args.sae_batch_size, device,
        )
        pruning["splits"][split] = classification_metrics(
            split_features[split], pruned_reconstructed, split_metadata[split],
            classifier_weight, classifier_bias, patient_threshold,
        )
    metrics["pruning"] = pruning
    val_metrics = metrics["splits"]["val"]
    metrics["core_fidelity_gate"] = {
        "maximum_patient_auc_drop": 0.01,
        "minimum_patient_prediction_agreement": 0.95,
        "minimum_mean_cosine": 0.90,
        "minimum_recovered_cross_entropy": 0.95,
        "patient_auc_drop": float(
            val_metrics["patient_original_auc"] - val_metrics["patient_reconstructed_auc"]
        ),
        "passed": bool(
            val_metrics["patient_original_auc"] - val_metrics["patient_reconstructed_auc"] <= 0.01
            and val_metrics["patient_prediction_agreement_at_locked_threshold"] >= 0.95
            and val_metrics["mean_cosine"] >= 0.90
            and val_metrics["recovered_cross_entropy"] >= 0.95
        ),
    }
    pd.DataFrame({"feature_id": np.arange(args.hidden_dim), "active_patient_count_train": active_counts, "kept_after_pruning": kept_mask}).to_csv(pruning_dir / "feature_pruning_decisions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curve).to_csv(pruning_dir / "pruning_curve.csv", index=False, encoding="utf-8-sig")
    (pruning_dir / "pruning_summary.json").write_text(json.dumps(json_ready(pruning), ensure_ascii=False, indent=2), encoding="utf-8")

    development_activations = np.concatenate([activations["train"], activations["val"]])
    development_metadata = pd.concat([split_metadata["train"], split_metadata["val"]], ignore_index=True)
    summary = build_feature_summary(
        development_activations, development_metadata, sae, classifier_weight,
    )
    summary["kept_after_pruning"] = kept_mask
    summary["pruning_active_patient_count_train"] = active_counts
    summary.to_csv(output / "feature_summary.csv", index=False, encoding="utf-8-sig")
    (output / "metrics.json").write_text(json.dumps(json_ready(metrics), ensure_ascii=False, indent=2), encoding="utf-8")

    make_overviews(
        model, capture, sae, development_activations, development_metadata, summary,
        args.branch, args, device, overview_dir,
    )
    capture.close()
    if module_sha(model) != frozen_sha_before:
        raise RuntimeError("冻结EfficientNet参数或BN统计在SAE流程中发生变化")

    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config = {
        **serialized_args, "output": str(output.resolve()), "input_dim": INPUT_DIM,
        "checkpoint": str(products["checkpoint"].resolve()),
        "checkpoint_sha256": file_sha256(products["checkpoint"]),
        "source_config": str(products["config"].resolve()),
        "source_config_sha256": file_sha256(products["config"]),
        "source_stage": source_config.get("stage", source_config.get("model")),
        "manifest": str(products["manifest"].resolve()),
        "manifest_sha256": file_sha256(products["manifest"]),
        "image_threshold_locked_on_val": image_threshold,
        "patient_threshold_locked_on_val": patient_threshold,
        "counts": {
            split: {"images": int(len(part)), "patients": int(part.patient_id.nunique())}
            for split, part in split_metadata.items()
        },
        "feature_layer": "features[8] output -> GAP",
        "margin_train_std": best.get("margin_train_std"),
        "test_evaluated": False, "external_evaluated": False,
        **git_snapshot(),
    }
    (output / "config.json").write_text(json.dumps(json_ready(config), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"最佳epoch={best['epoch']}；保留Feature={len(kept)}/{args.hidden_dim}")
    print(f"输出目录: {output}")


if __name__ == "__main__":
    main()
