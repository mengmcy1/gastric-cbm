#!/usr/bin/env python3
"""在冻结C-long学生的attention-pooled 1280维表示上训练SAE（文献重构新路线）。

解释对象是C-long实际送入分类头的向量：

    features[8] [1280,7,7] x attention [1,7,7] -> 空间加权求和 -> [1280]
    -> Dropout(eval关闭) -> Linear(1280->2)

与旧``efficientnet_sae_discovery.py``的关键差异：

1. 模型为attention-pooling学生，不使用GAP；
2. 训练前强制核验S0六项血缘SHA（学生/manifest/audit/教师/缓存/beta JSON）；
3. 图像/患者阈值直接使用S0冻结值，不从val重新扫描；
4. 提取特征后必须用缓存向量复算val概率与attention，与正式
   ``val_image_predictions.csv``比较，超差拒绝继续；
5. 剪枝没有不合格fallback：无非空合格方案时保留全部Feature；
6. decoder重复率只在非死亡Feature上计算；
7. Feature空间响应使用 C-long attention x (decoder方向 . 位置向量)，
   即每个空间位置对该Feature重构方向的逐位置贡献；
8. 特征缓存与C-long seed42绑定，SAE seed42/202/503复用同一缓存。

协议冻结：SAE实验进度与结果讨论.md，S2-S3已预注册（2026-08-19）。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
MAGE_CODE = PROJECT_ROOT / "程序/MAGE/正式代码"
TRAIN_CODE = PROJECT_ROOT / "程序/模型训练/正式代码"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(MAGE_CODE))
sys.path.insert(0, str(TRAIN_CODE))

from build_mage_teacher_roi_manifest import file_sha256  # noqa: E402
from train_mage_mg1b_attention_teacher import (  # noqa: E402
    GRID_SIZE,
    AttentionPoolingTeacher,
)
from train_mage_mg1_teacher import (  # noqa: E402
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    seed_everything,
)
from train_utils import git_snapshot, json_ready  # noqa: E402
from sae_discovery import (  # noqa: E402
    ACTIVE_EPS,
    SparseAutoencoder,
    build_feature_summary,
    patient_active_counts,
    patient_class_weights,
    pruning_thresholds,
    reconstruct_masked_features,
    weighted_head_metrics,
)
from efficientnet_sae_discovery import (  # noqa: E402
    TopKSparseAutoencoder,
    classification_metrics,
    heatmap_overlay,
    load_font,
    module_sha,
    ncc90,
    patient_mean,
    project,
    train_sae,
)


SEEDS = (42, 202, 503)
INPUT_DIM = 1280

# ---- S0冻结血缘（2026-08-19核验写入，见SAE进度文档S0节） ----
CLONG_RUN = PROJECT_ROOT / (
    "结果/MAGE/MG2L训练轮数敏感性_20260818/正式验证集筛选/"
    "mg2l_armc_efficientnet_b0_seed42"
)
CLONG_CHECKPOINT = CLONG_RUN / "mg2_armc_best_student.pth"
CLONG_CONFIG = CLONG_RUN / "config.json"
CLONG_VAL_PREDICTIONS = CLONG_RUN / "val_image_predictions.csv"
CLONG_CHECKPOINT_SHA256 = (
    "29e76977251fbd50c9fd9ecd6ebd26927eaf9b54eeb8e70a0b70d3d5a70ce014"
)
MANIFEST = PROJECT_ROOT / (
    "数据整理记录/MAGE/MG0b_独立轴动态扩边ROI_v3_20260817/"
    "independent_axis_dynamic/mage_teacher_roi_manifest_independent_axis_dynamic_v3.csv"
)
MANIFEST_SHA256 = "b1dfd24505cbba876fd28502e36b0b190206de8f063562f47347dc7c67f32e33"
V3_AUDIT = PROJECT_ROOT / (
    "数据整理记录/MAGE/MG0b_独立轴动态扩边ROI_v3_20260817/"
    "mg0b_independent_axis_dynamic_roi_v3_audit.json"
)
V3_AUDIT_SHA256 = "94f78dc94fce9ca95692ef05d9c423313acb78dfdb7a431ab57603193095017a"
TEACHER_CHECKPOINT = PROJECT_ROOT / (
    "结果/MAGE/MG1b注意力池化教师_20260818/正式验证集筛选/"
    "mg1b_attention_efficientnet_b0_seed42/mg1b_best_teacher.pth"
)
TEACHER_CHECKPOINT_SHA256 = (
    "f2cd3b13cbbc0e8b4df64e9090ec50343314137ba71a4c07d9fae9faa8af48f9"
)
TEACHER_CACHE = PROJECT_ROOT / "结果/MAGE/MG2全图学生蒸馏_20260818/teacher_cache_mg1b_v3.pt"
TEACHER_CACHE_SHA256 = "89f3d329d68b182b33cad6f9a074e4cd919fb910bca96fff9d9ab16427d9e7db"
BETA_JSON = PROJECT_ROOT / "结果/MAGE/MG2全图学生蒸馏_20260818/beta_calibration_seed42.json"
BETA_JSON_SHA256 = "47179b6228122096c00e6f0e59ab5c45663b3f163a506c77aa8811320803ac7f"

# S0冻结阈值，直接来自C-long正式config，禁止从val重新扫描。
FROZEN_IMAGE_THRESHOLD = 0.25923898816108704
FROZEN_PATIENT_THRESHOLD = 0.3074711561203003
# 缓存特征复算val概率/attention与正式CSV允许的最大绝对差。
RECOMPUTE_TOLERANCE = 1e-4

# S2预注册矩阵（2026-08-19冻结）。
FORMAL_WIDTHS = (512, 1280, 2560, 5120, 10240)
FORMAL_LAMBDAS = (2e-4, 5e-4, 1e-3)
FORMAL_GAMMA = 0.1
DIAGNOSTIC_NO_MARGIN = {"hidden_dim": 10240, "lambda_l1": 5e-4, "seed": 42}
TOPK_WIDTH = 10240
TOPK_K = 1024

# 四项正式冻结成功门槛 + 死亡/重复上限。
GATE_MAX_PATIENT_AUC_DROP = 0.01
GATE_MIN_PATIENT_AGREEMENT = 0.95
GATE_MIN_COSINE = 0.90
GATE_MIN_RECOVERED_CE = 0.95
GATE_MAX_DEAD_RATE = 0.10
GATE_MAX_DUPLICATE_RATE = 0.10
DUPLICATE_COSINE_THRESHOLD = 0.95

OUTPUT_ROOT = PROJECT_ROOT / "结果/SAE/CLong文献重构_20260819"

EXPECTED_COUNTS = {
    "train": {"images": 2350, "patients": 1212},
    "val": {"images": 497, "patients": 260},
}


def parse_args() -> argparse.Namespace:
    """解析宽度、稀疏机制、训练和debug参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument(
        "--activation-mode", choices=("relu_l1", "topk"), default="relu_l1",
        help="relu_l1使用L1软稀疏；topk为每个样本保留固定数量的最大非负激活。",
    )
    parser.add_argument("--top-k", type=int, help="topk模式下每张图保留的Feature数。")
    parser.add_argument("--lambda-l1", type=float, default=5e-4)
    parser.add_argument(
        "--margin-loss-weight", type=float, default=FORMAL_GAMMA,
        help="无标签分类margin保真损失权重；正式主矩阵固定0.1。",
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
    parser.add_argument("--experiment")
    parser.add_argument(
        "--feature-cache-from", type=Path,
        help="复用已核验血缘的特征缓存目录，仍在新目录保存独立副本。",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-patients-per-class", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _close(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def validate_formal_args(args: argparse.Namespace) -> None:
    """非debug运行必须落在S2预注册冻结的矩阵内。"""
    if args.activation_mode == "topk":
        if args.top_k is None or not 0 < args.top_k <= args.hidden_dim:
            raise ValueError("topk模式要求0 < top_k <= hidden_dim")
        if not _close(args.lambda_l1, 0.0):
            raise ValueError("topk模式固定lambda_l1=0，避免同时改变两种稀疏机制")
    elif args.top_k is not None:
        raise ValueError("top_k只用于topk模式")
    if args.margin_loss_weight < 0:
        raise ValueError("margin_loss_weight不能为负数")
    if args.debug:
        return
    if args.activation_mode == "topk":
        if args.hidden_dim != TOPK_WIDTH or args.top_k != TOPK_K:
            raise ValueError("正式Top-K备选固定为 hidden=10240, K=1024")
        if not _close(args.margin_loss_weight, FORMAL_GAMMA):
            raise ValueError("正式Top-K备选固定 gamma=0.1")
        return
    if args.hidden_dim not in FORMAL_WIDTHS:
        raise ValueError("正式L1 hidden_dim不在冻结五档网格{512,1280,2560,5120,10240}中")
    if not any(_close(args.lambda_l1, value) for value in FORMAL_LAMBDAS):
        raise ValueError("正式L1 lambda不在冻结网格{2e-4,5e-4,1e-3}中")
    if _close(args.margin_loss_weight, FORMAL_GAMMA):
        return
    if _close(args.margin_loss_weight, 0.0):
        diagnostic = (
            args.hidden_dim == DIAGNOSTIC_NO_MARGIN["hidden_dim"]
            and _close(args.lambda_l1, DIAGNOSTIC_NO_MARGIN["lambda_l1"])
            and args.seed == DIAGNOSTIC_NO_MARGIN["seed"]
        )
        if diagnostic:
            return
        raise ValueError("无margin诊断只允许 w10240/lambda=5e-4/seed42 一组")
    raise ValueError("正式运行gamma只能为0.1（主矩阵）或0（唯一无margin诊断组）")


def verify_s0_lineage() -> dict:
    """核验S0六项血缘SHA与C-long config关键字段，任一不符立即终止。"""
    expected = {
        "student_checkpoint": (CLONG_CHECKPOINT, CLONG_CHECKPOINT_SHA256),
        "manifest": (MANIFEST, MANIFEST_SHA256),
        "v3_audit": (V3_AUDIT, V3_AUDIT_SHA256),
        "teacher_checkpoint": (TEACHER_CHECKPOINT, TEACHER_CHECKPOINT_SHA256),
        "teacher_cache": (TEACHER_CACHE, TEACHER_CACHE_SHA256),
        "beta_calibration_json": (BETA_JSON, BETA_JSON_SHA256),
    }
    verified = {}
    for name, (path, sha) in expected.items():
        if not path.is_file():
            raise FileNotFoundError(f"S0资产缺失: {path}")
        actual = file_sha256(path)
        if actual != sha:
            raise ValueError(f"S0血缘核验失败: {name} SHA={actual}，期望{sha}")
        verified[name] = {"path": str(path), "sha256": sha}
    config = json.loads(CLONG_CONFIG.read_text(encoding="utf-8"))
    checks = {
        "arm": config.get("arm") == "C",
        "seed": int(config.get("seed", -1)) == 42,
        "debug": config.get("debug") is False,
        "checkpoint_sha256": config.get("checkpoint_sha256") == CLONG_CHECKPOINT_SHA256,
        "manifest_sha256": config.get("manifest_sha256") == MANIFEST_SHA256,
        "teacher_checkpoint_sha256": (
            config.get("teacher_checkpoint_sha256") == TEACHER_CHECKPOINT_SHA256
        ),
        "teacher_cache_sha256": config.get("teacher_cache_sha256") == TEACHER_CACHE_SHA256,
        "architecture": (
            config.get("architecture", {}).get("pooling")
            == "attention-weighted sum only; no GAP bypass"
        ),
        "image_threshold": _close(
            float(config["metrics"]["image_threshold_metrics"]["threshold"]),
            FROZEN_IMAGE_THRESHOLD,
        ),
        "patient_threshold": _close(
            float(config["metrics"]["patient_threshold_metrics"]["threshold"]),
            FROZEN_PATIENT_THRESHOLD,
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"C-long config与S0冻结值不一致: {failed}")
    verified["clong_config"] = {"path": str(CLONG_CONFIG), "checks": sorted(checks)}
    print("S0血缘核验通过：六项SHA与C-long config关键字段全部一致")
    return verified


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


def load_manifest_frame(args: argparse.Namespace) -> pd.DataFrame:
    """加载v3清单的train/val并落实完整性检查与正式计数。"""
    frame = pd.read_csv(MANIFEST, encoding="utf-8-sig", dtype={"patient_id": str})
    frame = frame.loc[frame.split.isin(["train", "val"])].reset_index(drop=True)
    required = {"image_relpath", "relative_path", "patient_id", "label", "split",
                "source", "center", "sha256"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"清单缺少字段: {sorted(missing)}")
    if frame.groupby("patient_id").split.nunique().gt(1).any():
        raise ValueError("清单存在患者跨split")
    if frame.groupby("patient_id").label.nunique().gt(1).any():
        raise ValueError("清单存在患者标签冲突")
    for split in ("train", "val"):
        if set(frame.loc[frame.split.eq(split), "label"]) != {0, 1}:
            raise ValueError(f"{split}未同时包含两类")
    frame["domain"] = np.where(frame.center.eq("武大省人民"), "省人民", "外院")
    frame["hospital"] = frame.center.astype(str)
    if args.debug:
        return select_debug_patients(frame, args.debug_patients_per_class, args.seed)
    for split, expected in EXPECTED_COUNTS.items():
        part = frame.loc[frame.split.eq(split)]
        actual = {"images": len(part), "patients": int(part.patient_id.nunique())}
        if actual != expected:
            raise ValueError(f"{split}计数{actual}与S0冻结值{expected}不一致")
    return frame


def load_clong_model(device: torch.device) -> nn.Module:
    """恢复冻结C-long学生，并拒绝参数训练。"""
    payload = torch.load(CLONG_CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("architecture") != "efficientnet_b0_attention_pooling_7x7":
        raise ValueError("checkpoint不是attention-pooling 7x7架构")
    model = AttentionPoolingTeacher(pretrained=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
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


class CLongFeatureDataset(Dataset):
    """以C-long eval预处理读取完整图：直接resize到224x224，无翻转无光度增强。"""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def load_pil(self, index: int) -> Image.Image:
        row = self.frame.iloc[index]
        with Image.open(PROJECT_ROOT / row.image_relpath) as source:
            image = source.convert("RGB")
        return image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.load_pil(index)
        tensor = TF.normalize(
            TF.pil_to_tensor(image).float().div(255.0), IMAGENET_MEAN, IMAGENET_STD
        )
        return tensor, index


@torch.no_grad()
def extract_features(
    model: nn.Module, capture: ActivationCapture, frame: pd.DataFrame,
    args: argparse.Namespace, device: torch.device, cache: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """逐split提取attention-pooled特征、7x7 attention和复算概率。"""
    features: dict[str, np.ndarray] = {}
    attentions: dict[str, np.ndarray] = {}
    metadata: dict[str, pd.DataFrame] = {}
    classifier = model.classifier
    for split in ("train", "val"):
        split_frame = frame.loc[frame.split.eq(split)].reset_index(drop=True)
        loader = DataLoader(
            CLongFeatureDataset(split_frame), batch_size=args.image_batch_size,
            shuffle=False, num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        pooled_batches, attention_batches, probability_batches = [], [], []
        for batch_index, (images, _) in enumerate(loader, 1):
            logits, attention = model(images.to(device, non_blocking=True))
            fmap = capture.output
            if fmap is None or fmap.shape[1:] != (INPUT_DIM, GRID_SIZE, GRID_SIZE):
                raise RuntimeError("捕获层形状不是[1280,7,7]")
            pooled = (fmap * attention).sum(dim=(2, 3))
            if not torch.allclose(classifier(pooled), logits, atol=1e-5):
                raise RuntimeError("attention-pooled向量经分类头不等于模型logits")
            pooled_batches.append(pooled.cpu().numpy())
            attention_batches.append(attention.flatten(1).cpu().numpy())
            probability_batches.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            print(f"提取 {split}: [{batch_index}/{len(loader)}]")
        array = np.concatenate(pooled_batches).astype(np.float32)
        attention_array = np.concatenate(attention_batches).astype(np.float32)
        split_frame["cancer_probability"] = np.concatenate(probability_batches)
        np.save(cache / f"{split}_pooled_features.npy", array)
        np.save(cache / f"{split}_attention_maps.npy", attention_array)
        split_frame.to_csv(cache / f"{split}_metadata.csv", index=False, encoding="utf-8-sig")
        features[split] = array
        attentions[split] = attention_array
        metadata[split] = split_frame
        print(f"{split}: {len(split_frame)}张，特征={array.shape}，attention={attention_array.shape}")
    return features, attentions, metadata


def verify_recompute_against_official(
    val_metadata: pd.DataFrame, val_attention: np.ndarray,
) -> dict:
    """缓存特征复算的val概率与attention必须与正式val_image_predictions.csv一致。"""
    official = pd.read_csv(
        CLONG_VAL_PREDICTIONS, encoding="utf-8-sig", dtype={"patient_id": str},
    )
    if len(official) != len(val_metadata):
        raise ValueError("正式val预测行数与清单val不一致")
    if official["relative_path"].tolist() != val_metadata["relative_path"].tolist():
        raise ValueError("正式val预测顺序与清单val顺序不一致，拒绝比较")
    probability_delta = np.abs(
        official["cancer_probability"].to_numpy(float)
        - val_metadata["cancer_probability"].to_numpy(float)
    )
    attention_columns = [
        f"attention_{row}_{col}" for row in range(GRID_SIZE) for col in range(GRID_SIZE)
    ]
    attention_delta = np.abs(
        official[attention_columns].to_numpy(float) - val_attention.astype(float)
    )
    result = {
        "probability_max_abs_delta": float(probability_delta.max()),
        "attention_max_abs_delta": float(attention_delta.max()),
        "tolerance": RECOMPUTE_TOLERANCE,
        "passed": bool(
            probability_delta.max() <= RECOMPUTE_TOLERANCE
            and attention_delta.max() <= RECOMPUTE_TOLERANCE
        ),
    }
    if not result["passed"]:
        raise ValueError(f"缓存特征复算与正式val预测不一致: {result}")
    print(
        f"复算自测通过：val概率最大差={result['probability_max_abs_delta']:.3e}，"
        f"attention最大差={result['attention_max_abs_delta']:.3e}"
    )
    return result


def reuse_feature_cache(
    source: Path, target: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """校验缓存血缘后复用attention-pooled特征，避免矩阵内重复跑Encoder。"""
    lineage_path = source / "cache_config.json"
    if not lineage_path.is_file():
        raise FileNotFoundError(f"缓存缺少血缘记录: {lineage_path}")
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    expected = {
        "student_checkpoint_sha256": CLONG_CHECKPOINT_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "feature_layer": FEATURE_LAYER_DESCRIPTION,
    }
    for key, value in expected.items():
        if lineage.get(key) != value:
            raise ValueError(f"复用特征缓存的{key}与S0冻结值不一致")
    if not lineage.get("recompute_self_test", {}).get("passed"):
        raise ValueError("源缓存未通过复算自测，拒绝复用")
    features, attentions, metadata = {}, {}, {}
    for split in ("train", "val"):
        for name in (f"{split}_pooled_features.npy", f"{split}_attention_maps.npy"):
            shutil.copy2(source / name, target / name)
        feature_path = source / f"{split}_pooled_features.npy"
        attention_path = source / f"{split}_attention_maps.npy"
        metadata_path = source / f"{split}_metadata.csv"
        features[split] = np.load(feature_path).astype(np.float32, copy=False)
        attentions[split] = np.load(attention_path).astype(np.float32, copy=False)
        metadata[split] = pd.read_csv(
            metadata_path, encoding="utf-8-sig", dtype={"patient_id": str},
            float_precision="round_trip",
        )
        shutil.copy2(metadata_path, target / metadata_path.name)
    print(f"复用冻结特征缓存: {source}")
    return features, attentions, metadata


FEATURE_LAYER_DESCRIPTION = "features[8] x attention -> attention-pooled 1280"


def annotate_frozen_thresholds(metadata: dict[str, pd.DataFrame]) -> None:
    """用S0冻结阈值标记TP/TN/FP/FN；不扫描任何新阈值。"""
    for frame in metadata.values():
        predicted = frame.cancer_probability.ge(FROZEN_IMAGE_THRESHOLD).astype(int)
        frame["pred_label"] = predicted
        frame["confusion_type"] = np.select(
            [
                frame.label.eq(1) & predicted.eq(1), frame.label.eq(0) & predicted.eq(0),
                frame.label.eq(0) & predicted.eq(1), frame.label.eq(1) & predicted.eq(0),
            ], ["TP", "TN", "FP", "FN"], default="",
        )


def duplicate_decoder_rate_nondead(
    weights: np.ndarray, nondead_mask: np.ndarray, threshold: float = DUPLICATE_COSINE_THRESHOLD,
) -> dict:
    """仅在非死亡Feature中统计decoder绝对余弦重复率（新协议口径）。

    旧实现对全部decoder方向计算；正式新协议要求显式传入非死亡掩码。
    重复定义：与另一个非死亡decoder方向的绝对余弦 >= threshold，每个Feature最多计一次。
    """
    selected = weights[nondead_mask]
    if len(selected) == 0:
        return {"nondead_feature_count": 0, "duplicate_feature_count": 0,
                "duplicate_rate": 0.0}
    normalized = selected / np.maximum(
        np.linalg.norm(selected, axis=1, keepdims=True), 1e-12
    )
    duplicated = np.zeros(len(selected), dtype=bool)
    block = 512
    for start in range(0, len(selected), block):
        similarities = np.abs(normalized[start:start + block] @ normalized.T)
        rows = np.arange(start, min(start + block, len(selected)))
        similarities[np.arange(len(rows)), rows] = 0
        duplicated[start:start + len(rows)] = similarities.max(1) >= threshold
    return {
        "nondead_feature_count": int(len(selected)),
        "duplicate_feature_count": int(duplicated.sum()),
        "duplicate_rate": float(duplicated.mean()),
    }


def select_pruning_threshold(curve: list[dict], ce_tolerance: float) -> int | None:
    """返回满足recovered CE容差的最大train激活患者数阈值；无合格方案返回None。

    纯函数，便于单元测试。调用方必须保证curve按阈值升序排列。
    """
    selected = None
    for row in curve:
        if row["kept_feature_count"] > 0 and row["recovered_ce_drop"] <= ce_tolerance:
            selected = row["active_patient_threshold"]
    return selected


def run_pruning_strict(
    sae: SparseAutoencoder, train_activations: np.ndarray, train_metadata: pd.DataFrame,
    val_features: np.ndarray, val_activations: np.ndarray, val_metadata: pd.DataFrame,
    classifier_weight: np.ndarray, classifier_bias: np.ndarray,
    hidden_dim: int, min_active_patients: int, ce_tolerance: float,
    batch_size: int, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """train生成候选阈值，val选择容差内最大阈值；无合格方案保留全部Feature。

    与旧``run_feature_pruning``的关键差异：不存在违反容差的fallback。
    若没有任何非空剪枝方案满足`val recovered CE drop <= ce_tolerance`，
    则保留全部Feature并标记``pruning_applied=False``。
    """
    active_counts = patient_active_counts(train_activations, train_metadata)
    all_mask = np.ones(hidden_dim, dtype=bool)
    baseline_reconstructed = reconstruct_masked_features(
        sae, val_activations, all_mask, batch_size, device,
    )
    baseline_metrics = weighted_head_metrics(
        val_features, baseline_reconstructed, val_metadata,
        classifier_weight, classifier_bias,
    )
    curve: list[dict] = []
    for threshold in pruning_thresholds(active_counts, min_active_patients):
        kept_mask = active_counts > threshold
        kept_count = int(kept_mask.sum())
        if kept_count == 0:
            continue
        reconstructed = reconstruct_masked_features(
            sae, val_activations, kept_mask, batch_size, device,
        )
        current = weighted_head_metrics(
            val_features, reconstructed, val_metadata, classifier_weight, classifier_bias,
        )
        curve.append({
            "active_patient_threshold": int(threshold),
            "kept_feature_count": kept_count,
            "pruned_feature_count": int(hidden_dim - kept_count),
            "recovered_cross_entropy": current["recovered_cross_entropy"],
            "recovered_ce_drop": float(
                baseline_metrics["recovered_cross_entropy"]
                - current["recovered_cross_entropy"]
            ),
            "reconstructed_accuracy": current["reconstructed_accuracy"],
            "prediction_agreement": current["prediction_agreement"],
            "cancer_probability_mae": current["cancer_probability_mae"],
        })
    selected_threshold = select_pruning_threshold(curve, ce_tolerance)
    if selected_threshold is None:
        kept_mask = all_mask
        applied = False
        selected_metrics = baseline_metrics
        selected_drop = 0.0
    else:
        kept_mask = active_counts > selected_threshold
        applied = True
        row = next(
            item for item in curve
            if item["active_patient_threshold"] == selected_threshold
        )
        reconstructed = reconstruct_masked_features(
            sae, val_activations, kept_mask, batch_size, device,
        )
        selected_metrics = weighted_head_metrics(
            val_features, reconstructed, val_metadata, classifier_weight, classifier_bias,
        )
        selected_drop = row["recovered_ce_drop"]
    kept_indices = np.flatnonzero(kept_mask)
    summary = {
        "total_feature_count": int(hidden_dim),
        "kept_feature_count": int(len(kept_indices)),
        "pruned_feature_count": int(hidden_dim - len(kept_indices)),
        "selected_active_patient_threshold": (
            int(selected_threshold) if selected_threshold is not None else None
        ),
        "minimum_active_patients": int(min_active_patients),
        "recovered_ce_tolerance": float(ce_tolerance),
        "baseline_recovered_cross_entropy": baseline_metrics["recovered_cross_entropy"],
        "selected_recovered_cross_entropy": selected_metrics["recovered_cross_entropy"],
        "recovered_ce_drop": float(selected_drop),
        "pruning_applied": bool(applied),
        "selection_within_tolerance": True,
        "no_fallback_note": (
            "无合格非空剪枝方案时保留全部Feature；不允许输出违反容差的剪枝产品"
        ),
        "selected_reconstructed_accuracy": selected_metrics["reconstructed_accuracy"],
        "selected_prediction_agreement": selected_metrics["prediction_agreement"],
        "selected_cancer_probability_mae": selected_metrics["cancer_probability_mae"],
    }
    return kept_indices, active_counts, curve, summary


@torch.no_grad()
def make_overviews(
    model: nn.Module, capture: ActivationCapture, sae: SparseAutoencoder,
    activations: np.ndarray, metadata: pd.DataFrame, summary: pd.DataFrame,
    args: argparse.Namespace, device: torch.device, output: Path,
) -> None:
    """为高分类贡献Feature保存 原图/attention/逐位置贡献 三联图。

    逐位置贡献定义为 ``w_p * (d_j . f_p)``：C-long attention权重乘以该位置
    特征向量在decoder方向上的投影，即位置p对Feature j重构方向的贡献。
    这与旧代码"encoder方向与feature map直接点积"不同，后者对
    attention-pooling学生不成立。
    """
    candidates = summary.loc[
        summary.kept_after_pruning & summary.active_patient_count.ge(2)
    ].nlargest(args.overview_features, "mean_abs_margin_contribution")
    dataset = CLongFeatureDataset(metadata)
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
        direction = sae.decoder_weight[feature_id].detach()
        canvas = Image.new("RGB", (960, 65 + 285 * len(selected)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 8),
            f"Feature {feature_id} 癌方向={feature.cancer_margin_direction:.4f}",
            fill="black", font=title_font,
        )
        for row_number, index in enumerate(selected):
            input_image = dataset.load_pil(index)
            logits, attention = model(
                TF.normalize(
                    TF.pil_to_tensor(input_image).float().div(255.0),
                    IMAGENET_MEAN, IMAGENET_STD,
                ).unsqueeze(0).to(device)
            )
            del logits
            fmap = capture.output[0]
            alignment = (fmap * direction[:, None, None]).sum(0)
            contribution = (attention[0, 0] * alignment).cpu().numpy()
            attention_map = attention[0, 0].cpu().numpy()
            contribution_overlay = heatmap_overlay(
                input_image, np.maximum(contribution, 0)
            )
            attention_overlay = heatmap_overlay(input_image, attention_map)
            y = 65 + row_number * 285
            canvas.paste(ImageOps.fit(input_image, (320, 240)), (0, y))
            canvas.paste(ImageOps.fit(attention_overlay, (320, 240)), (320, y))
            canvas.paste(ImageOps.fit(contribution_overlay, (320, 240)), (640, y))
            item = metadata.iloc[index]
            draw.text(
                (8, y + 245),
                f"{item.patient_id} label={item.label} {item.source} "
                f"激活={activations[index, feature_id]:.4f}",
                fill="black", font=text_font,
            )
            records.append({
                "feature_id": feature_id, "rank": row_number + 1,
                "image_relpath": item.image_relpath, "patient_id": item.patient_id,
                "label": int(item.label),
                "activation": float(activations[index, feature_id]),
                "contribution_min": float(contribution.min()),
                "contribution_max": float(contribution.max()),
            })
        canvas.save(output / f"feature_{feature_id:04d}.png")
    pd.DataFrame(records).to_csv(
        output / "top_examples.csv", index=False, encoding="utf-8-sig"
    )


def run_self_test() -> None:
    """验证形式校验、Top-K、NCC90、非死亡重复率和剪枝无fallback逻辑。"""
    center = torch.zeros(8)
    sae = SparseAutoencoder(8, 4, center)
    reconstructed, hidden = sae(torch.randn(3, 8))
    assert reconstructed.shape == (3, 8) and hidden.shape == (3, 4)
    topk_sae = TopKSparseAutoencoder(8, 6, center, top_k=2)
    reconstructed, hidden = topk_sae(torch.randn(3, 8))
    assert reconstructed.shape == (3, 8) and hidden.shape == (3, 6)
    assert bool(hidden.gt(0).sum(1).le(2).all())
    assert ncc90(np.array([[4.0, 3.0, 2.0, 1.0]], dtype=np.float32)) == 3.0

    # 非死亡掩码：死亡Feature即使方向重复也不得计入。
    weights = np.array([
        [1.0, 0.0], [1.0, 1e-9], [0.0, 1.0],
    ], dtype=np.float32)
    nondead = np.array([True, False, True])
    result = duplicate_decoder_rate_nondead(weights, nondead, threshold=0.95)
    assert result["nondead_feature_count"] == 2
    assert result["duplicate_feature_count"] == 0

    # 剪枝选择：取容差内最大阈值；全部不合格时返回None（保留全部Feature）。
    curve = [
        {"active_patient_threshold": 4, "kept_feature_count": 100, "recovered_ce_drop": 0.004},
        {"active_patient_threshold": 9, "kept_feature_count": 60, "recovered_ce_drop": 0.008},
        {"active_patient_threshold": 19, "kept_feature_count": 30, "recovered_ce_drop": 0.020},
    ]
    assert select_pruning_threshold(curve, 0.01) == 9
    failing = [dict(row, recovered_ce_drop=0.5) for row in curve]
    assert select_pruning_threshold(failing, 0.01) is None
    empty_kept = [dict(row, kept_feature_count=0) for row in curve]
    assert select_pruning_threshold(empty_kept, 0.01) is None
    print("C-long SAE self-test passed")


def main() -> None:
    """执行S0血缘核验、特征提取/复用、SAE训练、严格剪枝、统计与三联概览。"""
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.seed is None or args.hidden_dim is None or not args.experiment:
        raise ValueError("非self-test运行必须提供 --seed、--hidden-dim 和 --experiment")
    validate_formal_args(args)
    seed_everything(args.seed)
    lineage = verify_s0_lineage()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    output = args.output_root / args.experiment
    if output.exists():
        raise FileExistsError(f"SAE输出已存在，拒绝覆盖: {output}")
    cache, model_dir, pruning_dir, overview_dir = (
        output / "特征缓存", output / "SAE模型", output / "feature筛选", output / "feature概览",
    )
    for directory in (cache, model_dir, pruning_dir, overview_dir):
        directory.mkdir(parents=True, exist_ok=False)

    frame = load_manifest_frame(args)
    model = load_clong_model(device)
    frozen_sha_before = module_sha(model)
    capture = ActivationCapture(model.features[8])
    if args.feature_cache_from:
        split_features, split_attentions, split_metadata = reuse_feature_cache(
            args.feature_cache_from.resolve(), cache,
        )
        recompute_result = {"reused_cache": True, "passed": True}
    else:
        split_features, split_attentions, split_metadata = extract_features(
            model, capture, frame, args, device, cache,
        )
        if args.debug:
            # debug子集只有少量患者，无法与497行正式val预测对齐，跳过复算自测；
            # debug缓存永不进入正式矩阵（正式复用要求passed=True）。
            recompute_result = {"skipped_debug": True, "passed": False}
        else:
            recompute_result = verify_recompute_against_official(
                split_metadata["val"], split_attentions["val"],
            )
    annotate_frozen_thresholds(split_metadata)
    for split, current in split_metadata.items():
        current.to_csv(cache / f"{split}_metadata.csv", index=False, encoding="utf-8-sig")
    cache_lineage = {
        "student_checkpoint_sha256": CLONG_CHECKPOINT_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "feature_layer": FEATURE_LAYER_DESCRIPTION,
        "counts": {
            split: {"images": int(len(part)), "patients": int(part.patient_id.nunique())}
            for split, part in split_metadata.items()
        },
        "recompute_self_test": recompute_result,
    }
    (cache / "cache_config.json").write_text(
        json.dumps(json_ready(cache_lineage), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    classifier = model.classifier[1]
    classifier_weight = classifier.weight.detach().cpu().numpy()
    classifier_bias = classifier.bias.detach().cpu().numpy()
    classifier_margin = classifier_weight[1] - classifier_weight[0]
    sae, best = train_sae(
        split_features["train"], split_metadata["train"],
        split_features["val"], split_metadata["val"], args, device, model_dir,
        classifier_margin,
    )

    activations: dict[str, np.ndarray] = {}
    metrics = {"best_epoch": int(best["epoch"]), "splits": {}}
    for split in ("train", "val"):
        reconstructed, current_activations = project(
            sae, split_features[split], args.sae_batch_size, device,
        )
        activations[split] = current_activations
        np.savez_compressed(
            cache / f"{split}_sae_projection.npz",
            reconstructed=reconstructed, activations=current_activations,
        )
        current_metrics = classification_metrics(
            split_features[split], reconstructed, split_metadata[split],
            classifier_weight, classifier_bias, FROZEN_PATIENT_THRESHOLD,
        )
        current_metrics.update({
            "mean_l0": float((current_activations > ACTIVE_EPS).sum(1).mean()),
            "mean_ncc90": ncc90(current_activations),
            "dead_feature_count": int(
                ((current_activations > ACTIVE_EPS).sum(0) == 0).sum()
            ),
        })
        metrics["splits"][split] = current_metrics

    kept, active_counts, curve, pruning = run_pruning_strict(
        sae, activations["train"], split_metadata["train"],
        split_features["val"], activations["val"], split_metadata["val"],
        classifier_weight, classifier_bias, args.hidden_dim,
        args.pruning_min_active_patients, args.pruning_ce_tolerance,
        args.sae_batch_size, device,
    )
    kept_mask = np.zeros(args.hidden_dim, dtype=bool)
    kept_mask[kept] = True
    dead_mask = (activations["train"] > ACTIVE_EPS).sum(0) == 0
    pruning["decoder_duplicate_nondead"] = duplicate_decoder_rate_nondead(
        sae.decoder_weight.detach().cpu().numpy(), ~dead_mask,
    )
    pruning["dead_feature_rate_train"] = float(dead_mask.mean())
    metrics["pruning"] = pruning

    val_metrics = metrics["splits"]["val"]
    dead_rate = float(dead_mask.mean())
    duplicate_rate = pruning["decoder_duplicate_nondead"]["duplicate_rate"]
    metrics["core_fidelity_gate"] = {
        "maximum_patient_auc_drop": GATE_MAX_PATIENT_AUC_DROP,
        "minimum_patient_prediction_agreement": GATE_MIN_PATIENT_AGREEMENT,
        "minimum_mean_cosine": GATE_MIN_COSINE,
        "minimum_recovered_cross_entropy": GATE_MIN_RECOVERED_CE,
        "maximum_dead_feature_rate": GATE_MAX_DEAD_RATE,
        "maximum_decoder_duplicate_rate": GATE_MAX_DUPLICATE_RATE,
        "patient_auc_drop": float(
            val_metrics["patient_original_auc"] - val_metrics["patient_reconstructed_auc"]
        ),
        "passed": bool(
            val_metrics["patient_original_auc"] - val_metrics["patient_reconstructed_auc"]
            <= GATE_MAX_PATIENT_AUC_DROP
            and val_metrics["patient_prediction_agreement_at_locked_threshold"]
            >= GATE_MIN_PATIENT_AGREEMENT
            and val_metrics["mean_cosine"] >= GATE_MIN_COSINE
            and val_metrics["recovered_cross_entropy"] >= GATE_MIN_RECOVERED_CE
            and dead_rate <= GATE_MAX_DEAD_RATE
            and duplicate_rate <= GATE_MAX_DUPLICATE_RATE
        ),
    }
    pd.DataFrame({
        "feature_id": np.arange(args.hidden_dim),
        "active_patient_count_train": active_counts,
        "kept_after_pruning": kept_mask,
    }).to_csv(
        pruning_dir / "feature_pruning_decisions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(curve).to_csv(
        pruning_dir / "pruning_curve.csv", index=False, encoding="utf-8-sig"
    )
    (pruning_dir / "pruning_summary.json").write_text(
        json.dumps(json_ready(pruning), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    development_activations = np.concatenate([activations["train"], activations["val"]])
    development_metadata = pd.concat(
        [split_metadata["train"], split_metadata["val"]], ignore_index=True
    )
    summary = build_feature_summary(
        development_activations, development_metadata, sae, classifier_weight,
    )
    summary["kept_after_pruning"] = kept_mask
    summary["pruning_active_patient_count_train"] = active_counts
    summary.to_csv(output / "feature_summary.csv", index=False, encoding="utf-8-sig")
    (output / "metrics.json").write_text(
        json.dumps(json_ready(metrics), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    make_overviews(
        model, capture, sae, development_activations, development_metadata, summary,
        args, device, overview_dir,
    )
    capture.close()
    if module_sha(model) != frozen_sha_before:
        raise RuntimeError("冻结C-long参数或BN统计在SAE流程中发生变化")

    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config = {
        **serialized_args, "output": str(output.resolve()), "input_dim": INPUT_DIM,
        "s0_lineage": lineage,
        "feature_layer": FEATURE_LAYER_DESCRIPTION,
        "margin_train_std": best.get("margin_train_std"),
        "image_threshold_frozen": FROZEN_IMAGE_THRESHOLD,
        "patient_threshold_frozen": FROZEN_PATIENT_THRESHOLD,
        "thresholds_source": "S0冻结自C-long正式config，本流程未扫描任何新阈值",
        "recompute_self_test": recompute_result,
        "counts": cache_lineage["counts"],
        "test_evaluated": False, "internal_test_evaluated": False,
        "external_evaluated": False,
        **git_snapshot(),
    }
    (output / "config.json").write_text(
        json.dumps(json_ready(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"最佳epoch={best['epoch']}；保留Feature={len(kept)}/{args.hidden_dim}")
    print(f"输出目录: {output}")


if __name__ == "__main__":
    main()
