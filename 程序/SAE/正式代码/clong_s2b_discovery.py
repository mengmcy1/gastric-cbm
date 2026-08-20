#!/usr/bin/env python3
"""C-long S2b 结构重构 SAE 正式入口。

实验臂：

* S4-B：attention-pooled 1280维 + BatchTopK；
* S4-C：7x7x1280 patch + 逐位置 Top-K；
* S4-D：7x7x1280 patch + BatchTopK；
* S4-N：pooled逐样本归一化 + Top-K，仅诊断。

S4-C/D 正式评价会从重构的 7x7 特征图重算注意力，不复用原始
attention。协议冻结于 ``SAE实验进度与结果讨论.md`` S2b（2026-08-20）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "程序/MAGE/正式代码"))
sys.path.insert(0, str(PROJECT_ROOT / "程序/模型训练/正式代码"))

from build_mage_teacher_roi_manifest import file_sha256  # noqa: E402
from clong_sae_discovery import (  # noqa: E402
    CLONG_CHECKPOINT_SHA256,
    FROZEN_IMAGE_THRESHOLD,
    FROZEN_PATIENT_THRESHOLD,
    INPUT_DIM,
    MANIFEST_SHA256,
    RECOMPUTE_TOLERANCE,
    ActivationCapture,
    CLongFeatureDataset,
    classification_metrics,
    confusion_at_threshold,
    duplicate_decoder_rate_nondead,
    load_clong_model,
    load_manifest_frame,
    patient_class_weights,
    verify_s0_lineage,
)
from clong_s2b_core import (  # noqa: E402
    ACTIVE_EPS,
    StructuredSparseAutoencoder,
    array_sequence_sha,
    attention_drift_metrics,
    attention_from_features,
    cell_overlap_map,
    normalized_margin_mse,
    patch_position_weights,
    pooled_from_features,
    project_decoder_gradient,
    solve_threshold_from_chunks,
    spatial_alignment_metrics,
)
from train_utils import git_snapshot, json_ready  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "结果/SAE/CLong_S2b结构重构_20260820"
CACHE_ROOT = OUTPUT_ROOT / "frozen_spatial_cache"
GAMMA_JSON = OUTPUT_ROOT / "gamma_pool_calibration_seed42.json"
GRID_SIZE = 7
HIDDEN_DIM = 10240
POOLED_K = 1024
PATCH_K = 128
GAMMA_MARGIN = 0.1
FORMAL_BUDGET = {
    "learning_rate": 1e-4,
    "epochs": 1000,
    "patience": 50,
    "warmup_fraction": 0.05,
    "image_batch_size": 32,
}


def parse_args() -> argparse.Namespace:
    """解析结构臂、训练预算与校准模式。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("B", "C", "D", "N"), default="C")
    parser.add_argument("--seed", type=int, choices=(42, 202, 503), default=42)
    parser.add_argument("--experiment")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--calibrate-gamma", action="store_true")
    parser.add_argument(
        "--postprocess-existing", action="store_true",
        help="仅加载已有checkpoint执行BatchTopK阈值与评价，不重训。",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-patients-per-class", type=int, default=2)
    parser.add_argument("--debug-hidden-dim", type=int, default=64)
    parser.add_argument("--debug-k", type=int, default=8)
    return parser.parse_args()


def formal_name(arm: str, seed: int) -> str:
    """返回预注册臂与seed的唯一正式实验名。"""
    return f"s4{arm.lower()}_clong_seed{seed}"


def validate_args(args: argparse.Namespace) -> None:
    """正式运行锁死S2b预算和实验名；debug与正式目录强制隔离。"""
    if args.debug:
        if args.experiment == formal_name(args.arm, args.seed):
            raise ValueError("debug禁止使用正式实验名")
        args.output_root = (
            OUTPUT_ROOT / "debug" /
            f"seed{args.seed}_p{args.debug_patients_per_class}"
        )
        return
    if args.postprocess_existing and args.arm not in {"B", "D"}:
        raise ValueError("--postprocess-existing只用于BatchTopK臂B/D")
    if args.experiment != formal_name(args.arm, args.seed):
        raise ValueError(f"正式实验名必须是{formal_name(args.arm, args.seed)}")
    if args.device != "cuda":
        raise ValueError("正式S2b必须显式使用--device cuda，保持与C-long复算路径一致")
    if args.arm in {"B", "N"} and args.seed != 42:
        raise ValueError("S4-B/N只在seed42运行")
    for field, expected in FORMAL_BUDGET.items():
        if not math.isclose(float(getattr(args, field)), expected, abs_tol=1e-12):
            raise ValueError(f"正式{field}必须等于{expected}")


def choose_device(name: str) -> torch.device:
    """解析运行设备，显式CUDA不可用时快速失败。"""
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("要求CUDA但PyTorch不可用")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else name)


def prepare_output(args: argparse.Namespace) -> Path:
    """新建单次实验目录，已存在时拒绝覆盖。"""
    output = args.output_root / (args.experiment or f"debug_s4{args.arm.lower()}_seed{args.seed}")
    if output.exists():
        raise FileExistsError(f"输出已存在，禁止覆盖: {output}")
    output.mkdir(parents=True)
    return output


def load_existing_training_product(
    args: argparse.Namespace, sae: StructuredSparseAutoencoder,
    device: torch.device,
) -> tuple[Path, StructuredSparseAutoencoder, dict]:
    """加载训练已完成但后处理中断的正式BatchTopK产物。"""
    output = args.output_root / args.experiment
    checkpoint_path = output / "sae_best.pth"
    history_path = output / "training_history.csv"
    if not checkpoint_path.is_file() or not history_path.is_file():
        raise FileNotFoundError("后处理恢复要求sae_best.pth和training_history.csv均存在")
    if (output / "config.json").exists():
        raise FileExistsError("完整config.json已存在，禁止重复后处理")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    expected = {
        "arm": args.arm, "seed": args.seed,
        "hidden_dim": sae.hidden_dim, "target_k": sae.target_k,
        "activation_mode": sae.activation_mode,
    }
    failed = [key for key, value in expected.items() if checkpoint.get(key) != value]
    if failed:
        raise ValueError(f"已有checkpoint与当前正式参数不一致: {failed}")
    sae.load_state_dict(checkpoint["sae_state_dict"], strict=True)
    return output, sae, checkpoint


def cache_paths(split: str) -> dict[str, Path]:
    """返回指定split的空间特征缓存文件集。"""
    return {
        "spatial": CACHE_ROOT / f"{split}_spatial_features.npy",
        "pooled": CACHE_ROOT / f"{split}_pooled_features.npy",
        "attention": CACHE_ROOT / f"{split}_attention.npy",
        "metadata": CACHE_ROOT / f"{split}_metadata.csv",
    }


@torch.no_grad()
def build_spatial_cache(
    model: torch.nn.Module, frame: pd.DataFrame, args: argparse.Namespace,
    device: torch.device, lineage: dict,
) -> None:
    """提取C-long冻结 ``[N,49,1280]`` 特征并做概率/attention复算自测。"""
    if CACHE_ROOT.exists():
        raise FileExistsError(f"空间缓存目录已存在: {CACHE_ROOT}")
    CACHE_ROOT.mkdir(parents=True)
    capture = ActivationCapture(model.features[8])
    files = {}
    try:
        for split in ("train", "val"):
            split_frame = frame.loc[frame.split.eq(split)].reset_index(drop=True)
            loader = DataLoader(
                CLongFeatureDataset(split_frame), batch_size=args.image_batch_size,
                shuffle=False, num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            spatial, pooled, attention, probabilities = [], [], [], []
            for batch_index, (images, _) in enumerate(loader, 1):
                logits, current_attention = model(images.to(device, non_blocking=True))
                fmap = capture.output
                if fmap is None or tuple(fmap.shape[1:]) != (INPUT_DIM, 7, 7):
                    raise RuntimeError("空间特征不是[1280,7,7]")
                current_spatial = fmap.flatten(2).transpose(1, 2)
                current_pooled = pooled_from_features(
                    current_spatial, current_attention.flatten(1)
                )
                if not torch.allclose(model.classifier(current_pooled), logits, atol=1e-5):
                    raise RuntimeError("缓存pooled重算不等于C-long logits")
                spatial.append(current_spatial.cpu().numpy())
                pooled.append(current_pooled.cpu().numpy())
                attention.append(current_attention.flatten(1).cpu().numpy())
                probabilities.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
                print(f"提取空间特征 {split} [{batch_index}/{len(loader)}]")
            arrays = {
                "spatial": np.concatenate(spatial).astype(np.float32),
                "pooled": np.concatenate(pooled).astype(np.float32),
                "attention": np.concatenate(attention).astype(np.float32),
            }
            split_frame["cancer_probability"] = np.concatenate(probabilities)
            paths = cache_paths(split)
            for key, array in arrays.items():
                np.save(paths[key], array)
            split_frame.to_csv(paths["metadata"], index=False, encoding="utf-8-sig")
            for path in paths.values():
                files[path.name] = file_sha256(path)

        official = pd.read_csv(
            PROJECT_ROOT / (
                "结果/MAGE/MG2L训练轮数敏感性_20260818/正式验证集筛选/"
                "mg2l_armc_efficientnet_b0_seed42/val_image_predictions.csv"
            ), encoding="utf-8-sig",
        )
        val_meta = pd.read_csv(cache_paths("val")["metadata"], encoding="utf-8-sig")
        val_attention = np.load(cache_paths("val")["attention"])
        attention_columns = [f"attention_{y}_{x}" for y in range(7) for x in range(7)]
        if args.debug:
            if official.relative_path.duplicated().any():
                raise RuntimeError("正式val预测relative_path不唯一")
            official = official.set_index("relative_path").loc[
                val_meta.relative_path.tolist()
            ].reset_index()
        if official.relative_path.tolist() != val_meta.relative_path.tolist():
            raise RuntimeError("缓存val行顺序与正式预测不一致")
        probability_delta = np.max(np.abs(
            official.cancer_probability.to_numpy() - val_meta.cancer_probability.to_numpy()
        ))
        attention_delta = np.max(np.abs(
            official[attention_columns].to_numpy() - val_attention
        ))
        tolerance = 2e-3 if args.debug and device.type == "cpu" else RECOMPUTE_TOLERANCE
        if max(probability_delta, attention_delta) > tolerance:
            raise RuntimeError(
                "空间缓存复算自测超容差: "
                f"probability={probability_delta:.3e}, attention={attention_delta:.3e}, "
                f"tolerance={tolerance:.1e}"
            )
        config = {
            "student_checkpoint_sha256": CLONG_CHECKPOINT_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "lineage": lineage,
            "files": files,
            "counts": {split: int(len(frame.loc[frame.split.eq(split)]))
                       for split in ("train", "val")},
            "shapes": {split: {
                "spatial": list(np.load(cache_paths(split)["spatial"], mmap_mode="r").shape),
                "pooled": list(np.load(cache_paths(split)["pooled"], mmap_mode="r").shape),
            } for split in ("train", "val")},
            "recompute_self_test": {
                "probability_max_abs_delta": float(probability_delta),
                "attention_max_abs_delta": float(attention_delta),
                "tolerance": tolerance,
                "debug_cpu_relaxed_tolerance": bool(
                    args.debug and device.type == "cpu"
                ),
                "passed": True,
            },
        }
        (CACHE_ROOT / "cache_config.json").write_text(
            json.dumps(json_ready(config), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    finally:
        capture.close()


def load_cache(frame: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    """逐文件SHA、shape和relative_path顺序校验空间缓存。"""
    config_path = CACHE_ROOT / "cache_config.json"
    if not config_path.is_file():
        raise FileNotFoundError("空间缓存未构建")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["student_checkpoint_sha256"] != CLONG_CHECKPOINT_SHA256:
        raise ValueError("缓存学生checkpoint SHA不一致")
    if config["manifest_sha256"] != MANIFEST_SHA256:
        raise ValueError("缓存manifest SHA不一致")
    spatial, pooled, attention, metadata = {}, {}, {}, {}
    for split in ("train", "val"):
        paths = cache_paths(split)
        for path in paths.values():
            if config["files"].get(path.name) != file_sha256(path):
                raise ValueError(f"缓存文件SHA不一致: {path.name}")
        metadata[split] = pd.read_csv(
            paths["metadata"], encoding="utf-8-sig", dtype={"patient_id": str},
            float_precision="round_trip",
        )
        expected = frame.loc[frame.split.eq(split), "relative_path"].tolist()
        if metadata[split].relative_path.tolist() != expected:
            raise ValueError(f"{split}缓存行顺序不一致")
        spatial[split] = np.load(paths["spatial"], mmap_mode="r")
        pooled[split] = np.load(paths["pooled"], mmap_mode="r")
        attention[split] = np.load(paths["attention"], mmap_mode="r")
    return spatial, pooled, attention, metadata


def make_sampler(metadata: pd.DataFrame, seed: int) -> WeightedRandomSampler:
    """生成类别平衡且同类内每患者总权重相等的采样器。"""
    weights = patient_class_weights(metadata)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), len(metadata), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def classifier_components(model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回冻结分类头权重、偏置与癌减非癌margin方向。"""
    linear = model.classifier[1]
    margin = (linear.weight[1] - linear.weight[0]).detach()
    return linear.weight.detach(), linear.bias.detach(), margin


def margin_scale(pooled: np.ndarray, metadata: pd.DataFrame, margin: np.ndarray) -> float:
    """计算train患者/类别平衡的margin标准差。"""
    weights = patient_class_weights(metadata).astype(np.float64)
    values = np.asarray(pooled) @ margin
    mean = np.average(values, weights=weights)
    std = float(np.sqrt(np.average(np.square(values - mean), weights=weights)))
    if std <= 1e-8:
        raise RuntimeError("train margin标准差过小")
    return std


def initialize_sae(
    arm: str, spatial: np.ndarray, pooled: np.ndarray, metadata: pd.DataFrame,
    args: argparse.Namespace, device: torch.device,
) -> StructuredSparseAutoencoder:
    weights = patient_class_weights(metadata).astype(np.float64)
    if arm in {"B", "N"}:
        features = np.asarray(pooled)
        if arm == "N":
            means = features.mean(1, keepdims=True)
            centered = features - means
            features = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)
        center = np.average(features, axis=0, weights=weights)
    else:
        center = np.average(np.asarray(spatial).mean(1), axis=0, weights=weights)
    hidden = args.debug_hidden_dim if args.debug else HIDDEN_DIM
    target_k = args.debug_k if args.debug else (POOLED_K if arm in {"B", "N"} else PATCH_K)
    mode = "batch_topk" if arm in {"B", "D"} else "topk"
    return StructuredSparseAutoencoder(
        INPUT_DIM, hidden, torch.from_numpy(center.astype(np.float32)).to(device),
        mode, target_k,
    ).to(device)


def gamma_lineage(cache_config: dict, batch_shas: list[str], median_patch: float,
                  median_pool: float, initialization_sha256: str) -> dict:
    """组装gamma_pool校准JSON的血缘与数值字段。"""
    if median_pool < 1e-8:
        raise RuntimeError("gamma_pool校准的median L_pool<1e-8")
    return {
        "protocol": "s2b_gamma_pool_seed42_one_balanced_epoch_74_batches",
        "seed": 42, "batch_size": 32, "batch_count": len(batch_shas),
        "batch_sha256": batch_shas,
        "s4c_initialization_sha256": initialization_sha256,
        "median_l_patch": median_patch, "median_l_pool": median_pool,
        "gamma_pool": 0.25 * median_patch / median_pool,
        "manifest_sha256": MANIFEST_SHA256,
        "student_checkpoint_sha256": CLONG_CHECKPOINT_SHA256,
        "cache_config_sha256": file_sha256(CACHE_ROOT / "cache_config.json"),
        "cache_files": cache_config["files"],
    }


@torch.no_grad()
def calibrate_gamma(
    model: torch.nn.Module, sae: StructuredSparseAutoencoder,
    train_spatial: np.ndarray, train_pooled: np.ndarray,
    train_attention: np.ndarray, train_metadata: pd.DataFrame,
    args: argparse.Namespace, device: torch.device,
) -> dict:
    """用seed42患者平衡的一个完train epoch冻结gamma_pool。"""
    initialization_digest = hashlib.sha256()
    for key, value in sae.state_dict().items():
        initialization_digest.update(key.encode())
        initialization_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    sampler = make_sampler(train_metadata, 42)
    loader = DataLoader(TensorDataset(torch.arange(len(train_metadata))), batch_size=32,
                        sampler=sampler)
    patch_losses, pool_losses, batch_shas = [], [], []
    for (indices,) in loader:
        index_np = indices.numpy()
        current = torch.from_numpy(np.asarray(train_spatial[index_np])).to(device)
        original_pool = torch.from_numpy(np.asarray(train_pooled[index_np])).to(device)
        original_attention = torch.from_numpy(np.asarray(train_attention[index_np])).to(device)
        reconstructed, _ = sae(current)
        weights = patch_position_weights(original_attention)
        patch_loss = ((reconstructed - current).square().mean(2) * weights).mean()
        rebuilt_attention = attention_from_features(reconstructed, model.attention_head)
        rebuilt_pool = pooled_from_features(reconstructed, rebuilt_attention)
        pool_loss = F.mse_loss(rebuilt_pool, original_pool)
        patch_losses.append(float(patch_loss))
        pool_losses.append(float(pool_loss))
        batch_shas.append(hashlib.sha256(index_np.astype(np.int64).tobytes()).hexdigest())
    if not args.debug and len(batch_shas) != 74:
        raise RuntimeError(f"正式gamma_pool校准必须为74批，当前{len(batch_shas)}")
    config = json.loads((CACHE_ROOT / "cache_config.json").read_text(encoding="utf-8"))
    result = gamma_lineage(
        config, batch_shas, float(np.median(patch_losses)), float(np.median(pool_losses)),
        initialization_digest.hexdigest(),
    )
    target = (args.output_root if args.debug else OUTPUT_ROOT) / GAMMA_JSON.name
    if target.exists():
        raise FileExistsError(f"gamma_pool JSON已存在: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"gamma_pool={result['gamma_pool']:.8f}; 保存至 {target}")
    return result


def load_gamma(args: argparse.Namespace) -> tuple[float, dict]:
    path = (args.output_root if args.debug else OUTPUT_ROOT) / GAMMA_JSON.name
    config = json.loads(path.read_text(encoding="utf-8"))
    cache_config = json.loads((CACHE_ROOT / "cache_config.json").read_text(encoding="utf-8"))
    checks = {
        "protocol": config.get("protocol") == "s2b_gamma_pool_seed42_one_balanced_epoch_74_batches",
        "seed": int(config.get("seed", -1)) == 42,
        "batch_size": int(config.get("batch_size", -1)) == 32,
        "manifest": config.get("manifest_sha256") == MANIFEST_SHA256,
        "student": config.get("student_checkpoint_sha256") == CLONG_CHECKPOINT_SHA256,
        "cache_config": config.get("cache_config_sha256") == file_sha256(CACHE_ROOT / "cache_config.json"),
        "cache_files": config.get("cache_files") == cache_config["files"],
    }
    if not all(checks.values()):
        raise ValueError(f"gamma_pool JSON血缘失败: {[k for k,v in checks.items() if not v]}")
    return float(config["gamma_pool"]), config


def normalized_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐样本返回去维度均值的单位方向、原均值与原模长。"""
    means = np.asarray(features).mean(1)
    centered = np.asarray(features) - means[:, None]
    norms = np.linalg.norm(centered, axis=1)
    directions = centered / np.maximum(norms[:, None], 1e-12)
    return directions.astype(np.float32), means.astype(np.float32), norms.astype(np.float32)


def batch_arrays(
    indices: np.ndarray, arm: str, spatial: np.ndarray, pooled: np.ndarray,
    attention: np.ndarray, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按图像索引取出当前臂输入、原pooled和原attention。"""
    if arm in {"B", "N"}:
        features = np.asarray(pooled[indices]).copy()
        if arm == "N":
            features = normalized_features(features)[0]
    else:
        features = np.asarray(spatial[indices]).copy()
    return (
        torch.from_numpy(features).to(device),
        torch.from_numpy(np.asarray(pooled[indices]).copy()).to(device),
        torch.from_numpy(np.asarray(attention[indices]).copy()).to(device),
    )


def loss_terms(
    arm: str, sae: StructuredSparseAutoencoder, features: torch.Tensor,
    original_pool: torch.Tensor, original_attention: torch.Tensor,
    model: torch.nn.Module, margin_vector: torch.Tensor, margin_std: float,
    gamma_pool: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """计算当前臂的总损失、分项损失和稀疏激活。"""
    reconstructed, hidden = sae(features)
    if arm in {"B", "N"}:
        mse = F.mse_loss(reconstructed, features)
        margin = normalized_margin_mse(
            features, reconstructed, margin_vector, margin_std
        )
        total = mse + GAMMA_MARGIN * margin
        return total, {"reconstruction": mse, "pool": torch.zeros_like(mse),
                       "margin": margin}, hidden
    weights = patch_position_weights(original_attention)
    patch = ((reconstructed - features).square().mean(2) * weights).mean()
    rebuilt_attention = attention_from_features(reconstructed, model.attention_head)
    rebuilt_pool = pooled_from_features(reconstructed, rebuilt_attention)
    pool = F.mse_loss(rebuilt_pool, original_pool)
    margin = normalized_margin_mse(
        original_pool, rebuilt_pool, margin_vector, margin_std
    )
    total = patch + gamma_pool * pool + GAMMA_MARGIN * margin
    return total, {"reconstruction": patch, "pool": pool, "margin": margin}, hidden


@torch.no_grad()
def evaluate_loss(
    arm: str, sae: StructuredSparseAutoencoder, spatial: np.ndarray,
    pooled: np.ndarray, attention: np.ndarray, metadata: pd.DataFrame,
    model: torch.nn.Module,
    margin_vector: torch.Tensor, margin_std: float, gamma_pool: float,
    batch_size: int, device: torch.device,
) -> dict:
    sae.eval()
    totals = np.zeros(5, dtype=np.float64)
    weights_all = patient_class_weights(metadata).astype(np.float64)
    weight_sum = 0.0
    for start in range(0, len(pooled), batch_size):
        indices = np.arange(start, min(start + batch_size, len(pooled)))
        features, original_pool, original_attention = batch_arrays(
            indices, arm, spatial, pooled, attention, device
        )
        reconstructed, hidden = sae(features)
        if arm in {"B", "N"}:
            reconstruction = (reconstructed - features).square().mean(1)
            pool_loss = torch.zeros_like(reconstruction)
            margin_loss = (
                ((reconstructed - features) @ margin_vector) / margin_std
            ).square()
            total = reconstruction + GAMMA_MARGIN * margin_loss
            l0 = hidden.gt(ACTIVE_EPS).sum(1).float()
        else:
            position_error = (reconstructed - features).square().mean(2)
            reconstruction = (
                position_error * patch_position_weights(original_attention)
            ).mean(1)
            rebuilt_attention = attention_from_features(reconstructed, model.attention_head)
            rebuilt_pool = pooled_from_features(reconstructed, rebuilt_attention)
            pool_loss = (rebuilt_pool - original_pool).square().mean(1)
            margin_loss = (
                ((rebuilt_pool - original_pool) @ margin_vector) / margin_std
            ).square()
            total = reconstruction + gamma_pool * pool_loss + GAMMA_MARGIN * margin_loss
            l0 = hidden.gt(ACTIVE_EPS).sum(2).float().mean(1)
        values = torch.stack((total, reconstruction, pool_loss, margin_loss, l0), 1)
        current_weights = weights_all[indices]
        totals += (values.cpu().numpy() * current_weights[:, None]).sum(0)
        weight_sum += current_weights.sum()
    values = totals / weight_sum
    return dict(zip(("total", "reconstruction", "pool", "margin", "l0"), values))


def train_sae(
    arm: str, sae: StructuredSparseAutoencoder, spatial: dict, pooled: dict,
    attention: dict, metadata: dict, model: torch.nn.Module, args: argparse.Namespace,
    device: torch.device, output: Path, gamma_pool: float,
) -> tuple[StructuredSparseAutoencoder, dict]:
    """患者类别平衡训练，只按各臂val总损失选checkpoint。"""
    _, _, margin = classifier_components(model)
    margin_np = margin.cpu().numpy()
    margin_features = (
        normalized_features(pooled["train"])[0]
        if arm == "N" else pooled["train"]
    )
    scale = margin_scale(margin_features, metadata["train"], margin_np)
    margin_device = margin.to(device)
    sampler = make_sampler(metadata["train"], args.seed)
    loader = DataLoader(TensorDataset(torch.arange(len(metadata["train"]))),
                        batch_size=args.image_batch_size, sampler=sampler)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.learning_rate)
    warmup_epochs = max(1, math.ceil(args.epochs * args.warmup_fraction))
    history, best, stale = [], math.inf, 0
    checkpoint_path = output / "sae_best.pth"
    for epoch in range(1, args.epochs + 1):
        warmup = min(1.0, epoch / warmup_epochs)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * warmup
        sae.train()
        totals = np.zeros(5, dtype=np.float64)
        seen = 0
        for (indices,) in loader:
            index_np = indices.numpy()
            features, original_pool, original_attention = batch_arrays(
                index_np, arm, spatial["train"], pooled["train"], attention["train"], device
            )
            optimizer.zero_grad(set_to_none=True)
            total, terms, hidden = loss_terms(
                arm, sae, features, original_pool, original_attention, model,
                margin_device, scale, gamma_pool,
            )
            # warmup同时线性引入pool和margin约束。
            if warmup < 1:
                base = terms["reconstruction"]
                total = base + warmup * (total - base)
            total.backward()
            with torch.no_grad():
                project_decoder_gradient(sae)
            optimizer.step()
            sae.normalize_decoder()
            count = len(index_np)
            totals += np.array([
                float(total.detach()), float(terms["reconstruction"].detach()),
                float(terms["pool"].detach()), float(terms["margin"].detach()),
                float(hidden.gt(ACTIVE_EPS).sum(-1).float().mean().detach()),
            ]) * count
            seen += count
        train_values = totals / seen
        val = evaluate_loss(
            arm, sae, spatial["val"], pooled["val"], attention["val"],
            metadata["val"], model,
            margin_device, scale, gamma_pool, args.image_batch_size, device,
        )
        row = {
            "epoch": epoch, "warmup": warmup, "learning_rate": args.learning_rate * warmup,
            "train_total": train_values[0], "train_reconstruction": train_values[1],
            "train_pool": train_values[2], "train_margin": train_values[3],
            "train_l0": train_values[4], **{f"val_{key}": value for key, value in val.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} warmup={warmup:.2f} "
            f"train={train_values[0]:.5f} rec={train_values[1]:.5f} "
            f"pool={train_values[2]:.5f} margin={train_values[3]:.5f} "
            f"L0={train_values[4]:.1f} | val={val['total']:.5f} "
            f"rec={val['reconstruction']:.5f} pool={val['pool']:.5f} "
            f"margin={val['margin']:.5f} L0={val['l0']:.1f}"
        )
        if epoch < warmup_epochs:
            continue
        if val["total"] < best:
            best, stale = float(val["total"]), 0
            torch.save({
                "sae_state_dict": sae.state_dict(), "arm": arm,
                "seed": args.seed, "hidden_dim": sae.hidden_dim,
                "target_k": sae.target_k, "activation_mode": sae.activation_mode,
                "gamma_pool": gamma_pool, "gamma_margin": GAMMA_MARGIN,
                "margin_train_std": scale, "epoch": epoch, "val_total": best,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop: {args.patience}轮无val总损失改善")
                break
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sae.load_state_dict(checkpoint["sae_state_dict"], strict=True)
    return sae, checkpoint


def threshold_chunks(
    sae: StructuredSparseAutoencoder, arm: str, spatial: np.ndarray,
    pooled: np.ndarray, batch_size: int, device: torch.device,
):
    """构造可重入的train预激活分块生成器，供全局阈值求解。"""
    @torch.no_grad()
    def factory():
        sae.eval()
        for start in range(0, len(pooled), batch_size):
            end = min(start + batch_size, len(pooled))
            values = (
                np.asarray(pooled[start:end]).copy()
                if arm == "B" else np.asarray(spatial[start:end]).copy()
            )
            dense = sae.preactivation(torch.from_numpy(values).to(device))
            yield dense.cpu().numpy()
    return factory


def freeze_batchtopk_threshold(
    sae: StructuredSparseAutoencoder, arm: str, spatial: np.ndarray,
    pooled: np.ndarray, metadata: pd.DataFrame, args: argparse.Namespace,
    device: torch.device, checkpoint_path: Path,
) -> dict:
    vectors = len(pooled) if arm == "B" else len(pooled) * 49
    result = solve_threshold_from_chunks(
        threshold_chunks(sae, arm, spatial, pooled, args.image_batch_size, device),
        vectors, sae.target_k,
    ).as_dict()
    result.update({
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "manifest_sha256": MANIFEST_SHA256,
        "cache_config_sha256": file_sha256(CACHE_ROOT / "cache_config.json"),
        "sample_order_sha256": array_sequence_sha(
            metadata.relative_path.astype(str).to_numpy(dtype="U")
        ),
        "comparison": "activation >= theta",
        "estimated_on": "train deterministic once; no replacement; no val recalibration",
        "auxiliary_dead_latent_loss": False,
    })
    return result


@torch.no_grad()
def project_pooled(
    sae: StructuredSparseAutoencoder, features: np.ndarray, arm: str,
    threshold: float | None, batch_size: int, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    reconstructed, activations = [], []
    for start in range(0, len(features), batch_size):
        values = np.asarray(features[start:start + batch_size]).copy()
        if arm == "N":
            values = normalized_features(values)[0]
        tensor = torch.from_numpy(values).to(device)
        hidden = sae.encode_threshold(tensor, threshold) if threshold is not None else sae.encode(tensor)
        reconstructed.append(sae.decode(hidden).cpu().numpy())
        activations.append(hidden.cpu().numpy())
    return np.concatenate(reconstructed), np.concatenate(activations)


@torch.no_grad()
def project_patch(
    sae: StructuredSparseAutoencoder, spatial: np.ndarray, original_attention: np.ndarray,
    metadata: pd.DataFrame, model: torch.nn.Module, threshold: float | None,
    batch_size: int, device: torch.device, collect_feature_coverage: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """流式重构patch，返回完整替换pooled/attention和稀疏分布统计。"""
    pooled_rows, attention_rows = [], []
    hidden_dim = sae.hidden_dim
    active_vectors = np.zeros(hidden_dim, dtype=np.int64)
    active_images = np.zeros(hidden_dim, dtype=np.int64)
    active_patients: list[set[str]] | None = (
        [set() for _ in range(hidden_dim)] if collect_feature_coverage else None
    )
    image_unique, all_l0, high_l0, low_l0 = [], [], [], []
    high_error, low_error = [], []
    inside_l0, outside_l0, boundary_l0 = [], [], []
    patch_cosines, image_patch_cosines = [], []
    for start in range(0, len(spatial), batch_size):
        end = min(start + batch_size, len(spatial))
        original = torch.from_numpy(np.asarray(spatial[start:end]).copy()).to(device)
        original_attn = torch.from_numpy(
            np.asarray(original_attention[start:end]).copy()
        ).to(device)
        hidden = sae.encode_threshold(original, threshold) if threshold is not None else sae.encode(original)
        reconstructed = sae.decode(hidden)
        rebuilt_attention = attention_from_features(reconstructed, model.attention_head)
        rebuilt_pool = pooled_from_features(reconstructed, rebuilt_attention)
        pooled_rows.append(rebuilt_pool.cpu().numpy())
        attention_rows.append(rebuilt_attention.cpu().numpy())
        active = hidden.gt(ACTIVE_EPS)
        active_vectors += active.sum((0, 1)).cpu().numpy()
        active_images += active.any(1).sum(0).cpu().numpy()
        image_unique.extend(active.any(1).sum(1).cpu().numpy().tolist())
        l0 = active.sum(2).cpu().numpy()
        all_l0.extend(l0.reshape(-1).tolist())
        attn_np = original_attn.cpu().numpy()
        errors = (reconstructed - original).square().mean(2).cpu().numpy()
        for row in range(end - start):
            order = np.argsort(attn_np[row])
            low, high = order[:12], order[-12:]
            low_l0.extend(l0[row, low].tolist()); high_l0.extend(l0[row, high].tolist())
            low_error.extend(errors[row, low].tolist()); high_error.extend(errors[row, high].tolist())
            source = metadata.iloc[start + row]
            if int(source.label) == 1 and all(
                np.isfinite(float(source.get(key, np.nan)))
                for key in ("bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm")
            ):
                box = source[["bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm"]].to_numpy(float)
                overlap = cell_overlap_map(box).reshape(-1)
                inside_l0.extend(l0[row, overlap >= 0.999].tolist())
                outside_l0.extend(l0[row, overlap <= 0.001].tolist())
                boundary_l0.extend(l0[row, (overlap > 0.001) & (overlap < 0.999)].tolist())
            if active_patients is not None:
                ids = np.flatnonzero(active[row].any(0).cpu().numpy())
                patient = str(source.patient_id)
                for feature_id in ids:
                    active_patients[int(feature_id)].add(patient)
        original_np = original.cpu().numpy().reshape(-1, INPUT_DIM)
        reconstructed_np = reconstructed.cpu().numpy().reshape(-1, INPUT_DIM)
        patch_cosines.extend((
            np.sum(original_np * reconstructed_np, 1) /
            (np.linalg.norm(original_np, axis=1) * np.linalg.norm(reconstructed_np, axis=1) + 1e-12)
        ).tolist())
        image_patch_cosines.extend((
            np.sum(original.cpu().numpy() * reconstructed.cpu().numpy(), axis=2) /
            (
                np.linalg.norm(original.cpu().numpy(), axis=2) *
                np.linalg.norm(reconstructed.cpu().numpy(), axis=2) + 1e-12
            )
        ).mean(1).tolist())
    coverage = {
        "mean_l0_per_position": float(np.mean(all_l0)),
        "mean_unique_features_per_image": float(np.mean(image_unique)),
        "feature_active_position_counts": active_vectors.tolist(),
        "feature_active_image_counts": active_images.tolist(),
        "feature_active_patient_counts": (
            [len(value) for value in active_patients] if active_patients is not None else None
        ),
        "high_attention_quartile_mean_l0": float(np.mean(high_l0)),
        "low_attention_quartile_mean_l0": float(np.mean(low_l0)),
        "high_attention_quartile_mse": float(np.mean(high_error)),
        "low_attention_quartile_mse": float(np.mean(low_error)),
        "bbox_inside_mean_l0": float(np.mean(inside_l0)) if inside_l0 else None,
        "bbox_outside_mean_l0": float(np.mean(outside_l0)) if outside_l0 else None,
        "bbox_boundary_mean_l0": float(np.mean(boundary_l0)) if boundary_l0 else None,
        "mean_patch_cosine": float(np.mean(patch_cosines)),
        "per_image_mean_patch_cosine": image_patch_cosines,
    }
    return np.concatenate(pooled_rows), np.concatenate(attention_rows), coverage


def patient_frame(metadata: pd.DataFrame, original: np.ndarray, reconstructed: np.ndarray) -> pd.DataFrame:
    """按患者平均聚合原始与重构癌概率。"""
    frame = metadata[["patient_id", "label"]].copy()
    frame["original"] = original
    frame["reconstructed"] = reconstructed
    return frame.groupby("patient_id", as_index=False).agg(
        label=("label", "first"), original=("original", "mean"),
        reconstructed=("reconstructed", "mean"),
    )


def fidelity_extras(
    original: np.ndarray, reconstructed: np.ndarray, metadata: pd.DataFrame,
    weight: np.ndarray, bias: np.ndarray,
) -> dict:
    """补齐冻结阈值、margin和最大患者概率偏移指标。"""
    original_logits = original @ weight.T + bias
    reconstructed_logits = reconstructed @ weight.T + bias
    original_probability = torch.softmax(torch.from_numpy(original_logits), 1)[:, 1].numpy()
    reconstructed_probability = torch.softmax(
        torch.from_numpy(reconstructed_logits), 1
    )[:, 1].numpy()
    labels = metadata.label.to_numpy(int)
    patients = patient_frame(metadata, original_probability, reconstructed_probability)
    original_margin = original_logits[:, 1] - original_logits[:, 0]
    reconstructed_margin = reconstructed_logits[:, 1] - reconstructed_logits[:, 0]
    margin_delta = reconstructed_margin - original_margin
    deviation = np.abs(patients.reconstructed - patients.original)
    maximum = int(deviation.argmax())
    correlation = np.corrcoef(original_margin, reconstructed_margin)[0, 1]
    return {
        "image_threshold_original": confusion_at_threshold(
            labels, original_probability, FROZEN_IMAGE_THRESHOLD,
        ),
        "image_threshold_reconstructed": confusion_at_threshold(
            labels, reconstructed_probability, FROZEN_IMAGE_THRESHOLD,
        ),
        "patient_threshold_original": confusion_at_threshold(
            patients.label.to_numpy(int), patients.original.to_numpy(),
            FROZEN_PATIENT_THRESHOLD,
        ),
        "patient_threshold_reconstructed": confusion_at_threshold(
            patients.label.to_numpy(int), patients.reconstructed.to_numpy(),
            FROZEN_PATIENT_THRESHOLD,
        ),
        "margin_mse": float(np.square(margin_delta).mean()),
        "margin_mae": float(np.abs(margin_delta).mean()),
        "margin_pearson": float(correlation),
        "patient_probability_max_deviation": float(deviation.max()),
        "patient_probability_max_deviation_patient_id": str(
            patients.patient_id.iloc[maximum]
        ),
    }


def patch_cosine_strata(
    metadata: pd.DataFrame, values: list[float], lesion_bounds: np.ndarray,
) -> dict:
    """按标签、来源和冻结病灶大小报告每图patch cosine。"""
    frame = metadata.copy()
    frame["patch_cosine"] = np.asarray(values, dtype=float)
    result = {"label": {}, "source": {}, "lesion_size": {}}
    for label, group in frame.groupby("label"):
        result["label"][str(int(label))] = {
            "images": int(len(group)), "mean": float(group.patch_cosine.mean())
        }
    if "source" in frame:
        for source, group in frame.groupby(frame.source.fillna("__MISSING__")):
            result["source"][str(source)] = {
                "images": int(len(group)), "mean": float(group.patch_cosine.mean())
            }
    cancer = frame.loc[frame.label.eq(1)].copy()
    if "bbox_area_fraction" in cancer and len(cancer):
        cancer["_lesion"] = pd.cut(
            cancer.bbox_area_fraction,
            [-np.inf, lesion_bounds[0], lesion_bounds[1], np.inf],
            labels=["small", "medium", "large"], include_lowest=True,
        )
        result["lesion_size_bounds_train_frozen"] = [float(x) for x in lesion_bounds]
        for name, group in cancer.groupby("_lesion", observed=True):
            result["lesion_size"][str(name)] = {
                "images": int(len(group)), "mean": float(group.patch_cosine.mean())
            }
    return result


def scalar_auc(train_values: np.ndarray, train_labels: np.ndarray,
               val_values: np.ndarray, val_labels: np.ndarray) -> dict:
    """报告一个标量在train和val上的标签AUC。"""
    return {
        "train_auc": float(roc_auc_score(train_labels, train_values)),
        "val_auc": float(roc_auc_score(val_labels, val_values)),
    }


def lock_train_sensitivity_threshold(
    labels: np.ndarray, probabilities: np.ndarray, target: float = 0.90,
) -> float:
    """仅在train正类实际候选值中冻结最高Sens阈值。"""
    positives = probabilities[np.asarray(labels) == 1]
    for threshold in np.sort(np.unique(positives))[::-1]:
        if float(np.mean(positives >= threshold)) >= target:
            return float(threshold)
    raise RuntimeError("无法冻结train Sens>=0.90阈值")


def evaluate_normalization_diagnostic(
    sae: StructuredSparseAutoencoder, pooled: dict, metadata: dict,
    model: torch.nn.Module, args: argparse.Namespace, device: torch.device,
) -> dict:
    """S4-N只用train拟合标量旁路，val只做评价。"""
    directions, means, norms = {}, {}, {}
    recon_directions = {}
    for split in ("train", "val"):
        directions[split], means[split], norms[split] = normalized_features(pooled[split])
        recon_directions[split], _ = project_pooled(
            sae, pooled[split], "N", None, args.image_batch_size, device
        )
    train_labels = metadata["train"].label.to_numpy(int)
    val_labels = metadata["val"].label.to_numpy(int)
    scalar_model = LogisticRegression(random_state=42, max_iter=1000).fit(
        np.column_stack((means["train"], norms["train"])), train_labels
    )
    scalar_train = scalar_model.predict_proba(np.column_stack((means["train"], norms["train"])))[:, 1]
    scalar_val = scalar_model.predict_proba(np.column_stack((means["val"], norms["val"])))[:, 1]
    scalar_sens_threshold = lock_train_sensitivity_threshold(train_labels, scalar_train)
    scalar_patient = {}
    for split, probabilities in (("train", scalar_train), ("val", scalar_val)):
        frame = metadata[split][["patient_id", "label"]].copy()
        frame["probability"] = probabilities
        scalar_patient[split] = frame.groupby("patient_id", as_index=False).agg(
            label=("label", "first"), probability=("probability", "mean")
        )
    scalar_patient_threshold = lock_train_sensitivity_threshold(
        scalar_patient["train"].label.to_numpy(int),
        scalar_patient["train"].probability.to_numpy(),
    )
    median_mean, median_norm = np.median(means["train"]), np.median(norms["train"])
    weight, bias, _ = classifier_components(model)
    weight_np, bias_np = weight.cpu().numpy(), bias.cpu().numpy()
    def restore(direction, mean, norm):
        return direction * norm[:, None] + mean[:, None]
    val_true = restore(recon_directions["val"], means["val"], norms["val"])
    val_median = restore(
        recon_directions["val"], np.full(len(val_labels), median_mean),
        np.full(len(val_labels), median_norm),
    )
    return {
        "role": "diagnostic_only_not_for_selection",
        "mean_scalar": scalar_auc(means["train"], train_labels, means["val"], val_labels),
        "norm_scalar": scalar_auc(norms["train"], train_labels, norms["val"], val_labels),
        "mean_norm_logistic": {
            "train_auc": float(roc_auc_score(train_labels, scalar_train)),
            "val_auc": float(roc_auc_score(val_labels, scalar_val)),
            "val_threshold_0_5": confusion_at_threshold(val_labels, scalar_val, 0.5),
            "train_locked_sensitivity_threshold": scalar_sens_threshold,
            "val_train_locked_sensitivity_threshold": confusion_at_threshold(
                val_labels, scalar_val, scalar_sens_threshold,
            ),
            "val_patient_auc": float(roc_auc_score(
                scalar_patient["val"].label, scalar_patient["val"].probability
            )),
            "val_patient_threshold_0_5": confusion_at_threshold(
                scalar_patient["val"].label.to_numpy(int),
                scalar_patient["val"].probability.to_numpy(), 0.5,
            ),
            "train_locked_patient_sensitivity_threshold": scalar_patient_threshold,
            "val_patient_train_locked_sensitivity_threshold": confusion_at_threshold(
                scalar_patient["val"].label.to_numpy(int),
                scalar_patient["val"].probability.to_numpy(), scalar_patient_threshold,
            ),
        },
        "direction_with_true_scalars": classification_metrics(
            pooled["val"], val_true, metadata["val"], weight_np, bias_np,
            FROZEN_PATIENT_THRESHOLD,
        ),
        "direction_with_train_median_scalars": classification_metrics(
            pooled["val"], val_median, metadata["val"], weight_np, bias_np,
            FROZEN_PATIENT_THRESHOLD,
        ),
        "train_median_mean": float(median_mean),
        "train_median_norm": float(median_norm),
    }


@torch.no_grad()
def evaluate_product(
    arm: str, sae: StructuredSparseAutoencoder, spatial: dict, pooled: dict,
    attention: dict, metadata: dict, model: torch.nn.Module,
    threshold_info: dict | None, args: argparse.Namespace, device: torch.device,
) -> dict:
    weight, bias, _ = classifier_components(model)
    weight_np, bias_np = weight.cpu().numpy(), bias.cpu().numpy()
    threshold = threshold_info["threshold"] if threshold_info else None
    train_counts = None
    if arm in {"B", "N"}:
        reconstructed, val_activations = project_pooled(
            sae, pooled["val"], arm, threshold, args.image_batch_size, device
        )
        _, train_activations = project_pooled(
            sae, pooled["train"], arm, threshold, args.image_batch_size, device
        )
        train_counts = (train_activations > ACTIVE_EPS).sum(0)
        metrics = classification_metrics(
            pooled["val"], reconstructed if arm == "B" else pooled["val"],
            metadata["val"], weight_np, bias_np, FROZEN_PATIENT_THRESHOLD,
        )
        distribution = {
            "val_mean_l0": float((val_activations > ACTIVE_EPS).sum(1).mean()),
            "train_mean_l0": float((train_activations > ACTIVE_EPS).sum(1).mean()),
        }
        metrics.update(fidelity_extras(
            pooled["val"], reconstructed, metadata["val"], weight_np, bias_np
        ))
        if arm == "N":
            return {"normalization_diagnostic": evaluate_normalization_diagnostic(
                sae, pooled, metadata, model, args, device
            ), "sparsity": distribution}
    else:
        train_pool, train_rebuilt_attention, train_distribution = project_patch(
            sae, spatial["train"], attention["train"], metadata["train"], model,
            threshold, args.image_batch_size, device, True,
        )
        rebuilt_pool, rebuilt_attention, distribution = project_patch(
            sae, spatial["val"], attention["val"], metadata["val"], model,
            threshold, args.image_batch_size, device, False,
        )
        train_counts = np.asarray(train_distribution["feature_active_position_counts"])
        metrics = classification_metrics(
            pooled["val"], rebuilt_pool, metadata["val"], weight_np, bias_np,
            FROZEN_PATIENT_THRESHOLD,
        )
        metrics.update(fidelity_extras(
            pooled["val"], rebuilt_pool, metadata["val"], weight_np, bias_np
        ))
        labels = metadata["val"].label.to_numpy(int)
        bbox_columns = ["bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm"]
        bboxes = metadata["val"][bbox_columns].to_numpy(float)
        original_spatial = spatial_alignment_metrics(attention["val"], labels, bboxes)
        rebuilt_spatial = spatial_alignment_metrics(rebuilt_attention, labels, bboxes)
        distribution["train"] = train_distribution
        distribution["attention_drift"] = attention_drift_metrics(
            attention["val"], rebuilt_attention
        )
        distribution["spatial_original"] = original_spatial
        distribution["spatial_reconstructed"] = rebuilt_spatial
        distribution["normalized_aib_drop"] = (
            original_spatial["mean_normalized_aib"] - rebuilt_spatial["mean_normalized_aib"]
        )
        distribution["pga_drop"] = original_spatial["pga"] - rebuilt_spatial["pga"]
        train_cancer_area = metadata["train"].loc[
            metadata["train"].label.eq(1), "bbox_area_fraction"
        ].to_numpy(float)
        lesion_bounds = np.quantile(train_cancer_area, [1 / 3, 2 / 3])
        distribution["patch_cosine_strata"] = patch_cosine_strata(
            metadata["val"], distribution["per_image_mean_patch_cosine"], lesion_bounds
        )
        # 固定原attention只作诊断，不进门槛。
        fixed_rows = []
        for start in range(0, len(spatial["val"]), args.image_batch_size):
            original = torch.from_numpy(
                np.asarray(spatial["val"][start:start + args.image_batch_size]).copy()
            ).to(device)
            original_attn = torch.from_numpy(
                np.asarray(attention["val"][start:start + args.image_batch_size]).copy()
            ).to(device)
            hidden = sae.encode_threshold(original, threshold) if threshold is not None else sae.encode(original)
            fixed_rows.append(pooled_from_features(sae.decode(hidden), original_attn).cpu().numpy())
        fixed_pool = np.concatenate(fixed_rows)
        distribution["fixed_original_attention_diagnostic"] = classification_metrics(
            pooled["val"], fixed_pool, metadata["val"], weight_np, bias_np,
            FROZEN_PATIENT_THRESHOLD,
        )
    nondead = train_counts > 0
    duplicate = duplicate_decoder_rate_nondead(
        sae.decoder_weight.detach().cpu().numpy(), nondead
    )
    metrics["patient_auc_drop"] = (
        metrics["patient_original_auc"] - metrics["patient_reconstructed_auc"]
    )
    return {
        "fidelity": metrics,
        "sparsity": distribution,
        "dead_feature_count": int((~nondead).sum()),
        "dead_feature_rate": float((~nondead).mean()),
        "duplicate": duplicate,
    }


def write_config(output: Path, payload: dict) -> None:
    """以严格标准JSON写入正式配置，禁止NaN。"""
    (output / "config.json").write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_protocol_failure(
    output: Path, args: argparse.Namespace, checkpoint: dict, error: Exception,
) -> Path:
    """将预注册门槛失败写成可汇总、可追溯的正式记录。"""
    path = output / "protocol_failure.json"
    payload = {
        "stage": "S2b", "arm": args.arm, "seed": args.seed,
        "experiment": args.experiment, "debug": args.debug,
        "failure_stage": "freeze_batchtopk_train_threshold",
        "reason": str(error),
        "rule": (
            "fail when positive train preactivations are fewer than N*K; "
            "zero activations must not be used to fill K"
        ),
        "student_checkpoint_sha256": CLONG_CHECKPOINT_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "cache_config_sha256": file_sha256(CACHE_ROOT / "cache_config.json"),
        "checkpoint_sha256": file_sha256(output / "sae_best.pth"),
        "best_epoch": int(checkpoint["epoch"]),
        "training": dict(FORMAL_BUDGET),
        "test_evaluated": False, "internal_test_evaluated": False,
        "external_evaluated": False,
    }
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return path


def main() -> None:
    """执行血缘校验、缓存、校准/训练、评价和产物落盘。"""
    global CACHE_ROOT
    args = parse_args()
    validate_args(args)
    if args.debug:
        CACHE_ROOT = args.output_root / "frozen_spatial_cache"
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    lineage = verify_s0_lineage()
    frame = load_manifest_frame(args)
    model = load_clong_model(device)
    if not (CACHE_ROOT / "cache_config.json").exists():
        build_spatial_cache(model, frame, args, device, lineage)
    spatial, pooled, attention, metadata = load_cache(frame)
    sae = initialize_sae(
        args.arm, spatial["train"], pooled["train"], metadata["train"], args, device
    )
    if args.calibrate_gamma:
        if args.arm != "C" or args.seed != 42:
            raise ValueError("gamma_pool只能由S4-C seed42初始化校准")
        calibrate_gamma(
            model, sae, spatial["train"], pooled["train"], attention["train"],
            metadata["train"], args, device,
        )
        return
    gamma_pool, gamma_config = (0.0, None)
    if args.arm in {"C", "D"}:
        gamma_pool, gamma_config = load_gamma(args)
    if args.postprocess_existing:
        output, sae, checkpoint = load_existing_training_product(args, sae, device)
    else:
        output = prepare_output(args)
        sae, checkpoint = train_sae(
            args.arm, sae, spatial, pooled, attention, metadata, model, args, device,
            output, gamma_pool,
        )
    threshold_info = None
    if args.arm in {"B", "D"}:
        try:
            threshold_info = freeze_batchtopk_threshold(
                sae, args.arm, spatial["train"], pooled["train"], metadata["train"],
                args, device, output / "sae_best.pth",
            )
        except RuntimeError as error:
            failure_path = write_protocol_failure(output, args, checkpoint, error)
            print(f"BatchTopK协议失败已记录: {failure_path}")
            raise
        (output / "batchtopk_threshold.json").write_text(
            json.dumps(json_ready(threshold_info), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    evaluation = evaluate_product(
        args.arm, sae, spatial, pooled, attention, metadata, model,
        threshold_info, args, device,
    )
    payload = {
        "stage": "S2b", "arm": args.arm, "seed": args.seed,
        "experiment": args.experiment, "debug": args.debug,
        "student_checkpoint_sha256": CLONG_CHECKPOINT_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "cache_config_sha256": file_sha256(CACHE_ROOT / "cache_config.json"),
        "checkpoint_sha256": file_sha256(output / "sae_best.pth"),
        "architecture": {
            "input": "pooled_1280" if args.arm in {"B", "N"} else "shared_patch_49x1280",
            "hidden_dim": sae.hidden_dim, "target_k": sae.target_k,
            "activation_mode": sae.activation_mode,
            "batchtopk_auxiliary_dead_latent_loss": False,
        },
        "training": {
            "learning_rate": args.learning_rate, "epochs": args.epochs,
            "patience": args.patience, "warmup_fraction": args.warmup_fraction,
            "image_batch_size": args.image_batch_size,
            "gamma_margin": GAMMA_MARGIN, "gamma_pool": gamma_pool,
            "best_epoch": checkpoint["epoch"],
        },
        "gamma_calibration_json_sha256": (
            file_sha256((args.output_root if args.debug else OUTPUT_ROOT) / GAMMA_JSON.name)
            if gamma_config is not None else None
        ),
        "batchtopk_threshold": threshold_info,
        "evaluation": evaluation,
        "test_evaluated": False, "internal_test_evaluated": False,
        "external_evaluated": False,
        "git": git_snapshot(),
    }
    write_config(output, payload)
    print(f"输出目录: {output}")


if __name__ == "__main__":
    main()
