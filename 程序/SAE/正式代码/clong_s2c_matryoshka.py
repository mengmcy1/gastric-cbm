#!/usr/bin/env python3
"""C-long S2c Matryoshka patch SAE 正式入口。

协议冻结于 ``SAE实验进度与结果讨论.md`` S2c（2026-08-20冻结）：
同一 ``1280 -> 10240 -> 1280`` 共享字典上联合训练
``K={64,128,256,512,1024}`` 五层嵌套Top-K；每层损失为
``L_patch + gamma_pool * L_pool + 0.1 * L_margin``，总损失为五层均匀平均；
``gamma_pool`` 由seed42固定初始化在train-only 74个平衡批次上按
``0.25*median(joint_patch)/median(joint_pool)`` 一次校准冻结。

训练后仅用冻结val对每个K独立执行完整替换评价（重构7x7x1280、重算冻结
注意力、重新汇聚、进入冻结分类头），沿用S2b八项硬门槛；死亡/重复率按
五层train激活并集定义。正式产品K为同时通过八项门槛的最小K，不做二次
挑选；全部失败则无正式产品并停止。test/internal test/external全程锁定。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "程序/MAGE/正式代码"))
sys.path.insert(0, str(PROJECT_ROOT / "程序/模型训练/正式代码"))

from build_mage_teacher_roi_manifest import file_sha256  # noqa: E402
from clong_sae_discovery import (  # noqa: E402
    CLONG_CHECKPOINT_SHA256,
    FROZEN_PATIENT_THRESHOLD,
    INPUT_DIM,
    MANIFEST_SHA256,
    classification_metrics,
    duplicate_decoder_rate_nondead,
    load_clong_model,
    load_manifest_frame,
    patient_class_weights,
    verify_s0_lineage,
)
from clong_s2b_core import (  # noqa: E402
    ACTIVE_EPS,
    attention_drift_metrics,
    attention_from_features,
    normalized_margin_mse,
    patch_position_weights,
    pooled_from_features,
    project_decoder_gradient,
    spatial_alignment_metrics,
)
from clong_s2b_discovery import (  # noqa: E402
    GAMMA_MARGIN,
    HIDDEN_DIM,
    choose_device,
    classifier_components,
    fidelity_extras,
    load_cache,
    make_sampler,
    margin_scale,
    patch_cosine_strata,
    patient_frame,
    project_patch,
)
import clong_s2b_discovery as s2b_module  # noqa: E402


def cache_root() -> Path:
    """返回当前空间缓存根目录；debug模式下在main中被切换到独立缓存。"""
    return s2b_module.CACHE_ROOT
from clong_s2c_core import (  # noqa: E402
    K_LIST,
    MatryoshkaSparseAutoencoder,
    SingleKView,
    fvu_from_sums,
    gamma_from_medians,
    select_min_passing_k,
    union_nondead_mask,
)
from summarize_clong_s2b import (  # noqa: E402
    MAX_AUC_DROP,
    MAX_DEAD,
    MAX_DUPLICATE,
    MAX_NAIB_DROP,
    MAX_PGA_DROP,
    MIN_AGREEMENT,
    MIN_COSINE,
    MIN_RECOVERED_CE,
)
from train_utils import git_snapshot, json_ready  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT / "结果/SAE/CLong_S2c_Matryoshka_20260821"
GAMMA_JSON_NAME = "gamma_pool_calibration_seed42.json"
GAMMA_PROTOCOL = "s2c_gamma_pool_seed42_one_balanced_epoch_74_batches_matryoshka5"
FORMAL_BUDGET = {
    "learning_rate": 1e-4,
    "epochs": 1000,
    "patience": 50,
    "warmup_fraction": 0.05,
    "image_batch_size": 32,
}


def parse_args() -> argparse.Namespace:
    """解析seed、训练预算与校准模式。

    无参数对象；返回argparse.Namespace。正式模式的预算字段会被
    ``validate_args`` 锁死到S2c冻结值，只有debug可以缩小规模。
    """
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-patients-per-class", type=int, default=2)
    parser.add_argument("--debug-hidden-dim", type=int, default=64)
    parser.add_argument("--debug-ks", default="4,8,16")
    return parser.parse_args()


def formal_name(seed: int) -> str:
    """返回S2c唯一正式实验名。"""
    return f"s2c_clong_seed{seed}"


def validate_args(args: argparse.Namespace) -> None:
    """正式运行锁死S2c预算、实验名与设备；debug与正式目录强制隔离。"""
    if args.debug:
        if args.experiment == formal_name(args.seed):
            raise ValueError("debug禁止使用正式实验名")
        args.output_root = (
            OUTPUT_ROOT / "debug" / f"seed{args.seed}_p{args.debug_patients_per_class}"
        )
        return
    if args.experiment != formal_name(args.seed):
        raise ValueError(f"正式实验名必须是{formal_name(args.seed)}")
    if args.device != "cuda":
        raise ValueError("正式S2c必须显式使用--device cuda，保持与C-long复算路径一致")
    if args.output_root.resolve() != OUTPUT_ROOT.resolve():
        raise ValueError(f"正式output_root必须固定为{OUTPUT_ROOT}")
    for field, expected in FORMAL_BUDGET.items():
        if not math.isclose(float(getattr(args, field)), expected, abs_tol=1e-12):
            raise ValueError(f"正式{field}必须等于{expected}")


def debug_k_list(args: argparse.Namespace, hidden_dim: int) -> tuple[int, ...]:
    """解析debug用缩小K-list，正式运行始终返回冻结K_LIST。"""
    if not args.debug:
        return K_LIST
    values = tuple(int(v) for v in str(args.debug_ks).split(","))
    if not values or max(values) > hidden_dim:
        raise ValueError(f"debug K-list {values}必须非空且不超过hidden_dim={hidden_dim}")
    return values


def prepare_output(args: argparse.Namespace) -> Path:
    """新建单次实验目录，已存在时拒绝覆盖。"""
    output = args.output_root / (args.experiment or f"debug_s2c_seed{args.seed}")
    if output.exists():
        raise FileExistsError(f"输出已存在，禁止覆盖: {output}")
    output.mkdir(parents=True)
    return output


def initialization_sha(sae: MatryoshkaSparseAutoencoder) -> str:
    """返回初始化state_dict的稳定SHA，用于绑定gamma_pool校准。"""
    digest = hashlib.sha256()
    for key, value in sae.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize_matryoshka(
    spatial_train: np.ndarray, metadata_train: pd.DataFrame,
    args: argparse.Namespace, device: torch.device,
) -> MatryoshkaSparseAutoencoder:
    """以train患者/类别平衡的中心初始化共享字典。

    参数:
        spatial_train (np.ndarray): train空间特征，shape [N,49,1280]。
        metadata_train (pd.DataFrame): train元数据，提供患者/类别权重。
        args (argparse.Namespace): debug规模开关。
        device (torch.device): 运行设备。

    返回:
        MatryoshkaSparseAutoencoder: 已移至device的初始化模型。
    """
    weights = patient_class_weights(metadata_train).astype(np.float64)
    center = np.average(np.asarray(spatial_train).mean(1), axis=0, weights=weights)
    hidden = args.debug_hidden_dim if args.debug else HIDDEN_DIM
    return MatryoshkaSparseAutoencoder(
        INPUT_DIM, hidden, torch.from_numpy(center.astype(np.float32)).to(device),
        debug_k_list(args, hidden),
    ).to(device)


def joint_layer_losses(
    sae: MatryoshkaSparseAutoencoder, features: torch.Tensor,
    original_pool: torch.Tensor, original_attention: torch.Tensor,
    model: torch.nn.Module, margin_vector: torch.Tensor, margin_std: float,
    gamma_pool: float,
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, torch.Tensor]]]:
    """对一批图像计算五个嵌套层的分项损失。

    参数:
        sae (MatryoshkaSparseAutoencoder): 当前模型。
        features (torch.Tensor): 空间特征，shape [B,49,1280]。
        original_pool (torch.Tensor): 原始pooled表示，shape [B,1280]。
        original_attention (torch.Tensor): 原始注意力，shape [B,49]。
        model (torch.nn.Module): 冻结C-long学生，只用其attention头。
        margin_vector (torch.Tensor): 癌减非癌margin方向，shape [1280]。
        margin_std (float): train平衡margin标准差。
        gamma_pool (float): 冻结的pool项权重。

    返回:
        tuple: ``L_i`` dict与每层 ``{patch, pool, margin}`` 分项dict。
    """
    weights = patch_position_weights(original_attention)
    totals, terms = {}, {}
    for k, (reconstructed, _hidden) in sae(features).items():
        patch = ((reconstructed - features).square().mean(2) * weights).mean()
        rebuilt_attention = attention_from_features(reconstructed, model.attention_head)
        rebuilt_pool = pooled_from_features(reconstructed, rebuilt_attention)
        pool = F.mse_loss(rebuilt_pool, original_pool)
        margin = normalized_margin_mse(
            original_pool, rebuilt_pool, margin_vector, margin_std
        )
        terms[k] = {"patch": patch, "pool": pool, "margin": margin}
        totals[k] = patch + gamma_pool * pool + GAMMA_MARGIN * margin
    return totals, terms


def joint_total(
    totals: dict[int, torch.Tensor],
) -> torch.Tensor:
    """五层uniform平均的总损失。"""
    return torch.stack(list(totals.values())).mean()


@torch.no_grad()
def calibrate_gamma(
    model: torch.nn.Module, sae: MatryoshkaSparseAutoencoder,
    train_spatial: np.ndarray, train_pooled: np.ndarray,
    train_attention: np.ndarray, train_metadata: pd.DataFrame,
    args: argparse.Namespace, device: torch.device,
) -> dict:
    """用seed42固定初始化在train-only平衡批次上冻结gamma_pool。

    每批先算 ``joint_patch=mean_K(L_patch,K)`` 与
    ``joint_pool=mean_K(L_pool,K)``，再取74批中位数冻结
    ``gamma_pool=0.25*median(joint_patch)/median(joint_pool)``。
    只看train初始化损失，不看val，禁止手工传入。
    """
    init_sha = initialization_sha(sae)
    sampler = make_sampler(train_metadata, 42)
    loader = DataLoader(TensorDataset(torch.arange(len(train_metadata))),
                        batch_size=32, sampler=sampler)
    patch_losses, pool_losses, batch_shas = [], [], []
    for (indices,) in loader:
        index_np = indices.numpy()
        current = torch.from_numpy(np.asarray(train_spatial[index_np])).to(device)
        original_pool = torch.from_numpy(np.asarray(train_pooled[index_np])).to(device)
        original_attention = torch.from_numpy(
            np.asarray(train_attention[index_np])
        ).to(device)
        _totals, terms = joint_layer_losses(
            sae, current, original_pool, original_attention, model,
            torch.zeros(sae.input_dim, device=device), 1.0, 1.0,
        )
        patch_losses.append(float(torch.stack([t["patch"] for t in terms.values()]).mean()))
        pool_losses.append(float(torch.stack([t["pool"] for t in terms.values()]).mean()))
        batch_shas.append(hashlib.sha256(index_np.astype(np.int64).tobytes()).hexdigest())
    if not args.debug and len(batch_shas) != 74:
        raise RuntimeError(f"正式gamma_pool校准必须为74批，当前{len(batch_shas)}")
    median_patch = float(np.median(patch_losses))
    median_pool = float(np.median(pool_losses))
    cache_config = json.loads((cache_root() / "cache_config.json").read_text(encoding="utf-8"))
    result = {
        "protocol": GAMMA_PROTOCOL,
        "seed": 42, "batch_size": 32, "batch_count": len(batch_shas),
        "batch_sha256": batch_shas,
        "k_list": list(sae.k_list),
        "s2c_initialization_sha256": init_sha,
        "median_joint_patch": median_patch, "median_joint_pool": median_pool,
        "gamma_pool": gamma_from_medians(median_patch, median_pool),
        "manifest_sha256": MANIFEST_SHA256,
        "student_checkpoint_sha256": CLONG_CHECKPOINT_SHA256,
        "cache_config_sha256": file_sha256(cache_root() / "cache_config.json"),
        "cache_files": cache_config["files"],
    }
    target = (args.output_root if args.debug else OUTPUT_ROOT) / GAMMA_JSON_NAME
    if target.exists():
        raise FileExistsError(f"gamma_pool JSON已存在: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(json_ready(result), ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"gamma_pool={result['gamma_pool']:.8f}; 保存至 {target}")
    return result


def load_gamma(
    args: argparse.Namespace, expected_k_list: tuple[int, ...],
    expected_initialization_sha256: str | None,
) -> tuple[float, dict]:
    """加载并逐项核验冻结的S2c gamma_pool校准JSON。

    参数:
        args (argparse.Namespace): 运行参数，决定读取正式还是debug校准文件。
        expected_k_list (tuple[int, ...]): 当前模型实际K-list；正式运行为
            冻结K_LIST，debug为缩小列表，JSON必须与之一致。
        expected_initialization_sha256 (str | None): seed42训练前重算的
            校准初始化SHA；复现seed传None，改由seed42正式config绑定校准JSON。

    返回:
        tuple[float, dict]: gamma_pool数值与完整校准记录。
    """
    path = (args.output_root if args.debug else OUTPUT_ROOT) / GAMMA_JSON_NAME
    if not path.is_file():
        raise FileNotFoundError(f"缺少gamma_pool校准JSON: {path}；先运行--calibrate-gamma")
    config = json.loads(path.read_text(encoding="utf-8"))
    cache_config = json.loads((cache_root() / "cache_config.json").read_text(encoding="utf-8"))
    recorded_batch_shas = config.get("batch_sha256", [])
    recorded_batch_count = len(recorded_batch_shas)
    recorded_gamma = float(config.get("gamma_pool", float("nan")))
    recomputed_gamma = gamma_from_medians(
        float(config.get("median_joint_patch", float("nan"))),
        float(config.get("median_joint_pool", float("nan"))),
    )
    checks = {
        "protocol": config.get("protocol") == GAMMA_PROTOCOL,
        "seed": int(config.get("seed", -1)) == 42,
        "batch_size": int(config.get("batch_size", -1)) == 32,
        "k_list": tuple(config.get("k_list", ())) == tuple(expected_k_list),
        "manifest": config.get("manifest_sha256") == MANIFEST_SHA256,
        "student": config.get("student_checkpoint_sha256") == CLONG_CHECKPOINT_SHA256,
        "cache_config": config.get("cache_config_sha256") == file_sha256(cache_root() / "cache_config.json"),
        "cache_files": config.get("cache_files") == cache_config["files"],
        "batch_count": int(config.get("batch_count", -1)) == recorded_batch_count,
        "batch_shas": recorded_batch_count > 0 and all(
            isinstance(value, str) and len(value) == 64 for value in recorded_batch_shas
        ),
        "formal_74_batches": args.debug or recorded_batch_count == 74,
        "initialization": (
            expected_initialization_sha256 is None
            or config.get("s2c_initialization_sha256") == expected_initialization_sha256
        ),
        "gamma_formula": math.isfinite(recorded_gamma)
        and math.isclose(recorded_gamma, recomputed_gamma, rel_tol=0.0, abs_tol=1e-12),
    }
    if not all(checks.values()):
        raise ValueError(f"gamma_pool JSON血缘失败: {[k for k, v in checks.items() if not v]}")
    return recorded_gamma, config


@torch.no_grad()
def evaluate_loss(
    sae: MatryoshkaSparseAutoencoder, spatial: np.ndarray, pooled: np.ndarray,
    attention: np.ndarray, metadata: pd.DataFrame, model: torch.nn.Module,
    margin_vector: torch.Tensor, margin_std: float, gamma_pool: float,
    batch_size: int, device: torch.device,
) -> dict:
    """在完整val上按患者/类别权重计算五层联合损失。

    返回 ``total``（五层平均）及 ``per_k_total``、联合分项均值，
    checkpoint只按 ``total`` 选择，不按单一K选轮次。
    """
    sae.eval()
    weights_all = patient_class_weights(metadata).astype(np.float64)
    k_list = sae.k_list
    totals = np.zeros(4, dtype=np.float64)
    per_k = np.zeros(len(k_list), dtype=np.float64)
    weight_sum = 0.0
    for start in range(0, len(pooled), batch_size):
        end = min(start + batch_size, len(pooled))
        features = torch.from_numpy(np.asarray(spatial[start:end]).copy()).to(device)
        original_pool = torch.from_numpy(np.asarray(pooled[start:end]).copy()).to(device)
        original_attention = torch.from_numpy(
            np.asarray(attention[start:end]).copy()
        ).to(device)
        weights = patch_position_weights(original_attention)
        layers = sae(features)
        k_values, patch_list, pool_list, margin_list = [], [], [], []
        for k, (reconstructed, _hidden) in layers.items():
            patch = ((reconstructed - features).square().mean(2) * weights).mean(1)
            rebuilt_attention = attention_from_features(reconstructed, model.attention_head)
            rebuilt_pool = pooled_from_features(reconstructed, rebuilt_attention)
            pool = (rebuilt_pool - original_pool).square().mean(1)
            margin = (((rebuilt_pool - original_pool) @ margin_vector) / margin_std).square()
            k_values.append(patch + gamma_pool * pool + GAMMA_MARGIN * margin)
            patch_list.append(patch)
            pool_list.append(pool)
            margin_list.append(margin)
        stacked = torch.stack(k_values, 1)  # [B,n_K]
        per_k += (stacked.cpu().numpy() * weights_all[start:end, None]).sum(0)
        rows = torch.stack((
            stacked.mean(1), torch.stack(patch_list, 1).mean(1),
            torch.stack(pool_list, 1).mean(1), torch.stack(margin_list, 1).mean(1),
        ), 1)
        totals += (rows.cpu().numpy() * weights_all[start:end, None]).sum(0)
        weight_sum += weights_all[start:end].sum()
    values = totals / weight_sum
    return {
        "total": float(values[0]), "reconstruction": float(values[1]),
        "pool": float(values[2]), "margin": float(values[3]),
        "per_k_total": {
            str(k): float(value) for k, value in zip(k_list, per_k / weight_sum)
        },
    }


def train_matryoshka(
    sae: MatryoshkaSparseAutoencoder, spatial: dict, pooled: dict,
    attention: dict, metadata: dict, model: torch.nn.Module,
    args: argparse.Namespace, device: torch.device, output: Path,
    gamma_pool: float,
) -> tuple[MatryoshkaSparseAutoencoder, dict]:
    """患者类别平衡训练，只按五层联合val总损失选checkpoint。"""
    _, _, margin = classifier_components(model)
    margin_device = margin.to(device)
    scale = margin_scale(np.asarray(pooled["train"]), metadata["train"],
                         margin.cpu().numpy())
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
        totals = np.zeros(4, dtype=np.float64)
        seen = 0
        for (indices,) in loader:
            index_np = indices.numpy()
            features = torch.from_numpy(
                np.asarray(spatial["train"][index_np]).copy()
            ).to(device)
            original_pool = torch.from_numpy(
                np.asarray(pooled["train"][index_np]).copy()
            ).to(device)
            original_attention = torch.from_numpy(
                np.asarray(attention["train"][index_np]).copy()
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            totals_k, terms = joint_layer_losses(
                sae, features, original_pool, original_attention, model,
                margin_device, scale, gamma_pool,
            )
            total = joint_total(totals_k)
            joint_patch = torch.stack([t["patch"] for t in terms.values()]).mean()
            # warmup同时线性引入pool和margin约束。
            if warmup < 1:
                total = joint_patch + warmup * (total - joint_patch)
            total.backward()
            with torch.no_grad():
                project_decoder_gradient(sae)
            optimizer.step()
            sae.normalize_decoder()
            count = len(index_np)
            totals += np.array([
                float(total.detach()), float(joint_patch.detach()),
                float(torch.stack([t["pool"] for t in terms.values()]).mean().detach()),
                float(torch.stack([t["margin"] for t in terms.values()]).mean().detach()),
            ]) * count
            seen += count
        train_values = totals / seen
        val = evaluate_loss(
            sae, spatial["val"], pooled["val"], attention["val"], metadata["val"],
            model, margin_device, scale, gamma_pool, args.image_batch_size, device,
        )
        row = {
            "epoch": epoch, "warmup": warmup,
            "learning_rate": args.learning_rate * warmup,
            "train_total": train_values[0], "train_reconstruction": train_values[1],
            "train_pool": train_values[2], "train_margin": train_values[3],
            "val_total": val["total"], "val_reconstruction": val["reconstruction"],
            "val_pool": val["pool"], "val_margin": val["margin"],
            **{f"val_total_k{k}": v for k, v in val["per_k_total"].items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} warmup={warmup:.2f} "
            f"train={train_values[0]:.5f} rec={train_values[1]:.5f} "
            f"pool={train_values[2]:.5f} margin={train_values[3]:.5f} | "
            f"val={val['total']:.5f} rec={val['reconstruction']:.5f} "
            f"pool={val['pool']:.5f} margin={val['margin']:.5f}"
        )
        if epoch < warmup_epochs:
            continue
        if val["total"] < best:
            best, stale = float(val["total"]), 0
            torch.save({
                "sae_state_dict": sae.state_dict(), "stage": "S2c",
                "seed": args.seed, "hidden_dim": sae.hidden_dim,
                "k_list": list(sae.k_list),
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


@torch.no_grad()
def l0_profile(
    view: SingleKView, spatial: np.ndarray, batch_size: int, device: torch.device,
) -> dict:
    """逐K报告mean L0与正激活不足比例（真实L0<K的位置占比）。"""
    l0_values = []
    for start in range(0, len(spatial), batch_size):
        features = torch.from_numpy(
            np.asarray(spatial[start:start + batch_size]).copy()
        ).to(device)
        hidden = view.encode(features)
        l0_values.append(hidden.gt(ACTIVE_EPS).sum(-1).cpu().numpy())
    l0 = np.concatenate(l0_values).reshape(-1)
    return {
        "mean_l0_per_position": float(l0.mean()),
        "shortfall_ratio_l0_below_k": float((l0 < view.k).mean()),
    }


def frozen_variance_reference(
    train_spatial: np.ndarray, train_pooled: np.ndarray,
) -> dict:
    """从确定性完整train计算patch/pooled的冻结均值。

    该统计只为FVU诊断提供参考基线，不进入训练、checkpoint或K选择。
    采用无重采样的完整train算术均值，避免把val统计量用于参考中心。
    """
    patch_sum = np.asarray(train_spatial).sum(axis=(0, 1), dtype=np.float64)
    patch_count = int(train_spatial.shape[0] * train_spatial.shape[1])
    pooled_sum = np.asarray(train_pooled).sum(axis=0, dtype=np.float64)
    return {
        "mu_patch_train": patch_sum / patch_count,
        "mu_pooled_train": pooled_sum / int(train_pooled.shape[0]),
    }


@torch.no_grad()
def variance_diagnostics(
    view: SingleKView, spatial: np.ndarray, pooled: np.ndarray,
    rebuilt_pool: np.ndarray, reference: dict, batch_size: int,
    device: torch.device,
) -> dict:
    """计算patch与完整替换pooled的FVU及解释方差。

    分子在冻结val上累计重构残差平方和，分母使用原始val相对完整train
    冻结均值的平方和。四项均为纯诊断字段，不参与任何选择。
    """
    patch_residual_ss = 0.0
    patch_reference_ss = 0.0
    patch_mean = np.asarray(reference["mu_patch_train"], dtype=np.float64)
    for start in range(0, len(spatial), batch_size):
        original_np = np.asarray(spatial[start:start + batch_size])
        original = torch.from_numpy(original_np.copy()).to(device)
        reconstructed = view.decode(view.encode(original)).cpu().numpy()
        patch_residual_ss += float(np.square(
            reconstructed.astype(np.float64) - original_np.astype(np.float64)
        ).sum())
        patch_reference_ss += float(np.square(
            original_np.astype(np.float64) - patch_mean
        ).sum())

    pooled_np = np.asarray(pooled, dtype=np.float64)
    rebuilt_np = np.asarray(rebuilt_pool, dtype=np.float64)
    pooled_mean = np.asarray(reference["mu_pooled_train"], dtype=np.float64)
    patch_metrics = fvu_from_sums(patch_residual_ss, patch_reference_ss)
    pooled_metrics = fvu_from_sums(
        float(np.square(rebuilt_np - pooled_np).sum()),
        float(np.square(pooled_np - pooled_mean).sum()),
    )
    return {
        "patch_fvu": {"value": patch_metrics["fvu"], "diagnostic_only": True},
        "patch_explained_variance": {
            "value": patch_metrics["explained_variance"], "diagnostic_only": True,
        },
        "pooled_fvu": {"value": pooled_metrics["fvu"], "diagnostic_only": True},
        "pooled_explained_variance": {
            "value": pooled_metrics["explained_variance"], "diagnostic_only": True,
        },
    }


def value_strata(
    metadata: pd.DataFrame, values: np.ndarray, lesion_bounds: np.ndarray,
) -> dict:
    """按冻结标签、来源、分辨率、画中画和病灶大小汇总逐图指标。"""
    frame = metadata.copy()
    frame["_value"] = np.asarray(values, dtype=float)
    result: dict[str, dict] = {}
    for column in ("label", "source", "size_group", "pip_present_original"):
        if column not in frame:
            continue
        result[column] = {}
        for name, group in frame.groupby(frame[column].fillna("__MISSING__")):
            result[column][str(name)] = {
                "images": int(len(group)), "mean": float(group["_value"].mean()),
            }
    cancer = frame.loc[frame.label.eq(1)].copy()
    result["lesion_size"] = {}
    if len(cancer) and "bbox_area_fraction" in cancer:
        cancer["_lesion"] = pd.cut(
            cancer.bbox_area_fraction,
            [-np.inf, lesion_bounds[0], lesion_bounds[1], np.inf],
            labels=["small", "medium", "large"], include_lowest=True,
        )
        for name, group in cancer.groupby("_lesion", observed=True):
            result["lesion_size"][str(name)] = {
                "images": int(len(group)), "mean": float(group["_value"].mean()),
            }
    return result


@torch.no_grad()
def fixed_attention_pooled(
    view: SingleKView, spatial: np.ndarray, attention: np.ndarray,
    batch_size: int, device: torch.device,
) -> np.ndarray:
    """固定原注意力只加权重构内容，作为每K的诊断口径（不进门槛）。"""
    rows = []
    for start in range(0, len(spatial), batch_size):
        original = torch.from_numpy(
            np.asarray(spatial[start:start + batch_size]).copy()
        ).to(device)
        original_attn = torch.from_numpy(
            np.asarray(attention[start:start + batch_size]).copy()
        ).to(device)
        rows.append(
            pooled_from_features(view.decode(view.encode(original)), original_attn)
            .cpu().numpy()
        )
    return np.concatenate(rows)


def threshold_band_agreement(patient: pd.DataFrame, band: float) -> dict:
    """报告原始概率在冻结阈值±band内患者的一致率（诊断指标）。"""
    near = patient.loc[
        (patient.original - FROZEN_PATIENT_THRESHOLD).abs() <= band
    ]
    if not len(near):
        return {"patients": 0, "agreement": None}
    agree = (
        (near.original >= FROZEN_PATIENT_THRESHOLD)
        == (near.reconstructed >= FROZEN_PATIENT_THRESHOLD)
    ).mean()
    return {"patients": int(len(near)), "agreement": float(agree)}


def replication_contract(args: argparse.Namespace) -> tuple[int | None, dict | None]:
    """冻结seed202/503复现目标K，禁止在seed42产品产生前越级运行。

    seed42或debug返回 ``(None, None)``。正式复现seed必须读取seed42 config，
    核验其产品状态、数据锁、血缘和gamma JSON SHA，并返回唯一冻结K。
    """
    if args.debug or args.seed == 42:
        return None, None
    seed42_dir = OUTPUT_ROOT / formal_name(42)
    config_path = seed42_dir / "config.json"
    checkpoint_path = seed42_dir / "sae_best.pth"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("seed202/503只能在seed42正式产品完整落盘后运行")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gamma_path = OUTPUT_ROOT / GAMMA_JSON_NAME
    selected = config.get("evaluation", {}).get("selected_k")
    checks = {
        "status": config.get("status") == "s2c_k_selected",
        "seed": int(config.get("seed", -1)) == 42,
        "selected_k": selected in K_LIST,
        "student": config.get("student_checkpoint_sha256") == CLONG_CHECKPOINT_SHA256,
        "manifest": config.get("manifest_sha256") == MANIFEST_SHA256,
        "seed42_checkpoint": config.get("checkpoint_sha256") == file_sha256(checkpoint_path),
        "gamma_json": gamma_path.is_file() and config.get(
            "gamma_calibration_json_sha256"
        ) == file_sha256(gamma_path),
        "test_locked": config.get("test_evaluated") is False,
        "internal_locked": config.get("internal_test_evaluated") is False,
        "external_locked": config.get("external_evaluated") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"seed42复现交接失败: {[k for k, v in checks.items() if not v]}")
    return int(selected), config


def evaluate_all_k(
    sae: MatryoshkaSparseAutoencoder, spatial: dict, pooled: dict,
    attention: dict, metadata: dict, model: torch.nn.Module,
    args: argparse.Namespace, device: torch.device,
    replication_target_k: int | None = None,
) -> dict:
    """按K升序对每个K独立执行完整替换评价并判定八项门槛。

    返回 ``per_k`` 评价、五层并集的死亡/重复率、门槛判定与最小合格K。
    """
    weight, bias, _ = classifier_components(model)
    weight_np, bias_np = weight.cpu().numpy(), bias.cpu().numpy()
    labels = metadata["val"].label.to_numpy(int)
    bboxes = metadata["val"][
        ["bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm"]
    ].to_numpy(float)
    train_cancer_area = metadata["train"].loc[
        metadata["train"].label.eq(1), "bbox_area_fraction"
    ].to_numpy(float)
    lesion_bounds = np.quantile(train_cancer_area, [1 / 3, 2 / 3])
    original_spatial = spatial_alignment_metrics(attention["val"], labels, bboxes)
    variance_reference = frozen_variance_reference(
        spatial["train"], pooled["train"],
    )

    per_k, train_position_counts = {}, {}
    for k in sae.k_list:
        view = SingleKView(sae, k)
        _train_pool, _train_attn, train_cov = project_patch(
            view, spatial["train"], attention["train"], metadata["train"], model,
            None, args.image_batch_size, device, False,
        )
        train_position_counts[k] = np.asarray(
            train_cov["feature_active_position_counts"]
        )
        rebuilt_pool, rebuilt_attention, val_cov = project_patch(
            view, spatial["val"], attention["val"], metadata["val"], model,
            None, args.image_batch_size, device, False,
        )
        metrics = classification_metrics(
            np.asarray(pooled["val"]), rebuilt_pool, metadata["val"],
            weight_np, bias_np, FROZEN_PATIENT_THRESHOLD,
        )
        metrics.update(fidelity_extras(
            np.asarray(pooled["val"]), rebuilt_pool, metadata["val"],
            weight_np, bias_np,
        ))
        metrics["patient_auc_drop"] = (
            metrics["patient_original_auc"] - metrics["patient_reconstructed_auc"]
        )
        rebuilt_spatial = spatial_alignment_metrics(rebuilt_attention, labels, bboxes)
        fixed_pool = fixed_attention_pooled(
            view, spatial["val"], attention["val"], args.image_batch_size, device,
        )
        original_probability = torch.softmax(torch.from_numpy(
            np.asarray(pooled["val"]) @ weight_np.T + bias_np
        ), 1)[:, 1].numpy()
        rebuilt_probability = torch.softmax(torch.from_numpy(
            rebuilt_pool @ weight_np.T + bias_np
        ), 1)[:, 1].numpy()
        patient = patient_frame(
            metadata["val"], original_probability, rebuilt_probability,
        )
        pooled_cosine_per_image = np.sum(
            np.asarray(pooled["val"]) * rebuilt_pool, axis=1,
        ) / (
            np.linalg.norm(np.asarray(pooled["val"]), axis=1)
            * np.linalg.norm(rebuilt_pool, axis=1) + 1e-12
        )
        coverage_scalars = {
            key: value for key, value in val_cov.items()
            if key not in {
                "feature_active_position_counts", "feature_active_image_counts",
                "feature_active_patient_counts", "per_image_mean_patch_cosine",
            }
        }
        per_k[k] = {
            "fidelity": metrics,
            "sparsity": {
                "train": l0_profile(view, spatial["train"], args.image_batch_size, device),
                "val": l0_profile(view, spatial["val"], args.image_batch_size, device),
                "train_used_feature_count": int((train_position_counts[k] > 0).sum()),
                "train_used_feature_ratio": float((train_position_counts[k] > 0).mean()),
                "mean_unique_features_per_image": val_cov["mean_unique_features_per_image"],
                "mean_patch_cosine": val_cov["mean_patch_cosine"],
                "patch_cosine_strata": patch_cosine_strata(
                    metadata["val"], val_cov["per_image_mean_patch_cosine"], lesion_bounds
                ),
                "patch_cosine_extended_strata": value_strata(
                    metadata["val"], val_cov["per_image_mean_patch_cosine"], lesion_bounds,
                ),
                "pooled_cosine_strata": value_strata(
                    metadata["val"], pooled_cosine_per_image, lesion_bounds,
                ),
                "coverage_diagnostics": coverage_scalars,
                "attention_drift": attention_drift_metrics(
                    attention["val"], rebuilt_attention
                ),
                "spatial_original": original_spatial,
                "spatial_reconstructed": rebuilt_spatial,
                "normalized_aib_drop": (
                    original_spatial["mean_normalized_aib"]
                    - rebuilt_spatial["mean_normalized_aib"]
                ),
                "pga_drop": original_spatial["pga"] - rebuilt_spatial["pga"],
            },
            "fixed_original_attention_diagnostic": classification_metrics(
                np.asarray(pooled["val"]), fixed_pool, metadata["val"],
                weight_np, bias_np, FROZEN_PATIENT_THRESHOLD,
            ),
            "threshold_band_agreement": {
                "pm0_05": threshold_band_agreement(patient, 0.05),
                "pm0_10": threshold_band_agreement(patient, 0.10),
            },
            "variance_diagnostics": variance_diagnostics(
                view, spatial["val"], pooled["val"], rebuilt_pool,
                variance_reference, args.image_batch_size, device,
            ),
        }
        print(f"K={k}评价完成: cosine={metrics['mean_cosine']:.5f} "
              f"agreement={metrics['patient_prediction_agreement_at_locked_threshold']:.5f}")

    nondead = union_nondead_mask(train_position_counts)
    duplicate = duplicate_decoder_rate_nondead(
        sae.decoder_weight.detach().cpu().numpy(), nondead
    )
    union = {
        "dead_feature_count": int((~nondead).sum()),
        "dead_feature_rate": float((~nondead).mean()),
        "definition": "五层K的train激活并集中从未激活",
        "duplicate": duplicate,
    }
    gates = {}
    for k, result in per_k.items():
        fidelity = result["fidelity"]
        sparsity = result["sparsity"]
        gates[k] = {
            "patient_auc_drop_le_0_01": fidelity["patient_auc_drop"] <= MAX_AUC_DROP,
            "patient_agreement_ge_0_95": (
                fidelity["patient_prediction_agreement_at_locked_threshold"] >= MIN_AGREEMENT
            ),
            "pooled_cosine_ge_0_90": fidelity["mean_cosine"] >= MIN_COSINE,
            "recovered_ce_ge_0_95": (
                fidelity["recovered_cross_entropy"] >= MIN_RECOVERED_CE
            ),
            "dead_rate_le_0_10": union["dead_feature_rate"] <= MAX_DEAD,
            "duplicate_rate_le_0_10": duplicate["duplicate_rate"] <= MAX_DUPLICATE,
            "normalized_aib_drop_le_0_05": (
                sparsity["normalized_aib_drop"] <= MAX_NAIB_DROP
            ),
            "pga_drop_le_0_05": sparsity["pga_drop"] <= MAX_PGA_DROP,
        }
    minimum_passing_k = select_min_passing_k(gates)
    if replication_target_k is None:
        selected = minimum_passing_k
        selection_rule = "seed42同时通过八项门槛的最小K，不二次挑选"
    else:
        if replication_target_k not in gates:
            raise ValueError(f"复现目标K={replication_target_k}不在冻结K-list")
        selected = (
            replication_target_k
            if all(gates[replication_target_k].values()) else None
        )
        selection_rule = (
            f"复现seed只判定seed42冻结K={replication_target_k}，不得重新选择K"
        )

    selected_feature_coverage = None
    if selected is not None:
        selected_feature_coverage = {}
        selected_view = SingleKView(sae, selected)
        for split in ("train", "val"):
            _pool, _attention, coverage = project_patch(
                selected_view, spatial[split], attention[split], metadata[split],
                model, None, args.image_batch_size, device, True,
            )
            selected_feature_coverage[split] = {
                "feature_active_position_counts": coverage["feature_active_position_counts"],
                "feature_active_image_counts": coverage["feature_active_image_counts"],
                "feature_active_patient_counts": coverage["feature_active_patient_counts"],
            }

    cache_config_path = cache_root() / "cache_config.json"
    return {
        "per_k": {str(k): value for k, value in per_k.items()},
        "union_dictionary_health": union,
        "gates": {str(k): value for k, value in gates.items()},
        "selected_k": selected,
        "seed42_minimum_passing_k_diagnostic": minimum_passing_k,
        "replication_target_k": replication_target_k,
        "selection_rule": selection_rule,
        "selected_k_feature_coverage": selected_feature_coverage,
        "variance_reference": {
            "mu_patch_train": variance_reference["mu_patch_train"].tolist(),
            "mu_patch_train_shape": list(variance_reference["mu_patch_train"].shape),
            "mu_pooled_train": variance_reference["mu_pooled_train"].tolist(),
            "mu_pooled_train_shape": list(variance_reference["mu_pooled_train"].shape),
            "source_cache_config_sha256": file_sha256(cache_config_path),
            "source": "deterministic_full_train_unweighted_mean",
            "diagnostic_only": True,
        },
    }


def main() -> None:
    """执行血缘校验、校准/训练、逐K评价、K选择与产物落盘。"""
    args = parse_args()
    validate_args(args)
    if args.debug:
        s2b_module.CACHE_ROOT = args.output_root / "frozen_spatial_cache"
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    lineage = verify_s0_lineage()
    frame = load_manifest_frame(args)
    model = load_clong_model(device)
    if not (cache_root() / "cache_config.json").exists():
        s2b_module.build_spatial_cache(model, frame, args, device, lineage)
    spatial, pooled, attention, metadata = load_cache(frame)
    # SAE初始化必须与“本次是否先创建缓存”无关，否则校准与训练的seed42 SHA会漂移。
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sae = initialize_matryoshka(spatial["train"], metadata["train"], args, device)
    if args.calibrate_gamma:
        if args.seed != 42:
            raise ValueError("gamma_pool只能由seed42初始化校准")
        calibrate_gamma(
            model, sae, spatial["train"], pooled["train"], attention["train"],
            metadata["train"], args, device,
        )
        return
    replication_target_k, seed42_config = replication_contract(args)
    expected_initialization_sha = initialization_sha(sae) if args.seed == 42 else None
    gamma_pool, gamma_config = load_gamma(
        args, sae.k_list, expected_initialization_sha,
    )
    output = prepare_output(args)
    sae, checkpoint = train_matryoshka(
        sae, spatial, pooled, attention, metadata, model, args, device, output,
        gamma_pool,
    )
    evaluation = evaluate_all_k(sae, spatial, pooled, attention, metadata, model,
                                args, device, replication_target_k)
    selected = evaluation["selected_k"]
    payload = {
        "stage": "S2c", "seed": args.seed,
        "experiment": args.experiment, "debug": args.debug,
        "student_checkpoint_sha256": CLONG_CHECKPOINT_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "cache_config_sha256": file_sha256(cache_root() / "cache_config.json"),
        "checkpoint_sha256": file_sha256(output / "sae_best.pth"),
        "architecture": {
            "input": "shared_patch_49x1280", "hidden_dim": sae.hidden_dim,
            "k_list": list(sae.k_list),
            "activation_mode": "matryoshka_nested_topk_single_sort",
            "nesting": "同一正预激活降序排序的递增前缀mask",
        },
        "training": {
            "learning_rate": args.learning_rate, "epochs": args.epochs,
            "patience": args.patience, "warmup_fraction": args.warmup_fraction,
            "image_batch_size": args.image_batch_size,
            "gamma_margin": GAMMA_MARGIN, "gamma_pool": gamma_pool,
            "layer_weighting": "uniform_1_over_5",
            "best_epoch": checkpoint["epoch"],
        },
        "gamma_calibration_json_sha256": file_sha256(
            (args.output_root if args.debug else OUTPUT_ROOT) / GAMMA_JSON_NAME
        ),
        "evaluation": evaluation,
        "replication_contract": {
            "target_k": replication_target_k,
            "seed42_config_sha256": (
                file_sha256(OUTPUT_ROOT / formal_name(42) / "config.json")
                if seed42_config is not None else None
            ),
        },
        "status": (
            "s2c_k_selected" if args.seed == 42 and selected is not None else
            "no_product_stop_s2c" if args.seed == 42 else
            "s2c_replication_pass" if selected is not None else
            "s2c_replication_fail"
        ),
        "next_step": (
            f"冻结K={selected}并用SAE seed202/503在同一特征缓存上复现"
            if args.seed == 42 and selected is not None else
            "停止S2c；不放宽门槛、不追加K-list/reverse weighting/Gated SAE到本实验"
            if args.seed == 42 else
            "汇总三SAE seed的冻结K复现结果，进入跨seed Feature对齐"
            if selected is not None else
            "如实记录该SAE seed复现失败，不重新选择其他K"
        ),
        "test_evaluated": False, "internal_test_evaluated": False,
        "external_evaluated": False,
        "git": git_snapshot(),
    }
    (output / "config.json").write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"输出目录: {output}")
    print(f"状态: {payload['status']}; selected_k={selected}")


if __name__ == "__main__":
    main()
