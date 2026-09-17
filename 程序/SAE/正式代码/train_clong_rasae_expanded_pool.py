#!/usr/bin/env python3
"""固定计算预算，仅扩大RA-SAE每轮可抽样的train患者池。

候选从原pilot的确定性初始状态重新开始，固定代表点、中心、encoder解析校准、
模型结构、损失、K、优化器起点和100轮×13次更新预算。唯一改变是每轮有放回抽样
的候选池从196图/128人扩展到每人最多2图的1673图/1212人。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_clong_rasae_medical_feedback import load_models  # noqa: E402
from clong_rasae_core import ArchetypalMatryoshkaSAE, calibrate_initial_encoder  # noqa: E402
from clong_s2b_core import attention_from_features, pooled_from_features  # noqa: E402
from clong_s2c_matryoshka import joint_layer_losses  # noqa: E402
from clong_sae_discovery import patient_class_weights  # noqa: E402
from run_clong_rasae_pilot import read_subset  # noqa: E402


BASE = ROOT / "结果/SAE/RA_SAE_Pilot_20260908"
CACHE = ROOT / "结果/SAE/CLong_S2b结构重构_20260820/frozen_spatial_cache"
REFERENCE = BASE / "duration100"
SEED = 1701
FORMAL_EPOCHS = 100
FORMAL_DRAWS_PER_EPOCH = 196
BATCH_SIZE = 16


def expanded_train_pool() -> pd.DataFrame:
    """构建完整train中每位患者最多2张图的固定候选池。

    Args:
        None.

    Returns:
        pd.DataFrame: 1673图/1212人，``source_row``指向冻结train缓存。
    """
    full = pd.read_csv(CACHE / "train_metadata.csv").reset_index().rename(columns={"index": "source_row"})
    if full.groupby("patient_id").label.nunique().max() != 1:
        raise RuntimeError("train存在跨标签患者")
    pool = (full.sort_values(["patient_id", "relative_path"], kind="stable")
            .groupby("patient_id", sort=False).head(2).reset_index(drop=True))
    if len(pool) != 1673 or pool.patient_id.nunique() != 1212:
        raise RuntimeError(f"扩大池不是1673图/1212人: {len(pool)}/{pool.patient_id.nunique()}")
    return pool


def sampling_orders(
    pool: pd.DataFrame, image_weights: np.ndarray, epochs: int, draws_per_epoch: int,
) -> tuple[list[np.ndarray], pd.DataFrame, dict]:
    """使用原方式每轮有放回抽样，并记录实际患者覆盖。

    Args:
        pool (pd.DataFrame): 扩大后的候选池。
        image_weights (np.ndarray): 重算后的患者/类别平衡图像权重。
        epochs (int): 训练轮数。
        draws_per_epoch (int): 每轮有放回抽样图像数。

    Returns:
        tuple: 每轮索引、逐轮覆盖表和100轮累计覆盖摘要。
    """
    if len(image_weights) != len(pool) or not np.isclose(image_weights.sum(), 1):
        raise ValueError("抽样权重与扩大池不一致")
    rng = np.random.default_rng(SEED)
    orders = [rng.choice(len(pool), draws_per_epoch, replace=True, p=image_weights)
              for _ in range(epochs)]
    cumulative_images: set[int] = set()
    cumulative_patients: set[str] = set()
    rows = []
    for epoch, order in enumerate(orders, 1):
        sampled = pool.iloc[order]
        cumulative_images.update(map(int, sampled.source_row))
        cumulative_patients.update(map(str, sampled.patient_id))
        unique_patients = sampled[["patient_id", "label"]].drop_duplicates()
        rows.append({
            "epoch": epoch, "draws": len(order), "optimizer_steps": int(np.ceil(len(order) / BATCH_SIZE)),
            "last_batch_size": int(len(order) % BATCH_SIZE or BATCH_SIZE),
            "unique_images_this_epoch": int(sampled.source_row.nunique()),
            "unique_patients_this_epoch": int(sampled.patient_id.nunique()),
            "cancer_patients_this_epoch": int(unique_patients.label.sum()),
            "noncancer_patients_this_epoch": int((unique_patients.label == 0).sum()),
            "cumulative_unique_images": len(cumulative_images),
            "cumulative_unique_patients": len(cumulative_patients),
        })
    sampled_all = pool.iloc[np.concatenate(orders)]
    patients_all = sampled_all[["patient_id", "label"]].drop_duplicates()
    summary = {
        "candidate_pool_images": len(pool), "candidate_pool_patients": int(pool.patient_id.nunique()),
        "draws_total": int(epochs * draws_per_epoch),
        "optimizer_steps_total": int(epochs * np.ceil(draws_per_epoch / BATCH_SIZE)),
        "last_batch_size": int(draws_per_epoch % BATCH_SIZE or BATCH_SIZE),
        "sampled_unique_images": int(sampled_all.source_row.nunique()),
        "sampled_unique_patients": int(sampled_all.patient_id.nunique()),
        "sampled_unique_cancer_patients": int(patients_all.label.sum()),
        "sampled_unique_noncancer_patients": int((patients_all.label == 0).sum()),
    }
    return orders, pd.DataFrame(rows), summary


def verify_initialization(
    model: ArchetypalMatryoshkaSAE, pilot_data: dict, pilot_frame: pd.DataFrame,
    batch_size: int,
) -> dict:
    """用原pilot数据恢复encoder校准，并逐项对照原初始化记录。

    Args:
        model (ArchetypalMatryoshkaSAE): 刚刚从原代表点/中心确定性构造的RA模型。
        pilot_data (dict): 原196张pilot的spatial/attention张量。
        pilot_frame (pd.DataFrame): 原pilot元数据。
        batch_size (int): 校准批量。

    Returns:
        dict: 新计算初始化记录与对照误差。
    """
    weights = patient_class_weights(pilot_frame).astype(np.float64)
    weights /= weights.sum()
    current = calibrate_initial_encoder(
        model, pilot_data["spatial"], pilot_data["attention"],
        torch.as_tensor(weights, device=pilot_data["spatial"].device), batch_size,
    )
    reference = json.loads((REFERENCE / "ra_initialization.json").read_text())
    differences = {"scale": abs(current["scale"] - reference["scale"])}
    for k in ("64", "128", "256"):
        for metric in ("before", "after", "center_only"):
            differences[f"{k}_{metric}"] = abs(
                current["per_k"][k][metric] - reference["per_k"][k][metric]
            )
    if max(differences.values()) > 1e-5:
        raise RuntimeError(f"初始化记录未复原: {differences}")
    if bool(model.relaxation.ne(0).any()):
        raise RuntimeError("新RA模型的初始松弛项不是0")
    return {"recomputed": current, "reference": reference, "absolute_differences": differences,
            "initialization_record_matches": True, "relaxation_zero": True}


def main() -> None:
    """运行扩大可抽样池的单RA臂固定预算训练。

    Args:
        None: 输出、设备和debug由CLI提供。

    Returns:
        None: 写入配置、初始化核对、抽样覆盖、历史和最终checkpoint。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    epochs = 1 if args.debug else FORMAL_EPOCHS
    draws_per_epoch = 20 if args.debug else FORMAL_DRAWS_PER_EPOCH
    old_config = json.loads((REFERENCE / "config.json").read_text())
    config = {
        "scope": "expanded_optimization_sampling_pool_single_factor",
        "seed": SEED, "epochs": epochs, "draws_per_epoch": draws_per_epoch,
        "batch": BATCH_SIZE, "updates_per_epoch": int(np.ceil(draws_per_epoch / BATCH_SIZE)),
        "hidden_dim": 2560, "k_list": [64, 128, 256], "delta": 0.2,
        "learning_rate": 1e-4, "weight_decay": 0.0, "warmup_epochs": 2,
        "gamma_pool": old_config["gamma_pool"], "margin_std": old_config["margin_std"],
        "initialization": "unique", "encoder_step_policy": "base_lr_times_original_pilot_scale",
        "representative_points": "fixed from current duration100 RA checkpoint",
        "feature_center": "fixed from current duration100 RA checkpoint",
        "initial_encoder_calibration": "recomputed on original pilot196 and matched to saved record",
        "single_changed_factor": "available optimization sampling pool",
        "checkpoint_rule": "fixed final epoch; no val selection",
        "debug": args.debug, "new_training": True, "test_read": False, "external_read": False,
    }
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    device = torch.device(args.device)
    pool = expanded_train_pool()
    if args.debug:
        keep_patients = pool[["patient_id", "label"]].drop_duplicates().groupby("label", group_keys=False).head(4)
        pool = pool[pool.patient_id.isin(keep_patients.patient_id)].reset_index(drop=True)
    pool.to_csv(args.output / "expanded_train_pool.csv", index=False)
    pilot_frame = pd.read_csv(REFERENCE / "train_subset.csv")
    pilot_data = read_subset(pilot_frame, "train", device)
    train_data = read_subset(pool, "train", device)
    current_sae, head, classifier_weight, _ = load_models(device)
    current_checkpoint = torch.load(REFERENCE / "ra_final.pth", map_location=device, weights_only=False)
    state = current_checkpoint["state_dict"]
    model = ArchetypalMatryoshkaSAE(
        state["points"], state["decoder_bias"], 2560, (64, 128, 256), 0.2,
        True, SEED, "unique",
    ).to(device)
    initialization = verify_initialization(model, pilot_data, pilot_frame, BATCH_SIZE)
    (args.output / "initialization_verification.json").write_text(
        json.dumps(initialization, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del current_sae, pilot_data
    weights = patient_class_weights(pool).astype(np.float64)
    weights /= weights.sum()
    orders, sampling_frame, sampling_summary = sampling_orders(
        pool, weights, epochs, draws_per_epoch
    )
    sampling_frame.to_csv(args.output / "sampling_coverage_by_epoch.csv", index=False)
    (args.output / "sampling_summary.json").write_text(
        json.dumps(sampling_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.debug:
        if sampling_summary["optimizer_steps_total"] != 1300 or sampling_summary["last_batch_size"] != 4:
            raise RuntimeError("正式更新预算不是1300次且末批4张")
    max_attention_error = max_pool_error = 0.0
    with torch.no_grad():
        for start in range(0, len(pool), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(pool))
            spatial = train_data["spatial"][start:end]
            attention = attention_from_features(spatial, head)
            pooled = pooled_from_features(spatial, attention)
            max_attention_error = max(max_attention_error, float(
                (attention - train_data["attention"][start:end]).abs().max()
            ))
            max_pool_error = max(max_pool_error, float(
                (pooled - train_data["pooled"][start:end]).abs().max()
            ))
    if max(max_attention_error, max_pool_error) > 1e-4:
        raise RuntimeError("扩大池缓存与下游复算不一致")
    encoder_lr_scale = initialization["recomputed"]["scale"]
    optimizer = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr_scale": encoder_lr_scale},
        {"params": [parameter for name, parameter in model.named_parameters()
                    if not name.startswith("encoder.")], "lr_scale": 1.0},
    ], lr=config["learning_rate"], weight_decay=0)
    margin = classifier_weight[1] - classifier_weight[0]
    history = []
    for epoch, order in enumerate(orders, 1):
        model.train()
        warmup = min(1.0, epoch / config["warmup_epochs"])
        for group in optimizer.param_groups:
            group["lr"] = config["learning_rate"] * warmup * group["lr_scale"]
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            indices = torch.as_tensor(order[start:start + BATCH_SIZE], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss_k, terms = joint_layer_losses(
                model, train_data["spatial"][indices], train_data["pooled"][indices],
                train_data["attention"][indices], SimpleNamespace(attention_head=head),
                margin, config["margin_std"], config["gamma_pool"],
            )
            loss = torch.stack(list(loss_k.values())).mean()
            patch = torch.stack([term["patch"] for term in terms.values()]).mean()
            loss = patch + warmup * (loss - patch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"epoch{epoch}损失非有限")
            loss.backward()
            optimizer.step()
            model.normalize_decoder()
            total_loss += float(loss.detach()) * len(indices)
        row = {"epoch": epoch, "train_loss": total_loss / len(order),
               "draws": len(order), "optimizer_steps": int(np.ceil(len(order) / BATCH_SIZE))}
        history.append(row)
        pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
        print(json.dumps(row), flush=True)
    checkpoint = {"state_dict": model.state_dict(), "config": config,
                  "sampling_summary": sampling_summary,
                  "initialization_verification": initialization}
    if not args.debug:
        torch.save(checkpoint, args.output / "coverage_ra_final.pth")
    verification = {
        "initialization_record_matches": initialization["initialization_record_matches"],
        "relaxation_zero_at_start": initialization["relaxation_zero"],
        "optimizer_newly_created": True,
        "loaded_trained_parameters": False,
        "max_attention_cache_error": max_attention_error,
        "max_pool_cache_error": max_pool_error,
        "fixed_final_epoch": epochs,
        "test_read": False, "external_read": False,
    }
    (args.output / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"sampling": sampling_summary, "verification": verification}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
