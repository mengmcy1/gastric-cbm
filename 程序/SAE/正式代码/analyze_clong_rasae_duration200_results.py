#!/usr/bin/env python3
"""汇总扩大池同一连续轨迹第100轮与第200轮的固定终点结果。

复用共同train1673图的原损失/注意力路径口径，分开原pilot和新增患者；
同时读取已完成的完整val第100/200轮结果做预定方向判定。不训练、不选epoch。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_clong_rasae_coverage_loss_paths import (  # noqa: E402
    aggregate_metrics,
    common_cohort,
    evaluate_model,
    evaluation_weights,
    paired_differences,
)
from analyze_clong_rasae_medical_feedback import load_models  # noqa: E402
from clong_rasae_core import ArchetypalMatryoshkaSAE  # noqa: E402
from run_clong_rasae_pilot import read_subset  # noqa: E402


BASE = ROOT / "结果/SAE/RA_SAE_Pilot_20260908"
DURATION_ROOT = BASE / "expanded_pool_duration200_20260916"
K_LIST = (64, 128, 256)


def load_sae(path: Path, device: torch.device) -> ArchetypalMatryoshkaSAE:
    """从完整续训checkpoint读取冻结RA-SAE。

    Args:
        path (Path): 含``state_dict``和``config``的checkpoint。
        device (torch.device): 评价设备。

    Returns:
        ArchetypalMatryoshkaSAE: 宽2560、三层K的冻结模型。
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state, config = checkpoint["state_dict"], checkpoint["config"]
    if config["hidden_dim"] != 2560 or tuple(config["k_list"]) != K_LIST:
        raise RuntimeError(f"checkpoint结构不符合预期: {path}")
    model = ArchetypalMatryoshkaSAE(
        state["points"], state["decoder_bias"], config["hidden_dim"],
        tuple(config["k_list"]), config["delta"], True, config["seed"],
        config["initialization"],
    ).to(device)
    model.load_state_dict(state)
    model.requires_grad_(False).eval()
    return model


def get(source: dict, path: tuple[str, ...]) -> float:
    """按键路径读取JSON数值。

    Args:
        source (dict): 已解析JSON。
        path (tuple[str, ...]): 递归键路径。

    Returns:
        float: 目标数值。
    """
    value = source
    for key in path:
        value = value[key]
    return float(value)


def compare_val() -> dict:
    """比较同一轨迹第100轮与第200轮完整val结果。

    Args:
        None.

    Returns:
        dict: 五项预定指标、方向条件和联合状态。
    """
    old = json.loads((DURATION_ROOT / "full_val_epoch100/summary.json").read_text())
    new = json.loads((DURATION_ROOT / "full_val_epoch200/summary.json").read_text())
    paths = {
        "patch_fvu": ("reconstruction", "patch_fvu"),
        "pooled_fvu": ("reconstruction", "pooled_fvu"),
        "patient_probability_abs_change_mean": (
            "prediction", "patient_probability_abs_change", "mean"
        ),
        "patient_prediction_agreement": ("prediction", "patient_prediction_agreement"),
        "reconstructed_patient_auc": ("prediction", "reconstructed_patient_auc"),
    }
    metrics = {}
    for name, path in paths.items():
        value100, value200 = get(old, path), get(new, path)
        metrics[name] = {
            "epoch100": value100, "epoch200": value200,
            "delta_epoch200_minus_epoch100": value200 - value100,
        }
    conditions = {
        "patch_fvu_lower": metrics["patch_fvu"]["epoch200"] < metrics["patch_fvu"]["epoch100"],
        "pooled_fvu_lower": metrics["pooled_fvu"]["epoch200"] < metrics["pooled_fvu"]["epoch100"],
        "patient_probability_abs_change_not_higher": (
            metrics["patient_probability_abs_change_mean"]["epoch200"]
            <= metrics["patient_probability_abs_change_mean"]["epoch100"]
        ),
        "patient_prediction_agreement_not_lower": (
            metrics["patient_prediction_agreement"]["epoch200"]
            >= metrics["patient_prediction_agreement"]["epoch100"]
        ),
        "reconstructed_patient_auc_not_lower": (
            metrics["reconstructed_patient_auc"]["epoch200"]
            >= metrics["reconstructed_patient_auc"]["epoch100"]
        ),
    }
    return {
        "metrics": metrics, "conditions": conditions,
        "conditions_met": int(sum(conditions.values())), "conditions_total": len(conditions),
        "joint_status": (
            "representation_improved_but_prediction_fidelity_not_fully_preserved"
            if conditions["patch_fvu_lower"] and conditions["pooled_fvu_lower"]
            and not all(conditions.values()) else
            "joint_target_met" if all(conditions.values()) else "joint_target_not_met"
        ),
    }


def main() -> None:
    """执行train分层及val固定终点汇总。

    Args:
        None: 输出、设备和批量由CLI提供。

    Returns:
        None: 写入train逐图/聚合表、val比较及验证JSON。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame, _pilot_patients = common_cohort()
    weights = evaluation_weights(frame)
    device = torch.device(args.device)
    data = read_subset(frame, "train", device)
    unused, head, classifier_weight, _classifier_bias = load_models(device)
    del unused
    config = json.loads((DURATION_ROOT / "config.json").read_text())
    margin_vector = classifier_weight[1] - classifier_weight[0]
    epoch100 = load_sae(DURATION_ROOT / "replay_epoch100_full.pth", device)
    rows100, verify100 = evaluate_model(
        "baseline", epoch100, data, frame, head, margin_vector,
        float(config["margin_std"]), float(config["gamma_pool"]), args.batch_size,
    )
    del epoch100
    torch.cuda.empty_cache() if device.type == "cuda" else None
    epoch200 = load_sae(DURATION_ROOT / "duration200_full.pth", device)
    rows200, verify200 = evaluate_model(
        "expanded_pool", epoch200, data, frame, head, margin_vector,
        float(config["margin_std"]), float(config["gamma_pool"]), args.batch_size,
    )
    images = pd.concat([rows100, rows200], ignore_index=True)
    aggregate = aggregate_metrics(images, weights, frame)
    differences = paired_differences(aggregate)
    images["checkpoint"] = images.checkpoint.replace({
        "baseline": "replay_epoch100", "expanded_pool": "duration200"
    })
    aggregate["checkpoint"] = aggregate.checkpoint.replace({
        "baseline": "replay_epoch100", "expanded_pool": "duration200"
    })
    rename = {
        column: column.replace("_baseline", "_epoch100").replace("_expanded", "_epoch200")
        for column in differences.columns
    }
    differences = differences.rename(columns=rename)
    images.to_csv(args.output / "train_image_metrics_internal.csv", index=False)
    aggregate.to_csv(args.output / "train_aggregate_metrics.csv", index=False)
    differences.to_csv(args.output / "train_epoch200_minus_epoch100.csv", index=False)
    val_comparison = compare_val()
    (args.output / "val_comparison.json").write_text(
        json.dumps(val_comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    verification = {
        "train_rows": len(images), "aggregate_rows": len(aggregate),
        "all_train_outputs_finite": bool(
            np.isfinite(images.select_dtypes(include=[np.number]).to_numpy()).all()
        ),
        "epoch100": verify100, "epoch200": verify200,
        "strict_continuous_trajectory_endpoints": True,
        "val_conditions_met": val_comparison["conditions_met"],
        "val_conditions_total": val_comparison["conditions_total"],
        "new_training": False, "test_read": False, "external_read": False,
    }
    (args.output / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"val": val_comparison, "verification": verification}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
