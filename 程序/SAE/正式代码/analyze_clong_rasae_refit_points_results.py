#!/usr/bin/env python3
"""比较扩大池拟合代表点候选与旧代表点第200轮对照。

在共同train1673图上复用三层联合/K256及固定/重算注意力路径，
并读取完整val固定终点结果执行五项预定方向判定。不训练、不选epoch。
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
from analyze_clong_rasae_duration200_results import load_sae  # noqa: E402
from analyze_clong_rasae_medical_feedback import load_models  # noqa: E402
from run_clong_rasae_pilot import read_subset  # noqa: E402


BASE = ROOT / "结果/SAE/RA_SAE_Pilot_20260908"
CONTROL_ROOT = BASE / "expanded_pool_duration200_20260916"
CANDIDATE_ROOT = BASE / "expanded_pool_refit_points_duration200_20260917"


def nested_float(source: dict, path: tuple[str, ...]) -> float:
    """按键路径读取JSON浮点。

    Args:
        source (dict): 已解析JSON。
        path (tuple[str, ...]): 嵌套键。

    Returns:
        float: 目标数值。
    """
    value = source
    for key in path:
        value = value[key]
    return float(value)


def compare_val() -> dict:
    """比较旧/新代表点的完整val第200轮结果。

    Args:
        None.

    Returns:
        dict: 五项指标、条件及5/5清晰收益判定。
    """
    control = json.loads((CONTROL_ROOT / "full_val_epoch200/summary.json").read_text())
    candidate = json.loads((CANDIDATE_ROOT / "full_val/summary.json").read_text())
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
        old, new = nested_float(control, path), nested_float(candidate, path)
        metrics[name] = {
            "old_points_epoch200": old, "expanded_points_epoch200": new,
            "delta_candidate_minus_control": new - old,
        }
    conditions = {
        "patch_fvu_lower": metrics["patch_fvu"]["expanded_points_epoch200"] < metrics["patch_fvu"]["old_points_epoch200"],
        "pooled_fvu_lower": metrics["pooled_fvu"]["expanded_points_epoch200"] < metrics["pooled_fvu"]["old_points_epoch200"],
        "patient_probability_abs_change_not_higher": (
            metrics["patient_probability_abs_change_mean"]["expanded_points_epoch200"]
            <= metrics["patient_probability_abs_change_mean"]["old_points_epoch200"]
        ),
        "patient_prediction_agreement_not_lower": (
            metrics["patient_prediction_agreement"]["expanded_points_epoch200"]
            >= metrics["patient_prediction_agreement"]["old_points_epoch200"]
        ),
        "reconstructed_patient_auc_not_lower": (
            metrics["reconstructed_patient_auc"]["expanded_points_epoch200"]
            >= metrics["reconstructed_patient_auc"]["old_points_epoch200"]
        ),
    }
    passed = int(sum(conditions.values()))
    if passed == len(conditions):
        status = "clear_fidelity_benefit_5_of_5"
    elif conditions["patch_fvu_lower"] and conditions["pooled_fvu_lower"]:
        status = "representation_improved_with_prediction_tradeoff"
    else:
        status = "expected_joint_representation_benefit_not_achieved"
    return {
        "metrics": metrics, "conditions": conditions,
        "conditions_met": passed, "conditions_total": len(conditions),
        "joint_status": status,
        "negative_boundary": (
            "Under fixed old center, pilot calibration protocol, and 2600 updates, "
            "changing representative-point fit data did not achieve the expected benefit; "
            "this does not exclude representative-point limitations."
        ) if passed < len(conditions) else None,
    }


def main() -> None:
    """执行共同train与完整val的旧/新代表点汇总。

    Args:
        None: 输出、设备和批量由CLI提供。

    Returns:
        None: 写入train逐图/聚合表、val比较和验证JSON。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame, _ = common_cohort()
    weights = evaluation_weights(frame)
    device = torch.device(args.device)
    data = read_subset(frame, "train", device)
    unused, head, classifier_weight, _ = load_models(device)
    del unused
    control_config = json.loads((CONTROL_ROOT / "config.json").read_text())
    margin = classifier_weight[1] - classifier_weight[0]
    control = load_sae(CONTROL_ROOT / "duration200_full.pth", device)
    control_rows, control_check = evaluate_model(
        "baseline", control, data, frame, head, margin,
        float(control_config["margin_std"]), float(control_config["gamma_pool"]), args.batch_size,
    )
    del control
    torch.cuda.empty_cache() if device.type == "cuda" else None
    candidate = load_sae(CANDIDATE_ROOT / "refit_points_duration200_full.pth", device)
    candidate_rows, candidate_check = evaluate_model(
        "expanded_pool", candidate, data, frame, head, margin,
        float(control_config["margin_std"]), float(control_config["gamma_pool"]), args.batch_size,
    )
    images = pd.concat([control_rows, candidate_rows], ignore_index=True)
    aggregate = aggregate_metrics(images, weights, frame)
    differences = paired_differences(aggregate)
    images["checkpoint"] = images.checkpoint.replace({
        "baseline": "old_points_epoch200", "expanded_pool": "expanded_points_epoch200"
    })
    aggregate["checkpoint"] = aggregate.checkpoint.replace({
        "baseline": "old_points_epoch200", "expanded_pool": "expanded_points_epoch200"
    })
    differences = differences.rename(columns={
        column: column.replace("_baseline", "_old_points").replace("_expanded", "_expanded_points")
        for column in differences.columns
    })
    images.to_csv(args.output / "train_image_metrics_internal.csv", index=False)
    aggregate.to_csv(args.output / "train_aggregate_metrics.csv", index=False)
    differences.to_csv(args.output / "train_expanded_points_minus_old_points.csv", index=False)
    val = compare_val()
    (args.output / "val_comparison.json").write_text(
        json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    verification = {
        "train_rows": len(images), "aggregate_rows": len(aggregate),
        "all_train_outputs_finite": bool(
            np.isfinite(images.select_dtypes(include=[np.number]).to_numpy()).all()
        ),
        "control": control_check, "candidate": candidate_check,
        "same_images_weights_budget_k_loss": True,
        "old_center_and_pilot_calibration_protocol_fixed": True,
        "val_conditions_met": val["conditions_met"],
        "val_conditions_total": val["conditions_total"],
        "new_training": False, "test_read": False, "external_read": False,
    }
    (args.output / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"val": val, "verification": verification}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
