#!/usr/bin/env python3
"""汇总全局A1c固定配置的三种子保真结果，不读取test或external。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINE_RUN = PROJECT_ROOT / (
    "结果/SAE/EfficientNet全局局部_0804/SAE-A1c_Margin保真/"
    "a1c_global_efficientnet_b0_seed42_h10240_topk1024_margin01"
)
OUTPUT_ROOT = PROJECT_ROOT / "结果/SAE/EfficientNet全局局部_0804/SAE-C全局多种子复现"


def run_path(seed: int) -> Path:
    """返回一个seed的冻结正式结果目录；seed42沿用既有A1c产物。"""
    if seed == 42:
        return BASELINE_RUN
    return OUTPUT_ROOT / f"sae_c_global_efficientnet_b0_seed{seed}_h10240_topk1024_margin01"


def collect(seed: int) -> dict | None:
    """读取完整结果并按预注册的六项门槛生成一行；未完成seed返回None。"""
    run = run_path(seed)
    config_file, metrics_file = run / "config.json", run / "metrics.json"
    if not config_file.is_file() or not metrics_file.is_file():
        return None
    config = json.loads(config_file.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    val, pruning = metrics["splits"]["val"], metrics["pruning"]
    passed = bool(
        metrics["core_fidelity_gate"]["passed"]
        and pruning["dead_feature_rate_train"] <= 0.10
        and pruning["decoder_duplicate_feature_rate_abs_cosine_ge_0p95"] <= 0.10
    )
    return {
        "seed": seed,
        "run": run.name,
        "hidden_dim": config["hidden_dim"],
        "top_k": config["top_k"],
        "margin_loss_weight": config["margin_loss_weight"],
        "best_epoch": metrics["best_epoch"],
        "val_cosine": val["mean_cosine"],
        "patient_auc_original": val["patient_original_auc"],
        "patient_auc_reconstructed": val["patient_reconstructed_auc"],
        "patient_auc_drop": metrics["core_fidelity_gate"]["patient_auc_drop"],
        "patient_agreement": val["patient_prediction_agreement_at_locked_threshold"],
        "recovered_ce": val["recovered_cross_entropy"],
        "dead_rate": pruning["dead_feature_rate_train"],
        "duplicate_rate": pruning["decoder_duplicate_feature_rate_abs_cosine_ge_0p95"],
        "kept_features": pruning["kept_feature_count"],
        "passed_all_frozen_gates": passed,
        "test_evaluated": config["test_evaluated"],
        "external_evaluated": config["external_evaluated"],
    }


def main() -> None:
    """保存阶段性CSV；允许任务中途调用，便于每完成一个seed就审阅。"""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [row for seed in (42, 202, 503) if (row := collect(seed)) is not None]
    frame = pd.DataFrame(rows).sort_values("seed")
    output = OUTPUT_ROOT / "sae_c_global_replication_summary.csv"
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    print(frame.to_string(index=False))
    passed = int(frame.passed_all_frozen_gates.sum())
    print(f"\n当前完成{len(frame)}/3个seed，通过{passed}/{len(frame)}；汇总: {output}")


if __name__ == "__main__":
    main()
