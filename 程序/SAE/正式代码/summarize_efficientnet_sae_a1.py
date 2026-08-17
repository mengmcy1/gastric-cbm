#!/usr/bin/env python3
"""汇总SAE-A0/A1/A1b/A1c核心保真、稀疏性和字典质量，不读取test/external。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
A0 = PROJECT_ROOT / "结果/SAE/EfficientNet全局局部_0804/SAE-A0_S0工程验证"
A1 = PROJECT_ROOT / "结果/SAE/EfficientNet全局局部_0804/SAE-A1_宽度L1比较"
A1B = PROJECT_ROOT / "结果/SAE/EfficientNet全局局部_0804/SAE-A1b_TopK比较"
A1C = PROJECT_ROOT / "结果/SAE/EfficientNet全局局部_0804/SAE-A1c_Margin保真"


def collect(run: Path) -> dict:
    """读取一个完整run并转换为同一比较口径。"""
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    val, pruning = metrics["splits"]["val"], metrics["pruning"]
    quality = (
        metrics["core_fidelity_gate"]["passed"]
        and pruning["dead_feature_rate_train"] <= 0.10
        and pruning["decoder_duplicate_feature_rate_abs_cosine_ge_0p95"] <= 0.10
    )
    return {
        "run": run.name, "branch": config["branch"], "hidden_dim": config["hidden_dim"],
        "activation_mode": config.get("activation_mode", "relu_l1"),
        "top_k": config.get("top_k"),
        "lambda_l1": config["lambda_l1"],
        "margin_loss_weight": config.get("margin_loss_weight", 0.0),
        "best_epoch": metrics["best_epoch"],
        "val_mse": val["mse"], "val_cosine": val["mean_cosine"],
        "val_l0": val["mean_l0"], "val_ncc90": val["mean_ncc90"],
        "patient_auc_original": val["patient_original_auc"],
        "patient_auc_reconstructed": val["patient_reconstructed_auc"],
        "patient_auc_drop": metrics["core_fidelity_gate"]["patient_auc_drop"],
        "patient_agreement": val["patient_prediction_agreement_at_locked_threshold"],
        "recovered_ce": val["recovered_cross_entropy"],
        "dead_rate": pruning["dead_feature_rate_train"],
        "duplicate_rate": pruning["decoder_duplicate_feature_rate_abs_cosine_ge_0p95"],
        "kept_features": pruning["kept_feature_count"],
        "passed_all_frozen_gates": quality,
    }


def main() -> None:
    """保存机器可读CSV并打印按分支排序的完整开发结果。"""
    runs = [path for root in (A0, A1, A1B, A1C) if root.exists() for path in root.iterdir()
            if path.is_dir() and (path / "config.json").is_file() and (path / "metrics.json").is_file()]
    frame = pd.DataFrame(collect(run) for run in runs).sort_values(
        ["branch", "passed_all_frozen_gates", "patient_auc_drop", "val_l0"],
        ascending=[True, False, True, True],
    )
    output_root = A1C if A1C.exists() else (A1B if A1B.exists() else A1)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "sae_a0_a1b_a1c_summary.csv"
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    print(frame.to_string(index=False))
    for branch, part in frame.groupby("branch", sort=False):
        passed = part.loc[part.passed_all_frozen_gates]
        print(f"\n{branch}: 合格 {len(passed)}/{len(part)}")
        if len(passed):
            print(passed[["run", "val_cosine", "val_l0", "patient_auc_drop", "patient_agreement"]].to_string(index=False))
    print(f"\n汇总: {output}")


if __name__ == "__main__":
    main()
