#!/usr/bin/env python3
"""汇总来源内1:1.3平衡M0的6组验证集结果。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_m0_full_matrix import (
    METRICS,
    align_predictions,
    clustered_auc_delta_bootstrap,
)


MODELS = ("resnet50", "efficientnet_b0")
SEEDS = (42, 202, 503)


def parse_args():
    project = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=project / "结果/M0平衡_0804/正式验证集筛选",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def load_runs(root):
    runs = {}
    for model in MODELS:
        for seed in SEEDS:
            name = f"m0_balanced_keep_{model}_seed{seed}"
            run_dir = root / name
            required = [
                "config.json", "training_history.csv",
                "val_image_predictions.csv", "val_patient_predictions.csv",
                "val_subgroup_metrics.csv",
            ]
            missing = [path for path in required if not (run_dir / path).is_file()]
            if missing:
                raise FileNotFoundError(f"{name}缺少: {missing}")
            if list(run_dir.glob("test_*")):
                raise RuntimeError(f"{name}存在禁止的test输出")
            runs[(model, seed)] = {
                "name": name,
                "config": json.loads((run_dir / "config.json").read_text(encoding="utf-8")),
                "history": pd.read_csv(run_dir / "training_history.csv"),
                "image": pd.read_csv(run_dir / "val_image_predictions.csv"),
                "patient": pd.read_csv(run_dir / "val_patient_predictions.csv"),
            }
    return runs


def run_table(runs):
    rows = []
    for (model, seed), run in sorted(runs.items()):
        config = run["config"]
        best = run["history"].loc[run["history"].val_patient_AUC.idxmax()]
        row = {
            "model": model,
            "seed": seed,
            "run_name": run["name"],
            "selected_stage": config["selected_stage"],
            "best_epoch": int(best.epoch),
            "threshold": float(config["threshold_selection"]["threshold"]),
        }
        for level in ("patient", "image"):
            for metric in METRICS:
                row[f"{level}_{metric}"] = config["metrics"][f"val_{level}"][metric]
        rows.append(row)
    return pd.DataFrame(rows)


def cross_seed(table):
    rows = []
    columns = [
        f"{level}_{metric}"
        for level in ("patient", "image")
        for metric in METRICS
    ]
    for model, group in table.groupby("model", sort=True):
        row = {"model": model, "seeds": len(group)}
        for column in columns:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1))
            row[f"{column}_worst"] = float(values.min())
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_model_delta(runs, samples, seed):
    rows = []
    rng = np.random.default_rng(seed)
    for run_seed in SEEDS:
        for level in ("patient", "image"):
            paired = align_predictions(
                runs[("resnet50", run_seed)][level],
                runs[("efficientnet_b0", run_seed)][level],
                level,
            )
            result = clustered_auc_delta_bootstrap(paired, level, samples, rng)
            rows.append({
                "comparison": "efficientnet_b0_minus_resnet50",
                "seed": run_seed,
                "level": level,
                **result,
            })
    return pd.DataFrame(rows)


def markdown(run_metrics, summary, bootstrap):
    lines = [
        "# M0来源内1:1.3平衡验证矩阵汇总",
        "",
        "> 仅使用val；internal test和外部test不参与模型选择。",
        "",
        "## 逐运行",
        "",
        "| model | seed | patient AUC | patient Acc | patient Sens | patient Spec | image AUC | image Acc | image Sens | image Spec | threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in run_metrics.to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['seed']} | {row['patient_AUC']:.4f} "
            f"| {row['patient_Accuracy']:.4f} | {row['patient_Sensitivity']:.4f} "
            f"| {row['patient_Specificity']:.4f} | {row['image_AUC']:.4f} "
            f"| {row['image_Accuracy']:.4f} | {row['image_Sensitivity']:.4f} "
            f"| {row['image_Specificity']:.4f} | {row['threshold']:.5f} |"
        )
    lines.extend([
        "", "## 跨种子", "",
        "| model | patient AUC mean±SD | patient Acc | patient Sens | patient Spec | image AUC mean±SD | image Acc | image Sens | image Spec |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ])
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['patient_AUC_mean']:.4f}±{row['patient_AUC_std']:.4f} "
            f"| {row['patient_Accuracy_mean']:.4f} | {row['patient_Sensitivity_mean']:.4f} "
            f"| {row['patient_Specificity_mean']:.4f} | "
            f"{row['image_AUC_mean']:.4f}±{row['image_AUC_std']:.4f} "
            f"| {row['image_Accuracy_mean']:.4f} | {row['image_Sensitivity_mean']:.4f} "
            f"| {row['image_Specificity_mean']:.4f} |"
        )
    lines.extend([
        "",
        "图片级阈值沿用同一模型的val患者筛查阈值，没有单独调优。",
        "EfficientNet-B0与ResNet50的AUC配对bootstrap见CSV。",
    ])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    runs = load_runs(args.result_root)
    run_metrics = run_table(runs)
    summary = cross_seed(run_metrics)
    bootstrap = bootstrap_model_delta(runs, args.bootstrap, args.seed)
    output = args.result_root / "自动汇总"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"自动汇总目录非空: {output}")
    output.mkdir(parents=True, exist_ok=False)
    run_metrics.to_csv(output / "m0_balanced_run_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "m0_balanced_cross_seed_summary.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(output / "m0_balanced_paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    (output / "M0来源内1to1p3平衡验证矩阵汇总.md").write_text(
        markdown(run_metrics, summary, bootstrap), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"汇总目录: {output}")


if __name__ == "__main__":
    main()
