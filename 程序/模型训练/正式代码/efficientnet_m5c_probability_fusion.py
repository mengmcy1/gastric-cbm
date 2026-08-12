#!/usr/bin/env python3
"""EfficientNet-B0 M5c：冻结全局/局部患者概率的简单后期融合。

M5c由M5b验证集诊断形成，属于运行前冻结的后验验证实验：
- A（主检验）：患者级全局概率与局部概率固定等权平均；
- B（辅助）：仅在train患者上拟合两变量Logistic Regression；
- C（辅助）：用train患者经验分布把两路概率转为分位数后等权平均。

脚本复用M5b-B已冻结的全局Encoder、局部Encoder和局部分类头，只做确定性前向与
患者级概率融合，不训练CNN、不使用gate/q_region/框几何，也不读取internal test或external。
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from efficientnet_m1_localization import EfficientNetM1  # noqa: E402
from efficientnet_m5b_fusion import (  # noqa: E402
    M5bDataset,
    M5bModel,
    build_transforms,
    load_roi_manifest,
    patient_top2_mean,
    precompute_features,
)
from train_utils import (  # noqa: E402
    file_sha256,
    git_snapshot,
    json_ready,
    seed_everything,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "结果/M5c概率融合_0804/正式验证集筛选"
ROI_ROOT = PROJECT_ROOT / "结果/M5预测ROI融合_0804/冻结ROI清单"
M5B_ROOT = PROJECT_ROOT / "结果/M5b残差融合_0804/正式验证集筛选"
SEEDS = (42, 202, 503)
EPS = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, choices=SEEDS)
    parser.add_argument("--roi-manifest", type=Path, default=None)
    parser.add_argument("--m5b-run", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-units", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def default_roi_manifest(seed):
    return ROI_ROOT / f"m5_roi_manifest_seed{seed}.csv"


def default_m5b_run(seed):
    return M5B_ROOT / f"m5b_gate_local_balanced_keep_efficientnet_b0_seed{seed}"


def sigmoid(values):
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))


def logit(values):
    values = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    return np.log(values / (1.0 - values))


def empirical_percentile(reference, values):
    """用冻结的train经验分布把任意概率映射到[0,1]分位数。"""
    reference = np.sort(np.asarray(reference, dtype=float))
    if len(reference) == 0:
        raise ValueError("rank融合缺少train参考概率")
    return np.searchsorted(reference, np.asarray(values), side="right") / len(reference)


def aggregate_patient_probabilities(image_table):
    """分别执行top2_mean后合并全局/局部患者概率，避免改变既有聚合口径。"""
    table = image_table.copy()
    table["p_global"] = sigmoid(table.m_global)
    global_patients = patient_top2_mean(table, "p_global").rename(
        columns={"patient_probability": "p_global"}
    )
    local_patients = patient_top2_mean(table, "p_local").rename(
        columns={"patient_probability": "p_local"}
    )
    merged = global_patients.merge(
        local_patients[["patient_id", "label", "p_local"]],
        on=["patient_id", "label"], validate="one_to_one",
    )
    return merged.sort_values("patient_id").reset_index(drop=True)


def fit_and_apply_fusions(train_patients, val_patients, seed):
    """冻结三种融合公式；LR和rank映射只接触train患者。"""
    train = train_patients.copy()
    val = val_patients.copy()
    for frame in (train, val):
        frame["p_fixed_mean"] = 0.5 * (frame.p_global + frame.p_local)

    x_train = np.column_stack([logit(train.p_global), logit(train.p_local)])
    x_val = np.column_stack([logit(val.p_global), logit(val.p_local)])
    lr = LogisticRegression(
        C=1.0, class_weight="balanced", solver="lbfgs",
        max_iter=1000, random_state=seed,
    )
    lr.fit(x_train, train.label.to_numpy(dtype=int))
    train["p_lr"] = lr.predict_proba(x_train)[:, 1]
    val["p_lr"] = lr.predict_proba(x_val)[:, 1]

    for frame in (train, val):
        frame["rank_global"] = empirical_percentile(train.p_global, frame.p_global)
        frame["rank_local"] = empirical_percentile(train.p_local, frame.p_local)
        frame["p_rank_mean"] = 0.5 * (frame.rank_global + frame.rank_local)
    lr_record = {
        "features": ["logit_p_global", "logit_p_local"],
        "coefficient": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "C": 1.0,
        "class_weight": "balanced",
        "fit_split": "train_patients",
    }
    return train, val, lr_record


def paired_patient_bootstrap(frame, target, iterations, seed):
    """对患者行有放回抽样，报告目标融合相对全局AUC差的配对区间。"""
    labels = frame.label.to_numpy(dtype=int)
    baseline = frame.p_global.to_numpy(dtype=float)
    candidate = frame[target].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        index = rng.choice(len(frame), size=len(frame), replace=True)
        sampled_labels = labels[index]
        if np.unique(sampled_labels).size < 2:
            continue
        differences.append(
            roc_auc_score(sampled_labels, candidate[index])
            - roc_auc_score(sampled_labels, baseline[index])
        )
    if not differences:
        raise RuntimeError("bootstrap未获得同时含两类患者的有效抽样")
    return {
        "delta_mean": float(np.mean(differences)),
        "delta_ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
        "iterations_used": len(differences),
    }


def evaluate_methods(val_patients, bootstrap, seed):
    """按预注册主次关系评估A/B/C，不在三者之间做验证集择优。"""
    global_auc = float(roc_auc_score(val_patients.label, val_patients.p_global))
    methods = {
        "A_fixed_mean_primary": "p_fixed_mean",
        "B_train_lr_auxiliary": "p_lr",
        "C_train_ecdf_rank_mean_auxiliary": "p_rank_mean",
    }
    results = {}
    for offset, (name, column) in enumerate(methods.items()):
        auc = float(roc_auc_score(val_patients.label, val_patients[column]))
        results[name] = {
            "column": column,
            "patient_auc": auc,
            "delta_vs_global": auc - global_auc,
            "paired_bootstrap": paired_patient_bootstrap(
                val_patients, column, bootstrap, seed + offset,
            ),
        }
    return global_auc, results


def prepare_output(args):
    run_name = args.run_name or f"m5c_balanced_keep_efficientnet_b0_seed{args.seed}"
    output = args.output_root / (run_name + ("_debug" if args.debug else ""))
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output}")
    output.mkdir(parents=True, exist_ok=args.overwrite)
    return output


def self_test():
    reference = np.array([0.1, 0.2, 0.3, 0.4])
    mapped = empirical_percentile(reference, np.array([0.05, 0.2, 0.5]))
    if not np.allclose(mapped, [0.0, 0.5, 1.0]):
        raise AssertionError("train ECDF分位映射错误")
    frame = pd.DataFrame({
        "patient_id": list("ABCDEF"), "label": [0, 0, 0, 1, 1, 1],
        "p_global": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        "same": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
    })
    result = paired_patient_bootstrap(frame, "same", 100, 42)
    if result["delta_ci95"] != [0.0, 0.0]:
        raise AssertionError("相同患者预测的配对bootstrap差异不为0")


def main():
    """校验冻结产物链，生成患者概率，执行三种固定融合并保存完整审计结果。"""
    args = parse_args()
    if args.self_test:
        self_test()
        print("M5c自测通过: train ECDF + 患者配对bootstrap")
        return
    if args.batch_size != 32:
        raise ValueError("M5c正式推理batch size冻结为32")
    seed_everything(args.seed)
    roi_manifest = (args.roi_manifest or default_roi_manifest(args.seed)).resolve()
    m5b_run = (args.m5b_run or default_m5b_run(args.seed)).resolve()
    checkpoint = m5b_run / "m5b_best.pth"
    m5b_config_path = m5b_run / "config.json"
    if not checkpoint.is_file() or not m5b_config_path.is_file():
        raise FileNotFoundError(f"缺少冻结M5b-B产物: {m5b_run}")
    m5b_config = json.loads(m5b_config_path.read_text(encoding="utf-8"))
    if m5b_config.get("alpha_mode") != "gate_local" \
            or int(m5b_config.get("seed", -1)) != args.seed:
        raise ValueError("M5c只复用同seed正式M5b-B(gate_local)产物")
    if m5b_config.get("test_evaluated") or m5b_config.get("external_evaluated"):
        raise ValueError("M5b-B来源产物违反test/external锁定边界")
    if file_sha256(roi_manifest) != m5b_config.get("roi_manifest_sha256"):
        raise ValueError("当前ROI清单与冻结M5b-B产物不同源")

    frame = load_roi_manifest(
        roi_manifest, args.image_root, args.debug, args.debug_units, args.seed,
    )
    output = prepare_output(args)
    eval_transform, _, roi_eval_transform = build_transforms()
    datasets = {
        split: M5bDataset(
            frame.loc[frame.split.eq(split)].copy(), args.image_root,
            eval_transform, roi_eval_transform,
        )
        for split in ("train", "val")
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = EfficientNetM1()
    model = M5bModel(base.backbone, "gate_local").to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    print(f"设备: {device}; seed={args.seed}; train={len(datasets['train'])}张; "
          f"val={len(datasets['val'])}张; 冻结来源={m5b_run.name}")

    image_tables = precompute_features(
        model, datasets, device, args.batch_size, args.num_workers,
    )
    train_patients = aggregate_patient_probabilities(image_tables["train"])
    val_patients = aggregate_patient_probabilities(image_tables["val"])
    train_patients, val_patients, lr_record = fit_and_apply_fusions(
        train_patients, val_patients, args.seed,
    )
    global_auc, methods = evaluate_methods(
        val_patients, args.bootstrap, args.seed,
    )
    config = {
        **{key: str(value) if isinstance(value, Path) else value
           for key, value in vars(args).items()},
        "experiment_origin": "post_hoc_hypothesis_from_m5b_val_diagnostics",
        "primary_method": "A_fixed_mean_primary",
        "method_selection_on_val": False,
        "roi_manifest": str(roi_manifest),
        "roi_manifest_sha256": file_sha256(roi_manifest),
        "frozen_m5b_run": str(m5b_run),
        "frozen_m5b_checkpoint": str(checkpoint),
        "frozen_m5b_checkpoint_sha256": file_sha256(checkpoint),
        "n_train_images": len(image_tables["train"]),
        "n_val_images": len(image_tables["val"]),
        "n_train_patients": len(train_patients),
        "n_val_patients": len(val_patients),
        "patient_aggregation": "top2_mean_each_branch_before_fusion",
        "lr_model": lr_record,
        "global_patient_auc": global_auc,
        "methods": methods,
        "test_evaluated": False,
        "external_evaluated": False,
        **git_snapshot(),
    }
    train_patients.to_csv(
        output / "train_patient_predictions.csv", index=False, encoding="utf-8-sig",
    )
    val_patients.to_csv(
        output / "val_patient_predictions.csv", index=False, encoding="utf-8-sig",
    )
    (output / "config.json").write_text(
        json.dumps(json_ready(config), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    shutil.copy2(Path(__file__), output / "source_entry.py")
    print(f"全局患者AUC={global_auc:.4f}")
    for name, result in methods.items():
        print(f"{name}: AUC={result['patient_auc']:.4f} "
              f"Δ={result['delta_vs_global']:+.4f} "
              f"CI={result['paired_bootstrap']['delta_ci95']}")
    print(f"输出目录: {output}")


if __name__ == "__main__":
    main()
