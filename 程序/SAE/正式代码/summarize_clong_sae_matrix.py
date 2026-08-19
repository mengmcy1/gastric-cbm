#!/usr/bin/env python3
"""汇总C-long SAE矩阵（17组）并按预注册规则选出唯一正式配置。

输入：`结果/SAE/CLong文献重构_20260819/` 下17个实验目录，每个须包含完整产物
（config.json、metrics.json、SAE模型、特征缓存、feature筛选）。

判定逻辑（S2-S3预注册，2026-08-19冻结）：

1. 完整性校验：17组全部存在且config血缘、阈值、参数与冻结值一致，否则拒绝汇总；
2. 硬门槛：四项分类保真 + 死亡率 <= 10% + 非死亡重复率 <= 10%；
3. Pareto前沿：患者AUC下降↓、患者一致率↑、cosine↑、recovered CE↑、
   val mean L0↓、剪枝后保留Feature数↓；
4. 决胜链：前沿候选依次按 患者AUC下降更小 → 患者一致率更高 → val mean L0更低
   → 剪枝后保留数更少 → 字典宽度更小，选出唯一正式配置；
5. Top-K规则：L1有合格候选时Top-K只作机制对照；L1全败时Top-K升级为备选候选，
   过同样门槛后进入选择池；两者皆无合格时报告"本阶段无正式产品"。

输出：`clong_sae_matrix_summary_seed42.json` 与 `clong_sae_matrix_seed42.csv`。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import clong_sae_discovery as csd  # noqa: E402

RUN_ROOT = csd.OUTPUT_ROOT
SUMMARY_JSON = RUN_ROOT / "clong_sae_matrix_summary_seed42.json"
SUMMARY_CSV = RUN_ROOT / "clong_sae_matrix_seed42.csv"

# 与 run_clong_sae_matrix.sh 完全一致的17组固定矩阵。
RUNS = [
    {"name": f"clong_w{width}_l{label}_seed42", "mode": "relu_l1",
     "hidden_dim": width, "lambda_l1": value, "gamma": 0.1, "top_k": None}
    for width, pairs in (
        (512, (("2e-4", 2e-4), ("5e-4", 5e-4), ("1e-3", 1e-3))),
        (1280, (("2e-4", 2e-4), ("5e-4", 5e-4), ("1e-3", 1e-3))),
        (2560, (("2e-4", 2e-4), ("5e-4", 5e-4), ("1e-3", 1e-3))),
        (5120, (("2e-4", 2e-4), ("5e-4", 5e-4), ("1e-3", 1e-3))),
        (10240, (("2e-4", 2e-4), ("5e-4", 5e-4), ("1e-3", 1e-3))),
    )
    for label, value in pairs
]
RUNS.append({"name": "clong_w10240_l5e-4_gamma0_seed42", "mode": "relu_l1",
             "hidden_dim": 10240, "lambda_l1": 5e-4, "gamma": 0.0, "top_k": None})
RUNS.append({"name": "clong_w10240_topk1024_seed42", "mode": "topk",
             "hidden_dim": 10240, "lambda_l1": 0.0, "gamma": 0.1, "top_k": 1024})

REQUIRED_FILES = (
    "config.json", "metrics.json", "SAE模型/sae_best.pth",
    "特征缓存/cache_config.json", "feature_summary.csv",
    "feature筛选/pruning_summary.json",
)


def check_run_integrity(spec: dict) -> dict:
    """核验单组产物完整性与config血缘，返回config；任一不符立即终止。"""
    run_dir = RUN_ROOT / spec["name"]
    for relative in REQUIRED_FILES:
        if not (run_dir / relative).is_file():
            raise FileNotFoundError(f"{spec['name']} 缺少 {relative}，拒绝汇总")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("debug") is not False:
        raise ValueError(f"{spec['name']} 是debug运行，不能进入正式汇总")
    if int(config.get("seed", -1)) != 42:
        raise ValueError(f"{spec['name']} seed不是42")
    for field, expected in (
        ("hidden_dim", spec["hidden_dim"]),
        ("activation_mode", spec["mode"]),
    ):
        if config.get(field) != expected:
            raise ValueError(f"{spec['name']} 的{field}与矩阵定义不一致")
    if not csd._close(float(config.get("lambda_l1")), spec["lambda_l1"]):
        raise ValueError(f"{spec['name']} 的lambda与矩阵定义不一致")
    if not csd._close(float(config.get("margin_loss_weight")), spec["gamma"]):
        raise ValueError(f"{spec['name']} 的gamma与矩阵定义不一致")
    if spec["mode"] == "topk" and int(config.get("top_k", -1)) != spec["top_k"]:
        raise ValueError(f"{spec['name']} 的top_k与矩阵定义不一致")
    lineage = config.get("s0_lineage", {})
    sha_expectations = {
        "student_checkpoint": csd.CLONG_CHECKPOINT_SHA256,
        "manifest": csd.MANIFEST_SHA256,
        "v3_audit": csd.V3_AUDIT_SHA256,
        "teacher_checkpoint": csd.TEACHER_CHECKPOINT_SHA256,
        "teacher_cache": csd.TEACHER_CACHE_SHA256,
        "beta_calibration_json": csd.BETA_JSON_SHA256,
    }
    for name, sha in sha_expectations.items():
        if lineage.get(name, {}).get("sha256") != sha:
            raise ValueError(f"{spec['name']} 的S0血缘{name}与冻结值不一致")
    if not csd._close(float(config.get("patient_threshold_frozen")),
                      csd.FROZEN_PATIENT_THRESHOLD):
        raise ValueError(f"{spec['name']} 的冻结患者阈值与S0不一致")
    if not csd._close(float(config.get("image_threshold_frozen")),
                      csd.FROZEN_IMAGE_THRESHOLD):
        raise ValueError(f"{spec['name']} 的冻结图像阈值与S0不一致")
    if not config.get("recompute_self_test", {}).get("passed"):
        raise ValueError(f"{spec['name']} 的缓存复算自测未通过")
    if config.get("feature_layer") != csd.FEATURE_LAYER_DESCRIPTION:
        raise ValueError(f"{spec['name']} 的特征层描述与预注册不一致")
    return config


def collect_row(spec: dict) -> dict:
    """从metrics.json提取判定所需全部指标。"""
    metrics = json.loads(
        (RUN_ROOT / spec["name"] / "metrics.json").read_text(encoding="utf-8")
    )
    val = metrics["splits"]["val"]
    pruning = metrics["pruning"]
    duplicate = pruning["decoder_duplicate_nondead"]
    auc_drop = val["patient_original_auc"] - val["patient_reconstructed_auc"]
    row = {
        "name": spec["name"], "mode": spec["mode"],
        "hidden_dim": spec["hidden_dim"], "lambda_l1": spec["lambda_l1"],
        "gamma": spec["gamma"], "top_k": spec["top_k"],
        "best_epoch": metrics["best_epoch"],
        "val_mean_cosine": val["mean_cosine"],
        "patient_original_auc": val["patient_original_auc"],
        "patient_reconstructed_auc": val["patient_reconstructed_auc"],
        "patient_auc_drop": auc_drop,
        "patient_agreement": val["patient_prediction_agreement_at_locked_threshold"],
        "recovered_ce": val["recovered_cross_entropy"],
        "val_mean_l0": val["mean_l0"],
        "val_mean_ncc90": val["mean_ncc90"],
        "dead_feature_rate": pruning["dead_feature_rate_train"],
        "duplicate_rate_nondead": duplicate["duplicate_rate"],
        "kept_feature_count": pruning["kept_feature_count"],
        "pruning_applied": pruning["pruning_applied"],
    }
    row["gate_passed"] = bool(
        auc_drop <= csd.GATE_MAX_PATIENT_AUC_DROP
        and row["patient_agreement"] >= csd.GATE_MIN_PATIENT_AGREEMENT
        and row["val_mean_cosine"] >= csd.GATE_MIN_COSINE
        and row["recovered_ce"] >= csd.GATE_MIN_RECOVERED_CE
        and row["dead_feature_rate"] <= csd.GATE_MAX_DEAD_RATE
        and row["duplicate_rate_nondead"] <= csd.GATE_MAX_DUPLICATE_RATE
    )
    return row


def dominates(a: pd.Series, b: pd.Series) -> bool:
    """Pareto支配：a在全部六维不劣于b且至少一维严格更优。"""
    better_or_equal = (
        a.patient_auc_drop <= b.patient_auc_drop
        and a.patient_agreement >= b.patient_agreement
        and a.val_mean_cosine >= b.val_mean_cosine
        and a.recovered_ce >= b.recovered_ce
        and a.val_mean_l0 <= b.val_mean_l0
        and a.kept_feature_count <= b.kept_feature_count
    )
    strictly_better = (
        a.patient_auc_drop < b.patient_auc_drop
        or a.patient_agreement > b.patient_agreement
        or a.val_mean_cosine > b.val_mean_cosine
        or a.recovered_ce > b.recovered_ce
        or a.val_mean_l0 < b.val_mean_l0
        or a.kept_feature_count < b.kept_feature_count
    )
    return bool(better_or_equal and strictly_better)


def pareto_frontier(candidates: pd.DataFrame) -> pd.DataFrame:
    """返回不被任何其他候选支配的候选子集。"""
    keep = []
    for index, row in candidates.iterrows():
        dominated = any(
            dominates(other, row)
            for other_index, other in candidates.iterrows()
            if other_index != index
        )
        keep.append(not dominated)
    return candidates.loc[keep]


def select_unique_config(frontier: pd.DataFrame) -> pd.Series:
    """按冻结决胜链从前沿候选中选出唯一正式配置。"""
    ordered = frontier.sort_values(
        by=["patient_auc_drop", "patient_agreement", "val_mean_l0",
            "kept_feature_count", "hidden_dim"],
        ascending=[True, False, True, True, True],
        kind="mergesort",
    )
    return ordered.iloc[0]


def main() -> None:
    """核验17组产物，输出矩阵表、Pareto前沿与唯一正式配置。"""
    for spec in RUNS:
        check_run_integrity(spec)
    rows = [collect_row(spec) for spec in RUNS]
    table = pd.DataFrame(rows)
    table.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    l1_main = table.loc[(table["mode"] == "relu_l1") & (table["gamma"] > 0)]
    diagnostic = table.loc[table["gamma"] == 0].iloc[0]
    topk = table.loc[table["mode"] == "topk"].iloc[0]

    l1_passed = l1_main.loc[l1_main.gate_passed]
    topk_role = "mechanism_control" if len(l1_passed) > 0 else "fallback_candidate"
    candidate_pool = l1_passed.copy()
    if topk_role == "fallback_candidate" and bool(topk.gate_passed):
        candidate_pool = pd.concat([candidate_pool, topk.to_frame().T])

    summary = {
        "matrix_size": int(len(table)),
        "l1_main_runs": int(len(l1_main)),
        "l1_passed_runs": int(len(l1_passed)),
        "topk_role": topk_role,
        "topk_gate_passed": bool(topk.gate_passed),
        "gates": {
            "maximum_patient_auc_drop": csd.GATE_MAX_PATIENT_AUC_DROP,
            "minimum_patient_agreement": csd.GATE_MIN_PATIENT_AGREEMENT,
            "minimum_mean_cosine": csd.GATE_MIN_COSINE,
            "minimum_recovered_cross_entropy": csd.GATE_MIN_RECOVERED_CE,
            "maximum_dead_feature_rate": csd.GATE_MAX_DEAD_RATE,
            "maximum_decoder_duplicate_rate_nondead": csd.GATE_MAX_DUPLICATE_RATE,
        },
        "no_margin_diagnostic": {
            "name": diagnostic["name"],
            "patient_auc_drop": float(diagnostic.patient_auc_drop),
            "patient_agreement": float(diagnostic.patient_agreement),
            "val_mean_cosine": float(diagnostic.val_mean_cosine),
            "recovered_ce": float(diagnostic.recovered_ce),
            "val_mean_l0": float(diagnostic.val_mean_l0),
            "reference_with_margin": "clong_w10240_l5e-4_seed42",
        },
    }

    if len(candidate_pool) == 0:
        summary["status"] = "no_formal_product"
        summary["decision"] = (
            "15组L1与固定Top-K备选均未通过全部硬门槛；按预注册停止规则，"
            "本阶段无正式产品，回到表示对象或SAE结构设计，不追加超参数。"
        )
    else:
        frontier = pareto_frontier(candidate_pool)
        selected = select_unique_config(frontier)
        summary["status"] = "formal_config_selected"
        summary["pareto_frontier"] = frontier["name"].tolist()
        summary["selected_config"] = {
            key: (float(value) if isinstance(value, float) else
                  int(value) if isinstance(value, (int,)) else value)
            for key, value in selected.to_dict().items()
        }
        summary["next_step"] = (
            "按S5对选中配置运行SAE seed202/503复现，报告3/3、2/3或1/3稳定性"
        )
    SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"矩阵汇总完成: {SUMMARY_JSON}")
    print(f"状态: {summary['status']}；L1合格 {len(l1_passed)}/15；Top-K角色={topk_role}")
    if summary["status"] == "formal_config_selected":
        print(f"唯一正式配置: {summary['selected_config']['name']}")


if __name__ == "__main__":
    main()
