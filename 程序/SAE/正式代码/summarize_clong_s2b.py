#!/usr/bin/env python3
"""S2b seed42 B/C/D/N 完整性校验、门槛判定和唯一patch产品选择。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "程序/MAGE/正式代码"))

from build_mage_teacher_roi_manifest import file_sha256  # noqa: E402
from clong_s2b_discovery import (  # noqa: E402
    CLONG_CHECKPOINT_SHA256,
    FORMAL_BUDGET,
    MANIFEST_SHA256,
    OUTPUT_ROOT,
    formal_name,
)
from train_utils import json_ready  # noqa: E402

MAX_AUC_DROP = 0.01
MIN_AGREEMENT = 0.95
MIN_COSINE = 0.90
MIN_RECOVERED_CE = 0.95
MAX_DEAD = 0.10
MAX_DUPLICATE = 0.10
MAX_NAIB_DROP = 0.05
MAX_PGA_DROP = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_paths(root: Path, arm: str, seed: int) -> tuple[Path, Path, Path, Path]:
    """返回单个正式实验的目录及三类结果文件。"""
    run = root / formal_name(arm, seed)
    return run, run / "config.json", run / "protocol_failure.json", run / "sae_best.pth"


def load_run(root: Path, arm: str, seed: int) -> dict:
    """读取并验证一个成功完成评价的正式运行。"""
    run, config_path, _, checkpoint = run_paths(root, arm, seed)
    history = run / "training_history.csv"
    for path in (config_path, checkpoint, history):
        if not path.is_file():
            raise FileNotFoundError(f"S4-{arm}产物不完整: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checks = {
        "stage": config.get("stage") == "S2b",
        "arm": config.get("arm") == arm,
        "seed": int(config.get("seed", -1)) == seed,
        "experiment": config.get("experiment") == formal_name(arm, seed),
        "debug": config.get("debug") is False,
        "student": config.get("student_checkpoint_sha256") == CLONG_CHECKPOINT_SHA256,
        "manifest": config.get("manifest_sha256") == MANIFEST_SHA256,
        "checkpoint": config.get("checkpoint_sha256") == file_sha256(checkpoint),
        "locked_data": not any(config.get(key, True) for key in (
            "test_evaluated", "internal_test_evaluated", "external_evaluated"
        )),
    }
    for key, value in FORMAL_BUDGET.items():
        checks[f"budget_{key}"] = math.isclose(
            float(config["training"][key]), float(value), abs_tol=1e-12
        )
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"S4-{arm}完整性校验失败: {failed}")
    if arm in {"B", "D"}:
        threshold = config.get("batchtopk_threshold")
        if not threshold or threshold.get("relative_deviation", 1) > 0.01:
            raise ValueError(f"S4-{arm} BatchTopK阈值记录不合格")
        if threshold.get("auxiliary_dead_latent_loss") is not False:
            raise ValueError(f"S4-{arm}意外启用了aux损失")
    return config


def load_protocol_failure(root: Path, arm: str, seed: int) -> dict:
    """读取并验证BatchTopK在预注册阈值冻结处的正式失败。"""
    run, config_path, failure_path, checkpoint = run_paths(root, arm, seed)
    history = run / "training_history.csv"
    if arm not in {"B", "D"}:
        raise ValueError(f"S4-{arm}不允许记录BatchTopK协议失败")
    if config_path.exists():
        raise ValueError(f"S4-{arm}同时存在config与protocol_failure")
    for path in (failure_path, checkpoint, history):
        if not path.is_file():
            raise FileNotFoundError(f"S4-{arm}协议失败产物不完整: {path}")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    checks = {
        "stage": failure.get("stage") == "S2b",
        "arm": failure.get("arm") == arm,
        "seed": int(failure.get("seed", -1)) == seed,
        "experiment": failure.get("experiment") == formal_name(arm, seed),
        "debug": failure.get("debug") is False,
        "failure_stage": failure.get("failure_stage") == "freeze_batchtopk_train_threshold",
        "student": failure.get("student_checkpoint_sha256") == CLONG_CHECKPOINT_SHA256,
        "manifest": failure.get("manifest_sha256") == MANIFEST_SHA256,
        "checkpoint": failure.get("checkpoint_sha256") == file_sha256(checkpoint),
        "locked_data": not any(failure.get(key, True) for key in (
            "test_evaluated", "internal_test_evaluated", "external_evaluated"
        )),
    }
    for key, value in FORMAL_BUDGET.items():
        checks[f"budget_{key}"] = math.isclose(
            float(failure.get("training", {}).get(key, float("nan"))),
            float(value), abs_tol=1e-12,
        )
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"S4-{arm}协议失败完整性校验失败: {failed}")
    return failure


def load_outcome(root: Path, arm: str, seed: int) -> dict:
    """区分成功结果、预注册协议失败和残缺产物。"""
    _, config_path, failure_path, _ = run_paths(root, arm, seed)
    if config_path.is_file() and failure_path.exists():
        raise ValueError(f"S4-{arm}同时存在成功与失败结果")
    if config_path.is_file():
        return {"status": "success", "config": load_run(root, arm, seed)}
    if failure_path.is_file():
        return {
            "status": "protocol_failure",
            "failure": load_protocol_failure(root, arm, seed),
        }
    raise FileNotFoundError(f"S4-{arm}既无config.json也无protocol_failure.json")


def common_gates(config: dict) -> dict:
    evaluation = config["evaluation"]
    fidelity = evaluation["fidelity"]
    duplicate = evaluation["duplicate"]
    return {
        "patient_auc_drop_le_0_01": fidelity["patient_auc_drop"] <= MAX_AUC_DROP,
        "patient_agreement_ge_0_95": (
            fidelity["patient_prediction_agreement_at_locked_threshold"] >= MIN_AGREEMENT
        ),
        "pooled_cosine_ge_0_90": fidelity["mean_cosine"] >= MIN_COSINE,
        "recovered_ce_ge_0_95": fidelity["recovered_cross_entropy"] >= MIN_RECOVERED_CE,
        "dead_rate_le_0_10": evaluation["dead_feature_rate"] <= MAX_DEAD,
        "duplicate_rate_le_0_10": duplicate["duplicate_rate"] <= MAX_DUPLICATE,
    }


def patch_gates(config: dict) -> dict:
    gates = common_gates(config)
    distribution = config["evaluation"]["sparsity"]
    gates.update({
        "normalized_aib_drop_le_0_05": distribution["normalized_aib_drop"] <= MAX_NAIB_DROP,
        "pga_drop_le_0_05": distribution["pga_drop"] <= MAX_PGA_DROP,
    })
    return gates


def patch_selection_key(config: dict) -> tuple:
    evaluation = config["evaluation"]
    fidelity = evaluation["fidelity"]
    sparsity = evaluation["sparsity"]
    # min(key)：五级决胜链，最后C比D更简单。
    return (
        fidelity["patient_auc_drop"],
        -fidelity["patient_prediction_agreement_at_locked_threshold"],
        -fidelity["mean_cosine"],
        sparsity["mean_l0_per_position"],
        sparsity["mean_unique_features_per_image"],
        0 if config["arm"] == "C" else 1,
    )


def summarize(root: Path, seed: int) -> dict:
    outcomes = {arm: load_outcome(root, arm, seed) for arm in ("B", "C", "D", "N")}
    if outcomes["C"]["status"] != "success" or outcomes["N"]["status"] != "success":
        raise RuntimeError("S4-C/N必须完成正式评价，不接受协议失败替代")

    gate_b = (
        common_gates(outcomes["B"]["config"])
        if outcomes["B"]["status"] == "success" else None
    )
    gate_c = patch_gates(outcomes["C"]["config"])
    gate_d = (
        patch_gates(outcomes["D"]["config"])
        if outcomes["D"]["status"] == "success" else None
    )
    eligible = [outcomes["C"]["config"]] if all(gate_c.values()) else []
    if gate_d is not None and all(gate_d.values()):
        eligible.append(outcomes["D"]["config"])
    selected = min(eligible, key=patch_selection_key) if eligible else None
    if selected is None:
        status = "no_patch_product_stop_s2b"
    elif len(eligible) == 1:
        status = "one_patch_candidate_selected"
    else:
        status = "two_patch_candidates_tiebreak_selected"
    return {
        "stage": "S2b", "seed": seed, "status": status,
        "pooled_s4b": {
            "status": outcomes["B"]["status"],
            "passed": gate_b is not None and all(gate_b.values()),
            "gates": gate_b,
            "failure": outcomes["B"].get("failure"),
        },
        "patch_s4c": {"passed": all(gate_c.values()), "gates": gate_c},
        "patch_s4d": {
            "status": outcomes["D"]["status"],
            "passed": gate_d is not None and all(gate_d.values()),
            "gates": gate_d,
            "failure": outcomes["D"].get("failure"),
        },
        "normalization_s4n": {
            "role": "diagnostic_only_not_for_selection",
            "results": outcomes["N"]["config"]["evaluation"]["normalization_diagnostic"],
        },
        "selected_patch_arm": selected["arm"] if selected else None,
        "selected_experiment": selected["experiment"] if selected else None,
        "selection_key": list(patch_selection_key(selected)) if selected else None,
        "next_step": (
            "freeze selected patch arm and rerun SAE seeds 202/503"
            if selected else
            "stop; do not add width/K/normalization/cosine/multiview hyperparameters"
        ),
    }


def main() -> None:
    args = parse_args()
    result = summarize(args.output_root, args.seed)
    output = args.output_root / f"clong_s2b_summary_seed{args.seed}.json"
    output.write_text(
        json.dumps(json_ready(result), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    rows = []
    for key in ("pooled_s4b", "patch_s4c", "patch_s4d"):
        rows.append({
            "arm": key,
            "status": result[key].get("status", "success"),
            "passed": result[key]["passed"],
            **(result[key]["gates"] or {}),
        })
    pd.DataFrame(rows).to_csv(
        args.output_root / f"clong_s2b_gates_seed{args.seed}.csv", index=False,
        encoding="utf-8-sig",
    )
    print(f"S2b汇总完成: {output}")
    print(f"状态={result['status']}; 唯一patch选择={result['selected_patch_arm']}")


if __name__ == "__main__":
    main()
