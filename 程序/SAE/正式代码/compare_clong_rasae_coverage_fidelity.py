#!/usr/bin/env python3
"""逐项比较RA-SAE扩大可抽样患者池前后的完整val纯重构保真。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "结果/SAE/RA_SAE_Pilot_20260908"
BASELINE = BASE / "full_val_fidelity_20260915/summary.json"


def main() -> None:
    """读取基线与候选的全精度JSON，写入逐项差值和联合目标判定。

    Args:
        None: 候选summary、抽样summary和输出文件由CLI提供。

    Returns:
        None: 不进行任何新评价或模型选择。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--sampling-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline = json.loads(BASELINE.read_text())
    candidate = json.loads(args.candidate_summary.read_text())
    sampling = json.loads(args.sampling_summary.read_text())
    paths = {
        "patient_probability_abs_change_mean": ("prediction", "patient_probability_abs_change", "mean"),
        "patient_prediction_agreement": ("prediction", "patient_prediction_agreement"),
        "patch_fvu": ("reconstruction", "patch_fvu"),
        "pooled_fvu": ("reconstruction", "pooled_fvu"),
        "reconstructed_patient_auc": ("prediction", "reconstructed_patient_auc"),
    }

    def get(source: dict, path: tuple[str, ...]) -> float:
        value = source
        for key in path:
            value = value[key]
        return float(value)

    metrics = {}
    for name, path in paths.items():
        old, new = get(baseline, path), get(candidate, path)
        metrics[name] = {"baseline": old, "candidate": new, "delta_candidate_minus_baseline": new - old}
    conditions = {
        "patient_probability_abs_change_lower": (
            metrics["patient_probability_abs_change_mean"]["candidate"]
            < metrics["patient_probability_abs_change_mean"]["baseline"]
        ),
        "patient_prediction_agreement_higher": (
            metrics["patient_prediction_agreement"]["candidate"]
            > metrics["patient_prediction_agreement"]["baseline"]
        ),
        "patch_fvu_lower": metrics["patch_fvu"]["candidate"] < metrics["patch_fvu"]["baseline"],
        "pooled_fvu_lower": metrics["pooled_fvu"]["candidate"] < metrics["pooled_fvu"]["baseline"],
        "reconstructed_patient_auc_not_lower": (
            metrics["reconstructed_patient_auc"]["candidate"]
            >= metrics["reconstructed_patient_auc"]["baseline"]
        ),
    }
    passed = sum(conditions.values())
    if passed == len(conditions):
        status = "joint_target_met_directional_single_seed"
    elif passed:
        status = "improvements_inconsistent_joint_target_not_met"
    else:
        status = "no_directional_metric_improved"
    result = {
        "question": "Does a broader optimization sampling pool improve reconstruction fidelity under fixed representatives, structure, and update budget?",
        "metrics": metrics, "conditions": conditions,
        "joint_status": status, "conditions_met": passed, "conditions_total": len(conditions),
        "sampling": sampling,
        "limits": [
            "single seed directional evidence only",
            "medical semantics not evaluated",
            "new model Feature IDs do not inherit old medical labels",
            "fidelity change does not establish a cause of semantic mixing",
        ],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
