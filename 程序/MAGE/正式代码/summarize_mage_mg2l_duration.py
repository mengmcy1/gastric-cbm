#!/usr/bin/env python3
"""Summarize MG2-L and select the attention-first SAE handoff product.

MG2-L compares A-long (CE) with C-long (CE + logit KD + attention KD) under
the same extended stage-B budget.  It does not revise the original MG2 gate
decision.  A candidate can enter SAE only when spatial alignment improves and
classification remains inside the frozen safety envelope.  If both original
MG2-C and C-long are eligible, selection is lexicographic by normalized AiB,
then PGA, then patient AUC.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from build_mage_teacher_roi_manifest import PROJECT_ROOT, file_sha256
from summarize_mage_mg2 import (
    DEFAULT_BETA_JSON,
    DEFAULT_M0F_REFERENCE,
    load_arm_products,
    load_m0f_reference,
)
from train_mage_mg2_student import (
    DEFAULT_TEACHER_CACHE,
    FROZEN_BETA_CALIBRATION_SHA256,
    FROZEN_TEACHER_SHA256,
    load_beta_calibration,
)


MG2_ROOT = PROJECT_ROOT / "结果/MAGE/MG2全图学生蒸馏_20260818/正式验证集筛选"
MG2L_ROOT = PROJECT_ROOT / "结果/MAGE/MG2L训练轮数敏感性_20260818"
DEFAULT_RUN_ROOT = MG2L_ROOT / "正式验证集筛选"
DEFAULT_OUTPUT = MG2L_ROOT / "mg2l_attention_sae_summary_seed42.json"

PATIENT_AUC_TOLERANCE = 0.01
IMAGE_AUC_DROP_TOLERANCE = 0.005
MIN_SPATIAL_GAIN = 0.05
MAX_EXTRA_FN = 1
MAX_SPECIFICITY_DROP = 0.03
MAX_SMALL_PGA_DROP = 0.05
MIN_STRATUM_IMAGES = 15


def parse_args() -> argparse.Namespace:
    """Parse result roots and the machine-readable output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_dir(root: Path, arm: str, long_run: bool) -> Path:
    """Return the frozen result directory for one original or long arm."""
    prefix = "mg2l" if long_run else "mg2"
    return root / f"{prefix}_arm{arm.lower()}_efficientnet_b0_seed42"


def metric_view(config: dict) -> dict:
    """Extract metrics needed by the attention-first safety gates."""
    metrics = config["metrics"]
    return {
        "patient_auc": float(metrics["val_patient_auc"]),
        "image_auc": float(metrics["val_image_auc"]),
        "spatial": metrics["spatial"],
        "threshold": metrics["patient_threshold_metrics"],
        "checkpoint": config["checkpoint"],
        "checkpoint_sha256": config["checkpoint_sha256"],
    }


def verify_pair(configs: dict, calibration: dict) -> None:
    """Validate the A-long/C-long product lineage and equal training budget."""
    for arm, config in configs.items():
        if int(config.get("seed", -1)) != 42 or bool(config.get("debug")):
            raise ValueError(f"{arm}-long的seed/debug标记不符合正式协议")
        for flag in ("test_evaluated", "internal_test_evaluated", "external_evaluated"):
            if config.get(flag):
                raise ValueError(f"{arm}-long的{flag}非false")
        checkpoint = Path(config["checkpoint"])
        if not checkpoint.is_file() or file_sha256(checkpoint) != config["checkpoint_sha256"]:
            raise ValueError(f"{arm}-long checkpoint缺失或SHA不一致")
        if config.get("teacher_checkpoint_sha256") != FROZEN_TEACHER_SHA256:
            raise ValueError(f"{arm}-long教师SHA不符合冻结值")

    for field in ("manifest_sha256", "v3_audit_sha256", "teacher_checkpoint_sha256"):
        values = {configs[arm].get(field) for arm in "AC"}
        if len(values) != 1:
            raise ValueError(f"A-long/C-long的{field}不一致: {values}")

    comparable_training = (
        "batch_size", "stage_a_epochs", "stage_b_epochs", "stage_a_lr",
        "stage_b_lr", "weight_decay", "patience", "optimizer",
        "label_smoothing", "sampler", "selection",
    )
    for field in comparable_training:
        values = {json.dumps(configs[arm]["training"].get(field), sort_keys=True) for arm in "AC"}
        if len(values) != 1:
            raise ValueError(f"A-long/C-long训练字段{field}不一致: {values}")
    if int(configs["A"]["training"]["stage_b_epochs"]) != 40:
        raise ValueError("MG2-L stageB上限不是冻结的40轮")
    if configs["A"]["input_protocol"] != configs["C"]["input_protocol"]:
        raise ValueError("A-long/C-long输入协议不一致")
    if configs["A"]["architecture"] != configs["C"]["architecture"]:
        raise ValueError("A-long/C-long架构不一致")

    c_training = configs["C"]["training"]
    if not c_training.get("beta_frozen"):
        raise ValueError("C-long beta未标记为冻结校准值")
    if c_training.get("beta_calibration_json_sha256") != calibration["_self_sha256"]:
        raise ValueError("C-long记录的beta校准JSON SHA不一致")
    if not math.isclose(float(c_training["beta"]), float(calibration["beta"]), abs_tol=1e-12):
        raise ValueError("C-long beta与冻结校准JSON不一致")


def attention_sae_gates(control: dict, candidate: dict, m0f_auc: float) -> dict:
    """Evaluate the frozen five attention-first SAE handoff requirements."""
    a, c = metric_view(control), metric_view(candidate)
    a_spatial, c_spatial = a["spatial"], c["spatial"]
    naib_gain = float(c_spatial["mean_normalized_aib"]) - float(
        a_spatial["mean_normalized_aib"]
    )
    pga_gain = float(c_spatial["pga"]) - float(a_spatial["pga"])
    fn_a = int(a["threshold"]["confusion_matrix"][2])
    fn_c = int(c["threshold"]["confusion_matrix"][2])
    spec_drop = float(a["threshold"]["specificity"]) - float(
        c["threshold"]["specificity"]
    )
    small_a = a_spatial["lesion_size_strata"]["small"]
    small_c = c_spatial["lesion_size_strata"]["small"]
    small_images = int(small_c["images"])
    small_drop = float(small_a["pga"]) - float(small_c["pga"])
    gates = {
        "spatial_alignment": {
            "actual": {"normalized_aib_gain": naib_gain, "pga_gain": pga_gain},
            "passed": bool(naib_gain >= 0 and pga_gain >= 0 and max(naib_gain, pga_gain) >= MIN_SPATIAL_GAIN),
        },
        "patient_auc_safety": {
            "actual": c["patient_auc"], "threshold": m0f_auc - PATIENT_AUC_TOLERANCE,
            "passed": bool(c["patient_auc"] >= m0f_auc - PATIENT_AUC_TOLERANCE),
        },
        "image_auc_safety": {
            "actual": c["image_auc"] - a["image_auc"], "threshold": -IMAGE_AUC_DROP_TOLERANCE,
            "passed": bool(c["image_auc"] >= a["image_auc"] - IMAGE_AUC_DROP_TOLERANCE),
        },
        "clinical_safety": {
            "actual": {"extra_fn": fn_c - fn_a, "specificity_drop": spec_drop},
            "passed": bool(fn_c - fn_a <= MAX_EXTRA_FN and spec_drop <= MAX_SPECIFICITY_DROP),
        },
        "small_lesion_pga": {
            "actual": small_drop, "images": small_images,
            "degraded": small_images < MIN_STRATUM_IMAGES,
            "passed": None if small_images < MIN_STRATUM_IMAGES else bool(small_drop <= MAX_SMALL_PGA_DROP),
        },
    }
    eligible = all(
        gate["passed"] is not False for gate in gates.values()
    )
    return {"eligible_for_sae": eligible, "gates": gates}


def main() -> None:
    """Validate both experiments, evaluate gates and freeze one SAE product."""
    args = parse_args()
    run_root = args.run_root.resolve()
    long_configs = {
        arm: load_arm_products(run_dir(run_root, arm, True), arm) for arm in "AC"
    }
    calibration = load_beta_calibration(
        DEFAULT_BETA_JSON.resolve(), long_configs["C"]["manifest_sha256"],
        DEFAULT_TEACHER_CACHE.resolve(), 42,
        int(long_configs["C"]["training"]["batch_size"]), False,
    )
    if calibration["_self_sha256"] != FROZEN_BETA_CALIBRATION_SHA256:
        raise ValueError("MG2-L校准JSON不是已冻结正式文件")
    verify_pair(long_configs, calibration)

    current_configs = {
        arm: load_arm_products(run_dir(MG2_ROOT, arm, False), arm) for arm in "AC"
    }
    m0f_auc, m0f_source, warning = load_m0f_reference(
        DEFAULT_M0F_REFERENCE.resolve(), debug=False
    )
    candidates = {
        "MG2_C_epoch20": {
            "control": metric_view(current_configs["A"]),
            "candidate": metric_view(current_configs["C"]),
            **attention_sae_gates(current_configs["A"], current_configs["C"], m0f_auc),
        },
        "MG2L_C_long": {
            "control": metric_view(long_configs["A"]),
            "candidate": metric_view(long_configs["C"]),
            **attention_sae_gates(long_configs["A"], long_configs["C"], m0f_auc),
        },
    }
    eligible = [name for name, value in candidates.items() if value["eligible_for_sae"]]
    if not eligible:
        raise RuntimeError("MG2与MG2-L均未通过注意力优先SAE交接口径")
    selected = max(
        eligible,
        key=lambda name: (
            candidates[name]["candidate"]["spatial"]["mean_normalized_aib"],
            candidates[name]["candidate"]["spatial"]["pga"],
            candidates[name]["candidate"]["patient_auc"],
        ),
    )
    output = {
        "stage": "MG2L_attention_first_SAE_handoff",
        "seed": 42,
        "m0f_reference": {"patient_auc": m0f_auc, "source": m0f_source, "warning": warning},
        "selection_order": ["mean_normalized_aib", "PGA", "patient_AUC"],
        "candidates": candidates,
        "selected": selected,
        "selected_checkpoint": candidates[selected]["candidate"]["checkpoint"],
        "selected_checkpoint_sha256": candidates[selected]["candidate"]["checkpoint_sha256"],
        "original_mg2_decision_unchanged": "stop_diagnostic_spatial_gain_only",
        "test_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
    }
    output_path = args.output.resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"汇总输出已存在，拒绝覆盖: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, value in candidates.items():
        metric = value["candidate"]
        print(
            f"{name}: eligible={value['eligible_for_sae']} "
            f"patient AUC={metric['patient_auc']:.4f} "
            f"image AUC={metric['image_auc']:.4f} "
            f"nAiB={metric['spatial']['mean_normalized_aib']:.4f} "
            f"PGA={metric['spatial']['pga']:.4f}"
        )
    print(f"SAE交接选择: {selected}")
    print(f"checkpoint: {output['selected_checkpoint']}")
    print(f"汇总JSON: {output_path}")


if __name__ == "__main__":
    main()
