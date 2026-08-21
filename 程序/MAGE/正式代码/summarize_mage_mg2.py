#!/usr/bin/env python3
"""汇总MG2 A/B/C实验矩阵，并判断八条冻结放行门槛。

本工具读取三组实验产物、冻结beta校准JSON和M0-F seed42参照，校验全部
SHA绑定及产物完整性，再按2026-08-18协议评估冻结门槛：

1. arm A val patient AUC >= M0-F patient AUC - 0.01;
2. arm C val patient AUC >= M0-F patient AUC - 0.005;
3. C - A val patient AUC >= 0.005;
4. C val image AUC >= A val image AUC - 0.005;
5. C cancer mean normalized AiB and PGA both not below A, with at least one
   gain >= 0.05;
6. B val patient AUC >= A val patient AUC - 0.01 (logit KD sanity);
7. at each arm's own frozen threshold, C misses at most 1 more val cancer
   patient than A and C patient specificity drops by at most 0.03;
8. small-lesion stratum (train-frozen lesion_area_fraction tertiles) PGA drop
   <= 0.05; if the stratum has fewer than 15 images this gate degrades to
   report-only and the summary explicitly discloses it.

残缺或被改动的产物不会被汇总为成功。M0-F患者AUC从其配置JSON读取，只有
文件不可读时才使用预注册回退常量并明确警告。本工具只读取由训练/验证得到的
产物，测试集和外部数据均不参与；输出机器可读JSON和便于阅读的表格。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from build_mage_teacher_roi_manifest import PROJECT_ROOT, file_sha256
from train_mage_mg2_student import (
    DEFAULT_DEBUG_OUTPUT,
    DEFAULT_DEBUG_TEACHER_CACHE,
    DEFAULT_OUTPUT,
    DEFAULT_TEACHER_CACHE,
    MG2_ROOT,
    load_beta_calibration,
)


DEFAULT_M0F_REFERENCE = PROJECT_ROOT / (
    "结果/M0全量诊断_0804/正式验证集筛选/"
    "m0_full_keep_efficientnet_b0_seed42/config.json"
)
M0F_FALLBACK_PATIENT_AUC = 0.9133
DEFAULT_BETA_JSON = MG2_ROOT / "beta_calibration_seed42.json"
DEFAULT_DEBUG_BETA_JSON = DEFAULT_DEBUG_OUTPUT / "beta_calibration_debug.json"

GATE3_MIN_PATIENT_AUC_GAIN = 0.005
GATE4_MAX_IMAGE_AUC_DROP = 0.005
GATE5_MIN_SPATIAL_GAIN = 0.05
GATE6_MAX_PATIENT_AUC_DROP = 0.01
GATE7_MAX_EXTRA_FN_PATIENTS = 1
GATE7_MAX_SPECIFICITY_DROP = 0.03
GATE8_MAX_SMALL_PGA_DROP = 0.05
STRATUM_MIN_IMAGES = 15

DECISION_PROCEED = "proceed_to_MG3"
DECISION_DIAGNOSTIC = "stop_diagnostic_spatial_gain_only"
DECISION_STOP = "stop"


def parse_args() -> argparse.Namespace:
    """解析汇总输入，默认路径严格区分正式与调试产物。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=None,
                        help="三组arm产物根目录；缺省按正式/debug目录解析")
    parser.add_argument("--beta-calibration-json", type=Path, default=None)
    parser.add_argument("--m0f-reference", type=Path, default=DEFAULT_M0F_REFERENCE)
    parser.add_argument("--teacher-cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="机器可读汇总JSON；缺省写入MG2结果根（或debug目录）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true",
                        help="汇总debug产物，仅用于链路回归，不构成正式判定")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def arm_run_name(arm: str, seed: int, debug: bool) -> str:
    """返回各实验组冻结的运行目录名称。"""
    name = f"mg2_arm{arm.lower()}_efficientnet_b0_seed{seed}"
    return name + ("_debug" if debug else "")


def load_arm_products(run_dir: Path, arm: str) -> dict:
    """完成完整性与锁定标记校验后加载一个实验组配置。

    参数:
        run_dir (Path): 该arm输出目录。
        arm (str): ``"A"``/``"B"``/``"C"``，必须与config记录一致。
    返回:
        dict: 校验通过的config；缺config/checkpoint/预测CSV/训练历史任一
            文件、三项锁定标记任一非false、或arm标记不符，均快速失败，
            残缺产物拒绝被汇总当作成功。
    """
    required = [
        "config.json", "training_history.csv",
        "val_image_predictions.csv", "val_patient_predictions.csv",
        f"mg2_arm{arm.lower()}_best_student.pth",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"arm{arm}产物残缺，拒绝汇总: {run_dir} 缺少 {missing}"
        )
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("arm") != arm:
        raise ValueError(f"{run_dir} 的config arm标记为{config.get('arm')}，期望{arm}")
    for flag in ("test_evaluated", "internal_test_evaluated", "external_evaluated"):
        if config.get(flag):
            raise ValueError(f"arm{arm}的config锁定标记{flag}非false，拒绝汇总")
    return config


def extract_arm_metrics(config: dict) -> dict:
    """从已校验配置中提取放行门槛所需指标。

    参数:
        config (dict): load_arm_products校验过的config。
    返回:
        dict: val患者/图像AUC、空间汇总（含病灶大小分层）与患者阈值指标。
    """
    metrics = config["metrics"]
    return {
        "val_patient_auc": float(metrics["val_patient_auc"]),
        "val_image_auc": float(metrics["val_image_auc"]),
        "spatial": metrics["spatial"],
        "patient_threshold_metrics": metrics["patient_threshold_metrics"],
    }


def load_m0f_reference(
    reference_path: Path, debug: bool
) -> tuple[float, str, str | None]:
    """读取M0-F seed42安全参照患者AUC。

    参数:
        reference_path (Path): M0-F正式run的config.json路径。
        debug (bool): True时读取失败允许回退预注册常量0.9133并显式警告；
            正式模式下读取失败直接抛错终止（快速失败）。
    返回:
        tuple: ``(patient_auc, source, warning)``；正常source为
            ``"config_json"``；debug回退时source为 ``"fallback_constant"``。
    """
    try:
        payload = json.loads(reference_path.read_text(encoding="utf-8"))
        value = float(payload["best_val_patient_auc"])
        if not math.isfinite(value):
            raise ValueError("M0-F参照患者AUC非有限值")
        return value, "config_json", None
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if not debug:
            raise RuntimeError(
                f"正式模式必须读取M0-F参照config，拒绝回退常量: "
                f"{reference_path} ({exc})"
            ) from exc
        warning = (
            f"警告: 未能从{reference_path}读取M0-F参照({exc})，"
            f"debug模式回退预注册常量{M0F_FALLBACK_PATIENT_AUC}"
        )
        print(warning)
        return M0F_FALLBACK_PATIENT_AUC, "fallback_constant", warning


def verify_cross_arm_consistency(configs: dict, calibration: dict) -> None:
    """在判断门槛前校验三组实验的深层一致性。

    参数:
        configs (dict): ``{"A"/"B"/"C": config}``，每个config已经过
            load_arm_products完整性校验并附加 ``_run_dir``。
        calibration (dict): load_beta_calibration校验通过的校准payload，
            含 ``_self_path``/``_self_sha256``。
    返回: 无；以下任一不符即抛ValueError：每组checkpoint实际SHA256等于
        config记录值；三组seed均为42；三组batch size、阶段epoch上限、
        学习率、weight decay、patience、采样器标识、增强配置与学生架构
        完全一致；三组教师checkpoint SHA、manifest SHA、v3 audit SHA一致；
        B/C绑定同一教师缓存（路径与SHA）；C组config的beta校准JSON路径与
        SHA等于正式冻结校准JSON。
    """
    arms = ("A", "B", "C")
    for arm in arms:
        config = configs[arm]
        checkpoint = Path(config["_run_dir"]) / f"mg2_arm{arm.lower()}_best_student.pth"
        actual_sha = file_sha256(checkpoint)
        if actual_sha != config.get("checkpoint_sha256"):
            raise ValueError(f"arm{arm} checkpoint SHA与config记录不一致")
        if int(config.get("seed", -1)) != 42:
            raise ValueError(f"arm{arm} seed不是42: {config.get('seed')}")

    def training_key(arm: str) -> dict:
        """提取A/B/C必须一致的训练配置，排除实验条件混杂。"""
        training = configs[arm]["training"]
        return {
            name: training[name]
            for name in (
                "batch_size", "stage_a_epochs", "stage_b_epochs",
                "stage_a_lr", "stage_b_lr", "weight_decay", "patience", "sampler",
            )
        }

    for arm in ("B", "C"):
        if training_key(arm) != training_key("A"):
            raise ValueError(f"arm{arm}与armA的训练超参/采样器标识不一致")
        if configs[arm]["input_protocol"] != configs["A"]["input_protocol"]:
            raise ValueError(f"arm{arm}与armA的增强配置不一致")
        if configs[arm]["architecture"] != configs["A"]["architecture"]:
            raise ValueError(f"arm{arm}与armA的学生架构不一致")
        for field in ("teacher_checkpoint_sha256", "manifest_sha256", "v3_audit_sha256"):
            if configs[arm].get(field) != configs["A"].get(field):
                raise ValueError(f"arm{arm}与armA的{field}不一致")
    for arm in ("B", "C"):
        for field in ("teacher_cache", "teacher_cache_sha256"):
            if configs["B"].get(field) != configs[arm].get(field):
                raise ValueError("B/C未绑定同一教师缓存")
    c_training = configs["C"]["training"]
    if c_training.get("beta_calibration_json") != calibration["_self_path"]:
        raise ValueError("armC config的beta校准JSON路径与正式冻结校准JSON不一致")
    if c_training.get("beta_calibration_json_sha256") != calibration["_self_sha256"]:
        raise ValueError("armC config的beta校准JSON SHA与正式冻结校准JSON不一致")


def evaluate_gates(arm_metrics: dict, m0f_patient_auc: float) -> dict:
    """以纯函数方式判断八条冻结MG2放行门槛。

    参数:
        arm_metrics (dict): ``{"A"/"B"/"C": extract_arm_metrics结果}``。
        m0f_patient_auc (float): M0-F seed42患者AUC参照值。
    返回:
        dict: ``gates`` 每条门槛的实际值/门槛值/通过与否（gate8降级时
            passed=None且degraded=True并附披露文本）与整体 ``decision``：
            八条全过为 ``proceed_to_MG3``；门槛5通过但门槛1-4或7-8有失败为
            ``stop_diagnostic_spatial_gain_only``；其余为 ``stop``。
    """
    a = arm_metrics["A"]
    b = arm_metrics["B"]
    c = arm_metrics["C"]
    a_patient, b_patient, c_patient = (
        a["val_patient_auc"], b["val_patient_auc"], c["val_patient_auc"]
    )
    a_image, c_image = a["val_image_auc"], c["val_image_auc"]
    a_naib = float(a["spatial"]["mean_normalized_aib"])
    c_naib = float(c["spatial"]["mean_normalized_aib"])
    a_pga = float(a["spatial"]["pga"])
    c_pga = float(c["spatial"]["pga"])
    a_threshold = a["patient_threshold_metrics"]
    c_threshold = c["patient_threshold_metrics"]
    fn_a = int(a_threshold["confusion_matrix"][2])
    fn_c = int(c_threshold["confusion_matrix"][2])
    spec_a = float(a_threshold["specificity"])
    spec_c = float(c_threshold["specificity"])
    small = c["spatial"]["lesion_size_strata"]["small"]
    small_images = int(small["images"])
    small_pga_drop = None
    if small_images >= STRATUM_MIN_IMAGES:
        small_pga_drop = float(a["spatial"]["lesion_size_strata"]["small"]["pga"]) - float(
            small["pga"]
        )

    gates = {
        "gate1_armA_patient_auc_floor": {
            "description": "A组val患者AUC >= M0-F患者AUC - 0.01",
            "actual": a_patient,
            "threshold": m0f_patient_auc - 0.01,
            "passed": bool(a_patient >= m0f_patient_auc - 0.01),
        },
        "gate2_armC_patient_auc_floor": {
            "description": "C组val患者AUC >= M0-F患者AUC - 0.005",
            "actual": c_patient,
            "threshold": m0f_patient_auc - 0.005,
            "passed": bool(c_patient >= m0f_patient_auc - 0.005),
        },
        "gate3_patient_auc_gain_C_minus_A": {
            "description": "C-A组val患者AUC >= 0.005",
            "actual": c_patient - a_patient,
            "threshold": GATE3_MIN_PATIENT_AUC_GAIN,
            "passed": bool(c_patient - a_patient >= GATE3_MIN_PATIENT_AUC_GAIN),
        },
        "gate4_image_auc_C_not_below_A": {
            "description": "C组val图像AUC不低于A组-0.005",
            "actual": c_image - a_image,
            "threshold": -GATE4_MAX_IMAGE_AUC_DROP,
            "passed": bool(c_image - a_image >= -GATE4_MAX_IMAGE_AUC_DROP),
        },
        "gate5_spatial_gain_C_over_A": {
            "description": (
                "C组癌图mean normalized AiB与PGA均不低于A组，且至少一项提升>=0.05"
            ),
            "actual": {
                "normalized_aib_A": a_naib, "normalized_aib_C": c_naib,
                "normalized_aib_gain": c_naib - a_naib,
                "pga_A": a_pga, "pga_C": c_pga, "pga_gain": c_pga - a_pga,
            },
            "threshold": {"min_gain_either": GATE5_MIN_SPATIAL_GAIN},
            "passed": bool(
                c_naib >= a_naib and c_pga >= a_pga and (
                    c_naib - a_naib >= GATE5_MIN_SPATIAL_GAIN
                    or c_pga - a_pga >= GATE5_MIN_SPATIAL_GAIN
                )
            ),
        },
        "gate6_armB_logit_kd_sanity": {
            "description": "B组val患者AUC >= A组 - 0.01",
            "actual": b_patient - a_patient,
            "threshold": -GATE6_MAX_PATIENT_AUC_DROP,
            "passed": bool(b_patient >= a_patient - GATE6_MAX_PATIENT_AUC_DROP),
        },
        "gate7_clinical_safety_C_over_A": {
            "description": (
                "各自冻结阈值下C相对A最多多漏诊1名val癌患者，"
                "且C患者Specificity下降不超过0.03"
            ),
            "actual": {
                "extra_missed_cancer_patients": fn_c - fn_a,
                "specificity_drop": spec_a - spec_c,
                "fn_A": fn_a, "fn_C": fn_c,
                "specificity_A": spec_a, "specificity_C": spec_c,
                "threshold_A": a_threshold["threshold"],
                "threshold_C": c_threshold["threshold"],
            },
            "threshold": {
                "max_extra_missed": GATE7_MAX_EXTRA_FN_PATIENTS,
                "max_specificity_drop": GATE7_MAX_SPECIFICITY_DROP,
            },
            "passed": bool(
                fn_c - fn_a <= GATE7_MAX_EXTRA_FN_PATIENTS
                and spec_a - spec_c <= GATE7_MAX_SPECIFICITY_DROP
            ),
        },
    }
    gate8 = {
        "description": "新口径small病灶组PGA下降不超过0.05；不足15张降级为只报告",
        "small_stratum_images": small_images,
        "min_images_for_hard_gate": STRATUM_MIN_IMAGES,
        "degraded": small_images < STRATUM_MIN_IMAGES,
    }
    if small_images >= STRATUM_MIN_IMAGES:
        gate8.update({
            "actual": small_pga_drop,
            "threshold": GATE8_MAX_SMALL_PGA_DROP,
            "passed": bool(small_pga_drop <= GATE8_MAX_SMALL_PGA_DROP),
            "disclosure": None,
        })
    else:
        gate8.update({
            "actual": {
                "pga_A": a["spatial"]["lesion_size_strata"]["small"]["pga"],
                "pga_C": small["pga"],
            },
            "threshold": None,
            "passed": None,
            "disclosure": (
                f"small病灶组仅{small_images}张(<{STRATUM_MIN_IMAGES})，"
                "门槛8降级为只报告，不参与放行判定"
            ),
        })
    gates["gate8_small_lesion_pga"] = gate8

    classification_gates = [
        gates["gate1_armA_patient_auc_floor"]["passed"],
        gates["gate2_armC_patient_auc_floor"]["passed"],
        gates["gate3_patient_auc_gain_C_minus_A"]["passed"],
        gates["gate4_image_auc_C_not_below_A"]["passed"],
        gates["gate7_clinical_safety_C_over_A"]["passed"],
        gate8["passed"] is not False,
    ]
    if (
        all(classification_gates)
        and gates["gate5_spatial_gain_C_over_A"]["passed"]
        and gates["gate6_armB_logit_kd_sanity"]["passed"]
    ):
        decision = DECISION_PROCEED
    elif gates["gate5_spatial_gain_C_over_A"]["passed"] and not all(classification_gates):
        decision = DECISION_DIAGNOSTIC
    else:
        decision = DECISION_STOP
    return {"gates": gates, "decision": decision}


def print_table(summary: dict) -> None:
    """打印便于阅读的门槛表格和总体结论。"""
    print(f"M0-F参照患者AUC: {summary['m0f_reference']['patient_auc']:.4f} "
          f"({summary['m0f_reference']['source']})")
    if summary["m0f_reference"]["warning"]:
        print(summary["m0f_reference"]["warning"])
    for arm in "ABC":
        metrics = summary["arms"][arm]["metrics"]
        print(
            f"arm {arm}: 患者AUC={metrics['val_patient_auc']:.4f} "
            f"图像AUC={metrics['val_image_auc']:.4f} "
            f"nAiB={metrics['spatial']['mean_normalized_aib']:.4f} "
            f"PGA={metrics['spatial']['pga']:.4f}"
        )
    print(f"{'门槛':<44} {'实际值':<28} {'门槛值':<24} 结果")
    for name, gate in summary["gates"].items():
        actual = gate.get("actual")
        threshold = gate.get("threshold")
        if gate.get("degraded"):
            verdict = "降级只报告"
        else:
            verdict = "通过" if gate["passed"] else "未通过"
        actual_text = (
            json.dumps(actual, ensure_ascii=False)
            if isinstance(actual, dict) else
            ("-" if actual is None else f"{actual:.4f}")
        )
        threshold_text = (
            json.dumps(threshold, ensure_ascii=False)
            if isinstance(threshold, dict) else
            ("-" if threshold is None else f"{threshold:.4f}")
        )
        print(f"{name:<44} {actual_text:<28} {threshold_text:<24} {verdict}")
        if gate.get("disclosure"):
            print(f"  披露: {gate['disclosure']}")
    print(f"整体判定: {summary['decision']}")


def main() -> None:
    """校验绑定、加载三组实验、判断门槛并写出汇总。"""
    args = parse_args()
    run_root = args.run_root
    if run_root is None:
        run_root = DEFAULT_DEBUG_OUTPUT if args.debug else DEFAULT_OUTPUT
    run_root = run_root.resolve()
    calibration_path = args.beta_calibration_json
    if calibration_path is None:
        calibration_path = (
            DEFAULT_DEBUG_BETA_JSON if args.debug else DEFAULT_BETA_JSON
        )
    cache_path = args.teacher_cache
    if cache_path is None:
        cache_path = DEFAULT_DEBUG_TEACHER_CACHE if args.debug else DEFAULT_TEACHER_CACHE

    configs = {}
    for arm in "ABC":
        run_dir = run_root / arm_run_name(arm, args.seed, args.debug)
        configs[arm] = load_arm_products(run_dir, arm)
        configs[arm]["_run_dir"] = str(run_dir)
        if bool(configs[arm].get("debug")) != bool(args.debug):
            raise ValueError(f"arm{arm} config的debug标记与汇总模式不一致")

    # beta强绑定交叉核验：三组config的manifest SHA、C组beta与校准JSON必须一致。
    manifest_shas = {configs[arm]["manifest_sha256"] for arm in "ABC"}
    if len(manifest_shas) != 1:
        raise ValueError(f"三组config的manifest SHA不一致: {manifest_shas}")
    c_training = configs["C"]["training"]
    calibration = load_beta_calibration(
        calibration_path.resolve(), manifest_shas.pop(), cache_path.resolve(),
        int(configs["C"]["seed"]),
        int(c_training["batch_size"]), bool(configs["C"]["debug"]),
    )
    if not c_training.get("beta_frozen"):
        raise ValueError("armC config的beta未标记为校准冻结，拒绝汇总")
    if not math.isclose(
        float(c_training["beta"]), float(calibration["beta"]), abs_tol=1e-12
    ):
        raise ValueError("armC config的beta与校准JSON不一致")
    verify_cross_arm_consistency(configs, calibration)

    m0f_auc, m0f_source, m0f_warning = load_m0f_reference(
        args.m0f_reference.resolve(), args.debug
    )
    arm_metrics = {arm: extract_arm_metrics(configs[arm]) for arm in "ABC"}
    verdict = evaluate_gates(arm_metrics, m0f_auc)

    summary = {
        "stage": "MG2_gate_summary",
        "debug": bool(args.debug),
        "seed": args.seed,
        "run_root": str(run_root),
        "beta_calibration": {
            "path": calibration["_self_path"],
            "sha256": calibration["_self_sha256"],
            "beta": float(calibration["beta"]),
            "rule": calibration["rule"],
            "seed": calibration["seed"],
            "batch_size": calibration["batch_size"],
            "device": calibration.get("device"),
            "binding_validated": True,
        },
        "m0f_reference": {
            "path": str(args.m0f_reference.resolve()),
            "patient_auc": m0f_auc,
            "source": m0f_source,
            "warning": m0f_warning,
        },
        "arms": {
            arm: {
                "run_dir": str(run_root / arm_run_name(arm, args.seed, args.debug)),
                "complete": True,
                "metrics": arm_metrics[arm],
            }
            for arm in "ABC"
        },
        "gates": verdict["gates"],
        "decision": verdict["decision"],
        "test_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
    }
    output = args.output
    if output is None:
        name = "mg2_gate_summary_debug.json" if args.debug else "mg2_gate_summary_seed42.json"
        output = (DEFAULT_DEBUG_OUTPUT if args.debug else MG2_ROOT) / name
    output = output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"汇总输出已存在，拒绝覆盖（或显式--overwrite）: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_table(summary)
    print(f"汇总JSON: {output}")
    if args.debug:
        print("注意: 本汇总基于debug产物，仅验证判定与披露链路，不构成正式结论。")


if __name__ == "__main__":
    main()
