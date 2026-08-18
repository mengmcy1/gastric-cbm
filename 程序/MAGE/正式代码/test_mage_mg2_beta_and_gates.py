#!/usr/bin/env python3
"""Unit tests for the MG2 beta binding, calibration epoch and gate judgement.

Covers the frozen defenses added after protocol review (2026-08-18):

1. beta JSON strong binding (``load_beta_calibration``/``resolve_beta_binding``):
   manual ``--beta`` on formal arm C is rejected, missing/tampered calibration
   JSON is rejected, a correctly bound JSON passes and supplies beta; arms A/B
   reject calibration files;
2. calibration-epoch completeness (``verify_calibration_epoch``): exactly one
   complete sampling epoch (74 batches / 2350 draws at batch 32) is required;
   73/75 batches or a wrong draw count fail;
3. with-replacement sampling audit fields (``calibration_sampling_audit``);
4. the eight-gate judgement (``summarize_mage_mg2.evaluate_gates``): all-pass,
   diagnostic-stop and stop paths, plus the gate-8 degraded disclosure;
5. incomplete arm products are refused (``load_arm_products``).

Runnable both as ``pytest test_mage_mg2_beta_and_gates.py`` and directly as
``python test_mage_mg2_beta_and_gates.py``.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_mage_teacher_roi_manifest import file_sha256  # noqa: E402
from summarize_mage_mg2 import (  # noqa: E402
    DECISION_DIAGNOSTIC,
    DECISION_PROCEED,
    DECISION_STOP,
    M0F_FALLBACK_PATIENT_AUC,
    evaluate_gates,
    load_arm_products,
    load_m0f_reference,
    verify_cross_arm_consistency,
)
from train_mage_mg2_student import (  # noqa: E402
    BETA_CALIBRATION_RULE,
    BETA_CALIBRATION_STAGE,
    DEFAULT_BETA,
    FROZEN_BETA_CALIBRATION_SHA256,
    FROZEN_TEACHER_SHA256,
    calibration_sampling_audit,
    load_beta_calibration,
    resolve_beta_binding,
    verify_calibration_epoch,
    verify_frozen_calibration,
)

MANIFEST_SHA = "m" * 64


def _write_calibration(tmp: Path, cache_path: Path, **overrides) -> Path:
    """Write one synthetic calibration JSON bound to ``cache_path``."""
    payload = {
        "stage": BETA_CALIBRATION_STAGE,
        "rule": BETA_CALIBRATION_RULE,
        "seed": 42,
        "batch_size": 32,
        "debug": False,
        "beta": 0.123,
        "manifest_sha256": MANIFEST_SHA,
        "teacher_checkpoint_sha256": FROZEN_TEACHER_SHA256,
        "teacher_cache": str(cache_path.resolve()),
        "teacher_cache_sha256": file_sha256(cache_path),
    }
    payload.update(overrides)
    path = tmp / "beta_calibration.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _args(**overrides) -> SimpleNamespace:
    """Build one minimal argument namespace for resolve_beta_binding."""
    base = dict(arm="C", beta=None, beta_calibration_json=None,
                seed=42, batch_size=32, debug=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_beta_json_strong_binding() -> None:
    """beta强绑定：手工beta/缺文件/任一SHA或参数不符被拒，正确JSON通过。

    字段级绑定在debug模式下用合成JSON验证（正式模式另由冻结SHA防线覆盖，
    见test_frozen_calibration_binding与test_formal_mode_requires_frozen_sha）。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cache = tmp / "cache.pt"
        cache.write_bytes(b"synthetic-cache")
        calibration = _write_calibration(tmp, cache, debug=True)

        # 正确绑定：beta只能来自JSON。
        beta, payload = resolve_beta_binding(
            _args(beta_calibration_json=calibration, debug=True), MANIFEST_SHA, cache
        )
        assert beta == 0.123 and payload["_self_sha256"] == file_sha256(calibration)
        assert payload["_frozen_sha_verified"] is False

        # 正式C组手工--beta被拒（等价于被禁的--beta 1.0 --beta-frozen路径）。
        for bad_args, message in [
            (_args(beta=1.0, beta_calibration_json=calibration), "正式C手工beta"),
            (_args(), "正式C缺校准JSON"),
            (_args(arm="A", beta_calibration_json=calibration), "A组接受校准文件"),
            (_args(arm="B", beta_calibration_json=calibration), "B组接受校准文件"),
        ]:
            try:
                resolve_beta_binding(bad_args, MANIFEST_SHA, cache)
            except (ValueError, FileNotFoundError):
                pass
            else:
                raise AssertionError(f"未拒绝: {message}")

        # 校准文件不存在即快速失败。
        try:
            resolve_beta_binding(
                _args(beta_calibration_json=tmp / "missing.json", debug=True),
                MANIFEST_SHA, cache,
            )
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("未拒绝缺失校准JSON")

        # 逐项篡改SHA/seed/batch size/debug标记，全部必须被拒。
        tampered = [
            {"manifest_sha256": "x" * 64},
            {"teacher_checkpoint_sha256": "0" * 64},
            {"teacher_cache_sha256": "0" * 64},
            {"seed": 1},
            {"batch_size": 16},
            {"debug": False},
            {"rule": "beta = hand"},
        ]
        for override in tampered:
            options = {"debug": True}
            options.update(override)
            broken = _write_calibration(tmp, cache, **options)
            try:
                resolve_beta_binding(
                    _args(beta_calibration_json=broken, debug=True),
                    MANIFEST_SHA, cache,
                )
            except ValueError:
                pass
            else:
                raise AssertionError(f"未检测到篡改: {override}")

        # debug占位路径保留：无校准文件时回退占位beta。
        beta_debug, payload_debug = resolve_beta_binding(
            _args(debug=True), MANIFEST_SHA, cache
        )
        assert beta_debug == DEFAULT_BETA and payload_debug is None
        beta_manual, _ = resolve_beta_binding(
            _args(debug=True, beta=2.0), MANIFEST_SHA, cache
        )
        assert beta_manual == 2.0


def _epoch_records(batches: int, draws: int) -> list[dict]:
    """Build synthetic batch records with the requested batch/draw counts."""
    records = []
    remaining = draws
    for index in range(batches):
        size = min(32, remaining - 32 * (batches - index - 1))
        records.append({"batch_index": index, "sha256": ["s"] * size})
        remaining -= size
    assert remaining == 0
    return records


def test_calibration_epoch_completeness() -> None:
    """校准epoch完整性：恰好74批/2350次通过；73/75批或抽样数不符被拒。"""
    verify_calibration_epoch(_epoch_records(74, 2350), 2350, 32)
    for batches, draws, message in [
        (73, 2336, "73批不足"),
        (75, 2350 + 14, "75批超出"),
        (74, 2351, "抽样次数不符"),
    ]:
        try:
            verify_calibration_epoch(_epoch_records(batches, draws), 2350, 32)
        except ValueError:
            pass
        else:
            raise AssertionError(f"未拒绝: {message}")
    # debug子集口径：12张batch 32→恰好1批12次。
    verify_calibration_epoch(_epoch_records(1, 12), 12, 32)


def test_sampling_audit_fields() -> None:
    """采样审计：总数/唯一图/重复数/标签分布/患者分布与显式口径声明。"""
    train = pd.DataFrame({
        "sha256": ["a", "b", "c", "d"],
        "label": [0, 0, 1, 1],
        "patient_id": ["p0", "p1", "p2", "p3"],
    })
    records = [
        {"batch_index": 0, "sha256": ["a", "a", "c"]},
        {"batch_index": 1, "sha256": ["c", "d", "a"]},
    ]
    audit = calibration_sampling_audit(records, train)
    assert audit["total_draws"] == 6
    assert audit["unique_images"] == 3
    assert audit["duplicate_draws"] == 3
    assert audit["max_draws_per_image"] == 3
    assert audit["images_not_drawn"] == 1
    assert audit["label_distribution_draws"] == {"0": 3, "1": 3}
    assert audit["unique_patients"] == 3
    assert audit["patient_draws_min"] == 1 and audit["patient_draws_max"] == 3
    assert "有放回" in audit["sampling_note"]


def _spatial(naib: float, pga: float, small_pga: float, small_images: int = 20) -> dict:
    """Build one synthetic spatial summary for gate tests."""
    return {
        "mean_normalized_aib": naib,
        "pga": pga,
        "lesion_size_strata": {
            "small": {"images": small_images, "pga": small_pga},
            "medium": {"images": 20, "pga": 0.9},
            "large": {"images": 20, "pga": 0.9},
        },
    }


def _threshold(fn: int, specificity: float) -> dict:
    """Build one synthetic patient threshold metric block."""
    tp = 60 - fn
    tn = int(round(specificity * 100))
    return {
        "threshold": 0.3,
        "sensitivity": tp / 60.0,
        "specificity": specificity,
        "confusion_matrix": [tn, 100 - tn, fn, tp],
    }


def _arm_metrics(
    patient_auc: float, image_auc: float, naib: float, pga: float,
    small_pga: float, fn: int, spec: float, small_images: int = 20,
) -> dict:
    """Assemble one synthetic arm metric set for evaluate_gates."""
    return {
        "val_patient_auc": patient_auc,
        "val_image_auc": image_auc,
        "spatial": _spatial(naib, pga, small_pga, small_images),
        "patient_threshold_metrics": _threshold(fn, spec),
    }


def _passing_metrics() -> dict:
    """Return synthetic arm metrics that pass all eight gates."""
    return {
        "A": _arm_metrics(0.910, 0.880, 0.40, 0.80, 0.80, 6, 0.64),
        "B": _arm_metrics(0.905, 0.878, 0.41, 0.80, 0.80, 6, 0.64),
        "C": _arm_metrics(0.916, 0.881, 0.46, 0.82, 0.78, 6, 0.63),
    }


def test_evaluate_gates_all_pass() -> None:
    """门槛判定：八条全过（含gate8硬性评估）→ proceed_to_MG3。"""
    verdict = evaluate_gates(_passing_metrics(), 0.9133)
    assert verdict["decision"] == DECISION_PROCEED
    assert all(
        gate["passed"] for gate in verdict["gates"].values() if not gate.get("degraded")
    )


def test_evaluate_gates_diagnostic_stop() -> None:
    """门槛5通过但门槛1失败 → stop_diagnostic_spatial_gain_only。"""
    metrics = _passing_metrics()
    metrics["A"]["val_patient_auc"] = 0.90  # gate1失败
    metrics["B"]["val_patient_auc"] = 0.895  # 保持gate6通过
    metrics["C"]["val_patient_auc"] = 0.916  # gate2/3仍过
    verdict = evaluate_gates(metrics, 0.9133)
    assert verdict["gates"]["gate5_spatial_gain_C_over_A"]["passed"]
    assert not verdict["gates"]["gate1_armA_patient_auc_floor"]["passed"]
    assert verdict["decision"] == DECISION_DIAGNOSTIC


def test_evaluate_gates_plain_stop_paths() -> None:
    """门槛5失败、或门槛6/7失败 → stop。"""
    metrics = _passing_metrics()
    metrics["C"]["spatial"]["mean_normalized_aib"] = 0.41  # 增益不足0.05
    metrics["C"]["spatial"]["pga"] = 0.81
    verdict = evaluate_gates(metrics, 0.9133)
    assert not verdict["gates"]["gate5_spatial_gain_C_over_A"]["passed"]
    assert verdict["decision"] == DECISION_STOP

    metrics = _passing_metrics()
    metrics["B"]["val_patient_auc"] = 0.89  # gate6失败
    verdict = evaluate_gates(metrics, 0.9133)
    assert verdict["decision"] == DECISION_STOP

    metrics = _passing_metrics()
    metrics["C"]["patient_threshold_metrics"] = _threshold(8, 0.64)  # 多漏诊2名
    verdict = evaluate_gates(metrics, 0.9133)
    assert not verdict["gates"]["gate7_clinical_safety_C_over_A"]["passed"]
    assert verdict["decision"] == DECISION_DIAGNOSTIC


def test_evaluate_gates_gate8_degraded_disclosure() -> None:
    """门槛8降级：small组<15张时只报告、附披露，其余全过仍放行。"""
    metrics = _passing_metrics()
    for arm in "ABC":
        metrics[arm]["spatial"]["lesion_size_strata"]["small"]["images"] = 10
    metrics["C"]["spatial"]["lesion_size_strata"]["small"]["pga"] = 0.50  # 大降但降级
    verdict = evaluate_gates(metrics, 0.9133)
    gate8 = verdict["gates"]["gate8_small_lesion_pga"]
    assert gate8["degraded"] and gate8["passed"] is None
    assert "降级为只报告" in gate8["disclosure"]
    assert verdict["decision"] == DECISION_PROCEED


def test_incomplete_arm_products_refused() -> None:
    """残缺产物拒绝汇总：缺checkpoint/预测CSV即快速失败。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "mg2_arma_efficientnet_b0_seed42"
        run_dir.mkdir()
        (run_dir / "config.json").write_text(json.dumps({
            "arm": "A", "test_evaluated": False,
            "internal_test_evaluated": False, "external_evaluated": False,
        }), encoding="utf-8")
        try:
            load_arm_products(run_dir, "A")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("未拒绝残缺产物")
        for name in ("training_history.csv", "val_image_predictions.csv",
                     "val_patient_predictions.csv", "mg2_arma_best_student.pth"):
            (run_dir / name).write_text("x", encoding="utf-8")
        config = load_arm_products(run_dir, "A")
        assert config["arm"] == "A"
        # 锁定标记非false同样拒绝。
        (run_dir / "config.json").write_text(json.dumps({
            "arm": "A", "test_evaluated": True,
            "internal_test_evaluated": False, "external_evaluated": False,
        }), encoding="utf-8")
        try:
            load_arm_products(run_dir, "A")
        except ValueError:
            pass
        else:
            raise AssertionError("未拒绝锁定标记违规")


def test_m0f_reference_formal_fails_debug_falls_back() -> None:
    """M0-F参照：正式模式读取失败快速失败；debug模式回退常量并显式警告。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        valid = tmp / "m0f.json"
        valid.write_text(json.dumps({"best_val_patient_auc": 0.9133}), encoding="utf-8")
        value, source, warning = load_m0f_reference(valid, debug=False)
        assert value == 0.9133 and source == "config_json" and warning is None

        missing = tmp / "missing.json"
        try:
            load_m0f_reference(missing, debug=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("正式模式未拒绝M0-F读取失败")
        broken = tmp / "broken.json"
        broken.write_text(json.dumps({"other_field": 1}), encoding="utf-8")
        try:
            load_m0f_reference(broken, debug=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("正式模式未拒绝M0-F字段缺失")

        value, source, warning = load_m0f_reference(missing, debug=True)
        assert value == M0F_FALLBACK_PATIENT_AUC
        assert source == "fallback_constant" and warning


def _frozen_payload(**overrides) -> dict:
    """Build one field-complete synthetic formal calibration payload."""
    median_nonspatial, median_attention = 0.8, 1.6
    payload = {
        "median_nonspatial": median_nonspatial,
        "median_attention": median_attention,
        "beta": 0.25 * median_nonspatial / median_attention,
        "batches": 74,
        "sampling_audit": {"total_draws": 2350},
        "alpha": 0.25,
        "tau": 4.0,
        "test_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
    }
    payload.update(overrides)
    return payload


def test_frozen_calibration_binding() -> None:
    """冻结校准强绑定：SHA逐字符核验、beta重算、74批/2350次/常量/锁定标记。"""
    verify_frozen_calibration(_frozen_payload(), FROZEN_BETA_CALIBRATION_SHA256)

    def expect_failure(payload, sha, message):
        try:
            verify_frozen_calibration(payload, sha)
        except ValueError:
            return
        raise AssertionError(f"未拒绝: {message}")

    # sidecar重生成/内容相同但字节不同的场景：SHA不符即拒。
    expect_failure(
        _frozen_payload(), "0" * 64, "SHA不符（含重生成场景）"
    )
    expect_failure(
        _frozen_payload(beta=0.5), FROZEN_BETA_CALIBRATION_SHA256, "beta重算不符"
    )
    expect_failure(
        _frozen_payload(batches=73), FROZEN_BETA_CALIBRATION_SHA256, "batches=73"
    )
    expect_failure(
        _frozen_payload(batches=75), FROZEN_BETA_CALIBRATION_SHA256, "batches=75"
    )
    expect_failure(
        _frozen_payload(sampling_audit={"total_draws": 2349}),
        FROZEN_BETA_CALIBRATION_SHA256, "total_draws不符",
    )
    expect_failure(
        _frozen_payload(alpha=0.3), FROZEN_BETA_CALIBRATION_SHA256, "alpha不符"
    )
    expect_failure(
        _frozen_payload(tau=3.0), FROZEN_BETA_CALIBRATION_SHA256, "tau不符"
    )
    expect_failure(
        _frozen_payload(external_evaluated=True),
        FROZEN_BETA_CALIBRATION_SHA256, "锁定标记违规",
    )


def test_formal_mode_requires_frozen_sha_debug_skips() -> None:
    """端到端：正式模式拒绝任何非冻结SHA的校准JSON；debug模式显式放宽。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cache = tmp / "cache.pt"
        cache.write_bytes(b"synthetic-cache")
        calibration = _write_calibration(tmp, cache)
        try:
            load_beta_calibration(calibration, MANIFEST_SHA, cache, 42, 32, False)
        except ValueError:
            pass
        else:
            raise AssertionError("正式模式未拒绝非冻结SHA的校准JSON")
        debug_calibration = _write_calibration(tmp, cache, debug=True)
        payload = load_beta_calibration(
            debug_calibration, MANIFEST_SHA, cache, 42, 32, True
        )
        assert payload["_frozen_sha_verified"] is False


def _arm_config(arm: str, tmp: Path, **overrides) -> dict:
    """Build one synthetic arm config with a real checkpoint file."""
    run_dir = tmp / f"run_{arm}"
    run_dir.mkdir(exist_ok=True)
    checkpoint = run_dir / f"mg2_arm{arm.lower()}_best_student.pth"
    checkpoint.write_bytes(f"ckpt-{arm}".encode())
    config = {
        "arm": arm,
        "seed": 42,
        "debug": False,
        "_run_dir": str(run_dir),
        "checkpoint_sha256": file_sha256(checkpoint),
        "training": {
            "batch_size": 32, "stage_a_epochs": 10, "stage_b_epochs": 20,
            "stage_a_lr": 1e-3, "stage_b_lr": 1e-4, "weight_decay": 1e-4,
            "patience": 8, "sampler": "patient_and_class_balanced_with_replacement",
            "beta_calibration_json": None,
            "beta_calibration_json_sha256": None,
        },
        "input_protocol": {"student": "full_rgb_224", "teacher": "luma_roi"},
        "architecture": {"backbone": "efficientnet_b0"},
        "teacher_checkpoint_sha256": FROZEN_TEACHER_SHA256,
        "manifest_sha256": MANIFEST_SHA,
        "v3_audit_sha256": "v" * 64,
        "teacher_cache": "/cache.pt" if arm in "BC" else None,
        "teacher_cache_sha256": "c" * 64 if arm in "BC" else None,
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


def _consistent_configs(tmp: Path) -> tuple[dict, dict]:
    """Build three mutually consistent synthetic arm configs + calibration."""
    configs = {arm: _arm_config(arm, tmp) for arm in "ABC"}
    calibration = {"_self_path": "/cal.json", "_self_sha256": "s" * 64}
    configs["C"]["training"]["beta_calibration_json"] = "/cal.json"
    configs["C"]["training"]["beta_calibration_json_sha256"] = "s" * 64
    return configs, calibration


def test_cross_arm_consistency() -> None:
    """跨组一致性：一致通过；checkpoint/seed/超参/增强/架构/SHA/缓存/beta绑定不符均拒。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        configs, calibration = _consistent_configs(tmp)
        verify_cross_arm_consistency(configs, calibration)

        def expect_failure(mutate, message):
            broken, cal = _consistent_configs(tmp)
            mutate(broken, cal)
            try:
                verify_cross_arm_consistency(broken, cal)
            except ValueError:
                return
            raise AssertionError(f"未拒绝: {message}")

        expect_failure(
            lambda c, _: c["A"].update(checkpoint_sha256="0" * 64), "checkpoint SHA"
        )
        expect_failure(lambda c, _: c["B"].update(seed=202), "seed非42")
        expect_failure(
            lambda c, _: c["C"]["training"].update(batch_size=16), "batch size不一致"
        )
        expect_failure(
            lambda c, _: c["C"]["training"].update(stage_b_epochs=19), "epoch上限不一致"
        )
        expect_failure(
            lambda c, _: c["B"]["training"].update(stage_b_lr=2e-4), "学习率不一致"
        )
        expect_failure(
            lambda c, _: c["C"]["training"].update(weight_decay=1e-3), "weight decay不一致"
        )
        expect_failure(
            lambda c, _: c["B"]["training"].update(patience=6), "patience不一致"
        )
        expect_failure(
            lambda c, _: c["C"]["training"].update(sampler="other"), "采样器不一致"
        )
        expect_failure(
            lambda c, _: c["B"].update(input_protocol={"student": "changed"}),
            "增强配置不一致",
        )
        expect_failure(
            lambda c, _: c["C"].update(architecture={"backbone": "resnet50"}),
            "学生架构不一致",
        )
        expect_failure(
            lambda c, _: c["B"].update(manifest_sha256="x" * 64), "manifest SHA不一致"
        )
        expect_failure(
            lambda c, _: c["C"].update(v3_audit_sha256="x" * 64), "v3 audit SHA不一致"
        )
        expect_failure(
            lambda c, _: c["C"].update(teacher_cache_sha256="x" * 64),
            "B/C教师缓存SHA不一致",
        )
        expect_failure(
            lambda c, _: c["B"].update(teacher_cache="/other.pt"),
            "B/C教师缓存路径不一致",
        )
        expect_failure(
            lambda c, _: c["C"]["training"].update(beta_calibration_json="/other.json"),
            "C组beta JSON路径不一致",
        )
        expect_failure(
            lambda _, cal: cal.update(_self_sha256="0" * 64),
            "C组beta JSON SHA不一致",
        )


def test_launcher_dry_run() -> None:
    """启动器dry-run：打印A→B→C计划与自动汇总步骤，不执行训练。"""
    env = dict(os.environ, MG2_MATRIX_DRY_RUN="1")
    result = subprocess.run(
        ["bash", str(SCRIPT_DIR / "run_mage_mg2_matrix.sh")],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "[DRY-RUN]" in result.stdout
    assert "--arm A" in result.stdout and "--arm C" in result.stdout
    assert "--beta-calibration-json" in result.stdout
    assert "summarize_mage_mg2.py" in result.stdout


def run_all() -> None:
    """Execute every ``test_*`` function in definition order and report."""
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"MG2防线单元测试全部通过: {len(tests)}项")


if __name__ == "__main__":
    run_all()
