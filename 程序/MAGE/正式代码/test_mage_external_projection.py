#!/usr/bin/env python3
"""Unit tests for the external descriptive projection script defenses.

Covers the fixes after the 2026-08-19 code review:

1. ``stratified_auc`` patient level must not raise KeyError (aggregated
   probability column) and must mark undersized strata NA;
2. ``select_attention_subset`` must return exactly 60 unique rows, with the
   random group drawn outside the 48 extreme rows;
3. ``verify_checkpoint_sha`` hard-binds all three model checkpoints to the
   preregistered SHA256 values;
4. ``threshold_metrics`` applies the 1e-7 float tolerance at the frozen
   threshold (thresholds equal an observed val probability by construction);
5. ``patient_mean`` rejects cross-label patients.
6. archived M0-F probabilities must pass all pinned SHA/lineage checks and
   exactly reproduce the published image/patient AUC values.

Runnable both as ``pytest test_mage_external_projection.py`` and directly as
``python test_mage_external_projection.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_mage_external_projection import (  # noqa: E402
    MODEL_SPECS,
    build_external_frame,
    load_archived_m0f_probabilities,
    patient_mean,
    select_attention_subset,
    stratified_auc,
    threshold_metrics,
    verify_checkpoint_sha,
)


def synthetic_frame() -> pd.DataFrame:
    """构建16患者32图的双类别合成清单（含一个仅2图的碎层）。

    返回:
        pd.DataFrame: patient_id/label/original_width/progress_bar_detected/
            crop_status及M0-F、A-long两列概率；癌图概率整体高于非癌。
    """
    rows = []
    for patient_index in range(16):
        label = patient_index % 2
        for image_index in range(2):
            rows.append({
                "patient_id": f"P{patient_index:02d}",
                "label": label,
                "original_width": 1000,
                "progress_bar_detected": False,
                "crop_status": "cropped",
                "M0-F": 0.7 if label else 0.3,
                "A-long": 0.8 if label else 0.2,
            })
    # 碎层：两张宽700的图，触发分层NA规则。
    for extra in range(2):
        rows.append({
            "patient_id": f"P{extra:02d}",
            "label": extra % 2,
            "original_width": 700,
            "progress_bar_detected": False,
            "crop_status": "cropped",
            "M0-F": 0.6,
            "A-long": 0.6,
        })
    return pd.DataFrame(rows)


def test_stratified_auc_patient_level_no_keyerror() -> None:
    """患者级分层应正常出AUC，碎层记NA且不抛KeyError。"""
    records = stratified_auc(synthetic_frame(), ["M0-F", "A-long"])
    table = pd.DataFrame(records)
    adequate = table.loc[
        table.value.eq("801-1100") & table.level.eq("patient")
    ]
    assert len(adequate) == 2
    assert adequate.auc.notna().all()
    assert adequate.auc.eq(1.0).all()
    tiny = table.loc[table.value.eq("<=800")]
    assert tiny.auc.isna().all()
    assert tiny.na_reason.notna().all()


def test_attention_subset_exactly_60_unique() -> None:
    """极值48张+随机12张必须互不重叠，合计恰好60个唯一行号。"""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "patient_id": [f"Q{i:03d}" for i in range(300)],
        "label": [1] * 100 + [0] * 200,
        "C-long": rng.uniform(size=300),
    })
    subset = select_attention_subset(frame, per_group=12)
    fixed = {
        i for key, indices in subset.items() if not key.startswith("random")
        for i in indices
    }
    random_group = set(subset["random_seed20260819"])
    assert len(fixed) == 48
    assert len(random_group) == 12
    assert not fixed & random_group
    assert len(fixed | random_group) == 60


def test_verify_checkpoint_sha_binding() -> None:
    """三模型协议SHA硬绑定：一致通过，被篡改即拒绝。"""
    for name, spec in MODEL_SPECS.items():
        verify_checkpoint_sha(name, spec, spec["checkpoint_sha256"])
        try:
            verify_checkpoint_sha(name, spec, "0" * 64)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{name}篡改SHA未被拒绝")


def test_threshold_metrics_float_tolerance() -> None:
    """贴阈值1e-7内的概率判阳性侧，超过容差的判阴性侧。"""
    within = threshold_metrics([0, 1], [0.2, 0.5 - 5e-8], 0.5)
    assert within["confusion_matrix"] == [1, 0, 0, 1]
    beyond = threshold_metrics([0, 1], [0.2, 0.5 - 1e-6], 0.5)
    assert beyond["confusion_matrix"] == [1, 0, 1, 0]
    exact = threshold_metrics([0, 1], [0.2, 0.5], 0.5)
    assert exact["confusion_matrix"] == [1, 0, 0, 1]


def test_patient_mean_cross_label_rejected() -> None:
    """跨标签患者必须抛错而非静默聚合。"""
    frame = pd.DataFrame({
        "patient_id": ["X", "X"],
        "label": [0, 1],
        "probability": [0.1, 0.9],
    })
    try:
        patient_mean(frame, "probability")
    except ValueError:
        pass
    else:
        raise AssertionError("跨标签患者未被拒绝")


def test_archived_m0f_probabilities_reproduce_published_auc() -> None:
    """历史M0-F概率须通过SHA/血缘校验并精确复算既有AUC。"""
    frame = build_external_frame()
    probabilities = load_archived_m0f_probabilities(frame)
    image_metrics = threshold_metrics(frame.label, probabilities, 0.20749907195568085)
    patients = frame[["patient_id", "label"]].copy()
    patients["probability"] = probabilities
    patient_frame = patient_mean(patients, "probability")
    patient_metrics = threshold_metrics(
        patient_frame.label,
        patient_frame.probability,
        0.20749907195568085,
    )
    assert len(probabilities) == 1941
    assert abs(image_metrics["auc"] - 0.7213138535960055) < 1e-12
    assert abs(patient_metrics["auc"] - 0.7408803568496525) < 1e-12


def main() -> None:
    """逐个运行全部测试并打印结果；无参数，返回None。"""
    tests = [
        test_stratified_auc_patient_level_no_keyerror,
        test_attention_subset_exactly_60_unique,
        test_verify_checkpoint_sha_binding,
        test_threshold_metrics_float_tolerance,
        test_patient_mean_cross_label_rejected,
        test_archived_m0f_probabilities_reproduce_published_auc,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} 测试全部通过")


if __name__ == "__main__":
    main()
