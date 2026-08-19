#!/usr/bin/env python3
"""clong_sae_discovery 的单元测试（C-long SAE文献重构新路线）。

运行方式：

    /home/mcy/miniconda3/envs/gastric-cbm/bin/python \
        /home/mcy/gastric-cbm/程序/SAE/正式代码/test_clong_sae.py

覆盖：正式矩阵参数校验、S0血缘核验、冻结阈值使用、非死亡重复率口径、
剪枝无fallback选择逻辑、attention-pooling身份等式和复算自测输入检查。
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import clong_sae_discovery as csd  # noqa: E402


def make_args(**overrides) -> types.SimpleNamespace:
    """构造validate_formal_args所需的最小参数命名空间。"""
    base = dict(
        seed=42, hidden_dim=1280, activation_mode="relu_l1", top_k=None,
        lambda_l1=5e-4, margin_loss_weight=0.1, debug=False,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def expect_error(args: types.SimpleNamespace, fragment: str) -> None:
    """断言validate_formal_args抛出包含指定片段的错误。"""
    try:
        csd.validate_formal_args(args)
    except ValueError as exc:
        assert fragment in str(exc), f"错误信息缺少'{fragment}': {exc}"
        return
    raise AssertionError(f"应当拒绝但未拒绝: {args}")


def test_formal_matrix_validation() -> None:
    """正式运行必须落在冻结的五宽度x三lambda网格和固定备选内。"""
    for width in csd.FORMAL_WIDTHS:
        for lambda_l1 in csd.FORMAL_LAMBDAS:
            csd.validate_formal_args(
                make_args(hidden_dim=width, lambda_l1=lambda_l1)
            )
    # 无margin诊断只允许 w10240/lambda=5e-4/seed42。
    csd.validate_formal_args(make_args(
        hidden_dim=10240, lambda_l1=5e-4, margin_loss_weight=0.0,
    ))
    expect_error(
        make_args(hidden_dim=512, margin_loss_weight=0.0), "无margin诊断",
    )
    expect_error(
        make_args(hidden_dim=10240, lambda_l1=2e-4, margin_loss_weight=0.0),
        "无margin诊断",
    )
    expect_error(make_args(hidden_dim=768), "五档网格")
    expect_error(make_args(lambda_l1=3e-4), "冻结网格")
    expect_error(make_args(margin_loss_weight=0.2), "gamma只能为0.1")
    # Top-K备选固定 hidden=10240, K=1024, lambda=0, gamma=0.1。
    csd.validate_formal_args(make_args(
        activation_mode="topk", hidden_dim=10240, top_k=1024, lambda_l1=0.0,
    ))
    expect_error(
        make_args(activation_mode="topk", hidden_dim=10240, top_k=512,
                  lambda_l1=0.0),
        "hidden=10240, K=1024",
    )
    expect_error(
        make_args(activation_mode="topk", hidden_dim=10240, top_k=1024,
                  lambda_l1=5e-4),
        "lambda_l1=0",
    )
    # debug模式不受正式网格限制。
    csd.validate_formal_args(make_args(debug=True, hidden_dim=64, lambda_l1=1.0))
    print("PASS test_formal_matrix_validation")


def test_frozen_thresholds_match_clong_config() -> None:
    """脚本内冻结阈值必须与C-long正式config逐位一致。"""
    config = json.loads(csd.CLONG_CONFIG.read_text(encoding="utf-8"))
    image = float(config["metrics"]["image_threshold_metrics"]["threshold"])
    patient = float(config["metrics"]["patient_threshold_metrics"]["threshold"])
    assert image == csd.FROZEN_IMAGE_THRESHOLD
    assert patient == csd.FROZEN_PATIENT_THRESHOLD
    assert config["checkpoint_sha256"] == csd.CLONG_CHECKPOINT_SHA256
    assert config["manifest_sha256"] == csd.MANIFEST_SHA256
    print("PASS test_frozen_thresholds_match_clong_config")


def test_s0_lineage_verification() -> None:
    """真实资产必须通过S0血缘核验。"""
    lineage = csd.verify_s0_lineage()
    expected_names = {
        "student_checkpoint", "manifest", "v3_audit",
        "teacher_checkpoint", "teacher_cache", "beta_calibration_json",
    }
    assert expected_names <= set(lineage)
    print("PASS test_s0_lineage_verification")


def test_annotate_uses_frozen_threshold_only() -> None:
    """标注只使用S0冻结图像阈值，不发生任何扫描。"""
    frame = pd.DataFrame({
        "patient_id": ["p1", "p2", "p3", "p4"],
        "label": [1, 1, 0, 0],
        "cancer_probability": [
            csd.FROZEN_IMAGE_THRESHOLD + 1e-6,
            csd.FROZEN_IMAGE_THRESHOLD - 1e-6,
            csd.FROZEN_IMAGE_THRESHOLD - 1e-6,
            csd.FROZEN_IMAGE_THRESHOLD + 1e-6,
        ],
    })
    metadata = {"val": frame}
    csd.annotate_frozen_thresholds(metadata)
    assert frame["pred_label"].tolist() == [1, 0, 0, 1]
    assert frame["confusion_type"].tolist() == ["TP", "FN", "TN", "FP"]
    print("PASS test_annotate_uses_frozen_threshold_only")


def test_duplicate_rate_excludes_dead_features() -> None:
    """重复率只在非死亡Feature上计算，且分母为非死亡数。"""
    # 两个活着的Feature方向不同；一个死亡Feature与活着的完全同向。
    weights = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float32)
    nondead = np.array([True, True, False])
    result = csd.duplicate_decoder_rate_nondead(weights, nondead)
    assert result["nondead_feature_count"] == 2
    assert result["duplicate_feature_count"] == 0
    assert result["duplicate_rate"] == 0.0
    # 两个活着的Feature互为重复时，两者都计入。
    nondead2 = np.array([True, False, True])
    weights2 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.9999999, 1e-4, 0.0],
    ], dtype=np.float32)
    result2 = csd.duplicate_decoder_rate_nondead(weights2, nondead2)
    assert result2["nondead_feature_count"] == 2
    assert result2["duplicate_feature_count"] == 2
    assert result2["duplicate_rate"] == 1.0
    # 全部死亡时返回0而不是报错。
    result3 = csd.duplicate_decoder_rate_nondead(weights, np.zeros(3, dtype=bool))
    assert result3["duplicate_rate"] == 0.0
    print("PASS test_duplicate_rate_excludes_dead_features")


def test_pruning_selection_never_violates_tolerance() -> None:
    """选择逻辑：容差内最大阈值；无合格方案返回None（调用方保留全部）。"""
    curve = [
        {"active_patient_threshold": 4, "kept_feature_count": 100,
         "recovered_ce_drop": 0.004},
        {"active_patient_threshold": 9, "kept_feature_count": 60,
         "recovered_ce_drop": 0.009},
        {"active_patient_threshold": 19, "kept_feature_count": 30,
         "recovered_ce_drop": 0.030},
    ]
    assert csd.select_pruning_threshold(curve, 0.01) == 9
    assert csd.select_pruning_threshold(
        [dict(row, recovered_ce_drop=0.5) for row in curve], 0.01,
    ) is None
    # 非单调曲线：中间失败但后面合格的，仍取容差内最大阈值。
    nonmonotonic = [
        {"active_patient_threshold": 4, "kept_feature_count": 100,
         "recovered_ce_drop": 0.004},
        {"active_patient_threshold": 9, "kept_feature_count": 60,
         "recovered_ce_drop": 0.030},
        {"active_patient_threshold": 19, "kept_feature_count": 30,
         "recovered_ce_drop": 0.008},
    ]
    assert csd.select_pruning_threshold(nonmonotonic, 0.01) == 19
    print("PASS test_pruning_selection_never_violates_tolerance")


def test_attention_pooling_identity() -> None:
    """attention-pooled向量经分类头必须等于模型logits，attention和为1。"""
    model = csd.AttentionPoolingTeacher(pretrained=False)
    model.eval()
    images = torch.randn(2, 3, csd.IMAGE_SIZE, csd.IMAGE_SIZE)
    with torch.no_grad():
        logits, attention = model(images)
        fmap = model.features(images)
        pooled = (fmap * attention).sum(dim=(2, 3))
    assert pooled.shape == (2, csd.INPUT_DIM)
    assert torch.allclose(model.classifier(pooled), logits, atol=1e-5)
    assert torch.allclose(attention.sum(dim=(2, 3)), torch.ones(2, 1), atol=1e-5)
    print("PASS test_attention_pooling_identity")


def test_debug_subset_keeps_patient_integrity() -> None:
    """debug子集按完整患者抽样，不拆散患者、不混split。"""
    rows = []
    for split in ("train", "val"):
        for label in (0, 1):
            for patient_index in range(5):
                patient = f"{split}_{label}_{patient_index}"
                for image_index in range(3):
                    rows.append({
                        "patient_id": patient, "label": label, "split": split,
                    })
    frame = pd.DataFrame(rows)
    subset = csd.select_debug_patients(frame, 2, 42)
    counts = subset.groupby("patient_id").size()
    assert (counts == 3).all(), "患者图像被拆散"
    assert subset.groupby("patient_id")["split"].nunique().max() == 1
    assert subset.groupby("patient_id")["label"].nunique().max() == 1
    print("PASS test_debug_subset_keeps_patient_integrity")


def test_pareto_and_selection_chain() -> None:
    """Pareto前沿保留互不支配候选；决胜链从前沿中选出唯一配置。"""
    import summarize_clong_sae_matrix as smz

    # A全面优于B（B被支配）；C与A互有优劣（A的AUC更好，C更稀疏）。
    table = pd.DataFrame([
        {"name": "A", "patient_auc_drop": 0.005, "patient_agreement": 0.96,
         "val_mean_cosine": 0.93, "recovered_ce": 0.96, "val_mean_l0": 100.0,
         "kept_feature_count": 200, "hidden_dim": 2560},
        {"name": "B", "patient_auc_drop": 0.008, "patient_agreement": 0.95,
         "val_mean_cosine": 0.91, "recovered_ce": 0.95, "val_mean_l0": 150.0,
         "kept_feature_count": 300, "hidden_dim": 5120},
        {"name": "C", "patient_auc_drop": 0.009, "patient_agreement": 0.96,
         "val_mean_cosine": 0.93, "recovered_ce": 0.96, "val_mean_l0": 60.0,
         "kept_feature_count": 150, "hidden_dim": 512},
    ])
    assert smz.dominates(table.iloc[0], table.iloc[1])
    assert not smz.dominates(table.iloc[0], table.iloc[2])
    assert not smz.dominates(table.iloc[2], table.iloc[0])
    frontier = smz.pareto_frontier(table)
    assert frontier["name"].tolist() == ["A", "C"]
    # 决胜链第一级是患者AUC下降更小，因此前沿中A胜出。
    assert smz.select_unique_config(frontier)["name"] == "A"
    # AUC下降相同时进入第二级患者一致率。
    tied = frontier.assign(patient_auc_drop=0.005)
    tied.loc[tied["name"] == "C", "patient_agreement"] = 0.97
    assert smz.select_unique_config(tied)["name"] == "C"
    print("PASS test_pareto_and_selection_chain")


def main() -> None:
    test_formal_matrix_validation()
    test_frozen_thresholds_match_clong_config()
    test_s0_lineage_verification()
    test_annotate_uses_frozen_threshold_only()
    test_duplicate_rate_excludes_dead_features()
    test_pruning_selection_never_violates_tolerance()
    test_attention_pooling_identity()
    test_debug_subset_keeps_patient_integrity()
    test_pareto_and_selection_chain()
    print("全部 9 项测试通过")


if __name__ == "__main__":
    main()
