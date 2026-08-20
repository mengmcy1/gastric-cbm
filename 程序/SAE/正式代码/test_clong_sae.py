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
    """构造validate_formal_args所需的最小参数命名空间（默认合法正式配置）。"""
    base = dict(
        seed=42, hidden_dim=1280, activation_mode="relu_l1", top_k=None,
        lambda_l1=5e-4, margin_loss_weight=0.1, debug=False,
        experiment="clong_w1280_l5e-4_seed42",
        learning_rate=1e-4, epochs=1000, patience=50, warmup_fraction=0.05,
        image_batch_size=32, sae_batch_size=32,
        pruning_ce_tolerance=0.01, pruning_min_active_patients=5,
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
    labels = (("2e-4", 2e-4), ("5e-4", 5e-4), ("1e-3", 1e-3))
    for width in csd.FORMAL_WIDTHS:
        for label, lambda_l1 in labels:
            csd.validate_formal_args(
                make_args(
                    hidden_dim=width, lambda_l1=lambda_l1,
                    experiment=f"clong_w{width}_l{label}_seed42",
                )
            )
    # 无margin诊断只允许 w10240/lambda=5e-4/seed42。
    csd.validate_formal_args(make_args(
        hidden_dim=10240, lambda_l1=5e-4, margin_loss_weight=0.0,
        experiment="clong_w10240_l5e-4_gamma0_seed42",
    ))
    expect_error(
        make_args(hidden_dim=512, margin_loss_weight=0.0), "绑定hidden_dim",
    )
    expect_error(
        make_args(hidden_dim=10240, lambda_l1=2e-4, margin_loss_weight=0.0),
        "绑定hidden_dim",
    )
    expect_error(make_args(hidden_dim=768), "绑定hidden_dim")
    expect_error(make_args(lambda_l1=3e-4), "绑定lambda_l1")
    expect_error(make_args(margin_loss_weight=0.2), "绑定margin_loss_weight")
    # Top-K备选固定 hidden=10240, K=1024, lambda=0, gamma=0.1。
    csd.validate_formal_args(make_args(
        activation_mode="topk", hidden_dim=10240, top_k=1024, lambda_l1=0.0,
        experiment="clong_w10240_topk1024_seed42",
    ))
    expect_error(
        make_args(activation_mode="topk", hidden_dim=10240, top_k=512,
                  lambda_l1=0.0),
        "绑定activation_mode",
    )
    expect_error(
        make_args(activation_mode="topk", hidden_dim=10240, top_k=1024,
                  lambda_l1=5e-4),
        "lambda_l1=0",
    )
    # debug模式不受正式网格限制，但禁止使用正式实验名。
    csd.validate_formal_args(make_args(
        debug=True, hidden_dim=64, lambda_l1=1.0, experiment="debug_x",
    ))
    expect_error(
        make_args(debug=True, experiment="clong_w1280_l5e-4_seed42"),
        "debug运行禁止",
    )
    print("PASS test_formal_matrix_validation")


def test_formal_budget_and_name_lock() -> None:
    """正式运行的训练预算和实验名被冻结，逐项篡改都会被拒绝。"""
    csd.validate_formal_args(make_args())
    expect_error(make_args(learning_rate=2e-4), "learning_rate必须等于冻结预算")
    expect_error(make_args(epochs=500), "epochs必须等于冻结预算")
    expect_error(make_args(patience=10), "patience必须等于冻结预算")
    expect_error(make_args(warmup_fraction=0.1), "warmup_fraction必须等于冻结预算")
    expect_error(make_args(image_batch_size=16), "image_batch_size必须等于冻结预算")
    expect_error(make_args(sae_batch_size=64), "sae_batch_size必须等于冻结预算")
    expect_error(
        make_args(pruning_ce_tolerance=0.02), "pruning_ce_tolerance必须等于冻结预算",
    )
    expect_error(make_args(experiment="clong_w999_l2e-4_seed42"), "17个名称")
    expect_error(make_args(experiment="my_experiment"), "17个名称")
    expect_error(
        make_args(
            experiment="clong_w512_l2e-4_seed42", hidden_dim=1280,
            lambda_l1=2e-4,
        ),
        "绑定hidden_dim=512",
    )
    # 17个名称覆盖15组L1、1组无margin诊断和1组Top-K，按seed变化。
    names42 = csd.formal_experiment_names(42)
    assert len(names42) == 17
    assert "clong_w10240_topk1024_seed42" in names42
    assert "clong_w10240_l5e-4_gamma0_seed42" in names42
    assert "clong_w512_l2e-4_seed202" in csd.formal_experiment_names(202)
    print("PASS test_formal_budget_and_name_lock")


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


def test_cache_reuse_verifies_file_sha() -> None:
    """复用缓存逐文件核验SHA/shape/行顺序；篡改任一文件必须被拒绝。"""
    import tempfile

    frame = pd.DataFrame({
        "patient_id": ["p1", "p2", "p3", "p4", "p5", "p6"],
        "label": [1, 0, 1, 0, 1, 0],
        "split": ["train"] * 4 + ["val"] * 2,
        "relative_path": [f"img_{i}.jpg" for i in range(6)],
        "sha256": [f"sha_{i}" for i in range(6)],
    })
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source"
        target = Path(tmp) / "target"
        source.mkdir()
        target.mkdir()
        for split, count in (("train", 4), ("val", 2)):
            np.save(source / f"{split}_pooled_features.npy",
                    np.random.rand(count, csd.INPUT_DIM).astype(np.float32))
            np.save(source / f"{split}_attention_maps.npy",
                    np.random.rand(count, 49).astype(np.float32))
            frame.loc[frame.split.eq(split)].reset_index(drop=True).to_csv(
                source / f"{split}_metadata.csv", index=False, encoding="utf-8-sig",
            )
        lineage = {
            "student_checkpoint_sha256": csd.CLONG_CHECKPOINT_SHA256,
            "manifest_sha256": csd.MANIFEST_SHA256,
            "feature_layer": csd.FEATURE_LAYER_DESCRIPTION,
            "counts": {"train": {"images": 4, "patients": 4},
                       "val": {"images": 2, "patients": 2}},
            "files": csd.cache_file_shas(source),
            "recompute_self_test": {"passed": True},
        }
        (source / "cache_config.json").write_text(
            json.dumps(lineage, ensure_ascii=False), encoding="utf-8"
        )
        features, attentions, metadata = csd.reuse_feature_cache(source, target, frame)
        assert features["train"].shape == (4, csd.INPUT_DIM)
        assert attentions["val"].shape == (2, 49)
        assert len(metadata["train"]) == 4
        # 篡改一个缓存文件后必须拒绝复用。
        np.save(source / "val_pooled_features.npy",
                np.random.rand(2, csd.INPUT_DIM).astype(np.float32))
        target2 = Path(tmp) / "target2"
        target2.mkdir()
        try:
            csd.reuse_feature_cache(source, target2, frame)
        except ValueError as exc:
            assert "SHA" in str(exc)
        else:
            raise AssertionError("缓存文件被篡改但未被拒绝")
        assert not any(target2.iterdir()), "核验失败时不应留下部分拷贝"
        # 行顺序错乱同样必须拒绝。
        lineage["files"] = csd.cache_file_shas(source)
        (source / "cache_config.json").write_text(
            json.dumps(lineage, ensure_ascii=False), encoding="utf-8"
        )
        shuffled = frame.iloc[::-1].reset_index(drop=True)
        target3 = Path(tmp) / "target3"
        target3.mkdir()
        try:
            csd.reuse_feature_cache(source, target3, shuffled)
        except ValueError as exc:
            assert "顺序" in str(exc) or "sha256" in str(exc)
        else:
            raise AssertionError("清单行顺序错乱但未被拒绝")
    print("PASS test_cache_reuse_verifies_file_sha")


def test_extended_fidelity_metrics() -> None:
    """扩展指标：margin保真、双阈值指标、患者偏移和密度直方图齐全。"""
    rng = np.random.default_rng(42)
    features = rng.normal(size=(40, csd.INPUT_DIM)).astype(np.float32)
    weight = rng.normal(size=(2, csd.INPUT_DIM)).astype(np.float32)
    bias = rng.normal(size=2).astype(np.float32)
    metadata = pd.DataFrame({
        "patient_id": [f"p{i // 2}" for i in range(40)],
        "label": [i % 2 for i in range(40)],
    })
    activations = (rng.random((40, 16)) > 0.5).astype(np.float32) * rng.random((40, 16))
    # 完美重构时margin相关为1、MSE为0，阈值指标原始=重构。
    perfect = csd.extended_fidelity_metrics(
        features, features.copy(), activations, metadata, weight, bias,
    )
    assert perfect["margin_mse"] == 0.0
    assert abs(perfect["margin_pearson"] - 1.0) < 1e-6
    assert (perfect["image_threshold_original"]
            == perfect["image_threshold_reconstructed"])
    assert (perfect["patient_threshold_original"]
            == perfect["patient_threshold_reconstructed"])
    assert perfect["patient_probability_max_deviation"] == 0.0
    assert isinstance(perfect["patient_probability_max_deviation_patient"], str)
    histogram = perfect["feature_density_histogram"]
    assert sum(histogram["counts"]) == 16
    assert len(histogram["bin_edges"]) == len(histogram["counts"]) + 1
    # 加噪重构时margin指标退化但字段齐全。
    noisy = csd.extended_fidelity_metrics(
        features, features + 0.1 * rng.normal(size=features.shape).astype(np.float32),
        activations, metadata, weight, bias,
    )
    assert noisy["margin_mse"] > 0
    assert noisy["patient_probability_max_deviation"] > 0
    print("PASS test_extended_fidelity_metrics")


def test_confusion_at_threshold_edge() -> None:
    """阈值指标在极端输入下返回None而不是NaN。"""
    labels = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.2, 0.1])
    result = csd.confusion_at_threshold(labels, probabilities, 0.5)
    assert result["sensitivity"] == 1.0 and result["specificity"] == 1.0
    assert result["accuracy"] == 1.0 and result["f1"] == 1.0
    assert (result["tp"], result["tn"], result["fp"], result["fn"]) == (2, 2, 0, 0)
    empty_positive = csd.confusion_at_threshold(
        np.array([0, 0]), np.array([0.1, 0.2]), 0.5,
    )
    assert empty_positive["sensitivity"] is None
    assert empty_positive["f1"] is None
    zero_hit = csd.confusion_at_threshold(
        np.array([1, 0]), np.array([0.1, 0.9]), 0.5,
    )
    assert zero_hit["f1"] == 0.0
    print("PASS test_confusion_at_threshold_edge")


def test_summary_json_sanitizer() -> None:
    """汇总JSON不允许非标准NaN，json_scalar把NaN转为None。"""
    import summarize_clong_sae_matrix as smz

    assert smz.json_scalar(None) is None
    assert smz.json_scalar(float("nan")) is None
    assert smz.json_scalar(np.float64("nan")) is None
    assert smz.json_scalar(np.float64(0.5)) == 0.5
    assert smz.json_scalar(np.int64(3)) == 3
    assert smz.json_scalar("text") == "text"
    payload = {"top_k": smz.json_scalar(np.float64("nan")), "v": smz.json_scalar(1.5)}
    json.dumps(payload, allow_nan=False)
    print("PASS test_summary_json_sanitizer")


def main() -> None:
    test_formal_matrix_validation()
    test_formal_budget_and_name_lock()
    test_frozen_thresholds_match_clong_config()
    test_s0_lineage_verification()
    test_annotate_uses_frozen_threshold_only()
    test_duplicate_rate_excludes_dead_features()
    test_pruning_selection_never_violates_tolerance()
    test_attention_pooling_identity()
    test_debug_subset_keeps_patient_integrity()
    test_pareto_and_selection_chain()
    test_cache_reuse_verifies_file_sha()
    test_extended_fidelity_metrics()
    test_confusion_at_threshold_edge()
    test_summary_json_sanitizer()
    print("全部 14 项测试通过")


if __name__ == "__main__":
    main()
