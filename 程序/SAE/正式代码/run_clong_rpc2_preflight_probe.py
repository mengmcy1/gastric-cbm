#!/usr/bin/env python3
"""执行 RP-C2 全对象工程 preflight 或四对象中间剂量 probe。

两个阶段均只读取 matching v2 冻结清单，不执行重新 matching，也不生成
RP-C2 正式科学摘要。preflight 核对 alpha=1 identity 与 alpha=0 RP-C1
逐层产物；probe 验证四个冻结工程对象及其 matched references 的五档路径。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clong_rpc2_core import (
    ALPHAS,
    fixed_attention_delta_margin,
    intermediate_curve,
    matched_reference_summary,
)
from clong_rpc_core import (
    aggregate_active_by_patient,
    aggregate_image_matrix_by_patient,
    intervention_block,
    label_seed_metrics,
)
from clong_rpa_prepare_seed import checkpoint_path, load_sae
from clong_rpb_core import SEEDS, file_sha256
from clong_sae_discovery import FROZEN_PATIENT_THRESHOLD, load_clong_model
from run_clong_rpc1_effect_screen import (
    CACHE_ROOT,
    RPA_ROOT,
    SPATIAL_ROOT,
    model_weights,
    run_identity_gate,
    screen_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path(__file__).resolve().parent
MATCHING_ROOT = (
    PROJECT_ROOT
    / "结果/SAE/RP_C2_Matching_Feasibility_20260825/matching_v2_freeze_rational_ties"
)
INVALID_MATCHING_ROOT = (
    PROJECT_ROOT / "结果/SAE/RP_C2_Matching_Feasibility_20260825/matching_v2_freeze"
)
STUDY_PATH = (
    PROJECT_ROOT
    / "结果/SAE/RP_C2_Matching_Feasibility_20260825/study_objects/rpc2_study_objects.csv"
)
CONTROL_PATH = MATCHING_ROOT / "rpc2_matched_control_manifest_v2.csv"
TARGET_PATH = MATCHING_ROOT / "rpc2_target_matching_summary_v2.csv"
RPC1_ROOT = PROJECT_ROOT / "结果/SAE/RP_C1_Effect_Screen_20260825/formal"
OUTPUT_ROOT = PROJECT_ROOT / "结果/SAE/RP_C2_Intervention_20260825"
PROTOCOL_PATH = CODE_ROOT / "rpc2_intervention_protocol_v1.json"
MATCHING_PROTOCOL_PATH = CODE_ROOT / "rpc2_matching_protocol_v2.json"
PROBE_ANCHORS = ("a00139", "a00987", "a00816", "a00019")
ARRAY_NAMES = (
    "image_delta_margin",
    "image_delta_probability",
    "image_attention_cosine",
    "image_attention_l1",
    "image_peak_changed",
    "image_active_image",
    "patient_delta_margin",
    "patient_delta_probability",
    "patient_attention_cosine",
    "patient_attention_l1",
    "patient_peak_changed",
    "patient_active",
    "original_image_margin",
    "original_image_probability",
    "original_patient_probability",
)


def parse_args() -> argparse.Namespace:
    """解析互斥工程阶段和设备参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "probe"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--image-batch", type=int, default=32)
    parser.add_argument("--feature-block", type=int, default=64)
    return parser.parse_args()


def verify_frozen_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """核验正式目录和协议绑定的四项 matching/RP-C1 SHA。"""
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_rpc2_intervention_2026-08-25":
        raise RuntimeError("RP-C2干预协议未冻结")
    frozen = protocol["frozen_inputs"]
    paths = {
        "study_object_manifest_sha256": STUDY_PATH,
        "matching_protocol_v2_sha256": MATCHING_PROTOCOL_PATH,
        "matched_control_manifest_sha256": CONTROL_PATH,
        "target_matching_summary_sha256": TARGET_PATH,
        "rpc1_protocol_sha256": CODE_ROOT / "rpc_effect_screen_protocol_v1.json",
        "rpc1_effect_master_sha256": RPC1_ROOT / "summary/rpc1_anchor_effect_master.csv",
    }
    for key, path in paths.items():
        if file_sha256(path) != frozen[key]:
            raise RuntimeError(f"冻结输入SHA不一致: {key}")
    matching_config = json.loads((MATCHING_ROOT / "config.json").read_text(encoding="utf-8"))
    if matching_config.get("status") != "matching_v2_frozen_before_rpc2_intervention":
        raise RuntimeError("matching v2正式目录状态非法")
    if INVALID_MATCHING_ROOT.resolve() == MATCHING_ROOT.resolve():
        raise RuntimeError("正式runner指向已作废matching目录")
    study = pd.read_csv(STUDY_PATH, dtype={"anchor_id": str})
    controls = pd.read_csv(CONTROL_PATH, dtype={"anchor_id": str})
    if len(study) != 149 or len(controls) != 44424:
        raise RuntimeError("RP-C2冻结清单行数不一致")
    return study, controls


def load_inputs() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载 train-only 空间表示以及固定图像/患者顺序。"""
    images = pd.read_csv(CACHE_ROOT / "seed42/train_images.csv", dtype={"patient_id": str})
    patients = pd.read_csv(CACHE_ROOT / "seed42/train_patients.csv", dtype={"patient_id": str})
    metadata = pd.read_csv(SPATIAL_ROOT / "train_metadata.csv", dtype={"patient_id": str})
    spatial = np.load(SPATIAL_ROOT / "train_spatial_features.npy", mmap_mode="r")
    if not images.relative_path.astype(str).equals(metadata.relative_path.astype(str)):
        raise RuntimeError("RP-A激活与空间特征图像顺序不一致")
    for seed in SEEDS:
        current = pd.read_csv(CACHE_ROOT / f"seed{seed}/train_images.csv", dtype={"patient_id": str})
        if not images.equals(current):
            raise RuntimeError(f"seed{seed}图像表不一致")
    return spatial, images, patients, metadata


def compare_exact_array(candidate: Path, reference: Path, columns: np.ndarray | None) -> None:
    """按 RP-C1 已保存 dtype 对数组执行精确相等回归。"""
    current = np.load(candidate, mmap_mode="r")
    expected = np.load(reference, mmap_mode="r")
    if columns is not None and expected.ndim == 2:
        expected = expected[:, columns]
    if current.dtype != expected.dtype or current.shape != expected.shape:
        raise RuntimeError(f"alpha=0数组dtype/shape不一致: {candidate.name}")
    if not np.array_equal(np.asarray(current), np.asarray(expected), equal_nan=True):
        raise RuntimeError(f"alpha=0数组未精确复现RP-C1: {candidate.name}")


def compare_seed_metrics(candidate: Path, reference: Path, anchor_ids: list[str]) -> None:
    """核对 alpha=0 的 seed-anchor 指标，忽略仅表示列位置的 anchor_column。"""
    current = pd.read_csv(candidate, dtype={"anchor_id": str}).set_index("anchor_id")
    expected = pd.read_csv(reference, dtype={"anchor_id": str}).set_index("anchor_id").loc[anchor_ids]
    if not current.index.equals(expected.index):
        raise RuntimeError("alpha=0 seed-anchor顺序不一致")
    columns = [column for column in expected.columns if column != "anchor_column"]
    for column in columns:
        left = current[column].to_numpy()
        right = expected[column].to_numpy()
        if left.dtype.kind in "iufcb" and right.dtype.kind in "iufcb":
            equal = np.array_equal(left, right, equal_nan=True)
        else:
            equal = np.array_equal(left.astype(str), right.astype(str))
        if not equal:
            raise RuntimeError(f"alpha=0 seed-anchor指标未精确复现: {column}")


def run_preflight(
    study: pd.DataFrame, spatial: np.ndarray, images: pd.DataFrame,
    patients: pd.DataFrame, metadata: pd.DataFrame, model: torch.nn.Module,
    device: torch.device, image_batch: int, feature_block: int, target: Path,
) -> None:
    """全149对象执行 alpha=1 identity 与 alpha=0 RP-C1回归。"""
    identity = run_identity_gate(model, spatial, metadata, device, debug=False)
    target.mkdir(parents=True)
    (target / "alpha1_identity_gate.json").write_text(
        json.dumps(identity, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    all_anchors = pd.read_csv(
        RPA_ROOT / "full_train_matching/formal/development_anchors.csv",
        dtype={"anchor_id": str},
    )
    anchor_lookup = {anchor: index for index, anchor in enumerate(all_anchors.anchor_id)}
    columns = np.asarray([anchor_lookup[anchor] for anchor in study.anchor_id], dtype=np.int64)
    for seed in SEEDS:
        screen_seed(
            seed, study, spatial, images, patients, model, device,
            image_batch, feature_block, target,
        )
        current_root = target / f"seed{seed}"
        reference_root = RPC1_ROOT / f"seed{seed}"
        for name in ARRAY_NAMES:
            candidate = current_root / f"{name}.npy"
            reference = reference_root / f"{name}.npy"
            subset = columns if np.load(reference, mmap_mode="r").ndim == 2 else None
            compare_exact_array(candidate, reference, subset)
        compare_seed_metrics(
            current_root / "seed_anchor_metrics.csv",
            reference_root / "seed_anchor_metrics.csv",
            study.anchor_id.tolist(),
        )
    payload = {
        "status": "rpc2_preflight_passed",
        "study_object_count": 149,
        "target_seed_count": 447,
        "alpha1_identity": identity,
        "alpha0_rpc1_exact_reproduction": True,
        "scientific_summary": False,
        "train_only": True,
        "val_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
    }
    (target / "config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )


@torch.no_grad()
def evaluate_probe_seed(
    seed: int, feature_ids: np.ndarray, target_ids: set[int], spatial: np.ndarray,
    images: pd.DataFrame, patients: pd.DataFrame, model: torch.nn.Module,
    device: torch.device, image_batch: int, feature_block: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算一个seed的probe Feature五档指标和目标 fixed-attention 诊断。"""
    cache = CACHE_ROOT / f"seed{seed}"
    activations = np.load(cache / "train_image_activations.npy", mmap_mode="r")
    decoder_all = np.load(cache / "decoder_weight.npy", mmap_mode="r")
    patient_ids = patients.patient_id.astype(str).to_numpy()
    image_patient_ids = images.patient_id.astype(str).to_numpy()
    labels = patients.label.to_numpy(int)
    attention_weight, attention_bias, margin_weight, margin_bias = model_weights(model, device)
    rows: list[pd.DataFrame] = []
    fixed_rows: list[dict[str, object]] = []
    target_positions = np.asarray(
        [index for index, feature in enumerate(feature_ids) if int(feature) in target_ids],
        dtype=np.int64,
    )
    for alpha in ALPHAS:
        matrices = {
            "delta_margin": np.empty((len(images), len(feature_ids)), dtype=np.float32),
            "delta_probability": np.empty((len(images), len(feature_ids)), dtype=np.float32),
            "attention_cosine": np.empty((len(images), len(feature_ids)), dtype=np.float32),
            "attention_l1": np.empty((len(images), len(feature_ids)), dtype=np.float32),
            "peak_changed": np.empty((len(images), len(feature_ids)), dtype=bool),
            "active_image": np.empty((len(images), len(feature_ids)), dtype=bool),
        }
        original_probability = np.empty(len(images), dtype=np.float32)
        fixed_target = np.empty((len(images), len(target_positions)), dtype=np.float32)
        for image_start in range(0, len(images), image_batch):
            image_stop = min(image_start + image_batch, len(images))
            features = torch.from_numpy(np.asarray(spatial[image_start:image_stop]).copy()).to(device)
            attention_logits = features @ attention_weight + attention_bias
            original_attention = torch.softmax(attention_logits, dim=1)
            for block_start in range(0, len(feature_ids), feature_block):
                block_stop = min(block_start + feature_block, len(feature_ids))
                ids = feature_ids[block_start:block_stop]
                hidden = torch.from_numpy(
                    np.asarray(np.take(activations[image_start:image_stop], ids, axis=2)).copy()
                ).to(device)
                decoder = torch.from_numpy(np.asarray(decoder_all[ids]).copy()).to(device)
                result = intervention_block(
                    features, hidden, decoder, attention_weight, attention_bias,
                    margin_weight, margin_bias, float(alpha),
                )
                if block_start == 0:
                    original_probability[image_start:image_stop] = (
                        result["original_probability"].cpu().numpy()
                    )
                for name, values in matrices.items():
                    values[image_start:image_stop, block_start:block_stop] = (
                        result[name].cpu().numpy()
                    )
                local_targets = [
                    (global_position, global_position - block_start)
                    for global_position in target_positions
                    if block_start <= global_position < block_stop
                ]
                if local_targets:
                    global_positions = np.asarray([item[0] for item in local_targets])
                    local_positions = np.asarray([item[1] for item in local_targets])
                    fixed_target[image_start:image_stop, np.searchsorted(target_positions, global_positions)] = (
                        fixed_attention_delta_margin(
                            original_attention,
                            hidden[:, :, local_positions],
                            decoder[local_positions], margin_weight, float(alpha),
                        ).cpu().numpy()
                    )
        patient_outputs = {
            name: aggregate_image_matrix_by_patient(values, image_patient_ids, patient_ids)
            for name, values in matrices.items()
            if name not in {"active_image"}
        }
        active_patient = aggregate_active_by_patient(
            matrices["active_image"], image_patient_ids, patient_ids,
        )
        original_patient_probability = aggregate_image_matrix_by_patient(
            original_probability[:, None], image_patient_ids, patient_ids,
        )[:, 0]
        dose = pd.DataFrame(label_seed_metrics(
            patient_outputs["delta_margin"], patient_outputs["delta_probability"],
            patient_outputs["attention_cosine"], patient_outputs["attention_l1"],
            patient_outputs["peak_changed"], active_patient, labels,
            original_patient_probability, FROZEN_PATIENT_THRESHOLD,
        ))
        dose["feature_id"] = feature_ids
        dose["seed"] = seed
        dose["alpha"] = float(alpha)
        rows.append(dose)
        fixed_patient = aggregate_image_matrix_by_patient(
            fixed_target, image_patient_ids, patient_ids,
        )
        for local, position in enumerate(target_positions):
            fixed_rows.append({
                "seed": seed,
                "feature_id": int(feature_ids[position]),
                "alpha": float(alpha),
                "fixed_attention_overall_mean_abs_delta_margin": float(
                    np.abs(fixed_patient[:, local]).mean()
                ),
            })
    return pd.concat(rows, ignore_index=True), pd.DataFrame(fixed_rows)


def run_probe(
    study: pd.DataFrame, controls: pd.DataFrame, spatial: np.ndarray,
    images: pd.DataFrame, patients: pd.DataFrame, model: torch.nn.Module,
    device: torch.device, image_batch: int, feature_block: int, target: Path,
) -> None:
    """运行四个冻结工程对象和其实际matched references的五档路径。"""
    probe_study = study.set_index("anchor_id").loc[list(PROBE_ANCHORS)].reset_index()
    dose_frames = []
    fixed_frames = []
    reference_rows = []
    for seed in SEEDS:
        seed_controls = controls[
            controls.sae_seed.eq(seed) & controls.anchor_id.isin(PROBE_ANCHORS)
        ]
        target_features = set(probe_study[f"feature_{seed}"].astype(int))
        feature_ids = np.asarray(sorted(
            target_features | set(seed_controls.control_feature_id.astype(int))
        ), dtype=np.int64)
        dose, fixed = evaluate_probe_seed(
            seed, feature_ids, target_features, spatial, images, patients,
            model, device, image_batch, feature_block,
        )
        dose_frames.append(dose)
        fixed_frames.append(fixed)
        overall = dose.pivot(index="feature_id", columns="alpha", values="overall_mean_abs_delta_margin")
        alpha_columns = [float(value) for value in ALPHAS]
        overall = overall.loc[:, alpha_columns]
        curves = {
            int(feature): float(intermediate_curve(row.to_numpy()))
            for feature, row in overall.iterrows()
        }
        for item in probe_study.itertuples(index=False):
            anchor = str(item.anchor_id)
            target_feature = int(getattr(item, f"feature_{seed}"))
            control_ids = seed_controls.loc[
                seed_controls.anchor_id.eq(anchor), "control_feature_id"
            ].astype(int).to_numpy()
            summary = matched_reference_summary(
                curves[target_feature], np.asarray([curves[int(value)] for value in control_ids]),
            )
            reference_rows.append({
                "anchor_id": anchor,
                "seed": seed,
                "target_feature_id": target_feature,
                "intermediate_curve_overall_effect": curves[target_feature],
                **summary,
            })
    target.mkdir(parents=True)
    pd.concat(dose_frames, ignore_index=True).to_csv(target / "probe_seed_dose_metrics.csv", index=False)
    pd.concat(fixed_frames, ignore_index=True).to_csv(
        target / "probe_fixed_attention_metrics.csv", index=False,
    )
    pd.DataFrame(reference_rows).to_csv(target / "probe_matched_reference_summary.csv", index=False)
    payload = {
        "status": "rpc2_engineering_probe_complete",
        "probe_anchors": list(PROBE_ANCHORS),
        "probe_anchor_count": 4,
        "probe_target_seed_count": 12,
        "probe_results_enter_scientific_summary": False,
        "protocol_changes_based_on_probe_effect_shape": False,
        "scientific_summary": False,
        "train_only": True,
        "val_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
    }
    (target / "config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )


def main() -> None:
    """核验冻结输入后执行且仅执行所选工程阶段。"""
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RP-C2正式工程preflight/probe必须使用CUDA")
    study, controls = verify_frozen_inputs()
    spatial, images, patients, metadata = load_inputs()
    target = OUTPUT_ROOT / args.stage
    if target.exists():
        raise FileExistsError(f"RP-C2工程输出已存在，禁止覆盖: {target}")
    device = torch.device("cuda")
    model = load_clong_model(device)
    if args.stage == "preflight":
        run_preflight(
            study, spatial, images, patients, metadata, model, device,
            args.image_batch, args.feature_block, target,
        )
    else:
        run_probe(
            study, controls, spatial, images, patients, model, device,
            args.image_batch, args.feature_block, target,
        )
    print(f"RP-C2 {args.stage}完成: {target}")


if __name__ == "__main__":
    main()
