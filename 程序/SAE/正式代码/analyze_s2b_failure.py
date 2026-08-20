#!/usr/bin/env python3
"""S2b 失败诊断：只分析既有正式产物，不训练新模型，不修改任何冻结门槛。

输入全部为已冻结资产：C-long 学生、空间特征缓存、S4-C/D 的
``sae_best.pth`` 与 S4-D 的 ``batchtopk_threshold.json``。复算口径与
``clong_s2b_discovery.py`` 完全一致，并先把复算指标与正式 config 逐位
对齐（容差1e-6）作为自检，再输出诊断。

回答四个问题：

1. 哪些患者在冻结阈值下预测发生变化，变化时是否本来就贴近阈值；
2. 正式pooled cosine与patch cosine偏低的图像集中在哪些分层
   （标签/来源/分辨率/画中画/病灶大小），最差的是哪些图；
3. 偏差来自特征内容重构还是注意力重算（固定原注意力对照）；
4. C 与 D 的失败患者是否为同一批。

只读取train/val，不触碰internal test与external。分析结论不用于修改
S2b 门槛，只作为后续路线决策依据。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "程序/MAGE/正式代码"))
sys.path.insert(0, str(PROJECT_ROOT / "程序/模型训练/正式代码"))

from build_mage_teacher_roi_manifest import file_sha256  # noqa: E402
from clong_sae_discovery import (  # noqa: E402
    FROZEN_IMAGE_THRESHOLD,
    FROZEN_PATIENT_THRESHOLD,
    INPUT_DIM,
    load_clong_model,
    load_manifest_frame,
    verify_s0_lineage,
)
from clong_s2b_core import (  # noqa: E402
    StructuredSparseAutoencoder,
    pooled_from_features,
)
from clong_s2b_discovery import (  # noqa: E402
    CACHE_ROOT,
    HIDDEN_DIM,
    PATCH_K,
    classifier_components,
    load_cache,
    patient_frame,
    project_patch,
)
from train_utils import json_ready  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT / "结果/SAE/CLong_S2b结构重构_20260820"
ARMS = {
    "C": {"experiment": "s4c_clong_seed42", "activation_mode": "topk"},
    "D": {"experiment": "s4d_clong_seed42", "activation_mode": "batch_topk"},
}
REPRODUCE_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_ROOT / "failure_diagnosis_seed42"
    )
    return parser.parse_args()


def load_formal_sae(
    arm: str, device: torch.device
) -> tuple[StructuredSparseAutoencoder, float | None, dict]:
    """加载正式臂checkpoint，逐项核验与config.json记录一致。"""
    info = ARMS[arm]
    directory = OUTPUT_ROOT / info["experiment"]
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    checkpoint_path = directory / "sae_best.pth"
    if config["checkpoint_sha256"] != file_sha256(checkpoint_path):
        raise ValueError(f"S4-{arm} checkpoint SHA与config不一致")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = {
        "arm": arm, "seed": 42, "hidden_dim": HIDDEN_DIM,
        "target_k": PATCH_K, "activation_mode": info["activation_mode"],
    }
    failed = [key for key, value in expected.items() if checkpoint.get(key) != value]
    if failed:
        raise ValueError(f"S4-{arm} checkpoint字段不一致: {failed}")
    sae = StructuredSparseAutoencoder(
        INPUT_DIM, HIDDEN_DIM, torch.zeros(INPUT_DIM),
        info["activation_mode"], PATCH_K,
    )
    sae.load_state_dict(checkpoint["sae_state_dict"], strict=True)
    sae.to(device).eval()
    threshold = None
    if arm == "D":
        threshold_info = json.loads(
            (directory / "batchtopk_threshold.json").read_text(encoding="utf-8")
        )
        if threshold_info["checkpoint_sha256"] != config["checkpoint_sha256"]:
            raise ValueError("S4-D冻结阈值与checkpoint SHA不一致")
        threshold = float(threshold_info["threshold"])
    return sae, threshold, config


@torch.no_grad()
def fixed_attention_pooled(
    sae: StructuredSparseAutoencoder, spatial: np.ndarray,
    attention: np.ndarray, threshold: float | None, batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """固定原注意力只加权重构内容，与正式评价的完整替换口径对照。"""
    rows = []
    for start in range(0, len(spatial), batch_size):
        original = torch.from_numpy(
            np.asarray(spatial[start:start + batch_size]).copy()
        ).to(device)
        original_attn = torch.from_numpy(
            np.asarray(attention[start:start + batch_size]).copy()
        ).to(device)
        hidden = (
            sae.encode_threshold(original, threshold)
            if threshold is not None else sae.encode(original)
        )
        rows.append(pooled_from_features(sae.decode(hidden), original_attn).cpu().numpy())
    return np.concatenate(rows)


def image_probabilities(pooled: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    logits = pooled @ weight.T + bias
    return torch.softmax(torch.from_numpy(logits), 1)[:, 1].numpy()


def row_cosine(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """计算每张图pooled向量的cosine，口径与正式保真门槛一致。"""
    numerator = np.sum(original * reconstructed, axis=1)
    denominator = np.linalg.norm(original, axis=1) * np.linalg.norm(reconstructed, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def per_image_attention_drift(original: np.ndarray, rebuilt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐图注意力KL(original‖rebuilt)与cosine。"""
    original = np.clip(np.asarray(original, dtype=np.float64), 1e-12, None)
    rebuilt = np.clip(np.asarray(rebuilt, dtype=np.float64), 1e-12, None)
    kl = (original * np.log(original / rebuilt)).sum(1)
    cosine = (original * rebuilt).sum(1) / (
        np.linalg.norm(original, axis=1) * np.linalg.norm(rebuilt, axis=1)
    )
    return kl, cosine


def lesion_tiers(metadata: dict) -> tuple[pd.Series, list[float]]:
    """按train癌图冻结的三分位给val癌图分small/medium/large。"""
    train_area = metadata["train"].loc[
        metadata["train"].label.eq(1), "bbox_area_fraction"
    ].to_numpy(float)
    bounds = np.quantile(train_area, [1 / 3, 2 / 3]).tolist()
    area = metadata["val"]["bbox_area_fraction"].to_numpy(float)
    is_cancer = metadata["val"].label.eq(1).to_numpy()
    tier = pd.Series(np.where(is_cancer, "unassigned", "non_cancer"), dtype=object)
    valid = is_cancer & np.isfinite(area)
    tier.loc[valid & (area <= bounds[0])] = "small"
    tier.loc[valid & (area > bounds[0]) & (area <= bounds[1])] = "medium"
    tier.loc[valid & (area > bounds[1])] = "large"
    if (tier == "unassigned").any():
        raise RuntimeError("存在无法分层的val癌图")
    return tier, bounds


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"诊断输出目录已存在: {args.output}")
    device = torch.device(args.device)
    verify_s0_lineage()
    frame = load_manifest_frame(SimpleNamespace(debug=False))
    model = load_clong_model(device)
    spatial, pooled, attention, metadata = load_cache(frame)
    weight, bias, _ = classifier_components(model)
    weight_np, bias_np = weight.cpu().numpy(), bias.cpu().numpy()

    val_meta = metadata["val"].reset_index(drop=True)
    original_probability = image_probabilities(
        np.asarray(pooled["val"]), weight_np, bias_np
    )
    tiers, tier_bounds = lesion_tiers(metadata)

    per_arm = {}
    for arm in ARMS:
        sae, threshold, config = load_formal_sae(arm, device)
        rebuilt_pool, rebuilt_attention, coverage = project_patch(
            sae, spatial["val"], attention["val"], val_meta, model,
            threshold, args.batch_size, device, False,
        )
        fixed_pool = fixed_attention_pooled(
            sae, spatial["val"], attention["val"], threshold,
            args.batch_size, device,
        )
        # 自检：复算指标必须与正式config逐位一致。
        full_patient = patient_frame(val_meta, original_probability,
                                     image_probabilities(rebuilt_pool, weight_np, bias_np))
        agreement = float((
            (full_patient.original >= FROZEN_PATIENT_THRESHOLD)
            == (full_patient.reconstructed >= FROZEN_PATIENT_THRESHOLD)
        ).mean())
        expected = config["evaluation"]["fidelity"]
        checks = {
            "patient_agreement": (
                agreement,
                expected["patient_prediction_agreement_at_locked_threshold"],
            ),
            "mean_cosine": (
                float(np.mean([
                    float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                    for a, b in zip(np.asarray(pooled["val"]), rebuilt_pool)
                ])),
                expected["mean_cosine"],
            ),
        }
        failed = {
            key: (actual, want) for key, (actual, want) in checks.items()
            if abs(actual - want) > REPRODUCE_TOLERANCE
        }
        if failed:
            raise RuntimeError(f"S4-{arm}复算与正式config不一致: {failed}")
        kl, attn_cosine = per_image_attention_drift(attention["val"], rebuilt_attention)
        per_arm[arm] = {
            "rebuilt_probability": image_probabilities(rebuilt_pool, weight_np, bias_np),
            "fixed_probability": image_probabilities(fixed_pool, weight_np, bias_np),
            "pooled_cosine": row_cosine(np.asarray(pooled["val"]), rebuilt_pool),
            "per_image_patch_cosine": np.asarray(coverage["per_image_mean_patch_cosine"]),
            "attention_kl": kl,
            "attention_cosine": attn_cosine,
        }
        print(f"S4-{arm}复算自检通过: agreement={agreement:.6f}")

    # 逐图诊断表。
    image_table = val_meta[[
        "relative_path", "patient_id", "label", "source", "center",
        "size_group", "style_group", "pip_present_original", "bbox_area_fraction",
    ]].copy()
    image_table["lesion_size_tier"] = tiers
    image_table["original_probability"] = original_probability
    for arm in ARMS:
        image_table[f"rebuilt_probability_{arm}"] = per_arm[arm]["rebuilt_probability"]
        image_table[f"fixed_probability_{arm}"] = per_arm[arm]["fixed_probability"]
        image_table[f"pooled_cosine_{arm}"] = per_arm[arm]["pooled_cosine"]
        image_table[f"patch_cosine_{arm}"] = per_arm[arm]["per_image_patch_cosine"]
        image_table[f"attention_kl_{arm}"] = per_arm[arm]["attention_kl"]
        image_table[f"attention_cosine_{arm}"] = per_arm[arm]["attention_cosine"]
        image_table[f"image_flipped_{arm}_full"] = (
            (original_probability >= FROZEN_IMAGE_THRESHOLD)
            != (per_arm[arm]["rebuilt_probability"] >= FROZEN_IMAGE_THRESHOLD)
        )
        image_table[f"image_flipped_{arm}_fixed"] = (
            (original_probability >= FROZEN_IMAGE_THRESHOLD)
            != (per_arm[arm]["fixed_probability"] >= FROZEN_IMAGE_THRESHOLD)
        )

    # 患者级诊断表。
    patient_table = None
    variants = [("rebuilt", arm) for arm in ARMS] + [("fixed", arm) for arm in ARMS]
    for variant, arm in variants:
        key = f"{variant}_probability_{arm}"
        aggregated = patient_frame(
            val_meta, original_probability, per_arm[arm][f"{variant}_probability"],
        ).rename(columns={"original": "original_probability", "reconstructed": key})
        if patient_table is None:
            patient_table = aggregated
        else:
            patient_table = patient_table.merge(
                aggregated[["patient_id", key]],
                on="patient_id", how="left", validate="one_to_one",
            )
    patient_table["abs_distance_to_threshold"] = (
        patient_table["original_probability"] - FROZEN_PATIENT_THRESHOLD
    ).abs()
    for variant in ("rebuilt", "fixed"):
        for arm in ARMS:
            patient_table[f"flipped_{arm}_{variant}"] = (
                (patient_table["original_probability"] >= FROZEN_PATIENT_THRESHOLD)
                != (patient_table[f"{variant}_probability_{arm}"] >= FROZEN_PATIENT_THRESHOLD)
            )
    image_counts = val_meta.groupby("patient_id").size().rename("n_images")
    patient_table = patient_table.merge(
        image_counts, on="patient_id", how="left", validate="one_to_one",
    )

    args.output.mkdir(parents=True)
    image_table.to_csv(args.output / "per_image_diagnosis.csv",
                       index=False, encoding="utf-8-sig")
    patient_table.to_csv(args.output / "changed_patients.csv",
                         index=False, encoding="utf-8-sig")

    # 汇总四个诊断问题。
    summary = {"stage": "S2b_failure_diagnosis", "seed": 42,
               "frozen_patient_threshold": FROZEN_PATIENT_THRESHOLD,
               "frozen_image_threshold": FROZEN_IMAGE_THRESHOLD,
               "lesion_size_bounds_train_frozen": tier_bounds,
               "reproduction_check": "两臂患者一致率与pooled cosine复算均与正式config一致(容差1e-6)"}
    q1 = {}
    for arm in ARMS:
        flipped = patient_table.loc[patient_table[f"flipped_{arm}_rebuilt"]]
        rest = patient_table.loc[~patient_table[f"flipped_{arm}_rebuilt"]]
        q1[f"s4{arm.lower()}"] = {
            "changed_patient_count": int(len(flipped)),
            "changed_patient_ids": flipped.patient_id.tolist(),
            "label_breakdown": flipped.label.value_counts().to_dict(),
            "median_abs_distance_changed": float(flipped.abs_distance_to_threshold.median()),
            "median_abs_distance_unchanged": float(rest.abs_distance_to_threshold.median()),
            "changed_within_pm0_05": int((flipped.abs_distance_to_threshold <= 0.05).sum()),
            "changed_within_pm0_10": int((flipped.abs_distance_to_threshold <= 0.10).sum()),
        }
    both = patient_table.loc[
        patient_table.flipped_C_rebuilt & patient_table.flipped_D_rebuilt
    ]
    q1["cross_arm_overlap"] = int(len(both))
    summary["q1_changed_patients"] = q1

    q3 = {"strata": {}}
    for column in ("label", "source", "size_group", "pip_present_original", "lesion_size_tier"):
        cosine_columns = [
            "pooled_cosine_C", "pooled_cosine_D", "patch_cosine_C", "patch_cosine_D"
        ]
        grouped = image_table.groupby(column)[cosine_columns].mean()
        q3["strata"][column] = {
            str(index): {"images": int((image_table[column] == index).sum()),
                         "pooled_cosine_C": float(row.pooled_cosine_C),
                         "pooled_cosine_D": float(row.pooled_cosine_D),
                         "patch_cosine_C": float(row.patch_cosine_C),
                         "patch_cosine_D": float(row.patch_cosine_D)}
            for index, row in grouped.iterrows()
        }
    worst = image_table.nsmallest(30, "patch_cosine_C")
    q3["worst30_by_patch_cosine_C"] = {
        "relative_paths": worst.relative_path.tolist(),
        "label_breakdown": worst.label.value_counts().to_dict(),
        "pip_present_breakdown": worst.pip_present_original.value_counts().to_dict(),
        "size_group_breakdown": worst.size_group.value_counts().to_dict(),
    }
    worst_pooled = image_table.nsmallest(30, "pooled_cosine_C")
    q3["worst30_by_pooled_cosine_C"] = {
        "relative_paths": worst_pooled.relative_path.tolist(),
        "label_breakdown": worst_pooled.label.value_counts().to_dict(),
        "pip_present_breakdown": worst_pooled.pip_present_original.value_counts().to_dict(),
        "size_group_breakdown": worst_pooled.size_group.value_counts().to_dict(),
    }
    summary["q3_low_cosine_concentration"] = q3

    q4 = {}
    for arm in ARMS:
        flip_full = patient_table[f"flipped_{arm}_rebuilt"]
        flip_fixed = patient_table[f"flipped_{arm}_fixed"]
        q4[f"s4{arm.lower()}"] = {
            "agreement_full_replacement": float((~flip_full).mean()),
            "agreement_fixed_attention": float((~flip_fixed).mean()),
            "flips_full": int(flip_full.sum()),
            "flips_fixed": int(flip_fixed.sum()),
            "content_driven_flips_flip_in_both": int((flip_full & flip_fixed).sum()),
            "attention_driven_flips_only_full": int((flip_full & ~flip_fixed).sum()),
            "attention_recompute_corrected_fixed_only_flip": int((~flip_full & flip_fixed).sum()),
            "stable_in_both": int((~flip_full & ~flip_fixed).sum()),
            "attention_kl_mean": float(per_arm[arm]["attention_kl"].mean()),
            "attention_cosine_mean": float(per_arm[arm]["attention_cosine"].mean()),
            "corr_attention_kl_vs_patch_cosine": float(np.corrcoef(
                per_arm[arm]["attention_kl"], per_arm[arm]["per_image_patch_cosine"]
            )[0, 1]),
            "corr_attention_kl_vs_pooled_cosine": float(np.corrcoef(
                per_arm[arm]["attention_kl"], per_arm[arm]["pooled_cosine"]
            )[0, 1]),
        }
    summary["q4_content_vs_attention"] = q4
    summary["data_locks"] = {"test_evaluated": False, "internal_test_evaluated": False,
                             "external_evaluated": False}
    (args.output / "diagnosis_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"诊断输出目录: {args.output}")


if __name__ == "__main__":
    main()
