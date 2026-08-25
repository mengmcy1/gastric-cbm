#!/usr/bin/env python3
"""运行 RP-C2 正式五档 target 与 deterministic matched-reference 分析。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clong_rpc2_core import (
    ALPHAS,
    aligned_dose_spearman,
    direction_code,
    intermediate_curve,
    matched_reference_summary,
)
from clong_rpa_prepare_seed import checkpoint_path
from clong_rpb_core import SEEDS, file_sha256
from clong_sae_discovery import CLONG_CHECKPOINT, load_clong_model
from run_clong_rpc2_preflight_probe import (
    CACHE_ROOT,
    CONTROL_PATH,
    MATCHING_PROTOCOL_PATH,
    MATCHING_ROOT,
    PROTOCOL_PATH,
    RPC1_ROOT,
    SPATIAL_ROOT,
    STUDY_PATH,
    TARGET_PATH,
    evaluate_probe_seed,
    load_inputs,
    verify_frozen_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "结果/SAE/RP_C2_Intervention_20260825/formal"
RPC1_MASTER = RPC1_ROOT / "summary/rpc1_anchor_effect_master.csv"
ROLE_COLUMNS = (
    "primary_candidate",
    "low_effect_control",
    "post_rpc1_secondary_exploratory",
)
ALPHA_NAMES = {1.0: "1p00", 0.75: "0p75", 0.5: "0p50", 0.25: "0p25", 0.0: "0p00"}


def parse_args() -> argparse.Namespace:
    """解析冻结计算分块；正式运行只允许CUDA。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--image-batch", type=int, default=32)
    parser.add_argument("--feature-block", type=int, default=64)
    return parser.parse_args()


def prepare_output_root(path: Path) -> None:
    """正式目录必须全新；任何既有完整或残缺目录均拒绝覆盖。"""
    if path.exists():
        raise FileExistsError(f"RP-C2 formal目录已存在，禁止覆盖: {path}")
    path.mkdir(parents=True)


def unique_feature_universe(
    study: pd.DataFrame, controls: pd.DataFrame, seed: int,
) -> np.ndarray:
    """返回一个seed的study与冻结control Feature并集。"""
    target = set(study[f"feature_{seed}"].astype(int))
    reference = set(controls.loc[controls.sae_seed.eq(seed), "control_feature_id"].astype(int))
    output = np.asarray(sorted(target | reference), dtype=np.int64)
    if not target.issubset(set(output)):
        raise RuntimeError(f"seed{seed} unique universe缺少study target")
    return output


def alpha_zero_gate(
    study: pd.DataFrame, spatial: np.ndarray, images: pd.DataFrame,
    patients: pd.DataFrame, model: torch.nn.Module, device: torch.device,
    image_batch: int, feature_block: int, root: Path,
) -> dict[str, object]:
    """用正式五档核心独立复现全部447个target-seed的RP-C1 alpha=0。"""
    all_anchors = pd.read_csv(
        PROJECT_ROOT / "结果/SAE/RP_A_Development_20260824/full_train_matching/formal/development_anchors.csv",
        dtype={"anchor_id": str},
    )
    anchor_column = {anchor: index for index, anchor in enumerate(all_anchors.anchor_id)}
    checked_metrics = 0
    for seed in SEEDS:
        feature_to_anchor = {
            int(row[f"feature_{seed}"]): str(row.anchor_id)
            for _, row in study.iterrows()
        }
        feature_ids = np.asarray(sorted(feature_to_anchor), dtype=np.int64)
        capture = root / f"seed{seed}"
        metrics, _fixed = evaluate_probe_seed(
            seed, feature_ids, set(feature_ids.tolist()), spatial, images, patients,
            model, device, image_batch, feature_block,
            alphas=np.asarray([0.0]), capture_target_dir=capture,
        )
        metrics.to_csv(capture / "formal_core_alpha0_seed_metrics.csv", index=False)
        columns = np.asarray(
            [anchor_column[feature_to_anchor[int(feature)]] for feature in feature_ids],
            dtype=np.int64,
        )
        comparisons = (
            (capture / "image_delta_margin_alpha_0p00.npy", RPC1_ROOT / f"seed{seed}/image_delta_margin.npy"),
            (capture / "patient_delta_margin_alpha_0p00.npy", RPC1_ROOT / f"seed{seed}/patient_delta_margin.npy"),
        )
        for candidate_path, reference_path in comparisons:
            candidate = np.load(candidate_path, mmap_mode="r")
            reference = np.load(reference_path, mmap_mode="r")[:, columns]
            if candidate.dtype != reference.dtype or candidate.shape != reference.shape:
                raise RuntimeError(f"Gate A seed{seed} dtype/shape不一致: {candidate_path.name}")
            if not np.array_equal(np.asarray(candidate), np.asarray(reference), equal_nan=True):
                raise RuntimeError(f"Gate A seed{seed}未逐位复现: {candidate_path.name}")
        old = pd.read_csv(
            RPC1_ROOT / f"seed{seed}/seed_anchor_metrics.csv", dtype={"anchor_id": str},
        ).set_index("anchor_id")
        current = metrics.set_index("feature_id")
        excluded = {"anchor_column", "feature_id", "seed", "alpha"}
        for feature in feature_ids:
            anchor = feature_to_anchor[int(feature)]
            for column in old.columns:
                if column in excluded or column not in current.columns:
                    continue
                left = old.loc[anchor, column]
                right = current.loc[int(feature), column]
                if isinstance(left, str):
                    equal = left == right
                else:
                    equal = (
                        (pd.isna(left) and pd.isna(right))
                        or np.asarray(left).dtype.kind != "f" and left == right
                        or np.asarray(left).dtype.kind == "f"
                        and np.float64(left).tobytes() == np.float64(right).tobytes()
                    )
                if not equal:
                    raise RuntimeError(f"Gate A seed{seed}/{anchor}/{column}未逐位复现")
                checked_metrics += 1
    payload = {
        "status": "rpc2_formal_core_alpha0_gate_passed",
        "target_seed_count": 447,
        "image_delta_margin_exact": True,
        "patient_delta_margin_exact": True,
        "seed_level_metric_values_checked": checked_metrics,
        "scientific_summary": False,
    }
    (root / "config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    return payload


def source_rows(
    seed: int, study: pd.DataFrame, patients: pd.DataFrame, capture: Path,
) -> list[dict[str, object]]:
    """按冻结N>=15规则汇总study target的label×source患者效应。"""
    feature_ids = np.load(capture / "target_feature_ids.npy")
    feature_column = {int(feature): index for index, feature in enumerate(feature_ids)}
    rows: list[dict[str, object]] = []
    for alpha in ALPHAS:
        suffix = f"{float(alpha):.2f}".replace(".", "p")
        delta = np.load(capture / f"patient_delta_margin_alpha_{suffix}.npy", mmap_mode="r")
        for item in study.itertuples(index=False):
            feature = int(getattr(item, f"feature_{seed}"))
            values = np.asarray(delta[:, feature_column[feature]])
            for (label, source), indices in patients.groupby(["label", "source"], sort=True).indices.items():
                selected = values[np.asarray(indices)]
                supported = len(selected) >= 15
                rows.append({
                    "anchor_id": str(item.anchor_id),
                    "seed": seed,
                    "feature_id": feature,
                    "alpha": float(alpha),
                    "label": int(label),
                    "source": str(source),
                    "n_patients": int(len(selected)),
                    "source_effect_status": "reported" if supported else "insufficient_support",
                    "patient_balanced_mean_delta_margin": float(selected.mean()) if supported else None,
                    "patient_balanced_median_delta_margin": float(np.median(selected)) if supported else None,
                })
    return rows


def dose_vector(frame: pd.DataFrame, feature_id: int, metric: str) -> np.ndarray:
    """按冻结alpha顺序读取单Feature五档seed-level指标。"""
    selected = frame.loc[frame.feature_id.eq(feature_id)].set_index("alpha")
    return np.asarray([selected.loc[float(alpha), metric] for alpha in ALPHAS], dtype=np.float64)


def build_target_evidence(
    study: pd.DataFrame, controls: pd.DataFrame, rpc1: pd.DataFrame,
    dose_by_seed: dict[int, pd.DataFrame], fixed_by_seed: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """从unique Feature事实层映射447行target-seed证据。"""
    rpc1 = rpc1.set_index("anchor_id")
    rows: list[dict[str, object]] = []
    metric_names = {
        "overall": "overall_mean_abs_delta_margin",
        "cancer": "cancer_mean_delta_margin",
        "noncancer": "noncancer_mean_delta_margin",
        "separation": "class_separation_delta_D",
    }
    for item in study.itertuples(index=False):
        anchor = str(item.anchor_id)
        master = rpc1.loc[anchor]
        directions = {}
        for name, metric in (
            ("cancer", "cancer_mean_delta_margin"),
            ("noncancer", "noncancer_mean_delta_margin"),
            ("separation", "class_separation_delta_D"),
        ):
            directions[name] = direction_code(np.asarray([
                master[f"seed{seed}_{metric}"] for seed in SEEDS
            ], dtype=np.float64))
        functional_pattern = (
            "bidirectional_label_supporting_3of3"
            if directions["cancer"][0] == "negative" and directions["cancer"][1] == 3
            and directions["noncancer"][0] == "positive" and directions["noncancer"][1] == 3
            else "mixed_functional_pattern"
        )
        for seed in SEEDS:
            target_feature = int(getattr(item, f"feature_{seed}"))
            dose = dose_by_seed[seed]
            vectors = {
                name: dose_vector(dose, target_feature, metric)
                for name, metric in metric_names.items()
            }
            primary = float(intermediate_curve(vectors["overall"]))
            seed_controls = controls.loc[
                controls.sae_seed.eq(seed) & controls.anchor_id.eq(anchor),
                "control_feature_id",
            ].astype(int).to_numpy()
            control_values = np.asarray([
                float(intermediate_curve(dose_vector(dose, int(feature), metric_names["overall"])))
                for feature in seed_controls
            ])
            matched = matched_reference_summary(primary, control_values)
            fixed = fixed_by_seed[seed].set_index(["feature_id", "alpha"])
            row: dict[str, object] = {
                "anchor_id": anchor,
                "seed": seed,
                "target_feature_id": target_feature,
                **{column: bool(getattr(item, column)) for column in ROLE_COLUMNS},
                "functional_pattern": functional_pattern,
                "intermediate_curve_overall_effect": primary,
                **matched,
            }
            for name in ("cancer", "noncancer", "separation"):
                expected, count, pattern = directions[name]
                curve = vectors[name]
                row[f"expected_{name}_direction"] = expected
                row[f"rpc1_{name}_matching_seed_count"] = count
                row[f"rpc1_{name}_seed_sign_pattern"] = pattern
                row[f"intermediate_curve_{name}_raw"] = float(intermediate_curve(curve))
                row[f"intermediate_curve_{name}_aligned"] = (
                    None if expected == "zero" else
                    float(intermediate_curve(curve)) * (1.0 if expected == "positive" else -1.0)
                )
                rho = aligned_dose_spearman(curve, expected)
                row[f"{name}_dose_spearman"] = rho
                row[f"{name}_dose_curve_regular"] = None if rho is None else bool(rho >= 0.9)
            for alpha_index, alpha in enumerate(ALPHAS):
                alpha_name = ALPHA_NAMES[float(alpha)]
                for name, curve in vectors.items():
                    row[f"{name}_alpha_{alpha_name}"] = float(curve[alpha_index])
                fixed_effect = float(fixed.loc[(target_feature, float(alpha)), "fixed_attention_overall_mean_abs_delta_margin"])
                row[f"fixed_attention_overall_alpha_{alpha_name}"] = fixed_effect
                row[f"attention_rearrangement_associated_difference_alpha_{alpha_name}"] = (
                    float(vectors["overall"][alpha_index]) - fixed_effect
                )
            rows.append(row)
    output = pd.DataFrame(rows).sort_values(["anchor_id", "seed"]).reset_index(drop=True)
    if len(output) != 447:
        raise RuntimeError("target-seed evidence不是447行")
    return output


def build_anchor_master(evidence: pd.DataFrame) -> pd.DataFrame:
    """先保留三seed原值，再生成149行anchor描述汇总。"""
    rows = []
    for anchor, group in evidence.groupby("anchor_id", sort=True):
        group = group.sort_values("seed")
        if tuple(group.seed.astype(int)) != tuple(SEEDS):
            raise RuntimeError(f"{anchor}缺少三个seed")
        row: dict[str, object] = {
            "anchor_id": anchor,
            **{column: bool(group.iloc[0][column]) for column in ROLE_COLUMNS},
            "functional_pattern": group.iloc[0].functional_pattern,
        }
        for item in group.itertuples(index=False):
            prefix = f"seed{int(item.seed)}"
            for metric in (
                "target_feature_id",
                "intermediate_curve_overall_effect",
                "matched_midrank_percentile",
                "matched_plus_one_tail_fraction",
                "effect_ratio_to_control_median",
                "n_control",
            ):
                row[f"{prefix}_{metric}"] = getattr(item, metric)
        row["median_intermediate_curve_overall_effect"] = float(
            group.intermediate_curve_overall_effect.median()
        )
        row["median_matched_midrank_percentile"] = float(group.matched_midrank_percentile.median())
        row["minimum_matched_midrank_percentile"] = float(group.matched_midrank_percentile.min())
        rows.append(row)
    output = pd.DataFrame(rows)
    if len(output) != 149:
        raise RuntimeError("anchor evidence master不是149行")
    return output


def provenance(args: argparse.Namespace) -> dict[str, object]:
    """记录决定正式计算的代码、模型、缓存和CUDA环境。"""
    code_files = (
        "run_clong_rpc2_formal.py",
        "run_clong_rpc2_preflight_probe.py",
        "clong_rpc2_core.py",
        "clong_rpc_core.py",
        "rpc2_intervention_protocol_v1.json",
    )
    checkpoints = {}
    cache_configs = {}
    for seed in SEEDS:
        path = checkpoint_path(argparse.Namespace(
            debug=False, debug_patients_per_class=0, debug_epochs=0, seed=seed,
        ))
        checkpoints[str(seed)] = {"path": str(path), "sha256": file_sha256(path)}
        cache_path = CACHE_ROOT / f"seed{seed}/config.json"
        cache_configs[str(seed)] = {"path": str(cache_path), "sha256": file_sha256(cache_path)}
    device = torch.cuda.current_device()
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        ).strip(),
        "code_file_sha256": {name: file_sha256(CODE_ROOT / name) for name in code_files},
        "input_sha256": {
            "study_manifest": file_sha256(STUDY_PATH),
            "matching_protocol": file_sha256(MATCHING_PROTOCOL_PATH),
            "matched_control_manifest": file_sha256(CONTROL_PATH),
            "target_matching_summary": file_sha256(TARGET_PATH),
            "rpc1_effect_master": file_sha256(RPC1_MASTER),
            "rpc1_protocol": file_sha256(CODE_ROOT / "rpc_effect_screen_protocol_v1.json"),
            "clong_checkpoint": file_sha256(CLONG_CHECKPOINT),
            "spatial_metadata": file_sha256(SPATIAL_ROOT / "train_metadata.csv"),
            "spatial_cache_config": file_sha256(SPATIAL_ROOT / "cache_config.json"),
        },
        "sae_checkpoints": checkpoints,
        "analysis_cache_configs": cache_configs,
        "runtime": {
            "image_batch": int(args.image_batch),
            "feature_block": int(args.feature_block),
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
    }


def main() -> None:
    """Gate A通过后执行去重五档计算并最后原子发布完成config。"""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    study, controls = verify_frozen_inputs()
    spatial, images, patients, _metadata = load_inputs()
    rpc1 = pd.read_csv(RPC1_MASTER, dtype={"anchor_id": str})
    device = torch.device("cuda")
    model = load_clong_model(device)
    prepare_output_root(OUTPUT_ROOT)
    start_provenance = provenance(args)
    (OUTPUT_ROOT / "run_provenance_start.json").write_text(
        json.dumps(start_provenance, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    gate = alpha_zero_gate(
        study, spatial, images, patients, model, device,
        args.image_batch, args.feature_block, OUTPUT_ROOT / "gate_a_alpha0",
    )
    if gate["status"] != "rpc2_formal_core_alpha0_gate_passed":
        raise RuntimeError("Gate A未通过，禁止启动controls")

    dose_by_seed: dict[int, pd.DataFrame] = {}
    fixed_by_seed: dict[int, pd.DataFrame] = {}
    source_output: list[dict[str, object]] = []
    facts_root = OUTPUT_ROOT / "unique_feature_facts"
    for seed in SEEDS:
        universe = unique_feature_universe(study, controls, seed)
        target_ids = set(study[f"feature_{seed}"].astype(int))
        seed_root = facts_root / f"seed{seed}"
        capture = seed_root / "target_patient_effects"
        dose, fixed = evaluate_probe_seed(
            seed, universe, target_ids, spatial, images, patients, model, device,
            args.image_batch, args.feature_block, alphas=ALPHAS,
            capture_target_dir=capture,
        )
        seed_root.mkdir(parents=True, exist_ok=True)
        np.save(seed_root / "unique_feature_ids.npy", universe)
        dose.to_csv(seed_root / "unique_feature_dose_metrics.csv", index=False)
        fixed.to_csv(seed_root / "study_fixed_attention_metrics.csv", index=False)
        dose_by_seed[seed] = dose
        fixed_by_seed[seed] = fixed
        source_output.extend(source_rows(seed, study, patients, capture))

    evidence = build_target_evidence(study, controls, rpc1, dose_by_seed, fixed_by_seed)
    master = build_anchor_master(evidence)
    evidence.to_csv(OUTPUT_ROOT / "rpc2_target_seed_evidence.csv", index=False)
    master.to_csv(OUTPUT_ROOT / "rpc2_anchor_evidence_master.csv", index=False)
    pd.DataFrame(source_output).to_csv(OUTPUT_ROOT / "rpc2_source_descriptive.csv", index=False)

    role_counts = {column: int(master[column].sum()) for column in ROLE_COLUMNS}
    expected_counts = {
        "primary_candidate": 122,
        "low_effect_control": 20,
        "post_rpc1_secondary_exploratory": 8,
    }
    if role_counts != expected_counts or len(master) != 149:
        raise RuntimeError(f"正式summary角色计数错误: {role_counts}")
    overlap = master.loc[
        master.primary_candidate & master.low_effect_control, "anchor_id"
    ].tolist()
    if overlap != ["a00987"]:
        raise RuntimeError(f"多角色Anchor错误: {overlap}")

    end_provenance = provenance(args)
    if end_provenance != start_provenance:
        raise RuntimeError("正式运行期间Git/code/input/environment provenance发生变化")

    output_sha = {
        "target_seed_evidence": file_sha256(OUTPUT_ROOT / "rpc2_target_seed_evidence.csv"),
        "anchor_evidence_master": file_sha256(OUTPUT_ROOT / "rpc2_anchor_evidence_master.csv"),
        "source_descriptive": file_sha256(OUTPUT_ROOT / "rpc2_source_descriptive.csv"),
        "gate_a_config": file_sha256(OUTPUT_ROOT / "gate_a_alpha0/config.json"),
        "run_provenance_start": file_sha256(OUTPUT_ROOT / "run_provenance_start.json"),
    }
    for seed in SEEDS:
        seed_root = facts_root / f"seed{seed}"
        output_sha[f"seed{seed}_unique_feature_ids"] = file_sha256(
            seed_root / "unique_feature_ids.npy"
        )
        output_sha[f"seed{seed}_unique_feature_dose_metrics"] = file_sha256(
            seed_root / "unique_feature_dose_metrics.csv"
        )
        output_sha[f"seed{seed}_study_fixed_attention_metrics"] = file_sha256(
            seed_root / "study_fixed_attention_metrics.csv"
        )

    payload = {
        "status": "rpc2_formal_complete",
        "scientific_summary": True,
        "confirmatory_significance_test": False,
        "stage_level_pass_fail": False,
        "unique_study_objects": 149,
        "target_seed_rows": 447,
        "matched_control_mapping_rows": 44424,
        "role_flag_counts": role_counts,
        "known_overlap": overlap,
        "gate_a": gate,
        "provenance": start_provenance,
        "output_sha256": output_sha,
        "train_only": True,
        "val_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
    }
    temporary = OUTPUT_ROOT / "config.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    os.replace(temporary, OUTPUT_ROOT / "config.json")
    print(f"RP-C2 formal完成: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
