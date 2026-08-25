#!/usr/bin/env python3
"""Finalize an existing CD0 run without repeating YOLO inference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from cade_cd0_metrics import box_iou, match_predictions
from evaluate_cade_cd0 import (
    DEFAULT_OUTPUT,
    SEEDS,
    load_external_cohort,
    load_products,
    load_train_lesion_size_bounds,
    load_val_cohort,
    basic_stratified_summary,
    render_review_images,
)


def parse_args() -> argparse.Namespace:
    """解析已完成 CD0 目录，不接受模型或数据选择参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh-stratification-only", action="store_true")
    return parser.parse_args()


def trigger_metrics(
    images: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold: float,
) -> dict:
    """计算冻结部署阈值的图像/患者触发率与候选框分布。"""
    triggered_keys = set(predictions.loc[predictions.confidence.ge(threshold), "image_key"])
    image_frame = images[["image_key", "patient_id", "label"]].copy()
    image_frame["triggered"] = image_frame.image_key.isin(triggered_keys)
    patient = image_frame.groupby(["patient_id", "label"], as_index=False).triggered.max()
    noncancer_images = image_frame[image_frame.label.eq(0)]
    noncancer_patients = patient[patient.label.eq(0)]
    counts = predictions.groupby("image_key").size().reindex(images.image_key, fill_value=0)
    return {
        "noncancer_image_trigger_rate": float(noncancer_images.triggered.mean()),
        "noncancer_patient_trigger_rate": float(noncancer_patients.triggered.mean()),
        "all_image_trigger_rate": float(image_frame.triggered.mean()),
        "prediction_boxes_at_0p001": int(len(predictions)),
        "candidate_boxes_per_image_mean": float(counts.mean()),
        "confidence_quantiles": {
            str(key): float(value)
            for key, value in predictions.confidence.quantile([0, 0.25, 0.5, 0.75, 0.9, 1]).items()
        },
    }


def stable_fn_table(run_dir: Path, cohort: str, images: pd.DataFrame) -> pd.DataFrame:
    """将同一病灶在三 seed 的 FN 合并为一个复核案例。"""
    frames = []
    for seed in SEEDS:
        frame = pd.read_csv(run_dir / f"seed{seed}_{cohort}_fn.csv", encoding="utf-8-sig")
        frame["seed"] = seed
        frames.append(frame)
    errors = pd.concat(frames, ignore_index=True)
    metadata = images.set_index("image_key")
    records = []
    for image_key, group in errors.groupby("image_key", sort=False):
        row = group.iloc[0]
        seeds = sorted(group.seed.unique())
        records.append({
            "cohort": cohort,
            "error_kind": "FN",
            "image_key": image_key,
            "image_path": row.image_path,
            "patient_id": row.patient_id,
            "seed": ",".join(map(str, seeds)),
            "seed_count": len(seeds),
            "stability": "stable_3of3" if len(seeds) == 3 else "unstable_1or2",
            "confidence": math.nan,
            "gt_x1": row.x1, "gt_y1": row.y1, "gt_x2": row.x2, "gt_y2": row.y2,
            "pred_x1": math.nan, "pred_y1": math.nan,
            "pred_x2": math.nan, "pred_y2": math.nan,
            "automatic_error": row.automatic_error,
            "source": metadata.at[image_key, "source"] if "source" in metadata else "unknown",
            "size_group": metadata.at[image_key, "size_group"] if "size_group" in metadata else "unknown",
        })
    return pd.DataFrame(records)


def fp_clusters(run_dir: Path, cohort: str, images: pd.DataFrame) -> pd.DataFrame:
    """以同图 IoU>=0.30 将不同 seed 的 FP 聚成稳定或不稳定区域。"""
    frames = []
    for seed in SEEDS:
        frame = pd.read_csv(run_dir / f"seed{seed}_{cohort}_fp.csv", encoding="utf-8-sig")
        frame["seed"] = seed
        frames.append(frame)
    errors = pd.concat(frames, ignore_index=True)
    metadata = images.set_index("image_key")
    records = []
    for image_key, group in errors.groupby("image_key", sort=False):
        clusters: list[list[pd.Series]] = []
        for _, row in group.sort_values("confidence", ascending=False).iterrows():
            box = row[["x1", "y1", "x2", "y2"]].to_numpy(float)
            destination = None
            for cluster in clusters:
                if row.seed in {int(item.seed) for item in cluster}:
                    continue
                cluster_boxes = np.stack([
                    item[["x1", "y1", "x2", "y2"]].to_numpy(float) for item in cluster
                ])
                if np.any(box_iou(box, cluster_boxes) >= 0.30):
                    destination = cluster
                    break
            if destination is None:
                clusters.append([row])
            else:
                destination.append(row)
        for cluster_index, cluster in enumerate(clusters):
            representative = max(cluster, key=lambda item: float(item.confidence))
            seeds = sorted({int(item.seed) for item in cluster})
            records.append({
                "cohort": cohort,
                "error_kind": "FP",
                "image_key": image_key,
                "image_path": representative.image_path,
                "patient_id": representative.patient_id,
                "seed": ",".join(map(str, seeds)),
                "seed_count": len(seeds),
                "stability": "stable_3of3" if len(seeds) == 3 else "unstable_1or2",
                "confidence": float(representative.confidence),
                "gt_x1": math.nan, "gt_y1": math.nan,
                "gt_x2": math.nan, "gt_y2": math.nan,
                "pred_x1": representative.x1, "pred_y1": representative.y1,
                "pred_x2": representative.x2, "pred_y2": representative.y2,
                "automatic_error": representative.automatic_error,
                "cluster_index": cluster_index,
                "source": metadata.at[image_key, "source"] if "source" in metadata else "unknown",
                "size_group": metadata.at[image_key, "size_group"] if "size_group" in metadata else "unknown",
            })
    return pd.DataFrame(records)


def choose_review_cases(fn: pd.DataFrame, fp: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """冻结为全部去重FN、稳定高置信FP与分层抽样的不稳定FP。"""
    selected_fp = []
    counts = {}
    for cohort, group in fp.groupby("cohort", sort=False):
        stable = group[group.stability.eq("stable_3of3")].nlargest(100, "confidence")
        unstable_pool = group[group.stability.eq("unstable_1or2")].copy()
        unstable_pool["confidence_group"] = pd.qcut(
            unstable_pool.confidence, q=4, labels=False, duplicates="drop"
        ).astype(str)
        unstable_pool["sampling_key"] = (
            unstable_pool.source.astype(str) + "|"
            + unstable_pool.size_group.astype(str) + "|"
            + unstable_pool.confidence_group
        )
        shuffled = unstable_pool.sample(frac=1.0, random_state=20260825)
        ordered_parts = [
            group.reset_index(drop=True)
            for _, group in shuffled.groupby("sampling_key", sort=True)
        ]
        round_robin = []
        for index in range(max((len(part) for part in ordered_parts), default=0)):
            round_robin.extend(part.iloc[index] for part in ordered_parts if index < len(part))
        unstable = pd.DataFrame(round_robin).head(min(50, len(unstable_pool)))
        selected_fp.extend([stable, unstable])
        counts[cohort] = {
            "stable_fp_candidates": int(group.stability.eq("stable_3of3").sum()),
            "unstable_fp_candidates": int(group.stability.eq("unstable_1or2").sum()),
            "stable_fp_selected": int(len(stable)),
            "unstable_fp_selected": int(len(unstable)),
        }
    selected = pd.concat([fn, *selected_fp], ignore_index=True, sort=False)
    selected = selected.sort_values(
        ["cohort", "error_kind", "stability", "confidence"],
        ascending=[True, True, True, False], na_position="last",
    ).reset_index(drop=True)
    selected["case_id"] = [f"CD0-{index:04d}" for index in range(1, len(selected) + 1)]
    selected["reviewer_requirement"] = "single"
    selected["needs_expert_review"] = ""
    dual_count = int(round(len(selected) * 0.20))
    if dual_count:
        dual_indices = np.linspace(0, len(selected) - 1, dual_count, dtype=int)
        selected.loc[dual_indices, "reviewer_requirement"] = "independent_dual"
    for column in (
        "primary_error_category", "secondary_contributors", "free_note",
        "reviewer_1_primary", "reviewer_1_secondary", "reviewer_1_note",
        "reviewer_2_primary", "reviewer_2_secondary", "reviewer_2_note",
        "adjudicated_primary", "adjudicated_secondary", "adjudication_note",
    ):
        selected[column] = ""
    return selected, {
        "fn_selected_after_cross_seed_deduplication": int(len(fn)),
        "fp": counts,
        "dual_review_selected": dual_count,
        "dual_review_fraction": dual_count / len(selected),
        "random_seed": 20260825,
    }


def cross_seed_summary(summary: dict) -> tuple[pd.DataFrame, dict]:
    """对预注册的核心指标计算三 seed 均值、样本标准差和方向。"""
    rows = []
    result = {}
    for cohort in ("development_val", "external_development"):
        metric_paths = {
            "primary_sensitivity": ("primary_froc_at_most_0p5_fp_per_image", "lesion_sensitivity"),
            "primary_achieved_fp_per_image": ("primary_froc_at_most_0p5_fp_per_image", "fp_per_image"),
            "deployment_sensitivity": ("deployment", "lesion_sensitivity"),
            "deployment_fp_per_image": ("deployment", "fp_per_image"),
            "deployment_iou_ge_0p5": ("deployment", "iou_ge_0p5_all_lesions"),
            "deployment_mean_iou": ("deployment", "mean_iou_all_lesions"),
            "ap50": ("ap50",),
            "map50_95": ("map50_95",),
        }
        result[cohort] = {}
        for metric, path in metric_paths.items():
            values = []
            for seed in SEEDS:
                value = summary["seeds"][str(seed)][cohort]
                for key in path:
                    value = value[key]
                values.append(float(value))
            record = {
                "cohort": cohort, "metric": metric,
                **{f"seed{seed}": value for seed, value in zip(SEEDS, values)},
                "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
            }
            rows.append(record)
            result[cohort][metric] = record
    result["external_minus_internal"] = {}
    for metric in result["development_val"]:
        gaps = [
            result["external_development"][metric][f"seed{seed}"]
            - result["development_val"][metric][f"seed{seed}"]
            for seed in SEEDS
        ]
        result["external_minus_internal"][metric] = {
            **{f"seed{seed}": gap for seed, gap in zip(SEEDS, gaps)},
            "mean": float(np.mean(gaps)),
            "same_direction_3of3": bool(all(gap >= 0 for gap in gaps) or all(gap <= 0 for gap in gaps)),
        }
    return pd.DataFrame(rows), result


def main() -> None:
    """补齐已有 CD0 的触发率、跨seed汇总和去重人工复核包。"""
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "cd0_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("debug") or summary.get("internal_temporal_test_read"):
        raise ValueError("只允许终结未读内部时间测试的正式CD0")
    products = load_products()
    val_images, _ = load_val_cohort(False)
    external_images, _ = load_external_cohort(False)
    cohorts = {
        "development_val": val_images,
        "external_development": external_images,
    }
    lesion_size_bounds = load_train_lesion_size_bounds()
    if args.refresh_stratification_only:
        ground_truths = {
            "development_val": load_val_cohort(False)[1],
            "external_development": load_external_cohort(False)[1],
        }
        for cohort, images in cohorts.items():
            ground_truth = ground_truths[cohort]
            for seed in SEEDS:
                predictions = pd.read_csv(
                    run_dir / f"seed{seed}_{cohort}_all_predictions.csv"
                )
                threshold = products[seed]["deployment_threshold"]
                result = summary["seeds"][str(seed)][cohort]
                result["stratified"] = basic_stratified_summary(
                    images, predictions, ground_truth, threshold, lesion_size_bounds
                )
                result["lesion_size_definition"] = {
                    "source": "development_train",
                    "bbox_area_q33": lesion_size_bounds[0],
                    "bbox_area_q67": lesion_size_bounds[1],
                }
        summary["lesion_size_definition"] = {
            "source": "development_train",
            "bbox_area_q33": lesion_size_bounds[0],
            "bbox_area_q67": lesion_size_bounds[1],
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        review_path = run_dir / "CD0人工复核正式版/CD0人工复核去重清单.csv"
        review = pd.read_csv(review_path, encoding="utf-8-sig")
        review["needs_expert_review"] = ""
        review.to_csv(review_path, index=False, encoding="utf-8-sig")
        print(
            f"CD0分层已刷新: development_train q33={lesion_size_bounds[0]:.8f}, "
            f"q67={lesion_size_bounds[1]:.8f}; YOLO推理未重跑"
        )
        return
    all_fn, all_fp = [], []
    for cohort, images in cohorts.items():
        all_fn.append(stable_fn_table(run_dir, cohort, images))
        all_fp.append(fp_clusters(run_dir, cohort, images))
        for seed in SEEDS:
            predictions = pd.read_csv(run_dir / f"seed{seed}_{cohort}_all_predictions.csv")
            summary["seeds"][str(seed)][cohort]["trigger_and_distribution"] = trigger_metrics(
                images, predictions, products[seed]["deployment_threshold"]
            )
    fn = pd.concat(all_fn, ignore_index=True)
    fp = pd.concat(all_fp, ignore_index=True)
    review, review_config = choose_review_cases(fn, fp)
    review_dir = run_dir / "CD0人工复核正式版"
    review_dir.mkdir()
    review.to_csv(review_dir / "CD0人工复核去重清单.csv", index=False, encoding="utf-8-sig")
    render_review_images(review, review_dir / "两联图")
    (review_dir / "review_selection_config.json").write_text(
        json.dumps(review_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cross_frame, cross_json = cross_seed_summary(summary)
    cross_frame.to_csv(run_dir / "cd0_cross_seed_summary.csv", index=False, encoding="utf-8-sig")
    summary["cross_seed_summary"] = cross_json
    summary["formal_review_package"] = review_config
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(
        f"CD0终结完成: 去重FN={len(fn)}, FP聚类={len(fp)}, "
        f"复核案例={len(review)}, 双人={review_config['dual_review_selected']}"
    )


if __name__ == "__main__":
    main()
