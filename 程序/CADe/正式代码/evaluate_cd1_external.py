#!/usr/bin/env python3
"""将冻结CD1 RGB/Gray检测器只读投影到External Development。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINING_CODE = PROJECT_ROOT / "程序/模型训练/正式代码"
sys.path.insert(0, str(TRAINING_CODE))

from run_y1_yolo26_smoke import assert_locked_library_behavior, file_sha256  # noqa: E402
from train_utils import git_snapshot  # noqa: E402
from cade_cd0_metrics import box_iou, lesion_size_groups  # noqa: E402
from cade_cd1_metrics import detection_errors  # noqa: E402
from evaluate_cade_cd0 import load_external_cohort  # noqa: E402
from evaluate_cd1_val import (  # noqa: E402
    BOOTSTRAP_REPEATS,
    SEEDS,
    evaluate_pair,
    image_detection_sensitivity,
    predict_all_boxes,
    validate_rgb_reproduction,
)


CD1_VAL_DECISION = PROJECT_ROOT / (
    "结果/CADe/CD1_Gray_Qualification_20260825/"
    "Development_Val正式评价/CD1_VAL_DECISION.json"
)
CD0_ROOT = PROJECT_ROOT / "结果/CADe/CD0错误地图_20260825"
CD0_SUMMARY = CD0_ROOT / "cd0_summary.json"
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "结果/CADe/CD1_Gray_Qualification_20260825/External_Development只读投影"
)


def parse_args() -> argparse.Namespace:
    """解析CUDA设备、输出目录和只读预检/debug选项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_frozen_val_decision() -> dict:
    """读取内部冻结决策，并拒绝未完成或已越过安全边界的产品。"""
    decision = json.loads(CD1_VAL_DECISION.read_text(encoding="utf-8"))
    if (
        decision["stage"] != "CD1_VAL"
        or decision["debug"]
        or not decision["qualification_evaluated"]
        or not decision["deployment_thresholds_frozen_for_formal_use"]
        or decision["decision"]["decision"] != "INCONCLUSIVE"
        or decision["external_read"]
        or decision["internal_temporal_test_read"]
    ):
        raise ValueError("CD1 Val冻结决策状态不符合External只读投影要求")
    for seed in SEEDS:
        for modality in ("rgb", "gray"):
            product = decision["products"][str(seed)][modality]
            checkpoint = Path(product["checkpoint"])
            if file_sha256(checkpoint) != product["checkpoint_sha256"]:
                raise ValueError(f"seed{seed} {modality} checkpoint SHA不一致")
    return decision


def build_gray_external_view(
    images: pd.DataFrame,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """确定性生成三通道Gray PNG，并返回同键Gray队列与实际图像尺寸。"""
    view = output / "gray_external_view"
    view.mkdir(parents=True)
    gray = images.copy()
    gray_paths = []
    dimensions = []
    for index, row in enumerate(images.itertuples(index=False)):
        source = (PROJECT_ROOT / str(row.image_path)).resolve()
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"无法读取External图像: {source}")
        height, width = image.shape[:2]
        luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        three_channel = cv2.cvtColor(luma, cv2.COLOR_GRAY2BGR)
        target = view / f"{index:06d}.png"
        if not cv2.imwrite(str(target), three_channel):
            raise RuntimeError(f"无法写入External Gray图像: {target}")
        gray_paths.append(str(target))
        dimensions.append({
            "image_key": str(row.image_key),
            "processed_width": int(width),
            "processed_height": int(height),
        })
    gray["image_path"] = gray_paths
    return gray, pd.DataFrame(dimensions), view


def image_trigger_summary(
    images: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold: float,
) -> dict:
    """报告冻结阈值下癌图和非癌图至少触发一个框的比例。"""
    top_scores = predictions.groupby("image_key").confidence.max()
    values = images.image_key.map(top_scores).fillna(0.0)
    return {
        "all_images": float(np.mean(values >= threshold)),
        "cancer_images": float(np.mean(values[images.label.eq(1)] >= threshold)),
        "noncancer_images": float(np.mean(values[images.label.eq(0)] >= threshold)),
    }


def gt_candidate_evidence(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[tuple[str, int], tuple[float, float]]:
    """为每个GT保存全部候选中最佳IoU及该候选置信度。"""
    evidence = {}
    prediction_groups = {
        str(key): group for key, group in predictions.groupby("image_key", sort=False)
    }
    for row in ground_truth.itertuples(index=False):
        key = (str(row.image_key), int(row.gt_index))
        candidates = prediction_groups.get(key[0])
        if candidates is None or candidates.empty:
            evidence[key] = (0.0, 0.0)
            continue
        ious = box_iou(
            np.array([row.x1, row.y1, row.x2, row.y2], dtype=float),
            candidates[["x1", "y1", "x2", "y2"]].to_numpy(float),
        )
        best = int(np.argmax(ious))
        evidence[key] = (
            float(ious[best]), float(candidates.iloc[best].confidence),
        )
    return evidence


def build_seed_quadrants(
    seed: int,
    images: pd.DataFrame,
    ground_truth: pd.DataFrame,
    rgb_predictions: pd.DataFrame,
    gray_predictions: pd.DataFrame,
    rgb_threshold: float,
    gray_threshold: float,
    lesion_size_bounds: tuple[float, float],
    operating_point: str,
) -> pd.DataFrame:
    """构建一个seed在指定工作点的逐GT四象限明细。"""
    rgb_detected, _, _ = detection_errors(rgb_predictions, ground_truth, rgb_threshold)
    gray_detected, _, _ = detection_errors(gray_predictions, ground_truth, gray_threshold)
    rgb_evidence = gt_candidate_evidence(ground_truth, rgb_predictions)
    gray_evidence = gt_candidate_evidence(ground_truth, gray_predictions)
    metadata_columns = [
        column for column in (
            "image_key", "patient_id", "source", "center", "size_group",
            "original_width", "original_height", "processed_width", "processed_height",
        ) if column in images
    ]
    metadata = images[metadata_columns].drop_duplicates("image_key")
    table = ground_truth.merge(metadata, on="image_key", how="left", validate="many_to_one")
    table["lesion_size_group"] = lesion_size_groups(
        table.bbox_area_fraction, lesion_size_bounds
    ).astype(str)
    records = []
    for row in table.itertuples(index=False):
        key = (str(row.image_key), int(row.gt_index))
        rgb_hit = key in rgb_detected
        gray_hit = key in gray_detected
        if rgb_hit and gray_hit:
            quadrant = "both_detected"
        elif rgb_hit:
            quadrant = "rgb_only"
        elif gray_hit:
            quadrant = "gray_only_rescue"
        else:
            quadrant = "neither"
        record = row._asdict()
        record.update({
            "seed": seed,
            "operating_point": operating_point,
            "rgb_threshold": float(rgb_threshold),
            "gray_threshold": float(gray_threshold),
            "rgb_detected": rgb_hit,
            "gray_detected": gray_hit,
            "quadrant": quadrant,
            "gray_rescue_rgb_fn": bool(not rgb_hit and gray_hit),
            "rgb_best_iou": rgb_evidence[key][0],
            "rgb_best_iou_candidate_confidence": rgb_evidence[key][1],
            "gray_best_iou": gray_evidence[key][0],
            "gray_best_iou_candidate_confidence": gray_evidence[key][1],
        })
        records.append(record)
    return pd.DataFrame(records)


def summarize_rescue_stability(quadrants: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """按同一GT汇总0/3至3/3 Gray rescue及每seed四象限。"""
    keys = ["image_key", "gt_index"]
    identity = [
        "patient_id", "bbox_area_fraction", "lesion_size_group", "source", "center",
        "size_group", "original_width", "original_height",
        "processed_width", "processed_height",
    ]
    identity = [column for column in identity if column in quadrants]
    base = quadrants[keys + identity].drop_duplicates(keys).set_index(keys)
    rescue = quadrants.pivot(index=keys, columns="seed", values="gray_rescue_rgb_fn")
    quadrant = quadrants.pivot(index=keys, columns="seed", values="quadrant")
    rescue.columns = [f"seed{int(seed)}_gray_rescue_rgb_fn" for seed in rescue.columns]
    quadrant.columns = [f"seed{int(seed)}_quadrant" for seed in quadrant.columns]
    output = base.join(rescue).join(quadrant).reset_index()
    rescue_columns = [column for column in output if column.endswith("gray_rescue_rgb_fn")]
    output["gray_rescue_seed_count"] = output[rescue_columns].sum(axis=1).astype(int)
    output["gray_rescue_stability"] = output.gray_rescue_seed_count.map(
        lambda count: f"{count}/3"
    )
    counts = output.gray_rescue_seed_count.value_counts().reindex(range(4), fill_value=0)
    summary = {f"{count}/3": int(counts[count]) for count in range(4)}
    summary["at_least_one_seed"] = int((output.gray_rescue_seed_count >= 1).sum())
    summary["stable_3_of_3"] = int((output.gray_rescue_seed_count == 3).sum())
    return output, summary


def validate_rgb_external_reproduction(seed: int, actual: dict, expected: dict) -> None:
    """确认复用的RGB预测仍逐项复现CD0 External正式记录。"""
    validate_rgb_reproduction(seed, actual, expected)
    for metric in ("ap50", "map50_95"):
        if not math.isclose(
            float(actual[metric]), float(expected[metric]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"seed{seed} RGB External {metric}未复现CD0")


def main() -> None:
    """执行External只读预检、debug或唯一一次正式描述性投影。"""
    args = parse_args()
    decision = load_frozen_val_decision()
    cd0 = json.loads(CD0_SUMMARY.read_text(encoding="utf-8"))
    if cd0["stage"] != "CD0" or cd0["debug"] or cd0["internal_temporal_test_read"]:
        raise ValueError("CD0 External正式参照状态异常")
    images, ground_truth = load_external_cohort(args.debug)
    lesion_size_bounds = (
        float(decision["lesion_size_definition"]["bbox_area_q33"]),
        float(decision["lesion_size_definition"]["bbox_area_q67"]),
    )
    if args.preflight_only:
        print(
            f"CD1 External预检通过: {len(images)}张/{images.patient_id.nunique()}人/"
            f"{len(ground_truth)}病灶; internal_decision=INCONCLUSIVE; "
            "second_qualification=False; internal_temporal_test_read=False"
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，拒绝CD1 External Gray推理")
    output = args.output.with_name(args.output.name + ("_debug" if args.debug else ""))
    if output.exists():
        raise FileExistsError(f"CD1 External输出目录已存在，拒绝覆盖: {output}")
    output.mkdir(parents=True)
    gray_images, dimensions, gray_source = build_gray_external_view(images, output)
    images = images.merge(dimensions, on="image_key", how="left", validate="one_to_one")
    gray_images = gray_images.merge(dimensions, on="image_key", how="left", validate="one_to_one")

    seed_results = []
    quadrant_tables = {"primary": [], "val_frozen_deployment": []}
    product_records = {}
    for seed in SEEDS:
        products = decision["products"][str(seed)]
        rgb_predictions = pd.read_csv(
            CD0_ROOT / f"seed{seed}_external_development_all_predictions.csv"
        )
        if args.debug:
            rgb_predictions = rgb_predictions[
                rgb_predictions.image_key.isin(images.image_key)
            ].reset_index(drop=True)
        gray_model = YOLO(products["gray"]["checkpoint"])
        gray_behavior = assert_locked_library_behavior(gray_model)
        gray_predictions = predict_all_boxes(
            gray_model, gray_images, gray_source,
            int(products["gray"]["imgsz"]), args.device,
        )
        rgb_threshold = float(products["rgb"]["deployment_threshold"])
        gray_threshold = float(products["gray"]["deployment_threshold"])
        rgb_predictions.to_csv(output / f"seed{seed}_rgb_predictions.csv", index=False)
        gray_predictions.to_csv(output / f"seed{seed}_gray_predictions.csv", index=False)
        result = evaluate_pair(
            seed, images, ground_truth, rgb_predictions, gray_predictions,
            rgb_threshold, gray_threshold, lesion_size_bounds, output, args.debug,
        )
        result["deployment_image_trigger_rate"] = {
            "rgb": image_trigger_summary(images, rgb_predictions, rgb_threshold),
            "gray": image_trigger_summary(images, gray_predictions, gray_threshold),
        }
        if not args.debug:
            validate_rgb_external_reproduction(
                seed, result["rgb"], cd0["seeds"][str(seed)]["external_development"]
            )
        rgb_primary = float(
            result["rgb"]["primary_froc_at_most_0p5_fp_per_image"]["threshold"]
        )
        gray_primary = float(
            result["gray"]["primary_froc_at_most_0p5_fp_per_image"]["threshold"]
        )
        quadrant_tables["primary"].append(build_seed_quadrants(
            seed, images, ground_truth, rgb_predictions, gray_predictions,
            rgb_primary, gray_primary, lesion_size_bounds, "external_primary",
        ))
        quadrant_tables["val_frozen_deployment"].append(build_seed_quadrants(
            seed, images, ground_truth, rgb_predictions, gray_predictions,
            rgb_threshold, gray_threshold, lesion_size_bounds, "val_frozen_deployment",
        ))
        seed_results.append(result)
        product_records[str(seed)] = {
            "rgb": products["rgb"],
            "gray": products["gray"],
            "gray_model_behavior": gray_behavior,
        }
        print(
            f"seed{seed}: RGB Primary={result['rgb']['primary_froc_at_most_0p5_fp_per_image']['lesion_sensitivity']:.4f}; "
            f"Gray Primary={result['gray']['primary_froc_at_most_0p5_fp_per_image']['lesion_sensitivity']:.4f}; "
            f"Gray rescue={result['gray_rescue']:.4f}; FN Jaccard={result['fn_jaccard']:.4f}"
        )
        del gray_model
        torch.cuda.empty_cache()

    stability = {}
    for name, tables in quadrant_tables.items():
        long_table = pd.concat(tables, ignore_index=True)
        stable_table, summary = summarize_rescue_stability(long_table)
        long_table.to_csv(
            output / f"external_{name}_gt_quadrants_long.csv",
            index=False, encoding="utf-8-sig",
        )
        stable_table.to_csv(
            output / f"external_{name}_gt_rescue_stability.csv",
            index=False, encoding="utf-8-sig",
        )
        stability[name] = summary

    code_version = git_snapshot()
    summary = {
        "stage": "CD1_EXTERNAL_PROJECTION",
        "debug": args.debug,
        "training_performed": False,
        "qualification_evaluated": False,
        "second_pass_fail_decision": False,
        "internal_cd1_decision_unchanged": "INCONCLUSIVE",
        "external_read": True,
        "internal_temporal_test_read": False,
        "code_git_commit": code_version["git_commit"],
        "git_dirty": code_version["git_dirty"],
        "evaluation_script_sha256": file_sha256(Path(__file__)),
        "cd1_val_decision": str(CD1_VAL_DECISION),
        "cd1_val_decision_sha256": file_sha256(CD1_VAL_DECISION),
        "rgb_cd0_reference": str(CD0_SUMMARY),
        "rgb_cd0_reference_sha256": file_sha256(CD0_SUMMARY),
        "images": int(len(images)),
        "patients": int(images.patient_id.nunique()),
        "lesions": int(len(ground_truth)),
        "matching_iou_threshold": 0.30,
        "primary_fp_per_image_limit": 0.5,
        "bootstrap_repetitions": 0 if args.debug else BOOTSTRAP_REPEATS,
        "lesion_size_definition": decision["lesion_size_definition"],
        "products": product_records,
        "seed_results": seed_results,
        "gray_rescue_stability": stability,
        "interpretation_boundary": (
            "External Development is descriptive only and cannot change the frozen "
            "CD1 Val INCONCLUSIVE decision."
        ),
    }
    filename = "cd1_external_debug_summary.json" if args.debug else "CD1_EXTERNAL_PROJECTION.json"
    path = output / filename
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"CD1 External {'debug' if args.debug else '正式只读投影'}完成: {path}")


if __name__ == "__main__":
    main()
