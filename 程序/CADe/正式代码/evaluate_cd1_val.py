#!/usr/bin/env python3
"""在Development Val上配对评价三组RGB与Gray检测器并冻结CD1决策。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINING_CODE = PROJECT_ROOT / "程序/模型训练/正式代码"
sys.path.insert(0, str(TRAINING_CODE))

from evaluate_y2_yolo26 import (  # noqa: E402
    TARGET_SENSITIVITY,
    lock_recall_threshold,
)
from run_y1_yolo26_smoke import assert_locked_library_behavior, file_sha256  # noqa: E402
from train_utils import git_snapshot  # noqa: E402
from cade_cd0_metrics import summarize_detection  # noqa: E402
from cade_cd1_metrics import (  # noqa: E402
    classify_cd1,
    detection_errors,
    fn_complementarity,
    fp_overlap,
    paired_patient_bootstrap,
)
from evaluate_cade_cd0 import (  # noqa: E402
    basic_stratified_summary,
    load_products,
    load_train_lesion_size_bounds,
    load_val_cohort,
    predict_all_boxes,
)


SEEDS = (42, 202, 503)
GRAY_ROOT = PROJECT_ROOT / "数据整理记录/CADe_CD1_Gray_20260825"
GRAY_RUN_ROOT = PROJECT_ROOT / "结果/CADe/CD1_Gray_Qualification_20260825/训练"
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "结果/CADe/CD1_Gray_Qualification_20260825/Development_Val正式评价"
)
CD0_SUMMARY = PROJECT_ROOT / "结果/CADe/CD0错误地图_20260825/cd0_summary.json"
BOOTSTRAP_REPEATS = 5000


def parse_args() -> argparse.Namespace:
    """解析显卡、输出目录、预检和独立debug选项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_gray_products() -> dict[int, dict]:
    """读取三组正式Gray训练产品并核对其数据和安全边界。"""
    products = {}
    commits = set()
    manifests = set()
    for seed in SEEDS:
        run = GRAY_RUN_ROOT / f"cd1_gray_yolo26s_640_seed{seed}"
        config_path = run / "cd1_gray_train_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        checkpoint = run / "weights/best.pt"
        if config["stage"] != "CD1" or config["role"] != "gray_detector":
            raise ValueError(f"Gray产品角色异常: seed{seed}")
        if int(config["seed"]) != seed or config["debug"]:
            raise ValueError(f"Gray产品seed/debug异常: seed{seed}")
        if any([
            config["qualification_evaluated"], config["deployment_threshold_frozen"],
            config["external_read"], config["internal_temporal_test_read"],
        ]):
            raise ValueError(f"Gray训练产品越过CD1训练边界: seed{seed}")
        if file_sha256(checkpoint) != config["products"]["best_pt_sha256"]:
            raise ValueError(f"Gray checkpoint SHA不一致: seed{seed}")
        commits.add(config["code_git_commit"])
        manifests.add(config["gray_data"]["manifest_sha256"])
        products[seed] = {
            "checkpoint": checkpoint,
            "checkpoint_sha256": config["products"]["best_pt_sha256"],
            "train_config": config_path,
            "train_config_sha256": file_sha256(config_path),
            "code_git_commit": config["code_git_commit"],
            "git_dirty": bool(config["git_dirty"]),
            "imgsz": int(config["training_protocol"]["imgsz"]),
        }
    if len(commits) != 1 or len(manifests) != 1:
        raise ValueError("三组Gray产品的代码版本或manifest不一致")
    return products


def build_gray_val_frame(images: pd.DataFrame) -> pd.DataFrame:
    """把原始val元数据一对一映射到Gray PNG，同时保留共同image_key。"""
    manifest = pd.read_csv(GRAY_ROOT / "gray_manifest.csv", encoding="utf-8-sig")
    gray = manifest[manifest.split.eq("val")].copy()
    if len(gray) != 497 or gray.patient_id.nunique() != 260:
        raise ValueError(f"Gray val规模异常: {len(gray)}张/{gray.patient_id.nunique()}人")
    mapped = images.merge(
        gray[["source_relpath", "gray_relpath", "patient_id", "label"]],
        left_on="image_key", right_on="source_relpath", how="left",
        validate="one_to_one", suffixes=("", "_gray"),
    )
    if mapped.gray_relpath.isna().any():
        raise ValueError("Development Val无法完整映射到Gray PNG")
    if not mapped.patient_id.astype(str).eq(mapped.patient_id_gray.astype(str)).all():
        raise ValueError("Gray val患者映射不一致")
    if not mapped.label.eq(mapped.label_gray).all():
        raise ValueError("Gray val标签映射不一致")
    mapped["image_path"] = mapped.gray_relpath.map(lambda value: str(GRAY_ROOT / value))
    return mapped[images.columns]


def build_inference_view(images: pd.DataFrame, output: Path) -> Path:
    """为当前精确队列建立稳定编号软链接，避免目录推理越过清单。"""
    output.mkdir(parents=True)
    for index, row in images.reset_index(drop=True).iterrows():
        source = (PROJECT_ROOT / str(row.image_path)).resolve()
        target = output / f"{index:06d}{source.suffix.lower() or '.jpg'}"
        target.symlink_to(source)
    return output


def lock_image_recall_threshold(
    images: pd.DataFrame,
    predictions: pd.DataFrame,
    target: float = TARGET_SENSITIVITY,
) -> float:
    """按癌图最高候选置信度冻结达到目标图像召回率的最高实际阈值。"""
    top_scores = predictions.groupby("image_key").confidence.max()
    cancer_keys = images.loc[images.label.eq(1), "image_key"]
    values = cancer_keys.map(top_scores).fillna(0.0).to_numpy(float)
    threshold = lock_recall_threshold(values, target)
    if threshold is None:
        raise RuntimeError(f"Gray检测器无法达到冻结图像召回目标{target:.2f}")
    return float(threshold)


def image_detection_sensitivity(
    images: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold: float,
) -> float:
    """计算给定阈值下癌图至少出现一个候选框的比例。"""
    top_scores = predictions.groupby("image_key").confidence.max()
    cancer_keys = images.loc[images.label.eq(1), "image_key"]
    values = cancer_keys.map(top_scores).fillna(0.0).to_numpy(float)
    return float(np.mean(values >= threshold))


def fp_overlap_by_image_label(
    images: pd.DataFrame,
    rgb_predictions: pd.DataFrame,
    gray_predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    rgb_threshold: float,
    gray_threshold: float,
) -> dict:
    """分别汇总非癌图主FP重合和癌图额外FP重合。"""
    _, _, rgb_fp = detection_errors(rgb_predictions, ground_truth, rgb_threshold)
    _, _, gray_fp = detection_errors(gray_predictions, ground_truth, gray_threshold)
    output = {}
    for name, label in (("negative_images", 0), ("positive_images_auxiliary", 1)):
        keys = set(images.loc[images.label.eq(label), "image_key"].astype(str))
        output[name] = fp_overlap(
            rgb_fp[rgb_fp.image_key.astype(str).isin(keys)],
            gray_fp[gray_fp.image_key.astype(str).isin(keys)],
        )
    return output


def bootstrap_interval(frame: pd.DataFrame) -> dict:
    """提取配对Sensitivity差及两模型FP/image的百分位区间。"""
    quantiles = frame[[
        "sensitivity_delta", "rgb_fp_per_image", "gray_fp_per_image"
    ]].quantile([0.025, 0.975])
    return {
        column: [float(quantiles.loc[0.025, column]), float(quantiles.loc[0.975, column])]
        for column in quantiles.columns
    }


def validate_rgb_reproduction(seed: int, actual: dict, expected: dict) -> None:
    """确认新入口重跑的RGB主指标逐项复现既有CD0正式记录。"""
    paths = (
        ("primary_froc_at_most_0p5_fp_per_image", "lesion_sensitivity"),
        ("primary_froc_at_most_0p5_fp_per_image", "fp_per_image"),
        ("deployment", "lesion_sensitivity"),
        ("deployment", "fp_per_image"),
    )
    mismatches = {}
    for section, metric in paths:
        observed = float(actual[section][metric])
        reference = float(expected[section][metric])
        if not math.isclose(observed, reference, rel_tol=0.0, abs_tol=1e-12):
            mismatches[f"{section}.{metric}"] = {
                "expected": reference, "actual": observed,
            }
    if mismatches:
        raise RuntimeError(f"seed{seed} RGB未复现CD0正式记录: {mismatches}")


def evaluate_pair(
    seed: int,
    images: pd.DataFrame,
    ground_truth: pd.DataFrame,
    rgb_predictions: pd.DataFrame,
    gray_predictions: pd.DataFrame,
    rgb_threshold: float,
    gray_threshold: float,
    lesion_size_bounds: tuple[float, float],
    output: Path,
    debug: bool,
) -> dict:
    """计算一个seed在Primary与部署点的完整RGB-Gray配对指标。"""
    rgb_summary, rgb_curve, _ = summarize_detection(
        images, rgb_predictions, ground_truth, rgb_threshold
    )
    gray_summary, gray_curve, _ = summarize_detection(
        images, gray_predictions, ground_truth, gray_threshold
    )
    rgb_primary = float(rgb_summary["primary_froc_at_most_0p5_fp_per_image"]["threshold"])
    gray_primary = float(gray_summary["primary_froc_at_most_0p5_fp_per_image"]["threshold"])

    primary_complementarity = fn_complementarity(
        rgb_predictions, gray_predictions, ground_truth, rgb_primary, gray_primary
    )
    deployment_complementarity = fn_complementarity(
        rgb_predictions, gray_predictions, ground_truth, rgb_threshold, gray_threshold
    )
    primary_fp_overlap = fp_overlap_by_image_label(
        images, rgb_predictions, gray_predictions, ground_truth, rgb_primary, gray_primary
    )
    deployment_fp_overlap = fp_overlap_by_image_label(
        images, rgb_predictions, gray_predictions, ground_truth, rgb_threshold, gray_threshold
    )

    primary_bootstrap_interval = None
    deployment_bootstrap_interval = None
    if not debug:
        primary_bootstrap = paired_patient_bootstrap(
            images, ground_truth, rgb_predictions, gray_predictions,
            rgb_primary, gray_primary, repeats=BOOTSTRAP_REPEATS, seed=20260825 + seed,
        )
        deployment_bootstrap = paired_patient_bootstrap(
            images, ground_truth, rgb_predictions, gray_predictions,
            rgb_threshold, gray_threshold, repeats=BOOTSTRAP_REPEATS, seed=20260826 + seed,
        )
        primary_bootstrap.to_csv(output / f"seed{seed}_primary_bootstrap.csv", index=False)
        deployment_bootstrap.to_csv(output / f"seed{seed}_deployment_bootstrap.csv", index=False)
        primary_bootstrap_interval = bootstrap_interval(primary_bootstrap)
        deployment_bootstrap_interval = bootstrap_interval(deployment_bootstrap)
    rgb_curve.to_csv(output / f"seed{seed}_rgb_froc.csv", index=False)
    gray_curve.to_csv(output / f"seed{seed}_gray_froc.csv", index=False)

    rgb_summary["stratified"] = basic_stratified_summary(
        images, rgb_predictions, ground_truth, rgb_threshold, lesion_size_bounds
    )
    gray_summary["stratified"] = basic_stratified_summary(
        images, gray_predictions, ground_truth, gray_threshold, lesion_size_bounds
    )
    primary_delta = float(
        gray_summary["primary_froc_at_most_0p5_fp_per_image"]["lesion_sensitivity"]
        - rgb_summary["primary_froc_at_most_0p5_fp_per_image"]["lesion_sensitivity"]
    )
    return {
        "seed": seed,
        "rgb": rgb_summary,
        "gray": gray_summary,
        "primary_delta": primary_delta,
        "gray_rescue": float(primary_complementarity["gray_rescue"]),
        "fn_jaccard": float(primary_complementarity["fn_jaccard"]),
        "primary_complementarity": primary_complementarity,
        "deployment_complementarity": deployment_complementarity,
        "primary_fp_overlap": primary_fp_overlap,
        "deployment_fp_overlap": deployment_fp_overlap,
        "primary_bootstrap": primary_bootstrap_interval,
        "deployment_bootstrap": deployment_bootstrap_interval,
    }


def main() -> None:
    """执行预检、独立debug或唯一一次正式Development Val评价。"""
    args = parse_args()
    rgb_products = load_products()
    gray_products = load_gray_products()
    code_version = git_snapshot()
    cd0 = json.loads(CD0_SUMMARY.read_text(encoding="utf-8"))
    if cd0["stage"] != "CD0" or cd0["debug"] or cd0["internal_temporal_test_read"]:
        raise ValueError("CD0 RGB正式参照状态异常")
    images, ground_truth = load_val_cohort(args.debug)
    gray_images = build_gray_val_frame(images)
    lesion_size_bounds = load_train_lesion_size_bounds()
    if args.preflight_only:
        print(
            f"CD1 val预检通过: {len(images)}张/{len(ground_truth)}病灶; "
            f"seeds={list(SEEDS)}; external_read=False; internal_temporal_test_read=False"
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，拒绝CD1 val推理")
    output = args.output.with_name(args.output.name + ("_debug" if args.debug else ""))
    if output.exists():
        raise FileExistsError(f"CD1 val输出目录已存在，拒绝覆盖: {output}")
    output.mkdir(parents=True)
    rgb_source = build_inference_view(images, output / "rgb_val_view")
    gray_source = build_inference_view(gray_images, output / "gray_val_view")

    seed_results = []
    products = {}
    for seed in SEEDS:
        rgb_product = rgb_products[seed]
        gray_product = gray_products[seed]
        rgb_model = YOLO(str(rgb_product["checkpoint"]))
        gray_model = YOLO(str(gray_product["checkpoint"]))
        rgb_behavior = assert_locked_library_behavior(rgb_model)
        gray_behavior = assert_locked_library_behavior(gray_model)
        rgb_predictions = predict_all_boxes(
            rgb_model, images, rgb_source,
            rgb_product["imgsz"], args.device,
        )
        gray_predictions = predict_all_boxes(
            gray_model, gray_images, gray_source,
            gray_product["imgsz"], args.device,
        )
        gray_threshold = lock_image_recall_threshold(gray_images, gray_predictions)
        rgb_threshold = float(rgb_product["deployment_threshold"])
        rgb_predictions.to_csv(output / f"seed{seed}_rgb_predictions.csv", index=False)
        gray_predictions.to_csv(output / f"seed{seed}_gray_predictions.csv", index=False)
        result = evaluate_pair(
            seed, images, ground_truth, rgb_predictions, gray_predictions,
            rgb_threshold, gray_threshold, lesion_size_bounds, output, args.debug,
        )
        result["deployment_image_level_sensitivity"] = {
            "rgb": image_detection_sensitivity(images, rgb_predictions, rgb_threshold),
            "gray": image_detection_sensitivity(images, gray_predictions, gray_threshold),
        }
        if not args.debug:
            validate_rgb_reproduction(
                seed, result["rgb"], cd0["seeds"][str(seed)]["development_val"]
            )
        seed_results.append(result)
        products[str(seed)] = {
            "rgb": {
                "checkpoint": str(rgb_product["checkpoint"]),
                "checkpoint_sha256": rgb_product["checkpoint_sha256"],
                "config": str(rgb_product["config"]),
                "config_sha256": rgb_product["config_sha256"],
                "deployment_threshold": rgb_threshold,
                "model_behavior": rgb_behavior,
            },
            "gray": {
                **{key: str(value) if isinstance(value, Path) else value for key, value in gray_product.items()},
                "deployment_threshold": gray_threshold,
                "model_behavior": gray_behavior,
            },
        }
        print(
            f"seed{seed}: RGB Primary={result['rgb']['primary_froc_at_most_0p5_fp_per_image']['lesion_sensitivity']:.4f}; "
            f"Gray Primary={result['gray']['primary_froc_at_most_0p5_fp_per_image']['lesion_sensitivity']:.4f}; "
            f"delta={result['primary_delta']:+.4f}"
        )
        del rgb_model, gray_model
        torch.cuda.empty_cache()

    base = {
        "stage": "CD1_VAL",
        "debug": args.debug,
        "training_performed": False,
        "code_git_commit": code_version["git_commit"],
        "git_dirty": code_version["git_dirty"],
        "evaluation_script_sha256": file_sha256(Path(__file__)),
        "external_read": False,
        "internal_temporal_test_read": False,
        "seeds": SEEDS,
        "images": int(len(images)),
        "patients": int(images.patient_id.nunique()),
        "lesions": int(len(ground_truth)),
        "matching_iou_threshold": 0.30,
        "primary_fp_per_image_limit": 0.5,
        "gray_deployment_target_image_sensitivity": TARGET_SENSITIVITY,
        "bootstrap_repetitions": 0 if args.debug else BOOTSTRAP_REPEATS,
        "rgb_reference_summary": str(CD0_SUMMARY),
        "rgb_reference_summary_sha256": file_sha256(CD0_SUMMARY),
        "lesion_size_definition": {
            "source": "development_train",
            "bbox_area_q33": lesion_size_bounds[0],
            "bbox_area_q67": lesion_size_bounds[1],
        },
        "products": products,
        "seed_results": seed_results,
    }
    if args.debug:
        path = output / "cd1_val_debug_summary.json"
        base["qualification_evaluated"] = False
        base["deployment_thresholds_frozen_for_formal_use"] = False
    else:
        path = output / "CD1_VAL_DECISION.json"
        base["qualification_evaluated"] = True
        base["deployment_thresholds_frozen_for_formal_use"] = True
        base["decision"] = classify_cd1(seed_results)
    path.write_text(json.dumps(base, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"CD1 val {'debug' if args.debug else '正式决策'}完成: {path}")
    if not args.debug:
        print(f"冻结结论: {base['decision']['decision']}")


if __name__ == "__main__":
    main()
