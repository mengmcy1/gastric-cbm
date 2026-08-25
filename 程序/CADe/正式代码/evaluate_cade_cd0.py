#!/usr/bin/env python3
"""Run the frozen CD0 error atlas for three Y3-F YOLO26 detectors.

CD0 performs no training and never reads the locked internal temporal test set.
It evaluates the development val cohort and the already-used external development
cohort, then exports FROC curves, frozen-threshold errors and review images.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINING_CODE = PROJECT_ROOT / "程序/模型训练/正式代码"
sys.path.insert(0, str(TRAINING_CODE))

from evaluate_y2_yolo26 import (  # noqa: E402
    GEOMETRY_CONFIDENCE,
    GEOMETRY_MAX_DET,
    INFERENCE_BATCH_SIZE,
)
from evaluate_y5_yolo26_full_external import (  # noqa: E402
    build_external_view,
    resolution_group,
    validate_manifest,
)
from run_y1_yolo26_smoke import assert_locked_library_behavior, file_sha256  # noqa: E402
from cade_cd0_metrics import (  # noqa: E402
    box_iou,
    froc_curve,
    match_predictions,
    primary_froc_point,
    summarize_detection,
)


SEEDS = (42, 202, 503)
Y3_ROOT = PROJECT_ROOT / "结果/YOLO26定位_0804/Y3固定640三种子"
Y0F_ROOT = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/"
    "11_Y0F_YOLO26完整诊断数据_20260813"
)
EXTERNAL_BBOX = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃镜多中心测试集_M0Keep预处理_v1_20260805/"
    "03_外部癌图标注整合_v2_20260824/外部多中心癌图bbox空间评价清单_v2.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/CADe/CD0错误地图_20260825"


def parse_args() -> argparse.Namespace:
    """解析显卡、输出目录和只读预检选项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_products() -> dict[int, dict]:
    """读取三个 Y3-F 正式产品及各自的 val 部署阈值。"""
    products = {}
    for seed in SEEDS:
        run = Y3_ROOT / f"y3_full_yolo26s_640_seed{seed}"
        config_path = run / "y3_geometry_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        checkpoint = Path(config["checkpoint"])
        if config["stage"] != "Y3-F" or config["role"] != "full" or config["seed"] != seed:
            raise ValueError(f"Y3-F配置角色异常: {config_path}")
        if file_sha256(checkpoint) != config["checkpoint_sha256"]:
            raise ValueError(f"Y3-F权重SHA不一致: seed{seed}")
        products[seed] = {
            "checkpoint": checkpoint,
            "checkpoint_sha256": config["checkpoint_sha256"],
            "config": config_path,
            "config_sha256": file_sha256(config_path),
            "deployment_threshold": float(config["geometry"]["deployment_threshold"]),
            "imgsz": int(config["geometry_protocol"]["imgsz"]),
        }
    return products


def load_val_cohort(debug: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构建 Y0-F val 图片表和病灶真值表，不读取 test 图像。"""
    mapping = pd.read_csv(Y0F_ROOT / "y0f_mapping.csv", encoding="utf-8-sig")
    frame = mapping[mapping.split.eq("val")].copy().reset_index(drop=True)
    if len(frame) != 497 or frame.patient_id.nunique() != 260:
        raise ValueError(f"Y0-F val规模异常: {len(frame)}张/{frame.patient_id.nunique()}人")
    frame["image_key"] = frame.image_relpath.astype(str)
    frame["image_path"] = frame.image_relpath.astype(str)
    frame["year_group"] = frame.year.fillna("unknown").astype(str)
    frame["cohort"] = "development_val"
    ground_truth = frame[frame.label.eq(1)][[
        "image_key", "bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm",
        "bbox_area_fraction",
    ]].rename(columns={
        "bbox_x1_norm": "x1", "bbox_y1_norm": "y1",
        "bbox_x2_norm": "x2", "bbox_y2_norm": "y2",
    })
    ground_truth["gt_index"] = 0
    if debug:
        keep = pd.concat([
            frame[frame.label.eq(1)].head(8), frame[frame.label.eq(0)].head(8)
        ]).index
        frame = frame.loc[keep].reset_index(drop=True)
        ground_truth = ground_truth[ground_truth.image_key.isin(frame.image_key)].reset_index(drop=True)
    return frame, ground_truth


def load_external_cohort(debug: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """将 539 张有框癌图与 1399 张非癌图组成外部开发 CADe 队列。"""
    full = validate_manifest(
        PROJECT_ROOT / "结果/M0全量诊断_0804/外部多中心完整测试_Keep_v1/external_keep_manifest.csv"
    )
    bbox = pd.read_csv(EXTERNAL_BBOX, encoding="utf-8-sig", dtype={"patient_id": str})
    cancer = full[full.label.eq(1)].merge(
        bbox[[
            "source_keep_path", "bbox_x1_norm", "bbox_y1_norm",
            "bbox_x2_norm", "bbox_y2_norm", "bbox_area_fraction",
        ]],
        left_on="image_path", right_on="source_keep_path", how="inner", validate="one_to_one",
    )
    frame = pd.concat([full[full.label.eq(0)], cancer], ignore_index=True, sort=False)
    if len(frame) != 1938 or int(frame.label.eq(1).sum()) != 539:
        raise ValueError(f"外部CADe队列规模异常: {len(frame)}张/癌{frame.label.eq(1).sum()}")
    frame["image_key"] = frame.image_relpath.astype(str)
    frame["cohort"] = "external_development"
    frame["source"] = frame.original_relpath.map(lambda value: str(value).split("/")[1])
    ground_truth = frame[frame.label.eq(1)][[
        "image_key", "bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm",
        "bbox_area_fraction",
    ]].rename(columns={
        "bbox_x1_norm": "x1", "bbox_y1_norm": "y1",
        "bbox_x2_norm": "x2", "bbox_y2_norm": "y2",
    })
    ground_truth["gt_index"] = 0
    if debug:
        frame = pd.concat([
            frame[frame.label.eq(1)].head(8), frame[frame.label.eq(0)].head(8)
        ]).reset_index(drop=True)
        ground_truth = ground_truth[ground_truth.image_key.isin(frame.image_key)].reset_index(drop=True)
    return frame, ground_truth


def predict_all_boxes(
    model: YOLO,
    images: pd.DataFrame,
    source_dir: Path,
    imgsz: int,
    device: str,
) -> pd.DataFrame:
    """YOLO26 流式推理并保留每图所有 confidence>=0.001 候选框。"""
    rows_by_path = {
        (PROJECT_ROOT / row.image_path).resolve(): row.image_key
        for row in images.itertuples(index=False)
    }
    results = model.predict(
        source=str(source_dir), imgsz=imgsz, conf=GEOMETRY_CONFIDENCE,
        max_det=GEOMETRY_MAX_DET, device=device, batch=INFERENCE_BATCH_SIZE,
        stream=True, augment=False, nms=False, verbose=False,
    )
    records = []
    for result in results:
        path = Path(result.path).resolve()
        if path not in rows_by_path:
            raise RuntimeError(f"CD0预测路径不在队列: {path}")
        image_key = rows_by_path.pop(path)
        confidence = result.boxes.conf.detach().cpu().numpy()
        boxes = result.boxes.xyxyn.detach().cpu().numpy()
        order = np.argsort(-confidence)
        for prediction_index, index in enumerate(order):
            box = boxes[index].astype(float)
            records.append({
                "image_key": image_key,
                "prediction_index": prediction_index,
                "confidence": float(confidence[index]),
                "x1": float(box[0]), "y1": float(box[1]),
                "x2": float(box[2]), "y2": float(box[3]),
            })
    if rows_by_path:
        raise RuntimeError(f"CD0推理缺少{len(rows_by_path)}张图")
    return pd.DataFrame(records, columns=[
        "image_key", "prediction_index", "confidence", "x1", "y1", "x2", "y2"
    ])


def basic_stratified_summary(
    images: pd.DataFrame,
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    threshold: float,
) -> list[dict]:
    """按来源、分辨率、年份和病灶大小输出部署点分层诊断。"""
    records = []
    definitions: list[tuple[str, pd.Series]] = []
    for column in ("source", "center", "size_group", "year_group", "aspect_group", "frame_profile"):
        if column in images:
            for value in images[column].dropna().unique():
                definitions.append((f"{column}={value}", images[column].eq(value)))
    if "bbox_area_fraction" in ground_truth and len(ground_truth):
        bounds = ground_truth.bbox_area_fraction.quantile([1 / 3, 2 / 3]).to_numpy()
        lesion_group = pd.cut(
            ground_truth.bbox_area_fraction,
            [-math.inf, bounds[0], bounds[1], math.inf],
            labels=["lesion_small", "lesion_medium", "lesion_large"],
        )
        for value in lesion_group.unique():
            keys = ground_truth.loc[lesion_group.eq(value), "image_key"]
            definitions.append((str(value), images.image_key.isin(keys)))
    for name, mask in definitions:
        subset = images[mask]
        if subset.empty:
            continue
        subset_predictions = predictions[predictions.image_key.isin(subset.image_key)]
        subset_gt = ground_truth[ground_truth.image_key.isin(subset.image_key)]
        curve = froc_curve(subset, subset_predictions, subset_gt)
        matches = match_predictions(subset_predictions, subset_gt, threshold, 0.30)
        records.append({
            "stratum": name,
            "images": int(len(subset)),
            "patients": int(subset.patient_id.nunique()),
            "lesions": int(len(subset_gt)),
            "deployment_sensitivity": (
                float(matches.is_tp.sum() / len(subset_gt)) if len(subset_gt) else math.nan
            ),
            "deployment_fp_per_image": (
                float((~matches.is_tp).sum() / len(subset)) if len(matches) else 0.0
            ),
            "primary_sensitivity_at_0p5_fp_per_image": primary_froc_point(curve)["lesion_sensitivity"],
        })
    return records


def error_tables(
    images: pd.DataFrame,
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    matches: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成所有 FN 和 FP 的自动初分表，保留人工主/次错误列。"""
    matched_pairs = {
        (row.image_key, int(row.matched_gt_index))
        for row in matches[matches.is_tp].itertuples(index=False)
    }
    fn_records = []
    for image_key, group in ground_truth.groupby("image_key", sort=False):
        image_predictions = predictions[
            predictions.image_key.eq(image_key) & predictions.confidence.ge(threshold)
        ]
        for index, gt in group.reset_index(drop=True).iterrows():
            if (image_key, index) in matched_pairs:
                continue
            best_iou = 0.0
            if len(image_predictions):
                best_iou = float(box_iou(
                    gt[["x1", "y1", "x2", "y2"]].to_numpy(float),
                    image_predictions[["x1", "y1", "x2", "y2"]].to_numpy(float),
                ).max())
            record = gt.to_dict()
            record.update({
                "image_key": image_key,
                "automatic_error": "no_box_above_threshold" if image_predictions.empty else "geometry_miss",
                "best_iou_above_threshold": best_iou,
                "primary_error": "", "secondary_errors": "", "reviewer": "", "review_note": "",
            })
            fn_records.append(record)
    false_positive = matches[~matches.is_tp].merge(
        images[["image_key", "patient_id", "label", "image_path"]],
        on="image_key", how="left", validate="many_to_one",
    )
    if len(false_positive):
        false_positive["automatic_error"] = np.where(
            false_positive.label.eq(0), "negative_image_fp", "extra_box_on_positive"
        )
        for column in ("primary_error", "secondary_errors", "reviewer", "review_note"):
            false_positive[column] = ""
    fn = pd.DataFrame(fn_records)
    if len(fn):
        fn = fn.merge(
            images[["image_key", "patient_id", "label", "image_path"]],
            on="image_key", how="left", validate="many_to_one",
        )
    return fn, false_positive


def render_review_images(candidates: pd.DataFrame, output: Path) -> None:
    """将入选 FN/FP 生成原图与框叠加两联图，原数据只读。"""
    output.mkdir(parents=True, exist_ok=True)
    for order, row in enumerate(candidates.itertuples(index=False), start=1):
        image = plt.imread(PROJECT_ROOT / row.image_path)
        height, width = image.shape[:2]
        figure, axes = plt.subplots(1, 2, figsize=(10, 5))
        for axis in axes:
            axis.imshow(image)
            axis.axis("off")
        axes[0].set_title("Original")
        axes[1].set_title(f"{row.error_kind} | seed={row.seed}")
        if not pd.isna(row.gt_x1):
            axes[1].add_patch(Rectangle(
                (row.gt_x1 * width, row.gt_y1 * height),
                (row.gt_x2 - row.gt_x1) * width,
                (row.gt_y2 - row.gt_y1) * height,
                fill=False, edgecolor="#20c05c", linewidth=2.5,
            ))
        if not pd.isna(row.pred_x1):
            axes[1].add_patch(Rectangle(
                (row.pred_x1 * width, row.pred_y1 * height),
                (row.pred_x2 - row.pred_x1) * width,
                (row.pred_y2 - row.pred_y1) * height,
                fill=False, edgecolor="#f04b3f", linewidth=2.5,
            ))
        figure.tight_layout()
        figure.savefig(output / f"{order:04d}_{row.cohort}_{row.error_kind}_seed{row.seed}.png", dpi=120)
        plt.close(figure)


def main() -> None:
    """执行预检或完整 CD0 三种子内部val+外部开发评价。"""
    args = parse_args()
    products = load_products()
    val_images, val_gt = load_val_cohort(args.debug)
    external_images, external_gt = load_external_cohort(args.debug)
    if args.preflight_only:
        print(
            f"CD0预检通过: val={len(val_images)}张/{len(val_gt)}病灶; "
            f"external={len(external_images)}张/{len(external_gt)}病灶; "
            f"seeds={list(products)}; internal_temporal_read=False"
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，拒绝正式CD0推理")
    output = args.output.with_name(args.output.name + ("_debug" if args.debug else ""))
    if output.exists():
        raise FileExistsError(f"CD0输出目录已存在，拒绝覆盖: {output}")
    output.mkdir(parents=True)
    val_source = build_external_view(val_images, output / "development_val_view")
    external_source = build_external_view(
        external_images, output / "external_development_view"
    )

    summaries = {}
    for seed, product in products.items():
        model = YOLO(product["checkpoint"])
        behavior = assert_locked_library_behavior(model)
        summaries[str(seed)] = {}
        for cohort, images, ground_truth, source in (
            ("development_val", val_images, val_gt, val_source),
            ("external_development", external_images, external_gt, external_source),
        ):
            predictions = predict_all_boxes(
                model, images, source, product["imgsz"], args.device
            )
            summary, curve, matches = summarize_detection(
                images, predictions, ground_truth, product["deployment_threshold"]
            )
            summary["stratified"] = basic_stratified_summary(
                images, predictions, ground_truth, product["deployment_threshold"]
            )
            prefix = f"seed{seed}_{cohort}"
            predictions.to_csv(output / f"{prefix}_all_predictions.csv", index=False)
            curve.to_csv(output / f"{prefix}_froc.csv", index=False)
            fn, fp = error_tables(
                images, predictions, ground_truth, matches, product["deployment_threshold"]
            )
            fn.to_csv(output / f"{prefix}_fn.csv", index=False, encoding="utf-8-sig")
            fp.to_csv(output / f"{prefix}_fp.csv", index=False, encoding="utf-8-sig")
            summaries[str(seed)][cohort] = summary
            print(
                f"seed{seed} {cohort}: Primary Sens={summary['primary_froc_at_most_0p5_fp_per_image']['lesion_sensitivity']:.4f}; "
                f"deployment Sens={summary['deployment']['lesion_sensitivity']:.4f}, "
                f"FP/image={summary['deployment']['fp_per_image']:.4f}"
            )
        summaries[str(seed)]["product"] = product
        summaries[str(seed)]["model_behavior"] = behavior
        del model
        torch.cuda.empty_cache()

    config = {
        "stage": "CD0",
        "debug": args.debug,
        "training_performed": False,
        "internal_temporal_test_read": False,
        "matching_iou_threshold": 0.30,
        "primary_fp_per_image_limit": 0.5,
        "review_selection": "run finalize_cade_cd0.py after inference",
        "seeds": summaries,
    }
    (output / "cd0_summary.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"CD0完成: {output}")


if __name__ == "__main__":
    main()
