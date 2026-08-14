#!/usr/bin/env python3
"""将冻结Y3-F三seed YOLO26投影到无bbox外部多中心Keep集。

Y5只描述检测器响应，不评价外部定位IoU。checkpoint与阈值均来自Y3-F验证集，
外部标签不得用于选seed、重定阈值或修改后处理。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from evaluate_y2_yolo26 import (
    GEOMETRY_CONFIDENCE,
    GEOMETRY_MAX_DET,
    INFERENCE_BATCH_SIZE,
)
from run_y1_yolo26_smoke import PROJECT_ROOT, assert_locked_library_behavior, file_sha256


SEEDS = (42, 202, 503)
DEFAULT_MANIFEST = PROJECT_ROOT / (
    "结果/M0全量诊断_0804/外部多中心完整测试_Keep_v1/external_keep_manifest.csv"
)
EXPECTED_MANIFEST_SHA256 = "adb1e471286f0533344ec69373c574b03816599473be419ec5a165649dbd9925"
Y3_ROOT = PROJECT_ROOT / "结果/YOLO26定位_0804/Y3固定640三种子"
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/YOLO26定位_0804/Y5外部诊断_Full_20260814"


def parse_args() -> argparse.Namespace:
    """解析CUDA设备、冻结外部清单和唯一输出目录。

    Returns:
        argparse.Namespace: 正式Y5、只读预检或合成自测所需参数。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def load_products() -> dict[int, dict]:
    """校验三组Full checkpoint、配置SHA和验证集部署阈值。

    Returns:
        dict: 以seed为键的Y3-F配置、权重、SHA和冻结阈值。
    """
    products = {}
    for seed in SEEDS:
        run = Y3_ROOT / f"y3_full_yolo26s_640_seed{seed}"
        config_path = run / "y3_geometry_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        checkpoint = Path(config["checkpoint"])
        if config.get("role") != "full" or int(config.get("seed")) != seed:
            raise ValueError(f"Y3-F角色或seed不一致: {config_path}")
        if not checkpoint.is_file() or file_sha256(checkpoint) != config["checkpoint_sha256"]:
            raise ValueError(f"Y3-F checkpoint缺失或SHA不一致: seed{seed}")
        products[seed] = {
            "config": config_path,
            "config_sha256": file_sha256(config_path),
            "checkpoint": checkpoint,
            "checkpoint_sha256": file_sha256(checkpoint),
            "deployment_threshold": float(config["geometry"]["deployment_threshold"]),
        }
    return products


def resolution_group(width: int, height: int) -> str:
    """按原图最长边生成冻结分辨率桶。

    Args:
        width: 原图宽度像素。
        height: 原图高度像素。

    Returns:
        str: ``small/medium/large/very_large``四级分辨率组。
    """
    maximum = max(int(width), int(height))
    if maximum <= 640:
        return "small_le640"
    if maximum <= 1024:
        return "medium_641_1024"
    if maximum <= 1600:
        return "large_1025_1600"
    return "very_large_gt1600"


def patient_year(patient_id: str) -> str:
    """从稳定患者编号提取年份，无法识别时返回unknown。

    Args:
        patient_id: 外部清单中的患者标识。

    Returns:
        str: 四位年份或``unknown``。
    """
    match = re.search(r"(?:^|_)(20\d{2})(?:_|$)", str(patient_id))
    return match.group(1) if match else "unknown"


def validate_manifest(path: Path) -> pd.DataFrame:
    """读取既有外部Keep清单并校验规模、标签、路径和SHA来源。

    Args:
        path: 冻结外部图片级清单CSV。

    Returns:
        pd.DataFrame: 增加项目相对路径、分辨率组和年份的1941张队列。
    """
    if file_sha256(path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Y5外部清单SHA与冻结记录不一致")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"patient_id": str})
    required = {
        "patient_id", "label", "image_path", "original_sha256",
        "original_width", "original_height", "input_variant",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Y5外部清单缺少字段: {sorted(missing)}")
    if len(frame) != 1941 or frame.patient_id.nunique() != 1329:
        raise ValueError(f"Y5外部规模异常: {len(frame)}张/{frame.patient_id.nunique()}人")
    if frame.groupby("patient_id").label.nunique().gt(1).any():
        raise ValueError("Y5外部患者跨标签")
    resolved = []
    for value in frame.image_path:
        image = Path(value).resolve()
        if not image.is_file() or not image.is_relative_to(PROJECT_ROOT):
            raise FileNotFoundError(f"Y5图像缺失或不在项目根目录: {image}")
        resolved.append(image)
    if len(set(resolved)) != len(resolved):
        raise ValueError("Y5外部清单存在重复解析图像路径")
    frame = frame.copy()
    frame["image_relpath"] = [str(path.relative_to(PROJECT_ROOT)) for path in resolved]
    frame["size_group"] = [
        resolution_group(width, height)
        for width, height in zip(frame.original_width, frame.original_height)
    ]
    frame["year_group"] = frame.patient_id.map(patient_year)
    return frame.reset_index(drop=True)


def build_external_view(frame: pd.DataFrame, output: Path) -> Path:
    """为Ultralytics建立稳定编号的只读外部图片软链接目录。

    Args:
        frame: 已校验的外部图片清单。
        output: 本次Y5唯一输出目录。

    Returns:
        Path: 供YOLO流式读取的软链接目录。
    """
    view = output / "locked_external_view"
    view.mkdir(parents=True)
    for index, row in frame.iterrows():
        source = (PROJECT_ROOT / row.image_relpath).resolve()
        target = view / f"{index:06d}{source.suffix.lower() or '.jpg'}"
        target.symlink_to(source)
    return view


def predict_external(
    model: YOLO,
    frame: pd.DataFrame,
    source_dir: Path,
    imgsz: int,
    device: str,
    threshold: float,
) -> pd.DataFrame:
    """提取外部每图Top-1框、置信度和冻结阈值以上候选数。

    Args:
        model: 一个冻结Y3-F YOLO26模型。
        frame: 外部图片清单，必须每路径唯一。
        source_dir: 稳定软链接推理目录。
        imgsz: 冻结输入边长，当前为640。
        device: 当前可见CUDA编号。
        threshold: 对应seed在Y3-F val冻结的部署阈值。

    Returns:
        pd.DataFrame: 每图检测响应；不含任何外部几何真值或IoU。
    """
    rows_by_path = {
        (PROJECT_ROOT / row.image_relpath).resolve(): (order, row)
        for order, (_, row) in enumerate(frame.iterrows())
    }
    results = model.predict(
        source=str(source_dir), imgsz=imgsz, conf=GEOMETRY_CONFIDENCE,
        max_det=GEOMETRY_MAX_DET, device=device, batch=INFERENCE_BATCH_SIZE,
        stream=True, augment=False, nms=False, verbose=False,
    )
    records = []
    for result in results:
        result_path = Path(result.path).resolve()
        if result_path not in rows_by_path:
            raise RuntimeError(f"Y5预测路径不在冻结外部清单: {result.path}")
        order, row = rows_by_path.pop(result_path)
        confidence = result.boxes.conf.detach().cpu().numpy()
        boxes = result.boxes.xyxyn.detach().cpu().numpy()
        if len(confidence):
            top = int(np.argmax(confidence))
            score = float(confidence[top])
            box = boxes[top].astype(float)
            area = float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
        else:
            score, box, area = 0.0, None, 0.0
        record = row.to_dict()
        record.update({
            "_order": order,
            "top1_confidence": score,
            "triggered": score >= threshold,
            "candidate_count_at_0p001": int(len(confidence)),
            "candidate_count_at_frozen_threshold": int((confidence >= threshold).sum()),
            "top1_x1": math.nan if box is None else float(box[0]),
            "top1_y1": math.nan if box is None else float(box[1]),
            "top1_x2": math.nan if box is None else float(box[2]),
            "top1_y2": math.nan if box is None else float(box[3]),
            "top1_area_fraction": area,
        })
        records.append(record)
    if rows_by_path or len(records) != len(frame):
        raise RuntimeError(f"Y5预测数量不完整: {len(records)} != {len(frame)}")
    return pd.DataFrame(records).sort_values("_order").drop(columns="_order").reset_index(drop=True)


def rate_record(frame: pd.DataFrame, decision: str) -> dict:
    """按标签汇总癌检出率与非癌触发率。

    Args:
        frame: 图片级或患者级预测表。
        decision: 布尔触发列名。

    Returns:
        dict: 两类样本量、癌触发率、非癌触发率和特异度。
    """
    cancer = frame.label.eq(1)
    noncancer = frame.label.eq(0)
    return {
        "cancer_count": int(cancer.sum()),
        "noncancer_count": int(noncancer.sum()),
        "cancer_trigger_rate": float(frame.loc[cancer, decision].mean()),
        "noncancer_trigger_rate": float(frame.loc[noncancer, decision].mean()),
        "specificity": float((~frame.loc[noncancer, decision]).mean()),
    }


def aggregate_patients(images: pd.DataFrame) -> pd.DataFrame:
    """以任一图触发为患者触发，保留最大置信度和图片数。

    Args:
        images: 单seed图片级检测响应。

    Returns:
        pd.DataFrame: 每位患者一行的标签、触发状态和最大置信度。
    """
    return images.groupby("patient_id", as_index=False).agg(
        label=("label", "first"), triggered=("triggered", "max"),
        top1_confidence=("top1_confidence", "max"), image_count=("image_path", "size"),
    )


def bootstrap_rates(patient: pd.DataFrame, iterations: int, seed: int) -> dict:
    """按患者标签分层有放回估计癌/非癌触发率95%CI。

    Args:
        patient: 每位患者一行的触发结果。
        iterations: bootstrap重复次数。
        seed: 仅控制重采样复现性的随机种子。

    Returns:
        dict: 癌触发率和非癌触发率的percentile 95%CI。
    """
    groups = {label: patient.loc[patient.label.eq(label)] for label in (0, 1)}
    rng = np.random.default_rng(seed)
    values = {"cancer_trigger_rate": [], "noncancer_trigger_rate": []}
    for _ in range(iterations):
        cancer = groups[1].iloc[rng.integers(0, len(groups[1]), len(groups[1]))]
        noncancer = groups[0].iloc[rng.integers(0, len(groups[0]), len(groups[0]))]
        values["cancer_trigger_rate"].append(float(cancer.triggered.mean()))
        values["noncancer_trigger_rate"].append(float(noncancer.triggered.mean()))
    return {
        name: [float(value) for value in np.quantile(samples, [0.025, 0.975])]
        for name, samples in values.items()
    }


def distribution_record(frame: pd.DataFrame) -> dict:
    """汇总置信度、框面积和候选数量的描述性分位数。

    Args:
        frame: 一个图片分层子集。

    Returns:
        dict: 样本量以及三个响应量的均值和四分位数。
    """
    def describe(column: str) -> dict:
        values = frame[column].to_numpy(dtype=float)
        return {
            "mean": float(values.mean()),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
        }
    return {
        "images": int(len(frame)),
        "top1_confidence": describe("top1_confidence"),
        "top1_area_fraction": describe("top1_area_fraction"),
        "candidate_count_at_frozen_threshold": describe(
            "candidate_count_at_frozen_threshold"
        ),
    }


def stratified_records(images: pd.DataFrame) -> list[dict]:
    """按标签与分辨率、年份、输入变体输出检测响应分层。

    Args:
        images: 单seed完整外部图片预测。

    Returns:
        list[dict]: 每个分层的图像数、患者数和触发率。
    """
    rows = []
    for column in ("size_group", "year_group", "input_variant"):
        for (value, label), group in images.groupby([column, "label"], dropna=False):
            rows.append({
                "stratifier": column, "group": str(value), "label": int(label),
                "images": int(len(group)), "patients": int(group.patient_id.nunique()),
                "trigger_rate": float(group.triggered.mean()),
                "top1_confidence_mean": float(group.top1_confidence.mean()),
                "top1_area_fraction_mean": float(group.top1_area_fraction.mean()),
            })
    return rows


def self_test() -> None:
    """用合成记录检查患者聚合、触发率与分辨率分桶。

    Returns:
        None: 失败时抛出``AssertionError``，且不读取external。
    """
    images = pd.DataFrame({
        "patient_id": ["a", "a", "b"], "label": [1, 1, 0],
        "triggered": [False, True, False], "top1_confidence": [0.1, 0.9, 0.2],
        "image_path": ["a1", "a2", "b1"],
    })
    patients = aggregate_patients(images)
    if rate_record(patients, "triggered")["cancer_trigger_rate"] != 1.0:
        raise AssertionError("Y5患者触发聚合错误")
    if resolution_group(1880, 1880) != "very_large_gt1600":
        raise AssertionError("Y5分辨率分桶错误")


def main() -> None:
    """运行Full三seed冻结外部投影并保存逐图、分层和稳定性汇总。

    Returns:
        None: 正式模式写出三seed预测、诊断JSON和跨seed汇总。
    """
    args = parse_args()
    if args.self_test:
        self_test()
        print("Y5 Full外部诊断自测通过；未读取external。")
        return
    products = load_products()
    if args.preflight_only:
        print(f"Y5 Full产品链预检通过: {len(products)}组；未读取external。")
        return
    if args.output.exists():
        raise FileExistsError(f"Y5输出已存在，拒绝重复外部投影: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，拒绝执行Y5正式外部投影")

    frame = validate_manifest(args.manifest)
    args.output.mkdir(parents=True)
    source_dir = build_external_view(frame, args.output)
    seed_summaries = {}
    all_strata = []
    trigger_columns = []
    model_behavior = None
    combined = frame[["patient_id", "label", "image_relpath"]].copy()

    for seed in SEEDS:
        product = products[seed]
        model = YOLO(str(product["checkpoint"]))
        behavior = assert_locked_library_behavior(model)
        if model_behavior is None:
            model_behavior = behavior
        elif behavior != model_behavior:
            raise ValueError("Y5三组YOLO模型行为不一致")
        images = predict_external(
            model, frame, source_dir, 640, args.device, product["deployment_threshold"]
        )
        patients = aggregate_patients(images)
        image_rates = rate_record(images, "triggered")
        patient_rates = rate_record(patients, "triggered")
        strata = stratified_records(images)
        for row in strata:
            row["seed"] = seed
        all_strata.extend(strata)
        seed_summaries[str(seed)] = {
            "deployment_threshold": product["deployment_threshold"],
            "image_rates": image_rates,
            "patient_rates_any_image": patient_rates,
            "patient_bootstrap_ci95": bootstrap_rates(
                patients, args.bootstrap, 20260814 + seed
            ),
            "cancer_response_distribution": distribution_record(images.loc[images.label.eq(1)]),
            "noncancer_response_distribution": distribution_record(images.loc[images.label.eq(0)]),
        }
        images.to_csv(
            args.output / f"y5_full_seed{seed}_external_image_predictions.csv",
            index=False, encoding="utf-8-sig",
        )
        patients.to_csv(
            args.output / f"y5_full_seed{seed}_external_patient_predictions.csv",
            index=False, encoding="utf-8-sig",
        )
        trigger = f"trigger_seed{seed}"
        trigger_columns.append(trigger)
        combined[trigger] = images.triggered.to_numpy(dtype=bool)
        print(
            f"Full seed{seed}: image癌触发={image_rates['cancer_trigger_rate']:.4f}, "
            f"非癌触发={image_rates['noncancer_trigger_rate']:.4f}; "
            f"patient癌触发={patient_rates['cancer_trigger_rate']:.4f}, "
            f"非癌触发={patient_rates['noncancer_trigger_rate']:.4f}", flush=True,
        )
        del model
        torch.cuda.empty_cache()

    combined["trigger_count_3seeds"] = combined[trigger_columns].sum(axis=1)
    combined["majority_trigger"] = combined.trigger_count_3seeds.ge(2)
    combined.to_csv(
        args.output / "y5_full_cross_seed_trigger_agreement.csv",
        index=False, encoding="utf-8-sig",
    )
    patient_agreement = combined.groupby("patient_id", as_index=False).agg(
        label=("label", "first"),
        **{column: (column, "max") for column in trigger_columns},
    )
    patient_agreement["trigger_count_3seeds"] = patient_agreement[trigger_columns].sum(axis=1)
    patient_agreement["majority_trigger"] = patient_agreement.trigger_count_3seeds.ge(2)
    patient_agreement.to_csv(
        args.output / "y5_full_cross_seed_patient_agreement.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(all_strata).to_csv(
        args.output / "y5_full_stratified_diagnostics.csv",
        index=False, encoding="utf-8-sig",
    )
    summary = {
        "stage": "Y5 post-Y4 exploratory Full external diagnostic",
        "analysis_role": "external_response_only_without_bbox",
        "external_evaluated": True,
        "internal_test_reused": False,
        "external_bbox_available": False,
        "threshold_reselected_on_external": False,
        "seed_selected_on_external": False,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "cohort": {
            "images": int(len(frame)), "patients": int(frame.patient_id.nunique()),
            "image_labels": {str(k): int(v) for k, v in frame.label.value_counts().sort_index().items()},
            "patient_labels": {
                str(k): int(v) for k, v in frame[["patient_id", "label"]].drop_duplicates()
                .label.value_counts().sort_index().items()
            },
        },
        "protocol": {
            "imgsz": 640, "confidence": GEOMETRY_CONFIDENCE,
            "max_det": GEOMETRY_MAX_DET, "nms": False,
            "inference_batch_size": INFERENCE_BATCH_SIZE,
            "threshold_source": "each seed's Y3-F validation geometry config",
            "source_stratification": (
                "not reported: external directory roots are label-defined and source is perfectly confounded"
            ),
        },
        "model_behavior": model_behavior,
        "products": {
            str(seed): {
                key: str(value) if isinstance(value, Path) else value
                for key, value in products[seed].items()
            }
            for seed in SEEDS
        },
        "seeds": seed_summaries,
        "cross_seed": {
            "image_majority_trigger_rates": rate_record(combined, "majority_trigger"),
            "image_all_three_agreement_rate": float(
                combined.trigger_count_3seeds.isin([0, 3]).mean()
            ),
            "patient_majority_trigger_rates": rate_record(
                patient_agreement, "majority_trigger"
            ),
            "patient_all_three_agreement_rate": float(
                patient_agreement.trigger_count_3seeds.isin([0, 3]).mean()
            ),
        },
    }
    (args.output / "y5_full_external_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Y5 Full外部诊断完成: {args.output}")


if __name__ == "__main__":
    main()
