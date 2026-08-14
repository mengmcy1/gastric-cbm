#!/usr/bin/env python3
"""一次性评价锁定内部test上的M1、Balanced YOLO和Full YOLO。

Y4只读取Y2/Y3在验证集冻结的checkpoint与部署阈值。所有模型在同一份
198张/153人队列上推理；内部test不参与阈值、checkpoint或配方选择。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO

from efficientnet_m1_localization import EfficientNetM1, M1Dataset, decode_boxes
from evaluate_y2_yolo26 import (
    GEOMETRY_CONFIDENCE,
    GEOMETRY_MAX_DET,
    INFERENCE_BATCH_SIZE,
    SAFETY_MARGIN,
    box_metrics,
    predict_val,
)
from evaluate_y3_yolo26 import m1_run_dir
from run_y1_yolo26_smoke import PROJECT_ROOT, assert_locked_library_behavior, file_sha256


SEEDS = (42, 202, 503)
ROLES = ("balanced", "full")
DEFAULT_QUEUE = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/"
    "10_Y0_YOLO26检测数据_20260813/locked_internal_test_queue.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/YOLO26定位_0804/Y4锁定内部测试_20260814"
EXPECTED_QUEUE_SHA256 = "7ba25f23ad12ddfdf282efe47e95726751bf8df42b31ae2e4223384ae3ffd694"
Y2_BALANCED_SEED42 = PROJECT_ROOT / (
    "结果/YOLO26定位_0804/Y2平衡分辨率预筛/"
    "y2b_yolo26s_640_seed42/y2_geometry_config.json"
)
Y3_ROOT = PROJECT_ROOT / "结果/YOLO26定位_0804/Y3固定640三种子"


def parse_args() -> argparse.Namespace:
    """解析唯一输出目录、CUDA设备和冻结bootstrap次数。

    Returns:
        argparse.Namespace: 正式揭盲、自测或只读产品链预检所需参数。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--m1-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def geometry_config_path(role: str, seed: int) -> Path:
    """返回对应Y2/Y3正式产品的冻结几何配置。

    Args:
        role: ``balanced``主线或``full``诊断线。
        seed: 正式重复种子42、202或503。

    Returns:
        Path: 已完成产品的几何配置JSON路径。
    """
    if role == "balanced" and seed == 42:
        return Y2_BALANCED_SEED42
    return Y3_ROOT / f"y3_{role}_yolo26s_640_seed{seed}/y3_geometry_config.json"


def validate_queue(path: Path) -> pd.DataFrame:
    """读取并校验Y0-B已冻结的内部test队列及逐图SHA。

    Args:
        path: 预注册内部test队列CSV。

    Returns:
        pd.DataFrame: 198张图像的稳定顺序、标签和归一化真值框。
    """
    if file_sha256(path) != EXPECTED_QUEUE_SHA256:
        raise ValueError("Y4冻结队列SHA与预注册记录不一致")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"patient_id": str})
    required = {
        "split", "image_relpath", "patient_id", "label", "sha256",
        "localization_supervision", "bbox_x1_norm", "bbox_y1_norm",
        "bbox_x2_norm", "bbox_y2_norm",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Y4队列缺少字段: {sorted(missing)}")
    if len(frame) != 198 or frame.patient_id.nunique() != 153:
        raise ValueError(f"Y4队列规模异常: {len(frame)}张/{frame.patient_id.nunique()}人")
    if not frame.split.eq("test").all() or frame.image_relpath.duplicated().any():
        raise ValueError("Y4队列split或图像唯一性异常")
    if frame.groupby("patient_id").label.nunique().gt(1).any():
        raise ValueError("Y4队列存在患者跨标签")
    cancer = frame.label.eq(1)
    if int(cancer.sum()) != 86 or not frame.loc[cancer, "localization_supervision"].eq(1).all():
        raise ValueError("Y4癌图数量或bbox监督异常")
    for row in frame.itertuples(index=False):
        image_path = PROJECT_ROOT / row.image_relpath
        if not image_path.is_file() or file_sha256(image_path) != row.sha256:
            raise ValueError(f"Y4图像缺失或SHA不一致: {image_path}")
    return frame.reset_index(drop=True)


def load_frozen_products() -> dict[tuple[str, int], dict]:
    """校验六组YOLO/M1 checkpoint及其验证集冻结阈值。

    Returns:
        dict: 以``(role, seed)``为键的checkpoint、SHA和部署阈值。
    """
    products = {}
    for role in ROLES:
        for seed in SEEDS:
            config_path = geometry_config_path(role, seed)
            if not config_path.is_file():
                raise FileNotFoundError(config_path)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            yolo_checkpoint = Path(config["checkpoint"])
            m1_checkpoint = m1_run_dir(role, seed) / "m1_best_warmup_localization.pth"
            for checkpoint in (yolo_checkpoint, m1_checkpoint):
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
            if file_sha256(yolo_checkpoint) != config["checkpoint_sha256"]:
                raise ValueError(f"YOLO checkpoint SHA不一致: {role} seed{seed}")
            m1_record = config["m1_paired_baseline"]
            if file_sha256(m1_checkpoint) != m1_record["checkpoint_sha256"]:
                raise ValueError(f"M1 checkpoint SHA不一致: {role} seed{seed}")
            products[(role, seed)] = {
                "config_path": config_path,
                "config_sha256": file_sha256(config_path),
                "yolo_checkpoint": yolo_checkpoint,
                "yolo_checkpoint_sha256": file_sha256(yolo_checkpoint),
                "yolo_threshold": float(config["geometry"]["deployment_threshold"]),
                "m1_checkpoint": m1_checkpoint,
                "m1_checkpoint_sha256": file_sha256(m1_checkpoint),
                "m1_threshold": float(m1_record["threshold"]),
            }
    return products


def build_yolo_test_view(queue: pd.DataFrame, output: Path) -> Path:
    """用稳定编号软链接建立只读YOLO test推理目录。

    Args:
        queue: 已校验的内部test图片清单。
        output: 本次Y4唯一输出目录。

    Returns:
        Path: 供Ultralytics流式读取的稳定软链接目录。
    """
    view = output / "locked_test_view"
    view.mkdir(parents=True)
    for index, row in queue.iterrows():
        source = (PROJECT_ROOT / row.image_relpath).resolve()
        suffix = source.suffix.lower() or ".jpg"
        target = view / f"{index:06d}{suffix}"
        target.symlink_to(source)
    return view


@torch.no_grad()
def predict_m1(
    queue: pd.DataFrame,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    """在锁定test上运行一个M1产品并返回Top-1框和定位置信度。

    Args:
        queue: 按冻结顺序排列的图片级test清单。
        checkpoint: 同role、同seed的M1 warmup-only权重。
        device: 当前可见CUDA设备。
        batch_size: M1确定性推理批大小。
        num_workers: DataLoader读取进程数。

    Returns:
        pd.DataFrame: 每图置信度和``[0,1]``归一化XYXY预测框。
    """
    dataset = M1Dataset(queue, PROJECT_ROOT, training=False)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = EfficientNetM1().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    records = []
    for batch in loader:
        outputs = model(batch["image"].to(device, non_blocking=True))
        boxes, confidence = decode_boxes(outputs)
        for offset, row_index in enumerate(batch["row_index"].tolist()):
            box = boxes[offset].cpu().numpy()
            records.append({
                "row_index": int(row_index),
                "m1_confidence": float(confidence[offset]),
                "m1_x1": float(box[0]), "m1_y1": float(box[1]),
                "m1_x2": float(box[2]), "m1_y2": float(box[3]),
            })
    if len(records) != len(queue):
        raise RuntimeError(f"M1 test预测数量不完整: {len(records)} != {len(queue)}")
    return pd.DataFrame(records).sort_values("row_index").reset_index(drop=True)


def attach_m1_geometry(frame: pd.DataFrame, m1: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """将M1预测按冻结顺序配对，并计算所有癌图的几何指标。

    Args:
        frame: 内部test真值与元数据。
        m1: 同顺序M1预测表。
        threshold: 从验证集冻结的M1定位阈值。

    Returns:
        pd.DataFrame: 增加检出、IoU、覆盖率和中心命中的图片级表。
    """
    paired = pd.concat([frame.reset_index(drop=True), m1.drop(columns="row_index")], axis=1)
    paired["m1_detected"] = paired.m1_confidence.ge(threshold)
    paired["m1_iou"] = np.nan
    paired["m1_coverage"] = np.nan
    paired["m1_center_hit"] = np.nan
    for index in paired.index[paired.label.eq(1)]:
        row = paired.loc[index]
        pred = row[["m1_x1", "m1_y1", "m1_x2", "m1_y2"]].to_numpy(dtype=float)
        gt = row[["bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm"]].to_numpy(dtype=float)
        iou, coverage, center_hit, _ = box_metrics(pred, gt)
        paired.loc[index, ["m1_iou", "m1_coverage", "m1_center_hit"]] = (
            iou, coverage, center_hit
        )
    return paired


def summarize_model(frame: pd.DataFrame, prefix: str) -> dict:
    """汇总一个模型在癌图上的检出和Top-1几何表现。

    Args:
        frame: 已同时包含M1与YOLO预测的图片级配对表。
        prefix: ``m1``或``yolo``，用于选择对应列。

    Returns:
        dict: 癌图Sensitivity、几何指标及非癌触发率。
    """
    cancer = frame.loc[frame.label.eq(1)]
    noncancer = frame.loc[frame.label.eq(0)]
    detected = f"{prefix}_detected"
    iou = "m1_iou" if prefix == "m1" else "top1_iou"
    coverage = "m1_coverage" if prefix == "m1" else "lesion_coverage"
    center_hit = "m1_center_hit" if prefix == "m1" else "center_hit"
    return {
        "cancer_images": int(len(cancer)),
        "noncancer_images": int(len(noncancer)),
        "sensitivity": float(cancer[detected].mean()),
        "noncancer_trigger_rate": float(noncancer[detected].mean()),
        "mean_iou": float(cancer[iou].mean()),
        "median_iou": float(cancer[iou].median()),
        "iou_ge_0p5": float(cancer[iou].ge(0.5).mean()),
        "mean_lesion_coverage": float(cancer[coverage].mean()),
        "center_hit_rate": float(cancer[center_hit].mean()),
    }


def lesion_size_rows(
    frame: pd.DataFrame, role: str, seed: int, boundaries: list[float]
) -> list[dict]:
    """按训练集冻结面积三分位输出癌图大小分层诊断。

    Args:
        frame: 一个role/seed的内部test配对预测。
        role: Balanced主线或Full诊断线。
        seed: 当前重复种子。
        boundaries: 相应训练集癌框面积的两个三分位边界。

    Returns:
        list[dict]: small、medium、large的检出率及M1/YOLO几何指标。
    """
    cancer = frame.loc[frame.label.eq(1)].copy()
    cancer["lesion_size_group"] = pd.cut(
        cancer.bbox_area_fraction,
        bins=[-np.inf, boundaries[0], boundaries[1], np.inf],
        labels=["small", "medium", "large"],
        include_lowest=True,
    )
    rows = []
    for name, group in cancer.groupby("lesion_size_group", observed=True):
        rows.append({
            "role": role, "seed": seed, "group": str(name), "images": int(len(group)),
            "m1_sensitivity": float(group.m1_detected.mean()),
            "yolo_sensitivity": float(group.yolo_detected.mean()),
            "m1_iou_ge_0p5": float(group.m1_iou.ge(0.5).mean()),
            "yolo_iou_ge_0p5": float(group.top1_iou.ge(0.5).mean()),
            "m1_mean_iou": float(group.m1_iou.mean()),
            "yolo_mean_iou": float(group.top1_iou.mean()),
        })
    return rows


def patient_cluster_bootstrap(frames: list[pd.DataFrame], iterations: int, seed: int) -> dict:
    """以癌患者为单位估计YOLO减M1差值的95%CI。

    Args:
        frames: 一个seed或同role三seed的配对预测表。
        iterations: 患者有放回重复抽样次数。
        seed: 仅控制bootstrap重复性的随机种子。

    Returns:
        dict: Sensitivity、IoU50和mean IoU的平均差值及percentile 95%CI。
    """
    grouped_frames = []
    for frame in frames:
        cancer = frame.loc[frame.label.eq(1)]
        grouped_frames.append({key: group for key, group in cancer.groupby("patient_id", sort=False)})
    patients = np.array(list(grouped_frames[0]), dtype=object)
    if any(set(groups) != set(patients) for groups in grouped_frames[1:]):
        raise ValueError("三seed bootstrap患者集合不一致")
    rng = np.random.default_rng(seed)
    values = {"sensitivity": [], "iou_ge_0p5": [], "mean_iou": []}
    for _ in range(iterations):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        seed_differences = {name: [] for name in values}
        for groups in grouped_frames:
            sample = pd.concat([groups[patient] for patient in sampled], ignore_index=True)
            seed_differences["sensitivity"].append(
                float(sample.yolo_detected.mean() - sample.m1_detected.mean())
            )
            seed_differences["iou_ge_0p5"].append(
                float(sample.top1_iou.ge(0.5).mean() - sample.m1_iou.ge(0.5).mean())
            )
            seed_differences["mean_iou"].append(
                float(sample.top1_iou.mean() - sample.m1_iou.mean())
            )
        for name in values:
            values[name].append(float(np.mean(seed_differences[name])))
    return {
        name: {
            "mean_difference": float(np.mean(samples)),
            "percentile_95_ci": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        }
        for name, samples in values.items()
    }


def self_test() -> None:
    """用合成患者检查缺失框计零和三seed患者bootstrap。

    Returns:
        None: 检查不通过时抛出``AssertionError``，且不读取test。
    """
    gt = np.array([0.2, 0.2, 0.6, 0.6])
    if box_metrics(None, gt)[:3] != (0.0, 0.0, 0.0):
        raise AssertionError("缺失候选框未按零几何处理")
    base = pd.DataFrame({
        "patient_id": ["a", "b"], "label": [1, 1],
        "m1_detected": [False, True], "yolo_detected": [True, True],
        "m1_iou": [0.2, 0.6], "top1_iou": [0.7, 0.8],
        "m1_coverage": [0.4, 0.8], "lesion_coverage": [0.8, 0.9],
        "m1_center_hit": [0.0, 1.0], "center_hit": [1.0, 1.0],
    })
    if summarize_model(base, "yolo")["iou_ge_0p5"] != 1.0:
        raise AssertionError("Y4模型汇总指标错误")
    result = patient_cluster_bootstrap([base, base, base], 20, 20260814)
    if result["mean_iou"]["mean_difference"] <= 0:
        raise AssertionError("Y4合成bootstrap差值方向错误")


def main() -> None:
    """校验全部输入后一次性推理六组产品并给出Y4主检验结论。

    Returns:
        None: 保存逐图预测、逐seed比较、分层诊断和Y4汇总JSON。
    """
    args = parse_args()
    if args.self_test:
        self_test()
        print("Y4内部test评价自测通过；未读取test队列。")
        return
    if args.preflight_only:
        products = load_frozen_products()
        print(f"Y4产品链预检通过: {len(products)}组；未读取test队列。")
        return
    if args.output.exists():
        raise FileExistsError(f"Y4输出已存在，拒绝重复揭盲: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，拒绝执行Y4正式揭盲")

    products = load_frozen_products()
    queue = validate_queue(args.queue)
    args.output.mkdir(parents=True)
    test_view = build_yolo_test_view(queue, args.output)
    device = torch.device(f"cuda:{args.device}")
    frames_by_role: dict[str, list[pd.DataFrame]] = {role: [] for role in ROLES}
    rows = []
    size_rows = []
    per_seed_summaries = {}
    model_behavior = None

    for role in ROLES:
        for seed in SEEDS:
            product = products[(role, seed)]
            m1_predictions = predict_m1(
                queue, product["m1_checkpoint"], device,
                args.m1_batch_size, args.num_workers,
            )
            paired = attach_m1_geometry(queue.copy(), m1_predictions, product["m1_threshold"])
            yolo = YOLO(str(product["yolo_checkpoint"]))
            behavior = assert_locked_library_behavior(yolo)
            if model_behavior is None:
                model_behavior = behavior
            elif behavior != model_behavior:
                raise ValueError("六组YOLO产品的锁定模型行为不一致")
            yolo_predictions = predict_val(
                yolo, queue, 640, args.device, source_dir=test_view,
            )
            yolo_columns = [
                "image_relpath", "top1_confidence", "candidate_count_at_0p001",
                "top1_x1", "top1_y1", "top1_x2", "top1_y2", "top1_iou",
                "lesion_coverage", "center_hit", "predicted_area_fraction",
                "all_confidences",
            ]
            paired = paired.merge(
                yolo_predictions[yolo_columns], on="image_relpath", how="left",
                validate="one_to_one",
            )
            if paired.top1_confidence.isna().any():
                raise RuntimeError(f"YOLO与Y4队列配对失败: {role} seed{seed}")
            paired["yolo_detected"] = paired.top1_confidence.ge(product["yolo_threshold"])
            m1_metrics = summarize_model(paired, "m1")
            yolo_metrics = summarize_model(paired, "yolo")
            differences = {
                "sensitivity": yolo_metrics["sensitivity"] - m1_metrics["sensitivity"],
                "iou_ge_0p5": yolo_metrics["iou_ge_0p5"] - m1_metrics["iou_ge_0p5"],
                "mean_iou": yolo_metrics["mean_iou"] - m1_metrics["mean_iou"],
            }
            per_seed_bootstrap = patient_cluster_bootstrap(
                [paired], args.bootstrap, 20260814 + seed + (1000 if role == "full" else 0),
            )
            per_seed_summaries[f"{role}_seed{seed}"] = {
                "m1": m1_metrics,
                "yolo": yolo_metrics,
                "paired_differences_yolo_minus_m1": differences,
                "patient_cluster_bootstrap": per_seed_bootstrap,
                "sensitivity_engineering_gate": differences["sensitivity"] >= SAFETY_MARGIN,
                "sensitivity_statistical_noninferiority": (
                    per_seed_bootstrap["sensitivity"]["percentile_95_ci"][0] > SAFETY_MARGIN
                ),
            }
            source_config = json.loads(product["config_path"].read_text(encoding="utf-8"))
            size_rows.extend(lesion_size_rows(
                paired, role, seed, source_config["geometry"]["train_lesion_area_terciles"]
            ))
            rows.append({
                "role": role, "seed": seed,
                "m1_sensitivity": m1_metrics["sensitivity"],
                "yolo_sensitivity": yolo_metrics["sensitivity"],
                "sensitivity_difference": differences["sensitivity"],
                "m1_iou_ge_0p5": m1_metrics["iou_ge_0p5"],
                "yolo_iou_ge_0p5": yolo_metrics["iou_ge_0p5"],
                "iou_ge_0p5_difference": differences["iou_ge_0p5"],
                "m1_mean_iou": m1_metrics["mean_iou"],
                "yolo_mean_iou": yolo_metrics["mean_iou"],
                "mean_iou_difference": differences["mean_iou"],
                "m1_noncancer_trigger": m1_metrics["noncancer_trigger_rate"],
                "yolo_noncancer_trigger": yolo_metrics["noncancer_trigger_rate"],
                "passed_sensitivity_engineering_gate": differences["sensitivity"] >= SAFETY_MARGIN,
                "passed_iou50_gain": differences["iou_ge_0p5"] > 0,
            })
            frames_by_role[role].append(paired)
            paired.to_csv(
                args.output / f"y4_{role}_seed{seed}_image_predictions.csv",
                index=False, encoding="utf-8-sig",
            )
            print(
                f"{role} seed{seed}: Sens {m1_metrics['sensitivity']:.4f} -> "
                f"{yolo_metrics['sensitivity']:.4f}; IoU50 {m1_metrics['iou_ge_0p5']:.4f} -> "
                f"{yolo_metrics['iou_ge_0p5']:.4f}; meanIoU {m1_metrics['mean_iou']:.4f} -> "
                f"{yolo_metrics['mean_iou']:.4f}", flush=True,
            )
            del yolo
            torch.cuda.empty_cache()

    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.output / "y4_seed_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(size_rows).to_csv(
        args.output / "y4_lesion_size_stratified.csv", index=False, encoding="utf-8-sig"
    )
    role_summaries = {}
    for role in ROLES:
        role_rows = comparison.loc[comparison.role.eq(role)]
        bootstrap = patient_cluster_bootstrap(
            frames_by_role[role], args.bootstrap, 20260814 if role == "balanced" else 20260815,
        )
        role_summaries[role] = {
            "status": "primary" if role == "balanced" else "preregistered_diagnostic",
            "mean_seed_differences": {
                "sensitivity": float(role_rows.sensitivity_difference.mean()),
                "iou_ge_0p5": float(role_rows.iou_ge_0p5_difference.mean()),
                "mean_iou": float(role_rows.mean_iou_difference.mean()),
            },
            "patient_cluster_bootstrap": bootstrap,
            "all_seeds_iou50_improved": bool(role_rows.passed_iou50_gain.all()),
            "all_seeds_passed_sensitivity_gate": bool(
                role_rows.passed_sensitivity_engineering_gate.all()
            ),
            "passed_y4_rule": bool(
                role_rows.passed_iou50_gain.all()
                and role_rows.passed_sensitivity_engineering_gate.all()
                and bootstrap["iou_ge_0p5"]["percentile_95_ci"][0] > 0
            ),
        }
    summary = {
        "stage": "Y4 locked internal test",
        "test_evaluated": True,
        "external_evaluated": False,
        "queue": str(args.queue.resolve()),
        "queue_sha256": file_sha256(args.queue),
        "cohort": {
            "images": int(len(queue)), "patients": int(queue.patient_id.nunique()),
            "image_labels": {str(k): int(v) for k, v in queue.label.value_counts().sort_index().items()},
            "patient_labels": {
                str(k): int(v) for k, v in queue[["patient_id", "label"]].drop_duplicates()
                .label.value_counts().sort_index().items()
            },
        },
        "frozen_protocol": {
            "imgsz": 640, "geometry_confidence": GEOMETRY_CONFIDENCE,
            "geometry_max_det": GEOMETRY_MAX_DET, "nms": False,
            "yolo_inference_batch_size": INFERENCE_BATCH_SIZE,
            "m1_inference_batch_size": args.m1_batch_size,
            "sensitivity_safety_margin": SAFETY_MARGIN,
            "bootstrap_repetitions": args.bootstrap,
            "threshold_source": "Y2/Y3 validation geometry config; never internal test",
        },
        "model_behavior": model_behavior,
        "products": {
            f"{role}_seed{seed}": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in products[(role, seed)].items()
            }
            for role in ROLES for seed in SEEDS
        },
        "per_seed": per_seed_summaries,
        "roles": role_summaries,
        "primary_y4_success": role_summaries["balanced"]["passed_y4_rule"],
    }
    (args.output / "y4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Y4主检验通过={summary['primary_y4_success']}; "
        f"Balanced IoU50平均差CI={role_summaries['balanced']['patient_cluster_bootstrap']['iou_ge_0p5']['percentile_95_ci']}"
    )
    print(f"Y4结果: {args.output}")


if __name__ == "__main__":
    main()
