#!/usr/bin/env python3
"""M0-F/A-long/C-long 新内部时间测试集一次性正式评估。

输入固定为经人工Keep验收、重复图处理和CVAT bbox整合后的112张/78人清单。
三个模型、checkpoint、阈值、患者聚合和图像变换完全复用已验证的外部投影协议；
本脚本不在测试集选择阈值或模型。A-long/C-long额外按52张癌图bbox计算AiB、
normalized AiB和PGA，评价注意力是否落在病灶区域。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from evaluate_mage_external_projection import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    MODEL_SPECS,
    PROJECT_ROOT,
    collect_predictions,
    load_model_spec,
    patient_mean,
    run_val_self_test,
    sha256_of,
    shared_patient_bootstrap,
    threshold_metrics,
)


FINAL_MANIFEST = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/"
    "内部测试集省人民260612_Keep预处理_v1_20260819/"
    "06_内部测试集标注整合_v1_1/内部测试集最终评估清单.csv"
)
FINAL_MANIFEST_SHA256 = (
    "9d2437aafaa551b052f9ea2c2f10d80ae1824a86445118f30916252764e6cc1a"
)
BBOX_MANIFEST = FINAL_MANIFEST.parent / "内部测试集癌图bbox冻结.csv"
BBOX_MANIFEST_SHA256 = (
    "4abbc9ae271d688d52fa2e4add511118eef68e414225ead99e03e57d4de17f0c"
)
ANNOTATIONS_XML = FINAL_MANIFEST.parent / "annotations2_frozen.xml"
ANNOTATIONS_XML_SHA256 = (
    "40293f2e038200d85bf982db99e2cfa6e0b12e532f5628e84843ce131f572eb5"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/MAGE/内部时间测试集一次性评估_20260819"
GRID_SIZE = 7
NORMALIZED_AIB_MAX_BBOX_AREA = 0.99
LESION_TERCILE_BOUNDS = (0.18661894999626696, 0.3384438676553876)
SPATIAL_BOOTSTRAP_SEED = 20260820


def parse_args() -> argparse.Namespace:
    """解析运行参数；正式输出已存在时拒绝覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda")
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="只核验清单、SHA、文件与bbox，不加载模型或生成预测。",
    )
    return parser.parse_args()


def build_internal_frame() -> pd.DataFrame:
    """加载并严格核验112张最终内部测试清单。"""
    frozen = {
        FINAL_MANIFEST: FINAL_MANIFEST_SHA256,
        BBOX_MANIFEST: BBOX_MANIFEST_SHA256,
        ANNOTATIONS_XML: ANNOTATIONS_XML_SHA256,
    }
    for path, expected_sha in frozen.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha = sha256_of(path)
        if actual_sha != expected_sha:
            raise ValueError(f"冻结输入SHA不符: {path}: {actual_sha} != {expected_sha}")
    frame = pd.read_csv(
        FINAL_MANIFEST, encoding="utf-8-sig", dtype={"patient_id": "string"}
    )
    expected = {
        "images": 112,
        "patients": 78,
        "cancer_images": 52,
        "cancer_patients": 32,
        "noncancer_images": 60,
        "noncancer_patients": 46,
    }
    actual = {
        "images": len(frame),
        "patients": frame.patient_id.nunique(),
        "cancer_images": int(frame.label.eq(1).sum()),
        "cancer_patients": frame.loc[frame.label.eq(1), "patient_id"].nunique(),
        "noncancer_images": int(frame.label.eq(0).sum()),
        "noncancer_patients": frame.loc[frame.label.eq(0), "patient_id"].nunique(),
    }
    if actual != expected:
        raise ValueError(f"内部最终清单计数不符: {actual} != {expected}")
    if frame.groupby("patient_id").label.nunique().gt(1).any():
        raise ValueError("内部测试清单存在跨标签患者")
    if frame.geometry_id.nunique() != len(frame):
        raise ValueError("内部测试清单geometry_id不唯一")
    if frame.keep_sha256.nunique() != len(frame):
        raise ValueError("内部测试清单仍含Keep字节重复图")
    frame["image_path"] = frame.masked.map(
        lambda value: str((PROJECT_ROOT / value).resolve())
    )
    missing = [path for path in frame.image_path if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"内部测试Keep图缺失{len(missing)}张")
    for row in frame.itertuples(index=False):
        if sha256_of(Path(row.image_path)) != row.keep_sha256:
            raise ValueError(f"Keep图SHA与最终清单不一致: {row.relative_path}")
    cancer = frame.label.eq(1)
    if frame.loc[cancer, "bbox_available"].ne(True).any():
        raise ValueError("癌图存在bbox缺失")
    if frame.loc[~cancer, "bbox_available"].ne(False).any():
        raise ValueError("非癌图不应带bbox")
    for axis, size in (("x", "keep_width"), ("y", "keep_height")):
        frame[f"bbox_{axis}1_norm"] = frame[f"bbox_{axis}1"] / frame[size]
        frame[f"bbox_{axis}2_norm"] = frame[f"bbox_{axis}2"] / frame[size]
    valid = frame.loc[cancer]
    inside = (
        valid.bbox_x1_norm.ge(0) & valid.bbox_y1_norm.ge(0)
        & valid.bbox_x2_norm.le(1) & valid.bbox_y2_norm.le(1)
        & valid.bbox_x2_norm.gt(valid.bbox_x1_norm)
        & valid.bbox_y2_norm.gt(valid.bbox_y1_norm)
    )
    if not inside.all():
        raise ValueError("归一化bbox越界或无效")
    return frame.sort_values("image_index").reset_index(drop=True)


def grid_overlap_with_bbox(box: np.ndarray) -> np.ndarray:
    """返回7x7网格单元被bbox覆盖的面积比例。"""
    x1, y1, x2, y2 = map(float, box)
    overlap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    cell = 1.0 / GRID_SIZE
    for grid_y in range(GRID_SIZE):
        gy1, gy2 = grid_y * cell, (grid_y + 1) * cell
        for grid_x in range(GRID_SIZE):
            gx1, gx2 = grid_x * cell, (grid_x + 1) * cell
            width = max(0.0, min(x2, gx2) - max(x1, gx1))
            height = max(0.0, min(y2, gy2) - max(y1, gy1))
            overlap[grid_y, grid_x] = width * height / (cell * cell)
    return overlap


def spatial_predictions(
    frame: pd.DataFrame, attention: np.ndarray, model_name: str
) -> pd.DataFrame:
    """按冻结AiB/nAiB/PGA口径计算一模型的52张癌图空间指标。"""
    records = []
    for index, row in frame.loc[frame.label.eq(1)].iterrows():
        mass = attention[index].reshape(GRID_SIZE, GRID_SIZE).astype(np.float64)
        if not np.isclose(mass.sum(), 1.0, atol=1e-6):
            raise ValueError(f"{model_name} attention行和不为1: index={index}")
        box = np.array([
            row.bbox_x1_norm, row.bbox_y1_norm,
            row.bbox_x2_norm, row.bbox_y2_norm,
        ], dtype=np.float64)
        area = float((box[2] - box[0]) * (box[3] - box[1]))
        aib = float((mass * grid_overlap_with_bbox(box)).sum())
        peak_y, peak_x = divmod(int(mass.argmax()), GRID_SIZE)
        center_x = (peak_x + 0.5) / GRID_SIZE
        center_y = (peak_y + 0.5) / GRID_SIZE
        pga = float(
            box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3]
        )
        if area <= LESION_TERCILE_BOUNDS[0]:
            lesion_group = "small"
        elif area <= LESION_TERCILE_BOUNDS[1]:
            lesion_group = "medium"
        else:
            lesion_group = "large"
        records.append({
            "image_index": int(row.image_index),
            "patient_id": row.patient_id,
            "relative_path": row.relative_path,
            "model": model_name,
            "lesion_area_fraction": area,
            "lesion_size_group": lesion_group,
            "aib": aib,
            "normalized_aib": (
                (aib - area) / (1.0 - area)
                if area < NORMALIZED_AIB_MAX_BBOX_AREA else np.nan
            ),
            "pga": pga,
            "peak_grid_y": peak_y,
            "peak_grid_x": peak_x,
        })
    return pd.DataFrame(records)


def summarize_spatial(frame: pd.DataFrame) -> dict:
    """汇总总体及冻结病灶大小分层的空间指标。"""
    strata = {}
    for name in ("small", "medium", "large"):
        group = frame.loc[frame.lesion_size_group.eq(name)]
        strata[name] = {
            "images": int(len(group)),
            "patients": int(group.patient_id.nunique()),
            "mean_aib": float(group.aib.mean()),
            "mean_normalized_aib": float(group.normalized_aib.mean()),
            "pga": float(group.pga.mean()),
        }
    return {
        "cancer_images": int(len(frame)),
        "cancer_patients": int(frame.patient_id.nunique()),
        "mean_aib": float(frame.aib.mean()),
        "mean_normalized_aib": float(frame.normalized_aib.mean()),
        "pga": float(frame.pga.mean()),
        "lesion_size_strata": strata,
    }


def paired_spatial_bootstrap(
    a_frame: pd.DataFrame, c_frame: pd.DataFrame,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict:
    """以癌患者为单位配对重采样，估计C-long减A-long空间指标差值。"""
    metrics = ("aib", "normalized_aib", "pga")
    a_patient = a_frame.groupby("patient_id", as_index=False)[list(metrics)].mean()
    c_patient = c_frame.groupby("patient_id", as_index=False)[list(metrics)].mean()
    merged = a_patient.merge(
        c_patient, on="patient_id", suffixes=("_A", "_C"), validate="one_to_one"
    ).sort_values("patient_id")
    rng = np.random.default_rng(SPATIAL_BOOTSTRAP_SEED)
    records = {metric: [] for metric in metrics}
    for _ in range(iterations):
        sampled = rng.integers(0, len(merged), size=len(merged))
        for metric in metrics:
            diff = (
                merged[f"{metric}_C"].to_numpy()[sampled]
                - merged[f"{metric}_A"].to_numpy()[sampled]
            )
            records[metric].append(float(np.nanmean(diff)))
    return {
        metric: {
            "mean_difference": float(
                (merged[f"{metric}_C"] - merged[f"{metric}_A"]).mean()
            ),
            "ci95": np.quantile(records[metric], [0.025, 0.975]).tolist(),
        }
        for metric in metrics
    }


def render_attention_bbox_review(
    frame: pd.DataFrame, attentions: dict[str, np.ndarray], output: Path
) -> None:
    """为52张癌图生成原图bbox及A/C注意力三联图。"""
    output.mkdir(parents=True, exist_ok=True)
    for index, row in frame.loc[frame.label.eq(1)].iterrows():
        with Image.open(row.image_path) as handle:
            image = np.asarray(handle.convert("RGB"))
        vmax = max(
            float(attentions["A-long"][index].max()),
            float(attentions["C-long"][index].max()),
        )
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(image)
        axes[0].add_patch(Rectangle(
            (row.bbox_x1, row.bbox_y1),
            row.bbox_x2 - row.bbox_x1,
            row.bbox_y2 - row.bbox_y1,
            fill=False, edgecolor="lime", linewidth=2,
        ))
        axes[0].set_title("Keep image + lesion bbox")
        for axis, name in zip(axes[1:], ("A-long", "C-long")):
            axis.imshow(image)
            axis.imshow(
                attentions[name][index].reshape(GRID_SIZE, GRID_SIZE),
                cmap="jet", alpha=0.45, vmin=0.0, vmax=vmax,
                extent=(0, image.shape[1], image.shape[0], 0),
                interpolation="bilinear",
            )
            axis.add_patch(Rectangle(
                (row.bbox_x1, row.bbox_y1),
                row.bbox_x2 - row.bbox_x1,
                row.bbox_y2 - row.bbox_y1,
                fill=False, edgecolor="lime", linewidth=2,
            ))
            axis.set_title(f"{name} attention")
        for axis in axes:
            axis.axis("off")
        fig.suptitle(
            f"index={int(row.image_index):04d} patient={row.patient_id} "
            f"label={int(row.label)}"
        )
        fig.tight_layout()
        fig.savefig(output / f"{int(row.image_index):04d}_attention_bbox.png", dpi=110)
        plt.close(fig)


def main() -> None:
    """执行血缘预检、val闸门、一次性测试推理、指标与图像导出。"""
    args = parse_args()
    frame = build_internal_frame()
    print(
        f"[preflight] 内部最终清单通过: {len(frame)}张/"
        f"{frame.patient_id.nunique()}人; bbox={int(frame.bbox_available.sum())}",
        flush=True,
    )
    if args.preflight_only:
        return
    if args.output.exists():
        raise FileExistsError(f"正式输出目录已存在，拒绝覆盖: {args.output}")
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定cuda但不可用")

    loaded = {
        name: load_model_spec(name, spec, device)
        for name, spec in MODEL_SPECS.items()
    }
    val_self_test = {}
    for name in MODEL_SPECS:
        val_self_test[name] = run_val_self_test(
            name, loaded[name], args.batch_size, args.num_workers, device
        )
    print("[formal] 三模型val闸门通过，开始一次性内部测试", flush=True)

    attentions = {}
    for name, spec in MODEL_SPECS.items():
        print(f"[formal] {name} 推理112张", flush=True)
        probabilities, attention = collect_predictions(
            loaded[name]["model"], spec["kind"], frame,
            args.batch_size, args.num_workers, device,
        )
        frame[name] = probabilities
        if attention is not None:
            attentions[name] = attention

    metrics = {}
    patient_frames = {}
    for name in MODEL_SPECS:
        patients = patient_mean(frame[["patient_id", "label", name]], name)
        patient_frames[name] = patients
        metrics[name] = {
            "image": threshold_metrics(
                frame.label, frame[name], loaded[name]["image_threshold"]
            ),
            "patient": threshold_metrics(
                patients.label, patients.probability,
                loaded[name]["patient_threshold"],
            ),
        }
    patient_bootstrap = shared_patient_bootstrap(
        patient_frames, BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED
    )

    spatial_frames = {
        name: spatial_predictions(frame, attentions[name], name)
        for name in ("A-long", "C-long")
    }
    spatial_summary = {
        name: summarize_spatial(spatial_frames[name]) for name in spatial_frames
    }
    spatial_bootstrap = paired_spatial_bootstrap(
        spatial_frames["A-long"], spatial_frames["C-long"]
    )

    # 所有模型闸门、推理和统计成功后才创建正式目录，
    # 避免中途失败留下可被误认为完整结果的半成品。
    args.output.mkdir(parents=True)
    attention_columns = pd.DataFrame({
        f"{name}_attention_{cell}": attentions[name][:, cell]
        for name in attentions for cell in range(GRID_SIZE * GRID_SIZE)
    })
    image_output = pd.concat([frame, attention_columns], axis=1)
    image_output.to_csv(
        args.output / "internal_image_predictions.csv",
        index=False, encoding="utf-8-sig",
    )
    patient_output = patient_frames["M0-F"][["patient_id", "label"]].copy()
    for name in MODEL_SPECS:
        patient_output[f"{name}_probability"] = patient_frames[name].probability
    patient_output.to_csv(
        args.output / "internal_patient_predictions.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.concat(spatial_frames.values(), ignore_index=True).to_csv(
        args.output / "internal_spatial_predictions.csv",
        index=False, encoding="utf-8-sig",
    )
    classification_rows = []
    for name in MODEL_SPECS:
        for level in ("image", "patient"):
            classification_rows.append({
                "model": name,
                "level": level,
                **metrics[name][level],
            })
    pd.DataFrame(classification_rows).to_csv(
        args.output / "internal_classification_metrics.csv",
        index=False, encoding="utf-8-sig",
    )
    render_attention_bbox_review(
        frame, attentions, args.output / "attention_bbox_review"
    )

    summary = {
        "stage": "MAGE_internal_temporal_test_once",
        "protocol_frozen": "2026-08-19_before_predictions",
        "cohort": {
            "images": 112,
            "patients": 78,
            "cancer_images": 52,
            "cancer_patients": 32,
            "noncancer_images": 60,
            "noncancer_patients": 46,
        },
        "inputs": {
            "final_manifest": str(FINAL_MANIFEST),
            "final_manifest_sha256": FINAL_MANIFEST_SHA256,
            "bbox_manifest_sha256": BBOX_MANIFEST_SHA256,
            "annotations_xml_sha256": ANNOTATIONS_XML_SHA256,
        },
        "models": {
            name: {
                "role": spec["role"],
                "checkpoint_sha256": loaded[name]["weight_sha256"],
                "image_threshold": loaded[name]["image_threshold"],
                "patient_threshold": loaded[name]["patient_threshold"],
                "threshold_source": "各模型val冻结，未在内部测试重扫",
            }
            for name, spec in MODEL_SPECS.items()
        },
        "val_self_test": val_self_test,
        "metrics": metrics,
        "patient_bootstrap": patient_bootstrap,
        "spatial": spatial_summary,
        "spatial_paired_bootstrap_C_minus_A": spatial_bootstrap,
        "boundaries": [
            "内部测试只运行一次，不回灌训练或checkpoint选择",
            "分类阈值来自val冻结，未在测试集重新扫描",
            "患者概率为患者内图像癌概率算术均值",
            "AiB/nAiB/PGA只在52张有bbox癌图上计算",
            "样本量较小，结论须结合bootstrap区间解释",
        ],
        "internal_test_evaluated": True,
        "external_evaluated_in_this_run": False,
    }
    (args.output / "internal_test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[formal] 完成: {args.output}", flush=True)
    for name in MODEL_SPECS:
        image_metric = metrics[name]["image"]
        patient_metric = metrics[name]["patient"]
        print(
            f"  {name}: patient AUC={patient_metric['auc']:.4f} "
            f"image AUC={image_metric['auc']:.4f} "
            f"Sens={patient_metric['sensitivity']:.4f} "
            f"Spec={patient_metric['specificity']:.4f}",
            flush=True,
        )
    for name in ("A-long", "C-long"):
        metric = spatial_summary[name]
        print(
            f"  {name} spatial: AiB={metric['mean_aib']:.4f} "
            f"nAiB={metric['mean_normalized_aib']:.4f} "
            f"PGA={metric['pga']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
