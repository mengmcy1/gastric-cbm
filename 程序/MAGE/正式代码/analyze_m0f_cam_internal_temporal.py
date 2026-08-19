#!/usr/bin/env python3
"""M0-F在新内部时间测试集上的后验CAM空间诊断。

M0-F使用GAP+线性二分类头，没有A-long/C-long那样直接参与分类的
内置注意力。本脚本使用癌logit与非癌logit的线性权重差生成7x7后验CAM，
仅作诊断性空间比较，不改写一次性分类结果。
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
from torch.utils.data import DataLoader

from evaluate_mage_external_projection import (
    BOOTSTRAP_ITERATIONS,
    MODEL_SPECS,
    ProjectionDataset,
    load_model_spec,
    sha256_of,
)
from evaluate_mage_internal_temporal_test import (
    GRID_SIZE,
    build_internal_frame,
    spatial_predictions,
    summarize_spatial,
)


FORMAL_ROOT = Path(
    "/home/mcy/gastric-cbm/结果/MAGE/内部时间测试集一次性评估_20260819"
)
FORMAL_SUMMARY = FORMAL_ROOT / "internal_test_summary.json"
FORMAL_SUMMARY_SHA256 = (
    "10bb2818a5d4628d4f96f97fb143308b7e887fb46174ca4d96537e9c0ec4745e"
)
FORMAL_IMAGE_PREDICTIONS = FORMAL_ROOT / "internal_image_predictions.csv"
FORMAL_IMAGE_PREDICTIONS_SHA256 = (
    "1a73486757c25d728091747901c7eb8939a32be99047bfa85bb4ada09e729ff8"
)
FORMAL_SPATIAL_PREDICTIONS = FORMAL_ROOT / "internal_spatial_predictions.csv"
DEFAULT_OUTPUT = FORMAL_ROOT / "m0f_cam_diagnostic"
CAM_BOOTSTRAP_SEED = 20260821


def parse_args() -> argparse.Namespace:
    """解析诊断参数；正式输出已存在时拒绝覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def verify_formal_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """核验一次性正式产物SHA、行序与A/C空间预测完整性。"""
    pinned = {
        FORMAL_SUMMARY: FORMAL_SUMMARY_SHA256,
        FORMAL_IMAGE_PREDICTIONS: FORMAL_IMAGE_PREDICTIONS_SHA256,
    }
    for path, expected in pinned.items():
        if not path.is_file() or sha256_of(path) != expected:
            raise ValueError(f"正式产物缺失或SHA不符: {path}")
    summary = json.loads(FORMAL_SUMMARY.read_text(encoding="utf-8"))
    if not summary.get("internal_test_evaluated"):
        raise ValueError("一次性正式评估未标记完成")
    images = pd.read_csv(
        FORMAL_IMAGE_PREDICTIONS, encoding="utf-8-sig", dtype={"patient_id": "string"}
    )
    spatial = pd.read_csv(
        FORMAL_SPATIAL_PREDICTIONS,
        encoding="utf-8-sig",
        dtype={"patient_id": "string"},
    )
    if len(images) != 112 or images.image_index.nunique() != 112:
        raise ValueError("正式逐图预测计数或image_index异常")
    counts = spatial.groupby("model").size().to_dict()
    if counts != {"A-long": 52, "C-long": 52}:
        raise ValueError(f"A/C空间预测不完整: {counts}")
    return images.sort_values("image_index").reset_index(drop=True), spatial


def collect_m0f_cam(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """返回M0-F癌概率与归一化后验CAM。

    CAM使用二类logit margin的线性权重差 ``w_cancer-w_noncancer``
    加权7x7特征图；不加空间上均匀的bias。ReLU仅保留正向癌证据，
    再将每张图49个位置归一化为和1。
    """
    loader = DataLoader(
        ProjectionDataset(frame),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    probabilities = np.zeros(len(frame), dtype=np.float64)
    attentions = np.zeros((len(frame), GRID_SIZE * GRID_SIZE), dtype=np.float64)
    classifier = model.classifier[1]
    margin_weight = (classifier.weight[1] - classifier.weight[0]).detach()
    model.eval()
    with torch.inference_mode():
        for batch_index, (images, indices) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=device.type == "cuda")
            feature_map = model.features(images)
            pooled = model.avgpool(feature_map)
            logits = model.classifier(torch.flatten(pooled, 1))
            raw_cam = torch.einsum("bchw,c->bhw", feature_map, margin_weight)
            positive_cam = torch.relu(raw_cam)
            mass = positive_cam.flatten(1).sum(dim=1)
            if torch.any(mass <= 1e-12):
                failed = indices[mass.cpu() <= 1e-12].tolist()
                raise RuntimeError(f"M0-F存在无正向癌证据CAM的图像: {failed}")
            attention = positive_cam / mass[:, None, None]
            row_indices = indices.numpy()
            probabilities[row_indices] = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            attentions[row_indices] = attention.flatten(1).double().cpu().numpy()
            if batch_index == len(loader):
                print(f"[CAM] batch {batch_index}/{len(loader)}", flush=True)
    return probabilities, attentions


def paired_patient_spatial_difference(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict:
    """按癌患者配对bootstrap候选模型减M0-F CAM的空间差值。"""
    metrics = ("aib", "normalized_aib", "pga")
    ref = reference.groupby("patient_id", as_index=False)[list(metrics)].mean()
    cand = candidate.groupby("patient_id", as_index=False)[list(metrics)].mean()
    merged = ref.merge(
        cand, on="patient_id", suffixes=("_ref", "_cand"), validate="one_to_one"
    ).sort_values("patient_id")
    rng = np.random.default_rng(CAM_BOOTSTRAP_SEED)
    samples = {metric: [] for metric in metrics}
    for _ in range(iterations):
        selected = rng.integers(0, len(merged), size=len(merged))
        for metric in metrics:
            difference = (
                merged[f"{metric}_cand"].to_numpy()[selected]
                - merged[f"{metric}_ref"].to_numpy()[selected]
            )
            samples[metric].append(float(np.nanmean(difference)))
    return {
        metric: {
            "mean_difference": float(
                (merged[f"{metric}_cand"] - merged[f"{metric}_ref"]).mean()
            ),
            "ci95": np.quantile(samples[metric], [0.025, 0.975]).tolist(),
        }
        for metric in metrics
    }


def render_four_panel_review(
    frame: pd.DataFrame,
    m0_attention: np.ndarray,
    formal_images: pd.DataFrame,
    output: Path,
) -> None:
    """渲染原图bbox、M0-F CAM、A-long和C-long注意力四联图。"""
    output.mkdir(parents=True, exist_ok=True)
    a_attention = formal_images[
        [f"A-long_attention_{index}" for index in range(49)]
    ].to_numpy()
    c_attention = formal_images[
        [f"C-long_attention_{index}" for index in range(49)]
    ].to_numpy()
    for index, row in frame.loc[frame.label.eq(1)].iterrows():
        with Image.open(row.image_path) as handle:
            image = np.asarray(handle.convert("RGB"))
        maps = {
            "M0-F post-hoc CAM": m0_attention[index],
            "A-long built-in attention": a_attention[index],
            "C-long built-in attention": c_attention[index],
        }
        vmax = max(float(values.max()) for values in maps.values())
        figure, axes = plt.subplots(1, 4, figsize=(18, 5))
        axes[0].imshow(image)
        axes[0].set_title("Keep image + lesion bbox")
        for axis, (title, values) in zip(axes[1:], maps.items()):
            axis.imshow(image)
            axis.imshow(
                values.reshape(GRID_SIZE, GRID_SIZE),
                cmap="jet", alpha=0.45, vmin=0.0, vmax=vmax,
                extent=(0, image.shape[1], image.shape[0], 0),
                interpolation="bilinear",
            )
            axis.set_title(title)
        for axis in axes:
            axis.add_patch(Rectangle(
                (row.bbox_x1, row.bbox_y1),
                row.bbox_x2 - row.bbox_x1,
                row.bbox_y2 - row.bbox_y1,
                fill=False, edgecolor="lime", linewidth=2,
            ))
            axis.axis("off")
        figure.suptitle(f"index={int(row.image_index):04d} patient={row.patient_id}")
        figure.tight_layout()
        figure.savefig(
            output / f"{int(row.image_index):04d}_m0f_a_c_attention.png", dpi=110
        )
        plt.close(figure)


def main() -> None:
    """执行血缘核验、M0-F CAM推理、正式概率复验与空间比较。"""
    args = parse_args()
    frame = build_internal_frame()
    formal_images, formal_spatial = verify_formal_inputs()
    if not np.array_equal(frame.image_index.to_numpy(), formal_images.image_index.to_numpy()):
        raise ValueError("冻结清单与正式逐图预测行序不一致")
    print("[preflight] 冻结清单与一次性正式产物通过", flush=True)
    if args.preflight_only:
        return
    if args.output.exists():
        raise FileExistsError(f"诊断输出已存在，拒绝覆盖: {args.output}")
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定cuda但不可用")
    loaded = load_model_spec("M0-F", MODEL_SPECS["M0-F"], device)
    probabilities, attention = collect_m0f_cam(
        loaded["model"], frame, args.batch_size, args.num_workers, device
    )
    max_probability_difference = float(
        np.max(np.abs(probabilities - formal_images["M0-F"].to_numpy()))
    )
    if max_probability_difference > 1e-7:
        raise RuntimeError(
            f"M0-F CAM前向概率与一次性正式预测不一致: "
            f"max_diff={max_probability_difference:.3e}"
        )

    m0_spatial = spatial_predictions(frame, attention, "M0-F post-hoc CAM")
    all_spatial = {"M0-F": m0_spatial}
    for name in ("A-long", "C-long"):
        all_spatial[name] = formal_spatial.loc[
            formal_spatial.model.eq(name)
        ].copy()
    summaries = {name: summarize_spatial(values) for name, values in all_spatial.items()}
    comparisons = {
        f"{name}_minus_M0-F": paired_patient_spatial_difference(m0_spatial, all_spatial[name])
        for name in ("A-long", "C-long")
    }

    # 前向概率、CAM和统计全部成功后再创建正式诊断目录。
    args.output.mkdir(parents=True)
    m0_spatial.to_csv(
        args.output / "m0f_cam_spatial_predictions.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame([
        {
            "model": name,
            "attention_type": (
                "post-hoc CAM" if name == "M0-F" else "built-in attention"
            ),
            "mean_aib": values["mean_aib"],
            "mean_normalized_aib": values["mean_normalized_aib"],
            "pga": values["pga"],
            "peak_in_bbox_count": int(round(values["pga"] * 52)),
            "cancer_images": values["cancer_images"],
        }
        for name, values in summaries.items()
    ]).to_csv(
        args.output / "all_models_spatial_comparison.csv",
        index=False, encoding="utf-8-sig",
    )
    render_four_panel_review(
        frame, attention, formal_images, args.output / "four_panel_review"
    )
    summary = {
        "stage": "M0-F_posthoc_CAM_internal_temporal_diagnostic",
        "protocol_frozen_before_cam_inference": "2026-08-19",
        "formal_summary_sha256": FORMAL_SUMMARY_SHA256,
        "formal_image_predictions_sha256": FORMAL_IMAGE_PREDICTIONS_SHA256,
        "m0f_checkpoint_sha256": loaded["weight_sha256"],
        "cam_definition": {
            "feature_map": "EfficientNet-B0 model.features output, 7x7",
            "class_direction": "classifier.weight[cancer]-classifier.weight[noncancer]",
            "bias_included": False,
            "positive_evidence": "ReLU(raw CAM)",
            "normalization": "per-image sum to 1",
        },
        "m0f_probability_max_abs_difference_vs_formal": max_probability_difference,
        "spatial": summaries,
        "paired_patient_bootstrap": comparisons,
        "boundaries": [
            "M0-F CAM为分类后验诊断，A/C为前向分类实际使用的内置注意力",
            "M0-F与A/C可用同一空间指标描述，但不宣称为完全同构机制比较",
            "本诊断不改变一次性分类结果或模型选择",
        ],
    }
    (args.output / "m0f_cam_diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] {args.output}", flush=True)
    for name in ("M0-F", "A-long", "C-long"):
        values = summaries[name]
        print(
            f"  {name}: AiB={values['mean_aib']:.4f} "
            f"nAiB={values['mean_normalized_aib']:.4f} "
            f"PGA={values['pga']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
