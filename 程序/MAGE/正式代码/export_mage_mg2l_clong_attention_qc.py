#!/usr/bin/env python3
"""Export fixed-rule C-long attention overlays for manual clinical QC.

This is a read-only, post-selection risk audit before the MAGE student SAE
preregistration. It never retrains, never re-selects checkpoints and never
modifies any existing artifact. Three frozen sample classes (no manual image
picking allowed):

1. teacher_student: the same 36 val cancer images used by the MG1b attention
   QC, rendered with teacher ROI/attention panels for direct comparison;
2. spatial_failure: all val cancer images with PGA failure; if fewer than 24,
   topped up by ascending normalized AiB (deduplicated);
3. high_risk: the top 12 non-cancer false positives by probability under the
   frozen threshold 0.3074711561203003, plus up to 6 images whose attention
   peak cell sits on the outermost grid ring and outside the crop box
   (edge/black-border suspicion, any label, deduplicated).

Student attention comes from the frozen C-long ``val_image_predictions.csv``
(49 columns, row sums verified); teacher attention comes from the MG2 teacher
cache (flip0 state). All coordinates are full-image normalized [0,1]. Titles
are ASCII, following the fixed MG1b QC practice. Outputs: per-image multi-panel
PNGs, one index CSV with prefilled objective fields plus blank manual-review
columns, and qc_config.json recording the frozen rules and SHA bindings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from build_mage_teacher_roi_manifest import PROJECT_ROOT, file_sha256
from train_mage_mg1_teacher import DEFAULT_MANIFEST
from train_mage_mg1b_attention_teacher import (
    GRID_SIZE,
    bbox_in_crop,
    cell_overlap_map,
)
from train_mage_mg2_student import CACHE_FORMAT, FROZEN_TEACHER_SHA256


DEFAULT_CLONG_RUN = PROJECT_ROOT / (
    "结果/MAGE/MG2L训练轮数敏感性_20260818/正式验证集筛选/"
    "mg2l_armc_efficientnet_b0_seed42"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/MAGE/MG2L训练轮数敏感性_20260818/clong_attention_qc"
DEFAULT_MG1B_QC_INDEX = PROJECT_ROOT / (
    "结果/MAGE/MG1b注意力池化教师_20260818/正式验证集筛选/"
    "mg1b_attention_efficientnet_b0_seed42/attention_qc/attention_qc_index.csv"
)
DEFAULT_TEACHER_CACHE = (
    PROJECT_ROOT / "结果/MAGE/MG2全图学生蒸馏_20260818/teacher_cache_mg1b_v3.pt"
)
FROZEN_CLONG_CHECKPOINT_SHA256 = (
    "29e76977251fbd50c9fd9ecd6ebd26927eaf9b54eeb8e70a0b70d3d5a70ce014"
)
FROZEN_THRESHOLD = 0.3074711561203003
SPATIAL_FAILURE_TARGET = 24
HIGH_RISK_FP_COUNT = 12
HIGH_RISK_RING_MAX = 6
ATTENTION_COLUMNS = [
    f"attention_{grid_y}_{grid_x}"
    for grid_y in range(GRID_SIZE) for grid_x in range(GRID_SIZE)
]
MANUAL_COLUMNS = [
    "主峰在病灶", "关注有效黏膜", "关注器械", "关注反光黏液黑边文字",
    "过于弥散", "三级结论",
]
CLASS_DIRS = {
    "teacher_student": "01_teacher_student",
    "spatial_failure": "02_spatial_failure",
    "high_risk": "03_high_risk",
}


def parse_args() -> argparse.Namespace:
    """Parse frozen C-long product, lineage inputs and output location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clong-run", type=Path, default=DEFAULT_CLONG_RUN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--mg1b-qc-index", type=Path, default=DEFAULT_MG1B_QC_INDEX)
    parser.add_argument("--teacher-cache", type=Path, default=DEFAULT_TEACHER_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def attention_grid(row: pd.Series) -> np.ndarray:
    """Return one image's 7x7 attention grid from the 49 prediction columns.

    参数:
        row (pd.Series): val_image_predictions的一行，含49列attention。
    返回:
        np.ndarray: shape ``(7,7)`` float64，行=y列=x，总和约1。
    """
    return np.array([row[name] for name in ATTENTION_COLUMNS], dtype=np.float64).reshape(
        GRID_SIZE, GRID_SIZE
    )


def peak_flags(
    attention: np.ndarray, crop_box: np.ndarray
) -> tuple[int, int, bool, bool]:
    """Locate the attention peak cell and classify its position.

    参数:
        attention (np.ndarray): shape ``(7,7)`` 全图学生注意力。
        crop_box (np.ndarray): shape ``(4,)`` 全图归一化 ``[x1,y1,x2,y2]``。
    返回:
        tuple: ``(peak_y, peak_x, peak_in_crop, peak_outer_ring)``；cell中心
            坐标为 ``(index+0.5)/7``，最外圈指行或列属于 ``{0,6}``。
    """
    peak = int(attention.argmax())
    peak_y, peak_x = divmod(peak, GRID_SIZE)
    center_x = (peak_x + 0.5) / GRID_SIZE
    center_y = (peak_y + 0.5) / GRID_SIZE
    in_crop = bool(
        crop_box[0] <= center_x <= crop_box[2]
        and crop_box[1] <= center_y <= crop_box[3]
    )
    outer_ring = bool(peak_y in (0, GRID_SIZE - 1) or peak_x in (0, GRID_SIZE - 1))
    return peak_y, peak_x, in_crop, outer_ring


def load_and_verify(
    clong_run: Path, manifest_path: Path
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Load the frozen C-long run and verify checkpoint/threshold/attention.

    参数:
        clong_run (Path): C-long正式run目录（含config/checkpoint/预测CSV）。
        manifest_path (Path): v3主清单。
    返回:
        tuple: ``(config, predictions, manifest)``；predictions已按
            relative_path连接manifest坐标列并补峰值标记。checkpoint实际SHA、
            冻结阈值、attention逐行总和、癌图AiB/PGA列与按现有口径重算值
            不一致时快速失败。
    """
    config = json.loads((clong_run / "config.json").read_text(encoding="utf-8"))
    for flag in ("test_evaluated", "internal_test_evaluated", "external_evaluated"):
        if config.get(flag):
            raise ValueError(f"C-long config锁定标记{flag}非false，拒绝导出QC")
    checkpoint_sha = file_sha256(clong_run / "mg2_armc_best_student.pth")
    if checkpoint_sha != FROZEN_CLONG_CHECKPOINT_SHA256:
        raise ValueError(f"C-long checkpoint SHA与冻结值不一致: {checkpoint_sha}")
    if config.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("C-long config记录的checkpoint SHA与实际文件不一致")
    threshold = float(config["metrics"]["patient_threshold_metrics"]["threshold"])
    if abs(threshold - FROZEN_THRESHOLD) > 1e-12:
        raise ValueError(f"C-long患者阈值与冻结值不一致: {threshold}")
    manifest_sha = file_sha256(manifest_path)
    if config.get("manifest_sha256") != manifest_sha:
        raise ValueError("C-long config的manifest SHA与当前清单不一致")

    predictions = pd.read_csv(
        clong_run / "val_image_predictions.csv", encoding="utf-8-sig",
        dtype={"patient_id": str}, float_precision="round_trip",
    )
    missing = set(ATTENTION_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"C-long预测缺少attention列: {sorted(missing)[:3]}")
    sum_error = float((predictions[ATTENTION_COLUMNS].sum(axis=1) - 1.0).abs().max())
    if sum_error > 1e-5:
        raise ValueError(f"C-long attention逐行求和异常，最大误差{sum_error}")

    manifest = pd.read_csv(
        manifest_path, encoding="utf-8-sig", dtype={"patient_id": str},
        float_precision="round_trip",
    )
    coordinate_columns = [
        "relative_path", "sha256", "image_relpath",
        "base_crop_x1", "base_crop_y1", "base_crop_x2", "base_crop_y2",
        "bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm",
    ]
    merged = predictions.merge(
        manifest[coordinate_columns], on="relative_path", how="left",
        validate="one_to_one",
    )
    if merged["sha256"].isna().any():
        raise ValueError("C-long预测存在无法匹配manifest的行")

    # 口径自检：癌图AiB/PGA列必须与attention+bbox按协议口径的重算值一致。
    max_aib_diff = 0.0
    peak_records = []
    for _, row in merged.iterrows():
        attention = attention_grid(row)
        crop = np.array([
            row.base_crop_x1, row.base_crop_y1, row.base_crop_x2, row.base_crop_y2
        ], dtype=np.float64)
        peak_y, peak_x, in_crop, outer_ring = peak_flags(attention, crop)
        peak_records.append({
            "peak_grid_y": peak_y, "peak_grid_x": peak_x,
            "peak_in_crop": in_crop, "peak_outer_ring": outer_ring,
        })
        if int(row.label) == 1:
            bbox = np.array([
                row.bbox_x1_norm, row.bbox_y1_norm,
                row.bbox_x2_norm, row.bbox_y2_norm,
            ], dtype=np.float64)
            if (bbox < -1e-9).any() or (bbox > 1 + 1e-9).any() or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError(f"癌图lesion_bbox非法: {row.relative_path}")
            aib = float((attention * cell_overlap_map(bbox)).sum())
            max_aib_diff = max(max_aib_diff, abs(aib - float(row.aib)))
    if max_aib_diff > 1e-6:
        raise ValueError(f"AiB列与重算口径不一致，最大差{max_aib_diff}")
    for name, values in {
        "base_crop": merged[["base_crop_x1", "base_crop_y1", "base_crop_x2", "base_crop_y2"]],
    }.items():
        array = values.to_numpy(dtype=float)
        if (array < -1e-9).any() or (array > 1 + 1e-9).any():
            raise ValueError(f"{name}坐标超出[0,1]")
        if (array[:, 2] <= array[:, 0]).any() or (array[:, 3] <= array[:, 1]).any():
            raise ValueError(f"{name}存在无面积矩形")
    return config, merged.join(pd.DataFrame(peak_records)), manifest


def load_teacher_cache(cache_path: Path, manifest_sha256: str) -> dict:
    """Load the MG2 teacher cache with SHA binding checks (flip0 only used).

    参数:
        cache_path (Path): 教师缓存 ``.pt`` 路径。
        manifest_sha256 (str): 当前manifest SHA256。
    返回:
        dict: ``{sha256: {"flip0": {...}, "flip1": {...}}}`` 条目映射；
            格式/教师checkpoint SHA/manifest SHA不符即快速失败。
    """
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if cache.get("format") != CACHE_FORMAT:
        raise ValueError("教师缓存格式标记不符")
    if cache.get("teacher_checkpoint_sha256") != FROZEN_TEACHER_SHA256:
        raise ValueError("教师缓存的教师checkpoint SHA与冻结值不一致")
    if cache.get("manifest_sha256") != manifest_sha256:
        raise ValueError("教师缓存的manifest SHA与当前清单不一致")
    return cache


def select_samples(
    predictions: pd.DataFrame, mg1b_index: pd.DataFrame
) -> pd.DataFrame:
    """Select the three frozen QC sample classes (no manual picking).

    参数:
        predictions (pd.DataFrame): load_and_verify返回的val预测（含坐标与
            峰值标记）。
        mg1b_index (pd.DataFrame): MG1b QC索引（同一批36张val癌图）。
    返回:
        pd.DataFrame: 每图一行，``qc_class`` 记录所属类别（跨类重叠以``+``
            连接），并按类内规则排序：类1沿用MG1b顺序、类2为PGA失败在前
            再按nAiB升序、类3为假阳性概率降序在前再边缘嫌疑图。
    """
    cancer = predictions.loc[predictions.label.eq(1)]
    membership = {}

    def add(relative_path: str, class_name: str, order_key) -> None:
        entry = membership.setdefault(relative_path, {"classes": [], "order": {}})
        if class_name not in entry["classes"]:
            entry["classes"].append(class_name)
        entry["order"][class_name] = order_key

    # 类1：MG1b QC同一批36张val癌图，保持MG1b索引顺序。
    for rank, relative_path in enumerate(mg1b_index.relative_path.astype(str)):
        add(relative_path, "teacher_student", rank)

    # 类2：全部PGA失败癌图在前，再按nAiB升序补足至24张（去重）。
    pga_fail = cancer.loc[cancer.pga.eq(0)].sort_values(
        ["normalized_aib", "relative_path"], na_position="last"
    )
    chosen2 = list(pga_fail.relative_path.astype(str))
    fill = cancer.loc[~cancer.relative_path.astype(str).isin(chosen2)].sort_values(
        ["normalized_aib", "relative_path"], na_position="last"
    )
    for relative_path in fill.relative_path.astype(str):
        if len(chosen2) >= SPATIAL_FAILURE_TARGET:
            break
        chosen2.append(relative_path)
    for rank, relative_path in enumerate(chosen2):
        add(relative_path, "spatial_failure", rank)

    # 类3a：冻结阈值下非癌假阳性按概率降序前12。
    false_positive = predictions.loc[
        predictions.label.eq(0)
        & predictions.cancer_probability.ge(FROZEN_THRESHOLD)
    ].sort_values(["cancer_probability", "relative_path"], ascending=[False, True])
    chosen3 = []
    for rank, relative_path in enumerate(
        false_positive.relative_path.astype(str)[:HIGH_RISK_FP_COUNT]
    ):
        add(relative_path, "high_risk", rank)
        chosen3.append(relative_path)
    # 类3b：峰值cell在最外圈且在crop外，最多6张（与类3a去重）。
    ring = predictions.loc[
        predictions.peak_outer_ring & ~predictions.peak_in_crop
        & ~predictions.relative_path.astype(str).isin(chosen3)
    ].sort_values(
        ["cancer_probability", "relative_path"], ascending=[False, True]
    )
    for offset, relative_path in enumerate(
        ring.relative_path.astype(str)[:HIGH_RISK_RING_MAX]
    ):
        add(relative_path, "high_risk", HIGH_RISK_FP_COUNT + offset)

    rows = []
    for relative_path, entry in membership.items():
        source = predictions.loc[predictions.relative_path.eq(relative_path)].iloc[0]
        classes = sorted(
            entry["classes"], key=lambda name: list(CLASS_DIRS).index(name)
        )
        rows.append({
            "row": source,
            "qc_class": "+".join(classes),
            "primary_class": classes[0],
            "order": min(entry["order"].values()),
        })
    selected = pd.DataFrame(rows).sort_values(
        ["primary_class", "order"]
    ).reset_index(drop=True)
    return selected


def draw_box(axis: plt.Axes, box: np.ndarray, color: str, width: float = 2.0) -> None:
    """Draw one xyxy pixel rectangle without adding a legend entry."""
    x1, y1, x2, y2 = map(float, box)
    axis.add_patch(plt.Rectangle(
        (x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=width
    ))


def overlay_attention(
    axis: plt.Axes, base: np.ndarray, attention: np.ndarray, size: tuple[int, int]
) -> None:
    """Overlay one 7x7 attention grid (jet, alpha) onto a base image array.

    参数:
        axis (plt.Axes): 目标坐标轴。
        base (np.ndarray): 底图数组（RGB或灰度）。
        attention (np.ndarray): shape ``(7,7)``，按最大值归一化。
        size (tuple[int, int]): 上采样目标 ``(width, height)`` 像素。
    返回: 无。
    """
    attention_image = Image.fromarray(
        np.uint8(np.clip(attention / max(float(attention.max()), 1e-12), 0, 1) * 255)
    ).resize(size, Image.Resampling.BICUBIC)
    axis.imshow(base)
    axis.imshow(attention_image, cmap="jet", alpha=0.62, vmin=0, vmax=255)


def export_one(
    row: pd.Series, teacher_attention: np.ndarray | None, output_path: Path
) -> None:
    """Render one multi-panel QC figure (2 panels, or 4 with teacher panels).

    参数:
        row (pd.Series): 单图完整记录（预测+坐标+峰值标记）。
        teacher_attention (np.ndarray | None): 类1样本的教师7x7 attention
            （flip0），其余类传None。
        output_path (Path): 输出PNG路径。
    返回: 无。
    """
    with Image.open(PROJECT_ROOT / row.image_relpath) as handle:
        full = handle.convert("RGB")
    width, height = full.size
    crop_pixels = np.array([
        row.base_crop_x1 * width, row.base_crop_y1 * height,
        row.base_crop_x2 * width, row.base_crop_y2 * height,
    ])
    is_cancer = int(row.label) == 1
    lesion_pixels = np.array([
        row.bbox_x1_norm * width, row.bbox_y1_norm * height,
        row.bbox_x2_norm * width, row.bbox_y2_norm * height,
    ]) if is_cancer else None
    student_attention = attention_grid(row)

    panels = 4 if teacher_attention is not None else 2
    figure, axes = plt.subplots(1, panels, figsize=(4 * panels, 4), dpi=160)
    axes = np.atleast_1d(axes)

    axes[0].imshow(full)
    draw_box(axes[0], crop_pixels, "#f5c542")
    if lesion_pixels is not None:
        draw_box(axes[0], lesion_pixels, "#25b04b")
    axes[0].set_title("Full: crop yellow, lesion green")

    overlay_attention(axes[1], np.asarray(full), student_attention, (width, height))
    draw_box(axes[1], crop_pixels, "#f5c542")
    if lesion_pixels is not None:
        draw_box(axes[1], lesion_pixels, "#25b04b")
    axes[1].set_title("C-long student attention")

    if teacher_attention is not None:
        pixels = crop_pixels.round().astype(int)
        pixels[[0, 2]] = np.clip(pixels[[0, 2]], 0, width)
        pixels[[1, 3]] = np.clip(pixels[[1, 3]], 0, height)
        roi = full.crop(tuple(pixels)).resize((224, 224), Image.Resampling.BILINEAR)
        luma = np.asarray(roi.convert("L"))
        relative_bbox = bbox_in_crop(row)
        bbox_224 = relative_bbox * np.array([224, 224, 224, 224])
        axes[2].imshow(luma, cmap="gray", vmin=0, vmax=255)
        draw_box(axes[2], bbox_224, "#25b04b")
        axes[2].set_title("Teacher luma ROI")
        overlay_attention(axes[3], luma, teacher_attention, (224, 224))
        draw_box(axes[3], bbox_224, "#25b04b")
        axes[3].set_title("Teacher attention")

    for axis in axes:
        axis.axis("off")
    naib = row.normalized_aib
    naib_text = f"{naib:.3f}" if pd.notna(naib) else "NA"
    pga_text = str(int(row.pga)) if pd.notna(row.pga) else "NA"
    figure.suptitle(
        f"{Path(row.relative_path).stem} | label={int(row.label)} "
        f"p={row.cancer_probability:.3f} | nAiB={naib_text} PGA={pga_text} | "
        f"peak_in_box={pga_text} peak_in_crop={int(row.peak_in_crop)} "
        f"outer_ring={int(row.peak_outer_ring)}",
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Verify lineage, select the three frozen classes and export QC artifacts."""
    args = parse_args()
    clong_run = args.clong_run.resolve()
    manifest_path = args.manifest.resolve()
    config, predictions, _ = load_and_verify(clong_run, manifest_path)
    manifest_sha = file_sha256(manifest_path)
    cache = load_teacher_cache(args.teacher_cache.resolve(), manifest_sha)
    mg1b_index = pd.read_csv(args.mg1b_qc_index.resolve(), encoding="utf-8-sig")
    if len(mg1b_index) != 36 or not mg1b_index.label.eq(1).all():
        raise ValueError("MG1b QC索引不是预期的36张val癌图")

    selected = select_samples(predictions, mg1b_index)
    counts = {
        name: int(selected.qc_class.str.contains(name).sum()) for name in CLASS_DIRS
    }
    if counts["teacher_student"] != 36:
        raise ValueError(f"类1样本数异常: {counts['teacher_student']}")
    expected_class2 = max(
        int(predictions.loc[predictions.label.eq(1), "pga"].eq(0).sum()),
        min(SPATIAL_FAILURE_TARGET, int(predictions.label.eq(1).sum())),
    )
    if counts["spatial_failure"] != expected_class2:
        raise ValueError(
            f"类2样本数异常: {counts['spatial_failure']} != 预期{expected_class2}"
        )
    if not (1 <= counts["high_risk"] <= HIGH_RISK_FP_COUNT + HIGH_RISK_RING_MAX):
        raise ValueError(f"类3样本数异常: {counts['high_risk']}")
    if len(selected) != len(set(
        item.relative_path for item in selected["row"]
    )):
        raise ValueError("三类样本去重后仍存在重复图")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"QC输出已存在，拒绝覆盖: {output_dir}")
    for name in CLASS_DIRS.values():
        (output_dir / name).mkdir(parents=True)

    index_rows = []
    for item in selected.itertuples(index=False):
        row = item.row
        teacher_attention = None
        if "teacher_student" in item.qc_class:
            entry = cache["entries"][str(row.sha256)]["flip0"]
            teacher_attention = torch.as_tensor(
                entry["attention"], dtype=torch.float32
            ).reshape(GRID_SIZE, GRID_SIZE).numpy().astype(np.float64)
        filename = f"{item.order + 1:02d}_{Path(row.relative_path).stem}_attention.png"
        relative_qc = (Path(CLASS_DIRS[item.primary_class]) / filename).as_posix()
        export_one(row, teacher_attention, output_dir / relative_qc)
        index_rows.append({
            "qc_class": item.qc_class,
            "relative_path": row.relative_path,
            "patient_id": row.patient_id,
            "label": int(row.label),
            "split": row.split,
            "cancer_probability": float(row.cancer_probability),
            "aib": float(row.aib) if pd.notna(row.aib) else np.nan,
            "normalized_aib": (
                float(row.normalized_aib) if pd.notna(row.normalized_aib) else np.nan
            ),
            "pga": float(row.pga) if pd.notna(row.pga) else np.nan,
            "peak_in_lesion_box": (
                bool(row.pga == 1) if pd.notna(row.pga) else ""
            ),
            "peak_in_crop": bool(row.peak_in_crop),
            "peak_outer_ring": bool(row.peak_outer_ring),
            "peak_grid_y": int(row.peak_grid_y),
            "peak_grid_x": int(row.peak_grid_x),
            "source": row.source,
            "size_group": row.size_group,
            "style_group": row.style_group,
            "lesion_area_fraction": (
                float(row.lesion_area_fraction)
                if pd.notna(row.lesion_area_fraction) else np.nan
            ),
            "qc_file": relative_qc,
            **{name: "" for name in MANUAL_COLUMNS},
        })
    index = pd.DataFrame(index_rows)
    index.to_csv(output_dir / "clong_attention_qc_index.csv", index=False,
                 encoding="utf-8-sig")

    qc_config = {
        "stage": "MG2L_Clong_attention_manual_qc",
        "selection_role": "read-only pre-SAE risk audit; no retraining, no reselection",
        "data_declaration": "只读train/val；test/internal test/external未读取",
        "frozen_threshold": FROZEN_THRESHOLD,
        "sample_rules": {
            "teacher_student": "MG1b QC同一批36张val癌图，师生对照",
            "spatial_failure": (
                "val癌图全部PGA失败图；不足"
                f"{SPATIAL_FAILURE_TARGET}张时按nAiB升序补足（去重）"
            ),
            "high_risk": (
                f"冻结阈值下非癌假阳性概率降序前{HIGH_RISK_FP_COUNT}张，"
                f"另加峰值cell位于最外圈且在crop外最多{HIGH_RISK_RING_MAX}张（去重）"
            ),
        },
        "counts": {**counts, "total_unique_images": int(len(index))},
        "clong_run": str(clong_run),
        "clong_checkpoint_sha256": FROZEN_CLONG_CHECKPOINT_SHA256,
        "clong_predictions_sha256": file_sha256(
            clong_run / "val_image_predictions.csv"
        ),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "mg1b_qc_index": str(args.mg1b_qc_index.resolve()),
        "mg1b_qc_index_sha256": file_sha256(args.mg1b_qc_index.resolve()),
        "teacher_cache": str(args.teacher_cache.resolve()),
        "teacher_cache_sha256": file_sha256(args.teacher_cache.resolve()),
        "manual_columns": MANUAL_COLUMNS,
        "test_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
    }
    (output_dir / "qc_config.json").write_text(
        json.dumps(qc_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"C-long注意力QC已生成: {output_dir}")
    print(f"样本: {counts}, 总计{len(index)}张唯一图")


if __name__ == "__main__":
    main()
