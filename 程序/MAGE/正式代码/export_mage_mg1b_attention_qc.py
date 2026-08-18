#!/usr/bin/env python3
"""Export stratified MG1b attention overlays for manual clinical QC.

This is a read-only post-selection diagnostic. It uses the frozen validation
predictions and v3 manifest, samples evenly across normalized-AiB ranks within
each lesion-size stratum, and never participates in checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from build_mage_teacher_roi_manifest import PROJECT_ROOT, file_sha256
from train_mage_mg1b_attention_teacher import (
    DEFAULT_MANIFEST,
    GRID_SIZE,
    bbox_in_crop,
    lesion_group,
)


DEFAULT_RUN = PROJECT_ROOT / (
    "结果/MAGE/MG1b注意力池化教师_20260818/正式验证集筛选/"
    "mg1b_attention_efficientnet_b0_seed42"
)


def parse_args() -> argparse.Namespace:
    """Parse frozen MG1b product and deterministic QC sampling options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--images-per-group", type=int, default=12)
    return parser.parse_args()


def evenly_spaced_rows(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Select deterministic rows across the full normalized-AiB ranking."""
    ordered = frame.sort_values(
        ["normalized_aib", "relative_path"], na_position="last"
    ).reset_index(drop=True)
    if len(ordered) <= count:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return ordered.iloc[np.unique(positions)].reset_index(drop=True)


def crop_pixels(row: pd.Series, width: int, height: int) -> np.ndarray:
    """Convert normalized v3 crop coordinates to clipped source-image pixels."""
    box = np.array([
        row.base_crop_x1, row.base_crop_y1, row.base_crop_x2, row.base_crop_y2
    ])
    pixels = (box * np.array([width, height, width, height])).round().astype(int)
    pixels[[0, 2]] = np.clip(pixels[[0, 2]], 0, width)
    pixels[[1, 3]] = np.clip(pixels[[1, 3]], 0, height)
    return pixels


def draw_box(axis: plt.Axes, box: np.ndarray, color: str, width: float = 2.0) -> None:
    """Draw one xyxy rectangle without adding a legend entry."""
    x1, y1, x2, y2 = map(float, box)
    axis.add_patch(plt.Rectangle(
        (x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=width
    ))


def export_one(row: pd.Series, output_path: Path) -> None:
    """Save full image, luma ROI and attention overlay for one validation cancer image."""
    with Image.open(PROJECT_ROOT / row.image_relpath) as handle:
        full = handle.convert("RGB")
    width, height = full.size
    pixels = crop_pixels(row, width, height)
    roi = full.crop(tuple(pixels)).resize((224, 224), Image.Resampling.BILINEAR)
    luma = np.asarray(roi.convert("L"))
    relative_bbox = bbox_in_crop(row)
    bbox_224 = relative_bbox * np.array([224, 224, 224, 224])
    attention = np.array([
        row[f"attention_{grid_y}_{grid_x}"]
        for grid_y in range(GRID_SIZE) for grid_x in range(GRID_SIZE)
    ]).reshape(GRID_SIZE, GRID_SIZE)
    attention_image = Image.fromarray(
        np.uint8(np.clip(attention / max(float(attention.max()), 1e-12), 0, 1) * 255)
    ).resize((224, 224), Image.Resampling.BICUBIC)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=160)
    axes[0].imshow(full)
    draw_box(axes[0], pixels, "#f5c542")
    lesion_pixels = np.array([
        row.bbox_x1_norm * width, row.bbox_y1_norm * height,
        row.bbox_x2_norm * width, row.bbox_y2_norm * height,
    ])
    draw_box(axes[0], lesion_pixels, "#25b04b")
    axes[0].set_title("Full: crop yellow, lesion green")

    axes[1].imshow(luma, cmap="gray", vmin=0, vmax=255)
    draw_box(axes[1], bbox_224, "#25b04b")
    axes[1].set_title("Teacher luma ROI")

    axes[2].imshow(luma, cmap="gray", vmin=0, vmax=255)
    axes[2].imshow(attention_image, cmap="jet", alpha=0.62, vmin=0, vmax=255)
    draw_box(axes[2], bbox_224, "#25b04b")
    axes[2].set_title(
        f"Attention: AiB={row.aib:.3f}, nAiB={row.normalized_aib:.3f}, PGA={int(row.pga)}"
    )
    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        f"row={int(row.row_index)} | {row.lesion_size_group} | "
        f"p(cancer)={row.cancer_probability:.3f}",
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Validate lineage, sample all lesion strata and export manual-QC artifacts."""
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if not config.get("passed_mg1b"):
        raise ValueError("只允许从通过MG1b门槛的正式产品导出QC")
    manifest_path = args.manifest.resolve()
    if file_sha256(manifest_path) != config["manifest_sha256"]:
        raise ValueError("QC manifest SHA与MG1b正式产品不一致")
    predictions = pd.read_csv(
        run_dir / "val_image_predictions.csv", encoding="utf-8-sig",
        dtype={"patient_id": str}, float_precision="round_trip",
    )
    manifest = pd.read_csv(
        manifest_path, encoding="utf-8-sig", dtype={"patient_id": str},
        float_precision="round_trip",
    )
    attention_columns = [
        f"attention_{grid_y}_{grid_x}"
        for grid_y in range(GRID_SIZE) for grid_x in range(GRID_SIZE)
    ]
    missing = set(attention_columns) - set(predictions.columns)
    if missing:
        raise ValueError(f"MG1b预测缺少attention列: {sorted(missing)[:3]}")
    source_columns = [
        "relative_path", "image_relpath", "base_crop_x1", "base_crop_y1",
        "base_crop_x2", "base_crop_y2", "bbox_x1_norm", "bbox_y1_norm",
        "bbox_x2_norm", "bbox_y2_norm",
    ]
    cancer = predictions.loc[predictions.label.eq(1)].merge(
        manifest[source_columns], on="relative_path", how="left", validate="one_to_one"
    )
    bounds = tuple(config["lesion_tercile_bounds_from_train"])
    cancer["lesion_size_group"] = cancer.lesion_area.map(
        lambda value: lesion_group(float(value), bounds)
    )

    output_dir = run_dir / "attention_qc"
    if output_dir.exists():
        raise FileExistsError(f"QC输出已存在，拒绝覆盖: {output_dir}")
    selected_frames = []
    output_dir.mkdir(parents=True)
    for group_name in ("small", "medium", "large"):
        group = evenly_spaced_rows(
            cancer.loc[cancer.lesion_size_group.eq(group_name)], args.images_per_group
        )
        group_dir = output_dir / group_name
        group_dir.mkdir()
        for rank, (_, row) in enumerate(group.iterrows(), start=1):
            filename = f"{rank:02d}_{Path(row.relative_path).stem}_attention.png"
            export_one(row, group_dir / filename)
            record = row.to_dict()
            record["qc_file"] = str((Path(group_name) / filename).as_posix())
            selected_frames.append(record)
    selected = pd.DataFrame(selected_frames)
    selected.to_csv(output_dir / "attention_qc_index.csv", index=False, encoding="utf-8-sig")
    summary = {
        "stage": "MG1b_attention_manual_qc",
        "selection_role": "post-selection diagnostic only",
        "checkpoint_sha256": config["checkpoint_sha256"],
        "manifest_sha256": config["manifest_sha256"],
        "images": int(len(selected)),
        "groups": selected.lesion_size_group.value_counts().to_dict(),
        "test_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
    }
    (output_dir / "qc_config.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"MG1b人工QC已生成: {output_dir} ({len(selected)}张)")


if __name__ == "__main__":
    main()
