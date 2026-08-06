#!/usr/bin/env python3
"""对已裁剪内镜图像执行第二阶段 FOV 分割和圆外统一遮黑。

本脚本只生成派生结果：
- masks/：二值有效视野掩膜；
- masked/：保持输入尺寸、将掩膜外统一填为纯黑的 JPEG；
- previews/：原图、掩膜和遮黑结果三联预览；
- mapping.csv：输入、输出、质量指标和复核原因。

原始图像及第一阶段裁剪图始终只读，非空输出目录拒绝覆盖。
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from config import STAGE1_CROPS, STAGE2_OUTPUT
from onnx_infer import (
    DEFAULT_MODEL,
    clean_component,
    create_inference_session,
    encode_and_save,
    extract_probability,
    inference_backend_name,
    largest_component,
    list_images,
    preprocess,
    read_image,
    relative_image_path,
    verify_session,
)


LOW_COVERAGE = 0.45
HIGH_COVERAGE = 0.98
SECONDARY_COMPONENT_MIN_RATIO = 0.003
PREVIEW_PANEL_WIDTH = 480
PREVIEW_PANEL_HEIGHT = 400


def parse_args():
    """命令行参数：onnx模型/输入/输出/阈值/内缩像素。"""
    parser = argparse.ArgumentParser(
        description="使用 shiye_V1 分割已裁剪图像的有效视野，并统一遮黑 FOV 外区域。"
    )
    parser.add_argument("--onnx", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, default=STAGE1_CROPS)
    parser.add_argument("--output", type=Path, default=STAGE2_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--erode-pixels",
        type=int,
        default=1,
        help="在 256×256 模型掩膜上向内收缩的像素数。",
    )
    parser.add_argument(
        "--black-threshold",
        type=int,
        default=12,
        help="剔除与边缘连通的近黑区域；0 表示关闭。",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=30,
        help="最多保存多少张三联预览；0 表示不保存。",
    )
    return parser.parse_args()


def validate_args(args):
    """校验输入存在、输出目录非空拒绝覆盖。"""
    if not args.onnx.is_file():
        raise FileNotFoundError(f"ONNX 模型不存在：{args.onnx}")
    if not args.input.exists():
        raise FileNotFoundError(f"输入不存在：{args.input}")
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold 必须位于 (0, 1)")
    if args.erode_pixels < 0:
        raise ValueError("--erode-pixels 不能为负数")
    if not 0 <= args.black_threshold <= 255:
        raise ValueError("--black-threshold 必须位于 [0, 255]")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须为正整数")
    if args.preview_limit < 0:
        raise ValueError("--preview-limit 不能为负数")

    input_resolved = args.input.resolve()
    output_resolved = args.output.resolve()
    if args.input.is_dir() and (
        output_resolved == input_resolved or input_resolved in output_resolved.parents
    ):
        raise ValueError("输出目录不能位于输入目录内部")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录不是空目录，拒绝覆盖：{args.output}")


def fill_external_contour(mask):
    """填充掩膜最大外轮廓内部（圆内区域全保留）。"""
    """填充最大外轮廓内部孔洞，避免暗腔等医学区域被误遮黑。"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise ValueError("有效视野掩膜没有外轮廓")
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
    return filled


def component_metrics(raw_mask):
    """掩膜连通域统计：主/次连通域覆盖率。"""
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        raw_mask.astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        return 0, 0.0
    areas = sorted(
        (int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)),
        reverse=True,
    )
    secondary_area = sum(
        area
        for area in areas[1:]
        if area / raw_mask.size >= SECONDARY_COMPONENT_MIN_RATIO
    )
    return count - 1, secondary_area / raw_mask.size


def fit_panel(image):
    """按面板尺寸等比缩放入画布（预览用）。"""
    scale = min(
        PREVIEW_PANEL_WIDTH / image.shape[1],
        PREVIEW_PANEL_HEIGHT / image.shape[0],
    )
    width = max(1, round(image.shape[1] * scale))
    height = max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    panel = np.zeros(
        (PREVIEW_PANEL_HEIGHT, PREVIEW_PANEL_WIDTH, 3),
        dtype=np.uint8,
    )
    x = (PREVIEW_PANEL_WIDTH - width) // 2
    y = (PREVIEW_PANEL_HEIGHT - height) // 2
    panel[y : y + height, x : x + width] = resized
    return panel


def make_preview(image, display_mask, masked, review_reasons):
    """原图/掩膜/遮黑结果三联预览。"""
    mask_visual = cv2.cvtColor(display_mask * 255, cv2.COLOR_GRAY2BGR)
    panels = [fit_panel(item) for item in (image, mask_visual, masked)]
    preview = cv2.hconcat(panels)
    labels = ("input", "mask", "masked")
    for index, label in enumerate(labels):
        cv2.putText(
            preview,
            label,
            (index * PREVIEW_PANEL_WIDTH + 8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    review_label = (
        "review=" + "|".join(review_reasons)
        if review_reasons
        else "review=auto_pass_candidate"
    )
    cv2.putText(
        preview,
        review_label,
        (8, PREVIEW_PANEL_HEIGHT - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255) if review_reasons else (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return preview


def build_review_reasons(mask_coverage, secondary_component_ratio):
    """按覆盖率/次连通域生成复核原因。"""
    reasons = []
    if mask_coverage < LOW_COVERAGE:
        reasons.append("low_mask_coverage")
    if mask_coverage > HIGH_COVERAGE:
        reasons.append("high_mask_coverage")
    if secondary_component_ratio >= SECONDARY_COMPONENT_MIN_RATIO:
        reasons.append("secondary_fov_component")
    return reasons


def process_one(session, input_name, image_path, relative, args, save_preview):
    """单张：推理→掩膜清洗→视野外遮黑→落盘，返回mapping行。"""
    image = read_image(image_path)
    probability = extract_probability(
        session.run(None, {input_name: preprocess(image)})[0]
    )
    raw_mask = (probability > args.threshold).astype(np.uint8)
    raw_component_count, secondary_component_ratio = component_metrics(raw_mask)
    component, _ = largest_component(raw_mask)
    cleaned = clean_component(
        component,
        image,
        args.erode_pixels,
        args.black_threshold,
    )
    filled = fill_external_contour(cleaned)

    height, width = image.shape[:2]
    display_mask = cv2.resize(
        filled,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    masked = image.copy()
    masked[display_mask == 0] = 0
    mask_coverage = float(display_mask.mean())
    removed_pixel_ratio = 1.0 - mask_coverage
    removed_nonblack_ratio = float(
        ((display_mask == 0) & (image.max(axis=2) > args.black_threshold)).mean()
    )
    review_reasons = build_review_reasons(
        mask_coverage,
        secondary_component_ratio,
    )

    mask_path = args.output / "masks" / relative.with_suffix(".png")
    masked_path = args.output / "masked" / relative.with_suffix(".jpg")
    encode_and_save(mask_path, display_mask * 255)
    encode_and_save(masked_path, masked)

    preview_path = ""
    if save_preview:
        preview_file = args.output / "previews" / relative.with_suffix(".jpg")
        encode_and_save(
            preview_file,
            make_preview(image, display_mask, masked, review_reasons),
        )
        preview_path = str(preview_file)

    return {
        "source": str(image_path),
        "relative_path": str(relative),
        "mask": str(mask_path),
        "masked": str(masked_path),
        "preview": preview_path,
        "width": width,
        "height": height,
        "threshold": args.threshold,
        "erode_pixels": args.erode_pixels,
        "black_threshold": args.black_threshold,
        "mask_coverage": mask_coverage,
        "removed_pixel_ratio": removed_pixel_ratio,
        "removed_nonblack_ratio": removed_nonblack_ratio,
        "raw_component_count": raw_component_count,
        "secondary_component_ratio": secondary_component_ratio,
        "review_status": "pending" if review_reasons else "auto_pass_candidate",
        "review_reasons": "|".join(review_reasons),
    }


def error_row(image_path, relative, error):
    """单张失败时的空错误行（批量不中断）。"""
    return {
        "source": str(image_path),
        "relative_path": str(relative),
        "mask": "",
        "masked": "",
        "preview": "",
        "width": "",
        "height": "",
        "threshold": "",
        "erode_pixels": "",
        "black_threshold": "",
        "mask_coverage": "",
        "removed_pixel_ratio": "",
        "removed_nonblack_ratio": "",
        "raw_component_count": "",
        "secondary_component_ratio": "",
        "review_status": "error",
        "review_reasons": f"processing_error:{error}",
    }


def main():
    """入口：解析→建session→逐张遮罩→写mapping/manifest。"""
    args = parse_args()
    validate_args(args)
    image_paths = list_images(args.input)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"未找到图像：{args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    session = create_inference_session(args.onnx)
    print(f"推理后端：{inference_backend_name(session)}")
    input_name = verify_session(session)

    rows = []
    for index, image_path in enumerate(image_paths, start=1):
        relative = relative_image_path(image_path, args.input)
        try:
            row = process_one(
                session,
                input_name,
                image_path,
                relative,
                args,
                save_preview=index <= args.preview_limit,
            )
        except Exception as error:
            row = error_row(image_path, relative, error)
        rows.append(row)
        print(
            f"[{index}/{len(image_paths)}] {relative} -> "
            f"{row['review_status']} {row['review_reasons']}"
        )

    mapping_path = args.output / "mapping.csv"
    with mapping_path.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["review_status"] for row in rows)
    print(f"完成：{len(rows)} 张；复核状态：{dict(counts)}")
    print(f"映射表：{mapping_path}")


if __name__ == "__main__":
    main()
