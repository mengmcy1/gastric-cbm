#!/usr/bin/env python3
"""用原图 FOV 预测复核第一阶段裁剪是否误删暗部有效视野。

本脚本只生成风险指标、保守裁剪候选和人工对照图，不覆盖第一阶段结果，也不自动
决定正式裁剪框。正式采用前必须在分中心、标签和设备风格抽样上完成人工验收。
"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from config import MODEL_PATH, PROJECT_ROOT, STAGE1_OUTPUT
from fov_mask_v1 import fill_external_contour
from onnx_infer import (
    clean_component,
    create_inference_session,
    encode_and_save,
    extract_probability,
    inference_backend_name,
    largest_component,
    preprocess,
    read_image,
    verify_session,
)


PANEL_WIDTH = 460
PANEL_HEIGHT = 380
COMPARISON_PROFILES = (
    ("A_敏感_框外1.5pct_单边2pct", 0.015, 0.02),
    ("B_中等_框外3pct_单边4pct", 0.03, 0.04),
    ("C_保守_框外5pct_单边6pct", 0.05, 0.06),
    ("D_强风险_框外10pct_单边10pct", 0.10, 0.10),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="用原图 FOV 掩膜筛查暗部有效视野被第一阶段裁掉的风险。"
    )
    parser.add_argument("--stage1-mapping", type=Path, default=STAGE1_OUTPUT / "mapping.csv")
    parser.add_argument("--onnx", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=STAGE1_OUTPUT.parent / "01b_暗部裁剪安全复核",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--outside-ratio-threshold",
        type=float,
        default=0.05,
        help="预测有效视野落在当前框外的比例达到该值时进入复核。",
    )
    parser.add_argument(
        "--side-extension-ratio-threshold",
        type=float,
        default=0.06,
        help="FOV 外接框单边超出当前框达到原图对应边比例时进入复核。",
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.01,
        help="保守候选在当前框与 FOV 框并集外追加的原图短边比例。",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=200,
        help="最多保存多少张风险三联图；0 表示不保存。",
    )
    return parser.parse_args()


def validate_args(args):
    if not args.stage1_mapping.is_file():
        raise FileNotFoundError(f"第一阶段映射表不存在：{args.stage1_mapping}")
    if not args.onnx.is_file():
        raise FileNotFoundError(f"FOV ONNX 模型不存在：{args.onnx}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录不是空目录，拒绝覆盖：{args.output}")
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold 必须位于 (0, 1)")
    if not 0 <= args.outside_ratio_threshold <= 1:
        raise ValueError("--outside-ratio-threshold 必须位于 [0, 1]")
    if not 0 <= args.side_extension_ratio_threshold <= 1:
        raise ValueError("--side-extension-ratio-threshold 必须位于 [0, 1]")
    if not 0 <= args.margin_ratio <= 0.1:
        raise ValueError("--margin-ratio 必须位于 [0, 0.1]")
    if args.preview_limit < 0:
        raise ValueError("--preview-limit 不能为负数")


def resolve_existing_path(value, mapping_path):
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((PROJECT_ROOT / path, mapping_path.parent / path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"映射表引用的文件不存在：{value}")


def predict_fov_mask(session, input_name, image, threshold):
    probability = extract_probability(
        session.run(None, {input_name: preprocess(image)})[0]
    )
    raw_mask = (probability > threshold).astype(np.uint8)
    component, component_count = largest_component(raw_mask)
    cleaned = clean_component(component, image, erode_pixels=0, black_threshold=12)
    filled = fill_external_contour(cleaned)
    height, width = image.shape[:2]
    mask = cv2.resize(filled, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.uint8), component_count


def mask_bbox(mask):
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("FOV 掩膜为空")
    x, y, width, height = cv2.boundingRect(points)
    return x, y, x + width, y + height


def clamp_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    x1 = min(max(0, int(x1)), width - 1)
    y1 = min(max(0, int(y1)), height - 1)
    x2 = min(max(x1 + 1, int(x2)), width)
    y2 = min(max(y1 + 1, int(y2)), height)
    return x1, y1, x2, y2


def directional_bbox(
    current,
    fov,
    width,
    height,
    margin_ratio,
    extensions,
    extension_threshold,
    lock_bottom,
):
    margin = round(min(width, height) * margin_ratio)
    candidate = list(current)
    expanded_sides = []
    blocked_sides = []
    if extensions["left"] >= extension_threshold:
        candidate[0] = min(current[0], fov[0]) - margin
        expanded_sides.append("left")
    if extensions["top"] >= extension_threshold:
        candidate[1] = min(current[1], fov[1]) - margin
        expanded_sides.append("top")
    if extensions["right"] >= extension_threshold:
        candidate[2] = max(current[2], fov[2]) + margin
        expanded_sides.append("right")
    if extensions["bottom"] >= extension_threshold:
        if lock_bottom:
            blocked_sides.append("bottom_progress_bar")
        else:
            candidate[3] = max(current[3], fov[3]) + margin
            expanded_sides.append("bottom")
    return (
        clamp_bbox(candidate, width, height),
        expanded_sides,
        blocked_sides,
    )


def as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fit_panel(image):
    scale = min(PANEL_WIDTH / image.shape[1], PANEL_HEIGHT / image.shape[0])
    width = max(1, round(image.shape[1] * scale))
    height = max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    offset_x = (PANEL_WIDTH - width) // 2
    offset_y = (PANEL_HEIGHT - height) // 2
    panel[offset_y : offset_y + height, offset_x : offset_x + width] = resized
    return panel, scale, offset_x, offset_y


def draw_box(panel, bbox, scale, offset_x, offset_y, color, thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(
        panel,
        (offset_x + round(x1 * scale), offset_y + round(y1 * scale)),
        (offset_x + round(x2 * scale), offset_y + round(y2 * scale)),
        color,
        thickness,
    )


def make_preview(image, mask, current, conservative, outside_ratio, reasons):
    overlay, scale, offset_x, offset_y = fit_panel(image)
    mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if mask_contours:
        contour = max(mask_contours, key=cv2.contourArea).astype(np.float32)
        contour[:, :, 0] = contour[:, :, 0] * scale + offset_x
        contour[:, :, 1] = contour[:, :, 1] * scale + offset_y
        cv2.drawContours(overlay, [contour.astype(np.int32)], -1, (255, 255, 0), 1)
    draw_box(overlay, current, scale, offset_x, offset_y, (0, 255, 0), 2)
    draw_box(overlay, conservative, scale, offset_x, offset_y, (0, 165, 255), 2)

    x1, y1, x2, y2 = current
    current_panel, _, _, _ = fit_panel(image[y1:y2, x1:x2])
    x1, y1, x2, y2 = conservative
    conservative_panel, _, _, _ = fit_panel(image[y1:y2, x1:x2])
    masked = image.copy()
    masked[mask == 0] = 0
    masked_panel, _, _, _ = fit_panel(masked[y1:y2, x1:x2])
    preview = cv2.hconcat(
        (overlay, current_panel, conservative_panel, masked_panel)
    )
    labels = (
        "original: green=current orange=directional cyan=FOV",
        "current",
        "directional before FOV",
        "directional after FOV",
    )
    for index, label in enumerate(labels):
        cv2.putText(
            preview,
            label,
            (index * PANEL_WIDTH + 8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        preview,
        f"outside={outside_ratio:.4f} review={'|'.join(reasons)}",
        (8, PANEL_HEIGHT - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    return preview


def write_contact_sheets(preview_paths, output):
    if not preview_paths:
        return []
    pages = []
    page_size = 6
    thumb_width = 690
    for page_index in range(math.ceil(len(preview_paths) / page_size)):
        paths = preview_paths[page_index * page_size : (page_index + 1) * page_size]
        thumbs = []
        for path in paths:
            image = read_image(path)
            scale = thumb_width / image.shape[1]
            thumbs.append(
                cv2.resize(
                    image,
                    (thumb_width, max(1, round(image.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            )
        row_height = max(item.shape[0] for item in thumbs)
        canvas = np.full((row_height * len(thumbs), thumb_width, 3), 255, np.uint8)
        for index, thumb in enumerate(thumbs):
            canvas[index * row_height : index * row_height + thumb.shape[0]] = thumb
        page_path = output / "人工复核分页" / f"page_{page_index + 1:02d}.jpg"
        encode_and_save(page_path, canvas)
        pages.append(str(page_path))
    return pages


def profile_hit(row, outside_threshold, side_threshold):
    if row.get("review_status") == "error":
        return False
    outside = float(row["fov_outside_crop_ratio"])
    side_max = max(
        float(row[key])
        for key in (
            "fov_extend_left_ratio",
            "fov_extend_top_ratio",
            "fov_extend_right_ratio",
            "fov_extend_bottom_ratio",
        )
    )
    return outside >= outside_threshold or side_max >= side_threshold


def write_threshold_comparison(rows, output):
    comparison_root = output / "多阈值联图"
    summaries = []
    pages_by_profile = {}
    for profile_name, outside_threshold, side_threshold in COMPARISON_PROFILES:
        selected = [
            row
            for row in rows
            if row.get("preview")
            and profile_hit(row, outside_threshold, side_threshold)
        ]
        selected.sort(
            key=lambda row: float(row["fov_outside_crop_ratio"]), reverse=True
        )
        profile_output = comparison_root / profile_name
        pages = write_contact_sheets(
            [Path(row["preview"]) for row in selected], profile_output
        )
        summaries.append(
            {
                "profile": profile_name,
                "outside_ratio_threshold": outside_threshold,
                "side_extension_ratio_threshold": side_threshold,
                "candidate_count": len(selected),
                "total_count": len(rows),
                "candidate_ratio": len(selected) / len(rows),
            }
        )
        pages_by_profile[profile_name] = pages

    summary_path = comparison_root / "阈值候选数量汇总.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return summaries, pages_by_profile


def process_row(session, input_name, source_row, mapping_path, args, save_preview):
    source = resolve_existing_path(source_row["source"], mapping_path)
    current_crop = resolve_existing_path(source_row["crop"], mapping_path)
    image = read_image(source)
    height, width = image.shape[:2]
    current = clamp_bbox(
        tuple(int(source_row[key]) for key in ("crop_x1", "crop_y1", "crop_x2", "crop_y2")),
        width,
        height,
    )
    mask, component_count = predict_fov_mask(
        session, input_name, image, args.threshold
    )
    fov = mask_bbox(mask)
    total_fov = int(mask.sum())
    inside_fov = int(mask[current[1] : current[3], current[0] : current[2]].sum())
    outside_ratio = (total_fov - inside_fov) / total_fov
    extensions = {
        "left": max(0, current[0] - fov[0]) / width,
        "top": max(0, current[1] - fov[1]) / height,
        "right": max(0, fov[2] - current[2]) / width,
        "bottom": max(0, fov[3] - current[3]) / height,
    }
    reasons = []
    if outside_ratio >= args.outside_ratio_threshold:
        reasons.append("fov_outside_crop")
    for side, ratio in extensions.items():
        if ratio >= args.side_extension_ratio_threshold:
            reasons.append(f"fov_extends_{side}")
    mask_coverage = total_fov / mask.size
    if mask_coverage < 0.2 or mask_coverage > 0.99:
        reasons.append("fov_prediction_unreliable")

    comparison_outside_threshold = min(item[1] for item in COMPARISON_PROFILES)
    comparison_side_threshold = min(item[2] for item in COMPARISON_PROFILES)
    comparison_hit = (
        outside_ratio >= comparison_outside_threshold
        or max(extensions.values()) >= comparison_side_threshold
    )
    conservative, expanded_sides, blocked_sides = directional_bbox(
        current,
        fov,
        width,
        height,
        args.margin_ratio,
        extensions,
        comparison_side_threshold,
        lock_bottom=as_bool(source_row.get("progress_bar_detected", False)),
    )
    crop_root = (mapping_path.parent / "crops").resolve()
    try:
        relative = current_crop.resolve().relative_to(crop_root)
    except ValueError as error:
        raise ValueError(
            f"第一阶段输出不在预期crops目录内: {current_crop}"
        ) from error
    candidate_path = ""
    masked_candidate_path = ""
    preview_path = ""
    if reasons or comparison_hit:
        candidate_file = args.output / "保守裁剪候选" / relative
        x1, y1, x2, y2 = conservative
        encode_and_save(candidate_file, image[y1:y2, x1:x2])
        candidate_path = str(candidate_file)
        masked = image.copy()
        masked[mask == 0] = 0
        masked_candidate_file = args.output / "保守裁剪_FOV遮罩候选" / relative
        encode_and_save(masked_candidate_file, masked[y1:y2, x1:x2])
        masked_candidate_path = str(masked_candidate_file)
        if save_preview:
            preview_file = args.output / "三联对照" / relative
            encode_and_save(
                preview_file,
                make_preview(image, mask, current, conservative, outside_ratio, reasons),
            )
            preview_path = str(preview_file)

    return {
        "source": str(source),
        "relative_path": source_row["relative_path"],
        "current_crop": str(current_crop),
        "conservative_candidate": candidate_path,
        "conservative_fov_masked_candidate": masked_candidate_path,
        "preview": preview_path,
        "source_width": width,
        "source_height": height,
        "current_x1": current[0],
        "current_y1": current[1],
        "current_x2": current[2],
        "current_y2": current[3],
        "fov_x1": fov[0],
        "fov_y1": fov[1],
        "fov_x2": fov[2],
        "fov_y2": fov[3],
        "safe_x1": conservative[0],
        "safe_y1": conservative[1],
        "safe_x2": conservative[2],
        "safe_y2": conservative[3],
        "fov_mask_coverage": mask_coverage,
        "fov_outside_crop_ratio": outside_ratio,
        "fov_extend_left_ratio": extensions["left"],
        "fov_extend_top_ratio": extensions["top"],
        "fov_extend_right_ratio": extensions["right"],
        "fov_extend_bottom_ratio": extensions["bottom"],
        "raw_component_count": component_count,
        "expanded_sides": "|".join(expanded_sides),
        "blocked_sides": "|".join(blocked_sides),
        "review_status": "pending" if reasons else "auto_pass_candidate",
        "review_reasons": "|".join(reasons),
        "decision": "pending" if reasons else "keep_current",
    }


def main():
    args = parse_args()
    validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)
    with args.stage1_mapping.open("r", encoding="utf-8-sig", newline="") as file:
        source_rows = list(csv.DictReader(file))
    if not source_rows:
        raise ValueError("第一阶段映射表为空")

    session = create_inference_session(args.onnx)
    input_name = verify_session(session)
    rows = []
    preview_paths = []
    risk_count = 0
    for index, source_row in enumerate(source_rows, start=1):
        save_preview = risk_count < args.preview_limit if args.preview_limit else False
        try:
            row = process_row(
                session, input_name, source_row, args.stage1_mapping, args, save_preview
            )
        except Exception as error:
            row = {
                "source": source_row.get("source", ""),
                "relative_path": source_row.get("relative_path", ""),
                "review_status": "error",
                "review_reasons": f"processing_error:{error}",
                "decision": "pending",
            }
        rows.append(row)
        if row["review_status"] == "pending":
            risk_count += 1
            if row.get("preview"):
                preview_paths.append(Path(row["preview"]))
        print(
            f"[{index}/{len(source_rows)}] {row['relative_path']} -> "
            f"{row['review_status']} {row['review_reasons']}"
        )

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    mapping_output = args.output / "mapping.csv"
    with mapping_output.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    pages = write_contact_sheets(preview_paths, args.output)
    threshold_summaries, threshold_pages = write_threshold_comparison(
        rows, args.output
    )
    counts = Counter(row["review_status"] for row in rows)
    manifest = {
        "stage": "dark_crop_safety_review_v1",
        "purpose": "candidate_review_only_no_automatic_replacement",
        "stage1_mapping": str(args.stage1_mapping.resolve()),
        "onnx": str(args.onnx.resolve()),
        "backend": inference_backend_name(session),
        "fov_threshold": args.threshold,
        "outside_ratio_threshold": args.outside_ratio_threshold,
        "side_extension_ratio_threshold": args.side_extension_ratio_threshold,
        "margin_ratio": args.margin_ratio,
        "counts": dict(counts),
        "contact_sheets": pages,
        "threshold_comparison": threshold_summaries,
        "threshold_comparison_pages": threshold_pages,
    }
    with (args.output / "run_manifest.json").open("x", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(f"完成：{len(rows)} 张；状态：{dict(counts)}")
    print(f"映射表：{mapping_output}")
    if pages:
        print(f"人工复核分页：{args.output / '人工复核分页'}")
    print("多阈值候选数：")
    for summary in threshold_summaries:
        print(f"  {summary['profile']}: {summary['candidate_count']}/{len(rows)}")


if __name__ == "__main__":
    main()
