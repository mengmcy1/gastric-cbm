#!/usr/bin/env python3
"""基于亮度轮廓和四边扫描的内镜图像裁剪（gastric-cbm迁移版）。

特点：
- 递归读取患者目录并保留相对路径；
- 原始图像只读，输出到独立结果目录；
- 不提供整目录覆盖/删除选项；
- 裁剪后保持原始宽高比，不强制拉伸成正方形；
- 记录坐标、质量指标和疑似界面残留，供人工复核。
"""

import argparse
import csv
import hashlib
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from config import INPUT_ROOT, STAGE1_OUTPUT


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
DETECT_MAX_SIDE = 512
FOREGROUND_THRESHOLD = 10
BLUR_KERNEL = 5
CLOSE_KERNEL_RATIO = 0.02
CENTER_WINDOW_RATIO = 0.50
MIN_CONTOUR_AREA_RATIO = 0.25
MIN_BBOX_DIM_RATIO = 0.45
MAX_CENTER_OFFSET_RATIO = 0.20
INSET_RATIO = 0.015
FULL_FRAME_MARGIN_RATIO = 0.01

EDGE_BRIGHTNESS_THRESHOLD = 25
EDGE_CONTENT_PERCENTILE = 50
EDGE_SCAN_MAX_RATIO = 0.35
EDGE_BAND_START = 0.20
EDGE_BAND_END = 0.80
EDGE_CONTENT_RUN_RATIO = 0.01
EDGE_EXPAND_RATIO = 0.015
EDGE_ALWAYS_ALLOW_TRIM_RATIO = 0.03
EDGE_WIDE_TRIM_BLACK_RATIO = 0.12
MIN_REFINED_DIM_RATIO = 0.45

PROGRESS_SEARCH_START_RATIO = 0.78
PROGRESS_MAX_BOUNDARY_RATIO = 0.99
PROGRESS_MIN_TAIL_RATIO = 0.015
PROGRESS_MIN_JUMP = 14.0
PROGRESS_MAX_TAIL_STD = 12.0

BLACK_THRESHOLD = 10
HIGH_BLACK_RATIO = 0.12
SECONDARY_COMPONENT_MIN_RATIO = 0.003
OVERLAY_MAX_SATURATION = 55
OVERLAY_MIN_VALUE = 155
OVERLAY_EDGE_X_RATIO = 0.15
OVERLAY_EDGE_Y_RATIO = 0.12
OVERLAY_COMPONENT_MAX_AREA_RATIO = 0.002
OVERLAY_COMPONENT_MAX_WIDTH_RATIO = 0.12
OVERLAY_COMPONENT_MAX_HEIGHT_RATIO = 0.08
OVERLAY_COMPONENT_MIN_COUNT = 10
OVERLAY_COMPONENT_MIN_AREA_RATIO = 0.0001
JPEG_QUALITY = 95
PREVIEW_PANEL_WIDTH = 560
PREVIEW_PANEL_HEIGHT = 420


def parse_args():
    parser = argparse.ArgumentParser(
        description="基于亮度轮廓和真正四边扫描的内镜图像安全裁剪。"
    )
    parser.add_argument("--input", type=Path, default=INPUT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=STAGE1_OUTPUT,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="患者抽样和每患者限额之后的全局图像上限。",
    )
    parser.add_argument(
        "--patient-limit",
        type=int,
        default=None,
        help="从患者目录排序范围内等间距选择多少位患者。",
    )
    parser.add_argument(
        "--images-per-patient",
        type=int,
        default=None,
        help="每位患者最多选择多少张图像。",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=20,
        help="最多保存多少张裁剪前后预览；0 表示不保存。",
    )
    return parser.parse_args()


def validate_args(args):
    if not args.input.exists():
        raise FileNotFoundError(f"输入不存在：{args.input}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须为正整数")
    if args.patient_limit is not None and args.patient_limit <= 0:
        raise ValueError("--patient-limit 必须为正整数")
    if args.images_per_patient is not None and args.images_per_patient <= 0:
        raise ValueError("--images-per-patient 必须为正整数")
    if args.input.is_file() and (
        args.patient_limit is not None or args.images_per_patient is not None
    ):
        raise ValueError("单文件输入不能使用患者抽样参数")
    if args.preview_limit < 0:
        raise ValueError("--preview-limit 不能为负数")
    input_resolved = args.input.resolve()
    output_resolved = args.output.resolve()
    if args.input.is_dir() and (
        output_resolved == input_resolved or input_resolved in output_resolved.parents
    ):
        raise ValueError("输出目录不能位于原始输入目录内部")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录不是空目录，拒绝覆盖：{args.output}")


def list_images(input_path):
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"不支持的图像格式：{input_path}")
        return [input_path]
    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def patient_key(image_path, input_path):
    relative = image_path.relative_to(input_path)
    # gastric-cbm常见层级是“类别/中心/患者/图片”，不能把第一层类别
    # 误当成患者。以图片直接父目录的完整相对路径作为稳定抽样单元。
    return relative.parent.as_posix() if len(relative.parts) > 1 else "."


def evenly_spaced_keys(keys, limit):
    if limit is None or limit >= len(keys):
        return keys
    if limit == 1:
        return [keys[len(keys) // 2]]
    indices = [
        round(index * (len(keys) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return [keys[index] for index in indices]


def select_images(images, args):
    if args.input.is_file():
        return images[: args.limit] if args.limit is not None else images
    groups = {}
    for image_path in images:
        groups.setdefault(patient_key(image_path, args.input), []).append(image_path)
    selected_keys = evenly_spaced_keys(sorted(groups), args.patient_limit)
    selected = []
    for key in selected_keys:
        patient_images = groups[key]
        if args.images_per_patient is not None:
            patient_images = patient_images[: args.images_per_patient]
        selected.extend(patient_images)
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def relative_path(image_path, input_path):
    return Path(image_path.name) if input_path.is_file() else image_path.relative_to(input_path)


def build_output_relative_paths(images, input_path):
    """为统一JPEG输出生成无冲突的确定性相对路径。"""
    relatives = [relative_path(path, input_path) for path in images]
    normalized = [relative.with_suffix(".jpg") for relative in relatives]
    counts = Counter(path.as_posix() for path in normalized)
    outputs = []
    for relative, candidate in zip(relatives, normalized):
        if counts[candidate.as_posix()] == 1:
            outputs.append(candidate)
            continue
        suffix = relative.suffix.lower().lstrip(".") or "none"
        outputs.append(
            relative.parent / f"{relative.stem}__src_{suffix}.jpg"
        )
    output_keys = [path.as_posix() for path in outputs]
    if len(output_keys) != len(set(output_keys)):
        raise ValueError("统一JPEG输出仍存在相对路径冲突，请检查同名源文件")
    return dict(zip(images, zip(relatives, outputs)))


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV 无法解码：{path}")
    return image


def write_jpeg(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not ok:
        raise ValueError(f"OpenCV 无法编码 JPEG：{path}")
    encoded.tofile(str(path))


def file_sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def odd_kernel_size(short_side):
    size = max(5, int(round(short_side * CLOSE_KERNEL_RATIO)))
    return size if size % 2 else size + 1


def resize_for_detection(image):
    height, width = image.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(width, height))
    detect_width = max(1, round(width * scale))
    detect_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image,
        (detect_width, detect_height),
        interpolation=interpolation,
    )
    return resized, scale


def foreground_mask(detect_image):
    gray = cv2.cvtColor(detect_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
    raw = (gray > FOREGROUND_THRESHOLD).astype(np.uint8) * 255
    size = odd_kernel_size(min(gray.shape))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
    return gray, raw, closed


def contour_intersects_center(contour, width, height):
    window_width = width * CENTER_WINDOW_RATIO
    window_height = height * CENTER_WINDOW_RATIO
    x1 = (width - window_width) / 2
    y1 = (height - window_height) / 2
    x2 = x1 + window_width
    y2 = y1 + window_height
    x, y, width_box, height_box = cv2.boundingRect(contour)
    return not (
        x + width_box <= x1
        or x >= x2
        or y + height_box <= y1
        or y >= y2
    )


def full_frame_result(width, height, reason):
    return {
        "bbox": (0, 0, width, height),
        "crop_status": "fallback_full_frame",
        "crop_method": "full_frame",
        "failure_reason": reason,
        "contour_area_ratio": 0.0,
        "center_offset_ratio": 0.0,
        "secondary_edge_component_ratio": 0.0,
    }


def secondary_edge_component_ratio(raw_mask, main_bbox):
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (raw_mask > 0).astype(np.uint8), connectivity=8
    )
    height, width = raw_mask.shape
    image_area = width * height
    main_x, main_y, main_w, main_h = main_bbox
    best = 0.0
    for index in range(1, count):
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        component_width = int(stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        ratio = area / image_area
        if ratio < SECONDARY_COMPONENT_MIN_RATIO:
            continue
        component_center = (x + component_width / 2, y + component_height / 2)
        inside_main = (
            main_x <= component_center[0] <= main_x + main_w
            and main_y <= component_center[1] <= main_y + main_h
        )
        near_edge = (
            x <= width * 0.2
            or y <= height * 0.2
            or x + component_width >= width * 0.8
            or y + component_height >= height * 0.8
        )
        if near_edge and not inside_main:
            best = max(best, ratio)
    return best


def detect_initial_roi(image):
    original_height, original_width = image.shape[:2]
    detect_image, scale = resize_for_detection(image)
    detect_height, detect_width = detect_image.shape[:2]
    _, raw_mask, closed_mask = foreground_mask(detect_image)
    contours, _ = cv2.findContours(
        closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = [
        contour
        for contour in contours
        if contour_intersects_center(contour, detect_width, detect_height)
    ]
    if not candidates:
        return full_frame_result(
            original_width, original_height, "no_center_contour"
        )

    contour = max(candidates, key=cv2.contourArea)
    x, y, width_box, height_box = cv2.boundingRect(contour)
    contour_area_ratio = cv2.contourArea(contour) / (detect_width * detect_height)
    bbox_width_ratio = width_box / detect_width
    bbox_height_ratio = height_box / detect_height
    center_x = x + width_box / 2
    center_y = y + height_box / 2
    center_offset = math.hypot(
        center_x - detect_width / 2,
        center_y - detect_height / 2,
    ) / math.hypot(detect_width, detect_height)
    secondary_ratio = secondary_edge_component_ratio(
        raw_mask, (x, y, width_box, height_box)
    )

    failures = []
    if contour_area_ratio < MIN_CONTOUR_AREA_RATIO:
        failures.append("small_contour_area")
    if (
        bbox_width_ratio < MIN_BBOX_DIM_RATIO
        or bbox_height_ratio < MIN_BBOX_DIM_RATIO
    ):
        failures.append("small_bbox")
    if center_offset > MAX_CENTER_OFFSET_RATIO:
        failures.append("off_center")
    if failures:
        result = full_frame_result(
            original_width, original_height, "|".join(failures)
        )
        result.update(
            contour_area_ratio=contour_area_ratio,
            center_offset_ratio=center_offset,
            secondary_edge_component_ratio=secondary_ratio,
        )
        return result

    scale_x = original_width / detect_width
    scale_y = original_height / detect_height
    x1 = max(0, math.floor(x * scale_x))
    y1 = max(0, math.floor(y * scale_y))
    x2 = min(original_width, math.ceil((x + width_box) * scale_x))
    y2 = min(original_height, math.ceil((y + height_box) * scale_y))
    margins = (
        x1 / original_width,
        y1 / original_height,
        (original_width - x2) / original_width,
        (original_height - y2) / original_height,
    )
    if all(margin < FULL_FRAME_MARGIN_RATIO for margin in margins):
        bbox = (0, 0, original_width, original_height)
        method = "full_frame"
        status = "full_frame"
    else:
        inset = round(min(x2 - x1, y2 - y1) * INSET_RATIO)
        bbox = (x1 + inset, y1 + inset, x2 - inset, y2 - inset)
        method = "center_largest_contour_inset"
        status = "cropped"

    return {
        "bbox": bbox,
        "crop_status": status,
        "crop_method": method,
        "failure_reason": "",
        "contour_area_ratio": contour_area_ratio,
        "center_offset_ratio": center_offset,
        "secondary_edge_component_ratio": secondary_ratio,
    }


def first_content_offset(scores):
    run = max(3, round(len(scores) * EDGE_CONTENT_RUN_RATIO))
    limit = min(len(scores), round(len(scores) * EDGE_SCAN_MAX_RATIO))
    for offset in range(max(0, limit - run + 1)):
        if np.all(scores[offset : offset + run] > EDGE_BRIGHTNESS_THRESHOLD):
            return offset
    return 0


def four_edge_offsets(gray):
    height, width = gray.shape
    band_y1 = round(height * EDGE_BAND_START)
    band_y2 = round(height * EDGE_BAND_END)
    band_x1 = round(width * EDGE_BAND_START)
    band_x2 = round(width * EDGE_BAND_END)
    column_scores = np.percentile(
        gray[band_y1:band_y2], EDGE_CONTENT_PERCENTILE, axis=0
    )
    row_scores = np.percentile(
        gray[:, band_x1:band_x2], EDGE_CONTENT_PERCENTILE, axis=1
    )
    return (
        first_content_offset(column_scores),
        first_content_offset(row_scores),
        first_content_offset(column_scores[::-1]),
        first_content_offset(row_scores[::-1]),
    )


def detect_progress_bar(gray, left, right):
    height = gray.shape[0]
    content_width = max(1, right - left)
    x1 = left + round(content_width * EDGE_BAND_START)
    x2 = left + round(content_width * EDGE_BAND_END)
    band = gray[:, x1:x2].astype(np.float32)
    row_means = band.mean(axis=1)
    row_stds = band.std(axis=1)
    differences = np.abs(np.diff(row_means))
    search_start = round(height * PROGRESS_SEARCH_START_RATIO)
    search_end = min(
        len(differences), round(height * PROGRESS_MAX_BOUNDARY_RATIO)
    )
    if search_end <= search_start:
        return height, False, 0.0, 0.0
    # 工具栏常有多条横线；最大跳变可能位于栏内或栏底。按从上到下的
    # 顺序选择第一个满足“明显跳变 + 后续低纹理”的边界，避免只裁掉栏底。
    best_jump = 0.0
    best_tail_std = float("inf")
    for boundary in range(search_start, search_end):
        tail_start = boundary + 1
        tail_length = height - tail_start
        tail_std = (
            float(np.median(row_stds[tail_start:]))
            if tail_length
            else float("inf")
        )
        jump = float(differences[boundary])
        if jump > best_jump:
            best_jump = jump
            best_tail_std = tail_std
        if (
            tail_length >= round(height * PROGRESS_MIN_TAIL_RATIO)
            and jump >= PROGRESS_MIN_JUMP
            and tail_std <= PROGRESS_MAX_TAIL_STD
        ):
            return tail_start, True, jump, tail_std
    return height, False, best_jump, best_tail_std


def refine_by_four_edges(image, bbox):
    original_height, original_width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return bbox, {
            "edge_refined": False,
            "progress_bar_detected": False,
            "edge_trim_left": 0,
            "edge_trim_top": 0,
            "edge_trim_right": 0,
            "edge_trim_bottom": 0,
            "progress_jump": 0.0,
            "progress_tail_std": 0.0,
        }

    detect_roi, scale = resize_for_detection(roi)
    gray = cv2.cvtColor(detect_roi, cv2.COLOR_BGR2GRAY)
    left, top, right_trim, bottom_trim = four_edge_offsets(gray)
    right = gray.shape[1] - right_trim
    bottom = gray.shape[0] - bottom_trim
    progress_bottom, progress_detected, jump, tail_std = detect_progress_bar(
        gray, left, right
    )
    bottom = min(bottom, progress_bottom)

    black_ratio = float((gray <= BLACK_THRESHOLD).mean())
    allow_wide = progress_detected or black_ratio >= EDGE_WIDE_TRIM_BLACK_RATIO
    offsets = [left, top, right_trim, gray.shape[0] - bottom]
    dimensions = [gray.shape[1], gray.shape[0], gray.shape[1], gray.shape[0]]
    if not allow_wide:
        offsets = [
            0 if value > dimension * EDGE_ALWAYS_ALLOW_TRIM_RATIO else value
            for value, dimension in zip(offsets, dimensions)
        ]
    left, top, right_trim, bottom_trim = offsets
    right = gray.shape[1] - right_trim
    bottom = gray.shape[0] - bottom_trim

    expand = max(1, round(min(gray.shape) * EDGE_EXPAND_RATIO))
    left = max(0, left - expand)
    top = max(0, top - expand)
    right = min(gray.shape[1], right + expand)
    # 已识别工具栏时，bottom 就是栏顶；向下扩张会把工具栏重新带回裁剪。
    bottom = (
        min(gray.shape[0], bottom)
        if progress_detected
        else min(gray.shape[0], bottom + expand)
    )
    raw_candidate = (
        x1 + math.floor(left / scale),
        y1 + math.floor(top / scale),
        x1 + math.ceil(right / scale),
        y1 + math.ceil(bottom / scale),
    )
    raw_candidate = (
        max(x1, raw_candidate[0]),
        max(y1, raw_candidate[1]),
        min(x2, raw_candidate[2]),
        min(y2, raw_candidate[3]),
    )
    cx1, cy1, cx2, cy2 = raw_candidate
    horizontal_valid = (
        cx2 > cx1
        and cx2 - cx1 >= original_width * MIN_REFINED_DIM_RATIO
        and cx1 <= original_width / 2 <= cx2
        and cx1 - x1 <= original_width * EDGE_SCAN_MAX_RATIO
        and x2 - cx2 <= original_width * EDGE_SCAN_MAX_RATIO
    )
    vertical_valid = (
        cy2 > cy1
        and cy2 - cy1 >= original_height * MIN_REFINED_DIM_RATIO
        and cy1 <= original_height / 2 <= cy2
        and cy1 - y1 <= original_height * EDGE_SCAN_MAX_RATIO
        and y2 - cy2 <= original_height * EDGE_SCAN_MAX_RATIO
    )
    if not horizontal_valid:
        cx1, cx2 = x1, x2
    if not vertical_valid:
        cy1, cy2 = y1, y2
    candidate = (cx1, cy1, cx2, cy2)
    trim = (cx1 - x1, cy1 - y1, x2 - cx2, y2 - cy2)
    return candidate, {
        "edge_refined": candidate != bbox,
        "progress_bar_detected": progress_detected,
        "edge_trim_left": trim[0],
        "edge_trim_top": trim[1],
        "edge_trim_right": trim[2],
        "edge_trim_bottom": trim[3],
        "progress_jump": jump,
        "progress_tail_std": tail_std,
    }


def black_pixel_ratio(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float((gray <= BLACK_THRESHOLD).mean())


def edge_overlay_metrics(image):
    """统计靠近边缘的白色低饱和小组件，作为文字/刻度残留的复核提示。"""
    height, width = image.shape[:2]
    image_area = height * width
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 1] < OVERLAY_MAX_SATURATION)
        & (hsv[:, :, 2] > OVERLAY_MIN_VALUE)
    ).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    component_count = 0
    component_area = 0
    for index in range(1, count):
        component_width = int(stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        center_x, center_y = centroids[index]
        near_edge = (
            center_x < width * OVERLAY_EDGE_X_RATIO
            or center_x > width * (1 - OVERLAY_EDGE_X_RATIO)
            or center_y < height * OVERLAY_EDGE_Y_RATIO
            or center_y > height * (1 - OVERLAY_EDGE_Y_RATIO)
        )
        small_component = (
            2 <= area <= image_area * OVERLAY_COMPONENT_MAX_AREA_RATIO
            and component_width <= width * OVERLAY_COMPONENT_MAX_WIDTH_RATIO
            and component_height <= height * OVERLAY_COMPONENT_MAX_HEIGHT_RATIO
        )
        if near_edge and small_component:
            component_count += 1
            component_area += area
    return component_count, component_area / image_area


def build_review_reasons(
    initial,
    refine,
    crop_black_ratio,
    overlay_component_count,
    overlay_component_area_ratio,
):
    reasons = []
    if initial["crop_status"] == "fallback_full_frame":
        reasons.append("fallback_full_frame")
    if initial["secondary_edge_component_ratio"] >= SECONDARY_COMPONENT_MIN_RATIO:
        reasons.append("secondary_edge_component")
    if crop_black_ratio >= HIGH_BLACK_RATIO:
        reasons.append("high_black_ratio")
    if (
        overlay_component_count >= OVERLAY_COMPONENT_MIN_COUNT
        and overlay_component_area_ratio >= OVERLAY_COMPONENT_MIN_AREA_RATIO
    ):
        reasons.append("possible_edge_overlay")
    if (
        refine["progress_jump"] >= PROGRESS_MIN_JUMP
        and not refine["progress_bar_detected"]
    ):
        reasons.append("possible_progress_bar")
    return reasons


def fit_panel(image):
    scale = min(
        PREVIEW_PANEL_WIDTH / image.shape[1],
        PREVIEW_PANEL_HEIGHT / image.shape[0],
    )
    width = max(1, round(image.shape[1] * scale))
    height = max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (width, height), cv2.INTER_AREA)
    panel = np.zeros(
        (PREVIEW_PANEL_HEIGHT, PREVIEW_PANEL_WIDTH, 3), dtype=np.uint8
    )
    x = (PREVIEW_PANEL_WIDTH - width) // 2
    y = (PREVIEW_PANEL_HEIGHT - height) // 2
    panel[y : y + height, x : x + width] = resized
    return panel, scale, x, y


def make_preview(original, cropped, bbox, review_reasons):
    left, scale, offset_x, offset_y = fit_panel(original)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(
        left,
        (
            offset_x + round(x1 * scale),
            offset_y + round(y1 * scale),
        ),
        (
            offset_x + round(x2 * scale),
            offset_y + round(y2 * scale),
        ),
        (0, 255, 0),
        2,
    )
    right, _, _, _ = fit_panel(cropped)
    combined = cv2.hconcat((left, right))
    label = "review=" + ("|".join(review_reasons) if review_reasons else "ok")
    cv2.putText(
        combined,
        label,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255) if review_reasons else (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return combined


def process_one(image_path, relative, output_relative, args, save_preview):
    image = read_image(image_path)
    original_height, original_width = image.shape[:2]
    initial = detect_initial_roi(image)
    bbox, refine = refine_by_four_edges(image, initial["bbox"])
    x1, y1, x2, y2 = bbox
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        raise ValueError("裁剪结果为空")
    crop_black_ratio = black_pixel_ratio(cropped)
    overlay_component_count, overlay_component_area_ratio = edge_overlay_metrics(
        cropped
    )
    review_reasons = build_review_reasons(
        initial,
        refine,
        crop_black_ratio,
        overlay_component_count,
        overlay_component_area_ratio,
    )

    crop_path = args.output / "crops" / output_relative
    write_jpeg(crop_path, cropped)
    preview_path = ""
    if save_preview:
        preview_file = args.output / "previews" / output_relative
        write_jpeg(
            preview_file,
            make_preview(image, cropped, bbox, review_reasons),
        )
        preview_path = str(preview_file)

    return {
        "source": str(image_path),
        "relative_path": str(relative),
        "crop": str(crop_path),
        "preview": preview_path,
        "source_width": original_width,
        "source_height": original_height,
        "crop_x1": x1,
        "crop_y1": y1,
        "crop_x2": x2,
        "crop_y2": y2,
        "crop_width": x2 - x1,
        "crop_height": y2 - y1,
        "retained_area_ratio": (
            (x2 - x1) * (y2 - y1) / (original_width * original_height)
        ),
        "crop_method": (
            initial["crop_method"] + "+four_edge_scan_v2"
            if refine["edge_refined"]
            else initial["crop_method"]
        ),
        "crop_status": initial["crop_status"],
        "failure_reason": initial["failure_reason"],
        "contour_area_ratio": initial["contour_area_ratio"],
        "center_offset_ratio": initial["center_offset_ratio"],
        "secondary_edge_component_ratio": initial[
            "secondary_edge_component_ratio"
        ],
        **refine,
        "black_pixel_ratio": crop_black_ratio,
        "overlay_component_count": overlay_component_count,
        "overlay_component_area_ratio": overlay_component_area_ratio,
        "review_status": "pending" if review_reasons else "auto_pass_candidate",
        "review_reasons": "|".join(review_reasons),
        "crop_sha256": file_sha256(crop_path),
    }


def main():
    args = parse_args()
    validate_args(args)
    images = select_images(list_images(args.input), args)
    if not images:
        raise FileNotFoundError(f"未找到图像：{args.input}")
    selected_patients = (
        1
        if args.input.is_file()
        else len({patient_key(path, args.input) for path in images})
    )
    print(f"选择：{selected_patients} 位患者，{len(images)} 张图像")
    args.output.mkdir(parents=True, exist_ok=True)
    output_paths = build_output_relative_paths(images, args.input)

    rows = []
    for index, image_path in enumerate(images, start=1):
        relative, output_relative = output_paths[image_path]
        try:
            row = process_one(
                image_path,
                relative,
                output_relative,
                args,
                save_preview=index <= args.preview_limit,
            )
        except Exception as error:
            row = {
                "source": str(image_path),
                "relative_path": str(relative),
                "crop": "",
                "preview": "",
                "source_width": "",
                "source_height": "",
                "crop_x1": "",
                "crop_y1": "",
                "crop_x2": "",
                "crop_y2": "",
                "crop_width": "",
                "crop_height": "",
                "retained_area_ratio": "",
                "crop_method": "",
                "crop_status": "error",
                "failure_reason": str(error),
                "contour_area_ratio": "",
                "center_offset_ratio": "",
                "secondary_edge_component_ratio": "",
                "edge_refined": False,
                "progress_bar_detected": False,
                "edge_trim_left": "",
                "edge_trim_top": "",
                "edge_trim_right": "",
                "edge_trim_bottom": "",
                "progress_jump": "",
                "progress_tail_std": "",
                "black_pixel_ratio": "",
                "overlay_component_count": "",
                "overlay_component_area_ratio": "",
                "review_status": "error",
                "review_reasons": "processing_error",
                "crop_sha256": "",
            }
        rows.append(row)
        print(
            f"[{index}/{len(images)}] {image_path} -> "
            f"{row['crop_status']} {row['review_status']} "
            f"{row['review_reasons']}"
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
