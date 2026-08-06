#!/usr/bin/env python3
"""画中画自动筛查 v2：矩形提议、边界内容确认与人工复核输出。

本脚本只生成质量标记和候选预览，不删除、移动或修改任何输入图像。

检测分为两级：
1. 矩形提议：在左上区域寻找可相交的水平/垂直边界；如果垂直边界
   因黑色背景而不可见，则由明确的下边界配合垂直梯度投影回退估计。
2. 内容确认：在矩形右边界和下边界两侧使用互不重叠的窄条采样，
   分别验证亮度、颜色和梯度突变。

用法：
    # 30 张验证集，默认输出到独立的 v2 验证目录
    python pip_detector.py --validate-only

    # 全量筛查，默认输出到独立的 v2 全量目录
    python pip_detector.py
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import (
    PIP_VALIDATION_LABELS,
    STAGE1_CROPS,
    STAGE2_MAPPING,
    STAGE2_MASKS,
    STAGE3_OUTPUT,
    STAGE3_VALIDATION_LIST,
    STAGE3_VALIDATION_OUTPUT,
)

VALIDATION_LIST = STAGE3_VALIDATION_LIST
VALIDATION_LABELS = PIP_VALIDATION_LABELS
VALIDATION_OUTPUT = STAGE3_VALIDATION_OUTPUT
FULL_OUTPUT = STAGE3_OUTPUT

# 矩形提议参数
SCAN_RATIO = 0.40
CANNY_LOW = 30
CANNY_HIGH = 100
HOUGH_THRESHOLD = 40
MIN_LINE_LENGTH_RATIO = 0.10
MAX_LINE_GAP_RATIO = 0.02
ANGLE_TOLERANCE = 12
INTERSECTION_TOLERANCE_RATIO = 0.05
MIN_RECT_AREA_RATIO = 0.02
MAX_RECT_AREA_RATIO = 0.50
MIN_ASPECT_RATIO = 0.50
MAX_ASPECT_RATIO = 2.00
MAX_CLUSTERED_LINES = 20
FALLBACK_MIN_VERTICAL_COHERENCE = 8.0
FALLBACK_MIN_BOTTOM_COHERENCE = 8.0

# 融合与分层参数
GEOMETRY_WEIGHT = 0.55
CONTENT_WEIGHT = 0.45
DEFAULT_CANDIDATE_THRESHOLD = 0.45
DEFAULT_REVIEW_THRESHOLD = 0.30
HIGH_CONFIDENCE_MIN_CONTENT = 0.55
PREVIEW_PANEL = 720
OPENCV_RNG_SEED = 0

# 候选矩形内 FOV 覆盖率门槛：低于该值说明矩形大部分落在 FOV 圆外，
# 是圆弧边缘被误认成矩形边界的伪影；且圆外内容会被 FOV 遮罩遮黑，
# 本身不需要 notch。
MIN_RECT_FOV_COVERAGE = 0.5

# 画中画可出现在四个角。通过翻转把每个角映射成“虚拟左上角”，
# 复用下方以左上角为锚的矩形提议与内容验证逻辑。
CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")


def corner_transform(image, corner):
    """把指定角翻转到虚拟左上角；尺寸不变。"""
    if corner == "top_right":
        return cv2.flip(image, 1)
    if corner == "bottom_left":
        return cv2.flip(image, 0)
    if corner == "bottom_right":
        return cv2.flip(image, -1)
    return image


def _map_point_back(x, y, corner, width, height):
    """单个点从虚拟角坐标映射回原图坐标。"""
    if corner == "top_right":
        return width - 1 - x, y
    if corner == "bottom_left":
        return x, height - 1 - y
    if corner == "bottom_right":
        return width - 1 - x, height - 1 - y
    return x, y


def map_rect_back(rect, corner, width, height):
    """矩形(x,y,w,h)从虚拟角坐标映射回原图坐标。"""
    x, y, w, h = rect
    if corner == "top_right":
        return width - x - w, y, w, h
    if corner == "bottom_left":
        return x, height - y - h, w, h
    if corner == "bottom_right":
        return width - x - w, height - y - h, w, h
    return rect


def map_line_back(line, corner, width, height):
    """线段两端点从虚拟角坐标映射回原图坐标。"""
    x1, y1, x2, y2 = line
    x1b, y1b = _map_point_back(x1, y1, corner, width, height)
    x2b, y2b = _map_point_back(x2, y2, corner, width, height)
    return x1b, y1b, x2b, y2b


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="画中画自动筛查 v2")
    parser.add_argument("--crops", type=Path, default=STAGE1_CROPS)
    parser.add_argument("--masks", type=Path, default=STAGE2_MASKS)
    parser.add_argument("--stage2-mapping", type=Path, default=STAGE2_MAPPING)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--candidate-threshold", type=float,
        default=DEFAULT_CANDIDATE_THRESHOLD,
    )
    parser.add_argument(
        "--review-threshold", type=float,
        default=DEFAULT_REVIEW_THRESHOLD,
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validation-list", type=Path, default=VALIDATION_LIST)
    parser.add_argument("--validation-labels", type=Path, default=VALIDATION_LABELS)
    parser.add_argument(
        "--min-fov-coverage", type=float, default=MIN_RECT_FOV_COVERAGE,
        help="候选矩形内FOV覆盖率门槛；低于则判为圆外伪影",
    )
    return parser.parse_args()


def resolve_output(args):
    """未显式指定输出时，按验证/全量模式选默认目录。"""
    if args.output is not None:
        return args.output
    return VALIDATION_OUTPUT if args.validate_only else FULL_OUTPUT


def validate_args(args, output):
    """校验输入存在、输出目录为空、阈值顺序合法。"""
    required = [args.crops, args.stage2_mapping]
    if args.validate_only:
        required.extend([args.validation_list, args.validation_labels])
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"输入不存在：{path}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{output}")
    if not 0 < args.review_threshold < args.candidate_threshold < 1:
        raise ValueError(
            "阈值必须满足 0 < review-threshold < candidate-threshold < 1"
        )


def read_image(path, flags=cv2.IMREAD_COLOR):
    """读取图像为BGR数组，失败抛错。"""
    image = cv2.imread(str(path), flags)
    if image is None:
        raise ValueError(f"图像读取失败：{path}")
    return image


def cluster_axis_lines(lines, axis, tolerance):
    """按边界坐标合并同一条边产生的多条霍夫线。"""
    if not lines:
        return []
    coord_index = 1 if axis == "horizontal" else 0
    ordered = sorted(lines, key=lambda item: item[coord_index])
    groups = []
    for line in ordered:
        coord = line[coord_index]
        if not groups or abs(coord - np.mean(
            [item[coord_index] for item in groups[-1]]
        )) > tolerance:
            groups.append([line])
        else:
            groups[-1].append(line)
    return [max(group, key=lambda item: item[4]) for group in groups]


def line_features(image):
    """返回聚类后的水平线、垂直线以及原始/聚类线条数量。"""
    height, width = image.shape[:2]
    short_side = min(height, width)
    scan_h = max(1, int(height * SCAN_RATIO))
    scan_w = max(1, int(width * SCAN_RATIO))
    gray = cv2.cvtColor(image[:scan_h, :scan_w], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        cv2.GaussianBlur(gray, (3, 3), 0), CANNY_LOW, CANNY_HIGH
    )
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=max(20, int(short_side * MIN_LINE_LENGTH_RATIO)),
        maxLineGap=max(5, int(short_side * MAX_LINE_GAP_RATIO)),
    )

    horizontal = []
    vertical = []
    if lines is not None:
        # OpenCV 4常返回[N, 1, 4]，当前gastric-cbm的OpenCV 5返回
        # [N, 4]；统一展平，避免把单个numpy.int32当成一条线。
        for raw in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = [int(value) for value in raw]
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            length = math.hypot(x2 - x1, y2 - y1)
            if angle < ANGLE_TOLERANCE or angle > 180 - ANGLE_TOLERANCE:
                horizontal.append((
                    min(x1, x2), (y1 + y2) / 2,
                    max(x1, x2), (y1 + y2) / 2, length,
                ))
            elif 90 - ANGLE_TOLERANCE < angle < 90 + ANGLE_TOLERANCE:
                vertical.append((
                    (x1 + x2) / 2, min(y1, y2),
                    (x1 + x2) / 2, max(y1, y2), length,
                ))

    raw_count = len(horizontal) + len(vertical)
    merge_tolerance = max(3, int(short_side * 0.005))
    horizontal = cluster_axis_lines(
        horizontal, "horizontal", merge_tolerance
    )
    vertical = cluster_axis_lines(vertical, "vertical", merge_tolerance)
    clustered_count = len(horizontal) + len(vertical)
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "raw_count": raw_count,
        "clustered_count": clustered_count,
        "scan_h": scan_h,
        "scan_w": scan_w,
        "short_side": short_side,
    }


def rect_is_valid(rect, scan_w, scan_h):
    """按面积占比和宽高比过滤过小/过大矩形。"""
    _, _, rect_w, rect_h = rect
    if rect_w <= 0 or rect_h <= 0:
        return False
    area_ratio = (rect_w * rect_h) / max(1, scan_w * scan_h)
    aspect_ratio = rect_w / rect_h
    return (
        MIN_RECT_AREA_RATIO <= area_ratio <= MAX_RECT_AREA_RATIO
        and MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO
    )


def line_count_quality(clustered_count):
    """线条数量作为软惩罚，不以单个阈值直接清零。"""
    if clustered_count <= 10:
        return 1.0
    if clustered_count <= 15:
        return 0.70
    if clustered_count <= MAX_CLUSTERED_LINES:
        return 0.40
    return max(0.05, 0.40 - 0.04 * (clustered_count - 20))


def pair_proposals(features):
    """横竖线相交生成左上锚定矩形候选（几何分）。"""
    proposals = []
    tolerance = max(
        10, int(features["short_side"] * INTERSECTION_TOLERANCE_RATIO)
    )
    count_quality = line_count_quality(features["clustered_count"])

    for h_line in features["horizontal"]:
        hx1, h_y, hx2, _, h_len = h_line
        for v_line in features["vertical"]:
            v_x, vy1, _, vy2, v_len = v_line
            intersects = (
                hx1 - tolerance <= v_x <= hx2 + tolerance
                and vy1 - tolerance <= h_y <= vy2 + tolerance
            )
            if not intersects:
                continue
            rect = (0, 0, int(round(v_x)), int(round(h_y)))
            if not rect_is_valid(
                rect, features["scan_w"], features["scan_h"]
            ):
                continue
            _, _, rect_w, rect_h = rect
            horizontal_coverage = min(1.0, h_len / rect_w)
            vertical_coverage = min(1.0, v_len / rect_h)
            intersection_error = math.hypot(
                max(0, hx1 - v_x, v_x - hx2),
                max(0, vy1 - h_y, h_y - vy2),
            )
            intersection_score = max(0.0, 1 - intersection_error / tolerance)
            # 线条数量是候选可信度的门控项。正常FOV圆环和复杂组织
            # 往往产生大量近似横/竖线，不能让较长的偶然交叉线抵消该证据。
            geometry_base = (
                0.38 * horizontal_coverage
                + 0.38 * vertical_coverage
                + 0.24 * intersection_score
            )
            geometry_score = geometry_base * count_quality
            proposals.append({
                "rect": rect,
                "geometry_score": min(1.0, geometry_score),
                "proposal_type": "line_pair",
                "h_line": tuple(int(round(v)) for v in h_line[:4]),
                "v_line": tuple(int(round(v)) for v in v_line[:4]),
                "horizontal_coverage": horizontal_coverage,
                "vertical_coverage": vertical_coverage,
            })
    return proposals


def vertical_projection(gray_float, bottom_y, start_x, end_x, strip):
    """在给定下边界上方搜索最连贯的垂直亮度突变。"""
    y1 = max(strip, 5)
    y2 = min(gray_float.shape[0] - strip, bottom_y - 5)
    if y2 <= y1 or end_x <= start_x:
        return None

    scores = []
    for x in range(start_x, end_x + 1):
        inner = gray_float[y1:y2, x - strip:x]
        outer = gray_float[y1:y2, x:x + strip]
        if inner.size == 0 or outer.size == 0:
            continue
        per_row = np.abs(inner.mean(axis=1) - outer.mean(axis=1))
        coherence = float(np.median(per_row))
        scores.append((coherence, x))
    if not scores:
        return None
    return max(scores)


def bottom_boundary_coherence(gray_float, rect, strip):
    """矩形下边界两侧的横向亮度突变一致性。"""
    _, _, rect_w, rect_h = rect
    x1 = max(strip, 5)
    x2 = min(gray_float.shape[1] - strip, rect_w - 5)
    if x2 <= x1 or rect_h - strip < 0 or rect_h + strip > gray_float.shape[0]:
        return 0.0
    inner = gray_float[rect_h - strip:rect_h, x1:x2]
    outer = gray_float[rect_h:rect_h + strip, x1:x2]
    if inner.size == 0 or outer.size == 0:
        return 0.0
    per_col = np.abs(inner.mean(axis=0) - outer.mean(axis=0))
    return float(np.median(per_col))


def fallback_proposals(image, features):
    """由明确下边界和垂直梯度投影补回黑背景中的隐形右边界。"""
    height, width = image.shape[:2]
    gray_float = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    strip = max(2, int(round(features["short_side"] * 0.004)))
    count_quality = line_count_quality(features["clustered_count"])
    proposals = []

    for h_line in features["horizontal"]:
        hx1, h_y_float, hx2, _, h_len = h_line
        bottom_y = int(round(h_y_float))
        if not int(height * 0.08) <= bottom_y <= int(height * 0.32):
            continue

        start_x = max(
            strip + 1,
            int(bottom_y * MIN_ASPECT_RATIO),
            int(hx2),
        )
        end_x = min(
            features["scan_w"] - strip - 1,
            int(bottom_y * MAX_ASPECT_RATIO),
        )
        projected = vertical_projection(
            gray_float, bottom_y, start_x, end_x, strip
        )
        if projected is None:
            continue
        vertical_coherence, right_x = projected
        rect = (0, 0, int(right_x), bottom_y)
        if not rect_is_valid(
            rect, features["scan_w"], features["scan_h"]
        ):
            continue
        bottom_coherence = bottom_boundary_coherence(
            gray_float, rect, strip
        )
        if (
            vertical_coherence < FALLBACK_MIN_VERTICAL_COHERENCE
            or bottom_coherence < FALLBACK_MIN_BOTTOM_COHERENCE
        ):
            continue

        horizontal_coverage = min(1.0, h_len / max(1, right_x))
        vertical_score = min(
            1.0, vertical_coherence / 18.0
        )
        bottom_score = min(1.0, bottom_coherence / 25.0)
        geometry_base = (
            0.25 * horizontal_coverage
            + 0.30 * vertical_score
            + 0.30 * bottom_score
            + 0.15
        )
        geometry_score = geometry_base * count_quality
        proposals.append({
            "rect": rect,
            "geometry_score": min(0.85, geometry_score),
            "proposal_type": "horizontal_projection_fallback",
            "h_line": tuple(int(round(v)) for v in h_line[:4]),
            "v_line": (right_x, 0, right_x, bottom_y),
            "horizontal_coverage": horizontal_coverage,
            "vertical_coverage": vertical_score,
            "vertical_coherence": vertical_coherence,
            "bottom_coherence": bottom_coherence,
        })
    return proposals


def deduplicate_proposals(proposals, short_side):
    """合并近似重复候选，按几何分保留前5个。"""
    tolerance = max(8, int(short_side * 0.015))
    selected = []
    for proposal in sorted(
        proposals, key=lambda item: item["geometry_score"], reverse=True
    ):
        rect = proposal["rect"]
        duplicate = any(
            abs(rect[2] - kept["rect"][2]) <= tolerance
            and abs(rect[3] - kept["rect"][3]) <= tolerance
            for kept in selected
        )
        if not duplicate:
            selected.append(proposal)
    return selected[:5]


def propose_rectangles(image):
    """汇总直线提议+回退提议+去重，返回候选矩形列表。"""
    features = line_features(image)
    proposals = pair_proposals(features)
    proposals.extend(fallback_proposals(image, features))
    proposals = deduplicate_proposals(proposals, features["short_side"])
    return proposals, features


def boundary_metrics(gray_float, lab_float, rect, side, strip):
    """计算矩形单侧边界（右/下）的亮度、颜色、梯度分。"""
    _, _, rect_w, rect_h = rect
    trim = max(4, strip)
    if side == "right":
        start, end = trim, rect_h - trim
        if end <= start:
            return None
        inner_gray = gray_float[start:end, rect_w - strip:rect_w]
        outer_gray = gray_float[start:end, rect_w:rect_w + strip]
        inner_lab = lab_float[start:end, rect_w - strip:rect_w]
        outer_lab = lab_float[start:end, rect_w:rect_w + strip]
        per_point = np.abs(
            inner_gray.mean(axis=1) - outer_gray.mean(axis=1)
        )
    else:
        start, end = trim, rect_w - trim
        if end <= start:
            return None
        inner_gray = gray_float[rect_h - strip:rect_h, start:end]
        outer_gray = gray_float[rect_h:rect_h + strip, start:end]
        inner_lab = lab_float[rect_h - strip:rect_h, start:end]
        outer_lab = lab_float[rect_h:rect_h + strip, start:end]
        per_point = np.abs(
            inner_gray.mean(axis=0) - outer_gray.mean(axis=0)
        )

    if inner_gray.size == 0 or outer_gray.size == 0:
        return None
    brightness_diff = float(abs(inner_gray.mean() - outer_gray.mean()))
    color_diff = float(np.linalg.norm(
        inner_lab.reshape(-1, 3).mean(axis=0)
        - outer_lab.reshape(-1, 3).mean(axis=0)
    ))
    gradient_median = float(np.median(per_point))
    gradient_p75 = float(np.percentile(per_point, 75))

    brightness_score = min(1.0, brightness_diff / 30.0)
    color_score = min(1.0, color_diff / 25.0)
    edge_score = (
        min(1.0, gradient_median / 15.0)
        * min(1.0, gradient_p75 / 30.0)
    )
    score = (
        0.25 * brightness_score
        + 0.20 * color_score
        + 0.55 * edge_score
    )
    return {
        "score": score,
        "brightness_diff": brightness_diff,
        "color_diff": color_diff,
        "gradient_median": gradient_median,
        "gradient_p75": gradient_p75,
    }


def verify_rectangle_content(image, rect):
    """验证矩形右+下两条内边界有内容突变证据。"""
    height, width = image.shape[:2]
    _, _, rect_w, rect_h = rect
    short_side = min(height, width)
    strip = max(4, min(12, int(round(short_side * 0.008))))
    if rect_w <= strip or rect_h <= strip:
        return 0.0, {"error": "rect_too_small"}
    if rect_w + strip > width or rect_h + strip > height:
        return 0.0, {"error": "rect_out_of_bounds"}

    gray_float = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    ).astype(np.float32)
    lab_float = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).astype(np.float32)
    right = boundary_metrics(gray_float, lab_float, rect, "right", strip)
    bottom = boundary_metrics(gray_float, lab_float, rect, "bottom", strip)
    if right is None or bottom is None:
        return 0.0, {"error": "empty_boundary_strip"}

    # 两条边都应有证据：较弱边占 60%，避免一条强皱襞掩盖另一条弱边。
    weak = min(right["score"], bottom["score"])
    average = (right["score"] + bottom["score"]) / 2
    score = 0.60 * weak + 0.40 * average
    return min(1.0, score), {
        "strip_width": strip,
        "right": right,
        "bottom": bottom,
    }


def fuse_proposal(proposal, content_score):
    """几何分与内容分融合成最终候选分。"""
    score = min(
        1.0,
        proposal["geometry_score"] * GEOMETRY_WEIGHT
        + content_score * CONTENT_WEIGHT,
    )
    # 没有基本的双边内容突变时，不能只凭正常解剖结构中的偶然直线
    # 进入人工复核队列。
    if content_score < 0.08:
        score *= 0.60
    return score


def detect_best_corner(
    image, fov_mask=None, min_fov_coverage=MIN_RECT_FOV_COVERAGE
):
    """四角各检测一次，在各自翻转空间验证，返回全局最优（映射回原坐标）。

    每个角通过 corner_transform 变成虚拟左上角，复用左上角锚定的矩形提议和
    右/下内边界验证；验证在翻转空间进行，保证“哪两条边是内边界”随角自动正确。
    fov_mask 非空时，过滤掉矩形内 FOV 覆盖率低于门槛的候选（FOV 圆弧边缘
    伪影；且圆外内容会被 FOV 遮罩遮黑，本身不需要 notch）。
    """
    height, width = image.shape[:2]
    best = None
    for corner in CORNERS:
        transformed = corner_transform(image, corner)
        proposals, features = propose_rectangles(transformed)
        for proposal in proposals:
            content_score, content = verify_rectangle_content(
                transformed, proposal["rect"]
            )
            fused = fuse_proposal(proposal, content_score)
            rect_orig = map_rect_back(
                proposal["rect"], corner, width, height
            )
            if fov_mask is not None:
                rx, ry, rw, rh = rect_orig
                if rw <= 0 or rh <= 0:
                    continue
                coverage = fov_mask[ry:ry + rh, rx:rx + rw].mean()
                if coverage < min_fov_coverage:
                    continue
            if best is None or fused > best[0]:
                mapped = dict(proposal)
                mapped["rect"] = rect_orig
                mapped["corner"] = corner
                mapped["raw_count"] = features["raw_count"]
                mapped["clustered_count"] = features["clustered_count"]
                for key in ("h_line", "v_line"):
                    if mapped.get(key):
                        mapped[key] = map_line_back(
                            mapped[key], corner, width, height
                        )
                best = (fused, mapped, content_score, content)
    return best


def load_paths(path):
    """读取输入清单并校验 relative_path 唯一。"""
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "relative_path" not in rows[0]:
        raise ValueError(f"映射表缺少 relative_path：{path}")
    paths = [row["relative_path"].strip() for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError(f"映射表存在重复 relative_path：{path}")
    return paths


def load_labels(path):
    """读取验证标签（pip/non_pip/uncertain/unreviewed）。"""
    labels = {}
    if not path.exists():
        return labels
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            relative_path = row["relative_path"].strip()
            label = row["label"].strip()
            if label not in {"pip", "non_pip", "uncertain", "unreviewed"}:
                raise ValueError(
                    f"不支持的验证标签 {label!r}：{relative_path}"
                )
            labels[relative_path] = {
                "label": label,
                "review_note": row.get("review_note", "").strip(),
            }
    return labels


def tier_for_score(
    score, content_score, candidate_threshold, review_threshold
):
    """按分数与内容分层：高置信 / 灰区 / 阴性。"""
    if (
        score >= candidate_threshold
        and content_score >= HIGH_CONFIDENCE_MIN_CONTENT
    ):
        return "high_confidence"
    if score >= review_threshold:
        return "manual_review"
    return "negative"


def serialise_rect(rect):
    """矩形转逗号分隔字符串（x,y,w,h）。"""
    return ",".join(str(int(value)) for value in rect) if rect else ""


def patient_key(relative_path):
    """用图片父目录作为患者单元。"""
    parent = Path(relative_path).parent
    return parent.as_posix() if str(parent) != "." else "."


def make_preview(image, mask, row, proposal):
    """生成带FOV轮廓、检测框、线条和标题的预览图。"""
    height, width = image.shape[:2]
    scale = min(1.0, PREVIEW_PANEL / max(height, width))
    preview = cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )

    if mask is not None:
        resized_mask = cv2.resize(
            mask, (preview.shape[1], preview.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        contours, _ = cv2.findContours(
            resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(preview, contours, -1, (0, 180, 0), 1)

    rect = proposal["rect"]
    rx, ry, rect_w, rect_h = rect
    cv2.rectangle(
        preview,
        (int(rx * scale), int(ry * scale)),
        (int((rx + rect_w) * scale), int((ry + rect_h) * scale)),
        (0, 255, 255), 2,
    )
    for key, color in [("h_line", (255, 0, 255)), ("v_line", (255, 255, 0))]:
        line = proposal.get(key)
        if line:
            x1, y1, x2, y2 = line
            cv2.line(
                preview,
                (int(x1 * scale), int(y1 * scale)),
                (int(x2 * scale), int(y2 * scale)),
                color, 2,
            )

    header_h = 52
    canvas = np.full(
        (preview.shape[0] + header_h, preview.shape[1], 3),
        35, dtype=np.uint8,
    )
    canvas[header_h:] = preview
    line1 = (
        f"score={row['pip_score']:.3f} tier={row['auto_tier']} "
        f"corner={row.get('pip_corner', '')} type={row['proposal_type']}"
    )
    line2 = (
        f"geo={row['geometry_score']:.3f} "
        f"content={row['content_score']:.3f} "
        f"lines={row['clustered_line_count']} | "
        f"{Path(row['relative_path']).name}"
    )
    cv2.putText(
        canvas, line1[:130], (6, 19),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255),
        1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, line2[:150], (6, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255),
        1, cv2.LINE_AA,
    )
    return canvas


def write_preview(path, preview):
    """预览图写入磁盘（JPEG 92）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path), preview, [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    if not ok:
        raise OSError(f"预览图写入失败：{path}")


def evaluate(rows):
    """用参考标签计算检测的TP/FP/FN/TN、召回率与精确率。"""
    labelled = [
        row for row in rows if row["reference_label"] in {"pip", "non_pip"}
    ]
    if not labelled:
        return {}
    tp = sum(
        row["reference_label"] == "pip"
        and row["auto_tier"] != "negative"
        for row in labelled
    )
    fn = sum(
        row["reference_label"] == "pip"
        and row["auto_tier"] == "negative"
        for row in labelled
    )
    fp = sum(
        row["reference_label"] == "non_pip"
        and row["auto_tier"] != "negative"
        for row in labelled
    )
    tn = sum(
        row["reference_label"] == "non_pip"
        and row["auto_tier"] == "negative"
        for row in labelled
    )
    return {
        "labelled_images": len(labelled),
        "tp_including_review_tier": tp,
        "fn": fn,
        "fp_including_review_tier": fp,
        "tn": tn,
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
    }


def script_sha256():
    """返回本脚本SHA256，用于运行可复现性。"""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main():
    """编排逐图检测，输出quality_flags、预览与run_manifest。"""
    # HoughLinesP may otherwise change borderline proposals across runs when
    # OpenCV schedules work differently. Candidate generation must be reproducible
    # before thresholds and manual review lists are frozen.
    cv2.setNumThreads(1)
    cv2.setRNGSeed(OPENCV_RNG_SEED)
    args = parse_args()
    output = resolve_output(args)
    validate_args(args, output)
    source_list = args.validation_list if args.validate_only else args.stage2_mapping
    relative_paths = load_paths(source_list)
    labels = load_labels(args.validation_labels) if args.validate_only else {}

    output.mkdir(parents=True, exist_ok=False)
    preview_root = output / "candidate_previews"
    rows = []
    preview_jobs = []

    print(f"待筛查图像：{len(relative_paths)}")
    for index, relative_path in enumerate(relative_paths, start=1):
        crop_path = args.crops / relative_path
        mask_path = args.masks / Path(relative_path).with_suffix(".png")
        reference = labels.get(relative_path, {})
        row = {
            "relative_path": relative_path,
            "patient_id": patient_key(relative_path),
            "reference_label": reference.get("label", "unreviewed"),
            "review_note": reference.get("review_note", ""),
            "processing_status": "ok",
            "processing_error": "",
            "pip_score": 0.0,
            "auto_tier": "negative",
            "proposal_type": "",
            "pip_corner": "",
            "rect_xywh": "",
            "geometry_score": 0.0,
            "content_score": 0.0,
            "raw_line_count": 0,
            "clustered_line_count": 0,
            "right_gradient_median": 0.0,
            "bottom_gradient_median": 0.0,
            "vertical_coherence": 0.0,
            "bottom_coherence": 0.0,
        }

        try:
            image = read_image(crop_path)
            fov_mask = None
            raw_mask = cv2.imread(
                str(mask_path), cv2.IMREAD_GRAYSCALE
            )
            if raw_mask is not None:
                fov_mask = (raw_mask > 127).astype(np.uint8)
            best = detect_best_corner(
                image, fov_mask, args.min_fov_coverage
            )
            row["raw_line_count"] = (
                best[1].get("raw_count", 0) if best else 0
            )
            row["clustered_line_count"] = (
                best[1].get("clustered_count", 0) if best else 0
            )

            if best is not None:
                score, proposal, content_score, content = best
                row.update({
                    "pip_score": round(score, 4),
                    "auto_tier": tier_for_score(
                        score,
                        content_score,
                        args.candidate_threshold,
                        args.review_threshold,
                    ),
                    "proposal_type": proposal["proposal_type"],
                    "pip_corner": proposal["corner"],
                    "rect_xywh": serialise_rect(proposal["rect"]),
                    "geometry_score": round(
                        proposal["geometry_score"], 4
                    ),
                    "content_score": round(content_score, 4),
                    "right_gradient_median": round(
                        content.get("right", {}).get(
                            "gradient_median", 0.0
                        ), 2
                    ),
                    "bottom_gradient_median": round(
                        content.get("bottom", {}).get(
                            "gradient_median", 0.0
                        ), 2
                    ),
                    "vertical_coherence": round(
                        proposal.get("vertical_coherence", 0.0), 2
                    ),
                    "bottom_coherence": round(
                        proposal.get("bottom_coherence", 0.0), 2
                    ),
                })
                if row["auto_tier"] != "negative":
                    preview_jobs.append((
                        relative_path, image, mask_path, row, proposal
                    ))
        except Exception as exc:
            row["processing_status"] = "error"
            row["processing_error"] = str(exc)
        rows.append(row)

        if index % 200 == 0 or index == len(relative_paths):
            tiers = Counter(item["auto_tier"] for item in rows)
            print(
                f"[{index}/{len(relative_paths)}] "
                f"高置信={tiers['high_confidence']} "
                f"灰区={tiers['manual_review']} "
                f"错误={sum(r['processing_status'] == 'error' for r in rows)}"
            )

    fieldnames = list(rows[0].keys()) if rows else []
    flags_path = output / "quality_flags.csv"
    with flags_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for relative_path, image, mask_path, row, proposal in preview_jobs:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        preview = make_preview(image, mask, row, proposal)
        preview_path = (
            preview_root / row["auto_tier"]
            / Path(relative_path).with_suffix(".jpg")
        )
        write_preview(preview_path, preview)

    tier_counts = Counter(row["auto_tier"] for row in rows)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "script": str(Path(__file__)),
        "script_sha256": script_sha256(),
        "mode": "validation" if args.validate_only else "full",
        "input_list": str(source_list),
        "crops": str(args.crops),
        "masks": str(args.masks),
        "output": str(output),
        "candidate_threshold": args.candidate_threshold,
        "review_threshold": args.review_threshold,
        "high_confidence_min_content": HIGH_CONFIDENCE_MIN_CONTENT,
        "min_fov_coverage": args.min_fov_coverage,
        "total_images": len(rows),
        "tier_counts": dict(tier_counts),
        "processing_errors": sum(
            row["processing_status"] == "error" for row in rows
        ),
        "label_metrics": evaluate(rows),
    }
    with (output / "run_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["processing_errors"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
