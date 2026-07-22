"""第二批胃镜图像预处理 v1.1：轮廓裁剪后独立扫描四边和视频进度条。"""

import argparse
import csv
import hashlib
import math
import os
import re
import shutil
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(PROJECT_DIR, '数据', '第二批整理后')
MANIFEST_PATH = os.path.join(SOURCE_DIR, 'dataset_manifest.csv')
DATA_OUTPUT_ROOT = os.path.join(PROJECT_DIR, '数据')
RECORD_ROOT = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '预处理_v1_1')

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

OUTPUT_SIZE = 512
JPEG_QUALITY = 95
SHARPNESS_SIZE = 224
BLACK_THRESHOLD = 10
RANDOM_SEED = 42

EDGE_BRIGHTNESS_THRESHOLD = 25
EDGE_CONTENT_PERCENTILE = 50
EDGE_SCAN_MAX_RATIO = 0.35
EDGE_ORTHOGONAL_START = 0.20
EDGE_ORTHOGONAL_END = 0.80
EDGE_CONTENT_RUN_RATIO = 0.01
EDGE_EXPAND_RATIO = 0.015
EDGE_ALWAYS_ALLOW_TRIM_RATIO = 0.03
EDGE_WIDE_TRIM_BLACK_RATIO = 0.12
MIN_REFINED_DIM_RATIO = 0.45
PROGRESS_SEARCH_START_RATIO = 0.80
PROGRESS_MAX_BOUNDARY_RATIO = 0.97
PROGRESS_MIN_TAIL_RATIO = 0.03
PROGRESS_MIN_JUMP = 20.0
PROGRESS_MAX_TAIL_STD = 12.0

DEFAULT_DEBUG_PER_GROUP = 10
DEFAULT_TOP_RESOLUTIONS = 8
CONTACT_SHEET_ROWS = 4
CONTACT_SHEET_COLS = 4
CONTACT_PANEL_SIZE = 256

YEAR_PATTERN = re.compile(r'^(20\d{2})$')
DATE_PREFIX_PATTERN = re.compile(r'^(\d{2})(\d{2})(\d{2})')
UNIX_TIMESTAMP_PATTERN = re.compile(r'(1[4-9]\d{8})(?:\D|$)')


def parse_args():
    parser = argparse.ArgumentParser(description='第二批胃镜图像预处理 v1.1')
    parser.add_argument('--mode', choices=('debug', 'full'), default='debug')
    parser.add_argument('--debug-per-group', type=int, default=DEFAULT_DEBUG_PER_GROUP)
    parser.add_argument('--top-resolutions', type=int, default=DEFAULT_TOP_RESOLUTIONS)
    parser.add_argument('--focus-manifest', default='')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('OpenCV 无法解码图片')
    return image


def write_jpeg(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    success, encoded = cv2.imencode(
        '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not success:
        raise ValueError('OpenCV 无法编码 JPEG')
    encoded.tofile(path)


def file_sha256(path):
    with open(path, 'rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def odd_kernel_size(short_side):
    size = max(5, int(round(short_side * CLOSE_KERNEL_RATIO)))
    return size if size % 2 == 1 else size + 1


def contour_intersects_center(contour, width, height):
    window_w = width * CENTER_WINDOW_RATIO
    window_h = height * CENTER_WINDOW_RATIO
    center_x1 = (width - window_w) / 2
    center_y1 = (height - window_h) / 2
    center_x2 = center_x1 + window_w
    center_y2 = center_y1 + window_h
    x, y, w, h = cv2.boundingRect(contour)
    return not (
        x + w <= center_x1 or x >= center_x2
        or y + h <= center_y1 or y >= center_y2
    )


def full_frame_bbox(width, height, status, reason):
    return {
        'bbox': (0, 0, width, height),
        'crop_status': status,
        'crop_method': 'full_frame',
        'failure_reason': reason,
        'contour_area_ratio': 1.0 if status == 'full_frame' else 0.0,
        'center_offset_ratio': 0.0,
    }


def detect_roi(image):
    original_h, original_w = image.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(original_w, original_h))
    detect_w = max(1, int(round(original_w * scale)))
    detect_h = max(1, int(round(original_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    detect_image = cv2.resize(image, (detect_w, detect_h), interpolation=interpolation)

    gray = cv2.cvtColor(detect_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
    mask = (gray > FOREGROUND_THRESHOLD).astype(np.uint8) * 255
    kernel_size = odd_kernel_size(min(detect_w, detect_h))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [
        contour for contour in contours
        if contour_intersects_center(contour, detect_w, detect_h)
    ]
    if not candidates:
        return full_frame_bbox(
            original_w, original_h, 'fallback_full_frame', 'no_center_contour'
        )

    contour = max(candidates, key=cv2.contourArea)
    contour_area_ratio = cv2.contourArea(contour) / (detect_w * detect_h)
    x, y, w, h = cv2.boundingRect(contour)
    bbox_w_ratio = w / detect_w
    bbox_h_ratio = h / detect_h
    contour_center_x = x + w / 2
    contour_center_y = y + h / 2
    center_distance = math.hypot(
        contour_center_x - detect_w / 2,
        contour_center_y - detect_h / 2,
    )
    center_offset_ratio = center_distance / math.hypot(detect_w, detect_h)

    failure_reasons = []
    if contour_area_ratio < MIN_CONTOUR_AREA_RATIO:
        failure_reasons.append('small_contour_area')
    if bbox_w_ratio < MIN_BBOX_DIM_RATIO or bbox_h_ratio < MIN_BBOX_DIM_RATIO:
        failure_reasons.append('small_bbox')
    if center_offset_ratio > MAX_CENTER_OFFSET_RATIO:
        failure_reasons.append('off_center')
    if failure_reasons:
        result = full_frame_bbox(
            original_w,
            original_h,
            'fallback_full_frame',
            '|'.join(failure_reasons),
        )
        result['contour_area_ratio'] = contour_area_ratio
        result['center_offset_ratio'] = center_offset_ratio
        return result

    scale_x = original_w / detect_w
    scale_y = original_h / detect_h
    x1 = max(0, int(math.floor(x * scale_x)))
    y1 = max(0, int(math.floor(y * scale_y)))
    x2 = min(original_w, int(math.ceil((x + w) * scale_x)))
    y2 = min(original_h, int(math.ceil((y + h) * scale_y)))

    margins = (
        x1 / original_w,
        y1 / original_h,
        (original_w - x2) / original_w,
        (original_h - y2) / original_h,
    )
    if all(margin < FULL_FRAME_MARGIN_RATIO for margin in margins):
        result = full_frame_bbox(original_w, original_h, 'full_frame', '')
        result['contour_area_ratio'] = contour_area_ratio
        result['center_offset_ratio'] = center_offset_ratio
        return result

    inset = int(round(min(x2 - x1, y2 - y1) * INSET_RATIO))
    inset_x1 = x1 + inset
    inset_y1 = y1 + inset
    inset_x2 = x2 - inset
    inset_y2 = y2 - inset
    if inset_x2 <= inset_x1 or inset_y2 <= inset_y1:
        result = full_frame_bbox(
            original_w, original_h, 'fallback_full_frame', 'invalid_inset_bbox'
        )
        result['contour_area_ratio'] = contour_area_ratio
        result['center_offset_ratio'] = center_offset_ratio
        return result

    return {
        'bbox': (inset_x1, inset_y1, inset_x2, inset_y2),
        'crop_status': 'cropped',
        'crop_method': 'center_largest_contour_inset',
        'failure_reason': '',
        'contour_area_ratio': contour_area_ratio,
        'center_offset_ratio': center_offset_ratio,
    }


def resize_square(image, size):
    height, width = image.shape[:2]
    interpolation = cv2.INTER_AREA if max(height, width) > size else cv2.INTER_CUBIC
    return cv2.resize(image, (size, size), interpolation=interpolation)


def image_sharpness(image):
    resized = resize_square(image, SHARPNESS_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def black_pixel_ratio(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float((gray <= BLACK_THRESHOLD).mean())


def first_content_offset(scores):
    run = max(3, int(round(len(scores) * EDGE_CONTENT_RUN_RATIO)))
    limit = min(len(scores), int(round(len(scores) * EDGE_SCAN_MAX_RATIO)))
    for offset in range(max(0, limit - run + 1)):
        if np.all(scores[offset:offset + run] > EDGE_BRIGHTNESS_THRESHOLD):
            return offset
    return 0


def edge_scan_bbox(gray):
    height, width = gray.shape
    band_y1 = int(round(height * EDGE_ORTHOGONAL_START))
    band_y2 = int(round(height * EDGE_ORTHOGONAL_END))
    column_scores = np.percentile(
        gray[band_y1:band_y2], EDGE_CONTENT_PERCENTILE, axis=0
    )
    left = first_content_offset(column_scores)
    right = width - first_content_offset(column_scores[::-1])
    expand = max(1, int(round(min(width, height) * EDGE_EXPAND_RATIO)))
    return (
        max(0, left - expand),
        0,
        min(width, right + expand),
        height,
    )


def detect_progress_bar(gray, x1, x2):
    height = gray.shape[0]
    content_width = max(1, x2 - x1)
    band_x1 = x1 + int(round(content_width * EDGE_ORTHOGONAL_START))
    band_x2 = x1 + int(round(content_width * EDGE_ORTHOGONAL_END))
    band = gray[:, band_x1:band_x2].astype(np.float32)
    row_means = band.mean(axis=1)
    row_stds = band.std(axis=1)
    differences = np.abs(np.diff(row_means))
    search_start = int(round(height * PROGRESS_SEARCH_START_RATIO))
    search_end = min(
        len(differences),
        int(round(height * PROGRESS_MAX_BOUNDARY_RATIO)),
    )
    if search_end <= search_start:
        return height, False, 0.0, 0.0
    boundary = search_start + int(np.argmax(differences[search_start:search_end]))
    tail_start = boundary + 1
    tail_length = height - tail_start
    tail_std = (
        float(np.median(row_stds[tail_start:])) if tail_length else float('inf')
    )
    jump = float(differences[boundary])
    detected = (
        tail_length >= int(round(height * PROGRESS_MIN_TAIL_RATIO))
        and jump >= PROGRESS_MIN_JUMP
        and tail_std <= PROGRESS_MAX_TAIL_STD
    )
    return (tail_start if detected else height), detected, jump, tail_std


def refine_roi_by_edge_scan(image, bbox):
    original_h, original_w = image.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(original_w, original_h))
    detect_w = max(1, int(round(original_w * scale)))
    detect_h = max(1, int(round(original_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    detect_image = cv2.resize(
        image, (detect_w, detect_h), interpolation=interpolation
    )
    gray = cv2.cvtColor(detect_image, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = bbox
    dx1 = max(0, int(math.floor(x1 * scale)))
    dy1 = max(0, int(math.floor(y1 * scale)))
    dx2 = min(detect_w, int(math.ceil(x2 * scale)))
    dy2 = min(detect_h, int(math.ceil(y2 * scale)))
    roi_gray = gray[dy1:dy2, dx1:dx2]
    empty_metrics = {
        'edge_refined': False,
        'progress_bar_detected': False,
        'edge_trim_left': 0,
        'edge_trim_top': 0,
        'edge_trim_right': 0,
        'edge_trim_bottom': 0,
        'progress_jump': 0.0,
        'progress_tail_std': 0.0,
    }
    if roi_gray.size == 0:
        return bbox, empty_metrics

    rx1, ry1, rx2, ry2 = edge_scan_bbox(roi_gray)
    progress_bottom, progress_detected, progress_jump, progress_tail_std = (
        detect_progress_bar(roi_gray, rx1, rx2)
    )
    pre_refine_black_ratio = float((roi_gray <= BLACK_THRESHOLD).mean())
    allow_wide_trim = (
        progress_detected
        or pre_refine_black_ratio >= EDGE_WIDE_TRIM_BLACK_RATIO
    )
    if not allow_wide_trim:
        if rx1 > roi_gray.shape[1] * EDGE_ALWAYS_ALLOW_TRIM_RATIO:
            rx1 = 0
        if roi_gray.shape[1] - rx2 > roi_gray.shape[1] * EDGE_ALWAYS_ALLOW_TRIM_RATIO:
            rx2 = roi_gray.shape[1]
    ry2 = min(ry2, progress_bottom)
    candidate = (
        x1 if rx1 == 0 else max(x1, int(math.floor((dx1 + rx1) / scale))),
        y1 if ry1 == 0 else max(y1, int(math.floor((dy1 + ry1) / scale))),
        (
            x2 if rx2 == roi_gray.shape[1]
            else min(x2, int(math.ceil((dx1 + rx2) / scale)))
        ),
        (
            y2 if ry2 == roi_gray.shape[0]
            else min(y2, int(math.ceil((dy1 + ry2) / scale)))
        ),
    )
    cx1, cy1, cx2, cy2 = candidate
    trim = (cx1 - x1, cy1 - y1, x2 - cx2, y2 - cy2)
    valid = (
        cx2 > cx1
        and cy2 > cy1
        and cx2 - cx1 >= original_w * MIN_REFINED_DIM_RATIO
        and cy2 - cy1 >= original_h * MIN_REFINED_DIM_RATIO
        and cx1 <= original_w / 2 <= cx2
        and cy1 <= original_h / 2 <= cy2
        and all(
            value <= dimension * EDGE_SCAN_MAX_RATIO
            for value, dimension in zip(
                trim,
                (original_w, original_h, original_w, original_h),
            )
        )
    )
    refined = valid and candidate != bbox
    if not refined:
        empty_metrics['progress_jump'] = progress_jump
        empty_metrics['progress_tail_std'] = progress_tail_std
        return bbox, empty_metrics
    return candidate, {
        'edge_refined': True,
        'progress_bar_detected': progress_detected,
        'edge_trim_left': trim[0],
        'edge_trim_top': trim[1],
        'edge_trim_right': trim[2],
        'edge_trim_bottom': trim[3],
        'progress_jump': progress_jump,
        'progress_tail_std': progress_tail_std,
    }


def infer_center(source_path):
    parts = source_path.replace('\\', '/').split('/')
    if len(parts) >= 3 and parts[1] == '外院数据':
        return parts[2]
    return '武大省人民'


def infer_year(source_path, image_name):
    parts = source_path.replace('\\', '/').split('/')
    for part in parts:
        match = YEAR_PATTERN.match(part)
        if match:
            return int(match.group(1)), 'source_path'

    prefix = DATE_PREFIX_PATTERN.match(image_name)
    if prefix:
        year = 2000 + int(prefix.group(1))
        month = int(prefix.group(2))
        day = int(prefix.group(3))
        if 2000 <= year <= datetime.now().year and 1 <= month <= 12 and 1 <= day <= 31:
            return year, 'filename_date'

    timestamp_match = UNIX_TIMESTAMP_PATTERN.search(image_name)
    if timestamp_match:
        timestamp = int(timestamp_match.group(1))
        try:
            year = datetime.fromtimestamp(timestamp).year
            if 2000 <= year <= datetime.now().year:
                return year, 'filename_timestamp'
        except (OverflowError, OSError, ValueError):
            pass
    return '', 'missing'


def inspect_dimensions(manifest):
    widths = []
    heights = []
    errors = []
    for row in manifest.itertuples(index=False):
        source = os.path.join(SOURCE_DIR, row.image_path)
        try:
            image = read_image(source)
            height, width = image.shape[:2]
            widths.append(width)
            heights.append(height)
            errors.append('')
        except Exception as error:
            widths.append(0)
            heights.append(0)
            errors.append(str(error))
    result = manifest.copy()
    result['original_width'] = widths
    result['original_height'] = heights
    result['inspect_error'] = errors
    result['resolution'] = (
        result['original_width'].astype(str) + 'x' + result['original_height'].astype(str)
    )
    result['center'] = result['source_path'].map(infer_center)
    return result


def evenly_sample(group, count, seed_offset):
    if len(group) <= count:
        return group.index.tolist()
    return group.sample(n=count, random_state=RANDOM_SEED + seed_offset).index.tolist()


def select_debug_rows(manifest, per_group, top_resolutions):
    selected = set()
    valid = manifest[manifest['inspect_error'] == '']

    for seed_offset, (_, group) in enumerate(valid.groupby(['center', 'label'], sort=True)):
        group = group.sort_values(['resolution', 'image_path'])
        selected.update(evenly_sample(group, per_group, seed_offset))

    resolution_counts = valid['resolution'].value_counts()
    frequent_resolutions = resolution_counts.head(top_resolutions).index
    resolution_rows = valid[valid['resolution'].isin(frequent_resolutions)]
    start_offset = valid.groupby(['center', 'label']).ngroups
    for offset, (_, group) in enumerate(
        resolution_rows.groupby(['label', 'resolution'], sort=True)
    ):
        selected.update(evenly_sample(group, per_group, start_offset + offset))

    selected.update(manifest[manifest['inspect_error'] != ''].index.tolist())
    return manifest.loc[sorted(selected)].copy()


def prepare_output(path, overwrite):
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f'输出目录已存在，请检查后使用 --overwrite: {path}')
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def draw_preview(original, processed, bbox, title):
    original_panel = resize_square(original, CONTACT_PANEL_SIZE)
    scale_x = CONTACT_PANEL_SIZE / original.shape[1]
    scale_y = CONTACT_PANEL_SIZE / original.shape[0]
    x1, y1, x2, y2 = bbox
    cv2.rectangle(
        original_panel,
        (int(x1 * scale_x), int(y1 * scale_y)),
        (int(x2 * scale_x), int(y2 * scale_y)),
        (0, 255, 0),
        2,
    )
    processed_panel = resize_square(processed, CONTACT_PANEL_SIZE)
    pair = np.concatenate([original_panel, processed_panel], axis=1)
    cv2.rectangle(pair, (0, 0), (pair.shape[1], 24), (255, 255, 255), -1)
    cv2.putText(
        pair,
        title[:72],
        (5, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return pair


def save_contact_sheets(previews, output_dir):
    if not previews:
        return
    os.makedirs(output_dir, exist_ok=True)
    per_page = CONTACT_SHEET_ROWS * CONTACT_SHEET_COLS
    pair_h = CONTACT_PANEL_SIZE
    pair_w = CONTACT_PANEL_SIZE * 2
    for page_start in range(0, len(previews), per_page):
        page_items = previews[page_start:page_start + per_page]
        canvas = np.full(
            (CONTACT_SHEET_ROWS * pair_h, CONTACT_SHEET_COLS * pair_w, 3),
            245,
            dtype=np.uint8,
        )
        for position, preview in enumerate(page_items):
            row = position // CONTACT_SHEET_COLS
            col = position % CONTACT_SHEET_COLS
            y1 = row * pair_h
            x1 = col * pair_w
            canvas[y1:y1 + pair_h, x1:x1 + pair_w] = preview
        page_number = page_start // per_page + 1
        write_jpeg(os.path.join(output_dir, f'裁剪对照_{page_number:03d}.jpg'), canvas)


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def process_rows(rows, output_dir, collect_previews):
    processed_rows = []
    previews = []
    for number, row in enumerate(rows.itertuples(index=False), start=1):
        source_path = os.path.join(SOURCE_DIR, row.image_path)
        output_relative = os.path.splitext(row.image_path)[0] + '.jpg'
        destination = os.path.join(output_dir, output_relative)
        base = row._asdict()
        try:
            original = read_image(source_path)
            original_h, original_w = original.shape[:2]
            roi = detect_roi(original)
            pre_refine_bbox = roi['bbox']
            refined_bbox, refine = refine_roi_by_edge_scan(original, pre_refine_bbox)
            x1, y1, x2, y2 = refined_bbox
            cropped = original[y1:y2, x1:x2]
            processed = resize_square(cropped, OUTPUT_SIZE)
            write_jpeg(destination, processed)
            year, year_source = infer_year(row.source_path, row.image_name)
            result = {
                **base,
                'center': infer_center(row.source_path),
                'year': year,
                'year_source': year_source,
                'original_width': original_w,
                'original_height': original_h,
                'pre_refine_crop_x1': pre_refine_bbox[0],
                'pre_refine_crop_y1': pre_refine_bbox[1],
                'pre_refine_crop_x2': pre_refine_bbox[2],
                'pre_refine_crop_y2': pre_refine_bbox[3],
                'crop_x1': x1,
                'crop_y1': y1,
                'crop_x2': x2,
                'crop_y2': y2,
                'crop_width': x2 - x1,
                'crop_height': y2 - y1,
                'roi_area_ratio': ((x2 - x1) * (y2 - y1)) / (original_w * original_h),
                'contour_area_ratio': roi['contour_area_ratio'],
                'center_offset_ratio': roi['center_offset_ratio'],
                'crop_method': (
                    roi['crop_method'] + '+edge_scan_v1_1'
                    if refine['edge_refined'] else roi['crop_method']
                ),
                'crop_status': 'cropped' if refine['edge_refined'] else roi['crop_status'],
                'failure_reason': roi['failure_reason'],
                **refine,
                'processed_path': output_relative,
                'processed_sha256': file_sha256(destination),
                'sharpness': image_sharpness(processed),
                'black_pixel_ratio': black_pixel_ratio(processed),
                'review_status': 'pending',
            }
            processed_rows.append(result)
            title = (
                f'{number:04d} y={row.label} edge={int(refine["edge_refined"])} '
                f'bar={int(refine["progress_bar_detected"])}'
            )
            if collect_previews:
                previews.append(draw_preview(original, processed, refined_bbox, title))
        except Exception as error:
            year, year_source = infer_year(row.source_path, row.image_name)
            processed_rows.append({
                **base,
                'center': infer_center(row.source_path),
                'year': year,
                'year_source': year_source,
                'original_width': getattr(row, 'original_width', 0),
                'original_height': getattr(row, 'original_height', 0),
                'pre_refine_crop_x1': '',
                'pre_refine_crop_y1': '',
                'pre_refine_crop_x2': '',
                'pre_refine_crop_y2': '',
                'crop_x1': '',
                'crop_y1': '',
                'crop_x2': '',
                'crop_y2': '',
                'crop_width': '',
                'crop_height': '',
                'roi_area_ratio': '',
                'contour_area_ratio': '',
                'center_offset_ratio': '',
                'crop_method': '',
                'crop_status': 'decode_error',
                'failure_reason': str(error),
                'edge_refined': False,
                'progress_bar_detected': False,
                'edge_trim_left': '',
                'edge_trim_top': '',
                'edge_trim_right': '',
                'edge_trim_bottom': '',
                'progress_jump': '',
                'progress_tail_std': '',
                'processed_path': '',
                'processed_sha256': '',
                'sharpness': '',
                'black_pixel_ratio': '',
                'review_status': 'pending',
            })
        if number % 100 == 0 or number == len(rows):
            print(f'处理进度: {number}/{len(rows)}')
    return processed_rows, previews


def save_summary(rows, record_dir, mode, total_manifest_rows):
    status_counts = Counter(row['crop_status'] for row in rows)
    label_status = Counter((row['label'], row['crop_status']) for row in rows)
    center_status = Counter((row['center'], row['crop_status']) for row in rows)
    edge_refined_count = sum(bool(row['edge_refined']) for row in rows)
    progress_bar_count = sum(bool(row['progress_bar_detected']) for row in rows)

    summary_lines = [
        f'模式: {mode}',
        f'原始清单: {total_manifest_rows} 张',
        f'本次处理: {len(rows)} 张',
        f'四边扫描收紧: {edge_refined_count} 张',
        f'检测并移除进度条: {progress_bar_count} 张',
        '',
        '裁剪状态:',
    ]
    for status, count in sorted(status_counts.items()):
        summary_lines.append(f'  {status}: {count} ({count / len(rows):.2%})')
    summary_lines.extend(['', '按标签裁剪状态:'])
    for (label, status), count in sorted(label_status.items()):
        denominator = sum(row['label'] == label for row in rows)
        summary_lines.append(f'  label={label} {status}: {count}/{denominator} ({count / denominator:.2%})')
    summary_lines.extend(['', '按中心裁剪状态:'])
    for (center, status), count in sorted(center_status.items()):
        denominator = sum(row['center'] == center for row in rows)
        summary_lines.append(f'  {center} {status}: {count}/{denominator} ({count / denominator:.2%})')

    summary_path = os.path.join(record_dir, '处理汇总.txt')
    with open(summary_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(summary_lines) + '\n')
    print('\n'.join(summary_lines))


def main():
    args = parse_args()
    suffix = '_debug' if args.mode == 'debug' else ''
    output_dir = os.path.join(DATA_OUTPUT_ROOT, f'第二批裁剪后_v1_1{suffix}')
    record_dir = os.path.join(RECORD_ROOT, args.mode)
    prepare_output(output_dir, args.overwrite)
    prepare_output(record_dir, args.overwrite)

    manifest = pd.read_csv(MANIFEST_PATH, encoding='utf-8-sig')
    inspected = inspect_dimensions(manifest)
    if args.mode == 'debug':
        if args.focus_manifest:
            focus = pd.read_csv(args.focus_manifest, encoding='utf-8-sig')
            requested = focus['image_path'].drop_duplicates().tolist()
            indexed = inspected.set_index('image_path', drop=False)
            missing = sorted(set(requested) - set(indexed.index))
            if missing:
                raise ValueError(f'focus manifest 中有 {len(missing)} 个路径不存在')
            rows = indexed.loc[requested].reset_index(drop=True)
        else:
            rows = select_debug_rows(
                inspected,
                per_group=args.debug_per_group,
                top_resolutions=args.top_resolutions,
            )
        rows.to_csv(
            os.path.join(record_dir, '调试样本清单.csv'),
            index=False,
            encoding='utf-8-sig',
        )
    else:
        rows = inspected

    processed_rows, previews = process_rows(
        rows,
        output_dir,
        collect_previews=args.mode == 'debug',
    )
    fields = list(processed_rows[0].keys())
    write_csv(
        os.path.join(output_dir, 'processed_manifest.csv'),
        processed_rows,
        fields,
    )
    write_csv(
        os.path.join(record_dir, 'processed_manifest.csv'),
        processed_rows,
        fields,
    )
    save_contact_sheets(previews, os.path.join(record_dir, '裁剪对照'))
    save_summary(processed_rows, record_dir, args.mode, len(manifest))
    print(f'\n处理后数据: {output_dir}')
    print(f'处理记录: {record_dir}')


if __name__ == '__main__':
    main()
