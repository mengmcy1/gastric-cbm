"""第二批胃镜图像预处理 v1：自动裁剪黑边、统一输出并生成审计清单。"""

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
RECORD_ROOT = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '预处理_v1')

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

DEFAULT_DEBUG_PER_GROUP = 10
DEFAULT_TOP_RESOLUTIONS = 8
CONTACT_SHEET_ROWS = 4
CONTACT_SHEET_COLS = 4
CONTACT_PANEL_SIZE = 256

YEAR_PATTERN = re.compile(r'^(20\d{2})$')
DATE_PREFIX_PATTERN = re.compile(r'^(\d{2})(\d{2})(\d{2})')
UNIX_TIMESTAMP_PATTERN = re.compile(r'(1[4-9]\d{8})(?:\D|$)')


def parse_args():
    parser = argparse.ArgumentParser(description='第二批胃镜图像预处理 v1')
    parser.add_argument('--mode', choices=('debug', 'full'), default='debug')
    parser.add_argument('--debug-per-group', type=int, default=DEFAULT_DEBUG_PER_GROUP)
    parser.add_argument('--top-resolutions', type=int, default=DEFAULT_TOP_RESOLUTIONS)
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
            x1, y1, x2, y2 = roi['bbox']
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
                'crop_x1': x1,
                'crop_y1': y1,
                'crop_x2': x2,
                'crop_y2': y2,
                'crop_width': x2 - x1,
                'crop_height': y2 - y1,
                'roi_area_ratio': ((x2 - x1) * (y2 - y1)) / (original_w * original_h),
                'contour_area_ratio': roi['contour_area_ratio'],
                'center_offset_ratio': roi['center_offset_ratio'],
                'crop_method': roi['crop_method'],
                'crop_status': roi['crop_status'],
                'failure_reason': roi['failure_reason'],
                'processed_path': output_relative,
                'processed_sha256': file_sha256(destination),
                'sharpness': image_sharpness(processed),
                'black_pixel_ratio': black_pixel_ratio(processed),
                'review_status': 'pending',
            }
            processed_rows.append(result)
            title = f'{number:04d} y={row.label} {result["center"]} {roi["crop_status"]}'
            if collect_previews:
                previews.append(draw_preview(original, processed, roi['bbox'], title))
        except Exception as error:
            year, year_source = infer_year(row.source_path, row.image_name)
            processed_rows.append({
                **base,
                'center': infer_center(row.source_path),
                'year': year,
                'year_source': year_source,
                'original_width': getattr(row, 'original_width', 0),
                'original_height': getattr(row, 'original_height', 0),
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

    summary_lines = [
        f'模式: {mode}',
        f'原始清单: {total_manifest_rows} 张',
        f'本次处理: {len(rows)} 张',
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
    output_dir = os.path.join(DATA_OUTPUT_ROOT, f'第二批裁剪后_v1{suffix}')
    record_dir = os.path.join(RECORD_ROOT, args.mode)
    prepare_output(output_dir, args.overwrite)
    prepare_output(record_dir, args.overwrite)

    manifest = pd.read_csv(MANIFEST_PATH, encoding='utf-8-sig')
    inspected = inspect_dimensions(manifest)
    if args.mode == 'debug':
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
