"""对医学生整理概念集复用第二批已验收的 v1.1 裁剪规则。"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / '数据整理脚本'))

from preprocess_second_batch_v1_1 import (  # noqa: E402
    OUTPUT_SIZE,
    detect_roi,
    draw_preview,
    file_sha256,
    image_sharpness,
    black_pixel_ratio,
    read_image,
    refine_roi_by_edge_scan,
    resize_square,
    save_contact_sheets,
    write_jpeg,
)


AUDIT_MANIFEST = (
    PROJECT_DIR / '数据整理记录' / '概念提取训练集_v1'
    / '原始审计' / 'original_manifest.csv'
)
OUTPUT_ROOT = PROJECT_DIR / '数据'
RECORD_ROOT = (
    PROJECT_DIR / '数据整理记录' / '概念提取训练集_v1' / '预处理_v1_1'
)
RANDOM_SEED = 42
FIRST_COMPETITION_SOURCE = '第一届早癌大赛'
VIEWPORT_SATURATION_THRESHOLD = 35
VIEWPORT_VALUE_THRESHOLD = 30
VIEWPORT_COLUMN_RATIO = 0.25
VIEWPORT_ROW_RATIO = 0.20
VIEWPORT_CONTENT_RUN = 12
VIEWPORT_EXPAND_RATIO = 0.015
MIN_ROI_AREA_HARD = 0.40
MIN_ROI_AREA_WARNING = 0.45


def first_content_run(scores, threshold):
    content = scores > threshold
    for index in range(len(content) - VIEWPORT_CONTENT_RUN + 1):
        if content[index:index + VIEWPORT_CONTENT_RUN].all():
            return index
    return None


def detect_first_competition_viewport(image):
    """用彩色黏膜占比排除第一届大赛截图两侧的灰黑设备界面。"""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_mask = (
        (hsv[:, :, 1] > VIEWPORT_SATURATION_THRESHOLD)
        & (hsv[:, :, 2] > VIEWPORT_VALUE_THRESHOLD)
    )
    column_scores = color_mask[
        int(round(height * 0.10)):int(round(height * 0.90))
    ].mean(axis=0)
    row_scores = color_mask[
        :, int(round(width * 0.10)):int(round(width * 0.90))
    ].mean(axis=1)
    left = first_content_run(column_scores, VIEWPORT_COLUMN_RATIO)
    right_offset = first_content_run(column_scores[::-1], VIEWPORT_COLUMN_RATIO)
    top = first_content_run(row_scores, VIEWPORT_ROW_RATIO)
    bottom_offset = first_content_run(row_scores[::-1], VIEWPORT_ROW_RATIO)
    if None in (left, right_offset, top, bottom_offset):
        return None

    right = width - right_offset
    bottom = height - bottom_offset
    expand = max(1, int(round(min(width, height) * VIEWPORT_EXPAND_RATIO)))
    candidate = (
        max(0, left - expand),
        max(0, top - expand),
        min(width, right + expand),
        min(height, bottom + expand),
    )
    x1, y1, x2, y2 = candidate
    area_ratio = ((x2 - x1) * (y2 - y1)) / (width * height)
    valid = (
        x2 > x1
        and y2 > y1
        and area_ratio >= 0.45
        and x1 <= width / 2 <= x2
        and y1 <= height / 2 <= y2
    )
    return candidate if valid else None


def parse_args():
    parser = argparse.ArgumentParser(description='医学生整理概念集 v1.1 裁剪')
    parser.add_argument('--mode', choices=('debug', 'full'), default='debug')
    parser.add_argument('--debug-per-source', type=int, default=8)
    parser.add_argument('--debug-per-resolution', type=int, default=4)
    parser.add_argument('--top-resolutions', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def prepare_directory(path, overwrite):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f'输出目录已存在，请检查后加 --overwrite: {path}')
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def select_debug_rows(manifest, args):
    selected = set()
    for offset, (_, group) in enumerate(
        manifest.groupby(['label', 'source'], sort=True)
    ):
        count = min(args.debug_per_source, len(group))
        selected.update(
            group.sample(count, random_state=RANDOM_SEED + offset).index
        )

    seed_offset = manifest.groupby(['label', 'source']).ngroups
    for label in sorted(manifest['label'].unique()):
        label_rows = manifest[manifest['label'] == label]
        top = label_rows['resolution'].value_counts().head(args.top_resolutions).index
        for offset, resolution in enumerate(top):
            group = label_rows[label_rows['resolution'] == resolution]
            count = min(args.debug_per_resolution, len(group))
            selected.update(
                group.sample(
                    count,
                    random_state=RANDOM_SEED + seed_offset + label * 100 + offset,
                ).index
            )

    selected.update(manifest[manifest['review_flags'].fillna('') != ''].index)
    return manifest.loc[sorted(selected)].copy()


def output_relpath(original_relpath):
    return str(Path(original_relpath).with_suffix('.jpg'))


def process_rows(rows, output_dir, collect_previews):
    processed_rows = []
    previews = []
    for number, row in enumerate(rows.itertuples(index=False), start=1):
        destination = output_dir / output_relpath(row.original_relpath)
        try:
            original = read_image(row.original_path)
            original_h, original_w = original.shape[:2]
            roi = detect_roi(original)
            pre_refine_bbox = roi['bbox']
            bbox, refine = refine_roi_by_edge_scan(original, pre_refine_bbox)
            device_viewport_applied = False
            if row.source == FIRST_COMPETITION_SOURCE:
                device_bbox = detect_first_competition_viewport(original)
                if device_bbox is not None:
                    bbox = device_bbox
                    device_viewport_applied = True
            x1, y1, x2, y2 = bbox
            cropped = original[y1:y2, x1:x2]
            processed = resize_square(cropped, OUTPUT_SIZE)
            write_jpeg(str(destination), processed)
            result = {
                **row._asdict(),
                'processed_relpath': str(destination.relative_to(output_dir)),
                'processed_path': str(destination.resolve()),
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
                'crop_status': (
                    'cropped'
                    if device_viewport_applied or refine['edge_refined']
                    else roi['crop_status']
                ),
                'crop_method': (
                    'hsv_viewport_first_competition_v1'
                    if device_viewport_applied
                    else (
                        roi['crop_method'] + '+edge_scan_v1_1'
                        if refine['edge_refined'] else roi['crop_method']
                    )
                ),
                'failure_reason': roi['failure_reason'],
                **refine,
                'device_viewport_applied': device_viewport_applied,
                'processed_sha256': file_sha256(destination),
                'sharpness': image_sharpness(processed),
                'black_pixel_ratio': black_pixel_ratio(processed),
            }
            processed_rows.append(result)
            if collect_previews:
                title = (
                    f'{number:04d} y={row.label} {row.source[:12]} '
                    f'edge={int(refine["edge_refined"])} '
                    f'bar={int(refine["progress_bar_detected"])} '
                    f'view={int(device_viewport_applied)}'
                )
                previews.append(draw_preview(original, processed, bbox, title))
        except Exception as error:
            processed_rows.append({
                **row._asdict(),
                'processed_relpath': '',
                'processed_path': '',
                'crop_status': 'decode_error',
                'crop_method': '',
                'failure_reason': str(error),
            })
        if number % 100 == 0 or number == len(rows):
            print(f'裁剪进度: {number}/{len(rows)}')
    return pd.DataFrame(processed_rows), previews


def validate_results(results):
    valid = results[results['crop_status'] != 'decode_error'].copy()
    output_exists = valid['processed_path'].map(lambda path: Path(path).exists())
    checks = {
        'row_count': len(results),
        'decode_errors': int((results['crop_status'] == 'decode_error').sum()),
        'missing_outputs': int((~output_exists).sum()),
        'duplicate_output_paths': int(
            valid['processed_path'].duplicated().sum()
        ),
        'minimum_roi_area_ratio': float(valid['roi_area_ratio'].min()),
        'low_roi_area_warnings': int(
            (valid['roi_area_ratio'] < MIN_ROI_AREA_WARNING).sum()
        ),
        'cropped': int((valid['crop_status'] == 'cropped').sum()),
        'full_frame': int((valid['crop_status'] == 'full_frame').sum()),
        'fallback_full_frame': int(
            (valid['crop_status'] == 'fallback_full_frame').sum()
        ),
        'edge_refined': int(valid['edge_refined'].sum()),
        'progress_bar_detected': int(valid['progress_bar_detected'].sum()),
        'device_viewport_applied': int(valid['device_viewport_applied'].sum()),
    }
    checks['passed'] = (
        checks['decode_errors'] == 0
        and checks['missing_outputs'] == 0
        and checks['duplicate_output_paths'] == 0
        and checks['minimum_roi_area_ratio'] >= MIN_ROI_AREA_HARD
    )
    return checks


def main():
    args = parse_args()
    suffix = '_debug' if args.mode == 'debug' else ''
    output_dir = OUTPUT_ROOT / f'胃早癌概念提取训练集_裁剪后_v1_1{suffix}'
    record_dir = RECORD_ROOT / args.mode
    prepare_directory(output_dir, args.overwrite)
    prepare_directory(record_dir, args.overwrite)

    manifest = pd.read_csv(
        AUDIT_MANIFEST, encoding='utf-8-sig', dtype={'patient_id': str}
    )
    rows = select_debug_rows(manifest, args) if args.mode == 'debug' else manifest
    relative_outputs = rows['original_relpath'].map(output_relpath)
    if relative_outputs.duplicated().any():
        duplicates = relative_outputs[relative_outputs.duplicated(False)].tolist()
        raise ValueError(f'输出相对路径冲突: {duplicates[:10]}')

    results, previews = process_rows(
        rows, output_dir, collect_previews=args.mode == 'debug'
    )
    results.to_csv(
        record_dir / 'processed_manifest.csv', index=False, encoding='utf-8-sig'
    )
    if previews:
        save_contact_sheets(previews, str(record_dir / '裁剪对照'))
    validation = validate_results(results)
    with open(record_dir / 'validation_summary.json', 'w', encoding='utf-8') as file:
        json.dump(validation, file, ensure_ascii=False, indent=2)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f'裁剪数据: {output_dir}')
    print(f'处理记录: {record_dir}')
    if not validation['passed']:
        raise RuntimeError('自动验收未通过，请检查 validation_summary.json')


if __name__ == '__main__':
    main()
