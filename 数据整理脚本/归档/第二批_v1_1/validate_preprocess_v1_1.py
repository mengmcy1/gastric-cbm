"""验收第二批预处理结果，输出硬性检查与肉眼复核清单。"""

import argparse
import hashlib
import os
from collections import Counter

import cv2
import numpy as np
import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_DIR, '数据')
RECORD_ROOT = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '预处理_v1_1')

EXPECTED_SIZE = 512
MAX_FAILURE_RATE = 0.05
MAX_FAILURE_GAP = 0.05
MIN_ROI_DIM_RATIO = 0.45
MAX_CENTER_OFFSET_RATIO = 0.20
HIGH_BLACK_RATIO = 0.15
CONTACT_SHEET_SIZE = 16
REVIEW_PER_CENTER_LABEL = 3
RANDOM_SEED = 42

HARD_FAILURE_FLAGS = {
    'missing_file',
    'decode_error',
    'wrong_size',
    'wrong_channels',
    'hash_mismatch',
    'invalid_bbox',
    'small_roi',
    'off_center',
    'source_center_outside_bbox',
    'invalid_refinement',
}


def parse_args():
    parser = argparse.ArgumentParser(description='第二批预处理结果验收 v1.1')
    parser.add_argument('--mode', choices=('debug', 'full'), default='debug')
    return parser.parse_args()


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def file_sha256(path):
    with open(path, 'rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def output_paths(mode):
    suffix = '_debug' if mode == 'debug' else ''
    data_dir = os.path.join(DATA_ROOT, f'第二批裁剪后_v1_1{suffix}')
    record_dir = os.path.join(RECORD_ROOT, mode)
    manifest_path = os.path.join(data_dir, 'processed_manifest.csv')
    return data_dir, record_dir, manifest_path


def split_flags(value):
    if not value:
        return set()
    return set(value.split('|'))


def validate_row(row, data_dir):
    hard_flags = []
    warning_flags = []
    processed_path = os.path.join(data_dir, str(row.processed_path))

    if row.crop_status == 'decode_error':
        hard_flags.append('decode_error')
    elif row.crop_status == 'fallback_full_frame':
        warning_flags.append('fallback_full_frame')

    trim_values = [
        int(row.edge_trim_left),
        int(row.edge_trim_top),
        int(row.edge_trim_right),
        int(row.edge_trim_bottom),
    ]
    pre_bbox = [
        int(row.pre_refine_crop_x1),
        int(row.pre_refine_crop_y1),
        int(row.pre_refine_crop_x2),
        int(row.pre_refine_crop_y2),
    ]
    final_bbox = [
        int(row.crop_x1),
        int(row.crop_y1),
        int(row.crop_x2),
        int(row.crop_y2),
    ]
    final_is_subset = (
        final_bbox[0] >= pre_bbox[0]
        and final_bbox[1] >= pre_bbox[1]
        and final_bbox[2] <= pre_bbox[2]
        and final_bbox[3] <= pre_bbox[3]
    )
    refinement_consistent = (
        bool(row.edge_refined) == any(trim_values)
        and trim_values[1] == 0
        and (trim_values[3] > 0) == bool(row.progress_bar_detected)
    )
    if not final_is_subset or not refinement_consistent:
        hard_flags.append('invalid_refinement')
    if not os.path.isfile(processed_path):
        hard_flags.append('missing_file')
        return hard_flags, warning_flags

    image = read_image(processed_path)
    if image is None:
        hard_flags.append('decode_error')
        return hard_flags, warning_flags
    if image.shape[:2] != (EXPECTED_SIZE, EXPECTED_SIZE):
        hard_flags.append('wrong_size')
    if image.ndim != 3 or image.shape[2] != 3:
        hard_flags.append('wrong_channels')
    if file_sha256(processed_path) != row.processed_sha256:
        hard_flags.append('hash_mismatch')

    bbox_values = [row.crop_x1, row.crop_y1, row.crop_x2, row.crop_y2]
    if any(pd.isna(value) for value in bbox_values):
        hard_flags.append('invalid_bbox')
    else:
        x1, y1, x2, y2 = map(int, bbox_values)
        valid_bbox = (
            0 <= x1 < x2 <= int(row.original_width)
            and 0 <= y1 < y2 <= int(row.original_height)
        )
        if not valid_bbox:
            hard_flags.append('invalid_bbox')
        else:
            source_center_x = int(row.original_width) / 2
            source_center_y = int(row.original_height) / 2
            if not (x1 <= source_center_x <= x2 and y1 <= source_center_y <= y2):
                hard_flags.append('source_center_outside_bbox')

    width_ratio = float(row.crop_width) / float(row.original_width)
    height_ratio = float(row.crop_height) / float(row.original_height)
    if width_ratio < MIN_ROI_DIM_RATIO or height_ratio < MIN_ROI_DIM_RATIO:
        hard_flags.append('small_roi')
    if float(row.center_offset_ratio) > MAX_CENTER_OFFSET_RATIO:
        hard_flags.append('off_center')
    if float(row.black_pixel_ratio) > HIGH_BLACK_RATIO:
        warning_flags.append('high_black_ratio')
    if float(row.roi_area_ratio) < 0.60:
        warning_flags.append('low_roi_area')
    if float(row.center_offset_ratio) > 0.10:
        warning_flags.append('high_center_offset')
    if float(row.sharpness) < 60:
        warning_flags.append('low_sharpness')
    return hard_flags, warning_flags


def group_failure_rates(manifest):
    failure_mask = manifest['hard_flags'].fillna('').map(
        lambda value: bool(split_flags(value) & HARD_FAILURE_FLAGS)
    )
    rates = []
    for group_name, columns in (
        ('label', ['label']),
        ('center', ['center']),
        ('center_label', ['center', 'label']),
    ):
        grouped = manifest.assign(hard_failure=failure_mask).groupby(columns, dropna=False)
        for key, group in grouped:
            key = key if isinstance(key, tuple) else (key,)
            rates.append({
                'group_type': group_name,
                'group_value': ' | '.join(map(str, key)),
                'count': len(group),
                'hard_failures': int(group['hard_failure'].sum()),
                'failure_rate': float(group['hard_failure'].mean()),
            })
    return pd.DataFrame(rates)


def select_review_rows(manifest, mode):
    selected = set(manifest.index[manifest['hard_flags'].fillna('').astype(bool)])
    selected.update(manifest.index[manifest['warning_flags'].fillna('').astype(bool)])
    selected.update(manifest.nsmallest(10, 'roi_area_ratio').index)
    selected.update(manifest.nlargest(10, 'center_offset_ratio').index)
    selected.update(manifest.nlargest(10, 'black_pixel_ratio').index)

    for offset, (_, group) in enumerate(manifest.groupby(['center', 'label'], sort=True)):
        count = min(REVIEW_PER_CENTER_LABEL, len(group))
        sampled = group.sample(n=count, random_state=RANDOM_SEED + offset)
        selected.update(sampled.index)

    review = manifest.loc[sorted(selected)].copy()
    review.insert(0, 'manifest_row', review.index + 1)
    if mode == 'debug':
        review.insert(1, 'contact_page', review['manifest_row'].map(
            lambda value: (value - 1) // CONTACT_SHEET_SIZE + 1
        ))
        review.insert(2, 'contact_slot', review['manifest_row'].map(
            lambda value: (value - 1) % CONTACT_SHEET_SIZE + 1
        ))
    columns = [
        'manifest_row',
        *(['contact_page', 'contact_slot'] if mode == 'debug' else []),
        'label',
        'center',
        'image_path',
        'crop_status',
        'hard_flags',
        'warning_flags',
        'roi_area_ratio',
        'center_offset_ratio',
        'black_pixel_ratio',
        'sharpness',
    ]
    return review[columns]


def count_output_images(data_dir):
    count = 0
    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.lower().endswith(('.jpg', '.jpeg')):
                count += 1
    return count


def main():
    args = parse_args()
    data_dir, record_dir, manifest_path = output_paths(args.mode)
    manifest = pd.read_csv(manifest_path, encoding='utf-8-sig')

    hard_values = []
    warning_values = []
    for row in manifest.itertuples(index=False):
        hard_flags, warning_flags = validate_row(row, data_dir)
        hard_values.append('|'.join(sorted(set(hard_flags))))
        warning_values.append('|'.join(sorted(set(warning_flags))))
    manifest['hard_flags'] = hard_values
    manifest['warning_flags'] = warning_values

    duplicate_paths = int(manifest['processed_path'].duplicated().sum())
    duplicate_hashes = int(manifest['processed_sha256'].duplicated().sum())
    hash_label_counts = manifest.groupby('processed_sha256')['label'].nunique()
    cross_label_hashes = int((hash_label_counts > 1).sum())
    output_image_count = count_output_images(data_dir)
    hard_failure_count = int(manifest['hard_flags'].map(
        lambda value: bool(split_flags(value) & HARD_FAILURE_FLAGS)
    ).sum())

    group_rates = group_failure_rates(manifest)
    excessive_groups = group_rates[group_rates['failure_rate'] > MAX_FAILURE_RATE]
    label_rates = group_rates[group_rates['group_type'] == 'label']['failure_rate']
    label_failure_gap = float(label_rates.max() - label_rates.min()) if len(label_rates) else 0.0

    global_checks = {
        'manifest_nonempty': len(manifest) > 0,
        'output_count_matches_manifest': output_image_count == len(manifest),
        'no_duplicate_processed_paths': duplicate_paths == 0,
        'no_cross_label_processed_hashes': cross_label_hashes == 0,
        'no_hard_row_failures': hard_failure_count == 0,
        'group_failure_rate_within_limit': excessive_groups.empty,
        'label_failure_gap_within_limit': label_failure_gap <= MAX_FAILURE_GAP,
    }
    passed = all(global_checks.values())

    os.makedirs(record_dir, exist_ok=True)
    manifest.to_csv(
        os.path.join(record_dir, '自动验收逐图结果.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    group_rates.to_csv(
        os.path.join(record_dir, '自动验收分组失败率.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    review = select_review_rows(manifest, args.mode)
    review.to_csv(
        os.path.join(record_dir, '需要肉眼复核.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    warning_counter = Counter()
    for value in manifest['warning_flags']:
        warning_counter.update(split_flags(value))

    lines = [
        f'自动验收结论: {"PASS" if passed else "FAIL"}',
        f'模式: {args.mode}',
        f'manifest 行数: {len(manifest)}',
        f'输出图片数: {output_image_count}',
        f'逐图硬失败: {hard_failure_count}',
        f'重复 processed_path: {duplicate_paths}',
        f'重复 processed_sha256: {duplicate_hashes}',
        f'跨标签重复 processed_sha256: {cross_label_hashes}',
        f'标签失败率差: {label_failure_gap:.2%}',
        f'肉眼复核清单: {len(review)} 张',
        '',
        '全局检查:',
    ]
    lines.extend(f'  {name}: {"PASS" if value else "FAIL"}' for name, value in global_checks.items())
    lines.append('')
    lines.append('警告（不自动否决）:')
    if warning_counter:
        lines.extend(f'  {name}: {count}' for name, count in sorted(warning_counter.items()))
    else:
        lines.append('  无')
    lines.extend([
        '',
        '说明: 自动 PASS 仅表示文件、清单、裁剪几何及失败率满足 v1 硬门槛；',
        '是否切到病灶或保留不合理界面仍必须查看裁剪联系图。',
    ])
    report = '\n'.join(lines) + '\n'
    with open(os.path.join(record_dir, '自动验收报告.txt'), 'w', encoding='utf-8') as file:
        file.write(report)
    print(report)

    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
