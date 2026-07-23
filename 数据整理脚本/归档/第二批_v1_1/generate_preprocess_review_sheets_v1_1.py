"""为 v1.1 全量预处理结果生成风险导向的分层肉眼复核联系图。"""

import argparse
import importlib.util
import os
import shutil

import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_DIR, '数据', '第二批整理后')
RECORD_ROOT = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '预处理_v1_1')
RANDOM_SEED = 42
EXTREME_COUNT = 32
STRATIFIED_PER_GROUP = 2
REFINED_PER_GROUP = 2
PROGRESS_PER_GROUP = 4


def load_preprocess_module():
    path = os.path.join(SCRIPT_DIR, 'preprocess_second_batch_v1_1.py')
    spec = importlib.util.spec_from_file_location('preprocess_second_batch_v1_1', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description='生成 v1.1 预处理肉眼复核联系图')
    parser.add_argument('--mode', choices=('debug', 'full'), default='full')
    return parser.parse_args()


def add_reason(selected, indexes, reason):
    for index in indexes:
        selected.setdefault(index, set()).add(reason)


def add_group_samples(selected, frame, count, reason, seed_offset):
    for offset, (_, group) in enumerate(frame.groupby(['center', 'label'], sort=True)):
        sample_count = min(count, len(group))
        sample = group.sample(n=sample_count, random_state=RANDOM_SEED + seed_offset + offset)
        add_reason(selected, sample.index, reason)


def select_rows(manifest):
    selected = {}
    hard = manifest['hard_flags'].fillna('').astype(bool)
    fallback = manifest['crop_status'].eq('fallback_full_frame')
    high_black = manifest['warning_flags'].fillna('').str.contains('high_black_ratio')
    refined = manifest['edge_refined'].astype(bool)
    progress = manifest['progress_bar_detected'].astype(bool)

    add_reason(selected, manifest.index[hard], 'hard')
    add_reason(selected, manifest.index[fallback], 'fallback')
    add_reason(selected, manifest.index[high_black], 'high_black')
    add_reason(selected, manifest.nlargest(EXTREME_COUNT, 'edge_trim_left').index, 'left_trim')
    add_reason(selected, manifest.nlargest(EXTREME_COUNT, 'edge_trim_right').index, 'right_trim')
    add_reason(selected, manifest.nlargest(EXTREME_COUNT, 'edge_trim_bottom').index, 'bottom_trim')
    add_reason(selected, manifest.nlargest(EXTREME_COUNT, 'black_pixel_ratio').index, 'black')
    add_reason(selected, manifest.nlargest(EXTREME_COUNT, 'center_offset_ratio').index, 'center')
    add_reason(selected, manifest.nsmallest(EXTREME_COUNT, 'roi_area_ratio').index, 'roi')
    add_reason(selected, manifest.nsmallest(EXTREME_COUNT, 'sharpness').index, 'sharp')

    add_group_samples(selected, manifest, STRATIFIED_PER_GROUP, 'sample', 0)
    add_group_samples(selected, manifest.loc[refined], REFINED_PER_GROUP, 'refined_sample', 100)
    add_group_samples(selected, manifest.loc[progress], PROGRESS_PER_GROUP, 'progress_sample', 200)

    review = manifest.loc[sorted(selected)].copy()
    review.insert(0, 'manifest_row', review.index + 1)
    review.insert(1, 'review_reason', [
        '|'.join(sorted(selected[index])) for index in review.index
    ])
    return review


def main():
    args = parse_args()
    suffix = '_debug' if args.mode == 'debug' else ''
    data_dir = os.path.join(PROJECT_DIR, '数据', f'第二批裁剪后_v1_1{suffix}')
    record_dir = os.path.join(RECORD_ROOT, args.mode)
    validation_path = os.path.join(record_dir, '自动验收逐图结果.csv')
    output_dir = os.path.join(record_dir, '裁剪对照_分层复核')
    review_path = os.path.join(record_dir, '分层肉眼复核清单.csv')

    manifest = pd.read_csv(validation_path, encoding='utf-8-sig')
    review = select_rows(manifest)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    preprocess = load_preprocess_module()
    previews = []
    for row in review.itertuples(index=False):
        original = preprocess.read_image(os.path.join(SOURCE_DIR, row.image_path))
        processed = preprocess.read_image(os.path.join(data_dir, row.processed_path))
        bbox = tuple(map(int, (row.crop_x1, row.crop_y1, row.crop_x2, row.crop_y2)))
        title = (
            f'{row.manifest_row:04d} y={row.label} {row.review_reason} '
            f'L{row.edge_trim_left}/R{row.edge_trim_right}/B{row.edge_trim_bottom} '
            f'bar={int(row.progress_bar_detected)} roi={row.roi_area_ratio:.2f}'
        )
        previews.append(preprocess.draw_preview(original, processed, bbox, title))

    preprocess.save_contact_sheets(previews, output_dir)
    review.to_csv(review_path, index=False, encoding='utf-8-sig')
    print(f'分层复核样本: {len(review)} 张')
    print(f'复核清单: {review_path}')
    print(f'裁剪对照: {output_dir}')


if __name__ == '__main__':
    main()
