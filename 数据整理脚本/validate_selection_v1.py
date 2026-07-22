"""独立验收第二批 P2 患者内去重与限图结果。"""

import argparse
import os
import re

import cv2
import numpy as np
import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, '数据', '第二批裁剪后_v1_1')
SOURCE_MANIFEST_PATH = os.path.join(DATA_DIR, 'processed_manifest.csv')
RECORD_ROOT = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '筛图_v1')
DHASH_THRESHOLD = 4
MAX_IMAGES_PER_PATIENT = 3
HEX_PATTERN = re.compile(r'^[0-9a-f]{16}$')


def parse_args():
    parser = argparse.ArgumentParser(description='验收第二批 P2 筛图结果')
    parser.add_argument('--mode', choices=('debug', 'full'), default='debug')
    return parser.parse_args()


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'OpenCV 无法解码图片: {path}')
    return image


def independent_dhash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (resized[:, 1:] > resized[:, :-1]).reshape(-1)
    return ''.join(f'{sum(int(bit) << (7 - offset) for offset, bit in enumerate(bits[i:i + 8])):02x}'
                   for i in range(0, 64, 8))


def hamming_hex(left, right):
    return (int(left, 16) ^ int(right, 16)).bit_count()


def expected_components(hashes):
    adjacency = {index: set() for index in range(len(hashes))}
    for left in range(len(hashes)):
        for right in range(left + 1, len(hashes)):
            if hamming_hex(hashes[left], hashes[right]) <= DHASH_THRESHOLD:
                adjacency[left].add(right)
                adjacency[right].add(left)
    components = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        stack = [start]
        component = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        unseen -= component
        components.append(component)
    return components


def bool_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().map({'true': True, 'false': False})


def validate_patient(group):
    flags = {index: [] for index in group.index}
    selected = bool_series(group['selected'])
    representatives = bool_series(group['is_duplicate_representative'])
    patient_flags = []

    if group['label'].nunique() != 1:
        patient_flags.append('mixed_label')
    if group['center'].nunique() != 1:
        patient_flags.append('mixed_center')
    selected_count = int(selected.sum())
    group_count = int(group['duplicate_group'].nunique())
    if selected_count != min(MAX_IMAGES_PER_PATIENT, group_count):
        patient_flags.append('wrong_selected_count')
    if (selected & ~representatives).any():
        patient_flags.append('selected_nonrepresentative')
    ranks = sorted(group.loc[selected, 'selection_rank'].dropna().astype(int).tolist())
    if ranks != list(range(1, selected_count + 1)):
        patient_flags.append('invalid_selection_ranks')
    if not group.loc[selected, 'review_status'].eq('pending').all():
        patient_flags.append('invalid_review_status')

    hashes = group['dhash_hex'].tolist()
    path_by_local = group['processed_path'].tolist()
    actual_group_by_local = group['duplicate_group'].tolist()
    for component in expected_components(hashes):
        actual_groups = {actual_group_by_local[index] for index in component}
        if len(actual_groups) != 1:
            patient_flags.append('duplicate_component_split')
        component_group = next(iter(actual_groups))
        actual_members = {
            index for index, value in enumerate(actual_group_by_local) if value == component_group
        }
        if actual_members != component:
            patient_flags.append('duplicate_component_merged')

    for duplicate_group, duplicate_rows in group.groupby('duplicate_group', sort=False):
        representative_rows = duplicate_rows.loc[
            bool_series(duplicate_rows['is_duplicate_representative'])
        ]
        if len(representative_rows) != 1:
            patient_flags.append('invalid_representative_count')
            continue
        expected = duplicate_rows.sort_values(
            ['sharpness', 'processed_path'], ascending=[False, True], kind='stable'
        ).iloc[0]
        representative = representative_rows.iloc[0]
        if representative['processed_path'] != expected['processed_path']:
            patient_flags.append('wrong_representative')
        if not duplicate_rows['duplicate_group_size'].eq(len(duplicate_rows)).all():
            patient_flags.append('wrong_duplicate_group_size')
        if not duplicate_rows['duplicate_representative_path'].eq(
            representative['processed_path']
        ).all():
            patient_flags.append('wrong_representative_path')

    if selected_count:
        first = group.loc[group['selection_rank'].eq(1)]
        expected_first = group.loc[representatives].sort_values(
            ['sharpness', 'processed_path'], ascending=[False, True], kind='stable'
        ).iloc[0]
        if len(first) != 1 or first.iloc[0]['processed_path'] != expected_first['processed_path']:
            patient_flags.append('wrong_first_selection')

    for index in group.index:
        flags[index].extend(sorted(set(patient_flags)))
    return flags


def main():
    args = parse_args()
    record_dir = os.path.join(RECORD_ROOT, args.mode)
    selection_path = os.path.join(record_dir, 'selection_manifest.csv')
    limited_path = os.path.join(record_dir, 'patient_limited_manifest.csv')
    result_path = os.path.join(record_dir, '自动验收逐图结果.csv')
    summary_path = os.path.join(record_dir, '自动验收汇总.txt')

    source = pd.read_csv(SOURCE_MANIFEST_PATH, encoding='utf-8-sig')
    selection = pd.read_csv(selection_path, encoding='utf-8-sig')
    limited = pd.read_csv(limited_path, encoding='utf-8-sig')
    row_flags = {index: [] for index in selection.index}

    if selection['processed_path'].duplicated().any():
        for index in selection.index[selection['processed_path'].duplicated(keep=False)]:
            row_flags[index].append('duplicate_processed_path')
    if not set(selection['processed_path']).issubset(set(source['processed_path'])):
        missing = ~selection['processed_path'].isin(source['processed_path'])
        for index in selection.index[missing]:
            row_flags[index].append('not_in_source_manifest')
    if args.mode == 'full' and set(selection['processed_path']) != set(source['processed_path']):
        for index in selection.index:
            row_flags[index].append('full_input_coverage_mismatch')

    for position, row in enumerate(selection.itertuples(), start=1):
        if not HEX_PATTERN.fullmatch(str(row.dhash_hex)):
            row_flags[row.Index].append('invalid_dhash_format')
            continue
        path = os.path.join(DATA_DIR, row.processed_path)
        if not os.path.isfile(path):
            row_flags[row.Index].append('missing_processed_file')
            continue
        computed = independent_dhash(read_image(path))
        if computed != row.dhash_hex:
            row_flags[row.Index].append('dhash_mismatch')
        expected_style = f'{row.size_group}|{row.aspect_group}|{row.frame_profile}'
        if row.style_group != expected_style:
            row_flags[row.Index].append('style_group_mismatch')
        if position % 500 == 0 or position == len(selection):
            print(f'验收进度: {position}/{len(selection)}')

    for _, group in selection.groupby('patient_id', sort=True):
        patient_flags = validate_patient(group)
        for index, flags in patient_flags.items():
            row_flags[index].extend(flags)

    selected = bool_series(selection['selected'])
    expected_limited = set(selection.loc[selected, 'processed_path'])
    if set(limited['processed_path']) != expected_limited or len(limited) != len(expected_limited):
        for index in selection.index:
            row_flags[index].append('limited_manifest_mismatch')

    selection['hard_flags'] = [
        '|'.join(sorted(set(row_flags[index]))) for index in selection.index
    ]
    selection.to_csv(result_path, index=False, encoding='utf-8-sig')
    hard_failures = int(selection['hard_flags'].astype(bool).sum())
    conclusion = 'PASS' if hard_failures == 0 else 'FAIL'
    summary = (
        f'自动验收结论: {conclusion}\n'
        f'模式: {args.mode}\n'
        f'输入图片: {len(selection)}\n'
        f'输入患者: {selection.patient_id.nunique()}\n'
        f'入选图片: {int(selected.sum())}\n'
        f'近重复图片: {int((~bool_series(selection.is_duplicate_representative)).sum())}\n'
        f'硬失败图片: {hard_failures}\n'
        f'最大每患者入选数: {int(selection.loc[selected].groupby("patient_id").size().max())}\n'
    )
    with open(summary_path, 'w', encoding='utf-8') as file:
        file.write(summary)
    print('\n' + summary)
    print(f'逐图结果: {result_path}')
    if conclusion != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
