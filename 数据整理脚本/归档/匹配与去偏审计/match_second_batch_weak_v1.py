"""P4：在中心内按患者级风格和年份进行级联最小成本匹配。"""

import argparse
import hashlib
import os
import shutil

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTION_ROOT = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '筛图_v1_1', 'full')
SELECTION_MANIFEST_PATH = os.path.join(SELECTION_ROOT, 'selection_manifest.csv')
LIMITED_MANIFEST_PATH = os.path.join(SELECTION_ROOT, 'patient_limited_manifest.csv')
RECORD_DIR = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '匹配_v1')

RANDOM_SEED = 42
YEAR_TOLERANCE = 2.0
MISSING_YEAR_DISTANCE = 25.0
MAX_YEAR_DISTANCE = 25.0

LEVEL2_BASE = 10**9
LEVEL3_BASE = 10**15
SIZE_DISTANCE_WEIGHT = 10**12
ASPECT_MISMATCH_WEIGHT = 10**9
FRAME_MISMATCH_WEIGHT = 10**6
YEAR_DISTANCE_WEIGHT = 10**3

SIZE_ORDER = {
    'small_le512': 0,
    'medium_513_1024': 1,
    'large_1025_1600': 2,
    'xlarge_gt1600': 3,
}


def parse_args():
    parser = argparse.ArgumentParser(description='第二批弱标注患者级匹配 v1')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def stable_mode(series):
    values = series.dropna().astype(str)
    if values.empty:
        return ''
    counts = values.value_counts()
    maximum = counts.max()
    return sorted(counts.index[counts.eq(maximum)])[0]


def build_patient_signatures(limited):
    rows = []
    for patient_id, group in limited.groupby('patient_id', sort=True):
        labels = group['label'].unique()
        centers = group['center'].unique()
        if len(labels) != 1 or len(centers) != 1:
            raise ValueError(f'患者标签或中心不唯一: {patient_id}')
        size = stable_mode(group['size_group'])
        aspect = stable_mode(group['aspect_group'])
        frame = stable_mode(group['frame_profile'])
        years = pd.to_numeric(group['year'], errors='coerce').dropna()
        patient_year = float(years.median()) if not years.empty else np.nan
        rows.append({
            'patient_id': patient_id,
            'label': int(labels[0]),
            'center': centers[0],
            'selected_image_count': len(group),
            'patient_size_group': size,
            'patient_aspect_group': aspect,
            'patient_frame_profile': frame,
            'patient_style_group': f'{size}|{aspect}|{frame}',
            'patient_year': patient_year,
            'year_missing': bool(years.empty),
            'year_observation_count': len(years),
            'review_status': 'pending',
        })
    return pd.DataFrame(rows)


def year_distance(case, control):
    if pd.isna(case.patient_year) or pd.isna(control.patient_year):
        return MISSING_YEAR_DISTANCE, True
    return min(abs(float(case.patient_year) - float(control.patient_year)), MAX_YEAR_DISTANCE), False


def deterministic_jitter(case_id, control_id, ratio, slot):
    value = f'{RANDOM_SEED}|{ratio}|{slot}|{case_id}|{control_id}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(value).digest()[:2], 'big') % 1000


def pair_features(case, control, ratio, slot):
    same_size = case.patient_size_group == control.patient_size_group
    same_aspect = case.patient_aspect_group == control.patient_aspect_group
    same_frame = case.patient_frame_profile == control.patient_frame_profile
    same_style = same_size and same_aspect and same_frame
    size_distance = abs(
        SIZE_ORDER[case.patient_size_group] - SIZE_ORDER[control.patient_size_group]
    )
    aspect_mismatch = int(not same_aspect)
    frame_mismatch = int(not same_frame)
    distance_year, year_missing = year_distance(case, control)
    jitter = deterministic_jitter(case.patient_id, control.patient_id, ratio, slot)

    if same_style and not year_missing and distance_year <= YEAR_TOLERANCE:
        match_level = 1
        cost = int(round(distance_year * YEAR_DISTANCE_WEIGHT)) + jitter
    elif same_style:
        match_level = 2
        cost = LEVEL2_BASE + int(round(distance_year * YEAR_DISTANCE_WEIGHT)) + jitter
    else:
        match_level = 3
        cost = (
            LEVEL3_BASE
            + size_distance * SIZE_DISTANCE_WEIGHT
            + aspect_mismatch * ASPECT_MISMATCH_WEIGHT
            + frame_mismatch * FRAME_MISMATCH_WEIGHT
            + int(round(distance_year * YEAR_DISTANCE_WEIGHT))
            + jitter
        )
    return {
        'match_level': match_level,
        'same_style_group': same_style,
        'size_group_distance': size_distance,
        'aspect_group_mismatch': aspect_mismatch,
        'frame_profile_mismatch': frame_mismatch,
        'year_distance': distance_year,
        'year_missing_in_pair': year_missing,
        'assignment_cost': cost,
    }


def match_center(center_signatures, ratio):
    cases = center_signatures.loc[center_signatures['label'].eq(1)].sort_values('patient_id')
    controls = center_signatures.loc[center_signatures['label'].eq(0)].sort_values('patient_id')
    required = len(cases) * ratio
    if len(controls) < required:
        raise ValueError(
            f'{center_signatures.center.iloc[0]} 非癌患者不足: '
            f'需要 {required}，实际 {len(controls)}'
        )

    case_slots = [
        (case, slot)
        for case in cases.itertuples(index=False)
        for slot in range(1, ratio + 1)
    ]
    control_rows = list(controls.itertuples(index=False))
    costs = np.empty((len(case_slots), len(control_rows)), dtype=np.float64)
    features = {}
    for row_index, (case, slot) in enumerate(case_slots):
        for column_index, control in enumerate(control_rows):
            pair = pair_features(case, control, ratio, slot)
            costs[row_index, column_index] = pair['assignment_cost']
            features[(row_index, column_index)] = pair

    assigned_rows, assigned_columns = linear_sum_assignment(costs)
    pairs = []
    for row_index, column_index in zip(assigned_rows, assigned_columns):
        case, slot = case_slots[row_index]
        control = control_rows[column_index]
        pair = features[(row_index, column_index)]
        pairs.append({
            'case_patient_id': case.patient_id,
            'control_patient_id': control.patient_id,
            'center': case.center,
            'control_slot': slot,
            'case_style_group': case.patient_style_group,
            'control_style_group': control.patient_style_group,
            'case_year': case.patient_year,
            'control_year': control.patient_year,
            **pair,
        })
    return pairs


def match_all_centers(signatures, ratio):
    pairs = []
    for _, center_group in signatures.groupby('center', sort=True):
        pairs.extend(match_center(center_group, ratio))
    pairs = pd.DataFrame(pairs).sort_values(
        ['center', 'case_patient_id', 'control_slot'], kind='stable'
    ).reset_index(drop=True)
    case_order = {
        patient_id: number
        for number, patient_id in enumerate(sorted(pairs['case_patient_id'].unique()), start=1)
    }
    pairs.insert(0, 'match_group_id', [
        f'match_{ratio}to1_{case_order[patient_id]:04d}'
        for patient_id in pairs['case_patient_id']
    ])
    pairs.insert(1, 'pair_id', [
        f'{group_id}_control{slot}'
        for group_id, slot in zip(pairs['match_group_id'], pairs['control_slot'])
    ])
    return pairs


def matched_image_manifest(limited, pairs, ratio):
    rows = []
    for match_group_id, pair_group in pairs.groupby('match_group_id', sort=True):
        case_id = pair_group['case_patient_id'].iloc[0]
        controls = pair_group.sort_values('control_slot')
        control_ids = controls['control_patient_id'].tolist()
        case_images = limited.loc[limited['patient_id'].eq(case_id)].copy()
        case_images['match_group_id'] = match_group_id
        case_images['match_role'] = 'case'
        case_images['matched_patient_ids'] = '|'.join(control_ids)
        case_images['match_levels'] = '|'.join(map(str, controls['match_level'].astype(int)))
        case_images['max_match_level'] = int(controls['match_level'].max())
        case_images['match_ratio'] = f'1:{ratio}'
        rows.append(case_images)

        for pair in controls.itertuples(index=False):
            control_images = limited.loc[limited['patient_id'].eq(pair.control_patient_id)].copy()
            control_images['match_group_id'] = match_group_id
            control_images['match_role'] = 'control'
            control_images['matched_patient_ids'] = case_id
            control_images['match_levels'] = str(int(pair.match_level))
            control_images['max_match_level'] = int(pair.match_level)
            control_images['match_ratio'] = f'1:{ratio}'
            rows.append(control_images)
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(
        ['match_group_id', 'match_role', 'patient_id', 'selection_rank'], kind='stable'
    ).reset_index(drop=True)


def add_match_status(signatures, pairs_1to1):
    matched_ids = set(pairs_1to1['case_patient_id']) | set(pairs_1to1['control_patient_id'])
    result = signatures.copy()
    result['matched_1to1'] = result['patient_id'].isin(matched_ids)
    result['match_role_1to1'] = np.where(
        result['patient_id'].isin(pairs_1to1['case_patient_id']),
        'case',
        np.where(result['patient_id'].isin(pairs_1to1['control_patient_id']), 'control', 'unmatched'),
    )
    return result


def write_summary(signatures, pairs_1to1, pairs_1to2, path):
    lines = [
        f'患者总数: {len(signatures)}',
        f'癌患者: {int(signatures.label.eq(1).sum())}',
        f'非癌患者: {int(signatures.label.eq(0).sum())}',
        f'1:1 匹配组: {pairs_1to1.match_group_id.nunique()}',
        f'1:1 匹配患者: {pairs_1to1.match_group_id.nunique() * 2}',
        f'1:2 匹配组: {pairs_1to2.match_group_id.nunique()}',
        f'1:2 匹配患者: {pairs_1to2.match_group_id.nunique() * 3}',
        '',
        '1:1 match_level:',
    ]
    lines.extend(
        f'  Level {int(level)}: {count}'
        for level, count in pairs_1to1['match_level'].value_counts().sort_index().items()
    )
    lines.append('')
    lines.append('1:2 match_level:')
    lines.extend(
        f'  Level {int(level)}: {count}'
        for level, count in pairs_1to2['match_level'].value_counts().sort_index().items()
    )
    with open(path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    args = parse_args()
    if os.path.exists(RECORD_DIR):
        if not args.overwrite:
            raise FileExistsError(f'输出目录已存在，请使用 --overwrite: {RECORD_DIR}')
        shutil.rmtree(RECORD_DIR)
    os.makedirs(RECORD_DIR, exist_ok=True)

    selection = pd.read_csv(SELECTION_MANIFEST_PATH, encoding='utf-8-sig')
    limited = pd.read_csv(LIMITED_MANIFEST_PATH, encoding='utf-8-sig')
    signatures = build_patient_signatures(limited)
    pairs_1to1 = match_all_centers(signatures, ratio=1)
    pairs_1to2 = match_all_centers(signatures, ratio=2)
    matched_1to1 = matched_image_manifest(limited, pairs_1to1, ratio=1)
    matched_1to2 = matched_image_manifest(limited, pairs_1to2, ratio=2)
    signatures = add_match_status(signatures, pairs_1to1)

    matched_1to1_ids = set(matched_1to1['patient_id'])
    usable_unmatched = limited.loc[~limited['patient_id'].isin(matched_1to1_ids)].copy()
    excluded = selection.loc[~selection['selected'].astype(bool)].copy()

    signatures.to_csv(
        os.path.join(RECORD_DIR, 'patient_signatures.csv'), index=False, encoding='utf-8-sig'
    )
    selection.to_csv(
        os.path.join(RECORD_DIR, 'full_manifest.csv'), index=False, encoding='utf-8-sig'
    )
    limited.to_csv(
        os.path.join(RECORD_DIR, 'patient_limited_manifest.csv'), index=False, encoding='utf-8-sig'
    )
    pairs_1to1.to_csv(
        os.path.join(RECORD_DIR, 'matched_pairs_1to1_seed42.csv'), index=False, encoding='utf-8-sig'
    )
    pairs_1to2.to_csv(
        os.path.join(RECORD_DIR, 'matched_pairs_1to2_seed42.csv'), index=False, encoding='utf-8-sig'
    )
    matched_1to1.to_csv(
        os.path.join(RECORD_DIR, 'matched_weak_relaxed_all_cases_1to1_seed42.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    matched_1to2.to_csv(
        os.path.join(RECORD_DIR, 'matched_weak_relaxed_all_cases_1to2_seed42.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    usable_unmatched.to_csv(
        os.path.join(RECORD_DIR, 'usable_unmatched_manifest.csv'), index=False, encoding='utf-8-sig'
    )
    excluded.to_csv(
        os.path.join(RECORD_DIR, 'excluded_manifest.csv'), index=False, encoding='utf-8-sig'
    )
    write_summary(
        signatures,
        pairs_1to1,
        pairs_1to2,
        os.path.join(RECORD_DIR, '匹配汇总.txt'),
    )
    print(f'\n匹配记录: {RECORD_DIR}')


if __name__ == '__main__':
    main()
