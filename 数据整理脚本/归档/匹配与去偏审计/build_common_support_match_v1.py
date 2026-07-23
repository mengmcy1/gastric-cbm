"""构建同中心、同图像风格且每对图像数相同的严格共同支持集。"""

import hashlib
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTION_ROOT = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '筛图_v1_1', 'full')
LIMITED_PATH = os.path.join(SELECTION_ROOT, 'patient_limited_manifest.csv')
MATCH_ROOT = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '匹配_v1')
SIGNATURE_PATH = os.path.join(MATCH_ROOT, 'patient_signatures.csv')
RANDOM_SEED = 42
MISSING_YEAR_DISTANCE = 25.0
MAX_YEAR_DISTANCE = 25.0
UNMATCHED_COST = 10**12
INCOMPATIBLE_COST = 2 * 10**12


def deterministic_jitter(case_id, control_id):
    value = f'{RANDOM_SEED}|strict|{case_id}|{control_id}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(value).digest()[:2], 'big') % 1000


def patient_style_counts(limited):
    return {
        patient_id: group['style_group'].value_counts().to_dict()
        for patient_id, group in limited.groupby('patient_id', sort=True)
    }


def choose_shared_style(case_id, control_id, style_counts):
    case_counts = style_counts[case_id]
    control_counts = style_counts[control_id]
    shared = set(case_counts) & set(control_counts)
    if not shared:
        return None
    return sorted(
        shared,
        key=lambda style: (
            -min(case_counts[style], control_counts[style]),
            -(case_counts[style] + control_counts[style]),
            style,
        ),
    )[0]


def year_distance(case, control):
    if pd.isna(case.patient_year) or pd.isna(control.patient_year):
        return MISSING_YEAR_DISTANCE, True
    return min(abs(float(case.patient_year) - float(control.patient_year)), MAX_YEAR_DISTANCE), False


def match_center(center_signatures, style_counts):
    cases = list(
        center_signatures.loc[center_signatures.label.eq(1)]
        .sort_values('patient_id')
        .itertuples(index=False)
    )
    controls = list(
        center_signatures.loc[center_signatures.label.eq(0)]
        .sort_values('patient_id')
        .itertuples(index=False)
    )
    costs = np.full(
        (len(cases), len(controls) + len(cases)),
        INCOMPATIBLE_COST,
        dtype=np.float64,
    )
    pair_meta = {}
    for case_index, case in enumerate(cases):
        for control_index, control in enumerate(controls):
            shared_style = choose_shared_style(case.patient_id, control.patient_id, style_counts)
            if shared_style is None:
                continue
            distance_year, missing_year = year_distance(case, control)
            cost = int(round(distance_year * 1000)) + deterministic_jitter(
                case.patient_id, control.patient_id
            )
            costs[case_index, control_index] = cost
            pair_meta[(case_index, control_index)] = (
                shared_style,
                distance_year,
                missing_year,
            )
        costs[case_index, len(controls):] = UNMATCHED_COST + case_index

    assigned_rows, assigned_columns = linear_sum_assignment(costs)
    pairs = []
    for case_index, control_index in zip(assigned_rows, assigned_columns):
        if control_index >= len(controls) or (case_index, control_index) not in pair_meta:
            continue
        case = cases[case_index]
        control = controls[control_index]
        shared_style, distance_year, missing_year = pair_meta[(case_index, control_index)]
        image_count = min(
            style_counts[case.patient_id][shared_style],
            style_counts[control.patient_id][shared_style],
        )
        pairs.append({
            'case_patient_id': case.patient_id,
            'control_patient_id': control.patient_id,
            'center': case.center,
            'shared_style_group': shared_style,
            'matched_images_per_role': image_count,
            'case_year': case.patient_year,
            'control_year': control.patient_year,
            'year_distance': distance_year,
            'year_missing_in_pair': missing_year,
        })
    return pairs


def build_pairs(signatures, style_counts):
    rows = []
    for _, center_group in signatures.groupby('center', sort=True):
        rows.extend(match_center(center_group, style_counts))
    pairs = pd.DataFrame(rows).sort_values(
        ['center', 'case_patient_id'], kind='stable'
    ).reset_index(drop=True)
    pairs.insert(0, 'match_group_id', [
        f'strict_match_{number:04d}' for number in range(1, len(pairs) + 1)
    ])
    return pairs


def choose_images(group, count):
    return group.sort_values(
        ['selection_rank', 'sharpness', 'processed_path'],
        ascending=[True, False, True],
        kind='stable',
    ).head(count).copy()


def build_image_manifest(limited, pairs):
    rows = []
    for pair in pairs.itertuples(index=False):
        for role, patient_id, matched_id in [
            ('case', pair.case_patient_id, pair.control_patient_id),
            ('control', pair.control_patient_id, pair.case_patient_id),
        ]:
            candidates = limited.loc[
                limited.patient_id.eq(patient_id)
                & limited.style_group.eq(pair.shared_style_group)
            ]
            selected = choose_images(candidates, int(pair.matched_images_per_role))
            selected['match_group_id'] = pair.match_group_id
            selected['match_role'] = role
            selected['matched_patient_ids'] = matched_id
            selected['strict_style_group'] = pair.shared_style_group
            selected['match_ratio'] = '1:1'
            selected['match_strategy'] = 'strict_common_support'
            rows.append(selected)
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(
        ['match_group_id', 'match_role', 'selection_rank'], kind='stable'
    ).reset_index(drop=True)


def main():
    limited = pd.read_csv(LIMITED_PATH, encoding='utf-8-sig')
    signatures = pd.read_csv(SIGNATURE_PATH, encoding='utf-8-sig')
    style_counts = patient_style_counts(limited)
    pairs = build_pairs(signatures, style_counts)
    manifest = build_image_manifest(limited, pairs)

    pair_path = os.path.join(MATCH_ROOT, 'strict_common_support_pairs_1to1_seed42.csv')
    manifest_path = os.path.join(
        MATCH_ROOT, 'matched_weak_strict_common_support_v1_seed42.csv'
    )
    pairs.to_csv(pair_path, index=False, encoding='utf-8-sig')
    manifest.to_csv(manifest_path, index=False, encoding='utf-8-sig')

    summary = (
        f'严格共同支持匹配对: {len(pairs)}\n'
        f'患者数: {manifest.patient_id.nunique()}\n'
        f'图像数: {len(manifest)}\n'
        f'癌图: {int(manifest.label.eq(1).sum())}\n'
        f'非癌图: {int(manifest.label.eq(0).sum())}\n'
        f'覆盖癌患者比例: {len(pairs) / int(signatures.label.eq(1).sum()):.2%}\n'
    )
    with open(
        os.path.join(MATCH_ROOT, '严格共同支持集汇总.txt'), 'w', encoding='utf-8'
    ) as file:
        file.write(summary)
    print(summary)
    print(f'严格清单: {manifest_path}')


if __name__ == '__main__':
    main()
