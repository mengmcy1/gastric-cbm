"""构建医学生整理概念集的患者限图、松弛1:1和严格共同支持清单。"""

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


PROJECT_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_DIR / '数据整理记录' / '概念提取训练集_v1'
    / '预处理_v1_1' / 'full' / 'processed_manifest.csv'
)
OUTPUT_ROOT = (
    PROJECT_DIR / '数据整理记录' / '概念提取训练集_v1' / '核心清单_v1'
)
MAX_IMAGES_PER_PATIENT = 3
RANDOM_SEED = 42
DHASH_SIZE = 8


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'OpenCV无法解码: {path}')
    return image


def dhash64(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(
        gray, (DHASH_SIZE + 1, DHASH_SIZE), interpolation=cv2.INTER_AREA
    )
    differences = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in differences.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left, right):
    return (left ^ right).bit_count()


def frame_profile(row):
    if bool(row.device_viewport_applied):
        return 'device_viewport'
    if bool(row.progress_bar_detected):
        return 'progress_bar_ui'
    width = max(1, int(row.original_width))
    trim_ratio = max(float(row.edge_trim_left), float(row.edge_trim_right)) / width
    if trim_ratio > 0.03:
        return 'side_ui'
    if row.crop_status == 'cropped':
        return 'contour_frame'
    return 'full_frame'


def add_style_and_hashes(manifest):
    result = manifest.copy()
    result['frame_profile'] = [frame_profile(row) for row in result.itertuples()]
    result['style_group'] = (
        result['source'].astype(str) + '|'
        + result['size_group'].astype(str) + '|'
        + result['aspect_group'].astype(str) + '|'
        + result['frame_profile'].astype(str)
    )
    hashes = []
    for index, path in enumerate(result['processed_path'], start=1):
        hashes.append(dhash64(read_image(path)))
        if index % 500 == 0 or index == len(result):
            print(f'dHash进度: {index}/{len(result)}')
    result['dhash_int'] = hashes
    result['dhash_hex'] = [f'{value:016x}' for value in hashes]
    return result


def canonical_exact_duplicates(manifest):
    result = manifest.copy()
    result['exact_duplicate_keep'] = True
    result['exact_duplicate_reason'] = ''
    for _, group in result.groupby('original_sha256', sort=True):
        if len(group) == 1:
            continue

        def candidate_key(row):
            filename_matches_patient = str(row.image_name).startswith(str(row.patient_id))
            return (
                -int(filename_matches_patient),
                -float(row.sharpness),
                str(row.original_relpath),
            )

        ordered = sorted(group.itertuples(), key=candidate_key)
        keep_index = ordered[0].Index
        drop_indexes = [row.Index for row in ordered[1:]]
        result.loc[drop_indexes, 'exact_duplicate_keep'] = False
        reason = (
            'cross_patient_exact_duplicate'
            if group['patient_key'].nunique() > 1
            else 'same_patient_exact_duplicate'
        )
        result.loc[drop_indexes, 'exact_duplicate_reason'] = reason
        result.loc[keep_index, 'exact_duplicate_reason'] = 'canonical_exact_duplicate'
    return result


def choose_diverse_images(group):
    candidates = group[group['exact_duplicate_keep']].copy()
    if candidates.empty:
        return []
    first = candidates.sort_values(
        ['sharpness', 'processed_path'], ascending=[False, True], kind='stable'
    ).index[0]
    selected = [first]
    while len(selected) < min(MAX_IMAGES_PER_PATIENT, len(candidates)):
        remaining = [index for index in candidates.index if index not in selected]

        def key(index):
            minimum_distance = min(
                hamming_distance(
                    int(candidates.at[index, 'dhash_int']),
                    int(candidates.at[chosen, 'dhash_int']),
                )
                for chosen in selected
            )
            return (
                -minimum_distance,
                -float(candidates.at[index, 'sharpness']),
                str(candidates.at[index, 'processed_path']),
            )

        selected.append(sorted(remaining, key=key)[0])
    return selected


def build_patient_limited(manifest):
    result = canonical_exact_duplicates(manifest)
    result['eligible_unambiguous_patient'] = ~result['cross_label_patient'].astype(bool)
    result['selected'] = False
    result['selection_rank'] = pd.Series(pd.NA, index=result.index, dtype='Int64')
    result['selection_reason'] = ''
    result['exclusion_reason'] = result['exact_duplicate_reason'].where(
        ~result['exact_duplicate_keep'], ''
    )
    result.loc[
        ~result['eligible_unambiguous_patient'], 'exclusion_reason'
    ] = 'cross_label_patient_pending_review'

    eligible = result[
        result['eligible_unambiguous_patient'] & result['exact_duplicate_keep']
    ]
    for _, group in eligible.groupby('patient_key', sort=True):
        selected = choose_diverse_images(group)
        for rank, index in enumerate(selected, start=1):
            result.at[index, 'selected'] = True
            result.at[index, 'selection_rank'] = rank
            result.at[index, 'selection_reason'] = (
                'highest_sharpness' if rank == 1 else 'max_dhash_diversity'
            )
        unselected = [index for index in group.index if index not in selected]
        result.loc[unselected, 'exclusion_reason'] = 'patient_image_limit'
    return result


def patient_table(limited):
    rows = []
    for patient_key, group in limited.groupby('patient_key', sort=True):
        style_counts = Counter(group['style_group'])
        dominant_style = sorted(
            style_counts, key=lambda value: (-style_counts[value], value)
        )[0]
        representative = group[group['style_group'] == dominant_style].sort_values(
            ['selection_rank', 'processed_path'], kind='stable'
        ).iloc[0]
        rows.append({
            'patient_key': patient_key,
            'patient_id': representative['patient_id'],
            'label': int(representative['label']),
            'class_name': representative['class_name'],
            'source': representative['source'],
            'image_count': len(group),
            'dominant_style_group': dominant_style,
            'size_group': representative['size_group'],
            'aspect_group': representative['aspect_group'],
            'frame_profile': representative['frame_profile'],
        })
    return pd.DataFrame(rows)


def style_cost(cancer, control):
    return (
        4.0 * (cancer.size_group != control.size_group)
        + 3.0 * (cancer.aspect_group != control.aspect_group)
        + 2.0 * (cancer.frame_profile != control.frame_profile)
        + 0.25 * abs(cancer.image_count - control.image_count)
    )


def relaxed_pairs(patients):
    pairs = []
    for source, source_rows in patients.groupby('source', sort=True):
        cancers = source_rows[source_rows['label'] == 1].reset_index(drop=True)
        controls = source_rows[source_rows['label'] == 0].reset_index(drop=True)
        if len(controls) < len(cancers):
            raise ValueError(
                f'{source}非癌患者不足: 癌{len(cancers)}，非癌{len(controls)}'
            )
        costs = np.zeros((len(cancers), len(controls)), dtype=float)
        for left, cancer in enumerate(cancers.itertuples()):
            for right, control in enumerate(controls.itertuples()):
                costs[left, right] = style_cost(cancer, control) + right * 1e-9
        cancer_indexes, control_indexes = linear_sum_assignment(costs)
        for number, (left, right) in enumerate(
            zip(cancer_indexes, control_indexes), start=1
        ):
            cancer = cancers.iloc[left]
            control = controls.iloc[right]
            exact_style = (
                cancer['size_group'] == control['size_group']
                and cancer['aspect_group'] == control['aspect_group']
                and cancer['frame_profile'] == control['frame_profile']
            )
            pairs.append({
                'match_id': f'relaxed_{source}_{number:04d}',
                'source': source,
                'cancer_patient_key': cancers.iloc[left]['patient_key'],
                'control_patient_key': controls.iloc[right]['patient_key'],
                'match_cost': float(costs[left, right]),
                'match_level': 'exact_style' if exact_style else 'nearest_style',
            })
    return pd.DataFrame(pairs)


def strict_pairs(patients):
    pairs = []
    grouped = patients.groupby(['source', 'dominant_style_group'], sort=True)
    pair_number = 0
    for (source, style), group in grouped:
        cancers = group[group['label'] == 1].sort_values('patient_key')
        controls = group[group['label'] == 0].sort_values('patient_key')
        count = min(len(cancers), len(controls))
        if count == 0:
            continue
        cancer_sample = cancers.sample(count, random_state=RANDOM_SEED).sort_values(
            'patient_key'
        )
        control_sample = controls.sample(count, random_state=RANDOM_SEED).sort_values(
            'patient_key'
        )
        for cancer, control in zip(
            cancer_sample.itertuples(), control_sample.itertuples()
        ):
            pair_number += 1
            pairs.append({
                'match_id': f'strict_{pair_number:04d}',
                'source': source,
                'dominant_style_group': style,
                'cancer_patient_key': cancer.patient_key,
                'control_patient_key': control.patient_key,
                'match_cost': 0.0,
                'match_level': 'exact_style',
            })
    return pd.DataFrame(pairs)


def expand_pairs(pairs, limited, restrict_to_pair_style=False):
    pair_lookup = {}
    style_lookup = {}
    for pair in pairs.itertuples(index=False):
        pair_lookup[pair.cancer_patient_key] = (pair.match_id, 'cancer')
        pair_lookup[pair.control_patient_key] = (pair.match_id, 'control')
        if restrict_to_pair_style:
            style_lookup[pair.match_id] = pair.dominant_style_group
    selected = limited[limited['patient_key'].isin(pair_lookup)].copy()
    selected['match_id'] = selected['patient_key'].map(
        lambda patient: pair_lookup[patient][0]
    )
    selected['match_role'] = selected['patient_key'].map(
        lambda patient: pair_lookup[patient][1]
    )
    if restrict_to_pair_style:
        selected = selected[
            selected['style_group']
            == selected['match_id'].map(style_lookup)
        ].copy()
    return selected.sort_values(
        ['match_id', 'label', 'selection_rank'], kind='stable'
    )


def pair_image_balance(matched):
    rows = []
    for _, group in matched.groupby('match_id', sort=True):
        count = int(group.groupby('label').size().min())
        for _, label_rows in group.groupby('label', sort=True):
            rows.append(label_rows.nsmallest(count, 'selection_rank'))
    return pd.concat(rows, ignore_index=True) if rows else matched.iloc[0:0].copy()


def summarize(selection, limited, patients, relaxed, strict):
    return {
        'input_images': len(selection),
        'input_patient_keys': int(selection['patient_key'].nunique()),
        'cross_label_patient_keys_excluded': int(
            selection.loc[
                ~selection['eligible_unambiguous_patient'], 'patient_key'
            ].nunique()
        ),
        'exact_duplicate_files_excluded': int(
            (~selection['exact_duplicate_keep']).sum()
        ),
        'patient_limited_images': len(limited),
        'patient_limited_patients': int(limited['patient_key'].nunique()),
        'patient_counts': patients['label'].value_counts().sort_index().to_dict(),
        'relaxed_pairs': len(relaxed),
        'relaxed_match_levels': relaxed['match_level'].value_counts().to_dict(),
        'strict_pairs': len(strict),
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(
        MANIFEST_PATH, encoding='utf-8-sig', dtype={'patient_id': str}
    )
    manifest = add_style_and_hashes(manifest)
    selection = build_patient_limited(manifest)
    limited = selection[selection['selected']].copy()
    patients = patient_table(limited)
    relaxed = relaxed_pairs(patients)
    strict = strict_pairs(patients)

    relaxed_manifest = expand_pairs(relaxed, limited)
    strict_manifest = expand_pairs(strict, limited, restrict_to_pair_style=True)
    relaxed_balanced = pair_image_balance(relaxed_manifest)
    strict_balanced = pair_image_balance(strict_manifest)

    selection.to_csv(
        OUTPUT_ROOT / 'selection_manifest.csv', index=False, encoding='utf-8-sig'
    )
    limited.to_csv(
        OUTPUT_ROOT / 'patient_limited_manifest.csv', index=False, encoding='utf-8-sig'
    )
    patients.to_csv(
        OUTPUT_ROOT / 'patient_summary.csv', index=False, encoding='utf-8-sig'
    )
    relaxed.to_csv(OUTPUT_ROOT / 'relaxed_pairs.csv', index=False, encoding='utf-8-sig')
    strict.to_csv(OUTPUT_ROOT / 'strict_pairs.csv', index=False, encoding='utf-8-sig')
    relaxed_manifest.to_csv(
        OUTPUT_ROOT / 'relaxed_1to1_manifest.csv', index=False, encoding='utf-8-sig'
    )
    strict_manifest.to_csv(
        OUTPUT_ROOT / 'strict_common_support_manifest.csv',
        index=False,
        encoding='utf-8-sig',
    )
    relaxed_balanced.to_csv(
        OUTPUT_ROOT / 'relaxed_1to1_pair_image_balanced_manifest.csv',
        index=False,
        encoding='utf-8-sig',
    )
    strict_balanced.to_csv(
        OUTPUT_ROOT / 'strict_pair_image_balanced_manifest.csv',
        index=False,
        encoding='utf-8-sig',
    )
    summary = summarize(selection, limited, patients, relaxed, strict)
    with open(OUTPUT_ROOT / 'summary.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'核心清单: {OUTPUT_ROOT}')


if __name__ == '__main__':
    main()
