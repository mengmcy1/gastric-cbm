"""审计 P4 级联匹配前后的患者级元数据平衡。"""

import os

import numpy as np
import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD_DIR = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '匹配_v1')
SIZE_ORDER = {
    'small_le512': 0,
    'medium_513_1024': 1,
    'large_1025_1600': 2,
    'xlarge_gt1600': 3,
}
BALANCE_LIMIT = 0.10


def total_variation(frame, column):
    table = pd.crosstab(frame[column], frame['label'], normalize='columns')
    for label in (0, 1):
        if label not in table:
            table[label] = 0.0
    return float(0.5 * (table[1] - table[0]).abs().sum())


def standardized_mean_difference(frame, column):
    case = pd.to_numeric(frame.loc[frame.label.eq(1), column], errors='coerce').dropna()
    control = pd.to_numeric(frame.loc[frame.label.eq(0), column], errors='coerce').dropna()
    denominator = np.sqrt((case.var(ddof=1) + control.var(ddof=1)) / 2)
    if not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return float((case.mean() - control.mean()) / denominator)


def dataset_metrics(name, signatures):
    known_case = signatures.loc[signatures.label.eq(1), 'patient_year'].notna().mean()
    known_control = signatures.loc[signatures.label.eq(0), 'patient_year'].notna().mean()
    return {
        'dataset': name,
        'patients': len(signatures),
        'case_patients': int(signatures.label.eq(1).sum()),
        'control_patients': int(signatures.label.eq(0).sum()),
        'center_tv': total_variation(signatures, 'center'),
        'size_group_tv': total_variation(signatures, 'patient_size_group'),
        'aspect_group_tv': total_variation(signatures, 'patient_aspect_group'),
        'frame_profile_tv': total_variation(signatures, 'patient_frame_profile'),
        'style_group_tv': total_variation(signatures, 'patient_style_group'),
        'size_group_smd': standardized_mean_difference(signatures, 'size_order'),
        'year_smd': standardized_mean_difference(signatures, 'patient_year'),
        'selected_image_count_smd': standardized_mean_difference(
            signatures, 'selected_image_count'
        ),
        'year_known_rate_case': known_case,
        'year_known_rate_control': known_control,
        'year_known_rate_gap': known_case - known_control,
    }


def long_distribution(dataset, frame, column):
    table = (
        frame.groupby(['label', column], dropna=False)
        .size()
        .rename('patient_count')
        .reset_index()
    )
    totals = frame.groupby('label').size().rename('label_total')
    table = table.join(totals, on='label')
    table['proportion'] = table['patient_count'] / table['label_total']
    table.insert(0, 'dataset', dataset)
    table.insert(1, 'field', column)
    table = table.rename(columns={column: 'value'})
    return table[['dataset', 'field', 'label', 'value', 'patient_count', 'proportion']]


def strict_pair_signatures(pairs):
    rows = []
    for pair in pairs.itertuples(index=False):
        size, aspect, frame = pair.shared_style_group.split('|')
        for label, patient_id, year in [
            (1, pair.case_patient_id, pair.case_year),
            (0, pair.control_patient_id, pair.control_year),
        ]:
            rows.append({
                'patient_id': patient_id,
                'label': label,
                'center': pair.center,
                'selected_image_count': pair.matched_images_per_role,
                'patient_size_group': size,
                'patient_aspect_group': aspect,
                'patient_frame_profile': frame,
                'patient_style_group': pair.shared_style_group,
                'patient_year': year,
                'size_order': SIZE_ORDER[size],
            })
    return pd.DataFrame(rows)


def main():
    signatures = pd.read_csv(
        os.path.join(RECORD_DIR, 'patient_signatures.csv'), encoding='utf-8-sig'
    )
    signatures['size_order'] = signatures['patient_size_group'].map(SIZE_ORDER)
    manifest_1to1 = pd.read_csv(
        os.path.join(RECORD_DIR, 'matched_weak_relaxed_all_cases_1to1_seed42.csv'),
        encoding='utf-8-sig',
    )
    manifest_1to2 = pd.read_csv(
        os.path.join(RECORD_DIR, 'matched_weak_relaxed_all_cases_1to2_seed42.csv'),
        encoding='utf-8-sig',
    )
    strict_pairs = pd.read_csv(
        os.path.join(RECORD_DIR, 'strict_common_support_pairs_1to1_seed42.csv'),
        encoding='utf-8-sig',
    )
    datasets = {
        'patient_limited_pre_match': signatures,
        'relaxed_all_cases_1to1': signatures.loc[
            signatures.patient_id.isin(manifest_1to1.patient_id.unique())
        ],
        'relaxed_all_cases_1to2': signatures.loc[
            signatures.patient_id.isin(manifest_1to2.patient_id.unique())
        ],
        'strict_common_support_1to1': strict_pair_signatures(strict_pairs),
    }
    metrics = pd.DataFrame([
        dataset_metrics(name, frame) for name, frame in datasets.items()
    ])
    metrics['balance_pass'] = (
        metrics['style_group_tv'].le(BALANCE_LIMIT)
        & metrics['size_group_smd'].abs().le(BALANCE_LIMIT)
        & metrics['year_smd'].abs().le(BALANCE_LIMIT)
    )
    metrics.to_csv(
        os.path.join(RECORD_DIR, '平衡审计汇总.csv'), index=False, encoding='utf-8-sig'
    )

    distributions = []
    for name, frame in datasets.items():
        for column in [
            'center',
            'patient_size_group',
            'patient_aspect_group',
            'patient_frame_profile',
            'patient_style_group',
        ]:
            distributions.append(long_distribution(name, frame, column))
    pd.concat(distributions, ignore_index=True).to_csv(
        os.path.join(RECORD_DIR, '患者元数据分布_匹配前后.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    level_rows = []
    for ratio, filename in [
        ('1:1', 'matched_pairs_1to1_seed42.csv'),
        ('1:2', 'matched_pairs_1to2_seed42.csv'),
    ]:
        pairs = pd.read_csv(os.path.join(RECORD_DIR, filename), encoding='utf-8-sig')
        table = pairs.groupby(['center', 'match_level']).size().rename('pair_count').reset_index()
        table.insert(0, 'ratio', ratio)
        level_rows.append(table)
    pd.concat(level_rows, ignore_index=True).to_csv(
        os.path.join(RECORD_DIR, '匹配层级按中心.csv'), index=False, encoding='utf-8-sig'
    )

    matched = metrics.loc[metrics.dataset.eq('relaxed_all_cases_1to1')].iloc[0]
    strict = metrics.loc[metrics.dataset.eq('strict_common_support_1to1')].iloc[0]
    conclusion = 'PASS' if bool(matched.balance_pass) else 'FAIL_COMMON_SUPPORT'
    summary = (
        f'平衡审计结论: {conclusion}\n'
        f'1:1 center TV: {matched.center_tv:.4f}\n'
        f'1:1 style_group TV: {matched.style_group_tv:.4f}\n'
        f'1:1 size SMD: {matched.size_group_smd:.4f}\n'
        f'1:1 year SMD: {matched.year_smd:.4f}\n'
        f'1:1 selected_image_count SMD: {matched.selected_image_count_smd:.4f}\n'
        f'strict pairs: {len(strict_pairs)}\n'
        f'strict style_group TV: {strict.style_group_tv:.4f}\n'
        f'strict size SMD: {strict.size_group_smd:.4f}\n'
        f'strict year SMD: {strict.year_smd:.4f}\n'
        f'strict selected_image_count SMD: {strict.selected_image_count_smd:.4f}\n'
        '说明: 级联匹配覆盖全部癌患者并平衡中心，但因关键风格缺少非癌共同支持，'
        '不得将该清单表述为已完成设备风格去偏。\n'
    )
    with open(os.path.join(RECORD_DIR, '平衡审计结论.txt'), 'w', encoding='utf-8') as file:
        file.write(summary)
    print(summary)


if __name__ == '__main__':
    main()
