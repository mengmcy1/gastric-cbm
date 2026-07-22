"""P4/P6：为四套冻结清单生成患者/匹配组级固定训练划分。"""

import argparse
import hashlib
import os
import shutil

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCH_ROOT = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '匹配_v1')
OUTPUT_DIR = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '训练划分_v1')
RANDOM_SEED = 42
N_FOLDS = 5

SINGLE_SPLIT_DATASETS = {
    'full': 'full_manifest.csv',
    'relaxed_all_cases_1to1': 'matched_weak_relaxed_all_cases_1to1_seed42.csv',
    'relaxed_all_cases_1to2': 'matched_weak_relaxed_all_cases_1to2_seed42.csv',
}
STRICT_FILENAME = 'matched_weak_strict_common_support_v1_seed42.csv'


def parse_args():
    parser = argparse.ArgumentParser(description='生成第二批固定训练划分 v1')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def stable_mode(series):
    values = series.dropna().astype(str)
    counts = values.value_counts()
    return sorted(counts.index[counts.eq(counts.max())])[0]


def source_domain(center):
    return 'provincial' if center == '武大省人民' else 'external'


def unit_table(frame, grouped):
    unit_column = 'match_group_id' if grouped else 'patient_id'
    rows = []
    for unit_id, group in frame.groupby(unit_column, sort=True):
        labels = sorted(pd.to_numeric(group['label'], errors='raise').astype(int).unique())
        if grouped and labels != [0, 1]:
            raise ValueError(f'匹配组未同时包含两类标签: {unit_id}')
        if not grouped and len(labels) != 1:
            raise ValueError(f'患者标签不唯一: {unit_id}')
        centers = group['center'].dropna().astype(str).unique()
        if len(centers) != 1:
            raise ValueError(f'划分单位中心不唯一: {unit_id}')
        center = centers[0]
        style_column = 'strict_style_group' if 'strict_style_group' in group else 'style_group'
        rows.append({
            'unit_id': unit_id,
            'label': labels[0] if not grouped else -1,
            'center': center,
            'domain': source_domain(center),
            'style_group': stable_mode(group[style_column]),
            'patient_count': group['patient_id'].nunique(),
            'image_count': len(group),
        })
    return pd.DataFrame(rows)


def collapse_rare_strata(raw_strata, fallback, minimum):
    counts = raw_strata.value_counts()
    return raw_strata.where(raw_strata.map(counts).ge(minimum), fallback)


def assign_single_split(units, grouped):
    if grouped:
        raw = units['center']
        strata = collapse_rare_strata(raw, units['domain'], minimum=7)
    else:
        strata = units['label'].astype(str) + '|' + units['domain']
    train_val, test = train_test_split(
        units,
        test_size=0.15,
        stratify=strata,
        random_state=RANDOM_SEED,
    )
    train_val_strata = strata.loc[train_val.index]
    train, val = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        stratify=train_val_strata,
        random_state=RANDOM_SEED,
    )
    assignment = pd.concat([
        train[['unit_id']].assign(split='train'),
        val[['unit_id']].assign(split='val'),
        test[['unit_id']].assign(split='test'),
    ])
    return assignment


def assign_folds(units):
    raw = units['domain'] + '|' + units['style_group']
    strata = collapse_rare_strata(raw, units['domain'], minimum=N_FOLDS)
    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    assignment = []
    for fold, (_, test_index) in enumerate(splitter.split(units, strata)):
        assignment.extend(
            {'unit_id': unit_id, 'fold': fold}
            for unit_id in units.iloc[test_index]['unit_id']
        )
    return pd.DataFrame(assignment)


def attach_assignment(frame, assignment, grouped, column):
    unit_column = 'match_group_id' if grouped else 'patient_id'
    result = frame.merge(
        assignment.rename(columns={'unit_id': unit_column}),
        on=unit_column,
        how='left',
        validate='many_to_one',
    )
    if result[column].isna().any():
        raise ValueError(f'存在未分配的样本: {column}')
    if column == 'fold':
        result[column] = result[column].astype(int)
    return result


def file_sha256(path):
    with open(path, 'rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def write_summary(outputs):
    rows = []
    for dataset, frame, assignment_type in outputs:
        for assignment, group in frame.groupby(assignment_type, sort=True):
            rows.append({
                'dataset': dataset,
                'assignment_type': assignment_type,
                'assignment': assignment,
                'patients': group['patient_id'].nunique(),
                'images': len(group),
                'case_patients': group.loc[group.label.eq(1), 'patient_id'].nunique(),
                'control_patients': group.loc[group.label.eq(0), 'patient_id'].nunique(),
                'case_images': int(group.label.eq(1).sum()),
                'control_images': int(group.label.eq(0).sum()),
                'match_groups': (
                    group['match_group_id'].nunique()
                    if 'match_group_id' in group.columns else 0
                ),
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, '划分汇总.csv'), index=False, encoding='utf-8-sig'
    )


def main():
    args = parse_args()
    if os.path.exists(OUTPUT_DIR):
        if not args.overwrite:
            raise FileExistsError(f'输出目录已存在，请使用 --overwrite: {OUTPUT_DIR}')
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    outputs = []
    frozen_files = []
    for dataset, filename in SINGLE_SPLIT_DATASETS.items():
        frame = pd.read_csv(os.path.join(MATCH_ROOT, filename), encoding='utf-8-sig')
        grouped = 'match_group_id' in frame.columns
        units = unit_table(frame, grouped)
        assignment = assign_single_split(units, grouped)
        frozen = attach_assignment(frame, assignment, grouped, 'split')
        output_path = os.path.join(OUTPUT_DIR, f'{dataset}_split_seed42.csv')
        frozen.to_csv(output_path, index=False, encoding='utf-8-sig')
        outputs.append((dataset, frozen, 'split'))
        frozen_files.append(output_path)

    strict = pd.read_csv(os.path.join(MATCH_ROOT, STRICT_FILENAME), encoding='utf-8-sig')
    strict_units = unit_table(strict, grouped=True)
    fold_assignment = assign_folds(strict_units)
    strict_frozen = attach_assignment(strict, fold_assignment, grouped=True, column='fold')
    strict_path = os.path.join(OUTPUT_DIR, 'strict_common_support_1to1_folds5_seed42.csv')
    strict_frozen.to_csv(strict_path, index=False, encoding='utf-8-sig')
    outputs.append(('strict_common_support_1to1', strict_frozen, 'fold'))
    frozen_files.append(strict_path)

    write_summary(outputs)
    with open(os.path.join(OUTPUT_DIR, '清单SHA256.txt'), 'w', encoding='utf-8') as file:
        for path in frozen_files:
            file.write(f'{file_sha256(path)}  {os.path.basename(path)}\n')
    print(pd.read_csv(os.path.join(OUTPUT_DIR, '划分汇总.csv'), encoding='utf-8-sig').to_string(index=False))
    print(f'\n输出目录: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
