"""独立验收 P6 固定划分的患者、哈希和匹配组隔离。"""

import os

import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_ROOT = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '训练划分_v1')
FILES = {
    'full': ('full_split_seed42.csv', 'split'),
    'relaxed_all_cases_1to1': ('relaxed_all_cases_1to1_split_seed42.csv', 'split'),
    'relaxed_all_cases_1to2': ('relaxed_all_cases_1to2_split_seed42.csv', 'split'),
    'strict_common_support_1to1': ('strict_common_support_1to1_folds5_seed42.csv', 'fold'),
}


def main():
    checks = []
    for dataset, (filename, assignment) in FILES.items():
        frame = pd.read_csv(os.path.join(SPLIT_ROOT, filename), encoding='utf-8-sig')
        prefix = f'{dataset}:'
        expected = {'train', 'val', 'test'} if assignment == 'split' else set(range(5))
        checks.extend([
            (prefix + 'assignments_complete', set(frame[assignment].unique()) == expected),
            (
                prefix + 'patient_not_cross_assignment',
                not frame.groupby('patient_id')[assignment].nunique().gt(1).any(),
            ),
            (
                prefix + 'sha_not_cross_assignment',
                not frame.groupby('processed_sha256')[assignment].nunique().gt(1).any(),
            ),
            (prefix + 'labels_valid', set(frame['label'].unique()) == {0, 1}),
            (prefix + 'paths_present', frame['processed_path'].notna().all()),
        ])
        if 'match_group_id' in frame.columns:
            checks.append((
                prefix + 'match_group_not_cross_assignment',
                not frame.groupby('match_group_id')[assignment].nunique().gt(1).any(),
            ))
        for value, group in frame.groupby(assignment):
            checks.append((
                f'{prefix}{assignment}_{value}_has_both_labels',
                set(group['label'].unique()) == {0, 1},
            ))

    result = pd.DataFrame(checks, columns=['check', 'passed'])
    result.to_csv(
        os.path.join(SPLIT_ROOT, '自动验收检查.csv'), index=False, encoding='utf-8-sig'
    )
    passed = bool(result['passed'].all())
    text = (
        f'自动验收结论: {"PASS" if passed else "FAIL"}\n'
        f'检查项: {len(result)}\n'
        f'失败项: {int((~result.passed).sum())}\n'
    )
    with open(os.path.join(SPLIT_ROOT, '自动验收汇总.txt'), 'w', encoding='utf-8') as file:
        file.write(text)
    print(text)
    if not passed:
        print(result.loc[~result.passed].to_string(index=False))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
