"""独立验收 P5 元数据基线的 OOF 完整性、分组隔离和汇总指标。"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(
    PROJECT_DIR, '数据整理记录', '第二批', '匹配_v1', '元数据基线_v1'
)
EXPECTED_DATASETS = 4
EXPECTED_FEATURE_SETS = 6
EXPECTED_MODELS = 2
EXPECTED_FOLDS = 5


def main():
    folds = pd.read_csv(os.path.join(RESULT_DIR, '折级指标.csv'), encoding='utf-8-sig')
    predictions = pd.read_csv(os.path.join(RESULT_DIR, 'OOF预测.csv'), encoding='utf-8-sig')
    summary = pd.read_csv(
        os.path.join(RESULT_DIR, '元数据基线汇总.csv'), encoding='utf-8-sig'
    )
    patients = pd.read_csv(
        os.path.join(RESULT_DIR, '患者级元数据表.csv'), encoding='utf-8-sig'
    )
    keys = ['dataset', 'feature_set', 'model']
    checks = []

    expected_configs = EXPECTED_DATASETS * EXPECTED_FEATURE_SETS * EXPECTED_MODELS
    checks.append(('summary_configuration_count', len(summary) == expected_configs))
    checks.append(('fold_row_count', len(folds) == expected_configs * EXPECTED_FOLDS))
    checks.append(('oof_patient_unique', not predictions.duplicated(keys + ['patient_id']).any()))
    checks.append((
        'match_group_not_cross_fold',
        not predictions.groupby(keys + ['cv_group'])['fold'].nunique().gt(1).any(),
    ))
    checks.append(('all_probabilities_finite', np.isfinite(predictions['probability']).all()))
    checks.append((
        'all_probabilities_in_range',
        predictions['probability'].between(0.0, 1.0).all(),
    ))

    patient_counts = patients.groupby('dataset')['patient_id'].nunique()
    oof_counts = predictions.groupby(keys)['patient_id'].nunique()
    checks.append((
        'every_configuration_covers_all_patients',
        all(count == patient_counts.loc[dataset] for (dataset, _, _), count in oof_counts.items()),
    ))

    summary_indexed = summary.set_index(keys)
    auc_matches = []
    for key, group in predictions.groupby(keys, sort=True):
        recalculated = roc_auc_score(group['label'], group['probability'])
        reported = summary_indexed.loc[key, 'pooled_auc']
        auc_matches.append(np.isclose(recalculated, reported, atol=1e-12))
    checks.append(('pooled_auc_recalculation', all(auc_matches)))

    result = pd.DataFrame(checks, columns=['check', 'passed'])
    result.to_csv(
        os.path.join(RESULT_DIR, '自动验收检查.csv'), index=False, encoding='utf-8-sig'
    )
    passed = bool(result['passed'].all())
    summary_text = (
        f'自动验收结论: {"PASS" if passed else "FAIL"}\n'
        f'配置数: {len(summary)}\n'
        f'折级记录数: {len(folds)}\n'
        f'OOF预测数: {len(predictions)}\n'
        f'失败项: {int((~result.passed).sum())}\n'
    )
    with open(
        os.path.join(RESULT_DIR, '自动验收汇总.txt'), 'w', encoding='utf-8'
    ) as file:
        file.write(summary_text)
    print(summary_text)
    if not passed:
        print(result.loc[~result.passed].to_string(index=False))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
