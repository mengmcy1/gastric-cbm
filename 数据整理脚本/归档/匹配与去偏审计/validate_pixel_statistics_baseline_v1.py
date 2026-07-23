"""独立验收 P5 低层像素统计基线结果。"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(
    PROJECT_DIR, '数据整理记录', '第二批', '匹配_v1', '像素统计基线_v1'
)
EXPECTED_IMAGES = 4099
EXPECTED_DATASETS = 4
EXPECTED_FEATURE_SETS = 4
EXPECTED_MODELS = 2
EXPECTED_FOLDS = 5


def main():
    images = pd.read_csv(
        os.path.join(RESULT_DIR, '图像级低层统计.csv'), encoding='utf-8-sig'
    )
    patients = pd.read_csv(
        os.path.join(RESULT_DIR, '患者级低层统计.csv'), encoding='utf-8-sig'
    )
    folds = pd.read_csv(os.path.join(RESULT_DIR, '折级指标.csv'), encoding='utf-8-sig')
    predictions = pd.read_csv(os.path.join(RESULT_DIR, 'OOF预测.csv'), encoding='utf-8-sig')
    summary = pd.read_csv(
        os.path.join(RESULT_DIR, '像素统计基线汇总.csv'), encoding='utf-8-sig'
    )
    keys = ['dataset', 'feature_set', 'model']
    expected_configs = EXPECTED_DATASETS * EXPECTED_FEATURE_SETS * EXPECTED_MODELS
    checks = [
        ('image_count', len(images) == EXPECTED_IMAGES),
        ('image_path_unique', images['processed_path'].is_unique),
        ('image_features_finite', np.isfinite(images.select_dtypes('number')).all().all()),
        ('summary_configuration_count', len(summary) == expected_configs),
        ('fold_row_count', len(folds) == expected_configs * EXPECTED_FOLDS),
        ('oof_patient_unique', not predictions.duplicated(keys + ['patient_id']).any()),
        (
            'match_group_not_cross_fold',
            not predictions.groupby(keys + ['cv_group'])['fold'].nunique().gt(1).any(),
        ),
        ('all_probabilities_finite', np.isfinite(predictions['probability']).all()),
        ('all_probabilities_in_range', predictions['probability'].between(0, 1).all()),
    ]

    patient_counts = patients.groupby('dataset')['patient_id'].nunique()
    oof_counts = predictions.groupby(keys)['patient_id'].nunique()
    checks.append((
        'every_configuration_covers_all_patients',
        all(count == patient_counts.loc[dataset] for (dataset, _, _), count in oof_counts.items()),
    ))
    indexed = summary.set_index(keys)
    auc_matches = []
    for key, group in predictions.groupby(keys, sort=True):
        auc_matches.append(np.isclose(
            roc_auc_score(group['label'], group['probability']),
            indexed.loc[key, 'pooled_auc'],
            atol=1e-12,
        ))
    checks.append(('pooled_auc_recalculation', all(auc_matches)))

    result = pd.DataFrame(checks, columns=['check', 'passed'])
    result.to_csv(
        os.path.join(RESULT_DIR, '自动验收检查.csv'), index=False, encoding='utf-8-sig'
    )
    passed = bool(result['passed'].all())
    text = (
        f'自动验收结论: {"PASS" if passed else "FAIL"}\n'
        f'图像统计数: {len(images)}\n'
        f'配置数: {len(summary)}\n'
        f'折级记录数: {len(folds)}\n'
        f'OOF预测数: {len(predictions)}\n'
        f'失败项: {int((~result.passed).sum())}\n'
    )
    with open(
        os.path.join(RESULT_DIR, '自动验收汇总.txt'), 'w', encoding='utf-8'
    ) as file:
        file.write(text)
    print(text)
    if not passed:
        print(result.loc[~result.passed].to_string(index=False))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
