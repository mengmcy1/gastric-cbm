"""P5：用非图像元数据预测患者标签，审计残余来源/设备捷径。"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCH_ROOT = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '匹配_v1')
OUTPUT_DIR = os.path.join(MATCH_ROOT, '元数据基线_v1')
RANDOM_SEED = 42
N_SPLITS = 5

DATASETS = {
    'patient_limited_pre_match': 'patient_limited_manifest.csv',
    'relaxed_all_cases_1to1': 'matched_weak_relaxed_all_cases_1to1_seed42.csv',
    'relaxed_all_cases_1to2': 'matched_weak_relaxed_all_cases_1to2_seed42.csv',
    'strict_common_support_1to1': 'matched_weak_strict_common_support_v1_seed42.csv',
}

FEATURE_SETS = {
    'dimensions_only': {
        'numeric': [
            'original_width',
            'original_height',
            'original_aspect_ratio',
            'log_original_area',
        ],
        'categorical': [],
    },
    'style_source_without_year': {
        'numeric': [],
        'categorical': ['center', 'size_group', 'aspect_group', 'frame_profile'],
    },
    'year_only': {
        'numeric': ['patient_year', 'year_missing'],
        'categorical': [],
    },
    'preprocessing_only': {
        'numeric': [
            'selected_image_count',
            'roi_area_ratio',
            'contour_area_ratio',
            'black_pixel_ratio',
            'edge_refined_rate',
            'progress_bar_rate',
        ],
        'categorical': ['crop_method'],
    },
    'source_style_with_year': {
        'numeric': ['patient_year', 'year_missing'],
        'categorical': ['center', 'size_group', 'aspect_group', 'frame_profile'],
    },
    'combined_safe_metadata': {
        'numeric': [
            'original_width',
            'original_height',
            'original_aspect_ratio',
            'log_original_area',
            'patient_year',
            'year_missing',
            'selected_image_count',
            'roi_area_ratio',
            'contour_area_ratio',
            'black_pixel_ratio',
            'edge_refined_rate',
            'progress_bar_rate',
        ],
        'categorical': [
            'center',
            'size_group',
            'aspect_group',
            'frame_profile',
            'crop_method',
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='P5 患者级元数据预测基线 v1')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def stable_mode(series):
    values = series.dropna().astype(str)
    if values.empty:
        return '__MISSING__'
    counts = values.value_counts()
    return sorted(counts.index[counts.eq(counts.max())])[0]


def numeric_median(group, column):
    values = pd.to_numeric(group[column], errors='coerce').dropna()
    return float(values.median()) if not values.empty else np.nan


def patient_table(images):
    rows = []
    for patient_id, group in images.groupby('patient_id', sort=True):
        labels = pd.to_numeric(group['label'], errors='raise').astype(int).unique()
        if len(labels) != 1:
            raise ValueError(f'患者标签不唯一: {patient_id}')
        width = pd.to_numeric(group['original_width'], errors='coerce')
        height = pd.to_numeric(group['original_height'], errors='coerce')
        valid_size = width.gt(0) & height.gt(0)
        aspect = width.loc[valid_size] / height.loc[valid_size]
        area = width.loc[valid_size] * height.loc[valid_size]
        years = pd.to_numeric(group['year'], errors='coerce').dropna()
        match_group = (
            stable_mode(group['match_group_id'])
            if 'match_group_id' in group.columns
            else patient_id
        )
        rows.append({
            'patient_id': patient_id,
            'label': int(labels[0]),
            'cv_group': match_group,
            'original_width': float(width.median()),
            'original_height': float(height.median()),
            'original_aspect_ratio': float(aspect.median()),
            'log_original_area': float(np.log1p(area).median()),
            'patient_year': float(years.median()) if not years.empty else np.nan,
            'year_missing': int(years.empty),
            'selected_image_count': len(group),
            'roi_area_ratio': numeric_median(group, 'roi_area_ratio'),
            'contour_area_ratio': numeric_median(group, 'contour_area_ratio'),
            'black_pixel_ratio': numeric_median(group, 'black_pixel_ratio'),
            'edge_refined_rate': float(group['edge_refined'].astype(bool).mean()),
            'progress_bar_rate': float(group['progress_bar_detected'].astype(bool).mean()),
            'center': stable_mode(group['center']),
            'size_group': stable_mode(group['size_group']),
            'aspect_group': stable_mode(group['aspect_group']),
            'frame_profile': stable_mode(group['frame_profile']),
            'crop_method': stable_mode(group['crop_method']),
        })
    result = pd.DataFrame(rows)
    if result['label'].nunique() != 2:
        raise ValueError('数据集必须同时包含癌和非癌患者')
    return result


def make_preprocessor(spec):
    transformers = []
    if spec['numeric']:
        numeric = Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', StandardScaler()),
        ])
        transformers.append(('numeric', numeric, spec['numeric']))
    if spec['categorical']:
        categorical = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ])
        transformers.append(('categorical', categorical, spec['categorical']))
    return ColumnTransformer(transformers, remainder='drop')


def make_model(model_name, spec):
    if model_name == 'logistic_regression':
        estimator = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=3000,
            random_state=RANDOM_SEED,
        )
    elif model_name == 'random_forest':
        estimator = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight='balanced_subsample',
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    else:
        raise ValueError(f'未知模型: {model_name}')
    return Pipeline([
        ('preprocessor', make_preprocessor(spec)),
        ('classifier', estimator),
    ])


def metrics(y_true, probability):
    prediction = (np.asarray(probability) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        'auc': roc_auc_score(y_true, probability),
        'accuracy': accuracy_score(y_true, prediction),
        'balanced_accuracy': balanced_accuracy_score(y_true, prediction),
        'sensitivity': tp / (tp + fn) if tp + fn else np.nan,
        'specificity': tn / (tn + fp) if tn + fp else np.nan,
    }


def split_iterator(frame, paired):
    if paired:
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED,
        )
        return splitter.split(frame, frame['label'], groups=frame['cv_group'])
    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    return splitter.split(frame, frame['label'])


def evaluate_configuration(dataset, frame, feature_set, spec, model_name, paired):
    fold_rows = []
    prediction_rows = []
    features = frame[spec['numeric'] + spec['categorical']]
    labels = frame['label'].to_numpy(dtype=int)
    for fold, (train_index, test_index) in enumerate(split_iterator(frame, paired), start=1):
        model = make_model(model_name, spec)
        model.fit(features.iloc[train_index], labels[train_index])
        probability = model.predict_proba(features.iloc[test_index])[:, 1]
        fold_metric = metrics(labels[test_index], probability)
        fold_rows.append({
            'dataset': dataset,
            'feature_set': feature_set,
            'model': model_name,
            'fold': fold,
            'train_patients': len(train_index),
            'test_patients': len(test_index),
            **fold_metric,
        })
        for index, prob in zip(test_index, probability):
            prediction_rows.append({
                'dataset': dataset,
                'feature_set': feature_set,
                'model': model_name,
                'fold': fold,
                'patient_id': frame.iloc[index]['patient_id'],
                'cv_group': frame.iloc[index]['cv_group'],
                'label': labels[index],
                'probability': prob,
            })
    return fold_rows, prediction_rows


def summarize(folds, predictions):
    rows = []
    keys = ['dataset', 'feature_set', 'model']
    for key, group in folds.groupby(keys, sort=True):
        selected = predictions
        for column, value in zip(keys, key):
            selected = selected.loc[selected[column].eq(value)]
        pooled = metrics(selected['label'].to_numpy(), selected['probability'].to_numpy())
        row = dict(zip(keys, key))
        row.update({
            'patients': selected['patient_id'].nunique(),
            'case_patients': int(selected.drop_duplicates('patient_id')['label'].eq(1).sum()),
            'control_patients': int(selected.drop_duplicates('patient_id')['label'].eq(0).sum()),
            'pooled_auc': pooled['auc'],
            'pooled_accuracy': pooled['accuracy'],
            'pooled_balanced_accuracy': pooled['balanced_accuracy'],
            'pooled_sensitivity': pooled['sensitivity'],
            'pooled_specificity': pooled['specificity'],
        })
        for metric in ['auc', 'accuracy', 'balanced_accuracy', 'sensitivity', 'specificity']:
            row[f'fold_{metric}_mean'] = group[metric].mean()
            row[f'fold_{metric}_std'] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def risk_level(auc):
    if auc >= 0.80:
        return 'HIGH'
    if auc >= 0.70:
        return 'MODERATE'
    return 'LOW'


def write_conclusion(summary, path):
    lines = [
        'P5 患者级元数据基线（5折交叉验证）',
        '注意：HIGH/MODERATE/LOW 表示元数据捷径预测风险，不表示医学模型性能。',
        '',
    ]
    for dataset in DATASETS:
        lines.append(dataset)
        subset = summary.loc[
            summary['dataset'].eq(dataset)
            & summary['model'].eq('random_forest')
        ]
        for row in subset.itertuples(index=False):
            lines.append(
                f'  {row.feature_set}: AUC={row.pooled_auc:.3f}, '
                f'BalancedAcc={row.pooled_balanced_accuracy:.3f}, '
                f'风险={risk_level(row.pooled_auc)}'
            )
        lines.append('')
    strict_auc = summary.loc[
        summary['dataset'].eq('strict_common_support_1to1')
        & summary['feature_set'].eq('combined_safe_metadata')
        & summary['model'].eq('random_forest'),
        'pooled_auc',
    ].iloc[0]
    relaxed_auc = summary.loc[
        summary['dataset'].eq('relaxed_all_cases_1to1')
        & summary['feature_set'].eq('combined_safe_metadata')
        & summary['model'].eq('random_forest'),
        'pooled_auc',
    ].iloc[0]
    lines.extend([
        f'松弛1:1综合元数据风险: {risk_level(relaxed_auc)} (AUC={relaxed_auc:.3f})',
        f'严格共同支持综合元数据风险: {risk_level(strict_auc)} (AUC={strict_auc:.3f})',
        '最终训练定位应结合本文件、平衡审计和样本覆盖率共同判断。',
    ])
    with open(path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    args = parse_args()
    if os.path.exists(OUTPUT_DIR):
        if not args.overwrite:
            raise FileExistsError(f'输出目录已存在，请使用 --overwrite: {OUTPUT_DIR}')
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_folds = []
    all_predictions = []
    patient_tables = []
    for dataset, filename in DATASETS.items():
        images = pd.read_csv(os.path.join(MATCH_ROOT, filename), encoding='utf-8-sig')
        patients = patient_table(images)
        patients.insert(0, 'dataset', dataset)
        patient_tables.append(patients)
        paired = 'match_group_id' in images.columns
        for feature_set, spec in FEATURE_SETS.items():
            for model_name in ['logistic_regression', 'random_forest']:
                folds, predictions = evaluate_configuration(
                    dataset,
                    patients,
                    feature_set,
                    spec,
                    model_name,
                    paired,
                )
                all_folds.extend(folds)
                all_predictions.extend(predictions)

    folds = pd.DataFrame(all_folds)
    predictions = pd.DataFrame(all_predictions)
    summary = summarize(folds, predictions)
    pd.concat(patient_tables, ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, '患者级元数据表.csv'), index=False, encoding='utf-8-sig'
    )
    folds.to_csv(
        os.path.join(OUTPUT_DIR, '折级指标.csv'), index=False, encoding='utf-8-sig'
    )
    predictions.to_csv(
        os.path.join(OUTPUT_DIR, 'OOF预测.csv'), index=False, encoding='utf-8-sig'
    )
    summary.to_csv(
        os.path.join(OUTPUT_DIR, '元数据基线汇总.csv'), index=False, encoding='utf-8-sig'
    )
    write_conclusion(summary, os.path.join(OUTPUT_DIR, '元数据基线结论.txt'))
    print(f'\n输出目录: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
