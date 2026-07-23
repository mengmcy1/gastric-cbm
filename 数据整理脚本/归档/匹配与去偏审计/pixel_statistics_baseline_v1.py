"""P5：用低层像素、边界和压缩统计预测患者标签。"""

import argparse
import os
import shutil

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from metadata_baseline_v1 import make_model, metrics, risk_level, summarize


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCH_ROOT = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '匹配_v1')
IMAGE_ROOT = os.path.join(PROJECT_DIR, '数据', '第二批裁剪后_v1_1')
OUTPUT_DIR = os.path.join(MATCH_ROOT, '像素统计基线_v1')
RANDOM_SEED = 42
N_SPLITS = 5
ANALYSIS_SIZE = 224
BORDER_RATIO = 0.10

DATASETS = {
    'patient_limited_pre_match': 'patient_limited_manifest.csv',
    'relaxed_all_cases_1to1': 'matched_weak_relaxed_all_cases_1to1_seed42.csv',
    'relaxed_all_cases_1to2': 'matched_weak_relaxed_all_cases_1to2_seed42.csv',
    'strict_common_support_1to1': 'matched_weak_strict_common_support_v1_seed42.csv',
}

COLOR_FEATURES = [
    'gray_mean', 'gray_std', 'gray_p10', 'gray_p50', 'gray_p90',
    'blue_mean', 'green_mean', 'red_mean',
    'blue_std', 'green_std', 'red_std',
    'hue_mean', 'saturation_mean', 'saturation_std', 'value_mean',
    'lab_a_mean', 'lab_b_mean', 'dark_ratio', 'bright_ratio', 'gray_entropy',
]
BORDER_FEATURES = [
    'border_gray_mean', 'border_gray_std', 'border_dark_ratio',
    'center_gray_mean', 'center_gray_std', 'center_dark_ratio',
    'border_center_mean_difference', 'border_center_std_difference',
]
TEXTURE_FEATURES = [
    'edge_density', 'laplacian_variance', 'jpeg_blockiness',
    'file_size_kb', 'jpeg_bytes_per_pixel',
]
FEATURE_SETS = {
    'color_intensity': COLOR_FEATURES,
    'border_composition': BORDER_FEATURES,
    'texture_compression': TEXTURE_FEATURES,
    'all_low_level_statistics': COLOR_FEATURES + BORDER_FEATURES + TEXTURE_FEATURES,
}


def parse_args():
    parser = argparse.ArgumentParser(description='P5 患者级低层像素统计基线 v1')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'无法解码图像: {path}')
    return image


def entropy(gray):
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probability = histogram[histogram > 0] / histogram.sum()
    return float(-(probability * np.log2(probability)).sum())


def jpeg_blockiness(gray):
    array = gray.astype(np.float32)
    vertical = np.abs(np.diff(array, axis=1))
    horizontal = np.abs(np.diff(array, axis=0))
    vertical_boundary = np.arange(vertical.shape[1]) % 8 == 7
    horizontal_boundary = np.arange(horizontal.shape[0]) % 8 == 7
    boundary = np.concatenate([
        vertical[:, vertical_boundary].ravel(),
        horizontal[horizontal_boundary, :].ravel(),
    ])
    interior = np.concatenate([
        vertical[:, ~vertical_boundary].ravel(),
        horizontal[~horizontal_boundary, :].ravel(),
    ])
    return float(boundary.mean() - interior.mean())


def region_masks(height, width):
    border_y = max(1, int(round(height * BORDER_RATIO)))
    border_x = max(1, int(round(width * BORDER_RATIO)))
    center = np.zeros((height, width), dtype=bool)
    center[border_y:height - border_y, border_x:width - border_x] = True
    return ~center, center


def extract_features(relative_path):
    absolute_path = os.path.join(IMAGE_ROOT, relative_path)
    source_image = read_image(absolute_path)
    source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    source_height, source_width = source_gray.shape
    image = source_image
    image = cv2.resize(image, (ANALYSIS_SIZE, ANALYSIS_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    border_mask, center_mask = region_masks(*gray.shape)
    border = gray[border_mask]
    center = gray[center_mask]
    blue, green, red = cv2.split(image)
    file_size = os.path.getsize(absolute_path)
    values = {
        'processed_path': relative_path,
        'gray_mean': gray.mean(),
        'gray_std': gray.std(),
        'gray_p10': np.percentile(gray, 10),
        'gray_p50': np.percentile(gray, 50),
        'gray_p90': np.percentile(gray, 90),
        'blue_mean': blue.mean(),
        'green_mean': green.mean(),
        'red_mean': red.mean(),
        'blue_std': blue.std(),
        'green_std': green.std(),
        'red_std': red.std(),
        'hue_mean': hsv[..., 0].mean(),
        'saturation_mean': hsv[..., 1].mean(),
        'saturation_std': hsv[..., 1].std(),
        'value_mean': hsv[..., 2].mean(),
        'lab_a_mean': lab[..., 1].mean(),
        'lab_b_mean': lab[..., 2].mean(),
        'dark_ratio': np.mean(gray < 20),
        'bright_ratio': np.mean(gray > 240),
        'gray_entropy': entropy(gray),
        'border_gray_mean': border.mean(),
        'border_gray_std': border.std(),
        'border_dark_ratio': np.mean(border < 20),
        'center_gray_mean': center.mean(),
        'center_gray_std': center.std(),
        'center_dark_ratio': np.mean(center < 20),
        'border_center_mean_difference': border.mean() - center.mean(),
        'border_center_std_difference': border.std() - center.std(),
        'edge_density': np.mean(cv2.Canny(gray, 50, 150) > 0),
        'laplacian_variance': cv2.Laplacian(gray, cv2.CV_64F).var(),
        'jpeg_blockiness': jpeg_blockiness(source_gray),
        'file_size_kb': file_size / 1024.0,
        'jpeg_bytes_per_pixel': file_size / float(source_height * source_width),
    }
    return values


def build_image_feature_table(base_manifest):
    paths = sorted(base_manifest['processed_path'].dropna().astype(str).unique())
    rows = []
    for index, path in enumerate(paths, start=1):
        rows.append(extract_features(path))
        if index % 500 == 0 or index == len(paths):
            print(f'低层统计: {index}/{len(paths)}', flush=True)
    return pd.DataFrame(rows)


def stable_mode(series):
    values = series.dropna().astype(str)
    counts = values.value_counts()
    return sorted(counts.index[counts.eq(counts.max())])[0]


def patient_feature_table(images, image_features):
    merged = images.merge(
        image_features,
        on='processed_path',
        how='left',
        validate='many_to_one',
    )
    if merged[COLOR_FEATURES + BORDER_FEATURES + TEXTURE_FEATURES].isna().any().any():
        raise ValueError('存在未提取到的图像低层统计')
    rows = []
    all_features = COLOR_FEATURES + BORDER_FEATURES + TEXTURE_FEATURES
    for patient_id, group in merged.groupby('patient_id', sort=True):
        labels = pd.to_numeric(group['label'], errors='raise').astype(int).unique()
        if len(labels) != 1:
            raise ValueError(f'患者标签不唯一: {patient_id}')
        row = {
            'patient_id': patient_id,
            'label': int(labels[0]),
            'cv_group': (
                stable_mode(group['match_group_id'])
                if 'match_group_id' in group.columns
                else patient_id
            ),
        }
        row.update({feature: float(group[feature].median()) for feature in all_features})
        rows.append(row)
    return pd.DataFrame(rows)


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


def evaluate(dataset, frame, feature_set, features, model_name, paired):
    fold_rows = []
    prediction_rows = []
    labels = frame['label'].to_numpy(dtype=int)
    spec = {'numeric': features, 'categorical': []}
    for fold, (train_index, test_index) in enumerate(split_iterator(frame, paired), start=1):
        model = make_model(model_name, spec)
        model.fit(frame.iloc[train_index][features], labels[train_index])
        probability = model.predict_proba(frame.iloc[test_index][features])[:, 1]
        fold_rows.append({
            'dataset': dataset,
            'feature_set': feature_set,
            'model': model_name,
            'fold': fold,
            'train_patients': len(train_index),
            'test_patients': len(test_index),
            **metrics(labels[test_index], probability),
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


def write_conclusion(summary, path):
    lines = [
        'P5 患者级低层像素统计基线（5折交叉验证）',
        '注意：该基线不使用空间病灶语义；高AUC提示颜色、边界或压缩捷径风险。',
        '',
    ]
    for dataset in DATASETS:
        lines.append(dataset)
        subset = summary.loc[
            summary['dataset'].eq(dataset) & summary['model'].eq('random_forest')
        ]
        for row in subset.itertuples(index=False):
            lines.append(
                f'  {row.feature_set}: AUC={row.pooled_auc:.3f}, '
                f'BalancedAcc={row.pooled_balanced_accuracy:.3f}, '
                f'风险={risk_level(row.pooled_auc)}'
            )
        lines.append('')
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

    base = pd.read_csv(
        os.path.join(MATCH_ROOT, DATASETS['patient_limited_pre_match']),
        encoding='utf-8-sig',
    )
    image_features = build_image_feature_table(base)
    image_features.to_csv(
        os.path.join(OUTPUT_DIR, '图像级低层统计.csv'), index=False, encoding='utf-8-sig'
    )

    all_folds = []
    all_predictions = []
    patient_tables = []
    for dataset, filename in DATASETS.items():
        images = pd.read_csv(os.path.join(MATCH_ROOT, filename), encoding='utf-8-sig')
        patients = patient_feature_table(images, image_features)
        patients.insert(0, 'dataset', dataset)
        patient_tables.append(patients)
        paired = 'match_group_id' in images.columns
        for feature_set, features in FEATURE_SETS.items():
            for model_name in ['logistic_regression', 'random_forest']:
                folds, predictions = evaluate(
                    dataset, patients, feature_set, features, model_name, paired
                )
                all_folds.extend(folds)
                all_predictions.extend(predictions)

    folds = pd.DataFrame(all_folds)
    predictions = pd.DataFrame(all_predictions)
    summary = summarize(folds, predictions)
    pd.concat(patient_tables, ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, '患者级低层统计.csv'), index=False, encoding='utf-8-sig'
    )
    folds.to_csv(os.path.join(OUTPUT_DIR, '折级指标.csv'), index=False, encoding='utf-8-sig')
    predictions.to_csv(
        os.path.join(OUTPUT_DIR, 'OOF预测.csv'), index=False, encoding='utf-8-sig'
    )
    summary.to_csv(
        os.path.join(OUTPUT_DIR, '像素统计基线汇总.csv'), index=False, encoding='utf-8-sig'
    )
    write_conclusion(summary, os.path.join(OUTPUT_DIR, '像素统计基线结论.txt'))
    print(f'输出目录: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
