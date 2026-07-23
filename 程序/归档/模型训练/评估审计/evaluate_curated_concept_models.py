"""在医学生整理概念集上对照旧版与v1.1裁剪重训练模型。"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from efficientnet_train_debiased import build_model as build_efficientnet  # noqa: E402
from inference import load_model as load_legacy_model  # noqa: E402
from resnet_train_debiased import build_model as build_resnet  # noqa: E402
from train_utils import build_transforms  # noqa: E402


CORE_ROOT = (
    PROJECT_DIR / '数据整理记录' / '概念提取训练集_v1' / '核心清单_v1'
)
INPUT_MANIFEST = CORE_ROOT / 'patient_limited_manifest.csv'
OLD_SPLIT = PROJECT_DIR / '结果' / '数据划分' / 'image_split_seed42.csv'
NEW_SPLIT = (
    PROJECT_DIR / '结果' / '去偏重训练_v1'
    / 'expA_full_resnet50_seed42' / 'frozen_split_snapshot.csv'
)
OUTPUT_ROOT = (
    PROJECT_DIR / '结果' / '去偏重训练_v1' / '医学生整理概念集模型审计_v1'
)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SPECS = {
    'legacy_resnet50': {
        'architecture': 'resnet50',
        'input_column': 'original_path',
        'image_threshold': 0.30,
        'patient_threshold': 0.30,
        'weight': PROJECT_DIR / '结果/模型权重/resnet50_transfer_best.pth',
        'exposure_column': 'legacy_exposure',
    },
    'legacy_efficientnet_b0': {
        'architecture': 'efficientnet_b0',
        'input_column': 'original_path',
        'image_threshold': 0.22,
        'patient_threshold': 0.22,
        'weight': PROJECT_DIR / '结果/模型权重/efficientnet_b0_best.pth',
        'exposure_column': 'legacy_exposure',
    },
    'debiased_resnet50': {
        'architecture': 'resnet50',
        'input_column': 'processed_path',
        'image_threshold': 0.15484211,
        'patient_threshold': 0.3849397003650665,
        'weight': (
            PROJECT_DIR / '结果/去偏重训练_v1/expA_full_resnet50_seed42'
            / 'resnet50_debiased_best.pth'
        ),
        'exposure_column': 'debiased_exposure',
    },
    'debiased_efficientnet_b0': {
        'architecture': 'efficientnet_b0',
        'input_column': 'processed_path',
        'image_threshold': 0.2224278,
        'patient_threshold': 0.3894016146659851,
        'weight': (
            PROJECT_DIR / '结果/去偏重训练_v1/expA_full_efficientnet_b0_seed42'
            / 'efficientnet_b0_debiased_best.pth'
        ),
        'exposure_column': 'debiased_exposure',
    },
}


class ImagePathDataset(Dataset):
    def __init__(self, frame, path_column, transform):
        self.frame = frame.reset_index(drop=True)
        self.path_column = path_column
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        with Image.open(self.frame.iloc[index][self.path_column]) as source:
            image = source.convert('RGB')
        return self.transform(image), index


def load_model(name, spec):
    architecture = spec['architecture']
    if name.startswith('legacy_'):
        return load_legacy_model(architecture, spec['weight'])
    builders = {
        'resnet50': build_resnet,
        'efficientnet_b0': build_efficientnet,
    }
    checkpoint = torch.load(
        spec['weight'], map_location=DEVICE, weights_only=False
    )
    model = builders[architecture](pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict(model, frame, path_column, batch_size=64, num_workers=4):
    _, transform = build_transforms()
    dataset = ImagePathDataset(frame, path_column, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DEVICE.type == 'cuda',
        persistent_workers=num_workers > 0,
    )
    probabilities = np.zeros(len(dataset), dtype=np.float32)
    for images, indexes in loader:
        logits = model(images.to(DEVICE, non_blocking=True))
        probabilities[indexes.numpy()] = (
            torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        )
    return probabilities


def add_exposure(manifest):
    legacy = pd.read_csv(OLD_SPLIT, encoding='utf-8-sig')[['sha256', 'split']]
    new = pd.read_csv(NEW_SPLIT, encoding='utf-8-sig')[['sha256', 'split']]
    legacy_map = legacy.drop_duplicates('sha256').set_index('sha256')['split']
    new_map = new.drop_duplicates('sha256').set_index('sha256')['split']
    result = manifest.copy()
    result['legacy_exposure'] = result['original_sha256'].map(legacy_map).fillna('unseen')
    result['debiased_exposure'] = result['original_sha256'].map(new_map).fillna('unseen')
    return result


def add_subset_membership(manifest):
    result = manifest.copy()
    for name, filename in (
        ('relaxed_balanced', 'relaxed_1to1_pair_image_balanced_manifest.csv'),
        ('strict_balanced', 'strict_pair_image_balanced_manifest.csv'),
    ):
        subset = pd.read_csv(CORE_ROOT / filename, encoding='utf-8-sig')
        result[name] = result['image_id'].isin(set(subset['image_id']))
    return result


def metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        'auc': float(roc_auc_score(labels, probabilities)),
        'accuracy': float(accuracy_score(labels, predictions)),
        'sensitivity': float(tp / (tp + fn)),
        'specificity': float(tn / (tn + fp)),
        'precision': float(precision_score(labels, predictions, zero_division=0)),
        'f1': float(f1_score(labels, predictions, zero_division=0)),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def aggregate_patients(images):
    return images.groupby(['patient_key', 'label'], as_index=False).agg(
        cancer_probability=('cancer_probability', 'mean'),
        max_probability=('cancer_probability', 'max'),
        image_count=('image_id', 'size'),
        source=('source', 'first'),
        legacy_exposure=('legacy_exposure', lambda values: '|'.join(sorted(set(values)))),
        debiased_exposure=('debiased_exposure', lambda values: '|'.join(sorted(set(values)))),
        relaxed_balanced=('relaxed_balanced', 'max'),
        strict_balanced=('strict_balanced', 'max'),
    )


def evaluate_subsets(model_name, spec, images):
    rows = []
    scenarios = [('all', images)]
    for source in sorted(images['source'].unique()):
        scenarios.append((f'source:{source}', images[images['source'] == source]))
    for subset in ('relaxed_balanced', 'strict_balanced'):
        scenarios.append((subset, images[images[subset]]))
    exposure_column = spec['exposure_column']
    patient_exposure = images.groupby('patient_key')[exposure_column].agg(
        lambda values: values.iloc[0] if values.nunique() == 1 else 'mixed'
    )
    for exposure in ('train', 'val', 'test', 'unseen', 'mixed'):
        patient_keys = patient_exposure[patient_exposure == exposure].index
        image_subset = images[images['patient_key'].isin(patient_keys)]
        scenarios.append((f'{exposure_column}:{exposure}', image_subset))

    for scenario, image_frame in scenarios:
        patient_frame = aggregate_patients(image_frame) if not image_frame.empty else pd.DataFrame()
        for level, frame, threshold in (
            ('image', image_frame, spec['image_threshold']),
            ('patient_mean', patient_frame, spec['patient_threshold']),
        ):
            if frame.empty or frame['label'].nunique() < 2:
                continue
            rows.append({
                'model': model_name,
                'scenario': scenario,
                'level': level,
                'threshold': threshold,
                'samples': len(frame),
                'patients': (
                    frame['patient_key'].nunique()
                    if 'patient_key' in frame else len(frame)
                ),
                **metrics(frame['label'], frame['cancer_probability'], threshold),
            })
    return rows


def paired_bootstrap(
    predictions, baseline, candidate, level, scenario='all', iterations=2000
):
    baseline_frame = predictions[baseline]
    candidate_frame = predictions[candidate]
    if scenario == 'strict_balanced':
        baseline_frame = baseline_frame[baseline_frame['strict_balanced']]
        candidate_frame = candidate_frame[candidate_frame['strict_balanced']]
    elif scenario == 'common_unseen':
        pure_unseen = baseline_frame.groupby('patient_key').apply(
            lambda group: bool(
                group['legacy_exposure'].eq('unseen').all()
                and group['debiased_exposure'].eq('unseen').all()
            ),
            include_groups=False,
        )
        patient_keys = pure_unseen[pure_unseen].index
        baseline_frame = baseline_frame[
            baseline_frame['patient_key'].isin(patient_keys)
        ]
        candidate_frame = candidate_frame[
            candidate_frame['patient_key'].isin(patient_keys)
        ]
    if level == 'image':
        key = 'image_id'
        left = baseline_frame[[key, 'label', 'cancer_probability']]
        right = candidate_frame[[key, 'cancer_probability']]
    else:
        key = 'patient_key'
        left = aggregate_patients(baseline_frame)[
            [key, 'label', 'cancer_probability']
        ]
        right = aggregate_patients(candidate_frame)[
            [key, 'cancer_probability']
        ]
    paired = left.merge(right, on=key, suffixes=('_baseline', '_candidate'))
    labels = paired['label'].to_numpy(dtype=int)
    baseline_probability = paired['cancer_probability_baseline'].to_numpy()
    candidate_probability = paired['cancer_probability_candidate'].to_numpy()
    observed = (
        roc_auc_score(labels, candidate_probability)
        - roc_auc_score(labels, baseline_probability)
    )
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(iterations):
        indexes = np.concatenate((
            rng.choice(negative, len(negative), replace=True),
            rng.choice(positive, len(positive), replace=True),
        ))
        differences.append(
            roc_auc_score(labels[indexes], candidate_probability[indexes])
            - roc_auc_score(labels[indexes], baseline_probability[indexes])
        )
    low, high = np.percentile(differences, [2.5, 97.5])
    values = np.asarray(differences)
    p_value = min(1.0, 2 * min((values <= 0).mean(), (values >= 0).mean()))
    return {
        'architecture': baseline.replace('legacy_', ''),
        'level': level,
        'baseline': baseline,
        'candidate': candidate,
        'scenario': scenario,
        'samples': len(paired),
        'auc_difference_candidate_minus_baseline': float(observed),
        'ci95_low': float(low),
        'ci95_high': float(high),
        'p_value': float(p_value),
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(
        INPUT_MANIFEST, encoding='utf-8-sig', dtype={'patient_id': str}
    )
    manifest = add_subset_membership(add_exposure(manifest))
    predictions = {}
    metric_rows = []
    for model_name, spec in MODEL_SPECS.items():
        print(f'评估 {model_name} ({spec["input_column"]})')
        model = load_model(model_name, spec)
        result = manifest.copy()
        result['cancer_probability'] = predict(
            model, result, spec['input_column']
        )
        result['prediction'] = (
            result['cancer_probability'] >= spec['image_threshold']
        ).astype(int)
        patients = aggregate_patients(result)
        predictions[model_name] = result
        metric_rows.extend(evaluate_subsets(model_name, spec, result))
        model_dir = OUTPUT_ROOT / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(model_dir / 'image_predictions.csv', index=False, encoding='utf-8-sig')
        patients.to_csv(model_dir / 'patient_predictions.csv', index=False, encoding='utf-8-sig')
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(OUTPUT_ROOT / 'metrics.csv', index=False, encoding='utf-8-sig')
    comparisons = []
    for baseline, candidate in (
        ('legacy_resnet50', 'debiased_resnet50'),
        ('legacy_efficientnet_b0', 'debiased_efficientnet_b0'),
    ):
        for scenario in ('all', 'strict_balanced', 'common_unseen'):
            for level in ('image', 'patient_mean'):
                comparisons.append(
                    paired_bootstrap(
                        predictions, baseline, candidate, level, scenario=scenario
                    )
                )
    with open(OUTPUT_ROOT / 'paired_comparisons.json', 'w', encoding='utf-8') as file:
        json.dump(comparisons, file, ensure_ascii=False, indent=2)

    config = {
        'device': str(DEVICE),
        'input_manifest': str(INPUT_MANIFEST.resolve()),
        'images': len(manifest),
        'patients': int(manifest['patient_key'].nunique()),
        'models': {
            name: {
                key: str(value) if isinstance(value, Path) else value
                for key, value in spec.items()
            }
            for name, spec in MODEL_SPECS.items()
        },
    }
    with open(OUTPUT_ROOT / 'config.json', 'w', encoding='utf-8') as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    print(metrics_frame[
        (metrics_frame['scenario'] == 'all')
        & (metrics_frame['level'].isin(['image', 'patient_mean']))
    ].to_string(index=False))
    print(f'结果: {OUTPUT_ROOT}')


if __name__ == '__main__':
    main()
