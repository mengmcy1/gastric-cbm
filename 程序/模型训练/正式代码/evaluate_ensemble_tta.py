#!/usr/bin/env python3
"""在冻结验证/测试集上评估患者聚合、双模型集成和水平翻转 TTA。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0, resnet50


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
DEFAULT_RESULT_ROOT = PROJECT_DIR / '结果' / '去偏重训练_v1'
DEFAULT_RESNET_RUN = DEFAULT_RESULT_ROOT / 'expA_full_resnet50_seed42'
DEFAULT_EFFICIENTNET_RUN = DEFAULT_RESULT_ROOT / 'expA_full_efficientnet_b0_seed42'
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_ROOT / 'expA_ensemble_tta_v1'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(
        description='冻结验证/测试集上的患者聚合、模型集成与水平翻转TTA评估'
    )
    parser.add_argument('--resnet-run-dir', type=Path, default=DEFAULT_RESNET_RUN)
    parser.add_argument(
        '--efficientnet-run-dir', type=Path, default=DEFAULT_EFFICIENTNET_RUN
    )
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--target-sensitivity', type=float, default=0.90)
    parser.add_argument('--ensemble-weights', default='0,0.25,0.5,0.75,1')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--include-test', action='store_true',
        help='显式确认读取冻结internal test；缺少该参数时拒绝运行',
    )
    return parser.parse_args()


class EvaluationDataset(Dataset):
    def __init__(self, frame, image_root):
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        path = Path(str(self.frame.iloc[index]['image_relpath']))
        if not path.is_absolute():
            path = self.image_root / path
        with Image.open(path) as source:
            image = self.transform(source.convert('RGB'))
        return image


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_frozen_split(run_dir):
    config = load_json(run_dir / 'config.json')
    frame = pd.read_csv(
        run_dir / 'frozen_split_snapshot.csv',
        encoding='utf-8-sig',
        dtype={'patient_id': str},
    )
    required = {'patient_id', 'label', 'image_relpath', 'split'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'{run_dir}: 冻结划分缺少字段 {sorted(missing)}')
    frame['patient_id'] = frame['patient_id'].astype(str)
    frame['label'] = pd.to_numeric(frame['label'], errors='raise').astype(int)
    return config, frame


def validate_matching_splits(resnet_frame, efficientnet_frame):
    keys = ['patient_id', 'label', 'image_relpath', 'split']
    left = resnet_frame[keys].sort_values(keys).reset_index(drop=True)
    right = efficientnet_frame[keys].sort_values(keys).reset_index(drop=True)
    if not left.equals(right):
        raise ValueError('ResNet50 与 EfficientNet-B0 的冻结划分不一致')
    if left.groupby('patient_id')['split'].nunique().gt(1).any():
        raise ValueError('检测到患者跨 train/val/test')


def build_model(model_name):
    if model_name == 'resnet50':
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == 'efficientnet_b0':
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    else:
        raise ValueError(model_name)
    return model


def load_model(model_name, run_dir, model_slug, device):
    model = build_model(model_name)
    checkpoint = torch.load(
        run_dir / f'{model_slug}_best.pth',
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def predict_original_and_flip(model, loader, device):
    original_probabilities = []
    flipped_probabilities = []
    for images in loader:
        images = images.to(device, non_blocking=True)
        original = torch.softmax(model(images), dim=1)[:, 1]
        flipped = torch.softmax(
            model(torch.flip(images, dims=[3])), dim=1
        )[:, 1]
        original_probabilities.append(original.cpu().numpy())
        flipped_probabilities.append(flipped.cpu().numpy())
    return (
        np.concatenate(original_probabilities),
        np.concatenate(flipped_probabilities),
    )


def metric_values(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    return {
        'auc': float(roc_auc_score(labels, probabilities)),
        'accuracy': float((predictions == labels).mean()),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1': float(
            2 * precision * sensitivity / (precision + sensitivity)
            if precision + sensitivity
            else 0.0
        ),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def choose_threshold(labels, probabilities, target_sensitivity):
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    eligible = []
    for threshold in candidates:
        values = metric_values(labels, probabilities, float(threshold))
        if values['sensitivity'] + 1e-12 >= target_sensitivity:
            eligible.append((values['specificity'], float(threshold), values))
    if not eligible:
        raise RuntimeError('验证集没有满足目标敏感度的阈值')
    return max(eligible, key=lambda item: (item[0], item[1]))[1]


def aggregate_patient_probabilities(image_frame, probability_column, method):
    rows = []
    for patient_id, group in image_frame.groupby('patient_id', sort=True):
        labels = group['label'].unique()
        if len(labels) != 1:
            raise ValueError(f'患者标签不唯一: {patient_id}')
        values = group[probability_column].to_numpy(dtype=float)
        if method == 'mean':
            probability = values.mean()
        elif method == 'max':
            probability = values.max()
        elif method == 'top2_mean':
            probability = np.sort(values)[-min(2, len(values)):].mean()
        else:
            raise ValueError(method)
        rows.append({
            'patient_id': patient_id,
            'label': int(labels[0]),
            'image_count': int(len(group)),
            'cancer_probability': float(probability),
        })
    return pd.DataFrame(rows)


def evaluate_candidate(
    image_frame,
    probability_column,
    aggregation,
    target_sensitivity,
    threshold=None,
):
    patients = aggregate_patient_probabilities(
        image_frame, probability_column, aggregation
    )
    if threshold is None:
        threshold = choose_threshold(
            patients['label'].to_numpy(),
            patients['cancer_probability'].to_numpy(),
            target_sensitivity,
        )
    values = metric_values(
        patients['label'], patients['cancer_probability'], threshold
    )
    return patients, float(threshold), values


def validate_reproduced_probabilities(run_dir, split, reproduced):
    saved = pd.read_csv(
        run_dir / f'{split}_image_predictions.csv',
        encoding='utf-8-sig',
        dtype={'patient_id': str},
    )
    expected = saved['cancer_probability'].to_numpy(dtype=float)
    if len(expected) != len(reproduced):
        raise RuntimeError(f'{run_dir.name}/{split}: 预测数量不一致')
    difference = float(np.max(np.abs(expected - reproduced)))
    if difference > 1e-5:
        raise RuntimeError(
            f'{run_dir.name}/{split}: 原图预测未能复现，最大差值={difference:.8f}'
        )
    return difference


def candidate_rank(row):
    return (
        row['val_specificity'],
        row['val_auc'],
        row['val_accuracy'],
        -abs(row['resnet_weight'] - 0.5),
    )


def main():
    args = parse_args()
    if not args.include_test:
        raise RuntimeError('该入口会读取internal test，必须显式传入--include-test')
    if args.output_dir.exists():
        raise FileExistsError(f'输出目录已存在，请更换路径: {args.output_dir}')
    args.output_dir.mkdir(parents=True)

    weights = sorted({
        float(value.strip())
        for value in args.ensemble_weights.split(',')
        if value.strip()
    })
    if not weights or min(weights) < 0 or max(weights) > 1:
        raise ValueError('--ensemble-weights 必须位于 [0, 1]')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    resnet_config, resnet_frame = load_frozen_split(args.resnet_run_dir)
    efficientnet_config, efficientnet_frame = load_frozen_split(
        args.efficientnet_run_dir
    )
    validate_matching_splits(resnet_frame, efficientnet_frame)
    if Path(resnet_config['image_root']) != Path(efficientnet_config['image_root']):
        raise ValueError('两个模型的 image_root 不一致')

    evaluation_frame = resnet_frame.loc[
        resnet_frame['split'].isin(['val', 'test'])
    ].copy()
    image_root = Path(resnet_config['image_root'])
    split_frames = {}
    reproduction_differences = {}

    model_specs = [
        (
            'resnet',
            'resnet50',
            args.resnet_run_dir,
            resnet_config['model_slug'],
        ),
        (
            'efficientnet',
            'efficientnet_b0',
            args.efficientnet_run_dir,
            efficientnet_config['model_slug'],
        ),
    ]
    for split in ['val', 'test']:
        frame = evaluation_frame.loc[evaluation_frame['split'].eq(split)].copy()
        split_frames[split] = frame.reset_index(drop=True)

    for prefix, model_name, run_dir, model_slug in model_specs:
        print(f'\n加载 {model_name}: {run_dir}')
        model = load_model(model_name, run_dir, model_slug, device)
        for split in ['val', 'test']:
            dataset = EvaluationDataset(split_frames[split], image_root)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == 'cuda',
                persistent_workers=args.num_workers > 0,
            )
            original, flipped = predict_original_and_flip(model, loader, device)
            difference = validate_reproduced_probabilities(
                run_dir, split, original
            )
            reproduction_differences[f'{prefix}_{split}'] = difference
            split_frames[split][f'{prefix}_base_probability'] = original
            split_frames[split][f'{prefix}_flip_probability'] = flipped
            split_frames[split][f'{prefix}_tta_probability'] = (
                original + flipped
            ) / 2
            print(
                f'{split}: {len(original)}张，原预测最大复现差值={difference:.2e}'
            )
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # 第一步：分别比较两个模型的患者聚合方式。
    aggregation_rows = []
    for view in ['base', 'tta']:
        for model_prefix in ['resnet', 'efficientnet']:
            column = f'{model_prefix}_{view}_probability'
            for aggregation in ['mean', 'max', 'top2_mean']:
                _, threshold, values = evaluate_candidate(
                    split_frames['val'],
                    column,
                    aggregation,
                    args.target_sensitivity,
                )
                aggregation_rows.append({
                    'view': view,
                    'model': model_prefix,
                    'aggregation': aggregation,
                    'threshold': threshold,
                    **{f'val_{key}': value for key, value in values.items()},
                })
    aggregation_frame = pd.DataFrame(aggregation_rows)

    # 第二步：仅用原图验证集选择共享聚合方式和集成权重。
    ensemble_rows = []
    for weight in weights:
        column = f'ensemble_base_w{weight:g}'
        for split, frame in split_frames.items():
            frame[column] = (
                weight * frame['resnet_base_probability']
                + (1 - weight) * frame['efficientnet_base_probability']
            )
        for aggregation in ['mean', 'max', 'top2_mean']:
            _, threshold, values = evaluate_candidate(
                split_frames['val'],
                column,
                aggregation,
                args.target_sensitivity,
            )
            ensemble_rows.append({
                'resnet_weight': weight,
                'efficientnet_weight': 1 - weight,
                'aggregation': aggregation,
                'threshold': threshold,
                **{f'val_{key}': value for key, value in values.items()},
            })
    ensemble_frame = pd.DataFrame(ensemble_rows)
    selected_ensemble = max(
        ensemble_rows,
        key=candidate_rank,
    )
    selected_weight = selected_ensemble['resnet_weight']
    selected_aggregation = selected_ensemble['aggregation']

    # 第三步：锁定权重和聚合，仅在验证集决定是否启用水平翻转 TTA。
    view_rows = []
    patient_predictions = {}
    for view in ['base', 'tta']:
        column = f'selected_{view}_probability'
        for split, frame in split_frames.items():
            frame[column] = (
                selected_weight * frame[f'resnet_{view}_probability']
                + (1 - selected_weight) * frame[f'efficientnet_{view}_probability']
            )
        val_patients, threshold, val_values = evaluate_candidate(
            split_frames['val'],
            column,
            selected_aggregation,
            args.target_sensitivity,
        )
        test_patients, _, test_values = evaluate_candidate(
            split_frames['test'],
            column,
            selected_aggregation,
            args.target_sensitivity,
            threshold=threshold,
        )
        patient_predictions[(view, 'val')] = val_patients
        patient_predictions[(view, 'test')] = test_patients
        view_rows.append({
            'view': view,
            'threshold': threshold,
            **{f'val_{key}': value for key, value in val_values.items()},
            **{f'test_{key}': value for key, value in test_values.items()},
        })
    view_frame = pd.DataFrame(view_rows)
    selected_view = max(
        view_rows,
        key=lambda row: (
            row['val_specificity'],
            row['val_auc'],
            row['val_accuracy'],
            row['view'] == 'base',
        ),
    )

    for split, frame in split_frames.items():
        frame.to_csv(
            args.output_dir / f'{split}_image_predictions.csv',
            index=False,
            encoding='utf-8-sig',
        )
    for (view, split), frame in patient_predictions.items():
        frame.to_csv(
            args.output_dir / f'{split}_patient_predictions_{view}.csv',
            index=False,
            encoding='utf-8-sig',
        )
    aggregation_frame.to_csv(
        args.output_dir / 'aggregation_validation_results.csv',
        index=False,
        encoding='utf-8-sig',
    )
    ensemble_frame.to_csv(
        args.output_dir / 'ensemble_validation_results.csv',
        index=False,
        encoding='utf-8-sig',
    )
    view_frame.to_csv(
        args.output_dir / 'tta_locked_test_results.csv',
        index=False,
        encoding='utf-8-sig',
    )

    summary = {
        'protocol': (
            '验证集先选择聚合方式与集成权重，再在锁定组合下决定是否启用水平翻转TTA；'
            '各候选阈值均满足验证Sensitivity目标并最大化Specificity；测试集不参与选择。'
        ),
        'target_validation_sensitivity': args.target_sensitivity,
        'device': str(device),
        'resnet_run_dir': str(args.resnet_run_dir.resolve()),
        'efficientnet_run_dir': str(args.efficientnet_run_dir.resolve()),
        'image_root': str(image_root.resolve()),
        'validation_images': int(len(split_frames['val'])),
        'validation_patients': int(split_frames['val']['patient_id'].nunique()),
        'test_images': int(len(split_frames['test'])),
        'test_patients': int(split_frames['test']['patient_id'].nunique()),
        'reproduction_max_absolute_difference': reproduction_differences,
        'selected_ensemble_without_tta': selected_ensemble,
        'selected_aggregation': selected_aggregation,
        'selected_resnet_weight': selected_weight,
        'selected_efficientnet_weight': 1 - selected_weight,
        'selected_view': selected_view['view'],
        'selected_view_results': selected_view,
        'all_view_results': view_rows,
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    print('\n===== 验证集选择结果 =====')
    print(
        f"聚合={selected_aggregation}; ResNet权重={selected_weight:.2f}; "
        f"EfficientNet权重={1-selected_weight:.2f}; "
        f"TTA={'启用' if selected_view['view'] == 'tta' else '不启用'}"
    )
    print(
        f"验证: AUC={selected_view['val_auc']:.4f}, "
        f"Acc={selected_view['val_accuracy']:.4f}, "
        f"Sens={selected_view['val_sensitivity']:.4f}, "
        f"Spec={selected_view['val_specificity']:.4f}"
    )
    print(
        f"测试: AUC={selected_view['test_auc']:.4f}, "
        f"Acc={selected_view['test_accuracy']:.4f}, "
        f"Sens={selected_view['test_sensitivity']:.4f}, "
        f"Spec={selected_view['test_specificity']:.4f}, "
        f"F1={selected_view['test_f1']:.4f}"
    )
    print(f'结果目录: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
