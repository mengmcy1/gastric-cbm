"""去偏重训练共享的数据、增强、训练和评估工具。"""

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DEFAULT_IMAGE_ROOT = os.path.join(BASE_DIR, '数据', '第二批裁剪后_v1_1')
DEFAULT_OUTPUT_ROOT = os.path.join(BASE_DIR, '结果', '去偏重训练_v1')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

AUGMENTATION_CONFIG = {
    'resize_before_augmentation': 256,
    'random_resized_crop_size': 224,
    'random_resized_crop_scale': [0.85, 1.0],
    'random_resized_crop_ratio': [0.90, 1.10],
    'horizontal_flip_probability': 0.5,
    'rotation_degrees': 10,
    'color_jitter_brightness': 0.15,
    'color_jitter_contrast': 0.15,
    'color_jitter_saturation': 0.10,
    'downsample_probability': 0.5,
    'downsample_scale': [0.50, 0.85],
    'jpeg_probability': 0.5,
    'jpeg_quality': [60, 95],
    'gaussian_blur_probability': 0.3,
    'gaussian_blur_sigma': [0.1, 1.2],
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class RandomDownsampleUpsample:
    def __init__(self, scale=(0.50, 0.85)):
        self.scale = scale

    def __call__(self, image):
        width, height = image.size
        factor = random.uniform(*self.scale)
        small = (
            max(32, int(round(width * factor))),
            max(32, int(round(height * factor))),
        )
        image = image.resize(small, Image.Resampling.BILINEAR)
        return image.resize((width, height), Image.Resampling.BILINEAR)


class RandomJPEGCompression:
    def __init__(self, quality=(60, 95)):
        self.quality = quality

    def __call__(self, image):
        quality = random.randint(*self.quality)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert('RGB').copy()


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomApply([
            RandomDownsampleUpsample(tuple(AUGMENTATION_CONFIG['downsample_scale']))
        ], p=AUGMENTATION_CONFIG['downsample_probability']),
        transforms.RandomApply([
            RandomJPEGCompression(tuple(AUGMENTATION_CONFIG['jpeg_quality']))
        ], p=AUGMENTATION_CONFIG['jpeg_probability']),
        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=5,
                sigma=tuple(AUGMENTATION_CONFIG['gaussian_blur_sigma']),
            )
        ], p=AUGMENTATION_CONFIG['gaussian_blur_probability']),
        transforms.RandomResizedCrop(
            224,
            scale=tuple(AUGMENTATION_CONFIG['random_resized_crop_scale']),
            ratio=tuple(AUGMENTATION_CONFIG['random_resized_crop_ratio']),
        ),
        transforms.RandomHorizontalFlip(
            p=AUGMENTATION_CONFIG['horizontal_flip_probability']
        ),
        transforms.RandomRotation(AUGMENTATION_CONFIG['rotation_degrees']),
        transforms.ColorJitter(
            brightness=AUGMENTATION_CONFIG['color_jitter_brightness'],
            contrast=AUGMENTATION_CONFIG['color_jitter_contrast'],
            saturation=AUGMENTATION_CONFIG['color_jitter_saturation'],
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, eval_transform


class GastricDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = normalize_manifest_columns(df).reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = os.path.join(self.img_dir, row['image_relpath'])
        with Image.open(path) as source:
            image = source.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row['label'])


def normalize_manifest_columns(frame):
    result = frame.copy()
    if 'label' not in result and '瘤变标签' in result:
        result['label'] = result['瘤变标签']
    if 'image_relpath' not in result:
        for column in ['processed_path', 'image_path', '图片名字']:
            if column in result:
                result['image_relpath'] = result[column]
                break
    required = {'patient_id', 'label', 'image_relpath'}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f'manifest 缺少字段: {sorted(missing)}')
    result['label'] = pd.to_numeric(result['label'], errors='raise').astype(int)
    if not set(result['label'].unique()).issubset({0, 1}):
        raise ValueError('标签必须为 0/1')
    return result


def file_sha256(path):
    with open(path, 'rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def assign_split_from_fold(frame, fold):
    available = sorted(pd.to_numeric(frame['fold'], errors='raise').astype(int).unique())
    if fold is None:
        raise ValueError(f'manifest 使用 fold 列，必须指定 --fold；可选: {available}')
    if fold not in available:
        raise ValueError(f'--fold {fold} 不存在；可选: {available}')
    if len(available) < 3:
        raise ValueError('fold 数量不足，无法独立划分 train/val/test')
    validation_fold = available[(available.index(fold) + 1) % len(available)]
    result = frame.copy()
    result['split'] = 'train'
    result.loc[result['fold'].eq(validation_fold), 'split'] = 'val'
    result.loc[result['fold'].eq(fold), 'split'] = 'test'
    return result, validation_fold


def validate_frozen_split(frame, image_root):
    if set(frame['split'].unique()) != {'train', 'val', 'test'}:
        raise ValueError('split 必须完整包含 train/val/test')
    if frame.groupby('patient_id')['split'].nunique().gt(1).any():
        raise ValueError('检测到患者跨 train/val/test')
    if 'processed_sha256' in frame and frame.groupby('processed_sha256')['split'].nunique().gt(1).any():
        raise ValueError('检测到相同图片哈希跨 train/val/test')
    if 'match_group_id' in frame and frame.groupby('match_group_id')['split'].nunique().gt(1).any():
        raise ValueError('检测到匹配组跨 train/val/test')
    for split, group in frame.groupby('split'):
        if set(group['label'].unique()) != {0, 1}:
            raise ValueError(f'{split} 未同时包含两类标签')
    missing_paths = [
        path for path in frame['image_relpath'].astype(str)
        if not os.path.isfile(os.path.join(image_root, path))
    ]
    if missing_paths:
        raise FileNotFoundError(f'存在 {len(missing_paths)} 张缺失图像，示例: {missing_paths[0]}')


def debug_subset(frame, image_root, units_per_class, seed):
    selected = []
    for split, split_frame in frame.groupby('split', sort=False):
        if 'match_group_id' in split_frame:
            units = split_frame[['match_group_id']].drop_duplicates()
            units = units.sample(
                n=min(units_per_class, len(units)),
                random_state=seed,
            )
            chosen = split_frame.loc[split_frame['match_group_id'].isin(units['match_group_id'])]
        else:
            patients = split_frame[['patient_id', 'label']].drop_duplicates()
            sampled = pd.concat([
                group.sample(n=min(units_per_class, len(group)), random_state=seed)
                for _, group in patients.groupby('label', sort=True)
            ])
            chosen = split_frame.loc[split_frame['patient_id'].isin(sampled['patient_id'])]
        selected.append(chosen)
        print(f'[DEBUG] {split}: {chosen.patient_id.nunique()}人/{len(chosen)}张')
    result = pd.concat(selected, ignore_index=True)
    validate_frozen_split(result, image_root)
    return result


def load_frozen_manifest(manifest_path, image_root, fold=None, debug=False, debug_units=4, seed=42):
    frame = normalize_manifest_columns(pd.read_csv(manifest_path, encoding='utf-8-sig'))
    validation_fold = None
    if 'split' in frame:
        frame['split'] = frame['split'].astype(str)
    elif 'fold' in frame:
        frame, validation_fold = assign_split_from_fold(frame, fold)
    else:
        raise ValueError('冻结 manifest 必须包含 split 或 fold 列，禁止训练时临时随机划分')
    validate_frozen_split(frame, image_root)
    if debug:
        frame = debug_subset(frame, image_root, debug_units, seed)
    return frame, validation_fold


def build_loaders(frame, image_root, batch_size, num_workers, seed):
    train_transform, eval_transform = build_transforms()
    datasets = {
        split: GastricDataset(
            frame.loc[frame['split'].eq(split)].copy(),
            image_root,
            train_transform if split == 'train' else eval_transform,
        )
        for split in ['train', 'val', 'test']
    }
    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == 'train',
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=generator,
            persistent_workers=num_workers > 0,
        )
        for split, dataset in datasets.items()
    }
    return loaders


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, y_prob) if np.unique(y_true).size == 2 else np.nan
    return {
        'AUC': float(auc),
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Sensitivity': float(tp / (tp + fn)) if tp + fn else 0.0,
        'Specificity': float(tn / (tn + fp)) if tn + fp else 0.0,
        'Precision': float(tp / (tp + fp)) if tp + fp else 0.0,
        'F1': float(f1_score(y_true, y_pred, zero_division=0)),
        'CM': (int(tn), int(fp), int(fn), int(tp)),
    }


def select_screening_threshold(y_true, y_prob, minimum_sensitivity=0.90):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if np.unique(y_true).size != 2:
        raise ValueError('阈值选择数据必须同时包含两类')
    candidates = np.unique(np.concatenate(([0.0], y_prob, [1.0])))
    rows = []
    for threshold in candidates:
        metrics = compute_metrics(y_true, y_prob, float(threshold))
        rows.append({'threshold': float(threshold), **metrics})
    eligible = [
        row for row in rows
        if row['Sensitivity'] + 1e-12 >= minimum_sensitivity
    ]
    if not eligible:
        raise RuntimeError(
            f'无阈值能满足Sensitivity >= {minimum_sensitivity:.3f}'
        )
    best = max(
        eligible,
        key=lambda row: (
            row['Specificity'], row['Sensitivity'], row['threshold']
        ),
    )
    return best['threshold'], pd.DataFrame(rows)


def format_metrics(values):
    return (
        f"AUC={values['AUC']:.4f} | Acc={values['Accuracy']:.4f} | "
        f"Sens={values['Sensitivity']:.4f} | Spec={values['Specificity']:.4f} | "
        f"Prec={values['Precision']:.4f} | F1={values['F1']:.4f} | CM={values['CM']}"
    )


@torch.no_grad()
def predict_loader(model, loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    labels = []
    probabilities = []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(images)
        if criterion is not None:
            total_loss += criterion(logits, target).item() * images.size(0)
        labels.extend(target.cpu().numpy())
        probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    loss = total_loss / len(loader.dataset) if criterion is not None else np.nan
    return loss, np.asarray(labels), np.asarray(probabilities)


def evaluate(model, loader, criterion, device=None):
    if device is None:
        device = next(model.parameters()).device
    loss, labels, probabilities = predict_loader(model, loader, device, criterion)
    return loss, compute_metrics(labels, probabilities)


def prediction_frame(model, loader, device, criterion):
    loss, labels, probabilities = predict_loader(model, loader, device, criterion)
    frame = loader.dataset.df.copy().reset_index(drop=True)
    if len(frame) != len(labels):
        raise RuntimeError('评估 DataLoader 顺序与 manifest 不一致')
    frame['label'] = labels
    frame['cancer_probability'] = probabilities
    frame['prediction'] = (probabilities >= 0.5).astype(int)
    return loss, frame


def patient_prediction_frame(image_predictions):
    rows = []
    for patient_id, group in image_predictions.groupby('patient_id', sort=True):
        labels = group['label'].unique()
        if len(labels) != 1:
            raise ValueError(f'患者标签不唯一: {patient_id}')
        row = {
            'patient_id': patient_id,
            'label': int(labels[0]),
            'cancer_probability': float(group['cancer_probability'].mean()),
            'image_count': len(group),
        }
        for column in [
            'source',
            'center',
            'year',
            'style_group',
            'size_group',
            'aspect_group',
            'frame_profile',
            'branch',
        ]:
            if column in group:
                values = group[column].dropna().astype(str)
                row[column] = sorted(values.mode())[0] if not values.empty else ''
        if 'pip_present_original' in group:
            pip_values = (
                group['pip_present_original']
                .fillna(False)
                .astype(str)
                .str.lower()
                .isin({'true', '1', 'yes'})
            )
            row['pip_present_original'] = bool(pip_values.any())
        rows.append(row)
    result = pd.DataFrame(rows)
    result['prediction'] = (result['cancer_probability'] >= 0.5).astype(int)
    return result


def subgroup_metrics(predictions, level, threshold=0.5):
    rows = []
    fields = [
        'source',
        'center',
        'year',
        'style_group',
        'size_group',
        'aspect_group',
        'frame_profile',
        'pip_present_original',
        'analysis_group',
        'branch',
    ]
    for field in fields:
        if field not in predictions:
            continue
        values = predictions[field].fillna('__MISSING__').astype(str)
        for value, group in predictions.assign(_group=values).groupby('_group', sort=True):
            metric = compute_metrics(
                group['label'], group['cancer_probability'], threshold
            )
            rows.append({
                'level': level,
                'field': field,
                'value': value,
                'samples': len(group),
                'cases': int(group['label'].eq(1).sum()),
                'controls': int(group['label'].eq(0).sum()),
                **metric,
            })
    return pd.DataFrame(rows)


def freeze_bn_stats(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    freeze_bn_stats(model)
    total_loss = 0.0
    start = time.time()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset), time.time() - start


def run_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    stage_name,
    history,
    scheduler=None,
    early_stop_patience=None,
):
    best_auc = -np.inf
    best_state = None
    no_improve = 0
    for epoch in range(1, epochs + 1):
        train_loss, elapsed = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_predictions = prediction_frame(
            model, val_loader, device, criterion
        )
        val_patients = patient_prediction_frame(val_predictions)
        val_image_metrics = compute_metrics(
            val_predictions['label'], val_predictions['cancer_probability']
        )
        val_patient_metrics = compute_metrics(
            val_patients['label'], val_patients['cancer_probability']
        )
        current_lr = optimizer.param_groups[0]['lr']
        history.append({
            'stage': stage_name,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': current_lr,
            'elapsed_seconds': elapsed,
            **{
                f'val_image_{key}': value
                for key, value in val_image_metrics.items()
            },
            **{
                f'val_patient_{key}': value
                for key, value in val_patient_metrics.items()
            },
        })
        print(
            f'{stage_name} Epoch {epoch:02d}/{epochs} | {elapsed:.0f}s | '
            f'TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | '
            f'LR={current_lr:.2e}\n'
            f'  Val image: {format_metrics(val_image_metrics)}\n'
            f'  Val patient: {format_metrics(val_patient_metrics)}'
        )
        if val_patient_metrics['AUC'] > best_auc:
            best_auc = val_patient_metrics['AUC']
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if scheduler is not None:
            scheduler.step()
        if early_stop_patience and no_improve >= early_stop_patience:
            print(
                f'Early Stop: 连续 {early_stop_patience} epoch '
                '验证患者级 AUC 未提升'
            )
            break
    if best_state is None:
        raise RuntimeError(f'{stage_name} 未产生有效模型状态')
    model.load_state_dict(best_state)
    return float(best_auc), best_state


def create_training_parser(model_name):
    parser = argparse.ArgumentParser(description=f'{model_name} 去偏重训练 v1')
    parser.add_argument('--manifest', required=True, help='包含 split 或 fold 的冻结 CSV')
    parser.add_argument('--image-root', default=DEFAULT_IMAGE_ROOT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--run-name', default='')
    parser.add_argument('--fold', type=int, default=None, help='fold 清单的测试折；下一折作验证')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--stage1-epochs', type=int, default=10)
    parser.add_argument('--stage2-epochs', type=int, default=20)
    parser.add_argument('--stage1-lr', type=float, default=1e-3)
    parser.add_argument('--stage2-lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--early-stop-patience', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug-units', type=int, default=4)
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument(
        '--defer-test',
        action='store_true',
        help='只用验证集选模型和阈值，暂不读取内部测试集预测',
    )
    parser.add_argument(
        '--crop-scale-min', type=float, default=0.85,
        help='RandomResizedCrop scale 下界（A0=0.85, A1=0.7, A2=0.5）',
    )
    parser.add_argument(
        '--crop-scale-max', type=float, default=1.0,
        help='RandomResizedCrop scale 上界',
    )
    parser.add_argument('--overwrite', action='store_true')
    return parser


def git_snapshot():
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=BASE_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=BASE_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip())
        return {'git_commit': commit, 'git_dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'git_commit': '', 'git_dirty': None}


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def prepare_run_directory(args, model_slug):
    manifest_stem = os.path.splitext(os.path.basename(args.manifest))[0]
    suffix = f'_fold{args.fold}' if args.fold is not None else ''
    debug_suffix = '_debug' if args.debug else ''
    run_name = args.run_name or f'{model_slug}_{manifest_stem}{suffix}{debug_suffix}'
    run_dir = os.path.join(args.output_root, run_name)
    if os.path.exists(run_dir):
        if not args.overwrite:
            raise FileExistsError(f'运行目录已存在，请更换 --run-name 或使用 --overwrite: {run_dir}')
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def run_training(
    args,
    model_slug,
    model_display_name,
    build_model,
    set_trainable_stage1,
    set_trainable_stage2,
    stage2_description,
    label_smoothing,
):
    if args.debug:
        args.stage1_epochs = min(args.stage1_epochs, 1)
        args.stage2_epochs = min(args.stage2_epochs, 1)
        args.num_workers = 0
    seed_everything(args.seed)
    # 单变量 scale 消融：覆盖 RandomResizedCrop scale（A0=0.85/A1=0.7/A2=0.5）
    if not 0 < args.crop_scale_min <= args.crop_scale_max <= 1.0:
        raise ValueError('crop-scale 必须满足 0 < min <= max <= 1')
    AUGMENTATION_CONFIG['random_resized_crop_scale'] = [
        args.crop_scale_min, args.crop_scale_max
    ]
    run_dir = prepare_run_directory(args, model_slug)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frame, validation_fold = load_frozen_manifest(
        args.manifest,
        args.image_root,
        fold=args.fold,
        debug=args.debug,
        debug_units=args.debug_units,
        seed=args.seed,
    )
    loaders = build_loaders(
        frame,
        args.image_root,
        args.batch_size,
        args.num_workers,
        args.seed,
    )
    print(f'当前设备: {device}')
    print(f'模型: {model_display_name}')
    for split in ['train', 'val', 'test']:
        dataset = loaders[split].dataset.df
        print(
            f'{split}: {dataset.patient_id.nunique()}人/{len(dataset)}张, '
            f'标签={dataset.label.value_counts().sort_index().to_dict()}'
        )

    model = build_model(pretrained=not args.no_pretrained).to(device)
    label_counts = loaders['train'].dataset.df['label'].value_counts().reindex([0, 1])
    if label_counts.isna().any() or label_counts.le(0).any():
        raise ValueError('训练集必须同时包含两类标签')
    class_weights = label_counts.sum() / (2 * label_counts.to_numpy(dtype=float))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
    print(
        f'类别权重: 非癌={class_weights[0].item():.3f}, '
        f'癌/高级别={class_weights[1].item():.3f}'
    )

    history = []
    set_trainable_stage1(model)
    optimizer = optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.stage1_lr,
        weight_decay=args.weight_decay,
    )
    s1_auc, s1_state = run_stage(
        model,
        loaders['train'],
        loaders['val'],
        criterion,
        optimizer,
        device,
        args.stage1_epochs,
        'S1',
        history,
    )

    set_trainable_stage2(model)
    optimizer = optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.stage2_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.stage2_epochs)
    )
    s2_auc, s2_state = run_stage(
        model,
        loaders['train'],
        loaders['val'],
        criterion,
        optimizer,
        device,
        args.stage2_epochs,
        'S2',
        history,
        scheduler=scheduler,
        early_stop_patience=args.early_stop_patience,
    )
    selected_stage = 'S2' if s2_auc > s1_auc else 'S1'
    selected_auc = max(s1_auc, s2_auc)
    model.load_state_dict(s2_state if selected_stage == 'S2' else s1_state)
    print(
        f'阶段选择（验证患者级 AUC）: '
        f'S1={s1_auc:.4f}, S2={s2_auc:.4f}; 最终使用 {selected_stage}'
    )

    val_loss, val_predictions = prediction_frame(
        model, loaders['val'], device, criterion
    )
    val_patients = patient_prediction_frame(val_predictions)
    selected_threshold, threshold_scan = select_screening_threshold(
        val_patients['label'],
        val_patients['cancer_probability'],
        minimum_sensitivity=0.90,
    )
    val_predictions['prediction'] = (
        val_predictions['cancer_probability'] >= selected_threshold
    ).astype(int)
    val_patients['prediction'] = (
        val_patients['cancer_probability'] >= selected_threshold
    ).astype(int)
    metrics_by_level = {
        'val_image': compute_metrics(
            val_predictions.label,
            val_predictions.cancer_probability,
            selected_threshold,
        ),
        'val_patient': compute_metrics(
            val_patients.label,
            val_patients.cancer_probability,
            selected_threshold,
        ),
    }
    print(f'验证患者筛查阈值: {selected_threshold:.6f}')
    print(f'Val image: {format_metrics(metrics_by_level["val_image"])}')
    print(f'Val patient: {format_metrics(metrics_by_level["val_patient"])}')

    evaluated_splits = [('val', val_predictions, val_patients)]
    test_loss = None
    if args.defer_test:
        print('内部测试集评估已延后（--defer-test）')
    else:
        test_loss, test_predictions = prediction_frame(
            model, loaders['test'], device, criterion
        )
        test_patients = patient_prediction_frame(test_predictions)
        test_predictions['prediction'] = (
            test_predictions['cancer_probability'] >= selected_threshold
        ).astype(int)
        test_patients['prediction'] = (
            test_patients['cancer_probability'] >= selected_threshold
        ).astype(int)
        metrics_by_level['test_image'] = compute_metrics(
            test_predictions.label,
            test_predictions.cancer_probability,
            selected_threshold,
        )
        metrics_by_level['test_patient'] = compute_metrics(
            test_patients.label,
            test_patients.cancer_probability,
            selected_threshold,
        )
        evaluated_splits.append(('test', test_predictions, test_patients))
        print(f'Test image: {format_metrics(metrics_by_level["test_image"])}')
        print(f'Test patient: {format_metrics(metrics_by_level["test_patient"])}')

    pd.DataFrame(history).to_csv(
        os.path.join(run_dir, 'training_history.csv'), index=False, encoding='utf-8-sig'
    )
    frame.to_csv(
        os.path.join(run_dir, 'frozen_split_snapshot.csv'), index=False, encoding='utf-8-sig'
    )
    threshold_scan.to_csv(
        os.path.join(run_dir, 'val_patient_threshold_scan.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    for split, image_frame, patient_frame in evaluated_splits:
        image_frame.to_csv(
            os.path.join(run_dir, f'{split}_image_predictions.csv'),
            index=False,
            encoding='utf-8-sig',
        )
        patient_frame.to_csv(
            os.path.join(run_dir, f'{split}_patient_predictions.csv'),
            index=False,
            encoding='utf-8-sig',
        )
        pd.concat([
            subgroup_metrics(image_frame, 'image', selected_threshold),
            subgroup_metrics(patient_frame, 'patient', selected_threshold),
        ], ignore_index=True).to_csv(
            os.path.join(run_dir, f'{split}_subgroup_metrics.csv'),
            index=False,
            encoding='utf-8-sig',
        )

    entry_script = os.path.abspath(sys.argv[0])
    train_utils_script = os.path.abspath(__file__)
    shutil.copy2(entry_script, os.path.join(run_dir, 'source_entry.py'))
    shutil.copy2(train_utils_script, os.path.join(run_dir, 'source_train_utils.py'))
    config = {
        **vars(args),
        'model': model_display_name,
        'model_slug': model_slug,
        'manifest': os.path.abspath(args.manifest),
        'manifest_sha256': file_sha256(args.manifest),
        'entry_script_sha256': file_sha256(entry_script),
        'train_utils_sha256': file_sha256(train_utils_script),
        'image_root': os.path.abspath(args.image_root),
        'validation_fold': validation_fold,
        'label_smoothing': label_smoothing,
        'stage2_description': stage2_description,
        'augmentation': AUGMENTATION_CONFIG,
        'selection_metric': 'val_patient_auc',
        'threshold_selection': {
            'dataset': 'val_patient',
            'rule': 'maximum_specificity_subject_to_sensitivity_at_least_0.90',
            'threshold': selected_threshold,
            'minimum_sensitivity': 0.90,
            'minimum_specificity_screen': 0.50,
            'passes_minimum_specificity': (
                metrics_by_level['val_patient']['Specificity'] >= 0.50
            ),
        },
        'selected_stage': selected_stage,
        'best_val_patient_auc': selected_auc,
        'stage1_best_val_patient_auc': s1_auc,
        'stage2_best_val_patient_auc': s2_auc,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'metrics': metrics_by_level,
        **git_snapshot(),
    }
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as file:
        json.dump(json_ready(config), file, ensure_ascii=False, indent=2)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': json_ready(config),
        'metrics': metrics_by_level,
    }, os.path.join(run_dir, f'{model_slug}_best.pth'))
    print(f'运行结果: {run_dir}')
    return run_dir
