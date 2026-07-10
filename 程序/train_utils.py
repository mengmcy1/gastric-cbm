"""训练脚本共用的数据、评估和两阶段训练逻辑。"""

import os
import random
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, '数据', '胃图文带特征标签数据集 3600+ 1933瘤变')
CSV_PATH = os.path.join(PROJECT_DIR, '数据', '胃图文标签表格-添加瘤变标签.csv')
OUTPUT_DIR = os.path.join(PROJECT_DIR, '结果')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class GastricDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = os.path.join(DATA_DIR, row['图片名字'])
        image = self.transform(Image.open(image_path).convert('RGB'))
        return image, int(row['瘤变标签'])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_matched_dataframe(csv_path=CSV_PATH, image_dir=DATA_DIR):
    """读取标签表，保留 0/1 标签且图片存在的样本。"""
    dataframe = pd.read_csv(csv_path, encoding='gbk')
    dataframe = dataframe[dataframe['瘤变标签'].isin([0, 1])].copy()
    dataframe['图片名字'] = dataframe['图片名字'].astype(str)
    matched = dataframe[
        dataframe['图片名字'].map(lambda name: os.path.exists(os.path.join(image_dir, name)))
    ].copy()

    if matched.empty:
        raise RuntimeError('CSV 中没有与图片文件匹配的有效样本。')

    counts = matched['瘤变标签'].value_counts()
    print(f'有效样本数: {len(matched)}')
    print(f'标签分布 — 早癌/瘤变(1): {counts.get(1, 0)}, 非癌(0): {counts.get(0, 0)}')
    return matched


def split_dataframe(dataframe, seed):
    """按标签分层划分训练/验证/测试集 = 70/15/15。"""
    train_val, test = train_test_split(
        dataframe,
        test_size=0.15,
        stratify=dataframe['瘤变标签'],
        random_state=seed,
    )
    train, val = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        stratify=train_val['瘤变标签'],
        random_state=seed,
    )
    print(f'数据划分 — 训练: {len(train)}, 验证: {len(val)}, 测试: {len(test)}')
    return train, val, test


def build_loaders(batch_size, num_workers, seed, debug=False, debug_samples=200):
    dataframe = load_matched_dataframe()
    if debug and len(dataframe) > debug_samples:
        dataframe, _ = train_test_split(
            dataframe,
            train_size=debug_samples,
            stratify=dataframe['瘤变标签'],
            random_state=seed,
        )
        print(f'[DEBUG] 仅使用 {debug_samples} 张样本')

    datasets = [
        GastricDataset(split, transform)
        for split, transform in zip(split_dataframe(dataframe, seed), [TRAIN_TRANSFORM, EVAL_TRANSFORM, EVAL_TRANSFORM])
    ]
    common = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': DEVICE.type == 'cuda',
    }
    return (
        DataLoader(datasets[0], shuffle=True, **common),
        DataLoader(datasets[1], shuffle=False, **common),
        DataLoader(datasets[2], shuffle=False, **common),
    )


def set_trainable(model, modules):
    model.requires_grad_(False)
    for module in modules:
        module.requires_grad_(True)


def freeze_bn_stats(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)) and not any(
            parameter.requires_grad for parameter in module.parameters()
        ):
            module.eval()


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_prob) >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'AUC': roc_auc_score(y_true, y_prob),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Sensitivity': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'CM': (int(tn), int(fp), int(fn), int(tp)),
    }


def format_metrics(metrics):
    return (
        f"AUC={metrics['AUC']:.4f} | Acc={metrics['Accuracy']:.4f} | "
        f"Sens={metrics['Sensitivity']:.4f} | Spec={metrics['Specificity']:.4f} | "
        f"Prec={metrics['Precision']:.4f} | F1={metrics['F1']:.4f} | CM={metrics['CM']}"
    )


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    labels_all = []
    probs_all = []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * len(images)
        labels_all.extend(labels.cpu().numpy())
        probs_all.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

    loss = total_loss / len(loader.dataset)
    return loss, compute_metrics(labels_all, probs_all)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    freeze_bn_stats(model)
    total_loss = 0.0
    start = time.time()

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(images)

    return total_loss / len(loader.dataset), time.time() - start


def run_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    stage_name,
    scheduler=None,
    early_stop_patience=None,
):
    best_auc = float('-inf')
    best_state = deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss, elapsed = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_metrics = evaluate(model, val_loader, criterion)
        lr = optimizer.param_groups[0]['lr']
        lr_text = f'LR={lr:.2e} | ' if scheduler else ''
        print(
            f'{stage_name} Epoch {epoch:02d}/{epochs}, 耗时 {elapsed:.0f}s | '
            f'TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | '
            f'{lr_text}{format_metrics(val_metrics)}'
        )

        if val_metrics['AUC'] > best_auc:
            best_auc = val_metrics['AUC']
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if scheduler:
            scheduler.step()
        if early_stop_patience and no_improve >= early_stop_patience:
            print(f'Early Stop — 连续 {early_stop_patience} 轮验证 AUC 未提升')
            break

    model.load_state_dict(best_state)
    return best_auc


def run_training(
    *,
    model,
    stage1_modules,
    stage2_modules,
    stage1_name,
    stage2_name,
    save_filename,
    model_name,
    label_smoothing,
    batch_size=32,
    random_seed=42,
    stage1_epochs=10,
    stage2_epochs=20,
    stage1_lr=1e-3,
    stage2_lr=1e-4,
    weight_decay=1e-4,
    early_stop_patience=8,
    num_workers=4,
    debug=False,
    debug_samples=200,
):
    seed_everything(random_seed)
    weights_dir = os.path.join(OUTPUT_DIR, '模型权重')
    os.makedirs(weights_dir, exist_ok=True)
    print(f'当前设备: {DEVICE}')

    train_loader, val_loader, test_loader = build_loaders(
        batch_size, num_workers, random_seed, debug, debug_samples,
    )
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    stages = [
        (stage1_name, stage1_modules, stage1_epochs, stage1_lr, None),
        (stage2_name, stage2_modules, stage2_epochs, stage2_lr, early_stop_patience),
    ]
    best_auc = float('-inf')
    for index, (description, modules, epochs, lr, patience) in enumerate(stages, 1):
        print(f'\n{"=" * 60}\n第{index}阶段 — {description} ({epochs} epochs, LR={lr})\n{"=" * 60}')
        set_trainable(model, modules)
        optimizer = optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if index == 2 else None
        best_auc = run_stage(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            epochs,
            f'S{index}',
            scheduler,
            patience,
        )
        print(f'第{index}阶段完成 — 最佳 Val AUC: {best_auc:.4f}')

    test_loss, test_metrics = evaluate(model, test_loader, criterion)
    print(f'\n{"=" * 60}\n测试集最终评估\n{"=" * 60}')
    print(f'Test Loss: {test_loss:.4f}')
    print(format_metrics(test_metrics))

    config = {
        'model': model_name,
        'batch_size': batch_size,
        'stage1_epochs': stage1_epochs,
        'stage2_epochs': stage2_epochs,
        'stage1_lr': stage1_lr,
        'stage2_lr': stage2_lr,
        'weight_decay': weight_decay,
        'early_stop_patience': early_stop_patience,
        'random_seed': random_seed,
    }
    save_path = os.path.join(weights_dir, save_filename)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'best_val_auc': best_auc,
            'test_metrics': test_metrics,
            'config': config,
        },
        save_path,
    )
    print(f'\n模型已保存至: {save_path}')
