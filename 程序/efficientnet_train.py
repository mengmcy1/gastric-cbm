"""
EfficientNet-B0 迁移学习训练脚本（对照实验）
基于 ImageNet 预训练 EfficientNet-B0，在胃早癌白光胃镜图像上微调二分类模型

训练策略（两阶段）：
  第一阶段 — 冻结 backbone，仅训练 FC 分类头，较高学习率
  第二阶段 — 仅微调 features 后段 + FC，低学习率 + CosineAnnealing + Early Stop

参照 agents.md：
  仅使用 CSV 中能与图片匹配的 3314 张有效样本
  目标列：瘤变标签（1=早癌/瘤变，0=非癌）

与 ResNet 对照：
  本脚本作为 EfficientNet-B0 对照实验，与 resnet_train_final.py 保持相同的
  数据划分随机种子、训练策略和评估口径，便于公平对比。
"""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
import time
import random
from copy import deepcopy

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, f1_score,
)


# ============================================================
# 0. 全局配置
# ============================================================

# ---- 调试模式 ----
DEBUG = False
DEBUG_SAMPLES = 200

# 基于脚本自身位置构建绝对路径
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, '数据', '胃图文带特征标签数据集 3600+ 1933瘤变')
CSV_PATH   = os.path.join(BASE_DIR, '数据', '胃图文标签表格-添加瘤变标签.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '程序', '结果')

BATCH_SIZE     = 32
NUM_WORKERS    = 0 if DEBUG else 4
RANDOM_SEED    = 42

STAGE1_EPOCHS  = 2 if DEBUG else 10
STAGE2_EPOCHS  = 2 if DEBUG else 20
STAGE1_LR      = 1e-3
STAGE2_LR      = 1e-4
WEIGHT_DECAY   = 1e-4
EARLY_STOP_PATIENCE = 5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ImageNet 预训练模型的均值与标准差（EfficientNet 与 ResNet 一致）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ============================================================
# 1. 随机种子 & 数据增强
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# EfficientNet-B0 默认输入 224×224（与 ResNet 一致设置，便于公平对比）
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ============================================================
# 2. 数据集（与 ResNet 版本完全一致）
# ============================================================

class GastricDataset(Dataset):
    """胃镜图像二分类数据集，仅保留标签有效且图片存在的样本。"""

    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['图片名字'])
        image = Image.open(img_path).convert('RGB')
        label = int(row['瘤变标签'])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_matched_dataframe(csv_path, img_dir):
    df = pd.read_csv(csv_path, encoding='gbk')

    needed_cols = {'图片名字', '瘤变标签'}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f'CSV 缺少必要列: {missing}')

    df = df[df['瘤变标签'].isin([0, 1])].copy()
    df['图片名字'] = df['图片名字'].astype(str)

    exists_mask = df['图片名字'].apply(
        lambda name: os.path.exists(os.path.join(img_dir, name))
    )
    df_valid = df[exists_mask].copy()

    if len(df_valid) == 0:
        raise RuntimeError('没有找到 CSV 与图片文件成功匹配的有效样本。')

    print(f'有效样本数: {len(df_valid)}')
    print(f'标签分布 — 早癌/瘤变(1): {(df_valid["瘤变标签"]==1).sum()}, '
          f'非癌(0): {(df_valid["瘤变标签"]==0).sum()}')

    return df_valid


def split_dataframe(df):
    train_val_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df['瘤变标签'], random_state=RANDOM_SEED,
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.15 / 0.85,
        stratify=train_val_df['瘤变标签'], random_state=RANDOM_SEED,
    )
    print(f'数据划分 — 训练: {len(train_df)}, 验证: {len(val_df)}, 测试: {len(test_df)}')
    return train_df, val_df, test_df


def build_loaders():
    df_valid = load_matched_dataframe(CSV_PATH, DATA_DIR)

    if DEBUG and len(df_valid) > DEBUG_SAMPLES:
        df_valid, _ = train_test_split(
            df_valid, train_size=DEBUG_SAMPLES,
            stratify=df_valid['瘤变标签'], random_state=RANDOM_SEED,
        )
        print(f'[DEBUG] 仅使用 {DEBUG_SAMPLES} 张样本进行快速验证')

    train_df, val_df, test_df = split_dataframe(df_valid)

    train_set = GastricDataset(train_df, DATA_DIR, transform=train_transform)
    val_set   = GastricDataset(val_df,   DATA_DIR, transform=eval_transform)
    test_set  = GastricDataset(test_df,  DATA_DIR, transform=eval_transform)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    return train_loader, val_loader, test_loader


# ============================================================
# 3. 模型构建 & 冻结策略（EfficientNet 专用）
# ============================================================

def build_model(num_classes=2):
    """
    构建 ImageNet 预训练 EfficientNet-B0，替换分类头为二分类输出。

    EfficientNet-B0 结构概览：
        features   : 8 个 MBConv 阶段的 Sequential
        avgpool    : AdaptiveAvgPool2d(1)
        classifier : Sequential(Dropout(0.2), Linear(1280→1000))

    这里将最后的 Linear(1280→1000) 替换为 Linear(1280→2)。
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    # classifier 是 Sequential，最后一个是 Linear 层
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def unwrap(model):
    """DataParallel 包装时取 .module，否则直接返回。"""
    return model.module if hasattr(model, 'module') else model


def freeze_bn_stats(model):
    """将所有 requires_grad=False 的 BatchNorm 设为 eval，冻结 running mean/var。"""
    m = unwrap(model)
    for module in m.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()


def set_trainable_stage1(model):
    """第一阶段：冻结 backbone，仅训练 classifier 分类头。"""
    m = unwrap(model)
    for param in m.parameters():
        param.requires_grad = False
    for param in m.classifier.parameters():
        param.requires_grad = True


def set_trainable_stage2(model):
    """
    第二阶段：微调 features 后段（第 6-7 阶段）+ classifier。

    EfficientNet-B0 的 features 有 8 个阶段（索引 0-7），后段负责高层语义。
    解冻 features[5:] ≈ 最后 3 个阶段，类比 ResNet 的 layer4 微调策略。
    """
    m = unwrap(model)
    for param in m.parameters():
        param.requires_grad = False
    # features 是 nn.Sequential，用切片解冻后段
    for i in range(5, len(m.features)):
        for param in m.features[i].parameters():
            param.requires_grad = True
    for param in m.classifier.parameters():
        param.requires_grad = True


# ============================================================
# 4. 评估指标 & 训练工具（与 ResNet 版本一致）
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    return avg_loss, metrics


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        'AUC':         roc_auc_score(y_true, y_prob),
        'Accuracy':    accuracy_score(y_true, y_pred),
        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'Precision':   tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'F1':          f1_score(y_true, y_pred),
        'CM':          (int(tn), int(fp), int(fn), int(tp)),
    }


def format_metrics(metrics):
    return (
        f"AUC={metrics['AUC']:.4f} | "
        f"Acc={metrics['Accuracy']:.4f} | "
        f"Sens={metrics['Sensitivity']:.4f} | "
        f"Spec={metrics['Specificity']:.4f} | "
        f"Prec={metrics['Precision']:.4f} | "
        f"F1={metrics['F1']:.4f} | "
        f"CM={metrics['CM']}"
    )


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    freeze_bn_stats(model)           # 冻结 backbone BN 统计量，避免漂移
    total_loss = 0.0
    start = time.time()

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    elapsed = time.time() - start
    return total_loss / len(loader.dataset), elapsed


def run_stage(model, train_loader, val_loader, criterion, optimizer,
              epochs, stage_name, scheduler=None, early_stop_patience=None):
    best_auc = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        train_loss, elapsed = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_metrics = evaluate(model, val_loader, criterion)

        if scheduler is not None:
            scheduler.step()

        lr_str = f'LR={scheduler.get_last_lr()[0]:.2e} | ' if scheduler else ''
        print(f'{stage_name} Epoch {epoch+1:02d}/{epochs} , 耗时 {elapsed:.0f}s | '
              f'TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | '
              f'{lr_str}{format_metrics(val_metrics)}')

        if val_metrics['AUC'] > best_auc:
            best_auc = val_metrics['AUC']
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if early_stop_patience and no_improve >= early_stop_patience:
                print(f'Early Stop — 连续 {early_stop_patience} epoch 验证 AUC 未提升')
                break

    if best_state is None:
        raise RuntimeError(f'{stage_name} 未产生可用最佳模型。')

    model.load_state_dict(best_state)
    return model, best_auc


# ============================================================
# 5. 主训练流程
# ============================================================

def main():
    seed_everything(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'当前设备: {DEVICE}')
    if torch.cuda.is_available():
        print(f'GPU 数量: {torch.cuda.device_count()}')
        print(f'当前 GPU: {torch.cuda.get_device_name(0)}')

    train_loader, val_loader, test_loader = build_loaders()

    model = build_model(num_classes=2).to(DEVICE)

    # 多 GPU：DataParallel 自动利用所有可见 GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f'已启用 DataParallel，使用 {torch.cuda.device_count()} 张 GPU')

    criterion = nn.CrossEntropyLoss()

    # ===== 第一阶段：冻结 backbone，仅训练 classifier =====
    print('\n' + '=' * 60)
    print(f'第一阶段 — 冻结 backbone，仅训练 classifier 分类头 '
          f'({STAGE1_EPOCHS} epochs, LR={STAGE1_LR})')
    print('=' * 60)

    set_trainable_stage1(model)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE1_LR, weight_decay=WEIGHT_DECAY,
    )
    model, best_s1_auc = run_stage(
        model, train_loader, val_loader, criterion, optimizer,
        STAGE1_EPOCHS, 'S1',
    )
    print(f'第一阶段完成 — 最佳 Val AUC: {best_s1_auc:.4f}')

    # ===== 第二阶段：微调 features 后段 + classifier =====
    print('\n' + '=' * 60)
    print(f'第二阶段 — 微调 features[5:] + classifier '
          f'({STAGE2_EPOCHS} epochs, LR={STAGE2_LR}, EarlyStop={EARLY_STOP_PATIENCE})')
    print('=' * 60)

    set_trainable_stage2(model)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE2_LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE2_EPOCHS)

    model, best_s2_auc = run_stage(
        model, train_loader, val_loader, criterion, optimizer,
        STAGE2_EPOCHS, 'S2',
        scheduler=scheduler, early_stop_patience=EARLY_STOP_PATIENCE,
    )
    print(f'第二阶段完成 — 最佳 Val AUC: {best_s2_auc:.4f}')

    # ===== 测试集最终评估 =====
    print('\n' + '=' * 60)
    print('测试集最终评估')
    print('=' * 60)

    test_loss, test_metrics = evaluate(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test AUC:         {test_metrics["AUC"]:.4f}')
    print(f'Test Accuracy:    {test_metrics["Accuracy"]:.4f}')
    print(f'Test Sensitivity: {test_metrics["Sensitivity"]:.4f}  ← 关键：早癌召回率')
    print(f'Test Specificity: {test_metrics["Specificity"]:.4f}')
    print(f'Test Precision:   {test_metrics["Precision"]:.4f}')
    print(f'Test F1:          {test_metrics["F1"]:.4f}')
    print(f'Test Confusion Matrix (TN, FP, FN, TP): {test_metrics["CM"]}')

    # ===== 保存 =====
    # DataParallel 包装时需从 .module 取 state_dict
    state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    save_path = os.path.join(OUTPUT_DIR, 'efficientnet_b0_best.pth')
    torch.save({
        'model_state_dict': state,
        'best_val_auc': best_s2_auc,
        'test_metrics': test_metrics,
        'config': {
            'model': 'EfficientNet-B0',
            'batch_size': BATCH_SIZE,
            'stage1_epochs': STAGE1_EPOCHS,
            'stage2_epochs': STAGE2_EPOCHS,
            'stage1_lr': STAGE1_LR,
            'stage2_lr': STAGE2_LR,
            'weight_decay': WEIGHT_DECAY,
            'early_stop_patience': EARLY_STOP_PATIENCE,
            'random_seed': RANDOM_SEED,
        },
    }, save_path)
    print(f'\n模型已保存至: {save_path}')


if __name__ == '__main__':
    main()
