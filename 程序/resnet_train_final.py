"""
ResNet50 迁移学习训练脚本（最终版）
基于 ImageNet 预训练 ResNet50，在胃早癌白光胃镜图像上微调二分类模型

训练策略（两阶段）：
  第一阶段 — 冻结 backbone，仅训练 FC 分类头，较高学习率
  第二阶段 — 仅微调 layer4 + FC，低学习率 + CosineAnnealing + Early Stop

参照 agents.md：
  仅使用 CSV 中能与图片匹配的 3314 张有效样本
  目标列：瘤变标签（1=早癌/瘤变，0=非癌）
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
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, f1_score,
)


# ============================================================
# 0. 全局配置
# ============================================================

# ---- 调试模式：仅用少量数据快速跑通，检查有无 bug ----
DEBUG = False                # ← 正式训练时改为 False
DEBUG_SAMPLES = 200         # 调试时使用的总样本数

# 基于脚本自身位置构建绝对路径，无论从哪个目录启动都能正确找到数据
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, '数据', '胃图文带特征标签数据集 3600+ 1933瘤变')
CSV_PATH   = os.path.join(BASE_DIR, '数据', '胃图文标签表格-添加瘤变标签.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '程序', '结果')

BATCH_SIZE     = 32
NUM_WORKERS    = 0 if DEBUG else 4    # 调试时单进程，报错信息更清晰
RANDOM_SEED    = 42

STAGE1_EPOCHS  = 2 if DEBUG else 10  # 调试时各跑 2 轮即可
STAGE2_EPOCHS  = 2 if DEBUG else 20
STAGE1_LR      = 1e-3
STAGE2_LR      = 1e-4
WEIGHT_DECAY   = 1e-4
EARLY_STOP_PATIENCE = 5               # 连续 N 个 epoch Val AUC 未提升则停止

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ImageNet 预训练模型的均值与标准差
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ============================================================
# 1. 随机种子 & 数据增强
# ============================================================

def seed_everything(seed):
    """固定所有随机源，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 训练集增强：随机裁剪 + 翻转 + 旋转 + 颜色抖动（胃镜光照差异大）
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# 验证/测试集：仅缩放 + 归一化，不做增强
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ============================================================
# 2. 数据集
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
        image = Image.open(img_path).convert('RGB')          # 统一为 3 通道 RGB
        label = int(row['瘤变标签'])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_matched_dataframe(csv_path, img_dir):
    """
    读取 GBK 编码的 CSV，过滤出标签为 0/1 且图片文件真实存在的有效样本。
    返回干净的 DataFrame。
    """
    df = pd.read_csv(csv_path, encoding='gbk')

    # 校验必要列
    needed_cols = {'图片名字', '瘤变标签'}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f'CSV 缺少必要列: {missing}')

    # 过滤无效标签（888 等缺失值）
    df = df[df['瘤变标签'].isin([0, 1])].copy()
    df['图片名字'] = df['图片名字'].astype(str)

    # 向量化检查图片是否存在（比 iterrows 快）
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
    """按标签分层划分训练/验证/测试集 = 70/15/15。"""
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
    """构建训练/验证/测试 DataLoader。"""
    df_valid = load_matched_dataframe(CSV_PATH, DATA_DIR)

    # 调试模式：分层采样少量数据，快速验证代码能否跑通
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

    pin = torch.cuda.is_available()     # GPU 时启用 pinned memory，加速数据传输

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    return train_loader, val_loader, test_loader


# ============================================================
# 3. 模型构建 & 冻结策略
# ============================================================

def build_model(num_classes=2):
    """
    构建 ImageNet 预训练 ResNet50，替换 FC 层为二分类输出。

    torchvision 的 fc 层是 1000 类，替换为 2 类随机初始化；
    backbone 权重来自 ImageNet，不需要手动迁移。
    """
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_trainable_stage1(model):
    """第一阶段：冻结 backbone，仅训练 FC 分类头。"""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


def set_trainable_stage2(model):
    """
    第二阶段：仅微调 layer4（最高层语义特征）+ FC。

    相比解冻全网络，对 3314 张的中小医学数据集更保守、更不易过拟合。
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True


# ============================================================
# 4. 评估指标 & 训练工具
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion):
    """验证/测试：收集所有预测结果，返回 loss 和指标字典。"""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]    # 取「早癌」概率
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    return avg_loss, metrics


def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    计算二分类核心评估指标。

    Args:
        y_true: 真实标签 (0/1)
        y_prob: 模型预测的癌概率
        threshold: 硬判决阈值，默认 0.5
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(np.int64)

    # 混淆矩阵（显式指定 labels 顺序，确保 TN/FP/FN/TP 不会错位）
    # TN: 实际非癌 → 预测非癌 ✓   FP: 实际非癌 → 预测癌   ✗
    # FN: 实际癌   → 预测非癌 ✗   TP: 实际癌   → 预测癌   ✓
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        'AUC':         roc_auc_score(y_true, y_prob),         # ROC 曲线下面积
        'Accuracy':    accuracy_score(y_true, y_pred),        # 总体准确率
        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # 早癌检出率，关键指标
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,  # 非癌正确排除率
        'Precision':   tp / (tp + fp) if (tp + fp) > 0 else 0.0,  # 判癌中有多少是真癌
        'F1':          f1_score(y_true, y_pred),                    # P 和 R 的调和平均
        'CM':          (int(tn), int(fp), int(fn), int(tp)),        # 混淆矩阵原始值
    }


def format_metrics(metrics):
    """格式化输出一行指标。"""
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
    """一个 epoch 的训练，返回 (平均 loss, 耗时秒数)。"""
    model.train()
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
    """
    运行一个训练阶段，每个 epoch 后验证并择优保存。
    支持 LR scheduler 和 Early Stop。
    """
    best_auc = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        train_loss, elapsed = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_metrics = evaluate(model, val_loader, criterion)

        if scheduler is not None:
            scheduler.step()

        # 日志
        lr_str = f'LR={scheduler.get_last_lr()[0]:.2e} | ' if scheduler else ''
        print(f'{stage_name} Epoch {epoch+1:02d}/{epochs} , 耗时 {elapsed:.0f}s | '
              f'TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | '
              f'{lr_str}{format_metrics(val_metrics)}')

        # 择优
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
    train_loader, val_loader, test_loader = build_loaders()

    model = build_model(num_classes=2).to(DEVICE)

    # 不做类别加权（1918:1396 = 1.37:1，不平衡较轻）
    # 胃早癌筛查优先关注 Sensitivity，评估时重点看该指标即可
    criterion = nn.CrossEntropyLoss()

    # ===== 第一阶段：冻结 backbone，仅训练 FC =====
    print('\n' + '=' * 60)
    print(f'第一阶段 — 冻结 backbone，仅训练 FC 分类头 '
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

    # ===== 第二阶段：仅微调 layer4 + FC =====
    print('\n' + '=' * 60)
    print(f'第二阶段 — 微调 layer4 + FC '
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
    save_path = os.path.join(OUTPUT_DIR, 'resnet50_transfer_best.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_val_auc': best_s2_auc,
        'test_metrics': test_metrics,
        'config': {
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
