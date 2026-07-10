"""
迁移学习训练脚本
基于 ImageNet 预训练 ResNet50，在胃早癌白光胃镜图像上微调二分类模型

训练策略（两阶段）：
  第一阶段 — 冻结 backbone，仅训练分类头（FC 层），学习率较高
  第二阶段 — 解冻全网络，降低学习率，完整微调

参照 agents.md：
  仅使用 CSV 中能与图片匹配的 3314 张有效样本
  目标列：瘤变标签（1=早癌/瘤变，0=非癌）
"""

import os                                    # 文件路径操作
import random                                # Python 随机数
import numpy as np                           # 数值计算
import pandas as pd                          # CSV 表格读写
from PIL import Image                        # 胃镜图像读取
from tqdm import tqdm                        # 训练进度条
from copy import deepcopy                    # 模型参数深拷贝（保存最佳模型）

import torch                                 # PyTorch 核心库
import torch.nn as nn                        # 神经网络模块
import torch.optim as optim                  # 优化器
from torch.utils.data import Dataset, DataLoader  # 数据集加载
import torchvision.transforms as transforms       # 图像预处理与数据增强
import torchvision.models as torchvision_models   # 获取 ImageNet 预训练权重
from sklearn.model_selection import train_test_split  # 分层划分训练/验证/测试集
from sklearn.metrics import (                        # 评估指标
    roc_auc_score, accuracy_score, confusion_matrix,
    classification_report, f1_score
)

from resnet import ResNet50                   # 自定义 ResNet50 模型

# ============================================================
# 固定随机种子（保证每次运行的划分、初始化、增强顺序一致，实验可复现）
# ============================================================
SEED = 42

random.seed(SEED)                            # Python 标准库随机
np.random.seed(SEED)                         # NumPy 随机（sklearn 划分依赖）
torch.manual_seed(SEED)                      # PyTorch CPU 随机（DataLoader shuffle、权重初始化）
torch.cuda.manual_seed(SEED)                 # PyTorch 当前 GPU 随机
torch.cuda.manual_seed_all(SEED)             # PyTorch 所有 GPU 随机
torch.backends.cudnn.deterministic = True    # cuDNN 确定性模式（牺牲少量速度换可复现）
torch.backends.cudnn.benchmark = False       # 关闭 cuDNN 自动调优（避免不同 run 选不同算法）

# ============================================================
# 全局配置
# ============================================================

# 路径（根据实际情况调整）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, '数据', '胃图文带特征标签数据集 3600+ 1933瘤变')
CSV_PATH = os.path.join(BASE_DIR, '数据', '胃图文标签表格-添加瘤变标签.csv')

# 训练超参数
BATCH_SIZE    = 32
NUM_EPOCHS_S1 = 10      # 第一阶段：仅训练分类头
NUM_EPOCHS_S2 = 20      # 第二阶段：全网络微调
LR_S1         = 1e-3    # 第一阶段学习率
LR_S2         = 1e-4    # 第二阶段学习率（更低，避免破坏预训练特征）
NUM_WORKERS   = 4       # DataLoader 线程数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据增强与预处理
# 训练集：RandAugment 风格增强（胃镜图可容忍一定程度的翻转和颜色抖动）
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),              # 先放大到 256
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪 → 224
    transforms.RandomHorizontalFlip(p=0.5),     # 随机水平翻转
    transforms.RandomRotation(degrees=15),      # 小角度旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # 颜色抖动（胃镜光照差异大）
    transforms.ToTensor(),                      # [0,255] → [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet 均值
                         std=[0.229, 0.224, 0.225]),   # ImageNet 标准差
])

# 验证/测试集：仅缩放 + 归一化，不做增强
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ============================================================
# 1. 数据集
# ============================================================

class GastricDataset(Dataset):
    """
    胃镜图像数据集

    仅保留 CSV 中「瘤变标签」为 0 或 1 且图片文件存在的样本。
    """
    def __init__(self, df, img_dir, transform=None):
        """
        Args:
            df: pandas DataFrame，包含「图片名字」和「瘤变标签」两列
            img_dir: 图片所在文件夹路径
            transform: torchvision 变换
        """
        self.img_dir = img_dir
        self.transform = transform

        # 过滤无效标签（888 等缺失值）
        df = df[df['瘤变标签'].isin([0, 1])].copy()
        self.labels = df['瘤变标签'].values.astype(np.int64)
        self.img_names = df['图片名字'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def load_data(csv_path, img_dir):
    """
    读取带标签表格，匹配图片文件，划分训练/验证/测试集

    返回: (train_loader, val_loader, test_loader)
    """
    # GBK 编码读取（CSV 中含中文字段名）
    df = pd.read_csv(csv_path, encoding='gbk')

    # 筛选有对应图片且标签有效的样本
    valid_rows = []
    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, row['图片名字'])
        if os.path.exists(img_path) and row['瘤变标签'] in [0, 1]:
            valid_rows.append(row)

    df_valid = pd.DataFrame(valid_rows)
    print(f'有效样本数: {len(df_valid)}')
    print(f'标签分布 — 早癌(1): {(df_valid["瘤变标签"]==1).sum()}, '
          f'非癌(0): {(df_valid["瘤变标签"]==0).sum()}')

    # 分层划分：先分出测试集（15%），再分出验证集（15%/85% ≈ 17.6%）
    train_df, test_df = train_test_split(
        df_valid, test_size=0.15, stratify=df_valid['瘤变标签'], random_state=42
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15 / 0.85, stratify=train_df['瘤变标签'], random_state=42
    )

    print(f'划分 — 训练: {len(train_df)}, 验证: {len(val_df)}, 测试: {len(test_df)}')

    train_dataset = GastricDataset(train_df, img_dir, train_transform)
    val_dataset   = GastricDataset(val_df,   img_dir, eval_transform)
    test_dataset  = GastricDataset(test_df,  img_dir, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader


# ============================================================
# 2. 预训练权重迁移
# ============================================================

def load_pretrained_weights(model):
    """
    将 torchvision ResNet50 的 ImageNet 预训练权重迁移到自定义模型上

    策略：按参数名逐层匹配。自定义模型中与 torchvision 同名的层直接拷贝；
    stem 被包在 Sequential 中，需要按索引映射。
    """
    tv_model = torchvision_models.resnet50(weights='IMAGENET1K_V2')
    tv_state = tv_model.state_dict()
    custom_state = model.state_dict()

    # 建立 torchvision → 自定义模型的参数名映射
    mapping = {}

    # --- Stem 映射 ---
    # torchvision: conv1 / bn1 → 自定义: stem.0 / stem.1
    mapping['conv1.weight'] = 'stem.0.weight'
    mapping['bn1.weight']   = 'stem.1.weight'
    mapping['bn1.bias']     = 'stem.1.bias'
    mapping['bn1.running_mean'] = 'stem.1.running_mean'
    mapping['bn1.running_var']  = 'stem.1.running_var'
    mapping['bn1.num_batches_tracked'] = 'stem.1.num_batches_tracked'

    # --- 四个 Stage 同名映射 ---
    for stage in ['layer1', 'layer2', 'layer3', 'layer4']:
        for tv_name, tv_param in tv_state.items():
            if tv_name.startswith(stage):
                mapping[tv_name] = tv_name  # 同名直接映射

    # 注意：fc.weight/bias 不迁移（torchvision 是 1000 类，我们是 2 类，形状不匹配）
    # linear 层保持随机初始化，由第一阶段训练从头学习

    # --- 执行拷贝 ---
    loaded = 0
    skipped = 0
    for tv_name, tv_param in tv_state.items():
        if tv_name in mapping:
            custom_name = mapping[tv_name]
            if custom_name in custom_state:
                custom_state[custom_name] = tv_param
                loaded += 1
            else:
                skipped += 1
        else:
            # 不需要映射的参数（如 torchvision 中 stem 的 relu/maxpool，无参数层不需要拷贝）
            pass

    model.load_state_dict(custom_state)
    print(f'预训练权重已迁移: {loaded} 层加载，{skipped} 层跳过')
    return model


# ============================================================
# 3. 训练 & 评估工具
# ============================================================

def compute_metrics(y_true, y_prob):
    """
    计算二分类核心评估指标

    Args:
        y_true: 真实标签 (0/1)
        y_prob: 模型预测的癌概率

    Returns:
        dict: AUC / Accuracy / Sensitivity / Specificity / Precision / F1 / Confusion Matrix
    """
    # 概率 → 硬判决（阈值 0.5）
    y_pred = (y_prob >= 0.5).astype(int)

    # 混淆矩阵的四个元素
    # TN (True  Negative): 实际非癌 → 预测非癌 ✓  （正确排除）
    # FP (False Positive): 实际非癌 → 预测癌   ✗  （误报）
    # FN (False Negative): 实际癌   → 预测非癌 ✗  （漏诊，最危险）
    # TP (True  Positive): 实际癌   → 预测癌   ✓  （正确检出）
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        'AUC':         roc_auc_score(y_true, y_prob),       # ROC 曲线下面积，综合衡量模型区分癌/非癌的能力
        'Accuracy':    accuracy_score(y_true, y_pred),      # 总体准确率 = (TP+TN) / 全部
        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,   # 召回率/查全率 = TP/(TP+FN)，早癌检出率，临床最关键指标
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,   # 特异度 = TN/(TN+FP)，非癌正确排除率
        'Precision':   tp / (tp + fp) if (tp + fp) > 0 else 0,   # 精确率/查准率 = TP/(TP+FP)，模型判癌中有多少是真癌
        'F1':          f1_score(y_true, y_pred),                  # F1 = 2*P*R/(P+R)，Precision 和 Recall 的调和平均
        'CM':          (tn, fp, fn, tp),                          # 混淆矩阵原始值 (TN, FP, FN, TP)
    }


def validate(model, loader, criterion):
    """验证/测试阶段：计算 loss + 收集预测结果"""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 取「早癌」概率
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))

    return avg_loss, metrics


def train_one_epoch(model, loader, optimizer, criterion):
    """一个 epoch 的训练"""
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc='Training', leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def format_metrics(metrics):
    """格式化打印指标"""
    cm = metrics['CM']
    return (
        f"AUC={metrics['AUC']:.4f} | Acc={metrics['Accuracy']:.4f} | "
        f"Sens={metrics['Sensitivity']:.4f} | Spec={metrics['Specificity']:.4f} | "
        f"Prec={metrics['Precision']:.4f} | F1={metrics['F1']:.4f} | "
        f"CM={cm}"
    )


# ============================================================
# 4. 两阶段迁移学习
# ============================================================

def train():
    # ---- 准备数据 ----
    train_loader, val_loader, test_loader = load_data(CSV_PATH, DATA_DIR)

    # ---- 构建模型 + 加载预训练权重 ----
    model = ResNet50(num_classes=2)
    model = load_pretrained_weights(model)
    model = model.to(DEVICE)

    # 交叉熵损失（早癌 1918 : 非癌 1396，比例 1.37:1，不平衡较轻，不加类别权重）
    # 胃早癌筛查场景优先关注 Sensitivity（早癌召回率），评估时重点看该指标即可
    criterion = nn.CrossEntropyLoss()

    # ---- 第一阶段：冻结 backbone，仅训练分类头 ----
    print('\n' + '=' * 60)
    print(f'第一阶段 — 冻结 backbone，仅训练分类头（{NUM_EPOCHS_S1} epochs, LR={LR_S1}）')
    print('=' * 60)

    # 冻结除 fc 外的所有参数
    for name, param in model.named_parameters():
        if 'linear' not in name:  # 注意：我们的分类头叫 self.linear
            param.requires_grad = False

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_S1
    )

    best_val_auc = 0.0
    best_model_s1 = None

    for epoch in range(NUM_EPOCHS_S1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_metrics = validate(model, val_loader, criterion)

        print(f'S1 Epoch {epoch+1:2d}/{NUM_EPOCHS_S1} | '
              f'TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | '
              f'{format_metrics(val_metrics)}')

        if val_metrics['AUC'] > best_val_auc:
            best_val_auc = val_metrics['AUC']
            best_model_s1 = deepcopy(model.state_dict())

    model.load_state_dict(best_model_s1)
    print(f'第一阶段完成 — 最佳 Val AUC: {best_val_auc:.4f}')

    # ---- 第二阶段：解冻全网络，低学习率微调 ----
    print('\n' + '=' * 60)
    print(f'第二阶段 — 全网络微调（{NUM_EPOCHS_S2} epochs, LR={LR_S2}）')
    print('=' * 60)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=LR_S2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_S2)

    best_val_auc_s2 = 0.0
    best_model_s2 = None
    patience = 5                    # Early stop：连续 N 个 epoch 无提升则停止
    no_improve = 0

    for epoch in range(NUM_EPOCHS_S2):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_metrics = validate(model, val_loader, criterion)
        scheduler.step()

        print(f'S2 Epoch {epoch+1:2d}/{NUM_EPOCHS_S2} | '
              f'TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | '
              f'LR={scheduler.get_last_lr()[0]:.2e} | '
              f'{format_metrics(val_metrics)}')

        if val_metrics['AUC'] > best_val_auc_s2:
            best_val_auc_s2 = val_metrics['AUC']
            best_model_s2 = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stop — 连续 {patience} 个 epoch 验证 AUC 未提升')
                break

    model.load_state_dict(best_model_s2)
    print(f'第二阶段完成 — 最佳 Val AUC: {best_val_auc_s2:.4f}')

    # ---- 在测试集上最终评估 ----
    print('\n' + '=' * 60)
    print('测试集最终评估')
    print('=' * 60)

    test_loss, test_metrics = validate(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test AUC:         {test_metrics["AUC"]:.4f}')
    print(f'Test Accuracy:    {test_metrics["Accuracy"]:.4f}')
    print(f'Test Sensitivity: {test_metrics["Sensitivity"]:.4f}  ← 关键：早癌召回率')
    print(f'Test Specificity: {test_metrics["Specificity"]:.4f}')
    print(f'Test Precision:   {test_metrics["Precision"]:.4f}')
    print(f'Test F1:          {test_metrics["F1"]:.4f}')
    print(f'Test Confusion Matrix (TN,FP,FN,TP): {test_metrics["CM"]}')

    # ---- 保存模型 ----
    save_path = 'resnet50_gastric_best.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_auc': best_val_auc_s2,
        'test_metrics': test_metrics,
    }, save_path)
    print(f'\n模型已保存至: {save_path}')

    return model, test_metrics


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    train()
