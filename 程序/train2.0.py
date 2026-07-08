"""
ResNet50 迁移学习训练脚本。

任务：
    白光胃镜图像二分类。

标签：
    0 = 非癌
    1 = 早癌 / 瘤变

主要思路：
    1. 使用 torchvision 提供的 ImageNet 预训练 ResNet50。
    2. 将原本用于 ImageNet 1000 类分类的 fc 层替换为 2 类分类层。
    3. 采用两阶段训练：
       - 第一阶段：冻结主干网络，只训练新的 fc 分类层。
       - 第二阶段：解冻 layer4 和 fc，进行小学习率微调。
"""

import os
from copy import deepcopy

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


# ============================================================
# 0. 全局配置
# ============================================================

# 使用脚本自身位置来构建路径，这样无论从项目根目录还是从“程序”文件夹运行，
# 都能正确找到数据文件。
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "数据", "胃图文带特征标签数据集 3600+ 1933瘤变")
CSV_PATH = os.path.join(BASE_DIR, "数据", "胃图文标签表格-添加瘤变标签.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "程序", "outputs")

BATCH_SIZE = 32
NUM_WORKERS = 4
RANDOM_SEED = 42

STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 20
STAGE1_LR = 1e-3
STAGE2_LR = 1e-4
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. 图像预处理与数据增强
# ============================================================

# 这里使用 ImageNet 预训练模型对应的均值和标准差。
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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
# 2. 数据集与数据划分
# ============================================================

class GastricDataset(Dataset):
    """CSV 与图片成功匹配后的胃镜图像二分类数据集。"""

    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["图片名字"])

        image = Image.open(img_path).convert("RGB")
        label = int(row["瘤变标签"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def seed_everything(seed):
    """固定随机种子，让数据划分和训练过程尽量可复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_matched_dataframe(csv_path, img_dir):
    """
    读取 CSV，并且只保留满足以下条件的样本：
        - “瘤变标签”为 0 或 1
        - 对应图片文件真实存在
    """
    df = pd.read_csv(csv_path, encoding="gbk")

    needed_cols = {"图片名字", "瘤变标签"}
    missing_cols = needed_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV 缺少必要列: {missing_cols}")

    df = df[df["瘤变标签"].isin([0, 1])].copy()
    df["图片名字"] = df["图片名字"].astype(str)

    exists_mask = df["图片名字"].apply(
        lambda name: os.path.exists(os.path.join(img_dir, name))
    )
    df_valid = df[exists_mask].copy()

    if len(df_valid) == 0:
        raise RuntimeError("没有找到 CSV 与图片文件成功匹配的有效样本。")

    print(f"有效样本数: {len(df_valid)}")
    print(
        "标签分布: "
        f"早癌/瘤变(1)={(df_valid['瘤变标签'] == 1).sum()}, "
        f"非癌(0)={(df_valid['瘤变标签'] == 0).sum()}"
    )

    return df_valid


def split_dataframe(df):
    """按标签分层划分训练集、验证集和测试集，比例为 70/15/15。"""
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.15,
        stratify=df["瘤变标签"],
        random_state=RANDOM_SEED,
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.15 / 0.85,
        stratify=train_val_df["瘤变标签"],
        random_state=RANDOM_SEED,
    )

    print(
        f"数据划分: 训练={len(train_df)}, 验证={len(val_df)}, 测试={len(test_df)}"
    )

    return train_df, val_df, test_df


def build_loaders():
    df_valid = load_matched_dataframe(CSV_PATH, DATA_DIR)
    train_df, val_df, test_df = split_dataframe(df_valid)

    train_set = GastricDataset(train_df, DATA_DIR, transform=train_transform)
    val_set = GastricDataset(val_df, DATA_DIR, transform=eval_transform)
    test_set = GastricDataset(test_df, DATA_DIR, transform=eval_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


# ============================================================
# 3. 模型构建
# ============================================================

def build_model(num_classes=2):
    """
    构建 ImageNet 预训练 ResNet50，并替换最后的分类层。

    重要：
        ImageNet 原始 fc 层是 1000 类分类层，不能直接用于本项目。
        这里只复用预训练特征提取部分，新的二分类 fc 层随机初始化。
    """
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def set_trainable_stage1(model):
    """第一阶段：冻结主干网络，只训练新的 fc 分类层。"""
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True


def set_trainable_stage2(model):
    """
    第二阶段：微调 layer4 和 fc。

    相比直接解冻整个网络，只微调最后一个残差阶段更保守，
    对中小规模医学图像数据集通常更稳。
    """
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True


# ============================================================
# 4. 评估指标与训练辅助函数
# ============================================================

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "Precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "F1": f1_score(y_true, y_pred),
        "CM": (int(tn), int(fp), int(fn), int(tp)),
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
    total_loss = 0.0

    for images, labels in tqdm(loader, desc="训练中", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_probs)

    return avg_loss, metrics


def run_stage(model, train_loader, val_loader, criterion, optimizer, epochs, stage_name):
    best_auc = -1.0
    best_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_metrics = evaluate(model, val_loader, criterion)

        print(
            f"{stage_name} Epoch {epoch + 1:02d}/{epochs} | "
            f"TrainLoss={train_loss:.4f} | "
            f"ValLoss={val_loss:.4f} | "
            f"{format_metrics(val_metrics)}"
        )

        if val_metrics["AUC"] > best_auc:
            best_auc = val_metrics["AUC"]
            best_state = deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError(f"{stage_name} 没有产生可用的最佳模型。")

    model.load_state_dict(best_state)
    return model, best_auc


# ============================================================
# 5. 主训练流程
# ============================================================

def main():
    seed_everything(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"当前设备: {DEVICE}")
    train_loader, val_loader, test_loader = build_loaders()

    model = build_model(num_classes=2).to(DEVICE)

    # 第一版基线先不使用类别权重，便于解释和复现实验结果。
    # 后续如果希望进一步提高早癌敏感度，可以在验证集上调整判别阈值。
    criterion = nn.CrossEntropyLoss()

    print("\n" + "=" * 60)
    print("第一阶段：冻结主干网络，只训练 fc 分类层")
    print("=" * 60)
    set_trainable_stage1(model)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE1_LR,
        weight_decay=WEIGHT_DECAY,
    )
    model, best_s1_auc = run_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        STAGE1_EPOCHS,
        "S1",
    )
    print(f"第一阶段最佳验证集 AUC: {best_s1_auc:.4f}")

    print("\n" + "=" * 60)
    print("第二阶段：微调 layer4 和 fc")
    print("=" * 60)
    set_trainable_stage2(model)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE2_LR,
        weight_decay=WEIGHT_DECAY,
    )
    model, best_s2_auc = run_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        STAGE2_EPOCHS,
        "S2",
    )
    print(f"第二阶段最佳验证集 AUC: {best_s2_auc:.4f}")

    print("\n" + "=" * 60)
    print("最终测试集评估")
    print("=" * 60)
    test_loss, test_metrics = evaluate(model, test_loader, criterion)
    print(f"TestLoss={test_loss:.4f} | {format_metrics(test_metrics)}")

    save_path = os.path.join(OUTPUT_DIR, "resnet50_transfer_best.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_val_auc": best_s2_auc,
            "test_metrics": test_metrics,
            "config": {
                "batch_size": BATCH_SIZE,
                "stage1_epochs": STAGE1_EPOCHS,
                "stage2_epochs": STAGE2_EPOCHS,
                "stage1_lr": STAGE1_LR,
                "stage2_lr": STAGE2_LR,
                "weight_decay": WEIGHT_DECAY,
                "random_seed": RANDOM_SEED,
            },
        },
        save_path,
    )
    print(f"模型已保存: {save_path}")


if __name__ == "__main__":
    main()
