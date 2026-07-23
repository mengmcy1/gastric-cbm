"""
ResNet50 模型实现
用于胃早癌白光胃镜图像二分类任务（癌/非癌）
支持 ImageNet 预训练权重加载，结构清晰便于后续 Grad-CAM 和 MOCE 接入
"""

import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    """
    ResNet Bottleneck 残差块

    标准三明治结构：1×1降维 → 3×3特征提取 → 1×1升维 + 跳跃连接

    Args:
        in_channels: 输入通道数
        mid_channels: 中间通道数（输出通道 = mid_channels × expansion）
        stride: 3×3卷积步长（用于空间下采样）
        downsample: 捷径上的下采样模块（1×1 Conv + BN，无 ReLU）
    """
    expansion = 4  # 输出通道膨胀系数

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()

        # 1×1 降维：in_channels → mid_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 3×3 特征提取
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1×1 升维：mid_channels → mid_channels * 4
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 捷径连接：仅当通道/尺寸变化时需要

    def forward(self, x):
        identity = x

        # 主路径
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))           # conv3 后暂不加 ReLU

        # 捷径连接：调整通道/尺寸使其与主路径输出对齐
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity                             # 加法
        out = self.relu(out)                        # 加法后统一激活
        return out


# ============================================================
# ResNet50
# ============================================================
# 各阶段 Bottleneck 数量：[3, 4, 6, 3]
# 各阶段通道变化：[64→256] → [256→512] → [512→1024] → [1024→2048]
# ============================================================

class ResNet50(nn.Module):
    """
    ResNet50 模型

    结构概览：
        Stem   : 7×7 Conv(s=2) → BN → ReLU → 3×3 MaxPool(s=2)      # 4×下采样
        Stage1 : 3 × Bottleneck(64→256),  stride=1                   # 56×56
        Stage2 : 4 × Bottleneck(256→512), stride=2 (首块)            # 28×28
        Stage3 : 6 × Bottleneck(512→1024), stride=2 (首块)           # 14×14
        Stage4 : 3 × Bottleneck(1024→2048), stride=2 (首块)          # 7×7
        Head   : AdaptiveAvgPool → Flatten → FC(2048, num_classes)

    Args:
        num_classes: 分类数，默认为 2（癌 / 非癌）
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # ===== Stem：输入处理 =====
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),   # 224 → 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                     # 112 → 56
        )

        # ===== 四个 Stage =====
        self.layer1 = self.make_layer(64, 64, 3, stride=1)      #  56×56,  64→256
        self.layer2 = self.make_layer(256, 128, 4, stride=2)    #  28×28, 256→512
        self.layer3 = self.make_layer(512, 256, 6, stride=2)    #  14×14, 512→1024
        self.layer4 = self.make_layer(1024, 512, 3, stride=2)   #   7×7, 1024→2048

        # ===== 分类头 =====
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * Bottleneck.expansion, num_classes)  # 2048 → 2

    def make_layer(self, in_channels, mid_channels, blocks, stride=1):
        """
        构建一个 Stage，包含 blocks 个 Bottleneck

        同 Stage 内只有首块可能需要下采样 + 通道变换；
        后续块的输入/输出通道一致，stride=1，shortcut 为恒等映射。

        Args:
            in_channels: 该 Stage 的输入通道数
            mid_channels: Bottleneck 中间通道数
            blocks: 该 Stage 包含的 Bottleneck 数量
            stride: 首块的步长（用于空间下采样）
        """
        out_channels = mid_channels * Bottleneck.expansion
        downsample = None

        # 首块：通道或尺寸变化时需要 shortcut 投影
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(Bottleneck(in_channels, mid_channels, stride, downsample))

        # 后续块：输入通道 = 输出通道，stride = 1
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels, mid_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播

        Args:
            x: (B, 3, H, W) 胃镜图像

        Returns:
            (B, num_classes) 分类 logits
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x
