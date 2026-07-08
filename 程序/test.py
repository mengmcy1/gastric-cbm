import torch                         # PyTorch 核心库
import torch.nn as nn                # 神经网络模块
import torch.nn.functional as F      # 函数式操作（激活函数、池化等）
import torch.optim as optim          # 优化器
from torch.utils.data import DataLoader, Dataset  # 数据加载
import torchvision.transforms as transforms       # 图像预处理/增强
import torchvision.models as models               # 预训练模型

import numpy as np                   # 数值计算
import os                            # 文件路径操作
from glob import glob                # 批量文件匹配
from PIL import Image                # 图像读取
import matplotlib.pyplot as plt      # 可视化
from tqdm import tqdm                # 进度条
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # 评估指标


class ConvBlock(nn.Module):
    """
    卷积块模块
    
    实现了一个标准的卷积操作块，包含卷积层、批归一化层和ReLU激活函数
    
    Args:
        in_channel (int): 输入通道数
        out_channel (int): 输出通道数
        kernel_size (int): 卷积核大小
        stride (int): 卷积步长
        padding (int): 填充大小
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        # 卷积三件套
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = self.relu(self.bn(self.conv(x)))
        return x
    

class BodyBlock(nn.Module):
    """
    残差块模块
    
    实现了ResNet中的残差连接结构，包含多个卷积层和跳跃连接
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        copy_cnt (int): 卷积层重复次数
        specical_stride (int, optional): 特殊步长，默认为1
    """
    def __init__(self, in_channels, out_channels, copy_cnt, specical_stride=1):
        super().__init__()
        self.copy_cnt = copy_cnt
        # 标准Bottleneck结构中间通道数为输出通道数的1/4
        mid_channels = out_channels // 4
        
        # 第一个残差块的主路径
        self.conv1 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, kernel_size=1, stride=1, padding=0),  # 降维
            ConvBlock(mid_channels, mid_channels, kernel_size=3, stride=specical_stride, padding=1),  # 保持维度
            ConvBlock(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)  # 升维
        )
        
        # 第一个残差块的捷径连接，当输入输出通道不一致时需要调整
        self.conv2 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=specical_stride, padding=0)

        # 后续残差块的主路径
        self.conv3 = nn.Sequential(
            ConvBlock(out_channels, mid_channels, kernel_size=1, stride=1, padding=0),  # 降维
            ConvBlock(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),  # 保持维度
            ConvBlock(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)  # 升维
        )
    def forward(self, x):
        """
        前向传播

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 经过残差连接和多个卷积层处理后的输出
        """
        # 第一个残差块：主路径 + 捷径连接
        x = self.conv1(x) + self.conv2(x)

        # 后续残差块：主路径 + 恒等映射
        for _ in range(self.copy_cnt):
            identity = x
            x = self.conv3(x) + identity
        return x

net = nn.Sequential(
    # head
    nn.Sequential(
        ConvBlock(in_channel=3, out_channel=64, kernel_size=7, stride=2, padding=3),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ),
    # body
    nn.Sequential(
        BodyBlock(in_channels=64, out_channels=256, copy_cnt=3, specical_stride=1),
        BodyBlock(in_channels=256, out_channels=512, copy_cnt=4, specical_stride=2),
        BodyBlock(in_channels=512, out_channels=1024, copy_cnt=6, specical_stride=2),
        BodyBlock(in_channels=1024, out_channels=2048, copy_cnt=3, specical_stride=2)
        ),
    # tail
    nn.Sequential(
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(2048, 1000)
        )
)
