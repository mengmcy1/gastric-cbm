"""
胃早癌推理脚本
加载训练好的模型（ResNet50 / EfficientNet-B0），对单张或批量胃镜图像进行癌/非癌预测。

用法：
  python inference.py --img ../数据/某张图片.jpg                                # ResNet50 单张
  python inference.py --model efficientnet_b0 --img ../数据/某张图片.jpg         # EfficientNet 单张
  python inference.py --dir ../数据/某文件夹/                                    # 批量预测
  python inference.py --dir ../数据/某文件夹/ --output result.csv                # 批量 + 导出 CSV

阈值来源：
  tune_threshold.py 在验证集上 Sens≥0.90 约束下的推荐值。
"""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, efficientnet_b0

# ---- 配置 ----
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))

# 模型名 → (权重文件名, Sens≥0.90 最佳阈值)
MODEL_REGISTRY = {
    'resnet50':        ('resnet50_transfer_best.pth',         0.20),
    'efficientnet_b0': ('efficientnet_b0_best.pth',           0.22),
}

# ★ 默认推理图片 —— 不传参数时直接用这个，点 ▶ 按钮就能跑
DATA_DIR   = os.path.join(os.path.dirname(BASE_DIR), '数据', '胃图文带特征标签数据集 3600+ 1933瘤变')
DEFAULT_IMG = os.path.join(DATA_DIR, '01.0000000000296_16_2016-11-09_10_49_17.jpg')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = {0: '非癌', 1: '早癌/瘤变'}


# ---- 模型加载 ----
def load_model(model_name, model_path):
    """根据模型名构建对应架构并恢复权重。"""
    if model_name == 'resnet50':
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == 'efficientnet_b0':
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
    else:
        raise ValueError(f'不支持的模型: {model_name}')

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model


# ---- 图像预处理 ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---- 预测 ----
@torch.no_grad()
def predict_one(model, img_path, threshold):
    """预测单张图像，返回 (癌概率, 预测类别, 类别名)。"""
    image = Image.open(img_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    prob = torch.softmax(logits, dim=1)[0, 1].item()
    pred = 1 if prob >= threshold else 0
    return prob, pred, CLASS_NAMES[pred]


@torch.no_grad()
def predict_batch(model, img_dir, threshold):
    """预测文件夹内所有图像，返回 DataFrame。"""
    results = []
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    files = sorted([
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])

    if not files:
        print(f'未在 {img_dir} 中找到图像文件')
        return None

    for fname in files:
        fpath = os.path.join(img_dir, fname)
        prob, pred, cls_name = predict_one(model, fpath, threshold)
        results.append({
            '图片': fname,
            '癌概率': round(prob, 4),
            '预测类别': pred,
            '预测结果': cls_name,
        })

    return pd.DataFrame(results)


# ---- 主入口 ----
def main():
    parser = argparse.ArgumentParser(description='胃早癌图像推理')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet_b0'],
                        help='模型选择，默认 resnet50')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--img', type=str, help='单张图像路径（默认使用 DEFAULT_IMG）')
    group.add_argument('--dir', type=str, help='图像文件夹路径')
    parser.add_argument('--output', type=str, default=None, help='批量预测结果 CSV 路径')
    parser.add_argument('--threshold', type=float, default=None,
                        help='临时覆盖阈值（不修改注册表中的默认值）')
    args = parser.parse_args()

    weight_file, default_th = MODEL_REGISTRY[args.model]
    model_path = os.path.join(BASE_DIR, '结果', weight_file)
    THRESHOLD = args.threshold if args.threshold is not None else default_th

    # 没传参数 → 用默认单张图片
    if not args.img and not args.dir:
        args.img = DEFAULT_IMG
        print(f'未指定输入，使用默认图片: {args.img}')

    print(f'模型: {args.model}')
    print(f'推理阈值: {THRESHOLD}')
    print(f'设备: {DEVICE}')

    model = load_model(args.model, model_path)

    if args.img:
        prob, pred, cls_name = predict_one(model, args.img, THRESHOLD)
        print(f'图像: {args.img}')
        print(f'癌概率: {prob:.4f}')
        print(f'预测: {cls_name} ({pred})')

    elif args.dir:
        df = predict_batch(model, args.dir, THRESHOLD)
        if df is not None:
            print(df.to_string(index=False))
            if args.output:
                df.to_csv(args.output, index=False, encoding='utf-8-sig')
                print(f'\n结果已保存至: {args.output}')


if __name__ == '__main__':
    main()
