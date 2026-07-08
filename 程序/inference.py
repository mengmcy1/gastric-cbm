"""
胃早癌推理脚本
加载训练好的 ResNet50 模型，对单张或批量胃镜图像进行癌/非癌预测。

用法：
  python inference.py --img ../数据/某张图片.jpg                  # 单张预测
  python inference.py --dir ../数据/某文件夹/                       # 批量预测
  python inference.py --dir ../数据/某文件夹/ --output result.csv   # 批量 + 导出 CSV

阈值来源：
  tune_threshold.py 在验证集上调优后的推荐值。
  修改下面的 THRESHOLD 变量即可。
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50

# ---- 配置 ----
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '结果', 'resnet50_transfer_best.pth')

# ★ 部署阈值 —— 跑完 tune_threshold.py 后填入推荐值
# 当前默认 0.5，调优后改为推荐值（如 0.36）
THRESHOLD = 0.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = {0: '非癌', 1: '早癌/瘤变'}


# ---- 模型加载 ----
def load_model(model_path):
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
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
def predict_one(model, img_path):
    """预测单张图像，返回 (癌概率, 预测类别, 类别名)。"""
    image = Image.open(img_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    prob = torch.softmax(logits, dim=1)[0, 1].item()
    pred = 1 if prob >= THRESHOLD else 0
    return prob, pred, CLASS_NAMES[pred]


@torch.no_grad()
def predict_batch(model, img_dir):
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
        prob, pred, cls_name = predict_one(model, fpath)
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--img', type=str, help='单张图像路径')
    group.add_argument('--dir', type=str, help='图像文件夹路径')
    parser.add_argument('--output', type=str, default=None, help='批量预测结果 CSV 路径')
    parser.add_argument('--threshold', type=float, default=None,
                        help='临时覆盖阈值（不修改脚本默认值）')
    args = parser.parse_args()

    th = args.threshold if args.threshold is not None else THRESHOLD
    print(f'推理阈值: {th}')
    print(f'设备: {DEVICE}')

    model = load_model(MODEL_PATH)

    if args.img:
        prob, pred, cls_name = predict_one(model, args.img)
        # 临时覆盖阈值用于单张
        if args.threshold is not None:
            pred = 1 if prob >= args.threshold else 0
            cls_name = CLASS_NAMES[pred]
        print(f'图像: {args.img}')
        print(f'癌概率: {prob:.4f}')
        print(f'预测: {cls_name} ({pred})')

    elif args.dir:
        df = predict_batch(model, args.dir)
        if df is not None:
            # 如果用临时阈值覆盖，重算
            if args.threshold is not None:
                df['预测类别'] = (df['癌概率'] >= args.threshold).astype(int)
                df['预测结果'] = df['预测类别'].map(CLASS_NAMES)
            print(df.to_string(index=False))
            if args.output:
                df.to_csv(args.output, index=False, encoding='utf-8-sig')
                print(f'\n结果已保存至: {args.output}')


if __name__ == '__main__':
    main()
