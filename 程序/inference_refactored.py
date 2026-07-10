"""加载 ResNet50 或 EfficientNet-B0，对单张图片或文件夹执行推理。"""

import argparse
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, resnet50


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
WEIGHTS_DIR = os.path.join(PROJECT_DIR, '结果', '模型权重')
DATA_DIR = os.path.join(PROJECT_DIR, '数据', '胃图文带特征标签数据集 3600+ 1933瘤变')
DEFAULT_IMG = os.path.join(DATA_DIR, '01.0000000000296_16_2016-11-09_10_49_17.jpg')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = {0: '非癌', 1: '早癌/瘤变'}
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_resnet50():
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def build_efficientnet_b0():
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model


MODEL_REGISTRY = {
    'resnet50': {
        'builder': build_resnet50,
        'weight_file': 'resnet50_transfer_best.pth',
        'threshold': 0.20,
    },
    'efficientnet_b0': {
        'builder': build_efficientnet_b0,
        'weight_file': 'efficientnet_b0_best.pth',
        'threshold': 0.22,
    },
}


def load_model(model_name, model_path=None):
    """构建模型并恢复权重。"""
    spec = MODEL_REGISTRY[model_name]
    model_path = model_path or os.path.join(WEIGHTS_DIR, spec['weight_file'])
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = spec['builder']()
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict_one(model, image_path, threshold):
    image = Image.open(image_path).convert('RGB')
    tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    probability = torch.softmax(model(tensor), dim=1)[0, 1].item()
    prediction = int(probability >= threshold)
    return probability, prediction, CLASS_NAMES[prediction]


def predict_batch(model, image_dir, threshold):
    filenames = sorted(
        name for name in os.listdir(image_dir)
        if os.path.splitext(name)[1].lower() in VALID_EXTENSIONS
    )
    rows = []
    for filename in filenames:
        probability, prediction, class_name = predict_one(
            model, os.path.join(image_dir, filename), threshold,
        )
        rows.append({
            '图片': filename,
            '癌概率': round(probability, 4),
            '预测类别': prediction,
            '预测结果': class_name,
        })
    return pd.DataFrame(rows, columns=['图片', '癌概率', '预测类别', '预测结果'])


def main():
    parser = argparse.ArgumentParser(description='胃早癌图像推理')
    parser.add_argument('--model', choices=MODEL_REGISTRY, default='resnet50')
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--img', help='单张图像路径')
    inputs.add_argument('--dir', help='图像文件夹路径')
    parser.add_argument('--output', help='批量预测结果 CSV 路径')
    parser.add_argument('--threshold', type=float, help='临时覆盖默认阈值')
    args = parser.parse_args()

    spec = MODEL_REGISTRY[args.model]
    threshold = spec['threshold'] if args.threshold is None else args.threshold
    image_path = args.img or (DEFAULT_IMG if args.dir is None else None)

    print(f'模型: {args.model}')
    print(f'推理阈值: {threshold}')
    print(f'设备: {DEVICE}')
    model = load_model(args.model)

    if image_path:
        probability, prediction, class_name = predict_one(model, image_path, threshold)
        print(f'图像: {image_path}')
        print(f'癌概率: {probability:.4f}')
        print(f'预测: {class_name} ({prediction})')
    else:
        dataframe = predict_batch(model, args.dir, threshold)
        print(dataframe.to_string(index=False))
        if args.output:
            dataframe.to_csv(args.output, index=False, encoding='utf-8-sig')
            print(f'\n结果已保存至: {args.output}')


if __name__ == '__main__':
    main()
