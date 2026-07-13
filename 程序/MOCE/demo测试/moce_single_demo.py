"""MOCE 单图演示：从模型激活中提取、去重并评估候选概念区域。"""

import csv
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import torch
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))  # 项目根目录
TRAIN_DIR = os.path.join(PROJECT_DIR, '程序', '模型训练', '正式代码')  # 推理代码目录
sys.path.insert(0, TRAIN_DIR)

from inference import DEVICE, MODEL_REGISTRY, load_model


DATA_DIR = os.path.join(PROJECT_DIR, '数据', '胃图文带特征标签数据集 3600+ 1933瘤变')  # 图片目录
OUTPUT_DIR = os.path.join(PROJECT_DIR, '结果', 'MOCE单图演示')  # 演示结果目录

# 手动配置区：选用一张 EfficientNet 高置信度早癌图片。
MODEL_NAME = 'efficientnet_b0'  # 使用的分类模型
IMAGE_NAME = '01.0000000179724.0039.1615258257.jpg'  # 待解释图片
TARGET_CLASS = 1   # 要解释的类别：0=非癌，1=早癌/瘤变

# MOCE 论文默认设置：保留前 50% 通道，每张激活图取 top 10% 区域。
KEEP_CHANNEL_RATIO = 0.5    # 保留重要性最高的 50% 通道
GAMMA = 0.10                # 每个通道仅保留激活最高的 10% 区域
MIN_AREA_RATIO = 0.005      # 候选区域最小面积占比
MAX_JACCARD = 0.5           # 掩码重叠超过该值时视为重复
DISPLAY_PARTS = 8           # 演示图展示的候选区域数量

TARGET_LAYERS = {  # 不同模型用于提取高层语义特征的目标层
    'resnet50': lambda model: model.layer4[-1],
    'efficientnet_b0': lambda model: model.features[-1],
}
CLASS_NAMES = {0: '非癌', 1: '早癌/瘤变'}  # 类别编号对应名称
FONT = ImageFont.truetype(
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', size=22,
)
SMALL_FONT = ImageFont.truetype(
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', size=17,
)

MODEL_TRANSFORM = transforms.Compose([  # 原图和移除区域图的预处理
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

CONCEPT_TRANSFORM = transforms.Compose([  # 已缩放候选区域的预处理
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class ActivationCapture:
    """通过前向 Hook 保存目标层激活及其梯度。"""

    def __init__(self, layer):
        """注册目标层，layer 是需要观察的卷积层。"""
        self.output = None
        self.handle = layer.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        """每次前向传播时保存该层输出。"""
        self.output = output
        if output.requires_grad:
            output.retain_grad()

    def close(self):
        """移除 Hook，结束中间特征捕获。"""
        self.handle.remove()


def largest_component(mask):
    """从二值掩码中只保留面积最大的连通区域。"""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component = 1 + np.argmax(stats[1:count, cv2.CC_STAT_AREA])
    return (labels == component).astype(np.uint8)


def jaccard(mask_a, mask_b):
    """计算两个掩码的交并比，用于删除重叠候选。"""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / union


def crop_and_resize(original, mask):
    """裁剪掩码区域，保持宽高比放入 224×224 画布。"""
    ys, xs = np.where(mask)
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1

    masked = original * mask[..., None]
    crop = Image.fromarray(masked[y1:y2, x1:x2])
    crop = ImageOps.contain(crop, (224, 224), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (224, 224))
    position = ((224 - crop.width) // 2, (224 - crop.height) // 2)
    canvas.paste(crop, position)
    return canvas, tuple(map(int, (x1, y1, x2, y2)))


def extract_candidate_masks(model, capture, image, target_class):
    """根据目标类别梯度提取并去重候选概念掩码。"""
    tensor = MODEL_TRANSFORM(Image.fromarray(image)).unsqueeze(0).to(DEVICE)
    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    probability = torch.softmax(logits, dim=1)[0, target_class].item()         #只对早癌类别计算概率
    logits[0, target_class].backward()                                          

    activations = capture.output[0].detach()        #激活程度
    gradients = capture.output.grad[0].detach()     #梯度信息
    channel_scores = torch.relu(
        gradients.mean(dim=(1, 2)) * activations.mean(dim=(1, 2))
    )

    keep_count = round(len(channel_scores) * KEEP_CHANNEL_RATIO)                 #只保留最重要的50%的通道数量1280/2个
    channel_order = torch.argsort(channel_scores, descending=True)[:keep_count]
    height, width = image.shape[:2]
    candidates = []

    for channel in channel_order.tolist():
        activation = activations[channel].cpu().numpy()
        activation = cv2.resize(activation, (width, height), interpolation=cv2.INTER_CUBIC)  
        cutoff = np.quantile(activation, 1 - GAMMA)                      #90% 分位 = top 10%
        mask = largest_component((activation >= cutoff).astype(np.uint8))#生成掩码
        area_ratio = mask.mean()

        if area_ratio < MIN_AREA_RATIO:  #保留面积大于0.5%的候选区域
            continue
        if any(jaccard(mask, item['mask']) >= MAX_JACCARD for item in candidates): #交并比大于0.5的重复区域跳过
            continue

        candidates.append({
            'channel': channel,
            'channel_score': channel_scores[channel].item(),
            'area_ratio': area_ratio,
            'mask': mask,
        })

    return probability, candidates


@torch.no_grad()
def encode_and_evaluate(model, capture, original, candidates, target_class):
    """编码候选区域，并计算仅保留和移除后的类别概率。"""
    concept_images = []
    removed_images = []

    for item in candidates:
        concept, bbox = crop_and_resize(original, item['mask'])
        removed = original.copy()
        removed[item['mask'].astype(bool)] = 0
        item['concept_image'] = concept
        item['bbox'] = bbox
        concept_images.append(CONCEPT_TRANSFORM(concept))
        removed_images.append(MODEL_TRANSFORM(Image.fromarray(removed)))

    concept_batch = torch.stack(concept_images).to(DEVICE)
    concept_logits = model(concept_batch)
    concept_vectors = capture.output.mean(dim=(2, 3)).detach().cpu().numpy()
    keep_probabilities = torch.softmax(concept_logits, dim=1)[:, target_class].cpu().numpy()

    removed_batch = torch.stack(removed_images).to(DEVICE)
    removed_probabilities = torch.softmax(model(removed_batch), dim=1)[:, target_class]
    removed_probabilities = removed_probabilities.cpu().numpy()

    for index, item in enumerate(candidates):
        item['feature_vector'] = concept_vectors[index]
        item['keep_probability'] = float(keep_probabilities[index])
        item['removed_probability'] = float(removed_probabilities[index])


def mask_overlay(original, mask, bbox):
    """在原图上用红色覆盖掩码并绘制黄色边框。"""
    overlay = original.copy()
    color = np.zeros_like(original)
    color[..., 0] = 255
    overlay[mask.astype(bool)] = (
        original[mask.astype(bool)] * 0.55 + color[mask.astype(bool)] * 0.45
    ).astype(np.uint8)
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline=(255, 255, 0), width=4)
    return image


def fit_panel(image, size=(300, 230)):
    """保持图片比例，将其居中放入固定大小的展示面板。"""
    image = ImageOps.contain(image.convert('RGB'), size, Image.Resampling.LANCZOS)
    panel = Image.new('RGB', size)
    panel.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return panel


def save_visualization(original, candidates, full_probability, output_path):
    """将排名靠前的候选区域保存为三列对比图。"""
    shown = candidates[:DISPLAY_PARTS]
    panel_width, panel_height = 300, 230
    row_height = 285
    canvas = Image.new(
        'RGB',
        (panel_width * 3, 70 + row_height * len(shown)),
        'white',
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 10),
        f'{MODEL_NAME}  目标类别: {CLASS_NAMES[TARGET_CLASS]}  原图概率: {full_probability:.4f}',
        fill=(20, 20, 20),
        font=FONT,
    )
    draw.text(
        (12, 40),
        '左：候选区域　中：仅保留区域　右：移除区域后的目标类别概率',
        fill=(60, 60, 60),
        font=SMALL_FONT,
    )

    for rank, item in enumerate(shown, 1):
        y = 70 + (rank - 1) * row_height
        highlighted = mask_overlay(original, item['mask'], item['bbox'])
        removed = original.copy()
        removed[item['mask'].astype(bool)] = 0
        panels = [
            fit_panel(highlighted),
            fit_panel(item['concept_image']),
            fit_panel(Image.fromarray(removed)),
        ]
        for column, panel in enumerate(panels):
            canvas.paste(panel, (column * panel_width, y))

        drop = full_probability - item['removed_probability']
        text = (
            f'#{rank} ch={item["channel"]}  面积={item["area_ratio"]:.1%}  '
            f'仅保留={item["keep_probability"]:.3f}  '
            f'移除后={item["removed_probability"]:.3f}  下降={drop:+.3f}'
        )
        draw.text((10, y + panel_height + 8), text, fill=(20, 20, 20), font=SMALL_FONT)

    canvas.save(output_path)


def save_results(candidates, full_probability, output_dir):
    """保存候选指标 CSV 和用于后续聚类的特征向量。"""
    csv_path = os.path.join(output_dir, 'candidate_concepts.csv')
    fields = [
        'rank', 'channel', 'channel_score', 'area_ratio', 'bbox',
        'full_probability', 'keep_probability', 'removed_probability', 'probability_drop',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for rank, item in enumerate(candidates, 1):
            writer.writerow({
                'rank': rank,
                'channel': item['channel'],
                'channel_score': round(item['channel_score'], 8),
                'area_ratio': round(float(item['area_ratio']), 6),
                'bbox': item['bbox'],
                'full_probability': round(full_probability, 6),
                'keep_probability': round(item['keep_probability'], 6),
                'removed_probability': round(item['removed_probability'], 6),
                'probability_drop': round(full_probability - item['removed_probability'], 6),
            })

    np.savez_compressed(
        os.path.join(output_dir, 'candidate_features.npz'),
        features=np.stack([item['feature_vector'] for item in candidates]),
        channels=np.array([item['channel'] for item in candidates]),
    )


def main():
    """加载模型和图片，依次执行提取、评价与结果保存。"""
    image_path = os.path.join(DATA_DIR, IMAGE_NAME)
    weight_file = MODEL_REGISTRY[MODEL_NAME][0]
    weight_path = os.path.join(PROJECT_DIR, '结果', '模型权重', weight_file)
    output_dir = os.path.join(OUTPUT_DIR, MODEL_NAME, os.path.splitext(IMAGE_NAME)[0])
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(MODEL_NAME, weight_path)
    capture = ActivationCapture(TARGET_LAYERS[MODEL_NAME](model))
    original = np.asarray(Image.open(image_path).convert('RGB'))

    full_probability, candidates = extract_candidate_masks(
        model, capture, original, TARGET_CLASS,
    )
    encode_and_evaluate(model, capture, original, candidates, TARGET_CLASS)
    candidates.sort(
        key=lambda item: full_probability - item['removed_probability'],
        reverse=True,
    )

    save_visualization(
        original,
        candidates,
        full_probability,
        os.path.join(output_dir, 'moce_single_demo.png'),
    )
    save_results(candidates, full_probability, output_dir)
    capture.close()

    prediction = int(full_probability >= MODEL_REGISTRY[MODEL_NAME][1])
    print(f'模型: {MODEL_NAME}')
    print(f'图片: {IMAGE_NAME}')
    print(f'目标类别概率: {full_probability:.4f}')
    print(f'阈值预测: {CLASS_NAMES[prediction]} ({prediction})')
    print(f'去重后候选区域: {len(candidates)}')
    print(f'输出目录: {output_dir}')


if __name__ == '__main__':
    main()
