"""
批量生成全部有效样本 Grad-CAM 三联图
==============================
按第二批整理清单顺序逐张保存全部样本：原图 / Grad-CAM 热图 / 叠加图。

用法：
  1. 在下面“手动配置区”修改 RUN_MODEL 和 DEBUG_N
  2. 直接运行：python gradcam_batch.py
"""

import os
import sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import csv

import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import cv2

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, resnet50

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRAIN_DIR = os.path.join(PROJECT_DIR, '程序', '模型训练', '正式代码')
sys.path.insert(0, TRAIN_DIR)

from inference import MODEL_REGISTRY
from resnet_train_final import load_matched_dataframe


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, '数据', '第二批整理后')
CSV_PATH = os.path.join(DATA_DIR, 'dataset_manifest.csv')
OUTPUT_DIR = os.path.join(PROJECT_DIR, '结果')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = {0: '非癌', 1: '癌/高级别'}
MODEL_SPECS = {
    'resnet50': {
        'weight_file': MODEL_REGISTRY['resnet50'][0],
        'target_layer': lambda model: model.layer4[-1],
    },
    'efficientnet_b0': {
        'weight_file': MODEL_REGISTRY['efficientnet_b0'][0],
        'target_layer': lambda model: model.features[-1],
    },
}

# ---- 手动配置区 ----
# 一次只跑一个模型；可选：'efficientnet_b0' 或 'resnet50'
RUN_MODEL = 'efficientnet_b0'

# 调试时填数字，例如 5；正式全量生成时改为 None
DEBUG_N = None

# 三联图固定版式，保证不同原图的标题大小和位置一致
PANEL_WIDTH = 560
PANEL_HEIGHT = 420
PANEL_GAP = 12
LABEL_HEIGHT = 48
LABEL_FONT_SIZE = 22

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


LABEL_FONT = ImageFont.truetype(
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', size=LABEL_FONT_SIZE,
)


def fit_panel(image_array):
    image = Image.fromarray(image_array).convert('RGB')
    image = ImageOps.contain(image, (PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.LANCZOS)
    panel = Image.new('RGB', (PANEL_WIDTH, PANEL_HEIGHT), color=(0, 0, 0))
    x = (PANEL_WIDTH - image.width) // 2
    y = (PANEL_HEIGHT - image.height) // 2
    panel.paste(image, (x, y))
    return panel


def build_model(model_name):
    if model_name == 'resnet50':
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == 'efficientnet_b0':
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
    else:
        raise ValueError(f'不支持的模型: {model_name}')

    weight_path = os.path.join(
        OUTPUT_DIR, '模型权重', MODEL_SPECS[model_name]['weight_file'],
    )
    checkpoint = torch.load(weight_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._hook_handle_fwd = target_layer.register_forward_hook(self._save_activation)
        self._hook_handle_bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor, threshold):
        self.activations = None
        self.gradients = None

        img_tensor = img_tensor.to(DEVICE)
        img_tensor.requires_grad = True
        logits = self.model(img_tensor)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        pred = 1 if prob >= threshold else 0

        self.model.zero_grad(set_to_none=True)
        logits[0, pred].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)

        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min)

        heatmap = cam.detach().cpu().numpy()
        self.activations = None
        self.gradients = None
        return heatmap, prob, pred

    def cleanup(self):
        self._hook_handle_fwd.remove()
        self._hook_handle_bwd.remove()


def make_triptych(img_path, heatmap, prob, pred, true_label, save_path):
    original = Image.open(img_path).convert('RGB')
    original_rgb = np.array(original)
    h, w = original_rgb.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_rgb, 0.60, heatmap_rgb, 0.40, 0)

    panels = [
        fit_panel(original_rgb),
        fit_panel(heatmap_rgb),
        fit_panel(overlay),
    ]
    canvas_w = PANEL_WIDTH * 3 + PANEL_GAP * 2
    canvas_h = LABEL_HEIGHT + PANEL_HEIGHT
    output = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))

    x_positions = [0, PANEL_WIDTH + PANEL_GAP, (PANEL_WIDTH + PANEL_GAP) * 2]
    for x, panel in zip(x_positions, panels):
        output.paste(panel, (x, LABEL_HEIGHT))

    draw = ImageDraw.Draw(output)
    labels = [
        f'原图  真实: {CLASS_NAMES[true_label]} ({true_label})',
        'Grad-CAM 热图',
        f'叠加图  预测: {CLASS_NAMES[pred]} ({pred})  癌概率: {prob:.3f}',
    ]
    for x, label_text in zip(x_positions, labels):
        draw.text((x + 8, 9), label_text, fill=(20, 20, 20), font=LABEL_FONT)

    output.save(save_path)


def load_output_dataframe(debug_n):
    # 默认生成整理清单中的全部图片。
    df_valid = load_matched_dataframe(CSV_PATH, DATA_DIR).reset_index(drop=True)

    if debug_n is not None:
        df_valid = df_valid.head(debug_n).copy()
    return df_valid


def clear_old_outputs(out_dir):
    for filename in os.listdir(out_dir):
        if filename.endswith('.png') or filename == 'manifest.csv':
            os.remove(os.path.join(out_dir, filename))


def generate_for_model(model_name, df):
    threshold = MODEL_REGISTRY[model_name][1]
    out_dir = os.path.join(OUTPUT_DIR, '热图批量', '第二批', model_name)
    os.makedirs(out_dir, exist_ok=True)
    clear_old_outputs(out_dir)

    print(f'\n===== {model_name} =====')
    print(f'阈值: {threshold}')
    print(f'输出目录: {out_dir}')

    model = build_model(model_name)
    target_layer = MODEL_SPECS[model_name]['target_layer'](model)
    gradcam = GradCAM(model, target_layer)

    manifest_path = os.path.join(out_dir, 'manifest.csv')
    with open(manifest_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['order', 'patient_id', 'image_name', 'true_label', 'cancer_prob', 'pred_label', 'output_file'],
        )
        writer.writeheader()

        for i, row in df.iterrows():
            order = i + 1
            image_name = row['图片名字']
            patient_id = row['patient_id']
            true_label = int(row['瘤变标签'])
            img_path = os.path.join(DATA_DIR, image_name)

            image = Image.open(img_path).convert('RGB')
            img_tensor = eval_transform(image).unsqueeze(0)

            heatmap, prob, pred = gradcam.generate(img_tensor, threshold)

            stem = os.path.splitext(os.path.basename(image_name))[0]
            output_name = f'{order:04d}_{stem}_true{true_label}_pred{pred}.png'
            output_path = os.path.join(out_dir, output_name)
            make_triptych(img_path, heatmap, prob, pred, true_label, output_path)

            writer.writerow({
                'order': order,
                'patient_id': patient_id,
                'image_name': image_name,
                'true_label': true_label,
                'cancer_prob': round(prob, 4),
                'pred_label': pred,
                'output_file': output_name,
            })
            f.flush()
            print(f'[{order:04d}/{len(df):04d}] {image_name}  prob={prob:.3f} -> {output_name}')

    gradcam.cleanup()
    del model
    torch.cuda.empty_cache()
    print(f'完成。清单已保存: {manifest_path}')


def main():
    print(f'设备: {DEVICE}')
    print(f'手动指定模型: {RUN_MODEL}')
    print(f'DEBUG_N: {DEBUG_N}')

    df = load_output_dataframe(DEBUG_N)
    print(f'待生成图片数: {len(df)}')

    generate_for_model(RUN_MODEL, df)


if __name__ == '__main__':
    main()
