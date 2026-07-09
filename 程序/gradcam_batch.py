"""
批量生成测试集 Grad-CAM 三联图
==============================
按测试集顺序逐张保存：原图 / Grad-CAM 热图 / 叠加图。

用法：
  1. 在下面“手动配置区”修改 RUN_MODEL 和 DEBUG_N
  2. 直接运行：python gradcam_batch.py
"""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import csv

import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, resnet50

from resnet_train_final import load_matched_dataframe, split_dataframe

try:
    import cv2
except ImportError:
    cv2 = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, '数据', '胃图文带特征标签数据集 3600+ 1933瘤变')
CSV_PATH = os.path.join(PROJECT_DIR, '数据', '胃图文标签表格-添加瘤变标签.csv')
OUTPUT_DIR = os.path.join(PROJECT_DIR, '结果')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = {0: '非癌', 1: '早癌瘤变'}
MODEL_THRESHOLDS = {
    'resnet50': 0.20,
    'efficientnet_b0': 0.22,
}
MODEL_SPECS = {
    'resnet50': {
        'weight_file': 'resnet50_transfer_best.pth',
        'target_layer': lambda model: model.layer4[-1],
    },
    'efficientnet_b0': {
        'weight_file': 'efficientnet_b0_best.pth',
        'target_layer': lambda model: model.features[-1],
    },
}

# ---- 手动配置区 ----
# 一次只跑一个模型；可选：'efficientnet_b0' 或 'resnet50'
RUN_MODEL = 'efficientnet_b0'

# 调试时填数字，例如 5；正式全量生成时改为 None
DEBUG_N = 5


eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_label_font(size):
    font_paths = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


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
        img_tensor = img_tensor.to(DEVICE)
        logits = self.model(img_tensor)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        pred = 1 if prob >= threshold else 0

        self.model.zero_grad()
        logits[0, pred].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy(), prob, pred

    def cleanup(self):
        self._hook_handle_fwd.remove()
        self._hook_handle_bwd.remove()


def make_triptych(img_path, heatmap, prob, pred, true_label, save_path):
    original = Image.open(img_path).convert('RGB')
    original_rgb = np.array(original)
    h, w = original_rgb.shape[:2]

    if cv2 is not None:
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original_rgb, 0.60, heatmap_rgb, 0.40, 0)
    else:
        heatmap_img = Image.fromarray(heatmap)
        heatmap_img = heatmap_img.resize((w, h), resample=Image.Resampling.BICUBIC)
        heatmap_resized = np.array(heatmap_img, dtype=np.float32)
        heatmap_rgb = (
            matplotlib.colormaps['jet'](np.clip(heatmap_resized, 0, 1))[..., :3] * 255
        ).astype(np.uint8)
        overlay = (
            original_rgb.astype(np.float32) * 0.60 +
            heatmap_rgb.astype(np.float32) * 0.40
        ).clip(0, 255).astype(np.uint8)

    panel_h = max(original_rgb.shape[0], heatmap_rgb.shape[0], overlay.shape[0])
    gap = max(12, w // 40)
    label_h = max(54, h // 10)
    canvas = np.full((panel_h + label_h, w * 3 + gap * 2, 3), 255, dtype=np.uint8)
    canvas[label_h:label_h + h, :w] = original_rgb
    canvas[label_h:label_h + h, w + gap:w * 2 + gap] = heatmap_rgb
    canvas[label_h:label_h + h, w * 2 + gap * 2:w * 3 + gap * 2] = overlay

    output = Image.fromarray(canvas)
    draw = ImageDraw.Draw(output)
    font = load_label_font(max(18, min(34, w // 18)))

    labels = [
        f'原图  真实: {CLASS_NAMES[true_label]} ({true_label})',
        'Grad-CAM 热图',
        f'叠加图  预测: {CLASS_NAMES[pred]} ({pred})',
    ]
    x_positions = [0, w + gap, w * 2 + gap * 2]
    for x, label_text in zip(x_positions, labels):
        draw.text((x + 8, max(8, label_h // 4)), label_text, fill=(20, 20, 20), font=font)

    output.save(save_path)


def load_test_dataframe(debug_n):
    df_valid = load_matched_dataframe(CSV_PATH, DATA_DIR)
    _, _, test_df = split_dataframe(df_valid)
    test_df = test_df.reset_index(drop=True)
    if debug_n is not None:
        test_df = test_df.head(debug_n).copy()
    return test_df


def generate_for_model(model_name, test_df):
    threshold = MODEL_THRESHOLDS[model_name]
    out_dir = os.path.join(OUTPUT_DIR, '热图批量', model_name)
    os.makedirs(out_dir, exist_ok=True)

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
            fieldnames=['order', 'image_name', 'true_label', 'pred_label', 'output_file'],
        )
        writer.writeheader()

        for i, row in test_df.iterrows():
            order = i + 1
            image_name = row['图片名字']
            true_label = int(row['瘤变标签'])
            img_path = os.path.join(DATA_DIR, image_name)

            image = Image.open(img_path).convert('RGB')
            img_tensor = eval_transform(image).unsqueeze(0)

            heatmap, prob, pred = gradcam.generate(img_tensor, threshold)

            stem = os.path.splitext(image_name)[0]
            output_name = f'{order:04d}_{stem}_true{true_label}_pred{pred}.png'
            output_path = os.path.join(out_dir, output_name)
            make_triptych(img_path, heatmap, prob, pred, true_label, output_path)

            writer.writerow({
                'order': order,
                'image_name': image_name,
                'true_label': true_label,
                'pred_label': pred,
                'output_file': output_name,
            })
            print(f'[{order:04d}/{len(test_df):04d}] {image_name} -> {output_name}')

    gradcam.cleanup()
    del model
    torch.cuda.empty_cache()
    print(f'完成。清单已保存: {manifest_path}')


def main():
    if RUN_MODEL not in MODEL_SPECS:
        raise ValueError(f'RUN_MODEL 不支持: {RUN_MODEL}')

    print(f'设备: {DEVICE}')
    print(f'手动指定模型: {RUN_MODEL}')
    print(f'DEBUG_N: {DEBUG_N}')

    test_df = load_test_dataframe(DEBUG_N)
    print(f'待生成图片数: {len(test_df)}')

    generate_for_model(RUN_MODEL, test_df)


if __name__ == '__main__':
    main()
