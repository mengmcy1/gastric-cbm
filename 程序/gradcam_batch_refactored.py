"""按标签表顺序批量生成原图、Grad-CAM 热图和叠加图。"""

import csv
import glob
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

from matplotlib import colormaps
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import torch

from inference_refactored import CLASS_NAMES, EVAL_TRANSFORM, MODEL_REGISTRY, load_model
from train_utils import DATA_DIR, OUTPUT_DIR, load_matched_dataframe


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_LAYERS = {
    'resnet50': lambda model: model.layer4[-1],
    'efficientnet_b0': lambda model: model.features[-1],
}

# 手动配置：每次运行一个模型；DEBUG_N=None 表示生成全部样本。
RUN_MODEL = 'resnet50'
DEBUG_N = None

PANEL_WIDTH = 560
PANEL_HEIGHT = 420
PANEL_GAP = 12
LABEL_HEIGHT = 48
LABEL_FONT = ImageFont.truetype(
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', size=22,
)


def fit_panel(image_array):
    image = ImageOps.contain(
        Image.fromarray(image_array).convert('RGB'),
        (PANEL_WIDTH, PANEL_HEIGHT),
        Image.Resampling.LANCZOS,
    )
    panel = Image.new('RGB', (PANEL_WIDTH, PANEL_HEIGHT))
    position = ((PANEL_WIDTH - image.width) // 2, (PANEL_HEIGHT - image.height) // 2)
    panel.paste(image, position)
    return panel


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.hooks = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_inputs, grad_outputs):
        self.gradients = grad_outputs[0].detach()

    def generate(self, image_tensor, threshold):
        logits = self.model(image_tensor.to(DEVICE))
        probability = torch.softmax(logits, dim=1)[0, 1].item()
        prediction = int(probability >= threshold)

        self.model.zero_grad(set_to_none=True)
        logits[0, prediction].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1).squeeze(0))
        cam = cam / cam.max().clamp_min(1e-8)
        return cam.cpu().numpy(), probability, prediction

    def close(self):
        for hook in self.hooks:
            hook.remove()


def colorize_heatmap(heatmap, size):
    resized = Image.fromarray(heatmap).resize(size, Image.Resampling.BICUBIC)
    normalized = np.clip(np.asarray(resized, dtype=np.float32), 0, 1)
    return (colormaps['jet'](normalized)[..., :3] * 255).astype(np.uint8)


def make_triptych(image_path, heatmap, probability, prediction, true_label, save_path):
    original = np.asarray(Image.open(image_path).convert('RGB'))
    height, width = original.shape[:2]
    colored_heatmap = colorize_heatmap(heatmap, (width, height))
    overlay = (
        original.astype(np.float32) * 0.6 + colored_heatmap.astype(np.float32) * 0.4
    ).astype(np.uint8)

    panels = map(fit_panel, [original, colored_heatmap, overlay])
    x_positions = [0, PANEL_WIDTH + PANEL_GAP, 2 * (PANEL_WIDTH + PANEL_GAP)]
    canvas = Image.new(
        'RGB',
        (PANEL_WIDTH * 3 + PANEL_GAP * 2, LABEL_HEIGHT + PANEL_HEIGHT),
        'white',
    )
    for x, panel in zip(x_positions, panels):
        canvas.paste(panel, (x, LABEL_HEIGHT))

    labels = [
        f'原图  真实: {CLASS_NAMES[true_label]} ({true_label})',
        'Grad-CAM 热图',
        f'叠加图  预测: {CLASS_NAMES[prediction]} ({prediction})  癌概率: {probability:.3f}',
    ]
    draw = ImageDraw.Draw(canvas)
    for x, label in zip(x_positions, labels):
        draw.text((x + 8, 9), label, fill=(20, 20, 20), font=LABEL_FONT)
    canvas.save(save_path)


def clear_old_outputs(output_dir):
    for path in glob.glob(os.path.join(output_dir, '*.png')):
        os.remove(path)
    manifest_path = os.path.join(output_dir, 'manifest.csv')
    if os.path.exists(manifest_path):
        os.remove(manifest_path)


def generate_for_model(model_name, dataframe):
    threshold = MODEL_REGISTRY[model_name]['threshold']
    output_dir = os.path.join(OUTPUT_DIR, '热图批量', model_name)
    os.makedirs(output_dir, exist_ok=True)
    clear_old_outputs(output_dir)

    print(f'\n===== {model_name} =====')
    print(f'阈值: {threshold}')
    print(f'输出目录: {output_dir}')

    model = load_model(model_name)
    gradcam = GradCAM(model, TARGET_LAYERS[model_name](model))
    manifest_path = os.path.join(output_dir, 'manifest.csv')
    fields = ['order', 'image_name', 'true_label', 'cancer_prob', 'pred_label', 'output_file']

    with open(manifest_path, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for order, row in enumerate(dataframe.itertuples(index=False), 1):
            image_name = getattr(row, '图片名字')
            true_label = int(getattr(row, '瘤变标签'))
            image_path = os.path.join(DATA_DIR, image_name)
            image = Image.open(image_path).convert('RGB')
            image_tensor = EVAL_TRANSFORM(image).unsqueeze(0)
            heatmap, probability, prediction = gradcam.generate(image_tensor, threshold)

            stem = os.path.splitext(image_name)[0]
            output_name = f'{order:04d}_{stem}_true{true_label}_pred{prediction}.png'
            make_triptych(
                image_path,
                heatmap,
                probability,
                prediction,
                true_label,
                os.path.join(output_dir, output_name),
            )
            writer.writerow(dict(zip(fields, [
                order,
                image_name,
                true_label,
                round(probability, 4),
                prediction,
                output_name,
            ])))
            print(
                f'[{order:04d}/{len(dataframe):04d}] {image_name}  '
                f'prob={probability:.3f} -> {output_name}'
            )

    gradcam.close()
    print(f'完成。清单已保存: {manifest_path}')


def main():
    dataframe = load_matched_dataframe().reset_index(drop=True)
    if DEBUG_N is not None:
        dataframe = dataframe.head(DEBUG_N)

    print(f'设备: {DEVICE}')
    print(f'模型: {RUN_MODEL}')
    print(f'待生成图片数: {len(dataframe)}')
    generate_for_model(RUN_MODEL, dataframe)


if __name__ == '__main__':
    main()
