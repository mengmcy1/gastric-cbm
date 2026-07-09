"""
Grad-CAM 学习示例脚本
=====================
对单张胃镜图像生成 Grad-CAM 热图，可视化"模型在看哪里"。

Grad-CAM 核心思想（5 步）：
  1. 前向传播 → 拿到目标层的激活图 A（特征图，shape: C×H×W）
  2. 对目标类别计算梯度 → 反向传播到 A，拿到 ∂y/∂A^k （shape: C×H×W）
  3. 对梯度做全局平均池化（GAP）→ 得到每个通道的重要性权重 α^k
  4. 用 α^k 对 A 做加权求和 → 粗粒度热图（shape: H×W）
  5. ReLU 过滤负值 → 上采样到原图尺寸 → 叠加显示

用法：
  python gradcam_demo.py                                    # 默认测试图，两个模型都跑
  python gradcam_demo.py --img ../数据/某图片.jpg             # 指定图片
  python gradcam_demo.py --model resnet50                    # 只看 ResNet50
"""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import argparse
import numpy as np
from PIL import Image
try:
    import cv2
except ImportError:
    cv2 = None
import matplotlib
matplotlib.use('Agg')                  # 无 GUI 环境用 Agg 后端
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# CJK 字体：优先用 Noto Sans CJK SC，回退到 sans-serif
_CJK_FONT = None
for _candidate in ['Noto Sans CJK SC', 'Noto Serif CJK SC', 'WenQuanYi Micro Hei']:
    try:
        _CJK_FONT = FontProperties(family=_candidate)
        # 验证是否能找到：找不到时 get_name() 会返回不同值
        if _candidate in _CJK_FONT.get_name():
            break
    except Exception:
        continue
if _CJK_FONT is None:
    _CJK_FONT = FontProperties()  # 回退，中文会显示为豆腐块

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, efficientnet_b0

# ---- 路径配置 ----
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(os.path.dirname(BASE_DIR), '数据',
                          '胃图文带特征标签数据集 3600+ 1933瘤变')
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), '结果')

# 模型配置：名称 → (权重文件, 目标层获取函数, 架构构建)
MODEL_SPECS = {
    'resnet50': {
        'weight_file': 'resnet50_transfer_best.pth',
        'target_layer': lambda model: model.layer4[-1],   # ResNet 最后一层卷积 → 最高层语义
    },
    'efficientnet_b0': {
        'weight_file': 'efficientnet_b0_best.pth',
        'target_layer': lambda model: model.features[-1], # EfficientNet 最后一组 MBConv
    },
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = {0: '非癌', 1: '早癌/瘤变'}

# 预处理：与训练和推理完全一致
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---- 模型加载 ----
def build_model(model_name):
    spec = MODEL_SPECS[model_name]
    weight_path = os.path.join(OUTPUT_DIR, '模型权重', spec['weight_file'])

    if model_name == 'resnet50':
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == 'efficientnet_b0':
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)

    checkpoint = torch.load(weight_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model


# ============================================================
#  Grad-CAM 核心
# ============================================================

class GradCAM:
    """
    Grad-CAM：梯度加权类激活映射。

    原理（逐行注释见 forward 方法）：

      热图 = ReLU( Σ_k α^k · A^k )

    其中：
      - A^k 是目标层第 k 个通道的激活图（前向传播时捕获）
      - α^k = GAP(∂y_c/∂A^k)，即梯度在空间维度上取平均
      - y_c 是目标类别的 logit（未经过 softmax）
      - ReLU 只保留"对该类别有正向贡献"的区域

    为什么用 ReLU？
      负数区域 = 降低该类别分数的区域 → 不是模型关注该类别的原因，排除掉。

    为什么对梯度做 GAP？
      已经在 ϕ(CAM) 论文中证明，这个权重等价于"该通道对类别 c 的整体贡献"。
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        # 钩子存储
        self.activations = None     # 目标层的激活图 A，前向时填充
        self.gradients  = None      # 激活图的梯度 ∂y/∂A，反向时填充

        # 注册钩子
        self._hook_handle_fwd = target_layer.register_forward_hook(self._save_activation)
        self._hook_handle_bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """前向钩子：保存目标层的输出激活图。"""
        self.activations = output.detach()   # detach 断开计算图，只存值

    def _save_gradient(self, module, grad_input, grad_output):
        """反向钩子：保存目标层输出的梯度 ∂y/∂A。"""
        self.gradients = grad_output[0].detach()

    def _remove_hooks(self):
        self._hook_handle_fwd.remove()
        self._hook_handle_bwd.remove()

    @torch.no_grad()
    def predict(self, img_tensor):
        """纯推理：返回癌概率和类别，不触发梯度计算。"""
        logits = self.model(img_tensor.to(DEVICE))
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        pred = 1 if prob >= 0.5 else 0
        return prob, pred, CLASS_NAMES[pred]

    def forward(self, img_tensor, target_class=None):
        """
        核心方法：对一张图像生成 Grad-CAM 热图。

        Args:
            img_tensor: shape (1, 3, 224, 224)，已归一化
            target_class: 要对哪个类别生成热图。
                          None → 自动选模型预测的类别
                          0    → 对"非癌"生成热图（看模型为什么不认为是癌）
                          1    → 对"早癌"生成热图（看模型关注哪些癌相关区域）

        Returns:
            heatmap:   np.ndarray, shape (224, 224)，取值 [0, 1]
            prob:      癌概率
            pred:      预测类别 (0/1)
        """
        # 1. 前向传播
        #    钩子 _save_activation 会自动捕获目标层输出 → self.activations
        img_tensor = img_tensor.to(DEVICE)
        img_tensor.requires_grad = True   # ★ 输入需要梯度，否则中间层梯度流不过去
        logits = self.model(img_tensor)
        prob = torch.softmax(logits, dim=1)[0, 1].item()

        pred_class = logits.argmax(dim=1).item()
        if target_class is None:
            target_class = pred_class

        # 2. 反向传播 → 计算梯度
        #    对目标类别的 logit 反向求导
        #    钩子 _save_gradient 会自动捕获 → self.gradients = ∂y_target/∂A
        self.model.zero_grad()
        logits[0, target_class].backward()

        # 3. 计算每个通道的权重 α^k = GAP(∂y/∂A^k)
        #    self.gradients shape: (1, C, H', W')
        #    在 H' 和 W' 维度上取平均 → (1, C, 1, 1) → (C,)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

        # 4. 加权求和 + ReLU
        #    (1,C,1,1) * (1,C,H',W') → sum over C → (H', W')
        cam = (weights * self.activations).sum(dim=1).squeeze(0)  # (H', W')
        cam = torch.relu(cam)                                      # ReLU: 只保留正向贡献

        # 5. 归一化到 [0, 1]（避免除以 0）
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        heatmap = cam.cpu().numpy()
        return heatmap, prob, pred_class

    def cleanup(self):
        """移除钩子（用完后释放）。"""
        self._remove_hooks()


# ---- 可视化 ----
def visualize(img_path, heatmap, prob, pred_class, model_name, save_path):
    """
    生成三列对比图：原图 / 热图 / 叠加图
    """
    # 读取原图（显示用，不做归一化），保留原始分辨率，避免输出被压到 224x224。
    original_pil = Image.open(img_path).convert('RGB')
    original_rgb = np.array(original_pil)
    h, w = original_rgb.shape[:2]

    # Grad-CAM 原始分辨率通常很低，例如 7x7；先高质量上采样到原图尺寸。
    if cv2 is not None:
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
        heatmap_img = heatmap_img.resize((w, h), resample=Image.Resampling.BICUBIC)
        heatmap_resized = np.array(heatmap_img).astype(np.float32) / 255.0

    heatmap_resized = np.clip(heatmap_resized, 0, 1)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # 优先使用 OpenCV applyColorMap；没有 cv2 时用 matplotlib jet 色图兜底。
    if cv2 is not None:
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original_rgb, 0.60, heatmap_rgb, 0.40, 0)
    else:
        heatmap_rgb = (plt.get_cmap('jet')(heatmap_resized)[..., :3] * 255).astype(np.uint8)
        overlay = (original_rgb.astype(np.float32) * 0.60 +
                   heatmap_rgb.astype(np.float32) * 0.40).clip(0, 255).astype(np.uint8)

    # 同时保存单独热图和叠加图，方便后续给医生单独查看。
    root, ext = os.path.splitext(save_path)
    heatmap_path = f'{root}_heatmap{ext}'
    overlay_path = f'{root}_overlay{ext}'
    Image.fromarray(heatmap_rgb).save(heatmap_path)
    Image.fromarray(overlay).save(overlay_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_rgb)
    axes[0].set_title('原图', fontproperties=_CJK_FONT, fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(heatmap_rgb)
    axes[1].set_title('Grad-CAM 热图', fontproperties=_CJK_FONT, fontsize=13)
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(f'叠加图\n预测: {CLASS_NAMES[pred_class]} (癌概率 {prob:.3f})',
                      fontproperties=_CJK_FONT, fontsize=13)
    axes[2].axis('off')

    suptitle = f'{model_name.upper()}  —  Grad-CAM'
    fig.suptitle(suptitle, fontproperties=_CJK_FONT, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'  → 已保存: {save_path}')
    print(f'  → 单独热图: {heatmap_path}')
    print(f'  → 单独叠加: {overlay_path}')


# ---- 主入口 ----
def main():
    parser = argparse.ArgumentParser(description='Grad-CAM 学习示例')
    parser.add_argument('--img', type=str, default=None,
                        help='图像路径（默认用推理脚本的测试图）')
    parser.add_argument('--model', type=str, default='all',
                        choices=['resnet50', 'efficientnet_b0', 'all'],
                        help='模型选择，默认两个都跑')
    args = parser.parse_args()

    # 默认测试图
    if args.img is None:
        args.img = os.path.join(
            DATA_DIR, '01.0000000000296_16_2016-11-09_10_49_17.jpg'
        )
        print(f'未指定图片，使用默认测试图: {os.path.basename(args.img)}')
    else:
        print(f'图片: {args.img}')

    if not os.path.exists(args.img):
        print(f'❌ 文件不存在: {args.img}')
        return

    print(f'设备: {DEVICE}\n')

    # 读图像 → tensor
    image = Image.open(args.img).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)    # (1, 3, 224, 224)

    model_names = ['resnet50', 'efficientnet_b0'] if args.model == 'all' else [args.model]

    for name in model_names:
        print(f'--- {name.upper()} ---')

        # 构建模型
        spec = MODEL_SPECS[name]
        model = build_model(name)

        # 获取目标层
        target_layer = spec['target_layer'](model)
        print(f'  目标层: {target_layer.__class__.__name__}')

        # 创建 Grad-CAM
        gradcam = GradCAM(model, target_layer)

        # 先推理看预测结果
        prob, pred, cls_name = gradcam.predict(img_tensor)
        print(f'  Softmax 推理 → 癌概率: {prob:.4f}, 预测: {cls_name}')

        # 生成热图（对预测类别）
        heatmap, prob2, pred2 = gradcam.forward(img_tensor)
        print(f'  Grad-CAM 热图 → 激活范围: [{heatmap.min():.4f}, {heatmap.max():.4f}]')

        # 可视化保存
        basename = os.path.splitext(os.path.basename(args.img))[0]
        save_path = os.path.join(OUTPUT_DIR, '热图', f'gradcam_demo_{name}_{basename}.png')
        visualize(args.img, heatmap, prob2, pred2, name, save_path)

        # 清理钩子
        gradcam.cleanup()
        del model
        torch.cuda.empty_cache()
        print()


if __name__ == '__main__':
    main()
