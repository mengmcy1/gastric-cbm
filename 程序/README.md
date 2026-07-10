# 程序目录说明

代码按研究阶段分为 `模型训练`、`注意力热图`、`MOCE` 和 `SAE` 四条主线。每条主线内部统一使用以下目录：

- `正式代码`：当前实验实际使用的稳定脚本。
- `前期训练`：早期实现、预实验和参数探索代码。
- `demo测试`：小规模演示、单图测试和功能验证代码。

## 目录结构

```text
程序/
├── 模型训练/
│   ├── 正式代码/
│   ├── 前期训练/
│   └── demo测试/
├── 注意力热图/
│   ├── 正式代码/
│   ├── 前期训练/
│   └── demo测试/
├── MOCE/
│   ├── 正式代码/
│   ├── 前期训练/
│   └── demo测试/
├── SAE/
│   ├── 正式代码/
│   ├── 前期训练/
│   └── demo测试/
└── requirements.txt
```

## 当前研究流程

```text
模型训练 → 阈值评估与推理 → Grad-CAM → MOCE → SAE
```

## 当前运行入口

```bash
python 程序/模型训练/正式代码/efficientnet_train.py
python 程序/模型训练/正式代码/resnet_train_final.py
python 程序/模型训练/正式代码/tune_threshold.py --model efficientnet_b0
python 程序/模型训练/正式代码/inference.py --model efficientnet_b0 --img 图片路径
python 程序/注意力热图/正式代码/gradcam_batch.py
python 程序/MOCE/demo测试/moce_single_demo.py
```

各阶段的详细文件用途见对应目录中的 `README.md`。
