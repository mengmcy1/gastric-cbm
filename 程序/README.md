# 程序目录说明

代码按研究阶段分为 `模型训练`、`注意力热图`、`MOCE` 和 `SAE` 四条主线。

- `正式代码`：当前实验实际使用的稳定脚本和共享模块。
- `归档`：早期实现、旧基线、一次性审计、旁路验证和demo；仅用于追溯，不作为当前入口。

## 目录结构

```text
程序/
├── 模型训练/
│   └── 正式代码/
├── 注意力热图/
│   └── 正式代码/
├── MOCE/
│   └── 正式代码/
├── SAE/
│   └── 正式代码/
├── 归档/
└── requirements.txt
```

## 当前研究流程

```text
模型训练 → 阈值评估与推理 → Grad-CAM → MOCE → SAE
```

## 当前运行入口

```bash
python 程序/模型训练/正式代码/efficientnet_train_debiased.py --manifest 冻结清单.csv --run-name 运行名
python 程序/模型训练/正式代码/resnet_train_debiased.py --manifest 冻结清单.csv --run-name 运行名
python 程序/模型训练/正式代码/inference.py --model efficientnet_b0 --img 图片路径
python 程序/注意力热图/正式代码/gradcam_batch.py
python 程序/MOCE/正式代码/moce_curated_concept.py --model resnet50 --mode debug
```

各阶段的详细文件用途见对应目录中的 `README.md`。
