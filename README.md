# gastric-cbm

基于白光胃镜图像的胃早癌分类与可解释性研究项目。当前主线包括去偏分类器、
Grad-CAM、MOCE 概念聚类和 SAE 概念发现。

## 目录

- `程序/`：模型训练、推理、Grad-CAM、MOCE 和 SAE 正式代码。
- `数据整理脚本/`：预处理、裁剪、审计和数据集构建脚本。
- `数据去偏重训练_参数与报告/`：冻结参数、实验报告和项目进度。
- `文档/`：标注规范、迁移说明和历史方法文档。
- `模型资产/`：私有 GitHub Release 权重清单及恢复说明。
- `数据/`、`数据整理记录/`、`结果/`：服务器本地受控资产，不随普通 Git 提交。

## 环境

```bash
python -m pip install -r 程序/requirements.txt
```

PyTorch 和 torchvision 建议根据新服务器的 CUDA 版本使用官方安装命令单独安装。
当前已验证环境版本记录在
`文档/服务器迁移与GitHub资产管理.md`。

## 快速推理

```bash
python 程序/模型训练/正式代码/inference.py \
  --model efficientnet_b0 \
  --img /path/to/image.jpg
```

默认加载去偏重训练实验 A 的正式权重。权重不进入普通 Git 历史，从私有 GitHub
Release 下载并恢复原相对路径后即可运行。

## 数据安全

患者图像、视频、患者级清单、图片级预测、特征缓存和包含胃镜画面的解释结果禁止
上传 GitHub。仓库应保持私有；模型权重也只通过私有 Release 分发。
