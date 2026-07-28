# 服务器迁移与 GitHub 资产管理

## 管理边界

项目采用三层资产管理：

1. 普通 Git：源码、参数、无患者标识的报告、README 和资产清单。
2. 私有 GitHub Release：全部模型权重及其非敏感配置。
3. 院内受控存储：患者图像、视频、患者级清单、图片级结果和解释图。

GitHub 仓库必须保持私有。胃镜原图、裁剪图、Grad-CAM、MOCE 候选区域、SAE
Top 图片以及可回溯患者的 CSV 均不得上传。

## 当前容量基线

2026-07-28 检查结果：

| 目录 | 占用 |
| --- | ---: |
| 项目总计 | 30.90 GiB |
| `结果` | 21.15 GiB |
| `数据` | 9.18 GiB |
| `.git` | 259 MiB |
| `程序` | 1.4 MiB |

结果归档约 16 GiB，数据归档约 5.9 GiB。服务器迁移时应将当前资产和历史归档分开
传输、分别校验，不要通过 GitHub 搬运。

## 已验证环境

当前环境为 Python 3.12.13，主要版本：

- PyTorch 2.11.0+cu128
- torchvision 0.26.0+cu128
- NumPy 2.4.4
- pandas 3.0.3
- Pillow 12.2.0
- scikit-learn 1.9.0
- OpenCV 5.0.0
- SciPy 1.18.0
- Matplotlib 3.11.0
- Joblib 1.5.3
- openpyxl 3.1.5

新服务器先按 CUDA/驱动版本安装匹配的 PyTorch，再安装
`程序/requirements.txt`。不要直接复制 Conda 环境目录。

## 迁移顺序

1. 提交并推送源码、文档和模型资产清单。
2. 运行 `manage_model_assets.py --package all`。
3. 将三个 `.tar` 上传至私有 GitHub Release，并记录 Release 标签。
4. 在院内网络中加密同步 `数据`、`数据整理记录` 和 `结果`。
5. 新服务器克隆仓库并下载三个权重归档。
6. 在项目根目录解包，恢复权重原相对路径。
7. 运行 `manage_model_assets.py --verify`。
8. 抽取一张脱敏测试图运行 ResNet50、EfficientNet-B0、Grad-CAM。
9. 对数据和结果执行文件数、总容量和 SHA256 抽查。

## 当前注意事项

- Git 历史已有旧权重和旧 MOCE 二进制对象，`.git` 约 259 MiB。Release 上传并验证
  后再单独安排历史清理；历史重写前必须制作裸仓库备份。
- `MOCE聚类数量分析`仍在运行，本轮不打包其生成结果。
- 推理和 Grad-CAM 默认使用去偏实验 A 权重；可通过推理脚本的 `--weights` 临时
  指定其他权重。
- 权重清单不代替离线备份。GitHub 和院内存储至少各保留一份。
