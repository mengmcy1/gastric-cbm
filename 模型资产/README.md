# 模型资产

患者图像不能进入 GitHub，且现有模型难以重新训练，因此全部 `.pth` 权重都必须作为
不可再生研究资产保留。权重通过**私有 GitHub Release**发布，不进入普通 Git 历史。

## Release 分组

- `runtime_v1`：当前正式 ResNet50、EfficientNet-B0、SAE，以及仍被历史推理引用的
  两份旧基线权重。
- `classifier_reproducibility_v1`：A0、B、C 五折、D 等分类器复现实验权重。
- `sae_experiments_v1`：其余 SAE 参数扫描、调试和历史权重。

`模型资产清单.json` 和 `模型资产清单.csv` 记录每个权重的原相对路径、大小、
SHA256、Release 分组和用途。新服务器下载 Release 归档后，在项目根目录执行：

```bash
tar -xf runtime_v1.tar
tar -xf classifier_reproducibility_v1.tar
tar -xf sae_experiments_v1.tar
python 程序/项目维护/正式代码/manage_model_assets.py --verify
```

上传前应确认 GitHub 仓库和 Release 均为私有，并保留至少一份院内离线备份。
