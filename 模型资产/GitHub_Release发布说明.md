# GitHub Release 发布说明

## 发布内容

本项目使用一个私有 Release `model-assets-v1` 保存三个权重归档：

- `runtime_v1.tar`：5 个正式运行权重，约 219 MiB。
- `classifier_reproducibility_v1.tar`：16 个分类器复现权重，约 845 MiB。
- `sae_experiments_v1.tar`：14 个 SAE 实验权重，约 101 MiB。
- `SHA256SUMS`：三个归档的完整性校验值。

归档内共 35 个 `.pth`，原始总大小约 1.14 GiB。每个权重的独立 SHA256 见
`模型资产清单.json`。

## 发布命令

首次使用 GitHub CLI 时先登录：

```bash
gh auth login
```

确认当前仓库为私有仓库后，在项目根目录执行：

```bash
gh release create model-assets-v1 \
  发布资产/模型权重/runtime_v1.tar \
  发布资产/模型权重/classifier_reproducibility_v1.tar \
  发布资产/模型权重/sae_experiments_v1.tar \
  发布资产/模型权重/SHA256SUMS \
  --title "Private model assets v1" \
  --notes "35个不可再训练模型权重；仅限项目授权成员使用。"
```

上传后下载复核：

```bash
mkdir -p /tmp/gastric-cbm-model-assets-check
gh release download model-assets-v1 \
  --dir /tmp/gastric-cbm-model-assets-check
cd /tmp/gastric-cbm-model-assets-check
sha256sum -c SHA256SUMS
```

确认下载校验通过后，才能从普通 Git 历史中移除旧基线 `.pth`。历史重写不属于本次
整理范围，执行前必须单独备份并通知所有仓库使用者。
