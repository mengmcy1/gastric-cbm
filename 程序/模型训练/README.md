# 模型训练

## 正式代码

- `resnet_train_final.py`：ResNet50 两阶段迁移学习。
- `efficientnet_train.py`：EfficientNet-B0 两阶段迁移学习。
- `train_utils.py`：去偏重训练共享的冻结清单读取、增强、训练和多层级评估。
- `resnet_train_debiased.py`：ResNet50 去偏重训练入口。
- `efficientnet_train_debiased.py`：EfficientNet-B0 去偏重训练入口。
- `summarize_cv_results.py`：合并5折OOF预测，并用每折验证集选择临床阈值。
- `summarize_holdout_results.py`：固定划分实验在验证集选阈值并锁定测试集。
- `compare_cv_models.py`：在相同OOF患者上配对bootstrap比较两个模型。
- `evaluate_legacy_on_preprocessed.py`：旧权重不重训，在旧测试患者上比较原图、v1.1裁剪和限图结果。
- `tune_threshold.py`：验证集阈值扫描和测试集固定阈值评估。
- `inference.py`：单图及批量推理。

旧的 `*_train_final.py` / `efficientnet_train.py` 保留用于复现 2026-07-15 基线；新实验使用 `*_debiased.py`，且必须显式传入冻结清单。

### 固定划分示例

```bash
# 实验 B：松弛 1:1，EfficientNet-B0
python 程序/模型训练/正式代码/efficientnet_train_debiased.py \
  --manifest 数据整理记录/第二批/训练划分_v1/relaxed_all_cases_1to1_split_seed42.csv \
  --run-name expB_relaxed_1to1_efficientnet_seed42

# 实验 C：严格共同支持 5 折中的 fold 0，ResNet50
python 程序/模型训练/正式代码/resnet_train_debiased.py \
  --manifest 数据整理记录/第二批/训练划分_v1/strict_common_support_1to1_folds5_seed42.csv \
  --fold 0 \
  --run-name expC_strict_resnet_fold0_seed42
```

调试时增加 `--debug --debug-units 2 --no-pretrained`；正式训练不要使用这三个参数。完整冻结参数见项目根目录的 `数据训练参数_v1.md`。

## 前期训练

- `resnet.py`：早期手写 ResNet50。
- `resnet_train1.0.py`：第一版训练脚本。
- `resnet_train2.0.py`：第二版训练脚本。

## demo测试

- `test.py`：早期网络结构和训练实验草稿。
