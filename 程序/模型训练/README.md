# 模型训练

## 正式代码

- `train_utils.py`：去偏重训练共享的冻结清单读取、增强、训练和多层级评估。
- `resnet_train_debiased.py`：ResNet50 去偏重训练入口。
- `efficientnet_train_debiased.py`：EfficientNet-B0 去偏重训练入口。
- `summarize_cv_results.py`：合并5折OOF预测，并用每折验证集选择临床阈值。
- `summarize_holdout_results.py`：固定划分实验在验证集选阈值并锁定测试集。
- `compare_cv_models.py`：在相同OOF患者上配对bootstrap比较两个模型。
- `evaluate_debiased_multicenter.py`：v1.1裁剪重训练模型的多中心验证。
- `inference.py`：单图及批量推理。

旧训练、阈值扫描和一次性评估脚本已移动到`程序/归档/模型训练/`；新实验使用
`*_debiased.py`，且必须显式传入冻结清单。

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

调试时增加 `--debug --debug-units 2 --no-pretrained`；正式训练不要使用这三个参数。完整冻结参数见`数据去偏重训练_参数与报告/数据训练参数_v1.md`。

历史复现材料按`旧基线`、`评估审计`和`前期与demo`分类归档，没有删除。
