# 正式代码

`sae_discovery.py` 使用冻结的 ResNet50 提取 GAP 2048 维整图特征，按患者平衡训练
ReLU + L1 稀疏自编码器，并保存重构指标、feature 患者级统计、Top 图片和 decoder
方向概念热图。

小规模流程验证：

```bash
python 程序/SAE/正式代码/sae_discovery.py --demo
```

正式训练默认只使用固定划分中的训练集和验证集：

```bash
python 程序/SAE/正式代码/sae_discovery.py
```

正式默认参数参考 M-CBM 的 ISIC2018/ResNet50 配置：输入 2048 维、隐藏 512 维、
`lambda_l1=5e-4`、学习率 `1e-4`、SAE batch size 32、最多 1000 epochs、早停
patience 50。损失函数使用逐元素 MSE（对齐 M-CBM 官方 `L2ReconstructionLoss`），
decoder 每次更新前移除沿自身方向的梯度分量再归一化（梯度正交投影）。参数均可
通过命令行覆盖，正式定稿前仍需在验证集比较重构、L0、dead feature 和分类恢复率。

SAE 训练后按训练集激活患者数进行 feature 筛选：默认 feature 至少激活 5 位
训练患者，并在验证集上选择 recovered cross-entropy 下降不超过 0.01 的最严格
阈值。筛选决策、阈值曲线和剪枝后指标保存在 `feature筛选`，概览图只生成保留的
feature。可用 `--pruning-min-active-patients` 和 `--pruning-ce-tolerance` 覆盖默认值。
如最低患者门槛本身已超出容差，脚本仍保留至少 5 位患者激活的 feature，同时在
`selection_within_tolerance` 和终端警告中标记需要人工复核。

只有在 SAE 结构和参数已经锁定后，才使用 `--include-test` 投影测试集。脚本不训练
Concept Bottleneck Layer；CBL 需要医生确认后的概念级 0/1 标签。

锁定 SAE 后使用 `sae_project_analyze.py` 做独立投影。该脚本只加载已有 checkpoint
和 feature 筛选结果，不重新训练 SAE；先以 `--split val` 复核既有指标，通过后再以
`--split test` 一次性投影测试集。输出同时保留原模型、完整 SAE 和筛选后 SAE 的图像
及患者概率，并分别保存原模型 FP/FN、`TP_to_FN`、`TN_to_FP`、`FP_to_TN`、
`FN_to_TP` 病例和 feature 排名。

`mcbm_cbl_train.py` 是概念标注完成后的备用脚本：读取 SAE 实验缓存的 GAP 特征，
用医生共识标签训练线性 CBL，再用全部训练图片的癌/非癌标签训练 elastic-net 稀疏
分类器。默认在 `NCC95<=5` 的候选中按验证 AUC 选择，并保存、打印全部 C 候选的
AUC、NCC95 和非零权重数；可用 `--target-ncc` 修改目标。标注格式见
`文档/SAE/SAE概念标注规范.md`。在真实标注完成前不要运行该脚本。

真实标注完成后的命令格式：

```bash
python 程序/SAE/正式代码/mcbm_cbl_train.py \
  --sae-run 结果/SAE/第二批/resnet50/正式实验目录 \
  --concept-catalog 结果/SAE概念标注/第二批/resnet50/v1/concept_catalog.csv \
  --annotations 结果/SAE概念标注/第二批/resnet50/v1/annotations_consensus.csv
```
