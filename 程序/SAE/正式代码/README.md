# 正式代码

`sae_discovery.py` 使用冻结的 ResNet50 提取 GAP 2048 维整图特征，按患者平衡训练
ReLU + L1 稀疏自编码器，并保存重构指标、feature 患者级统计、Top 图片和 encoder
激活方向概念热图。分类贡献仍由 decoder 方向与 ResNet50 分类头共同计算。

当前默认输入已切换为实验A去偏重训练 ResNet50：使用`第二批裁剪后_v1_1`、实验A冻结
train/val患者划分、`resnet50_debiased_best.pth`，以及图片/患者锁定阈值
0.154842/0.384940。旧第二批原图SAE结果保留在`结果/SAE/第二批/resnet50/`，不会覆盖。

小规模流程验证：

```bash
python 程序/SAE/正式代码/sae_discovery.py --demo
```

正式训练默认只使用固定划分中的训练集和验证集。当前正式方案已经锁定为
`lambda_l1=5e-4`：

```bash
python 程序/SAE/正式代码/sae_discovery.py \
  --experiment formal_expA_resnet50_sae_l1_0005_20260724
```

正式默认参数参考 M-CBM 的 ISIC2018/ResNet50 配置：输入 2048 维、隐藏 512 维、
`lambda_l1=5e-4`、学习率 `1e-4`、SAE batch size 32、最多 1000 epochs、早停
patience 50。该 L1 权重已根据验证集 L0、dead feature、重构和分类恢复率锁定。
损失函数使用逐元素MSE（对齐M-CBM官方`L2ReconstructionLoss`），
decoder 每次更新前移除沿自身方向的梯度分量再归一化（梯度正交投影）。参数均可
通过命令行覆盖；训练日志分别保存 MSE、原始 L1、`lambda_l1 × L1` 和 L0。

SAE 训练后按训练集激活患者数进行 feature 筛选：默认 feature 至少激活 5 位
训练患者，并在验证集上选择 recovered cross-entropy 下降不超过 0.01 的最严格
阈值。筛选决策、阈值曲线和剪枝后指标保存在 `feature筛选`，概览图只生成保留的
feature。可用 `--pruning-min-active-patients` 和 `--pruning-ce-tolerance` 覆盖默认值。
如最低患者门槛本身已超出容差，脚本仍保留至少 5 位患者激活的 feature，同时在
`selection_within_tolerance` 和终端警告中标记需要人工复核。

只有在 SAE 结构和参数已经锁定后，才使用 `--include-test` 投影测试集。脚本不训练
Concept Bottleneck Layer；CBL 需要医生确认后的概念级 0/1 标签。

锁定 SAE 后使用 `sae_project_analyze.py` 做独立投影。该脚本只加载已有 checkpoint
和 feature 筛选结果，不重新训练 SAE，并从训练run的`config.json`恢复数据、权重及
图片/患者双阈值。先以 `--split val` 复核既有指标，通过后再以`--split test`一次性
投影测试集。输出同时保留原模型、完整SAE和筛选后SAE的图像
及患者概率，并分别保存原模型 FP/FN、`TP_to_FN`、`TN_to_FP`、`FP_to_TN`、
`FN_to_TP` 病例和 feature 排名。

验证集复核通过后，可将同一个冻结 SAE 投影到外部多中心完整测试集。该过程不会
重训练 SAE，不会根据外部标签修改剪枝掩码或阈值：

```bash
python 程序/SAE/正式代码/sae_project_analyze.py \
  --sae-run 结果/SAE/去偏重训练_v1/resnet50/formal_expA_resnet50_sae_l1_0005_20260724 \
  --external-manifest 结果/去偏重训练_v1/外部多中心完整测试_v1/preprocess_manifest.csv \
  --experiment external_multicenter_resnet50_sae_20260727
```

投影完成后使用`analyze_sae_results.py`生成不含checkpoint和大体量特征缓存的医学生
提交版。提交版包含Word阅读指南、核心指标、训练期候选feature图、独立投影概念图、
错误病例清单、概念命名表和必要追溯数据：

```bash
python 程序/SAE/正式代码/analyze_sae_results.py \
  --sae-run 结果/SAE/去偏重训练_v1/resnet50/formal_expA_resnet50_sae_l1_0005_20260724 \
  --projection-run 结果/SAE/去偏重训练_v1/resnet50/external_multicenter_resnet50_sae_20260728
```

`sae_image_centered.py`按单张图片展示激活最高的Top-K保留feature。每张图输出原图、
多颜色叠加图、各feature独立热图和统计表；统计表同时区分激活强度和癌方向贡献：

```bash
CUDA_VISIBLE_DEVICES=1 python 程序/SAE/正式代码/sae_image_centered.py \
  --sae-run 结果/SAE/去偏重训练_v1/resnet50/formal_expA_resnet50_sae_l1_0005_20260724 \
  --image /绝对路径/示例图.jpg \
  --top-k 5 \
  --rank-by activation \
  --color-mode contribution
```

默认贡献配色为正向促癌高饱和红色、负向抑癌高饱和蓝色，绝对贡献越大颜色越明显。
其叠加逻辑参考Grad-CAM：`--overlay-alpha`控制热点处的最大叠加强度（默认0.55），
`--response-gamma`控制中等空间响应的显色程度（默认0.75，数值越小越显眼）。若需要
让每个feature始终保持独立类别色，可改为`--color-mode feature`。

批量调试可传入manifest，默认仅处理前5张；添加`--all`才处理筛选后的全部图片：

```bash
CUDA_VISIBLE_DEVICES=1 python 程序/SAE/正式代码/sae_image_centered.py \
  --sae-run 结果/SAE/去偏重训练_v1/resnet50/formal_expA_resnet50_sae_l1_0005_20260724 \
  --manifest 结果/去偏重训练_v1/外部多中心完整测试_v1/preprocess_manifest.csv \
  --patient-id 患者ID --limit 3
```

`mcbm_cbl_train.py` 是概念标注完成后的备用脚本：读取 SAE 实验缓存的 GAP 特征，
用医生共识标签训练线性 CBL，再用全部训练图片的癌/非癌标签训练 elastic-net 稀疏
分类器。默认在 `NCC95<=5` 的候选中按验证 AUC 选择，并保存、打印全部 C 候选的
AUC、NCC95 和非零权重数；可用 `--target-ncc` 修改目标。标注格式见
`文档/SAE/SAE概念标注规范.md`。在真实标注完成前不要运行该脚本。

真实标注完成后的命令格式：

```bash
python 程序/SAE/正式代码/mcbm_cbl_train.py \
  --sae-run 结果/SAE/去偏重训练_v1/resnet50/正式实验目录 \
  --concept-catalog 结果/SAE概念标注/去偏重训练_v1/resnet50/v1/concept_catalog.csv \
  --annotations 结果/SAE概念标注/去偏重训练_v1/resnet50/v1/annotations_consensus.csv
```
