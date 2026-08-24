# 正式代码

> 归档边界：本目录的`sae_discovery.py`、`efficientnet_sae_discovery.py`等旧脚本用于复现
> 既有ResNet50和EfficientNet-B0 SAE结果，并提供新路线可复用的实现参考。2026-08-19启动的
> C-long文献重构路线正式入口为`clong_sae_discovery.py`（矩阵`run_clong_sae_matrix.sh`，
> 汇总`summarize_clong_sae_matrix.py`，测试`test_clong_sae.py`），协议见根目录
> `SAE实验进度与结果讨论.md`（S2-S3已于2026-08-19冻结预注册）。不得用旧命令生成或覆盖
> C-long正式结果。

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

锁定SAE完成内部test和外部投影后，使用`sae_cross_dataset_stability.py`对齐train、
val、internal test和external test中的同一Feature。候选名单只由train/val筛选；
test和external仅用于确认方向、覆盖和Top患者稳定性：

```bash
python 程序/SAE/正式代码/sae_cross_dataset_stability.py
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

## C-long S2b结构重构

`clong_s2b_discovery.py`不覆盖上一轮pooled SAE，它在独立目录中执行
2026-08-20冻结的S2b协议：

- S4-B：pooled `1280 -> 10240 -> 1280` BatchTopK，目标mean L0=1024；
- S4-C：共享patch字典 `49 x 1280 -> 10240 -> 49 x 1280` Top-K，每位置K=128；
- S4-D：同一patch字典的BatchTopK，目标每位置mean L0=128；
- S4-N：pooled逐样本归一化Top-K，仅用于mean/norm旁路诊断。

patch正式评价会从重构后的7x7特征图重算attention、pooled表示和分类
概率；固定原attention只是诊断口径。BatchTopK冻结checkpoint后仅用train
估计全局阈值，val不重新估计。完整seed42矩阵由以下脚本串行执行：

```bash
CUDA_DEVICE=<启动前核实的空闲GPU> bash \
  程序/SAE/正式代码/run_clong_s2b_matrix.sh
```

脚本固定顺序为S4-B、`gamma_pool` train-only校准、S4-C、S4-D、S4-N，
全部成功后才调用`summarize_clong_s2b.py`判定门槛和选择唯一patch臂。
`test_clong_s2b.py`覆盖稀疏预算、阈值并列、完整替换、空间指标、
正式预算锁和决胜链。

## C-long S2c Matryoshka patch SAE

`clong_s2c_matryoshka.py`实现2026-08-20冻结的S2c协议：在同一个
`1280 -> 10240 -> 1280`共享patch字典中联合学习
`K={64,128,256,512,1024}`五层嵌套Top-K。`clong_s2c_core.py`保存单排序
嵌套结构、五层并集死亡口径、gamma公式和最小合格K选择等纯数学逻辑；
`test_clong_s2c.py`覆盖这些防线及FVU诊断、gamma初始化SHA和跨seed冻结K交接。

正式seed42固定按“train-only gamma_pool校准 -> 五层联合训练 -> 逐K完整替换评价
-> 八门槛最小K选择”运行：

```bash
CUDA_DEVICE=<启动前核实的空闲GPU> bash \
  程序/SAE/正式代码/run_clong_s2c.sh
```

脚本日志写入`结果/SAE/CLong_S2c_Matryoshka_20260821/logs/`。FVU与
explained variance只作诊断，不参与checkpoint、K选择或成败判定。seed42没有
合格K时正式停止；seed202/503只有在seed42选出唯一K后才能运行，并且只能复现该
冻结K，不得重新选择其他K。

## RP-A eligible train-only校准

`clong_rpa_eligible.py`仅读取S2c seed42字典与train空间缓存，不训练
SAE、不读val/test/external。`audit`阶段生成患者聚合矩阵、逐Feature
审计CSV和`q* -> P_min`血缘；`calibrate`阶段复用这些SHA绑定产物
执行400次label×source患者分半，冻结`A_min`。

长时正式任务应在启动前检查GPU，并由服务器后台调用：

```bash
CUDA_DEVICE=<启动前核实的GPU> bash \
  程序/SAE/正式代码/run_clong_rpa_eligible.sh
```

静态方法定义在`rpa_eligible_protocol_v1.json`；正式产物保存在
`结果/SAE/RP_A_Eligible校准_20260821/`。若`audit`已完成而后半阶段中断，
仅可以`STAGE=calibrate`复用已校验SHA的聚合缓存。

2026-08-21正式train-only任务已完成，`SHA256SUMS.txt` 9/9通过。冻结结果为
`q=0.02`、`P_min_train=25`、`val Top-q=6`、
`A_min=0.25/49=0.00510204081632653`。上述值不得在后续seed重估。

## RP-A matching纯计算benchmark

`benchmark_clong_rpa_matching.py`仅用seed42冻结SAE构造3个Feature双射逻辑seed，
执行一次正式规模的3 pair / 6 direction matching和3折pseudo-confirm plumbing。
它不训练新seed、不运行bootstrap循环，也不持久化edge、anchor、p值或任何
稳定性指标。候选矩阵使用source block流式归约，批量中秩和best选择已通过与
冻结null/FDR纯函数的逐行等价测试。

2026-08-24最终benchmark使用`source_block=128`/`target_block=256`，完整replicate为
696.2秒，峰值RAM 3.35 GiB、VRAM 6.20 GiB。该数字只用于bootstrap工程预算，
不是RP-A统计结果。

## RP-A bootstrap冻结纯函数

`rpa_bootstrap_protocol_v1.json`与`clong_rpa_bootstrap.py`定义了6个
`label x source` strata内有放回患者抽样、multiplicity加权、重复患者
Top-25、`ACTIVE_EPS=1e-8`、静态worker分配和六门槛归并。当前协议状态为
`frozen_2026-08-24`。这只表示bootstrap子协议已冻结；RP-A整体仍未冻结，仍不得启动新seed
或正式bootstrap。

```bash
python 程序/SAE/正式代码/test_clong_rpa_bootstrap.py
```

14项测试覆盖计划确定性、六层实例数、weighted-vs-explicit-copy等价、
Top-25 occurrence并列、Spearman、spatial、结构失败、400条归并及full-train
self-consistency，并验证coordinator拒绝错误的结构失败reason及超出`[0,1]`的正式指标。
bootstrap不提前冻结spatial的可评价性和数值规则；完整matching worker和正式产物格式须在
关闭spatial/GPU数值口径后再实现。冻结SHA见`rpa_bootstrap_SHA256SUMS.txt`。
