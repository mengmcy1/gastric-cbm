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
`frozen_2026-08-24`。bootstrap须由RP-A整体冻结runner按bundle规则调用。

```bash
python 程序/SAE/正式代码/test_clong_rpa_bootstrap.py
```

14项测试覆盖计划确定性、六层实例数、weighted-vs-explicit-copy等价、
Top-25 occurrence并列、Spearman、spatial、结构失败、400条归并及full-train
self-consistency，并验证coordinator拒绝错误的结构失败reason及超出`[0,1]`的正式指标。
bootstrap不提前冻结spatial的可评价性和数值规则；完整matching worker和正式产物格式须在
关闭spatial/GPU数值口径后再实现。冻结SHA见`rpa_bootstrap_SHA256SUMS.txt`。

## RP-A spatial冻结核心

`rpa_spatial_protocol_v1.json`与`clong_rpa_spatial.py`冻结同图49位置spatial的presence、
NA/0语义、患者平衡聚合及train/bootstrap/val支持度下限。正式数值路径为float32 raw-map
cosine、关闭TF32、float64层级累积，不进行裁剪或舍入。

```bash
python 程序/SAE/正式代码/test_clong_rpa_spatial.py
```

6项测试覆盖`max_p h > ACTIVE_EPS`、单侧active为0、双侧inactive为NA、患者等权和支持度语义。
`audit_clong_rpa_spatial_numeric.py`只允许输出误差、NA一致性、耗时和显存；正式v2审计支持混合
精度路径且未生成任何matching统计。冻结SHA见`rpa_spatial_SHA256SUMS.txt`。

## RP-A GPU identity冻结核心

`rpa_gpu_identity_protocol_v1.json`冻结 residual-preserving 路径在`h'=h`时的五级数值闸门：
patch、attention、pooled、logits和probability分别使用独立`atol/rtol`。固定audit只读取seed42
train缓存的32位患者/85张图，并在两张RTX 5080上得到一致结果。

```bash
python 程序/SAE/正式代码/test_clong_rpa_gpu_identity.py
```

4项测试覆盖误差累计、容差生成公式和逐元素gate。正式任务必须保存五级完整误差，任一级失败
即停止干预解释。冻结SHA见`rpa_gpu_identity_SHA256SUMS.txt`。

## RP-A整体冻结与产物schema

`rpa_overall_protocol_v1.json`、`rpa_coverage_protocol_v1.json`和
`rpa_protocol_bundle_v1.json`关闭最终PASS、activation/energy覆盖、正式输出和失败分流。
overall bundle只包含7个冻结规则JSON，Git与代码版本在run manifest中另记：

```text
protocol_bundle_sha256=768da344bfd3d49ca518528bc043a4eb2ef5b1b4f76b449c42e00c7223093280
```

```bash
python 程序/SAE/正式代码/test_clong_rpa_artifacts.py
```

11项测试覆盖唯一best、RNN一一edge、严格3-clique、400 bootstrap records、六项指标、禁止完整
候选矩阵，以及科学失败/实现失败互斥。规则层面现已允许启动seed43/44；正式runner仍必须按该
schema实现并验收，不能因为开发结果修改bundle。

## RP-A seed43/44 development训练适配

`clong_rpa_train_development.py`复用S2c的Matryoshka训练函数，但将43/44明确限定为
development-calibration字典，不再按S2c八门槛决定产品。正式解释固定使用`K=1024`，并强制
继承seed42的gamma校准JSON及SHA，禁止逐seed重校准。`test_clong_rpa_development.py`覆盖seed
角色、设备、gamma血缘、strict 3-clique和三折六指标归约；CPU隔离debug已跑通。

正式训练启动器只训练字典，不提前执行matching/bootstrap：

```bash
CUDA_DEVICE=<启动前检查后选定的GPU> bash \
  程序/SAE/正式代码/run_clong_rpa_development_training.sh
```

日志位于`结果/SAE/RP_A_Development_20260824/logs/`。该训练脚本仍只供完整runner内部调用，
不应单独启动。

`clong_rpa_prepare_seed.py`把三个字典编码成统一患者/图像顺序的presence、ranking、mass、
active-frequency、energy和49位置激活缓存；`clong_rpa_match_development.py`执行三pair双向
完整搜索，只落盘唯一best、BH+RNN edge、strict anchor和六项full-train指标。CPU缩小字典
debug已跑通，未保存完整候选矩阵。`clong_rpa_bootstrap_worker.py`与coordinator执行400次静态
分工重匹配；`clong_rpa_validate_development.py`在val固定train eligible、strata和经验CDF，使用
冻结的Top-6/空间支持6重建独立graph；`clong_rpa_finalize_development.py`按schema组装成功或科学
失败结果。CPU debug已覆盖完整重匹配、bootstrap归并、val复现和0-anchor科学失败落盘。

正式运行统一使用：

```bash
CUDA_DEVICES=<启动前核实的GPU，可逗号分隔多卡> bash \
  程序/SAE/正式代码/run_clong_rpa_development.sh
```

runner会先打印`nvidia-smi`，训练与分析使用第一张卡，bootstrap按所列GPU静态并行；所有阶段均有
独立实时日志。strict anchors不足100、bootstrap门槛不可行或val复现失败均作为正式科学失败停止；
异常退出另存实现失败现场，且不会生成科学结论。

runner在任何正式计算前由`clong_rpa_provenance.py`独占创建`run_code_snapshot.json`，分别记录
启动时Git commit、冻结protocol bundle SHA和22个显式正式执行文件及直接依赖的逐文件SHA256。续跑与finalizer均
复验同一快照；运行期间Git提交、代码内容或代码清单发生变化都会快速失败。最终`run_manifest.json`
必须同时包含这三类血缘，缺任一项都会被artifact schema拒绝。

## RP-A-lite与RP-B技术分析

`clong_rpa_lite_*`只构建固定`0..99`的train-only探索性B100筛查，不修改或替代正式RP-A
B400。结果明确记录`is_formal_rpa_result=false`，不读取val、internal test或external。

RP-B使用1150个development strict anchors建立统一技术主表：

```bash
bash 程序/SAE/正式代码/run_clong_rpb_technical.sh
```

`rpb_technical_protocol_v1.json`冻结sharedness、标签内来源审计、technical family和代表选择；
`rpb_priority_protocol_v1.json`冻结看图前的多轨技术队列。RP-B不生成加权可信度总分，不把
technical family称为医学概念，也不等待医生完成命名。输出位于
`结果/SAE/RP_B_Technical_20260825/`，只使用train analysis cache。

```bash
python 程序/SAE/正式代码/test_clong_rpb.py
```

测试覆盖五类sharedness共识、BH校正、无边singleton保留、complete-link防链式合并和family
代表词典序。图片、医学命名、轻量干预和完整RP-C干预均不属于该入口。

RP-B v1 sharedness按冻结协议仅在label内bootstrap。以下入口另行执行
`label × source`敏感性复算并生成mixed原因码，输出为`diagnostic_only`，不覆盖
v1五类结果：

```bash
python 程序/SAE/正式代码/diagnose_clong_rpb_sharedness.py
```

## RP-C1 residual-preserving effect screen

`rpc_effect_screen_protocol_v1.json`在查看任何干预结果前冻结了全部1150个
Anchor的三seed `100%→0%` effect screen，以`delta_margin=ablated-original`为主指标。
每张图49个位置的目标激活全部删除，随后重算attention、pooling、logits和概率。
结果先按image聚合到patient，再对患者等权汇总；同时保留全体与active-only效应。

```bash
CUDA_DEVICE=<启动前nvidia-smi确认的GPU> bash \
  程序/SAE/正式代码/run_clong_rpc1.sh
```

入口先对seed42/43/44执行`alpha=1`的patch、attention、pooled、logits和
probability五级identity gate，任一失败都不进入正式screen。RP-C1不生成p值、
BH或PASS/FAIL，只测量和排序。RP-C2的Top 5%总效应、Top 5%类别分离、source-risk
内Top 25%、全部source-sensitivity sentinel、全部cancer-enriched sentinel和20个低效对照
均已在effect结果产生前写入协议，不使用加权总分。

## RP-C2 中间剂量工程 preflight/probe

`rpc2_intervention_protocol_v1.json`冻结五档activation retention、三档中间剂量主统计量、
deterministic matched-reference位置统计和fixed-attention诊断边界。正式干预runner禁止重新
matching，也不把matched tail fraction称为p值。

正式五档全量计算前只运行工程链：

```bash
CUDA_DEVICE=<启动前nvidia-smi确认的GPU> bash \
  程序/SAE/正式代码/run_clong_rpc2_preflight_probe.sh
```

第一阶段对149个study objects的三个seed执行alpha=1五级identity和alpha=0图像、患者、
seed-anchor三层RP-C1精确复现；第二阶段只运行冻结的`a00139/a00987/a00816/a00019`
及其matched references五档路径。两个阶段都明确`scientific_summary=false`，probe曲线不进入
科学摘要，也不得用于修改协议。

preflight/probe通过后，正式五档runner使用：

```bash
CUDA_DEVICE=<启动前nvidia-smi确认的GPU> bash \
  程序/SAE/正式代码/run_clong_rpc2_formal.sh
```

正式runner先用自身五档核心对149对象×3 seed执行alpha=0图像、患者、seed-level逐位回归；
Gate A失败时不会启动controls。通过后按`seed × unique Feature × dose`去重forward，再通过冻结
manifest还原44,424条target-control关系。输出分为unique Feature事实层、447行target-seed证据层和
149行Anchor汇总层；最终`config.json`仅在全部计算、角色计数、SHA和运行前后provenance复验通过后
原子写入。

## RP-D Technical Feature Atlas清单冻结

`rpd_atlas_protocol_v1.json`冻结seed42 canonical visualization、患者级High/Low/Mid/Zero
确定性抽样、Light/Heavy规模、hard negative、source panel、跨seed同图复核和医生盲包隐藏字段。
先只生成清单，不读取或渲染图像：

```bash
python 程序/SAE/正式代码/test_clong_rpd.py
python 程序/SAE/正式代码/build_clong_rpd_manifests.py
```

构建器读取冻结train激活和C-long pooled表示，输出`rpd_anchor_manifest.csv`、
`rpd_heavy_membership_v1.csv`、`rpd_case_manifest.csv`和`rpd_case_shortfalls.csv`。Zero不足按协议
留空；不得跨bin补人。正式渲染器只能使用经`selection_freeze_v1.json`固结SHA的清单，不能在
看图后重新选择病例。val、internal test和external不属于RP-D v1开发输入。
