# gastric-cbm SAE实验进度与结果讨论

## 新对话恢复项目的固定提示词

> 该提示词只规定恢复上下文与协作方法；即时状态以本文后续章节和正式输出为准。

```text
请继续协助我开展 gastric-cbm 的文献驱动 SAE 解释研究。

先确认项目根目录为 /home/mcy/gastric-cbm，并按顺序阅读：

1. AGENTS.md：长期背景、数据边界、目录、文件安全、Git和注释规则；
2. SAE实验进度与结果讨论.md：当前协议、即时状态、结果、阻塞项和下一步；
3. MAGE实验进度与结果讨论.md中C-long的模型血缘、注意力QC和外部描述性投影；
4. 先读文献/SAE可能相关/README_文献分级索引.md，再按任务读取对应分级目录，另读
   文献/2_M-CBM_DeSantis_ICLR2026.pdf和文献/3_ProtoMIL_Sun_MICCAI2025.pdf；
5. 本轮相关正式代码、冻结manifest、缓存、checkpoint和SHA。数字以正式产物为准。

开始操作前先只读检查git status、目标输出目录和运行状态。GPU任务先运行nvidia-smi，
不默认GPU编号。不得覆盖旧SAE、MAGE、分类器、internal test或external正式产物。

当前SAE路线解释的是冻结C-long学生实际送入分类头的attention-pooled 1280维向量，
不是普通GAP，也不是事后Grad-CAM。SAE不得修改C-long参数、BN统计、注意力图或预测。

实验原则：
- SAE只用train拟合，val选择epoch、字典宽度、稀疏度和剪枝；
- C-long只有seed42正式模型。复现指在同一冻结特征上独立训练SAE seed42/202/503，
  不能写成分类器三种子复现；
- 默认训练目标包含无标签分类margin保真项，不能只靠事后评价筛选分类敏感方向；
- internal test和external只能在方案锁定后描述性投影，不得反向选参；
- Feature是模型内部方向，不自动等于医学概念；必须结合跨患者Top图、空间响应、
  分类贡献、干预、来源关联、癌/非癌共享性和临床审核；
- 癌与非癌使用同一联合SAE字典。不得先各自提取后把共有特征误当重复噪声删除；
- 先self-test和debug，再正式矩阵；长任务同时提供启动命令和实时日志命令。

每轮按“做了什么、结果如何、能确认什么、不能确认什么、风险、下一步、产物”
详细且易懂地汇报，并及时更新本文。
```

## 文档边界与旧路线归档

本文于2026-08-19重构，开始维护“文献重新推导的C-long SAE路线”。重构前完整文档已
冻结归档为：

`文档/归档/SAE第二批/SAE实验进度与结果讨论_旧路线冻结_20260819.md`

旧A0/A1/A1b/A1c/B/C实验、脚本、结果和医学生材料均原地保留，只读追溯，不删除、不覆盖、
不移动，以免破坏README、命令和结果血缘。旧实现索引见：

`程序/归档/SAE/旧路线索引_20260819.md`

旧路线最重要的可复用内部证据是：纯特征MSE无法稳定保护分类margin；A1c加入无标签margin
保真后，患者阈值一致率由约0.9385提高到合格线以上。该证据只用于确定新路线默认包含
margin项，不把旧字典宽度、lambda或Feature选择直接继承为新路线最终答案。

## 当前状态

更新时间：2026-08-21

| 项目 | 状态 | 当前结论或阻塞项 |
| --- | --- | --- |
| 文献调研 | 四轮完成 | 已覆盖InterPLM、ProtoMIL、M-CBM、PatchSAE、Histoscope及病理SAE等工作；第四轮（2026-08-21）专门核验S2c后Feature复现、家族、癌/非癌共享性、医生盲审、语义反证与因果干预，详见“S2c后正式解释阶段规划”及本地文献核验笔记 |
| 旧SAE路线 | 已冻结归档 | 文档快照已保存；旧代码与结果原地只读保留 |
| 新解释对象 | S0已冻结 | C-long attention-pooled 1280维表示；checkpoint、manifest、教师、缓存、beta共6项SHA全部核验一致，结构与阈值已写死 |
| 新SAE结构 | S2c已正式结束，无正式产品 | seed42于2026-08-21完成；五个K均通过其余7项门槛，但患者冻结阈值一致率为0.9308–0.9423，未达到0.95；按预注册停止严格重构型SAE路线，不运行seed202/503 |
| RP-SAE新解释范式 | RP-A主体未冻结；eligible、null/FDR、两类质量覆盖、bootstrap与spatial子协议已冻结 | eligible已冻结`q=0.02`、`P_min_train=25`、`val Top-q=6`、`A_min=0.25/49`；null/FDR、activation、representation energy与bootstrap均于2026-08-24正式冻结；spatial的presence、NA/0、支持度和混合精度规则经6项测试、固定小块数值审计及SHA固结后于2026-08-24正式冻结。其余未决项关闭前仍禁止运行43/44/202/503/911 |
| 复现口径 | RP-A草案已分工，未冻结 | 只有一个C-long seed42；所有SAE seed使用同一冻结特征。42为开发，43/44仅作开发校准，202/503为2/2正式确认，911仅在后续协议冻结后作留出初始化复现 |
| 癌/非癌联合分析 | 已列为正式任务 | 同一字典内分析共有、癌富集、非癌富集、混合及重复概念家族 |
| internal test/external | 锁定 | 新SAE开发不得读取；规则冻结后仅作一次描述性投影 |
| 新路线代码 | 已实现，两轮debug验收通过 | `程序/SAE/正式代码/clong_sae_discovery.py` + 矩阵脚本 + 汇总器；输出根目录`结果/SAE/CLong文献重构_20260819/`；14项单元测试通过。审阅后加固：正式预算（lr/epoch/patience/warmup/batch/剪枝容差）逐项锁死、实验名限17个、debug强制隔离到`debug/`、缓存六文件SHA+shape+行顺序核验、S0交叉绑定补齐v3_audit与beta JSON、S3扩展指标（margin/双阈值/患者偏移/密度直方图）、汇总JSON禁止NaN |
| 正式矩阵 | 已运行完成（2026-08-20），no_formal_product | 17/17组完成；L1合格0/15，Top-K备选亦未过全部硬门槛；按预注册停止规则本阶段无正式产品，未追加任何超参数。主要卡点：val mean cosine最高仅0.8814（Top-K），未达0.90。详见"S2-S3正式矩阵结果"节 |
| S2b结构重构 | 已正式预注册，2026-08-20冻结 | 文献驱动比较pooled BatchTopK与patch级Top-K/BatchTopK；逐样本归一化仅作固定诊断，不参与产品选择；宽度/K、损失、阈值估计、门槛、选择、复现与停止规则均已冻结；test/internal test/external继续锁定 |
| S2b代码 | 已实现并完成debug验收 | 独立核心模块、正式入口、四臂矩阵脚本、自动汇总器和18项回归测试已落盘；CPU debug的B/C/D/N四臂均端到端跑通；启动器和汇总器已支持将BatchTopK阈值不可实现记为正式协议失败，不阻断独立实验臂 |
| S2b正式矩阵 | 已完成（2026-08-20），`no_patch_product_stop_s2b` | B/C/D/N四臂与自动汇总全部完成；S4-B因train正预激活数不足而协议失败；S4-C/D均通过8项门槛中的6项，同时未达到患者预测一致率`>=0.95`和pooled cosine`>=0.90`；无唯一patch正式产品，按预注册停止，不进入SAE seed202/503复现，不追加K/宽度/归一化/门槛调参 |
| S2b失败诊断 | v2已完成，2026-08-20 | C/D正式指标均以`1e-6`容差复现；两臂各翻转17/260位患者且多数位于冻结阈值附近；完整替换翻转主要由内容重构驱动；正式pooled cosine短板已确认在标签、来源、分辨率和画中画等分层中广泛存在，而非单一亚组崩溃 |
| S2c Matryoshka patch SAE | 已正式结束（2026-08-21），无正式产品 | seed42五个K均通过其余7项门槛，但患者冻结阈值一致率为0.9308–0.9423，未达到0.95；状态为`no_product_stop_s2c`，不运行原S2c的seed202/503复现，不把任何K交给医生正式命名 |

## 第一轮文献结论

### InterPLM

论文使用SAE分解蛋白语言模型多个层级的表示，重点贡献不是单一超参数，而是完整证据链：

- 对多个模型层分别训练SAE，比较不同层的概念含量；
- 使用8倍或32倍过完备字典，并对学习率和L1做训练早期warmup；
- 按高、中、低和零激活样本展示Feature，而不是只展示Top图；
- 将Feature激活归一化后进行跨Feature比较，并聚类decoder方向寻找概念家族；
- 在独立样本上验证自动描述能否预测Feature激活；
- 修改Feature激活并保留原重构误差，观察模型输出是否按预期改变；
- 使用随机化模型对照，区分模型所学知识与输入数据本身规律。

对本项目的主要启发是：必须增加空间定位、困难负例、跨样本验证和残差保留干预；8倍或
32倍扩维不直接照搬，因胃镜训练规模和临床审核能力远小于论文数据规模。

### ProtoMIL

ProtoMIL在冻结的512维病理patch特征上训练`512 -> 2048 -> 512` SAE，稀疏机制为
ReLU+L1。单个patch平均只激活约10个单元（约0.49%），2048个潜在单元中整个数据集上
曾激活的仅74个。医生通过每个Feature的Top patch
识别病理结构和墨水、失焦、组织边界等伪特征，再以SAE激活训练透明MIL分类器。

其可解释预测可拆成“局部概念激活 × 区域注意力 × 类别权重”。对确认的伪特征置零后重训，
分类性能大体保持，但AUC没有稳定提高。因此可支持“减少伪特征依赖并维持性能”，不能解释为
干预必然提高准确率。

对本项目的主要启发是：字典容量、实际激活Feature数和最终临床概念数必须分开报告；应同时
给医生看空间位置、原型图和分类贡献；去除器械/反光Feature必须通过干预后性能验证。

### M-CBM

M-CBM从黑箱骨干自身表示中提取SAE候选，再通过高激活与困难负例命名、部分概念存在性标注、
Concept Bottleneck Layer和稀疏线性分类器构造透明模型。其关键方法包括：

- 按删除Feature后原分类交叉熵恢复下降不超过约1%选择低密度剪枝阈值；
- 每个概念同时提供激活例、随机负例和外观相似的困难负例；
- 概念名只是假设，须再标注概念是否存在，不能直接把SAE单元名称当真值；
- 概念标注与类别无关，避免“癌图只标癌概念”造成标签泄漏；
- 用NCC衡量一次决策解释95%绝对贡献需要多少个概念。

最接近本项目的ISIC2018实验把2048维骨干特征压到512维字典，平均L0约17，最终保留73个
概念，仍恢复约99.6%的原分类交叉熵表现。这说明医学图像SAE不必默认扩维，字典宽度应由
保真度、稀疏度、可读性和标注成本共同决定。

## S0：冻结解释对象与边界

### 候选解释对象

C-long真正用于分类的表示为：

```text
完整胃镜图224x224
  -> EfficientNet-B0 features[8] [1280,7,7]
  -> C-long学习的attention [1,7,7]
  -> 空间加权求和
  -> attention-pooled向量 a [1280]
  -> 冻结分类头
  -> 癌/非癌logits
```

冻结checkpoint为：

`结果/MAGE/MG2L训练轮数敏感性_20260818/正式验证集筛选/`
`mg2l_armc_efficientnet_b0_seed42/mg2_armc_best_student.pth`

### S0冻结核验结果（2026-08-19）

全部SHA均从文件本体重新计算，与C-long正式config逐项一致：

| 资产 | SHA256 |
| --- | --- |
| 学生checkpoint `mg2_armc_best_student.pth` | `29e76977251fbd50c9fd9ecd6ebd26927eaf9b54eeb8e70a0b70d3d5a70ce014` |
| manifest（v3独立轴动态扩边，2847行） | `b1dfd24505cbba876fd28502e36b0b190206de8f063562f47347dc7c67f32e33` |
| v3 audit JSON | `94f78dc94fce9ca95692ef05d9c423313acb78dfdb7a431ab57603193095017a` |
| 教师checkpoint `mg1b_best_teacher.pth` | `f2cd3b13cbbc0e8b4df64e9090ec50343314137ba71a4c07d9fae9faa8af48f9` |
| 教师缓存 `teacher_cache_mg1b_v3.pt` | `89f3d329d68b182b33cad6f9a074e4cd919fb910bca96fff9d9ab16427d9e7db` |
| beta校准JSON | `47179b6228122096c00e6f0e59ab5c45663b3f163a506c77aa8811320803ac7f` |

split与数据血缘核验（来自manifest本体）：

- train 2350图/1212人（癌1181、非癌1169）；val 497图/260人（癌240、非癌257）；
- 患者跨split泄漏=0；2847个图像sha256全部唯一；
- 该config的test/internal test/external标记均为false（训练时记录）。此后external已按
  MAGE线规则做过一次描述性投影（见MAGE进度文档），但SAE开发仍不得使用其结果反向
  选参。

分类头结构核验（checkpoint state_dict）：

- `attention_head`：1x1 conv，1280 -> 1，49位置spatial softmax；
- pooling：attention加权求和，无GAP旁路（与config一致）；
- `classifier`：Dropout + Linear(1280 -> 2)，eval态Dropout关闭；
- margin方向 = `classifier.1.weight[1] - classifier.1.weight[0]`（index 1 = 癌）。

关键冻结数值（来自正式config）：

- 最佳checkpoint：stage B epoch 22，按val患者AUC选出；
- val患者AUC = 0.9101384270358055，val图像AUC = 0.8669909208819715；
- 患者级阈值 = 0.3074711561203003；图像级阈值 = 0.25923898816108704；
- 病灶面积三分位（train冻结）= (0.18661894999626696, 0.3384438676553876)；
- SAE特征提取必须使用C-long val/eval预处理：完整RGB直接resize到224x224，无翻转、
  无P6光度增强；实现时从`train_mage_mg2_student.py`的val transform复刻，并以
  "缓存特征复算概率与原val概率最大差"做self-test。

新SAE config生成时须复制上述全部SHA，并在每次训练前运行时校验。训练前后须校验C-long
参数、BN统计、注意力图和原始概率不变。

### 数据边界

- SAE参数只由原C-long的train特征训练；
- val只用于epoch、字典、lambda、剪枝和Feature候选选择；
- 同一患者全部图像保持在同一split；患者/类别平衡发生在图像采样层，不改变原划分；
- internal test和external均不得参与字典、gamma、lambda、剪枝、命名或Feature选择；
- C-long外部结果已经揭盲，但只能作为冻结模型背景，不能反向影响新SAE开发决策。

## S1：特征缓存与双层解释

### 主对象：分类实际使用的整图表示

第一版SAE解释attention-pooled 1280维向量。该表示与冻结分类头直接相连，适合评价重构后的
分类margin、患者AUC和个体预测一致率。

### 空间证据：同一Feature的7x7响应

整图Feature的空间图由SAE decoder方向与C-long `features[8]`位置向量计算，并与原C-long
attention同时展示。空间图用于判断Feature落在病灶、器械、反光、气泡、褶皱或背景哪里，
但不另行训练癌/非癌两套字典。

若这种反投影不能稳定定位，再单独预注册空间Token或组稀疏SAE；不得把后续空间实验与首版
整图SAE混成同一次参数搜索。

## S2：SAE结构、损失与候选矩阵

### 字典宽度

输入固定`D=1280`，第一轮在同一冻结C-long、同一特征缓存、SAE seed42上同轮比较：

| 编号 | 字典宽度 | 比例 | 文献定位 |
| --- | ---: | ---: | --- |
| W0 | 512 | 0.4x | M-CBM医学图像压缩字典参照 |
| W1 | 1280 | 1x | 等宽参照 |
| W2 | 2560 | 2x | 轻度过完备 |
| W3 | 5120 | 4x | M-CBM可控标注成本上限参照 |
| W4 | 10240 | 8x | A1c历史合格容量、InterPLM低倍端参照 |

8x必须进入首轮：它是旧路线唯一通过全部保真门槛的容量（A1c `1280 -> 10240 -> 1280`），
而"attention-pooled表示比GAP简单"目前只是未经验证的假设；多跑一档成本有限，却能获得
完整的宽度—保真曲线。原"W3显示高度复用才另行扩宽"的条件式闸门已取消，不再设置可能
被事后解释的触发条件。

### 默认训练损失

对输入表示`a`、重构`a_hat`和冻结二分类头，定义癌相对非癌margin：

```text
m(a) = logit_cancer(a) - logit_non_cancer(a)
```

新路线默认损失为：

```text
L = L_feature_MSE
  + lambda_l1 * L1(h)
  + gamma * L_margin_MSE
```

其中：

- `L_feature_MSE`保护1280维一般表示；
- `L1(h)`产生稀疏Feature激活；
- `L_margin_MSE`比较原表示与重构表示的标准化分类margin，不使用标签，是label-free保真项；
- `gamma=0.1`作为所有正式宽度的默认值，不根据val重新扫描。

原因是旧A1b/A1c内部诊断已经证明，分类权重方向只占总重构误差能量约0.2%，纯MSE可能在
整体重构看似良好时丢失个体分类方向。margin保真必须进入训练目标，不能只作为S3事后指标。

无margin诊断对照冻结为唯一一组：

```text
W4 = 10240
lambda = 5e-4
gamma = 0
SAE seed = 42
```

它只用于量化margin项的贡献，不参与正式候选选择，不得扩展成新的gamma网格，也不得
改成其他宽度或lambda。

### 稀疏训练实现

第一版遵循三篇文献可比、且是旧路线2x2实验（稀疏机制x损失）中唯一未跑过的空格：
`Linear + ReLU + L1 + margin保真`：

- encoder线性层加ReLU；
- decoder线性重构；
- decoder方向单位范数及必要的梯度正交投影；
- 学习率与L1在训练早期从0 warmup到冻结目标；
- 患者/类别平衡采样；
- 训练、验证分别记录feature、margin、L1加权项和L0。

### lambda冻结网格

lambda_l1固定为统一网格：

```text
lambda_l1 = {2e-4, 5e-4, 1e-3}
```

- 覆盖旧A1实验与M-CBM医学图像量级；
- 五个宽度共享相同的lambda预算，不为单个宽度单独拟合lambda，避免宽度比较混入不同的
  调参力度；
- 不根据test、external或临床可读性反向选择lambda；
- 明确承认：同一lambda在不同宽度下L0不会严格对齐。因此本矩阵不是"固定L0的纯宽度
  消融"，而是"相同lambda预算下，各宽度能够达到的保真—稀疏Pareto前沿比较"；
- 横向比较时必须同时报告绝对L0、`L0/字典宽度`和剪枝后保留Feature数，不能只按lambda
  对齐。

该网格在正式训练前冻结；看过正式val矩阵后不得追加新lambda。

### Top-K预登记备选分支

预先登记一个固定备选配置，不在看到L1结果后再设计：

```text
字典宽度 = 10240
K = 1024
gamma_margin = 0.1
训练、缓存和验收逻辑与主路线完全一致
```

执行规则：

- L1矩阵结束后，该Top-K固定配置始终运行一次，不存在"L1通过则跳过"的分支；
- 至少一个`L1 + margin`候选通过全部硬门槛：Top-K结果只作为机制对照报告，不替换
  主路线；
- 所有`L1 + margin`候选失败：Top-K结果升级为备选候选，参与正式选择；
- 不追加新的K或gamma网格。

三种结局各自回答一个问题：

- L1+margin通过：margin保真是关键因素，L1在该表示上并非不可用；
- L1+margin全败而Top-K通过：硬稀疏机制本身也重要；
- 两者都失败：问题更可能在表示对象或SAE结构，而不是继续调lambda。

Top-K避免了L1对全部非零激活持续向零收缩的问题，但编码器缩放、decoder方向与截断仍会
影响激活幅度，不能宣称其激活幅度无偏。

### L1收缩偏差的记录与解释约束

在固定decoder归一化约定下，L1持续对非零激活施加向零压力，可能产生收缩偏差。
为控制其影响：

- 同时报告原始激活、重构贡献`h_j × d_j`和Feature置零后的margin变化，三者分开呈现；
- 使用encoder/decoder互逆缩放进行Feature归一化，保持重构不变；
- S7干预采用残差保留重构，并与相同激活频率的随机Feature干预比较；
- 不把L1激活绝对值直接解释为医学概念强度。

### 门槛冻结分工

- 分类成功门槛在正式训练前直接冻结为固定数值：患者AUC下降不超过0.01、患者阈值
  一致率不少于0.95、cosine不少于0.90、recovered CE不少于0.95。这是正式冻结门槛，
  不来自train校准，也不因val结果事后放宽或收紧；
- 剪枝按S3分工执行：候选阈值只由train激活患者数产生，val选择使val recovered CE
  下降不超过0.01的最大阈值（详见S3剪枝节）；
- train只用于校准损失数值尺度（warmup、日志量纲），不能决定任何验证成功标准；
- 所有门槛在看过正式val矩阵后不得追加或修改。

### 训练预算与checkpoint规则

沿用旧路线已验证、且与现有`efficientnet_sae_discovery.py`默认实现一致的预算：

```text
optimizer = Adam, lr = 1e-4
batch size = 32（患者/类别平衡采样）
最大epoch = 1000, early stop patience = 50
学习率、L1与margin权重在前5% epoch从0线性warmup
checkpoint按患者平衡的完整val总损失（feature+margin+L1加权项）选择
```

五档宽度 x 三档lambda共15组L1候选、1组无margin诊断和1组Top-K备选使用完全相同的
预算；不扫描学习率、epoch上限或其他超参数。

## S3：质量评价、选择与剪枝

### 表示与稀疏质量

每个split至少报告：

- feature MSE和cosine；
- 平均L0、NCC90、死亡Feature率和Feature密度直方图；
- decoder近重复方向率；
- 每个Feature激活患者数、图片数和标签/来源分布；
- 不同宽度下实际活跃Feature数，而不是只报告字典容量。

### 分类保真质量

至少报告：

- 原始与重构margin的MSE、MAE和相关性；
- 原始与重构图像级、患者级AUC；
- 冻结阈值下预测一致率、Sensitivity、Specificity和混淆矩阵；
- recovered CE；
- 患者概率MAE及最大偏移病例。

margin项进入损失后，这些指标仍用于验收，但不再承担“从纯MSE候选中碰巧找到分类保真模型”
的任务。四项正式冻结门槛为：患者AUC下降不超过0.01、患者阈值一致率不少于0.95、
val cosine不少于0.90、val recovered CE不少于0.95，不因新结果事后放宽。

### 唯一正式配置选择顺序

从15组L1候选中按固定顺序选出唯一正式配置：

1. 先过全部硬门槛：四项分类保真门槛（AUC下降、一致率、cosine、recovered CE）加死亡
   Feature率不超过10%、decoder近重复方向率不超过10%；
2. 合格者内部按固定维度的保真—稀疏Pareto前沿比较，不得只按总损失或单一指标排序。
   Pareto维度与方向固定为：

   ```text
   患者AUC下降 ↓
   患者一致率 ↑
   cosine ↑
   recovered CE ↑
   val mean L0 ↓
   剪枝后保留Feature数 ↓
   ```

   死亡率和重复率只作为硬门槛，不再进入Pareto维度；
3. 若Pareto前沿包含多个候选（各候选指标互有优劣，不属于"并列"），则对所有前沿候选
   依次按：患者AUC下降更小 → 患者阈值一致率更高 → val mean L0更低 → 剪枝后保留
   Feature数更少 → 字典宽度更小，选出唯一正式配置。其中"活跃Feature数"一律指
   val mean L0（平均每图激活数），不与"train至少激活一次的Feature总数"或"剪枝后
   保留数"混用；
4. 只选出一个正式配置进入SAE seed202/503复现和后续S4-S8，不并列多个"正式"配置；
   其余合格候选作为敏感性参照报告。

死亡与重复Feature定义写死：

```text
死亡Feature：train所有样本激活均 <= ACTIVE_EPS = 1e-8
重复Feature：在非死亡Feature中，与另一个decoder方向的绝对余弦 >= 0.95；
            每个Feature最多计一次
硬门槛：死亡率 <= 10%，重复率 <= 10%
```

注意：旧实现（`efficientnet_sae_discovery.py`）对全部decoder方向计算重复率，未先排除
死亡Feature。正式新协议在旧阈值`abs cosine >= 0.95`基础上增加"只统计非死亡Feature"
的限制；新入口实现时必须显式传入非死亡掩码，不能直接照搬旧函数行为。

停止规则：若15组L1均失败且固定Top-K备选也未通过全部硬门槛，则本阶段无正式产品，
停止并回到表示对象或SAE结构设计，不继续追加超参数。

### 剪枝

剪枝冻结为可执行定义：

- 候选阈值只由train激活患者数产生；
- 在val上选择满足`val recovered CE drop <= 0.01`的最大阈值，drop相对于同一SAE的
  未剪枝完整重构结果计算；
- 若没有任何非空剪枝方案满足容差，则保留全部Feature，不允许输出违反容差的剪枝产品；
- 旧代码的fallback可能在无合格阈值时仍返回不合格方案（仅记录
  `selection_within_tolerance=false`），新实现不得照搬该行为；
- 同时报告剪枝前后cosine、margin、AUC和实际保留Feature数。

剪枝只表示“对当前分类影响很小”，不能把被剪Feature解释成没有医学信息；保留Feature也
不能自动称为医学概念。

## S4：原型展示与临床命名

每个进入审核的Feature按患者去重展示：

1. 最高激活患者；
2. 高、中、低激活分位样本；
3. 零激活随机样本；
4. 外观相似但不激活的困难负例；
5. SAE空间响应、C-long原attention和病灶框三者对照；
6. 癌margin贡献、来源相关性及置零消融预览。

临床审核字段至少包括：建议名称、是否单一语义、是否病灶相关、癌/非癌均可出现、器械、
反光、气泡、黏液、褶皱、暗腔、边缘、来源/设备风险和不确定。审核者优先看图像后再查看
标签富集统计，避免类别名称诱导命名。

Feature名称只是关于内部方向的假设。进入M-CBM前，必须再对独立图片标注该概念是否存在，
不能直接把SAE激活当作医生概念真值。

## S5：复现与稳定性

### 分阶段复现安排

不为全部15个L1候选（5宽度 x 3 lambda）都跑三种子：

1. SAE seed42完成宽度 x lambda开发矩阵；
2. 按S3冻结规则选择唯一正式配置；
3. 仅该配置用SAE seed202、503重训；
4. 最终报告SAE优化稳定性为3/3、2/3或1/3。

### SAE训练随机种子复现

C-long只有seed42冻结模型，因此不具备分类器seed202/503。正式复现定义为：

```text
同一C-long checkpoint
+ 同一train/val特征缓存
+ 同一超参数和数据顺序规则
+ SAE seed = 42 / 202 / 503独立训练
```

这只能证明SAE字典在优化随机性下是否稳定，不能宣称C-long模型跨训练种子复现。

跨seed Feature不按编号对应，至少结合：

- decoder方向余弦；
- 共同患者激活Spearman相关；
- Top患者Jaccard；
- 癌/非癌富集方向；
- 空间响应和临床语义。

分别报告3/3、2/3和单seed Feature，不为增加匹配数量而事后放宽阈值。

### 跨患者与跨数据集稳定

“跨患者复现”指同一Feature在多位不同患者中出现一致视觉模式；“跨数据集稳定”指冻结SAE
投影到internal test或external后仍保持方向和语义。两者与SAE seed复现分开报告。

test/external只能在字典、剪枝、Feature名单和临床规则完全冻结后投影一次，不参与任何选择。

## S6：癌与非癌共享、特异和重复Feature分析

### 基本原则

癌与非癌图像共同训练一个联合SAE。癌侧Feature和非癌侧Feature只是同一字典的两种统计视图，
不是两套独立概念库。这样天然允许发现“胃镜镜体、腔口、褶皱、反光”等两类共有模式。

### 患者级分类

每位患者先聚合其图片Feature激活，再将候选分为：

- 两类共有：癌和非癌患者中均稳定激活，标签区分度低；
- 癌富集：癌患者激活率或强度更高；
- 非癌富集：非癌患者激活率或强度更高；
- 混合/稀有：覆盖不足或方向不稳定。

共有Feature不自动删除。它可能是正常解剖、成像条件或诊断所需背景；只有结合分类权重、
空间位置和干预结果，才能判断是有用上下文还是伪特征。

### 癌侧与非癌侧概念家族合并

分别从癌富集、非癌富集和共有候选中取代表Feature，再通过以下证据寻找重复或相关家族：

- decoder方向余弦；
- 全体患者激活相关；
- Top患者/Top图重叠；
- 空间热点重叠；
- 医生给出的语义名称。

若癌侧与非癌侧候选都响应同一器械、反光或胃腔结构，应合并为一个共享概念家族并保留双方
统计，而不是重复命名或只从一侧删除。概念家族关系与单Feature结果同时保存，避免过度合并
掩盖细粒度差异。

## S7：因果干预

对Feature `j`置零或按比例缩放后，使用残差保留重构：

```text
residual = a - Decoder(Encoder(a))
a_intervened = Decoder(h_intervened) + residual
```

再通过冻结C-long分类头评价：

- 癌概率和margin变化；
- 图像级、患者级AUC及阈值指标变化；
- 癌/非癌、来源、病灶大小和伪特征子组变化；
- 与同激活频率随机Feature、随机方向和置乱激活对照的差异。

只有“视觉模式一致、跨患者复现、分类贡献稳定、干预方向符合预期”的Feature，才可升级为
高可信模型概念。相关性或Top图单独不足以支持因果结论。

## S8：透明模型候选

### ProtoMIL式诊断模型

先使用冻结SAE激活和稀疏线性分类器，验证少量Feature能否承担分类，并把预测拆成Feature
激活、空间权重和类别权重。该模型适合快速诊断，但SAE名称仍是假设。

### M-CBM式正式模型

临床概念审核后，对部分train/val图像标注概念存在、缺失或不确定，训练masked BCE概念层，
再以elastic-net稀疏分类器预测癌/非癌。最终同时报告任务性能、概念预测AUC、NCC95、泄漏
对照和随机概念基线。

概念标注必须class-agnostic：同一概念的正负例同时覆盖癌与非癌，不能让"是否被标注"本身
泄露疾病标签。

## S2c后正式解释阶段规划（2026-08-21确认）

> **历史条件规划：** 本节按“S2c产生正式产品”这一前提起草。S2c现已按预注册失败，本节
> 已被后文RP-SAE新协议替代，仅保留解释原则和历史决策依据，不代表当前执行路线。

本节只规定S2c成功后的解释流程、证据维度和定阈值原则，不修改S2c训练、八项门槛、
最小合格K或停止规则。具体跨seed匹配阈值、家族边阈值、富集边界和医生panel数量，
必须等S2c产生正式产品和三seed分布后，使用train/val只读统计单独预注册；不得查看医生
命名、干预结果、internal test或external后再调整。

长期原则冻结为：

> 共享不删，重复先组家族，伪特征经临床审核与因果干预后再处理。

同时冻结以下八条解释边界：

1. 癌/非癌共享不能作为删除依据；
2. Feature相似性只产生family candidate，不自动合并；
3. 技术Feature family与医学概念family必须区分；
4. 跨SAE seed对齐先于family构建；
5. 医生先盲审语义，再揭示类别、来源、空间和贡献；
6. 伪特征必须同时有视觉一致性、混杂关联和干预证据；
7. 单Feature与family-level干预并行；
8. M-CBM概念必须经过独立存在性标注，不能直接把SAE激活当医学真值。

### 后续证据链与进入条件

```text
S2c正式K
→ SAE seed42/202/503复现
→ 跨seed Feature对齐
→ 技术Feature family候选
→ 癌/非癌四维共享性审计
→ 医生两级盲审
→ 来源与伪特征审计
→ 单Feature与family残差保留干预
→ M-CBM候选概念
```

只有S2c seed42产生正式K并按冻结协议完成SAE seed202/503复现后，才启动本流程。若S2c
失败，本节不作为继续扫描K、修改`0.90`门槛或追加SAE结构的理由，应另立解释方法协议。

### 跨seed对齐与Feature family分工

跨seed alignment回答seed42 Feature在seed202/503中是否存在近似对应物，是优化随机性
复现证据；Feature family回答多个不同Feature是否属于同一粗粒度语义家族。两者不能合并
为一次聚类，也不能把跨seed匹配数量解释为唯一标准概念的数量。

跨seed匹配至少综合带符号decoder方向余弦、患者激活Spearman、Top患者Jaccard和空间响应
相似性，优先采用互为最近邻。医学家族使用带符号cosine；S2c质量控制中冻结的
`abs cosine >= 0.95`仍只用于重复率，不能改写，也不能直接用于医学家族合并。

家族构建以Feature为节点、多证据合格关系为边，层次聚类或HDBSCAN只生成候选结构。
医生最终可判为同一家族、同一粗概念的不同亚型、技术相关但语义不同或无法判断；所有
原始Feature编号、单Feature统计和family归属必须同时保留。

### 癌/非癌四维共享性审计

每个Feature和候选family分别报告：

1. 标签共享：癌/非癌患者覆盖率、激活均值与中位数、标签AUC及患者bootstrap置信区间；
2. 语义共享：两类Top图是否呈现同一视觉含义；
3. 空间共享：两类中是否在病灶、背景或相同解剖位置响应；
4. 因果共享：两类中干预后margin与癌概率是否呈相同方向变化。

标签类别最终至少包括癌富集、非癌富集、共享高覆盖、共享低覆盖/稀有、混合/不稳定。
“共享”不能只由AUC接近0.5定义；正式边界在三seed Feature分布完成后冻结。

### 来源、设备与伪特征审计

器械、反光、气泡、黏液、文字、黑边、画中画、分辨率、医院和设备风格分别审计。
来源关联必须同时报告全体患者、癌患者内部和非癌患者内部结果，或拟合患者级
`Feature ~ Label + Source`模型，防止把癌/非癌构成差异误判为来源特征。

高风险伪特征必须同时具备：视觉模式一致、在控制标签后仍与来源/伪影相关、且干预后
模型决策发生稳定变化。只满足其中一项时标为疑似风险，不得直接删除。

### 医生两级盲审与语义反证

Stage 1只审核“Feature是什么”：按患者去重展示高、中、低、零激活样本、跨癌/非癌样本、
空间响应和近邻困难负例，不显示标签AUC、富集名称、来源和分类贡献。医生记录医学病灶
形态、正常结构、共享背景、伪特征、单义、疑似多义或无法判断。

Stage 2再审核“模型如何使用Feature”：揭示癌/非癌富集、来源、病灶内外位置、margin和
干预结果，判断这种使用是否医学合理。近邻困难负例承担主动证伪作用：若医生初始命名为
“黏膜发红”，应检查外观同样发红但不激活的图，修订真正触发Feature的附加条件。

初筛和深审panel数量属于待定执行参数，只按医生工作量和train/val分布预注册，不把任何
文献中的样本数直接当作胃镜标准。

### 单Feature与家族级干预

沿用S7残差保留重构，同时对单Feature和整个候选family做多剂量缩放，比较原始、部分抑制
和完全置零下的margin、癌概率及患者级指标，检查剂量反应是否稳定。随机对照至少匹配
family成员数、患者激活频率和基线总重构/分类贡献。

单Feature效应小而family效应明显时，只能表述为“与Feature splitting或冗余补偿相容”，
不能宣称已经证明Feature absorption。最终进入M-CBM的候选需同时满足视觉一致、跨患者、
跨seed、空间合理、来源风险低或已解释、分类贡献稳定及干预方向合理。

### 文献边界

Histoscope、M-CBM、ProtoMIL、PICASSO和两篇病理SAE工作直接支持专家审核、共享概念、
困难负例、空间/家族组织和伪影干预；PatchSAE支持patch级空间归因。A is for Absorption、
Descriptive Collision、Sparse Autoencoders Do Not Find Canonical Units of Analysis以及
Revising and Falsifying SAE Feature Explanations来自语言模型，当前只作为方法学风险和
流程设计依据，不作为胃镜Feature已经发生分裂、吸收、命名碰撞或非原子性的实证。

## 第二轮文献调研：泛化差距与Patch级SAE（2026-08-20）

针对正式矩阵暴露的问题（Top-K train cosine 0.9979 / val 0.8814，泛化不足），定向核查了
本地文献与公开文献。核心结论：**文献中没有人在"几千个整图pooled向量"上训练SAE并讨论
其泛化差距——所有先例都在位置/patch/token级向量上训练，样本量比本项目大2~6个数量级，
泛化问题在源头就被规避了。**

### 问题一：train/val重构差距，文献有没有直接解答

没有直接讨论，但有三个间接答案：

1. **字典学习样本复杂度**：字典参数量应与独立训练样本数匹配。本项目pooled方案为
   2600万参数对2350个向量，远超出正常范围；这解释了为何8×宽度下val cosine仍只有
   0.857——不是宽度不够，是样本不够。
2. **PathAI（Le et al., NeurIPS 2024 workshop，病理基础模型PLUTO的SAE）**：训练数据
   多样性越高，死亡Feature和超稀疏Feature（<0.1%样本激活）越少，且Feature能跨染色
   类型泛化。这同时支持"增加训练视图"（翻转扩增的方向）和"改用patch级训练"两条路。
3. **Gallifant et al.（EMNLP 2025，SAE特征分类与迁移）**：SAE特征在下游数据有限时
   仍稳健，但其前提是SAE本身在大规模token上预训练；不适用于"小样本从头训练SAE"。

翻转扩增没有任何SAE文献先例，属于我们自己的启发式修补，只能作为低成本验证，
不能作为主路线的依据。

### 问题二：Patch级/位置级SAE的文献先例

全部关键先例都在位置级训练，且与本项目设计高度对应：

| 文献 | 训练对象 | 样本量级 | 与本项目对应点 |
| --- | --- | --- | --- |
| SAE-V（Stevens et al. 2025，OSU） | ViT残差流逐patch向量 | ImageNet-1K×196 patches | 逐位置编码；训练前均值归一化；残差保留干预（x'=e+x̂'）与我们S6一致；证明patch级SAE可支持分类与分割的因果编辑 |
| InterPLM（本地） | 蛋白质逐氨基酸位置向量 | 5M序列×数百残基 | 位置级分解天然给出可定位Feature |
| ProtoMIL（本地） | CONCH patch嵌入512维 | 数十万WSI patch | 4×扩维、ReLU+L1 λ=3e-4；发现的伪概念（墨水、马克笔、失焦）正好对应本项目的器械/反光担忧 |
| PathAI（Le et al.） | PLUTO patch CLS嵌入384维 | 110万patch | 8×扩维、死亡神经元重采样；HDBSCAN对decoder方向聚类得到概念家族——对应我们的重复概念家族分析 |
| SAE-Rad（Abdulaal et al. 2024） | 胸片ViT潜变量 | MIMIC-CXR规模 | 医学影像SAE能产生有意义概念的首例 |
| CytoSAE（Dasdelen et al. 2025） | 细胞图像嵌入 | 血液细胞图像集 | 小器官尺度医学概念发现可行 |

### 对本项目设计的具体启示

1. Patch SAE（49个位置向量/图，2350×49≈11.5万向量）是文献唯一支持的方向；
   pooled整图SAE在小样本下没有成功先例，本轮失败与此一致。
2. 训练前对位置向量做均值中心化（SAE-V/Anthropic惯例）；是否做单位范数归一化需在
   预注册中写死，因为它改变重构目标的语义。
3. patch级重构cosine预计会显著高于pooled级（正常黏膜背景占主导，易重构），因此
   分类保真必须继续通过pooled重构+冻结分类头的margin来把关，不能用patch级cosine
   替代分类门槛——与用户方案中的三项损失设计一致。
4. 位置向量间强相关（感受野重叠），11.5万不是独立样本数；val患者级隔离仍是主要
   泛化检验，文档与预注册中不得夸大为"样本量提升49倍"。
5. ProtoMIL/PathAI均在SAE中发现采集伪迹概念（染色、模糊、墨水）；本项目若发现
   器械/反光/气泡Feature属于预期行为，正是审计目标而非失败。
6. 超稀疏Feature和decoder方向聚类（HDBSCAN）两套分析可直接沿用PathAI做法。

### 结论

文献明确支持"停止pooled级参数扫描，转向patch级SAE"的判断。P1（翻转扩增）无文献
先例，最多作为一次低成本对照；Patch SAE应作为下一阶段主路线，其损失设计、归一化
口径、门槛语义和位置权重公式必须在新的预注册中先行冻结。

### 第三轮：用户文献分析的逐条核验（2026-08-20）

用户提交了含5个新引用的文献分析与S4四组矩阵草案。逐条核验结果：

**引用真实性：5篇全部真实存在，无虚构。**

1. PatchSAE（ICLR 2025，CLIP ViT）：摘要确认"提取patch级空间归因的可解释概念"属实。
   注意：论文主题是CLIP适配机制，摘要未直接出现"不同数据域共有/特有概念"和"两个图像
   共同激活Feature"的表述，引用该具体结论前需读全文确认。
2. BatchTopK（arXiv 2412.06410，Bussmann/Leask/Nanda）：描述全部属实——batch级
   top-k、可变每样本激活数、同平均稀疏度下重构优于Top-K、推理时用全局阈值θ（训练集
   batch上最小正激活的均值）+JumpReLU消除batch依赖。但有两处原文限定用户未提：
   ① 仅在GPT-2/Gemma-2上验证，未做视觉实验；② 论文明确说未评估可解释性；
   ③ 它改善的是分布内的重构-稀疏Pareto前沿，并未证明能改善train→val泛化差距。
3. Gated SAE（arXiv 2404.16014）：描述属实（分离"是否激活"与"激活幅度"，解决L1收缩）。
4. Matryoshka SAE：注意有两篇同名工作。用户描述的CLIP实验对应Zaigrajew/Baniecki/
   Biecek《Interpreting CLIP with Hierarchical Sparse Autoencoders》（ICML 2025，
   arXiv 2502.20578），"重构-稀疏Pareto前沿最优"表述属实；另一篇Bussmann/Leask
   （arXiv 2503.17547）是GPT-2上的层级Feature研究，引用时需区分。
5. Gao et al.（arXiv 2406.04093）归一化断言：原文逐字确认——"We subtract the mean
   over the d_model dimension and normalize all inputs to unit norm, prior to passing
   to the autoencoder (or computing reconstruction errors)"，即逐样本减维度均值+单位
   范数归一化+归一化空间算重构误差，用户描述准确。

**我们实现与经典Top-K的差异：属实。** `SparseAutoencoder`只用数据集级`feature_center`
（decoder_bias）做中心化，无逐样本单位范数归一化（efficientnet_sae_discovery.py:267-292）。

**归一化假设的实证预检（用现有缓存，只读）**：

```text
train norm = 12.23±3.72（q05=8.47, q95=18.10）
val   norm = 11.73±2.54（q05=8.49, q95=16.29）
KS检验 p=0.045，分布高度重叠
norm→标签 AUC：train 0.565 / val 0.528（弱信号）
维度均值→标签 AUC：train 0.340 / val 0.383
```

2026-08-20更正：此前把维度均值定性为"弱反向"有误。AUC低于0.5表示预测方向相反，
反转后等价于train 0.660 / val 0.617，属于中等标签信号；模长AUC 0.565/0.528仍属弱信号。
因此，"保存mean/norm→重构后原样还原"会形成一条携带标签信息、但未经SAE解释的
旁路。S2b正式候选不启用逐样本归一化；归一化只进入固定诊断臂S4-N，不参与正式产品
选择。

train/val模长分布只有有限差异，"模长分布不同导致cosine差距"最多解释一小部分；差距
主体更可能是患者级方向泛化。归一化诊断仍有价值，但不能把它预设为弥合差距的主办法。

**S4矩阵草案评估**：方向合理（S4-C/D为主候选正确），三点需收紧：

1. S4-B（pooled+BatchTopK）预期收益不确定：BatchTopK改善的是分布内Pareto，不是
   泛化；可能train/val cosine同时上升但差距依旧。保留为低成本对照可以，不要期待它
   单独解决问题。
2. BatchTopK的全局阈值θ估计协议（用多少train batch、是否只含train）必须在预注册
   写死；val评估时逐样本L0可变，门槛用val mean L0不变。
3. 归一化最终决定为：所有正式候选保持原始特征尺度；增加唯一固定诊断臂S4-N，不参与
   选择。S4-A历史基线保持不动，以维持可比性并避免mean/norm旁路污染正式结论。

## S2b：结构重构正式预注册（2026-08-20冻结）

说明：既有`S4`编号已经用于"原型展示与临床命名"，获得正式SAE产品前不能启动。为避免
重号，本轮结构修复正式编号为`S2b`；`S4-A/B/C/D/N`仅保留为实验臂简称，不代表覆盖
既有S4阶段。

### 研究问题与实验臂

本轮只回答三个问题：固定逐样本K是否限制重构、pooled表示的小样本是否是主要瓶颈、
逐样本归一化改善中有多少来自未解释标量旁路。冻结五臂如下：

| 实验臂 | SAE输入 | 稀疏机制 | 宽度/预算 | 正式角色 |
| --- | --- | --- | --- | --- |
| S4-A | pooled `1280`维 | Top-K | `10240/K=1024` | 已完成历史基线，只读引用，不重跑 |
| S4-B | pooled `1280`维 | BatchTopK | `10240/目标mean K=1024` | pooled结构对照；单独判断能否形成pooled产品 |
| S4-C | `7x7x1280`空间特征 | 逐位置Top-K | 共享字典`10240/K=128` | patch正式主候选 |
| S4-D | `7x7x1280`空间特征 | BatchTopK | 共享字典`10240/目标每位置mean K=128` | patch正式主候选增强版 |
| S4-N | pooled逐样本减维度均值并单位范数化 | Top-K | `10240/K=1024` | 唯一固定机制诊断，不参与产品选择与跨seed复现 |

`K=128`取自Top-K文献常见的低密度工作区，并把每个局部位置的预算限制为输入维度的10%；
本轮不扫描其他宽度、K、位置权重或归一化方式。S4-B不预设能解决患者级泛化，它只检验
可变激活预算在同平均稀疏度下是否优于S4-A。

### 数据、缓存与隔离

- 解释对象仍为S0冻结C-long `features[8]`输出；模型、attention head和分类头全程
  `eval()`且冻结；
- 使用同一manifest、train/val患者划分和S0六项SHA；test、internal test、external
  均不读取；
- pooled缓存沿用正式缓存；patch缓存新增`[N,49,1280]`空间特征、原始`[N,49]`
  attention、标签外元数据和病灶框；缓存config记录源checkpoint、manifest、代码、行顺序
  与全部数组SHA；
- patch缓存必须自测：由原始空间特征经过冻结attention head重新得到的attention、pooled
  向量和癌概率，与C-long正式val预测逐位一致（浮点容差`1e-4`），否则拒绝训练；
- patch字典在49个位置间共享，不拼接坐标。位置只用于回填热图和框内外评价，不能成为
  SAE输入，以免Feature退化为固定位置检测器；
- 第一轮不加入翻转或光度多视图。翻转扩增缺乏本任务直接证据，避免与表示层级同时变化。

### Patch训练目标

对每张图的原始空间特征`F_p`和冻结原始注意力`a_p`，定义位置权重：

```text
w_p = 0.5 + 0.5 * 49 * a_p
```

因为`sum(a_p)=1`，49个位置的`w_p`均值严格为1：低注意力区域仍保留0.5底座，高注意力
区域获得更高权重，但不会改变整项损失的平均尺度。该权重只用于train/val重构损失，
不得作为正式评价旁路。

```text
L_patch  = mean_p [ w_p * MSE(F_hat_p, F_p) ]
a_hat    = softmax(frozen_attention_head(F_hat))
z_hat    = sum_p a_hat_p * F_hat_p
L_pool   = MSE(z_hat, z_original)
L_margin = ((margin(z_hat) - margin(z_original)) / train_margin_std)^2

L_total = L_patch + gamma_pool * L_pool + 0.1 * L_margin
```

`gamma_margin=0.1`沿用S2-S3冻结值。`gamma_pool`不看val：固定使用S4-C的seed42初始化、
batch 32和患者类别平衡采样的一个完整train校准epoch（74批），记录初始化时`L_patch`
和`L_pool`中位数，并按
`gamma_pool = 0.25 * median(L_patch) / median(L_pool)`冻结，使pooled项初始约为patch项
的25%。校准JSON绑定S0、缓存、样本顺序、每批SHA和两项中位数；正式C/D必须从该JSON
读取同一个`gamma_pool`，禁止手工传值。若分母小于`1e-8`则快速失败。

S4-B沿用pooled目标`MSE + 0.1 * normalized margin MSE`；S4-N沿用相同目标，但只在
归一化方向空间训练和计算重构误差。

### BatchTopK训练与冻结推理阈值

- S4-B训练时在每个`B x H`预激活矩阵中保留最大的`B*K`项；
- S4-D以位置向量为基本单元，在每个图像batch的`(B*49) x H`预激活中保留最大的
  `B*49*K`项，使复杂图像/位置可获得更多Feature；
- S4-B/D不使用BatchTopK原文的死亡Feature辅助损失（`top-k_aux=512`、
  `alpha=1/32`），与本项目既有Top-K实现保持一致；上一轮宽度10240的Top-K死亡率为0，
  当前没有引入该额外机制的实证必要。若正式运行的死亡率超过`0.10`，按本轮
  停止规则处理，不得事后追加辅助损失；
- 训练采样仍以患者和类别平衡的图像为单位；同一被抽中图像的49个位置全部进入；
- val checkpoint选择使用固定行顺序、batch 32和固定batch边界的BatchTopK val总损失，
  不根据AUC、cosine或最终阈值结果选epoch；
- checkpoint冻结后，只用train、确定性DataLoader、每图一次、无有放回平衡采样估计
  全局推理阈值`theta`。实现使用分块Top-K归并求全体正预激活的目标分位数，不得一次
  物化patch组约十亿个预激活；
- 阈值选择目标为train实际mean L0最接近K；比较规则固定为`activation >= theta`；
- 保存checkpoint SHA、manifest/cache SHA、样本顺序SHA、目标K、实际train mean L0、
  `theta`、向量数、正激活数和并列计数；val只使用冻结`theta`，不得重新估计；
- 若全train正预激活总数少于目标总激活数，或求得`theta <= 0`，该运行快速失败；不得用
  零激活填满K；
- 若并列导致train mean L0相对目标K偏差超过1%，该BatchTopK运行判为协议失败，不进入
  正式门槛与产品选择。1%是实现有效性门槛，不是模型效果门槛。

### 训练预算与checkpoint

- seed42结构选择；Adam、学习率`1e-4`、图像batch 32、最多1000 epoch、patience 50、
  前5% epoch线性warmup，与S2-S3一致；
- decoder梯度正交投影、单位范数、患者类别平衡和冻结margin标准化保持不变；
- S4-B/C/D使用各自val总损失选择checkpoint，不扫描学习率、权重衰减、宽度、K或损失
  系数；
- S4-A只读引用既有结果；S4-N固定跑一次seed42，不参与任何checkpoint跨组选择。

### 正式评价与旁路诊断

Patch组正式评价必须走完整替换：

```text
重构7x7x1280特征
  -> 冻结attention head重新计算attention
  -> 重新汇聚1280维向量
  -> 冻结分类头
```

不得复用原始attention。另行计算"固定原始attention"诊断口径，两者之差用于区分内容
重构误差和attention漂移，不参与checkpoint选择。

共同报告患者/图像AUC、冻结阈值Sens/Spec/Acc/F1/CM、患者一致率、pooled cosine、
recovered CE、margin MSE/MAE/Pearson、最大患者概率偏移、死亡率和非死亡decoder重复率。

Patch组额外报告：

- 每位置mean L0、每图激活过的唯一Feature数；
- 每个Feature覆盖的图片数、患者数和空间位置数；
- 原始attention最高/最低四分位位置的L0与重构误差；
- 癌图病灶框内、框外与边界环的激活分布；
- patch cosine的均值及按患者、标签、来源、病灶大小分层；
- 重构attention相对原attention的KL、cosine、normalized AiB和PGA变化；
- Top激活patch、完整图位置热图及器械/反光/气泡人工QC。

pooled与patch的L0单位不同，禁止放入同一Pareto前沿直接比较。S4-B若通过原S3六项硬
门槛，可形成独立的pooled候选；patch正式产品只在S4-C/D之间选择。

### 归一化诊断S4-N

S4-N保存逐样本维度均值和单位范数并还原，仅用于量化经典Top-K归一化带来的收益和
旁路贡献。固定报告：

1. mean、norm各自的train单变量AUC和val单变量AUC；
2. 仅用train拟合`mean+norm`逻辑回归；同时冻结`0.5`参考阈值和train上满足
   `Sensitivity >= 0.90`的最高阈值，在val评价AUC及两套阈值指标；
3. 仅标量模型、归一化方向SAE使用train中位标量还原、归一化方向SAE使用真实标量还原
   三种结果；
4. 完整重构收益中方向部分与真实标量旁路分别贡献多少；
5. 所有拟合只用train，val只评价；不读取其他数据，不进入产品选择或复现。

### 硬门槛、选择和停止规则

S4-B沿用S3六项硬门槛。S4-C/D使用完整替换口径，必须同时满足：

1. val患者AUC相对原C-long下降不超过`0.01`；
2. 冻结患者阈值下一致率不低于`0.95`；
3. 重构pooled向量mean cosine不低于`0.90`；
4. recovered CE不低于`0.95`；
5. train死亡率和非死亡decoder重复率均不超过`0.10`；
6. 重构attention的normalized AiB与PGA相对原C-long各自下降不超过`0.05`。

patch mean cosine、attention KL/cosine只作分层诊断，不增加未经先验支持的事后硬门槛。
若C/D仅一个通过，选择该组；若均通过，依次按完整替换患者AUC下降更小、患者一致率更高、
pooled cosine更高、每位置mean L0更低、每图唯一Feature数更低选择唯一patch配置。若仍
完全相同，优先结构更简单的S4-C。

唯一patch配置随后固定结构和全部超参数，仅更换SAE随机种子202/503在同一冻结C-long
特征上重训。3/3通过为稳定产品，2/3为有限复现，1/3或0/3不形成正式产品。若seed42的
C/D均失败，本轮停止，不追加宽度、K、归一化、cosine损失或多视图超参数；回到
Matryoshka/Gated、解释层选择或数据规模设计另立新协议。

若最终得到patch产品，既有`S4：原型展示与临床命名`不能原样照搬pooled激活口径；进入
旧S4前须先冻结图像级聚合方式（如患者/图像内patch最大值或attention加权值）、Top patch
选择和空间图口径。该适配只能改变展示与汇总，不能重新训练或选择S2b产品。

## S2b实现与debug验收（2026-08-20）

本轮新增独立实现，不覆盖S2-S3的pooled SAE代码与结果：

- `clong_s2b_core.py`：逐向量Top-K、BatchTopK、分块Top-K归并阈值、
  patch完整替换与AiB/nAiB/PGA核心逻辑；
- `clong_s2b_discovery.py`：S0血缘校验、空间缓存、`gamma_pool`校准、
  B/C/D/N共享训练harness、train-only BatchTopK阈值和正式评价；
- `summarize_clong_s2b.py`：pooled与patch分开门槛，C/D五级决胜链；
- `run_clong_s2b_matrix.sh`：固定B→gamma校准→C→D→N→汇总顺序；
- `test_clong_s2b.py`：18项回归测试，包含BatchTopK协议失败的结构化记录与续跑回归。

已完成的debug验收：

1. S0六项SHA正常通过，空间缓存产出`[N,49,1280]`、pooled和attention；
2. CPU与正式GPU数值路径不同，debug-only复算容差显式记为`2e-3`；
   正式运行强制`--device cuda`且仍使用`1e-4`，两者不共用放宽口径；
3. `gamma_pool`debug校准、C完整替换、B/D全train阈值、N标量旁路均跑通；
4. 微型B/D的冻结阈值实际mean L0均精确等于目标4.0，并列数均为1；
5. 重构attention、完整替换分类指标、固定原attention诊断、patch覆盖统计和
   标准JSON均已成功落盘；test/internal test/external未读取。

这些debug数值只验证工程链路，不作为模型结论。

## S2b正式矩阵结果（2026-08-20）

正式seed42矩阵已按B→gamma校准→C→D→N→汇总的预注册顺序完成，
`test/internal test/external`均未读取。汇总状态为
`no_patch_product_stop_s2b`，未选出唯一patch正式产品。

| 实验臂 | 状态 | 原始→重构患者AUC | AUC下降 | 患者一致率 | pooled cosine | recovered CE | 死亡/重复率 | 空间门槛 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| S4-B pooled+BatchTopK | 协议失败 | 未进入正式评价 | - | - | - | - | - | train正预激活`2,158,956 < 2,406,400`，无法合法冻结K=1024 |
| S4-C patch+Top-K | 6/8门槛通过 | `0.91014→0.90921` | `0.00093` | `0.93462` | `0.88939` | `0.98600` | `0/0` | nAiB与PGA均通过 |
| S4-D patch+BatchTopK | 6/8门槛通过 | `0.91014→0.91207` | `-0.00193` | `0.93462` | `0.88211` | `1.01447` | `0/0` | nAiB与PGA均通过 |
| S4-N 归一化 | 诊断完成 | 不参与产品选择 | - | - | - | - | - | mean+norm联合val图像AUC`0.7104`、患者AUC`0.7723` |

S4-C/D均说明patch字典可以保持分类AUC、交叉熵和空间注意力，且没有死亡或
重复Feature问题；但两者均在同两项失败：患者预测一致率为`0.9346 < 0.95`，
pooled cosine为`0.8894/0.8821 < 0.90`。BatchTopK没有修复方向保真，反而略低于逐位置
Top-K。因此不将“AUC没降”解读为SAE已合格，也不用放宽已冻结的保真门槛换取产品。

S4-N进一步证实mean/norm标量组合含有显著标签信息，逐样本归一化后再原样回填
会形成一条未经SAE解释的旁路；这支持了“归一化只做诊断、不进正式产品”的预注册决策。
按停止规则，本轮不进入SAE seed202/503复现，不追加K、宽度、归一化、cosine或多视图
超参数；若继续SAE，须回到Matryoshka/Gated、解释层选择或数据规模设计另立新协议。

### S2b失败诊断口径审阅（2026-08-20）

v1诊断脚本对S4-C/D正式checkpoint和config SHA做了逐项校验，并以`1e-6`
容差复现正式患者一致率和pooled cosine。已经能确认：

1. C/D均翻转17/260位患者，仅9人重叠；翻转者的原始患者概率距冻结阈值
   中位数为`0.029/0.034`，未翻转者为`0.264`；两臂各有15/17人在阈值
   `+-0.10`内，说明一致率门槛主要受边界病例影响，但不因此放宽冻结门槛；
2. 在完整替换后翻转的17人中，C/D均有13人在固定原注意力时仍翻转，
   4人仅在重算注意力时翻转；因此影响主要来自特征内容重构，注意力漂移是较小的
   独立贡献。同时v1尚未单列“固定注意力翻转、重算后被纠正”的反向情况；
3. v1的分层结论实际使用每图49个位置的平均patch cosine（约`0.81/0.80`），
   不是触发正式失败的pooled cosine（`0.889/0.882`）。因此“正式cosine短板在
   所有亚组弥散”在v2运行前仍是待验证命题，不将patch cosine的分层结果冒充为正式
   pooled门槛的分层证据。

`analyze_s2b_failure.py`已补入逐图pooled cosine、同pool/patch分层、pooled最差30图、
完整翻转四象限和attention KL与pooled cosine的相关性。v2只读train/val并输出新目录，
不改写v1，不参与S2b产品选择。

### S2b失败诊断v2正式结果（2026-08-20）

v2已在`failure_diagnosis_seed42_v2`完成，并以`1e-6`容差逐项复现S4-C/D正式患者
一致率与pooled cosine；`test/internal test/external`均未读取。结论如下：

1. C/D仍各翻转17/260位患者，仅9人重叠；翻转患者距冻结阈值的中位距离仅为
   `0.0289/0.0341`，两臂均有15/17人在阈值`+-0.10`内，确认一致率短板主要由
   边界病例对小重构误差敏感所致，但冻结门槛不因此放宽；
2. 正式pooled cosine短板为弥散性问题：C在癌/非癌分别为`0.8879/0.8908`，
   D为`0.8783/0.8856`；来源、分辨率组和画中画分层之间差异同样很小。
   小病灶癌图相对更低（C/D为`0.8832/0.8757`），但不足以解释整体失败；
3. 完整替换翻转中，C/D均有13人即使固定原注意力仍会翻转，仅4人只在重算注意力
   后翻转；说明主要误差来自特征内容重构，注意力漂移是较小但独立的贡献；
4. 因此S2b失败不是某个来源、画幅或伪特征亚组单独崩溃，也不是Feature死亡或重复，
   而是单一K预算下的方向细节损失广泛存在。该结果支持检验S2c多粒度嵌套预算，
   但只构成假设依据，不预设Matryoshka一定通过冻结门槛。

## S2c：Matryoshka patch SAE正式预注册（2026-08-20冻结）

### 研究问题与实验定位

S2b说明单一`K=128`的patch Top-K能保持AUC、recovered CE和空间注意力，但
在患者阈值一致性和pooled方向保真上仍略有不足。S2c只回答一个新问题：
在不改变解释层、字典宽度和数据的前提下，同时学习从粗到细的嵌套Top-K重构，
能否让少量Feature表示主要概念，并在较大K下补回S2b丢失的特征方向细节？

本轮不将Matryoshka预设为必然有效。S2b诊断只是与“单一K下的弥散细节损失”假设
相容，不是Matryoshka必然通过的证据。

本轮同时冻结以下研究边界：patch化将train表示从每图一个pooled向量扩展为每图49个
局部位置，显著增加了字典训练所见到的局部表示覆盖和变化，但同一图像、同一患者内的
patch高度相关，不能将约11.5万个patch表述为约11.5万个独立样本，也不能据此断言
pooled SAE的泛化不足完全由有效样本量造成。S2c检验的是多粒度嵌套稀疏结构能否改善
当前保真短板，不检验单一的“样本量根因”。

### 冻结输入、结构与数据边界

- 解释对象与S2b相同：C-long seed42 `features[8]`的`7x7x1280`patch特征；
- 复用S2b已逐位校验的正式空间缓存、manifest、C-long checkpoint和全部SHA；
- train/val患者、图像、顺序、预处理和标签不变；`test/internal test/external`全程锁定；
- 共享字典为`1280 -> 10240 -> 1280`，49个位置不拼接坐标，不扫描新宽度；
- 嵌套粒度冻结为`K={64,128,256,512,1024}`，同一正预激活排序下保证
  `TopK64 subset TopK128 subset ... subset TopK1024`；
- 不启用逐样本mean/norm归一化、BatchTopK、死亡Feature辅助损失、位置坐标或
  多视图损失。

`K=128`与S4-C形成直接结构对照；`64`用于观察更粗概念；`256/512/1024`用于
逐步补回方向细节。文献对CLIP使用更长的K-list直至接近隐层宽度；本项目为保留临床
概念稀疏性而截断在`1024`，这是面向本医学任务的显式适配，不声称完全复刻论文。

### 损失、校准和checkpoint规则

对每个粒度`K_i`分别计算：

```text
L_i = L_patch_weighted_MSE
    + gamma_pool * L_recomputed_attention_pool_MSE
    + 0.1 * L_normalized_margin_MSE
L_total = mean(L_64, L_128, L_256, L_512, L_1024)
```

- 五层使用统一权重`1/5`（Matryoshka uniform weighting），本轮不同时比较reverse weighting；
  理由是当前瓶颈为保真，而非进一步强调最小K的稀疏性；
- patch位置权重、冻结注意力头重算、margin定义与S2b一致；
- `gamma_margin=0.1`继续冻结；`gamma_pool`不直接复用S2b数值，而是在同一train-only
  74批、seed42固定初始化上做一次校准。对每批先计算
  `joint_patch=mean_K(L_patch,K)`和`joint_pool=mean_K(L_pool,K)`，再冻结
  `gamma_pool=0.25*median_74(joint_patch)/median_74(joint_pool)`，使pool项初始量级约为patch项的25%；
  JSON必须绑定五层K-list、初始SAE state SHA、manifest/缓存/学生checkpoint SHA、74个批次SHA、
  两组中位数和最终`gamma_pool`；分母小于`1e-8`则快速失败，禁止手工传入或看val重校准；
- 预计算实现必须对一次排序结果构造嵌套mask，不得分别重复排序导致嵌套性漂移；
- 预激活为正的数量少于某个K时，该层真实L0可低于K，不用0填充假装激活；
  必须逐K报告train/val mean L0和正激活不足比例；
- 训练预算暂沿用`Adam lr=1e-4, batch=32, max=1000, patience=50, warmup=5%`；
  checkpoint只按五层加权val总损失选择，训练中不按单一K的AUC/cosine选轮次。

### 评价、K选择和成功门槛

训练结束后仅用冻结val，按`64 -> 128 -> 256 -> 512 -> 1024`顺序对每个K独立执行
完整替换评价：重构7x7x1280特征、重算冻结注意力、重新汇聚、再进入冻结分类头。
每个K继续使用S2b冻结的八项硬门槛：

1. 患者AUC下降`<=0.01`；
2. 冻结患者阈值下一致率`>=0.95`；
3. pooled cosine`>=0.90`；
4. recovered CE`>=0.95`；
5. 字典级train死亡Feature率`<=0.10`；死亡定义为一个Feature在五层K的train激活
   并集中从未激活，不按最终选定的较小K机械地把高粒度Feature计为死亡；
6. 在上述五层并集非死亡Feature中，decoder绝对cosine`>=0.95`的重复Feature率`<=0.10`；
7. normalized AiB下降`<=0.05`；
8. PGA下降`<=0.05`。

正式产品K定义为“同时通过八项门槛的最小K”，不再用AUC或图像观感从通过者中
二次挑选。若所有K均失败，S2c无正式产品，不放宽门槛、不追加K-list、
reverse weighting或Gated SAE到同一实验中。

诊断指标额外报告但不参与checkpoint/K选择：患者/图像概率MAE、患者最大概率偏移及ID、
阈值`+-0.05/+-0.10`内患者的一致率、逐K注意力KL/cosine、每位置L0、每图唯一Feature数、
概念覆盖的图像/患者/位置数，选定K下至少激活一次的实际使用Feature数及占字典比例，
以及标签/来源/分辨率/画中画/病灶大小分层。“选定K下使用Feature数”是信息性指标，
不替代五层并集的字典级死亡率硬门槛。

#### 诊断字段补充（2026-08-21，不改变冻结选择协议）

为避免将cosine当作唯一重构质量指标，S2c对每个K补充以下纯诊断字段：

```text
patch_FVU  = sum_val ||x_hat - x||^2
             / sum_val ||x - mu_patch_train||^2

pooled_FVU = sum_val ||p_hat - p||^2
             / sum_val ||p - mu_pooled_train||^2

patch_explained_variance  = 1 - patch_FVU
pooled_explained_variance = 1 - pooled_FVU
```

- `mu_patch_train`由确定性完整train的全部图像与49个位置计算，`mu_pooled_train`由完整
  train图像计算；二者只计算一次并将数值、shape和来源缓存SHA写入config；
- `x/x_hat`分别为原始/重构patch特征，`p/p_hat`分别为原始/完整替换并重算注意力后的
  pooled特征；分母小于`1e-12`时快速失败；
- explained variance不裁剪到`[0,1]`，允许负值如实表示重构比train均值基线更差；
- config对四项均写入`diagnostic_only=true`。它们不参与checkpoint、K选择、八项门槛、
  成败判定或后续seed放行，不构成第九项门槛。

### 复现、停止与医学生交付

- seed42存在合格K：冻结唯一K和checkpoint，在同一冻结空间特征上仅重训SAE seed202/503；
- 3个SAE seed均通过：进入概念稳定性对齐、原型图、逐图多概念热图和医学生命名；
- 复现只有2/3或1/3通过：如实报告，不把单seed结果包装成稳定产品；
- seed42无合格K：停止S2c，Gated SAE如需尝试必须新建独立协议。

S2c是当前“严格重构型SAE”路线的终点实验，而不是继续扫描结构和门槛的起点：

- 若S2c成功，正式产品仍为通过八项门槛的最小K；随后进入SAE seed202/503复现、
  跨seed Feature对齐、概念家族、临床审核、来源/伪特征审计和残差保留干预；
- 若只有较大K通过，可将其解释为`K_reconstruction`（保真预算），但不得在本轮另选较小
  `K_concept`。面向医生的较小概念预算必须在S2c之后建立独立解释协议；
- 若S2c失败，保留失败结论，不追加K、不修改`0.90`门槛、不临时加入Gated SAE、
  不继续调`gamma_pool`。后续只能另立更合适解释层、Gated SAE、概念/重构粒度分离或
  更大更多样训练表示等新研究问题；
- 无论成功或失败，后续研究重点均从SAE重构器调优转向Feature的跨患者/跨seed稳定性、
  医学语义一致性、空间可定位性、来源混杂和对模型决策的因果贡献。

最终医学生交付不只提供技术指标，而必须包含：通俗阅读指南、模型保真表、癌侧富集/非癌侧
富集/两类共有/疑似伪特征四类概念表、每概念多患者原型图、逐图多色概念热图、
Feature置零后的概率/margin变化，以及供医学生填写“病灶/正常结构/反光/气泡/器械/无法判断”
的人工审核表。癌与非癌共有Feature不自动删除，需结合空间位置和置零干预判断其临床含义。

## S2c seed42正式结果（2026-08-21）

正式运行完成1000 epoch，按五层联合val总损失选择的最佳checkpoint为epoch 998。输出：

`结果/SAE/CLong_S2c_Matryoshka_20260821/s2c_clong_seed42/`

状态为`no_product_stop_s2c`，`selected_k=None`。逐K核心结果如下：

| K | 重构患者AUC | AUC下降 | pooled cosine | 冻结阈值一致率 | 翻转患者/260 | recovered CE | 未通过门槛 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 0.90898 | 0.00116 | 0.92236 | 0.93846 | 16 | 1.00178 | 一致率 |
| 128 | 0.90906 | 0.00108 | 0.95410 | 0.93077 | 18 | 0.99978 | 一致率 |
| 256 | 0.90805 | 0.00209 | 0.96926 | 0.94231 | 15 | 0.98724 | 一致率 |
| 512 | 0.90890 | 0.00124 | 0.97383 | 0.94231 | 15 | 0.98268 | 一致率 |
| 1024 | 0.90960 | 0.00054 | 0.97526 | 0.94231 | 15 | 0.98604 | 一致率 |

原始C-long患者AUC为0.91014。五个K的AUC下降、pooled cosine、recovered CE、五层并集
死亡率、非死亡重复率、normalized AiB下降和PGA下降均通过冻结门槛；五层并集死亡率与
重复率均为0。每个位置的实际L0等于对应K，没有正激活不足。

### 正式判定

Matryoshka结构已经解决S2b的主要方向保真问题：最低K=64的cosine已达到0.92236，明显
超过冻结下限0.90；增大K后cosine最高达到0.97526，患者AUC下降始终不超过0.0021，空间
指标也保持稳定。因此本轮不是字典死亡、重复、空间注意力崩坏或AUC明显退化。

唯一失败项是冻结患者阈值一致率。不同K仍有15–18/260位患者跨过原冻结阈值；即使K从
256增加到1024，一致率仍停在0.94231，没有达到0.95。这说明残余问题主要是阈值附近患者
对微小概率变化敏感，而不是继续增加K即可稳定解决。冻结协议不允许因此放宽门槛。

结论：**S2c按预注册失败，严格重构型SAE路线正式停止。** 不选择“最接近”的K，不运行
SAE seed202/503，不启动跨seed Feature family正式流程，也不追加K、不修改0.95/0.90门槛、
不继续调`gamma_pool`。这些结果可作为结构诊断证据保留，但不能包装成稳定SAE产品交给
医生正式命名。后续若继续解释研究，必须另立Gated SAE、更合适解释层或概念粒度与重构
粒度分离等新的预注册问题。

## RP-A预注册草案：残差保留解释探针的技术确认（待确认，未冻结）

状态：**本节是新研究问题的可执行草案，不是S2c修订，也尚未冻结。** 在本节所有`TBD`
参数、校准函数、实现自测和输出血缘完成审阅前，不得运行seed43/44/202/503/911。

### 研究问题与S2c边界

S2c已经否定“SAE重构可作为C-long表示的严格替代产品”达到冻结标准；该失败结论保持不变。
RP-SAE改问：在正常预测始终使用原C-long特征的前提下，Matryoshka SAE能否提出跨患者、
跨独立SAE初始化稳定的稀疏Feature，并通过后续临床反证和残差保留干预显示解释价值。

RP-A只确认技术稳定性，不进行医生命名、正式Feature family合并或正式干预结论。S2c的
`agreement >= 0.95`保留为历史产品门槛与描述性诊断，不再作为RP-A成功门槛。

### 固定表示与残差定义

解释对象仍为C-long `features[8]`输出的`F in R^(49x1280)`。开发选择暂定：

```text
SAE结构：S2c同一10240宽Matryoshka patch SAE
K_coverage：1024（每位置最多1024个正激活Feature）
N_review：后续临床审核候选数量，不是Top-K；RP-A不冻结具体数量
```

`K_coverage=1024`只用于新问题中定义高覆盖SAE分量，不是S2c正式产品，也不改变S2c失败。
每个patch位置`p`定义：

```text
r_p = F_p - D(h_p)
F_p = D(h_p) + r_p
F'_p = F_p + D(h'_p) - D(h_p)
```

正常预测始终使用原始`F`。只有干预时构造`F'`；正式路径重新计算冻结C-long attention、
attention pooling、logit和概率，固定原attention只作机制分解诊断。

### 训练损失与gamma_pool血缘

RP-A所有新SAE seed继承S2c正式校准的同一个`gamma_pool`，不逐seed重校准：

```text
gamma_pool_policy = inherit_s2c_seed42
gamma_pool_value = 0.5095280077324069
gamma_pool_source_json = 结果/SAE/CLong_S2c_Matryoshka_20260821/gamma_pool_calibration_seed42.json
gamma_pool_source_json_sha256 = 3e427764415873cb5a3e6d7f8259350c96a968f48e82bd31c37a030b1dc7f35f
gamma_pool_source_initial_state_sha256 = 26848587ae80438c92ba64cb0d6eec794a5e90f9f8ff33819f568dca034359db
per_seed_recalibration = false
```

训练入口必须从上述JSON读取全精度值并核验协议、manifest、C-long学生、空间缓存和校准JSON
SHA；禁止CLI手工覆盖。每个新seed仍须保存自己的初始化state SHA，但不得据此重新校准
`gamma_pool`。这样跨seed实验只改变SAE初始化，不混入seed-specific loss calibration。

### 全部SAE激活分量消融：纯诊断

RP-A对train/val额外报告`all-latent ablation`，令所有latent为0：

```text
F'_p = F_p + D(0) - D(h_p)
```

若decoder含bias，则该差分形式会抵消bias对应的实现歧义，但结果不命名为“裸residual-only”。
它回答“SAE显式激活分量整体承载了多少可干预决策信号”，不是单Feature/family因果效应的
严格上限，也不假定非线性attention重算后的效应可加。

必须同时报告image/patient的margin与probability变化、AUC、冻结阈值预测变化、attention
KL/cosine；重算attention为主结果，固定原attention只作内容机制诊断。config固定写入
`all_latents_zero_diagnostic_only=true`，该结果不参与RP-A PASS/FAIL、Feature匹配、阈值、
K或seed选择。输出JSON必须把两条路径分别记录为`attention_mode=recomputed`和
`attention_mode=fixed`，不得共用或覆盖同一个`delta_margin`字段。

### seed角色与顺序

| seed | 角色 | 允许用途 |
| ---: | --- | --- |
| 42 | development | 已完成的S2c开发证据和新假设来源，不计确认成功数 |
| 43、44 | development-calibration | 仅用开发阶段技术统计校准匹配算法与实际稳定性门槛，不进入正式确认、临床结论或正式干预结论 |
| 202、503 | confirmation | 规则冻结后运行，必须2/2通过RP-A；不得重估阈值 |
| 911 | held-out SAE initialization replication | 仅在RP-A通过且RP-B/RP-C全部冻结后运行；不是独立数据或新患者验证 |

若202或503任一失败，RP-SAE停止，不运行911。911不得修改匹配阈值、family规则、
`N_review`、概念名单、医生panel、干预剂量、随机对照或成功标准。

### eligible Feature定义

以下规则及数值必须在运行43/44前冻结：

- 非死亡：train上激活超过`ACTIVE_EPS=1e-8`；
- 最低患者覆盖`P_min_train=25`，由冻结`q=0.02`与1212位train患者生成；
- 最低激活位置频率`A_min=0.25/49=0.00510204081632653`；
- Top患者规则冻结为全split患者的最高`q=0.02`，train取25人、val取6人；
- decoder按冻结单位范数约定比较；
- 所有激活统计先按患者聚合，避免多图患者获得更大权重；
- patch到image、image到patient的聚合不能只写一个笼统函数。必须分别冻结
  `presence/coverage`、`ranking`和`activation mass`三种用途的公式；候选可包括49位置
  max、均值、Top-q均值或C-long attention加权，但不得在看到seed43/44后选择；
- 同一患者多图的聚合方式也须逐用途冻结，并保存分母、缺失图像和并列处理规则。

eligible规则不得在看到43/44低频Feature稳定性后修改，否则`R_feature`分母可被人为改变。

### 激活聚合公式：已冻结（2026-08-21）

本小节已通过正式审计/校准JSON、代码SHA、协议SHA、9/9产物SHA和测试验收；
后续seed不得重估聚合公式或数值。
对患者`u`的第`i`张图、
49位置中的`p`和Feature `j`，令`h_uipj >= 0`为`K_coverage=1024`激活，且：

```text
z_uipj = 1[h_uipj > ACTIVE_EPS]
ACTIVE_EPS = 1e-8
```

三种用途不共用一个万能分数：

| 用途 | patch到image | image到patient | 后续用途 |
| --- | --- | --- | --- |
| presence | 任一位置`h>eps` | 任一图present | `P_min`与患者覆盖 |
| ranking | `max_p h_uipj` | `max_i image_score_uij` | Spearman、Top患者与临床原型 |
| activation mass | `mean_p h_uipj` | `mean_i image_mass_uij` | `R_activation_reference/confirmation` |

具体定义为：

```text
image_presence_uij   = 1[max_p h_uipj > eps]
patient_presence_uj  = 1[max_i image_presence_uij = 1]
patient_coverage_j   = sum_u patient_presence_uj

active_frequency_uj  = mean_i mean_p z_uipj
active_frequency_j   = mean_u active_frequency_uj

image_rank_uij       = max_p h_uipj
patient_rank_uj      = max_i image_rank_uij

image_mass_uij       = mean_p h_uipj
patient_mass_uj      = mean_i image_mass_uij
dataset_mass_j       = sum_u patient_mass_uj
```

presence使用患者任一图出现，避免只在一张病灶图出现的概念被其他图平均掉；同时用患者平衡
的`active_frequency_j`识别“很多患者偶尔蹭到一次、但实际位置频率极低”的Feature。ranking
服务最强局部证据，activation mass服务总体使用量；多图患者在mass中仍只贡献一个患者均值。

Top-patient按`patient_rank_uj`取全split患者的最高`q%`。候选集固定为
`Q={0.01,0.02,0.05,0.10}`，在seed42完整train全部非死亡Feature上选择满足非零患者支持的
最大q。正式train-only审计得到`q=0.02`、train人数`ceil(0.02*1212)=25`。
边界同分值最终以原始稳定`patient_id`升序打破，
不使用随机数。任何split正激活患者不足Top-q人数时，`top_q_behavior_evaluable=false`，不得
用零分患者补满；val对应人数为`ceil(0.02*260)=6`。

2026-08-21正式审计同时更正一处早期口径混淆：患者覆盖率的第1百分位为
`14.39%`（约174人），但这不是最小值；10240个 full-train non-dead Feature的
实际最小覆盖为`53/1212=4.37%`。由于冻结函数要求候选q对所有基础
Feature都不混入零激活患者，`q=0.05`不安全而`q=0.02`通过。这是执行
预先写死的函数所得结果，不是事后改规则。

跨seed患者Spearman只在union-positive患者集合上计算：双方同时为0者排除，单方为0者保留，
双方为正者保留。Top-q仍以全split患者数为分母，不能改为union-positive人数。ranking是
ordinal score，只允许用于Spearman、Top-q和原型排序；禁止跨Feature解释绝对幅度，也禁止
进入activation mass或representation energy。患者图数对max-ranking的共同偏差预计可被
同患者结构的分层null部分吸收，但不表述为已消除，并须保留ranking与患者图数相关性审计。

### 同图49位置空间复现：正式冻结（2026-08-24）

对同一输入图像，比较seed A Feature `j`和seed B Feature `k`的两个原始非负49维激活图；
不做min-max、softmax、减均值或单位方差标准化，只计算raw-map cosine：

```text
image_active = 1[max_p h_p > ACTIVE_EPS], ACTIVE_EPS=1e-8
both active:       spatial_image = cosine(raw h_A[49], raw h_B[49])
exactly one active: spatial_image = 0
both inactive:      spatial_image = NA

union_active_images_u = 该患者中至少一侧非零的图像
spatial_patient_u = mean(spatial_image over union_active_images_u)
spatial_pair = mean(spatial_patient_u over valid patients)
```

`ACTIVE_EPS`只判断整张激活图是否active，不把49维原始图中低于eps的元素再次置零。一侧active而
另一侧inactive记0，用于惩罚漏响应；两侧都inactive不能证明空间复现，故记NA而非1。
患者内先平均、患者间再平均，防止图多患者获得更大权重。只在双方共同激活图上计算的cosine
可在未来作为diagnostic候选，但不替代上述主候选。正式输出另存`n_evaluable_patients`、
`n_union_active_images`、`fraction_one_side_zero`和`fraction_both_active`，用于区分“双方非零
但位置正交”和“一侧未响应”。bootstrap中字段名固定为`n_evaluable_patient_instances`，所有图像
计数均按multiplicity展开；`fraction_one_side_zero + fraction_both_active = 1`，仅作诊断。

最低支持度直接继承已冻结的Top-q支持：train为25位患者，bootstrap为25个patient instances，
val为6位患者。患者无union-active图像则患者为NA；pair低于对应支持度则`spatial=NA`且candidate
invalid。支持度足够但所有患者值均为0时，0是合法的完全空间不一致证据，不得改成NA。

数值路径冻结为：SAE激活和49维图像cosine使用float32，关闭TF32；图像到患者、患者到pair及
bootstrap multiplicity加权使用float64累积，最终spatial保存为float64。不裁剪到`[0,1]`、不舍入、
不把小值强制置零；有效位置出现NaN/Inf或超出理论`[0,1]`时fast-fail。

固定数值审计只读取seed42 train缓存的前32位排序患者（85图），用`PCG64(20260824)`无放回选择
96个Feature并分成48×48 pair；不输出best Feature、edge、anchor、coverage、p值或门槛。相对CPU
float64 reference，全float32与混合精度的NA/finite语义均完全一致；最大绝对误差分别为
`3.5232e-8`与`7.9919e-9`，预热后中位耗时分别为`0.001575s`与`0.001568s`，混合精度只增加
`331264 bytes`峰值显存。因此正式采用混合精度路径。正式审计为
`结果/SAE/RP_A_Spatial数值审计_20260824/numerical_audit_v2.json`，SHA256为
`d485b397e6c44935410befcdabbb0b821565c5fedeacfff964eef75e2405867b`；首轮未预热计时产物仅为历史
调试记录，不参与冻结证据。

冻结实现与测试为`rpa_spatial_protocol_v1.json`、`clong_rpa_spatial.py`、
`audit_clong_rpa_spatial_numeric.py`和`test_clong_rpa_spatial.py`，6项语义测试通过；完整SHA见
`程序/SAE/正式代码/rpa_spatial_SHA256SUMS.txt`。本子协议只冻结spatial数值与可评价规则，不授权
启动新seed或正式matching。

### seed42 train-only聚合分布审计：输出先冻结

审计只读取seed42正式checkpoint和train空间缓存，不读取val、internal test、external，不训练
也不改变任何Feature。每个Feature只允许固定输出：

```text
feature_id
patient_coverage_count / patient_coverage_fraction
active_position_frequency
patient_rank q50/q75/q90/q95/q99/max
activation_mass total / patient_q50/patient_q90/patient_q99
positive_patient_count
top_q_nonzero_count（仅针对预先列明的候选q）
ranking_score与患者图数的Pearson相关（只作max聚合偏差审计）
representation_energy（定义已冻结，正式实现前保持字段预留）
```

总体只输出`patient_coverage`、`active_position_frequency`、patient ranking和activation mass
分布，以及患者覆盖率超过0.25/0.50/0.75/0.90的Feature比例。审计脚本、候选q列表、输入SHA、
输出schema和自动选值函数必须在执行前完成审阅；不得先生成更多统计再事后挑选。当前不引入
“大于Feature自身某分位数才算激活”的显著激活阈值。若presence严重饱和，只能按审计前
写死的不可识别规则处理，不能在看到43/44后临时更换定义。

### active_frequency_calibration_spec：已冻结（2026-08-21）

基础Feature universe固定为seed42完整train五层激活并集中的非死亡Feature：

```text
J_full_nondead = {j: Feature j在seed42完整train五层并集激活上为non-dead}

G_A = {0.25/49, 0.5/49, 1/49, 2/49}
B_split = 400
Q_low = 0.05
quantile_impl = numpy.quantile
quantile_method = lower
T_J = 0.80
T_rho = 0.90
N_E_min = 100
N_anchor_min = 100
```

`B_split=400`是预注册计算预算：经验CDF步长为0.0025，5%下尾取非插值的经验观测值；不把
400次高度重叠的分半解释为独立实验，也不声称分位数具有固定标准误。所有分位数使用
`numpy.quantile(..., method="lower")`，并记录NumPy版本。

每次按患者在`label x source` stratum内确定性分半，患者全部图随患者移动。禁止Python内置
`hash()`；UTF-8输入字段使用明确分隔符和十进制split seed，分别计算：

```text
patient_key_sha = SHA256(
  "patient_order" | protocol_sha | split_seed | stratum_id | patient_id
)
offset_sha = SHA256(
  "stratum_offset" | protocol_sha | split_seed | stratum_id
)
offset = int(offset_sha, 16) mod 2
half = (zero_based_rank_after_patient_key_sort + offset) mod 2
```

`stratum_id`和`patient_id`使用manifest稳定原始值的规范化UTF-8序列，不依赖Python隐式类型
格式化。每个stratum须满足`abs(n_A-n_B)<=1`；每次保存各stratum和全局A/B人数。split seed
固定为`BASE_SPLIT_SEED + b, b=0,...,399`，其中`BASE_SPLIT_SEED=20260821`；该值写入静态协议JSON并纳入文件SHA，不得修改。

每个half `s`与候选`g`定义：

```text
E_s(g) = {
  j in J_full_nondead:
  positive_patients_j,s >= ceil(0.02 * N_s)
  AND A_j,s >= g
}
```

每次计算`Jaccard(E_1,E_2)`、在`E_1 union E_2`上的
`Spearman(A_half1,A_half2)`及`min(|E_1|,|E_2|)`。若并集为空、少于2个Feature、任一侧为
常数或出现NaN/Inf，则该`g`整体FAIL，不能删除该split后继续。

400次后计算：

```text
J_low   = Q0.05_lower(J)
rho_low = Q0.05_lower(rho)
N_low   = Q0.05_lower(min_set_size)

PASS(g) iff J_low >= 0.80 AND rho_low >= 0.90 AND N_low >= 100
A_min = 按数值升序最小的PASS g
```

若无候选通过，状态为`active_frequency_calibration_infeasible`，不扩网格、不降低门槛、不换
分半算法。`N_E_min=100`只是建立100个一一anchor所需的最低必要Feature池，不保证会形成
100个anchor；开发阶段仍独立要求`N_anchor_3of3>=100`，否则
`development_calibration_infeasible`。这些阈值是本项目的预注册实际复现/规模门槛，不包装
为通用SAE标准或100个独立Feature样本。

### eligible工程实现状态（2026-08-21）

已新增静态协议`rpa_eligible_protocol_v1.json`、正式入口
`clong_rpa_eligible.py`、启动器`run_clong_rpa_eligible.sh`和回归测试
`test_clong_rpa_eligible.py`。实现分为两个明确阶段：

1. `audit`：从S2c seed42正式checkpoint在K=1024视图上逐图编码，按患者
   产出presence/ranking/mass/active-frequency四个矩阵，然后通过冻结函数产出
   `q*`、`P_min_train`和逐Feature审计CSV；
2. `calibrate`：重验protocol、代码、审计CSV和患者矩阵SHA后，执行400次
   确定性患者分半，产出1600行原始折结果、四个g候选的下分位数和
   唯一`A_min`或不可校准状态。

单元测试覆盖SHA域分离、奇数stratum offset、每层人数平衡、Top-q选择、
`Q0.05 lower`、Spearman并列/常量边界、固定non-dead universe和审计schema。
CPU真实checkpoint debug已端到端跑通；小样本返回
`active_frequency_calibration_infeasible`符合门槛定义，不解读为正式结果。

### eligible正式审计与校准结果（2026-08-21）

正式任务已读取2350张train图、1212位train患者和10240个 full-train
non-dead Feature；未读取val激活、internal test或external。主要结果为：

| 项目 | 正式结果 |
| --- | ---: |
| `q*` | 0.02 |
| `P_min_train=ceil(q*1212)` | 25 |
| val Top-q固定数`ceil(q*260)` | 6 |
| full-train non-dead Feature | 10240 |
| 冻结eligible Feature数 | 9418 |
| `A_min` | `0.25/49 = 0.00510204081632653` |

四个`g`候选的400次split均可计算且全部通过三项门槛，按预注册选取最小合格值：

| `g` | Jaccard Q0.05 lower | Spearman Q0.05 lower | min eligible Q0.05 lower | PASS |
| ---: | ---: | ---: | ---: | --- |
| 0.25/49 | 0.9717 | 0.9779 | 9377 | 是 |
| 0.5/49 | 0.8837 | 0.9648 | 5676 | 是 |
| 1/49 | 0.9478 | 0.9975 | 2144 | 是 |
| 2/49 | 0.9890 | 0.9983 | 1629 | 是 |

正式血缘：

```text
protocol SHA = 3bc1bed29b8fb8ba0c9118438aa8d299f52f2af4c1e67fd337aa0a1ef7443a91
code SHA     = 6b6192dbddf58ea6e159dade3f7f8868f96ecd308d5bb12e092cc28ed8faf69c
S2c checkpoint SHA = 1abf1c2366fed3aa94d70dce7e52df2c1c21001d8ebcded221ae9665ebc1c15f
audit summary SHA  = 2b16886e898c09062ee80b0d0eaa6be068aa3a07def5cb9d5419d3c2f013f5ba
feature audit CSV SHA = b354c8491e47fbb0c6646cccf5c0946cfab1bd3cc1d412ffb37c9b2cd2f7c172
split results CSV SHA = 9ea6733bd09906162227e95de25830aa5bbdcaf6dd84533d61b5f384045125f3
```

`SHA256SUMS.txt`已对全9个正式产物复验通过。至此eligible子协议正式
冻结；后续seed只能使用上述数值和SHA绑定规则，不得重新估计。bootstrap与spatial子协议已于
2026-08-24随后完成冻结；RP-A整体目前仍因GPU identity和正式产物协议未关闭而未冻结。

### 跨seed边匹配算法

每个seed-pair分别运行；同一字典内部pair不能代替跨seed null。以下规则已实现为
`rpa_null_fdr_protocol_v1.json`协议已于2026-08-24通过复审并正式冻结。本子协议的统计方法与实现
不再调整；RP-A整体仍未冻结，因此仍不允许启动新seed。

1. 对每个有向source eligible Feature，在另一seed的全部valid target eligible Feature中执行
   完整搜索；target分层只用于条件null，不限制候选搜索范围。
2. 四项原始指标为带符号decoder cosine、union-positive患者Spearman、Top患者Jaccard和
   同图49位置空间复现。任一best pair原始指标非有限值或`<=0`，该有向假设直接记`p=1`。
   空间复现只比较同一输入图像的相同49位置，不把跨患者7x7绝对位置解释为解剖配准。
3. train中每个source行的四项指标分别按全部train-valid target计算
   `(#train<x + 0.5*#train==x)/N`经验中秩百分位。定义
   `U_behavior=median(U_spearman,U_jaccard,U_spatial)`，再定义
   `S_edge=min(U_decoder,U_behavior)`；因此结构和行为任一侧偏低都会限制边分数。
4. best target按`S_edge`降序、原始带符号decoder cosine降序、target Feature ID升序唯一确定。
5. null不是随机单pair，也不使用有限次Monte Carlo置换。它在每个target stratum内，条件保持
   decoder百分位和整块behavior分数的边际分布，只随机二者对应关系。对观测阈值`s`，每层令
   `a=#(U_decoder>=s)`、`b=#(U_behavior>=s)`、候选数为`n`，则该层没有高-高重合的精确概率为
   `C(n-a,b)/C(n,b)`；各层相乘后取`1-product`，即完整候选搜索
   `P_null(max S_edge>=s)`的闭式右尾p值。故`B_null`不存在，也没有有限置换导致的p值下限。
6. target分层采用target seed的train eligible Feature，先按
   `(patient coverage, active frequency, Feature ID)`等频分4组，再在每组内按
   `(active frequency, patient coverage, Feature ID)`等频分4组，形成确定性平衡16层。
   val沿用train层归属，不重新分层。每个非空层至少32个且target总数至少512；否则该seed-pair
   判`null_stratification_infeasible`，禁止合并、重分箱或删掉不利层。seed42正式9418个
   eligible Feature预检得到每层588--589；此前“两个绝对四分位直接交叉”产生5个Feature小层，
   已在冻结前被预检否决，不属于正式协议。
7. 每个无序seed-pair的BH假设族包含两个方向全部eligible source Feature各自唯一的
   best-candidate假设，固定`q_FDR=0.05`。排序并列依次按p值、source seed、source Feature ID、
   target Feature ID升序处理。
8. 正式顺序固定为`完整搜索 -> 唯一best -> 精确p -> seed-pair内BH-FDR -> reciprocal nearest
   neighbour -> 一一性断言`。只有双方向都被BH拒绝且best互指才成边；由于best已唯一，RNN后
   若仍出现一对多属于实现错误，不做贪心冲突消解。
9. val固定train Feature universe、eligible IDs、target strata、每个source行的train指标CDF和
   精确null函数。val只重算患者依赖的行为指标，并用
   `(#train<x + 0.5*#train==x)/N`映射到冻结train CDF。`train_valid`只决定train reference，
   `val_valid`只决定val query；改变val不可评价候选不得改变train CDF。禁止val重新置换、
   重估null或改分层。

闭式p值已用小规模全排列穷举核验；strata还必须同时满足16层完整、target总数至少512且
每层至少32。经验CDF、独立valid mask、非有限值拒绝、并列、BH和RNN共20项纯函数测试通过。
2026-08-24仓库级审阅发现并修复了“train/val共用valid mask”、train/val tie convention相差
半个秩和strata硬断言不完整三项问题；最终复审进一步将train中秩函数的valid非有限输入从
静默NaN改为fast-fail，exact-null数学主体全程未改变。协议状态、代码与测试SHA已固结，
后续实现必须按该版本执行。

### development technical reference

先在train构建`42<->43`、`42<->44`、`43<->44`三组冻结技术匹配图。主technical anchor
要求同一组三个Feature分别来自42/43/44，且三条seed-pair边全部成立，即严格3-clique；只有
连通链而缺少一条边不能算3/3 anchor。2/3结果仅作诊断。

confirmation Feature要匹配某个anchor，须至少与三个开发成员中的两个分别通过冻结边规则。
同一confirmation Feature和同一anchor不得重复计数；多个Feature竞争同一anchor时，采用
预先冻结的一一分配算法。`202<->503`只作一致性诊断，不增加或替代主成功门槛。

开发报告必须把参考规模作为headline metric，至少包括三个seed的eligible数、三组pair边数、
`N_anchor_3of3`、anchor数量覆盖、activation覆盖和energy覆盖。运行43/44前冻结最小可用参考
规模`N_anchor_min=100`；若严格3-clique anchor少于该值，状态为
`development_calibration_infeasible`并停止当前RP-A设计，不能自动降级为2/3 anchor，也不能
把“校准不可行”误写成confirmation失败。

### 实用稳定性指标

对于两个普通seed A/B，eligible数为`N_A/N_B`、一一匹配数为`M`：

```text
R_feature = 2M / (N_A + N_B)

R_activation_symmetric_diagnostic = 0.5 * (
    activation_mass_A_matched / activation_mass_A_eligible
  + activation_mass_B_matched / activation_mass_B_eligible
)
```

该普通pair对称activation均值仅作描述性诊断，不参与confirmation PASS或绝对门槛校准。

### Representation component energy：指标定义已冻结（2026-08-24）

固定使用`K_coverage=1024`的正式激活。对患者`u`、图像`i`、49个位置中的位置`p`和
Feature `j`，`h_uipj>=0`为激活，`d_j`为decoder第`j`列。单个位置的显式解码分量和平方范数为：

```text
c_uipj = h_uipj * d_j
e_uipj = ||c_uipj||_2^2
```

正式Feature energy采用患者平衡的均值聚合：

```text
E_j = mean_u mean_i mean_p ||h_uipj * d_j||_2^2
```

计算范围包含当前split全部患者、全部图像和全49个位置；零激活精确贡献0。decoder bias
不属于任何Feature，不计入；连续energy不应用`ACTIVE_EPS`二值截断。activation mass与
representation energy都采用患者平衡聚合；现有activation mass的dataset-level原始量是患者
质量求和，energy则规范存储为患者均值。在固定split内二者的sum/mean常数因子不改变覆盖比例。

`representation_energy`衡量SAE Feature显式decoder component `h_j*d_j`的平方范数质量。decoder
directions不要求正交，因此不同Feature可包含相关或重叠方向；各Feature energy之和不是
重构向量总能量的正交分解，也不等价于explained variance、Feature importance或classification
contribution。稳定Feature的energy coverage只表示其覆盖了eligible dictionary中多少显式
decoder-component magnitude。真正分类贡献仅在RP-C中用残差保留干预的`delta margin`定义。

对开发seed `s in {42,43,44}`，eligible Feature集合为`eligible_s`，严格3/3 anchor集合为`A`，
anchor `a`在seed `s`中的成员为`j_s(a)`。开发anchor energy coverage先在每个seed内计算：

```text
C_anchor_energy_s = sum_{a in A} E_{s,j_s(a)} / sum_{j in eligible_s} E_{s,j}
C_anchor_energy_mean = mean_{s in {42,43,44}} C_anchor_energy_s
```

`C_anchor_energy_mean`是development reference描述量，不扩展为新的PASS综合分。不允许先将不同seed的
raw energy相加后再计算一个总ratio，以免绝对激活尺度较大的seed获得更高权重。

confirmation相对3-member anchor不强行套用普通pair的对称`R_feature`，至少分别报告：

```text
R_anchor_recall = 被confirmation复现的eligible 3/3 anchors / 全部eligible 3/3 anchors
R_confirm_coverage = 匹配到anchor的eligible confirmation Features / eligible confirmation Features
```

### Activation mass coverage：指标定义已冻结（2026-08-24）

继续使用已有患者平衡activation mass，不修改历史定义：

```text
M_uij = mean_p h_uipj
M_uj = mean_i M_uij
M_j = sum_u M_uj
```

对正式confirmation seed `c`，被复现的冻结development anchors为`A_c_reproduced`，
confirmation seed中匹配到anchors的Feature集合为`M_c_matched`。两侧指标冻结为：

```text
R_activation_reference_c = mean_{s in {42,43,44}} (
    sum_{a in A_c_reproduced} M_{s,j_s(a)}
    / sum_{a in A} M_{s,j_s(a)}
)

R_activation_confirmation_c =
    sum_{j in M_c_matched} M_{c,j}
    / sum_{j in eligible_c} M_{c,j}
```

reference-side必须先在每个development seed内计算anchor activation mass ratio，再对42/43/44等权平均；
禁止先合并三个seed的raw activation mass。confirmation-side使用confirmation seed自身全部eligible
Feature mass作分母。两侧必须分别进入PASS：

```text
R_activation_reference >= T_activation_reference
AND
R_activation_confirmation >= T_activation_confirmation
```

原对称均值仅保留为`R_activation_symmetric_diagnostic`，固定
`diagnostic_only=true`、`used_for_pass=false`、`used_for_threshold_calibration=false`。

energy的两侧指标正式冻结为：

```text
R_energy_reference_c = mean_s (
    reproduced_anchor_energy_in_seed_s / all_frozen_anchor_energy_in_seed_s
)

R_energy_confirmation_c =
    matched_confirmation_feature_energy / all_confirmation_eligible_feature_energy
```

reference-side先在每个development seed内计算被confirmation复现的anchor energy比例，再对
42/43/44等权平均；其分母是全部冻结development anchors的energy，而不是全部eligible
Features。confirmation-side的分母则是该confirmation seed全部eligible Feature energy。两者回答
不同问题，必须分别进入PASS：

```text
R_energy_reference >= T_energy_reference
AND
R_energy_confirmation >= T_energy_confirmation
```

原对称均值仅允许以`R_energy_symmetric_diagnostic`保留于表格，固定
`diagnostic_only=true`、`used_for_pass=false`、`used_for_threshold_calibration=false`。开发伪确认与正式确认必须
使用同名、同方向指标，不能在确认阶段临时改回对称pair公式。

### 校准函数：先冻结算法，再由43/44产生数值

仅用三个普通开发pair直接生成门槛，与confirmation相对3成员anchor的评价对象不同。
因此草案采用三次leave-one-development-seed-out伪确认：

```text
用43/44构建2/2 anchor，42作为伪确认seed
用42/44构建2/2 anchor，43作为伪确认seed
用42/43构建2/2 anchor，44作为伪确认seed
```

每个伪确认Feature必须同时匹配2/2 anchor成员。该过程与未来202/503匹配3/3 anchor时采用
相同的candidate search、行为门控、FDR和一一分配主体，只把“至少2个开发成员”固定为共同
判据。这样开发门槛与确认评价处于同一统计口径。

每一折分别输出`R_energy_reference`和`R_energy_confirmation`。伪确认的reference只有两个seed，
因此reference-side energy的分母必须是该折全部冻结2/2 anchors的energy；不得使用未来完整
3/3 anchor分母。未来202/503正式确认则使用42/43/44严格3/3 anchors，并以匹配至少2/3
开发成员定义成功复现；两个阶段的指标语义保持不变。

对伪确认折`f`，两个reference seeds为`S_f`，该折冻结2/2 anchors为`A_f`，被held-out seed复现的
anchors为`A_f_reproduced`。reference-side energy必须先在每个reference seed内计算ratio，再对两个seed
等权平均：

```text
R_energy_reference_f = mean_{s in S_f} (
    sum_{a in A_f_reproduced} E_{s,j_s(a)}
    / sum_{a in A_f} E_{s,j_s(a)}
)
```

禁止将两个reference seed的raw matched energy和raw total energy各自相加后内嵌为一个总ratio。
该规则与正式3-seed confirmation的“逐seed ratio后等权平均”原则一致。

activation使用完全同构的伪确认口径：

```text
R_activation_reference_f = mean_{s in S_f} (
    sum_{a in A_f_reproduced} M_{s,j_s(a)}
    / sum_{a in A_f} M_{s,j_s(a)}
)

R_activation_confirmation_f =
    sum_{j in M_f_matched} M_{f,j}
    / sum_{j in eligible_f} M_{f,j}
```

同样禁止将两个reference seeds的raw activation mass合并后计算总ratio。activation与energy均只冻结
两列伪确认输出和两个独立门槛接口，门槛数值由后续bootstrap子协议分别产生。

energy子协议只冻结两列伪确认输出和两个独立绝对门槛的接口：

```text
pseudo_confirmation_outputs:
- R_energy_reference
- R_energy_confirmation

thresholds:
- T_energy_reference
- T_energy_confirmation

combined_threshold: none
```

两个门槛由后续冻结的patient-level bootstrap calibration protocol分别生成。energy协议现在不决定
`Q_0.05`、bootstrap次数、完整重匹配还是固定图，也不提前写入任何门槛数值。

患者bootstrap使用同一批有放回抽样患者同时重算三个伪确认折。必须在冻结前决定是每个
bootstrap从eligible、行为统计、null percentile、匹配边到anchor全部重算，还是固定全train
匹配图只重算质量覆盖；若固定匹配图，则`R_feature`不会变化，不能伪装成其bootstrap区间。
当前推荐候选是完整重算matching pipeline，因为它能为结构复现率提供真实抽样不确定性。
2026-08-24已完成正式规模纯计算benchmark，证明精确完整重算在现有服务器上无OOM
且可执行；该选择因此从计算可行性上通过，但仍须与抽样/RNG/并行归并等条款
一起完成bootstrap子协议冻结后才能运行。

### Bootstrap source只读审计与结构失败口径（2026-08-24）

bootstrap分层字段固定为S0冻结train manifest的原始`source`列，不使用`center`或其他派生字段。
只读审计前已复验manifest SHA为
`b1dfd24505cbba876fd28502e36b0b190206de8f063562f47347dc7c67f32e33`，train为2350图/1212人。
原始字面类别及患者数为：

| label | source | 患者数 |
| ---: | --- | ---: |
| 0 | 武大省人民 | 830 |
| 0 | 第一届早癌大赛 | 32 |
| 0 | 第二届早癌大赛 | 38 |
| 1 | 武大省人民 | 261 |
| 1 | 第一届早癌大赛 | 19 |
| 1 | 第二届早癌大赛 | 32 |

`source`和`label`均无缺失；1212名患者均唯一对应一个`label`和一个`source`，无患者内
冲突。bootstrap必须直接使用上述6个`label x source` strata的原始类别；禁止众数回填、
重命名、合并稀有类别、拆分类别或用`center`替代。

结构失败口径固定为：一个bootstrap replicate中，若构建三次pseudo-confirm所必需的任一
seed-pair无法满足已冻结null strata条件（16层完整、target eligible Features总数至少512、
每层至少32），则整个replicate记为`replicate_structural_failure`，三折全部正式calibration
metrics统一记0：

```text
R_anchor_recall
R_confirm_coverage
R_activation_reference
R_activation_confirmation
R_energy_reference
R_energy_confirmation
```

不得删除该replicate、合并strata、重新分箱或放宽512/32条件。合法抽样产生的0 anchors或
0 reproduced anchors同样属于科学性低稳定，对应正式指标记0而不删除。若未重采样full train的
任一正式pseudo-confirm fold发生同类不可行，则直接记`development_calibration_infeasible`，
不启动bootstrap。

NaN/Inf进入有效集、ID或shape错位、重复Feature ID、RNN一对多等属于实现或数值错误，
必须fast-fail终止整个校准，不得记0继续。

### RP-A matching纯计算benchmark（2026-08-24）

benchmark只读seed42冻结SAE与train空间缓存，通过Feature维度的确定性双射构造
42/43/44三个逻辑seed。decoder行和activation列使用同一双射，不加噪声、不训练新SAE，
也不创造跨seed统计证据。surrogate replicate使用完整train且每位患者权重为1，
因此只回答同规模精确计算能否跑通和需要多少资源，不是bootstrap分布中的一个正式观测。

完整链路真实执行了K=1024空间编码、3个seed-pair的双向decoder cosine、union-positive
Spearman、Top-q Jaccard、same-image 49位置spatial similarity、row percentile、best candidate、
exact-null、BH-FDR、RNN和3个2/2 pseudo-confirm fold plumbing。候选矩阵按source block流式处理，
每块完成有向hypothesis后立即释放；未持久化edge、anchor、p值、coverage或任何`R_*`。

| 实现 | source block | 单replicate | 百分位+选best | spatial | Spearman | exact-null | 峰值RAM | 峰值VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 冻结纯函数逐行基线 | 32 | 3828.0 s | 3472.7 s | 155.4 s | 121.1 s | 76.7 s | 3.13 GiB | 5.30 GiB |
| 等价批量中秩+best | 32 | 714.3 s | 358.9 s | 156.5 s | 121.2 s | 73.9 s | 3.20 GiB | 5.30 GiB |
| 等价批量中秩+best | 128 | **696.2 s** | 347.5 s | 122.3 s | 151.3 s | 73.9 s | 3.35 GiB | 6.20 GiB |

批量中秩与冻结`train_midrank_percentile()`逐行结果逐位等价；best选择与冻结的
`score desc -> cosine desc -> target ID asc`逐行结果一致，因此优化只改变实现效率，
不改变null/FDR数学定义。block 128相对block 32只再快2.5%，停止继续调块参数。

最终工程投影为：一次预计算约6.0秒，400次精确完整重算在单GPU串行下约
77.4小时。多GPU理想值不作承诺，因为百分位和exact-null还会占用CPU，且服务器GPU为共享资源。
`B_boot=400`在随后完成的bootstrap子协议复审中正式冻结；本benchmark本身仍只提供工程可执行性
证据，不能单独授权启动bootstrap或seed43/44。

### Bootstrap正式冻结协议与纯函数核心（2026-08-24）

基于benchmark，fixed graph从候选中删除；每个replicate精确重算eligible membership、
患者行为、空间相似度、百分位、exact-null、BH-FDR、RNN、anchor和pseudo-confirm六项指标。
正式静态协议为`rpa_bootstrap_protocol_v1.json`，状态为`frozen_2026-08-24`。

抽样由CPU单进程coordinator在worker启动前一次性生成。使用：

```text
B_boot = 400
RNG = numpy.random.Generator(numpy.random.PCG64(20260824))
strata order =
  (0,武大省人民), (0,第一届早癌大赛), (0,第二届早癌大赛),
  (1,武大省人民), (1,第一届早癌大赛), (1,第二届早癌大赛)
base order within stratum = patient_id string ascending
sampling = 每层有放回抽取该层原患者数次
```

因此每个replicate始终有1212个patient instances，六层始终为
`830/32/38/261/19/32`。同一份multiplicity plan同时传给三个seed和三个pseudo-confirm folds；
禁止各GPU自行产生随机数。

患者`u`被抽中`w_u`次等价于`w_u`个独立bootstrap patient instances，每个instance携带
该患者全部原始图像。实现使multiplicity加权而不复制大数组，但必须与显式展开逐位等价：

```text
positive patient instances = sum_u w_u * presence_u
active frequency = (1/1212) * sum_u w_u * A_u
activation mass = sum_u w_u * M_u
representation energy = (1/1212) * sum_u w_u * E_u
spatial = sum_{u valid} w_u*S_u / sum_{u valid} w_u
```

full-train non-dead Feature universe保持冻结，但`positive>=25 AND A>=A_min`在每个replicate重算。
presence和union-positive必须使用`ACTIVE_EPS=1e-8`；benchmark中的`>0`只是纯工程surrogate简化，
不得进入正式bootstrap kernel。

Top-25对bootstrap instances排序，顺序固定为`ranking score desc -> patient_id asc -> occurrence_index asc`。
同一患者若被抽中多次，可在Top-25中出现多次；禁止按patient ID去重。Spearman、spatial、
activation和energy也都须与multiplicity显式展开等价。

每个replicate的每项正式指标先取三个pseudo-confirm folds的最小值，再独立生成六个门槛：

```text
T_m = numpy.quantile(
  [min(R_m,42^(b), R_m,43^(b), R_m,44^(b)) for b in 0...399],
  0.05,
  method="lower"
)
```

不使用BCa、不插值，不生成综合`T_feature`。该分位数只称
`development bootstrap lower robustness bound`，不称95%置信下界。任一门槛`<=0`，或未重采样
full train的三折六指标任一低于相应门槛，均记`bootstrap_calibration_infeasible`；不裁剪门槛、
不删最弱折、不更换quantile。

静态多worker分配为`replicate_index % worker_count == worker_rank`。所有worker必须使用同一冻结代码、
协议、dtype和算法路径；正式任务只使用同型号GPU组。单coordinator最终只接受恰好一次的
`0...399`记录，按index升序后归并；missing或duplicate不得生成门槛。

已冻结的`replicate_structural_failure`仅由任一必要pair的`null_stratification_infeasible`触发，三折六指标
全部记0。合法抽样得到0 anchor或0 reproduced anchor仍是`completed`的科学性低稳定结果，
对应指标自然记0，不扩大“结构失败”定义。NaN/Inf、ID/shape错位、重复Feature ID或RNN冲突必须
终止整个formal calibration，不得用不足400条记录生成门槛。

本子协议只冻结患者重采样、multiplicity语义、每次replicate完整重匹配、三折归约、门槛生成和
并行归并。spatial指标的最少可评价患者数、空集合与浮点累积规则继承随后单独冻结的RP-A spatial
子协议；本次bootstrap冻结不提前定义或修改这些数值规则。

`clong_rpa_bootstrap.py`已实现抽样、加权聚合、Top-25、worker分配、结构失败记录和门槛归并纯函数；
`test_clong_rpa_bootstrap.py` 14项测试已通过，其中coordinator会再次拒绝错误的结构失败reason，
并拒绝任何超出`[0,1]`的正式比例指标。真实train-only只读debug已证明400个计划可重复产生，
且每replicate六层数始终为冻结值。当前尚未实现完整matching worker和正式产物协议，
也未启动任何正式bootstrap或新seed。

冻结SHA：

```text
4738acd626d755b05e57d65f7476108280dad8031cf6cd6f44612b0877a997fc  rpa_bootstrap_protocol_v1.json
9033089219e7bfdfa65acc0f2422e76470f410ec0e037762335e5f84ecf61009  clong_rpa_bootstrap.py
c364bad3b1df93ac84b90e8b3d35fcf7ce1d6c4ba33809769d57989762a48743  test_clong_rpa_bootstrap.py
```

正式冻结的门槛生成规则为：

```text
R_min^(b) = min(R_fold42^(b), R_fold43^(b), R_fold44^(b))
T_metric  = Q_0.05({R_min^(b)})
```

其语义是“开发阶段三次伪确认中最弱一折稳定性的bootstrap分布第5百分位”，即开发期校准
的稳定性下界，不表述为正式non-inferiority test。Feature边真实性已经由经验p值与BH-FDR
相对完整搜索null单独控制，不再把null upper重复塞入R指标门槛。RP-A的严格性来自边真实性、
BH-FDR、实际覆盖门槛、val独立复现和202/503双确认的组合，而不是`Q_0.05`单独提供。最终
分别产生anchor recall、confirmation coverage、activation和energy门槛，不再笼统记为一个`T_feature`。`Q_0.05`、
bootstrap次数的正式冻结、是否BCa、患者重复权重和并行归并均为`TBD`，必须在运行43/44
前冻结。不能看到开发结果后在最小值、均值、中位数或排除某折之间选择。

### RP-A确认PASS与停止规则

seed202和seed503分别相对冻结的42/43/44 3/3 anchors评价。每个seed必须同时通过：

1. 训练与checkpoint血缘、预算和split一致；
2. 字典死亡率`<=0.10`、非死亡decoder重复率`<=0.10`，直接继承S2c冻结定义；另须通过
   finite loss、无NaN/Inf、实际L0合法及checkpoint/SHA一致等工程健康检查；
3. 正式GPU路径no-op identity硬门槛；
4. BH-FDR后的Feature边真实性规则；
5. `R_anchor_recall`和`R_confirm_coverage`分别达到冻结门槛；
6. `R_activation_reference`和`R_activation_confirmation`分别达到各自冻结的绝对门槛；
7. `R_energy_reference`和`R_energy_confirmation`分别达到各自冻结的绝对门槛；
8. train冻结规则在val独立复现，val不得重估任何阈值。

只有202和503均PASS才进入RP-B/RP-C。1/2通过仍判RP-A失败，不更换seed、不放宽门槛、
不运行911，也不以子空间诊断改判。

### val独立复现的可执行定义

train冻结Feature universe、eligible集合、3/3 development anchors、strata、null CDF、全部
匹配阈值、BH-FDR规则和一一分配算法。val不得重新筛eligible、重建开发anchor、重估null或
重新生成PASS门槛。

val只用val患者重新计算患者依赖行为：激活Spearman、Top-patient Jaccard、同图49位置空间
复现及其组合统计量；随后按train冻结的null CDF和绝对阈值，依次重新执行经验p值、BH-FDR、
RNN及冻结的一一分配算法，形成独立val matching graph。最后重新计算confirmation相对固定
development anchors的anchor recall、confirmation coverage、activation和energy覆盖。

val中的`M_j_val`和`E_j_val`使用val患者按各自冻结公式重新计算，但eligible Feature IDs、
anchor membership和匹配参考全部继承train冻结对象。val不得根据val激活重新应用`A_min`、
`P_min`或其他eligible规则，也不得改变activation或energy的Feature universe。

val经验p值只能由冻结的train null empirical CDF映射得到；禁止在val重新做permutation或生成
新的null，否则视为重新估计校准分布并直接违反协议。

val使用与train完全相同的绝对PASS门槛，不另设`val >= train - delta`容差。任一正式指标在
val低于冻结门槛，该confirmation seed即失败。

### no-op identity工程硬门槛

令`h'=h`，正式GPU路径必须逐级记录max/mean absolute error及max relative error：

```text
patch feature -> attention -> pooled vector -> logits -> probability
```

正式容差不使用单一`T_identity_GPU`，而是分别冻结patch、attention、pooled、logit和
probability各级的`atol/rtol`。容差生成方法、固定GPU数值路径和self-test输入必须在确认seed
前写死；CPU仅用于单元测试并使用单独容差。不得只保存PASS布尔值。任一级超差则该seed
直接失败，禁止解释干预结果。

### 局部子空间稳定性：纯诊断

不比较两个完整10240 decoder张成的整体空间。只有在比较Feature集合和rank均已train-only
冻结时，才计算principal-angle similarity或projection overlap，例如冻结anchor局部邻域。
结果必须记录`diagnostic_only=true`。Feature级门槛失败而局部子空间相似时，只能表述为
“与不同初始化选择不同基底表达相似局部子空间相容”，不得改判RP-A成功。

### train/val职责与后续冻结

- train：eligible、分层null、技术anchor、阈值校准、候选选择与匹配随机对照；
- val：只验证跨seed技术结构和覆盖门槛，不重估阈值；
- internal test/external：RP-A开发和确认期间继续锁定；
- 医生命名、正式family合并、`N_review`、癌/非癌富集边界和RP-C效应门槛均不在RP-A中
  事后确定。RP-A通过后，才根据已冻结允许的技术分布另立RP-B/RP-C预注册。

### 文献依据与边界

ICLR 2026的`Sparse Autoencoders Trained on the Same Data Learn Different Features`直接支持
同数据不同初始化可学习不同Feature，且其Top-K实验更依赖seed；论文中约30%共享只属于
特定LLM实验，不作为胃镜门槛。2026预印本`Unstable Features, Reproducible Subspaces`
支持Feature重现概率和局部子空间诊断，但不是医学影像实证，也不能挽救Feature级失败。

### 冻结前未决项

1. bootstrap子协议已于2026-08-24正式冻结并完成SHA固结；完整matching worker和正式产物协议尚未实现；
2. 各级GPU identity的`atol/rtol`生成方法与固定self-test输入；
3. RP-A确认输出、失败保留现场和整体协议SHA文件格式。

字典死亡率和重复率不再列为新TBD，直接继承S2c的双10%定义；BH假设族及FDR/RNN执行顺序
已在本草案中明确。以上未决项全部关闭、测试通过并由用户确认后，RP-A才可从“待确认、未冻结”升级为正式预注册；
随后先运行seed43/44，生成开发校准报告并冻结数值，最后才允许启动seed202/503。

## S2-S3正式矩阵结果（2026-08-20）

正式17组矩阵已于2026-08-20全部运行完成，汇总器输出：

```text
结果/SAE/CLong文献重构_20260819/clong_sae_matrix_summary_seed42.json
状态：no_formal_product；L1合格 0/15；Top-K角色=fallback_candidate（未过门槛）
```

### 逐组核心指标（val，汇总自`clong_sae_matrix_seed42.csv`）

| 配置 | cosine | 患者AUC下降 | 患者一致率 | recovered CE | mean L0 | 死亡率 | 非死亡重复率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| w512/l2e-4 | 0.7771 | -0.0005 | 0.9346 | 0.9739 | 215.3 | 0 | 0 |
| w512/l5e-4 | 0.7047 | 0.0043 | 0.9154 | 0.9353 | 97.5 | 0 | 0 |
| w512/l1e-3 | 0.6415 | 0.0055 | 0.8923 | 0.9167 | 44.0 | 0 | 0 |
| w1280/l2e-4 | 0.8249 | 0.0005 | 0.9462 | 0.9911 | 415.6 | 0 | 0 |
| w1280/l5e-4 | 0.7337 | 0.0053 | 0.9308 | 0.9541 | 158.5 | 0 | 0 |
| w1280/l1e-3 | 0.6554 | 0.0114 | 0.9231 | 0.8993 | 70.8 | 0 | 0 |
| w2560/l2e-4 | 0.8478 | -0.0009 | 0.9538 | 1.0071 | 565.2 | 0 | 0 |
| w2560/l5e-4 | 0.7576 | -0.0033 | 0.9500 | 1.0032 | 237.2 | 0 | 0 |
| w2560/l1e-3 | 0.6695 | -0.0032 | 0.9385 | 0.9773 | 109.7 | 0 | 0 |
| w5120/l2e-4 | 0.8567 | -0.0061 | 0.9462 | 1.0225 | 643.1 | 0 | 0 |
| w5120/l5e-4 | 0.7758 | -0.0051 | 0.9308 | 1.0068 | 307.6 | 0 | 0 |
| w5120/l1e-3 | 0.6778 | -0.0022 | 0.9346 | 0.9671 | 132.0 | 0.0021 | 0 |
| w10240/l2e-4 | 0.8574 | -0.0035 | 0.9423 | 0.9983 | 681.7 | 0 | 0 |
| w10240/l5e-4 | 0.7901 | -0.0044 | 0.9308 | 1.0043 | 385.7 | 0 | 0 |
| w10240/l1e-3 | 0.6941 | 0.0105 | 0.9308 | 0.8967 | 177.5 | 0.0185 | 0 |
| w10240/l5e-4/gamma0（诊断） | 0.7872 | 0.0098 | 0.9192 | 0.8744 | 374.4 | 0 | 0 |
| w10240/topk1024（备选） | 0.8814 | -0.0039 | 0.9577 | 1.0147 | 952.5 | 0 | 0 |

参考：原始C-long val患者AUC = 0.9101；六项硬门槛为 AUC下降≤0.01、一致率≥0.95、
cosine≥0.90、recovered CE≥0.95、死亡率≤10%、非死亡重复率≤10%。

### 门槛分析

- **唯一全面卡点是cosine**：17组无一达到0.90。最高为Top-K的0.8814，L1最高为
  w5120/l2e-4的0.8567与w10240/l2e-4的0.8574。宽度从0.4×增至8×，cosine仅从0.78升至
  0.86，边际收益明显递减，8×并不能解决保真瓶颈。
- 患者AUC下降门槛几乎全过，且10组为负值（重构后患者AUC反而高于原模型，最高
  w5120/l2e-4达0.9162），说明margin保真按设计保住了分类相关信息。
- 一致率≥0.95仅3组通过（w2560/l2e-4、w2560/l5e-4、Top-K）；recovered CE≥0.95有12组
  通过；死亡率与非死亡重复率全部通过（绝大多数为0）。
- Top-K备选仅cosine一项不合格（其余五项全过），是全部候选中最接近放行的配置，
  但按预注册不得因此放宽门槛或追加K值网格。

### 无margin诊断组结论

w10240/l5e-4/gamma0相对同宽度同lambda的margin组：一致率0.9192对0.9308、
recovered CE 0.8744对1.0043、AUC下降0.0098对-0.0044，cosine基本持平
（0.7872对0.7901）。证实gamma=0.1 margin项的作用主要是保分类方向而非全向量保真，
与旧路线教训一致。

### 解释

margin保真目标确实把"分类相关方向"保住了（重构患者AUC普遍不降反升），但
attention-pooled 1280维向量中分类无关的成分在当前宽度×稀疏度下无法被重构到
cosine≥0.90。这与旧路线在GAP表示上的经验同构：分类保真容易、全向量高保真难。
按2026-08-19冻结的停止规则，本阶段无正式产品，回到表示对象或SAE结构设计，
**不追加超参数、不事后放宽门槛**。

### 审计备注

- 汇总CSV与单组`metrics.json`抽查一致（Top-K组逐位核对）；
- internal test/external全程未读取；
- 特征缓存六文件SHA复用核验全部通过，17组共用同一份S0冻结特征；
- 门槛、预算与选择规则自冻结起未做任何修改。

## 下一步

1. ~~冻结S0~~：S0已完成，6项SHA、split、结构和阈值全部核验写入；
2. ~~建立独立实验名和输出根目录~~：输出根目录为`结果/SAE/CLong文献重构_20260819/`，
   未复用旧A1c/B/C目录；
3. ~~起草S2-S3可执行预注册~~：已完成，2026-08-19冻结（训练预算、checkpoint规则、
   四项成功门槛、Pareto选择顺序、剪枝规则、Top-K备选与停止规则全部写死）；
4. ~~决定新入口~~：新建`程序/SAE/正式代码/clong_sae_discovery.py`，复用旧SAE基类、
   Top-K、平衡采样和margin训练主体；血缘核验、冻结阈值、复算自测、严格剪枝、
   非死亡重复率和attention加权空间图均为新实现；
5. ~~实现分项日志与校验~~：feature+margin+L1分项日志、S0六项SHA运行时校验、
   同一缓存三seed复用和癌/非癌联合统计字段已实现；9项单元测试与debug冒烟通过
   （debug输出在`结果/SAE/CLong文献重构_20260819/debug/`，不进正式矩阵）；
6. ~~启动正式17组矩阵~~：已于2026-08-20完成，17/17组跑通，汇总器自动执行，
   结论为no_formal_product（L1合格0/15，Top-K备选未过cosine门槛）；结果与门槛分析
   见"S2-S3正式矩阵结果"节；
7. ~~审阅并冻结S2b结构重构协议~~：已于2026-08-20正式冻结，明确S4-A/B/C/D/N五臂、
   patch完整替换评价、BatchTopK全train阈值、归一化旁路诊断、损失校准、硬门槛、
   唯一配置选择、复现与停止规则；BatchTopK死亡Feature辅助损失明确不启用；
8. ~~实现与debug验收S2b~~：空间特征缓存、完整替换复算、
   Top-K/BatchTopK共享harness、分块阈值、归一化诊断、汇总器和启动器已实现；
   18项测试及B/C/D/N四臂CPU debug均通过，未读取test/internal test/external；
9. ~~seed42按S4-B→S4-C→S4-D→S4-N固定顺序运行~~：已于2026-08-20完成；
   B协议失败，C/D均仅通过6/8门槛，汇总为`no_patch_product_stop_s2b`；
   按停止规则不进入SAE seed202/503复现。旧`S4：原型展示与临床命名`继续等待后续新协议产品；
10. 若继续SAE，先起草新协议比较Matryoshka/Gated SAE、更合适的解释层或更大的训练患者规模，
    不在S2b上临时扫描K/宽度/门槛。
11. ~~运行S2b失败诊断v2~~：已完成，正式pooled cosine短板确认在标签、来源、分辨率、
    画中画等分层中广泛存在；完整翻转主要由内容重构驱动，结果未改写S2b失败判定；
12. ~~审阅并冻结S2c Matryoshka patch SAE~~：已于2026-08-20冻结`K={64,128,256,512,1024}`、
    uniform weighting、五层联合train-only `gamma_pool`校准、五层激活并集死亡率口径和
    “八门槛中最小合格K”选择规则；
13. ~~实现并验收S2c完整链路~~：嵌套单排序核心、联合`gamma_pool`校准、五层联合训练、
    逐K完整替换评价、八门槛最小K选择、FVU纯诊断、冻结K跨seed交接和启动器均已实现；
    24项S2c测试、18项S2b回归与CPU端到端debug通过。debug发现“首次建缓存会改变
    SAE初始化随机数顺序”，已改为缓存加载后重新固定seed，并由初始化SHA防线复验；
14. ~~正式运行S2c seed42~~：已于2026-08-21完成1000 epoch和五个K的完整评价；五个K均
    只在患者冻结阈值一致率上失败，状态为`no_product_stop_s2c`，按协议不放行seed202/503；
15. **条件路线，当前不触发**：若未来另立协议并产生通过门槛的稳定SAE产品，再完成
    SAE seed42/202/503复现，并另行冻结跨seed匹配阈值、
    Feature family候选边界、共享性分类边界和医生panel数量，再依次执行跨seed对齐、技术
    家族候选、癌/非癌四维共享性审计、医生两级盲审、来源/伪特征审计及单Feature与家族级
    残差保留干预。当前S2c已失败，因此本项不启动；不得据此追加K、放宽门槛或继续调S2c；
16. **下一项决策**：在保留S2c失败结论的前提下，单独讨论是否新建Gated SAE、更合适解释层、
    概念粒度与重构粒度分离，或直接转向不要求严格可逆重构的概念发现协议。任何新方向均需
    重新预注册，不能复用S2c“差一点通过”作为事后放宽依据；
17. **当前主线**：~~seed42正式聚合审计与active-frequency split-half校准~~已完成，
    9/9产物SHA验收通过，eligible子协议已冻结；null/FDR闭式精确协议经两轮审阅、20项测试及
    SHA固结后于2026-08-24正式冻结；activation与representation energy的两侧指标定义均于
    2026-08-24完成复审并正式冻结；bootstrap source只读审计和结构失败口径已关闭；
    RP-A matching正式规模纯计算benchmark已完成，精确完整重算约11.6分钟/replicate、
    400次单GPU投影约77.4小时，无OOM且未持久化任何统计输出；bootstrap抽样、RNG、400次、
    最弱折lower门槛、重复患者、结构失败和并行归并已于2026-08-24正式冻结，14项纯函数测试、
    真实train-only抽样debug和协议/实现/测试SHA固结均通过；spatial数值与可评价规则随后通过
    6项语义测试和固定小块numerical-only audit，于2026-08-24正式冻结。下一步关闭GPU identity。
    任何新seed均未
    启动；全部规则冻结后才按43/44开发校准、202/503确认、911留出初始化复现执行。
