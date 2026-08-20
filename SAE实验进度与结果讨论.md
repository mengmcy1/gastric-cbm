# gastric-cbm SAE实验进度与结果讨论

## 新对话恢复项目的固定提示词

> 该提示词只规定恢复上下文与协作方法；即时状态以本文后续章节和正式输出为准。

```text
请继续协助我开展 gastric-cbm 的文献驱动 SAE 解释研究。

先确认项目根目录为 /home/mcy/gastric-cbm，并按顺序阅读：

1. AGENTS.md：长期背景、数据边界、目录、文件安全、Git和注释规则；
2. SAE实验进度与结果讨论.md：当前协议、即时状态、结果、阻塞项和下一步；
3. MAGE实验进度与结果讨论.md中C-long的模型血缘、注意力QC和外部描述性投影；
4. 文献/SAE可能相关/、文献/2_M-CBM_DeSantis_ICLR2026.pdf和
   文献/3_ProtoMIL_Sun_MICCAI2025.pdf；
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

更新时间：2026-08-20

| 项目 | 状态 | 当前结论或阻塞项 |
| --- | --- | --- |
| 文献调研 | 两轮完成 | 首轮：已独立精读InterPLM、ProtoMIL和M-CBM；两份ProtoMIL为同一论文的预印本与正式版。第二轮（2026-08-20）：针对正式矩阵失败模式（train/val cosine差距）定向调研，详见"第二轮文献调研"节 |
| 旧SAE路线 | 已冻结归档 | 文档快照已保存；旧代码与结果原地只读保留 |
| 新解释对象 | S0已冻结 | C-long attention-pooled 1280维表示；checkpoint、manifest、教师、缓存、beta共6项SHA全部核验一致，结构与阈值已写死 |
| 新SAE结构 | S2-S3已预注册，2026-08-19冻结 | 同轮比较0.4x/1x/2x/4x/8x五档字典；Linear-ReLU+L1加`gamma=0.1` margin保真；lambda网格{2e-4,5e-4,1e-3}、训练预算、checkpoint规则、四项成功门槛、Pareto选择顺序、剪枝规则与Top-K备选均已冻结 |
| 复现口径 | 已明确 | 只有一个C-long seed42；复现为同一冻结特征上的SAE seed42/202/503 |
| 癌/非癌联合分析 | 已列为正式任务 | 同一字典内分析共有、癌富集、非癌富集、混合及重复概念家族 |
| internal test/external | 锁定 | 新SAE开发不得读取；规则冻结后仅作一次描述性投影 |
| 新路线代码 | 已实现，两轮debug验收通过 | `程序/SAE/正式代码/clong_sae_discovery.py` + 矩阵脚本 + 汇总器；输出根目录`结果/SAE/CLong文献重构_20260819/`；14项单元测试通过。审阅后加固：正式预算（lr/epoch/patience/warmup/batch/剪枝容差）逐项锁死、实验名限17个、debug强制隔离到`debug/`、缓存六文件SHA+shape+行顺序核验、S0交叉绑定补齐v3_audit与beta JSON、S3扩展指标（margin/双阈值/患者偏移/密度直方图）、汇总JSON禁止NaN |
| 正式矩阵 | 已运行完成（2026-08-20），no_formal_product | 17/17组完成；L1合格0/15，Top-K备选亦未过全部硬门槛；按预注册停止规则本阶段无正式产品，未追加任何超参数。主要卡点：val mean cosine最高仅0.8814（Top-K），未达0.90。详见"S2-S3正式矩阵结果"节 |
| S2b结构重构 | 已正式预注册，2026-08-20冻结 | 文献驱动比较pooled BatchTopK与patch级Top-K/BatchTopK；逐样本归一化仅作固定诊断，不参与产品选择；宽度/K、损失、阈值估计、门槛、选择、复现与停止规则均已冻结；test/internal test/external继续锁定 |
| S2b代码 | 已实现并完成debug验收 | 独立核心模块、正式入口、四臂矩阵脚本、自动汇总器和18项回归测试已落盘；CPU debug的B/C/D/N四臂均端到端跑通；启动器和汇总器已支持将BatchTopK阈值不可实现记为正式协议失败，不阻断独立实验臂 |
| S2b正式矩阵 | 已完成（2026-08-20），`no_patch_product_stop_s2b` | B/C/D/N四臂与自动汇总全部完成；S4-B因train正预激活数不足而协议失败；S4-C/D均通过8项门槛中的6项，同时未达到患者预测一致率`>=0.95`和pooled cosine`>=0.90`；无唯一patch正式产品，按预注册停止，不进入SAE seed202/503复现，不追加K/宽度/归一化/门槛调参 |
| S2b失败诊断 | v1已完成，v2口径修正待运行 | v1确认17位翻转患者主要集中在冻结阈值附近，且完整替换的翻转主要由内容重构驱动；审阅发现v1分层使用的是patch cosine，而正式失败门槛为pooled cosine。诊断脚本已补入逐图pooled cosine分层、最差30图和完整翻转四象限，将输出到新v2目录，不覆盖v1 |
| S2c Matryoshka patch SAE | 预注册草案，待用户审阅，未冻结 | 单一seed42模型在同一字典中联合学习`K={64,128,256,512,1024}`五层嵌套粒度；不启用归一化旁路、BatchTopK或新宽度扫描；训练后选择通过八项门槛的最小K，通过后才进入SAE seed202/503复现和医学生概念命名 |

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

## S2c：Matryoshka patch SAE预注册草案（待用户审阅，未冻结）

### 研究问题与实验定位

S2b说明单一`K=128`的patch Top-K能保持AUC、recovered CE和空间注意力，但
在患者阈值一致性和pooled方向保真上仍略有不足。S2c只回答一个新问题：
在不改变解释层、字典宽度和数据的前提下，同时学习从粗到细的嵌套Top-K重构，
能否让少量Feature表示主要概念，并在较大K下补回S2b丢失的特征方向细节？

本轮不将Matryoshka预设为必然有效。S2b诊断只是与“单一K下的弥散细节损失”假设
相容，不是Matryoshka必然通过的证据。

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
  74批、seed42固定初始化上，按五层联合损失重新校准并绑定JSON/SHA；
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
5. 选定K下train死亡Feature率`<=0.10`；
6. 非死亡decoder绝对cosine`>=0.95`的重复Feature率`<=0.10`；
7. normalized AiB下降`<=0.05`；
8. PGA下降`<=0.05`。

正式产品K定义为“同时通过八项门槛的最小K”，不再用AUC或图像观感从通过者中
二次挑选。若所有K均失败，S2c无正式产品，不放宽门槛、不追加K-list、
reverse weighting或Gated SAE到同一实验中。

诊断指标额外报告但不参与checkpoint/K选择：患者/图像概率MAE、患者最大概率偏移及ID、
阈值`+-0.05/+-0.10`内患者的一致率、逐K注意力KL/cosine、每位置L0、每图唯一Feature数、
概念覆盖的图像/患者/位置数，以及标签/来源/分辨率/画中画/病灶大小分层。

### 复现、停止与医学生交付

- seed42存在合格K：冻结唯一K和checkpoint，在同一冻结空间特征上仅重训SAE seed202/503；
- 3个SAE seed均通过：进入概念稳定性对齐、原型图、逐图多概念热图和医学生命名；
- 复现只有2/3或1/3通过：如实报告，不把单seed结果包装成稳定产品；
- seed42无合格K：停止S2c，Gated SAE如需尝试必须新建独立协议。

最终医学生交付不只提供技术指标，而必须包含：通俗阅读指南、模型保真表、癌侧富集/非癌侧
富集/两类共有/疑似伪特征四类概念表、每概念多患者原型图、逐图多色概念热图、
Feature置零后的概率/margin变化，以及供医学生填写“病灶/正常结构/反光/气泡/器械/无法判断”
的人工审核表。癌与非癌共有Feature不自动删除，需结合空间位置和置零干预判断其临床含义。

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
11. 运行S2b失败诊断v2：使用新目录输出逐图pooled cosine分层和完整翻转四象限，
    核对后再将“正式cosine是否弥散”的结论升级为正式诊断结果；
12. 审阅并冻结S2c Matryoshka patch SAE草案：重点确认`K={64,128,256,512,1024}`、
    uniform weighting、五层联合`gamma_pool`校准和“八门槛中最小合格K”选择规则；
    冻结前不实现代码或启动训练。
