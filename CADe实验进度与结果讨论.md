# gastric-cbm CADe实验进度与结果讨论

## 新对话恢复项目的固定提示词

> 该提示词只规定恢复上下文与协作方法；即时状态以本文后续章节和正式输出为准。

```text
请继续协助我开展gastric-cbm的胃早癌/HGD计算机辅助检测（CADe）研究。

先确认项目根目录为/home/mcy/gastric-cbm，并按顺序阅读：

1. AGENTS.md：长期背景、数据/test边界、目录、隐私、Git、GPU和注释规则；
2. CADe实验进度与结果讨论.md：当前协议、即时状态、结果、阻塞和下一步；
3. YOLO26实验进度与结果讨论.md：Y0-Y6历史检测基线、数据、阈值和已揭盲队列边界；
4. MAGE实验进度与结果讨论.md：可复用的OOF困难ROI、灰度结构教师和空间关注证据；
5. SAE实验进度与结果讨论.md用于了解并行解释路线；CADe任务不得改写其状态；
6. 本轮相关正式代码、manifest、CSV、JSON和checkpoint。数字以正式产物为准。

开始操作前只读检查git status、目标输出目录和运行状态。GPU任务先运行nvidia-smi，同时
检查显存、利用率和进程，不默认GPU编号。先self-test和debug，再运行正式任务；长任务同时
提供启动和tail -F日志命令。

CADe回答“病灶是否被检出、在哪里以及误报多少”，不得用CADx分类AUC替代检测终点。
必须区分：Development Train、Development Val、External Development CADe Cohort、
Locked Internal Temporal CADe Test和未来CD8独立外部确证队列。External Development可用于
诊断并决定下一项研究问题，但不得用于逐epoch、loss权重、阈值或seed选择；锁定内部时间集
在CD1-CD5全部冻结前不得读取YOLO检测结果。

每轮按“做了什么、结果如何、能确认什么、不能确认什么、风险、下一步、产物”详细且易懂
地汇报并更新本文。只有长期规则、稳定入口或重要里程碑变化时才更新AGENTS.md。
```

## 文档边界

本文从2026-08-25开始维护独立CADe路线。`YOLO26实验进度与结果讨论.md`保存Y0-Y6历史，
不再追加；CADe候选技术、正式协议、即时进度和结果统一记录在本文，旧临时讨论稿已删除。
MAGE和SAE继续在各自文档独立推进，任何一条线的新结论不得静默改写其他路线的历史结论。

CADe与CADx的任务边界固定为：

- CADe：输入完整白光胃镜图像，输出可疑病灶框与置信度，主要评价病灶检出和误框；
- CADx：输出图像或患者癌/HGD概率，主要评价AUC、Sensitivity和Specificity；
- CADe检测响应不能直接解释为癌诊断，CADx注意力图也不能直接解释为检测框。

## 当前状态

更新时间：2026-08-25

| 项目 | 状态 | 当前结论或阻塞 |
| --- | --- | --- |
| CADe独立立项 | 已确认 | MAGE-Det只作为候选干预，不预设为答案；先完成CD0错误机制诊断 |
| RGB检测基线 | 已存在 | 冻结Y3-F YOLO26s-640 seed42/202/503；验证集框几何明显优于M1，但小病灶和非癌误触发仍是风险 |
| External Development CADe Cohort | 已具备 | 外部癌图542张中539张有有效bbox、覆盖121位癌患者；另有1399张非癌图；该队列已参与历史工程判断，只作开发诊断 |
| Locked Internal Temporal CADe Test | 保持锁定 | 112张/78人，癌52张/32人、非癌60张/46人；相对CADe开发尚未按YOLO错误返调，CD1-CD5全部冻结前不得读取检测结果 |
| CD0定量诊断 | 已完成 | 三seed内部val+外部开发队列已运行；外部Primary敏感度平均下降0.0889，小病灶是最弱分层 |
| CD0人工归因 | 非阻塞并行 | 已冻结510例去重复核包：210例FN+300例FP，其中102例双人独立归因；等待临床时间，不再阻塞CD1 |
| CD1 Gray Qualification | 已完成：INCONCLUSIVE；External只读投影完成 | Val平均Primary下降0.0208、Gray救回31.2%的RGB漏检；External平均下降0.0334、救回30.8%，互补结构跨中心复现但Gray单模型仍较弱 |
| CD2-CD8 | 路线骨架已冻结，逐阶段协议未冻结 | 根据CD0/CD1证据选择下一项假设；不得一次性堆叠Gray、LUPI、多个KD损失和分割 |

## 已确认的前置证据

1. M1轻量定位头可以粗略找到病灶，但框范围不稳定；历史mean IoU约0.446。
2. Y3-F三种子在开发val癌图Sensitivity固定为0.9000时，mean IoU平均约0.6343、
   IoU>=0.50比例约0.7681；专用检测器改善框几何的结论已经成立。
3. Y4 Balanced主检验因部分seed的小病灶Sensitivity安全门槛失败，不能写成正式整体成功；
   Full是后续探索基线，不追溯修改该结论。
4. Y5外部无框阶段只证明检测器多数会对癌患者产生候选，同时非癌患者误触发约一半；现在
   外部癌图bbox已经补齐，可在CD0重新回答框是否真正覆盖病灶。
5. MAGE C-long在内部时间集和外部539张标框癌图上均显示更稳定的病灶峰值对齐，支持灰度
   结构和训练期空间指导值得研究；该分类证据不自动证明Gray YOLO或检测KD一定有效。

## 数据角色与隔离规则

### Development Train

使用Y0-F/Y3-F既有train，共2350张/1212人。用于后续新模型拟合。CD0不重新训练。

### Development Val

使用Y0-F/Y3-F既有val，共497张/260人，其中癌图240张。用于：

- checkpoint和正式部署阈值选择；
- CD0内部开发指标与错误机制诊断；
- 后续阶段预注册允许的模型选择。

不得使用外部开发集或锁定内部时间集替代val做epoch、阈值、loss权重或seed选择。

### External Development CADe Cohort

当前正式角色为外部开发性质CADe队列：癌图原542张，其中539张有有效bbox、3张病灶不清
按临床反馈排除，覆盖121位癌患者；非癌1399张/1208人。该队列可以：

- 描述内部到外部的检测退化；
- 完成FN/FP人工归因；
- 决定下一项研究假设更应优先Gray、病灶加权KD、定位KD、hard-negative或polygon pilot。

该队列不得：

- 选择单个epoch、seed或checkpoint；
- 扫描并选择训练超参数、蒸馏权重或部署阈值；
- 写成全新独立外部确证集。

### Locked Internal Temporal CADe Test

112张/78人，其中癌52张/32人、非癌60张/46人。该队列虽已用于MAGE一次性评价，但尚未
根据YOLO检测错误修改CADe。CADe线从现在起继续锁定：CD0不读取；CD1-CD5候选方案、参数、
阈值和唯一产品全部冻结后才允许一次性评价。届时只能称“相对CADe开发保持隔离的锁定内部
时间测试”，不得称整个项目从未接触的全新测试集。

### Future CD8 External Confirmation

最终确证需要新的、未参与CADe路线选择的多中心队列，至少包含患者标签、癌图bbox和足够
非癌图；若进入分割，还需冻结的polygon/mask子集。

## 正式阶段骨架

| 阶段 | 正式问题 | 进入与停止边界 |
| --- | --- | --- |
| CD0 Baseline Diagnosis | 当前RGB CADe真实失败模式是什么 | 不训练；输出指标、分层、错误归因和决策报告 |
| CD1 Gray Qualification | 去颜色结构信息是否更强或互补 | RGB/Gray同结构同预算；Gray失败不阻塞其他路线 |
| CD2a Lesion-weighted KD | 重点蒸馏病灶区域是否有价值 | bbox只控制蒸馏权重，不作为教师输入；教师来源须在该阶段前冻结 |
| CD2b Explicit LUPI | 显式bbox/mask特权信息是否还有增益 | 癌与非癌均有区域提示；必须通过mask-only、random-shift和geometry-only审计 |
| CD3 Stability Confirmation | 入选蒸馏方案能否跨三seed稳定复现 | 不再改损失与阈值；失败则停止该方案 |
| CD4 Localization KD | 直接蒸馏定位知识是否进一步改善检出与框质量 | 仅在前一蒸馏方案成立后增加，不与多个新变量同时加入 |
| CD5 Hard-negative Suppression | 能否针对真实非癌困难结构降低误报 | 使用患者级OOF困难区域；不得用同患者训练模型生成训练监督 |
| CD6 Polygon Pilot | 精细边界是否值得额外标注成本 | 约150-200张分层pilot，其中30-50张双人独立标注 |
| CD7 Segmentation | mask监督是否足以支持分割CADe | 仅在标注一致性和模型增益同时成立后进入 |
| CD8 External Confirmation | 新多中心数据上是否真正泛化 | 全新队列一次性确证，不返调CD1-CD7 |

CD0-CD8是决策树，不是必须全部执行的流水线。Gray失败时不强行Gray KD；若CD0显示非癌
困难结构是主导问题，可优先CD5；若框边界和标注不稳定是主导问题，可提前CD6。

## CD0正式协议（2026-08-25冻结）

### 研究问题

CD0不产生新checkpoint，只回答：

1. 现有Y3-F RGB YOLO在内部开发val和外部开发队列上能检出多少真实病灶；
2. 在不同误报负担下，病灶Sensitivity如何变化；
3. 框一旦检出后是否准确，内部到外部下降发生在检出、几何还是非癌误报；
4. FN和FP主要属于哪些视觉或标注类型；
5. 下一项最有依据的干预是Gray、病灶加权KD、定位KD、hard-negative还是polygon pilot。

### 模型与输入

- 固定使用Y3-F YOLO26s-640 seed42/202/503三个`best.pt`；
- 每个seed保留自身Y3-F val冻结部署阈值；
- 不根据CD0外部结果挑选seed或重选checkpoint；
- 开发val与External Development分别评价，不混池计算单一指标；
- Locked Internal Temporal CADe Test不读取、不生成预测。

### TP、FP与FN匹配

正式病灶检出匹配阈值为`IoU>=0.30`。预测按置信度从高到低与GT一对一匹配：

- 首个满足`IoU>=0.30`的未占用预测与GT匹配为TP；
- GT没有匹配预测为FN；
- 未匹配预测均为FP，包括癌图上额外的皱襞、反光或重复框；
- 非癌图上的所有预测均为FP；
- 同一GT周围多个预测只有最高置信成功匹配者为TP，其余为FP；
- 当前队列一张癌图至多一个已知病灶框，但实现不得依赖这一假设，需支持未来一图多GT。

`IoU>=0.50`不作为主要“是否检出”定义，而作为关键框几何质量指标。

### FROC与Primary

全队列FP/image定义为：

```text
所有未匹配预测框数量 / 评价集中全部图像数量
```

同时补充`negative-only FP/image`，即非癌图预测框总数除以非癌图数。

主要终点固定为：

```text
Lesion Sensitivity at achieved FP/image <= 0.5
```

在所有真实唯一置信度工作点中，选择`FP/image<=0.5`时可达到的最高Sensitivity；Primary
不做线性插值，并保存对应真实阈值、TP/FP/FN和实际FP/image。这样该值对应真实可执行工作点，
而非只能由两点之间数学插值得到的数值。

研究型辅助报告绘制完整FROC，并在线性插值口径下报告`Sensitivity@0.1/0.25/0.5/1.0
FP/image`，同时保存每个目标点相邻的两个真实工作点。若目标点超出实际曲线覆盖范围，记为
不可估计，不外推。

### 冻结部署工作点

FROC扫描只评价模型排序能力，不产生新的部署阈值。每个seed原Y3-F val冻结阈值原封不动
应用于开发val和External Development，报告：

- lesion Sensitivity（IoU>=0.30）；
- FP/image与negative-only FP/image；
- 非癌图和非癌患者trigger rate；
- TP/FP/FN数量；
- IoU>=0.50、mean/median IoU、center-hit和lesion coverage。

不得把外部FROC扫描得到的阈值写成正式部署阈值。

### 其他指标与三种子汇总

每个seed独立报告：完整FROC、Primary、冻结工作点、mAP50、mAP50-95、mean/median IoU、
IoU>=0.50、center-hit、coverage、预测框数和置信度分布。三seed只报告均值、标准差和方向
一致性，不把三seed预测混池为一个大样本，也不选择最好seed作为产品。

分层至少包括：

- 训练集冻结病灶面积边界定义的small/medium/large；
- bbox面积连续分位；
- 已可靠获得的来源/中心；
- 原始分辨率或既有size_group；
- 外部已有的年份、画幅和裁剪状态字段。

样本过少或仅含单一类别的层只报告计数和可计算指标，不强行计算AUC或稳定结论。

### 内部到外部性能分解

CD0必须区分以下模式：

- small明显下降而medium/large稳定：支持小病灶表征不足；
- 所有大小同步下降：更支持整体域偏移；
- Sensitivity稳定但FP/image上升：支持非癌背景/良性结构域偏移；
- Sensitivity稳定但IoU下降：支持定位几何泛化不足。

这些是根据指标形成的机制假设，不自动证明因果；下一阶段仍需受控实验验证。

### FN与FP人工归因

人工归因基于每个seed原val冻结部署阈值，不基于FROC插值点。每个案例包含：

- `primary_error_category`：只能选择一个主要原因；
- `secondary_contributors`：允许多个次级因素；
- `needs_expert_review`：是否需要医生复核；
- 自由备注。

FN主要类别冻结为：

```text
small lesion
低对比/边界模糊
反光/过曝
模糊/远距离
黏液/气泡
靠边/部分出画
复杂背景/皱襞
标注存疑
其他
```

FP主要类别冻结为：

```text
皱襞
反光
炎症/糜烂
黏液/气泡
开口/暗腔
出血/明显色差
器械
图像边缘/界面
其他
```

FN全部进入复核材料，并分为3/3 seed均漏检的稳定FN和仅1-2个seed漏检的不稳定FN。FP优先
纳入3/3 seed在相近区域均响应的稳定高置信误框，其余按seed、来源、尺寸和置信度分层随机
抽样；具体抽样数量必须在查看渲染结果前根据候选总量写入CD0实现config。约20%的错误案例
由第二位标注者独立归因，报告主要类别一致率；若样本量允许，再报告Cohen's kappa。双人
不一致案例由医生或双方讨论形成裁定版，但原始两份判断必须保留。

### CD0产物与完成标准

正式产物至少包含：

```text
CD0/
├── detection_metrics/
├── stratified_analysis/
├── error_taxonomy/
└── decision_report/
```

其中应保存逐图/逐框预测、FROC点、逐seed汇总、分层表、FN/FP复核索引、可视化材料模板和
最终决策报告。患者图像和逐图结果只保存在服务器受控目录，不进入公共Git。

CD0完成要求：三seed在val与External Development全部计算完成；Primary和冻结工作点均可
追溯；所有FN完成主要归因；预冻结FP样本完成归因；约20%双标完成；`decision_report`明确
下一阶段优先研究问题。CD0不以某个性能门槛判成功或失败，因为它是诊断阶段。

## CD0定量结果（2026-08-25）

### 做了什么

新增`程序/CADe/正式代码/`下的纯指标模块、正式评价入口、终结脚本和回归测试。
正式运行固定Y3-F YOLO26s-640 seed42/202/503，读取497张开发val和1938张
External Development CADe图像，保留`confidence>=0.001`的全部候选框。未读取
Locked Internal Temporal CADe Test，未训练或选择任何新checkpoint。

### 主要结果

Primary是每图误框不超过0.5时的实际可达病灶Sensitivity；下表为三seed均值±样本标准差。

| 队列 | Primary Sensitivity | 冻结阈值Sensitivity | 冻结阈值FP/image | IoU>=0.50 | mean IoU | AP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development Val | 0.8236±0.0168 | 0.8639±0.0146 | 0.6814±0.0793 | 0.7319±0.0096 | 0.5899±0.0050 | 0.5618±0.0235 | 0.2437±0.0130 |
| External Development | 0.7347±0.0165 | 0.7928±0.0330 | 0.7546±0.1019 | 0.6580±0.0247 | 0.5440±0.0187 | 0.3332±0.0183 | 0.1564±0.0033 |
| 外部-内部平均差 | -0.0889 | -0.0711 | +0.0731 | -0.0739 | -0.0459 | -0.2286 | -0.0872 |

三个seed在外部队列上都同时出现Primary敏感度下降、冻结阈值敏感度下降、FP/image
上升、IoU下降和AP下降。这不是某一seed的偶然失败，也不只是阈值校准问题：外部退化同时
发生在“是否检出”、“框得是否准确”和“非癌误框”三个环节。

### 病灶大小分层

2026-08-25代码复核后修正分层口径：初版实现曾在val和external内各自重算面积
三分位，无法严格比较同一绝对尺度。现已统一改为Development Train癌图一次性冻结的
`bbox_area_fraction` 边界：`q33=0.18661895`、`q67=0.33844387`。同一边界原封不动应用于
Development Val、External Development及未来锁定测试队列。本次只用已保存预测重算
分层汇总，没有重跑YOLO，不影响总体FROC、部署点、IoU、AP或复核案例。

| 队列 | small冻结阈值Sensitivity | medium | large |
| --- | ---: | ---: | ---: |
| Development Val | 0.7404（95张） | 0.9351（77张） | 0.9461（68张） |
| External Development | 0.6707（247张） | 0.8676（146张） | 0.9247（146张） |

small仍是最弱分层，且在同一训练集冻结尺度下从内部到外部平均下降约7个百分点；
medium也出现接近7个百分点的下降，
large相对稳定。因此“小/低对比病灶表征不足”是下一阶段需重点核对的假设，但仍需
人工FN类型证据，不能仅根据bbox面积直接下因果结论。

### 复核包与完成边界

逐seed的433条FN已按`(image_key, gt_index)`合并为210个逐图GT错误案例：内部val 50例（稳定3/3漏检17），
外部开发160例（稳定3/3漏检64）。FP按同图框`IoU>=0.30`跨seed聚类后共2980簇；
冻结选入每队列稳定高置信FP 100例和不稳定FP固定随机50例。最终复核包共510例，
其102例（20%）标记为双人独立归因。

定量CD0已完成；人工主/次错误归因与一致性统计作为临床并行任务继续。现在可以说“跨域定位
退化和小病灶风险已定量确认”，还不能宣布Gray、KD或hard-negative中哪个已被证明有效。
CD1只验证Gray假设，不以等待人工归因为前置条件。

### 正式产物

- 指标、FROC、逐框预测与跨seed汇总：`结果/CADe/CD0错误地图_20260825/`；
- 去重复核清单与510张两联图：同目录下`CD0人工复核正式版/`；
- 初版逐seed重复材料仅供追溯：`初版未去重复核材料_归档/`；
- 正式代码：`程序/CADe/正式代码/`。

## CD1 Gray Qualification正式预注册（2026-08-25冻结）

### 研究问题与决策范围

CD1只回答：去除颜色后，Gray检测器是否比冻结RGB基线具有更好的病灶检出性能，或在总体
性能基本不下降时提供稳定的互补检出。它不是Gray蒸馏实验，也不证明某类视觉结构的因果
作用。CD1结论只允许为`performance pass`、`complementary pass`、`fail`或`inconclusive`。
Gray失败不阻塞CD2b、CD5或CD6，也不得在结果揭示后追加Gray补救调参。

### 数据角色与执行顺序

1. 训练与checkpoint选择只使用既有Development Train和Development Val患者划分；
2. 先完成三seed内部val评价并写出不可覆盖的`CD1_VAL_DECISION.json`；
3. 仅当内部决策冻结后，才允许在独立入口中只读投影External Development；外部结果不得
   修改内部决策、checkpoint、阈值或超参数；
4. Locked Internal Temporal CADe Test继续锁定，CD1不读取、不生成预测；
5. CD0人工归因并行推进，不作为CD1启动或结束的门槛。

### 冻结RGB基线与Gray输入

- RGB基线固定为Y3-F YOLO26s-640 seed42/202/503既有`best.pt`，不重新训练；
- Gray使用同一YOLO26s-640结构、预训练来源、患者split、seed、训练预算、优化器、几何增强、
  batch size、checkpoint选择规则和推理设置；
- 唯一主要训练变量是颜色：Gray设置`hsv_h=0`、`hsv_s=0`，保留`hsv_v=0.4`；
- Gray数据离线无损生成：`cv2.imread`读取原图，`BGR2GRAY`转灰度，再复制为三个相同通道，
  以PNG保存，不增加JPEG重压缩；
- 生成后核对图像数、患者划分、标签、bbox和原图映射一一不变，并验证三个通道逐像素相等；
  训练实现还需在真实增强后抽样确认通道仍相等、路径未越过train/val边界。

### 训练、checkpoint与部署阈值

Gray三seed沿用Y3-F正式参数；除Gray数据路径和上述颜色参数外，持久化训练参数必须与对应
RGB run一致。checkpoint仍只由Ultralytics验证`fitness`（mAP50-95）选择，不使用Primary、
rescue、Jaccard或External Development挑epoch。

每个Gray seed在Development Val上独立冻结部署阈值，严格复用
`evaluate_y3_yolo26.py`既有实现：在癌图最高检测置信度上选择达到图像级Sensitivity
`>=0.90`的最高实际阈值，后续比较统一使用`>=`。RGB继续使用各自历史冻结阈值，禁止RGB与
Gray共用一个阈值。部署阈值与Primary FROC工作点是两个不同口径，必须分别报告。

### 主要终点与统计单位

主要终点与CD0一致：在`IoU>=0.30`一对一匹配下，报告实际达到
`FP/image<=0.5`的最高病灶Sensitivity，不做插值。这里的“病灶Sensitivity”统计单位明确为
逐图GT实例`(image_key, gt_index)`；同一生物学病灶若出现在多张图中，会作为多个逐图实例。

同时报告冻结部署阈值下的Sensitivity、全队列FP/image、negative-only FP/image、mean IoU、
`IoU>=0.50`和AP。small/medium/large继续复用Development Train冻结边界
`q33=0.18661895`、`q67=0.33844387`，但病灶大小只作机制分层报告，不进入CD1放行门槛。

### 错误互补指标

在Primary工作点和冻结部署阈值两个口径分别计算：

```text
Gray rescue = RGB漏检而Gray检出的逐图GT实例数 / RGB漏检逐图GT实例数
RGB reverse rescue = Gray漏检而RGB检出的逐图GT实例数 / Gray漏检逐图GT实例数
FN Jaccard = 两模型共同漏检实例数 / 两模型漏检实例并集
```

FP重合主要在非癌图上评价；癌图额外FP作为辅助结果。两模型在同一图上的FP框使用
`IoU>=0.30`构图，先求最大匹配数量，再在最大数量解中取总IoU最高的二分图匹配；不得按
置信度贪心配对。定义
`FP Jaccard=N_matched/(N_RGB_FP+N_Gray_FP-N_matched)`，并报告匹配FP比例、各自独有FP
比例及计数。

### 内部val冻结判据

设每个seed的`Delta Primary = Gray Primary Sensitivity - RGB Primary Sensitivity`。一个
Development Val队列共有240个逐图GT实例，允许的单实例波动为`1/240=0.00417`。

按以下顺序裁决：

1. `performance pass`：每个seed `Delta Primary>=-1/240`，且三seed平均
   `Delta Primary>=+0.01`；
2. 若未通过1，检查`complementary pass`：三seed平均`Delta Primary>=-0.02`、每个seed
   `Delta Primary>=-0.03`、平均Gray rescue `>=0.20`且至少2/3 seed达到0.20、平均FN
   Jaccard `<=0.75`且至少2/3 seed不高于0.75；
3. 若未通过前两项，只有以下四项同时成立才记为`fail`：平均`Delta Primary<=-0.02`；至少
   2/3 seed的`Delta Primary<=-0.02`；平均Gray rescue `<0.15`；平均FN Jaccard `>0.80`；
4. 其余结果记为`inconclusive`。

上述顺序固定，结果揭示后不得修改阈值、增加small门槛或另选seed。`performance pass`表示
Gray可作为性能候选；`complementary pass`只表示Gray有资格作为后续互补教师/集成候选，
不表示其单模型更优。

### Bootstrap与External Development

内部统计使用5000次患者簇配对bootstrap；每次按患者有放回抽样，并保留同一患者的全部
图像。Primary及冻结部署工作点都使用全量Development Val预先确定的工作阈值，bootstrap
内部不得重新扫描阈值。报告Gray-RGB配对Sensitivity差的95% CI，并同步记录每次对应的
FP/image；CI用于描述不确定性，不替代上述预注册裁决树。

External Development只在`CD1_VAL_DECISION.json`存在后运行，使用已冻结checkpoint和val
部署阈值，报告同样的Primary、部署点、几何、rescue、FN/FP重合与固定大小分层。外部结果
只能描述域稳健性，不能覆盖内部val的四态决策。颜色扰动稳定性不属于CD1正式PASS条件，
如后续需要，必须作为另行预注册的诊断实验。

`CD1_VAL_DECISION.json`至少保存四态结论、三seed指标、Gray checkpoint路径与SHA、对应训练
配置SHA、RGB/Gray Primary实际工作点和冻结部署阈值。External入口必须读取这些已冻结记录，
不得自行重选checkpoint或重算val决策。

## CD1实现与训练入口验收（2026-08-25）

已完成以下前置、训练入口与debug验收：

- `prepare_cd1_gray_data.py`：从Y0-F的train/val视图生成三通道Gray PNG，复制YOLO标签并保存
  源图、Gray图、患者、split、标签和SHA映射；
- `smoke_cd1_gray_augmentation.py`：直接调用锁定Ultralytics训练数据管线，检查真实增强后的
  三通道是否仍相等；
- `cade_cd1_metrics.py`：实现逐图GT错误集合、Gray rescue、reverse rescue、FN Jaccard、
  FP最大基数且最大总IoU二分图匹配、固定阈值患者簇bootstrap和四态纯判定；
- `test_cade_cd1.py`：覆盖数据、指标、匹配、bootstrap、PASS、COMPLEMENTARY PASS、FAIL、
  INCONCLUSIVE及边界条件。
- `train_cd1_gray.py`：只负责单seed Gray YOLO26s-640训练，逐项核对Gray数据、增强smoke、
  同seed Y3-F参数参照和Ultralytics实际持久化参数，并保存checkpoint与完整产品血缘；该入口
  不计算Primary、rescue、Jaccard或部署阈值，也不作CD1资格判断；训练启动时另记录Git commit
  和工作区dirty状态用于后续Teacher血缘追溯，但dirty状态只作审计、不作为训练门槛；
- `test_train_cd1_gray.py`：验证Gray相对Y3-F只改变颜色相关参数、debug/formal命名隔离、
  持久化参数漂移拒绝和训练产品完整性。

正式Gray视图位于`数据整理记录/CADe_CD1_Gray_20260825/`，规模为train 2350张/1212人、
val 497张/260人，共2847张PNG和2847份标签；Gray manifest SHA256为
`28e5d567dc9afd58b9bfe735349b06de60726e8f436fa37c0914d6c0bc6031c0`。未导出test或external。
真实增强smoke抽取32张训练样本，tensor尺寸为`3x640x640`，最大通道差为0。CD0回归
`12/12`、CD1回归`20/20`、训练入口回归`12/12`均通过；FP匹配另经1至4框随机小矩阵
穷举复核通过。

seed42 debug已在完整Development Train（2350张）上完成1轮，独立输出目录、`args.yaml`、
`results.csv`、`best.pt`、`last.pt`和`cd1_gray_train_config.json`均通过验收；正式冻结参数为
`imgsz=640`、`batch=16`、`optimizer=AdamW`、`hsv_h=0`、`hsv_s=0`、`hsv_v=0.4`。该次运行
只验证工程链路，`mAP50-95=0`不作性能解释。训练期间Ultralytics绘图工作线程曾对早期无效
预测框报告`x1 must be greater than or equal to x0`，但主训练、验证、checkpoint保存和重载均
正常完成；当前将其界定为非阻塞的可视化告警，不修改冻结`plots=True`参数。

上述训练入口验收后已按冻结配置完成三seed正式训练；Development Val评价入口及专项测试
随后实现并通过。External投影入口仍未实现，内部test与External均未读取。

## CD1正式Development Val结果（2026-08-26冻结）

Gray seed42/202/503均使用提交`7ae549d`训练，正式评价入口提交为`8fc66f8`；三组训练和评价
配置均记录`git_dirty=false`。正式决策文件位于
`结果/CADe/CD1_Gray_Qualification_20260825/Development_Val正式评价/CD1_VAL_DECISION.json`，
文件SHA256为`d969ac04b88996b54d33f64ce3adf592990d38771fb74bae3ec0c49f703d05c8`，
绑定评价脚本SHA256 `c6d13d6925fe5231424ee21102f16e96271c2cf4a23b1ae5f446b4a476c34b1f`。
RGB三seed主指标逐项复现CD0正式记录；Gray各自冻结的部署阈值均在240张癌图上达到精确
90%图像级召回。External与Locked Internal Temporal CADe Test均未读取。

### Primary与互补性

Primary继续使用`IoU>=0.30`且实际`FP/image<=0.5`时的最高病灶Sensitivity，不插值。

| seed | RGB Primary | Gray Primary | Gray-RGB | Gray救回RGB FN | FN Jaccard | 配对95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.8083 | 0.7875 | -0.0208 | 0.3478（16/46） | 0.4478 | [-0.0700, 0.0288] |
| 202 | 0.8417 | 0.8125 | -0.0292 | 0.2632（10/38） | 0.5091 | [-0.0707, 0.0100] |
| 503 | 0.8208 | 0.8083 | -0.0125 | 0.3256（14/43） | 0.4833 | [-0.0667, 0.0400] |
| 平均 | 0.8236 | 0.8028 | **-0.0208** | **0.3122** | **0.4801** | 描述性汇总见各seed |

冻结四态结论为`INCONCLUSIVE`：

1. 未通过`performance pass`：三seed均下降，平均变化不是`>=+0.01`；
2. 未通过`complementary pass`：救回率和FN差异均满足要求，三seed单独下降也都未越过
   `-0.03`，但平均下降`-0.020833`略低于冻结下限`-0.020000`；差值仅0.000833不能成为
   事后放宽规则的理由；
3. 不属于`fail`：虽然性能下降条件成立，但平均Gray rescue为0.3122而非`<0.15`，平均
   FN Jaccard为0.4801而非`>0.80`，说明Gray与RGB错误并不高度重合，Gray确实检出了一批
   RGB漏诊病灶。

因此，Gray不能作为已验证的单模型性能替代，也不能按本轮规则直接晋级为正式互补教师；
但其互补信号真实存在，结论应保留为“证据接近门槛但不足”，不能写成Gray完全无效。

### 部署点、定位质量与分层

| seed | RGB病灶Sens | Gray病灶Sens | RGB FP/image | Gray FP/image | RGB mean IoU | Gray mean IoU |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.8625 | 0.8542 | 0.7686 | 0.7887 | 0.5945 | 0.5619 |
| 202 | 0.8792 | 0.8375 | 0.6620 | 0.6016 | 0.5906 | 0.5678 |
| 503 | 0.8500 | 0.8500 | 0.6137 | 0.7223 | 0.5845 | 0.5588 |

Gray三seed的AP50和mAP50-95也均低于RGB，定位mean IoU均下降；因此Gray的互补性不是建立在
整体定位质量提升之上。冻结病灶大小分层中，Gray对small Primary三seed均下降；medium在
seed42/503略高、seed202接近，large总体接近。该现象只作机制描述，不进入四态判定。

非癌图Primary FP Jaccard为0.397至0.435，表示两模型约四成FP区域可匹配，同时双方仍各有
大量独有FP。这与FN rescue共同说明：去颜色后模型确实改变了错误分布，但变化没有稳定转化
为总体Primary收益。

首次全量评价在评价代码提交前运行，核心指标与正式重跑完全一致；该产物完整归档为
`Development_Val正式评价_precommit_attempt_20260826`，不作为正式决策。正式重跑绑定干净
提交和脚本SHA，生成21个预测、FROC、bootstrap与决策文件。

## 当前下一步

1. CD1内部决策已经冻结；不得改阈值、换seed、补训Gray或重写四态门槛；
2. 已决定执行CD1 External Development只读投影；它只能描述Gray互补性是否跨域存在，不能把
   内部`INCONCLUSIVE`改成PASS；
3. Gray本轮不能直接作为已放行的CD2 Gray Teacher；下一项CADe干预应重新依据CD0错误地图、
   Gray互补证据和临床归因可用性，在病灶加权KD、定位KD或hard-negative中冻结一个单一假设；
4. CD0人工归因由医学生并行完成；Locked Internal Temporal CADe Test继续锁定。

## CD1 External Development只读投影协议（2026-08-26冻结）

### 目的与边界

本步骤只回答：Development Val中观察到的Gray与RGB漏检互补性，是否也能在跨中心External
Development中观察到。它不训练模型、不更换checkpoint、不重新冻结val部署阈值，也不产生
第二次`PASS/FAIL`。无论External结果如何，CD1内部正式结论始终保持`INCONCLUSIVE`；Locked
Internal Temporal CADe Test继续不读取。

评价使用`CD1_VAL_DECISION.json`中冻结的三组RGB/Gray checkpoint和各自val部署阈值。RGB
External预测必须逐项复现CD0正式External记录；Gray只将同一External图像确定性转为三通道
灰度，不改变尺寸、裁剪或病例构成。病灶大小沿用Development Train冻结边界
`q33=0.18661895`、`q67=0.33844387`。

### 预先规定的描述性输出

每个seed分别报告以下两套工作点，二者不得混为同一指标：

1. External Primary：RGB和Gray各自在External FROC中实际`FP/image<=0.5`时的最高病灶
   Sensitivity，用于描述外部域的排序能力；
2. Val冻结部署点：直接使用Development Val冻结阈值，报告病灶Sensitivity、FP/image、
   mean IoU、AP50、mAP50-95及图像级触发率，用于描述不调阈值时的跨域表现。

Primary和部署点均报告Gray rescue RGB FN、RGB reverse rescue、FN Jaccard，以及非癌图FP数量、
独有FP、匹配FP和FP Jaccard。结果解释必须把FN rescue与新增FP负担放在一起，不能把更高报警
倾向直接解释成更好的结构知识。来源、中心、分辨率和small/medium/large只作机制分层，不进入
任何资格门槛。

另输出逐GT实例四象限表：`both_detected`、`rgb_only`、`gray_only_rescue`、`neither`。每个实例
保留bbox面积与冻结大小组、来源/中心、图像分辨率，以及三seed下RGB/Gray最佳IoU和对应置信度。
在Primary和val冻结部署点分别统计同一GT被Gray稳定救回的`3/3`、`2/3`、`1/3`和`0/3`数量；
该稳定性用于区分偶然救回与跨随机初始化重复出现的互补信号。

本步骤不预设External rescue、FN Jaccard或FP负担的新数值门槛。External结果只形成后续研究
问题的机制证据；即使互补性跨域保持，也不能直接启动Gray Teacher/Student训练。若后续提出
“将Gray独有结构线索迁移到RGB且不继承Gray性能损失”的假设，必须另行预注册。

独立入口`evaluate_cd1_external.py`及专项测试已实现；纯逻辑测试`14/14`、CD1原有回归
`20/20`、Val评价回归`11/11`和全量preflight均通过。16张debug队列已完成三seed GPU推理，
逐GT长表、稳定性表及安全边界字段计数闭合。正式投影必须在代码提交后从干净版本运行。

## CD1 External Development只读投影结果（2026-08-26）

正式入口使用提交`a938b14`运行，记录`git_dirty=false`。输出位于
`结果/CADe/CD1_Gray_Qualification_20260825/External_Development只读投影/`，队列为
1938张/1329人/539个GT病灶。RGB预测逐项复现CD0 External正式记录；Locked Internal Temporal
CADe Test未读取。本节没有External资格判定，内部`INCONCLUSIVE`保持不变。

### External Primary与FN互补性

| seed | RGB Primary | Gray Primary | Gray-RGB | Gray救回RGB FN | RGB反向救回 | FN Jaccard | 配对95% CI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.7477 | 0.6957 | -0.0519 | 0.3088（42/136） | 0.4268 | 0.4563 | [-0.0942, -0.0101] |
| 202 | 0.7403 | 0.7069 | -0.0334 | 0.3143（44/140） | 0.3924 | 0.4752 | [-0.0743, 0.0090] |
| 503 | 0.7161 | 0.7013 | -0.0148 | 0.3007（46/153） | 0.3354 | 0.5169 | [-0.0565, 0.0226] |
| 平均 | 0.7347 | 0.7013 | **-0.0334** | **0.3079** | 0.3849 | **0.4828** | 各seed分别报告 |

Gray在External上的平均rescue为30.8%，与Development Val的31.2%几乎一致；FN Jaccard平均
0.483，也与Val的0.480接近。三seed方向一致，因此Val中观察到的错误互补并非只在单一内部
队列或单一初始化中出现。但Gray Primary三seed仍全部低于RGB，平均下降3.34个百分点，不能
解释成Gray单模型更优。

Primary工作点两模型均约束在`FP/image<=0.5`。非癌图FP Jaccard为0.400至0.409，Gray独有FP
分别为308、316、300个；说明Gray确实改变了报警区域，rescue必须与这些新增错误一起理解。
不过两模型的Primary FP负担已基本相同，所以30.8%的rescue并非简单来自Gray在该工作点整体
输出更多框。

### Val冻结部署点与定位质量

| seed | RGB Sens | Gray Sens | RGB FP/image | Gray FP/image | RGB mean IoU | Gray mean IoU | RGB AP50 | Gray AP50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.8219 | 0.7792 | 0.8617 | 0.8535 | 0.5611 | 0.5160 | 0.3499 | 0.2881 |
| 202 | 0.7996 | 0.7328 | 0.7430 | 0.5913 | 0.5469 | 0.4988 | 0.3360 | 0.3262 |
| 503 | 0.7570 | 0.7644 | 0.6589 | 0.7374 | 0.5241 | 0.5180 | 0.3136 | 0.3121 |
| 平均 | **0.7928** | **0.7588** | **0.7546** | **0.7274** | **0.5440** | **0.5109** | **0.3332** | **0.3088** |

不调阈值直接跨域时，Gray平均病灶Sensitivity下降3.40个百分点，但平均FP/image没有增加，
反而从0.755略降至0.727。Gray的mean IoU、AP50和mAP50-95均值仍低于RGB，说明其互补检出
没有转化为更好的整体定位质量。按冻结病灶大小分层，Gray对small Primary三seed均下降；
medium三seed均高于RGB，large总体接近。该分层只提示互补线索更可能出现在部分中等病灶，
不作为新门槛或因果结论。

### 跨seed rescue稳定性

| 工作点 | 0/3 | 1/3 | 2/3 | 3/3 | 至少1个seed rescue |
|---|---:|---:|---:|---:|---:|
| External Primary | 443 | 70 | 16 | **10** | 96 |
| Val冻结部署点 | 460 | 55 | 18 | **6** | 79 |

Primary下10个严格`3/3` rescue病灶包括small 3个、medium 6个、large 1个。它们是后续机制分析
最值得优先查看的技术样本；但数量只占539个GT的1.9%，不能把全部31%平均rescue都描述为
高度稳定。逐GT长表和稳定性表已保留每个seed的四象限、最佳IoU、对应置信度、bbox面积、
冻结大小组和可用分辨率字段。冻结External清单没有可靠的中心字段，因此本轮不虚构中心归属；
未来如补充中心映射，可在不重跑推理的情况下并入逐GT表作描述性分析。

### 阶段结论

CD1最终结论仍为`INCONCLUSIVE`。External证据支持：Gray不是合格的RGB替代检测器，但其独有
检出线索具有跨中心、跨seed可重复性，而且在冻结部署点并未以更高的平均FP负担为代价。
下一项研究问题可以据此收敛为“能否将Gray独有结构线索迁移到RGB，同时不继承Gray的总体
Sensitivity、IoU和AP损失”。这需要另行预注册Teacher/Student目标、对照组和停止条件，不能
把本次描述性结果直接当作CD2放行证据。
