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
| CD0人工归因 | 待医学生复核 | 已冻结510例去重复核包：210例FN+300例FP，其102例双人独立归因 |
| CD1-CD8 | 路线骨架已冻结，逐阶段协议未冻结 | 根据CD0选择下一项假设；不得一次性堆叠Gray、LUPI、多个KD损失和分割 |

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

逐seed的433条FN已合并为210个独立病灶案例：内部val 50例（稳定3/3漏检17），
外部开发160例（稳定3/3漏检64）。FP按同图框`IoU>=0.30`跨seed聚类后共2980簇；
冻结选入每队列稳定高置信FP 100例和不稳定FP固定随机50例。最终复核包共510例，
其102例（20%）标记为双人独立归因。

定量CD0已完成；整个CD0尚缺人工主/次错误归因与一致性统计，所以现在可以说“跨域定位
退化和小病灶风险已定量确认”，还不能宣布Gray、KD或hard-negative中哪个已被证明应优先。

### 正式产物

- 指标、FROC、逐框预测与跨seed汇总：`结果/CADe/CD0错误地图_20260825/`；
- 去重复核清单与510张两联图：同目录下`CD0人工复核正式版/`；
- 初版逐seed重复材料仅供追溯：`初版未去重复核材料_归档/`；
- 正式代码：`程序/CADe/正式代码/`。

## CD1预备边界（尚未冻结具体训练参数）

CD1只回答Gray是否有资格作为后续教师或互补模型。RGB与Gray必须保持dataset、患者split、
YOLO26s-640、预训练来源、optimizer、epoch、augmentation预算、checkpoint规则和
seed42/202/503一致，唯一主要变量为输入颜色。

Gray资格不能只看总体mAP。正式协议需预先量化：总体与病灶大小分层FROC、外部开发FROC、
Gray对RGB FN的rescue rate、RGB对Gray FN的反向rescue rate、FN/FP Jaccard重合、颜色扰动
稳定性和三seed一致性。若Gray总体、small、域稳健性均更差且错误高度重合，则停止Gray教师
路线；Gray失败不阻塞CD2b、CD5或CD6。

## 当前下一步

1. 将`CD0人工复核正式版`交给医学生，先统一FN/FP主类别的判定示例；
2. 完成210例FN与300例FP的主/次错误归因，102例保持双人盲法独立记录；
3. 统计主类别一致率，样本允许时计算Cohen's kappa，对分歧案例裁定；
4. 根据稳定FN是否主要由小病灶/低对比主导，决定首个受控实验是CD1 Gray还是CD2a病灶加权KD；
5. 若稳定FP主要为相同的皱襞/反光/炎症结构，则将CD5 OOF hard-negative提至优先候选；
6. 人工归因完成前不解锁内部时间测试，不实现新网络或扫描新超参。
