# gastric-cbm YOLO26实验进度与结果讨论

## 新对话恢复项目的固定提示词

> 该提示词只规定恢复上下文与协作方法；即时状态以本文后续章节和正式输出为准。

```text
请继续协助我开展 gastric-cbm 的YOLO26胃早癌病灶定位研究。

先确认项目根目录为 /home/mcy/gastric-cbm，并按顺序阅读：

1. AGENTS.md：长期项目背景、数据/test边界、目录、文件安全、Git和代码注释规则；
2. YOLO26实验进度与结果讨论.md：当前阶段、冻结协议、最新结果、阻塞项和下一步；
3. 实验进度与结果讨论.md 的“阶段决策”：M0-M5历史结论和启动YOLO路线的原因；
4. 本轮相关的正式代码、CSV/JSON/config和冻结manifest。数字以正式产物为准，不只依赖Markdown。

开始操作前先只读检查git status、目标输出目录和现有运行状态，不要重复运行完整产物。
GPU任务必须先运行nvidia-smi，同时检查显存、利用率和进程，再选择设备；不得默认GPU编号。

每轮先用通俗语言说明：解决什么问题、输入和输出是什么、是否接触internal test/external、
通过后进入哪一步。正式训练前必须确认超参数、checkpoint选择规则、阈值和停止条件已经冻结。

实验原则：
- patient-level split优先，患者不得跨split；
- Y0-Y3只使用train/val，internal test在Y4一次性揭盲，external只作Y5诊断；
- 一轮只改变预注册变量，不按test/external结果返调模型、阈值或框后处理；
- 没有预测框的癌图仍进入几何分母，IoU/coverage/center-hit记0；
- 自动坐标检查不能代替临床确认多病灶标注完整性；
- 先smoke再全量，不覆盖冻结清单、权重、历史结果或人工标注；
- 不批量删除，不上传患者图像、预测明细或权重到公共仓库；
- 长任务提供运行命令时，同时提供实时日志监看命令。

每完成一个正式阶段，按“做了什么、结果如何、确认与未确认结论、风险、下一步、文件与验证”
详细但易懂地汇报，并实时更新YOLO26实验进度与结果讨论.md。只有长期规则、稳定事实、正式入口
或重要阶段结论变化时才更新AGENTS.md。旧实验进度与结果讨论.md已经结束维护，不再追加新日志。
```

## 文档边界

本文只维护2026-08-13开始的YOLO26专用检测器阶段。此前第二批数据、去偏分类器、
M0-M5、MOCE和SAE的完整历史保留在`实验进度与结果讨论.md`，该文件已结束维护。

## 当前状态

更新时间：2026-08-13

| 阶段 | 状态 | 当前结论或阻塞项 |
| --- | --- | --- |
| Y0数据转换与审计 | 工程验收完成，待临床确认 | 文件、SHA、尺寸、坐标和精确重复通过；47组跨患者pHash候选逐对确认均非重复；仅剩多病灶标注完整性待临床确认 |
| Y1 YOLO26n smoke | 未开始 | Y0人工项确认后安装并锁定环境，只验证流程 |
| Y2 YOLO26s 640/960 | 未开始 | 仅seed42预筛 |
| Y3三种子稳定性 | 未开始 | 使用Y2唯一入选配置 |
| Y4 internal test | 锁定 | Y3完成并冻结后只评估一次 |
| Y5 external诊断 | 锁定 | 不用外部结果选模型或阈值 |
| Y6 ROI分类 | 条件候选 | 只有Y4定位成功才启动 |

## 前置结论

- 冻结M0 EfficientNet-B0仍是当前正式癌症分类器。
- 冻结M1能粗略定位，内部test mean IoU约`0.446`、center-hit约`0.907`。
- M3/M3b抑制、M3c-B硬门控和M4-M5c预测ROI分类干预均未在锁定internal test形成稳定产品收益。
- 因此新路线只检验：专用检测器能否在保护癌图检出的同时，提高病灶框几何质量。

## 冻结路线

```text
Y0  数据转换、bbox完整性、患者频次、精确/近重复审计
Y1  YOLO26n-640 smoke test，只验证代码和格式
Y2  YOLO26s-640 vs YOLO26s-960，固定seed42受控预筛
Y3  Y2唯一入选配置运行seed 42/202/503
Y4  一次性锁定internal test，与同seed M1作患者配对比较
Y5  无bbox external诊断投影，不用外部结果选模型或阈值
Y6  仅Y4成功后进行预测ROI分类
```

## Y0协议与结果

输入为M1来源内1:1.3 Keep主清单：`1373张/1020人`。类别固定为
`early_cancer_or_HGD`。train/val转换为YOLO单类别视图；internal test只生成带SHA的
冻结队列，不进入`data.yaml`，未生成预测或指标。

正式输出：

```text
数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/
10_Y0_YOLO26检测数据_20260813/
```

| split | 非癌图片/患者 | 癌图片/患者 | 是否导出到YOLO |
| --- | ---: | ---: | :---: |
| train | 551/402 | 424/312 | 是 |
| val | 113/86 | 87/67 | 是 |
| test | 112/86 | 86/67 | 否，仅冻结队列 |

自动验收：1373张全部可解码，文件SHA和记录尺寸一致；癌图框均存在、合法且界内；非癌图
定位监督关闭；患者不跨split；精确重复为0。每患者1-3张，各split/类别的中位数均为1、
最大值均为3，未见单患者图片数量失控。

近重复审计使用64-bit DCT pHash、Hamming distance<=6，共提出跨患者47组，其中跨split
19组。2026-08-13已逐对查看全部47组：均为构图、亮度或低频纹理相近，但黏膜纹理、皱襞、
反光、开口或病灶形态不同；未发现同图、同帧或可明确判定的相邻帧泄漏，不修改split。

框面积占全图比例：train中位数`0.235`、val `0.199`、锁定test `0.299`。该差异先作为
数据描述记录，不使用test分布调YOLO参数。

Y0工程侧已经放行，仍需一项临床确认：

1. 临床确认癌图中所有早癌/HGD目标均已框出。当前CSV每图只能表达一个框；若存在多处
   分离目标，必须升级为一图多框标签。非癌空标签只表示无早癌/HGD，不代表无其他异常。

## Y1-Y4预注册

Y1安装并锁定Ultralytics、torch和CUDA版本，保存完整`args.yaml`。Y2/Y3使用相同预训练
来源、optimizer、学习率、weight decay、batch、epoch、patience和增强；只允许改变
`imgsz`、模型规模和seed。正式checkpoint统一采用锁定Ultralytics版本按内置validation
fitness保存的`best.pt`，不得按internal test或单seed重新挑epoch。Y1只做smoke，不产生
性能结论；Y2开始前把实际官方默认值完整展开并冻结，不能只记录“default”。

Top-1几何规则：训练一致的letterbox；`imgsz`等于对应实验输入；geometry confidence
固定`0.001`、IoU/NMS参数`0.70`、`max_det=100`、`agnostic_nms=false`；Top-1为标准
后处理后置信度最高框。YOLO26路径若不使用NMS，仍记录实际行为。无候选时IoU、coverage、
center-hit均记0，所有癌图都进入分母。

deployment threshold只在val癌图上冻结为满足图像级Sensitivity>=0.90的最高置信阈值。
主要终点为癌图Top-1 IoU>=0.50比例；次要终点为mean/median IoU、coverage、center-hit、
预测框面积、非癌positive-trigger rate、每张非癌框数、mAP50、mAP50-95及病灶大小分层。

安全性界值是`Sensitivity_YOLO - Sensitivity_M1 >= -0.02`的点估计，并报告患者聚类
bootstrap 95%CI。只有CI下界高于`-0.02`时才称统计学非劣效，否则只称通过工程安全门槛。

Y2先过安全门槛，再以Top-1 IoU50选择640或960；平手依次比较mean IoU、非癌触发率和
计算成本。Y4正式成功要求：三seed IoU50均高于同seed M1，三seed平均配对提升的患者
bootstrap 95%CI下界高于0，且每seed均通过Sensitivity安全门槛。失败则停止，不进入Y6。

## Y5-Y6边界

Y5无bbox，只报告框触发率、框数、置信度、面积和来源/尺寸分层，不把置信度AUC写成定位
IoU。Y6开始前重新冻结局部分类器数据；第一版优先用GT bbox加位置、宽高和尺度jitter
模拟预测误差，OOF预测ROI仅作高成本候选。存在有效ROI时才融合全局与局部概率；无ROI时
必须回退`p_global`，不得把`p_local=0`后继续平均。

## 下一步

1. 请临床确认一图多病灶的标注完整性边界；
2. Y1可先作为纯工程smoke进行：先检查GPU，再安装/核对YOLO26环境并运行YOLO26n-640；
3. 临床完整性确认和Y1均通过后，完整冻结Y2训练参数，再运行YOLO26s-640与960的
   seed42受控比较；临床确认未完成前不得启动Y2正式性能实验。
