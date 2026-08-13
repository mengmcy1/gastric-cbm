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
| Y0数据转换与审计 | 已完成并放行 | 文件、SHA、尺寸、坐标和精确重复通过；47组跨患者pHash候选逐对确认均非重复；当前数据暂无已知多处分离目标，单框结构可用于本阶段 |
| Y1 YOLO26n smoke | 已完成并放行 | 环境、数据、空标签、end-to-end检测头、GPU训练/验证和标准产物均通过；5轮指标不作性能结论 |
| Y2-B平衡集YOLO26s 640/960 | 协议与代码已冻结，正式待运行 | 只用来源内平衡train/val和seed42选择唯一分辨率；960 debug全链通过 |
| Y3-B平衡主实验三种子 | 未开始 | 使用Y2-B唯一入选配置运行42/202/503 |
| Y0-F/Y3-F完整集诊断 | 待构建 | 单独审计完整集；复用Y2-B配置，不参与选参 |
| Y4 internal test | 锁定 | Balanced与Full全部冻结后，在同一198张队列上只评估一次 |
| Y5 external诊断 | 锁定 | 不用外部结果选模型或阈值 |
| Y6 ROI分类 | 条件候选 | 只有Y4定位成功才启动 |

## 前置结论

- 冻结M0 EfficientNet-B0仍是当前正式癌症分类器。
- 冻结M1能粗略定位，内部test mean IoU约`0.446`、center-hit约`0.907`。
- M3/M3b抑制、M3c-B硬门控和M4-M5c预测ROI分类干预均未在锁定internal test形成稳定产品收益。
- 因此新路线只检验：专用检测器能否在保护癌图检出的同时，提高病灶框几何质量。

## 冻结路线

```text
Y0-B  来源内平衡数据转换、bbox完整性、患者频次、精确/近重复审计
Y1    YOLO26n-640 smoke test，只验证代码和格式
Y2-B  平衡集YOLO26s-640 vs YOLO26s-960，固定seed42受控预筛
Y3-B  Y2-B唯一入选配置在平衡集运行seed 42/202/503（正式主实验）
Y0-F  完整集单独转换与审计，不读取其test作调参
Y3-F  完整集复用Y2-B唯一配置运行seed 42/202/503（预注册诊断分支）
Y4    一次性锁定同一198张internal test；M1、Balanced YOLO、Full YOLO作患者配对比较
Y5  无bbox external诊断投影，不用外部结果选模型或阈值
Y6  仅Y4成功后进行预测ROI分类
```

Balanced与Full的角色从本阶段起固定。Balanced来源内1:1.3清单为主实验，负责选择输入
分辨率并形成主要科学结论；Full完整清单只在配置冻结后增加训练监督，不能反向选择分辨率、
optimizer、epoch、阈值或后处理。完整清单共`3348张/1732人`，癌/非癌图片为
`1670/1678`，但癌/非癌患者为`446/1286`，患者及来源/风格明显不平衡；其`1670张`癌图
bbox覆盖100%，因此能检验更多框监督是否改善定位，同时必须报告可能的来源捷径。

Full训练只使用自身train/val（train `2350张`、val `497张`），其原有test `501张`不参与
模型选择，也不作为另一套正式测试。Y4中M1、Balanced YOLO与Full YOLO统一评价Y0-B已经
锁定的同一`198张/153人`internal test，保证所有配对差值来自相同患者和图像。Balanced为
主要结果，Full为预注册次要诊断；不能用Full结果补救Balanced失败。

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

多病灶边界确认（2026-08-13）：根据当前数据认知，暂无已知同图存在多个分离的早癌/HGD
目标而只标一处的情况，因此单框CSV结构满足本阶段训练要求，Y0正式放行。该结论不扩展为
“任何胃镜图都不可能多病灶”；后续一旦发现多处分离目标，必须补为一图多框YOLO标签，
并重新冻结受影响的清单。非癌空标签仍只表示无早癌/HGD，不代表没有其他异常。

## Y1-Y4预注册

Y1安装并锁定Ultralytics、torch和CUDA版本，保存完整`args.yaml`。Y2/Y3使用相同预训练
来源、optimizer、学习率、weight decay、batch、epoch、patience和增强；只允许改变
`imgsz`、模型规模和seed。正式checkpoint统一采用锁定Ultralytics版本按内置validation
fitness保存的`best.pt`。当前锁定版本的Detection fitness须在Y1中由源码核验并记录；若为
当前官方定义，则`best.pt`实质按val mAP50-95选择。不得根据Top-1 IoU50、癌召回、非癌FP、
internal test或单seed重新挑epoch。Y1只做smoke，不产生性能结论；Y2开始前把实际官方
默认值完整展开并冻结，不能只记录“default”。

Top-1几何规则：正式主分析使用YOLO26默认`end2end=True`路径和训练一致的letterbox；
`imgsz`等于对应实验输入；geometry confidence固定`0.001`、`max_det=100`；不额外施加
NMS，NMS IoU不适用于该正式路径；Top-1为end-to-end最终输出中置信度最高框。Y1必须由
实际模型和源码核验并记录`end2end`状态。如果锁定版本实际走`end2end=False`传统路径，
则停止正式运行并单独记录`NMS=True, iou=0.70`，不得静默混入口径。无候选时IoU、coverage、
center-hit均记0，所有癌图都进入分母。

deployment threshold只在val癌图上冻结为满足图像级Sensitivity>=0.90的最高置信阈值。
主要终点为癌图Top-1 IoU>=0.50比例；次要终点为mean/median IoU、coverage、center-hit、
预测框面积、非癌positive-trigger rate、每张非癌框数、mAP50、mAP50-95及病灶大小分层。

安全性界值是`Sensitivity_YOLO - Sensitivity_M1 >= -0.02`的点估计，并报告患者聚类
bootstrap 95%CI。只有CI下界高于`-0.02`时才称统计学非劣效，否则只称通过工程安全门槛。

Y2先过安全门槛，再以Top-1 IoU50选择640或960；平手依次比较mean IoU、非癌触发率和
计算成本。Y4描述性地按`42/202/503`对应展示YOLO和M1结果，但配对来自两个模型评价完全
相同的冻结患者/图像，并非“随机种子数字相同”。正式成功要求：三seed IoU50均高于对应
M1，三seed平均配对提升的患者bootstrap 95%CI下界高于0，且每seed均通过Sensitivity
安全门槛。bootstrap单位为患者：每次有放回抽取test患者并带入其全部图像，分别计算每seed
的YOLO-M1差值，再对三seed差值求平均；重复5000次，使用percentile 95%CI。不得把三个
seed当成`n=3`进行t检验。失败则停止，不进入Y6。

## Y1工程冒烟结果

2026-08-13在启动前检查GPU显存、利用率和进程后，选用当时资源相对充足的GPU 1运行
`YOLO26n-640`、seed 42、batch 16、5 epoch。Y1只验证整条工程链能否正确执行，不参与
Y2模型、分辨率、阈值或checkpoint选择。

环境与关键源码行为已经锁定：

| 项目 | Y1实际值 |
| --- | --- |
| Python | `3.12.13` |
| Ultralytics | `8.4.118` |
| PyTorch / torchvision | `2.11.0+cu128` / `0.26.0+cu128` |
| PyTorch CUDA runtime | `12.8` |
| 检测头 | `Detect`，head与model均为`end2end=True` |
| 推理后处理 | `nms=False`，正式YOLO26路径不额外NMS |
| checkpoint fitness | `[0, 0, 0, 1]`，即val `mAP50-95` |

数据入口再次核验为train/val共`1175张`：癌图`511张`均有单类框标签，非癌`664张`均为空
标签；internal test `198张`未导出到YOLO视图，`data.yaml`不含test，external未读取。
训练完成5/5 epoch，AMP、GPU前向/反向、验证和标准结果保存均正常。`best.pt`、`last.pt`、
`args.yaml`、`results.csv`、训练批次图、val标签图和预测图均存在；人工抽查看到框坐标与图像
对齐，空标签背景样本可正常参与训练。Y1使用完整train/val而非少量伪数据，因此也验证了
正式数据读取链。

第5轮val `mAP50=0.06579`、`mAP50-95=0.02693`。这只是5轮冒烟值：预测仍稀疏，模型尚未
完成正式收敛，不能据此判断YOLO26路线成功或失败，也不能与M1作性能比较。本次`best.pt`
与`last.pt` SHA相同，表示5轮范围内内置fitness最优点恰为最后一轮，不改变正式checkpoint
规则。

正式输出：

```text
结果/YOLO26定位_0804/Y1_smoke/y1_yolo26n_640_seed42/
  args.yaml
  results.csv
  y1_smoke_config.json
  weights/best.pt
  weights/last.pt
```

备注：首次外层`tee`的日志父目录未预先创建，因此shell最终返回非零，但Python训练与产物
验收已完整成功，模型输出不受影响。后续长任务命令必须先创建明确的日志目录再启动`tee`。
Y1的`args.yaml`中`end2end: null`表示调用端没有覆盖模型设置；实际模型头和加载后的
`best.pt`均已核验为`end2end=True`。其中Ultralytics常规val的`max_det=300`用于框检测mAP，
正式项目Top-1几何分析仍按预注册单独使用`confidence=0.001, max_det=100, nms=False`；两种
口径用途不同，Y2实现时必须分别记录。

## Y2-B冻结参数与实现

Y2-B只比较`YOLO26s-640`和`YOLO26s-960`，固定seed 42。预训练权重为官方
`yolo26s.pt`，SHA-256为
`646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`；模型约
`9.95M`参数，检测头与加载后的checkpoint必须保持`end2end=True`。两候选除`imgsz`外
使用完全相同的参数：

| 类别 | 冻结值 |
| --- | --- |
| 训练长度 | `epochs=100, patience=20` |
| batch/worker | 训练`batch=16, workers=4` |
| optimizer | 显式`AdamW, lr0=0.002, momentum=0.9, weight_decay=0.0005` |
| 调度 | `lrf=0.01, cos_lr=False, warmup_epochs=3, warmup_momentum=0.8, warmup_bias_lr=0` |
| loss权重 | `box=7.5, cls=0.5, dfl=1.5` |
| 色彩增强 | `hsv_h/s/v=0.015/0.7/0.4` |
| 几何增强 | `translate=0.1, scale=0.5, fliplr=0.5`；旋转、剪切、透视、上下翻转均为0 |
| 组合增强 | `mosaic=1.0, close_mosaic=10`；mixup/cutmix/copy-paste均为0 |
| 其他 | `amp=True, deterministic=True, rect=False, multi_scale=0, cache=False` |
| checkpoint | Ultralytics内置val fitness，即val `mAP50-95`最高的`best.pt` |

不使用`optimizer=auto`：在当前锁定源码中，平衡集100轮约6100 iterations会自动选AdamW，
完整集约14700 iterations则可能自动切成MuSGD；显式AdamW可避免Balanced与Full因数据量
不同而静默改变优化器。`lr0=0.002, momentum=0.9, warmup_bias_lr=0`与Y1中单类别数据触发
的auto实际选择一致。

标准检测验证继续使用Ultralytics val口径（`max_det=300`）计算mAP；项目Top-1几何评估
单独使用`confidence=0.001, max_det=100, nms=False`。几何推理batch固定为4，并把val目录
交给`LoadImagesAndVideos`分批读取；不得传`list[str]`，因为锁定版本会把路径列表一次性
转成内存图像并将全部200张作为一个batch，造成960推理显存溢出。deployment threshold、
M1配对、5000次患者整簇bootstrap、病灶大小分层和无候选记0均由统一评估入口完成。

Y2-B选择层级固定为：先通过`Sensitivity_YOLO-Sensitivity_M1>=-0.02`点估计安全门槛；
再按IoU>=0.50比例、mean IoU、非癌触发率、计算成本依次选择。mAP只作标准辅助指标，
不越过项目主终点直接选择分辨率。

2026-08-13已完成`YOLO26s-960`最坏显存路径debug：完整train `975张`跑1 epoch，确实包含
`424张`癌图和`551张`非癌图；训练峰值约`11GB`，AdamW冻结参数正确落盘；200张val按
batch 4完成Top-1推理、路径回配、阈值、M1配对、bootstrap与分层输出。该一轮模型的
IoU50=`0.0345`、mean IoU=`0.1230`仅证明评估能识别尚未收敛的模型，不作性能结论。
最初`fraction=0.1`恰按文件顺序抽到98张全非癌图，已归档诊断现场并把debug改为完整train
一轮；正式路径从未使用fraction抽样。

实现入口：

```text
程序/模型训练/正式代码/train_y2_yolo26.py
程序/模型训练/正式代码/evaluate_y2_yolo26.py
程序/模型训练/正式代码/summarize_y2_yolo26.py
程序/模型训练/正式代码/run_y2_yolo26_matrix.sh
```

## Y5-Y6边界

Y5无bbox，只报告框触发率、框数、置信度、面积和来源/尺寸分层，不把置信度AUC写成定位
IoU。Y6开始前重新冻结局部分类器数据；第一版优先用GT bbox加位置、宽高和尺度jitter
模拟预测误差，OOF预测ROI仅作高成本候选。存在有效ROI时才融合全局与局部概率；无ROI时
必须回退`p_global`，不得把`p_local=0`后继续平均。

## 下一步

1. 运行Y2-B正式640/960矩阵并按冻结层级选择唯一分辨率；运行前重新检查GPU，不默认设备；
2. Y2-B入选后运行Y3-B三种子，同时构建并审计Y0-F完整集YOLO视图；Y3-F只能复用同一配置；
3. Balanced与Full全部冻结后才一次性进入同一198张internal test；若发现多病灶漏标，立即
   停止正式训练并回到Y0修正标签和冻结清单。
