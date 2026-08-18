# gastric-cbm MAGE实验进度与结果讨论

## 新对话恢复项目的固定提示词

> 该提示词只规定恢复上下文与协作方法；即时状态以本文后续章节和正式输出为准。

```text
请继续协助我开展gastric-cbm的MAGE-like bbox引导知识蒸馏研究。

先确认项目根目录为/home/mcy/gastric-cbm，并按顺序阅读：

1. AGENTS.md：长期背景、数据/test边界、目录、隐私、Git、GPU和注释规则；
2. MAGE实验进度与结果讨论.md：当前协议、即时状态、结果、阻塞和下一步；
3. YOLO26实验进度与结果讨论.md末尾：Y0-F数据、M0-F全局模型、Y3-F检测器、Y6局部
   模型及其正式/探索性边界；
4. SAE实验进度与结果讨论.md用于了解正在并行推进的解释路线；MAGE任务不得改写其状态；
5. 本轮相关正式代码、冻结manifest、CSV、JSON和checkpoint SHA。数字以正式产物为准。

开始操作前只读检查git status、目标输出目录和运行状态。GPU任务先运行nvidia-smi，不默认
GPU编号。先self-test和debug，再运行正式矩阵；长任务同时给启动和tail -F日志命令。

本路线必须始终区分：原MAGE论文与本项目bbox适配；局部教师与部署学生；GT癌框与非癌
OOF困难框；train/val开发与已经揭盲的internal test/external描述性投影。不得使用test或
external选择教师、蒸馏损失、epoch、阈值或seed。

每轮按“做了什么、结果如何、能确认什么、不能确认什么、风险、下一步、产物”详细且易懂
地汇报并更新本文。只有长期规则或稳定里程碑变化时才更新AGENTS.md。
```

## 文档边界

本文从2026-08-17开始维护MAGE-like bbox引导蒸馏路线。原第二批、M0-M5、MOCE和旧SAE
见`实验进度与结果讨论.md`；Y0-Y6见`YOLO26实验进度与结果讨论.md`。EfficientNet-B0 SAE
由正在并行维护的`SAE实验进度与结果讨论.md`单独记录；两条线互不覆盖，MAGE逐轮日志只
写入本文。

## 当前状态

更新时间：2026-08-18

| 项目 | 状态 | 当前结论或阻塞 |
| --- | --- | --- |
| 原论文核对 | 已完成 | 原MAGE使用两类病灶的像素级mask；本项目只有癌/HGD矩形框，不能称完整复现 |
| MG0a数据血缘与OOF分折 | 已完成 | train 2350张/1212人完成5折患者级分层；test/internal test/external均未读取 |
| MG0b非癌困难ROI | 已完成并冻结v3主清单 | v3源框覆盖100%，解决细长框短轴过扩；完整几何患者AUC=0.6092 |
| MG1灰度局部教师 | 正式配对已完成，预注册判定失败 | 真实患者AUC=0.8572，但shuffle=0.8609；空间形态独立增益未建立 |
| MG1b bbox约束注意力教师 | 正式五门槛与人工QC均通过 | 患者AUC=0.8736；主体关注病灶，少量器械响应作为后续风险保留 |
| MG2全图学生蒸馏 | 代码闭环与beta冻结完成，待正式运行 | 17/17测试与debug回归通过；教师缓存SHA 89f3d329；beta=0.1936（校准JSON SHA 47179b62）；尚未启动正式A/B/C |
| internal test/external | 锁定 | 已在旧路线揭盲，只能在全部规则冻结后作描述性投影，不能用于选择 |

## 研究问题与命名边界

原MAGE是`Masked Achromatic Guidance Expert`：训练时让局部专家只看病灶mask内的灰度
结构，再把类别logit和空间关注蒸馏给接收完整彩色WLI的主模型；推理时只保留主模型。
原论文任务为腺瘤与癌，两类都有像素级病灶mask。本项目任务为早癌/HGD与非癌，只有癌图
矩形bbox，非癌没有病灶mask。因此正式名称固定为：

`MAGE-like bbox-guided color-invariant and spatial knowledge distillation`

不得写成原论文完整复现，也不得把矩形框声称为像素级病灶分割。

本路线的最终目标不是部署YOLO+ROI分类器，而是把bbox作为**仅训练期特权信息**：

```text
训练：局部灰度教师（需要bbox/困难ROI） -> logits与空间关注 -> 完整彩色图学生
推理：完整彩色图 -> 学生 -> 癌概率
```

## 与原MAGE相比的必要适配

1. **标签不同**：原论文区分两类已知肿瘤，本项目区分癌/HGD与非癌。
2. **标注不同**：原论文两类均有像素mask，本项目仅癌图有矩形bbox。
3. **非癌输入必须补齐**：train非癌使用患者级OOF YOLO最高置信候选；没有有效框时按
   train癌框尺寸分布生成确定性回退ROI。所有图都有ROI，禁止让“是否存在框”泄露标签。
4. **教师首版使用灰度ROI裁图**：bbox扩边后裁图、缩放至224并复制为3通道；不在输入中
   绘制框，不把框外黑底或框位置直接交给教师分类头。
5. **病灶框与裁图框必须分开**：`lesion_bbox`保存医生原始病灶范围，用于AiB/PGA和病灶
   评价；`crop_box`保存宽高独立扩边和jitter后的实际矩形裁图范围，用于教师ROI注意力回填。
   二者不得混用。
6. **空间蒸馏需坐标回填**：教师ROI特征关注图先按实际`crop_box`映射回完整图坐标，再与
   学生完整图同层关注图比较。禁止直接把ROI的14x14图与全图14x14图逐位置相减。
7. **非癌空间监督降为消融**：首版logit KD可覆盖癌与非癌；主空间KD只在有GT bbox的癌/HGD
   图计算。OOF非癌框没有真实病灶空间语义，将其纳入空间KD只能作为预注册的后续消融。
8. **外部集边界改变**：既有external已参与旧工程决策，只能作探索性描述；最终确证仍需
   新的独立外部队列。

## 正式阶段路线

### MG0a：冻结数据血缘与患者级OOF分折

输入固定为Y0-F mapping：3348张/1732人，其中只读取train 2350张/1212人和val
497张/260人；原501张test不进入开发。使用train患者做5折分层交叉拟合，折分随机种子固定
42。每位train患者恰好属于一个holdout fold，同患者所有图片必须同折。

输出：`mage_g0_manifest.csv`、患者折分摘要、split/label摘要、SHA与边界config。MG0a不训练
模型，只为后续OOF检测器和教师/学生数据流建立唯一索引。

### MG0b：生成OOF非癌困难ROI

对5个fold分别执行：只用其余4折train患者训练YOLO26s，在其内部再划fit/monitor患者，
不得使用项目正式val作早停；随后只推理当前holdout fold。训练参数继承冻结Y3-F的640输入、
预训练权重和优化配置，不重新搜索分辨率或YOLO规模。仅关闭不参与优化和模型选择的
Ultralytics异步训练图片绘图，避免锁定Pillow版本对端到端框绘制时偶发反向坐标异常。

- train癌图教师ROI：GT bbox；
- train非癌教师ROI：该图对应OOF YOLO的Top-1预测框，不按部署阈值先删掉困难负；
- 无任何预测框：使用来源/尺寸组内癌框宽高分布和图像内容边界生成确定性回退ROI；
- val癌图：GT bbox；
- val非癌图：冻结seed42 Y3-F的Top-1框，同样允许确定性回退。

MG0b验收：train每张图仅有一个ROI来源；train非癌预测框均来自未见过该患者的fold模型；
无患者跨fold；ROI合法且图内；癌GT覆盖不变；回退率、面积、中心、来源和分辨率分层全部
报告。额外记录非癌Top-1置信度P25/P50/P75/P90，并用仅在train拟合的逻辑回归报告bbox
几何对标签的图像级和患者级AUC。该几何模型只是泄漏风险审计，因为实际教师不直接接收
bbox坐标。内部test和external不得进入ROI清单。

### MG1：灰度局部教师

教师使用EfficientNet-B0。所有ROI按冻结`margin=0.20`在宽、高两个方向独立扩边，各轴最多
扩到归一化0.85，源框本身更大时不缩；矩形ROI直接缩放到224x224，再转luma灰度并复制
3通道。该固定尺寸表示与ROIAlign思路相同，目的是避免细长框被补成大方框；不使用黑边
letterbox，以免引入可见的框形状捷径。静态清单保存不随轻微bbox jitter改变的`lesion_bbox`
和确定性`base_crop_box`；Dataset每次取样必须同时返回实际jitter后的`crop_box`，供后续教师
注意力按本次真实裁图坐标回填。癌标签为1，非癌困难ROI标签为0。教师从ImageNet初始化，
不能直接把Y6彩色局部模型称为MAGE教师。

教师损失首版只用标签交叉熵：

```text
L_teacher = CrossEntropy(teacher_logits, y, label_smoothing=0.1)
```

同时训练三个对照以识别捷径：真实灰度ROI教师、仅bbox几何的逻辑回归、保留灰度统计但
破坏空间形态的patch-shuffle对照。常量输入只作工程冒烟，不替代patch-shuffle科学对照。

MG1协议于2026-08-17在训练前冻结如下：

- **输入**：只使用`independent_axis_dynamic`主清单；矩形v3裁图直接resize到224×224，
  luma后复制3通道，不正方形化、不letterbox、首版不做bbox jitter。训练几何增强仅水平
  翻转；val无随机增强。每个样本继续分别保存`lesion_bbox`和矩形`crop_box`，后续attention
  按矩形坐标回填；绿色病灶框只用于癌侧AiB/PGA评价。
- **真实内容对照**：正常输入灰度ROI。**patch-shuffle对照**：同一灰度ROI切成固定7×7个
  32×32 patch后打乱位置；train按样本/epoch确定性变化，val按图像SHA固定排列。两组使用
  相同ImageNet初始化、采样、epoch、优化器、增强和checkpoint规则。
- **geometry-only对照**：固定使用同一v3主清单在train拟合、val评价的中心+尺寸与形状
  逻辑回归，患者AUC=`0.6092`；不得换用v2或尺寸匹配清单重新定义基线。
- **采样与训练**：seed42；患者/类别平衡有放回采样；batch size 32；AdamW，weight decay
  `1e-4`。阶段A冻结backbone，仅训classifier 5 epoch，学习率`1e-3`；阶段B解冻
  `features[5:] + classifier`最多20 epoch，学习率`1e-4`，patience 6。患者概率为图片概率
  均值。checkpoint按val患者AUC最高、再按图像AUC最高、再按val loss最低选择。
- **固定教师策略**：MG1只训练seed42真实与patch-shuffle配对实验；若通过，seed42真实
  教师作为MG2/MG3共享冻结教师，学生再跑42/202/503三种子。这样不利用多个教师挑最好者。
- **成功门槛，三条必须同时满足**：真实教师val患者AUC `>=0.80`；真实教师患者AUC减
  geometry-only患者AUC `>=0.10`；真实教师患者AUC减patch-shuffle患者AUC `>=0.05`。
  patch-shuffle差值不足0.05即判定“空间形态证据未建立”，即便绝对AUC较高也不能进入MG2。
  图像AUC、Sensitivity、Specificity、F1和混淆矩阵同时报告，但不替代上述主门槛。
- **尺寸匹配边界**：v3尺寸匹配分支只保留为已完成的几何诊断，不训练教师、不参与
  checkpoint选择，也不参与MG1成功判定。

只有真实内容教师同时通过三条冻结门槛，才允许进入MG2；否则停止，不把一个主要依赖ROI
几何或局部灰度统计的教师继续蒸馏给全图学生。

### MG2：seed42全图学生蒸馏可行性

> 历史草案：本段写于MG1b成功之前，其学生架构、A组基线定义、attention提取方式、beta
> 公式和成功门槛均已被下文“MG2全图学生蒸馏适配协议（2026-08-18冻结）”整体取代，
> 保留仅供追溯，不再作为执行依据。

局部教师必须处于`eval()`且完全冻结，前向结果不保留教师梯度。学生为接收完整彩色224x224
WLI的EfficientNet-B0。首版以M0-F同协议的全图分类为受控基线，并使用原论文提供的
`tau=4`、`alpha=0.25`作为logit蒸馏起点。

```text
L_total = alpha * CE(student_logits, y)
        + (1-alpha) * tau^2 * KL(teacher_soft || student_soft)
        + beta * L_attention
```

`L_attention`比较归一化空间关注图。教师ROI关注图必须按实际`crop_box`映射回学生完整图
坐标，`lesion_bbox`只用于癌侧AiB/PGA评价。首版主实验只在癌/HGD图计算空间KD，避免把
YOLO在非癌图上的假阳性位置直接当成可靠空间真值；非癌困难ROI仍参与教师训练和所有样本
的logit KD。`beta`不能直接照搬原论文的300，因为bbox回填和归一化会改变损失尺度；先在
固定50至100个train debug批次记录三个raw loss中位数，再按预先写入config的规则冻结beta，
不能看val/test/external后调整。

MG2主矩阵至少包含：A普通全图控制、B所有样本logit KD、C所有样本logit KD加癌侧attention
KD；将癌与非癌均纳入attention KD只作为可选D消融。A/B/C必须使用相同初始化、训练预算、
患者采样、数据顺序、增强、学习率与checkpoint选择规则，不能让KD组额外获得更多训练。

共享几何增强必须先同步作用于完整图和bbox，再从变换后的图裁教师ROI；教师与学生可使用
分支独立的光度增强。首版只保留已验证的同步水平翻转，关闭随机透视、平移和复杂随机裁剪。
Attention提取层、通道压缩、归一化、距离函数及框外区域是否进入损失，必须在实现前写入
config并通过坐标回填单元测试；不得为了val结果临时改变。首版不强制学生框外注意力为零，
避免把“病灶优先”误实现为“完全禁止全局上下文”。

MG2只在seed42 train拟合、val选择。比较患者AUC、图像AUC、Sensitivity/Specificity、AiB
（bbox内关注质量占比）、PGA（关注峰值是否入框）、颜色扰动稳定性及来源/尺寸分层。

### MG3：三种子复现

MG2配置冻结后独立运行seed42/202/503，不再改损失、epoch、bbox规则或checkpoint选择。
正式成功要求分类与关注两方面同时成立：

- 3/3 seed的val患者AUC均高于同seed普通全图控制，且平均绝对提升不少于0.01；
- 3/3 seed的AiB与PGA不低于控制，平均至少一项提升不少于0.05；
- 没有来源、尺寸或小病灶主要分层出现预注册安全界限外的系统性退化。

若分类未达标但关注稳定改善，只能结论为“空间关注改进、分类收益未证实”；若教师未过MG1
内容有效性门槛，则停止，不把教师噪声继续蒸馏给学生。

### MG4：锁定投影与解释

MG3全部选择冻结后，才可对已揭盲internal test和external作一次描述性投影。既有external
不能用于挑seed、集成权重或返调蒸馏参数。MG4同时比较M0-F、MAGE-like学生和旧YOLO+Y6
工程版本，并用Grad-CAM/AiB/PGA说明学生是否更聚焦病灶。若MG3通过，可为MAGE学生另训
SAE；旧M0/Y6 SAE Feature编号不能直接迁移。

## 当前下一步

1. MG0a清单构建、自测和正式审计：已完成；
2. MG0b五折数据视图、GPU冒烟和五个正式OOF检测器：已完成；
3. 合并五折预测并生成唯一教师ROI清单：已完成；癌用GT框，非癌用OOF Top-1，61张train
   非癌走确定性回退；val使用冻结seed42 Y3-F，19张非癌走回退；
4. 非癌置信度、fallback、面积/中心/来源/尺寸和geometry-only审计：已完成；原清单几何
   患者AUC=0.6054，拆分确认信号主要来自裁图大小；
5. 原始、旧尺寸匹配、动态扩边v2及独立轴动态扩边v3审计：已完成；人工QC发现v2仍会让
   细长框在短轴吞入大量背景，v3已解决且两分支源框覆盖均为100%；
6. 独立轴动态v3已人工确认并冻结为MG1唯一训练清单；v3尺寸匹配只归档为诊断。MG1采样、
   两阶段训练、checkpoint规则、固定seed42教师策略及三条数值门槛均已预注册并实现；
7. MG1真实内容与patch-shuffle正式配对已完成：绝对AUC与geometry差值门通过，但真实减
   shuffle门失败，正式决策为`stop_before_MG2`；
8. MG1b已作为新研究问题完成预注册、实现、自测和CPU debug；下一步只运行冻结seed42正式
   实验。其结果不能修改MG1失败结论，也不得利用test/external返调门槛或参数；
9. MG1b seed42正式实验已完成并通过全部五门槛，36张注意力图人工QC也已通过；MG2适配协议
   经用户审阅修正后已于2026-08-18正式冻结（修正KL框外语义、beta按完整非空间目标25%校准、
   新增M0-F安全参照与临床安全门槛、病灶大小改为占完整图面积口径；旧MG2段落标为历史草案）。
   不得直接沿用旧MG1教师或跳过坐标回填验证；
10. MG2代码闭环完成：17/17单元测试、CPU debug与防线回归通过；教师缓存与beta=0.1936已
    冻结（beta协议经用户批准修订为74批完整校准epoch，并新增校准JSON强绑定防线）。下一步
    只剩GPU正式A/B/C矩阵与八门槛汇总，启动前需重新检查GPU并经用户确认。截至2026-08-18，
    MAGE线全部代码与本文改动尚未提交Git，不主动提交。

## MG0a正式结果（2026-08-17）

正式输出：`数据整理记录/MAGE/MG0a_患者级OOF分折_20260817/`。输入Y0-F mapping SHA
与冻结记录一致，仅导出train/val共2847张/1472人；原501张test未导出。

| fold | 非癌图片/患者 | 癌图片/患者 |
| ---: | ---: | ---: |
| 0 | 244/180 | 248/63 |
| 1 | 232/180 | 228/63 |
| 2 | 226/180 | 216/62 |
| 3 | 229/180 | 239/62 |
| 4 | 238/180 | 250/62 |

每位train患者只属于一个holdout fold，五折覆盖全部1212位train患者且无遗漏。val保持独立，
不分配OOF fold。首次正式构建因pandas可空整数哈希序列化失败，半成品保留在同级
`*_failed_nullable_hash`目录；修复后重跑、二次验收和`git diff --check`均通过。该失败不
影响分折内容，也没有读取受锁定数据。

## MG0b实现与冒烟（2026-08-17）

新增正式入口：

- `程序/MAGE/正式代码/build_mage_oof_yolo_views.py`：为每个外层fold建立患者互斥的
  fit/monitor/holdout软链接视图；
- `程序/MAGE/正式代码/train_mage_oof_yolo.py`：单fold训练、断点恢复和未见holdout
  Top-1预测；
- `程序/MAGE/正式代码/run_mage_oof_yolo_matrix.sh`：五折串行、失败保留、完整产物幂等
  跳过和实时日志。

五个fold均通过二次审计：每个fold的fit、monitor、holdout患者互斥；每份mapping完整覆盖
2350张train图；五个holdout合并后恰好覆盖全部2350张且不重复。每个fold只用其余4折中的
患者再按15%患者级分层划monitor，项目正式val不参与OOF检测器早停。

物理GPU 1上的fold0单epoch debug完整跑通：1596张fit、262张monitor、492张holdout；
holdout为248张癌图和244张非癌图，全部获得一一对应预测。最低置信度0.001下共234张有
候选，其中非癌107/244张有候选。该数字只证明候选与回退路径能被执行，**不是检测性能
结论**；1 epoch权重不得进入MG1。

第一次debug的Ultralytics异步绘图线程在Pillow中报告反向显示坐标，但主进程正常完成。
正式入口随后只关闭不参与优化或模型选择的`plots`，保留全部训练参数；第二次干净debug
退出码为0、无Traceback，权重/config/492张holdout预测均通过验收。首次debug现场保存在
同结果目录的`*_plots_thread_warning`，不作为正式产物。

## MG0b五折正式结果（2026-08-17）

正式输出：`结果/MAGE/MG0b_OOF_YOLO_20260817/`。物理GPU 1串行完成5个fold；五份日志均
以`DONE`结束，未发现`FAILED`、Traceback或RuntimeError。每折使用各自`best.pt`对从未参与
该检测器fit/monitor的holdout患者执行一次低阈值Top-1推理。

| fold | 完成epoch | 最佳epoch | monitor mAP50 | monitor mAP50-95 | holdout图片/患者 | 非癌候选覆盖 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 96 | 0.5478 | 0.2346 | 492/243 | 229/244（93.9%） |
| 1 | 100 | 95 | 0.4658 | 0.1993 | 460/243 | 202/232（87.1%） |
| 2 | 100 | 97 | 0.5834 | 0.2707 | 442/242 | 216/226（95.6%） |
| 3 | 92（早停） | 72 | 0.4002 | 0.1844 | 468/242 | 228/229（99.6%） |
| 4 | 100 | 86 | 0.5178 | 0.2461 | 488/242 | 233/238（97.9%） |

五折预测合并后为2350张/1212人，与冻结train队列完全一致：`relative_path`唯一2350，无重复、
无遗漏；标签为非癌1169张、癌1181张；`prediction_provenance`分别记录`oof_fold_0`至
`oof_fold_4`。最低候选置信度0.001下，全部图片2287/2350（97.3%）有候选；非癌1108/1169
（94.8%）有候选，剩余61张（5.2%）需要确定性回退ROI。

这些结果证明OOF数据血缘和困难ROI候选供给成立，不代表最终分类效果，也不要求五个检测器
mAP完全一致。MG1放行仍取决于合并ROI后的置信度、几何可分性、来源/尺寸分层和人工裁图
质控；当前不得仅凭94.8%的候选覆盖率直接开始正式教师训练。

## MG0b教师ROI审计与尺寸匹配敏感性（2026-08-17）

原始正式清单输出：
`数据整理记录/MAGE/MG0b_教师ROI清单与审计_20260817/`。清单唯一覆盖train/val共
2847张；train癌1181张全部使用GT框，train非癌1108张使用患者级OOF Top-1、61张使用
确定性回退；val癌240张使用GT框，val非癌238张使用冻结Y3-F seed42 Top-1、19张回退。
全部框合法且在图内，无患者跨split，未读取test/internal test/external。

train非癌有效候选置信度P25/P50/P75/P90依次为0.01842、0.07947、0.25677、0.47333。
自动抽查显示：癌侧大多覆盖医生标注病灶；非癌侧主要落在皱襞、隆起、幽门、黏膜纹理、
反光和少量器械/边缘等模型容易误判的结构；回退框位于有效黏膜内。因而困难负ROI内容本身
可用，未发现批量越界或空裁图。

但原清单存在中等强度的裁图尺寸信号。仅用train拟合逻辑回归、只在val评价的结果如下：

| 几何输入 | 原清单图像AUC | 原清单患者AUC |
| --- | ---: | ---: |
| 中心坐标 | 0.5439 | 0.5328 |
| 归一化宽高与面积 | 0.6240 | 0.6050 |
| 像素正方形边长 | 0.6270 | 0.6105 |
| 中心加像素边长 | 0.6281 | 0.6116 |
| 中心、宽高与面积 | 0.6246 | 0.6042 |

这说明原几何AUC约0.61几乎全部来自癌GT裁图整体大于非癌预测裁图，而不是ROI位置或原始
画幅。教师虽然不直接接收坐标，但缩放到224后仍可能从视野范围和组织尺度学到标签捷径。

为此另建了不覆盖原清单的敏感性候选：
`数据整理记录/MAGE/MG0b_教师ROI尺寸匹配敏感性_20260817/`。癌图裁图逐坐标保持不变；
非癌以原YOLO/回退中心为放置目标，边长由train癌图在相同来源和尺寸组的分布按图像SHA
确定性抽取，缺组时依次回退到来源级和全局train癌池。大框靠近边缘时，为保证不越界会
向图内作最小平移：1426张非癌中725张发生平移，发生平移者的归一化中心距离中位数为
0.0952。它不使用val标签拟合尺寸池，也不读取锁定数据。

| 几何输入 | 尺寸匹配图像AUC | 尺寸匹配患者AUC |
| --- | ---: | ---: |
| 中心坐标 | 0.5366 | 0.5205 |
| 归一化宽高与面积 | 0.4617 | 0.3823 |
| 像素正方形边长 | 0.4617 | 0.3817 |
| 中心加像素边长 | 0.5506 | 0.5046 |
| 中心、宽高与面积 | 0.5457 | 0.4893 |

完整几何患者AUC从0.6042降至0.4893，说明可利用的联合几何信号已接近随机。尺寸单项AUC
低于0.5反映val癌框与train癌框分布并非完全相同，不能解释为反向临床判别能力；因此仍需
在MG1报告几何基线，而不是宣称泄露被永久消除。视觉抽查确认匹配后仍围绕YOLO候选或其
图内约束位置取景，但部分非癌裁图扩大到接近全视野，局部信息会被稀释。

该阶段曾暂定“旧尺寸匹配作主实验、原始ROI作敏感性”，但随后发现旧像素正方形会截断
部分长条GT框，因此这项角色安排已被下述动态扩边v2审计取代，不再作为当前执行口径。

## MG0b动态扩边ROI v2（2026-08-17）

进一步审计发现，旧版像素正方形必须限制在图内，遇到宽度或高度超过图像短边的长条框时
无法完整包含源框。train/val共1421张癌图中74张（5.2%）存在该几何冲突，23张癌图的旧
黄色框覆盖不到绿色GT框面积的90%，最差覆盖率约81.0%。因此，仅给旧像素正方形增加0.85
短边上限仍可能截断病灶。

v2改为与原图保持相同像素宽高比的裁图。在归一化坐标中令裁图宽高相等，定义：

```text
source_scale = max(source_box归一化宽度, source_box归一化高度)
target_scale = min(1.4 * source_scale, max(source_scale, 0.85))
```

小框仍获得每侧20%上下文；大框最多扩到0.85；源框本身超过0.85时保持源框尺度、不缩小。
裁图在图内平移以完整包含源框，随后统一缩放到224×224，其宽高比变换与现有完整图分类
预处理一致。正式候选输出：
`数据整理记录/MAGE/MG0b_动态扩边ROI_v2_20260817/`。

| 分支与标签 | P50尺度 | P75尺度 | P90尺度 | 撑满全图 | 最小源框覆盖 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 动态v2 train非癌 | 0.6711 | 0.8500 | 0.8500 | 2/1169 | 100% |
| 动态v2 train癌 | 0.8407 | 0.8500 | 0.8701 | 0/1181 | 100% |
| 动态v2 val非癌 | 0.6918 | 0.8500 | 0.8500 | 0/257 | 100% |
| 动态v2 val癌 | 0.7902 | 0.8500 | 0.8529 | 1/240 | 100% |

v2将旧方案中常见的全图撑满降至3/2847，同时把所有源框覆盖恢复为100%。自动视觉抽查
确认绿色框始终位于黄色框内，长条病灶、边缘病灶及非癌困难结构均可见。

几何审计结果：

| 几何输入 | 动态v2图像AUC | 动态v2患者AUC | 动态+尺寸匹配图像AUC | 动态+尺寸匹配患者AUC |
| --- | ---: | ---: | ---: | ---: |
| 中心 | 0.5378 | 0.5310 | 0.5372 | 0.5242 |
| 尺寸 | 0.6033 | 0.6047 | 0.5195 | 0.6244 |
| 中心+尺寸 | 0.6113 | 0.6076 | 0.5676 | 0.6031 |

尺寸匹配令图像级尺寸AUC接近随机，但患者级没有同步下降。原因是癌患者通常拥有更多图片，
并且必须以真实大GT框作为裁图下限；逐图确定性匹配后，多图均值仍保留患者级尺度分布差异。
继续根据val结果调匹配参数会构成开发集追参，因此当前停止调整，不宣称v2尺寸匹配已完成
患者级去偏。

v2解决了病灶截断和大部分全图撑满，但随后人工QC发现它仍把归一化宽高设为同一尺度：细长
绿色源框会在短轴方向被补成很大的黄色框。故v2不再作为MG1首选，只保留为“固定图像宽高
比裁图”的历史对照；当前执行口径由下述独立轴动态扩边v3取代。

## MG0b独立轴动态扩边ROI v3（2026-08-17）

针对v2细长框短轴过扩，v3不再计算统一`target_scale`，而是分别对归一化宽、高执行：

```text
target_width  = min(1.4 * source_width,  max(source_width,  0.85), 1.0)
target_height = min(1.4 * source_height, max(source_height, 0.85), 1.0)
```

因此每个方向仍保留每侧20%上下文，但一个方向很长不会迫使另一个方向同步变长。裁图允许
保持长方形，完整包含绿色源框并位于图内，之后直接缩放为224x224灰度教师输入。正式候选
输出：`数据整理记录/MAGE/MG0b_独立轴动态扩边ROI_v3_20260817/`。

全量2847张的源框覆盖率均为100%，没有全幅裁图。裁图面积相对源框面积的倍率由v2的中位
2.28、P90 3.29、最大8.05，收敛为v3的中位/P90/最大均1.96。用户指出的两张细长癌图中，
黄色框占全图面积分别由72.25%降至48.26%、由59.70%降至27.10%，且绿色GT框完整保留。
专项对照位于`independent_axis_dynamic/qc/manual_reported_elongated_gt_examples.jpg`。

| 分支与标签 | 宽P50 | 高P50 | 面积P50 | 裁图/源框面积P50 | 全幅裁图 | 最小源框覆盖 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 动态v3 train非癌 | 0.5240 | 0.6154 | 0.3262 | 1.96 | 0/1169 | 100% |
| 动态v3 train癌 | 0.6600 | 0.7601 | 0.4832 | 1.96 | 0/1181 | 100% |
| 动态v3 val非癌 | 0.5268 | 0.6464 | 0.3409 | 1.96 | 0/257 | 100% |
| 动态v3 val癌 | 0.6219 | 0.7262 | 0.4171 | 1.96 | 0/240 | 100% |

仅在train拟合、只在val评价的几何审计如下。v3加入宽高比与像素宽高比是为了审计矩形形状
本身是否泄露标签；这些坐标不会直接输入教师分类器。

| 几何输入 | 动态v3图像AUC | 动态v3患者AUC | v3尺寸匹配图像AUC | v3尺寸匹配患者AUC |
| --- | ---: | ---: | ---: | ---: |
| 中心 | 0.5428 | 0.5351 | 0.5522 | 0.5433 |
| 尺寸与形状 | 0.6128 | 0.6033 | 0.5948 | 0.6708 |
| 中心+尺寸与形状 | 0.6196 | 0.6092 | 0.6144 | 0.6577 |

独立轴动态v3明显改善局部紧致度，但没有消除“癌GT框通常比非癌预测框大”的固有几何差异，
所以正式教师仍必须同时报告geometry-only与patch-shuffle对照。v3尺寸匹配虽然按train癌图
成对抽取宽高，患者级完整几何AUC反而升至0.6577，且非癌裁图/源框面积P90达到约10.25；
它既未完成患者级去偏，也再次稀释了局部内容。因此**冻结独立轴动态v3为MG1主候选，v3
尺寸匹配仅保留为失败的诊断敏感性，不进入首版正式训练**。不得继续根据val追调匹配规则。

## MG1实现与冒烟（2026-08-17）

新增正式入口：

- `train_mage_mg1_teacher.py`：同一入口通过`--input-mode real|patch_shuffle`运行配对实验；
- `summarize_mage_mg1.py`：绑定v3几何患者AUC并执行三条冻结成功门槛；
- `run_mage_mg1_pair.sh`：GPU前置检查、真实/shuffle串行运行、残缺产物拒绝、实时日志与统一
  汇总。

patch-shuffle单元测试确认7×7重排前后像素多重集合完全一致，同一图像/epoch/seed排列可复现；
患者类别平衡采样测试通过。CPU debug各取train每类3位患者和val每类3位患者，两种输入均完成
阶段A、阶段B、最佳checkpoint复算、图片/患者预测、config和数据边界落盘。debug真实患者
AUC 0.8889、shuffle患者AUC 0.7778只反映极小样本工程链路，**不是MG1性能结果，也不能用于
放行MG2**。正式结论必须等待全量seed42配对运行和统一汇总。

## MG1正式结果（2026-08-17）

正式输出：`结果/MAGE/MG1灰度局部教师_20260817/正式验证集筛选/`。GPU 1串行完成真实
灰度ROI和patch-shuffle两组；两组均使用同一v3 manifest SHA、seed42、ImageNet初始化、
患者/类别平衡采样和两阶段训练。真实组最佳为stage B epoch 4，shuffle组最佳为stage B
epoch 16。未读取test、internal test或external。

| 对照 | val图像AUC | val患者AUC | 患者Sens | 患者Spec | 患者FPR | 患者FNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 真实灰度v3 ROI | 0.8142 | 0.8572 | 0.9104 | 0.6010 | 0.3990 | 0.0896 |
| 7×7 patch-shuffle | 0.7812 | 0.8609 | 0.9104 | 0.6321 | 0.3679 | 0.0896 |
| geometry-only | 未作为图像分类器比较 | 0.6092 | - | - | - | - |

Sensitivity/Specificity在各自val患者级满足Sensitivity不低于0.90的最高实际阈值处报告；真实
阈值为0.2169，混淆矩阵`TN=116, FP=77, FN=6, TP=61`；shuffle阈值为0.2322，混淆矩阵
`TN=122, FP=71, FN=6, TP=61`。

冻结三门判定：

| 门槛 | 实际 | 是否通过 |
| --- | ---: | --- |
| 真实患者AUC >= 0.80 | 0.8572 | 是 |
| 真实减geometry患者AUC >= 0.10 | +0.2481 | 是 |
| 真实减shuffle患者AUC >= 0.05 | -0.0036 | **否** |

真实教师在图像级比shuffle高0.0330，说明完整空间排列对部分单张图可能有帮助；但患者均值
聚合后shuffle反而高0.0036，没有建立预注册要求的患者级空间形态独立增益。更合理的解释是：
当前局部教师的主要可复现信号来自灰度纹理、亮度分布和局部patch集合，EfficientNet加全局
平均池化可以在空间顺序被破坏后继续利用这些“纹理袋”信息。不能据此声称病灶形态完全无用，
但也不能把当前教师称为已证明依赖病灶空间结构的教师。

因此正式汇总`mg1_pair_summary.json`记录`passed_mg1=false`、
`decision=stop_before_MG2`。本轮MG1按预注册判定失败，当前checkpoint保留作诊断，不进入
正式MG2。若继续，应新开MG1b并在运行前重新定义研究问题和对照，而不是放宽本轮0.05门槛。

## MG1b bbox约束注意力池化教师预注册（2026-08-18）

MG1b不修改MG1失败结论，也不再重复询问“普通GAP分类器能否自发依赖完整空间排列”。新的
问题固定为：**让分类特征必须经过bbox可监督的空间注意力池化后，能否在基本保持分类性能的
同时，形成可用于后续空间蒸馏的病灶对齐注意力？**

### 架构与数据流

输入仍唯一使用独立轴动态v3主清单的224×224 luma三通道ROI，不正方形化、不letterbox、
不使用尺寸匹配分支。EfficientNet-B0最后卷积特征为`[B,1280,7,7]`，新增一个`1×1 Conv`
生成49个attention logits，经空间softmax得到总和为1的注意力。分类不再使用普通GAP，而是：

```text
feature [B,1280,7,7]
  -> 1x1 Conv -> spatial softmax -> attention [B,1,7,7]
  -> sum(feature * attention) -> [B,1280]
  -> dropout + Linear -> cancer/non-cancer logits
```

因此分类头没有绕开attention的GAP旁路。癌图医生`lesion_bbox`先换算为黄色v3`crop_box`内的
相对矩形，再计算其与7×7每个cell的精确相交面积，归一化为目标空间分布；水平翻转时图像与
目标同步翻转。非癌没有真实病灶框，只计算分类损失，不伪造空间目标，但其attention仍通过
分类梯度学习。

### 损失与训练

```text
L_MG1b = CrossEntropy(logits, label, label_smoothing=0.1)
       + 0.25 * mean_cancer KL(target_bbox_distribution || attention)
```

`lambda_attention=0.25`在正式运行前固定，不根据val、test或external调整。seed42、患者/类别
平衡有放回采样、batch size 32、AdamW及weight decay `1e-4`沿用MG1。阶段A冻结backbone，
只训attention head与classifier 5 epoch、学习率`1e-3`；阶段B解冻`features[5:]`及两个head，
最多20 epoch、学习率`1e-4`、patience 6。首版不做bbox jitter，val无随机增强。

### 空间指标与冻结门槛

- `AiB`：癌图attention落入bbox的质量比例；随机均匀attention的期望等于bbox在crop内面积。
- `normalized_AiB=(AiB-bbox_area)/(1-bbox_area)`；0表示仅达到均匀基线，1表示全部质量入框。
- 当`bbox_area>=0.99`时该归一化分母接近0，normalized AiB记为不可评估并从其均值中排除，
  但仍报告AiB与PGA。正式val仅1/240张癌图达到该条件，不影响其余239张的主空间门槛；
  排除数必须写入config，禁止将其静默记为0或1。
- `PGA`：attention峰值cell中心落入bbox的癌图比例。
- 病灶大小三分位只用train癌图的`bbox_area/crop_area`冻结，再用于val分层；样本不足15张的组
  只报告、不作硬门槛。

MG1b epoch必须先同时满足以下条件才有资格成为产品：

1. val患者AUC `>=0.8472`，即不低于MG1真实教师0.8572超过0.01；
2. val图像AUC `>=0.8042`，即不低于MG1真实教师0.8142超过0.01；
3. val癌图平均normalized AiB `>=0.30`；
4. val癌图PGA `>=0.80`；
5. 样本数不少于15的small/medium/large组PGA均`>=0.70`。

合格epoch按患者AUC、图像AUC、normalized AiB、PGA、较低val总loss依次选择。若没有任何
epoch全部合格，则MG1b失败并停止，不保存可进入MG2的正式教师；可保存诊断checkpoint但必须
明确标记`eligible=false`。MG1b不以patch-shuffle为成功门槛，因为它回答的是一个新的、明确
受监督的空间教师可行性问题；但MG1原shuffle结果必须继续并列报告，不能被MG1b覆盖。

### MG1b实现与冒烟（2026-08-18）

正式入口为`程序/MAGE/正式代码/train_mage_mg1b_attention_teacher.py`，GPU启动器为
`run_mage_mg1b.sh`。实现已验证：bbox到7×7精确相交分布、水平翻转同步、癌图KL掩码、
attention总和为1、分类无GAP旁路、患者/类别平衡采样、冻结阶段切换、五门槛选择及失败产品
隔离。全量坐标审计覆盖1421张癌图，网格面积最大误差`2.4e-7`；正式val有2/240张癌图的
`bbox_area>=0.99`，按预注册只从normalized AiB均值排除，AiB/PGA仍报告。

CPU debug使用train 6张/2人、val 5张/2人完成两阶段端到端链路。小样本未通过空间门槛，
脚本按设计只保存`mg1b_best_diagnostic_ineligible.pth`并返回非零状态；这只证明失败隔离有效，
不构成性能结果。debug config确认未读取test、internal test或external，逐图49个attention权重
求和最大误差小于`1e-7`。

### MG1b正式结果（2026-08-18）

正式输出：`结果/MAGE/MG1b注意力池化教师_20260818/正式验证集筛选/`
`mg1b_attention_efficientnet_b0_seed42/`。使用GPU 1运行冻结seed42，阶段B第4轮成为正式产品；
第10轮后因连续6轮无预注册排序改善早停。修复仅影响float64病灶分层归属的实现细节后，
使用完全相同参数重跑，模型state逐tensor与修复前一致，最大绝对差为0。

| 指标 | 冻结门槛 | MG1b正式值 | 结果 |
| --- | ---: | ---: | --- |
| val患者AUC | >=0.8472 | 0.8736 | 通过 |
| val图像AUC | >=0.8042 | 0.8313 | 通过 |
| mean normalized AiB | >=0.30 | 0.5771 | 通过 |
| PGA | >=0.80 | 0.9458 | 通过 |
| 三个病灶大小组PGA | 每组>=0.70 | 0.9107 / 0.9839 / 0.9697 | 通过 |

与MG1真实灰度教师相比，患者AUC从`0.8572`升至`0.8736`（+0.0163），图像AUC从
`0.8142`升至`0.8313`（+0.0171）。在240张val癌图中，平均AiB=`0.8208`，表示约82.1%的
注意力质量落在医生bbox覆盖区域；扣除框面积带来的均匀基线后，normalized AiB=`0.5771`。
PGA=`0.9458`，即227/240张癌图的最高注意力网格中心位于bbox内。1张`bbox_area>=0.99`图
不参与normalized AiB均值，但仍参与AiB和PGA。

| 病灶大小组 | 图片/患者 | mean AiB | mean normalized AiB | PGA |
| --- | ---: | ---: | ---: | ---: |
| small | 112/57 | 0.8005 | 0.5927 | 0.9107 |
| medium | 62/37 | 0.8190 | 0.6131 | 0.9839 |
| large | 66/31 | 0.8569 | 0.5158 | 0.9697 |

按患者Sensitivity>=0.90的最高阈值`0.3049`，患者级Sensitivity=`0.9104`、Specificity=
`0.6373`、Accuracy=`0.7077`、F1=`0.6162`，混淆矩阵`TN=123, FP=70, FN=6, TP=61`；
FPR=`0.3627`，FNR=`0.0896`。该阈值指标用于描述，不参与checkpoint选择。

**能确认**：在无GAP旁路、分类必须经过attention池化的条件下，bbox监督同时保持并提高了
内部val分类AUC，而且注意力在总体及三种病灶大小组均与病灶框稳定对齐。因此MG1b按预注册
判定通过，`decision=proceed_to_MG2`。

**不能确认**：空间指标是bbox监督后的预期结果，不等于已证明模型关注了框内的正确病理
语义；仍需逐图人工QC排查框内反光、器械、黏液等伪特征。当前只有seed42内部val结果，尚未
证明蒸馏给全图学生后可提升三种子、internal test或外部泛化。test、internal test和external
均未读取，三项config标记均为`false`。

已生成36张只读人工QC三联图，small/medium/large各12张，按各组normalized AiB排序等距
抽样，覆盖低、中、高对齐表现。每张依次展示完整图上的黄色crop/绿色病灶框、教师实际灰度
ROI、jet注意力叠加图；索引与血缘记录位于正式产品的`attention_qc/`。该QC不参与checkpoint
选择，需由人工重点检查高响应是否落在真正病灶组织，而不是框内反光、器械、黏液或边缘。

2026-08-18人工复核结论：绝大部分抽查图的主要注意力与病灶区域一致，少部分图会同时关注
器械，但总体质量足以作为后续蒸馏教师使用。因此MG1b人工QC判定为“通过，保留器械伪特征
风险”，正式放行MG2协议设计。该结论不等于教师完全没有伪特征；MG2/MG3必须继续保存并
报告器械关注失败例，验证全图学生是否继承或放大这类响应。

## MG2全图学生蒸馏适配协议（2026-08-18冻结）

本节为正式预注册，经用户逐条审阅草案后修正冻结，整体取代“正式阶段路线”中的旧MG2段落
（该段保留为历史草案，仅供追溯）。相对旧段落的实质变更不止教师来源，还包括：学生架构、
A组定义、attention提取方式、beta公式和成功门槛。

### 教师来源（冻结）

教师固定为MG1b正式产品`mg1b_best_teacher.pth`，checkpoint SHA256
`f2cd3b13cbbc0e8b4df64e9090ec50343314137ba71a4c07d9fae9faa8af48f9`（用户已核验一致），
清单为独立轴动态v3主清单（SHA256
`b1dfd24505cbba876fd28502e36b0b190206de8f063562f47347dc7c67f32e33`）。
全程`eval()`、`requires_grad=False`，不重训、不微调、不替换。教师同时提供：
分类logits（logit KD用）和已经过空间softmax的`7x7`注意力（attention KD用，总和为1，
无需再从GAP模型额外提取关注图）。教师输入为按`base_crop_box`裁出、缩放到224x224的
luma三通道ROI；MG2首版不做bbox jitter，与MG1b输入口径一致。

### 学生架构（冻结，D1）

学生为接收完整彩色224x224 WLI的EfficientNet-B0（ImageNet初始化）。A/B/C三组统一使用
与MG1b教师相同的attention-pooling结构（`features[8] -> 1x1 Conv -> 空间softmax ->
加权和 -> dropout + Linear`，无GAP旁路），差异只在损失：

- A普通全图控制：仅CE，attention头仅靠分类梯度学习（无任何bbox/蒸馏监督）；
- B：A + 所有样本logit KD；
- C：B + 癌侧attention KD；
- 可选D消融：癌与非癌均纳入attention KD。

三组架构完全一致，AiB/PGA在每组都有定义，C相对A的空间对齐增益可归因于bbox引导蒸馏
而非架构差异。学生因此不是M0-F GAP的逐参复制品；为防A组本身退化，M0-F seed42在MG2
内部作为固定安全参照（见放行门槛1-2），MG4仍保留M0-F外部参照比较。

### 损失与beta冻结规则（冻结，D2）

```text
L_total = alpha * CE(student_logits, y, label_smoothing=0.1)
        + (1-alpha) * tau^2 * KL(teacher_soft || student_soft)      # B/C，全部样本
        + beta * mean_cancer KL(backfilled_teacher_attention || student_attention)  # 仅C/D
```

`tau=4`、`alpha=0.25`沿用原论文起点。坐标回填：把教师`7x7`注意力视为crop矩形上的
面积密度，按与实际`crop_box`（含同步翻转后的变换）的面积相交关系分摊到学生完整图
`7x7`网格，再归一化为总和1的目标分布；crop外目标质量为0。

框外语义（修正后口径）：虽然KL(target||student)在target=0的位置单项恒为0，但学生
注意力经全图spatial softmax后总质量固定为1，crop外质量越多、crop内质量越少、crop内
KL越大。因此完整KL的实际作用是**对crop外注意力形成软抑制、把质量推向crop内**，而不是
硬掩膜置零；分类CE仍可抵抗这种压力、保留有诊断价值的全局上下文。首版保留完整KL
（不改为crop内条件归一化KL，因为提高整体AiB本来就是目标之一），同时必须报告学生
crop内/外注意力质量。

beta冻结规则（2026-08-18经用户批准修订，替代原“固定100个train批次”表述）：正式矩阵
运行前，用冻结教师和seed42、batch size 32、患者/类别平衡有放回采样器，在固定seed42下
运行**一个完整校准epoch，共74个固定批次、2350次有放回抽样**，记录每个批次的完整非空间
目标与attention项，再按

```text
L_nonspatial = alpha * CE + (1-alpha) * tau^2 * KL_logit   # 加权后的分类+logit蒸馏总量
beta = 0.25 * median(L_nonspatial_batch) / median(L_attention_batch)
```

冻结并写入config。语义明确为：attention项初始量级约为“分类+logit蒸馏总量”的25%，
作为首版保守强度。修订原因：train仅2350张，batch 32下一个epoch只有74个批次，原冻结
数字100与该规模不自洽；74批与正式训练的batch size和采样分布完全一致，中位数已足够
稳定。注意有放回采样下2350次抽样**不是2350张图各出现一次**，允许重复也可能漏图；因此
校准口径写为“覆盖一个完整的正式训练采样epoch”，校准JSON必须保存每批SHA、重复数、
唯一图片数和标签/患者分布，并记录校准设备（遵循`--device`，与正式训练保持一致）。
该规则在查看任何MG2 val指标之前执行；debug批次的loss数值本身不构成val结果，允许记录。
正式矩阵运行后不得再调beta；若训练明显不稳，只能停止并作为新实验重新预注册。

beta来源强绑定（冻结防线）：C组正式训练必须提供`--beta-calibration-json`，beta只允许
从校准JSON读取，禁止手工输入`--beta`并声明已冻结；训练前必须校验JSON中的manifest SHA、
教师checkpoint SHA、教师缓存SHA、seed、batch size和校准规则与当前运行一致，校准JSON
自身SHA写入训练config。正式校准完成后，其JSON SHA256冻结为
`47179b6228122096c00e6f0e59ab5c45663b3f163a506c77aa8811320803ac7f`；正式C组必须逐字符
核验该校验和等于冻结值，并重算`beta == 0.25 * median_nonspatial / median_attention`、
核验74批/2350次抽样、`alpha=0.25`、`tau=4`与三项数据锁定标记，即使JSON被修改后重新
生成sidecar也不得通过。A/B组不涉及attention项，不需要校准文件，传入即拒绝。

### 数据与增强（冻结）

- 数据唯一来自v3主清单的train 2350张/1212人与val 497张/260人；test、internal test、
  external不进入任何环节，config三项标记必须全为`false`。
- 共享几何增强只有同步水平翻转（p=0.5）：先翻转完整图并同步变换`lesion_bbox`与
  `base_crop_box`，再从变换后图像裁教师ROI。关闭M0-F P6中的RandomResizedCrop、
  rotation等一切几何增强。
- 学生分支保留P6光度增强（ColorJitter、downsample、JPEG、高斯模糊，均为非几何操作），
  三组一致；教师分支不做任何光度增强，保持luma ROI口径。
- 教师前向只依赖图像与翻转状态，正式运行前对`2847张 x 2种翻转`预计算并缓存教师logits
  与attention，缓存绑定教师checkpoint与manifest SHA；B/C必须使用同一份缓存。

### 训练与checkpoint（冻结）

seed42；学生从ImageNet初始化；两阶段沿用M0-F口径：阶段A冻结backbone，只训
attention head与classifier，10 epoch、lr `1e-3`；阶段B解冻`features[5:]`及两个head，
最多20 epoch、lr `1e-4`、weight decay `1e-4`、patience 8（对齐M0-F正式配置）；batch 32；
患者/类别平衡有放回采样；AdamW。三组使用相同初始化、数据顺序、采样、增强与预算。
checkpoint按val患者AUC最高、再按图像AUC最高、再按val loss最低选择，与MG1/MG1b一致。

### 评价指标（冻结）

- 学生侧AiB/PGA在val癌图上以完整图坐标下的GT `lesion_bbox`计算；`bbox_area`指
  `lesion_bbox面积/完整图面积`，>=0.99时normalized AiB记为不可评估并从均值排除，
  AiB/PGA仍报告，排除数写入config。
- 病灶大小分层重定义为`lesion_area_fraction = GT bbox面积 / 完整图面积`，表示病灶在
  完整胃镜视野中的真实相对大小；三分位只能由train癌图冻结，再应用到val；样本不足15张
  的组只报告、不作硬门槛。
- 同时报告：患者/图像AUC、各组按“val患者Sensitivity>=0.90取最高Specificity”规则选定
  阈值后的Sensitivity/Specificity/Accuracy/F1、混淆矩阵与FP/FN/FPR/FNR、来源/尺寸分层、
  学生crop内/外注意力质量、学生灰度化val副本的颜色扰动稳定性（仅描述）、以及学生器械
  关注失败例抽查（承接MG1b QC保留风险，验证学生是否继承或放大该响应）。

### MG2放行门槛（冻结，D3，八条须同时满足）

MG2全部在seed42 train拟合、val评价。进入MG3必须同时满足：

1. A组val患者AUC `>= 0.9033`（M0-F seed42患者AUC 0.9133 - 0.01，架构更换的安全参照）；
2. C组val患者AUC `>= 0.9083`（M0-F seed42 - 0.005）；
3. C组val患者AUC - A组val患者AUC `>= 0.005`；
4. C组val图像AUC `>=` A组val图像AUC `- 0.005`；
5. C组癌图mean normalized AiB与PGA均不低于A组，且至少一项提升 `>= 0.05`；
6. B组val患者AUC `>=` A组 `- 0.01`（logit KD本身不显著伤害分类的sanity门槛）；
7. 在各自冻结阈值下，C相对A最多多漏诊1名val癌患者，且C患者Specificity下降不超过
   `0.03`；
8. 新口径small病灶组PGA下降不超过`0.05`；若该组不足15张，门槛8降级为只报告，并在
   结论中显式披露。

八条全过才冻结配置进入MG3三种子复现。若门槛5通过但门槛1-4或7-8中有失败，停止并只
保留“空间关注改进、分类收益未证实”的诊断结论，不进入MG3；其余失败同样停止，不得放宽
门槛或改用test/external补救。

### 实施顺序（冻结）

1. 实现`train_mage_mg2_student.py`等正式入口，先通过坐标回填单元测试（回填质量总和为1、
   全部位于crop内、翻转一致性、合成用例round-trip）和教师缓存一致性校验；
2. CPU小样本debug验收完整链路；
3. 按修订规则运行一个完整校准epoch（batch 32、74个固定批次、2350次有放回抽样），
   记录两项raw loss并按规则冻结beta；校准JSON保存每批SHA、重复数、唯一图片数、
   标签/患者分布、校准设备及自身SHA；
4. 补齐A/B/C顺序启动器（实时日志、失败保留现场）和八门槛汇总器（自动读取冻结beta
   JSON、三组完整性检查、门槛自动判定），自测与CPU debug回归通过后，GPU正式运行A/B/C
   三组，统一汇总并执行八条门槛判定。

## MG2实现、beta冻结与运行防线（2026-08-18）

正式入口均已落在`程序/MAGE/正式代码/`：`train_mage_mg2_student.py`（A/B/C同一入口）、
`build_mage_mg2_teacher_cache.py`、`summarize_mage_mg2.py`（八门槛汇总）、
`run_mage_mg2_matrix.sh`（GPU启动器，CUDA_DEVICE传入不硬编码）、
`test_mage_mg2_backfill.py`（9项）与`test_mage_mg2_beta_and_gates.py`（8项）。
两套单元测试17/17通过；CPU debug与防线回归通过（正式C缺校准JSON被拒、A/B收JSON被拒、
手工声明beta被拒、篡改缓存/SHA被拒、汇总器对旧占位产物拒汇总）。

教师缓存已建成：`结果/MAGE/MG2全图学生蒸馏_20260818/teacher_cache_mg1b_v3.pt`，
2847张x2翻转状态，SHA256 `89f3d329d68b182b33cad6f9a074e4cd919fb910bca96fff9d9ab16427d9e7db`，
与教师checkpoint、manifest三者SHA绑定校验通过。

beta已正式冻结（GPU 1执行一个完整校准epoch）：**beta=0.19362648121926898**；
74个固定批次、2350次有放回抽样，median(L_nonspatial)=0.8787、median(L_attention)=
1.1345；抽样审计为唯一图1352张、重复998次、最多单图8次、未抽中998张、标签抽样
1167/1183、唯一患者957人；校准设备cuda。校准JSON
`结果/MAGE/MG2全图学生蒸馏_20260818/beta_calibration_seed42.json`，自身SHA256
`47179b6228122096c00e6f0e59ab5c45663b3f163a506c77aa8811320803ac7f`（sidecar同名
`.sha256`）。该步骤只读train，未读取val/test/internal test/external。

过程中发现并修复一个实现bug：校准路径漏把label/teacher_logits/target/has_target搬到
GPU，首次CUDA校准报设备不一致；已按训练循环同款写法修复，修复后正式校准与CPU debug
回归均通过。该bug不影响此前任何结果（当时校准未产出）。

正式A/B/C矩阵于2026-08-18首次启动，A组完成stageA 10轮后后台进程无异常栈地结束；
未留下checkpoint、config或正式预测产物，不构成可用实验结果，也未读取test/internal test/
external。日志已保留，启动器改为追加日志后，将使用脱离终端的用户级systemd服务从A组重新完整运行；
A/B/C全部成功后由启动器自动执行八门槛汇总。

## 参考依据

- 原方法：MAGE: Color-Invariant and Spatial Knowledge Distillation for Gastric Neoplasm
  Classification，arXiv:2607.12663v1（2026-07-14）。
- 原论文训练输入为完整WLI、像素级mask和二分类标签；推理只保留完整图主模型。
- 原论文报告`tau=4`、`beta=300`、batch size 32、224输入，但未公开完整实现细节；本项目
  bbox回填后的attention尺度不同，因此beta必须在本项目内预注册校准。
