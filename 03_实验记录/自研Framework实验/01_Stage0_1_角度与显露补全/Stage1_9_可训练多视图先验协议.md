# Stage 1.9 可训练多视图先验协议

> 状态：**Stage 1.9-0/1与新模型CPU契约已通过：资格池冻结128 train / 32 val、35 / 9场景；`SourceFeatureTargetViewNet`核心39/39测试通过。下一步是单train端点过拟合。**

## 1. 阶段问题

Stage 1.8 已证明三件事：显式目标相机条件具有可学习性；公开 GenWarp/ViewCrafter 先验能明显优于黑洞式 warp；但它们在独立 val 的目标内容绝对误差均未达到 `MAE≤35`。因此下一步不再更换二维补洞器，而验证一个自训的、部署时仍保持单图输入的多视图监督先验。

这里的“多视图”指训练监督来自同场景不同真实相机，并不把多张目标图作为部署输入。部署契约仍是：单张源 RGB、冻结 RGB-only BaseDepth、源相机、显式目标相机。

## 2. 第一轮零下载数据扩展

HLP-TRAIN-03 的50个独立场景均已本地保存左端、中心、右端三帧 RGB/depth/position/pose。v1只使用约15°的四类有向请求：

1. 中心→左端；
2. 中心→右端；
3. 左端→中心；
4. 右端→中心。

禁止左端→右端或右端→左端，因为那会把“总范围30°”误写成单侧约30°。初始得到160 train、40 val，共200个候选端点，不读取三个 held-out test 场景，不使用P01。v1审计发现反向yaw并不因矩阵求逆而严格对称，实际范围6.51°–20.22°，因此按原12°–18°门失败并保留；v2拒绝5个越界train反向对，最终冻结155 train、40 val，共195个端点，绝对yaw范围12.52°–17.90°。

新增的100个端点源帧只需用现有 UniDepth v1 权重生成冻结 RGB-only BaseDepth；全部真实 RGB-D 和三分区掩码仍只属于 label/evaluator。资格门与 Stage 1.8 一致：相机/position-depth一致性通过，`occlusion_hidden`和`outside_source_fov`均非空，所有输入哈希固定。

## 3. 模型变量

旧 `TargetViewUNet` 只看到已经投到目标空间的RGB-D、coverage和目标射线；洞区输入本身为空，源图全局语义也在稀疏warp时丢失。Stage 1.9 的唯一结构变量是改为 `SourceFeatureTargetViewNet`：

- 源图多尺度encoder首先在完整源图上提取局部语义；
- 按冻结BaseDepth和相机把多尺度源特征投到目标空间，而不是只投RGB；
- 源图全局池化向量通过FiLM/广播条件进入目标decoder，使视锥外洞区仍能感知场景类别、颜色和布局；
- 目标射线、目标相机原点、warp coverage继续保留；
- 输出仍为目标RGB、Z-depth、两类独立support、geometry confidence与appearance confidence；
- observed区域使用warp残差路径，两个新显露区使用生成路径，但共享目标空间decoder。

首轮复用本地 ImageNet ResNet-50 权重做冻结encoder冒烟，避免下载；它只是初始化先验，不作为质量结论。若冻结encoder无效，再以单变量方式比较末两级微调，不能同时扩大数据和更换损失。

## 4. 执行阶梯与短路门

1. `Stage 1.9-0`：构建候选有向视图对并审计本地文件、真实角度、平移和泄漏；canonical v2已以195/195通过。
2. `Stage 1.9-1`：为新增95个端点源帧生成冻结BaseDepth，建立源相对三分区真值并形成严格合格池；资格不足时才决定是否定向下载更多Hypersim帧。
3. `Stage 1.9-2`：1个端点过拟合，要求复现Stage 1.8的可学习性门；失败则修模型实现。
4. `Stage 1.9-3`：8 train场景/2 val场景小候选，检查训练是否下降、双support是否分离；失败则不扩全量。
5. `Stage 1.9-4`：40 train/10 val正式候选。两类区域分别要求相对point-warp RGB改善≥15%、RGB MAE≤35、深度AbsRel≤0.30、support正负分离≥0.10。
6. 只有两类均通过，才一次性读取held-out HLP-GEO/HLP-APP；test通过后才接P01与Gaussian Spawn。

任何一级失败均保存现场并短路后续。不能用两类区域平均值掩盖单分支失败，也不能用目标RGB-D作为模型输入。

## 5. 下载边界

Stage 1.9-0至1.9-4原则上全部复用本地资产：HLP-TRAIN-03三帧数据、UniDepth、ResNet-50、真实相机和现有评价器。当前下载需求为零。

只有1.9-1资格审计证明195候选中合格数量不足，或1.9-4显示train明显通过但val受场景多样性限制时，才冻结新的精确场景/帧manifest并一次性定向下载；不得预先下载完整Hypersim、RealEstate10K或CO3D。

## 6. Stage 1.9-0 实际结果（2026-08-18）

- 50个场景的三帧RGB/depth/position/pose全部本地存在，200个初始有向对唯一且无held-out交集；
- v1没有假定反向角度对称，实测5个train反向对落在冻结12°–18°带外，因此状态failed；
- v2只剔除这5个越界对，不替换场景、不扩大角度，得到155 train / 40 val共195对；绝对yaw 12.5249°–17.9041°，平移0.1955–12.0006 m，全部门通过；
- 195对涉及145个唯一源帧，其中50个中心源BaseDepth可复用，新增95个端点源只需本地UniDepth推理，下载需求为零。

正式记录为`records/stage1_9_bidirectional_pair_pool_v1.json`（失败现场）和`records/stage1_9_bidirectional_pair_pool_v2.json`（canonical）。

## 7. Stage 1.9-1 实际结果（2026-08-18）

- 195对包含145个唯一源帧；复用50个中心源BaseDepth，为其余95个端点源生成只含RGB的输入证据，75 train / 20 val全部覆盖；
- 同一冻结UniDepth v1权重在GPU1完成95/95推理，全部深度有限且为正，峰值显存2,284,896,256字节，运行37.45秒；真值深度与真值内参均未作为模型输入；
- 对195个有向目标重算相对于各自源相机的`observed / occlusion_hidden / outside_source_fov / conflict / unresolved`，分类完整性195/195通过；
- 按原`depth↔position P99相对误差≤0.2%`、两类新显露均非空进行逐目标资格审计，最终160个目标合格：128 train / 32 val，覆盖35 / 9个独立场景；
- 合格池累计`occlusion_hidden=10,607,043`像素、`outside_source_fov=49,431,173`像素，超过冻结100/20目标与30/8场景最低门；下载需求仍为零。

正式记录为`records/stage1_9_endpoint_source_rgb_v1.json`、`records/stage1_9_endpoint_frozen_base_depth_v1.json`和`records/stage1_9_directed_visibility_pool_v1.json`；模型入口为`configs/stage1_9_qualified_directed_pairs_v1.json`。下一步先实现`SourceFeatureTargetViewNet`及CPU单元测试，再只选一个train端点做过拟合，不直接启动全量训练。

## 8. Stage 1.9-2 模型核心 CPU 契约（2026-08-18）

- 新增z-buffer winner对应的目标→源归一化采样网格；恒等相机下经`grid_sample(align_corners=True)`逐像素复现源图；
- `SourceFeatureTargetViewNet`先对完整源图做三级多尺度编码，再将融合特征投影到目标已覆盖位置；完整源图全局描述广播到全部目标像素，为视锥外洞区提供场景上下文；
- 输出语义保持目标RGB、Z-depth、两类support和两类confidence，不改变Stage 1.8 evaluator与短路门；
- 新旧核心共39/39 CPU测试通过。专门测试确认在warp全空时，只改变完整源图也会改变目标输出，证明洞区确实接入源图上下文。

正式记录为`records/stage1_9_core_source_feature_smoke_v11.json`。这只证明接口、形状和条件路径正确，不代表质量；下一步为新模型准备单端点输入并从头过拟合。
