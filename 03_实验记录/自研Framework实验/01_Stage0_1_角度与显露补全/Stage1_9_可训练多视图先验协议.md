# Stage 1.9 可训练多视图先验协议

> 状态：**Stage 1.9-0/1/2已通过；Stage 1.9-3 v1/v2小候选均未通过。冻结ResNet v2有效改善深度与总体score，但遮挡RGB 38.66、视野外深度0.417仍失败且视觉涂抹。下一步设计洞区空间化源特征代理投影；不扩40/10、不加训练步数。**

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

## 9. Stage 1.9-2 单端点过拟合（2026-08-19）

- 使用`ai_024_010/cam_00`的中心16帧→左端22帧（真实yaw -14.9683°）作为唯一训练目标；同源右端90帧只用于目标相机敏感性控制；
- 新定向加载器只从冻结证据包白名单读取`source_rgb_uint8`，深度输入只来自冻结RGB-only UniDepth；目标RGB-D和三分区真值仅用于loss/evaluator，未读取held-out test；
- 384×512输入中，遮挡后/视锥外分别有5,595/26,039个监督像素，point warp覆盖57.73%，所有输入有限且哈希链通过；
- GPU1从头训练2,000步耗时179.36秒，峰值CUDA allocated 859,270,144字节；遮挡后RGB MAE由27.39降至2.19（约92.0%），视锥外由47.69降至3.99（约91.6%），深度AbsRel为0.00356/0.00243；
- 双support正样本均值0.988/0.996、负样本均值0.023/0.011；同源不同目标相机输出平均差0.2184，有限值与全部12项门均通过；
- 完整核心回归39/39通过。但视觉预览仍比目标明显更软，画框纹理、天花板格栅和窗格细节不足；这一级只证明新结构可优化且确实依赖目标相机，不证明跨场景泛化或最终画质。

正式配置与记录为`configs/stage1_9_source_feature_single_overfit_v1.json`和`records/stage1_9_source_feature_single_overfit_v1.json`。下一步按冻结阶梯进入Stage 1.9-3，仅选8个train场景与2个val场景建立小候选；必须同时报告point-warp相对改善、绝对RGB-D、双support分离和视觉清晰度，失败则不扩全量。

## 10. Stage 1.9-3 从头卷积小候选（2026-08-19）

- 冻结8 train / 2 val独立场景，纳入全部严格合格定向对共31/6个目标；缩放后train遮挡后/视锥外为372,895/1,845,581像素，val为41,706/614,772像素；
- GPU1训练2,000 updates耗时754.21秒，峰值CUDA allocated 1,440,031,744字节；最佳checkpoint出现在400 updates，之后验证selection score总体恶化，排除“只需继续加步数”；
- val遮挡后RGB MAE由warp 90.99降至40.87、视锥外由143.33降至29.50，两类相对改善与support分离均通过；但遮挡后RGB未过≤35，深度AbsRel 0.392/0.509均未过≤0.30，正式状态`failed`；
- train最佳深度AbsRel也只有0.238/0.495，说明不只是val过拟合；视野外深度在400 updates后明显恶化；
- 视觉预览显示浴室结构和画廊内容被压成低频色块、波纹与涂抹，明显不可接受。当前从头三级卷积在无warp区主要依赖广播全局向量，不能稳定保留空间细节。

正式小池、配置与结果为`records/stage1_9_small_candidate_pool_v1.json`、`configs/stage1_9_source_feature_small_candidate_v1.json`和`records/stage1_9_source_feature_small_candidate_v1.json`。按短路门不扩40/10。协议原定的冻结ImageNet ResNet-50尚未实际接入；本地官方V2权重大小102,540,417字节、SHA256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`，torchvision严格加载通过。下一轮只替换源图encoder并保持小池、优化预算、loss和质量门不变。

## 11. Stage 1.9-3 冻结ResNet小候选v2（2026-08-19）

- v2唯一结构变量为将随机源encoder替换为冻结ImageNet ResNet-50 layer1–4；31/6小池、目标decoder、2,000 updates、loss、384×512、checkpoint选择和全部门不变；
- CPU契约确认backbone始终`eval`、参数无梯度、投影层有梯度，完整回归39/39通过；GPU1训练576.46秒，峰值CUDA allocated 1,378,062,848字节，最佳checkpoint为800 updates；
- 相对v1，val selection score由1.197降至1.008；遮挡RGB 40.87→38.66，遮挡深度0.392→0.269并通过；视野外RGB为32.45并通过，但深度0.417仍未过0.30；遮挡RGB仍未过35，正式状态`failed`；
- train深度已达0.136/0.219而val为0.269/0.417，说明预训练语义和尺度学习有效，但跨场景洞区泛化仍不足；
- 视觉仍有大面积涂抹和波纹。进一步覆盖审计显示val遮挡区63.5%虽有warp，但其覆盖部分RGB MAE仍为45.41，直接增加skip并不能使遮挡门达到35；视野外仅12.9%有warp，更不能靠残差路径解决。

正式配置与记录为`configs/stage1_9_source_feature_small_candidate_v2.json`和`records/stage1_9_source_feature_small_candidate_v2.json`。下一步不解冻ResNet、不增加步数；先冻结一个空间化洞区特征设计：按目标射线与源深度尺度构造源平面代理采样坐标，使无z-winner像素获得位置相关的源特征，并与真实z-winner特征分支显式区分。该设计需先过恒等相机、有限值、越界坐标和信息泄漏CPU门，再复用同一小池做v3。
