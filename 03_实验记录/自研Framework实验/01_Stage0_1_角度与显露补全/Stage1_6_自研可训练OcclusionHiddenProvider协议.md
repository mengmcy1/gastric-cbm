# Stage 1.6 自研可训练 OcclusionHiddenProvider 协议

> 状态：**Stage 1.6 已收口为负结果。正例、unknown与目标视角 verified free-space 监督契约已实现，30/30核心测试通过；单样本过拟合可通过，但8场景多帧和40个独立训练场景均未在独立 val 上学到可靠 support 分离。HLP-TRAIN-03 冻结门未通过，因此未读取 HLP-GEO-01 test、未接 P01、未进入隐藏外观或 Gaussian Spawn。下一步转向显式 LDI/目标视角条件化的结构化候选。**

## 1. 本阶段只回答的问题

在不修改 Flash3D、不使用 P01 调参、也不使用 HLP-GEO-01 测试真值训练的前提下，能否从独立 Hypersim train/val 多视图数据学习一个源相机锚定的 `OcclusionHiddenProvider`，并在统一 evaluator 上超过冻结 Flash3D 参考值。

本阶段只处理中心视锥内部的遮挡后表面。中心视锥外内容继续标记为 unsupported，不由本模型外推。

## 2. 数据划分与泄漏硬门

- 官方 `metadata_images_split_scene_v1.csv` 是唯一 split 来源；其 SHA256 已写入 `configs/stage1_6_hlp_train_01_candidates_v1.json`；
- `ai_001_010`、`ai_005_001`、`ai_008_005` 属于官方 test，也是冻结 HLP-GEO-01 场景，禁止参与训练、早停、超参数选择、阈值选择和失败样例驱动修改；
- P01 及其他项目图片不得用于训练或本阶段调参；
- 源视角真值深度只能离线构造监督标签，不得放进模型输入。模型输入中的可见深度必须来自显式、冻结且可替换的 BaseDepthProvider，并记录来源；
- 第一批8 train、2 val 只用于数据与标签管线冒烟，不代表最终训练规模。姿态预审不合格的候选必须保留失败原因，不得偷偷改成 test 场景。

## 3. 30°训练三元组

每个场景最多选择一个源/左/右三元组，目标为总范围30°，即相对源相机 `-15°/0°/+15°`。端点角度误差预先限制为不超过3°；必须使用真实相机旋转计算，不用帧号或时间差冒充角度。

姿态预审阶段只下载相机 frame indices、orientation、position 和场景米制比例。通过后才定向下载所选三帧的 RGB、depth/position；不得下载完整场景 ZIP。

## 4. 源相机隐藏层监督语义

`adaptive3dgs.supervision.build_source_hidden_evidence` 将目标视角 first-hit RGB-D 反投影到世界，再投到源相机：

1. 目标点必须落在源视锥内且具有正向有限 Z-depth；
2. 候选点必须同时比源 first hit 深至少1 cm和1%；
3. 同一源像素有多个候选时保留最近的遮挡后表面，并记录观测数和深度跨度；
4. 有候选的像素是可靠正证据；没有候选的像素是 unknown，禁止编码成“隐藏层不存在”的负标签。

原因是稀疏目标视图只能证明“看到了后方表面”，不能证明未覆盖的源射线后方没有内容。支持概率的负监督必须来自目标视角 visibility/render loss，或未来的场景网格二次射线；不得用 `1-positive_mask` 伪造。

## 5. 第一版模型接口

第一版采用显式依赖而不是固定双层：

- 输入：源 RGB、冻结 BaseDepthProvider 的可见正向 Z-depth、内参；训练时不输入源真值深度；
- 输出头1：严格为正的相对隐藏深度增量 `r`，训练标签为 `z_hidden_truth/z_source_truth-1`，部署隐藏深度为 `z_base_pred × (1+r)`；这既保证隐藏层严格在可替换基础深度后方，也避免把单目跨域尺度误差写进遮挡比例标签；
- 输出头2：独立 support probability；
- 输出头3：独立 geometry uncertainty/confidence，不复用 Alpha/Opacity；
- 外观头可以读取多视图正证据 RGB，但在几何单样本过拟合通过前不作为主优化目标；
- 输出经统一 `OcclusionHiddenProvider → CLB → Validator`，视锥外仍为 unsupported。

## 6. 损失与执行顺序

第一轮损失分层启用：

1. 正证据像素上的隐藏 Z-depth/log-depth 回归和正深度增量约束；
2. 目标视角重投影/渲染后的 `occlusion_hidden` 支持分类，`outside_source_fov` 排除；
3. 基于真实几何误差的异方差损失与校准指标，confidence 不参与遮挡合成；
4. 几何单样本可以过拟合后，再加入隐藏外观和 HLP-APP 指标。

执行门：先合成几何测试，再做一个真实三元组标签冒烟，再做单样本过拟合；任一步失败都不得扩大训练规模。

## 7. 当前成功标准

- 数据门：train/val/test 场景集合严格不相交，姿态和文件哈希可复现；
- 标签门：所有正证据深度有限、为正且严格在源 first hit 后方；unknown 未被当负标签；
- 过拟合门：单样本正证据深度误差显著下降，support/confidence 不坍缩为 Alpha；
- 候选质量门：只在独立 val 冻结后运行 HLP-GEO-01 test，一次性与 Flash3D 原始米制 `AbsRel 23.33% / δ<1.10 24.80%` 比较；未通过前不接 P01。

## 8. 数据与标签冒烟结果（2026-08-14）

- 官方划分共365个 train、46个 val、46个 test 场景；冻结的3个 HLP-GEO-01 场景全部属于 test；
- 8个 train 与2个 val 候选均找到真实 `-15°/0°/+15°` 三元组，每场景有68–93个合格源帧；所选端点角度误差合计最大约0.193°，两端平移均不小于0.1 m；
- 姿态预审只取40个文件、166,171字节；正式三元组定向取得130/130个 RGB/depth/position/pose 文件，共136,035,863字节，未下载完整场景 ZIP；
- 第一次标签运行在写出数组前发现 `ai_031_010` 使用非对角 `M_cam_from_uv`。没有替换场景；现已用 RQ 将一般投影分解为上三角 K 与相机旋转修正，并把反投影/投影推广到一般 K。失败证据保存在 `records/stage1_6_hlp_train_01_hidden_evidence_v1_failed.json`；
- canonical v2 在10个三元组上生成912,897个正证据像素，最小场景占比1.6127%，中位占比12.1352%，显著高于预设0.1%门；50项深度、覆盖、unknown语义门全部通过；
- canonical 记录为 `records/stage1_6_hlp_train_01_hidden_evidence_v2.json`；核心一般投影与正证据生成器20项测试记录为 `records/stage1_6_core_supervision_smoke_v6.json`。

## 9. 冻结 BaseDepthProvider 输入（2026-08-14）

- BaseDepthProvider 固定为 `unidepth-v1-vitl14-rgb-only-frozen-v1`。生产脚本只从证据 NPZ 读取 `source_rgb_uint8`，不访问 `source_depth_z_label_only_float32`，也不向 UniDepth 传入真值内参；源真值深度仍只留在标签侧；
- 首次离线初始化发现本机 Hugging Face 缓存只有权重、没有 `config.json`，在模型构造前退出且未读取数据数组。空失败目录和 `records/stage1_6_hlp_train_01_frozen_base_depth_v1_failed.json` 均保留；
- canonical v2 改为直接读取 UniDepth 仓库冻结的 `config_v1_vitl14.json` 和相同的本地权重（SHA256 `da8f7910dbcf1d14d2d9f9699d791a740fad8c41ba2ab0ae5729dcff20efbb60`），不是更换模型或权重；
- 物理 GPU 1 上完成10/10样本；全部预测为原生768×1024、有限且严格为正，峰值分配显存2,284,020,736字节，推理段总耗时6.14秒；
- 10个深度 NPZ 与10张诊断预览的20/20哈希复核通过。正式记录为 `records/stage1_6_hlp_train_01_frozen_base_depth_v2.json`，质量边界只是冻结无泄漏输入，不代表隐藏层预测质量通过。

**下一步**：实现输入 `RGB + log(base depth)` 的正深度增量、support probability、geometry uncertainty/confidence 三头网络及 unknown 隔离损失；先只对 `ai_001_001` 做单样本过拟合门。

## 10. 三头网络与损失契约（2026-08-14）

- 冻结输入统计发现 UniDepth 的 Hypersim 跨域尺度误差不可忽略：若以预测可见深度直接构造绝对 `hidden-base` 标签，部分场景大多数正证据会成为负值，与严格后方约束矛盾。因此冻结为上节的正相对增量定义；
- 新增 `adaptive3dgs.models.OcclusionHiddenUNet`：RGB与样本内稳健归一化的 log base depth 进入小型 U-Net，正相对增量、support logits、geometry uncertainty 使用三个独立头；confidence 明确定义为 uncertainty 的单调变换，不复用 Alpha；
- 几何与 uncertainty 损失只索引正证据像素；unknown 中即使写入 NaN/任意占位值也不改变这两项损失；
- support 使用 non-negative PU risk。unknown 只作为未标注混合分布参与风险估计，不逐像素编码为负类；
- 24/24 CPU 单元测试通过，覆盖严格后方、三头范围/独立性、坏基础深度拒绝、unknown 隔离和非正真值比例拒绝。记录为 `records/stage1_6_core_trainable_model_smoke_v7.json`。

**下一步**：在物理 GPU 空闲检查后，对 `ai_001_001` 单样本运行相对隐藏几何过拟合；只有相对深度误差显著下降且 support/uncertainty 不发生非法坍缩，才扩大到8 train/2 val。

## 11. 单样本过拟合结果（2026-08-14）

- v1 在首次前向发现 CUDA indexed median 不支持确定性算法，0步、0产物退出；归一化改为确定性的均值/平均绝对偏差，失败记录为 `records/stage1_6_hlp_train_01_single_overfit_v1_failed.json`；
- v2 使用 `log(1+r)` 回归，500步后比例 AbsRel 从3.440降到0.336但未过0.10门；v3 改为与乘性误差一致的 `log(r)` 后降到0.115，support正例均值0.595，仍未降低门槛，两个失败记录与产物均保留；
- v4 只在同一 train 样本上加入余弦学习率和更高 PU support 权重，未触碰 val/test。750步后比例 AbsRel为0.08091、相对初始改善97.65%，比例 MAE为0.02133；
- 正例/unknown support 均值分别为0.84581/0.02012；正例 confidence 均值/标准差为0.92120/0.10643，未非法复用 Alpha 或成为常数；所有输出有限、增量严格为正、confidence 在开区间(0,1)；
- GPU 1 峰值分配显存405,159,936字节、训练耗时36.01秒；checkpoint成功重载，7个产物哈希全部复核。正式记录为 `records/stage1_6_hlp_train_01_single_overfit_v4.json`；核心25项测试记录为 `records/stage1_6_core_trainable_model_smoke_v8.json`。

**阶段结论**：模型与监督至少具有单样本可学习性，允许进入8 train/2 val小规模候选训练。该通过不等于泛化、HLP-GEO test、外观或 P01 质量通过。

**下一步**：固定当前相对增量/PU/uncertainty 定义，在8个 train 场景联合训练并只用2个 val 场景选择 checkpoint；冻结候选后才允许一次性运行 HLP-GEO-01 test。

## 12. 8 train / 2 val 首个联合候选（未通过，2026-08-14）

- 在查看 val 结果前冻结150 epoch、余弦学习率、checkpoint选择分数和6项通过门，配置为 `configs/stage1_6_hlp_train_01_small_training_v1.json`；HLP-GEO test 与 P01 均未读取；
- 物理 GPU 1 完成运行，验证 selection score 从2.4511降到最佳1.5410、相对改善37.13%，验证比例 AbsRel从2.2998降到1.3452，说明不是完全没有跨场景信号；
- 但最佳 epoch 5 的验证 support 正例/unknown均值只差0.0076，未达到预冻结0.05门；对应训练比例 AbsRel改善47.68%，也未达到50%门；
- 后续 epoch 的训练 support 分离逐步升至约0.247，验证分离仍约0.004，表现为典型的小样本场景记忆而非可靠泛化；
- 正式记录 `records/stage1_6_hlp_train_01_small_candidate_v1.json` 保持 `failed`。不降低门槛、不运行 HLP-GEO test，也不把这个 checkpoint 冻结为候选。

**阶段结论**：每场景仅1个源视图的8张训练图不足以学习跨场景 support；当前10三元组的定位本来就是管线冒烟，而非最终训练规模。下一步从相同官方 train/val 场景的已审计姿态中扩展多源三元组，仍保持场景级 train/val/test 隔离。

## 13. HLP-TRAIN-02 多源扩展就绪（2026-08-14）

- 不新增或更换场景，在原8个train/2个val场景内将每场景源视图数从1扩到10；场景级train/val隔离保持不变，3个HLP-GEO test场景仍为零交集；
- 改进姿态选择器，使每场景按角度误差排序选择10个唯一源帧。共100/100个真实 `-15°/0°/+15°` 三元组通过端点误差和平移门，形成80 train / 20 val；
- 改进精确文件获取器，按场景合并并去重三元组成员，避免重复打开相同ZIP和覆盖相同文件；最终只需769个唯一成员；
- 769/769文件全部取得并可读，共1,093,518,150字节；10个完整场景ZIP总大小约46.4 GB，均未下载；
- 冻结配置/记录为 `configs/stage1_6_hlp_train_02_candidates_v1.json`、`configs/stage1_6_hlp_train_02_triplets_v1.json`、`records/stage1_6_hlp_train_02_pose_selection_v1.json` 与 `records/stage1_6_hlp_train_02_fetch_v1.json`。

**下一步**：按同一一般投影和positive-only语义生成100组正证据；通过后再生成100份冻结RGB-only UniDepth输入，不直接复用旧10份作为未登记的隐式缓存。

## 14. HLP-TRAIN-02 标签与冻结输入就绪（2026-08-14）

- 100组标签v1/v2均被同一个量化边界像素拦住：8,721,745个正像素中，1个候选在float64投影中严格超过1%，但落盘float32后与1%阈值相等；失败产物和记录均保留；
- 没有放宽1%门。生成器现按最终float32深度与float32阈值再次执行严格比较；落在边界的像素恢复为unknown，并清零观测数/颜色；对应量化边界测试加入核心；
- canonical v3 生成8,721,744个正像素，最小样本占比0.11317%、中位7.27832%，500/500门通过；100个NPZ哈希复核通过；
- 随后在物理GPU 1生成100/100份RGB-only UniDepth V1冻结输入，全部768×1024、有限且严格为正；峰值分配显存2,284,896,256字节，推理段40.54秒；100个NPZ与100张预览共200/200哈希通过；
- 正式记录为 `records/stage1_6_hlp_train_02_hidden_evidence_v3.json`、`records/stage1_6_hlp_train_02_frozen_base_depth_v1.json` 与26/26核心测试 `records/stage1_6_core_quantized_supervision_smoke_v9.json`。

**下一步**：冻结HLP-TRAIN-02训练配置，联合训练80个train样本并以20个val样本选择checkpoint；HLP-GEO test仍不读取。

## 15. Support 可辨识性与 train-only free-space 过拟合（2026-08-14）

- HLP-TRAIN-02 candidate v1 沿用positive-only nnPU，val support正例/unknown均收敛到约0.096，证明稀疏观测正例无法给source-only存在性提供可辨识类先验；状态failed，未运行test；
- 只读统计显示val正例与基础深度梯度AUC约0.494、RGB梯度约0.544，因此没有改成缺乏证据的边缘启发式；
- candidate v2/v3分别在确定性cuBLAS配置和“单样本无verified negative”loss契约处提前失败，0步/首epoch失败记录保留；
- candidate v4 引入目标视角free-space负证据：提案点若落在目标真值first-hit前的已观测自由空间则为可靠负例，后方仍unknown。它生成约11.7万val负例，但浅层/无内参模型train与val均未分离；
- candidate v5 补齐协议要求的相机射线输入、/16上下文与delta-conditioned独立support/uncertainty头，30/30核心测试通过；联合batch=1仍发生场景梯度冲突，未通过；
- 为区分标签无解与联合优化问题，新增 `scripts/stage1_6_single_free_space_overfit.py`，只用train `ai_001_001/0099`。1000步后比例AbsRel `2.9417→0.0913`，support正例/verified-negative `0.4650/0.4974→0.8652/0.1349`，分离0.7304，4/4门通过；
- 正式单样本记录为 `records/stage1_6_hlp_train_02_single_free_space_overfit_v1.json`。结论是free-space监督与架构可学习，联合优化需避免batch=1顺序遗忘。

**下一步**：使用4样本梯度平均、train-only已通过的1e-3学习率与support权重1.0训练联合候选；val门不变，test仍不读取。

## 16. HLP-TRAIN-02 联合优化结论（仍未通过，2026-08-14）

- candidate v6 使用4样本梯度平均、1e-3学习率与support权重1.0；这些值来自train-only free-space过拟合，没有根据val结果降低门；
- train support正例与verified-negative分离从约0缓慢升到约0.023，说明梯度平均缓解了顺序遗忘；但20个val样本始终约0，最佳val正例/负例为0.5235/0.5333；
- 最佳val比例AbsRel为1.3881，虽优于随机初始化2.2446，但最佳checkpoint在epoch 6，train改善未过预设门；正式状态failed；
- HLP-TRAIN-02把每个train场景从1帧扩到10帧，但场景数仍只有8。多源帧增加没有转化为跨场景support泛化，继续在同8场景调优化器的边际证据不足。

**阶段结论**：下一数据扩展必须增加独立train/val场景数，而不是继续增加同场景帧。HLP-TRAIN-03将从官方365/46场景池做场景类型分层预审，优先“一场景一三元组”；HLP-GEO test继续隔离。

## 17. HLP-TRAIN-03 多场景姿态集合冻结（2026-08-14）

- 从官方split和trajectory元数据按场景类型round-robin冻结60 train / 20 val候选；覆盖22种train场景类型，所有场景唯一、公开帧数≥80，held-out test交集为0；
- 仅定向获取80场景×4个姿态/尺度文件，共320/320项、1,329,409字节，全部可读且未下载完整ZIP；
- 真实姿态预审中79/80场景具有合格±15°三元组；1个train失败候选保留在 `records/stage1_6_hlp_train_03_pose_selection_preflight_v1.json`；
- 在读取RGB-D前，从59 train / 20 val合格池再次按场景类型round-robin冻结40 train / 10 val，每场景仅1组三元组；train覆盖22类、val覆盖10类，四项姿态/唯一性/隔离门全过；
- 正式配置为 `configs/stage1_6_hlp_train_03_triplets_v1.json`，冻结记录为 `records/stage1_6_hlp_train_03_triplet_finalize_v1.json`。

**下一步**：只获取这50个已冻结三元组的精确RGB-D/position成员；随后按相同标签和RGB-only BaseDepth协议生成训练产物。

## 18. HLP-TRAIN-03 数据、标签与BaseDepth就绪（2026-08-14）

- 50个独立场景三元组对应650/650个精确文件、701,705,049字节，全部HDF5可读；未下载完整场景ZIP；
- positive-only标签生成3,388,680个正像素，最小样本占比0.40487%、中位6.78482%，250/250门通过；
- 物理GPU 1在保留未知约2.0 GiB进程的前提下仍有充足余量，生成50/50份RGB-only UniDepth冻结输入；全部有限、严格为正，峰值分配显存2,284,896,256字节、推理段28.47秒；
- 正式记录为 `records/stage1_6_hlp_train_03_fetch_v1.json`、`records/stage1_6_hlp_train_03_hidden_evidence_v1.json` 与 `records/stage1_6_hlp_train_03_frozen_base_depth_v1.json`。

**下一步**：按 `configs/stage1_6_hlp_train_03_training_v1.json` 运行40独立train场景/10独立val场景候选；保持1000次optimizer更新与此前v6一致，使主要变量是场景多样性而非更新预算。

## 19. HLP-TRAIN-03 冻结候选与 Stage 1.6 收口（2026-08-14）

- HLP-TRAIN-03 按冻结配置完成40个独立 train 场景、10个独立 val 场景和1000次 optimizer update；最佳 checkpoint 只由 val selection score 选择，未读取 HLP-GEO test 或 P01；
- 验证 ratio AbsRel 从 `3.8958` 降至 `2.3451`，selection score 相对改善 `37.40%`；训练 ratio AbsRel 相对改善 `67.22%`，说明网络并非完全没有学习；
- 但最佳验证 support 正例/verified-negative 均值为 `0.55777/0.53573`，分离仅 `0.02204`，低于冻结的 `0.05` 门；训练集分离也只有 `0.02456`。扩大场景数没有解决 source-only support 的跨场景可辨识性；
- 正式记录 `records/stage1_6_hlp_train_03_candidate_v1.json` 保持 `failed`。不降低门槛、不运行 HLP-GEO-01 test、不接 P01，也不把 checkpoint 包装成可用 Provider；
- 当前实现合同由30/30测试冻结，记录为 `records/stage1_6_core_trainable_model_smoke_v10.json`。测试通过只证明射线输入、深层上下文、delta-conditioned独立头、verified free-space 与 unknown 隔离实现正确，不改变质量失败结论。

**Stage 1.6 结论**：小型从零训练的 source-only 三头 Provider 可以记住单个样本，却不能在当前数据预算下可靠泛化“哪条源射线存在遮挡后表面”。继续只调优化器、场景数或同类二维先验的收益不足。下一候选必须改变信息结构：优先审计完整 LDI 连通关系与逐层补全，或显式目标视角条件化；`OutsideFOVProvider` 仍保持独立未解决。
