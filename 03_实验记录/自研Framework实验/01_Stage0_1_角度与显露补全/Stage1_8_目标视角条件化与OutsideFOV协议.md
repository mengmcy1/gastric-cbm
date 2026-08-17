# Stage 1.8 目标视角条件化与 OutsideFOV 协议

> 状态：**Stage 1.8-0/1已通过；Stage 1.8-2首个64-train/14-val联合候选正式不通过。逐目标oracle尺度仍不能使两类深度过门，确认并非单一全局尺度问题；GenWarp multi2隔离环境、CUDA扩展和8.43GB权重已就绪并完成哈希校验，按用户要求暂停在GPU冒烟之前，明日只做单个val目标的16GB可运行性冒烟。**

## 1. 研究问题

Stage 1.6 证明 source-only 小型三头网络无法在独立场景可靠判断隐藏表面；Stage 1.7 官方完整 LDI 在 P01 总范围30°下仍出现左右端严重空洞和虚假结构。因此本阶段改变信息结构：把指定目标相机显式送入 Provider，让模型直接回答该视角应出现的内容。

第一轮只回答：在相同源图、基础深度和多视图真值下，目标相机条件能否使指定 ±15°端点的新显露 RGB-D 相对 source-only 基线明显改善。候选通过独立真值门前，不接 P01、不转 Gaussian Spawn、不扩展 P02/P05。

## 2. 能力边界与三类区域

目标画面按真值几何分为互斥区域：

1. `observed_from_source`：源图已有表面可重投影到目标视角；
2. `occlusion_hidden`：位于源视锥内、但在源图中被前景遮挡的后方表面；
3. `outside_source_fov`：不在源相机视锥内、却进入目标画面的新区域。

后两类必须分别报告覆盖率、RGB误差和深度误差。`OcclusionHiddenProvider` 不得用边缘外推冒充 `OutsideFOVProvider`；未分类的几何冲突和 unresolved 像素不进入质量门。

## 3. 冻结输入与泄漏边界

- 数据来源：HLP-TRAIN-03 的40个 train、10个 val独立场景，held-out HLP-GEO test场景继续封存；
- 角度：总范围30°，左右端点约为 -15°/+15°；
- 部署输入：源 RGB、冻结 RGB-only BaseDepth、源相机、指定目标相机、目标射线/相对位姿；
- 标签侧：目标 RGB-D及由世界坐标生成的三类区域掩码；
- 禁止把源深度真值、目标 RGB、目标深度或真值区域掩码送入模型输入；
- train/val按场景隔离，val只用于选择候选，冻结后才允许一次性读取HLP-GEO/HLP-APP test。

## 4. Stage 1.8-0 evaluator oracle 冒烟

先从冻结 HLP-TRAIN-03 中固定首个 train和首个val三元组，共2场景、4个目标端点。使用目标真值本身作为 oracle 预测，验证：

- 4/4目标数据、相机和掩码可读取；
- 三类掩码两两互斥；
- 每个目标同时存在 `occlusion_hidden` 与 `outside_source_fov` 真值；
- 两类新显露区域覆盖率均为1；
- 深度 AbsRel 不高于 `1e-7`；
- RGB MAE 为0。

本冒烟只校准“尺子”，不构成模型质量结果。配置固定为 `configs/stage1_8_target_view_oracle_smoke_v1.json`。

## 5. 后续模型门

oracle通过后按三级门推进：

1. 单个 train目标过拟合，确认相机条件、RGB-D损失和两分支输出可学习；
2. HLP-TRAIN-03 train/val联合候选，与冻结 source-only失败基线比较；
3. 冻结候选后一次性运行HLP-GEO/HLP-APP独立测试。

模型评价必须至少分别报告两类新显露区的覆盖率、RGB MAE/PSNR、深度 AbsRel、错误前景侵入率与置信度。某一分支未通过时，不得用另一分支的平均改善掩盖。

## 6. 进入 P01 和 Gaussian Spawn 的条件

只有 `occlusion_hidden` 与 `outside_source_fov` 均通过独立多视图真值门，才允许接入 P01 `true_arc` 61帧总范围30°，并与 Stage 1.7 LDI失败基线做完整视频 A/B。P01通过后，才把经过目标视角验证的 RGB-D 转换为世界坐标 CLB，并进入 SurfaceFusion 与 Gaussian Spawn。

## 7. Stage 1.8-0 实际结果（2026-08-17）

- v1固定首个train/val场景，但val左端没有`outside_source_fov`真值，因此按原门失败；
- v2只替换val为下一个冻结场景，左端仍无视锥外真值，同样保留为失败；没有放宽“4个端点均需两类新显露真值”的门；
- 随后对10个冻结val场景做只读资格审计，v3固定第一个左右端均含视锥外真值的val场景`ai_024_012`，train场景仍为`ai_024_010`；
- v3可见性真值4/4端点通过，累计`occlusion_hidden=191,988`像素、`outside_source_fov=728,090`像素，分类完整且两类区域均非空；
- canonical evaluator记录为`records/stage1_8_target_view_oracle_smoke_v5.json`：2场景、4端点、31/31门通过，两类区域覆盖率均为1、深度AbsRel为0、RGB MAE为0；
- Adaptive3DGS核心测试现为32/32通过。oracle只证明数据、掩码与评价器对齐，不证明任何模型能够预测新内容。

下一轮唯一改变变量是加入目标相机条件。先定义 source RGB/BaseDepth、源/目标相机与目标射线的输入契约，再在`ai_024_010`的一个端点做单样本过拟合；不同时改变基础深度、数据划分或高斯表示。

## 8. Stage 1.8-1 单端点过拟合结果（2026-08-17）

- 新增显式`TargetViewRequest`，模型输入为目标空间中的源RGB-D重投影、覆盖掩码、目标射线在源相机中的方向和目标相机原点；目标RGB-D和真值掩码只进入loss/evaluator；
- `TargetViewUNet`共享目标空间backbone，但为`occlusion_hidden`和`outside_source_fov`保留两个独立support头，同时输出目标RGB、Z-depth及独立几何/外观confidence；
- v1固定1200步，绝对RGB、深度、support、相机敏感性和有限值门均通过，但两区RGB相对改善为89.26%/88.17%，未过冻结90%门，正式保留为failed；
- v2唯一改动为1200→2000步，同一随机种子从头训练，不续接v1；两类新显露区RGB MAE分别从29.39/49.09降到2.15/3.88，改善约92.69%/92.09%；
- 两区深度AbsRel为0.00328/0.00238，正support为0.9888/0.9953，负区support为0.0183/0.0108；相反目标相机能改变预测，12/12门全部通过；
- 物理GPU 1运行，峰值分配显存约0.51GB。正式记录为`records/stage1_8_target_view_single_overfit_v2.json`。

**结论边界**：目标相机条件、几何重投影、双分支标签和完整RGB-D loss具备可学习性，但当前只是单个train端点记忆。下一阶段保持结构不变，扩展到40 train/10 val场景的左右端点；val只能用于checkpoint选择，仍不读取HLP-GEO test。

## 9. Stage 1.8-2 联合候选 v1 结果（2026-08-17）

- 100个端点的初始可见性记录因13个端点未过冻结`depth↔position P99相对误差≤0.2%`而状态fail；没有放宽门或删除现场；
- 资格脚本逐端点应用原门，并要求两类新显露均非空，最终得到78个严格合格目标：64 train、14 val；100份可见性数组哈希全部通过；
- v1保持单样本已通过的TargetViewUNet、384×512和loss，固定batch4、2000次更新；只用val selection score挑checkpoint，最佳为update 800；
- val遮挡后/视锥外support分离为0.147/0.634，均过0.10门，证明目标条件结构能跨场景判断两类区域；
- 但val RGB MAE为43.46/59.43，未过35；深度AbsRel为0.593/1.074，未过0.30。train RGB仍为48.75/45.49，深度为0.284/0.628，说明不仅是val过拟合，也存在表示/先验不足；
- 11项门中7项通过、4项失败，正式记录为`records/stage1_8_target_view_candidate_v1.json`；按协议未读取HLP-GEO/HLP-APP test或P01。

**阶段决定**：停止只增加同一模型的优化步数。下一步先量化“每目标全局深度尺度对齐”能解释多少误差，并审计可复用的冻结预训练图像/新视角先验；support头可保留为当前结构资产，RGB-D生成骨干需要改变。

## 10. 深度尺度诊断与预训练骨干审计（2026-08-17）

- 对联合候选最佳checkpoint在14个val目标上做不可部署的逐目标正标量oracle；该标量只用于判断误差来源，不写回候选、不改变其failed状态；
- 遮挡后区域深度AbsRel从0.593降到0.441，视锥外区域从1.074降到0.365，仍均高于冻结0.30门；14个oracle尺度范围为0.219至7.942，中位0.807；
- 因此误差不只是一个可校准的全局深度尺度，还包含局部几何和内容生成错误。正式记录为`records/stage1_8_depth_scale_diagnostic_v1.json`；
- 本机可复用权重只有Flash3D、SHARP与ResNet50，未发现MAT、LaMa或3D Photo补全权重；Flash3D已作为几何失败基线，InfiniSplat也已暂停，均不重复冒充新候选；
- 官方候选中，ViewCrafter 512版官方显存为13.8GB但一次生成25帧；Stable Virtual Camera为1.3B且需Hugging Face授权；MetaView依赖Qwen-Image-Edit与两个Depth Anything 3巨型模型。首选GenWarp `multi2`，因为它直接以源RGB、源深度及显式源/目标相机矩阵生成一个指定目标视角，并同时返回warp、mask和correspondence；
- GenWarp代码为MIT，官方权重为CC-BY-NC 4.0并带上游使用限制，只作为研究候选。最小权重集合约8.43GB，官方未给16GB显存数字，所以必须先做单目标可运行性冒烟，不可直接跑14目标正式val。

**下一步硬边界**：隔离安装GenWarp，只下载`multi2 + image_encoder + VAE`。先在一个已合格val端点检查峰值显存、有限输出、相机方向和输出尺寸；冒烟通过后才冻结14目标适配器与评价配置。GenWarp只负责目标RGB和几何引导，部署深度由现有RGB-only冻结深度模型从生成目标RGB推断，目标真值仍仅进入evaluator。

## 11. GenWarp 环境与权重就绪，暂停在 GPU 冒烟前（2026-08-17）

- 官方GenWarp冻结commit为`22c03b6ae349013b7099801f4815cf1d0fd09c35`，本机建立独立`genwarp` Conda环境；现有SHARP、Flash3D等环境未被修改；
- RTX 5080要求新PyTorch/CUDA架构，最终环境固定为Python 3.10、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、diffusers 0.29.2、transformers 4.41.2、accelerate 0.31.0及环境内CUDA nvcc 12.8.93；
- 官方`splatting` commit `1427d7c...`仍使用旧`frame.type()`分派，在PyTorch 2.7首次编译失败；本地只将两处改为`frame.scalar_type()`，随后面向`sm_120`编译通过。GenWarp与全部关键依赖静态导入通过；
- 仅下载`multi2 + image_encoder + sd-vae-ft-mse`共8个文件、8,429,409,786字节，逐文件字节数与SHA256全部完成；没有下载重复`multi1`；
- 源码、兼容补丁、Conda环境和权重均为本机可重建资产，不上传GitHub；正式版本、补丁语义、字节数和哈希见`records/stage1_8_genwarp_environment_v1.json`；
- 本阶段没有运行模型加载或GPU推理，没有读取held-out test、HLP-APP或P01。用户要求下载完成后暂停，服务器不遗留持续GPU进程。

**明日唯一恢复点**：先运行`nvidia-smi`；目标GPU资源充足时优先GPU1。加载`multi2`后只跑一个已合格val端点，记录峰值显存、有限输出、分辨率和左右相机方向；该冒烟通过前不启动14目标批量评价。
