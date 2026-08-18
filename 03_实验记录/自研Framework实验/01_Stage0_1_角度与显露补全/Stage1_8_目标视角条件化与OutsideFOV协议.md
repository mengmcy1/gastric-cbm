# Stage 1.8 目标视角条件化与 OutsideFOV 协议

> 状态：**Stage 1.8小型联合候选、GenWarp multi2与ViewCrafter 512均已在独立val正式不通过；冻结公开生成先验路线已收口，生成深度、test、P01与Gaussian Spawn均按协议短路。下一步冻结自训多视图先验的数据与模型契约。**

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

## 12. GenWarp 单目标适配与显存冒烟通过（2026-08-18）

- 固定首个合格val端点`ai_024_013 / source 35 → right 22 / 14.9816°`；输入为源RGB、冻结RGB-only UniDepth和源/目标相机，目标RGB只用于冒烟后的视觉对照，held-out test与P01均未读取；
- 768×1024输入采用中心768×768裁剪后缩放到官方512×512。裁剪和缩放后的非中心主点被写入一般OpenGL投影矩阵，不假设主点位于图像中心；
- OpenCV right/down/forward到OpenGL right/up/back固定为左乘`diag(1,-1,-1,1)`。4096个点与canonical OpenCV投影逐点比较，p99误差`2.62e-13 px`、最大`7.19e-13 px`，相机方向门通过；
- 初次预检发现官方half模式把投影和网络一同降为float16。该端点有26个远端离屏坐标超过float16有限上限，导致correspondence出现Inf。没有放宽有限性门；本地只把`forward_warper`的几何与soft splat提升为float32，网络权重和扩散仍保持float16，严格复跑后全部输出有限；
- 正式20步`multi2`推理含模型加载35.80秒，峰值allocated/reserved为4.61/5.04 GiB，远低于15.5 GiB门；`synthesized`为1×3×512×512且范围0.0515–1.0；
- 视觉预检只作风险提示：生成图补全了warp黑洞，但画框数量/位置、长凳和门洞相对val真值发生明显语义改写。因此本轮只判“适配和16GB可运行性通过”，不判质量通过。

正式记录为`records/stage1_8_genwarp_val_smoke_v1.json`；两次预检及float16溢出根因保存在`records/stage1_8_genwarp_val_smoke_preflight_v1_failed.json`。下一步使用同一适配器、模型参数和冻结val资格池运行14目标RGB评价；两类新显露区仍分别应用`相对warp改善≥15%且MAE≤35`，通过后才补推生成RGB的部署深度。

## 13. GenWarp 14目标val RGB质量门未通过（2026-08-18）

- 冻结8个独立val场景中的14个合格端点，保持`multi2`、512×512中心方形适配、20步、guidance 3.5及每目标固定seed；目标RGB和两类区域掩码仅进入evaluator；
- 相对黑洞式warp，遮挡后RGB MAE从96.22降到44.22，改善54.04%；视锥外从107.56降到60.51，改善43.74%。两类均通过`改善≥15%`；
- 绝对质量两类均未过`MAE≤35`：遮挡后44.22、视锥外60.51。逐目标也不是单个异常值拖累，仅5/14个端点过遮挡后绝对门、7/14过视锥外绝对门；
- 最差端点包括`ai_031_004-left`（77.51/115.15）与`ai_024_012-left`（94.78/77.17）；生成结果通常视觉连续但会重构错误的柱体、家具、门窗和纹理，符合首样本观察到的语义改写；
- 14目标总运行44.39秒，峰值allocated/reserved 4.63/5.04 GiB；失败不是显存或数值问题，而是生成内容与真实目标视角不一致。

正式记录为`records/stage1_8_genwarp_val_rgb_v1.json`，状态`failed_rgb_gate`。按预先冻结的短路规则，不运行生成RGB的UniDepth、不读取HLP-GEO/HLP-APP test或P01、不进入Gaussian Spawn。GenWarp保留为“视觉连续补洞”对照，不再作为当前可冻结的正确内容Provider。

## 14. ViewCrafter 后备候选协议审计（2026-08-18）

- 官方`ViewCrafter_25_512`为320×512、25帧视频扩散模型；官方A100/50步/逐帧VAE数据为13.8GB和50秒。checkpoint为10,437,386,907字节；
- 官方`single_view_eval`会读取目标视频帧并用DUSt3R估计整段相机，违反本实验“目标RGB只进evaluator”的边界，明确禁用；官方`single_view_target`只接受围绕中心深度球心的`d_phi/d_theta/d_r/pan`，也不能直接冒充真实4×4目标相机；
- 底层点渲染器接受显式PyTorch3D相机，因此冻结适配器改为：用现有源RGB、RGB-only BaseDepth和真实K反投影源点云，按真实源/目标外参插值25帧相机，render video作为唯一视频扩散条件，末帧才与val目标真值评价；
- 不下载2,285,005,731字节DUSt3R权重，避免改变BaseDepth与相机估计这两个变量；相对GenWarp唯一核心变化是单帧生成先验变为视频一致生成先验；
- 首目标仍固定`ai_024_013/right`，50步、seed 20260818、显存门15.5 GiB。冒烟通过前不批跑14 val；即使冒烟通过也只代表接口/显存通过。

正式审计记录为`records/stage1_8_viewcrafter_audit_v1.json`。下一步从现有RTX 5080兼容环境派生隔离`viewcrafter`环境，安装现代PyTorch3D与最小推理依赖，并只下载512 checkpoint。

## 15. ViewCrafter 隔离环境就绪（2026-08-18）

- 没有修改现有GenWarp、SHARP或Flash3D环境；最终使用全新`viewcrafter2`环境，固定Python 3.10、PyTorch 2.7.1+cu128和torchvision 0.22.1+cu128；
- 官方PyTorch3D commit `2bce7110...`面向RTX 5080的`sm_120`源码编译通过。首次缺`cusparse.h`、第二次因Conda CUDA头文件位于`targets/x86_64-linux/include`而缺`cuda_runtime_api.h`，均保留为环境诊断；显式加入该include/lib路径后第三次通过；
- `torch`、`pytorch3d`、`pytorch_lightning`与`open_clip`静态导入通过，CUDA可见；本阶段没有加载扩散权重或运行GPU推理；
- 失败的克隆环境`viewcrafter`不再使用，也未删除；canonical环境为`viewcrafter2`。源码、环境和后续checkpoint均为本机可重建资产，不进入Git。

正式记录为`records/stage1_8_viewcrafter_environment_v1.json`。下一步只下载官方`ViewCrafter_25_512/model.ckpt`并严格核对10,437,386,907字节与SHA256；完成前不编写成批评价结论、不读取test/P01。

## 16. ViewCrafter 真实相机适配 CPU 门通过（2026-08-18）

- 新增冻结配置与单目标脚本，首目标保持`ai_024_013 / source 35 → right 22 / 14.9816°`，没有因为GenWarp质量结果改变样本；
- 将768×1024源图中心裁为640×1024，再缩放至官方512模型的320×512；对源/目标K同步执行主点平移与0.5倍缩放；
- 以源相机为世界坐标原点，目标外参使用真实`target_w2c @ inv(source_w2c)`；25帧轨迹用旋转Slerp与平移线性插值，并强制首尾为精确源/目标姿态；
- 点云保持OpenCV RDF，按官方ViewCrafter列变换转换PyTorch3D相机。最终帧抽样2,560点与canonical OpenCV投影对比，p99为0.00155 px、最大0.00236 px，严格通过0.05/0.1 px门；
- v1只因顶层配置状态写成`frozen`而在数据读取前失败；改成现有加载器要求的`pass`后v2通过，模型、目标和门均未改变。两次均未加载扩散模型、未运行GPU、未读取test/P01。

正式结果为`records/stage1_8_viewcrafter_val_geometry_preflight_v2.json`，v1失败现场单独保留。权重仍在可续传下载；完成字节数/SHA256校验后，才运行同一脚本的点渲染+50步扩散显存冒烟。

## 17. ViewCrafter 点渲染条件视频门通过（2026-08-18）

- 在物理GPU1运行官方PyTorch3D点渲染路径，生成25×320×512×3条件视频，全部有限，峰值allocated/reserved为1.782/1.879 GiB；
- 首帧按官方做法精确替换为源RGB，末帧保持真实目标相机下的源点云重投影。末帧精确黑背景占26.81%，主要位于右侧视锥外区域；这正是扩散模型需补全的输入缺口，不是相机方向错误；
- 人工只读核对中，已覆盖区域的墙面、画框和房梁方向与目标真值一致，但源点云无法呈现目标图右侧新内容；这与0.00236 px最大投影误差共同支持适配正确；
- v1/v2/v3分别保留懒加载模块、Python名称遮蔽与inference tensor原地更新兼容失败。v4只做渲染、未读取checkpoint、未执行扩散；兼容修正不改变相机、点云或质量门。

正式记录为`records/stage1_8_viewcrafter_val_render_preflight_v4.json`。权重校验完成后，下一步只剩同一输入上的50步扩散与15.5 GiB峰值门。

## 18. 下载与存储一次性审计（2026-08-18）

为避免实验被零散下载反复打断，已把当前决策树审计到以下边界：ViewCrafter单目标冒烟→可能的14-val RGB→生成RGB的既有UniDepth深度门→满足质量门后才进P01。该范围内唯一尚缺的网络资产是固定revision的`ViewCrafter_25_512/model.ckpt`（10,437,386,907字节）；其余数据、BaseDepth、visibility、UniDepth、相机、P01输入、PyTorch3D和视频编码器均已本地就绪。

不会补下以下冗余资产：

- DUSt3R（2,285,005,731字节）：当前真实相机/BaseDepth适配不需要，且官方eval会读目标帧；
- OpenCLIP LAION初始化权重：官方ViewCrafter checkpoint严格包含编码器参数，适配器改为`pretrained=None`实例化后严格加载checkpoint，避免“下载后立即被覆盖”；
- ViewCrafter 1024版、Stable Virtual Camera、MetaView/Qwen/Depth Anything 3：均不在已冻结的下一步；
- 全量Hypersim、RealEstate10K或CO3D：若ViewCrafter失败，先冻结可训练多视图先验的数据契约；首轮仍可复用现有50场景650文件，不提前下载TB级数据。

空间清理方面，GenWarp 14-val两项绝对RGB门已正式失败，后续不再运行。其4个大权重在逐文件复核原记录SHA256后删除，释放8,427,337,711字节（7.849 GiB）；源码、小配置、正式哈希和全部评价产物保留，可按记录重新下载恢复。当时先列出`genwarp`与失败克隆`viewcrafter`两个环境为人工清理项；ViewCrafter路线随后也在第20节关闭，`viewcrafter2`现已一并转为人工清理项。

正式审计为`records/stage1_8_download_and_storage_audit_v1.json`。后续不再边跑边新增权重下载；只完成当前ViewCrafter断点并校验，随后连续推进到需要模型/质量决策为止。

## 19. ViewCrafter 单val 50步技术冒烟通过（2026-08-18）

- checkpoint完成10,437,386,907字节下载后，官方Hugging Face LFS元数据与本地SHA256共同确认内容哈希为`e15528ca...`；此前记录的`0864b08b...`只是未经内容验证的远端标识，已完整纠正且无需重下；
- 严格checkpoint加载包含OpenCLIP文本/图像编码器，因此实例化时设`pretrained=None`，避免重复下载后再覆盖；
- v1未装可选xFormers，普通空间attention在进程约12.90 GiB时还需3.05 GiB而OOM。采用PyTorch 2.7原生SDPA替换同一无mask空间attention，CPU等价测试最大绝对差`7.45e-8`；时间相对位置attention不变；
- v2已完整执行50步且不OOM，但薄适配器遗漏官方`run_diffusion`返回前的`clamp[-1,1]`，因此误触范围门；v3补齐官方后处理，不改模型、seed、相机或步数；
- canonical v3含模型加载84.95秒，峰值allocated/reserved为14.203/14.588 GiB，低于15.5 GiB；25帧3×320×512输出全部有限，原始最大1.0801，官方clamp后范围为[-0.9487,1.0]；投影门仍为p99 0.00155 px、最大0.00236 px。

视觉只读预检中，末帧能把26.81%黑缺口变为连续画廊，但画作、门洞和长凳仍有生成式变化。因此本阶段只判技术门通过，不以单样本宣称质量。正式记录为`records/stage1_8_viewcrafter_val_smoke_v3.json`；随后已按相同checkpoint、SDPA、50步和seed策略完成第20节的冻结14-val评价。

## 20. ViewCrafter 14-val RGB质量门未通过，公开先验路线收口（2026-08-18）

- 首轮批量运行在GPU1首目标生成时因既有跨卡进程占用288 MiB而OOM；未终止该进程，也未改模型。转到GPU2并启用PyTorch expandable segments后，同一冻结v2协议完整运行14/14端点；
- v1在第12端点遇到一个约`(-9151,14999) px`、目标深度仅0.014 m的极端离屏投影。没有放宽0.05/0.1 px阈值：v2保留所有正深度点p99门，把最大误差门明确限定为能影响目标画幅的点，并单列全部正深度最大值诊断。14端点CPU预检全部通过，最坏p99为0.00801 px、画幅内最大误差0.000170 px；
- 遮挡后区域从点渲染MAE 98.01降到49.04，相对改善49.96%；视锥外从101.54降到59.35，改善41.56%。两类均过“改善≥15%”，但均未过绝对`MAE≤35`，且各自只有2/14端点达到绝对门；
- 正式运行含模型加载625.40秒，峰值allocated/reserved为14.203/14.588 GiB。结果表明模型能补掉黑洞，但跨场景目标内容不够准确，不能作为30°补充高斯的RGB来源；
- 按冻结短路协议，未运行生成RGB的UniDepth、held-out test、P01或Gaussian Spawn，也不再下载DUSt3R、ViewCrafter 1024或其他未冻结候选。正式结果为`records/stage1_8_viewcrafter_val_rgb_v2.json`。

公开冻结先验路线至此正式收口。已核对10,437,386,907字节和`e15528ca...` SHA256后删除ViewCrafter checkpoint，释放9.721 GiB；连同此前GenWarp权重累计释放约17.57 GiB。源码、配置、记录和14-val视觉证据保留。下一步不是继续换二维补洞器，而是先冻结“使用现有HLP-TRAIN-03定向数据启动的可训练多视图先验”协议，再决定是否增量下载更多数据。
