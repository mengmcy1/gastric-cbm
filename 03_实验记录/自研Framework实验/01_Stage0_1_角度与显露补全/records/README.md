# 自研 Framework 可同步记录

状态：**Stage 1.8-2首个64-train/14-val目标条件候选已正式不通过；support分离通过但RGB/深度质量未过门。下一步为深度尺度上限与冻结预训练生成先验审计（2026-08-17）。**

本目录用于保存从本地大型 `outputs/` 提升出来、可以提交 Git 的小型正式证据。不要在这里保存 PLY、MP4、逐帧图像、Depth、Alpha、NPY、模型或日志。

## 当前 canonical 运行

`outputs/02_Stage1_true_arc/03_正式矩阵/P01_stage01_true_arc_formal_v4_fallback_20260807/`

该目录当前存在于 `win5060`；`linux5080` 是否持有副本以其本地清单为准。缺少大型输出不改变这次运行已经完成的事实。

## 已同步的小型证据

- `stage1_matrix_summary.json`：从 canonical 输出归一化得到，删除不可迁移的旧绝对路径，并写明人工结论；
- `stage1_matrix_metrics.csv`：六个端点的正式自动指标；
- `manual_review.json`：用户完整播放 5°/15°/30° A/B 后的正式评分；
- `artifact_manifest.json`：登记配置、输入、六个视频、30°融合 PLY和源汇总的字节数与完整 SHA256。
- `p01_ang30_mat_static_review.json`：登记 `linux5080` 上 MAT Places-512 固定30°左右端点静态运行、环境、权重来源、输出哈希和用户“不通过”结论；大型输出与 checkpoint 不进入 Git。
- `stage1_1_method_audit.json`：登记3D Photo Inpainting/SLIDE 方法、源码、许可证、环境和输入协议审计；
- `p01_ang30_3dphoto_adapter_v1_review.json`：登记3D Photo 三网络固定端点适配及用户左右严重度3、虚假建筑、静态门不通过结论；
- `p01_ang30_observed_reprojection_v1.json`：登记中心原图 RGB-D 向±15°端点重投影的几何验证和覆盖率。左右合计568,341个 accepted 像素中仅627个有中心来源，99.8897%仍无观测。
- Stage 1.3 协议本体保存在上级 `Stage1_3_隐藏层表示与验证协议.md`，机器规则保存在 `configs/stage1_3_hidden_layer_protocol_v1.json`。
- `clb_syn_01.json`：登记 `CLB-SYN-01` 受控合成表示实验。中心无隐藏层泄漏，两端均显露遮挡后方层和对应视锥外层，全部表示硬门通过。
- `clb_gs_syn_01_v1_failed.json`：保留 CLB→Gaussian 首转换中斜视端点覆盖不足和未分层边界深度指标失真的失败证据；
- `clb_gs_syn_01.json`：登记 canonical v2 CLB→Gaussian 合成真值转换。中心可靠内部泄漏、两端隐藏区改善、Alpha 覆盖和分表面深度门全部通过；全图边界带差异作为诊断值保留。
- `stage1_4_re10k_angle_triplets_v1.json/.csv`：按真实相机姿态从 Flash3D 官方验证序列筛选30°总范围三元组；48个序列中16个合格，保存48组候选，不把时间偏移当角度。
- `stage1_4_flash3d_provider_preflight_v1.json`：Flash3D 官方 commit、checkpoint SHA256、双高斯/深度增量 decoder 和角度记录的 CPU 静态预检；通过不等于 GPU 推理或质量通过。
- `stage1_4_flash3d_environment_v1.json`：记录 `linux5080` 独立环境、历史 UniDepth/xFormers commit、本地模型哈希、rasterizer/attention 兼容边界和最小推理通过状态。
- `stage1_4_flash3d_layer_smoke_v1.json`：Flash3D `re10k_v2` 在物理 GPU 1 上的最小层输出证据；运行0.895秒、峰值显存约2.49 GB，两层深度与 Gaussian 参数全部有限，第二层深度增量100%为正。该结果只通过 schema/数值门，不等于新视角质量通过。
- `stage1_4_hlp_app_01_v1.json` 及 `supplement1`–`supplement4`：保留5轮定向下载回执；12个候选中5个成功，7个因 YouTube BotDetection/远程断连失败，失败不被改写成数据质量失败。
- `stage1_4_hlp_app_01_final_v1.json`：`HLP-APP-01` 正式完整性记录；5个不同序列、15帧 RGB，所有文件哈希、帧时间戳元数据对应与三元组尺寸一致性均通过。该数据集无稠密深度；其目标位姿平移尺度已由后续 COLMAP/RANSAC 记录恢复。
- `stage1_4_hlp_app_01_colmap_v1.json`：从官方 `pcl.test.tar` 通过校验的 HTTP Range 定向取得5个稀疏点云成员；38次请求共接收约289.5 MB，没有下载10.31 GB全包，5个 gzip/pickle 均包含 `xys/p3D_ids/xyz` 并通过帧数校验。
- `stage1_4_hlp_app_01_scales_v1.json`：按 Flash3D 官方的源帧 UniDepth + COLMAP RANSAC 方法恢复5个序列的平移尺度；尺度为0.5291–1.4961，最低内点率52.83%，全部为正有限值。
- `stage1_4_flash3d_to_clb_v1.json`：5序列 Flash3D 第二层到 `occlusion_hidden` CLB 的原始适配证据；5/5 schema 通过，但模型无原生独立置信度，两类 confidence 均显式为0，状态为 `schema_pass_uncalibrated`，不允许据此进入已接受融合。
- `stage1_4_flash3d_layer_ab_v1.json`：同一源预测与尺度位姿下，第一层 vs 双层的10个左右端 A/B；双层在8/10个视角提升 PSNR、9/10提升 SSIM，平均变化 `+0.9378 dB / +0.0570`。当前状态是 `complete_metrics_partial_protocol`：LPIPS、DISTS、独立遮挡后区真值和置信度校准尚未完成，不等于 Provider 质量通过。
- `stage1_4_hlp_geo_01_fetch_v1.json`：3场景、39个 Hypersim 精确文件的远程 ZIP Range 获取回执；44,433,462字节全部完成且 HDF5 可读，没有下载约12.4 GB的完整场景 ZIP。
- `stage1_4_hlp_geo_01_visibility_v1.json`：保留首次使用1 cm 绝对 position→depth P99 门时对远距离 float16 教堂场景的失败；不覆盖、不写成通过。
- `stage1_4_hlp_geo_01_visibility_v2.json`：改用有量化依据的相对 P99 ≤0.2% 一致性门；6个目标分类完整，共524,138个 `occlusion_hidden` 像素与1,122,337个 `outside_source_fov` 像素。几何冲突区单独保留，不混入遮挡后真值。
- `stage1_4_flash3d_hypersim_geometry_v1_failed.json`：保留首轮 alpha 后处理 NumPy 参数错误；没有完成几何指标。
- `stage1_4_flash3d_hypersim_geometry_v3.json`：canonical 原始米制结果；6端点遮挡后平均覆盖98.86%，但 AbsRel 23.33%、δ<1.10 24.80%；源第一层平均 AbsRel 22.86%。
- `stage1_4_flash3d_hypersim_geometry_v4_oracle_scale_failed.json`：保留 oracle 尺度首运行的 inference tensor 原地修改失败；没有完成 oracle 目标指标。
- `stage1_4_flash3d_hypersim_geometry_v5_oracle_scale.json`：使用源深度真值做不可部署的场景尺度对齐诊断；对齐后 AbsRel 仍为18.17%、δ<1.10 39.50%，证明错误不只是全局尺度偏移。
- `stage1_5_core_interface_smoke_v1.json`：自研核心接口第一版4项测试记录，已由 v2 supersede，保留版本证据。
- `stage1_5_core_interface_smoke_v2.json`：加入 Camera/Bundle、安全 I/O 和更严格 Alpha/支持类型门后的6项测试记录，已由 v3 supersede。
- `stage1_5_clb_io_roundtrip_v1.json`：Stage 1.3 合成 CLB 经新核心库旧→新→旧 round-trip；3相机、5 patch、121,600有效像素的61项精确门、NPZ 哈希及旧验证器全部通过。
- `stage1_5_core_interface_smoke_v3.json`：加入模型无关遮挡后几何与 confidence evaluator 后9项测试全部通过，已由 v4 supersede。
- `stage1_5_hlp_geo_dataset_adapter_smoke_v1.json`：HLP-GEO 数据/相机适配 canonical 冒烟；3场景、6目标、524,138个遮挡后像素和1,122,337个视锥外像素与冻结可见性记录一致，默认不向 Provider 暴露源深度真值，60项门全部通过。
- `stage1_5_core_interface_smoke_v4.json`：加入 Hypersim 数据/相机适配层后12项测试全部通过，已由 v5 supersede。
- `stage1_5_core_interface_smoke_v5.json`：当前 canonical 核心记录，supersede v4；加入 Flash3D 第二层→未校准 CLB 与原生目标渲染→统一几何预测的 CPU 只读映射后15项测试全部通过。该记录不代表仓库后端或 GPU 指标已复现。
- `stage1_5_flash3d_reference_regression_v1.json`：物理 GPU 2 上的 Flash3D 只读仓库后端回归；3个 CLB 均可重载，6视角覆盖率相对冻结 v3 差值全0，AbsRel 最大差约`6.43e-10`，平均覆盖率/AbsRel 精确复现为`0.9886012228320182/0.23331674487628928`。该 pass 只证明接口忠实，不改变 Flash3D Provider 质量未通过结论。
- `stage1_6_hlp_train_01_pose_fetch_v1.json`：8 train、2 val候选的40个姿态/场景比例文件定向获取；166,171字节全部可读，held-out交集为0。
- `stage1_6_hlp_train_01_pose_selection_v1.json`：10个候选均选出真实`-15°/0°/+15°`三元组；每场景68–93个合格源帧，端点误差和平移门全部通过。
- `stage1_6_hlp_train_01_fetch_v1.json`：10个三元组130/130个RGB/depth/position/pose文件定向获取，共136,035,863字节；未下载完整场景ZIP。
- `stage1_6_hlp_train_01_hidden_evidence_v1_failed.json`：保留首次遇到倾斜传感器、旧适配器拒绝非对角`M_cam_from_uv`的失败；无数组写出，未删除或覆盖失败目录，也未替换场景。
- `stage1_6_hlp_train_01_hidden_evidence_v2.json`：canonical真实正证据生成；10个三元组共912,897个隐藏层正证据像素，最小/中位场景占比1.6127%/12.1352%，50项门和10个NPZ复核全部通过。未观测像素保持unknown，不作为负标签。
- `stage1_6_core_supervision_smoke_v6.json`：当前 canonical 核心记录，supersede Stage 1.5 v5；加入一般Hypersim投影RQ分解、train/val加载器和正证据生成器后20项测试全部通过。
- `stage1_6_hlp_train_01_frozen_base_depth_v1_failed.json`：离线 `from_pretrained` 因本地缓存缺少配置文件而在模型构造前失败；0个数据数组被访问、0个产物写出，canonical 由 v2 supersede。
- `stage1_6_hlp_train_01_frozen_base_depth_v2.json`：10个样本的冻结 RGB-only UniDepth V1 可见深度输入；不读取源真值深度、不传真值内参，10个NPZ与10张预览哈希复核通过，全部预测有限且为正。
- `stage1_6_core_trainable_model_smoke_v7.json`：当前 canonical 核心记录，supersede v6；三头 U-Net、严格正相对隐藏深度、non-negative PU support 与正证据 uncertainty 损失加入后24项测试全部通过。只证明实现契约，不代表训练或质量通过。
- `stage1_6_hlp_train_01_single_overfit_v1_failed.json`：确定性模式下 CUDA indexed median 在首次前向失败；0步、0产物，已用确定性均值/平均绝对偏差修正。
- `stage1_6_hlp_train_01_single_overfit_v2.json`：500步 `log(1+r)` 诊断；比例 AbsRel降至0.336但未过0.10门，状态 failed，保留用于证明小比例欠权重。
- `stage1_6_hlp_train_01_single_overfit_v3.json`：直接 `log(r)` 后比例 AbsRel降至0.115，support正例均值0.595，仍未过冻结门，状态 failed。
- `stage1_6_hlp_train_01_single_overfit_v4.json`：canonical单样本过拟合通过；750步后比例 AbsRel 0.08091、改善97.65%，正例/unknown support 0.84581/0.02012，6/6门通过，7个产物哈希与checkpoint重载复核通过。
- `stage1_6_core_trainable_model_smoke_v8.json`：当前 canonical 核心记录，supersede v7；确定性归一化、直接log比例损失与1%小比例梯度测试加入后25项全部通过。
- `stage1_6_hlp_train_01_small_candidate_v1.json`：8 train图/2 val图首个联合候选，val比例误差改善但support正例/unknown分离与train改善门未过，状态failed；未运行HLP-GEO test或P01，不作为冻结候选。
- `stage1_6_hlp_train_02_pose_selection_v1.json`：相同8 train/2 val场景各扩展10个唯一源帧，100/100组三元组通过真实±15°端点、平移与held-out零交集门。
- `stage1_6_hlp_train_02_fetch_v1.json`：HLP-TRAIN-02按场景去重后769/769个精确RGB/depth/position/pose文件，共1,093,518,150字节且全部HDF5可读；未下载完整场景ZIP。
- `stage1_6_hlp_train_02_hidden_evidence_v1.json` / `v2.json`：均保留100组标签中同一个float32量化边界像素导致严格相对门失败的证据；未放宽门，canonical由v3 supersede。
- `stage1_6_hlp_train_02_hidden_evidence_v3.json`：100组canonical正证据，8,721,744个正像素、500/500门通过；最终float32值仍严格满足1%与1cm门。
- `stage1_6_hlp_train_02_frozen_base_depth_v1.json`：80 train/20 val共100份RGB-only UniDepth冻结输入，全部有限且为正，100个NPZ与100张预览哈希复核通过。
- `stage1_6_core_quantized_supervision_smoke_v9.json`：当前canonical核心记录，supersede v8；增加float32落盘边界二次门和回归测试后26项全部通过。
- `stage1_6_hlp_train_02_candidate_v1.json`：80/20的positive-only nnPU联合候选；几何改善但support在val坍缩为常数，状态failed，未运行test。
- `stage1_6_hlp_train_02_candidate_v2_failed.json` / `v3_failed.json`：分别保留target-view free-space首轮确定性cuBLAS配置缺失与单样本无verified-negative时loss拒绝的实现失败；均不构成质量结论。
- `stage1_6_hlp_train_02_candidate_v4.json`：目标视角free-space负证据联合候选；约11.7万val负例已形成，但旧浅层/无相机射线模型未学会support分离，状态failed。
- `stage1_6_hlp_train_02_candidate_v5.json`：相机射线、/16上下文和delta-conditioned头加入后的batch=1联合候选；仍有顺序梯度冲突，train/val门未过，状态failed。
- `stage1_6_hlp_train_02_single_free_space_overfit_v1.json`：train-only free-space可学习性通过；比例AbsRel 0.0913，support正例/verified-negative 0.8652/0.1349，4/4门通过。
- `stage1_6_hlp_train_02_candidate_v6.json`：4样本梯度平均联合候选；train分离略有改善但val仍不分离，证明同8场景内扩到80帧不足以获得跨场景泛化，状态failed且未运行test。
- `stage1_6_hlp_train_03_candidate_build_v1.json`：官方元数据层按场景类型冻结60 train/20 val姿态候选，80场景唯一、公开帧≥80、held-out交集0。
- `stage1_6_hlp_train_03_pose_fetch_v1.json`：80场景320/320个姿态/尺度文件定向获取，1,329,409字节全部可读，未下载完整ZIP。
- `stage1_6_hlp_train_03_pose_selection_preflight_v1.json`：79/80候选找到合格±15°三元组，唯一失败train候选保留；该记录状态failed_selection用于完整审计。
- `stage1_6_hlp_train_03_triplet_finalize_v1.json`：从合格池按场景类型冻结40 train/10 val正式三元组；场景唯一、姿态与held-out隔离门全部通过。
- `stage1_6_hlp_train_03_fetch_v1.json`：50个独立场景650/650个精确文件、701,705,049字节全部可读；未下载完整ZIP。
- `stage1_6_hlp_train_03_hidden_evidence_v1.json`：50场景共3,388,680个正证据像素，最小/中位样本占比0.40487%/6.78482%，250/250门通过。
- `stage1_6_hlp_train_03_frozen_base_depth_v1.json`：40 train/10 val共50份RGB-only UniDepth冻结输入，全部有限且为正。
- `stage1_6_hlp_train_03_candidate_v1.json`：40独立train场景/10独立val场景的1000-update候选；几何误差有改善，但val support正例/verified-negative分离仅0.02204，未过0.05冻结门，状态failed且未读取test/P01。
- `stage1_6_core_trainable_model_smoke_v10.json`：当前canonical核心记录，supersede v9；相机射线、/16上下文、delta-conditioned独立头、target-view verified free-space与unknown隔离共30项测试全部通过。实现通过不改变HLP-TRAIN-03质量失败结论。
- `stage1_7_official_ldi_true_arc_v4.json`：官方完整LDI构造接入P01冻结深度、实测焦距和统一`true_arc` 30°轨迹的canonical记录；视频61帧、无裁剪，左右端灰背景自动占比27.81%/26.81%，当前等待人工评阅。
- `stage1_7_official_ldi_true_arc_v4_manual_review.json`：用户正式评分为空洞3/3、拉伸/虚假结构2、连续性尚可、总体不通过；停止3D Photo调参、P02/P05扩展与补充高斯转换。
- `stage1_8_target_view_visibility_v1.json` / `v2.json`：保留首两个校准样本选择中val左端没有`outside_source_fov`真值而失败的证据；没有降低4端点双分支非空门。
- `stage1_8_target_view_visibility_v3.json`：canonical三分区真值；2场景、4端点均同时包含遮挡后和视锥外区域，累计191,988/728,090像素，全部可见性门通过。
- `stage1_8_target_view_oracle_smoke_v3.json` / `v4.json`：保留目标视角evaluator早期通过记录；v4修正“协议排除区support”的字段语义，均由v5 supersede。
- `stage1_8_target_view_oracle_smoke_v5.json`：当前canonical evaluator校准记录；2场景、4端点、31/31门通过，两类新显露区oracle覆盖率1、深度AbsRel 0、RGB MAE 0，并登记runner/evaluator及输入SHA256。该pass不是模型质量结果。
- `stage1_8_target_view_single_overfit_v1.json`：1200步首轮单端点过拟合；绝对RGB/深度/support等门通过，但两区RGB相对改善89.26%/88.17%未过冻结90%门，状态failed并保留。
- `stage1_8_target_view_single_overfit_v2.json`：canonical单端点可学习性通过；唯一改变为2000步，两区RGB MAE降至2.15/3.88、深度AbsRel 0.33%/0.24%，双support分离及相机敏感性共12/12门通过。只证明单train目标可记忆。
- `stage1_8_target_view_training_visibility_v1.json`：100端点全量三分区真值；分类与规模门通过，但13端点未过原0.2% depth-position P99门，因此整体保留为fail，数组仍由后续逐端点资格脚本校验。
- `stage1_8_target_view_training_pool_v1.json`：逐端点严格资格记录；100份数组哈希全部一致，最终64 train/14 val共78目标合格，22个拒绝目标及原因完整登记。
- `stage1_8_target_view_candidate_v1.json`：首个联合候选，最佳update 800；val support分离0.147/0.634通过，但RGB MAE 43.46/59.43、深度AbsRel 0.593/1.074未过门，状态failed且未读取test/P01。
- `stage1_8_depth_scale_diagnostic_v1.json`：14个val目标的不可部署逐目标正尺度oracle诊断；两类深度AbsRel对齐后仍为0.441/0.365，均未过0.30，证明全局尺度不是充分修复。
- `stage1_8_pretrained_backbone_audit_v1.json`：显式相机条件预训练骨干审计；冻结GenWarp `multi2`为首个16GB可运行性候选，最小权重约8.43GB。只允许先做单val目标冒烟，未读取held-out test/P01。
- `stage1_8_genwarp_environment_v1.json`：`linux5080`隔离环境与权重就绪记录；GenWarp及splatting冻结commit、PyTorch/CUDA兼容版本、两处最小扩展补丁、8个权重文件共8,429,409,786字节及SHA256全部登记。状态暂停在GPU冒烟前。

运行时没有可靠记录项目 Git commit，因此 manifest 中保持 `null`，没有用后来的提交号代替。大型输出继续由 `.gitignore` 排除；需要在 `linux5080` 复核时按 manifest 选择性复制并校验。

## 两台机器的 outputs 统一结构

```text
outputs/
├─ 00_Stage0_角度协议/{01_dryrun,02_校准冒烟,03_正式A基线}/
├─ 01_Stage1_legacy开发基线/{01_组件冒烟,02_完整链路冒烟}/
├─ 02_Stage1_true_arc/{01_dryrun,02_功能冒烟,03_正式矩阵}/
└─ 90_失败与诊断归档/{01_早期脚本失败,02_深度质量门预期失败}/
```

`linux5080` 不需要复制 `win5060` 的全部 outputs。默认从 Git 获取脚本、配置、输入和本 records 后重新运行；新运行使用新的运行 ID和 `producer_machine_id=linux5080`。只有需要查看同一视频或继续使用同一 SHA256 的 PLY时才传输大型文件。
