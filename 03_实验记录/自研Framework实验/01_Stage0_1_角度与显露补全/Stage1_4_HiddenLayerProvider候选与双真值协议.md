# Stage 1.4 HiddenLayerProvider 候选与双真值协议

> 状态：**v1 边界已冻结（2026-08-13）；Flash3D 首候选已完成 `HLP-APP-01` 受控 A/B 和 `HLP-GEO-01` 稠密几何评价。原始米制几何 AbsRel 23.33%，oracle 尺度对齐后仍为18.17%，且 opacity/alpha 不能作为稳定 confidence；因此原样不通过 `OcclusionHiddenProvider`，固化为对照基线。下一步转 Stage 1.5 自研可替换 Provider 骨架；`OutsideFOVProvider` 尚未选定。**

## 1. 先拆接口，不让一个模型承担它做不到的区域

Stage 1.3 的 `HiddenLayerProvider` 现在拆成两个子接口：

- `OcclusionHiddenProvider`：预测中心视锥内部、可见前景后方的隐藏表面；
- `OutsideFOVProvider`：预测中心原图视锥外、但会在 ±15°端点进入画面的扩展表面；
- 两者都先输出 CLB，再经统一验证和融合进入 Gaussian Spawn。

Flash3D 和 MINE 都只在中心源相机视锥内建立表示。它们可以验证遮挡后层，但不能生成原图之外的世界。P01 30°的大块外侧缺口不得交给 Flash3D 后再被误记为“隐藏层已解决”。

## 2. 首候选为什么是 Flash3D

官方 Flash3D `re10k_v2` 默认每个源像素预测两个高斯：第一层深度来自 UniDepth，第二层通过正深度增量累加；两层分别带 opacity、scale、rotation 和颜色特征。它比 MINE 的连续 frustum density 更接近现有 CLB→Gaussian 接口，所以先作为 `OcclusionHiddenProvider`。

但适配器必须补齐四件事：

1. 把第二层从完整高斯输出中显式分离成 `occlusion_hidden`；
2. 把官方相机坐标、累计深度和颜色转成 CLB；
3. 另行产生并校准几何/外观置信度；官方 opacity 是渲染量，不能冒充置信度；
4. 对中心视锥外区域明确输出 unsupported，不得外推或复制边缘内容。

MINE 保留为第二候选/消融：它能输出任意深度 RGB 与 volume density，但到 CLB/Gaussian 的转换更间接，软件栈也更旧。

## 3. 双真值集，而不是一个数据集包办全部问题

### 3.1 `HLP-APP-01`：RealEstate10K 外观与新视角评价

- 使用官方 RealEstate10K 测试元数据和 Flash3D 官方 `re10k_mine_filtered` 验证序列；
- 新脚本按相机外参计算真实有符号水平夹角，不使用官方 `tgt5/tgt10` 时间偏移标签代替角度；
- 30°总范围固定为源视角0°、左右目标约 -15°/+15°，单端允许 ±2.5°筛选误差；
- 48个官方验证序列全部找到元数据，16个序列存在合格源帧，每序列最多保存3组，共48个候选三元组；
- `HLP-APP-01` 已完成5个不同序列、15帧 RGB 的小型集；文件哈希、帧时间戳与元数据对应、组内图像尺寸均已校验，正式记录为 `records/stage1_4_hlp_app_01_final_v1.json`；
- 官方 RealEstate10K 位姿的平移尺度不定。Flash3D 官方评价使用 `pcl.test.tar` 内的稀疏 COLMAP 点云与 UniDepth 估计深度尺度；未恢复该尺度前，15帧 RGB/位姿只能记为外观真值数据就绪，不得开始正式目标视角质量判定；
- 记录为 `records/stage1_4_re10k_angle_triplets_v1.json/.csv`。

该数据只提供目标 RGB、内参和位姿，可评价新视角 RGB 与视频一致性；它没有稠密深度，不能直接评价隐藏层深度，也不能可靠拆分遮挡后区与视野外区。

### 3.2 `HLP-GEO-01`：Hypersim 稠密几何评价

Hypersim 官方发布包含 RGB、逐像素相机距离、世界坐标、相机位姿和场景内参，适合构造真值可见性、遮挡后区和视野外区。其 `depth_meters` 是到光心的欧氏距离，不是 Z-depth，进入 CLB 前必须按官方内参转换。

全量数据约1.9TB，不进入当前机器。首版已冻结官方 test split 的3个 `cam_00` 三元组：`ai_001_010` 27/61/65、`ai_005_001` 21/2/25、`ai_008_005` 43/17/72；端点均约为 ±15°。只通过远程 ZIP Range 获取选中的39个 RGB/depth/position/姿态文件，共44.43 MB，不下载三个完整场景 ZIP。

已将每个目标像素投影到源相机，用 world position 到源相机的距离与源 `depth_meters` 区分 `observed_from_source`、`occlusion_hidden`、`outside_source_fov`、`geometry_conflict` 和 `unresolved`。v2 的 position→depth 相对 P99 全部不高于0.2%，6个目标均有非空遮挡后与视锥外真值；正式记录为 `records/stage1_4_hlp_geo_01_visibility_v2.json`。

## 4. 评价口径

- 外观：PSNR、SSIM、LPIPS、DISTS，分别报告 whole-target、`occlusion_hidden`、`outside_source_fov`；
- 几何：AbsRel、δ<1.05、深度顺序错误率、中心泄漏率、跨视角冲突率；
- 置信度：risk-coverage、selective AURC 和 ECE，验证“低置信度是否真的对应高误差”；
- RealEstate10K 只进入有真值的列，不补造深度列；Hypersim 的欧氏距离先转 Z-depth；
- 模型选择不看 P01。P01 只在候选通过小型真值门后做真实图人工外推复核。

## 5. 源码、权重和环境边界

- Flash3D 官方源码固定 commit `a71c9b92b07a76cf944f2ed894f384f0891a1960`；本地仓库顶层未包含 LICENSE 文件，模型 Hugging Face 页面虽标 MIT，也不能据此替代码许可证下结论；
- MINE 官方源码固定 commit `ef16cd83ae99c70f7970a583c533016c63587373`，仓库为 MIT；
- Hypersim 官方源码/元数据固定 commit `c85b2879c8da8ded2f4c24d2630c1ab4451999b2`，数据集为 CC BY-SA 3.0；
- Flash3D checkpoint 为690,161,102字节，SHA256 `4d717b5543941069fc120733ab4001e6d11fe73ea094d6c6054c5e8b48922420`；CPU `weights_only=True` 检查可读取390项模型状态并确认双高斯解码器存在；
- 官方推荐 PyTorch 2.2.2/CUDA 11.8，但 RTX 5080 属于新架构。本项目必须建立独立的现代 CUDA兼容环境验证，不能污染 `sharp` / `infinisplat`，也不能先假定官方老二进制可在 Blackwell 上运行。

## 6. 当前硬结论与下一步

截至2026-08-14，Provider **静态预检**、RealEstate10K **角度数据预检**和 Flash3D **RTX 5080 最小层输出推理**均已通过。最小推理固定 UniDepth commit `bebc4b2`（Flash3D 首发时最新版本），使用 xFormers 0.0.25.post1 的 Nyström 公式与显式多头折叠兼容层；不宣称与当年未固定环境逐比特一致。下一步顺序为：

1. 保留已通过的 `flash3d` 隔离环境与 `stage1_4_flash3d_layer_smoke_v1.json`，不再重复建环境；
2. ✅ 已建立5个不同序列、15帧 RGB 的 `HLP-APP-01`，7次受限下载失败亦保留回执；
3. ✅ 已从官方约10.31 GB `pcl.test.tar` 定向取得5个稀疏 COLMAP 成员，仅传输约289.5 MB Range 数据，成员结构与帧对齐全部通过；
4. ✅ 已按官方 UniDepth/COLMAP RANSAC 口径恢复5个序列的平移尺度，并完成 Flash3D 第二层到 CLB 的原始 schema 适配；由于原模型无独立置信度，当前两类 confidence 均固定为0，状态只是 `schema_pass_uncalibrated`；
5. 在同一源高斯与已恢复目标位姿上渲染“第一层” vs “第一+第二层” A/B，先评价 `occlusion_hidden`，不声称覆盖 `outside_source_fov`；
6. 远程扫描 Hypersim test 场景内相机姿态，冻结3个 `HLP-GEO-01` 三元组后才下载选定文件；
7. ✅ `HLP-GEO-01` 原始米制评价已完成：遮挡后覆盖高，但 AbsRel 23.33%、δ<1.10 24.80%；oracle 尺度对齐后仍只有18.17%/39.50%；
8. Flash3D 原样不通过 Provider 门，不接 P01，不继续针对它调参；转 Stage 1.5 实现自研可替换 Provider 骨架和可训练遮挡后几何分支。
