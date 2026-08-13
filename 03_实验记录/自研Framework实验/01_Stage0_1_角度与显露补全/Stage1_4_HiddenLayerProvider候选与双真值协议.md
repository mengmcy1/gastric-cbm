# Stage 1.4 HiddenLayerProvider 候选与双真值协议

> 状态：**v1 边界已冻结（2026-08-13）；Flash3D 确定为 `OcclusionHiddenProvider` 首候选，`OutsideFOVProvider` 尚未选定。**

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
- 48个官方验证序列全部找到元数据，16个序列存在合格源帧，当前每序列最多保存3组，共48个候选三元组；
- 记录为 `records/stage1_4_re10k_angle_triplets_v1.json/.csv`。

该数据只提供目标 RGB、内参和位姿，可评价新视角 RGB 与视频一致性；它没有稠密深度，不能直接评价隐藏层深度，也不能可靠拆分遮挡后区与视野外区。

### 3.2 `HLP-GEO-01`：Hypersim 稠密几何评价

Hypersim 官方发布包含 RGB、逐像素相机距离、世界坐标、相机位姿和场景内参，适合构造真值可见性、遮挡后区和视野外区。其 `depth_meters` 是到光心的欧氏距离，不是 Z-depth，进入 CLB 前必须按官方内参转换。

全量数据约1.9TB，不进入当前机器。只采用官方 test split，先从官方 ZIP 远程索引中筛相机轨迹和约 ±15°三元组，再按文件下载选中帧的 RGB、depth/position、相机姿态和必要元数据。首版目标为3个场景，每个场景1个三元组；在完成角度、遮挡显露量和下载体积预检前不批量下载。

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

已通过的只是 Provider **静态预检**与 RealEstate10K **角度数据预检**，尚未运行 Flash3D 推理。下一步顺序固定为：

1. 建立隔离、RTX 5080兼容的 `flash3d` 环境，先做官方 checkpoint 单图最小推理；
2. 从48个候选中选择5个不同序列，下载源/左右目标9至15张图，建立 `HLP-APP-01`；
3. 实现 Flash3D→CLB adapter，先只评价 `occlusion_hidden`，不声称覆盖 `outside_source_fov`；
4. 远程扫描 Hypersim test 场景内相机姿态，冻结3个 `HLP-GEO-01` 三元组后才下载选定文件；
5. 两个真值集通过后，才把候选接到 P01，不直接重跑30°视频。
