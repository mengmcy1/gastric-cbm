# InfiniSplat 官方跨场景样例筛选

## 目的

补足此前 ETH3D courtyard 四组相机对场景单一的问题，为 PPT 生成室内、道路和工业场景的官方输入结果池，让用户先选择最有代表性的案例。

## 协议边界

- 本轮统一使用官方 RGB-only checkpoint 和官方 Demo 视频轨迹；
- ScanNet++ 室内结果复用已完成产物；
- Waymo 与 ETH3D kicker/pipes 图来自官方 `lidar_demo`，但本轮先按 RGB-only 模式运行，目的是保持筛选矩阵模型一致；
- 因此本轮不是 LiDAR 模式实验，也不是论文真实 target 相机 Benchmark；
- 用户选中道路或工业案例后，再决定是否下载 LiDAR checkpoint 做正式模态对照。

## 候选

室内：ScanNet++、my_bedroom、old_livingroom、bedroom；道路：Waymo 9、Waymo 147；工业：ETH3D kicker、ETH3D pipes。

## 完成状态

7张新增图已在物理GPU2上一次加载 RGB checkpoint 后顺序完成 PLY 与60帧视频；GPU1当时有约6.4GB既有任务，未抢占。加上复用的 ScanNet++，PPT筛选池共8个场景。

快速选择结论：道路优先 Waymo147，失败反例用 Waymo9；室内优先 ScanNetPP 和 MyBedroom；工业优先 ETH3D Kicker。`Bedroom` 虽然近黑区域最少，但Demo轨迹视觉变化最弱。

RGB近黑区域代理汇总位于 `outputs/cross_scene_video_black_area_summary.{csv,json}`。该代理不是Alpha，不能作为正式空洞率。

## 30° `true_arc` 追加压力测试

2026-08-14 已将五个优先候选的冻结 RGB-only PLY 按统一总范围 30°（-15°→+15°）、61帧、原相机为中心的 `true_arc` 重渲。GPU1 当时有正在运行的 Python 进程，本轮使用空闲物理 GPU2，未终止任何既有进程。

端点硬空洞率（左/右）：ScanNetPP 7.55%/16.16%，MyBedroom 5.96%/7.37%，Waymo147 17.60%/20.06%，Waymo9 19.87%/16.56%，ETH3D Kicker 21.48%/16.42%。五组在覆盖区仍保持主体结构，但端点的顶部、底部和侧边界出现明显黑洞，部分近景边界伴随拉伸或柔化。

本结果是**合成相机压力测试**，不是真实 Target 位姿。汇总为 `outputs/large_angle_30deg_true_arc_summary.{csv,json}`，RGB/Alpha 总览为 `outputs/large_angle_30deg_true_arc_*overview.jpg`，可复现脚本为 `scripts/summarize_cross_scene_true_arc.py`。
