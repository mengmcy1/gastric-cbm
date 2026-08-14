# InfiniSplat 跨场景真实大基线

## 目标

复用 `03_论文大基线协议复现` 的严格口径，对 ETH3D 官方多视图场景生成更多真实 Source→Target 大基线样例：

- Target 使用真实照片与真实 COLMAP 外参；
- Source/Target 都使用 ETH3D 完整 PINHOLE 内参，不走 EXIF 或 30 mm 回退；
- 报告真实米制基线、相对旋转、序列间隔、共视代理和 Alpha 覆盖；
- 与 30° `true_arc` 合成压力测试分开记录。

## 当前状态（2026-08-14）

- 已确认官方 InfiniSplat 仓库自带的 ScanNetPP、MyBedroom、Waymo9 和 Waymo147 只有单张示例；本地 `.npz` 只提供稀疏深度/掩码，不含真实 Target 照片与 Target 位姿，因此不能冒充真实大基线。
- ETH3D Kicker 与 Pipes 官方 undistorted 多视图包具备完整相机标定，已开始下载；下载、解压、帧匹配和大基线筛选完成后再运行 GPU 推理。
- 当前不宣称已有 Kicker/Pipes 真实 Source→Target 结果。
