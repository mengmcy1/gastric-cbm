# InfiniSplat 统一相机协议记录

本目录保存可通过 Git 同步的小型证据，不保存 PLY、MP4 或逐帧图像。

- `diagnostic_controls_20260812.json`：官方示例轨迹对照与 P01 5° `true_arc` 诊断的产物身份、自动空洞指标、失败现场和结论边界。
- `official_pexels_masi_manual_review.json`：官方岩洞示例视频的用户完整播放结论。
- `p01_true_arc_5deg_manual_review.json`：P01 受控 5° `true_arc` 视频的用户完整播放结论，`overall_pass=false`。
- `p01_center_gaussian_density_diagnostic_v1.json`：官方150万 PLY 的中心投影密度、1σ足迹尺度、深度与高频保留诊断。
- `p01_sample_budget_comparison_v1.json`：75万/150万/200万只改采样预算的0°中心视角消融；结论为更多点改善覆盖，但150万后不改善远景细节。
- `p01_sample_budget_artifact_manifest_v1.json`：三档 PLY/中心帧的产出机、字节数和完整 SHA256；大型文件本体不进 Git。
- `official_style_scene_calibration_v1.json`：自有 P01 与官方仓库 ETH3D/ScanNet++ 示例在“水平横移+前后dolly+持续look-at”轨迹下的适用域校准，包含 60 帧 Alpha 统计、目观观察和论文协议边界。

大型输出只保存在产出机器 `linux5080` 的 `outputs/01_诊断对照/`。
