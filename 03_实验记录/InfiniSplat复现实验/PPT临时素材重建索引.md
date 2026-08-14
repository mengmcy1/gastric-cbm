# InfiniSplat PPT 临时素材重建索引

## 用途

根目录 `临时_PPT_InfiniSplat问题总结_20260814/` 按项目约定被 Git 忽略，只是 PPT 工作副本。2026-08-14 曾误删一次，并已从下列正式产物完整重建。今后不得把临时目录当作唯一证据源。

## 正式来源

- 官方岩洞与 P01 5°：`02_true_arc统一协议/outputs/01_诊断对照/`；
- 高斯预算与远景密度诊断：同目录下 `P01_sample_budget_*` 和 `P01_center_gaussian_density_*`；
- 三场景官方 Demo 校准：`P01_official_orbit_dolly_look_at_*`、`OFFICIAL_eth3d_orbit_dolly_look_at_*`、`OFFICIAL_scannetpp_orbit_dolly_look_at_*`；
- ETH3D 首组静态真实目标：`03_论文大基线协议复现/outputs/ETH3D_courtyard_DSC0313_to_DSC0317_real_pose_v1_linux5080_20260814_retry1/`；
- ETH3D 首组连续视频：`03_论文大基线协议复现/outputs/ETH3D_courtyard_DSC0313_to_DSC0317_pose_interpolation_video_v1_linux5080_20260814_retry1/`；
- 三组补充相机对：`03_论文大基线协议复现/outputs/ETH3D_courtyard_DSC0312_to_DSC0317_*` 与 `ETH3D_courtyard_DSC0321_to_DSC0320/0322_*`；
- 四组总览：`03_论文大基线协议复现/outputs/eth3d_courtyard_four_real_pairs_white_overview.jpg`。
- 跨场景筛选正式输出：`04_跨场景官方样例筛选/outputs/OFFICIAL_cross_scene_rgb_batch_v1_linux5080_20260814/`；PPT副本单独位于 `新增_跨场景候选_室内道路工业_20260814/`。

## 正式记录

- `02_true_arc统一协议/records/official_style_scene_calibration_v1.json`；
- `03_论文大基线协议复现/records/eth3d_courtyard_dsc0313_to_dsc0317_real_pose_v1.json`；
- `03_论文大基线协议复现/records/eth3d_courtyard_dsc0313_to_dsc0317_pose_interpolation_video_v1.json`；
- `03_论文大基线协议复现/records/eth3d_courtyard_four_real_pair_static_matrix_v1.json`。

## 重建校验

重建后至少应存在：12张基础图片、2段基础视频、三场景各5张关键帧与1段视频、首组23项真实大基线编号素材、三组补充相机对各9项素材，以及四组总览图。

正式 `outputs/` 若仍完整，不应重新执行 GPU 推理；直接复制可保持位级一致，并避免随机差异和无意义算力消耗。
