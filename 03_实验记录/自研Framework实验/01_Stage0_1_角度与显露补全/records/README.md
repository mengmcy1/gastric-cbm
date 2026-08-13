# 自研 Framework 可同步记录

状态：**`win5060` canonical v4、`linux5080` MAT/3D Photo/P01 重投影证据已补齐；Stage 1.3 CLB 合成链路已通过；Stage 1.4 Flash3D Provider 与 RealEstate10K 角度预检已通过，独立环境已建成，最小 GPU 推理暂阻塞于 RTX 5080/xFormers attention 兼容（2026-08-13）**。

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
- `stage1_4_flash3d_environment_v1.json`：记录 `linux5080` 独立环境、外部源码 commit、本地模型哈希、rasterizer 兼容补丁和首次 GPU 推理阻塞；其 `minimal_layer_smoke.status=blocked` 不得解读为 Provider 已通过。

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
