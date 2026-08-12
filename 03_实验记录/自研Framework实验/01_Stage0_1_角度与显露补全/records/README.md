# 自研 Framework 可同步记录

状态：**`win5060` canonical v4 证据与 `linux5080` MAT 静态候选证据均已补齐（2026-08-12）**。

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
