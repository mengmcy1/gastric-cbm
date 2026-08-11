# 自研 Framework 可同步记录

状态：**已由 `win5060` 补齐（2026-08-11）**。

本目录用于保存从本地大型 `outputs/` 提升出来、可以提交 Git 的小型正式证据。不要在这里保存 PLY、MP4、逐帧图像、Depth、Alpha、NPY、模型或日志。

## 当前 canonical 运行

`outputs/02_Stage1_true_arc/03_正式矩阵/P01_stage01_true_arc_formal_v4_fallback_20260807/`

该目录当前存在于 `win5060`；`linux5080` 是否持有副本以其本地清单为准。缺少大型输出不改变这次运行已经完成的事实。

## 已同步的小型证据

- `stage1_matrix_summary.json`：从 canonical 输出归一化得到，删除不可迁移的旧绝对路径，并写明人工结论；
- `stage1_matrix_metrics.csv`：六个端点的正式自动指标；
- `manual_review.json`：用户完整播放 5°/15°/30° A/B 后的正式评分；
- `artifact_manifest.json`：登记配置、输入、六个视频、30°融合 PLY和源汇总的字节数与完整 SHA256。

运行时没有可靠记录项目 Git commit，因此 manifest 中保持 `null`，没有用后来的提交号代替。大型输出继续由 `.gitignore` 排除；需要在 `linux5080` 复核时按 manifest 选择性复制并校验。
