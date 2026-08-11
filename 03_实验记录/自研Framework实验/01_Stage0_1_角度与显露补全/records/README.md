# 自研 Framework 可同步记录

状态：**待另一台电脑补齐（2026-08-11）**。

本目录用于保存从本地大型 `outputs/` 提升出来、可以提交 Git 的小型正式证据。不要在这里保存 PLY、MP4、逐帧图像、Depth、Alpha、NPY、模型或日志。

## 当前 canonical 运行

`outputs/02_Stage1_true_arc/03_正式矩阵/P01_stage01_true_arc_formal_v4_fallback_20260807/`

该目录目前只存在于已经完成输出整理和用户人工评分的另一台电脑。当前电脑不要根据进度文档重造 JSON，也不要移动本机旧布局中的输出。

## 另一台电脑需要提升的文件

- `stage1_matrix_summary.json`：从 canonical 运行的 `formal_summary/` 原样复制；
- `stage1_matrix_metrics.csv`：从 canonical 运行的 `formal_summary/` 原样复制；
- `manual_review.json`：从已填写的 `manual_review_template.json` 原样复制并使用正式文件名；
- `artifact_manifest.json`：登记运行 ID、产出机器简称、项目 Git commit、配置相对路径、输入/模型/关键产物的字节数与完整 SHA256、原始相对路径和同步层级。

若源目录中的真实文件名与上面不同，保留源文件内容与实际含义，在本 README 中记录映射；不要为了迎合清单伪造不存在的结果。

补齐后把本状态改为“已同步”，写明日期和对应 Git commit。大型输出继续留在原电脑。
