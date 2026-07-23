# 数据整理脚本

根目录只保留当前概念集主线和仍被复用的稳定预处理模块：

- `audit_curated_concept_dataset_v1.py`：原始医学生整理集审计。
- `preprocess_curated_concept_dataset_v1_1.py`：当前概念集全量裁剪。
- `build_curated_concept_core_v1.py`：限图、去重和严格/松弛匹配清单。
- `preprocess_second_batch_v1_1.py`：两套当前裁剪流程共同复用的v1.1核心模块。

`归档/`按第二批v1、第二批v1.1和匹配/去偏审计分类保存已完成阶段的脚本。
