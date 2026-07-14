# MOCE 正式代码

`moce_cluster.py` 在不修改单图 demo 的情况下扩展多患者概念聚类：

1. 从文件名解析患者/检查编号；
2. 每个类别中每位患者随机保留一张图片；
3. 复用单图 demo 提取候选掩码和1280维特征；
4. 对非癌和早癌候选区域分别执行 K-Means；
5. 保存聚类模型、分配清单、簇统计和代表区域总览；
6. 对同图同簇保留距离簇中心最近的区域，计算概念级 S^R、S^E 和 S_h；
7. 按重要性依次加入或移除前5个概念，生成 SSC/SDC 结果。

运行：

```bash
python 程序/MOCE/正式代码/moce_cluster.py
```

初次默认每类10位患者、25个概念簇。确认流程后，可将脚本中的
`PATIENTS_PER_CLASS` 改为 `None` 扩大到全部患者。

每个类别的主要输出：

- `候选区域/`：用于查看和聚类的概念区域；
- `候选掩码/`：原生分辨率二值掩码；
- `cluster_assignments.csv`：候选区域的聚类编号；
- `concept_importance.csv`：各概念簇的 S_h 和全局排名；
- `concept_scores_per_image.csv`：每张图的 S^R、S^E 及排名；
- `ssc_sdc_summary.csv`：逐步加入/移除概念后的准确率和概率；
- `ssc_sdc_per_image.csv`：SSC/SDC 的逐图明细。

当前患者编号仍由旧文件名临时推断，且概念发现和 SSC/SDC 使用同一批抽样图片，
因此结果只用于验证流程。院方患者级数据和新模型就绪后，应在概念发现集上建立
K-Means 与 S_h 排名，再在独立验证集上执行 SSC/SDC。
