# MOCE 正式代码

`moce_cluster.py` 在不修改单图 demo 的情况下扩展多患者概念聚类：

1. 从第二批整理清单读取患者编号，每个类别中每位患者保留一张图片；
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

正式运行使用 `PATIENTS_PER_CLASS=None` 和25个概念簇。

每个类别的主要输出：

- `候选区域/`：用于查看和聚类的概念区域；
- `候选掩码/`：原生分辨率二值掩码；
- `cluster_assignments.csv`：候选区域的聚类编号；
- `concept_importance.csv`：各概念簇的 S_h 和全局排名；
- `concept_scores_per_image.csv`：每张图的 S^R、S^E 及排名；
- `ssc_sdc_summary.csv`：逐步加入/移除概念后的准确率和概率；
- `ssc_sdc_per_image.csv`：SSC/SDC 的逐图明细。

聚类完成后生成分页清晰版：

```bash
python 程序/MOCE/正式代码/render_cluster_overview.py
```

生成自动分析、曲线、典型案例图、重要概念扩展图和医生命名表：

```bash
python 程序/MOCE/正式代码/analyze_moce_results.py
```

自动分析使用 `matplotlib` 和 `openpyxl`，输出位于：

```text
结果/MOCE分析/{model}/class_{label}/
├── 01_自动分析结果/
└── 02_MOCE原始结果/
```

`01_自动分析结果`保存统计表、曲线、案例图、扩展图、清晰版聚类图和医生命名表；
`02_MOCE原始结果`保存分析时需要追溯的MOCE原始CSV与总览图。正式聚类目录只会被读取，
不会被移动或覆盖。当前分析目录仍包含患者编号和原始图像信息，发送前必须匿名化。

当前概念发现和 SSC/SDC 使用同一批抽样图片，属于内部忠实性分析。正式论文还应
在概念发现集建立K-Means与S_h排名，再在独立验证集固定中心和排名执行SSC/SDC。
