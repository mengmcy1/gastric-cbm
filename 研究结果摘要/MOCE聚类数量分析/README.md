# MOCE K-Means聚类数量敏感性分析

## 范围

固定同一严格平衡概念集、模型权重、候选区域和随机种子，只改变K-Means聚类数量。
分析覆盖：

- K值：10、15、20、25、30、35、40、45、50；
- 模型：ResNet50、EfficientNet-B0；
- 类别：非癌、癌/高级别；
- 合计：36组。

## GitHub中保留的内容

- `跨K汇总/`：总体CSV、Excel及四张不含胃镜图像的定量汇总图。
- `逐K聚合表/`：每组的`cluster_summary.csv`、`concept_importance.csv`和
  `ssc_sdc_summary.csv`。
- Word说明：数据筛选原则、指标解释、临床审核流程及本轮自动量化初筛结论。
- `文件清单.json`：所有摘要文件的大小和SHA256。

## 未上传内容

以下内容可能包含患者标识、原始路径或胃镜图像，仅保留在服务器：

- `cluster_assignments.csv`、`concept_scores_per_image.csv`；
- `ssc_sdc_per_image.csv`、患者贡献和典型病例表；
- 候选区域、代表区域、扩展图、典型病例图和原图定位；
- K-Means模型、特征缓存及医生命名工作簿。

## 当前自动初筛

量化结果建议优先人工复核K=20、25、30，并以K=35作为过度拆分对照；K=25暂作为
共同首选候选值。该结论仍需医学生或医生完成概念一致性、可命名性、病灶相关性和
伪相关风险审核，不能视为临床验证结论。

安全摘要由以下命令重新生成：

```bash
python 程序/项目维护/正式代码/export_safe_moce_summary.py
```
