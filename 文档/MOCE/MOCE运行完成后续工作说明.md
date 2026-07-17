# MOCE运行完成后续工作说明

> 用途：记录本轮MOCE任务完成后的固定处理顺序，避免遗漏或误操作。
>
> 数据批次：第二批按患者整理数据
>
> 模型：ResNet50（主要模型）、EfficientNet-B0（对照模型）

## 一、先确认MOCE任务真正完成

不要仅根据后台进程消失判断任务完成。每个模型的 `class_0` 和 `class_1` 均应包含：

```text
candidate_features.npz
cluster_assignments.csv
cluster_summary.csv
concept_clusters.png
concept_importance.csv
concept_scores_per_image.csv
kmeans_model.joblib
ssc_sdc_per_image.csv
ssc_sdc_summary.csv
```

日志末尾应出现：

```text
候选区域总数
平均每簇患者数
最高重要性概念簇
SSC/SDC评估步数
输出目录
```

同时确认日志中没有 `Traceback`、`CUDA out of memory`、`Killed` 等错误。

## 二、生成清晰版概念聚类图

现有脚本：

```text
程序/MOCE/正式代码/render_cluster_overview.py
```

该脚本只读取已有聚类结果重新排版，不重新提取特征、不重新执行K-Means，也不覆盖原始 `concept_clusters.png`。

EfficientNet-B0完成后运行：

```bash
cd /home/mcy/gastric-cbm

/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
程序/MOCE/正式代码/render_cluster_overview.py \
--model efficientnet_b0 \
--class-label all
```

ResNet50完成后运行：

```bash
cd /home/mcy/gastric-cbm

/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
程序/MOCE/正式代码/render_cluster_overview.py \
--model resnet50 \
--class-label all
```

两个模型均完成后也可一次运行：

```bash
cd /home/mcy/gastric-cbm

/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
程序/MOCE/正式代码/render_cluster_overview.py
```

输出目录：

```text
结果/MOCE聚类/第二批/{model}/class_0/概念聚类清晰版/
结果/MOCE聚类/第二批/{model}/class_1/概念聚类清晰版/
```

清晰版采用宋体风格字体，每页展示3个概念簇，每簇展示5张300×250代表图。展示编号为1～25；原始CSV中的 `cluster_id` 仍为0～24。

生成后检查：

- 图片和文字是否清晰；
- 是否仍有文字重叠；
- 每个概念簇是否有5张代表图；
- 概念编号是否为1～25；
- 两个模型的两个类别是否都生成；
- 原始 `concept_clusters.png` 是否保留。

## 三、生成自动分析结果

现有脚本：

```text
程序/MOCE/正式代码/analyze_moce_results.py
```

该脚本已在EfficientNet-B0的class_0和class_1正式结果上运行并验证。ResNet50完成后可直接复用。

运行：

```bash
cd /home/mcy/gastric-cbm

/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
程序/MOCE/正式代码/analyze_moce_results.py
```

输出：

```text
analysis_summary.xlsx
important_clusters.csv
patient_contribution.csv
cluster_patient_dominance.csv
cluster_source_composition.csv
typical_cases.csv
ssc_sdc_accuracy_curve.png
ssc_sdc_probability_curve.png
cluster_quality.png
医生概念命名表.xlsx
典型案例图/
重要概念扩展图/
```

主要分析内容：

1. 汇总模型、类别、患者数、候选区域数、随机种子和 `PATIENTS_PER_CLASS`；
2. 统计每位患者贡献的候选向量数量；
3. 计算每个概念簇的单患者最大贡献比例和Top 5患者合计比例；
4. 将 `cluster_assignments.csv` 与 `dataset_manifest.csv` 关联，统计医院来源构成；
5. 综合S_h、重要性排名、患者覆盖和相对类别基线的来源富集程度筛选重要概念；
6. 筛选概率下降明显、单独保留概率较高和概率下降为负的典型/异常案例；
7. 绐制SSC/SDC准确率曲线和平均目标概率曲线；
8. 生成医生可填写的医学概念命名表。

检查原则：

- 自动筛选结果只是审核优先级，不是医学结论；
- 高重要性概念仍需检查患者覆盖和医院来源；
- 单患者贡献偏高时应回看原图；医院来源必须与该类别整体来源基线比较；
- `concept_number = cluster_id + 1`，所有医生材料统一使用1～25编号。

## 四、分开保存自动分析结果与MOCE原始结果

不再额外建立整理目录，也不需要运行新的整理脚本。第三步中的
`analyze_moce_results.py` 会直接在原有 `结果/MOCE分析` 内生成以下结构：

```text
结果/MOCE分析/{model}/class_{label}/
├── 01_自动分析结果/
└── 02_MOCE原始结果/
```

`01_自动分析结果`包括汇总Excel、曲线、典型案例图、重要概念扩展图、医生概念命名表
和清晰版概念聚类图。`02_MOCE原始结果`包括以下可追溯表格和原始总览图：

```text
concept_clusters.png
cluster_assignments.csv
cluster_summary.csv
concept_importance.csv
concept_scores_per_image.csv
ssc_sdc_summary.csv
ssc_sdc_per_image.csv
```

`candidate_features.npz`、`kmeans_model.joblib`、全部候选区域和全部候选掩码体积较大，
并非医学生常规分析材料，因此不重复复制；它们仍保留在正式聚类目录中。

分析目录不额外放置重复的文件夹说明，统一使用：

```text
文档/MOCE/MOCE聚类最终结果分析指南_医学生版.docx
```

阅读顺序为：

```text
自动分析结果
→ 医生填写概念命名表
→ 按需追溯MOCE原始结果
```

当前分析目录仍包含患者编号和原始图像信息，不能直接发送给医学生。医生需要审核时，
再单独建立一个经过匿名化的最终提交文件夹；平时不再维护重复的中间整理目录。

## 五、材料发送前匿名化

医生材料包不应直接包含患者身份信息。

发送前处理：

- 对逐图CSV中的 `patient_id` 进行匿名化；
- 对图像文件名中可能包含的姓名、检查号进行匿名化；
- 匿名编号与原患者编号映射表单独保存在内部，不放入医生材料包；
- 确认代表图和原图中没有可见姓名、检查号等身份信息；
- 通过院方允许的安全渠道发送。

不发送：

```text
candidate_features.npz
kmeans_model.joblib
全部候选区域/
全部候选掩码/
运行日志
incomplete目录
模型权重
患者身份映射表
```

逐图原始明细如 `cluster_assignments.csv`、`concept_scores_per_image.csv` 和 `ssc_sdc_per_image.csv` 应匿名化后再决定是否加入材料包。

## 六、医生审核任务

医生主要完成：

1. 给重要概念簇进行医学命名；
2. 描述客观形态，如发红、凹陷、表面不规则、边界改变等；
3. 判断概念是否与病灶相关；
4. 标记正常结构、疑似伪相关概念和杂乱簇；
5. 判断同一概念簇内部是否一致；
6. 审核概率下降明显和概率下降为负的案例；
7. 填写可信度和备注；
8. 允许保留“暂无法命名”，不强行赋予医学含义。

分析优先级：

- ResNet50 `class_1`：主要医学概念分析；
- ResNet50 `class_0`：非癌对照和偏差检查；
- EfficientNet-B0 `class_1`：跨模型概念对照；
- EfficientNet-B0 `class_0`：补充对照。

## 七、独立验证

医生命名和内部分析完成后，再实现独立验证脚本：

```text
程序/MOCE/正式代码/validate_moce.py
```

验证流程：

```text
在概念发现集建立K-Means和概念排名
→ 固定聚类中心与重要性排名
→ 测试图片提取候选向量
→ 最近邻匹配到已有概念簇
→ 不重新训练K-Means
→ 在独立验证集重新计算SSC/SDC
```

当前正式脚本在概念发现所用图片上计算SSC/SDC，属于内部忠实性分析。论文中的正式结论应补充独立验证结果。

## 八、固定工作顺序

```text
确认两个MOCE任务完整结束
→ 生成清晰版概念图
→ 检查字体、编号和版式
→ 编写并运行自动分析脚本
→ 检查患者主导、医院来源和重要概念
→ 生成SSC/SDC曲线和典型案例
→ 建立医生审核材料包
→ 完成匿名化检查
→ 医生进行命名和医学审核
→ 汇总医生意见
→ 在独立验证集执行固定概念验证
→ 整理论文图表和结果
```

在每一步确认完成前，不提前删除正式结果或重新运行聚类。
