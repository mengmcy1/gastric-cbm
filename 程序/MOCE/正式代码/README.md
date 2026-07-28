# MOCE 正式代码

当前入口为`moce_curated_concept.py`，读取冻结的严格1:1图片平衡清单，使用v1.1裁剪图
和实验A重训练权重，分别运行ResNet50与EfficientNet-B0。

共享模块：

- `moce_core.py`：梯度通道评分、候选掩码提取、区域裁剪和模型变换。
- `moce_cluster.py`：候选编码、K-Means、S_R/S_E/S_h及SSC/SDC实现；由当前入口调用，
  不再作为当前数据的直接运行命令。
- `render_curated_cluster_overview.py`：为严格平衡结果生成分页清晰版概念总览。
- `analyze_curated_moce_results.py`：生成`结果/MOCE分析`中的自动分析、医生命名表和原始追溯层。

每个类别输出候选区域、候选掩码、聚类模型、分配清单、概念重要性、代表区域总览和
SSC/SDC逐图及汇总结果。S_R仅对正向概率下降归一化；负下降保留在CSV中用于异常分析。

## 已完成 Debug

两个模型均已用相同20个匹配对完成40张全链路debug：

- ResNet50：非癌1836个、癌2092个候选区域。
- EfficientNet-B0：非癌2966个、癌3004个候选区域。
- 两类均形成25簇并完成5步SSC/SDC，结果位于
  `结果/MOCE聚类/概念严格平衡_v1/debug/{model}/`。

## 全量运行

全量由用户运行，每个模型使用严格清单全部502张：

```bash
python 程序/MOCE/正式代码/moce_curated_concept.py \
  --model resnet50 --mode full

python 程序/MOCE/正式代码/moce_curated_concept.py \
  --model efficientnet_b0 --mode full
```

脚本保存冻结清单快照、清单SHA、权重SHA和运行参数。已有同模型同模式输出时默认拒绝
覆盖；确认重跑才添加`--overwrite`。

当前概念发现和SSC/SDC仍使用同一批图片，属于内部忠实性分析。论文验证阶段应固定
K-Means中心和重要性排名，在独立数据上做1-NN概念匹配和SSC/SDC，不重新聚类。

旧第二批MOCE脚本和结果分别位于`程序/归档/MOCE/`与`结果/归档/历史MOCE第二批/`。

## K-Means聚类数量敏感性分析

`run_kmeans_sensitivity.py`复用正式K=25运行已经冻结的候选特征、候选区域和掩码，
仅重新执行K-Means、概念重要性及SSC/SDC，不重复运行候选区域提取。结果统一保存到：

```text
结果/MOCE聚类数量分析/
├── 原始结果/k_XX/{model}/class_{0,1}/
└── 自动分析/
    ├── k_XX/{model}/class_{0,1}/
    ├── K敏感性汇总.csv
    ├── K敏感性汇总.xlsx
    └── {model}_class_{label}_K敏感性汇总.png
```

默认使用`K=10,15,20,25,30,35,40,45,50`。每个K均生成原始纵向大图、分页清晰大图、
前10重要概念扩展图、典型案例、医生概念命名表、质量图及SSC/SDC曲线。K=25直接复用
现有正式结果，其余K固定`random_state=42`并保持候选区域完全相同。

两模型、两类别、九个K共36组已全部完成。脱敏后的跨K汇总、逐K概念簇级CSV和说明
文档位于`研究结果摘要/MOCE聚类数量分析/`；患者级和图片级原始结果仅保留在服务器。

建议分别运行两个模型，脚本会自动跳过已经完成的模型、类别和K，并累积更新跨K汇总：

```bash
CUDA_VISIBLE_DEVICES=1 python 程序/MOCE/正式代码/run_kmeans_sensitivity.py \
  --model resnet50

CUDA_VISIBLE_DEVICES=1 python 程序/MOCE/正式代码/run_kmeans_sensitivity.py \
  --model efficientnet_b0
```

如确需逐整数扫描10至50，可添加：

```bash
--cluster-min 10 --cluster-max 50 --cluster-step 1
```

逐整数运行会生成41套完整图表，计算、磁盘和临床审核成本显著高于默认九点网格。
K值不应只按聚类惯性最小选择；应共同检查惯性下降是否进入平台、相邻K的ARI/NMI、
小患者簇数量、单患者主导比例、SSC/SDC、重要概念稳定性及医生可命名性。若多个K表现
接近，优先选择更小、更容易解释的K。

## 自动分析整理

EfficientNet-B0全量MOCE完成后运行：

```bash
python 程序/MOCE/正式代码/render_curated_cluster_overview.py \
  --model efficientnet_b0 --class-label all
python 程序/MOCE/正式代码/analyze_curated_moce_results.py \
  --model efficientnet_b0 --class-label all
```

输出沿用历史层级，位于`结果/MOCE分析/efficientnet_b0/class_{0,1}/`；每类包含
`01_自动分析结果`和`02_MOCE原始结果`。特征缓存、K-Means模型、全部候选区域和掩码
仍保留在正式聚类目录，不重复复制。
