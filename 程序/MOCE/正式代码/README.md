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
