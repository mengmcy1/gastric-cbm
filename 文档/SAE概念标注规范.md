# SAE 候选 feature 命名与 M-CBM 概念标注规范

## 1. 文件用途

SAE 首先从 ResNet50 特征中发现未命名 feature。医生查看 Top 激活图片、非激活对照和
概念热图后提出候选名称，再对图片中的概念是否存在进行独立标注。只有完成概念级
标注后，才能训练 Concept Bottleneck Layer（CBL）。

SAE 激活不等于医学概念标签。不能把 `h_k>0` 直接转换成“概念存在”，也不能仅根据
癌/非癌标签推导概念标签。

## 2. 推荐目录

真实标注建议保存到新的版本目录，不覆盖旧版本：

```text
结果/SAE概念标注/第二批/resnet50/v1/
├── concept_catalog.csv
├── annotations_doctor_a.csv
├── annotations_doctor_b.csv
├── annotations_consensus.csv
└── annotation_log.md
```

模板位于：

```text
文档/SAE标注模板/concept_catalog_template.csv
文档/SAE标注模板/concept_annotations_template.csv
```

## 3. 概念目录 `concept_catalog.csv`

每行定义一个概念，字段如下：

| 字段 | 含义 |
|---|---|
| `concept_id` | 稳定编号，如 `C001`；名称修改时编号不变 |
| `concept_name` | 医生最终确认的简短名称 |
| `source_feature_ids` | 产生该概念的 SAE feature ID，多个用分号分隔 |
| `concept_group` | 病灶、正常结构、图像质量、伪影、域风格或不明确 |
| `definition` | 一句话定义概念指什么 |
| `positive_criteria` | 标为1必须满足的可见条件 |
| `negative_criteria` | 标为0的条件 |
| `exclusion_criteria` | 容易混淆但必须排除的表现 |
| `use_for_cbl` | `1` 纳入CBL，`0` 暂不纳入 |
| `version` | 概念定义版本，例如 `v1` |

在正式标注前锁定概念定义。定义发生实质变化时应升级版本，不要直接覆盖旧目录。

## 4. 图片概念标注 `annotations_*.csv`

采用长表格式：一行表示“一张图片的一个概念”。核心字段如下：

| 字段 | 含义 |
|---|---|
| `image_path` | 相对 `数据/第二批整理后` 的路径，必须与特征缓存中的 `图片名字` 一致 |
| `patient_id` | 患者ID，用于审计和患者级抽样 |
| `split` | `train`、`val` 或 `test`，必须沿用固定患者划分 |
| `concept_id` | 对应概念目录中的稳定编号 |
| `concept_label` | `1`存在，`0`不存在，`-1`无法判断/未标注 |
| `annotator` | 标注医生或标注者编号 |
| `annotation_round` | 标注轮次，例如 `independent_1`、`consensus_1` |
| `confidence` | 建议1～3：1低、2中、3高 |
| `notes` | 混淆原因、排除依据或其他备注 |

原始医生文件可以对同一图片—概念保留多行，但输入 CBL 的
`annotations_consensus.csv` 必须保证每个 `image_path + concept_id` 只有一行。

## 5. 标注顺序

1. 根据正式 SAE 的 train/val overview 筛选稳定 feature。
2. 医生在隐藏癌标签、来源和预测结果时先描述共同视觉模式。
3. 揭示来源和标签统计，判断其可能是病灶、伪影还是域风格。
4. 为候选概念写出阳性、阴性和排除标准。
5. 先用少量训练图片做试标，修正有歧义的定义。
6. 两位医生独立标注正式训练/验证样本。
7. 计算一致性并对分歧进行共识判定。
8. 锁定概念目录和共识标注后训练 CBL。
9. 测试集标注只用于最终概念识别评价，不能回头修改概念或模型。

## 6. 图片抽样原则

每个概念的标注集应包含：

- SAE 高激活图片；
- SAE 不激活图片；
- 外观相似但 feature 不激活的困难负样本；
- 多位不同患者，限制单一患者图片占比；
- 癌和非癌图片；
- 省人民和外院图片；
- train 和独立 val 患者。

不能只标高激活癌图，否则 CBL 可能把癌标签、医院来源或数据批次直接编码为概念。

建议先从3～5个定义清晰的概念开始试标，不需要一次给全部图片和全部 feature 标注。
CBL 支持部分标注：`-1` 位置不参与概念 BCE。

## 7. 训练与数据隔离

```text
train概念标注 → 训练CBL
val概念标注   → CBL早停、概念性能和分类器选择
test概念标注  → 方案锁定后的最终评价
```

患者不得跨 split。SAE 可以使用全部 train 图片；CBL 只使用 train 中已有概念标注的
图片；CBL 冻结后，最终稀疏癌/非癌分类器可以使用全部 train 图片，因为每张图片都可
由 CBL 自动生成概念 logits。

## 8. 质量控制和报告

每个概念至少报告：

- train/val/test 标注图片数和患者数；
- 阳性和阴性患者数；
- 省人民和外院构成；
- 医生间一致率及 Cohen's kappa；
- 无法判断比例；
- CBL 概念 AUC、Sensitivity、Specificity、Precision；
- 患者级概念稳定性；
- 概念概率校准情况；
- 最终分类器非零概念数和 NCC95。

概念概率表示“模型认为概念存在的概率”，不等同于病变严重程度。
