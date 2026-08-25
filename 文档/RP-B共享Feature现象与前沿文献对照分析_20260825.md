# RP-B 大量共享 Feature 现象与前沿文献对照分析（2026-08-25）

> 本文用于解释 `gastric-cbm` 当前 RP-B 中出现的大量癌/非癌共享 SAE Feature，并与项目此前参考的工作及截至 2026-08-25 的视觉/病理 SAE 前沿研究进行对照。
>
> 重要边界：本文把三类证据严格分开——**本项目实证结果**、**文献明确报告的现象**、**基于两者提出的机制假设**。文献中没有直接报告的结论，不替作者补充。

---

## 1. 本项目当前现象

当前 RP-B 使用 RP-A development 得到的 1150 个 strict 3-seed Anchor，针对 seed42/43/44 三套 SAE 字典完成 train-only 技术属性分析。

冻结 v1 结果为：

| 项目 | 数量 |
|---|---:|
| strict Anchor | 1150 |
| `shared_high` | 331 |
| `cancer_enriched` | 1 |
| `noncancer_enriched` | 0 |
| `mixed_uncertain` | 818 |
| `source_risk` | 198 |
| technical-family edge | 0 |
| technical family | 1150 个 singleton |

进一步诊断显示，818 个 `mixed_uncertain` 中：

- 806 个为 `coverage_equivalent_mass_not_equivalent`；
- 12 个为 `within_seed_joint_criteria_unresolved`。

因此约 98.5% 的 mixed Anchor 并不是“癌和非癌一边有、一边没有”，而是**患者级 presence 已经接近等价，但 activation mass 尚未等价**。

同时，本项目 1150 个 Anchor 的患者覆盖率整体极高：中位数为 1.0，5% 分位数约 0.9979。结合当前表示方式——每张图 49 个 patch、Matryoshka patch SAE 使用 `K_coverage=1024`、患者 presence 采用“任意图像任意位置出现过即为 active”的 OR 语义——患者级 presence 很容易出现天花板效应。

因此 RP-B 最重要的认知变化不是“只找到 1 个癌 Feature”，而是：

> **C-long 中跨 SAE 初始化稳定的表示，更像一套癌/非癌共同使用的胃镜视觉 vocabulary；类别信息可能主要编码在共享 Feature 的激活强度、空间组织、组合关系和模型决策依赖中，而不是编码为大量癌专属 Feature。**

这句话目前仍是机制假设，需要 RP-C residual-preserving intervention 进一步验证。

项目内部依据：

- `文档/RP-B技术分析_20260825.md`
- `结果/SAE/RP_B_Technical_20260825/`
- `SAE实验进度与结果讨论.md`
- `rpb_technical_protocol_v1.json`

---

## 2. 先区分三种不同的“共享”

后续讨论中必须区分三种概念，否则很容易把完全不同的问题混在一起。

### 2.1 跨类别共享（class-shared）

同一个 SAE Feature 在癌和非癌中都出现或具有相近总体激活。

这是 RP-B `shared_high` / `mixed_uncertain` 主要研究的问题。

### 2.2 跨 SAE 初始化共享（cross-seed reproducibility）

seed42 的某个 Feature 是否能在 seed43/44 中找到对应方向。

这是 RP-A / RP-A-lite 研究的问题，最终形成 1150 个 strict Anchor。

### 2.3 多 Feature 共享同一语义（semantic duplication / feature splitting / descriptive collision）

两个技术上不同的 SAE Feature 最终可能被专家命名为类似医学现象。

RP-B 的 technical family=0 只能说明：在冻结的 decoder/behavior/spatial 严格规则下，没有足够证据把两个 Anchor 当作同一个**技术解释单元**。它不能证明未来一定存在 1150 个互不重复的医学概念。

---

## 3. 文献总览：是否遇到类似“共享 Feature”现象

| 工作 | 场景 | 是否直接涉及共享/跨类复用 | 作者如何处理 | 与本项目关系 |
|---|---|---|---|---|
| **Histoscope (2026)** | 结直肠病理 FM + TopK SAE | **是，最接近本项目**。很多 Feature 跨多个 diagnostic classes 出现，但病理专家仍认为其 morphology 单义 | 用专家盲审区分“跨类别出现”和“polysemantic” | 直接支持 `class-shared != uninterpretable` |
| **PICASSO (2026)** | 32 癌种 pan-cancer pathology SAE | **是**。报告 cancer-type-specific signatures，同时存在跨癌种反复出现的 morphology | 建 concept activation profile，并做 concept intervention、artifact suppression | 支持“共享 vocabulary + 不同 activation profile” |
| **PatchSAE (ICLR 2025)** | CLIP ViT patch SAE | 不直接统计癌/非癌式 shared ratio，但明确发现适配增益多数来自**已有概念的重新映射**，不是大量新概念 | patch attribution + latent ablation + class behavior | 支持“类别/任务差异可来自现有概念使用方式变化” |
| **ProtoMIL (MICCAI 2025)** | WSI tumor/normal | 使用同一 SAE 字典和 normal/tumor probing patches，但没有系统报告 shared ratio | Top patches + pathologist review + concept contribution + intervention | 支持联合字典，不要求 Feature 必须 class-exclusive |
| **M-CBM (ICLR 2026)** | SAE → named concepts → CBM | 不以 shared ratio 为研究终点 | 从黑箱自身 Feature 建统一 concept bottleneck，再用稀疏分类器学习 concept→class 关系 | 支持“概念可以被多个类别共同使用，类别差异由下游权重决定”的建模方式 |
| **Zhao et al. FM-SAE (2026 preprint)** | TB/肺癌等病理 | 核心是**compositional overlap**：一个 patch 同时包含多个 histomorphologies | 多 Feature 分解、overlap clustering、专家 group、多尺度图 | 支持医学图像天然是组合式，不应期待单一 class-exclusive 原子概念 |
| **H&E gene-expression concept explanation (2026)** | 结直肠癌 H&E → 空间转录组 | 不以 class sharing 为中心 | 比较 activation-derived 与 relevance-derived concepts，发现 relevance 更直接连接 downstream prediction | 强烈支持 RP-C：activation/mass 不等于 decision relevance |
| **Vision SAE monosemanticity (NeurIPS 2025)** | CLIP/VLM SAE | 概念本质上是跨数据点共享属性，不要求绑定单一类别 | 用 top activating images 衡量 monosemanticity并做 feature steering | 支持“单义性”和“类别专属性”是不同维度 |
| **SAEs Do Not Find Canonical Units (ICLR 2025)** | LLM SAE | 不属于视觉跨类共享实证 | 证明 SAE Feature 未必是唯一、完整、原子的 canonical units | 提醒不要把 1150 Anchor 直接解释成 1150 医学原子概念 |
| **Descriptive Collision (2026)** | LLM SAE | 多 Feature 可获得同一描述 | 提出 discrimination/collision 风险 | 提醒 technical family=0 仍不排除未来医学命名重复；仅作方法学风险依据 |

---

## 4. Histoscope：与当前现象最接近的直接证据

**Histoscope: Expert-Grounded Inspection of Sparse Autoencoder Features in Histopathology Foundation Models** 是目前与本项目“类别共享”问题最直接的对照之一。

其主要设置为：在结直肠病理 foundation-model embedding 上训练 TopK SAE，并按 diagnostic-class selectivity 对 Feature 做自动筛查，再交给病理专家盲审。

公开结果报告：

- 病理专家总体认为约 82% 的 Feature panel 是 monosemantic；
- 系统判为 monosemantic 的 50 个 Feature，专家全部确认；
- 系统判为 polysemantic 的 38 个 Feature 中，有 32 个仍被专家判断为 monosemantic；
- 一个主要原因是：**同一种 morphology 会跨多个 diagnostic classes 出现。**

这说明一个非常重要的问题：

> **低 class selectivity 不等于 Feature 有多个语义。一个 Feature 完全可能只表示一种稳定形态，只是这种形态并不是某个疾病类别专属。**

这与本项目 `shared_high=331` 非常契合。

在胃镜场景中，未来完全可能出现类似的情况，例如某个 Anchor 稳定表示：

- 黏膜皱襞；
- 血管；
- 表面纹理；
- 腺体/黏膜结构；
- 黏液；
- 反光；
- 某种颜色或成像模式。

这些视觉模式本来就可能同时存在于癌和非癌图像中。因此不能因为 Feature 跨类出现，就认为其 SAE 分解失败或 Feature 不可解释。

参考：

- Histoscope workshop page: https://mechinterpworkshop.com/posters/virtual/

---

## 5. PICASSO：更接近“共享视觉词典 + 不同疾病 activation profile”

**PICASSO: Dissecting and directing pathology foundation models** 在超过 1.2 亿 pathology patches、32 个癌种上训练 SAE，建立 pan-cancer morphology concept atlas。

论文的 pan-cancer profiling 明确同时观察到两类现象：

1. 某些 concept 对特定 cancer type 明显富集；
2. 另一些 morphology 会在多个癌种中反复出现，例如部分 stroma、tumor-cell sheets、necrosis/tissue-crack 等模式。

因此 PICASSO 的疾病表征不是：

```text
Cancer A = A专属概念集合
Cancer B = B专属概念集合
```

而更接近：

```text
共享 morphology vocabulary
        +
不同癌种的 concept activation profile
```

随后 PICASSO 不是依据“是否跨类共享”删除 concept，而是继续：

- 测量 concept activation；
- 连接下游模型贡献；
- selective concept suppression / amplification；
- 检查 tumor probability 如何变化；
- 单独识别 tissue folding 等 artifact concepts 并干预。

其结果还显示，抑制少量高影响概念能够显著改变 tumor/normal prediction，而随机 concept removal 不能提供同样的信息。

这与本项目下一阶段 RP-C 非常接近：

> `sharedness/mass` 只是观察性属性；真正的 decision relevance 需要通过 Feature intervention 来确认。

参考：

- PICASSO PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC13307983/

---

## 6. PatchSAE：任务差异可以来自“已有概念的重新使用”，而不是大量新类别专属概念

**Sparse Autoencoders Reveal Selective Remapping of Visual Concepts During Adaptation (PatchSAE, ICLR 2025)** 在 CLIP ViT 中训练 patch-level SAE，能够得到局部概念及 patch-wise spatial attribution。

这篇论文并没有像本项目一样给出 `shared_high / cancer_enriched` 的明确类别共享比例，因此不能把它写成“PatchSAE 也发现了 30% shared Feature”。

但它报告了一个非常相关的现象：

> 在多个 downstream adaptation task 中，大部分适配收益可以由 foundation model 中**已经存在的概念**解释，而不需要假设模型主要通过产生大量全新概念完成适配。

作者进一步通过 SAE latent ablation 检查这些概念对分类行为的影响。

这对本项目的启发是：

> 癌/非癌分类差异完全可能主要来自“对共同视觉概念的重新映射、重新加权或不同空间使用”，而不是来自两套互不重叠的概念字典。

参考：

- ICLR 2025 paper: https://openreview.net/forum?id=imT03YXlG2
- arXiv: https://arxiv.org/abs/2412.05276

---

## 7. ProtoMIL：联合 tumor/normal 字典，但不把 shared ratio 作为主要研究问题

**Prototype-Based Multiple Instance Learning for Gigapixel Whole Slide Image Classification (ProtoMIL, MICCAI 2025)** 是本项目此前重点参考的医学 SAE 工作。

ProtoMIL 在同一个 SAE 中学习 pathology concepts，再通过 probing patches 进行解释。项目此前文献笔记记录：其 probing set 同时包含 normal 与 tumor patches，医生查看每个 Feature 的 Top patches，并识别病理结构、组织边界、失焦、墨水等视觉模式。

ProtoMIL 没有像本项目 RP-B 一样系统地把所有 Feature 分类为：

- cancer enriched；
- non-cancer enriched；
- shared；
- mixed。

因此不能拿 ProtoMIL 的结果直接回答“共享比例是否和我们的 331/1150 相同”。

但其方法学选择非常重要：

- 使用**统一 SAE 字典**而非 tumor SAE + normal SAE；
- 允许一个 concept 对不同类别均有激活；
- 下游 ProtoMIL 再利用 concept activation、区域聚合/注意力及分类权重形成预测；
- 对病理专家判断为无关或伪影的 Feature 做 intervention。

这意味着 ProtoMIL 的解释范式本身也不要求“一个好 Feature 必须属于唯一类别”。

参考：

- MICCAI 2025: https://papers.miccai.org/miccai-2025/paper/0542_paper.pdf
- arXiv: https://arxiv.org/abs/2503.08384

---

## 8. M-CBM：concept 与 class 的关系由下游模型学习，不要求概念 class-exclusive

**Learning Concept Bottleneck Models from Mechanistic Explanations (M-CBM, ICLR 2026)** 从黑箱模型内部用 SAE 提取候选 Feature，再命名、标注概念 presence，最终构建 Concept Bottleneck Model。

它的关键思想不是先人为规定一组“某类别专属概念”，而是：

```text
black-box representation
        ↓
SAE discovered concepts
        ↓
concept naming / presence annotation
        ↓
concept predictor
        ↓
sparse concept-to-class classifier
```

因此概念本身和类别预测是两个层级：

> 一个 concept 可以在多个类别中存在；真正决定分类的是 concept activation/presence 与下游 classifier weight 的组合。

M-CBM 本身并不以“有多少 concept 跨类别共享”为主要定量结果，因此它不能直接为本项目 331 个 shared-high 提供比例参照；但它强烈支持本项目不删除 shared Feature，而是在后续医学标注后让 M-CBM 学习哪些 concept 真正具有分类作用。

参考：

- ICLR 2026 / OpenReview paper
- arXiv: https://arxiv.org/abs/2603.07343

---

## 9. Zhao et al. 2026：病理形态本身就是 compositional，而不是一 patch 一概念

**Compositional and interpretable representation of histology using AI foundation models and sparse autoencoders**（2026 preprint）直接指出，传统 clustering 在病理解释中的一个根本问题是：

> 一个 image patch 往往同时包含多种 histopathologies。

例如一个区域可能同时含有免疫细胞、纤维化、血管或其他组织结构。把一个 patch 强行分到唯一 cluster，本身就不能完整描述真实形态。

作者因此使用 SAE，让单个 patch 表示为多个 Feature 的稀疏组合，并通过：

- continuous feature score；
- token-level saliency；
- spatial overlap；
- hierarchical clustering；
- 专家 grouping；
- multi-feature maps；

构建组合式组织形态解释。

这项工作虽然不是癌/非癌 sharedness 研究，但它解释了为什么在医学图像中，不应期待 Feature 天然形成互斥的疾病标签：

> **真实组织形态本身就是由大量共享、重叠、可组合视觉成分构成。**

参考：

- PMC preprint: https://pmc.ncbi.nlm.nih.gov/articles/PMC13252107/

---

## 10. 2026 H&E → gene-expression 工作：activation 强并不等于 downstream relevance 强

2026 年 8 月公开的 **Concept-based explanation of gene expression prediction from H&E images** 将 TopK SAE concept discovery 与 relevance propagation 结合，用于解释 H&E 图像到空间转录组预测。

作者明确比较 activation-derived concepts 与 relevance-derived concepts，并报告 relevance 与 downstream prediction 的连接更直接。

这对本项目非常重要，因为 RP-B 当前已经发现：

- 大量 Feature 两类都出现；
- 806 个 mixed 的主要问题是 mass 不等价。

但即使一个 Feature 的 cancer mass 明显更高，也仍不能直接推出：

```text
activation高
→ 模型依赖高
```

因此 RP-C 必须单独测量：

```text
Feature intervention
→ delta margin
```

即把 **activation evidence** 与 **decision relevance evidence** 分开。

参考：

- arXiv: https://arxiv.org/abs/2608.16669

---

## 11. 视觉 SAE 的 monosemanticity 研究：单义性不等于类别专属性

**Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models**（NeurIPS 2025）研究 CLIP/VLM SAE，并通过 Top activating images 和 human study 衡量 Feature 是否 monosemantic。

论文中的“concept”本身就是跨多个输入样本共享的属性。一个 Feature 是否单义，核心是：

> 强激活它的样本是否体现一致的视觉语义。

这与“它是否只出现在一个类别”不是同一个问题。

例如一个稳定表示“parrot-like visual pattern”的 Feature 可以非常 monosemantic；类似地，一个稳定表示“血管”“皱襞”“反光”的胃镜 Feature 也可以非常单义，即使癌和非癌两类都出现。

因此未来医学审核不应把 `shared_high` 当作不可解释的负面标签。

参考：

- arXiv: https://arxiv.org/abs/2504.02821

---

## 12. 为什么我们的共享现象看起来尤其强？

文献对照之后，本项目“大量 shared / presence-equivalent Feature”可以由几类原因共同解释。

### 12.1 胃早癌 vs 非癌本来就高度共享基础视觉组成

这不是“猫 vs 飞机”式类别。

癌和非癌胃镜天然共享：

- 胃黏膜；
- 血管；
- 皱襞；
- 表面纹理；
- 色彩与照明；
- 反光；
- 黏液/液体；
- 胃腔和观察距离；
- 大量正常背景结构。

早癌的判别信息常常来自这些共同基础形态的异常变化、组合或空间关系，而不是出现一个非癌绝不会出现的新物体。

因此从生物学/视觉任务本身出发，共享 vocabulary 是合理的。

### 12.2 SAE 分解的是 C-long 的内部表示，不是在监督寻找“癌概念”

SAE 的目标是分解 C-long 已经学到的表示，而 C-long 为了完成癌症分类，也需要编码正常胃部结构、成像模式和上下文。

因此 SAE 自然会抽取大量 generic gastric visual primitives，而不是只抽取 label-specific features。

### 12.3 当前 patient presence 语义很容易饱和

本项目当前分析对象为：

```text
7×7 = 49 patches / image
Matryoshka dictionary width = 10240
K_coverage = 1024 / patch
多图像 / patient
patient presence = 任意 image、任意 patch 激活过即为 1
```

设单 patch 激活概率不算特别高，但一个患者拥有大量 patch 机会，则：

```text
patient ever-active probability
```

会随着图像数和 patch 数快速接近 1。

所以患者级 presence 更接近：

> “这个患者身上是否曾出现过这个视觉方向？”

而不是：

> “这个视觉方向是不是该患者最重要的诊断特征？”

这正好解释了为什么当前 coverage 中位数达到 1.0，而 activation mass 仍然能区分大量 mixed Anchor。

### 12.4 K=1024 的解释表示并不是极端稀疏的 patient-presence 设计

本项目每个 patch 最多保留 1024 个 Feature，用它是为了获取较充分的解释覆盖和跨 seed 稳定结构，而不是为了让 patient presence 变成极低概率事件。

因此大量患者级 shared presence 本身并不是反常结果。

### 12.5 分类信息可能编码在组合，而不是单一 Feature exclusivity

一个癌判断可能依赖：

```text
共享Feature A：某类色彩
+
共享Feature B：某类表面纹理
+
共享Feature C：某类血管模式
+
特定空间关系
+
不同激活强度
```

单独看 A/B/C，每一个都可能同时存在于癌和非癌。

真正类别特异的是它们的：

- activation magnitude；
- spatial arrangement；
- co-activation；
- attention coupling；
- classifier dependence。

这也是为什么不能把“只有 1 个 `cancer_enriched`”解释为“只有一个 Feature 与癌有关”。

---

## 13. 为什么 technical family=0 与文献并不矛盾

本项目 technical family 使用非常严格的技术标准，第一关就要求跨 Anchor decoder cosine ≥ 0.90，并同时约束 patient behavior、Top-patient overlap 和 spatial similarity。

实际不同 Anchor 的 decoder similarity 远低于门槛，因此没有 family edge。

这个结果只能支持：

> **没有证据表明两个 Anchor 是近乎同一个跨 seed 技术方向。**

但它不能支持：

> **未来医生一定会命名出 1150 个彼此完全不同的医学概念。**

这里需要注意两类来自语言模型 SAE 的方法学风险文献：

### Sparse Autoencoders Do Not Find Canonical Units of Analysis

该工作利用 SAE stitching 和 meta-SAE 说明 SAE Feature 未必组成唯一、完整且原子的 canonical units；更宽字典可能出现新 Feature，已有 Feature 也可能进一步被分解。

参考：https://arxiv.org/abs/2502.04878

### Descriptive Collision

该工作在人类标注的 LLM SAE 数据中发现，不同 Feature 可能被赋予相同或高度相似的文字解释，并把这一问题称为 descriptive collision。

参考：https://arxiv.org/abs/2605.12874

这两篇不是胃镜或视觉实证，不能用于声称本项目已经发生 feature splitting/collision。但它们提醒我们：

> technical distinctness 和 medical semantic distinctness 必须分开报告。

因此 RP-B 保留 1150 singleton 是正确的；未来若医生把多个 Anchor 命名为相似医学概念，也不应反向修改 RP-B technical-family 结果。

---

## 14. 对当前 RP-B 结果的最合理解释

综合本项目结果和文献，当前最稳妥的解释是：

### 可以说

1. C-long 内部存在大量跨 SAE 初始化稳定的稀疏技术方向。
2. 这些稳定方向多数并不表现为“癌有、非癌没有”的严格类别专属性。
3. 331 个 Anchor 满足严格 `shared_high` 规则。
4. 818 个 mixed 中有 806 个已经表现为 coverage equivalent，而主要不确定性来自 activation mass。
5. 198 个 Anchor 存在 source association，需要作为审计旗标保留。
6. 严格 technical-family 规则没有发现跨 Anchor 技术重复。
7. 当前现象与 Histoscope、PICASSO 等视觉/病理 SAE 工作中“共享 morphology / reusable concept vocabulary”的观察相容。

### 不能说

1. 不能说“只有一个癌相关 Feature”。
2. 不能说 331 个 shared-high 都是正常医学结构。
3. 不能说 818 个 mixed 是无用或不可解释 Feature。
4. 不能说 198 个 source-risk 都是伪特征。
5. 不能说 1150 singleton 就等于 1150 个医学概念。
6. 不能仅依据 activation mass 推断 Feature 被分类器实际使用。

---

## 15. 文献对 RP-C 的直接启示

多篇工作最终都从“Feature 激活”继续走向“Feature 对模型输出的作用”：

- PatchSAE：latent ablation / classification behavior；
- PICASSO：concept suppression/amplification + downstream output；
- ProtoMIL：concept contribution + human intervention；
- 2026 H&E gene-expression work：明确区分 activation 与 downstream relevance；
- Vision SAE steering：通过干预 SAE neuron 改变模型输出。

因此本项目 RP-C 的核心问题可以正式写成：

> **对于癌与非癌广泛共享的稳定 SAE Feature，类别信息究竟来自 Feature 的存在本身，还是来自其激活强度、空间组织及模型对该 Feature 的差异化功能依赖？**

RP-C1 的 residual-preserving `100% → 0%` effect screen 正好用于回答最后一部分：

```text
Feature j 激活
      ↓
仅删除其 SAE decoder component
      ↓
保留 SAE 未重构 residual
      ↓
重新计算 C-long attention / pooling / classifier
      ↓
delta margin（primary）
```

若后续出现如下情况：

```text
shared_high Anchor
Cancer:    删除后 margin 明显下降
NonCancer: 删除后变化很小或方向不同
```

则可以得到一个非常有价值的技术结果：

> **statistical sharedness 不等于 functional sharedness。**

这会把本项目从“哪些 Feature 与癌相关”推进到“C-long 如何差异化使用共同视觉基元完成癌症判断”。

---

## 16. 当前最值得关注的 RP-C 候选类型

后续不应只优先看唯一 `cancer_enriched`。至少应同时关注：

1. `shared_high` 且 RP-C cancer effect 大的 Anchor；
2. `coverage_equivalent_mass_not_equivalent` 且 RP-C effect 与 mass 方向一致的 Anchor；
3. mass 差异明显但 RP-C effect 很小的 Anchor——用于证明“相关不等于依赖”；
4. `source_risk` 且 RP-C effect 大的 Anchor——这是最值得审计的潜在 shortcut；
5. sharedness 六层敏感性发生变化的 10 个边界 Anchor；
6. 唯一 `cancer_enriched` 的 `a00987`，作为 cancer-enrichment + source-risk sentinel；
7. RP-C effect 极低的稳定 Anchor，作为技术负对照。

---

## 17. 最终总结

当前 RP-B 的“大量共享 Feature”不是需要被修掉的异常，而更可能是一个值得继续验证的核心现象：

> **胃早癌分类模型可能不是依靠一套“癌专属视觉词典”工作，而是在大量癌/非癌共同存在的视觉基元上，通过激活强度、空间布局、组合方式和决策依赖的差异完成分类。**

这一解释与当前病理/视觉 SAE 文献中的多项观察相容：

- Histoscope 证明跨 diagnostic class 的 morphology 仍可以保持 monosemantic；
- PICASSO 展示 pan-cancer shared concepts 与 cancer-specific activation fingerprints 可以同时存在；
- PatchSAE 表明 downstream adaptation 很大程度可由已有概念的重新映射解释；
- ProtoMIL/M-CBM 都采用统一概念空间，再由下游预测关系区分类别；
- 2026 pathology relevance 工作进一步提示 activation 不能替代 decision relevance。

但目前仍不能把它写成 C-long 的已证实机制。真正的关键验证是 RP-C：

> **当我们在保留 residual 的前提下逐一删除稳定 Anchor 时，哪些共享视觉 Feature 会稳定、特异地改变癌/非癌 margin？**

如果该结果成立，那么“大量共享 Feature”将不是 RP-B 的尴尬结果，而会成为后续机制解释的核心：

```text
shared visual vocabulary
        +
class-differential activation / spatial organization
        +
class-differential decision dependence
        =
C-long cancer discrimination mechanism（待RP-C验证）
```

---

## 参考文献与链接

1. Lim H, Choi J, Choo J, Schneider S. **Sparse Autoencoders Reveal Selective Remapping of Visual Concepts During Adaptation (PatchSAE)**. ICLR 2025.  
   https://openreview.net/forum?id=imT03YXlG2  
   https://arxiv.org/abs/2412.05276

2. **Histoscope: Expert-Grounded Inspection of Sparse Autoencoder Features in Histopathology Foundation Models**. ICML 2026 Mechanistic Interpretability Workshop.  
   https://mechinterpworkshop.com/posters/virtual/

3. **Dissecting and directing pathology foundation models (PICASSO)**. 2026.  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC13307983/

4. Sun S, van Midden D, Litjens G, Baumgartner CF. **Prototype-Based Multiple Instance Learning for Gigapixel Whole Slide Image Classification (ProtoMIL)**. MICCAI 2025.  
   https://papers.miccai.org/miccai-2025/paper/0542_paper.pdf  
   https://arxiv.org/abs/2503.08384

5. De Santis A, Tong S, Brambilla M, Kagal L. **Learning Concept Bottleneck Models from Mechanistic Explanations (M-CBM)**. ICLR 2026.  
   https://arxiv.org/abs/2603.07343

6. Zhao Z et al. **Compositional and interpretable representation of histology using AI foundation models and sparse autoencoders**. 2026 preprint.  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC13252107/

7. Muench A et al. **Concept-based explanation of gene expression prediction from H&E images**. 2026.  
   https://arxiv.org/abs/2608.16669

8. Pach M, Karthik S, Bouniot Q, Belongie S, Akata Z. **Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models**. NeurIPS 2025.  
   https://arxiv.org/abs/2504.02821

9. Leask P et al. **Sparse Autoencoders Do Not Find Canonical Units of Analysis**. ICLR 2025.  
   https://arxiv.org/abs/2502.04878

10. McCann JF. **Descriptive Collision in Sparse Autoencoder Auto-Interpretability: When One Explanation Describes Many Features**. 2026.  
    https://arxiv.org/abs/2605.12874

---

## 文档状态

- 日期：2026-08-25
- 用途：RP-B → RP-C 技术解释与文献定位
- RP-B v1 结果：不修改
- val/internal test/external：本文不新增任何读取或结论
- 医学语义：未标注，本文不得被用作医学命名依据
- 下一阶段：RP-C1 residual-preserving Anchor effect screening
