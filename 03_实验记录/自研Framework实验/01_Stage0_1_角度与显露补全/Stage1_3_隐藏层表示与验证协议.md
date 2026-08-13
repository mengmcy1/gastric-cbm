# Stage 1.3 隐藏层表示与验证协议

> 状态：**v1 协议已冻结（2026-08-13）；`CLB-SYN-01` 受控合成表示能力实验已通过全部硬门。**
>
> 目标：先证明“已观测表面 + 遮挡后方层 + 中心视野外扩展层”能在 `true_arc` 30° 下被统一表示和渲染，再接入任何预测模型。

## 1. 为什么不再直接做端点二维补全

P01 Stage 1.2 已证明：30° 左右端点 accepted 区域合计 568,341 像素中，只有 627 像素能由中心 RGB-D 重投影获得，99.8897% 没有真实中心观测。Big-LaMa、MAT 和 3D Photo adapter 已分别产生涂抹、虚假建筑或两者兼有。

因此，不再把端点二维图像直接反投影为高斯。所有新内容必须先成为带有几何、来源和置信度的隐藏表面，经多视角重投影验证后才能进入 Gaussian Spawn。

## 2. 表示：Canonical Layer Bundle（CLB）

CLB 是面向当前工程的“多个带相机锚点的 RGB-D-α 表面块”集合，而不是固定两层或固定深度平面数。它吸收 LDI 的可变层数、SLIDE 的软可见性以及 TMPI 的局部深度复杂度思路，但不宣称复现其中任何方法。

### 2.1 三类表面块

| `support_type` | 含义 | 默认锚点 |
|---|---|---|
| `observed_surface` | 中心图真实可见表面 | `center` |
| `occlusion_hidden` | 同一中心射线上、前景后方的隐藏表面 | `center` |
| `outside_source_fov` | 中心视锥外、但会在 ±15° 进入画面的表面 | `left` 或 `right` |

传统中心视角 LDI 只能自然表达前两类。P01 端点大面积外侧缺口包含第三类，所以协议必须允许左/右相机锚定的扩展块，再统一转到世界坐标验证和融合。

### 2.2 每个表面块的必需数组

- `rgb_uint8`：`H×W×3`，sRGB 颜色；
- `depth_z_float32`：`H×W`，锚点相机的正 Z-depth，无效位置为 NaN；
- `alpha_float32`：`H×W`，表面覆盖/软可见性；
- `valid_mask_uint8`：`H×W`，声明哪些像素可参与几何处理；
- `provenance_uint8`：`H×W`，逐像素来源类型；
- `geometry_confidence_float32`：`H×W`，几何可信度；
- `appearance_confidence_float32`：`H×W`，颜色/结构可信度。

`confidence` 和 `alpha/opacity` 严格分离：前者用于审计、融合和拒绝，后者只描述渲染覆盖。禁止像旧脚本那样用“距离掩码边界的距离”代替来源可信度。

### 2.3 来源枚举

| 值 | 名称 | 含义 |
|---:|---|---|
| 0 | `invalid` | 无数据 |
| 1 | `observed_center` | 中心 RGB-D 直接观测 |
| 2 | `reprojected_observation` | 由真实观测重投影 |
| 3 | `predicted_hidden_geometry` | 隐藏几何为模型预测 |
| 4 | `predicted_hidden_appearance` | 隐藏外观/结构为模型预测 |
| 5 | `external_multiview` | 来自额外真实视角 |
| 6 | `synthetic_oracle` | 仅用于受控合成验证的真值 |

部署时单张 RGB 无法使用 5 或 6。它们只用于分析上限和评价，不得混入单图模型的正式指标。

## 3. 几何与可见性规则

1. 所有块都登记锚点相机内外参，并用 Z-depth 反投影到同一世界坐标系。
2. `occlusion_hidden` 在重叠射线上必须位于对应遮挡前景之后；不满足深度顺序的像素拒绝进入渲染。
3. 可见性首先由目标相机 z-buffer/α 合成决定，不再仅靠“来源端点角度渐入渐出”掩盖冲突。
4. 预测层可带角度支撑区间，但该区间是额外拒绝门，不是几何可见性的替代品。
5. 左/右扩展块若在世界空间或中间视角重叠，必须先做深度、颜色和来源冲突检查；不能直接把两批高斯 append 到基座。

## 4. 级联接口

```text
BaseProvider
→ center observed_surface
→ HiddenLayerProvider
   ├─ occlusion_hidden patches
   └─ outside_source_fov patches
→ CLBValidator
   ├─ schema / range / hash
   ├─ depth ordering / reprojection
   ├─ center leakage / cross-view conflict
   └─ provenance / confidence accounting
→ SurfaceFusion
→ GaussianSpawnProvider
```

`HiddenLayerProvider` 是可替换接口，可以先接合成真值，后续再接冻结预训练模型或自训模型。`GaussianSpawnProvider` 不允许跳过 CLBValidator。

## 5. 验证门

### 5.1 表示层硬门

- 必需数组存在，形状一致，数值范围合法；
- 所有 valid 像素有正有限深度、非 `invalid` 来源和 `[0,1]` 置信度；
- 观测层不得含预测来源；合成真值不得进入部署运行；
- 坐标系、深度类型、单位和相机矩阵必须显式登记，不允许根据文件名推断。

### 5.2 `CLB-SYN-01` 几何能力门

- 中心视角的隐藏层可见像素数为 0，观测层渲染与参考中心图逐像素一致；
- ±15° 端点必须同时显示 `occlusion_hidden` 和对应的 `outside_source_fov`；
- 在合成真值有效区，端点深度相对误差 P95 不高于 1%，错误遮挡顺序像素率不高于 0.1%；
- 左/右共视的同一表面 ID 重投影后中位像素距离不高于 0.25 px；
- 任一硬门失败时，不生成高斯，不用人工裁剪或角度渐变规避。

### 5.3 预测模型门（后续）

- 在有多视图真值的验证集上评价，不用 P01 的无真值端点选模型；
- 分开报告遮挡后方区与视野外扩展区，不只报整图平均；
- 除 PSNR/SSIM/LPIPS/DISTS 外，必须报告深度顺序错误、中心泄漏、跨视角冲突和预测来源占比；
- P01 人工门仍要求涂抹、虚假硬结构、深度错误和闪烁均不得达到严重度 2。

## 6. 第一个可运行候选：`CLB-SYN-01`

第一步不是下载新模型，而是建立一个小型受控合成场景：中心前景遮挡板、遮挡后方背景层，以及中心视锥左右外的扩展块。它使用已冻结的 `true_arc` 5°/15°/30° 语义，但只在 0° 和 ±15° 做首轮硬门。

实施顺序：

1. 实现 CLB 清单和 NPZ 数组校验器；
2. 生成 `CLB-SYN-01` 的三类表面块与真值相机；
3. 用表面块直接渲染中心和两端，通过第 5.2 节硬门；
4. 只有表示层通过后，才实现 `CLB → supplemental Gaussians`；
5. 再对候选预测模型做接口审计。

不直接把 TMPI 定为首个可跑模型：本地论文说明了其表示优点，但本轮未找到作者公开的官方代码/权重入口。SLIDE 官方项目页同样只提供论文、补充材料和演示。3D Photo 虽有官方代码，但 P01 固定端点适配已失败；其 LDI 思路保留为协议参考，不再继续局部调参。

## 7. 范围边界

- Stage 1.3 当前不训练网络、不重跑 P01 视频、不生成 P01 补充高斯；
- 合成真值只验证“表示和渲染能不能做对”，不证明单图模型能预测对；
- 该协议覆盖并取代旧 Stage 1 中“端点 RGB-D 直接反投影为高斯”的进入条件；旧结果仍作为失败证据保留。

## 8. `CLB-SYN-01` 执行结果（2026-08-13）

- 已用 `scripts/stage1_clb_synthetic_smoke.py` 生成中心观测背景、中心观测前景、遮挡后方层、左视锥外层和右视锥外层；
- `scripts/stage1_validate_clb.py` 对 5 个表面块的清单、数组类型、形状、数值范围和来源语义校验通过；
- 中心视角隐藏层泄漏为 0 像素；
- 左右端点均正确显示 2,470 像素遮挡后方层和 11,392 像素对应视锥外扩展层；
- 端点深度相对误差 P95 为 0，错误遮挡顺序率为 0.05208%，低于 0.1% 硬门；
- 共享背景平面的中心↔端点往返重投影中位误差为 `4.02e-14 px`；
- 正式小型结果保存在 `records/clb_syn_01.json`，大型/可再生输出保存在本机 `outputs/02_Stage1_true_arc/07_隐藏层表示/CLB-SYN-01_roundtrip/`。首次未计算往返误差的运行保留在同级 `CLB-SYN-01/`，不作为 canonical 记录。

该结果只证明 CLB 数据结构与直接表面渲染能同时表达两类隐藏区域，不证明单图预测质量，也不证明 P01 30° 已可用。下一个单变量任务是 `CLB → supplemental Gaussians` 的合成真值适配与同样硬门，仍不接 P01 生成模型。

## 9. `CLB-GS-SYN-01` 高斯转换结果（2026-08-13）

### 9.1 转换约定

- 新增 `scripts/stage1_clb_to_gaussian_smoke.py`，把 CLB valid 像素按各自锚点相机 Z-depth 反投影到共享世界坐标；
- 高斯的局部 XY 轴与锚点成像平面对齐，Z 轴保持较薄；颜色从 sRGB 转为 linear RGB；
- 来源、层 ID 和索引范围保存在 `gaussian_sidecar.json`。SHARP PLY 没有来源字段，因此单独传递 PLY 会丢失 CLB 审计信息，后续正式格式必须把 PLY 与 sidecar 视为不可分割的 bundle；
- 本实验不使用左右端点角度渐入，隐藏高斯的可见性仅由表面位置、视锥、高斯 Alpha 合成和 z-buffer 决定。

### 9.2 v1 失败与有依据的修正

- v1 使用 `scale_factor=0.42`，在斜视端点下单像素高斯投影太小，左右全图 `alpha>=0.95` 率都只有36.86%；
- 虽然 v1 左右隐藏区 RGB MAE 相对 observed-only 已下降97.97%和97.75%，但覆盖和深度门失败，不能放行；
- v1 深度指标还混入了前/背景边界的 Alpha 混合像素，因此 v2 按解析表面 owner 向内腐蚀3像素后计算可靠内部深度；
- v2 只把 `scale_factor` 改为0.70，不改 CLB、相机、颜色、深度、不透明度或硬门。v1 记录保存为 `records/clb_gs_syn_01_v1_failed.json`。

### 9.3 v2 canonical 结果

- 共生成121,600个高斯，其中观测层93,312个，隐藏/视锥外层28,288个；
- 中心可靠表面内部的 full-vs-observed RGB 最大绝对差为0.000713，低于 `1/255`泄漏门；全图最大诊断差异0.0973位于前景抗锯齿/高斯边界带，已保留报告，没有从结果中删除；
- 左端隐藏区 RGB MAE 相对 observed-only 下降98.18%，右端下降97.98%；
- 0°/±15° 三个视角的可靠表面内部 `alpha>=0.95` 率均为100%；
- 可靠内部深度相对误差 P95：中心0.0306%，左端1.0178%，右端1.0178%，均低于2%门；
- 5 项高斯转换硬门全部通过。正式记录为 `records/clb_gs_syn_01.json`，canonical 本机输出为 `outputs/02_Stage1_true_arc/07_隐藏层表示/CLB-GS-SYN-01_v2/`。

这证明 CLB 的两类隐藏支持区可以通过同一世界坐标高斯表示，无需端点角度渐入来遮掩几何冲突。它仍只是合成 oracle 上的表示能力证据，不是 P01 质量结论。下一阶段应审计一个能同时产生隐藏几何、外观与置信度的预测 Provider，先在有多视图真值的小数据上评价，不直接拿 P01 选模型。
