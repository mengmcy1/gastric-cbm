# 图像裁剪

本模块用于新数据正式训练前的内镜图像预处理候选验证。它不会替换或覆盖既有
`第二批裁剪后_v1_1`，新数据必须先小样本验收，再冻结参数并生成新的数据版本。

## 来源与迁移边界

代码迁移自本机`/home/mcy/Image-cropping`仓库的`bbf3485`提交，原仓库未修改。

- 第一阶段`brightness_crop_v2.py`的核心四边裁剪最初来自本项目旧v1.1预处理，迁出后
  完成了路径参数化、四边统一扫描、进度条检测和复核指标整理；本处复制的是通用化版本，
  旧`数据整理脚本/preprocess_second_batch_v1_1.py`保持不动，用于历史复现。
- 第二阶段使用`shiye_V1` ONNX模型分割有效内镜视野，并统一遮黑视野外设备界面。
- 原仓库中的画中画检测与notch候选生成逻辑已迁入，仅作为候选生成机制，不能在胃早癌
  数据上未经验证直接用于训练；会改变图像尺寸的right分支和三视图数据划分（E0/E1/E2）
  均未迁入正式路线。

## 目录

```text
程序/图像裁剪/
├── README.md
├── requirements.txt
├── 参考模板/
│   ├── validation_labels.example.csv
│   ├── confirmed_pip_labels.example.csv
│   └── pip_rect_overrides.example.csv
├── 模型/
│   └── README.md
└── 正式代码/
    ├── config.py
    ├── brightness_crop_v2.py
    ├── dark_crop_safety_review.py
    ├── onnx_infer.py
    ├── fov_mask_v1.py
    ├── pip_detector.py
    ├── pip_notch_full.py
    ├── build_keep_notch_manifests.py
    └── pip_box_qc_initial_screen.py
```

## 推荐流程

### 1. 亮度轮廓与四边扫描

这一阶段裁掉黑边、暗灰薄边和满足规则的视频进度条，保持裁剪后的原始宽高比，不执行
最终分类输入resize。

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/图像裁剪/正式代码/brightness_crop_v2.py \
  --input <新数据原图目录> \
  --output <候选输出>/01_亮度四边裁剪 \
  --patient-limit 20 \
  --images-per-patient 2 \
  --preview-limit 40
```

输出包含`crops/`、`previews/`和`mapping.csv`。非空输出目录会被拒绝，原图只读。
调试抽样默认把图片直接父目录的完整相对路径作为患者单元，兼容
`类别/中心/患者/图片`等多层目录；正式全量仍应由冻结manifest核对患者归属。

### 1b. 暗部裁剪安全复核

部分胃镜图的有效黏膜边缘本身较暗，单纯亮度轮廓可能把它误当黑边。第一阶段后可用
原图FOV预测检查当前裁剪框是否排除了较多有效视野：

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/图像裁剪/正式代码/dark_crop_safety_review.py \
  --stage1-mapping <候选输出>/01_亮度四边裁剪/mapping.csv \
  --onnx 程序/图像裁剪/模型/shiye_V1.onnx \
  --output <候选输出>/01b_暗部裁剪安全复核
```

脚本输出`mapping.csv`、风险图的`保守裁剪候选/`、`保守裁剪_FOV遮罩候选/`、四联对照
和人工复核分页。绿色框是当前裁剪，橙色框是按FOV越界方向单独放宽的保守候选，
青色轮廓是原图FOV预测；四联图最后一列是保守扩框经过FOV遮罩后的预期正式外观。扩框
可能暂时带回设备UI或视频进度条，只要它们位于FOV轮廓外，遮罩后会被统一置黑。
如果第一阶段已检出底部视频进度条，安全复核会锁住当前底边，只允许恢复顶部、左侧或
右侧暗部；`mapping.csv`中的`expanded_sides/blocked_sides`记录实际放宽和被阻止的方向。
默认阈值仅用于形成待复核候选，不能未经人工验收直接替换正式裁剪。无风险图片仍沿用
第一阶段结果；风险图片的最终选择必须记录为`keep_current`或`use_conservative`，之后再
生成完整、冻结的裁剪清单。

同一次运行还会在`多阈值联图/`输出敏感、中等、保守和强风险四档候选分页，并生成
`阈值候选数量汇总.csv`。四档只改变“哪些图片进入人工复核”，不改变FOV模型阈值或
候选裁剪内容；正式阈值必须在独立验收子集上、查看M0结果之前冻结。

现有数据89张小样本肉眼验收后，已冻结本轮正式复核阈值为：框外FOV比例`>=0.05`或
单边越界比例`>=0.06`（C保守档）。该阈值命中5/89张，数量与可见误裁风险较平衡。
脚本默认值已同步为该口径；敏感和中等档只保留为审计联图，不进入正式`pending`清单。

### 2. FOV分割与视野外遮罩

准备真实ONNX权重后运行。脚本优先使用可选的`onnxruntime`；未安装时自动使用当前
OpenCV自带的CPU DNN后端：

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/图像裁剪/正式代码/fov_mask_v1.py \
  --onnx 程序/图像裁剪/模型/shiye_V1.onnx \
  --input <候选输出>/01_亮度四边裁剪/crops \
  --output <候选输出>/02_FOV遮罩 \
  --limit 40 \
  --preview-limit 40
```

输出包含`masked/`、`masks/`、`previews/`和`mapping.csv`。该阶段只把预测视野外设为纯黑，
不把分割结果当作病灶分割，也不改变癌/非癌标签。

`onnx_infer.py`还提供`safe_rectangle`、`bbox`和`none`三种独立候选模式，用于方法比较；
正式分类输入采用哪种方式必须由小样本人工验收和去偏审计决定。

当前`gastric-cbm`环境已安装`onnxruntime 1.28.0`并作为首选CPU后端；OpenCV DNN保留为
自动回退。迁移验收样本上两种后端生成的最终mask逐像素一致。安装过程没有升级numpy、
OpenCV、PyTorch、torchvision或CUDA。

## 正式采用前的验收

1. 从每个中心、设备风格、标签和主要分辨率层抽取病例，不按全目录前若干张抽样。
2. 同时查看原图、第一阶段裁剪、FOV mask和最终遮罩图。
3. 检查病灶是否被裁掉、胃镜视野是否被过度侵蚀、设备文字是否仍残留。
4. 分中心汇总`fallback`、低/高mask coverage、二级连通域和处理失败率。
5. 比较“仅四边裁剪”和“四边裁剪+FOV遮罩”的边缘元数据可预测性。
6. 医学与工程验收通过后，冻结模型SHA256、阈值、内缩像素和输出版本名，再跑全量。

禁止直接用内部test或外部多中心test选择FOV阈值和裁剪策略。

## 病灶框标注顺序

正式病灶框采用“先冻结空间几何，再在Keep图上标注”的顺序：

1. 对新数据执行亮度四边裁剪，保留裁剪后的原始分辨率和宽高比；
2. 运行FOV遮罩，并按中心、标签和画面风格完成人工验收；
3. 冻结高分辨率Keep图的尺寸、SHA256、`geometry_id`和原图到Keep的变换；
4. 将无可视化框的Keep图交给医学生标注，bbox坐标单独保存；
5. PIP复核、Notch生成和M0可以与病灶标注并行；Notch仅改变像素，不改变坐标；
6. 模型训练时才把图像resize/crop到网络输入尺寸，并对bbox执行完全相同的几何变换。

不要先把图像压缩到`224x224`再交给医学生，小病灶边界会更难判断。带可视化矩形框的
图片只能用于复核，模型输入必须是无框图，坐标来自独立CSV/JSON。建议至少保存
`image_path/image_sha256/geometry_id/image_width/image_height/patient_id/label/lesion_visible/x1/y1/x2/y2/coordinate_convention`，
其中`coordinate_convention`固定为`xyxy_left_closed_right_open`（左闭右开）。

Notch保持画布尺寸不变，与Keep共享`geometry_id`和bbox坐标。若bbox与Notch黑块相交，
必须单独人工复核病灶是否仍可见；不得默认将该图用于定位损失。正式路线不使用会再次
改变尺寸和坐标系的Right分支。

## 画中画保留/去除消融

医学生确认新数据中画中画较常见，因此不预设画中画一定有害。首轮M0增加同图配对的
预处理消融：

- `pip_keep`：使用第一阶段自然裁剪图，保留画中画；
- `pip_remove_notch`：仅对人工确认的画中画图把画中画框及安全边距置黑，其余图与
  `pip_keep`相同；它保留原始视野、尺度和构图，是首轮主对照；

画中画自动检测只负责生成候选框和预览，不能直接决定训练图。正式顺序是：

1. 在按中心、标签和设备风格抽取的验证子集上人工标记`pip/non_pip`，校准检测阈值；
2. 全量运行`pip_detector.py`生成`quality_flags.csv`；
3. 人工复核所有高置信和灰区候选，形成`confirmed_pip_labels.csv`；
4. 运行`pip_notch_full.py`生成保持原尺寸的notch候选；
5. 使用`pip_box_qc_initial_screen.py`和联系表复核处理框是否完整；
6. 使用`build_keep_notch_manifests.py`从同一冻结图片清单生成Keep/Notch两个
   manifest，禁止通过扫描输出目录组装训练集。

示例命令：

```bash
# 全量筛查；输入默认来自前两阶段mapping和mask
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/图像裁剪/正式代码/pip_detector.py \
  --crops <候选输出>/01_亮度四边裁剪/crops \
  --masks <候选输出>/02_FOV遮罩/masks \
  --stage2-mapping <候选输出>/02_FOV遮罩/mapping.csv \
  --output <候选输出>/03_画中画筛查

# 人工确认CSV准备完成后生成去除候选
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/图像裁剪/正式代码/pip_notch_full.py \
  --image-root <候选输出>/02_FOV遮罩/masked \
  --quality-flags <候选输出>/03_画中画筛查/quality_flags.csv \
  --pip-labels <候选输出>/03_画中画筛查/confirmed_pip_labels.csv \
  --output <候选输出>/04_画中画保留去除候选

# 框QC和人工复核通过后，构建两份完整的配对manifest
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/图像裁剪/正式代码/build_keep_notch_manifests.py \
  --base-mapping <候选输出>/02_FOV遮罩/mapping.csv \
  --notch-mapping <候选输出>/04_画中画保留去除候选/mapping.csv \
  --output <候选输出>/05_Keep_Notch配对manifest
```

M0比较固定要求：

- 两组患者、图片ID、train/val/test split、随机种子、增强和训练参数完全相同；
- 无画中画图片在两份manifest中指向同一文件；
- 只允许train/val决定去留，test和外部test不参与选择；
- 主指标为val患者AUC，同时报告Sensitivity、Specificity、画中画子组、无画中画子组和
  分中心结果；
- 必须额外审计画中画/notch出现率在标签和中心间是否平衡，并检查Grad-CAM是否集中于
  左上黑块；notch虽然移除了画中画内容，但黑块仍可能暴露“这里原本有画中画”；
- 若notch版患者AUC不低于保留版超过预先冻结的非劣容忍范围（`0.005`仅为候选），且
  画中画子组、分中心稳定性和黑块关注审计均不恶化，优先选择notch版；否则保留版继续
  作为主输入。
