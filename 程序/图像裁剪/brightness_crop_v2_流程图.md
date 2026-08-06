# brightness_crop_v2.py 流程图

> 用途：代码 review / 讲解用。分两张图：**整批处理流程**（main）与**单张裁剪流程**（process_one）。
> 核心思路：**先粗检（亮度轮廓找 FOV）→ 再精修（四边黑边扫描）→ 质检留痕**。

## 1. 整批处理流程（main）

```mermaid
flowchart TD
    A["启动 brightness_crop_v2"] --> B["解析参数 + 校验<br/>输入存在 / 输出目录非空拒绝覆盖"]
    B --> C["list_images 递归收集图片<br/>+ select_images 分层抽样"]
    C --> D["build_output_relative_paths<br/>为每张源图构造输出相对路径"]
    D --> E["逐张循环调用 process_one()"]
    E --> F{"还有未处理图?"}
    F -- "是" --> E
    F -- "否" --> G["写 mapping.csv<br/>每张一行（成功/失败都记）"]
    G --> H["写 previews 抽样预览"]
    H --> I["结束"]
```

## 2. 单张裁剪流程（process_one）

```mermaid
flowchart TD
    P0["read_image 读原图"] --> P1["detect_initial_roi 粗检测<br/>resize缩512 → 亮度前景掩膜<br/>→ 轮廓 → FOV外接框"]
    P1 --> P1q{"粗检通过?<br/>轮廓面积/外接框比例/中心偏移"}
    P1q -- "否" --> P1f["full_frame<br/>判定不裁剪"]
    P1q -- "是" --> P2["refine_by_four_edges 精修<br/>four_edge_offsets 四边黑边扫描<br/>+ detect_progress_bar 进度条<br/>+ 校验/expand/clamp"]
    P1f --> P3["裁切 + 空检查"]
    P2 --> P3
    P3 --> P4["质检<br/>black_pixel_ratio 黑占比<br/>+ edge_overlay_metrics 界面残留"]
    P4 --> P5["build_review_reasons<br/>→ review_status = pending / auto_pass"]
    P5 --> P6["write_jpeg 保存裁剪图<br/>（保持宽高比，不缩放）"]
    P6 --> P7["make_preview 三联预览（可选）"]
    P7 --> P8["返回 mapping 行<br/>坐标/方法/状态/各指标/SHA"]
```

## 3. 关键设计点

- **粗检 → 精修 两级**：先亮度轮廓粗定位 FOV，再四边扫描裁黑边——避免单层误判；
- **full_frame 兜底**：粗检不合格或内容填满时判定不裁剪（防切到 FOV 圆）；
- **进度条感知**：检测到底部视频进度条时在栏顶截断，不把工具栏裁回；
- **只读不覆盖**：原图只读、输出独立目录、非空拒绝覆盖；
- **失败不中断**：单张异常记 `error` 行，批量继续；
- **坐标留痕**：`crop_x1..y2`、trim 量、SHA 全进 mapping，供后续 bbox 追溯。

## 4. 输出文件一览

| 文件 | 内容 |
| --- | --- |
| `crops/` | 裁剪后图像（保持宽高比、原分辨率） |
| `previews/` | 原图+裁剪框+结果 三联预览（抽样） |
| `mapping.csv` | 每张一行：坐标、是否 full_frame、进度条标志、trim 量、黑占比、界面残留、review_status、SHA |
