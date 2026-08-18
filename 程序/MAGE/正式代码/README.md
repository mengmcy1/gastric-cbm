# MAGE-like bbox引导蒸馏

本目录保存MAGE-like正式入口。该路线借鉴MAGE的灰度局部教师、logit蒸馏和空间关注蒸馏，
但本项目只有癌/HGD矩形bbox，不能称为原论文的像素mask完整复现。

当前入口：

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/build_mage_g0_manifest.py --self-test

/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/build_mage_g0_manifest.py
```

MG0a仅构建train/val患者级5折OOF索引，不训练模型，不读取Y0-F test、internal test或
external。后续OOF YOLO、灰度局部教师和全图学生入口必须在对应预注册冻结后再加入。

构建OOF YOLO数据视图：

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/build_mage_oof_yolo_views.py --self-test

/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/build_mage_oof_yolo_views.py
```

该步骤只为每个外层fold建立fit/monitor/holdout软链接。项目正式val不参与OOF检测器早停，
holdout只用于之后生成未见患者的Top-1预测。

训练单个fold前先检查GPU。以下仅为调用格式，物理编号必须按当时`nvidia-smi`替换：

```bash
CUDA_DEVICE=<空闲GPU> bash \
  程序/MAGE/正式代码/run_mage_oof_yolo_matrix.sh 0

tail -F 结果/MAGE/MG0b_OOF_YOLO_20260817/logs/mg0b_oof_yolo26s_fold0.log
```

单fold验收后再把最后参数改为`0,1,2,3,4`。启动器遇到失败会保留现场并继续后续fold，
最终以非零状态退出；完整fold会幂等跳过，中断且含`last.pt`的fold会自动恢复。

合并五折预测、构建教师ROI并执行几何/QC审计：

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/build_mage_teacher_roi_manifest.py --self-test

/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/build_mage_teacher_roi_manifest.py
```

正式清单发现裁图大小具有中等标签预测力后，可另建尺寸匹配敏感性清单。该入口不覆盖正式
清单：癌图保持不变，非癌以检测中心为放置目标并从train癌框的来源/尺寸分层分布确定性
抽取边长；若大框靠近图像边缘，则仅为保持框在图内而平移。

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/build_mage_size_matched_roi_sensitivity.py --self-test

/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/build_mage_size_matched_roi_sensitivity.py
```

动态扩边ROI v2改用与原图相同像素宽高比的裁图，保证绿色源框完整包含；小框每侧扩20%，
大框在归一化尺度0.85处停止继续扩张，源框本身超过0.85时不缩小。脚本同时输出动态版和
动态+尺寸匹配版。该版本已被人工QC发现仍会使细长框在短轴过度扩张，现仅作历史对照。

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/build_mage_dynamic_roi_v2.py --self-test

/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/build_mage_dynamic_roi_v2.py
```

独立轴动态扩边ROI v3分别对源框宽、高扩边和限幅，不再为了固定宽高比把细长框补成大方框。
矩形ROI直接缩放到224x224，完整保留源框且不添加可见padding。脚本同时生成主候选与宽高
成对尺寸匹配诊断分支；后者不能自动视为去偏成功，必须查看患者级几何审计。

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/build_mage_dynamic_roi_v3.py --self-test

/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/build_mage_dynamic_roi_v3.py
```

MG1冻结为一组配对实验：真实灰度v3 ROI与7x7 patch-shuffle使用相同初始化、采样和训练预算，
随后与v3 geometry-only患者AUC统一比较。正式启动前必须先查看`nvidia-smi`，再显式指定当时
空闲的物理GPU；启动器依次运行两组并自动生成冻结门槛汇总。

```bash
CUDA_DEVICE=<空闲GPU物理编号> bash \
  程序/MAGE/正式代码/run_mage_mg1_pair.sh

tail -F 结果/MAGE/MG1灰度局部教师_20260817/正式验证集筛选/logs/\
mg1_real_efficientnet_b0_seed42.log
```

真实分支完成后启动器会继续运行`patch_shuffle`。可另开终端同时跟踪两份日志：

```bash
tail -F \
  结果/MAGE/MG1灰度局部教师_20260817/正式验证集筛选/logs/\
mg1_real_efficientnet_b0_seed42.log \
  结果/MAGE/MG1灰度局部教师_20260817/正式验证集筛选/logs/\
mg1_patch_shuffle_efficientnet_b0_seed42.log
```

MG1正式配对未建立患者级空间排列增益，因此不能直接进入MG2。MG1b作为新的预注册问题，
用癌图bbox监督7x7空间attention，并强制分类特征只能经attention加权池化进入分类头；非癌图
仍参与分类，但不伪造空间目标。先执行自测和CPU debug，再检查GPU运行正式seed42：

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/train_mage_mg1b_attention_teacher.py --self-test

/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/train_mage_mg1b_attention_teacher.py \
  --debug --device cpu --run-name mg1b_attention_manual_smoke

CUDA_DEVICE=<空闲GPU物理编号> bash \
  程序/MAGE/正式代码/run_mage_mg1b.sh

tail -F 结果/MAGE/MG1b注意力池化教师_20260818/正式验证集筛选/logs/\
mg1b_attention_efficientnet_b0_seed42.log
```

正式产品必须同时通过患者/图像AUC、normalized AiB、PGA和病灶大小分层PGA门槛。失败时
脚本只保存`mg1b_best_diagnostic_ineligible.pth`并返回非零状态，不允许该权重进入MG2。

MG1b通过后，按病灶大小分层导出只读注意力QC三联图；该步骤不重新选择checkpoint：

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/export_mage_mg1b_attention_qc.py
```

MG2全图学生蒸馏（适配协议2026-08-18冻结）：教师固定为MG1b正式产品（SHA256
`f2cd3b13...af48f9`），学生为同架构attention-pooling全图彩色EfficientNet-B0，
A/B/C三组仅损失不同。顺序：单元测试→教师缓存（B/C共用同一份）→CPU debug验收
→一个完整校准epoch（batch 32、74个固定批次、2350次有放回抽样）冻结beta→
A/B/C矩阵→八门槛汇总。beta校准只看train，不看val：

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/test_mage_mg2_backfill.py

/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/test_mage_mg2_beta_and_gates.py

CUDA_VISIBLE_DEVICES=<空闲GPU物理编号> PYTHONUNBUFFERED=1 \
  /home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/build_mage_mg2_teacher_cache.py --device cuda

/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/train_mage_mg2_student.py --arm A --debug --device cpu

CUDA_VISIBLE_DEVICES=<空闲GPU物理编号> PYTHONUNBUFFERED=1 \
  /home/mcy/miniconda3/envs/gastric-cbm/bin/python -u \
  程序/MAGE/正式代码/train_mage_mg2_student.py --device cuda \
  --calibrate-beta 结果/MAGE/MG2全图学生蒸馏_20260818/beta_calibration_seed42.json
```

校准JSON保存每批SHA、抽样审计（重复数/唯一图/标签与患者分布）、校准设备与全部
SHA绑定，自身SHA写入同名`.sha256` sidecar。正式校准已完成并冻结：
`结果/MAGE/MG2全图学生蒸馏_20260818/beta_calibration_seed42.json`（SHA256
`47179b62...03ac7f`，beta=0.19362648121926898，74批/2350次有放回抽样，cuda），
禁止修改或重新生成；正式教师缓存`teacher_cache_mg1b_v3.pt`亦已构建。正式C组训练
会逐字符核验该校准JSON的SHA并重算beta（容差1e-12），任一不符即拒绝启动。

随后用矩阵启动器按A→B→C顺序正式运行（完整产物幂等跳过，失败保留现场；三组全部
成功后自动调用`summarize_mage_mg2.py`执行八门槛判定，汇总已存在时幂等跳过；
`MG2_MATRIX_DRY_RUN=1`只打印计划不执行）：

```bash
CUDA_DEVICE=<空闲GPU物理编号> bash \
  程序/MAGE/正式代码/run_mage_mg2_matrix.sh

tail -F 结果/MAGE/MG2全图学生蒸馏_20260818/正式验证集筛选/logs/\
mg2_arma_efficientnet_b0_seed42.log
```

汇总器自动读取冻结beta校准JSON并交叉校验SHA绑定与arm C config，逐组重算
checkpoint SHA，并核验三组seed/超参/增强/架构/教师与清单SHA完全一致；
M0-F参照患者AUC直接读自其正式config（正式模式读取失败即终止，仅debug允许回退
常量0.9133并显式警告）。单独执行判定时：

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/MAGE/正式代码/summarize_mage_mg2.py
```

缓存与学生训练都会校验教师checkpoint与manifest的SHA256绑定及逐图对齐；正式输出
目录`结果/MAGE/MG2全图学生蒸馏_20260818/正式验证集筛选/`已存在正式产物时拒绝覆盖，
debug产物只写入独立的`debug/`子目录。C组beta只能来自`--beta-calibration-json`
（正式模式禁止手工`--beta`）；A/B组不接受校准文件。单独运行单组时把启动器替换为
`train_mage_mg2_student.py --arm A|B|C --device cuda`（C组加
`--beta-calibration-json <校准JSON>`）。
