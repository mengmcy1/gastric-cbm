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
