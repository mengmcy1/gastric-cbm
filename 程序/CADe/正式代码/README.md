# CADe 正式代码

`evaluate_cade_cd0.py` 对冻结的 Y3-F YOLO26s-640 三种子做 CD0 错误地图。
它不训练模型，只读取开发 val 和已经用于工程判断的外部开发队列，
不读取新内部时间测试集。
病灶大小分层统一使用 Development Train 癌图冻结的 bbox 面积 q33/q67，不允许在
val或external队列内重算分位点。

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/CADe/正式代码/test_cade_cd0.py

/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/CADe/正式代码/evaluate_cade_cd0.py \
  --device 0 --preflight-only
```

正式运行前必须先用 `nvidia-smi` 检查显卡，再去掉 `--preflight-only`。
输出包括逐种子 FROC、部署阈值指标、FN/FP 清单与人工复核两联图。
三 seed 推理完成后，用终结入口去重错误、生成稳定性标记、三 seed 汇总与
20% 双人复核分配：

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/CADe/正式代码/finalize_cade_cd0.py
```

如果只需用已保存预测刷新冻结尺度分层，使用：

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/CADe/正式代码/finalize_cade_cd0.py \
  --refresh-stratification-only
```

## CD1 Gray Qualification

CD1正式协议、Gray数据和单seed训练入口均已冻结。先验证Gray数据血缘、错误互补指标、严格FP
二分图匹配、患者整簇bootstrap、四态决策及训练入口参数边界：

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/CADe/正式代码/test_cade_cd1.py

/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/CADe/正式代码/test_train_cd1_gray.py

/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/CADe/正式代码/test_evaluate_cd1_val.py
```

测试通过后，生成只包含Development Train/Val的三通道Gray PNG视图：

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/CADe/正式代码/prepare_cd1_gray_data.py

/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/CADe/正式代码/smoke_cd1_gray_augmentation.py
```

该入口不会导出Y0-F的test，也不会读取External Development或Locked Internal Temporal
CADe Test。数据审计与真实Ultralytics增强smoke通过后，可先用完整train split做单轮debug：

```bash
CUDA_VISIBLE_DEVICES=<物理GPU编号> /home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/CADe/正式代码/train_cd1_gray.py \
  --seed 42 \
  --device 0 \
  --debug
```

`CUDA_VISIBLE_DEVICES`会把选中的物理GPU重新映射为进程内`cuda:0`。例如使用物理GPU 2时，
应设置`CUDA_VISIBLE_DEVICES=2`并传`--device 0`；不要同时使用`CUDA_VISIBLE_DEVICES=2`和
`--device 2`。也可以不设置`CUDA_VISIBLE_DEVICES`，直接传物理编号`--device 2`。

启动任何GPU任务前必须先用`nvidia-smi`同时核对显存、利用率和进程。`train_cd1_gray.py`
只训练并保存可追溯产品，不计算Primary、rescue、Jaccard或部署阈值，也不作Gray资格判断。
正式三seed完成后必须由独立的Development Val评价入口统一生成四态结论。

三组正式训练产品齐全后，先运行只读预检，再在一张空闲GPU上执行唯一一次正式val评价：

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/CADe/正式代码/evaluate_cd1_val.py \
  --device 0 \
  --preflight-only

CUDA_VISIBLE_DEVICES=<物理GPU编号> /home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/CADe/正式代码/evaluate_cd1_val.py \
  --device 0
```

正式入口重新推理冻结RGB与Gray产品，只读取Development Val，使用5000次患者簇配对
bootstrap，并生成不可覆盖的`CD1_VAL_DECISION.json`。该文件冻结前后均不得读取External
Development或Locked Internal Temporal CADe Test。

内部四态决策冻结后，可运行External Development只读投影。该入口复用CD0正式RGB预测，
只新增三组Gray推理；External Primary与val冻结部署点分开报告，不产生第二次资格判断：

```bash
/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/CADe/正式代码/test_evaluate_cd1_external.py

/home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/CADe/正式代码/evaluate_cd1_external.py \
  --device 0 \
  --preflight-only

CUDA_VISIBLE_DEVICES=<物理GPU编号> /home/mcy/miniconda3/envs/gastric-cbm/bin/python \
  程序/CADe/正式代码/evaluate_cd1_external.py \
  --device 0
```

输出包括三seed的RGB/Gray预测、FROC、患者簇bootstrap、FN/FP互补指标，以及Primary和val
冻结部署点的逐GT四象限与`0/3`至`3/3` Gray rescue稳定性表。正式JSON固定记录内部结论仍为
`INCONCLUSIVE`、External仅作描述性投影，并继续声明未读取Locked Internal Temporal CADe Test。
