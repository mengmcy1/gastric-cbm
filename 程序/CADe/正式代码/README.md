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
