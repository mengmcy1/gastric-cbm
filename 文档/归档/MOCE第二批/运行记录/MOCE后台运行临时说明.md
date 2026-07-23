# MOCE 后台运行临时说明

> 记录日期：2026-07-16  
> 本文件记录本次 MOCE 全量任务的后台运行和停止方法。PID 只对本次运行有效，任务重新启动后需以新 PID 为准。

## 当前任务

| 模型 | GPU | PID | 日志文件 | 输出目录 |
| --- | ---: | ---: | --- | --- |
| ResNet50 | 0 | `1308950` | `结果/MOCE聚类/第二批/resnet50_run.log` | `结果/MOCE聚类/第二批/resnet50/` |
| EfficientNet-B0 | 1 | `1309589` | `结果/MOCE聚类/第二批/efficientnet_b0_run.log` | `结果/MOCE聚类/第二批/efficientnet_b0/` |

两个模型是相互独立的进程，使用不同 GPU 和不同输出目录，不会互相覆盖。

## 启动命令

先进入正式代码目录：

```bash
cd /home/mcy/gastric-cbm/程序/MOCE/正式代码
```

ResNet50 使用 GPU 0：

```bash
nohup env CUDA_VISIBLE_DEVICES=0 \
/home/mcy/miniconda3/envs/gastric-cbm/bin/python -u -c \
"import moce_cluster as m; m.MODEL_NAME='resnet50'; m.main()" \
> /home/mcy/gastric-cbm/结果/MOCE聚类/第二批/resnet50_run.log \
2>&1 < /dev/null &
```

EfficientNet-B0 使用 GPU 1：

```bash
nohup env CUDA_VISIBLE_DEVICES=1 \
/home/mcy/miniconda3/envs/gastric-cbm/bin/python -u -c \
"import moce_cluster as m; m.MODEL_NAME='efficientnet_b0'; m.main()" \
> /home/mcy/gastric-cbm/结果/MOCE聚类/第二批/efficientnet_b0_run.log \
2>&1 < /dev/null &
```

`nohup` 可避免关闭终端、断开网络或本地电脑休眠后任务随终端结束。`-u` 让 Python 日志及时写入文件。

## 查看任务状态

查看全部 MOCE 后台任务：

```bash
pgrep -af "import moce_cluster"
```

查看本次两个任务：

```bash
ps -p 1308950,1309589 -o pid,etime,stat,pcpu,pmem,args
```

查看 GPU 状态：

```bash
nvidia-smi
```

只要进程仍存在，或者日志和候选文件仍在增加，任务就仍在运行。候选提取、图片保存和 K-Means 阶段可能主要使用 CPU，因此某一时刻 GPU 利用率为 0% 不代表任务卡死。

## 查看日志

ResNet50：

```bash
tail -f /home/mcy/gastric-cbm/结果/MOCE聚类/第二批/resnet50_run.log
```

EfficientNet-B0：

```bash
tail -f /home/mcy/gastric-cbm/结果/MOCE聚类/第二批/efficientnet_b0_run.log
```

在 `tail -f` 界面按 `Ctrl+C` 只会退出日志查看，不会停止 MOCE 后台任务。

## 停止某一个任务

优先使用 `SIGINT`，相当于在前台按 `Ctrl+C`。

停止 ResNet50：

```bash
kill -INT 1308950
```

停止 EfficientNet-B0：

```bash
kill -INT 1309589
```

等待几秒后检查：

```bash
ps -p 1308950,1309589 -o pid,stat,args
```

如果目标进程仍未停止，再使用普通终止信号：

```bash
kill 进程PID
```

只有普通停止仍无效时才强制结束：

```bash
kill -9 进程PID
```

不要优先使用 `kill -9`，因为它不会给程序关闭文件和释放资源的机会。

## 中断后的目录处理

当前 MOCE 不支持从中断位置继续运行。任务中断后会留下不完整目录，不能直接当作正式结果，也不建议在原目录上直接重跑，否则可能残留旧候选文件。

重新运行前先把不完整目录改名，例如：

```bash
mv 结果/MOCE聚类/第二批/resnet50 \
   结果/MOCE聚类/第二批/resnet50_incomplete_日期
```

```bash
mv 结果/MOCE聚类/第二批/efficientnet_b0 \
   结果/MOCE聚类/第二批/efficientnet_b0_incomplete_日期
```

确认新结果完整后，再决定是否删除带 `_incomplete_日期` 的旧目录。

## 判断是否完整完成

每个模型应同时生成 `class_0` 和 `class_1`，并且每个类别目录至少包含：

- `候选区域/`
- `候选掩码/`
- `kmeans_model.joblib`
- `candidate_features.npz`
- `cluster_assignments.csv`
- `cluster_summary.csv`
- `concept_clusters.png`
- `concept_scores_per_image.csv`
- `concept_importance.csv`
- `ssc_sdc_per_image.csv`
- `ssc_sdc_summary.csv`

只有候选区域和候选掩码、没有上述聚类和重要性文件，说明任务没有完整结束。

## 注意事项

- 不要重复启动同一模型并写入同一输出目录。
- 两个模型可以同时运行，但建议使用不同 GPU。
- 运行期间不要再启动全量 Grad-CAM 等大量 GPU/磁盘任务。
- PID 每次启动都会变化，重启任务后需要同步更新本文档中的 PID。
- 输出日志位于 `结果/MOCE聚类/第二批/`，与模型结果目录分开保存。
