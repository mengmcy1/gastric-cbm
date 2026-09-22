# 胃镜教师与学生模型

本仓库当前入口用于教师训练、学生蒸馏、图像预处理、推理和患者级结果汇总。教师是灰度局部ROI的MG1b，学生是完整彩色图的C-long；两者均为EfficientNet-B0加7×7注意力池化。学生部署不需要教师或病灶框。

历史研究代码保存在 `research-archive-before-handoff` 分支。主分支保留当前流程及其实际依赖，不使用旧的普通GAP分类器 `inference.py` 加载C-long。

## 先准备什么

- 此私有仓库的访问权限。
- Release `mage-models` 中的 `模型权重.tar` 和 `SHA256SUMS`。
- 如需训练或复现原队列：由数据管理者通过受控渠道提供整个 `教师学生模型交接数据` 目录。患者图片、标注、清单、教师缓存不在GitHub。
- 只做新图片学生推理时，不需要研究数据目录，只需已完成预处理的图片和学生权重。

代码和权重可以运行推理；训练必须有授权的图像、患者划分和ROI/癌框。没有这些数据，不能复现原研究指标。内部时间外测试和外部多中心测试不是训练或调参集合。

## 环境配置

已验证环境：Linux x86_64、Python 3.12.13、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128。GPU训练环境使用RTX 5080；建议GPU显存至少16GB，实际需求随batch变化。CPU可以推理和跑小样本训练，完整训练建议GPU。原队列训练建议32GB系统内存，教师双翻转缓存构建会临时使用数GB内存。数据与中间产物建议预留至少15GB空间。

在任意目录克隆，不要求用户名或安装位置与原服务器相同：

```bash
git clone --depth 1 https://github.com/mengmcy1/gastric-cbm.git
cd gastric-cbm
conda create -n gastric-cbm python=3.12 -y
conda activate gastric-cbm
```

NVIDIA环境先确认驱动能支持CUDA 12.8，再安装：

```bash
nvidia-smi
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

只使用CPU时，将PyTorch安装命令改为：

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

若OpenCV提示缺少libGL或libgthread，在系统层安装相应运行库（Ubuntu常见为libgl1和libglib2.0-0）。不需要安装桌面GUI。

## 下载权重

在项目根目录执行。`gh`需要先通过GitHub授权账号登录，也可以在网页Release页面手动下载：

```bash
gh release download mage-models --repo mengmcy1/gastric-cbm --dir weights-download
cd weights-download
sha256sum -c SHA256SUMS
cd ..
tar -xf weights-download/模型权重.tar
python 程序/MAGE/正式代码/manage_mage_handoff.py verify-assets
export TORCH_HOME="$PWD/模型资产/torch"
```

包括正式MG1b教师、正式C-long学生、FOV ONNX模型和ImageNet初始化权重。另提供 `ROI辅助权重.tar`，保留生成教师ROI使用的5折OOF检测器和Y3-F检测器；学生部署和使用现成ROI清单训练不需要加载这些检测器。若重新运行ROI检测流程，额外安装 `ultralytics==8.4.118` 并解包该文件，相关清单仍由受控数据交接。使用随包的ImageNet初始化可避免训练时另行联网下载。默认模型仍是原C-long，不自动使用后来的非癌框微调候选或SAE实验模型。

## 数据准备

将接收到的目录放在 `数据/教师学生模型交接数据/`，其中包含：

- 训练集：2350张、1212位患者。
- 验证集：497张、260位患者。
- 留出测试集：501张、260位患者。
- 内部时间外测试集：112张、78位患者。
- 外部多中心测试集：1941张、1329位患者。
- 标注源文件、原始冻结清单、教师ROI与缓存、beta校准、原始输入图。

每个数据集的 `图片与标注清单.csv` 都有相对图片路径、患者编号、分类标签、框是否存在、框来源及归一化坐标。没有框不代表正常，也不能自行补全。现有非癌框是后补标注，不改变原教师/学生训练中的仅癌图定位监督。

兼容历史训练入口时执行：

```bash
python 程序/MAGE/正式代码/manage_mage_handoff.py restore-data
python 程序/MAGE/正式代码/manage_mage_handoff.py check-data
```

脚本只创建旧路径到新目录的相对软链接，不复制图片，不覆盖已有不同文件。数据目录位于其他磁盘时可传 `--data-root /实际目录/教师学生模型交接数据`。传输该数据目录时保持内容与目录结构完整；不要只传旧路径下的软链接。

## 图像预处理

正式推理输入是裁边和FOV遮罩后的图，不是带设备界面的原始截图。已有整理数据已经完成这些步骤，不要再次处理。

新原图示例（输出目录必须为空或不存在）：

```bash
python 程序/图像裁剪/正式代码/brightness_crop_v2.py --input /原图目录 --output 工作目录/裁边
python 程序/图像裁剪/正式代码/fov_mask_v1.py --input 工作目录/裁边/crops --output 工作目录/视野遮罩 --onnx 程序/图像裁剪/模型/shiye_V1.onnx
```

查看两步产生的mapping.csv和预览。对pending、明显误裁或特殊画中画，不能未经检查直接纳入正式训练。需要暗部裁剪复核时使用 `dark_crop_safety_review.py`；画中画相关工具位于同目录，不会在推理时自动猜测是否应切除病灶。当前模型使用Keep输入口径，不自动改成notch或其他裁图分支。

癌框及教师ROI应位于最终预处理图的坐标系。若标注在原图上，必须按实际裁剪框转换坐标；不能把原图框直接套到裁剪图。

模型输入处理由代码完成：学生全图RGB→双线性resize 224×224→除255→ImageNet均值/标准差归一化；教师按归一化ROI裁切、resize后转灰度三通道，再使用同样归一化。评价/推理没有随机裁剪或翻转。

## 使用现有模型推理

单张学生输入：

```bash
python 程序/MAGE/正式代码/infer_mage.py --image /已预处理图片.jpg --role student --device cpu --save-attention --output 推理结果/单张
```

批量输入CSV至少包含 `image_path,patient_id`。图片路径默认相对CSV所在目录，也可用 `--image-root` 指定。整理后的各数据集 `图片与标注清单.csv` 可直接作为学生推理清单，其他标注列会忽略。

```bash
python 程序/MAGE/正式代码/infer_mage.py --manifest 数据/教师学生模型交接数据/02_验证集/图片与标注清单.csv --role student --device cpu --save-attention --output 推理结果/验证集
```

教师推理需要额外提供局部ROI，坐标是完整预处理图上的x1,y1,x2,y2，范围0–1。它不是全图学生的替代入口：

```bash
python 程序/MAGE/正式代码/infer_mage.py --role teacher --image /已预处理图片.jpg --roi 0.1 0.2 0.8 0.9 --device cpu --output 推理结果/教师
```

教师批量CSV另需 `roi_x1,roi_y1,roi_x2,roi_y2`。未提供ROI会报错，不会默认整图推理。`--weights`可指定本仓库训练入口产生的新模型checkpoint。

GPU运行前检查占用，例如选定空闲卡后使用 `CUDA_VISIBLE_DEVICES=2 ... --device cuda`；不要抢占他人进程。

## 后处理

推理目录包含：

- `image_predictions.csv`：每图癌概率及图片阈值判定。
- `patient_predictions.csv`：同一patient_id图片概率的算术均值及患者阈值判定。
- `inference_config.json`：权重、阈值、输入口径。
- 指定 `--save-attention` 时保存 `attention_7x7.npy`，顺序与逐图CSV一致。学生坐标对应完整预处理图，教师坐标对应局部ROI；它不是原图分割。

当前学生图片阈值0.25923898816108704，患者阈值0.3074711561203003；教师分别为0.21375234425067902和0.3049298021942377。代码从对应checkpoint读取阈值，不混用。类别0为非癌，1为早癌/高级别瘤变。不会根据待测图片重新挑阈值。

有真值标签的同一清单可追加评价，沿用推理时保存的冻结阈值，不重新扫阈值：

```bash
python 程序/MAGE/正式代码/evaluate_mage_predictions.py --manifest 数据/教师学生模型交接数据/02_验证集/图片与标注清单.csv --predictions 推理结果/验证集 --output 推理结果/验证集/评价.json
```

清单必须与预测逐图一一对应。只评估已批准的数据集合，训练调参期间不要运行内部/外部测试评价。

## 训练教师，再训练学生

接收方的新训练使用 `train_mage_handoff.py`，复用现有模型、训练增强、患者/类别平衡抽样、冻结阶段与损失。输入需要train和val，且患者互不重叠；它不接受test或external。

清单必须含：`image_relpath,patient_id,label,split`，以及教师ROI `base_crop_x1,base_crop_y1,base_crop_x2,base_crop_y2`、癌图框 `bbox_x1_norm,bbox_y1_norm,bbox_x2_norm,bbox_y2_norm`。ROI对每张图片必填，癌框只用于癌图；全部坐标相对完整预处理图。新数据的非癌ROI需事先明确生成来源，不能用测试标签或临时默认框替代。本项目原ROI来自癌框、患者级OOF检测及既定回退规则，相关生成代码保留在MAGE目录。

原队列可直接使用已经交接的冻结ROI清单：

```bash
export TORCH_HOME="$PWD/模型资产/torch"
nvidia-smi
# 检查显存和进程后，设置空闲GPU；下面的编号只是示例。
export CUDA_VISIBLE_DEVICES=2
mkdir -p 新训练/logs
nohup python -u 程序/MAGE/正式代码/train_mage_handoff.py --role teacher --manifest 数据/教师学生模型交接数据/08_训练配套资产/教师学生_冻结ROI清单.csv --output 新训练/teacher --device cuda > 新训练/logs/teacher.log 2>&1 &
tail -F 新训练/logs/teacher.log
```

日志出现“完成教师训练”后，按Ctrl+C退出日志查看（不会停止后台训练），再运行学生：

```bash
nohup python -u 程序/MAGE/正式代码/train_mage_handoff.py --role student --manifest 数据/教师学生模型交接数据/08_训练配套资产/教师学生_冻结ROI清单.csv --teacher 新训练/teacher/model.pth --stage-a-epochs 10 --stage-b-epochs 40 --output 新训练/student --device cuda > 新训练/logs/student.log 2>&1 &
tail -F 新训练/logs/student.log
```

日志出现“完成学生蒸馏训练”后，可加载新学生推理：

```bash
python 程序/MAGE/正式代码/infer_mage.py --weights 新训练/student/model.pth --role student --image /已预处理图片.jpg --output 推理结果/新学生
```

训练输出model.pth（包含优化器）、history.csv、config.json及val逐图/逐患者预测；学生另外保存由指定教师生成的训练缓存。输出中有患者级资料，只能留在受控存储。

新训练入口固定保存末轮，只在val上确定该模型阈值，**不声称复现历史最佳epoch选择、资格判定或原论文结果**。默认ImageNet初始化，stage A仅训练两个头、stage B解冻features[5:]，学习率分别1e-3/1e-4，weight decay 1e-4。教师损失为CE+0.25×癌图框KL；学生为原C组CE、教师logit蒸馏和癌图attention蒸馏，默认beta沿用0.19362648121926898作为固定起点，不冒称新数据已经完成旧beta校准。

仅验证新机器训练链路时，可用少量患者的独立train/val清单，传 `--device cpu --batch-size 2 --stage-a-epochs 1 --stage-b-epochs 1 --random-init`。这种模型不用于诊断；不得把小样本链路成功写成模型性能复现。

## 复现原冻结研究流程

保留原入口，未移除已有资格、数据和权重校验：

- `train_mage_mg1b_attention_teacher.py`：原教师训练与资格判定。
- `build_mage_mg2_teacher_cache.py`：原冻结教师的双翻转缓存。
- `train_mage_mg2_student.py`：原A/B/C学生训练，C-long为stage A 10轮、stage B最多40轮，按原验证规则选择。

使用交接数据恢复旧路径后，原教师可在独立输出目录重训：

```bash
mkdir -p 结果/原流程复现/logs
nohup python -u 程序/MAGE/正式代码/train_mage_mg1b_attention_teacher.py --device cuda --output-root 结果/原流程复现 --run-name teacher > 结果/原流程复现/logs/teacher.log 2>&1 &
tail -F 结果/原流程复现/logs/teacher.log
```

教师复现任务完成后，确认GPU资源再运行学生复现：

```bash
nohup python -u 程序/MAGE/正式代码/train_mage_mg2_student.py --arm C --device cuda --stage-a-epochs 10 --stage-b-epochs 40 --beta-calibration-json 数据/教师学生模型交接数据/08_训练配套资产/冻结beta校准.json --output-root 结果/原流程复现 --run-name student > 结果/原流程复现/logs/student.log 2>&1 &
tail -F 结果/原流程复现/logs/student.log
```

原学生命令使用交接的原冻结教师缓存，不会自动采用刚重训的教师。换目录时只将冻结JSON内的原仓库路径映射到当前克隆位置，原JSON和缓存文件不改写，内容校验仍完整执行。原缓存与beta的内容绑定必须保留；即使数值相近，重新序列化的缓存也不能冒充原缓存。原教师重训未通过资格时会保存诊断产物并报错，这是研究结果，不应删除判断或强行接入原学生。若要将自己新训练的教师接到学生，使用前面的新训练入口。

## 验证范围

本次交接检查在现有Linux/CPU环境完成：原图裁边→FOV遮罩→学生推理→图片与患者结果；正式教师ROI预处理与原Dataset逐值一致；正式学生输入与原Dataset逐值一致、同设备输出与原实现一致；4张抽查图相对历史缓存最大概率差约0.000148，未据此宣称完整历史指标精确复现；小样本教师两阶段训练→该教师缓存→学生两阶段训练→新权重推理。没有重新训练完整研究模型，没有重新评价锁定测试集，也没有在新装操作系统上验证所有驱动组合。

数据目录属于受控交接，不随GitHub发布。请保留所有split，不把验证和测试样本加入训练。后续非癌框微调和SAE解释不是当前默认部署模型的一部分。
