# SHARP 单张真实图片测试脚本运行说明

本文件夹提供 Linux 服务器版和 Windows 笔记本版两套运行脚本。两套脚本完成相同的测试流程，但使用的终端、路径格式和默认 GPU 编号不同。

## 1. 文件与目录结构

```text
2Dto3D/
├── SHARP运行脚本/
│   ├── run_sharp_single_image_test_linux.sh
│   ├── run_sharp_single_image_test_windows.ps1
│   └── README_运行说明.md
├── SHARP真实图片测试/
│   ├── 输入图片/
│   └── 输出结果/
└── 源码/
    └── SHARP_APPLE注释/
        └── checkpoints/
            └── sharp_2572gikvuh.pt
```

- `run_sharp_single_image_test_linux.sh`：Linux 服务器版本，默认使用物理 GPU 1，即第二张 NVIDIA 显卡。
- `run_sharp_single_image_test_windows.ps1`：原生 Windows PowerShell 版本，默认使用 GPU 0，适合只有一张 NVIDIA 显卡的笔记本。
- `SHARP真实图片测试/输入图片`：放置自己准备的真实图片。
- `SHARP真实图片测试/输出结果`：脚本自动创建每次测试的独立结果目录。

两个脚本都会根据自身所在位置自动确定项目根目录，因此整个 `2Dto3D` 文件夹移动到其他位置后，一般不需要修改脚本中的项目路径。

## 2. 输入图片准备

将待测试图片放入：

```text
SHARP真实图片测试/输入图片/
```

建议优先使用：

- JPG 或 PNG 格式；
- 画面清晰、分辨率正常的原始照片；
- 尽量保留 EXIF 和焦距信息；
- 文件名不要包含过多特殊符号。

脚本也接受项目外部的图片路径，但会在本次输出目录中保存一份输入副本。

### 2.1 当前测试批次

目前输入图片已经按照测试目的分为三轮：

```text
SHARP真实图片测试/输入图片/
├── 01_第一轮_基础与主展示/       6张
├── 02_第二轮_困难与问题分析/     7张
└── 03_第三轮_补充低优先级/       4张
```

- 第一轮用于检查正常场景下的基础效果，并选择 PPT 主展示图；
- 第二轮用于主动测试细结构、遮挡、反射、虚化和复杂深度；
- 第三轮用于补充远景、道路、山景等低优先级场景。

脚本支持读取子文件夹中的图片，不需要因为图片归类而修改脚本代码，只需在运行命令中传入图片移动后的完整路径。

## 3. Linux 服务器运行方法

### 3.1 当前服务器配置

- Conda 环境：`sharp`
- 默认物理 GPU：GPU 1，即第二张显卡
- SHARP 权重：`源码/SHARP_APPLE注释/checkpoints/sharp_2572gikvuh.pt`

### 3.2 运行命令

先进入脚本目录：

```bash
cd /home/mcy/2Dto3D/SHARP运行脚本
```

建议先运行第一轮中的一张校园图片：

```bash
bash ./run_sharp_single_image_test_linux.sh \
  "../SHARP真实图片测试/输入图片/01_第一轮_基础与主展示/IMG_20220628_134917.jpg" \
  "第一轮_校园建筑"
```

第一个参数是输入图片路径，第二个参数是可选的测试名称。省略测试名称时，脚本使用图片文件名作为测试名称。

脚本会自动激活 `sharp` 环境、固定使用物理 GPU 1、加载现有权重，并依次完成预测、首次渲染、热启动渲染、关键帧提取和问题分析记录生成。

### 3.3 运行第一轮全部图片

确认单张图片可以成功运行后，可以串行处理第一轮全部6张图片：

```bash
cd /home/mcy/2Dto3D/SHARP运行脚本

for image in "../SHARP真实图片测试/输入图片/01_第一轮_基础与主展示/"*.jpg; do
    bash ./run_sharp_single_image_test_linux.sh "$image"
done
```

该循环会逐张运行，不会同时占用多份 GPU 显存。每张图片会创建独立的时间戳输出目录，不会覆盖其他测试结果。

不建议并行运行多张图片，否则可能因为多个 SHARP 进程同时占用 GPU 而出现显存不足。

### 3.4 指定其他 GPU

默认使用物理 GPU 1。如需临时指定 GPU 0：

```bash
SHARP_GPU_INDEX=0 bash /home/mcy/2Dto3D/SHARP运行脚本/run_sharp_single_image_test_linux.sh \
  "/home/mcy/2Dto3D/SHARP真实图片测试/输入图片/01_第一轮_基础与主展示/IMG_20220628_134917.jpg" \
  "第一轮_校园建筑_GPU0"
```

查看帮助：

```bash
bash /home/mcy/2Dto3D/SHARP运行脚本/run_sharp_single_image_test_linux.sh --help
```

### 3.5 观察 GPU 状态

运行脚本时，可以在另一个终端持续观察 GPU 和显存占用：

```bash
watch -n 1 nvidia-smi
```

脚本通过 `CUDA_VISIBLE_DEVICES=1` 选择物理 GPU 1。因为该进程只看得到这一张显卡，所以它在 PyTorch 和日志中显示为 `cuda:0`，这是正常的设备重编号现象。

## 4. Windows 笔记本运行方法

### 4.1 运行前提

Windows 版本建议在 **Anaconda PowerShell Prompt** 中运行。运行前应确保：

- NVIDIA 驱动可正常识别 RTX 5060 Laptop GPU；
- 已创建能够运行 SHARP 的 Conda 环境，默认环境名为 `sharp`；
- SHARP 源码和权重位于上述目录结构中；
- `sharp` 命令可以在 Conda 环境中调用；
- FFmpeg 已安装并加入 `PATH`，否则脚本会跳过关键帧提取；
- 如果 `gsplat` 需要现场编译，已经准备相匹配的 CUDA Toolkit 和 Visual Studio C++ Build Tools。

### 4.2 运行前检查

在 Anaconda PowerShell Prompt 中依次运行：

```powershell
nvidia-smi
py --version
conda --version
nvcc --version
conda run -n sharp python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA不可用')"
```

如果没有单独安装 CUDA Toolkit，`nvcc --version` 可能提示找不到命令；这不一定影响已有预编译包运行，但可能影响 `gsplat` 的现场编译。

### 4.3 运行命令

进入脚本文件夹：

```powershell
cd "C:\Users\MCY\Desktop\2D转3D\SHARP运行脚本"
```

运行测试：

```powershell
powershell -ExecutionPolicy Bypass -File `
  ".\run_sharp_single_image_test_windows.ps1" `
  "..\SHARP真实图片测试\输入图片\01_第一轮_基础与主展示\IMG_20220628_134917.jpg" `
  第一轮_校园建筑
```

Windows 笔记本通常只有一张可供 CUDA 使用的 NVIDIA 独立显卡，因此脚本默认使用 GPU 0。Intel 或 AMD 核显不会占用 CUDA 的 GPU 编号。

### 4.4 自定义 Conda 环境或 GPU

如果 Conda 环境不是 `sharp`：

```powershell
powershell -ExecutionPolicy Bypass -File `
  ".\run_sharp_single_image_test_windows.ps1" `
  "..\SHARP真实图片测试\输入图片\01_第一轮_基础与主展示\IMG_20220628_134917.jpg" `
  第一轮_校园建筑 `
  -CondaEnvironment "你的环境名"
```

如需指定其他 CUDA GPU：

```powershell
powershell -ExecutionPolicy Bypass -File `
  ".\run_sharp_single_image_test_windows.ps1" `
  "..\SHARP真实图片测试\输入图片\01_第一轮_基础与主展示\IMG_20220628_134917.jpg" `
  第一轮_校园建筑 `
  -GpuIndex 1
```

查看帮助：

```powershell
powershell -ExecutionPolicy Bypass -File ".\run_sharp_single_image_test_windows.ps1" -Help
```

## 5. 脚本运行流程

一次完整测试会依次执行：

1. 检查输入图片、SHARP 仓库、模型权重、Conda 和 CUDA 环境；
2. 将输入图片复制到本次测试目录；
3. 使用 SHARP 将单张图片预测为 3D Gaussian Splatting（3D 高斯泼溅，3DGS）PLY 文件；
4. 执行第一次渲染，用于观察冷启动和 `gsplat` 初始化开销；
5. 再次渲染，用于记录热启动性能；
6. 从热启动生成的彩色视频和深度视频中提取关键帧；
7. 保存环境、耗时和运行日志；
8. 生成 `问题分析记录.md`，用于检查深度、遮挡、新视角和高斯表示问题。

首次使用 `gsplat` 时可能需要编译或加载 CUDA 扩展，因此第一次渲染明显慢于第二次属于正常现象。

## 6. 输出目录说明

每次运行都会在下面的位置创建带时间戳的独立目录：

```text
SHARP真实图片测试/输出结果/年月日_时分秒_测试名称/
```

典型输出结构：

```text
年月日_时分秒_测试名称/
├── 输入图片/
│   └── input.jpg
├── 预测输出/
│   └── input.ply
├── 渲染输出_首次/
├── 渲染输出_热启动/
│   ├── input.mp4
│   └── input.depth.mp4
├── 关键帧/
│   ├── 彩色/
│   └── 深度/
├── 日志/
│   ├── environment.log
│   ├── predict.log
│   ├── render_first.log
│   ├── render_warm.log
│   └── completion.txt
└── 问题分析记录.md
```

重点查看：

- `input.ply`：SHARP 预测得到的高斯场景；
- `input.mp4`：新视角彩色视频；
- `input.depth.mp4`：对应的深度视频；
- `关键帧/`：便于逐张对比原视角和新视角；
- `问题分析记录.md`：用于记录失败位置、严重度和初步归因；
- 三个阶段日志：用于比较预测、首次渲染和热启动渲染时间。

## 7. 常见问题

### 7.1 找不到 Conda

Windows 下请使用 Anaconda PowerShell Prompt。普通 PowerShell 如果没有完成 Conda 初始化，可能找不到 `conda` 命令。

### 7.2 找不到模型权重

确认下面的文件存在：

```text
源码/SHARP_APPLE注释/checkpoints/sharp_2572gikvuh.pt
```

权重文件约 2.62 GiB，不要只复制源码而遗漏权重。

### 7.3 CUDA 不可用

运行：

```powershell
nvidia-smi
conda run -n sharp python -c "import torch; print(torch.cuda.is_available())"
```

如果输出为 `False`，重点检查 NVIDIA 驱动、PyTorch CUDA 版本和当前 Conda 环境。

### 7.4 RTX 5060 笔记本显存不足

RTX 5060 Laptop GPU 通常为 8GB 显存。SHARP 默认处理较高分辨率并输出约 118 万个高斯，运行时可能比较接近显存上限。

建议：

- 接通电源并开启独显高性能模式；
- 关闭游戏、浏览器硬件加速和其他占用显存的程序；
- 保持预测与渲染分阶段执行；
- 通过 `nvidia-smi` 观察显存占用；
- 如果仍然出现 `CUDA out of memory`，先保存完整错误日志，再根据实际占用调整环境或推理方案，不要直接随意修改模型输入尺寸。

### 7.5 找不到 FFmpeg

脚本仍可完成 SHARP 预测和视频渲染，但会跳过关键帧提取。安装 FFmpeg 并将其加入 Windows `PATH` 后重新运行即可。

检查命令：

```powershell
ffmpeg -version
```

### 7.6 第一次渲染很慢

首次渲染可能包含 `gsplat` CUDA 扩展初始化或编译时间。应同时查看：

- `日志/render_first.log`
- `日志/render_warm.log`

如果第二次渲染明显更快，通常说明主要差异来自冷启动开销，而不是每次渲染都会持续这么慢。

## 8. 回家后建议的验证顺序

1. 先执行 Windows 运行前检查命令；
2. 确认 PyTorch 可以识别 RTX 5060；
3. 使用一张普通、清晰、没有复杂反射的照片进行首次测试；
4. 记录是否出现显存不足、`gsplat` 编译失败或 FFmpeg 缺失；
5. 首次测试成功后，再测试人物头发、树枝、栏杆、透明物体和反射场景；
6. 将输出视频、关键帧、日志和问题分析记录一起用于后续问题定位。
