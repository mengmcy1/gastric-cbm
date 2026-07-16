[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$InputImage,
    [Parameter(Position = 1)]
    [string]$TestName,
    [string]$CondaEnvironment = "sharp",
    [int]$GpuIndex = 0,
    [switch]$Help
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SharpRepository = Join-Path $ProjectRoot "源码\SHARP_APPLE注释"
$Checkpoint = Join-Path $SharpRepository "checkpoints\sharp_2572gikvuh.pt"
$InputRoot = Join-Path $ProjectRoot "SHARP真实图片测试\输入图片"
$OutputRoot = Join-Path $ProjectRoot "SHARP真实图片测试\输出结果"

function Show-Usage {
    @"
用法：
  powershell -ExecutionPolicy Bypass -File .\run_sharp_single_image_test_windows.ps1 <图片路径> [测试名称]

示例：
  powershell -ExecutionPolicy Bypass -File .\run_sharp_single_image_test_windows.ps1 `"$InputRoot\room.jpg`" 房间人物

可选参数：
  -CondaEnvironment sharp  Conda 环境名，默认 sharp
  -GpuIndex 0              CUDA 物理 GPU，单显卡笔记本默认 0
"@
}

if ($Help -or [string]::IsNullOrWhiteSpace($InputImage)) {
    Show-Usage
    if ($Help) { exit 0 }
    exit 2
}

if (-not (Test-Path -LiteralPath $InputImage -PathType Leaf)) {
    throw "输入图片不存在：$InputImage"
}
if (-not (Test-Path -LiteralPath $SharpRepository -PathType Container)) {
    throw "找不到 SHARP 仓库：$SharpRepository"
}
if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
    throw "找不到 SHARP 权重：$Checkpoint"
}
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "找不到 conda。请在 Anaconda PowerShell Prompt 中运行本脚本。"
}

$InputPath = (Resolve-Path -LiteralPath $InputImage).Path
$InputItem = Get-Item -LiteralPath $InputPath
$Extension = $InputItem.Extension.ToLowerInvariant()
$SupportedExtensions = @(".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff")
if ($SupportedExtensions -notcontains $Extension) {
    Write-Warning "扩展名 $Extension 可能不在 SHARP 的支持列表中。"
}

New-Item -ItemType Directory -Force -Path $InputRoot, $OutputRoot | Out-Null
if ([string]::IsNullOrWhiteSpace($TestName)) {
    $TestLabel = [System.IO.Path]::GetFileNameWithoutExtension($InputItem.Name)
} else {
    $TestLabel = $TestName
}
foreach ($InvalidChar in [System.IO.Path]::GetInvalidFileNameChars()) {
    $TestLabel = $TestLabel.Replace([string]$InvalidChar, "_")
}
$TestLabel = $TestLabel.Replace(" ", "_")
$RunId = "{0}_{1}" -f (Get-Date -Format "yyyyMMdd_HHmmss"), $TestLabel
$RunDirectory = Join-Path $OutputRoot $RunId
$InputDirectory = Join-Path $RunDirectory "输入图片"
$PredictDirectory = Join-Path $RunDirectory "预测输出"
$RenderFirstDirectory = Join-Path $RunDirectory "渲染输出_首次"
$RenderWarmDirectory = Join-Path $RunDirectory "渲染输出_热启动"
$LogDirectory = Join-Path $RunDirectory "日志"
$ColorFrameDirectory = Join-Path $RunDirectory "关键帧\彩色"
$DepthFrameDirectory = Join-Path $RunDirectory "关键帧\深度"

New-Item -ItemType Directory -Force -Path `
    $InputDirectory, $PredictDirectory, $RenderFirstDirectory, $RenderWarmDirectory, `
    $LogDirectory, $ColorFrameDirectory, $DepthFrameDirectory | Out-Null

$CopiedInput = Join-Path $InputDirectory ("input" + $Extension)
Copy-Item -LiteralPath $InputPath -Destination $CopiedInput
$env:CUDA_VISIBLE_DEVICES = [string]$GpuIndex
$env:PYTHONUNBUFFERED = "1"

function Invoke-Conda {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [string]$LogPath
    )

    $Timer = [System.Diagnostics.Stopwatch]::StartNew()
    & conda run --no-capture-output -n $CondaEnvironment @Arguments 2>&1 |
        Tee-Object -FilePath $LogPath
    $ExitCode = $LASTEXITCODE
    $Timer.Stop()
    ("总耗时（秒）：{0:N2}" -f $Timer.Elapsed.TotalSeconds) |
        Tee-Object -FilePath $LogPath -Append
    if ($ExitCode -ne 0) {
        throw "命令失败，退出码 $ExitCode。日志：$LogPath"
    }
}

try {
    $EnvironmentLog = Join-Path $LogDirectory "environment.log"
    @(
        "测试开始时间：$(Get-Date -Format o)"
        "测试目录：$RunDirectory"
        "原始输入：$InputPath"
        "测试副本：$CopiedInput"
        "SHARP 仓库：$SharpRepository"
        "权重：$Checkpoint"
        "Conda 环境：$CondaEnvironment"
        "CUDA_VISIBLE_DEVICES：$($env:CUDA_VISIBLE_DEVICES)"
    ) | Tee-Object -FilePath $EnvironmentLog

    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & nvidia-smi 2>&1 | Tee-Object -FilePath $EnvironmentLog -Append
    } else {
        "警告：找不到 nvidia-smi。" | Tee-Object -FilePath $EnvironmentLog -Append
    }

    $CudaCheck = "import torch; print('PyTorch：', torch.__version__); print('CUDA可用：', torch.cuda.is_available()); print('进程可见GPU数量：', torch.cuda.device_count()); print('进程cuda:0：', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '不可用'); raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() == 1 else 1)"
    Invoke-Conda -Arguments @("python", "-c", $CudaCheck) -LogPath (Join-Path $LogDirectory "cuda_check.log")

    Push-Location $SharpRepository
    try {
        Write-Host "`n[1/3] 开始预测高斯。"
        Invoke-Conda -Arguments @(
            "sharp", "predict", "-i", $CopiedInput, "-o", $PredictDirectory,
            "-c", $Checkpoint, "--device", "cuda", "--no-render", "-v"
        ) -LogPath (Join-Path $LogDirectory "predict.log")

        $PlyPath = Join-Path $PredictDirectory "input.ply"
        if (-not (Test-Path -LiteralPath $PlyPath -PathType Leaf) -or
            (Get-Item -LiteralPath $PlyPath).Length -eq 0) {
            throw "没有生成有效 PLY：$PlyPath"
        }

        Write-Host "`n[2/3] 开始首次渲染。"
        Invoke-Conda -Arguments @(
            "sharp", "render", "-i", $PlyPath, "-o", $RenderFirstDirectory, "-v"
        ) -LogPath (Join-Path $LogDirectory "render_first.log")

        Write-Host "`n[3/3] 开始热启动渲染。"
        Invoke-Conda -Arguments @(
            "sharp", "render", "-i", $PlyPath, "-o", $RenderWarmDirectory, "-v"
        ) -LogPath (Join-Path $LogDirectory "render_warm.log")
    } finally {
        Pop-Location
    }

    $ColorVideo = Join-Path $RenderWarmDirectory "input.mp4"
    $DepthVideo = Join-Path $RenderWarmDirectory "input.depth.mp4"
    if (-not (Test-Path -LiteralPath $ColorVideo -PathType Leaf) -or
        -not (Test-Path -LiteralPath $DepthVideo -PathType Leaf)) {
        throw "没有生成完整的彩色和深度视频。"
    }

    $Ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($Ffmpeg) {
        $Timestamps = @("0.0", "0.5", "1.0", "1.5")
        $Labels = @("00_source", "05_side_a", "10_forward", "15_side_b")
        for ($Index = 0; $Index -lt $Timestamps.Count; $Index++) {
            & $Ffmpeg.Source -hide_banner -loglevel error -y `
                -ss $Timestamps[$Index] -i $ColorVideo -frames:v 1 `
                (Join-Path $ColorFrameDirectory ("color_{0}.png" -f $Labels[$Index]))
            if ($LASTEXITCODE -ne 0) { throw "彩色关键帧提取失败。" }

            & $Ffmpeg.Source -hide_banner -loglevel error -y `
                -ss $Timestamps[$Index] -i $DepthVideo -frames:v 1 `
                (Join-Path $DepthFrameDirectory ("depth_{0}.png" -f $Labels[$Index]))
            if ($LASTEXITCODE -ne 0) { throw "深度关键帧提取失败。" }
        }
    } else {
        "警告：找不到 ffmpeg，跳过关键帧提取。" |
            Tee-Object -FilePath $EnvironmentLog -Append
    }

    $PlySizeMb = [math]::Round((Get-Item -LiteralPath $PlyPath).Length / 1MB, 2)
    $ColorSizeMb = [math]::Round((Get-Item -LiteralPath $ColorVideo).Length / 1MB, 2)
    $DepthSizeMb = [math]::Round((Get-Item -LiteralPath $DepthVideo).Length / 1MB, 2)
    $ReportPath = Join-Path $RunDirectory "问题分析记录.md"
    $Report = @"
# SHARP 单张真实图片测试记录

## 1. 测试基本信息

- 测试编号：$RunId
- 原始输入：``$InputPath``
- PLY 大小：$PlySizeMb MB
- 彩色视频大小：$ColorSizeMb MB
- 深度视频大小：$DepthSizeMb MB
- 物理 GPU：GPU $GpuIndex（进程内显示为 ``cuda:0``）

运行环境见 ``日志/environment.log`` 和 ``日志/cuda_check.log``，运行耗时见三个阶段日志。

## 2. 画质检查

严重度：0=没有，1=轻微，2=明显，3=严重影响观看。

| 检查项 | 严重度 0–3 | 发生位置 | 具体表现 |
|---|---:|---|---|
| 原始视角重建 |  |  |  |
| 整体深度结构 |  |  |  |
| 前景边缘与新显露区域 |  |  |  |
| 细长结构 |  |  |  |
| 漂浮高斯或重影 |  |  |  |
| 反射/透明区域 |  |  |  |
| 连续帧稳定性 |  |  |  |
| 清晰度和视差幅度 |  |  |  |

## 3. 分阶段判断

- 原视角是否已经模糊或失真：待填写
- 是否只在新视角出现空洞、拖抹或背景复制：待填写
- 前景是否比背景更近：待填写
- 细结构是否与背景粘连：待填写
- EXIF 或焦距是否可能有问题：待填写

## 4. 初步归因

- Depth Pro 深度问题：待填写
- 双层深度分工问题：待填写
- Gaussian Decoder 问题：待填写
- 3DGS 覆盖/表示问题：待填写
- Windows/CLI 工程性能问题：待填写

## 5. 最值得优先解决的问题

1. 待填写
2. 待填写
3. 待填写
"@
    Set-Content -LiteralPath $ReportPath -Value $Report -Encoding UTF8

    $CompletionLog = Join-Path $LogDirectory "completion.txt"
    @(
        "测试完成。"
        "测试目录：$RunDirectory"
        "高斯 PLY：$PlyPath（$PlySizeMb MB）"
        "热启动彩色视频：$ColorVideo"
        "热启动深度视频：$DepthVideo"
        "关键帧目录：$(Split-Path -Parent $ColorFrameDirectory)"
        "问题分析记录：$ReportPath"
        "日志目录：$LogDirectory"
    ) | Tee-Object -FilePath $CompletionLog
} catch {
    [Console]::Error.WriteLine("测试失败：$($_.Exception.Message)")
    [Console]::Error.WriteLine("已生成的文件和日志保留在：$RunDirectory")
    exit 1
}
