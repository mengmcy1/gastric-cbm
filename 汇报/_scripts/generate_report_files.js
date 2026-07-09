const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const reportDir = path.resolve(__dirname, "..");
const rootDir = path.resolve(reportDir, "..");
const assetDir = path.join(reportDir, "_assets");
const sourceDocx = path.join(reportDir, "论文总结7.10_NeRF与3DGS.docx");
const outDocx = path.join(reportDir, "论文总结7.10_NeRF与3DGS_补图版.docx");
const outPptx = path.join(reportDir, "论文总结7.10_NeRF与3DGS_基础汇报.pptx");

function psPath(p) {
  return p.replace(/'/g, "''");
}

const images = {
  nerfTitle: path.join(assetDir, "nerf_paper_title.png"),
  nerfTask: path.join(assetDir, "nerf_fig1_task.png"),
  nerfPipeline: path.join(assetDir, "nerf_fig2_pipeline.png"),
  nerfMetrics: path.join(assetDir, "nerf_table1_metrics.png"),
  nerfResults: path.join(assetDir, "nerf_fig5_results.png"),
  gsTitle: path.join(assetDir, "3dgs_paper_title.png"),
  gsSpeed: path.join(assetDir, "3dgs_fig1_speed_quality.png"),
  gsPipeline: path.join(assetDir, "3dgs_fig2_pipeline.png"),
  gsDensity: path.join(assetDir, "3dgs_fig4_density_control.png"),
  gsMetrics: path.join(assetDir, "3dgs_table1_metrics.png"),
  gsResults: path.join(assetDir, "3dgs_fig5_results.png"),
};

for (const [name, file] of Object.entries(images)) {
  if (!fs.existsSync(file)) throw new Error(`Missing image asset ${name}: ${file}`);
}

const ps = `
$ErrorActionPreference = "Stop"
$sourceDocx = '${psPath(sourceDocx)}'
$outDocx = '${psPath(outDocx)}'
$outPptx = '${psPath(outPptx)}'

$img = @{
  nerfTitle = '${psPath(images.nerfTitle)}'
  nerfTask = '${psPath(images.nerfTask)}'
  nerfPipeline = '${psPath(images.nerfPipeline)}'
  nerfMetrics = '${psPath(images.nerfMetrics)}'
  nerfResults = '${psPath(images.nerfResults)}'
  gsTitle = '${psPath(images.gsTitle)}'
  gsSpeed = '${psPath(images.gsSpeed)}'
  gsPipeline = '${psPath(images.gsPipeline)}'
  gsDensity = '${psPath(images.gsDensity)}'
  gsMetrics = '${psPath(images.gsMetrics)}'
  gsResults = '${psPath(images.gsResults)}'
}

Copy-Item -LiteralPath $sourceDocx -Destination $outDocx -Force

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$wdCollapseEnd = 0
try {
  $doc = $word.Documents.Open($outDocx)

  function Clean-ParaText($text) {
    return ($text -replace "[`r`a]", "").Trim()
  }

  function Insert-FigureAfterExact($doc, [string]$needle, [string]$imagePath, [string]$caption, [double]$widthPt) {
    for ($i = 1; $i -le $doc.Paragraphs.Count; $i++) {
      $p = $doc.Paragraphs.Item($i)
      $t = Clean-ParaText $p.Range.Text
      if ($t -eq $needle) {
        $r = $p.Range.Duplicate
        $r.Collapse($wdCollapseEnd)
        $r.InsertParagraphAfter()
        $r.Collapse($wdCollapseEnd)
        $r.Text = $caption + "`r"
        $r.Font.Name = "Microsoft YaHei"
        $r.Font.Size = 9
        $r.Font.Color = 8421504
        $r.Collapse($wdCollapseEnd)
        $shape = $doc.InlineShapes.AddPicture($imagePath, $false, $true, $r)
        $shape.LockAspectRatio = -1
        $shape.Width = $widthPt
        $shape.Range.ParagraphFormat.Alignment = 1
        $shape.Range.InsertParagraphAfter()
        return
      }
    }
    throw "Cannot find paragraph: $needle"
  }

  Insert-FigureAfterExact $doc "第一篇：NeRF" $img.nerfTitle "图源：NeRF 原论文首页（ECCV 2020），用于确认论文来源与基本信息。" 360
  Insert-FigureAfterExact $doc "1.1 动机与问题" $img.nerfTask "图 1：NeRF 的任务流程：多张输入图像 → 优化神经辐射场 → 渲染新视角。" 430
  Insert-FigureAfterExact $doc "1.2 方法核心" $img.nerfPipeline "图 2：NeRF 方法总览：沿相机光线采样 5D 坐标，经 MLP 输出颜色和密度，再通过体积渲染合成像素。" 430
  Insert-FigureAfterExact $doc "1.3 关键实验结果" $img.nerfMetrics "图 3：NeRF 在多个数据集上的定量指标，对比 SRN、NV、LLFF 等方法。" 430
  Insert-FigureAfterExact $doc "第二篇：3D Gaussian Splatting" $img.gsTitle "图源：3D Gaussian Splatting 原论文首页（SIGGRAPH 2023），用于确认论文来源与基本信息。" 360
  Insert-FigureAfterExact $doc "2.1 动机与问题" $img.gsSpeed "图 4：3DGS 的核心动机：在接近 NeRF 质量的前提下，把渲染速度提升到实时级别。" 430
  Insert-FigureAfterExact $doc "2.2 方法核心" $img.gsPipeline "图 5：3DGS 方法流程：SfM 点云初始化 3D 高斯，经投影、密度控制与瓦片光栅化生成图像。" 430
  Insert-FigureAfterExact $doc "2.3 关键实验结果" $img.gsMetrics "图 6：3DGS 在真实场景数据集上的定量结果，突出训练时间、渲染速度与质量之间的优势。" 430

  $doc.Save()
  $doc.Close()
} finally {
  $word.Quit()
}

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = [Microsoft.Office.Core.MsoTriState]::msoTrue
$ppLayoutBlank = 12
$msoTextOrientationHorizontal = 1
$msoShapeRectangle = 1
$msoShapeLine = 9
$msoShapeOval = 9
$msoFalse = [Microsoft.Office.Core.MsoTriState]::msoFalse
$msoTrue = [Microsoft.Office.Core.MsoTriState]::msoTrue

function Add-Text($slide, [string]$text, [double]$left, [double]$top, [double]$width, [double]$height, [double]$size, [int]$color, [bool]$bold=$false) {
  $box = $slide.Shapes.AddTextbox($msoTextOrientationHorizontal, $left, $top, $width, $height)
  $box.TextFrame.TextRange.Text = $text
  $box.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
  $box.TextFrame.TextRange.Font.Size = $size
  $box.TextFrame.TextRange.Font.Color.RGB = $color
  $box.TextFrame.TextRange.Font.Bold = $(if ($bold) { -1 } else { 0 })
  $box.TextFrame.WordWrap = $msoTrue
  return $box
}

function Add-Title($slide, [string]$title, [string]$section="") {
  if ($section -ne "") { Add-Text $slide $section 48 24 300 24 12 10066329 $true | Out-Null }
  Add-Text $slide $title 48 50 860 48 26 13421823 $true | Out-Null
  $line = $slide.Shapes.AddShape($msoShapeRectangle, 48, 104, 80, 4)
  $line.Fill.ForeColor.RGB = 192
  $line.Line.Visible = $msoFalse
}

function Add-Footer($slide, [int]$num) {
  Add-Text $slide "2D 转浅 3D · NeRF 与 3DGS 论文总结 · 2026.7.10" 48 690 650 18 8 8421504 $false | Out-Null
  Add-Text $slide ([string]$num) 1170 690 40 18 8 8421504 $false | Out-Null
}

function Add-Bullets($slide, [string[]]$items, [double]$left, [double]$top, [double]$width, [double]$height, [double]$size=17) {
  $text = ($items | ForEach-Object { "• " + $_ }) -join "`r"
  $box = Add-Text $slide $text $left $top $width $height $size 0 $false
  $box.TextFrame.TextRange.ParagraphFormat.SpaceAfter = 8
  return $box
}

function Add-Image($slide, [string]$path, [double]$left, [double]$top, [double]$width, [double]$height) {
  $pic = $slide.Shapes.AddPicture($path, $msoFalse, $msoTrue, $left, $top, $width, $height)
  return $pic
}

function Add-Card($slide, [string]$label, [string]$body, [double]$left, [double]$top, [double]$width, [double]$height) {
  $shape = $slide.Shapes.AddShape($msoShapeRectangle, $left, $top, $width, $height)
  $shape.Fill.ForeColor.RGB = 16448250
  $shape.Line.ForeColor.RGB = 13421772
  Add-Text $slide $label ($left+14) ($top+12) ($width-28) 26 15 192 $true | Out-Null
  Add-Text $slide $body ($left+14) ($top+42) ($width-28) ($height-52) 13 0 $false | Out-Null
}

$prs = $ppt.Presentations.Add()
$prs.PageSetup.SlideWidth = 1280
$prs.PageSetup.SlideHeight = 720

$slides = @()
for ($i=1; $i -le 11; $i++) {
  $slides += $prs.Slides.Add($i, $ppLayoutBlank)
  $slides[$i-1].Background.Fill.ForeColor.RGB = 16777215
}

$s=$slides[0]
Add-Text $s "NeRF 与 3D Gaussian Splatting" 60 95 980 64 34 13421823 $true | Out-Null
Add-Text $s "2D 转浅 3D 项目 · 基础必读论文总结" 62 170 760 34 21 0 $false | Out-Null
Add-Image $s $img.nerfTask 650 250 510 142 | Out-Null
Add-Image $s $img.gsSpeed 165 430 950 224 | Out-Null
Add-Footer $s 1

$s=$slides[1]
Add-Title $s "为什么先读这两篇基石论文" "阅读目标"
Add-Bullets $s @(
  "NeRF 定义了“用神经网络表示 3D 场景并渲染新视角”的通用框架。",
  "3DGS 把同一类体积渲染思想改成 GPU 友好的显式高斯表示，解决了实时渲染问题。",
  "它们共同暴露出本课题要突破的问题：多图输入、相机位姿、逐场景训练。"
) 70 155 650 260 18 | Out-Null
Add-Card $s "本课题关键词" "单张 2D 图像 → 浅 3D 表达 → 可交互新视角" 770 160 360 150
Add-Footer $s 2

$s=$slides[2]
Add-Title $s "从多图新视角合成，走向单图浅 3D" "课题关联"
Add-Card $s "已有基石方法" "多张图像 + 已知相机位姿 → 优化一个场景表示 → 渲染新视角" 80 160 330 150
Add-Card $s "本项目目标" "单张或少量 2D 图像 → 快速生成浅层 3D 表达 → 支持移动端交互" 475 160 330 150
Add-Card $s "核心难点" "深度不确定、遮挡区域缺失、透明/反光材质、边缘伪影、实时约束" 870 160 330 150
Add-Bullets $s @("后续综述要围绕“减少输入、提升几何一致性、增强实时性”组织。") 90 390 1050 100 20 | Out-Null
Add-Footer $s 3

$s=$slides[3]
Add-Title $s "NeRF 用一个网络记住连续 3D 场景" "NeRF · 问题"
Add-Image $s $img.nerfTask 80 145 660 185 | Out-Null
Add-Bullets $s @(
  "输入：几十张不同视角照片，以及每张照片对应的相机位置。",
  "输出：任意新视角下的图像。",
  "关键想法：不用传统网格存场景，而是让 MLP 查询任意空间点的颜色和密度。"
) 790 150 370 250 16 | Out-Null
Add-Footer $s 4

$s=$slides[4]
Add-Title $s "NeRF 的核心是“沿光线采样再叠加”" "NeRF · 方法"
Add-Image $s $img.nerfPipeline 75 145 690 235 | Out-Null
Add-Card $s "位置编码" "把坐标展开到高频 sin/cos 特征，帮助网络学习清晰细节。" 800 140 330 105
Add-Card $s "体积渲染" "沿相机光线采样多个点，把颜色和密度按透明度叠加成一个像素。" 800 270 330 120
Add-Card $s "分层采样" "粗网络先找物体大概位置，细网络在关键区域采更多点。" 800 415 330 105
Add-Footer $s 5

$s=$slides[5]
Add-Title $s "NeRF 质量高，但离本课题目标还很远" "NeRF · 优势与局限"
Add-Image $s $img.nerfMetrics 70 155 650 160 | Out-Null
Add-Card $s "优势" "连续表示、画质高、存储小、端到端可训练。" 780 150 330 115
Add-Card $s "局限" "每个场景需训练 1-2 天；渲染慢；依赖几十张图和准确相机位姿。" 780 300 330 135
Add-Bullets $s @("对本项目的启示：单图浅 3D 不能直接照搬 NeRF，需要引入深度估计和前馈预测。") 80 500 1020 80 19 | Out-Null
Add-Footer $s 6

$s=$slides[6]
Add-Title $s "3DGS 把质量和实时速度同时推上来" "3DGS · 问题"
Add-Image $s $img.gsSpeed 80 150 900 215 | Out-Null
Add-Bullets $s @(
  "NeRF 的瓶颈是光线步进和大量 MLP 查询。",
  "3DGS 改用显式 3D 高斯椭球表示场景。",
  "渲染时把高斯投影到屏幕上并做 α 混合，GPU 能高效并行。"
) 150 440 900 150 18 | Out-Null
Add-Footer $s 7

$s=$slides[7]
Add-Title $s "3DGS 的流程更接近传统图形渲染管线" "3DGS · 方法"
Add-Image $s $img.gsPipeline 70 135 1040 305 | Out-Null
Add-Card $s "显式表示" "每个高斯保存位置、尺度、旋转、透明度和方向相关颜色。" 90 500 315 105
Add-Card $s "密度控制" "细节不足就克隆或分裂高斯，没用的高斯会被删除。" 480 500 315 105
Add-Card $s "瓦片光栅化" "按屏幕小块并行排序和混合，实现实时渲染。" 870 500 315 105
Add-Footer $s 8

$s=$slides[8]
Add-Title $s "NeRF 与 3DGS 是同一问题的两种取舍" "对比"
Add-Card $s "NeRF" "隐式 MLP 表示；存储小；质量高；训练和渲染慢。" 95 165 430 150
Add-Card $s "3DGS" "显式高斯表示；存储更大；训练快；渲染实时。" 655 165 430 150
Add-Bullets $s @(
  "共同点：都服务于新视角合成，都依赖多视角图像和相机位姿。",
  "关键差异：NeRF 用网络查询点，3DGS 用高斯投影和 GPU 光栅化。",
  "本课题不能只追求完整 3D，还要关注浅 3D 的速度、稳定性和移动端可用性。"
) 105 390 980 160 18 | Out-Null
Add-Footer $s 9

$s=$slides[9]
Add-Title $s "基石论文暴露出的局限，就是本课题的切入点" "课题启示"
Add-Card $s "输入限制" "现有基石方法通常需要几十张图，不适合单张照片直接生成。" 75 150 330 125
Add-Card $s "几何限制" "遮挡区域、细线边缘、透明/反光材质容易产生错误深度和伪影。" 475 150 330 125
Add-Card $s "效率限制" "移动端体验要求快速生成、低内存、可交互渲染。" 875 150 330 125
Add-Bullets $s @(
  "后续方案可以围绕：单目深度估计 + 分层表达 / 前馈 3DGS + 遮挡补全 + 渲染优化。",
  "评估时既看客观指标，也要看 30°左右视角变化时是否有裂缝、重影、纸片感。"
) 95 380 1040 150 19 | Out-Null
Add-Footer $s 10

$s=$slides[10]
Add-Title $s "下一步按“解决局限”的方向继续补论文" "阅读计划"
Add-Bullets $s @(
  "深度估计基础：Depth Anything V2、Depth Pro，用于理解单图深度从哪里来。",
  "单图/前馈 3DGS：pixelSplat、AnySplat、SHARP，关注不逐场景训练的生成方式。",
  "分层表达路线：3D Photo、SLIDE、TMPI、MINE，作为浅 3D 表达的另一条主线。",
  "渲染与质量优化：几何一致性、遮挡补全、透明/反光材质、实时内存控制。"
) 90 150 990 260 19 | Out-Null
Add-Card $s "后续 PPT 扩展方式" "每新增一篇论文，固定补：问题 → 方法 → 关键图 → 实验 → 局限 → 对课题启示。" 210 480 760 100
Add-Footer $s 11

$prs.SaveAs($outPptx)
$prs.Close()
$ppt.Quit()

Write-Output "DOCX=$outDocx"
Write-Output "PPTX=$outPptx"
`;

const encoded = Buffer.from(ps, "utf16le").toString("base64");
const result = spawnSync("powershell.exe", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded], {
  encoding: "utf8",
  stdio: ["ignore", "pipe", "pipe"],
});

process.stdout.write(result.stdout || "");
process.stderr.write(result.stderr || "");
if (result.status !== 0) process.exit(result.status);
