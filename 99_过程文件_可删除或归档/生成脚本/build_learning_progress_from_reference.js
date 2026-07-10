const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..', '..');
const template = path.join(root, 'Review of ICCV 2025 Papers on 3D Gaussian Splatting.pptx');
const output = path.join(root, '汇报', '学习进度汇报7.10_四篇论文基础版.pptx');
const pages = path.join(root, '99_过程文件_可删除或归档', '渲染检查缓存', '论文页预览_20260710');
const crops = path.join(root, '99_过程文件_可删除或归档', '渲染检查缓存', '论文配图裁剪_20260710');
const preview = path.join(root, '99_过程文件_可删除或归档', '渲染检查缓存', '学习进度汇报18页_20260710');

const q = (value) => `'${value.replace(/'/g, "''")}'`;

const ps = String.raw`
$ErrorActionPreference = 'Stop'
$template = ${q(template)}
$output = ${q(output)}
$pages = ${q(pages)}
$crops = ${q(crops)}
$preview = ${q(preview)}

function Rgb([int]$r, [int]$g, [int]$b) { $r + 256 * $g + 65536 * $b }
$blue = Rgb 13 86 154
$navy = Rgb 16 31 55
$cyan = Rgb 34 199 226
$green = Rgb 72 222 171
$orange = Rgb 255 168 55
$white = Rgb 255 255 255
$gray = Rgb 102 124 151
$light = Rgb 238 245 251
$line = Rgb 130 180 224
$red = Rgb 221 57 45

New-Item -ItemType Directory -Path $crops -Force | Out-Null
New-Item -ItemType Directory -Path $preview -Force | Out-Null

Add-Type -AssemblyName System.Drawing
function Crop-Image([string]$source, [string]$dest, [int]$x, [int]$y, [int]$w, [int]$h) {
    $img = [Drawing.Image]::FromFile($source)
    try {
        if ($x + $w -gt $img.Width) { $w = $img.Width - $x }
        if ($y + $h -gt $img.Height) { $h = $img.Height - $y }
        $bmp = New-Object Drawing.Bitmap($w, $h)
        try {
            $g = [Drawing.Graphics]::FromImage($bmp)
            try {
                $g.Clear([Drawing.Color]::White)
                $g.InterpolationMode = [Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
                $g.DrawImage($img, (New-Object Drawing.Rectangle(0, 0, $w, $h)), (New-Object Drawing.Rectangle($x, $y, $w, $h)), [Drawing.GraphicsUnit]::Pixel)
            } finally { $g.Dispose() }
            $bmp.Save($dest, [Drawing.Imaging.ImageFormat]::Png)
        } finally { $bmp.Dispose() }
    } finally { $img.Dispose() }
}

# Each crop contains a paper figure, comparison, or formula region only.
Crop-Image (Join-Path $pages 'nerf-02.png') (Join-Path $crops 'nerf_task.png') 42 32 770 150
Crop-Image (Join-Path $pages 'nerf-05.png') (Join-Path $crops 'nerf_method.png') 42 35 770 165
Crop-Image (Join-Path $pages '3dgs-01.png') (Join-Path $crops '3dgs_speed.png') 62 220 728 180
Crop-Image (Join-Path $pages '3dgs-05.png') (Join-Path $crops '3dgs_pipeline.png') 65 82 720 145
Crop-Image (Join-Path $pages '3dgs-06.png') (Join-Path $crops '3dgs_density.png') 63 72 350 330
Crop-Image (Join-Path $pages '3dphoto-01.png') (Join-Path $crops '3dphoto_compare.png') 65 292 700 205
Crop-Image (Join-Path $pages '3dphoto-04.png') (Join-Path $crops '3dphoto_ldi.png') 62 330 725 145
Crop-Image (Join-Path $pages '3dphoto-06.png') (Join-Path $crops '3dphoto_inpaint.png') 62 45 725 315
Crop-Image (Join-Path $pages 'slide-01.png') (Join-Path $crops 'slide_soft.png') 70 340 700 100
Crop-Image (Join-Path $pages 'slide-03.png') (Join-Path $crops 'slide_pipeline.png') 62 42 725 275
Crop-Image (Join-Path $pages 'slide-05.png') (Join-Path $crops 'slide_alpha.png') 68 45 710 130

function Set-Text($shape, [string]$text, [double]$size, [int]$color, [bool]$bold = $false, [int]$align = 1) {
    $text = $text.Replace('\n', [Environment]::NewLine)
    $shape.TextFrame2.TextRange.Text = $text
    $shape.TextFrame2.MarginLeft = 7
    $shape.TextFrame2.MarginRight = 7
    $shape.TextFrame2.MarginTop = 4
    $shape.TextFrame2.MarginBottom = 4
    $shape.TextFrame2.WordWrap = -1
    $shape.TextFrame2.AutoSize = 0
    $shape.TextFrame2.TextRange.Font.Name = 'Microsoft YaHei'
    $shape.TextFrame2.TextRange.Font.Size = $size
    $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $color
    $shape.TextFrame2.TextRange.Font.Bold = $(if ($bold) { -1 } else { 0 })
    $shape.TextFrame2.TextRange.ParagraphFormat.Alignment = $align
}

function Add-Text($slide, [double]$x, [double]$y, [double]$w, [double]$h, [string]$text, [double]$size, [int]$color, [bool]$bold = $false, [int]$align = 1) {
    try {
        $shape = $slide.Shapes.AddTextBox(1, $x, $y, $w, $h)
    } catch {
        throw ('AddTextBox failed: slide=' + $slide.SlideIndex + ', x=' + $x + ', y=' + $y + ', w=' + $w + ', h=' + $h + ', text=' + $text)
    }
    Set-Text $shape $text $size $color $bold $align
    $shape
}

function Add-Card($slide, [double]$x, [double]$y, [double]$w, [double]$h, [string]$title, [string]$body, [int]$accent, [double]$bodySize = 15) {
    $card = $slide.Shapes.AddShape(5, $x, $y, $w, $h)
    $card.Fill.ForeColor.RGB = $navy
    $card.Line.ForeColor.RGB = $blue
    $card.Line.Weight = 1.1
    Add-Text $slide ($x + 12) ($y + 8) ($w - 24) 42 $title 18 $accent $true | Out-Null
    Add-Text $slide ($x + 12) ($y + 55) ($w - 24) ($h - 65) $body $bodySize $white $false | Out-Null
    $card
}

function Add-LightCard($slide, [double]$x, [double]$y, [double]$w, [double]$h, [string]$title, [string]$body, [int]$accent) {
    $card = $slide.Shapes.AddShape(5, $x, $y, $w, $h)
    $card.Fill.ForeColor.RGB = $light
    $card.Line.ForeColor.RGB = $line
    $card.Line.Weight = 1
    if ($h -lt 75) {
        Add-Text $slide ($x + 10) ($y + 7) ($w - 20) ($h - 14) ($title + '　' + $body) 13.5 $accent $true 2 | Out-Null
        return $card
    }
    Add-Text $slide ($x + 10) ($y + 8) ($w - 20) 28 $title 17 $accent $true | Out-Null
    Add-Text $slide ($x + 10) ($y + 42) ($w - 20) ($h - 50) $body 13.5 $navy $false | Out-Null
    $card
}

function Add-Title($slide, [string]$title, [string]$subtitle, [int]$page) {
    Add-Text $slide 50 16 830 37 $title 25.5 $blue $true | Out-Null
    Add-Text $slide 52 53 780 19 $subtitle 10.5 $gray $false | Out-Null
}

function Add-Source($slide, [string]$source) {
    Add-Text $slide 55 488 850 20 ('图源：' + $source) 8.5 $gray $false | Out-Null
}

function Add-PictureFit($slide, [string]$file, [double]$x, [double]$y, [double]$w, [double]$h) {
    $pic = $slide.Shapes.AddPicture($file, 0, -1, $x, $y, -1, -1)
    $pic.LockAspectRatio = -1
    $scale = [Math]::Min($w / $pic.Width, $h / $pic.Height)
    $pic.Width = $pic.Width * $scale
    $pic.Height = $pic.Height * $scale
    $pic.Left = $x + (($w - $pic.Width) / 2)
    $pic.Top = $y + (($h - $pic.Height) / 2)
    $pic.Line.Visible = -1
    $pic.Line.ForeColor.RGB = $line
    $pic.Line.Weight = 0.8
    $pic
}

function Add-PictureStretch($slide, [string]$file, [double]$x, [double]$y, [double]$w, [double]$h) {
    $pic = $slide.Shapes.AddPicture($file, 0, -1, $x, $y, $w, $h)
    $pic.Line.Visible = -1
    $pic.Line.ForeColor.RGB = $line
    $pic.Line.Weight = 0.8
    $pic
}

function Add-Arrow($slide, [double]$x, [double]$y, [double]$w = 30, [double]$h = 22) {
    $shape = $slide.Shapes.AddShape(33, $x, $y, $w, $h)
    $shape.Fill.ForeColor.RGB = $cyan
    $shape.Line.Visible = 0
    $shape
}

function Clear-Slide($slide) {
    for ($i = $slide.Shapes.Count; $i -ge 1; $i--) { $slide.Shapes.Item($i).Delete() }
}

if (Test-Path -LiteralPath $output) { Remove-Item -LiteralPath $output -Force }

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = -1
$templateDeck = $null
$deck = $null

try {
    $templateDeck = $ppt.Presentations.Open($template, 1, 0, 0)
    $templateDeck.SaveCopyAs($output, 24)
    $templateDeck.Close()
    $templateDeck = $null

    $deck = $ppt.Presentations.Open($output, 0, 0, 0)
    for ($i = $deck.Slides.Count; $i -ge 3; $i--) { $deck.Slides.Item($i).Delete() }
    for ($i = 1; $i -le 16; $i++) { $deck.Slides.Item($deck.Slides.Count).Duplicate() | Out-Null }

    # Cover uses the reference deck's original cover layout.
    $cover = $deck.Slides.Item(1)
    foreach ($shape in @($cover.Shapes)) {
        try {
            if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame2.HasText -eq -1) {
                $text = $shape.TextFrame2.TextRange.Text.Trim()
                $size = $shape.TextFrame2.TextRange.Font.Size
                if ($size -ge 28) {
                    $shape.TextFrame2.TextRange.Text = '单张 2D 图像转浅 3D 表达' + [Environment]::NewLine + '研究进展'
                    $shape.TextFrame2.TextRange.Font.Size = 35
                } elseif ($text -like '汇报人*') {
                    $shape.TextFrame2.TextRange.Text = '汇报人：郭浩杰'
                } elseif ($text -like '时间*') {
                    $shape.TextFrame2.TextRange.Text = '时间：2026.7.10'
                }
            }
        } catch {}
    }

    for ($i = 2; $i -le 18; $i++) { Clear-Slide $deck.Slides.Item($i) }

    # 2. Progress overview.
    $s = $deck.Slides.Item(2)
    Add-Title $s '本阶段完成的工作' '两篇新视角合成基石论文 + 两篇单图浅 3D 核心论文' 2
    $thumbs = @('nerf_task.png','3dgs_speed.png','3dphoto_compare.png','slide_soft.png')
    $names = @('NeRF','3D Gaussian Splatting','3D Photography','SLIDE')
    $notes = @('连续场与体积渲染','显式高斯与实时渲染','显露区域与 LDI 补全','软分层与细结构')
    $xs = @(45, 275, 505, 735)
    for ($j = 0; $j -lt 4; $j++) {
        Add-PictureStretch $s (Join-Path $crops $thumbs[$j]) $xs[$j] 145 180 92 | Out-Null
        Add-Text $s $xs[$j] 260 180 26 $names[$j] 16.5 $blue $true 2 | Out-Null
        Add-Text $s $xs[$j] 296 180 45 $notes[$j] 12.5 $gray $false 2 | Out-Null
    }
    Add-LightCard $s 110 390 740 70 '阶段结果' '已形成“基础表示 → 实时渲染 → 遮挡补全 → 软分层”的知识链路。' $cyan | Out-Null
    Add-Source $s 'NeRF（ECCV 2020）、3DGS（SIGGRAPH 2023）、3D Photography（CVPR 2020）、SLIDE（2021）'

    # 3. Paper progression.
    $s = $deck.Slides.Item(3)
    Add-Title $s '四篇论文逐步接近本课题目标' '研究脉络：场景表示、渲染效率、遮挡补全与细结构建模' 3
    $pt = @('NeRF','3DGS','3D Photography','SLIDE')
    $pb = @('连续三维场景表示\n体积渲染基础','显式高斯表示\n实现实时渲染','单图显露区域补全\nLDI 与 Mesh','软可见性\n保留头发与细线')
    $pa = @($cyan,$green,$orange,$green)
    $px = @(45,275,505,735)
    for ($j = 0; $j -lt 4; $j++) {
        Add-Card $s $px[$j] 145 180 220 $pt[$j] $pb[$j] $pa[$j] 14 | Out-Null
        if ($j -lt 3) { Add-Arrow $s ($px[$j] + 195) 242 | Out-Null }
    }
    Add-Text $s 90 405 780 42 '研究重点由多视图场景表示逐步转向单图浅 3D 表达与移动端部署' 18 $blue $true 2 | Out-Null

    # 4. NeRF representation.
    $s = $deck.Slides.Item(4)
    Add-Title $s 'NeRF 用神经网络表示连续三维场景' 'Neural Radiance Fields，神经辐射场' 4
    Add-PictureFit $s (Join-Path $crops 'nerf_method.png') 45 115 470 300 | Out-Null
    Add-Card $s 540 112 370 305 'Method Overview' "采用 MLP 参数化连续 5D 辐射场。\n\n输入：三维位置 + 观察方向\n输出：颜色 RGB + 体积密度 σ\n\n沿相机光线采样，并通过可微体积渲染合成像素。" $cyan 15 | Out-Null
    Add-Source $s 'NeRF 原论文 Fig. 2'

    # 5. NeRF key designs.
    $s = $deck.Slides.Item(5)
    Add-Title $s 'NeRF 的三个关键设计' '连续表示、位置编码和分层采样共同保证质量' 5
    Add-Card $s 45 120 270 250 '① MLP 表示' '用网络参数存储连续场景。\n\n优点：表达连续、存储紧凑。\n代价：每个采样点都要查询网络。' $cyan 13.8 | Out-Null
    Add-Card $s 345 120 270 250 '② 位置编码' '将坐标映射到多频率 sin/cos 基底，以增强 MLP 对高频纹理、细线和边缘的表达能力。' $green 13.8 | Out-Null
    Add-Card $s 645 120 270 250 '③ 分层采样' '粗网络先定位物体，细网络在高密度区域增加采样，减少空白区域计算。' $orange 13.8 | Out-Null
    Add-LightCard $s 180 405 600 60 '位置编码公式' 'PE(p) = [sin(p), cos(p), sin(2p), cos(2p), …]' $blue | Out-Null
    Add-Source $s 'NeRF Sections 3–5；公式按原文简化排版'

    # 6. 3DGS representation.
    $s = $deck.Slides.Item(6)
    Add-Title $s '3DGS 用显式高斯椭球表示场景' '每个高斯直接保存位置、大小、方向、颜色和透明度' 6
    Add-PictureFit $s (Join-Path $crops '3dgs_pipeline.png') 45 115 500 250 | Out-Null
    Add-Card $s 570 112 340 300 '显式高斯' "位置 (x,y,z)\n缩放 s 与旋转 q\n不透明度 α\n球谐颜色 SH\n\n渲染时直接投影到屏幕，再进行 Alpha 混合。" $green 14.5 | Out-Null
    Add-LightCard $s 115 400 730 60 '协方差参数化' 'Σ = R S S^T R^T　→　保证形状可优化，并能投影成屏幕上的椭圆斑。' $blue | Out-Null
    Add-Source $s '3D Gaussian Splatting 原论文 Fig. 2 与 Eq. 6'

    # 7. Why 3DGS is fast.
    $s = $deck.Slides.Item(7)
    Add-Title $s '3DGS 的实时渲染机制' '由逐射线神经查询转向屏幕空间可微光栅化' 7
    Add-PictureFit $s (Join-Path $crops '3dgs_speed.png') 80 100 800 150 | Out-Null
    Add-Card $s 55 275 380 175 'NeRF：光线步进' '每根光线采样约 192 个点\n每个点查询 MLP\n串行累积颜色和密度' $orange 14 | Out-Null
    Add-Card $s 525 275 380 175 '3DGS：屏幕空间光栅化' '高斯投影到屏幕\n按深度排序\nGPU 并行 Alpha 混合' $green 14 | Out-Null
    Add-Text $s 375 251 210 22 '速度对比：135 fps vs 0.07 fps' 12.5 $cyan $true 2 | Out-Null
    Add-Source $s '3DGS 原论文 Fig. 1；速度为论文代表性场景结果'

    # 8. Comparison and gap.
    $s = $deck.Slides.Item(8)
    Add-Title $s 'NeRF 与 3DGS 仍不适合直接完成本课题' '两者解决“多图 → 新视角”，尚未解决“单图 → 浅 3D”' 8
    Add-Card $s 45 130 270 285 '共同要求' '需要多张输入图\n需要相机位姿\n依赖多视图几何约束\n新场景仍需优化' $cyan 15 | Out-Null
    Add-Card $s 345 130 270 285 'NeRF' '隐式 MLP\n连续表达\n存储相对紧凑\n训练和渲染较慢' $orange 15 | Out-Null
    Add-Card $s 645 130 270 285 '3DGS' '显式高斯\nGPU 并行\n渲染实时\n显式存储较大' $green 15 | Out-Null
    Add-LightCard $s 130 435 700 48 '关键差异' '本课题输入为单张普通 RGB 图像，不具备多视图观测与相机位姿。' $red | Out-Null

    # 9. Disocclusion.
    $s = $deck.Slides.Item(9)
    Add-Title $s '单图浅 3D 的核心困难是显露区域' '新视角会露出原图中根本不存在的背景内容' 9
    Add-PictureFit $s (Join-Path $crops '3dphoto_compare.png') 45 115 500 285 | Out-Null
    Add-Card $s 570 112 340 305 'Disocclusion Problem' "视点移动后，原本被前景遮挡的背景区域重新可见。\n\n直接重投影 → 空洞\n邻域像素扩展 → 模糊、撕裂与结构失真\n\n因此需要联合补全颜色与几何。" $orange 14.5 | Out-Null
    Add-Source $s '3D Photography 原论文 Fig. 1'

    # 10. 3D Photography pipeline.
    $s = $deck.Slides.Item(10)
    Add-Title $s '3D Photography 建立完整的单图浅 3D 流程' 'RGB-D → LDI → 深度边界补全 → Textured Mesh' 10
    Add-PictureFit $s (Join-Path $crops '3dphoto_ldi.png') 120 105 720 215 | Out-Null
    $steps = @('RGB-D 输入','深度预处理','构建 LDI','检测边界','补颜色 + 深度','Textured Mesh','实时渲染')
    for ($j = 0; $j -lt 7; $j++) {
        $x = 35 + ($j * 132)
        $box = $s.Shapes.AddShape(5, $x, 360, 112, 58)
        $box.Fill.ForeColor.RGB = $(if ($j -lt 4) { $light } else { $navy })
        $box.Line.ForeColor.RGB = $line
        Set-Text $box $steps[$j] 12.5 $(if ($j -lt 4) { $blue } else { $white }) $true 2
        if ($j -lt 6) { Add-Arrow $s ($x + 113) 378 20 18 | Out-Null }
    }
    Add-Text $s 140 440 680 30 'LDI 允许同一图像位置保存前景、背景等多个颜色和深度样本。' 15 $blue $true 2 | Out-Null
    Add-Source $s '3D Photography 原论文 Fig. 3'

    # 11. Local inpainting.
    $s = $deck.Slides.Item(11)
    Add-Title $s '围绕深度边界进行局部遮挡补全' '基于深度不连续边界构建局部上下文区域与待合成区域' 11
    Add-PictureFit $s (Join-Path $crops '3dphoto_inpaint.png') 40 115 510 300 | Out-Null
    Add-Card $s 575 110 335 315 '结构引导的补全顺序' "① 先补 depth edge\n② 再补背景 color\n③ 再补背景 depth\n④ 复杂遮挡继续迭代\n\n共同的结构线索让 RGB 与 depth 更一致。" $green 14.5 | Out-Null
    Add-Source $s '3D Photography 原论文 Fig. 6'

    # 12. Mobile relevance.
    $s = $deck.Slides.Item(12)
    Add-Title $s '3D Photography 对移动端方案的启示' '生成阶段执行深度与补全，交互阶段采用显式表达渲染' 12
    Add-LightCard $s 55 125 200 250 '深度估计' '普通 RGB → 相对深度或视差\n\n候选：Depth Anything V2、Depth Pro' $cyan | Out-Null
    Add-LightCard $s 275 125 200 250 '分层表示' 'LDI、简化 Mesh、TMPI 或轻量 3DGS\n\n按内存和速度取舍' $green | Out-Null
    Add-LightCard $s 495 125 200 250 '遮挡补全' '围绕深度不连续边界\n\n同时补颜色和深度' $orange | Out-Null
    Add-LightCard $s 715 125 200 250 '实时渲染' '离线生成场景表达\n\n在线阶段仅更新视点并渲染' $blue | Out-Null
    Add-LightCard $s 125 405 710 48 '路线判断' '耗时模块置于前处理阶段；移动端仅执行显式表达的轻量渲染。' $cyan | Out-Null

    # 13. SLIDE problem.
    $s = $deck.Slides.Item(13)
    Add-Title $s 'SLIDE 重点解决硬分层损伤细结构的问题' '头发、栏杆、桥索和树枝不是简单的前景/背景二值切割' 13
    Add-PictureFit $s (Join-Path $crops 'slide_soft.png') 45 120 500 270 | Out-Null
    Add-Card $s 570 115 340 300 'Problem Formulation' "细结构边缘像素通常混合前景与背景外观。\n\n硬分层容易导致：\n结构断裂、细节缺失与边界撕裂。\n\nSLIDE 使用连续 soft visibility 建模边界可见性。" $orange 14.5 | Out-Null
    Add-Source $s 'SLIDE 原论文 Fig. 1'

    # 14. SLIDE pipeline.
    $s = $deck.Slides.Item(14)
    Add-Title $s 'SLIDE 使用轻量的两层表达' '一次前向生成前景层和补全后的背景层' 14
    Add-PictureFit $s (Join-Path $crops 'slide_pipeline.png') 95 95 770 245 | Out-Null
    Add-LightCard $s 55 365 270 90 '输入与深度' 'RGB → MiDaS disparity\n轻微平滑 + max-pool' $cyan | Out-Null
    Add-LightCard $s 345 365 270 90 '两层表达' '前景：RGB + disparity + A\n背景：补全 RGB + disparity' $green | Out-Null
    Add-LightCard $s 635 365 270 90 '分层渲染' '两层反投影为 Mesh\n新视角阶段执行投影与 Alpha 合成' $orange | Out-Null
    Add-Source $s 'SLIDE 原论文 Fig. 2'

    # 15. Soft visibility and inpainting.
    $s = $deck.Slides.Item(15)
    Add-Title $s '软可见性和深度感知补全各解决一个问题' '前景软可见性 A 与软显露区域 S 的功能分工' 15
    Add-PictureFit $s (Join-Path $crops 'slide_alpha.png') 45 120 500 210 | Out-Null
    Add-LightCard $s 55 350 470 110 '软分层公式' 'A = exp(-β · ||∇D||²)\nS：由邻域视差差异得到的连续显露区域图' $blue | Out-Null
    Add-Card $s 570 115 340 345 '辅助信息与补全目标' "Alpha Matting\n补充深度图难以捕捉的头发与细线边界。\n\nRGBD Inpainting\n联合补全背景颜色与视差，缓解颜色和几何不一致。" $green 14.2 | Out-Null
    Add-Source $s 'SLIDE 原论文 Fig. 5 与 Eqs. 1–3；公式按原文简化排版'

    # 16. 3D Photography vs SLIDE.
    $s = $deck.Slides.Item(16)
    Add-Title $s '3D Photography 与 SLIDE 的方法对比' '表示能力、边界建模与计算效率的权衡' 16
    Add-Card $s 55 125 405 315 '3D Photography / LDI' "输入：RGB-D\n表示：多层 LDI\n补全：局部迭代\n\n优势：遮挡层次更细，可处理复杂多层深度。\n局限：流程复杂，硬分层对细结构不够友好。" $cyan 14 | Out-Null
    Add-Card $s 500 125 405 315 'SLIDE / Soft two-layer' "输入：普通 RGB\n表示：前景 + 背景两层\n补全：一次前向\n\n优势：细结构更稳定、速度更快。\n局限：复杂多层遮挡表达能力有限。" $green 14 | Out-Null
    Add-LightCard $s 160 460 640 38 '本课题可借鉴方案' '显著深度边界采用显式切分；头发、桥索等细结构采用 soft visibility。' $orange | Out-Null

    # 17. Project pipeline.
    $s = $deck.Slides.Item(17)
    Add-Title $s '本课题可以采用模块化浅 3D 管线' '生成阶段与交互渲染阶段分离' 17
    $nodes = @('普通 RGB','单目深度','边界增强','分层 + 补全','显式表达','移动端渲染')
    $nodeNotes = @('一张照片','相对深度/视差','深度 + RGB 边界','硬边界 + 软可见性','LDI / Mesh / TMPI','小幅视角变化')
    for ($j = 0; $j -lt 6; $j++) {
        $x = 35 + ($j * 153)
        $box = $s.Shapes.AddShape(5, $x, 155, 128, 125)
        $box.Fill.ForeColor.RGB = $(if ($j -lt 4) { $light } else { $navy })
        $box.Line.ForeColor.RGB = $line
        Add-Text $s ($x + 5) 170 118 30 $nodes[$j] 15 $(if ($j -lt 4) { $blue } else { $white }) $true 2 | Out-Null
        Add-Text $s ($x + 7) 215 114 45 $nodeNotes[$j] 11.5 $(if ($j -lt 4) { $gray } else { $cyan }) $false 2 | Out-Null
        if ($j -lt 5) { Add-Arrow $s ($x + 132) 205 21 18 | Out-Null }
    }
    Add-Card $s 90 330 780 120 'Implementation Stages' "首先实现 RGB → 深度 → Mesh 重投影，验证运动视差；\n随后加入边界增强、背景补全和两层表达；最后比较 LDI、TMPI 与轻量 3DGS。" $orange 14.5 | Out-Null

    # 18. Next work.
    $s = $deck.Slides.Item(18)
    Add-Title $s '下一步工作' '继续补齐表示方法，同时开始建立可比较的实验基线' 18
    Add-Card $s 60 130 400 310 '论文阅读' "TMPI\n规则 MPI 怎样通过局部 tile 降低冗余？\n\nMINE\n离散深度层与连续深度表示如何取舍？\n\n继续关注单图前馈 3DGS。" $cyan 14 | Out-Null
    Add-Card $s 500 130 400 310 '实验验证' "测试 Depth Anything V2、Depth Pro 的边界质量。\n\n建立空洞、拉伸、重影、细线断裂失败案例集。\n\n完成深度 → Mesh 重投影基础实验。" $orange 14 | Out-Null
    Add-LightCard $s 145 462 670 38 '下一阶段目标' '由文献调研转入可验证原型构建与对比实验。' $blue | Out-Null

    $deck.Save()
    $deck.Export($preview, 'PNG', 1600, 900)
    Write-Output ('OUTPUT=' + $output)
    Write-Output ('SLIDES=' + $deck.Slides.Count)
    Write-Output ('PREVIEW=' + $preview)
}
finally {
    if ($deck) { $deck.Close() }
    if ($templateDeck) { $templateDeck.Close() }
    $ppt.Quit()
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($ppt) | Out-Null
}
`;

const generatedPs1 = path.join(__dirname, 'build_learning_progress_from_reference.generated.ps1');
fs.writeFileSync(generatedPs1, '\uFEFF' + ps, { encoding: 'utf16le' });
const result = spawnSync(
  'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe',
  ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', generatedPs1],
  { cwd: root, encoding: 'utf8', timeout: 240000 }
);

if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
if (result.error) throw result.error;
process.exit(result.status ?? 1);
