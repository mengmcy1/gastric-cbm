const { spawnSync } = require('child_process');
const path = require('path');

const root = path.resolve(__dirname, '..', '..');
const source = path.join(root, '汇报', '单张2D图像转浅3D表达_学习进度汇报.pptx');
const output = path.join(root, '汇报', '学习进度汇报7.10_四篇论文基础版.pptx');
const assets = path.join(root, '99_过程文件_可删除或归档', '渲染检查缓存', '18页版_assets_20260710');
const preview = path.join(root, '99_过程文件_可删除或归档', '渲染检查缓存', '学习进度汇报18页_20260710');

const psQuote = (value) => `'${value.replace(/'/g, "''")}'`;

const ps = String.raw`
$ErrorActionPreference = 'Stop'

$source = ${psQuote(source)}
$output = ${psQuote(output)}
$assets = ${psQuote(assets)}
$preview = ${psQuote(preview)}

function Rgb([int]$r, [int]$g, [int]$b) {
    return $r + (256 * $g) + (65536 * $b)
}

$blue = Rgb 13 86 154
$navy = Rgb 16 31 55
$cyan = Rgb 34 199 226
$green = Rgb 72 222 171
$orange = Rgb 255 168 55
$white = Rgb 255 255 255
$gray = Rgb 112 131 158
$light = Rgb 235 243 251

function Set-ShapeText($shape, [string]$text, [double]$fontSize, [int]$color, [bool]$bold = $false) {
    $shape.TextFrame2.TextRange.Text = $text
    $shape.TextFrame2.MarginLeft = 8
    $shape.TextFrame2.MarginRight = 8
    $shape.TextFrame2.MarginTop = 5
    $shape.TextFrame2.MarginBottom = 5
    $shape.TextFrame2.WordWrap = -1
    $shape.TextFrame2.AutoSize = 0
    $shape.TextFrame2.TextRange.Font.Name = 'Microsoft YaHei'
    $shape.TextFrame2.TextRange.Font.Size = $fontSize
    $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $color
    $shape.TextFrame2.TextRange.Font.Bold = $(if ($bold) { -1 } else { 0 })
}

function Add-Text($slide, [double]$x, [double]$y, [double]$w, [double]$h, [string]$text, [double]$fontSize, [int]$color, [bool]$bold = $false, [int]$align = 1) {
    $shape = $slide.Shapes.AddTextBox(1, $x, $y, $w, $h)
    Set-ShapeText $shape $text $fontSize $color $bold
    $shape.TextFrame2.TextRange.ParagraphFormat.Alignment = $align
    return $shape
}

function Add-Card($slide, [double]$x, [double]$y, [double]$w, [double]$h, [string]$title, [string]$body, [int]$accent) {
    $card = $slide.Shapes.AddShape(5, $x, $y, $w, $h)
    $card.Fill.ForeColor.RGB = $navy
    $card.Line.ForeColor.RGB = $blue
    $card.Line.Weight = 1.2
    try { $card.Reflection.Type = 0 } catch {}
    Add-Text $slide ($x + 12) ($y + 10) ($w - 24) 32 $title 19 $accent $true | Out-Null
    Add-Text $slide ($x + 12) ($y + 50) ($w - 24) ($h - 62) $body 14.5 $white $false | Out-Null
    return $card
}

function Add-Arrow($slide, [double]$x, [double]$y) {
    $arrow = $slide.Shapes.AddShape(33, $x, $y, 30, 24)
    $arrow.Fill.ForeColor.RGB = $cyan
    $arrow.Line.Visible = 0
    return $arrow
}

function Clear-Content($slide, [double]$slideHeight) {
    for ($i = $slide.Shapes.Count; $i -ge 1; $i--) {
        $shape = $slide.Shapes.Item($i)
        if ($shape.Top -gt 82 -and $shape.Top -lt ($slideHeight - 32)) {
            $shape.Delete()
        }
    }
}

function Set-Header($slide, [string]$title, [string]$subtitle) {
    $titleShape = $null
    $subtitleShape = $null
    foreach ($shape in @($slide.Shapes)) {
        try {
            if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame2.HasText -eq -1 -and $shape.Top -lt 82 -and $shape.Left -lt 760) {
                $size = $shape.TextFrame2.TextRange.Font.Size
                if ($size -ge 24 -and $null -eq $titleShape) {
                    $titleShape = $shape
                } elseif ($size -lt 24 -and $null -eq $subtitleShape) {
                    $subtitleShape = $shape
                }
            }
        } catch {}
    }
    if ($titleShape) {
        $titleShape.TextFrame2.TextRange.Text = $title
        $titleShape.TextFrame2.TextRange.Font.Name = 'Microsoft YaHei'
        $titleShape.TextFrame2.TextRange.Font.Bold = -1
        $titleShape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $blue
    }
    if ($subtitleShape) {
        $subtitleShape.TextFrame2.TextRange.Text = $subtitle
        $subtitleShape.TextFrame2.TextRange.Font.Name = 'Microsoft YaHei'
        $subtitleShape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $gray
    }
}

function Normalize-Slide($slide, [int]$pageNumber, [double]$slideWidth, [double]$slideHeight) {
    foreach ($shape in @($slide.Shapes)) {
        try { $shape.Reflection.Type = 0 } catch {}
        try {
            if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame2.HasText -eq -1) {
                $text = $shape.TextFrame2.TextRange.Text.Trim()
                if ($shape.Top -gt ($slideHeight - 42) -and $text -match '^\d+$') {
                    $shape.TextFrame2.TextRange.Text = [string]$pageNumber
                }
                if ($shape.Top -gt ($slideHeight - 65) -and $text -match '^(原论文|来源|论文原图)') {
                    $shape.Top = $slideHeight - 66
                }
            }
        } catch {}
    }
}

if (Test-Path -LiteralPath $output) {
    Remove-Item -LiteralPath $output -Force
}
New-Item -ItemType Directory -Path $preview -Force | Out-Null

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = -1
$sourceDeck = $null
$deck = $null

try {
    $sourceDeck = $ppt.Presentations.Open($source, 1, 0, 0)
    $sourceDeck.SaveCopyAs($output, 24)
    $sourceDeck.Close()
    $sourceDeck = $null

    $deck = $ppt.Presentations.Open($output, 0, 0, 0)
    $slideWidth = $deck.PageSetup.SlideWidth
    $slideHeight = $deck.PageSetup.SlideHeight

    $keep = @(1, 2, 4, 5, 6, 9, 10, 12, 13, 15, 18, 19, 20, 21, 22, 23, 26, 28)
    for ($i = $deck.Slides.Count; $i -ge 1; $i--) {
        if ($keep -notcontains $i) {
            $deck.Slides.Item($i).Delete()
        }
    }

    # Put the process overview before the soft-visibility detail slide.
    $deck.Slides.Item(15).MoveTo(14)

    $titles = @(
        '单张 2D 图像转浅 3D 表达研究进展',
        '本阶段完成的工作',
        '四篇论文逐步接近本课题目标',
        'NeRF 用神经网络表示连续三维场景',
        'NeRF 的三个关键设计',
        '3DGS 用显式高斯椭球表示场景',
        '3DGS 为什么能实现实时渲染',
        'NeRF 与 3DGS 仍不适合直接完成本课题',
        '单图浅 3D 的核心困难是显露区域',
        '3D Photography 建立完整的单图浅 3D 流程',
        '围绕深度边界进行局部遮挡补全',
        '3D Photography 与移动端目标高度一致',
        'SLIDE 重点解决硬分层损伤细结构的问题',
        'SLIDE 使用轻量的两层表达',
        '软可见性和深度感知补全各解决一个问题',
        '3D Photography 与 SLIDE 的取舍',
        '本课题可以采用模块化浅 3D 管线',
        '下一步工作'
    )
    $subtitles = @(
        'NeRF、3D Gaussian Splatting、3D Photography 与 SLIDE',
        '从新视角合成基础，逐步收敛到单图浅 3D',
        '基础表示 → 实时渲染 → 遮挡补全 → 软分层',
        'Neural Radiance Fields，神经辐射场',
        '位置编码、分层采样和连续隐式表示',
        'Three-Dimensional Gaussian Splatting，三维高斯泼溅',
        '从沿光线查询网络，转向屏幕空间并行光栅化',
        '共同局限仍是多图、位姿和逐场景优化',
        '新视角会露出原图中不存在的背景内容',
        'RGB-D → LDI → 边界补全 → Textured Mesh',
        '哪里会露出来，就围绕哪里的深度边界补全',
        '生成阶段做深度与补全，交互阶段只做轻量渲染',
        '头发、栏杆、树枝等不适合简单二值切割',
        'RGB → disparity → soft layering → RGBD inpainting → rendering',
        'A 管前景如何显示，S 管背景哪里需要补',
        'LDI 擅长多层遮挡，Soft two-layer 擅长细结构和效率',
        '把生成与交互渲染拆开，优先完成可解释原型',
        '继续补齐表示方法，并开始建立可验证的实验基线'
    )

    for ($i = 2; $i -le 18; $i++) {
        Set-Header $deck.Slides.Item($i) $titles[$i - 1] $subtitles[$i - 1]
    }

    # Cover: move presenter/date inside the blue field so they are not clipped.
    $cover = $deck.Slides.Item(1)
    foreach ($shape in @($cover.Shapes)) {
        try {
            if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame2.HasText -eq -1) {
                $text = $shape.TextFrame2.TextRange.Text.Trim()
                if ($text -like '汇报人*') { $shape.Top = 414; $shape.Left = 58 }
                if ($text -like '时间*') { $shape.Top = 438; $shape.Left = 58 }
            }
        } catch {}
        try { $shape.Reflection.Type = 0 } catch {}
    }

    # Slide 2: four-paper progress overview.
    $s = $deck.Slides.Item(2)
    Clear-Content $s $slideHeight
    $paperImages = @('image3.png', 'image5.png', 'image2.png', 'image9.png')
    $paperNames = @('NeRF', '3D Gaussian Splatting', '3D Photography', 'SLIDE')
    $paperNotes = @(
        '连续神经场与体积渲染基础',
        '显式高斯与实时渲染',
        '单图显露区域与 LDI 补全',
        '软分层与细结构保真'
    )
    $xs = @(44, 273, 502, 731)
    for ($j = 0; $j -lt 4; $j++) {
        $img = Join-Path $assets $paperImages[$j]
        $pic = $s.Shapes.AddPicture($img, 0, -1, $xs[$j], 112, 145, 188)
        $pic.Line.Visible = -1
        $pic.Line.ForeColor.RGB = $light
        Add-Text $s ($xs[$j] - 18) 310 181 30 $paperNames[$j] 17 $blue $true 2 | Out-Null
        Add-Text $s ($xs[$j] - 22) 346 189 55 $paperNotes[$j] 12.5 $gray $false 2 | Out-Null
    }
    Add-Text $s 112 428 736 36 '已完成：两篇基础论文 + 两篇单图浅 3D 核心论文' 19 $cyan $true 2 | Out-Null

    # Slide 3: narrative progression.
    $s = $deck.Slides.Item(3)
    Clear-Content $s $slideHeight
    $progressTitles = @('NeRF', '3DGS', '3D Photography', 'SLIDE')
    $progressBodies = @(
        '连续场景表示\n体积渲染',
        '显式高斯表示\n实时光栅化',
        '显露区域补全\nLDI 与 Mesh',
        '软可见性\n细结构与效率'
    )
    $accents = @($cyan, $green, $orange, $green)
    $px = @(45, 275, 505, 735)
    for ($j = 0; $j -lt 4; $j++) {
        Add-Card $s $px[$j] 150 180 215 $progressTitles[$j] $progressBodies[$j] $accents[$j] | Out-Null
        if ($j -lt 3) { Add-Arrow $s ($px[$j] + 194) 245 | Out-Null }
    }
    Add-Text $s 95 405 770 42 '研究重点逐步从“多图场景表示”转向“单图可部署的浅 3D 表达”' 18 $blue $true 2 | Out-Null

    # Slide 7: add the representative speed comparison from the original papers.
    $s = $deck.Slides.Item(7)
    Add-Text $s 330 446 300 30 '代表结果：135 fps  vs  0.07 fps' 17 $cyan $true 2 | Out-Null

    # Slide 15: complement A/S with matting and RGBD inpainting.
    $s = $deck.Slides.Item(15)
    Add-Text $s 505 402 370 54 'Alpha Matting 补细边界；RGBD Inpainting 同时补颜色和视差。' 14 $orange $true 2 | Out-Null

    # Slide 17: clean project pipeline.
    $s = $deck.Slides.Item(17)
    Clear-Content $s $slideHeight
    $stageTitles = @('阶段 1', '阶段 2', '阶段 3', '阶段 4')
    $stagePipes = @(
        'RGB → 深度 → 点云 / 简化 Mesh',
        '边界增强 → 背景补全 → 两层 Mesh',
        'LDI / TMPI / 轻量 3DGS 对比',
        'Android / OpenGL ES / Vulkan'
    )
    $stageGoals = @(
        '先验证运动视差是否成立',
        '解决空洞、拉伸和细结构问题',
        '比较质量、内存和渲染速度',
        '生成一次，交互阶段只做图形渲染'
    )
    for ($j = 0; $j -lt 4; $j++) {
        $y = 112 + ($j * 91)
        $tag = $s.Shapes.AddShape(5, 58, $y, 120, 67)
        $tag.Fill.ForeColor.RGB = $navy
        $tag.Line.ForeColor.RGB = $blue
        Set-ShapeText $tag $stageTitles[$j] 18 $(if ($j -lt 2) { $cyan } else { $orange }) $true
        $pipe = $s.Shapes.AddShape(5, 200, $y, 405, 67)
        $pipe.Fill.ForeColor.RGB = $light
        $pipe.Line.ForeColor.RGB = $blue
        Set-ShapeText $pipe $stagePipes[$j] 17 $blue $true
        Add-Text $s 630 ($y + 9) 272 50 $stageGoals[$j] 13.5 $gray $false | Out-Null
    }

    # Slide 18: concrete next work.
    $s = $deck.Slides.Item(18)
    Clear-Content $s $slideHeight
    Add-Card $s 60 132 400 300 '论文阅读' "TMPI：规则平面怎样降低冗余\n\nMINE：离散深度层与连续深度表示如何取舍\n\n继续补充单图前馈 3DGS 方法" $cyan | Out-Null
    Add-Card $s 500 132 400 300 '实验验证' "测试 Depth Anything V2、Depth Pro 的边界质量\n\n建立空洞、拉伸、重影、细线断裂失败案例集\n\n完成“深度 → Mesh 重投影”基础实验" $orange | Out-Null
    Add-Text $s 145 455 670 28 '下一阶段目标：从“理解论文”进入“建立可比较的浅 3D 原型”' 17 $blue $true 2 | Out-Null

    for ($i = 1; $i -le $deck.Slides.Count; $i++) {
        Normalize-Slide $deck.Slides.Item($i) $i $slideWidth $slideHeight
    }

    $deck.Save()
    $deck.Export($preview, 'PNG', 1600, 900)
    Write-Output ('OUTPUT=' + $output)
    Write-Output ('SLIDES=' + $deck.Slides.Count)
    Write-Output ('PREVIEW=' + $preview)
}
finally {
    if ($deck) { $deck.Close() }
    if ($sourceDeck) { $sourceDeck.Close() }
    $ppt.Quit()
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($ppt) | Out-Null
}
`;

const encoded = Buffer.from(ps, 'utf16le').toString('base64');
const result = spawnSync(
  'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe',
  ['-NoProfile', '-EncodedCommand', encoded],
  { cwd: root, encoding: 'utf8', timeout: 180000 }
);

if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
if (result.error) throw result.error;
process.exit(result.status ?? 1);
