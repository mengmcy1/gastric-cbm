param(
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [Parameter(Mandatory = $true)][string]$OutputDir
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

if (-not (Test-Path -LiteralPath $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$SourceRoot = (Resolve-Path -LiteralPath $SourceRoot).Path
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

function Export-Crop {
    param(
        [string]$Source,
        [string]$Name,
        [double]$Left,
        [double]$Top,
        [double]$Right,
        [double]$Bottom
    )

    $image = [System.Drawing.Image]::FromFile($Source)
    try {
        $x = [int]($image.Width * $Left)
        $y = [int]($image.Height * $Top)
        $width = [int]($image.Width * ($Right - $Left))
        $height = [int]($image.Height * ($Bottom - $Top))
        $bitmap = New-Object System.Drawing.Bitmap $width, $height
        try {
            $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
            try {
                $graphics.Clear([System.Drawing.Color]::White)
                $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
                $src = New-Object System.Drawing.Rectangle $x, $y, $width, $height
                $dst = New-Object System.Drawing.Rectangle 0, 0, $width, $height
                $graphics.DrawImage($image, $dst, $src, [System.Drawing.GraphicsUnit]::Pixel)
            }
            finally { $graphics.Dispose() }
            $bitmap.Save((Join-Path $OutputDir $Name), [System.Drawing.Imaging.ImageFormat]::Png)
        }
        finally { $bitmap.Dispose() }
    }
    finally { $image.Dispose() }
}

$n = Join-Path $SourceRoot 'NeRF'
$g = Join-Path $SourceRoot '3DGS'
$p = Join-Path $SourceRoot '3DPhoto'
$s = Join-Path $SourceRoot 'SLIDE'

Export-Crop (Join-Path $n 'page-01.png') '01_nerf_cover.png' 0.04 0.03 0.96 0.31
Export-Crop (Join-Path $n 'page-02.png') '02_nerf_overview.png' 0.05 0.03 0.95 0.36
Export-Crop (Join-Path $n 'page-05.png') '03_nerf_pipeline.png' 0.04 0.03 0.96 0.42
Export-Crop (Join-Path $n 'page-07.png') '04_nerf_ablation.png' 0.04 0.03 0.96 0.42

Export-Crop (Join-Path $g 'page-01.png') '05_3dgs_cover.png' 0.03 0.03 0.97 0.48
Export-Crop (Join-Path $g 'page-05.png') '06_3dgs_pipeline.png' 0.04 0.04 0.96 0.42
Export-Crop (Join-Path $g 'page-06.png') '07_3dgs_density.png' 0.04 0.03 0.96 0.40
Export-Crop (Join-Path $g 'page-07.png') '08_3dgs_results.png' 0.03 0.03 0.97 0.91

Export-Crop (Join-Path $p 'page-01.png') '09_3dphoto_cover.png' 0.03 0.03 0.97 0.48
Export-Crop (Join-Path $p 'page-04.png') '10_3dphoto_preprocess_ldi.png' 0.03 0.03 0.97 0.70
Export-Crop (Join-Path $p 'page-06.png') '11_3dphoto_inpainting.png' 0.04 0.03 0.96 0.62
Export-Crop (Join-Path $p 'page-07.png') '12_3dphoto_results.png' 0.03 0.03 0.97 0.92

Export-Crop (Join-Path $s 'page-01.png') '13_slide_cover.png' 0.03 0.03 0.97 0.48
Export-Crop (Join-Path $s 'page-03.png') '14_slide_pipeline.png' 0.04 0.03 0.96 0.42
Export-Crop (Join-Path $s 'page-05.png') '15_slide_soft_layering.png' 0.03 0.03 0.97 0.57
Export-Crop (Join-Path $s 'page-06.png') '16_slide_inpainting.png' 0.03 0.03 0.97 0.55
Export-Crop (Join-Path $s 'page-07.png') '17_slide_results.png' 0.03 0.03 0.97 0.92

Write-Output 'ASSET_COUNT=17'
