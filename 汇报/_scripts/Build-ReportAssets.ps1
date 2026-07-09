Add-Type -AssemblyName System.Drawing

$reportDir = Split-Path -Parent $PSScriptRoot
$mediaDir = Join-Path $reportDir "_docx_extract\word\media"
$outDir = Join-Path $reportDir "_assets"
New-Item -ItemType Directory -Force $outDir | Out-Null

function Crop-Image {
    param(
        [string]$InputPath,
        [string]$OutputName,
        [int]$X,
        [int]$Y,
        [int]$W,
        [int]$H
    )

    $src = [System.Drawing.Bitmap]::FromFile($InputPath)
    try {
        $rect = New-Object System.Drawing.Rectangle($X, $Y, $W, $H)
        $dst = New-Object System.Drawing.Bitmap($W, $H)
        try {
            $g = [System.Drawing.Graphics]::FromImage($dst)
            try {
                $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
                $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
                $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
                $g.DrawImage($src, (New-Object System.Drawing.Rectangle(0, 0, $W, $H)), $rect, [System.Drawing.GraphicsUnit]::Pixel)
            } finally {
                $g.Dispose()
            }
            $outPath = Join-Path $outDir $OutputName
            $dst.Save($outPath, [System.Drawing.Imaging.ImageFormat]::Png)
            Write-Output $outPath
        } finally {
            $dst.Dispose()
        }
    } finally {
        $src.Dispose()
    }
}

Copy-Item (Join-Path $mediaDir "image1.png") (Join-Path $outDir "nerf_paper_title.png") -Force
Copy-Item (Join-Path $mediaDir "image7.png") (Join-Path $outDir "3dgs_paper_title.png") -Force

Crop-Image (Join-Path $mediaDir "image2.png") "nerf_fig1_task.png" 70 85 655 185
Crop-Image (Join-Path $mediaDir "image3.png") "nerf_fig2_pipeline.png" 75 70 655 230
Crop-Image (Join-Path $mediaDir "image5.png") "nerf_table1_metrics.png" 70 70 655 160
Crop-Image (Join-Path $mediaDir "image6.png") "nerf_fig5_results.png" 70 75 660 850
Crop-Image (Join-Path $mediaDir "image7.png") "3dgs_fig1_speed_quality.png" 100 335 1020 240
Crop-Image (Join-Path $mediaDir "image7.png") "3dgs_title_crop.png" 100 150 850 170
Crop-Image (Join-Path $mediaDir "image8.png") "3dgs_fig2_pipeline.png" 100 120 1030 300
Crop-Image (Join-Path $mediaDir "image9.png") "3dgs_fig4_density_control.png" 120 145 510 300
Crop-Image (Join-Path $mediaDir "image10.png") "3dgs_fig5_results.png" 105 140 1010 1160
Crop-Image (Join-Path $mediaDir "image11.png") "3dgs_table1_metrics.png" 105 120 1020 220
