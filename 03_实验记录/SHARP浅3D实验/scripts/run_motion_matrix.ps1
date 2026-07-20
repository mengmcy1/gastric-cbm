param(
    [string]$PythonExe = "python",
    [string[]]$SampleIds = @("P01", "P02", "P03", "P04", "P05")
)

$ErrorActionPreference = "Stop"
$experimentRoot = Split-Path -Parent $PSScriptRoot
$renderer = Join-Path $PSScriptRoot "sharp_experiment_render.py"
$pilotRoot = Join-Path $experimentRoot "01_先导样本"
$outputRoot = Join-Path $experimentRoot "outputs\03_运动实验"

$samples = [ordered]@{
    P01 = "P01_building\scene_full.ply"
    P02 = "P02_occlusion_flower\scene_full.ply"
    P03 = "P03_ferris_wheel\scene_full.ply"
    P04 = "P04_person_wall_shadow\scene_full.ply"
    P05 = "P05_night_reflection\scene_full.ply"
}

$levels = [ordered]@{
    md000 = "0"
    md002 = "0.02"
    md004 = "0.04"
    md008 = "0.08"
}

foreach ($sampleId in $SampleIds) {
    if (-not $samples.Contains($sampleId)) {
        throw "未知样本 ID：$sampleId"
    }
    $ply = Join-Path $pilotRoot $samples[$sampleId]
    if (-not (Test-Path -LiteralPath $ply -PathType Leaf)) {
        throw "PLY 不存在：$ply"
    }

    foreach ($levelName in $levels.Keys) {
        $maxDisparity = $levels[$levelName]
        $outputDir = Join-Path $outputRoot "$sampleId\${levelName}_swipe60_keep100_crop00"
        Write-Host "[$sampleId][$levelName] max_disparity=$maxDisparity -> $outputDir"
        & $PythonExe $renderer `
            --ply $ply `
            --output-dir $outputDir `
            --trajectory swipe `
            --max-disparity $maxDisparity `
            --num-steps 60 `
            --endpoint all `
            --fps 30 `
            --device cuda
        if ($LASTEXITCODE -ne 0) {
            throw "渲染失败：$sampleId / $levelName，退出码 $LASTEXITCODE"
        }
    }
}
