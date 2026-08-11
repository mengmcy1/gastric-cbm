#requires -Version 7.0
[CmdletBinding()]
param(
    [string]$ConfigPath = ""
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $PSCommandPath
$experimentRoot = Split-Path -Parent $scriptDir
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $experimentRoot "../../.."))

if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path $experimentRoot "configs/p01_official_rgb_video_smoke_v1.json"
}
$ConfigPath = [System.IO.Path]::GetFullPath($ConfigPath)
$config = Get-Content -LiteralPath $ConfigPath -Raw -Encoding utf8 | ConvertFrom-Json

function Resolve-RepoPath([string]$RelativePath) {
    $resolved = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $RelativePath))
    if (-not $resolved.StartsWith($repoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes repository root: $RelativePath"
    }
    return $resolved
}

$sourcePath = Resolve-RepoPath $config.source_path
$manifestPath = Resolve-RepoPath $config.input_manifest
$checkpointPath = Resolve-RepoPath $config.checkpoint_path
$outputDir = Resolve-RepoPath $config.output_dir
$logPath = Resolve-RepoPath $config.log_path
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$inputPath = Resolve-RepoPath $manifest.path

$nestedGit = Join-Path $sourcePath ".git"
if (Test-Path -LiteralPath $nestedGit) {
    $actualCommit = (& git -C $sourcePath rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $config.source_commit) {
        throw "Source commit mismatch: expected $($config.source_commit), actual $actualCommit"
    }
}
else {
    $sourceManifestPath = Join-Path $experimentRoot "inputs/infinisplat_source_manifest.json"
    $sourceManifest = Get-Content -LiteralPath $sourceManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
    if ($sourceManifest.source_path -ne $config.source_path -or $sourceManifest.upstream_commit -ne $config.source_commit) {
        throw "Vendored source manifest does not match the frozen config."
    }
}
if ((Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash -ne $manifest.sha256) {
    throw "Input SHA256 mismatch: $inputPath"
}
$checkpoint = Get-Item -LiteralPath $checkpointPath
if ($checkpoint.Length -ne [int64]$config.checkpoint_expected_bytes) {
    throw "Checkpoint size mismatch: expected $($config.checkpoint_expected_bytes), actual $($checkpoint.Length)"
}
if ((Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash -ne $config.checkpoint_expected_sha256) {
    throw "Checkpoint SHA256 mismatch: $checkpointPath"
}
if (Test-Path -LiteralPath $outputDir) {
    throw "Output directory already exists; refusing to overwrite: $outputDir"
}
if (Test-Path -LiteralPath $logPath) {
    throw "Log file already exists; refusing to overwrite: $logPath"
}

$envInfo = conda env list --json | ConvertFrom-Json
$infinisplatEnv = $envInfo.envs | Where-Object { (Split-Path -Leaf $_) -eq $config.conda_env } | Select-Object -First 1
$toolkitEnv = $envInfo.envs | Where-Object { (Split-Path -Leaf $_) -eq $config.cuda_toolkit_source_env } | Select-Object -First 1
if (-not $infinisplatEnv) { throw "Conda environment not found: $($config.conda_env)" }
if (-not $toolkitEnv) { throw "CUDA toolkit source environment not found: $($config.cuda_toolkit_source_env)" }

$backendPath = Join-Path $infinisplatEnv "Lib/site-packages/gsplat/cuda/_backend.py"
$cudaHome = Join-Path $toolkitEnv "Library"
$nvccPath = Join-Path $cudaHome "bin/nvcc.exe"
$cudaBackendPath = Join-Path $env:LOCALAPPDATA "torch_extensions/torch_extensions/Cache/py310_cu128/gsplat_cuda/gsplat_cuda.pyd"
if (-not (Test-Path -LiteralPath $nvccPath -PathType Leaf)) { throw "CUDA compiler is missing: $nvccPath" }
if (-not (Test-Path -LiteralPath $cudaBackendPath -PathType Leaf)) { throw "Compiled gsplat CUDA backend is missing: $cudaBackendPath" }
if ((Get-FileHash -LiteralPath $backendPath -Algorithm SHA256).Hash -ne $config.gsplat_windows_backend_sha256) {
    throw "Patched gsplat Windows backend hash mismatch: $backendPath"
}
if ((Get-FileHash -LiteralPath $cudaBackendPath -Algorithm SHA256).Hash -ne $config.gsplat_cuda_backend_sha256) {
    throw "Compiled gsplat CUDA backend hash mismatch: $cudaBackendPath"
}

New-Item -ItemType Directory -Path (Split-Path -Parent $outputDir) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $logPath) -Force | Out-Null

$arguments = @(
    "run", "-n", $config.conda_env, "--no-capture-output",
    "python", "-m", "src.demo.infer_batch_images",
    "--mode", $config.mode,
    "--checkpoint", $checkpointPath,
    "--input", $inputPath,
    "--output-dir", $outputDir,
    "--device", $config.device,
    "--no-export-html"
)

$previousCudaHome = $env:CUDA_HOME
$previousCudaPath = $env:CUDA_PATH
$previousPath = $env:PATH
$previousPythonUtf8 = $env:PYTHONUTF8
$previousPythonIoEncoding = $env:PYTHONIOENCODING
$startedAt = Get-Date
Push-Location $sourcePath
try {
    $env:CUDA_HOME = $cudaHome
    $env:CUDA_PATH = $cudaHome
    $env:PATH = "$(Join-Path $cudaHome 'bin');$env:PATH"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    & conda @arguments 2>&1 | Tee-Object -LiteralPath $logPath
    $exitCode = $LASTEXITCODE
}
finally {
    $env:CUDA_HOME = $previousCudaHome
    $env:CUDA_PATH = $previousCudaPath
    $env:PATH = $previousPath
    $env:PYTHONUTF8 = $previousPythonUtf8
    $env:PYTHONIOENCODING = $previousPythonIoEncoding
    Pop-Location
}

if ($exitCode -ne 0) {
    throw "InfiniSplat P01 video smoke failed with exit code $exitCode. Log: $logPath"
}

$caseDir = Join-Path $outputDir ([System.IO.Path]::GetFileNameWithoutExtension($inputPath))
$plyPath = Join-Path $caseDir "input.ply"
$videoPath = Join-Path $caseDir "input.mp4"
if (-not (Test-Path -LiteralPath $plyPath -PathType Leaf)) { throw "Expected PLY is missing: $plyPath" }
if (-not (Test-Path -LiteralPath $videoPath -PathType Leaf)) { throw "Expected video is missing: $videoPath" }

$elapsed = (Get-Date) - $startedAt
Write-Host "InfiniSplat P01 RGB video smoke completed in $([math]::Round($elapsed.TotalSeconds, 2)) seconds: $outputDir"
