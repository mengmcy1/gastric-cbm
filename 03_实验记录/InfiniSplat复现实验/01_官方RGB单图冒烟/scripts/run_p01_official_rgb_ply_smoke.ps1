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
    $ConfigPath = Join-Path $experimentRoot "configs/p01_official_rgb_ply_smoke_v1.json"
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

if (-not (Test-Path -LiteralPath $sourcePath -PathType Container)) {
    throw "InfiniSplat source directory is missing: $sourcePath"
}
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

if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
    throw "Input image is missing: $inputPath"
}
$inputHash = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash
if ($inputHash -ne $manifest.sha256) {
    throw "Input SHA256 mismatch: expected $($manifest.sha256), actual $inputHash"
}

if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
    throw "Checkpoint is missing: $checkpointPath"
}
$checkpoint = Get-Item -LiteralPath $checkpointPath
if ($checkpoint.Length -ne [int64]$config.checkpoint_expected_bytes) {
    throw "Checkpoint size mismatch: expected $($config.checkpoint_expected_bytes), actual $($checkpoint.Length)"
}
$checkpointHash = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash
if ($checkpointHash -ne $config.checkpoint_expected_sha256) {
    throw "Checkpoint SHA256 mismatch: expected $($config.checkpoint_expected_sha256), actual $checkpointHash"
}

if (Test-Path -LiteralPath $outputDir) {
    throw "Output directory already exists; refusing to overwrite: $outputDir"
}
if (Test-Path -LiteralPath $logPath) {
    throw "Log file already exists; refusing to overwrite: $logPath"
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
    "--no-video",
    "--no-export-html"
)

Push-Location $sourcePath
$previousPythonUtf8 = $env:PYTHONUTF8
$previousPythonIoEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    & conda @arguments 2>&1 | Tee-Object -LiteralPath $logPath
    $exitCode = $LASTEXITCODE
}
finally {
    $env:PYTHONUTF8 = $previousPythonUtf8
    $env:PYTHONIOENCODING = $previousPythonIoEncoding
    Pop-Location
}

if ($exitCode -ne 0) {
    throw "InfiniSplat P01 smoke failed with exit code $exitCode. Log: $logPath"
}

Write-Host "InfiniSplat P01 PLY smoke completed: $outputDir"
