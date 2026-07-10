param(
    [Parameter(Mandatory = $true)][string]$InputPptx,
    [Parameter(Mandatory = $true)][string]$OutputDir
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$InputPptx = (Resolve-Path -LiteralPath $InputPptx).Path
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

$powerPoint = New-Object -ComObject PowerPoint.Application
$powerPoint.Visible = -1
$presentation = $null

try {
    $presentation = $powerPoint.Presentations.Open($InputPptx, 1, 0, 0)
    $presentation.Export($OutputDir, 'PNG', 1600, 900)
    Write-Output ("EXPORTED_SLIDES={0}" -f $presentation.Slides.Count)
}
finally {
    if ($presentation) { $presentation.Close() }
    $powerPoint.Quit()
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($powerPoint) | Out-Null
}
