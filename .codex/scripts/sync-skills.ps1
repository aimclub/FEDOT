param(
    [string]$SourceRoot = (Join-Path $PSScriptRoot '..\skills-src'),
    [string]$TargetRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$resolvedSource = (Resolve-Path -LiteralPath $SourceRoot).Path

if (-not $TargetRoot -or [string]::IsNullOrWhiteSpace($TargetRoot)) {
    $codexHome = if ($env:CODEX_HOME -and -not [string]::IsNullOrWhiteSpace($env:CODEX_HOME)) {
        $env:CODEX_HOME
    } else {
        Join-Path $HOME '.codex'
    }
    $TargetRoot = Join-Path $codexHome 'skills'
}

if (-not (Test-Path -LiteralPath $TargetRoot)) {
    New-Item -ItemType Directory -Path $TargetRoot -Force | Out-Null
}

$resolvedTarget = [System.IO.Path]::GetFullPath($TargetRoot)
$skillDirs = Get-ChildItem -LiteralPath $resolvedSource -Directory | Where-Object { $_.Name -ne '.system' }

Write-Host "Source root: $resolvedSource"
Write-Host "Target root: $resolvedTarget"

foreach ($skillDir in $skillDirs) {
    $destination = Join-Path $resolvedTarget $skillDir.Name
    if (-not (Test-Path -LiteralPath $destination)) {
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
    }

    $items = Get-ChildItem -LiteralPath $skillDir.FullName -Force
    foreach ($item in $items) {
        Copy-Item -LiteralPath $item.FullName -Destination $destination -Recurse -Force
    }

    Write-Host "Synced $($skillDir.Name) -> $destination"
}

Write-Host 'Done.'