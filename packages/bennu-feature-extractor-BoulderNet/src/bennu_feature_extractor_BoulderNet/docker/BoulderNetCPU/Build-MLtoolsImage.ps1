<# Build-MLtoolsImage.ps1 — PowerShell 3.0 #>
param(
  [string]$Tag = "mltools:py3.10",
  [string]$Dockerfile = "Dockerfile",
  [string]$Context = ".",
  [switch]$NoCache
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Dockerfile)) {
  Write-Error "Dockerfile not found at '$Dockerfile'."; exit 1
}

$ContextPath = (Resolve-Path $Context).Path

# docker build
$args = @("build","-t",$Tag,"-f",$Dockerfile)
if ($NoCache) { $args += "--no-cache" }
$args += $ContextPath

Write-Host ("docker " + ($args -join " "))
docker @args
if ($LASTEXITCODE -ne 0) { Write-Error "Build failed ($LASTEXITCODE)."; exit $LASTEXITCODE }

Write-Host "Build OK: $Tag"
