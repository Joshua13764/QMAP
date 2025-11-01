param(
  [string]$Dest = "C:\SPICE\orex",
  [string]$BaseUrl = "https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels",
  [string]$MkRel  = "mk/orx_2019_v08.tm",
  [string]$MkUrl  = "",
  [switch]$AlsoDownloadDSK
)

# DSKs to fetch when -AlsoDownloadDSK is set
$dskList = @(
  "dsk/bennu_g_00880mm_alt_obj_0000n00000_v021.bds"
  # "dsk/bennu_g_00400mm_alt_ptm_0000n00000_v021.bds"
  # "dsk/bennu_g_03170mm_spc_obj_0000n00000_v020.bds"
)

$ErrorActionPreference = "Stop"
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

function Ensure-Dir([string]$p) {
  if (-not (Test-Path -LiteralPath $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
}

function Download($url, $outPath) {
  Ensure-Dir (Split-Path -Parent $outPath)
  if (Test-Path -LiteralPath $outPath) { Write-Host "Exists: $outPath"; return }
  Write-Host "Downloading: $url"
  Invoke-WebRequest -Uri $url -OutFile $outPath -UseBasicParsing
  Write-Host "Saved: $outPath"
}

function Normalize-Rel([string]$p) {
  $q = $p.Trim()
  $q = $q -replace '^\$KERNELS/',''
  $q = $q -replace '^(\./)+',''
  $q = $q -replace '^(\.\./)+',''
  return $q
}

function Get-KernelPathsFromMK([string]$mkPath) {
  $paths = @()
  $inBlock = $false
  foreach ($line in (Get-Content -LiteralPath $mkPath)) {
    $l = $line.Trim()
    if (-not $inBlock -and $l -match '^\s*KERNELS_TO_LOAD\b') { $inBlock = $true }
    if ($inBlock) {
      $mi = Select-String -InputObject $l -Pattern '"([^"]+)"' -AllMatches
      if ($mi) { foreach ($m in $mi.Matches) { $paths += $m.Groups[1].Value } }
      $mi2 = Select-String -InputObject $l -Pattern "'([^']+)'" -AllMatches
      if ($mi2) { foreach ($m in $mi2.Matches) { $paths += $m.Groups[1].Value } }
      if ($l -match '\)') { $inBlock = $false }
    }
  }
  if (-not $paths -or $paths.Count -eq 0) {
    $txt = Get-Content -LiteralPath $mkPath -Raw
    $mi = Select-String -InputObject $txt -Pattern '"([^"]+)"' -AllMatches
    if ($mi) { foreach ($m in $mi.Matches) { $paths += $m.Groups[1].Value } }
    $mi2 = Select-String -InputObject $txt -Pattern "'([^']+)'" -AllMatches
    if ($mi2) { foreach ($m in $mi2.Matches) { $paths += $m.Groups[1].Value } }
  }
  $paths = $paths | ForEach-Object { Normalize-Rel $_ } | Where-Object { $_ } | Sort-Object -Unique
  return $paths
}

# Prepare destination & local metakernel path
Ensure-Dir $Dest
$mkName  = if ($MkUrl) { Split-Path $MkUrl -Leaf } else { Split-Path $MkRel -Leaf }
$mkLocal = Join-Path $Dest (Join-Path "mk" $mkName)

# Download metakernel
if ($MkUrl) { Download $MkUrl $mkLocal } else { Download ("$BaseUrl/$MkRel") $mkLocal }

# Parse MK and download referenced kernels
$rel = Get-KernelPathsFromMK $mkLocal
if (-not $rel -or $rel.Count -eq 0) { throw "No kernel paths found in $mkLocal" }
Write-Host "Found $($rel.Count) kernel paths in metakernel."

foreach ($r in $rel) {
  $url = "$BaseUrl/$r"
  $out = Join-Path $Dest $r
  Download $url $out
}

# Optional DSKs
if ($AlsoDownloadDSK) {
  Write-Host "Downloading Bennu DSK(s)..."
  foreach ($r in $dskList) {
    $url = "$BaseUrl/$r"
    $out = Join-Path $Dest $r
    Download $url $out
  }
}

Write-Host "Done. Kernels in: $Dest"
Write-Host "Metakernel: $mkLocal"
