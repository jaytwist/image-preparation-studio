<#
.SYNOPSIS
  Image Preparation Studio - setup and launch (Windows). Everything is installed into .\.venv
  inside this folder. Nothing outside this folder is modified; ComfyUI / NH Maker 2 are never touched.

.EXAMPLE
  .\run.ps1                 # first run: creates .venv + installs deps, then launches the app
  .\run.ps1 -FetchModels    # user-directed, checksum-verified model download (GitHub releases)
  .\run.ps1 -Inspect        # read-only hardware/ComfyUI/model report -> reports\env_windows.json
  .\run.ps1 -Bench          # measure the 15 s Fast Profile on this GPU -> reports\bench_*.json
  .\run.ps1 -Test           # run the automated test suite
  .\run.ps1 -PrivacyAudit   # verify no outbound network connections during a workload
  .\run.ps1 -Cli prepare C:\photos --profiles archive,share
#>
param(
  [switch]$Setup, [switch]$FetchModels, [switch]$Inspect, [switch]$Bench, [switch]$BenchQuick,
  [switch]$Test, [switch]$PrivacyAudit, [string]$Python = "", [string]$TorchIndex = "https://download.pytorch.org/whl/cu128",
  [Parameter(ValueFromRemainingArguments = $true)][string[]]$Cli
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Venv = Join-Path $Root ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$env:ORT_DISABLE_TELEMETRY = "1"   # onnxruntime ships network telemetry; disabled (also enforced in code)
$env:HF_HUB_OFFLINE = "1"
$env:DO_NOT_TRACK = "1"

function Find-Python {
  if ($Python) { return $Python }
  foreach ($v in @("3.12", "3.11", "3.13")) {
    try { $p = & py -$v -c "import sys; print(sys.executable)" 2>$null; if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() } } catch {}
  }
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  throw "Python 3.11-3.13 not found. Install from python.org (it does not need to be on PATH if the 'py' launcher is installed)."
}

function Ensure-Venv {
  if ((Test-Path $Py) -and -not $Setup) { return }
  $base = Find-Python
  Write-Host "Creating isolated environment with $base" -ForegroundColor Cyan
  & $base -m venv $Venv
  & $Py -m pip install --upgrade pip wheel
  Write-Host "Installing PyTorch (CUDA build from $TorchIndex)..." -ForegroundColor Cyan
  & $Py -m pip install torch torchvision --index-url $TorchIndex
  & $Py -m pip install -r requirements-dev.txt
  & $Py -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
}

Ensure-Venv
New-Item -ItemType Directory -Force -Path (Join-Path $Root "reports") | Out-Null

if ($FetchModels) { & $Py tools\fetch_models.py --all; exit $LASTEXITCODE }
if ($Inspect)     { & $Py tools\inspect_env.py --out reports\env_windows.json; exit $LASTEXITCODE }
if ($Bench)       { & $Py tools\bench.py; exit $LASTEXITCODE }
if ($BenchQuick)  { & $Py tools\bench.py --quick; exit $LASTEXITCODE }
if ($PrivacyAudit){ & $Py tools\privacy_audit.py; exit $LASTEXITCODE }
if ($Test) {
  if (-not (Test-Path "fixtures\corpus\manifest.json")) { & $Py -m ips.eval.fixtures }
  & $Py -m pytest -q; exit $LASTEXITCODE
}
if ($Cli -and $Cli.Count -gt 0) { & $Py -m ips.cli @Cli; exit $LASTEXITCODE }
& $Py -m ips.app
