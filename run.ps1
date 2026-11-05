<#
.SYNOPSIS
  Image Preparation Studio - setup and launch (Windows). Everything is installed into .\.venv
  inside this folder. Nothing outside this folder is modified; ComfyUI / NH Maker 2 are never touched.

.EXAMPLE
  .\run.ps1                 # first run: creates .venv, installs deps, offers the model download, launches the app
  .\run.ps1 -Shortcut       # put an "Image Preparation Studio" shortcut on the desktop
  .\run.ps1 -FetchModels    # user-directed, checksum-verified model download (GitHub releases)
  .\run.ps1 -Inspect        # read-only hardware/ComfyUI/model report -> reports\env_windows.json
  .\run.ps1 -Bench          # measure the 15 s Fast Profile on this GPU -> reports\bench_*.json
  .\run.ps1 -Test           # run the automated test suite
  .\run.ps1 -PrivacyAudit   # verify no outbound network connections during a workload
  .\run.ps1 -SetupOnly      # create/refresh .venv only
  .\run.ps1 -Smoke          # launch the app, open the main window, exit after 8 s (installation check)
  .\run.ps1 -Cli prepare C:\photos --profiles archive,share
#>
param(
  [switch]$Setup, [switch]$FetchModels, [switch]$Inspect, [switch]$Bench, [switch]$BenchQuick,
  [switch]$Test, [switch]$PrivacyAudit, [switch]$SetupOnly, [switch]$Smoke, [switch]$Shortcut, [switch]$Yes,
  [string]$Python = "", [string]$TorchIndex = "https://download.pytorch.org/whl/cu128",
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

function Invoke-Checked([string]$What, [scriptblock]$Cmd) {
  & $Cmd
  if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE). Re-run .\run.cmd -Setup to retry." }
}

function Ensure-Venv {
  $ready = Join-Path $Venv ".ips_ready"
  if ((Test-Path $ready) -and -not $Setup) { return }
  $base = Find-Python
  Write-Host "Creating isolated environment in $Venv (using $base)" -ForegroundColor Cyan
  Invoke-Checked "Creating the virtual environment" { & $base -m venv $Venv }
  Invoke-Checked "Upgrading pip" { & $Py -m pip install --upgrade pip wheel }
  Write-Host "Installing PyTorch (CUDA build from $TorchIndex) - this is a large download..." -ForegroundColor Cyan
  Invoke-Checked "Installing PyTorch" { & $Py -m pip install torch torchvision --index-url $TorchIndex }
  Invoke-Checked "Installing dependencies" { & $Py -m pip install -r requirements-dev.txt }
  & $Py -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
  Set-Content -Path $ready -Value (Get-Date -Format s)
}

function Ensure-Models {
  $missing = & $Py tools\fetch_models.py --missing
  if ($LASTEXITCODE -eq 0) { return }
  Write-Host ""
  Write-Host "Model weights not installed yet:" -ForegroundColor Yellow
  Write-Host "  $missing"
  Write-Host "They are downloaded once (about 1.4 GB) from the projects' official GitHub releases and checked by SHA-256."
  Write-Host "Without them the app still runs, using classical (non-AI) processing only."
  $answer = if ($Yes) { "y" } else { Read-Host "Download now? [Y/n]" }
  if ($answer -eq "" -or $answer -match "^[Yy]") { & $Py tools\fetch_models.py --all }
}

function New-DesktopShortcut {
  $desktop = [Environment]::GetFolderPath("Desktop")
  $lnk = Join-Path $desktop "Image Preparation Studio.lnk"
  $sh = New-Object -ComObject WScript.Shell
  $sc = $sh.CreateShortcut($lnk)
  $sc.TargetPath = Join-Path $Root "run.cmd"
  $sc.WorkingDirectory = $Root
  $sc.WindowStyle = 7  # minimised console; the app window opens normally
  $sc.Description = "Image Preparation Studio"
  $sc.Save()
  Write-Host "Shortcut created: $lnk" -ForegroundColor Green
}

Ensure-Venv
if ($SetupOnly) { exit 0 }
New-Item -ItemType Directory -Force -Path (Join-Path $Root "reports") | Out-Null

if ($FetchModels) { & $Py tools\fetch_models.py --all; exit $LASTEXITCODE }
if ($Inspect)     { & $Py tools\inspect_env.py --out reports\env_windows.json; exit $LASTEXITCODE }
if ($Bench)       { & $Py tools\bench.py; exit $LASTEXITCODE }
if ($BenchQuick)  { & $Py tools\bench.py --quick; exit $LASTEXITCODE }
if ($PrivacyAudit){ & $Py tools\privacy_audit.py; exit $LASTEXITCODE }
if ($Test) {
  if (-not (Test-Path "fixtures\corpus\manifest.json")) { & $Py -m ips.eval.fixtures --fetch }
  & $Py -m pytest -q; exit $LASTEXITCODE
}
if ($Cli -and $Cli.Count -gt 0) { & $Py -m ips.cli @Cli; exit $LASTEXITCODE }
if ($Shortcut) { New-DesktopShortcut; exit 0 }
Ensure-Models
if ($Smoke) { $env:IPS_SMOKE_SECONDS = "8"; & $Py -m ips.app; exit $LASTEXITCODE }
Write-Host "Starting Image Preparation Studio..." -ForegroundColor Cyan
& $Py -m ips.app
