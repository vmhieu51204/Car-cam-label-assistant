# Fallback setup: native Windows 10/11 (PowerShell). WSL2 (setup_wsl.sh) is recommended:
# sam3 imports `triton` unconditionally and Triton has no official Windows build, so this
# script installs the community `triton-windows` wheel that matches torch 2.10.
# Requirements: Python 3.12 (py launcher), Git, a recent NVIDIA driver.
$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = if ($env:VENV) { $env:VENV } else { Join-Path $Here ".venv" }
$Sam3Commit = "0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc"
$TorchIndex = "https://download.pytorch.org/whl/cu128"

py -3.12 --version
if ($LASTEXITCODE -ne 0) { throw "Python 3.12 not found: install it from python.org (tick 'py launcher')." }
nvidia-smi
if ($LASTEXITCODE -ne 0) { Write-Warning "nvidia-smi failed - install/update the NVIDIA driver." }

py -3.12 -m venv $Venv
$Py = Join-Path $Venv "Scripts\python.exe"
& $Py -m pip install --upgrade pip wheel
& $Py -m pip install torch==2.10.0 torchvision==0.25.0 --index-url $TorchIndex
& $Py -m pip install "triton-windows>=3.6,<3.7"
& $Py -m pip install -r (Join-Path $Here "requirements.txt")
& $Py -m pip install --no-deps "git+https://github.com/facebookresearch/sam3.git@$Sam3Commit"
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

Write-Host ""
Write-Host "Done. Next:"
Write-Host "  $Venv\Scripts\Activate.ps1"
Write-Host "  hf auth login     # token with access to facebook/sam3 and/or facebook/sam3.1 on Hugging Face"
Write-Host "  NVIDIA Control Panel > Manage 3D settings > CUDA - Sysmem Fallback Policy = Prefer No Sysmem Fallback"
Write-Host "  python $Here\sam3\healthcheck_sam3.py --payload C:\data\payload.zip --checkpoint C:\models\sam3.pt --load-model"
Write-Host "  python $Here\sam3.1\healthcheck_sam31.py --payload C:\data\payload.zip --checkpoint C:\models\sam3.1_multiplex.pt --load-model"
