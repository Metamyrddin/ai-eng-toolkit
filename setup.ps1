# This is the setup script for Windows. It creates the virtual environment, installs the packages and
# downloads the models. It has NOT been tested on Windows.
#
# More technical notes:
# Windows (PowerShell) setup. NOT TESTED ON WINDOWS (the toolkit was built and run on Linux) - see README "Windows notes".
# Usage:  powershell -ExecutionPolicy Bypass -File setup.ps1 [-NoModels]
param([switch]$NoModels)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -c "import sys; assert sys.version_info >= (3,10), 'need Python >= 3.10'"
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
# Triton has no Windows wheels: install everything else.
(Get-Content requirements-cpu.txt) | Where-Object { $_ -notmatch '^triton' -and $_ -notmatch '^deepspeed' } | Set-Content requirements-cpu-windows.txt
python -m pip install -r requirements-cpu-windows.txt
python -m pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
if (-not $NoModels) { python tools/download_models.py }
Write-Host "Done. Next:  .\.venv\Scripts\Activate.ps1 ; pytest -q -m 'not slow'"
Write-Host "Not installed on native Windows by this script: Triton (interpreter demo; use WSL2) and DeepSpeed. gloo is listed as supported on Windows in the PyTorch docs but we have not tested it. See README."
