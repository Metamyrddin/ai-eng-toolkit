# This is the setup script for Windows. It creates the virtual environment, installs the packages and
# downloads the models. The health check runs this script when .venv is missing.
# Triton and DeepSpeed are left out, because they have no normal Windows installers. Use WSL2 for those two.
# GitHub Actions runs this script on windows-latest. It was not tried on a home PC, and WSL2 was not tried.
#
# Usage:  powershell -ExecutionPolicy Bypass -File setup.ps1 [-NoModels]
param([switch]$NoModels)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Windows PowerShell keeps going when python.exe fails. This function stops the script instead.
function Test-LastExit {
    param([string]$What)
    if ($LASTEXITCODE -ne 0) {
        throw "$What failed with exit code $LASTEXITCODE."
    }
}

# Find a Python 3.10 or newer. Windows machines name it python, python3, or py.
function Resolve-PythonLauncher {
    $candidates = @(
        @{ Exe = "python"; Args = @() },
        @{ Exe = "python3"; Args = @() },
        @{ Exe = "py"; Args = @("-3") }
    )
    foreach ($candidate in $candidates) {
        $cmd = Get-Command $candidate.Exe -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        & $candidate.Exe @($candidate.Args) -c "import sys; assert sys.version_info >= (3, 10), 'need Python >= 3.10'"
        if ($LASTEXITCODE -eq 0) {
            return $candidate
        }
    }
    throw "Python 3.10 or newer was not found. Install Python from python.org, and make sure the python, python3, or py command works. Then run this script again."
}

$launcher = Resolve-PythonLauncher
Write-Host ("Using Python command: " + $launcher.Exe + " " + ($launcher.Args -join " "))
& $launcher.Exe @($launcher.Args) -m venv .venv
Test-LastExit "Creating .venv"

$venvPython = Join-Path (Join-Path (Join-Path $PSScriptRoot ".venv") "Scripts") "python.exe"
if (-not (Test-Path $venvPython)) {
    throw "The virtual environment was created, but .venv\Scripts\python.exe is missing."
}

# Turning on the virtual environment can be blocked by the execution policy. The installs below call
# python.exe by its full path, so setup still finishes when Activate.ps1 cannot run.
$activate = Join-Path (Join-Path (Join-Path $PSScriptRoot ".venv") "Scripts") "Activate.ps1"
if (Test-Path $activate) {
    try {
        & $activate | Out-Null
    } catch {
        Write-Host "Could not turn on .venv in this window ($($_.Exception.Message)). The install will continue with .venv\Scripts\python.exe."
    }
}

& $venvPython -m pip install --upgrade pip
Test-LastExit "Upgrading pip"
& $venvPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu
Test-LastExit "Installing PyTorch"

# Keep every requirements line except Triton and DeepSpeed. Those two have no normal Windows installers.
$lines = @((Get-Content (Join-Path $PSScriptRoot "requirements-cpu.txt")) | Where-Object { $_ -notmatch '^triton' -and $_ -notmatch '^deepspeed' })
$reqFile = Join-Path ([System.IO.Path]::GetTempPath()) "ai-eng-toolkit-requirements-cpu-windows.txt"
# Write UTF-8 without a special header. Windows PowerShell's Set-Content can write a format pip cannot read.
[System.IO.File]::WriteAllLines($reqFile, [string[]]$lines)
& $venvPython -m pip install -r $reqFile
Test-LastExit "Installing requirements"
& $venvPython -m pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
Test-LastExit "Installing llama-cpp-python"
if (-not $NoModels) {
    & $venvPython (Join-Path $PSScriptRoot "tools\download_models.py")
    Test-LastExit "Downloading models"
}
Write-Host "Done. Next:  .\.venv\Scripts\Activate.ps1 ; pytest -q -m 'not slow'"
Write-Host "Not installed on native Windows by this script: Triton and DeepSpeed. They have no normal Windows installers. Use WSL2 for those two."
