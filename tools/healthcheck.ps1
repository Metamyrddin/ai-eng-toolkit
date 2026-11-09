# This script runs the health check from Windows PowerShell.
# If .venv is missing, it builds it with setup.ps1 and then runs the checks.
# If .venv is already there, it does not build it again.
# If the build fails, it stops with a clear message and a non-zero exit code.
# A guard stops a loop if setup starts this script again while a build is already running.
# Extra words, such as --fast, are passed on to healthcheck.py.
# Triton and DeepSpeed are not installed on native Windows. The health check marks Triton as SKIP.
# Use WSL2 for those two. WSL2 was not tested.
#
# Usage:  powershell -ExecutionPolicy Bypass -File tools/healthcheck.ps1 --fast
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$HealthArgs
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Join-Under {
    param([string]$Base, [string[]]$Child)
    $path = $Base
    foreach ($part in $Child) {
        $path = Join-Path $path $part
    }
    return $path
}

# Return the full path of the virtual environment's Python, or $null when it is missing.
# Activate.ps1 is tried so the window matches a normal setup, but python.exe is what we run.
function Get-VenvPython {
    $pythonExe = Join-Under $root @(".venv", "Scripts", "python.exe")
    if (-not (Test-Path $pythonExe)) {
        return $null
    }
    $activate = Join-Under $root @(".venv", "Scripts", "Activate.ps1")
    if (Test-Path $activate) {
        try {
            & $activate | Out-Null
        } catch {
            Write-Host "Could not run Activate.ps1 ($($_.Exception.Message)). The health check will use .venv\Scripts\python.exe."
        }
    }
    return $pythonExe
}

$python = Get-VenvPython
if (-not $python) {
    $venvDir = Join-Path $root ".venv"
    if (Test-Path $venvDir) {
        Write-Host "A .venv folder is already here, but .venv\Scripts\python.exe was not found. This script will not rebuild an existing .venv. Remove that folder, fix whatever went wrong, and run this command again."
        exit 1
    }
    if ($env:AI_ENG_HEALTHCHECK_BUILDING) {
        Write-Host "The health check started again while it was already building .venv. Stopping so setup and the health check do not call each other forever."
        exit 1
    }

    Write-Host "No .venv found. Building it now with setup.ps1; this can take several minutes and downloads packages and models."
    $env:AI_ENG_HEALTHCHECK_BUILDING = "1"
    try {
        & (Join-Path $root "setup.ps1")
    } catch {
        Write-Host "Building .venv with setup.ps1 failed. The health check did not run. If a .venv folder was left behind, remove it, fix the error above, and run this command again."
        if ($_.Exception.Message) {
            Write-Host $_.Exception.Message
        }
        exit 1
    } finally {
        Remove-Item Env:\AI_ENG_HEALTHCHECK_BUILDING -ErrorAction SilentlyContinue
    }

    try {
        & (Join-Path $PSScriptRoot "get_llamacpp.ps1")
    } catch {
        Write-Host "The packages were installed, but downloading the llama.cpp programs failed. The health check did not run. Fix the error above, then run this command again."
        if ($_.Exception.Message) {
            Write-Host $_.Exception.Message
        }
        exit 1
    }

    Set-Location $root
    $python = Get-VenvPython
    if (-not $python) {
        Write-Host "setup.ps1 finished, but .venv\Scripts\python.exe is still missing. The health check did not run."
        exit 1
    }
}

if (-not $HealthArgs) {
    $HealthArgs = @()
}
& $python (Join-Path $PSScriptRoot "healthcheck.py") @HealthArgs
exit $LASTEXITCODE
