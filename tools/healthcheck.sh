#!/usr/bin/env bash
# This small script runs the health check inside the project's virtual environment.
# It exists so that beginners can type one command: bash tools/healthcheck.sh
# Any extra words you type (for example --fast) are passed on to healthcheck.py.
#
# If the .venv folder is missing, this script builds it and then runs the checks.
# The first build can take several minutes, because it downloads packages and models.
# If .venv is already there, this script does not build it again.
#
# On Linux and macOS the build command is bash setup.sh.
# On Windows Git Bash the build command is setup.ps1, because a Windows virtual environment
# lives in .venv/Scripts and the Python command may be python, python3, or py.
# setup.sh and setup.ps1 were read before this change. Neither one calls the health check,
# so the two scripts do not call each other in a loop. A guard variable still stops a loop
# if setup ever starts this script again while a build is running.
#
# setup.sh and setup.ps1 do not download the llama.cpp programs. This script does that next,
# because the health check looks for llama-server. On Windows it downloads the official CPU zip.
#
# Triton and DeepSpeed have no normal Windows installers. On native Windows the health check
# marks Triton as SKIP and tells you to use WSL2. WSL2 and macOS were not tested.
set -uo pipefail
cd "$(dirname "$0")/.."                       # Move to the repository root, wherever you started from.

# Turn on an existing virtual environment. Return 0 when it worked, and 1 when no activate file exists.
activate_venv() {
  if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate                 # This is the activate file on Linux and macOS.
    return 0
  fi
  if [ -f .venv/Scripts/activate ]; then
    # This is the activate file in Git Bash on Windows.
    # Some copies use Windows line endings. Git Bash can reject the extra return character,
    # so those characters are removed and the same file is then read.
    # shellcheck disable=SC1091
    source <(tr -d '\r' < .venv/Scripts/activate) || true
    return 0
  fi
  return 1
}

# Pick powershell.exe when Git Bash can see it, and powershell otherwise.
find_powershell() {
  if command -v powershell.exe >/dev/null 2>&1; then
    echo powershell.exe
    return 0
  fi
  if command -v powershell >/dev/null 2>&1; then
    echo powershell
    return 0
  fi
  return 1
}

if ! activate_venv; then
  # A half-finished folder is still an existing .venv. Do not delete it and do not rebuild it.
  if [ -e .venv ]; then
    echo "A .venv folder is already here, but it has no activate script (.venv/bin/activate or .venv/Scripts/activate). This script will not rebuild an existing .venv. Remove that folder, fix whatever went wrong, and run this command again."
    exit 1
  fi

  # This stops a loop if setup, or something it runs, starts the health check again.
  if [ -n "${AI_ENG_HEALTHCHECK_BUILDING:-}" ]; then
    echo "The health check started again while it was already building .venv. Stopping so setup and the health check do not call each other forever."
    exit 1
  fi

  # These system names mean the script is running in Bash on Windows.
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
      echo "No .venv found. Building it now with setup.ps1; this can take several minutes and downloads packages and models."
      if command -v cygpath >/dev/null 2>&1; then
        WIN_SETUP=$(cygpath -w "$PWD/setup.ps1")
      else
        WIN_SETUP="$PWD/setup.ps1"
      fi
      if ! PS_BIN=$(find_powershell); then
        echo "Building .venv with setup.ps1 failed. The powershell command was not found, so the health check did not run."
        exit 1
      fi
      # MSYS_NO_PATHCONV stops Git Bash from rewriting the Windows path before PowerShell sees it.
      if ! AI_ENG_HEALTHCHECK_BUILDING=1 MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' "$PS_BIN" -NoProfile -ExecutionPolicy Bypass -File "$WIN_SETUP"; then
        echo "Building .venv with setup.ps1 failed. The health check did not run. If a .venv folder was left behind, remove it, fix the error above, and run this command again."
        exit 1
      fi
      ;;
    *)
      echo "No .venv found. Building it now with setup.sh; this can take several minutes and downloads packages and models."
      if ! AI_ENG_HEALTHCHECK_BUILDING=1 bash setup.sh; then
        echo "Building .venv with setup.sh failed. The health check did not run. If a .venv folder was left behind, remove it, fix the error above, and run this command again."
        exit 1
      fi
      ;;
  esac

  if ! bash tools/get_llamacpp.sh; then
    echo "The packages were installed, but downloading the llama.cpp programs failed. The health check did not run. Fix the error above, then run bash tools/get_llamacpp.sh and this command again."
    exit 1
  fi
  if ! activate_venv; then
    echo "The setup finished, but .venv still has no activate script. The health check did not run."
    exit 1
  fi
fi

# On Windows, call the virtual environment's python.exe by name so a system Python is not used by mistake.
# On Linux and macOS, activation puts python on the path.
if [ -f .venv/Scripts/python.exe ]; then
  PY_BIN=".venv/Scripts/python.exe"
elif command -v python >/dev/null 2>&1; then
  PY_BIN=python
elif command -v python.exe >/dev/null 2>&1; then
  PY_BIN=python.exe
else
  echo "The virtual environment is on, but the python command was not found. The health check did not run."
  exit 1
fi

"$PY_BIN" tools/healthcheck.py "$@"
