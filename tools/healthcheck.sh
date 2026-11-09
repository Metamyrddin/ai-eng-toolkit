#!/usr/bin/env bash
# This small script runs the health check inside the project's virtual environment.
# It exists so that beginners can type one command: bash tools/healthcheck.sh
# Any extra words you type (for example --fast) are passed on to healthcheck.py.
#
# If the .venv folder is missing, this script builds it and then runs the checks.
# The first build can take several minutes, because it downloads packages and models.
# If .venv is already there, this script does not build it again.
#
# The build command is bash setup.sh. That file was read before this change.
# It does not call the health check, so the two scripts do not call each other in a loop.
# A guard variable still stops a loop if setup.sh ever starts this script while a build is running.
# setup.sh installs the packages and downloads the models. It does not download the llama.cpp
# programs. This script runs bash tools/get_llamacpp.sh after setup.sh, because the health
# check looks for the llama-server program.
#
# Windows was not tested. setup.sh is for Linux and macOS. It uses the python3 command and
# creates .venv/bin/activate. Git Bash on Windows uses .venv/Scripts/activate, and the Python
# command there is often python instead of python3. On Windows, run
# powershell -ExecutionPolicy Bypass -File setup.ps1, or use WSL2, which is Linux inside
# Windows. Triton and DeepSpeed need WSL2, because they have no normal Windows installers.
set -uo pipefail
cd "$(dirname "$0")/.."                       # Move to the repository root, wherever you started from.

# Turn on an existing virtual environment. Return 0 when it worked, and 1 when no activate file exists.
activate_venv() {
  if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate                 # This is the activate file on Linux and macOS.
    return 0
  fi
  if [ -f .venv/Scripts/activate ]; then
    source .venv/Scripts/activate             # This is the activate file in Git Bash on Windows.
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

  # This stops a loop if setup.sh, or something it runs, starts the health check again.
  if [ -n "${AI_ENG_HEALTHCHECK_BUILDING:-}" ]; then
    echo "The health check started again while it was already building .venv. Stopping so setup and the health check do not call each other forever."
    exit 1
  fi

  # These system names mean the script is running in Bash on Windows. setup.sh cannot build that .venv.
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
      echo "No .venv found. This automatic build was not tested on Windows."
      echo "setup.sh is for Linux and macOS. It uses python3 and creates .venv/bin/activate."
      echo "Git Bash on Windows uses .venv/Scripts/activate, and the Python command is often python instead of python3."
      echo "On Windows, run: powershell -ExecutionPolicy Bypass -File setup.ps1"
      echo "Or use WSL2 (Linux inside Windows) and run this health check there. Triton and DeepSpeed need WSL2, because they have no normal Windows installers."
      exit 1
      ;;
  esac

  echo "No .venv found. Building it now with setup.sh; this can take several minutes and downloads packages and models."
  if ! AI_ENG_HEALTHCHECK_BUILDING=1 bash setup.sh; then
    echo "Building .venv with setup.sh failed. The health check did not run. If a .venv folder was left behind, remove it, fix the error above, and run this command again."
    exit 1
  fi
  if ! bash tools/get_llamacpp.sh; then
    echo "The packages were installed, but downloading the llama.cpp programs failed. The health check did not run. Fix the error above, then run bash tools/get_llamacpp.sh and this command again."
    exit 1
  fi
  if ! activate_venv; then
    echo "setup.sh finished, but .venv still has no activate script. The health check did not run."
    exit 1
  fi
fi

if ! command -v python >/dev/null 2>&1; then
  echo "The virtual environment is on, but the python command was not found. The health check did not run."
  exit 1
fi

python tools/healthcheck.py "$@"
