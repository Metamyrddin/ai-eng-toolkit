#!/usr/bin/env bash
# This is the setup script for Linux and macOS. It creates the virtual environment, installs the
# packages and downloads the models. It is the first thing a new user runs.
#
# More technical notes:
# Linux / macOS setup. Creates ./.venv, installs CPU requirements, downloads small models (~2.5 GB total).
# Usage: ./setup.sh [--no-models]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY -c 'import sys; assert sys.version_info >= (3, 10), "need Python >= 3.10"' 
$PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt || { echo "NOTE: if triton failed (macOS/ARM), remove it from requirements-cpu.txt and re-run"; exit 1; }
# prebuilt llama.cpp python wheel (no compiler needed); falls back to a source build only if you remove --only-binary
python -m pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
  || echo "WARNING: no prebuilt llama-cpp-python wheel for this platform - area 1 llama.cpp scripts and area 7 agent will not run"
if [[ "${1:-}" != "--no-models" ]]; then
  python tools/download_models.py
fi
echo
echo "Done. Next:  source .venv/bin/activate && pytest -q -m 'not slow'   (fast tests)"
echo "Offline afterwards:  export HF_HUB_OFFLINE=1"
