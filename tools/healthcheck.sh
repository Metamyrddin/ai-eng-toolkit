#!/usr/bin/env bash
# This small script runs the health check inside the project's virtual environment.
# It exists so that beginners can type one command: bash tools/healthcheck.sh
# Any extra words you type (for example --fast) are passed on to healthcheck.py.
set -uo pipefail
cd "$(dirname "$0")/.."                       # move to the repository root, wherever you started from
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate                   # Linux and macOS layout of a virtual environment
elif [ -f .venv/Scripts/activate ]; then
  source .venv/Scripts/activate               # Git Bash on Windows layout
else
  # No virtual environment was found in the usual Linux or Windows place.
  # Stop here with a clear message, so a beginner does not only see "python: command not found".
  echo "No virtual environment (.venv) was found. Run bash setup.sh first, or rebuild .venv from requirements-lock-cpu-linux-py313.txt."
  exit 1
fi
python tools/healthcheck.py "$@"
