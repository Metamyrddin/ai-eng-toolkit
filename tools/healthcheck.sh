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
fi
python tools/healthcheck.py "$@"
