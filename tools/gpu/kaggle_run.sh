#!/usr/bin/env bash
# This script runs the toolkit's GPU checks on a FREE Kaggle GPU from your own computer, using the Kaggle command line.
#
# How it fits in the project:
#   It generates the notebook for one area (make_notebooks.py), uploads it to Kaggle with "kaggle kernels push",
#   waits until Kaggle has finished running it, and then downloads the result files with "kaggle kernels output".
#   You never have to open the Kaggle website after the one-time setup. See tools/gpu/README.md for the step-by-step guide.
#
# Usage:   bash tools/gpu/kaggle_run.sh <area 01..08 or all> <your kaggle user name> <public GitHub repo url>
# Example: bash tools/gpu/kaggle_run.sh 04 myname https://github.com/Metamyrddin/ai-eng-toolkit.git
#
# STATUS: UNTESTED. The commands follow Kaggle's official documentation, but they have not been run against a real account.
set -euo pipefail

AREA="${1:?Please give an area: 01 to 08, or all}"
KUSER="${2:?Please give your Kaggle user name}"
REPO="${3:?Please give the public GitHub URL of this repository}"

cd "$(dirname "$0")/../.."                     # go to the repository root
source .venv/bin/activate 2>/dev/null || true  # use the project's virtual environment if it exists

# 1. Make sure the Kaggle command line is installed and an access token exists.
command -v kaggle >/dev/null || { echo "The 'kaggle' command was not found. Run: pip install kaggle"; exit 1; }
if [ -z "${KAGGLE_API_TOKEN:-}" ] && [ ! -f "$HOME/.kaggle/access_token" ] && [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "No Kaggle access token found. Follow 'Step 2' in tools/gpu/README.md first."; exit 1
fi

# 2. Generate the notebook and the kernel-metadata.json file for this area.
python tools/gpu/make_notebooks.py --repo-url "$REPO" --kaggle-user "$KUSER" >/dev/null
NAME="$AREA"; [ "$AREA" != "all" ] && NAME="area$AREA"
FOLDER="tools/gpu/generated/kaggle_kernels/$NAME"
SLUG="$KUSER/ai-eng-toolkit-gpu-$NAME"

# 3. Upload the notebook. Kaggle starts running it right away (GPU and internet are switched on in kernel-metadata.json).
kaggle kernels push -p "$FOLDER"

# 4. Wait until the run is finished. We check the status every 30 seconds, for at most 40 minutes.
for i in $(seq 1 80); do
  STATUS="$(kaggle kernels status "$SLUG" 2>&1 || true)"
  echo "[$i] $STATUS"
  case "$STATUS" in
    *COMPLETE*) break ;;
    *ERROR*|*CANCEL*) echo "The Kaggle run did not finish well. Open https://www.kaggle.com/code/$SLUG to read the log."; exit 1 ;;
  esac
  sleep 30
done

# 5. Download everything the notebook saved (the JSON result files) into tools/gpu/results/kaggle_<name>/.
OUTDIR="tools/gpu/results/kaggle_$NAME"
mkdir -p "$OUTDIR"
kaggle kernels output "$SLUG" -p "$OUTDIR" -o
echo "Done. Result files are in $OUTDIR"
ls -la "$OUTDIR"
