#!/usr/bin/env bash
# This script downloads the ready-made llama.cpp programs (such as llama-server) into the .bin folder.
#
# More technical notes:
# Download the official prebuilt llama.cpp CPU binaries (Linux x64) into ./.bin  (llama-server, llama-bench, ...)
# Windows: download llama-<tag>-bin-win-cpu-x64.zip from https://github.com/ggml-org/llama.cpp/releases manually.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .bin && cd .bin
# 'releases/latest' on this repo may point at a non-build tag, so scan recent releases for a build asset.
URL=$(curl -s "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print(next(a["browser_download_url"] for r in d for a in r["assets"] if a["name"].endswith("bin-ubuntu-x64.tar.gz")))')
echo "downloading $URL"; echo "$URL" > VERSION_URL.txt
curl -L -o llama.tar.gz "$URL" && tar xzf llama.tar.gz && rm llama.tar.gz
ls
