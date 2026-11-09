#!/usr/bin/env bash
# This script downloads the ready-made llama.cpp programs (such as llama-server) into the .bin folder.
# On Linux it downloads the official CPU tar file. On Windows Git Bash it runs tools/get_llamacpp.ps1,
# which downloads the official Windows CPU zip. The Linux tar file does not run on Windows.
#
# More technical notes:
# Download the official prebuilt llama.cpp CPU binaries into ./.bin (llama-server, llama-bench, ...).
set -euo pipefail
cd "$(dirname "$0")/.."
# These system names mean Bash is running on Windows. Hand the download to the PowerShell script.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    if command -v cygpath >/dev/null 2>&1; then
      WIN_PS1=$(cygpath -w "$PWD/tools/get_llamacpp.ps1")
    else
      WIN_PS1="$PWD/tools/get_llamacpp.ps1"
    fi
    if command -v powershell.exe >/dev/null 2>&1; then
      PS_BIN=powershell.exe
    elif command -v powershell >/dev/null 2>&1; then
      PS_BIN=powershell
    else
      echo "Downloading the llama.cpp programs failed. The powershell command was not found."
      exit 1
    fi
    MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' "$PS_BIN" -NoProfile -ExecutionPolicy Bypass -File "$WIN_PS1"
    exit $?
    ;;
esac
mkdir -p .bin && cd .bin
# 'releases/latest' on this repo may point at a non-build tag, so scan recent releases for a build asset.
URL=$(curl -s "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print(next(a["browser_download_url"] for r in d for a in r["assets"] if a["name"].endswith("bin-ubuntu-x64.tar.gz")))')
echo "downloading $URL"; echo "$URL" > VERSION_URL.txt
curl -L -o llama.tar.gz "$URL" && tar xzf llama.tar.gz && rm llama.tar.gz
ls
