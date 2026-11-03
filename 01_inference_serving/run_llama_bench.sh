#!/usr/bin/env bash
# This script runs llama.cpp's own speed benchmark for each quantization level. It belongs to area 1
# and gives us a trustworthy speed reference.
#
# More technical notes:
# Official llama.cpp micro-benchmark (pp = prompt processing tok/s, tg = token generation tok/s) per quant.
# Needs ./.bin from tools/get_llamacpp.sh. Output is llama-bench's own table.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=$(ls -d .bin/llama-b*/ | head -1); export LD_LIBRARY_PATH="$PWD/$BIN"
for Q in "$@"; do
  M=$(python -c "from common import env; print(env.gguf_path('$Q'))" 2>/dev/null)
  "$BIN/llama-bench" -m "$M" -p 128 -n 64 -r 3 -t "$(nproc)"
done
