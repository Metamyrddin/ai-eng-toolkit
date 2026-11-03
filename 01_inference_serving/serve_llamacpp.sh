#!/usr/bin/env bash
# This script starts the official llama.cpp server on the CPU, so that bench_serving.py has something
# to talk to. It belongs to area 1.
#
# More technical notes:
# Start the official llama.cpp `llama-server` (OpenAI-compatible, continuous batching) on CPU.
# Needs ./.bin from `tools/get_llamacpp.sh`. Tested on this box (Linux x64 CPU build).
# Usage: ./serve_llamacpp.sh [quant=q4_k_m] [port=8080] [slots=4]
#   slots = number of requests decoded concurrently in one batch (llama-server --parallel).
set -euo pipefail
cd "$(dirname "$0")/.."
QUANT="${1:-q4_k_m}"; PORT="${2:-8080}"; SLOTS="${3:-4}"
BIN=$(ls -d .bin/llama-b*/ | head -1)
MODEL=$(python -c "from common import env; print(env.gguf_path('$QUANT'))" 2>/dev/null)
export LD_LIBRARY_PATH="$PWD/$BIN"
# total ctx is shared across slots, so give each slot 1024 tokens.
exec "$BIN/llama-server" -m "$MODEL" --port "$PORT" -np "$SLOTS" -c $((SLOTS*1024)) -t "$(nproc)" --no-webui
