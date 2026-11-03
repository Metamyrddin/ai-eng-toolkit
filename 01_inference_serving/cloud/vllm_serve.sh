#!/usr/bin/env bash
# This script starts a vLLM model server on a machine with an NVIDIA GPU. It has NOT been tested,
# because the build computer has no GPU. It belongs to the cloud part of area 1.
#
# More technical notes:
# UNTESTED on this box (no GPU here). Flags were checked against the vLLM CLI reference
# (https://docs.vllm.ai/en/latest/cli/serve/), but never executed by us.
# Needs a Linux machine with an NVIDIA GPU and `pip install -r requirements-gpu.txt` (installs vllm).
#
# Usage: ./vllm_serve.sh [model] [tp_size] [port]
#   e.g. ./vllm_serve.sh Qwen/Qwen2.5-7B-Instruct 1 8000
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"; TP="${2:-1}"; PORT="${3:-8000}"
exec vllm serve "$MODEL" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --enable-prefix-caching \
  --port "$PORT"
