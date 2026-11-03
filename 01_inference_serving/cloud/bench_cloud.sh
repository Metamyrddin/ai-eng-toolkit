#!/usr/bin/env bash
# This script measures how fast a vLLM server answers, by running our benchmark client against it. It
# has NOT been tested, because it needs a GPU server. It belongs to the cloud part of area 1.
#
# More technical notes:
# UNTESTED against vLLM (no GPU). The client itself (bench_serving.py) IS tested against llama.cpp's server.
# Start a server first (vllm_serve.sh), then:
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
python ../bench_serving.py --url http://127.0.0.1:8000/v1/chat/completions --model "$MODEL" \
  --concurrency 1 4 16 64 --requests-per-level 64 --max-tokens 128 --out results_vllm.json
