#!/usr/bin/env bash
# This script repeats the check that showed the llama-cpp-python server failing on several
# simultaneous streaming requests. It belongs to area 1.
#
# More technical notes:
# Reproduces the caveat in RESULTS.md: llama-cpp-python's built-in server and CONCURRENT streaming requests.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
MODEL=$(python -c "from common import env; print(env.gguf_path('q4_k_m'))" 2>/dev/null)
python -m llama_cpp.server --model "$MODEL" --port 8081 --n_ctx 2048 > /tmp/llama_cpp_python_server.log 2>&1 &
PID=$!; sleep 12
python 01_inference_serving/bench_serving.py --url http://127.0.0.1:8081/v1/chat/completions --concurrency 1 4 \
   --requests-per-level 8 --max-tokens 32 --trials 3 --out 01_inference_serving/results/serving_llama_cpp_python_server.json
kill $PID
