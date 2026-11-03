#!/usr/bin/env bash
# This script runs every experiment of area 1 in the right order and so recreates all numbers in its
# RESULTS.md file. It belongs to area 1.
#
# More technical notes:
# Reproduce every measured number in RESULTS.md (about 6-10 minutes on an 8-core CPU).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
mkdir -p 01_inference_serving/results
python 01_inference_serving/kv_cache_demo.py
python 01_inference_serving/bench_hf.py --repeats 5
python 01_inference_serving/bench_llamacpp.py --repeats 5
python 01_inference_serving/quant_from_scratch.py
python 01_inference_serving/speculative_demo.py --repeats 3
python 01_inference_serving/batching_sim.py
python 01_inference_serving/batching_sim.py --arrival-rate 0.05 --out 01_inference_serving/results/batching_sim_overload.json
for NP in 1 4; do
  ./01_inference_serving/serve_llamacpp.sh q4_k_m 8080 $NP > /tmp/llama_server_$NP.log 2>&1 &
  sleep 8
  python 01_inference_serving/bench_serving.py --concurrency 1 2 4 --requests-per-level 16 --max-tokens 64 --trials 5 \
      --out 01_inference_serving/results/serving_slots$NP.json
  pkill -f llama-server || true; sleep 2
done
./01_inference_serving/run_llama_bench.sh q8_0 q4_k_m > 01_inference_serving/results/llama_bench.txt 2>&1
./01_inference_serving/run_llamacpp_python_server_check.sh || true   # reproduces the llama-cpp-python concurrency caveat
