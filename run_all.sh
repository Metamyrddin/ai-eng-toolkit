#!/usr/bin/env bash
# This script runs all eight areas one after the other and rebuilds every results file. It is the
# single command that recreates the whole project.
#
# More technical notes:
# Regenerates every results/*.json and RESULTS.md (about 1 hour on an 8-core CPU laptop; needs the models cached: python tools/download_models.py).
# Runs areas sequentially on purpose: parallel runs would distort the timing measurements.
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
bash tools/get_llamacpp.sh || echo "no llama.cpp binaries: serving benchmarks will be skipped/fail"
bash 01_inference_serving/run_area1.sh
bash 02_gpu_systems_perf/run_area2.sh
( cd 03_distributed_training
  torchrun --nproc_per_node=2 ddp_cpu.py; torchrun --nproc_per_node=2 fsdp_cpu.py; torchrun --nproc_per_node=2 collectives_demo.py
  python mixed_precision_demo.py
  python zero_memory.py > results/zero_memory_7.5B_64gpus.txt; python zero_memory.py --params-b 7 --gpus 8 > results/zero_memory_7B_8gpus.txt )
bash 04_post_training/run_area4.sh
( cd 05_evals_rigor
  python eval_harness.py --models HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M --n 200 --shots 0
  python contamination_check.py
  python ablation.py --shots 0 1 3 --seeds 0 1 2 3 4 --n 100 )
( cd 06_data_engineering
  python fetch_sample.py --n 3000; python minhash_dedup.py; python quality_filters.py; python streaming_pipeline.py --n-docs 3000 --source cache )
( cd 07_rag_agents
  python retrieval_eval.py; python rag_pipeline.py --n 30; python agent_loop.py --eval )
( cd 08_safety_interp
  python redteam_harness.py; python logit_lens.py )
python tools/make_results.py
