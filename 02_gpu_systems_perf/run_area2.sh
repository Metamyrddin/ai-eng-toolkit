#!/usr/bin/env bash
# This script runs every experiment of area 2 in the right order and so recreates its results. It
# belongs to area 2.
#
# More technical notes:
set -euo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate 2>/dev/null || true
mkdir -p 02_gpu_systems_perf/results
python 02_gpu_systems_perf/profile_torch.py
python 02_gpu_systems_perf/triton_kernels.py
python 02_gpu_systems_perf/flash_attention_from_scratch.py
python 02_gpu_systems_perf/sdpa_backends.py
# torch.compile: use the improvised toolchain only if g++ is missing
if ! command -v g++ >/dev/null 2>&1; then source tools/env_inductor_cpu_nocompiler.sh; fi
OMP_NUM_THREADS=1 python 02_gpu_systems_perf/compile_bench.py --threads 1 --steps 20 --out 02_gpu_systems_perf/results/compile_1thread.json
python 02_gpu_systems_perf/compile_bench.py --steps 20 --out 02_gpu_systems_perf/results/compile_8threads.json
