#!/usr/bin/env bash
# This script starts a training job on the several GPUs of one computer with torchrun. It has NOT been
# tested on GPUs. It belongs to the cloud part of area 3.
#
# More technical notes:
# UNTESTED on GPUs (this box has none). The same scripts ARE tested here on CPU/gloo with 2 processes.
# Single node, 8 GPUs, NCCL backend: ddp_cpu.py has an (untested) --backend nccl path that pins rank -> cuda:LOCAL_RANK.
# torchrun docs: https://docs.pytorch.org/docs/stable/elastic/run.html
set -euo pipefail
torchrun --standalone --nproc_per_node=8 ../ddp_cpu.py --backend nccl
