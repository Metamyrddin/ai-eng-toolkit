#!/usr/bin/env bash
# This script starts a training job on several computers at once with torchrun. It has NOT been
# tested. It belongs to the cloud part of area 3.
#
# More technical notes:
# UNTESTED. Run on EACH node; set NODE_RANK=0..N-1 and MASTER_ADDR to node 0's address.
# torchrun docs: https://docs.pytorch.org/docs/stable/elastic/run.html
set -euo pipefail
: "${NODE_RANK:?set NODE_RANK}"; : "${MASTER_ADDR:?set MASTER_ADDR}"; NNODES="${NNODES:-2}"
torchrun --nnodes="$NNODES" --node_rank="$NODE_RANK" --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:29500" ../ddp_cpu.py --backend nccl
