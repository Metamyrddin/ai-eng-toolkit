#!/usr/bin/env bash
# This script starts a vLLM server that uses speculative decoding to write text faster. It has NOT
# been tested, because it needs a GPU. It belongs to the cloud part of area 1.
#
# More technical notes:
# UNTESTED (no GPU). Speculative decoding in vLLM via --speculative-config (JSON), documented at
# https://docs.vllm.ai/en/latest/features/speculative_decoding/ ("method": draft_model | ngram | eagle3 | ...).
set -euo pipefail
TARGET="${1:-Qwen/Qwen2.5-7B-Instruct}"; DRAFT="${2:-Qwen/Qwen2.5-0.5B-Instruct}"
vllm serve "$TARGET" --port 8000 --gpu-memory-utilization 0.90 \
  --speculative-config "{\"method\": \"draft_model\", \"model\": \"$DRAFT\", \"num_speculative_tokens\": 4}"
# Prompt-lookup alternative (no draft model, good for summarisation/RAG/code editing):
#   --speculative-config '{"method": "ngram", "num_speculative_tokens": 4}'
