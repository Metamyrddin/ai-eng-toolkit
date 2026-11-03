#!/usr/bin/env bash
# This script starts a vLLM server with a model that was already quantized (shrunk). It has NOT been
# tested, because it needs a GPU. It belongs to the cloud part of area 1.
#
# More technical notes:
# UNTESTED (no GPU). Serve a pre-quantized checkpoint (AWQ/GPTQ/FP8...). vLLM reads quantization_config from the
# model's config.json, so --quantization is usually optional (see `--quantization` in the CLI reference).
# Replace MODEL with a real quantized repo that you have checked exists on the Hugging Face Hub.
set -euo pipefail
MODEL="${1:?pass the HF id of a quantized checkpoint}"
vllm serve "$MODEL" --port 8000 --gpu-memory-utilization 0.90
