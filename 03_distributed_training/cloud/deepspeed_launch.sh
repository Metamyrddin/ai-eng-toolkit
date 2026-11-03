#!/usr/bin/env bash
# This script starts a DeepSpeed training job with one of the settings files from the
# deepspeed_configs folder. It has NOT been tested on GPUs. It belongs to the cloud part of area 3.
#
# More technical notes:
# UNTESTED (no GPU). Launch a DeepSpeed training script with one of ../deepspeed_configs/*.json.
# The configs' keys were checked against https://www.deepspeed.ai/docs/config-json/ and validated with DeepSpeed's own
# config parser on CPU (see tests/test_deepspeed_configs.py) - but never used for a real GPU run by us.
# Your script must call deepspeed.initialize(model=..., config=<json path>) - see
# https://www.deepspeed.ai/getting-started/
set -euo pipefail
CONFIG="${1:-../deepspeed_configs/zero2_bf16.json}"; SCRIPT="${2:?your_train_script.py}"
deepspeed --num_gpus=8 "$SCRIPT" --deepspeed_config "$CONFIG"
