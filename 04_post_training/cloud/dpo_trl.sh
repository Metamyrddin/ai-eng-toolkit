#!/usr/bin/env bash
# This script runs DPO training with the trl command on a GPU. It has NOT been tested. It belongs to
# the cloud part of area 4.
#
# More technical notes:
# UNTESTED. Run after merging the SFT adapter into the base (see sft_lora.py for how: merge_and_unload()).
set -euo pipefail
SFT_MODEL=${SFT_MODEL:?path to merged SFT model}
trl dpo --model_name_or_path "$SFT_MODEL" --dataset_name trl-lib/ultrafeedback_binarized \
  --output_dir outputs/dpo_7b --use_peft --lora_r 16 --lora_alpha 32 --beta 0.1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --learning_rate 5e-6 \
  --max_length 1024 --bf16 true --num_train_epochs 1
