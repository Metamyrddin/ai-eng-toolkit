#!/usr/bin/env bash
# This script fine-tunes a 7-billion-parameter model with LoRA on a GPU. It has NOT been tested. It
# belongs to the cloud part of area 4.
#
# More technical notes:
# UNTESTED (no GPU on the build box). Flags verified only via `trl sft --help` (trl 1.17.1).
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-7B}
trl sft --model_name_or_path "$MODEL" \
  --dataset_name trl-lib/Capybara --output_dir outputs/sft_7b_lora \
  --use_peft --lora_r 16 --lora_alpha 32 --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --learning_rate 2e-4 \
  --max_length 2048 --bf16 true --num_train_epochs 1
