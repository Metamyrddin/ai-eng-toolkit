#!/usr/bin/env bash
# This script fine-tunes a large model with QLoRA, which keeps the base model in 4 bits to save
# memory. It has NOT been tested. It belongs to the cloud part of area 4.
#
# More technical notes:
# UNTESTED. Requires a CUDA GPU and `pip install bitsandbytes`. 4-bit NF4 base + LoRA adapters.
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-7B}
trl sft --model_name_or_path "$MODEL" --load_in_4bit \
  --dataset_name trl-lib/Capybara --output_dir outputs/sft_7b_qlora \
  --use_peft --lora_r 16 --lora_alpha 32 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --learning_rate 2e-4 \
  --max_length 2048 --bf16 true --num_train_epochs 1
