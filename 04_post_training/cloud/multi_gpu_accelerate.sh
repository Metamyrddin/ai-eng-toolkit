#!/usr/bin/env bash
# This script trains a LoRA model on several GPUs at once with Accelerate. It has NOT been tested. It
# belongs to the cloud part of area 4.
#
# More technical notes:
# UNTESTED. Data-parallel LoRA SFT on N GPUs. `--multi_gpu` is a documented `accelerate launch` flag.
set -euo pipefail
N=${N:-4}; MODEL=${MODEL:-Qwen/Qwen2.5-7B}
accelerate launch --multi_gpu --num_processes "$N" -m trl.scripts.sft \
  --model_name_or_path "$MODEL" --dataset_name trl-lib/Capybara --output_dir outputs/sft_multi \
  --use_peft --lora_r 16 --per_device_train_batch_size 2 --gradient_accumulation_steps 4 --bf16 true
