#!/usr/bin/env bash
# This script runs every experiment of area 4 in the right order and so recreates its results. It
# belongs to area 4.
#
# More technical notes:
# Reproduce every number in RESULTS.md (about 10 minutes on 8 CPU cores).
set -euo pipefail
cd "$(dirname "$0")"; source ../.venv/bin/activate 2>/dev/null || true
python synthetic_data.py --mode template --n 2000
python synthetic_data.py --mode rejection_sampling --n 1000 --k 4 --model HuggingFaceTB/SmolLM2-360M   # bigger model generates, 135M learns (verified-only)
python sft_lora.py                       # arithmetic task, template data
python sft_lora.py --data data/sft_rejection_sampling.jsonl --out-dir outputs/sft_rs --results results/sft_arith_rejection_sampling.json
python dpo_tiny.py
python reward_model_sketch.py            # from the base model (expected: fails to learn in 60 steps - see RESULTS)
python reward_model_sketch.py --model outputs/sft/merged --steps 150 --results results/reward_model_from_sft.json   # from the SFT model
python sft_lora.py --dataset capybara --model HuggingFaceTB/SmolLM2-135M-Instruct --out-dir outputs/sft_capybara   # real-text smoke test
python dpo_tiny.py --dataset ultrafeedback --base HuggingFaceTB/SmolLM2-135M-Instruct --n-pairs 64                # real-text smoke test
