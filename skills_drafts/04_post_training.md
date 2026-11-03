# Skill draft: area 4, post-training

## When to use this area

Use this area when someone wants to see supervised fine-tuning with LoRA, DPO, a reward model or synthetic data on a model small enough for a CPU.

## Exact commands

Run these commands from the folder `04_post_training`.

```bash
python synthetic_data.py --mode template --n 2000
python synthetic_data.py --mode rejection_sampling --n 1000 --k 4 --model HuggingFaceTB/SmolLM2-360M
python sft_lora.py                    # Measures the base model, trains with LoRA, merges and measures again.
python dpo_tiny.py                    # Runs DPO on the merged SFT model.
python reward_model_sketch.py         # Trains a Bradley-Terry reward model.
./run_area4.sh                        # Runs everything in the order used for RESULTS.md (about 10 minutes).
```

## Known traps and gotchas

- DPO did not help on the arithmetic task. The accuracy went from 0.565 to 0.50, which is within the noise of plus or minus 0.07 but certainly no improvement. The training log shows that the model mostly pushed the rejected answer down. We did not tune the settings to hide this, and you should not either.
- The reward model did not beat chance on the arithmetic pairs. It only passed an easy sanity task (accuracy 1.0), which proves that the code can learn.
- DPO with LoRA needs the SFT adapter to be merged first, because trl uses the model without the adapter as the reference. The merged model is in `outputs/sft/merged`, so run `sft_lora.py` before `dpo_tiny.py`.
- The test set has 200 problems, so the 95 percent interval is about plus or minus 7 points. Smaller differences are not evidence.
- The comparison between rejection-sampling data and template data is not controlled, because the two runs saw different numbers of examples. Draw no conclusion about which method is better.
- QLoRA needs bitsandbytes and a CUDA GPU, so it is UNTESTED.

## General traps that apply to every area

- Run every command inside the project's virtual environment, which you turn on with `source .venv/bin/activate` from the project folder.
- Run `python tools/download_models.py` once while you are online, and then set `export HF_HUB_OFFLINE=1` to work offline.
- The tests set the variable `AIETK_TEST_MODE`, which sends every results file to a temporary folder. This protects the real results from being overwritten, so do not be surprised that a test run leaves `results/` unchanged.
- Timings come from a shared virtual machine with 8 CPU cores, so they are noisy. Compare ratios and confidence intervals and do not quote single absolute numbers.
- Nothing has been run on a GPU, on Windows or on macOS. Treat the cloud scripts and the setup scripts for those systems as UNTESTED.
