# 4 · Post-training: SFT, LoRA, DPO, reward models, synthetic data

This area shows how a pretrained model is taught to behave better.
We do it with a model of 135 million parameters, on a CPU, in a few minutes.

All training here uses one **verifiable toy task** ([`task.py`](task.py)): 1–2 digit addition/subtraction, `Question: What is 86 minus 10?\nAnswer:`.
Answers can be checked by code, so every number is trustworthy; train and test problems are disjoint by construction. It is a *toy* — it
demonstrates mechanics, not general capability.

## Mechanisms in plain words

**SFT (supervised fine-tuning)** = ordinary next-token training on (prompt, desired answer) pairs, with the loss counted on the answer
tokens only. **LoRA** ([Hu et al. 2021](https://arxiv.org/abs/2106.09685)) freezes each weight matrix `W` and learns a low-rank correction
`ΔW = (α/r)·B·A`; only `A`,`B` are trained (3.5 % of parameters here, measured), and after training `ΔW` can be *merged* into `W`
(`sft_lora.py` verifies merged ≡ adapter outputs). Libraries: [peft](https://huggingface.co/docs/peft/index),
[trl SFTTrainer](https://huggingface.co/docs/trl/sft_trainer). QLoRA ([Dettmers et al. 2023](https://arxiv.org/abs/2305.14314)) additionally
keeps the frozen base in 4-bit — GPU-only (bitsandbytes), see `cloud/`.

**DPO** ([Rafailov et al. 2023](https://arxiv.org/abs/2305.18290)) learns from preference pairs *(prompt, chosen, rejected)* without a reward model or RL:
it raises the policy's log-probability of `chosen` relative to `rejected`, measured against a frozen reference model, scaled by β.
[trl DPOTrainer docs](https://huggingface.co/docs/trl/dpo_trainer). With LoRA, trl uses the adapter-disabled model as reference, so we merge the SFT
adapter first (`outputs/sft/merged`) and run DPO on top.

**Reward model.** A scalar-output head trained with the Bradley–Terry loss `−log σ(r_chosen − r_rejected)`; the first stage of RLHF
([Ouyang et al. 2022](https://arxiv.org/abs/2203.02155), [Stiennon et al. 2020](https://arxiv.org/abs/2009.01325)). `reward_model_sketch.py` is a plain PyTorch loop so
you can read every line, and reports held-out pairwise accuracy before/after training.

**Synthetic data.** (1) *Template generation*: programmatic, perfectly labelled. (2) *Rejection sampling / best-of-n*: sample answers from a
model, keep only those a **verifier** accepts, train on the survivors — the idea behind STaR ([Zelikman et al. 2022](https://arxiv.org/abs/2203.14465)) and RFT
([Yuan et al. 2023](https://arxiv.org/abs/2308.01825)). `synthetic_data.py` reports the real *yield* (fraction of samples passing the verifier).

## Laptop / offline path (tested, CPU, ~10 min total)

```bash
python synthetic_data.py --mode template --n 2000
python synthetic_data.py --mode rejection_sampling --n 1000 --k 4 --model HuggingFaceTB/SmolLM2-360M
python sft_lora.py                          # base → LoRA SFT → merge; before/after accuracy
python dpo_tiny.py                          # DPO on the merged SFT model
python reward_model_sketch.py               # Bradley–Terry reward model
python sft_lora.py --dataset capybara       # real-text smoke test (loss only)
python dpo_tiny.py --dataset ultrafeedback --n-pairs 64   # real-text smoke test (training dynamics only)
./run_area4.sh                              # everything, in the order used for RESULTS.md
```

## Cloud-GPU path — [`cloud/`](cloud/) — **UNTESTED**

The `cloud/` folder has templates for the `trl` command and for `accelerate launch`.
They are meant for models with 7 to 8 billion parameters, and they cover LoRA, QLoRA and DPO, plus a sketch of data generation with vLLM.
All of them are untested.

## Honest limitations

- The test set has 200 problems, so a 95 percent Wilson interval is about plus or minus 7 accuracy points. Differences smaller than that are not evidence (area 5 has the statistics tools).
- We used one seed per setting, to save CPU time. The file `ablation.py` in area 5 shows how to run several seeds.
- The `ultrafeedback` and `capybara` runs only show that the code for real data trains. They make **no** claims about quality.
