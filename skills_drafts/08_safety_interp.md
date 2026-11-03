# Skill draft: area 8, safety and interpretability

## When to use this area

Use this area when someone wants to test how a small chat model reacts to harmful requests, jailbreaks, prompt injection and secret leaks, or wants to look inside a small model with the logit lens and activation patching.

## Exact commands

Run these commands from the folder `08_safety_interp`.

```bash
python redteam_harness.py      # Tests 3 small chat models with 50 hand-written prompts.
python logit_lens.py           # Runs the logit lens and activation patching on SmolLM2-135M.
```

## Known traps and gotchas

- The refusal detector is a simple pattern search. It raises a false alarm on "I cannot stress enough how much I love this idea" and misses "Nope, not going to do it". Not refusing is also not the same as producing harmful content, and tiny models often write nonsense.
- With only 6 to 12 prompts per cell the confidence intervals are very wide. Qwen2.5-0.5B-Instruct refused all direct harmful requests by our pattern while both SmolLM2 models refused none, but we cannot explain why.
- The test prompts are deliberately mild, because they test the refusal behaviour and do not contain harmful instructions.
- The logit lens shows junk in the early and middle layers, which is a known weakness. The correct answer only appears in the last few layers.
- The patching result (information about "France" reaching the last token at block 23) comes from one prompt pair on one small model. It illustrates the method and is not a discovery about circuits.
- The hooks in `logit_lens.py` only fit Llama-style models, because they use `model.model.layers`, `model.model.norm` and `lm_head`.

## General traps that apply to every area

- Run every command inside the project's virtual environment, which you turn on with `source .venv/bin/activate` from the project folder.
- Run `python tools/download_models.py` once while you are online, and then set `export HF_HUB_OFFLINE=1` to work offline.
- The tests set the variable `AIETK_TEST_MODE`, which sends every results file to a temporary folder. This protects the real results from being overwritten, so do not be surprised that a test run leaves `results/` unchanged.
- Timings come from a shared virtual machine with 8 CPU cores, so they are noisy. Compare ratios and confidence intervals and do not quote single absolute numbers.
- Nothing has been run on a GPU, on Windows or on macOS. Treat the cloud scripts and the setup scripts for those systems as UNTESTED.
