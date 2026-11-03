# Skill draft: area 5, evaluation rigor

## When to use this area

Use this area when someone asks whether a model is really better, how big a test must be, how to compare two models fairly, how much a random seed matters, or whether a benchmark leaked into the training data.

## Exact commands

Run these commands from the folder `05_evals_rigor`.

```bash
python eval_harness.py --models HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M --n 200 --shots 0
python ablation.py --shots 0 1 3 --seeds 0 1 2 3 4 --n 100
python contamination_check.py
```

## Known traps and gotchas

- With 200 questions the 95 percent interval is about plus or minus 7 points. The 135M and 360M models were not significantly different (McNemar p between 0.13 and 0.23), so do not claim that the larger model is better.
- The two scores `acc` and `acc_norm` differ by several points for the same model. Always say which one you report.
- The seed changes the chosen examples and the chosen test questions, so the scores vary by about 0.02 to 0.06 between seeds. Always run at least 3 seeds, and 5 is better.
- The contamination detector looks for 8 identical words in a row. Its recall drops to almost nothing once about 10 to 20 percent of the words are changed, so it cannot find paraphrases, and finding no hits never proves that the data is clean.
- The detector flagged a few near-duplicate questions inside ARC-Easy itself. Finding nothing in WikiText-2 says nothing about the real training data of the models.

## General traps that apply to every area

- Run every command inside the project's virtual environment, which you turn on with `source .venv/bin/activate` from the project folder.
- Run `python tools/download_models.py` once while you are online, and then set `export HF_HUB_OFFLINE=1` to work offline.
- The tests set the variable `AIETK_TEST_MODE`, which sends every results file to a temporary folder. This protects the real results from being overwritten, so do not be surprised that a test run leaves `results/` unchanged.
- Timings come from a shared virtual machine with 8 CPU cores, so they are noisy. Compare ratios and confidence intervals and do not quote single absolute numbers.
- Nothing has been run on a GPU, on Windows or on macOS. Treat the cloud scripts and the setup scripts for those systems as UNTESTED.
