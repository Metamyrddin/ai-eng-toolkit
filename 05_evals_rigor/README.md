# 5 · Evals & Rigor

This area explains how to tell whether a model really got better or whether you were just lucky.

## Mechanisms in plain words

**Multiple-choice evals by log-likelihood.** For a question with choices A…D the model is never asked to "answer"; we compute
`log P(choice | question)` for each choice, summed over the choice's tokens, and pick the largest — the "cloze" protocol of EleutherAI's
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). `acc` uses the raw sum; `acc_norm` divides by the choice's character count to
reduce the bias toward short answers. `eval_harness.py` is a ~100-line version on ARC-Easy ([Clark et al. 2018](https://arxiv.org/abs/1803.05457)),
saves per-item correctness, and supports k-shot prompts drawn with a seeded RNG from the train split.

**Uncertainty.** An accuracy on n questions is a *sample*. `stats_utils.py` (unit-tested against scipy) provides: Wilson interval for a proportion;
percentile bootstrap; **paired** bootstrap and exact **McNemar** test for comparing two models on the *same* items (much more powerful than comparing two
independent intervals); and a t-interval across seeds. Background: [Dror et al. 2018](https://aclanthology.org/P18-1128/),
[Madaan et al. 2024](https://arxiv.org/abs/2406.10229). Rule of thumb printed by RESULTS.md: with n = 200, the 95 % half-width is ≈ ±7 points.

**Seeds & ablations.** `ablation.py` runs a (factor × seed) grid. In the demo the seed picks the few-shot exemplars and the test subsample, so
the spread across seeds is the *protocol variance* usually left unreported. `run_grid(fn, grid, seeds)` accepts any `(config, seed) → score`
function — e.g. a training run — so you can use it for real ablations on a GPU.

**Contamination.** If test questions appeared in training data, scores are inflated. `contamination_check.py` normalises text, builds word n-grams
(GPT-3 used 13-grams: [Brown et al. 2020, App. C](https://arxiv.org/abs/2005.14165)), and flags an item if ≥ a threshold fraction of its n-grams occur in the corpus.
It is **validated** inside the script: we plant eval items in a clean corpus after corrupting 0–30 % of their words and report detection recall and the false-positive rate for each corruption level.
It finds near-verbatim overlap only — not paraphrases — so "no hits" never proves "clean".

## Laptop / offline path (tested)
```bash
python eval_harness.py --models HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M --n 200 --shots 0
python ablation.py --shots 0 1 3 --seeds 0 1 2 3 4 --n 100
python contamination_check.py
```
The datasets are downloaded once, because `tools/download_models.py` also fetches them, and after that everything runs offline.

## Cloud path — [`cloud/README.md`](cloud/README.md) — UNTESTED
The cloud README explains how to run the same evaluation with the `lm-evaluation-harness` on bigger models and GPUs.
It also explains how to give your own training function to `run_grid`.

## Honest limitations
- The two small models score close to each other on ARC-Easy. The interesting result is the statistics, which show what we can and cannot conclude from 200 questions, and the scores themselves matter less.
- We used only ARC-Easy and a small number of questions, to keep the CPU time low.
