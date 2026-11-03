# Area 5 cloud path: testing bigger models (UNTESTED)

There was no GPU when this toolkit was built, so nothing in this file has been run.
Two parts carry over unchanged to bigger models. The file `stats_utils.py` uses only numpy and scipy, and the function `run_grid` in `ablation.py` accepts any function that turns a configuration and a seed into a score.

## Running bigger models with the lm-evaluation-harness

EleutherAI's lm-evaluation-harness is a standard tool for testing large models.
You can start it like this:

```bash
pip install lm-eval
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-7B,dtype=bfloat16 --tasks arc_easy,hellaswag --batch_size auto --device cuda:0 --output_path results/
```

The flags come from the [harness README](https://github.com/EleutherAI/lm-evaluation-harness#basic-usage).
To use vLLM as the engine, replace the model options with `--model vllm --model_args pretrained=...,dtype=auto`.
Afterwards, load the per-question logs that you get with `--log_samples`, and give the right and wrong results to `paired_bootstrap` or `mcnemar_exact` in `stats_utils.py`.
This compares two checkpoints with proper uncertainty.

## Using several seeds on real training runs

The following code shows how to run a grid of settings with several seeds:

```python
from ablation import run_grid
def fn(cfg, seed): ...   # Train with the settings in cfg and the given seed, then return the accuracy on held-out data.
table = run_grid(fn, {"lr": [1e-4, 2e-4], "rank": [8, 16]}, seeds=range(5))
```

Use at least 3 seeds, and 5 seeds is better.
Report the mean together with its t-interval, and never report only the best seed.
