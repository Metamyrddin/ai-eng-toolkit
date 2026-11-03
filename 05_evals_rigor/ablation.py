"""This file runs a grid of experiments with several settings and several random seeds, and reports
the averages with confidence intervals. It is part of area 5, which is about measuring models
honestly.

More technical notes:
Ablation + seed-variance harness: grid over factors x seeds, reported with mean / std / t-CI / paired tests.

Demo ablation (cheap, no training): effect of the NUMBER OF FEW-SHOT EXEMPLARS and of the SCORING RULE
(acc vs acc_norm) on ARC-Easy for a small model. The random seed controls (a) which exemplars are drawn and (b)
which test items are subsampled - so the spread across seeds is the *evaluation-protocol variance* that is often
ignored when people report one number.
Generic use: `run_grid(fn, grid, seeds)` accepts any function (config, seed) -> score, e.g. a training run.

Run: python ablation.py --model HuggingFaceTB/SmolLM2-135M --shots 0 1 3 --seeds 0 1 2 3 4 --n 100
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import eval_harness as eh  # noqa: E402
import stats_utils as su  # noqa: E402
from common import env  # noqa: E402


def run_grid(fn, grid: dict, seeds):
    """fn(config: dict, seed: int) -> float. Returns {config_key: {"scores": [...], "summary": {...}}}."""
    keys = list(grid)
    out = {}
    for combo in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(zip(keys, combo))
        scores = [fn(cfg, s) for s in seeds]
        out[str(cfg)] = {"config": cfg, "scores": scores, "summary": su.seed_summary(scores)}
    return out


def paired_over_seeds(res, key_a, key_b):
    """Seed-paired difference between two configs (same seeds -> same exemplar/subsample draws)."""
    a, b = res[key_a]["scores"], res[key_b]["scores"]
    d = [x - y for x, y in zip(a, b)]
    return {"a": key_a, "b": key_b, "diff_summary": su.seed_summary(d)}


def main():
    """Run the grid of settings and seeds, print the table with confidence intervals, and save
    the results.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=env.SMOL_135M)
    ap.add_argument("--shots", type=int, nargs="+", default=[0, 1, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "ablation_shots.json"))
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    train = eh.load_arc("train")
    cache = {}

    def fn(cfg, seed):
        """Return the score for one setting and one seed. Results are cached, so one pass over
        the data gives both accuracy variants.
        """
        key = (cfg["shots"], seed)
        if key not in cache:  # one pass yields both acc and acc_norm
            items = eh.load_arc("test", a.n, seed)
            _, acc, accn = eh.evaluate_model(model, tok, items, eh.make_shots_fn(train, cfg["shots"], seed))
            cache[key] = (sum(acc) / len(acc), sum(accn) / len(accn))
        return cache[key][0 if cfg["metric"] == "acc" else 1]

    res = run_grid(fn, {"shots": a.shots, "metric": ["acc", "acc_norm"]}, a.seeds)
    for k, v in res.items():
        s = v["summary"]; print(k, f"mean={s['mean']:.3f} std={s['std']:.3f} ci95=({s['ci'][0]:.3f},{s['ci'][1]:.3f})")
    comps = [paired_over_seeds(res, str({"shots": s, "metric": "acc_norm"}), str({"shots": 0, "metric": "acc_norm"}))
             for s in a.shots if s != 0]
    env.save_json({"environment": env.environment_info(), "model": a.model, "n_items_per_seed": a.n, "seeds": a.seeds,
                   "grid": res, "paired_vs_zero_shot_acc_norm": comps}, a.out)


if __name__ == "__main__":
    main()
