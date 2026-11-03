"""This file is a small test runner for multiple-choice questions. It asks the model how likely
each answer is and picks the most likely one, then reports the accuracy with confidence
intervals. It is part of area 5.

More technical notes:
A small, readable multiple-choice evaluation harness (log-likelihood scoring) with proper uncertainty.

Scoring follows the standard "cloze" protocol used by EleutherAI's lm-evaluation-harness
(https://github.com/EleutherAI/lm-evaluation-harness):  for each choice compute  log P(" choice" | prompt)  summed over
the continuation tokens; predict argmax.
   acc      : argmax of the summed log-likelihood
   acc_norm : argmax of (summed log-likelihood / number of characters in the choice)   (length-normalised)
Data: ARC-Easy test split (Clark et al. 2018, arXiv:1803.05457) from the Hugging Face hub id `allenai/ai2_arc`.
Prompt format: "Question: ...\nAnswer:" (+ optional k-shot exemplars drawn from the TRAIN split with a seeded RNG).
We report Wilson and bootstrap 95% CIs; per-item correctness is saved so models can be compared with paired tests.

Run: python eval_harness.py --models HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M --n 200 --shots 0
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import stats_utils as su  # noqa: E402
from common import env  # noqa: E402

LETTERS = None


def load_arc(split="test", n=None, seed=0):
    """Load the ARC-Easy questions, drop the broken ones, and return a reproducible random
    sample of the requested size.
    """
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    items = []
    for r in ds:
        if r["answerKey"] not in r["choices"]["label"]:
            continue
        items.append({"id": r["id"], "question": r["question"], "choices": r["choices"]["text"],
                      "label": r["choices"]["label"].index(r["answerKey"])})
    if n is not None:
        random.Random(seed).shuffle(items)  # fixed subsample (seed recorded in results)
        items = items[:n]
    return items


def format_item(it, with_answer=False):
    """Write one question in the 'Question: ... Answer:' format, with the correct answer added
    when it is used as an example.
    """
    s = f"Question: {it['question']}\nAnswer:"
    return s + (f" {it['choices'][it['label']]}\n\n" if with_answer else "")


def build_prompt(it, shots):
    """Build the full prompt: the example questions (shots) first, then the question to be
    answered.
    """
    return "".join(format_item(s, True) for s in shots) + format_item(it)


@torch.no_grad()
def continuation_loglik(model, tok, context: str, continuations: list[str], device="cpu"):
    """Sum log P(continuation tokens | context) for each continuation. Context/continuation are tokenised
    separately and concatenated (continuations start with a space, so boundaries are stable for BPE tokenizers)."""
    ctx = tok(context, add_special_tokens=False).input_ids
    conts = [tok(c, add_special_tokens=False).input_ids for c in continuations]
    seqs = [ctx + c for c in conts]
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=device)
    att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=device)
    logp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), dim=-1)
    out = []
    for i, c in enumerate(conts):
        pos = torch.arange(len(ctx) - 1, len(ctx) + len(c) - 1)
        out.append(logp[i, pos, torch.tensor(c)].sum().item())
    return out, [len(c) for c in conts]


def evaluate_model(model, tok, items, shots_fn):
    """Score every answer choice of every question with the model, and return the per-question
    rows and the lists of correct/incorrect results.
    """
    rows = []
    for it in items:
        conts = [" " + c for c in it["choices"]]
        ll, _ = continuation_loglik(model, tok, build_prompt(it, shots_fn(it)), conts)
        chars = [len(c) for c in it["choices"]]
        rows.append({"id": it["id"], "label": it["label"], "pred": int(np.argmax(ll)),
                     "pred_norm": int(np.argmax([l / c for l, c in zip(ll, chars)]))})
    acc = [float(r["pred"] == r["label"]) for r in rows]
    accn = [float(r["pred_norm"] == r["label"]) for r in rows]
    return rows, acc, accn


def summarize(scores):
    """Turn a list of right/wrong results into the accuracy together with its Wilson and
    bootstrap confidence intervals.
    """
    k, n = int(sum(scores)), len(scores)
    return {"acc": k / n, "n": n, "wilson95": su.wilson_interval(k, n), "bootstrap95": su.bootstrap_ci(scores)}


def make_shots_fn(train_items, k, seed):
    """Pick the example questions once, with a seeded random generator, and return a function
    that gives them to every question.
    """
    rng = random.Random(seed)
    shots = rng.sample(train_items, k) if k else []
    return lambda it: shots


def main():
    """Evaluate each model, compare the models with paired tests and save the results."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[env.SMOL_135M, "HuggingFaceTB/SmolLM2-360M"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "eval_arc_easy.json"))
    a = ap.parse_args()
    items = load_arc("test", a.n, a.seed)
    train = load_arc("train")
    res = {"environment": env.environment_info(), "task": "ARC-Easy test (log-likelihood choice)", "n": len(items),
           "subsample_seed": a.seed, "shots": a.shots, "models": {}}
    per_item = {}
    for m in a.models:
        tok = AutoTokenizer.from_pretrained(m)
        model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float32).eval()
        t0 = time.perf_counter()
        rows, acc, accn = evaluate_model(model, tok, items, make_shots_fn(train, a.shots, a.seed))
        res["models"][m] = {"acc": summarize(acc), "acc_norm": summarize(accn), "seconds": round(time.perf_counter() - t0, 1)}
        per_item[m] = {"acc": acc, "acc_norm": accn}
        print(m, res["models"][m], flush=True)
        del model
    ms = list(per_item)
    if len(ms) == 2:
        for metric in ("acc", "acc_norm"):
            res.setdefault("paired_comparison", {})[metric] = {
                "A": ms[0], "B": ms[1],
                "paired_bootstrap_B_minus_A": su.paired_bootstrap(per_item[ms[1]][metric], per_item[ms[0]][metric]),
                "mcnemar_exact": su.mcnemar_exact(per_item[ms[0]][metric], per_item[ms[1]][metric])}
    res["per_item_correct"] = per_item
    env.save_json(res, a.out)


if __name__ == "__main__":
    main()
