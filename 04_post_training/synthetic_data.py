"""This file creates training data automatically in two ways: from a fixed template, and by letting
a model write many answers and keeping only the correct ones (rejection sampling). It is part of
area 4.

More technical notes:
Synthetic data generation, two ways.

1. template : programmatic generator (task.py) -> perfectly labelled prompt/completion + preference pairs.
2. rejection_sampling (a.k.a. best-of-n / RFT / "context distillation"): sample answers from a MODEL using a few-shot
   prompt, keep only those a programmatic VERIFIER marks correct, then store them as zero-shot SFT examples.
   Reports the real yield (fraction of samples that pass the verifier).
   Idea references: STaR (Zelikman et al. 2022, arXiv:2203.14465), RFT (Yuan et al. 2023, arXiv:2308.01825).

Outputs JSONL files under data/. Run:
    python synthetic_data.py --mode template --n 2000
    python synthetic_data.py --mode rejection_sampling --n 300 --k 4 --model HuggingFaceTB/SmolLM2-135M
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import task  # noqa: E402

from common import env  # noqa: E402

DATA = Path(__file__).parent / "data"
FEWSHOT = "Question: What is 12 plus 30?\nAnswer: 42\n\nQuestion: What is 90 minus 45?\nAnswer: 45\n\nQuestion: What is 7 plus 8?\nAnswer: 15\n\n"


def write_jsonl(rows, path):
    """Write a list of objects to a file with one JSON object per line, creating the folder if
    needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def make_template(n, seed):
    """Create training examples and preference pairs by following a fixed template, with
    answers that are known to be correct.
    """
    test = task.make_split(200, seed=999)                       # fixed held-out set
    train = task.make_split(n, seed=seed, exclude={task.problem_key(p) for p in test})
    sft = [{"prompt": p["prompt"], "completion": task.completion_for(p)} for p in train]
    rng = random.Random(seed + 1)
    pairs = []
    for p in train:
        wrong = p["answer"] + rng.choice([-10, -2, -1, 1, 2, 10])
        pairs.append({"prompt": p["prompt"], "chosen": task.completion_for(p), "rejected": f" {wrong}"})
    write_jsonl(sft, DATA / "sft_template.jsonl"); write_jsonl(pairs, DATA / "pairs_template.jsonl")
    write_jsonl(test, DATA / "test_problems.jsonl")
    return {"mode": "template", "n_sft": len(sft), "n_pairs": len(pairs), "n_test": len(test)}


def rejection_sampling(n, k, model_name, seed, temperature=0.7):
    """Let a model write several answers for each question, keep only answers that the checker
    accepts, and return statistics such as the yield.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    env.seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(model_name); tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
    test = task.make_split(200, seed=999)
    probs = task.make_split(n, seed=seed, exclude={task.problem_key(p) for p in test})
    kept, total, passed = [], 0, 0
    for i in range(0, len(probs), 25):
        chunk = probs[i:i + 25]
        enc = tok([FEWSHOT + p["prompt"] for p in chunk], return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=6, do_sample=True, temperature=temperature, top_p=0.95,
                                 num_return_sequences=k, pad_token_id=tok.pad_token_id)
        out = out.view(len(chunk), k, -1)
        for p, samples in zip(chunk, out):
            got_one = False
            for s in samples:
                text = tok.decode(s[enc.input_ids.shape[1]:], skip_special_tokens=True).split("\n")[0]
                total += 1
                if task.parse_answer(text) == p["answer"]:
                    passed += 1
                    if not got_one:  # keep at most one verified sample per prompt (de-dup)
                        kept.append({"prompt": p["prompt"], "completion": task.completion_for(p)}); got_one = True
    write_jsonl(kept, DATA / "sft_rejection_sampling.jsonl")
    return {"mode": "rejection_sampling", "model": model_name, "prompts": len(probs), "samples_per_prompt": k,
            "samples_total": total, "samples_passing_verifier": passed,
            "sample_yield": round(passed / total, 4), "prompts_with_at_least_one_pass": len(kept),
            "prompt_coverage": round(len(kept) / len(probs), 4), "temperature": temperature}


def main():
    """Generate the requested kind of synthetic data, save it and print how much of it passed
    the checker.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["template", "rejection_sampling"], default="template")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=env.SMOL_135M)
    a = ap.parse_args()
    res = make_template(a.n, a.seed) if a.mode == "template" else rejection_sampling(a.n, a.k, a.model, a.seed)
    print(res)
    env.save_json(res, Path(__file__).parent / "results" / f"synthetic_{a.mode}.json")


if __name__ == "__main__":
    main()
