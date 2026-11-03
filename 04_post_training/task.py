"""This file defines the tiny arithmetic task that area 4 uses. Every answer can be checked
automatically, which makes it a clean way to test training. It is part of area 4.

More technical notes:
A tiny VERIFIABLE task used across SFT / DPO / reward-model demos: 1-2 digit addition & subtraction.

Why this task? (1) answers can be checked automatically (no judge model needed -> trustworthy numbers),
(2) a 135M model can measurably improve in minutes on CPU, (3) train/test problems are disjoint by construction.
It is a *toy*: improvements here demonstrate the mechanics of post-training, not general capability.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROMPT_TMPL = "Question: What is {a} {op} {b}?\nAnswer:"


def make_problem(rng: random.Random):
    """Create one random addition or subtraction question with its correct answer. Subtraction
    answers are never negative.
    """
    a, b = rng.randint(0, 99), rng.randint(0, 99)
    op = rng.choice(["plus", "minus"])
    if op == "minus" and b > a:
        a, b = b, a  # keep answers non-negative
    ans = a + b if op == "plus" else a - b
    return {"a": a, "op": op, "b": b, "prompt": PROMPT_TMPL.format(a=a, op=op, b=b), "answer": ans}


def make_split(n: int, seed: int, exclude: set | None = None):
    """n unique problems (by (a, op, b)), none in `exclude`."""
    rng = random.Random(seed)
    seen, out = set(exclude or ()), []
    while len(out) < n:
        p = make_problem(rng)
        key = (p["a"], p["op"], p["b"])
        if key in seen:
            continue
        seen.add(key); out.append(p)
    return out


def problem_key(p):
    """Return a short identifier of a question, used to make sure that training and test
    questions never overlap.
    """
    return (p["a"], p["op"], p["b"])


def parse_answer(text: str):
    """First (possibly negative) integer in the generated text, else None."""
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def completion_for(p) -> str:
    """Return the text that the model should write as its answer to a question."""
    return f" {p['answer']}"


def evaluate(model, tok, problems, batch_size=25, max_new_tokens=6, device="cpu"):
    """Greedy exact-match accuracy. Output is cut at the first newline (the format we trained)."""
    import torch
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    correct, parsed, outs = 0, 0, []
    for i in range(0, len(problems), batch_size):
        chunk = problems[i:i + batch_size]
        enc = tok([p["prompt"] for p in chunk], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for p, g in zip(chunk, gen):
            text = tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True).split("\n")[0]
            val = parse_answer(text)
            parsed += val is not None
            correct += val == p["answer"]
            outs.append((p["prompt"], text, p["answer"]))
    return {"accuracy": correct / len(problems), "parseable_rate": parsed / len(problems), "n": len(problems),
            "examples": outs[:5]}
