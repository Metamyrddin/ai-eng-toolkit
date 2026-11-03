"""This file trains a small reward model, which learns to give a higher score to the better answer
of each pair. It is part of area 4 and is the first step of reinforcement learning from human
feedback.

More technical notes:
Reward-model sketch: Bradley-Terry pairwise training with a scalar head (LoRA on SmolLM2-135M).

A reward model r(x, y) is trained so that preferred answers score higher:
        L = -log sigmoid( r(x, y_chosen) - r(x, y_rejected) )          (Bradley & Terry 1952; used in RLHF,
        Ouyang et al. 2022 arXiv:2203.02155 and Stiennon et al. 2020 arXiv:2009.01325)
Implemented with a plain PyTorch loop (short enough to read) on top of AutoModelForSequenceClassification(num_labels=1).
(trl also provides a RewardTrainer: https://huggingface.co/docs/trl/reward_trainer)
We report pairwise accuracy on held-out pairs before (~50% expected for a random head) and after training.
Run: python synthetic_data.py --mode template && python reward_model_sketch.py [--steps 60]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import task  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from common import env  # noqa: E402

HERE = Path(__file__).parent


def score(model, tok, prompts, answers):
    """Return the reward (one number per example) that the model gives to each prompt-plus-
    answer text.
    """
    enc = tok([p + a for p, a in zip(prompts, answers)], return_tensors="pt", padding=True)
    return model(**enc).logits.squeeze(-1)


@torch.no_grad()
def pair_accuracy(model, tok, pairs, bs=32):
    """Return how often the reward of the chosen answer is higher than the reward of the
    rejected answer, and the average gap.
    """
    model.eval(); ok = 0; margin = 0.0
    for i in range(0, len(pairs), bs):
        ch = pairs[i:i + bs]
        rc = score(model, tok, [p["prompt"] for p in ch], [p["chosen"] for p in ch])
        rr = score(model, tok, [p["prompt"] for p in ch], [p["rejected"] for p in ch])
        ok += (rc > rr).sum().item(); margin += (rc - rr).sum().item()
    return {"pairwise_accuracy": ok / len(pairs), "mean_margin": margin / len(pairs), "n": len(pairs)}


def main():
    """Train the reward model with the Bradley-Terry loss, measure pairwise accuracy before and
    after, and save the results.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=env.SMOL_135M)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--easy", action="store_true", help="sanity mode: rejected answer is a non-number word (learnable by mechanics alone)")
    ap.add_argument("--results", default=str(HERE / "results" / "reward_model.json"))
    a = ap.parse_args()
    env.seed_everything(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1, dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="SEQ_CLS"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train = [json.loads(l) for l in open(HERE / "data" / "pairs_template.jsonl")]
    test_probs = [json.loads(l) for l in open(HERE / "data" / "test_problems.jsonl")]
    rng = random.Random(7)
    if a.easy:  # replace every rejected answer by a non-numeric word
        words = [" banana", " none", " maybe", " red", " cat", " hello"]
        train = [{**p, "rejected": rng.choice(words)} for p in train]
    test = [{"prompt": p["prompt"], "chosen": task.completion_for(p),
             "rejected": (rng.choice(words) if a.easy else f" {p['answer'] + rng.choice([-10, -2, -1, 1, 2, 10])}")} for p in test_probs]
    res = {"environment": env.environment_info(), "model": a.model, "trainable_params": trainable,
           "easy": a.easy, "steps": a.steps, "batch": a.batch, "lr": a.lr, "seed": a.seed, "before": pair_accuracy(model, tok, test)}
    print("BEFORE", res["before"])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    losses, t0 = [], time.perf_counter()
    model.train()
    for step in range(a.steps):
        ch = [train[(step * a.batch + i) % len(train)] for i in range(a.batch)]
        rc = score(model, tok, [p["prompt"] for p in ch], [p["chosen"] for p in ch])
        rr = score(model, tok, [p["prompt"] for p in ch], [p["rejected"] for p in ch])
        loss = -F.logsigmoid(rc - rr).mean()
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        if step % 10 == 0:
            print(step, round(loss.item(), 4), flush=True)
    res["train_seconds"] = round(time.perf_counter() - t0, 1)
    res["loss_every_10"] = [round(l, 4) for l in losses[::10]]
    res["after"] = pair_accuracy(model, tok, test); print("AFTER", res["after"])
    env.save_json(res, a.results)


if __name__ == "__main__":
    main()
