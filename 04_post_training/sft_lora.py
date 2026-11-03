"""This file fine-tunes a small language model with supervised fine-tuning (SFT) and LoRA, which
trains only a few small extra matrices instead of the whole model. It measures the accuracy
before and after. It is part of area 4.

More technical notes:
SFT + LoRA with peft + trl on SmolLM2-135M (CPU, minutes).

LoRA (Hu et al. 2021, arXiv:2106.09685) freezes the pretrained weight W and learns a low-rank update:
        h = W x + (alpha/r) * B A x ,    A: r x d_in, B: d_out x r      (only A and B are trained)
so trainable parameters are only a few percent of the model (3.5% here, measured) and the update can be merged back into W for free inference.
Docs: peft https://huggingface.co/docs/peft/index   trl SFTTrainer https://huggingface.co/docs/trl/sft_trainer

Pipeline here (task.py = verifiable arithmetic toy task):
   1. measure exact-match accuracy of the base model on 200 held-out problems
   2. LoRA-SFT on synthetic prompt/completion pairs (completion-only loss)
   3. measure accuracy again; merge LoRA into the base weights; verify merged == adapter outputs; save to outputs/
Run: python synthetic_data.py --mode template --n 2000 && python sft_lora.py [--data data/sft_template.jsonl --epochs 1]
For a real-text smoke test instead:  python sft_lora.py --dataset capybara
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import task  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset, load_dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from common import env  # noqa: E402

HERE = Path(__file__).parent


def load_jsonl(p):
    """Read a file where every line is one JSON object, and return the objects as a list."""
    return [json.loads(l) for l in open(p)]


def main():
    """Measure the base model, fine-tune it with LoRA, check that merging the adapter keeps the
    answers, measure again and save the results.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=env.SMOL_135M)
    ap.add_argument("--data", default=str(HERE / "data" / "sft_template.jsonl"))
    ap.add_argument("--dataset", default="arith", choices=["arith", "capybara"])
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-capybara", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(HERE / "outputs" / "sft"))
    ap.add_argument("--results", default=None)
    a = ap.parse_args()
    env.seed_everything(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    test = load_jsonl(HERE / "data" / "test_problems.jsonl") if a.dataset == "arith" else None
    res = {"environment": env.environment_info(), "model": a.model, "dataset": a.dataset, "lora_rank": a.rank,
           "lr": a.lr, "epochs": a.epochs, "batch": a.batch, "seed": a.seed}
    lora = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")

    if a.dataset == "arith":
        res["before"] = task.evaluate(model, tok, test); print("BEFORE", res["before"])
        train = Dataset.from_list(load_jsonl(a.data)); eval_ds = None
        max_len = 64
    else:
        ds = load_dataset("trl-lib/Capybara", split=f"train[:{a.n_capybara + 50}]")
        train, eval_ds, max_len = ds.select(range(a.n_capybara)), ds.select(range(a.n_capybara, a.n_capybara + 50)), 256
    cfg = SFTConfig(output_dir=a.out_dir, per_device_train_batch_size=a.batch, num_train_epochs=a.epochs,
                    learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=5, logging_steps=max(1, min(5, len(train) // a.batch)),
                    max_length=max_len, use_cpu=True, report_to="none", save_strategy="no", seed=a.seed,
                    eval_strategy="no", per_device_eval_batch_size=8, bf16=False)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train, eval_dataset=eval_ds, processing_class=tok,
                         peft_config=lora)
    n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in trainer.model.parameters())
    res["trainable_params"], res["total_params"] = n_train, n_all
    res["trainable_pct"] = round(100 * n_train / n_all, 3)
    if eval_ds is not None:
        res["eval_loss_before"] = trainer.evaluate()["eval_loss"]
    t0 = time.perf_counter(); trainer.train(); res["train_seconds"] = round(time.perf_counter() - t0, 1)
    res["loss_log"] = [(h["step"], round(h["loss"], 4)) for h in trainer.state.log_history if "loss" in h]
    if eval_ds is not None:
        res["eval_loss_after"] = trainer.evaluate()["eval_loss"]
    if a.dataset == "arith":
        res["after_adapter"] = task.evaluate(trainer.model, tok, test); print("AFTER", res["after_adapter"])
        # merge adapter into base weights and verify the merged model is numerically the same function
        probe = tok(["Question: What is 12 plus 30?\nAnswer:"], return_tensors="pt")
        with torch.no_grad():
            before_merge = trainer.model(**probe).logits
        merged = trainer.model.merge_and_unload()
        with torch.no_grad():
            after_merge = merged(**probe).logits
        res["merge_max_abs_logit_diff"] = float((before_merge - after_merge).abs().max())
        res["after_merged"] = task.evaluate(merged, tok, test)
        out = Path(a.out_dir) / "merged"; merged.save_pretrained(out); tok.save_pretrained(out)
        res["merged_model_dir"] = str(out)
    if res["loss_log"]:
        res["loss_first_last"] = [res["loss_log"][0], res["loss_log"][-1]]
    env.save_json(res, a.results or HERE / "results" / f"sft_{a.dataset}.json")
    print({k: v for k, v in res.items() if k not in ("environment", "loss_log")})


if __name__ == "__main__":
    main()
