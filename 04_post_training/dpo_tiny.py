"""This file trains a model with DPO (Direct Preference Optimization), where the model learns from
pairs of a better and a worse answer. It starts from the model made by sft_lora.py. It is part
of area 4.

More technical notes:
Tiny DPO run (peft LoRA + trl DPOTrainer) on top of the SFT model from sft_lora.py.

DPO (Rafailov et al. 2023, arXiv:2305.18290) optimises the policy directly on preference pairs (x, y_w, y_l) with
        L = -log sigmoid( beta * [ log pi(y_w|x)/pi_ref(y_w|x)  -  log pi(y_l|x)/pi_ref(y_l|x) ] )
so no separate reward model or RL loop is needed; the reference policy pi_ref is the frozen starting model.
With LoRA, trl uses the base weights with the adapter DISABLED as pi_ref - therefore we first merge the SFT adapter
(sft_lora.py already saves outputs/sft/merged), so that pi_ref = the SFT model.
Docs: https://huggingface.co/docs/trl/dpo_trainer
Run: python dpo_tiny.py [--n-pairs 500]   |   real-text smoke test: python dpo_tiny.py --dataset ultrafeedback --n-pairs 64
Preference pairs here are synthetic (correct vs off-by-k answer); ultrafeedback is a *smoke test only* (no quality eval).
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
from trl import DPOConfig, DPOTrainer  # noqa: E402

from common import env  # noqa: E402

HERE = Path(__file__).parent


def main():
    """Fine-tune the model with DPO on preference pairs, report the accuracy before and after
    (for the arithmetic task) and save the training log.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(HERE / "outputs" / "sft" / "merged"))
    ap.add_argument("--dataset", default="arith", choices=["arith", "ultrafeedback"])
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    env.seed_everything(a.seed)
    default_base = str(HERE / "outputs" / "sft" / "merged")
    base = env.SMOL_135M if (a.base == default_base and not Path(a.base).exists()) else a.base  # hub ids are allowed
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    res = {"environment": env.environment_info(), "base_model": base, "dataset": a.dataset, "beta": a.beta,
           "lr": a.lr, "n_pairs": a.n_pairs, "seed": a.seed}
    if a.dataset == "arith":
        test = [json.loads(l) for l in open(HERE / "data" / "test_problems.jsonl")]
        res["before"] = task.evaluate(model, tok, test); print("BEFORE", res["before"])
        pairs = [json.loads(l) for l in open(HERE / "data" / "pairs_template.jsonl")][-a.n_pairs:]  # NOTE: same prompts as the SFT set; the 200 test problems are disjoint from both
        train = Dataset.from_list(pairs); max_len = 64
    else:
        train = load_dataset("trl-lib/ultrafeedback_binarized", split=f"train[:{a.n_pairs}]"); max_len = 256
    cfg = DPOConfig(output_dir=str(HERE / "outputs" / "dpo"), per_device_train_batch_size=a.batch,
                    num_train_epochs=a.epochs, learning_rate=a.lr, beta=a.beta, logging_steps=5, max_length=max_len,
                    use_cpu=True, report_to="none", save_strategy="no", seed=a.seed, bf16=False,
                    lr_scheduler_type="cosine", warmup_steps=3)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=train, processing_class=tok, peft_config=lora)
    t0 = time.perf_counter(); trainer.train(); res["train_seconds"] = round(time.perf_counter() - t0, 1)
    logs = [h for h in trainer.state.log_history if "loss" in h]
    keep = ["step", "loss", "rewards/accuracies", "rewards/margins", "rewards/chosen", "rewards/rejected"]
    res["log"] = [{k: (round(h[k], 4) if isinstance(h[k], float) else h[k]) for k in keep if k in h} for h in logs]
    if a.dataset == "arith":
        res["after"] = task.evaluate(trainer.model, tok, test); print("AFTER", res["after"])
    env.save_json(res, HERE / "results" / f"dpo_{a.dataset}.json")
    print({"first_log": res["log"][0], "last_log": res["log"][-1], **{k: res[k] for k in ("before", "after") if k in res}})


if __name__ == "__main__":
    main()
