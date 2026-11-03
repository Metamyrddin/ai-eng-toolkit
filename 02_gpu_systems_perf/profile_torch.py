"""This file uses the PyTorch profiler, a tool that records where time is spent, on one training
step of a small language model. It prints the slowest operations and saves a trace file. It is
part of area 2 and teaches how to find slow spots.

More technical notes:
PyTorch profiler walkthrough on a real tiny LLM (SmolLM2-135M): forward + backward of one training step.

Run: python profile_torch.py [--batch 2 --seq 128 --device cpu]
Outputs: results/profile_top_ops.json, results/profile_trace.json (open in https://ui.perfetto.dev or chrome://tracing).
On a GPU the same script (--device cuda) also records CUDA kernels (UNTESTED here: no GPU).
Docs: https://docs.pytorch.org/docs/stable/profiler.html
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from common import env  # noqa: E402


def main():
    """Profile a few training steps of the model, print the slowest operations and save the
    table and a Chrome trace.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=env.SMOL_135M)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()
    env.seed_everything(0)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(a.device).train()
    ids = torch.randint(0, model.config.vocab_size, (a.batch, a.seq), device=a.device)

    def step():
        """Do one training step: a forward pass, then a backward pass, then clear the
        gradients.
        """
        with record_function("forward"):
            loss = model(input_ids=ids, labels=ids).loss
        with record_function("backward"):
            loss.backward()
        model.zero_grad(set_to_none=True)

    step()  # warm-up (allocator, lazy init)
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if a.device == "cuda" else [])
    with profile(activities=acts, record_shapes=True, profile_memory=True) as prof:
        for _ in range(3):
            step()
    sort_key = "self_cuda_time_total" if a.device == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=a.rows, max_name_column_width=40))
    ka = prof.key_averages()
    rows = sorted(ka, key=lambda e: e.self_cpu_time_total, reverse=True)[: a.rows]
    total_self = sum(e.self_cpu_time_total for e in ka)
    top = [{"op": e.key, "calls": e.count, "self_cpu_ms": round(e.self_cpu_time_total / 1000, 2),
            "self_cpu_pct": round(100 * e.self_cpu_time_total / total_self, 1)} for e in rows]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(out / "profile_trace.json"))
    env.save_json({"environment": env.environment_info(), "model": a.model, "batch": a.batch, "seq": a.seq,
                   "steps_profiled": 3, "top_ops_by_self_cpu": top}, out / "profile_top_ops.json")


if __name__ == "__main__":
    main()
