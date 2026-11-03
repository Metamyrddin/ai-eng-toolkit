"""This file measures how fast a Hugging Face model writes text when we give it several prompts at
the same time. It reports the time to the first token and the number of tokens per second. It is
part of area 1 and gives us a plain baseline to compare other methods against.

More technical notes:
Latency / TTFT / throughput benchmark for HF transformers generate() with static batching.

Run:  python bench_hf.py --model HuggingFaceTB/SmolLM2-135M-Instruct --batch-sizes 1 2 4 8 --repeats 3

Definitions (all measured with time.perf_counter, CPU fp32 unless --dtype/--device given):
  TTFT            : wall time of generate(max_new_tokens=1)  (prefill + first token), per request batch
  decode tok/s    : (new_tokens-1) * batch / (time(new_tokens) - time(1 token))   -> aggregate throughput
  e2e latency     : wall time of generate(max_new_tokens=N)
Greedy decoding, fixed-length outputs (we suppress EOS with min_new_tokens=N) so that all batch sizes do equal work.
On a GPU run the same script with --device cuda --dtype bfloat16 (untested here: no GPU on this box).
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from common import env  # noqa: E402

PROMPTS = [
    "Explain what a KV cache is.", "Write a short poem about the sea.", "What is the capital of France and why is it famous?",
    "List three uses of a hash table.", "Describe gradient descent in two sentences.", "Why is the sky blue?",
    "Give me a tip for learning to program.", "What does a compiler do?",
]


def sync(device):
    """Wait until the graphics card has finished its queued work, so that the stopwatch is
    honest. Does nothing on the CPU.
    """
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def timed_generate(model, tok, prompts, n_new, device):
    """Generate the requested number of new tokens for all prompts at once and return the time
    it took and the padded prompt length.
    """
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    sync(device)
    t0 = time.perf_counter()
    model.generate(**enc, max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False,
                   pad_token_id=tok.pad_token_id)
    sync(device)
    return time.perf_counter() - t0, enc.input_ids.shape[1]


def main():
    """Measure speed for each batch size, print a table and save the numbers as a JSON file.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=env.SMOL_135M_INSTRUCT)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "hf_batching.json"))
    a = ap.parse_args()
    env.seed_everything(0)
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    rows = []
    timed_generate(model, tok, PROMPTS[:1], 4, a.device)  # warm-up
    for bs in a.batch_sizes:
        prompts = (PROMPTS * ((bs + len(PROMPTS) - 1) // len(PROMPTS)))[:bs]
        ttft, e2e = [], []
        for _ in range(a.repeats):
            t1, plen = timed_generate(model, tok, prompts, 1, a.device)
            tn, _ = timed_generate(model, tok, prompts, a.new_tokens, a.device)
            ttft.append(t1); e2e.append(tn)
        dec = [(a.new_tokens - 1) * bs / max(e - t, 1e-9) for e, t in zip(e2e, ttft)]
        row = {"batch": bs, "padded_prompt_len": plen,
               "ttft_ms_median": round(1000 * statistics.median(ttft), 1),
               "e2e_s_median": round(statistics.median(e2e), 3),
               "decode_tok_s_aggregate_median": round(statistics.median(dec), 1),
               "decode_tok_s_per_request": round(statistics.median(dec) / bs, 1),
               "repeats": a.repeats}
        rows.append(row)
        print(row, flush=True)
    env.save_json({"environment": env.environment_info(), "model": a.model, "device": a.device,
                   "dtype": a.dtype, "new_tokens": a.new_tokens, "rows": rows}, a.out)


if __name__ == "__main__":
    main()
