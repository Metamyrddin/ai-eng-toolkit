"""This file shows what the KV cache is: a memory that stores earlier work so the model does not
repeat it for every new word. It proves that the output stays the same, measures how much faster
the cache makes generation, and checks a formula for how much memory the cache needs. It is part
of area 1.

More technical notes:
KV-cache demo: (1) correctness  (2) speed  (3) memory formula vs. measured bytes.

Run:  python kv_cache_demo.py [--model HuggingFaceTB/SmolLM2-135M] [--new-tokens 128]

What it proves (all numbers are measured when you run it):
  1. Greedy decoding WITH a KV cache produces exactly the same tokens as recomputing the
     whole prefix every step (the cache is an optimisation, not an approximation).
  2. Time per generated token stays ~flat with cache, but grows with sequence length without it.
  3. The textbook formula  2 * layers * kv_heads * head_dim * seq_len * bytes_per_elem
     matches the bytes actually held by the cache object.
"""
import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

from common import env  # noqa: E402
from common.hf_compat import cache_drop_last  # noqa: E402


def cache_nbytes(cache) -> int:
    """Total bytes of K and V tensors in a transformers cache (handles API variants)."""
    total = 0
    if hasattr(cache, "layers"):  # transformers >= 4.56 style
        for layer in cache.layers:
            for name in ("keys", "values"):
                t = getattr(layer, name, None)
                if t is not None and torch.is_tensor(t):
                    total += t.numel() * t.element_size()
    elif hasattr(cache, "key_cache"):
        for k, v in zip(cache.key_cache, cache.value_cache):
            total += k.numel() * k.element_size() + v.numel() * v.element_size()
    else:
        raise RuntimeError("Unknown cache layout; update cache_nbytes() for this transformers version")
    return total


def kv_formula_bytes(cfg, seq_len: int, batch: int = 1, bytes_per_elem: int = 4) -> int:
    """Calculate how many bytes the KV cache needs, using the formula from the model's
    configuration.
    """
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    return 2 * cfg.num_hidden_layers * kv_heads * head_dim * seq_len * batch * bytes_per_elem


@torch.no_grad()
def greedy_no_cache(model, ids, n_new):
    """Re-run the whole prefix each step. Returns (tokens, per-step seconds)."""
    times = []
    for _ in range(n_new):
        t0 = time.perf_counter()
        logits = model(ids, use_cache=False).logits[:, -1]
        nxt = logits.argmax(-1, keepdim=True)
        times.append(time.perf_counter() - t0)
        ids = torch.cat([ids, nxt], dim=1)
    return ids, times


@torch.no_grad()
def greedy_with_cache(model, ids, n_new):
    """Prefill once, then feed only the newest token each step."""
    times = []
    cache = DynamicCache()
    t0 = time.perf_counter()
    out = model(ids, past_key_values=cache, use_cache=True)
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    times.append(time.perf_counter() - t0)  # step 0 includes prefill (= TTFT)
    ids = torch.cat([ids, nxt], dim=1)
    for _ in range(n_new - 1):
        t0 = time.perf_counter()
        out = model(nxt, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        times.append(time.perf_counter() - t0)
        ids = torch.cat([ids, nxt], dim=1)
    return ids, times, cache


@torch.no_grad()
def step_cost_sweep(model, lengths=(32, 64, 128, 256, 512), repeats=7):
    """Cost of producing ONE next token at prefix length L: full recompute vs cached decode step.

    More robust than per-step timing inside one generation (less sensitive to noise):
    median over `repeats` runs, after one warm-up.
    """
    rows = []
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(0)
    for L in lengths:
        ids = torch.randint(0, vocab, (1, L), generator=g)
        cache = DynamicCache()
        model(ids[:, :-1], past_key_values=cache, use_cache=True)  # prefill L-1 tokens
        full, cached = [], []
        for r in range(repeats + 1):
            t0 = time.perf_counter(); model(ids, use_cache=False); tf = time.perf_counter() - t0
                        # drop the token we just appended so the cache length stays L-1 for every repeat
            t0 = time.perf_counter(); model(ids[:, -1:], past_key_values=cache, use_cache=True)
            tc = time.perf_counter() - t0
            cache_drop_last(cache, 1)
            if r > 0:  # r == 0 is warm-up
                full.append(tf); cached.append(tc)
        rows.append({"prefix_len": L, "ms_full_recompute_median": round(1000 * statistics.median(full), 2),
                     "ms_cached_step_median": round(1000 * statistics.median(cached), 2)})
        rows[-1]["speedup"] = round(rows[-1]["ms_full_recompute_median"] / rows[-1]["ms_cached_step_median"], 1)
    return rows


def main():
    """Run the three KV cache experiments (same output, speed against prefix length, memory
    formula) and save the results.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=env.SMOL_135M)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--prompt", default="The key idea behind a key-value cache in transformer inference is")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "kv_cache.json"))
    a = ap.parse_args()
    env.seed_everything(0)
    torch.set_num_threads(max(1, (os.cpu_count() or 2)))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    ids = tok(a.prompt, return_tensors="pt").input_ids

    ids_nc, t_nc = greedy_no_cache(model, ids, a.new_tokens)
    ids_c, t_c, cache = greedy_with_cache(model, ids, a.new_tokens)
    same = bool(torch.equal(ids_nc, ids_c))
    print(f"identical tokens with/without cache: {same}")

    # bucketed per-token latency (ms) to show growth
    def bucket(ts, k=4):
        """Split a list of timings into k equal groups and return the average of each group in
        milliseconds.
        """
        n = len(ts) // k
        return [round(1000 * statistics.mean(ts[i * n:(i + 1) * n]), 2) for i in range(k)]

    measured = cache_nbytes(cache)
    seq = ids_c.shape[1] - 1  # last generated token has not been fed back yet
    formula = kv_formula_bytes(model.config, seq)
    print(f"KV bytes measured={measured} formula={formula} (seq_len in cache={seq})")
    sweep = step_cost_sweep(model)
    for r in sweep:
        print(r)
    res = {
        "step_cost_sweep": sweep,
        "environment": env.environment_info(),
        "model": a.model, "prompt_tokens": ids.shape[1], "new_tokens": a.new_tokens,
        "tokens_identical": same,
        "total_seconds_no_cache": round(sum(t_nc), 3),
        "total_seconds_with_cache": round(sum(t_c), 3),
        "ms_per_token_quartiles_no_cache": bucket(t_nc),
        "ms_per_token_quartiles_with_cache": bucket(t_c[1:]),  # exclude prefill step
        "ttft_ms_with_cache": round(1000 * t_c[0], 2),
        "kv_bytes_measured": measured, "kv_bytes_formula": formula,
        "kv_formula_matches": measured == formula,
        "note": "single run, fp32, CPU; use --new-tokens larger for a clearer slope. Not a repeated benchmark.",
    }
    env.save_json(res, a.out)
    print(res)
    if not same:
        sys.exit(1)


if __name__ == "__main__":
    main()
