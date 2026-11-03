"""This file computes attention in three ways: the plain way, a tiled way that follows the
FlashAttention idea of never building the full score table, and PyTorch's built-in version. It
checks that the answers match and compares memory use. It is part of area 2.

More technical notes:
Attention three ways: naive, tiled "online softmax" (the FlashAttention algorithm), and PyTorch SDPA.

FlashAttention (Dao et al. 2022, arXiv:2205.14135) computes EXACT softmax attention without ever
materialising the full N x N score matrix: it processes K/V in blocks and keeps running statistics
(row max m and row sum l) so partial results can be rescaled ("online softmax", Milakov & Gimelshein 2018,
arXiv:1805.02867). The real speed-up comes from doing this inside one GPU kernel so blocks stay in fast SRAM;
the pure-PyTorch version here demonstrates the *math* and the O(N) extra-memory property, not GPU speed.

Run:  python flash_attention_from_scratch.py          (prints correctness errors + measured peak memory/time)
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common import env  # noqa: E402


def naive_attention(q, k, v, causal=False):
    """q,k,v: [B, H, N, D]. Materialises the [N, N] score matrix."""
    d = q.shape[-1]
    s = q @ k.transpose(-2, -1) / math.sqrt(d)
    if causal:
        n, m = s.shape[-2:]
        mask = torch.ones(n, m, dtype=torch.bool).tril(diagonal=m - n)
        s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v


def flash_attention_tiled(q, k, v, causal=False, block_q=64, block_k=64):
    """Exact attention via tiling + online softmax. Extra memory is O(block_q * block_k), not O(N^2)."""
    B, H, N, D = q.shape
    M = k.shape[-2]
    scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    for qs in range(0, N, block_q):
        qe = min(qs + block_q, N)
        qb = q[:, :, qs:qe] * scale
        m_i = torch.full((B, H, qe - qs, 1), float("-inf"), dtype=q.dtype)  # running row max
        l_i = torch.zeros((B, H, qe - qs, 1), dtype=q.dtype)                # running row sum of exp
        acc = torch.zeros((B, H, qe - qs, D), dtype=q.dtype)                # running (unnormalised) output
        for ks in range(0, M, block_k):
            if causal and ks > qe - 1 + (M - N):  # whole block is in the future -> skip
                break
            ke = min(ks + block_k, M)
            s = qb @ k[:, :, ks:ke].transpose(-2, -1)
            if causal:
                qi = torch.arange(qs, qe).unsqueeze(-1) + (M - N)
                kj = torch.arange(ks, ke).unsqueeze(0)
                s = s.masked_fill(kj > qi, float("-inf"))
            m_new = torch.maximum(m_i, s.amax(-1, keepdim=True))
            p = torch.exp(s - m_new)
            alpha = torch.exp(m_i - m_new)  # rescale factor for what we accumulated so far
            l_i = l_i * alpha + p.sum(-1, keepdim=True)
            acc = acc * alpha + p @ v[:, :, ks:ke]
            m_i = m_new
        out[:, :, qs:qe] = acc / l_i
    return out


def _child_peak_rss(impl: str, n: int, heads: int, dim: int) -> float:
    """Run one implementation in a fresh process; return (peak RSS during the call - RSS before) in MB.

    RSS is sampled every 0.5 ms by a background thread (psutil), because ru_maxrss is a process-lifetime high-water
    mark that can already be above the working set of a small computation (we hit exactly that artifact).
    """
    import threading

    import psutil
    q, k, v = (torch.randn(1, heads, n, dim) for _ in range(3))
    fns = {"naive": lambda: naive_attention(q, k, v, True),
           "tiled_online_softmax": lambda: flash_attention_tiled(q, k, v, True),
           "torch_sdpa": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)}
    proc = psutil.Process()
    base = proc.memory_info().rss
    peak = [base]
    stop = threading.Event()

    def sampler():
        """Keep checking the memory used by this process and remember the largest value seen.
        """
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.0005)

    t = threading.Thread(target=sampler, daemon=True); t.start()
    fns[impl]()
    stop.set(); t.join()
    peak[0] = max(peak[0], proc.memory_info().rss)
    return (peak[0] - base) / 2**20


def measure_peak_memory(impl: str, n: int, heads: int, dim: int) -> float:
    """Measure peak memory of one attention method by running it in a fresh process, so earlier
    runs cannot disturb the result.
    """
    import subprocess
    out = subprocess.check_output([sys.executable, __file__, "--child", impl, str(n), str(heads), str(dim)])
    return float(out.decode().strip().splitlines()[-1])


def main():
    """Check that the methods agree, time them for several sequence lengths, measure memory,
    and save the results.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "attention.json"))
    ap.add_argument("--mem-seq-len", type=int, default=4096)
    ap.add_argument("--child", nargs=4, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:  # subprocess mode used by measure_peak_memory()
        print(_child_peak_rss(a.child[0], int(a.child[1]), int(a.child[2]), int(a.child[3])))
        return
    env.seed_everything(0)
    rows = []
    for N in a.seq_lens:
        q, k, v = (torch.randn(1, a.heads, N, a.dim, dtype=torch.float32) for _ in range(3))
        row = {"seq_len": N}
        ref = naive_attention(q.double(), k.double(), v.double(), causal=True).float()
        for name, fn in [("naive", lambda: naive_attention(q, k, v, True)),
                         ("tiled_online_softmax", lambda: flash_attention_tiled(q, k, v, True)),
                         ("torch_sdpa", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))]:
            fn()
            ts = []
            for _ in range(5):
                t0 = time.perf_counter(); o = fn(); ts.append(time.perf_counter() - t0)
            row[name] = {"ms_median": round(1000 * sorted(ts)[len(ts) // 2], 2),
                         "max_abs_err_vs_fp64_naive": float((o - ref).abs().max())}
        row["score_matrix_mb_fp32_naive"] = round(a.heads * N * N * 4 / 2**20, 1)
        row["score_tile_mb_fp32_tiled"] = round(a.heads * 64 * 64 * 4 / 2**20, 3)
        rows.append(row); print(row, flush=True)
    mem = {impl: round(measure_peak_memory(impl, a.mem_seq_len, a.heads, a.dim), 1)
           for impl in ("naive", "tiled_online_softmax", "torch_sdpa")}
    print("peak extra RSS (MB) at seq_len", a.mem_seq_len, mem)
    env.save_json({"environment": env.environment_info(), "heads": a.heads, "head_dim": a.dim, "causal": True,
                   "rows": rows, "peak_extra_rss_mb": {"seq_len": a.mem_seq_len, **mem},
                   "memory_method": "fresh subprocess per implementation; peak RSS (psutil, sampled every 0.5 ms) minus RSS before the call; includes output tensor"}, a.out)


if __name__ == "__main__":
    main()
