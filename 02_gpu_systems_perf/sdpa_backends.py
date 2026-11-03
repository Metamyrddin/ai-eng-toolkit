"""This file tests the different engines that PyTorch can use for attention and records which of
them work on this computer. It is part of area 2.

More technical notes:
PyTorch SDPA backend comparison: which attention kernel does F.scaled_dot_product_attention use?

torch.nn.attention.sdpa_kernel(...) forces a backend: MATH (reference), FLASH_ATTENTION, EFFICIENT_ATTENTION
(memory-efficient, xFormers-style), CUDNN_ATTENTION. Which ones exist depends on device + dtype + shapes; an
unavailable backend raises RuntimeError, which we record rather than hide.
Docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
      https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html
CPU (tested here): MATH and FLASH_ATTENTION (a CPU flash kernel exists in recent PyTorch). GPU: run with --device cuda
--dtype bfloat16 (UNTESTED here).
Run: python sdpa_backends.py [--device cpu --dtype float32 --seq 1024]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

from common import env  # noqa: E402


def main():
    """Try each attention backend of PyTorch on the same input and record which ones are
    available, how fast they are and how accurate.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "sdpa_backends.json"))
    a = ap.parse_args()
    env.seed_everything(0)
    dt = getattr(torch, a.dtype)
    q, k, v = (torch.randn(1, a.heads, a.seq, a.dim, device=a.device, dtype=dt) for _ in range(3))
    ref = None
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), is_causal=True)
    rows = {}
    for be in (SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION):
        try:
            with sdpa_kernel(be):
                F.scaled_dot_product_attention(q, k, v, is_causal=True)
                ts = []
                for _ in range(10):
                    if a.device == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    if a.device == "cuda":
                        torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
            rows[be.name] = {"available": True, "ms_median": round(1000 * sorted(ts)[len(ts) // 2], 3),
                             "max_abs_err_vs_fp64_math": float((o.double() - ref).abs().max())}
        except Exception as e:
            rows[be.name] = {"available": False, "reason": f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"}
        print(be.name, rows[be.name], flush=True)
    env.save_json({"environment": env.environment_info(), "device": a.device, "dtype": a.dtype, "seq": a.seq,
                   "heads": a.heads, "head_dim": a.dim, "backends": rows}, a.out)


if __name__ == "__main__":
    main()
