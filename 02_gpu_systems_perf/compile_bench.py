"""This file measures whether torch.compile makes a small transformer faster than normal PyTorch
(eager mode). It also checks that both give the same answers. It is part of area 2, which is
about making models run faster.

More technical notes:
torch.compile benchmark: eager vs compiled on a small transformer block (forward+backward).

torch.compile (docs: https://docs.pytorch.org/docs/stable/generated/torch.compile.html) traces Python code
with TorchDynamo and generates fused kernels with TorchInductor. On CPU, Inductor generates C++ and needs a C++
compiler (g++/clang); on GPU it generates Triton kernels. We report compile time (first call) separately from
steady-state step time, and verify numerical agreement with eager.

Run: python compile_bench.py [--backend inductor|aot_eager|eager] [--device cpu]
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
from torch import nn  # noqa: E402

from common import env  # noqa: E402


class Block(nn.Module):
    """Pre-LN transformer block with GELU MLP - plenty of elementwise ops for the compiler to fuse."""
    def __init__(self, d=256, h=4, ff=1024):
        """Create the layers of one small transformer block: two normalisation layers, the
        attention projections and the feed-forward layers.
        """
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.h = h

    def forward(self, x):
        """Run the block on a batch of vectors: attention first, then the feed-forward part,
        each added back to its input.
        """
        B, N, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return x + self.fc2(nn.functional.gelu(self.fc1(self.ln2(x))))


class Net(nn.Module):
    """A small stack of transformer blocks that we use as the test model for torch.compile.
    """
    def __init__(self, layers=4):
        """Build the stack of blocks."""
        super().__init__()
        self.blocks = nn.Sequential(*[Block() for _ in range(layers)])

    def forward(self, x):
        """Pass the input through all blocks in order."""
        return self.blocks(x)


def time_steps(model, x, n=10):
    """Run the forward and backward pass several times and return how long each run took."""
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(x).sum().backward()
        ts.append(time.perf_counter() - t0)
    return ts


def main():
    """Time the model without and with torch.compile, check that the answers agree, and save
    the results.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="inductor")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (also export OMP_NUM_THREADS before launch)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "compile.json"))
    a = ap.parse_args()
    env.seed_everything(0)
    if a.threads:
        torch.set_num_threads(a.threads)
    model = Net().to(a.device)
    x = torch.randn(a.batch, a.seq, 256, device=a.device)
    eager_out = model(x).detach()
    time_steps(model, x, 2)
    t_eager = time_steps(model, x, a.steps)
    res = {"environment": env.environment_info(), "backend": a.backend, "device": a.device,
           "torch_threads": torch.get_num_threads(), "OMP_NUM_THREADS": env.os.environ.get("OMP_NUM_THREADS"),
           "CXX": env.os.environ.get("CXX"),
           "eager_ms_median": round(1000 * statistics.median(t_eager), 2)}
    try:
        cm = torch.compile(model, backend=a.backend)
        t0 = time.perf_counter()
        out = cm(x).detach()
        res["first_call_compile_s"] = round(time.perf_counter() - t0, 2)
        res["max_abs_diff_vs_eager"] = float((out - eager_out).abs().max())
        cm(x).sum().backward()  # compile backward graph too
        t_c = time_steps(cm, x, a.steps)
        res["compiled_ms_median"] = round(1000 * statistics.median(t_c), 2)
        res["speedup"] = round(res["eager_ms_median"] / res["compiled_ms_median"], 2)
        res["status"] = "ok"
    except Exception as e:  # report real failures (e.g. missing C++ compiler) instead of hiding them
        res["status"] = "failed"
        res["error"] = f"{type(e).__name__}: {str(e)[:600]}"
    print(res)
    env.save_json(res, a.out)


if __name__ == "__main__":
    main()
