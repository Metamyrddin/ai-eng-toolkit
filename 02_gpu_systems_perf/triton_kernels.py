"""This file contains three small GPU programs (kernels) written in Triton: adding vectors, softmax
and matrix multiplication. They run on the CPU through Triton's interpreter mode, so no graphics
card is needed. It is part of area 2 and the same code would run on a real GPU.

More technical notes:
Triton kernels that run on CPU through Triton's INTERPRETER mode (TRITON_INTERPRET=1).

Interpreter mode executes the kernel body in Python/NumPy, one program at a time, so you can debug with
print() and breakpoints, with no GPU. It is SLOW and meant for correctness, not performance.
(Docs: https://triton-lang.org/main/programming-guide/chapter-3/debugging.html ,
 tutorials https://triton-lang.org/main/getting-started/tutorials/index.html)
The same kernels run unchanged on an NVIDIA GPU if you drop TRITON_INTERPRET and pass CUDA tensors (UNTESTED here).

Kernels: vector add, fused row softmax, tiled matmul (tl.dot with fp32 accumulation).
Run: python triton_kernels.py        (self-checks against PyTorch and prints max errors)
"""
import os
import sys
from pathlib import Path

if not os.environ.get("TRITON_FORCE_GPU"):
    os.environ["TRITON_INTERPRET"] = "1"  # must be set BEFORE importing triton
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from common import env  # noqa: E402

DEVICE = "cuda" if os.environ.get("TRITON_FORCE_GPU") else "cpu"


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # Triton kernel that adds two vectors. Each program instance handles one block of BLOCK
    # numbers, and the mask protects the end of the vector.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add(x, y, block=128):
    """Python wrapper that launches the add kernel on enough blocks to cover the whole vector.
    """
    out = torch.empty_like(x)
    n = x.numel()
    add_kernel[(triton.cdiv(n, block),)](x, y, out, n, BLOCK=block)
    return out


@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, stride, BLOCK: tl.constexpr):
    # Triton kernel that computes softmax for one row of a matrix. Each program instance handles
    # one row.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)           # subtract row max for numerical stability
    num = tl.exp(x)
    tl.store(out_ptr + row * stride + cols, num / tl.sum(num, axis=0), mask=mask)


def softmax(x):
    """Python wrapper that launches the softmax kernel with one program instance per row."""
    rows, cols = x.shape
    out = torch.empty_like(x)
    softmax_kernel[(rows,)](x, out, cols, x.stride(0), BLOCK=triton.next_power_of_2(cols))
    return out


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Triton kernel that multiplies two matrices tile by tile. Each program instance computes one
    # BM by BN tile of the result.
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * sam + (k0 + rk)[None, :] * sak,
                    mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k0 + rk)[:, None] * sbk + rn[None, :] * sbn,
                    mask=((k0 + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul(a, b, bm=16, bn=16, bk=16):
    """Python wrapper that launches the matrix multiplication kernel on a grid of tiles."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)
    matmul_kernel[(triton.cdiv(M, bm), triton.cdiv(N, bn))](a, b, c, M, N, K, a.stride(0), a.stride(1),
                                                            b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                                                            BM=bm, BN=bn, BK=bk)
    return c


def main():
    """Run the three kernels on awkward sizes (to test the masks), compare with PyTorch and
    save the largest errors.
    """
    torch.manual_seed(0)
    res = {"environment": env.environment_info(), "mode": "TRITON_INTERPRET=1 (CPU)" if DEVICE == "cpu" else "GPU"}
    x, y = torch.randn(1000, device=DEVICE), torch.randn(1000, device=DEVICE)  # 1000 not a multiple of block -> tests mask
    res["add_max_abs_err"] = float((add(x, y) - (x + y)).abs().max())
    s = torch.randn(7, 50, device=DEVICE) * 5
    res["softmax_max_abs_err"] = float((softmax(s) - torch.softmax(s, -1)).abs().max())
    a, b = torch.randn(37, 53, device=DEVICE), torch.randn(53, 29, device=DEVICE)  # odd sizes test masks
    res["matmul_max_abs_err"] = float((matmul(a, b) - a @ b).abs().max())
    res["matmul_max_rel_err"] = float(((matmul(a, b) - a @ b).abs() / (a @ b).abs().clamp_min(1e-3)).max())
    print(res)
    env.save_json(res, Path(__file__).parent / "results" / "triton_interpreter.json")
    assert res["add_max_abs_err"] < 1e-6 and res["softmax_max_abs_err"] < 1e-6 and res["matmul_max_abs_err"] < 1e-3


if __name__ == "__main__":
    main()
