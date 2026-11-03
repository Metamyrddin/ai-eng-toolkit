"""This file shows how number formats such as float16 and bfloat16 behave: how big and how precise
they are, why small gradients can vanish, and how loss scaling fixes this. It is part of area 3.

More technical notes:
Mixed precision on CPU: number formats, fp16 gradient underflow + loss scaling, bf16 autocast accuracy/speed.

Facts shown are *measured/computed here* (no GPU needed):
 1. torch.finfo of fp32 / fp16 / bf16 (range and precision).
 2. fp16 gradients below ~6e-8 flush to zero; scaling the loss before backward (the idea behind GradScaler /
    loss scaling, Micikevicius et al. 2017, arXiv:1710.03740) rescues them; bf16 has fp32's range so needs no scaling.
 3. bf16 autocast on CPU: loss difference vs fp32 on TinyLM and step time (CPU speed may be better OR worse
    than fp32 depending on whether the CPU has native bf16 instructions - we report what we measure).
Docs: https://docs.pytorch.org/docs/stable/amp.html
Run: python mixed_precision_demo.py
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dist_common as dc  # noqa: E402
import torch  # noqa: E402

from common import env  # noqa: E402


def finfo_table():
    """Return the largest number, smallest normal number and precision of float32, float16 and
    bfloat16.
    """
    out = {}
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        fi = torch.finfo(dt)
        out[str(dt)] = {"max": fi.max, "smallest_normal": fi.tiny, "eps": fi.eps, "bits": fi.bits}
    return out


def underflow_demo():
    """A weight with a tiny true gradient: g = 1e-8 * x. Compute in fp16 with and without loss scaling."""
    res = {}
    for scale in (1.0, 1024.0, 32768.0):
        w = torch.tensor([1.0], dtype=torch.float16, requires_grad=True)
        x = torch.tensor([1e-3], dtype=torch.float16)
        loss = (w * x * torch.tensor(1e-5, dtype=torch.float16)).sum() * scale  # d loss/dw = 1e-8 * scale
        loss.backward()
        grad_unscaled = w.grad.float().item() / scale
        res[f"loss_scale_{int(scale)}"] = {"fp16_grad_after_unscale": grad_unscaled, "true_grad": 1e-8}
    w = torch.tensor([1.0], dtype=torch.bfloat16, requires_grad=True)
    (w * torch.tensor(1e-3, dtype=torch.bfloat16) * torch.tensor(1e-5, dtype=torch.bfloat16)).sum().backward()
    res["bf16_no_scaling"] = {"grad": w.grad.float().item(), "true_grad": 1e-8}
    return res


def autocast_demo(steps=10):
    """Compare the loss and the speed of one training step in float32 and with bfloat16
    automatic mixed precision on the CPU.
    """
    x, y = dc.make_batches(1, 32)[0]
    out = {}
    base = dc.make_model()
    with torch.no_grad():
        l32 = dc.lm_loss(base, x, y).item()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            lbf = dc.lm_loss(base, x, y).item()
    out["loss_fp32"] = l32; out["loss_bf16_autocast"] = lbf; out["abs_diff"] = abs(l32 - lbf)
    for name, ctx in [("fp32", None), ("bf16_autocast", torch.bfloat16)]:
        m = dc.make_model(); opt = torch.optim.SGD(m.parameters(), lr=0.1); ts = []
        for _ in range(steps + 2):
            t0 = time.perf_counter()
            opt.zero_grad()
            if ctx is None:
                loss = dc.lm_loss(m, x, y)
            else:
                with torch.autocast("cpu", dtype=ctx):
                    loss = dc.lm_loss(m, x, y)
            loss.backward(); opt.step(); ts.append(time.perf_counter() - t0)
        out[f"step_ms_{name}"] = round(1000 * statistics.median(ts[2:]), 2)
    return out


def main():
    """Run all the mixed precision demonstrations, print them and save the results as a JSON
    file.
    """
    res = {"environment": env.environment_info(), "finfo": finfo_table(), "fp16_underflow": underflow_demo(),
           "bf16_autocast_cpu": autocast_demo()}
    import json; print(json.dumps(res, indent=1)[:3000])
    env.save_json(res, Path(__file__).parent / "results" / "mixed_precision.json")


if __name__ == "__main__":
    main()
