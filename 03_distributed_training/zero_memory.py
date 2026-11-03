"""This file calculates how much GPU memory a model needs under normal data parallelism and under
the ZeRO stages. It uses only arithmetic, so it runs instantly. It is part of area 3 and its
numbers are checked against a published paper in the tests.

More technical notes:
Back-of-envelope GPU memory for model states under DDP / ZeRO-1/2/3 (pure arithmetic, unit-tested).

Model of Rajbhandari et al. 2019 (ZeRO, arXiv:1910.02054, Section 3 / Figure 1) for mixed-precision Adam:
   fp16/bf16 params 2*Psi bytes, fp16 grads 2*Psi bytes, optimizer states K*Psi bytes with K = 12
   (fp32 master params 4 + momentum 4 + variance 4).
   baseline DDP          : (2 + 2 + K) * Psi
   ZeRO-1 (shard opt)    : 2*Psi + 2*Psi + K*Psi/N
   ZeRO-2 (+shard grads) : 2*Psi + (2 + K)*Psi/N
   ZeRO-3 (+shard params): (2 + 2 + K)*Psi/N
Activations, buffers, fragmentation and CUDA context are NOT included - real usage is higher.
Run: python zero_memory.py --params-b 7 --gpus 8
"""
import argparse


def zero_model_state_gb(psi: float, n_gpus: int, stage: int, k: int = 12) -> float:
    """Per-GPU model-state memory in GB (1 GB = 1e9 bytes, matching the ZeRO paper's convention). stage 0 = plain DDP."""
    if stage not in (0, 1, 2, 3):
        raise ValueError("stage must be 0..3")
    if n_gpus < 1:
        raise ValueError("n_gpus >= 1")
    p, g, o = 2.0, 2.0, float(k)
    if stage >= 1:
        o /= n_gpus
    if stage >= 2:
        g /= n_gpus
    if stage >= 3:
        p /= n_gpus
    return (p + g + o) * psi / 1e9


def main():
    """Print how much memory per GPU the model states need under plain data parallelism and
    under ZeRO stages 1, 2 and 3.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--params-b", type=float, default=7.5, help="parameters in billions")
    ap.add_argument("--gpus", type=int, default=64)
    a = ap.parse_args()
    psi = a.params_b * 1e9
    print(f"Psi = {a.params_b}B params, N = {a.gpus} GPUs (model states only)")
    for st, name in enumerate(["DDP baseline", "ZeRO-1", "ZeRO-2", "ZeRO-3"]):
        print(f"  {name:13s}: {zero_model_state_gb(psi, a.gpus, st):8.2f} GB / GPU")


if __name__ == "__main__":
    main()
