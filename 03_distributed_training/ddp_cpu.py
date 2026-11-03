"""This file trains a tiny model with DistributedDataParallel, which means two processes each
handle half of the data and share their gradients. It then proves that the result equals
training on one process. It is part of area 3.

More technical notes:
DistributedDataParallel on CPU with the gloo backend (2 processes) + an EXACTNESS check.

What DDP does: every rank holds a full model copy, sees a different slice of the batch, and gradients are
averaged with all-reduce during backward, so all replicas take identical optimizer steps.
Docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
Claim we verify: with the same global batch, DDP(2 ranks) after N SGD steps == single process on the whole batch.

Run:  torchrun --nproc_per_node=2 ddp_cpu.py        or      python launch_spawn.py ddp_cpu.py --world 2
"""
import argparse
import copy
import time

import dist_common as dc
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from common import env


def main():
    """Train the tiny model with DistributedDataParallel on several processes, compare with
    training on one process, and stop with an error if they differ.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--global-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", default="gloo", choices=["gloo", "nccl"], help="nccl = GPU path (UNTESTED here)")
    a = ap.parse_args()
    rank, world = dc.init_dist(a.backend)
    if a.backend == "nccl":  # GPU path: one GPU per rank (UNTESTED: no GPU on the build machine)
        import os
        dev = torch.device("cuda", int(os.environ["LOCAL_RANK"])); torch.cuda.set_device(dev)
    else:
        dev = torch.device("cpu"); torch.set_num_threads(max(1, torch.get_num_threads() // world))
    batches = dc.make_batches(a.steps, a.global_batch)

    # --- single-process reference (computed on every rank; cheap) ---
    ref = dc.make_model()
    opt_ref = torch.optim.SGD(ref.parameters(), lr=a.lr)
    ref_losses = []
    for x, y in batches:
        opt_ref.zero_grad(); loss = dc.lm_loss(ref, x, y); loss.backward(); opt_ref.step(); ref_losses.append(loss.item())

    # --- DDP ---
    model = DDP(dc.make_model().to(dev), device_ids=[dev.index] if dev.type == "cuda" else None)  # identical init on all ranks (same seed); DDP also broadcasts rank-0 weights
    opt = torch.optim.SGD(model.parameters(), lr=a.lr)
    ddp_losses, t0 = [], time.perf_counter()
    for x, y in batches:
        xs, ys = x[rank::world].to(dev), y[rank::world].to(dev)  # this rank's shard of the global batch
        opt.zero_grad(); loss = dc.lm_loss(model, xs, ys); loss.backward(); opt.step()
        l = loss.detach().clone(); dist.all_reduce(l); ddp_losses.append((l / world).item())  # mean of shard means
    elapsed = time.perf_counter() - t0

    max_diff = max((p.detach().cpu() - q.detach()).abs().max().item() for p, q in zip(model.module.parameters(), ref.parameters()))
    # replicas identical across ranks?
    flat = torch.cat([p.detach().flatten() for p in model.module.parameters()]).to(dev)
    gathered = [torch.zeros_like(flat) for _ in range(world)]
    dist.all_gather(gathered, flat)
    rank_diff = max((gathered[0] - g).abs().max().item() for g in gathered[1:])
    res = {"world_size": world, "backend": a.backend, "steps": a.steps, "global_batch": a.global_batch,
           "loss_first": round(ddp_losses[0], 4), "loss_last": round(ddp_losses[-1], 4),
           "ref_loss_last": round(ref_losses[-1], 4),
           "max_abs_param_diff_ddp_vs_single_process": max_diff, "max_abs_param_diff_between_ranks": rank_diff,
           "seconds": round(elapsed, 2), "environment": env.environment_info()}
    if rank == 0:
        print(res)
        env.save_json(res, a.out or (__import__("pathlib").Path(__file__).parent / "results" / "ddp.json"))
    dist.barrier(); dist.destroy_process_group()
    assert max_diff < 1e-5 and rank_diff == 0.0, "DDP result differs from single-process reference"


if __name__ == "__main__":
    main()
