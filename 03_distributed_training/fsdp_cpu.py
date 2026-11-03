"""This file trains the tiny model with FSDP2, where each process stores only a part (shard) of the
weights. It checks that sharding works and that the result still matches one-process training.
It is part of area 3.

More technical notes:
FSDP2 (`fully_shard`) on CPU with gloo, 2 processes: sharding + exactness check vs a single process.

FSDP shards parameters, gradients and optimizer state across ranks. Before a layer runs, its parameters are
all-gathered; after backward, gradients are reduce-scattered so each rank only keeps its shard. This is the
PyTorch implementation of the ZeRO-3 idea (Rajbhandari et al. 2019, arXiv:1910.02054).
Docs: https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
Verified here: (1) each rank stores ~1/world of the parameters, (2) training matches single-process SGD.

Run: torchrun --nproc_per_node=2 fsdp_cpu.py        or      python launch_spawn.py fsdp_cpu.py --world 2
"""
import argparse
import time

import dist_common as dc
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from common import env


def main():
    """Train the tiny model with fully sharded data parallelism (FSDP2), check that each
    process stores only half of the weights, and compare with one-process training.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--global-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.1)
    a = ap.parse_args()
    rank, world = dc.init_dist("gloo")
    torch.set_num_threads(max(1, torch.get_num_threads() // world))
    batches = dc.make_batches(a.steps, a.global_batch)

    ref = dc.make_model()
    opt_ref = torch.optim.SGD(ref.parameters(), lr=a.lr)
    for x, y in batches:
        opt_ref.zero_grad(); dc.lm_loss(ref, x, y).backward(); opt_ref.step()

    model = dc.make_model()
    total_params = sum(p.numel() for p in model.parameters())
    mesh = init_device_mesh("cpu", (world,))
    for layer in model.enc.layers:  # shard each transformer block as its own FSDP unit
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)   # root unit: embeddings + head
    local_params = sum(p.to_local().numel() for p in model.parameters())
    opt = torch.optim.SGD(model.parameters(), lr=a.lr)
    losses, t0 = [], time.perf_counter()
    for x, y in batches:
        opt.zero_grad(); loss = dc.lm_loss(model, x[rank::world], y[rank::world]); loss.backward(); opt.step()
        l = loss.detach().clone(); dist.all_reduce(l); losses.append((l / world).item())
    elapsed = time.perf_counter() - t0
    max_diff = max((p.full_tensor() - q.detach()).abs().max().item() for p, q in zip(model.parameters(), ref.parameters()))
    res = {"world_size": world, "total_params": total_params, "params_stored_on_this_rank": local_params,
           "fraction_stored_per_rank": round(local_params / total_params, 3),
           "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
           "max_abs_param_diff_fsdp_vs_single_process": max_diff, "seconds": round(elapsed, 2),
           "environment": env.environment_info()}
    if rank == 0:
        print(res)
        env.save_json(res, __import__("pathlib").Path(__file__).parent / "results" / "fsdp.json")
    dist.barrier(); dist.destroy_process_group()
    assert max_diff < 1e-5, "FSDP result differs from single-process reference"


if __name__ == "__main__":
    main()
