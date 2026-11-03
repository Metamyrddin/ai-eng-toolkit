"""This file demonstrates the basic communication steps that processes use when they train
together, such as all-reduce and all-gather. It checks that each one gives the right answer and
times them. It is part of area 3.

More technical notes:
The collective operations behind DDP/FSDP/ZeRO, with gloo on CPU: verify semantics + time them.

all_reduce (sum)       every rank ends with the elementwise sum            (DDP gradient averaging)
all_gather             every rank ends with the concatenation of all shards (FSDP/ZeRO-3 param gather)
reduce_scatter         sum, then each rank keeps ONE shard of the result    (FSDP/ZeRO gradient sharding)
broadcast              copy rank-0 tensor to everyone                       (initial weights)
Timings are localhost-gloo numbers on this box and say nothing about NCCL/InfiniBand.
Docs: https://docs.pytorch.org/docs/stable/distributed.html
Run: torchrun --nproc_per_node=2 collectives_demo.py
"""
import statistics
import time

import dist_common as dc
import torch
import torch.distributed as dist

from common import env


def timeit(fn, n=20):
    """Run a function several times (after one warm-up run) and return the median time in
    milliseconds.
    """
    fn(); dist.barrier()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return 1000 * statistics.median(ts)


def main():
    """Check that each collective operation gives the correct answer on every process, time
    all-reduce for several sizes, and save the results.
    """
    rank, world = dc.init_dist("gloo")
    r = torch.tensor([float(rank + 1)] * 4)
    x = r.clone(); dist.all_reduce(x)
    assert torch.equal(x, torch.full((4,), float(sum(range(1, world + 1))))), "all_reduce"
    g = [torch.zeros(4) for _ in range(world)]; dist.all_gather(g, r)
    assert all(torch.equal(g[i], torch.full((4,), float(i + 1))) for i in range(world)), "all_gather"
    inp = [torch.arange(4.0) + 10 * rank + 100 * i for i in range(world)]
    out = torch.zeros(4)
    try:
        dist.reduce_scatter(out, inp)
        expect = sum(torch.arange(4.0) + 10 * rk + 100 * rank for rk in range(world))
        assert torch.equal(out, expect), "reduce_scatter"
        rs_ok = True
    except RuntimeError as e:  # gloo support for reduce_scatter has varied by version; report, don't hide
        rs_ok = f"unsupported: {str(e)[:120]}"
    b = torch.tensor([42.0]) if rank == 0 else torch.zeros(1); dist.broadcast(b, src=0)
    assert b.item() == 42.0, "broadcast"
    rows = []
    for mb in (0.1, 1, 8, 32):
        t = torch.randn(int(mb * 2**20 / 4))
        rows.append({"size_mb": mb, "all_reduce_ms_median": round(timeit(lambda: dist.all_reduce(t)), 3)})
    if rank == 0:
        res = {"semantics_verified": {"all_reduce": True, "all_gather": True, "broadcast": True, "reduce_scatter": rs_ok},
               "world_size": world, "timings": rows, "environment": env.environment_info()}
        print(res)
        env.save_json(res, __import__("pathlib").Path(__file__).parent / "results" / "collectives.json")
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
