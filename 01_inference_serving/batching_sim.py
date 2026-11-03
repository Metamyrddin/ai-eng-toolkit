"""This file simulates two ways for a server to group user requests: static batching, where
everyone waits for the slowest request, and continuous batching, where finished requests leave
and new ones join at once. It uses made-up request times, so no real model is needed. It is part
of area 1 and shows why modern servers use continuous batching.

More technical notes:
Static vs continuous (iteration-level) batching: a deterministic scheduling SIMULATION.

This is NOT a measurement of a real server. It runs a discrete-time simulator where one decode step for a
batch costs `step_cost(batch)` time units (default: affine model `a + b*batch`, with a >> b like a
memory-bound decoder: loading weights dominates). The point is to expose the *scheduling* effect:
static batching waits for the longest request in a batch; continuous batching (Orca, OSDI'22; vLLM) refills
a freed slot every step.  Request lengths are drawn from a seeded distribution so results are reproducible.
You can plug in step costs you MEASURED (bench_hf.py gives e2e/decode numbers) via --a and --b.

Run: python batching_sim.py [--n 200 --max-batch 8 --seed 0]
"""
import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import env  # noqa: E402


def make_requests(n, seed, arrival_rate):
    """Create a made-up list of requests that arrive at random times and want a random number
    of output words.
    """
    rng = random.Random(seed)
    t, reqs = 0.0, []
    for i in range(n):
        t += rng.expovariate(arrival_rate)
        reqs.append({"id": i, "arrival": t, "out_len": max(1, int(rng.lognormvariate(3.0, 0.8)))})  # median ~20 tokens
    return reqs


def simulate_static(reqs, max_batch, cost):
    """Take up to max_batch queued requests, run them until the LONGEST finishes (finished ones idle)."""
    pending = sorted(reqs, key=lambda r: r["arrival"])
    now, idx, queue, lat, toks = 0.0, 0, [], [], 0
    while idx < len(pending) or queue:
        while idx < len(pending) and pending[idx]["arrival"] <= now:
            queue.append(pending[idx]); idx += 1
        if not queue:
            now = pending[idx]["arrival"]
            continue
        batch, queue = queue[:max_batch], queue[max_batch:]
        steps = max(r["out_len"] for r in batch)
        now += steps * cost(len(batch))
        for r in batch:
            lat.append(now - r["arrival"]); toks += r["out_len"]
    return lat, now, toks


def simulate_continuous(reqs, max_batch, cost):
    """Iteration-level scheduling: after EVERY decode step finished sequences leave and queued ones join."""
    pending = sorted(reqs, key=lambda r: r["arrival"])
    now, idx, queue, running, lat, toks = 0.0, 0, [], [], [], 0
    while idx < len(pending) or queue or running:
        while idx < len(pending) and pending[idx]["arrival"] <= now:
            queue.append(pending[idx]); idx += 1
        while queue and len(running) < max_batch:
            r = queue.pop(0)
            running.append([r, r["out_len"]])
        if not running:
            now = pending[idx]["arrival"]
            continue
        now += cost(len(running))
        for s in running:
            s[1] -= 1; toks += 1
            if s[1] == 0:
                lat.append(now - s[0]["arrival"])
        running = [s for s in running if s[1] > 0]
    return lat, now, toks


def main():
    """Run the simulation for both scheduling styles, print the results as a table and save
    them as a JSON file.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--a", type=float, default=15.0, help="fixed ms per decode step (weight loading)")
    ap.add_argument("--b", type=float, default=1.0, help="extra ms per sequence in the batch")
    ap.add_argument("--arrival-rate", type=float, default=0.004, help="requests per ms (0.004 = 4 req/s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "batching_sim.json"))
    args = ap.parse_args()
    reqs = make_requests(args.n, args.seed, args.arrival_rate)
    res = {"kind": "SIMULATION (not a real-server measurement)", "args": vars(args), "modes": {}}
    for mode in ("static", "continuous"):
        cost = lambda bs: args.a + args.b * bs  # noqa: E731
        sim = simulate_static if mode == "static" else simulate_continuous
        lat, makespan, toks = sim(reqs, args.max_batch, cost)
        res["modes"][mode] = {
            "mean_latency_ms": round(statistics.mean(lat), 1),
            "p95_latency_ms": round(sorted(lat)[int(0.95 * len(lat)) - 1], 1),
            "makespan_ms": round(makespan, 1), "useful_tokens": toks,
            "throughput_tok_per_s": round(1000 * toks / makespan, 1)}
        print(mode, res["modes"][mode])
    env.save_json(res, args.out)


if __name__ == "__main__":
    main()
