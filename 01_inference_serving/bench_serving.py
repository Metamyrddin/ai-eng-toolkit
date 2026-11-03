"""This file acts like many users chatting with a model server at once and measures how quickly the
server answers. It works with any server that speaks the OpenAI chat format. It is part of area
1 and is the tool that measured the llama.cpp server.

More technical notes:
Benchmark any OpenAI-compatible streaming server (llama.cpp llama-server, vLLM, SGLang, TGI, ...).

Uses only the standard library (urllib + threads) so it runs anywhere.
Measures per request: TTFT (time to first streamed content chunk), end-to-end latency, inter-chunk
latency (~TPOT), output chunk count (== tokens for llama.cpp/vLLM which stream one token per chunk;
we also report server-side `usage` when present). Aggregates over `--concurrency` simultaneous clients.

Local test:   ./serve_llamacpp.sh q4_k_m 8080 4 &
              python bench_serving.py --url http://127.0.0.1:8080/v1/chat/completions --concurrency 1 4
Cloud:        vllm serve <model>   then   python bench_serving.py --url http://HOST:8000/v1/chat/completions --model <model>
"""
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import env  # noqa: E402

PROMPTS = ["Explain what a hash table is in two sentences.", "Write a haiku about autumn.",
           "What is 17 * 23? Show your steps.", "Name three benefits of unit tests.",
           "Describe how a binary search works.", "Give a short definition of entropy.",
           "Why do we normalise inputs in neural networks?", "What is a mutex?"]


def one_request(url, model, prompt, max_tokens, api_key=None):
    """Send one streaming chat request to the server and return the timing facts: time to first
    token, time between tokens, and total time.
    """
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers)
    t0 = time.perf_counter()
    first, stamps, usage = None, [], None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if ch and (ch[0].get("delta", {}).get("content") or ch[0].get("text")):
                now = time.perf_counter()
                if first is None:
                    first = now - t0
                stamps.append(now)
    end = time.perf_counter() - t0
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return {"ttft": first, "e2e": end, "chunks": len(stamps), "mean_gap": statistics.mean(gaps) if gaps else None,
            "completion_tokens": (usage or {}).get("completion_tokens")}


def run_level(url, model, concurrency, n_requests, max_tokens, api_key):
    """Send many requests with a fixed number of clients at the same time, and summarise the
    speed, including how many requests failed.
    """
    results, lock = [], threading.Lock()
    counter = iter(range(n_requests))

    def worker():
        """Each worker thread repeatedly takes the next request number and sends that request
        until none are left.
        """
        while True:
            with lock:
                try:
                    i = next(counter)
                except StopIteration:
                    return
            r = one_request(url, model, PROMPTS[i % len(PROMPTS)], max_tokens, api_key)
            with lock:
                results.append(r)

    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker) for _ in range(concurrency)]
    [t.start() for t in ths]; [t.join() for t in ths]
    wall = time.perf_counter() - t0
    n_total = len(results)
    failed = sum(1 for r in results if r["ttft"] is None)  # empty/failed streams are reported, never hidden
    results = [r for r in results if r["ttft"] is not None] or results
    toks = sum((r["completion_tokens"] or r["chunks"]) for r in results)
    ttfts = sorted(r["ttft"] for r in results if r["ttft"] is not None) or [float("nan")]
    gaps = [r["mean_gap"] for r in results if r["mean_gap"]]
    return {"concurrency": concurrency, "requests_ok": n_total - failed, "requests_failed_or_empty": failed, "wall_s": round(wall, 2),
            "output_tokens_total": toks, "throughput_tok_s": round(toks / wall, 1),
            "ttft_ms_median": round(1000 * statistics.median(ttfts), 1),
            "ttft_ms_p95": round(1000 * ttfts[max(0, int(0.95 * len(ttfts)) - 1)], 1),
            "tpot_ms_mean_of_means": round(1000 * statistics.mean(gaps), 2) if gaps else None,
            "e2e_s_median": round(statistics.median(r["e2e"] for r in results), 2)}


def main():
    """Test each concurrency level several times, print the results and save them as a JSON
    file.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--model", default="default")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--requests-per-level", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--trials", type=int, default=3, help="repeat each concurrency level; the median-throughput trial is reported")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "serving_bench.json"))
    a = ap.parse_args()
    one_request(a.url, a.model, "Hello", 4, a.api_key)  # warm-up
    rows = []
    for c in a.concurrency:
        trials = [run_level(a.url, a.model, c, a.requests_per_level, a.max_tokens, a.api_key) for _ in range(a.trials)]
        row = dict(sorted(trials, key=lambda t: t["throughput_tok_s"])[len(trials) // 2])  # trial with median throughput
        row["trials"] = a.trials
        row["throughput_tok_s_all_trials"] = [t["throughput_tok_s"] for t in trials]
        row["failed_or_empty_all_trials"] = [t["requests_failed_or_empty"] for t in trials]
        rows.append(row); print(row, flush=True)
    env.save_json({"environment": env.environment_info(), "url": a.url, "model": a.model, "rows": rows}, a.out)


if __name__ == "__main__":
    main()
