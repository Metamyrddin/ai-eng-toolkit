"""This file measures how good several search methods are on the SciFact test set, with confidence
intervals. It is part of area 7.

More technical notes:
Retrieval evaluation on SciFact (BEIR): BM25 vs dense (MiniLM+FAISS) vs hybrid (RRF / weighted) vs hybrid+cross-encoder rerank.

Reports nDCG@10, Recall@100 and MRR@10 with per-query bootstrap 95% CIs and paired bootstrap tests, so differences can be judged
against noise (300 queries). Run: python retrieval_eval.py [--rerank-k 30]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "05_evals_rigor"))
import numpy as np  # noqa: E402

import retrieval_lib as rl  # noqa: E402
import stats_utils as su  # noqa: E402
from common import env  # noqa: E402


def main():
    """Build all search methods, run every test query, compute the metrics with confidence
    intervals and save the results.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank-k", type=int, default=30)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "retrieval_eval.json"))
    a = ap.parse_args()
    ids, texts, queries, qrels = rl.load_scifact()
    id2text = dict(zip(ids, texts))
    qids = list(queries)[: a.max_queries]
    t0 = time.perf_counter(); bm25 = rl.BM25Index(ids, texts); t_bm25_build = time.perf_counter() - t0
    t0 = time.perf_counter(); dense = rl.DenseIndex(ids, texts); t_dense_build = time.perf_counter() - t0
    rer = rl.Reranker()
    runs = {"bm25": {}, "dense": {}, "hybrid_rrf": {}, "hybrid_weighted_0.5": {}, "hybrid_rrf+rerank": {}, "dense+rerank": {}}
    lat = {k: [] for k in runs}
    for q in qids:
        text = queries[q]
        t = time.perf_counter(); b_ids, b_s = bm25.search(text, a.depth); lat["bm25"].append(time.perf_counter() - t)
        t = time.perf_counter(); d_ids, d_s = dense.search(text, a.depth); lat["dense"].append(time.perf_counter() - t)
        runs["bm25"][q], runs["dense"][q] = b_ids, d_ids
        t = time.perf_counter(); runs["hybrid_rrf"][q] = rl.rrf_fuse([b_ids, d_ids], top=a.depth); lat["hybrid_rrf"].append(time.perf_counter() - t + lat["bm25"][-1] + lat["dense"][-1])
        runs["hybrid_weighted_0.5"][q] = rl.weighted_fuse(b_ids, b_s, d_ids, d_s, 0.5, a.depth)
        top = runs["hybrid_rrf"][q][: a.rerank_k]
        t = time.perf_counter(); rr = rer.rerank(text, top, id2text)
        lat["hybrid_rrf+rerank"].append(time.perf_counter() - t + lat["hybrid_rrf"][-1])
        runs["hybrid_rrf+rerank"][q] = rr + runs["hybrid_rrf"][q][a.rerank_k:]
        runs["dense+rerank"][q] = rer.rerank(text, d_ids[: a.rerank_k], id2text) + d_ids[a.rerank_k:]
    res = {"environment": env.environment_info(), "dataset": "BEIR SciFact test", "n_queries": len(qids), "n_docs": len(ids),
           "rerank_k": a.rerank_k, "index_build_seconds": {"bm25": round(t_bm25_build, 1), "dense_incl_encoding_or_cache": round(t_dense_build, 1)},
           "dense_model": env.MINILM, "reranker": env.CROSS_ENCODER, "methods": {}}
    per_q = {}
    for name, run in runs.items():
        nd = [rl.ndcg_at_k(run[q], qrels[q], 10) for q in qids]
        per_q[name] = nd
        res["methods"][name] = {
            "ndcg@10": round(float(np.mean(nd)), 4), "ndcg@10_ci95": [round(x, 4) for x in su.bootstrap_ci(nd)],
            "recall@100": round(float(np.mean([rl.recall_at_k(run[q], qrels[q], 100) for q in qids])), 4),
            "mrr@10": round(float(np.mean([rl.mrr_at_k(run[q], qrels[q], 10) for q in qids])), 4),
            "mean_query_latency_ms": round(1000 * float(np.mean(lat[name])), 1)}
        print(name, res["methods"][name], flush=True)
    res["paired_bootstrap_ndcg@10"] = {
        f"{a_}_minus_{b_}": su.paired_bootstrap(per_q[a_], per_q[b_])
        for a_, b_ in [("hybrid_rrf", "bm25"), ("hybrid_rrf", "dense"), ("hybrid_rrf+rerank", "hybrid_rrf"), ("dense", "bm25")]}
    env.save_json(res, a.out)


if __name__ == "__main__":
    main()
