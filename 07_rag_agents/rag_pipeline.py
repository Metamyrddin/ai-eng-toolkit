"""This file builds a question-answering system with retrieval-augmented generation (RAG). It finds
the documents that fit the question and lets a local model answer using them. It is part of area
7.

More technical notes:
End-to-end RAG: hybrid retrieval (BM25 + dense via RRF) -> cross-encoder rerank -> local LLM answer with citations.

RAG (Lewis et al. 2020, arXiv:2005.11401): ground an LLM's answer in retrieved passages to reduce reliance on parametric memory.
LLM = Qwen2.5-0.5B-Instruct GGUF through llama-cpp-python (CPU). SciFact claims are used as questions.

What we MEASURE (no judge model, so every number is programmatic):
  * context_recall@k : is a gold-relevant abstract inside the k passages shown to the LLM?
  * citation_validity: of the [id] citations the LLM wrote, how many point to a passage that was actually in its context?
  * citation_rate    : fraction of answers that contain >=1 citation
  * cites_gold_rate  : fraction of answers whose citations include a gold-relevant abstract
We do NOT claim answer correctness: SciFact support/contradict labels are not used here. Sample answers are saved for reading.
Run: python rag_pipeline.py --n 30        |       python rag_pipeline.py --ask "Does vitamin D reduce fracture risk?"
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

import retrieval_lib as rl  # noqa: E402
from common import env  # noqa: E402

PROMPT = """Answer using ONLY the documents below. Say whether they support, contradict, or do not address the claim, in 1-2 sentences, and cite the supporting document ids in square brackets like [123]. If the documents are irrelevant, say so.

{docs}

Claim: {q}"""


class RAG:
    """Retrieval-augmented generation: find relevant documents first, then let the language
    model answer using only those documents.
    """
    def __init__(self, llm, k_ctx=3, rerank_k=20, use_rerank=True):
        """Load the dataset, build the keyword and embedding indexes and, if wanted, the
        reranker.
        """
        self.ids, self.texts, self.queries, self.qrels = rl.load_scifact()
        self.id2text = dict(zip(self.ids, self.texts))
        self.bm25, self.dense = rl.BM25Index(self.ids, self.texts), rl.DenseIndex(self.ids, self.texts)
        self.rer = rl.Reranker() if use_rerank else None
        self.llm, self.k_ctx, self.rerank_k = llm, k_ctx, rerank_k

    def retrieve(self, q):
        """Find candidate documents with both search methods, merge the two rankings and rerank
        the best candidates.
        """
        b, _ = self.bm25.search(q, 50); d, _ = self.dense.search(q, 50)
        cand = rl.rrf_fuse([b, d], top=self.rerank_k)
        return (self.rer.rerank(q, cand, self.id2text) if self.rer else cand)[: self.k_ctx]

    def answer(self, q):
        """Retrieve documents for the question, put them into the prompt and return the model's
        answer with the document ids.
        """
        ctx = self.retrieve(q)
        docs = "\n\n".join(f"[{i}] {self.id2text[i][:700]}" for i in ctx)
        out = self.llm.create_chat_completion(messages=[{"role": "user", "content": PROMPT.format(docs=docs, q=q)}],
                                              max_tokens=160, temperature=0.0)
        return out["choices"][0]["message"]["content"].strip(), ctx


def main():
    """Either answer one question or run the evaluation on many questions, then save the
    results.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ask", default=None)
    a = ap.parse_args()
    from llama_cpp import Llama
    import os
    llm = Llama(model_path=env.gguf_path("q4_k_m"), n_ctx=4096, n_threads=os.cpu_count(), verbose=False, seed=0)
    rag = RAG(llm, k_ctx=a.k)
    if a.ask:
        ans, ctx = rag.answer(a.ask); print(ans, "\ncontext ids:", ctx); return
    qids = sorted(rag.queries, key=int)[: a.n]
    rows, t0 = [], time.perf_counter()
    for q in qids:
        ans, ctx = rag.answer(rag.queries[q])
        cited = re.findall(r"\[(\d+)\]", ans)
        gold = {d for d, g in rag.qrels[q].items() if g > 0}
        rows.append({"qid": q, "claim": rag.queries[q], "answer": ans, "context_ids": ctx, "cited": cited,
                     "ctx_has_gold": bool(gold & set(ctx)), "n_cited": len(cited),
                     "n_cited_valid": sum(c in ctx for c in cited), "cites_gold": bool(gold & set(cited))})
    tot_c = sum(r["n_cited"] for r in rows)
    res = {"environment": env.environment_info(), "n_queries": len(rows), "k_ctx": a.k, "seconds": round(time.perf_counter() - t0, 1),
           "context_recall@k": round(float(np.mean([r["ctx_has_gold"] for r in rows])), 3),
           "citation_rate": round(float(np.mean([r["n_cited"] > 0 for r in rows])), 3),
           "citation_validity": round(sum(r["n_cited_valid"] for r in rows) / max(1, tot_c), 3), "n_citations": tot_c,
           "cites_gold_rate": round(float(np.mean([r["cites_gold"] for r in rows])), 3), "samples": rows[:6], "all_rows": rows}
    print({k: v for k, v in res.items() if k not in ("environment", "samples", "all_rows")})
    env.save_json(res, Path(__file__).parent / "results" / "rag_pipeline.json")


if __name__ == "__main__":
    main()
