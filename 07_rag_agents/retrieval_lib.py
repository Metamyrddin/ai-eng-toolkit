"""This file holds the building blocks for searching documents: loading the data, scoring
functions, a keyword index, an embedding index, rank fusion and a reranker. It is part of area
7.

More technical notes:
Retrieval building blocks: SciFact loader, IR metrics, BM25, dense (sentence-transformers + FAISS), RRF fusion, reranker.

Dataset: SciFact in BEIR format (Wadden et al. 2020 arXiv:2004.14974; BEIR benchmark Thakur et al. 2021 arXiv:2104.08663),
5,183 abstracts, 300 test claims with relevance judgements. HF ids: `BeIR/scifact`, `BeIR/scifact-qrels`.
Everything here is plain functions so it can be unit-tested (see tests/test_retrieval_metrics.py).
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

from common import env  # noqa: E402

CACHE = Path(__file__).parent / "cache"


# ----------------------------------------------------------------------------- data
def load_scifact():
    """Return (doc_ids, doc_texts, queries{qid:text}, qrels{qid:{docid:rel}}) for the TEST split."""
    from datasets import load_dataset
    corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
    queries = load_dataset("BeIR/scifact", "queries")["queries"]
    qrels_ds = load_dataset("BeIR/scifact-qrels")["test"]
    qrels: dict = {}
    for r in qrels_ds:
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    qtext = {str(r["_id"]): r["text"] for r in queries if str(r["_id"]) in qrels}
    doc_ids = [str(r["_id"]) for r in corpus]
    doc_texts = [(r["title"] + ". " + r["text"]).strip() for r in corpus]
    return doc_ids, doc_texts, qtext, qrels


# ----------------------------------------------------------------------------- metrics
def dcg(rels):
    """Add up the relevance scores of a ranked list, giving less weight to results that appear
    lower in the list.
    """
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_ids, rel: dict, k=10):
    """nDCG@k with linear gain `rel` (gain = relevance grade). Ideal ranking = sort judged docs by grade."""
    gains = [rel.get(d, 0) for d in ranked_ids[:k]]
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids, rel: dict, k=100):
    """Return the fraction of all relevant documents that appear in the first k results."""
    rel_docs = {d for d, g in rel.items() if g > 0}
    return len(rel_docs & set(ranked_ids[:k])) / len(rel_docs) if rel_docs else 0.0


def mrr_at_k(ranked_ids, rel: dict, k=10):
    """Return 1 divided by the position of the first relevant result within the first k
    results, or 0 if there is none.
    """
    for i, d in enumerate(ranked_ids[:k]):
        if rel.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


# ----------------------------------------------------------------------------- BM25
_TOK = re.compile(r"\w+")


def bm25_tokens(text: str):
    """Split a text into lower-case words for the keyword search."""
    return _TOK.findall(text.lower())


class BM25Index:
    """Okapi BM25 via `rank_bm25` (k1=1.5, b=0.75 are its defaults; BEIR's reference uses Elasticsearch BM25, so
    absolute numbers differ slightly from published ones)."""
    def __init__(self, doc_ids, doc_texts):
        """Build the keyword index for all documents."""
        from rank_bm25 import BM25Okapi
        self.ids = doc_ids
        self.bm25 = BM25Okapi([bm25_tokens(t) for t in doc_texts])

    def scores(self, query: str) -> np.ndarray:
        """Return the keyword relevance score of every document for the query."""
        return np.asarray(self.bm25.get_scores(bm25_tokens(query)))

    def search(self, query: str, k=100):
        """Return the ids and scores of the k best documents for the query."""
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return [self.ids[i] for i in top], s[top]


# ----------------------------------------------------------------------------- dense + FAISS
class DenseIndex:
    """Sentence-transformers embeddings + exact inner-product FAISS index (vectors L2-normalised => cosine)."""
    def __init__(self, doc_ids, doc_texts, model_name=env.MINILM, cache_name="scifact_minilm", batch_size=64):
        """Turn all documents into embedding vectors (or load them from the cache file) and put
        them into a FAISS index.
        """
        import faiss
        from sentence_transformers import SentenceTransformer
        self.ids, self.model = doc_ids, SentenceTransformer(model_name, device="cpu")
        path = CACHE / f"{cache_name}.npy"
        if path.exists():
            emb = np.load(path)
        else:
            CACHE.mkdir(exist_ok=True)
            emb = self.model.encode(doc_texts, batch_size=batch_size, normalize_embeddings=True,
                                    show_progress_bar=False, convert_to_numpy=True).astype("float32")
            np.save(path, emb)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.emb = emb

    def search(self, query: str, k=100):
        """Embed the query and return the ids and scores of the k most similar documents."""
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        s, i = self.index.search(q, k)
        return [self.ids[j] for j in i[0]], s[0]


# ----------------------------------------------------------------------------- fusion / rerank
def rrf_fuse(rankings: list[list[str]], k: int = 60, top: int = 100):
    """Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009): score(d) = sum_r 1/(k + rank_r(d)), rank from 1."""
    score: dict = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking, start=1):
            score[d] = score.get(d, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(score.items(), key=lambda x: -x[1])][:top]


def minmax(x):
    """Rescale a list of numbers so that the smallest becomes 0 and the largest becomes 1."""
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def weighted_fuse(ranked_a, scores_a, ranked_b, scores_b, w=0.5, top=100):
    """Convex combination of min-max-normalised scores (documents missing from a list get 0 for that list)."""
    na, nb = dict(zip(ranked_a, minmax(scores_a))), dict(zip(ranked_b, minmax(scores_b)))
    all_ids = set(na) | set(nb)
    fused = {d: w * na.get(d, 0.0) + (1 - w) * nb.get(d, 0.0) for d in all_ids}
    return [d for d, _ in sorted(fused.items(), key=lambda x: -x[1])][:top]


class Reranker:
    """Cross-encoder reranker (query and document are read TOGETHER, so it is more accurate but ~O(k) model calls/query)."""
    def __init__(self, model_name=env.CROSS_ENCODER):
        """Load the cross-encoder model that reads a question and a document together."""
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device="cpu", max_length=512)

    def rerank(self, query: str, doc_ids: list[str], id2text: dict, top=None):
        """Score each candidate document together with the query and return the candidates
        sorted from best to worst.
        """
        scores = self.model.predict([(query, id2text[d]) for d in doc_ids], batch_size=16, show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))
        out = [doc_ids[i] for i in order]
        return out[:top] if top else out
