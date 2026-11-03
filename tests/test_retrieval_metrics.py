"""These tests check area 7's search metrics, rank fusion, keyword search and FAISS, using small
examples that can be worked out by hand.
"""
import math

import numpy as np
import pytest

import retrieval_lib as rl


def test_ndcg_hand_computed():
    """Check nDCG against a value that we calculated by hand."""
    rel = {"a": 1, "b": 1, "c": 1}
    # ranking [x, a, y, b]: gains 0,1,0,1 -> DCG = 1/log2(3) + 1/log2(5); ideal (3 relevant, k=4) = 1 + 1/log2(3) + 1/log2(4)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert rl.ndcg_at_k(["x", "a", "y", "b"], rel, 4) == pytest.approx(dcg / ideal)
    assert rl.ndcg_at_k(["a", "b", "c"], rel, 10) == pytest.approx(1.0)
    assert rl.ndcg_at_k(["x", "y"], rel, 10) == 0.0
    assert rl.ndcg_at_k(["a"], {}, 10) == 0.0


def test_graded_gains():
    """Check that nDCG rewards putting the more relevant document first."""
    rel = {"a": 2, "b": 1}
    assert rl.ndcg_at_k(["b", "a"], rel, 2) < 1.0
    assert rl.ndcg_at_k(["a", "b"], rel, 2) == pytest.approx(1.0)


def test_recall_and_mrr():
    """Check recall and mean reciprocal rank on small examples."""
    rel = {"a": 1, "b": 1, "c": 0}
    assert rl.recall_at_k(["a", "x", "b"], rel, 2) == 0.5
    assert rl.recall_at_k(["a", "x", "b"], rel, 3) == 1.0
    assert rl.mrr_at_k(["x", "y", "b"], rel, 10) == pytest.approx(1 / 3)
    assert rl.mrr_at_k(["x", "y", "b"], rel, 2) == 0.0


def test_rrf_formula_and_order():
    """Check that reciprocal rank fusion follows its formula."""
    fused = rl.rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    # a: 1/61 + 1/62 ; b: 1/62 + 1/61 -> tie, then c (1/63) and d (1/63)
    assert set(fused[:2]) == {"a", "b"} and set(fused[2:]) == {"c", "d"}
    assert rl.rrf_fuse([["a", "b"], ["a", "c"]])[0] == "a"
    assert rl.rrf_fuse([["a"]], top=1) == ["a"]


def test_weighted_fuse_extremes():
    """Check that the weighted fusion follows only one ranking when its weight is 0 or 1."""
    ra, sa = ["a", "b", "c"], [3.0, 2.0, 1.0]
    rb, sb = ["c", "b", "d"], [3.0, 2.0, 1.0]
    assert rl.weighted_fuse(ra, sa, rb, sb, w=1.0)[:2] == ["a", "b"]   # c and d tie at 0 -> only top-2 is determined
    assert rl.weighted_fuse(ra, sa, rb, sb, w=0.0)[:2] == ["c", "b"]
    mid = rl.weighted_fuse(ra, sa, rb, sb, w=0.5)
    assert mid[0] in ("a", "c") or mid[0] == "b"   # b is 2nd in both lists -> 0.5*0.5+0.5*0.5 = 0.5 ties a/c at 0.5


def test_bm25_ranks_matching_doc_first():
    """Check that keyword search puts the matching document first."""
    ids = ["d1", "d2", "d3"]
    texts = ["cats purr and sleep all day", "stock markets fell sharply on monday", "the dog barked at the mailman"]
    bm = rl.BM25Index(ids, texts)
    assert bm.search("sleepy cats", 3)[0][0] == "d1"
    assert bm.search("stock market", 3)[0][0] == "d2"


def test_faiss_flat_ip_is_cosine_on_normalised_vectors():
    """Check that FAISS inner product equals cosine similarity for unit-length vectors."""
    import faiss
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 16)).astype("float32"); X /= np.linalg.norm(X, axis=1, keepdims=True)
    idx = faiss.IndexFlatIP(16); idx.add(X)
    q = X[7:8]
    s, i = idx.search(q, 3)
    assert i[0][0] == 7 and s[0][0] == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(s[0], (X @ q[0])[i[0]], atol=1e-5)
