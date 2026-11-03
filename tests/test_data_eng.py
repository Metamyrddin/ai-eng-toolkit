"""These tests check area 6: MinHash estimates, duplicate removal, quality filters, token packing
and the streaming duplicate remover.
"""
import random

import numpy as np
import pytest

import minhash_dedup as md
import quality_filters as qf
import streaming_pipeline as sp

BASE = ("the committee reviewed the proposal carefully and decided that the new library should open longer on weekends because "
        "many students need quiet places to study after classes and during the exam period every year in the city")


def words(n, seed):
    """Create a text of n random words from a small vocabulary, using a seed."""
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(500)]
    return " ".join(rng.choice(vocab) for _ in range(n))


def test_signature_estimates_jaccard_within_tolerance():
    """Check that the MinHash estimate is close to the exact similarity."""
    mh = md.MinHasher(256, seed=3)
    # construct sets with exact Jaccard: |A∩B| = c, |A∪B| = 200 + extra
    for c, extra in [(100, 100), (150, 50), (50, 150)]:
        A = {f"a{i}" for i in range(c)} | {f"x{i}" for i in range(100)}
        B = {f"a{i}" for i in range(c)} | {f"y{i}" for i in range(extra)}
        j = md.exact_jaccard(A, B)
        est = md.MinHasher.estimate(mh.signature(A), mh.signature(B))
        assert abs(est - j) < 4 * np.sqrt(j * (1 - j) / 256) + 1e-9   # 4 sigma


def test_exact_jaccard_edge_cases():
    """Check the exact similarity for empty sets and for sets with nothing in common."""
    assert md.exact_jaccard(set(), set()) == 1.0
    assert md.exact_jaccard({"a"}, {"b"}) == 0.0


def test_choose_bands_threshold_tradeoff():
    """Check that a higher threshold leads to more rows per band."""
    b_low, r_low = md.choose_bands(128, 0.3)
    b_hi, r_hi = md.choose_bands(128, 0.9)
    assert r_hi > r_low and b_low * r_low <= 128 and b_hi * r_hi <= 128


def test_dedup_removes_exact_and_near_dups_keeps_distinct():
    """Check that duplicates are removed and different documents are kept."""
    docs = [words(150, i) for i in range(20)]
    docs.append(docs[3])                                              # exact duplicate
    docs.append(md.make_near_duplicate(docs[7], 0.02, random.Random(0)))  # near duplicate
    keep, info = md.dedup_minhash(docs, threshold=0.7)
    assert keep[:20] == [True] * 20
    assert keep[20] is False and keep[21] is False
    assert md.dedup_datasketch(docs, 0.7)[:20] == [True] * 20


def test_near_duplicate_edit_rate_controls_similarity():
    """Check that more edits make two documents less similar."""
    t = words(400, 1)
    rng = random.Random(0)
    js = [md.exact_jaccard(md.shingles(t), md.shingles(md.make_near_duplicate(t, r, rng))) for r in (0.01, 0.05, 0.2)]
    assert js[0] > js[1] > js[2]


GOOD = ("The city council met on Tuesday to discuss the new library budget. Members of the public were invited to speak, and many "
        "residents said that longer opening hours would help students and families. After a long debate, the council voted to "
        "approve the plan, and the library will extend its weekend hours starting in the spring. The mayor thanked everyone for "
        "taking part in the process and said that she hoped the change would bring more people to the library.")


def test_quality_filters_accept_prose_reject_spam():
    """Check that normal text passes and spam is rejected."""
    assert qf.check(GOOD)[0]
    assert "repeated_ngram" in qf.check("Click here to buy now. " * 60)[1] or "duplicate_lines" in qf.check("Click here to buy now. " * 60)[1]
    assert not qf.check(" ".join(["BUY", "NOW", "CHEAP"] * 60))[0]
    assert "word_count" in qf.check("too short")[1]
    rng = random.Random(0)
    junk = " ".join("".join(rng.choice("#@$%^&*abc123") for _ in range(8)) for _ in range(120))
    assert not qf.check(junk)[0]


def test_auc_helper():
    """Check the AUC helper for perfect, reversed and tied scores."""
    assert qf.auc([3, 4, 5], [0, 1, 2]) == 1.0
    assert qf.auc([0, 1, 2], [3, 4, 5]) == 0.0
    assert qf.auc([1, 1], [1, 1]) == 0.5


def test_pack_tokens_preserves_order_and_length():
    """Check that packing keeps the token order and that the leftover tokens are carried over.
    """
    stream = [[1, 2, 3], [4, 5, 6, 7, 8], [9, 10]]
    rows = list(sp.pack_tokens(iter(stream), 4))
    assert [r.tolist() for r in rows] == [[1, 2, 3, 4], [5, 6, 7, 8]]  # remainder (9, 10) is carried over, not emitted


def test_packed_dataset_roundtrip(tmp_path):
    """Check that written shards can be read back unchanged."""
    a = np.arange(12, dtype=np.uint32).reshape(3, 4); b = np.arange(12, 20, dtype=np.uint32).reshape(2, 4)
    np.save(tmp_path / "shard_0000.npy", a); np.save(tmp_path / "shard_0001.npy", b)
    ds = sp.PackedDataset(tmp_path)
    assert len(ds) == 5 and ds[0].tolist() == [0, 1, 2, 3] and ds[3].tolist() == [12, 13, 14, 15] and ds[4].tolist() == [16, 17, 18, 19]


def test_streaming_deduper():
    """Check that the streaming duplicate remover catches exact and near duplicates."""
    dd = sp.StreamingDeduper(0.8)
    d1, d2 = words(200, 1), words(200, 2)
    assert dd.is_duplicate(d1) is False and dd.is_duplicate(d2) is False
    assert dd.is_duplicate(d1) is True
    assert dd.is_duplicate(md.make_near_duplicate(d2, 0.01, random.Random(0))) is True
