"""This file finds and removes near-duplicate documents using MinHash and LSH, which are fast ways
to guess how similar two texts are. We wrote it ourselves and compare it with the datasketch
library. It is part of area 6.

More technical notes:
Near-duplicate removal with MinHash + LSH: from-scratch implementation, cross-checked against `datasketch`.

Idea (Broder 1997, "On the resemblance and containment of documents"): represent a document as its set of word k-shingles;
the probability that two documents have the same MinHash under a random permutation equals their Jaccard similarity,
so a signature of m minima gives an unbiased Jaccard estimate with std ~ sqrt(J(1-J)/m).
LSH banding: split the m values into b bands of r rows; two docs become candidates if ANY band matches entirely, which
happens with probability 1-(1-J^r)^b - an S-curve whose threshold is tuned by (b, r).
LLM relevance: Lee et al. 2022, "Deduplicating Training Data Makes Language Models Better", arXiv:2107.06499.

Experiment (python minhash_dedup.py): take N real FineWeb docs, plant K near-duplicates made by random word deletions /
swaps with known edit rate, run (a) own implementation, (b) datasketch MinHashLSH, and report
   * accuracy of the signature Jaccard estimate vs EXACT Jaccard (bias, std),
   * recall on planted duplicates whose exact Jaccard >= threshold, precision of removed docs (exact verification),
   * agreement between (a) and (b) and wall-clock time.
"""
import argparse
import hashlib
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

from common import env  # noqa: E402

P = (1 << 31) - 1  # Mersenne prime 2^31-1; with a,b,h < 2^31 products stay < 2^62 (no uint64 overflow)
_TOKEN = re.compile(r"\w+")


def shingles(text: str, k: int = 5) -> set[str]:
    """Cut a text into overlapping groups of k words (called shingles). Two similar texts share
    many shingles.
    """
    w = _TOKEN.findall(text.lower())
    if len(w) < k:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}


def exact_jaccard(a: set, b: set) -> float:
    """Return the exact Jaccard similarity of two sets: the size of the overlap divided by the
    size of the union.
    """
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _h31(s: str) -> int:
    """Turn a piece of text into a large whole number with a hash function, in a way that
    always gives the same number for the same text.
    """
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") % P


class MinHasher:
    """Creates MinHash signatures: short lists of numbers that stand in for a whole set of
    shingles.
    """
    def __init__(self, num_perm: int = 128, seed: int = 1):
        """Choose the random numbers that define the hash functions. The seed makes the choice
        repeatable.
        """
        rng = np.random.default_rng(seed)
        self.m = num_perm
        self.a = rng.integers(1, P, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, P, size=num_perm, dtype=np.uint64)

    def signature(self, sh: set[str]) -> np.ndarray:
        """Return the signature of a set of shingles: for each hash function, the smallest hash
        value over all shingles.
        """
        if not sh:
            return np.full(self.m, P, dtype=np.uint64)
        h = np.fromiter((_h31(s) for s in sh), dtype=np.uint64, count=len(sh))
        return ((self.a[:, None] * h[None, :] + self.b[:, None]) % P).min(axis=1)

    @staticmethod
    def estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Estimate the Jaccard similarity of two documents as the fraction of signature
        positions that are equal.
        """
        return float(np.mean(sig_a == sig_b))


def choose_bands(num_perm: int, threshold: float, fp_weight=0.5, fn_weight=0.5):
    """(bands, rows) with bands*rows <= num_perm minimising weighted false-positive + false-negative area."""
    best, best_err = None, 1e9
    xs = np.linspace(0, 1, 201)
    for r in range(1, num_perm + 1):
        b = num_perm // r
        if b == 0:
            continue
        p = 1 - (1 - xs ** r) ** b
        fp = np.trapezoid(p[xs < threshold], xs[xs < threshold])
        fn = np.trapezoid(1 - p[xs >= threshold], xs[xs >= threshold])
        err = fp_weight * fp + fn_weight * fn
        if err < best_err:
            best, best_err = (b, r), err
    return best


class UnionFind:
    """A helper that keeps track of which documents belong to the same group of near-
    duplicates.
    """
    def __init__(self, n):
        """Start with every document in its own group."""
        self.p = list(range(n))

    def find(self, x):
        """Return the representative (the earliest document) of the group that a document
        belongs to.
        """
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x

    def union(self, a, b):
        """Merge the groups of two documents, keeping the earliest document as the
        representative.
        """
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)  # keep the earliest doc as the cluster representative


def dedup_minhash(docs: list[str], threshold=0.8, num_perm=128, k=5, seed=1):
    """Return (keep_mask, info). The first document of every near-duplicate cluster is kept."""
    mh = MinHasher(num_perm, seed)
    b, r = choose_bands(num_perm, threshold)
    sigs = [mh.signature(shingles(d, k)) for d in docs]
    buckets: dict = {}
    uf = UnionFind(len(docs))
    cand = 0
    for i, s in enumerate(sigs):
        for band in range(b):
            key = (band, s[band * r:(band + 1) * r].tobytes())
            for j in buckets.setdefault(key, []):
                if uf.find(i) != uf.find(j) and MinHasher.estimate(sigs[i], sigs[j]) >= threshold:
                    uf.union(i, j); cand += 1
            buckets[key].append(i)
    keep = [uf.find(i) == i for i in range(len(docs))]
    return keep, {"bands": b, "rows": r, "num_perm": num_perm, "threshold": threshold, "verified_merges": cand}


def dedup_datasketch(docs, threshold=0.8, num_perm=128, k=5):
    """Remove near-duplicates with the datasketch library. We use it to check our own
    implementation.
    """
    from datasketch import MinHash, MinHashLSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep = []
    for i, d in enumerate(docs):
        m = MinHash(num_perm=num_perm)
        for s in shingles(d, k):
            m.update(s.encode())
        if lsh.query(m):
            keep.append(False)
        else:
            lsh.insert(str(i), m); keep.append(True)
    return keep


def make_near_duplicate(text: str, edit_rate: float, rng: random.Random) -> str:
    """Delete or swap-with-neighbour ~edit_rate of words (keeps most 5-shingles intact for small rates)."""
    w = text.split()
    out, i = [], 0
    while i < len(w):
        if rng.random() < edit_rate:
            if rng.random() < 0.5:
                i += 1; continue  # deletion
            if i + 1 < len(w):
                out += [w[i + 1], w[i]]; i += 2; continue  # swap
        out.append(w[i]); i += 1
    return " ".join(out)


def main():
    """Plant near-duplicates in real text, run both implementations, measure the accuracy, and
    save the results.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--planted", type=int, default=150)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "minhash.json"))
    a = ap.parse_args()
    from fetch_sample import load_sample
    base = [d["text"] for d in load_sample(a.n)]
    base = [t for t in base if len(t.split()) >= 50]
    rng = random.Random(a.seed)
    src = rng.sample(range(len(base)), min(a.planted, len(base)))
    rates = [0.01, 0.03, 0.05, 0.10, 0.20]
    planted = [(i, make_near_duplicate(base[i], rates[j % len(rates)], rng), rates[j % len(rates)]) for j, i in enumerate(src)]
    docs = base + [p[1] for p in planted]
    meta = [("orig", i) for i in range(len(base))] + [("dup", p[0]) for p in planted]
    res = {"environment": env.environment_info(), "n_original_docs": len(base), "n_planted": len(planted), **vars(a)}

    # (1) signature accuracy vs exact Jaccard on planted pairs
    mh = MinHasher(a.num_perm)
    errs, rows = [], []
    for (i, dup, rate) in planted:
        sa, sb = shingles(base[i]), shingles(dup)
        ej = exact_jaccard(sa, sb)
        est = MinHasher.estimate(mh.signature(sa), mh.signature(sb))
        errs.append(est - ej); rows.append((rate, ej, est))
    res["signature_estimate_error"] = {"mean_error(bias)": float(np.mean(errs)), "std_error": float(np.std(errs)),
                                       "theory_std_at_J0.5": float(np.sqrt(0.25 / a.num_perm)), "n_pairs": len(errs)}
    res["exact_jaccard_by_edit_rate"] = {str(r): round(float(np.mean([e for rr, e, _ in rows if rr == r])), 3) for r in rates}

    # (2) own implementation
    t0 = time.perf_counter(); keep, info = dedup_minhash(docs, a.threshold, a.num_perm); t_own = time.perf_counter() - t0
    # (3) datasketch
    t0 = time.perf_counter(); keep_ds = dedup_datasketch(docs, a.threshold, a.num_perm); t_ds = time.perf_counter() - t0

    def score(keep_mask, name):
        """Check a keep-or-remove decision list against exact similarities, and return recall
        and precision.
        """
        sets = {}
        def sh(i):
            """Return the shingles of document i, computing them only the first time."""
            if i not in sets:
                sets[i] = shingles(docs[i])
            return sets[i]
        # recall: planted dups whose exact Jaccard >= threshold should be removed
        should = [k for k, (i, dup, rate) in enumerate(planted) if exact_jaccard(shingles(base[i]), shingles(dup)) >= a.threshold]
        removed_planted = [k for k in should if not keep_mask[len(base) + k]]
        # precision: every removed doc must have exact Jaccard >= (threshold - 0.15) with SOME kept doc (verify against all kept)
        removed = [i for i, k in enumerate(keep_mask) if not k]
        bad = 0
        kept_idx = [i for i, k in enumerate(keep_mask) if k]
        for i in removed:
            best = max(exact_jaccard(sh(i), sh(j)) for j in kept_idx if j < i) if any(j < i for j in kept_idx) else 0
            bad += best < a.threshold - 0.15
        return {"docs_removed": len(removed), "planted_pairs_above_threshold": len(should),
                "recall_on_planted_above_threshold": round(len(removed_planted) / max(1, len(should)), 4),
                "removed_docs_not_verified_as_similar(exact_J<thr-0.15)": bad}

    res["own_implementation"] = {**score(keep, "own"), "seconds": round(t_own, 2), **info}
    res["datasketch"] = {**score(keep_ds, "ds"), "seconds": round(t_ds, 2)}
    res["agreement_keep_masks"] = round(float(np.mean(np.array(keep) == np.array(keep_ds))), 4)
    print(res)
    env.save_json(res, a.out)


if __name__ == "__main__":
    main()
