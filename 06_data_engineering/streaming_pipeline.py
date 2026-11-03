"""This file connects all the steps of a data pipeline: read documents, filter them, remove
duplicates, turn them into token ids and write them into packed files. It never needs the whole
dataset in memory. It is part of area 6.

More technical notes:
Streaming data pipeline: stream -> quality filter -> MinHash dedup -> tokenize -> pack -> write shards.

Why streaming: web-scale corpora do not fit on a laptop disk/RAM. `datasets.load_dataset(..., streaming=True)` returns an
IterableDataset that downloads lazily, one record at a time (https://huggingface.co/docs/datasets/stream).
We keep memory bounded by using generators everywhere and writing fixed-size token shards (uint16/uint32 .npy, memmap-able).
The dedup index is the only state that grows (O(#kept docs) signatures) - we report peak RSS to show it.

Run:  python streaming_pipeline.py --n-docs 3000 [--source cache|stream]     (cache = data/fineweb_sample.jsonl, offline)
Reads the output back with PackedDataset to verify round-trip and prints throughput / stats.
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

import quality_filters as qf  # noqa: E402
from common import env  # noqa: E402
from minhash_dedup import MinHasher, choose_bands, shingles  # noqa: E402


class StreamingDeduper:
    """Online MinHash-LSH: returns True if doc is (near-)duplicate of an already-seen KEPT doc."""
    def __init__(self, threshold=0.8, num_perm=128):
        """Prepare an empty index that remembers the signatures of all documents seen so far.
        """
        self.mh, self.t = MinHasher(num_perm), threshold
        self.b, self.r = choose_bands(num_perm, threshold)
        self.sigs, self.buckets = [], {}

    def is_duplicate(self, text: str) -> bool:
        """Return True if the text is a near-duplicate of an earlier document. Otherwise
        remember it and return False.
        """
        sig = self.mh.signature(shingles(text))
        keys = [(band, sig[band * self.r:(band + 1) * self.r].tobytes()) for band in range(self.b)]
        for k in keys:
            for j in self.buckets.get(k, ()):
                if MinHasher.estimate(sig, self.sigs[j]) >= self.t:
                    return True
        idx = len(self.sigs); self.sigs.append(sig)
        for k in keys:
            self.buckets.setdefault(k, []).append(idx)
        return False


def doc_iter(source, n):
    """Yield documents one by one, either from the local cache file or from the internet
    stream.
    """
    if source == "stream":
        from fetch_sample import stream_fineweb
        for i, row in enumerate(stream_fineweb()):
            if i >= n:
                return
            yield row["text"]
    else:
        from fetch_sample import load_sample
        for d in load_sample(n):
            yield d["text"]


def pack_tokens(token_stream, seq_len):
    """Concatenate docs (separated by EOS, added by the caller) and cut into fixed-length rows; remainder is carried over."""
    buf = []
    for toks in token_stream:
        buf.extend(toks)
        while len(buf) >= seq_len:
            yield np.asarray(buf[:seq_len], dtype=np.uint32)
            del buf[:seq_len]


class PackedDataset:
    """Reads the shards written by this pipeline: indexable rows of seq_len tokens."""
    def __init__(self, shard_dir):
        """Open all shard files without loading them into memory, and remember where each shard
        starts.
        """
        self.files = sorted(Path(shard_dir).glob("shard_*.npy"))
        self.arrs = [np.load(f, mmap_mode="r") for f in self.files]
        self.cum = np.cumsum([len(a) for a in self.arrs])

    def __len__(self):
        """Return the number of rows of tokens in all shards together."""
        return int(self.cum[-1]) if len(self.cum) else 0

    def __getitem__(self, i):
        """Return row number i of the combined dataset by finding the shard that contains it.
        """
        s = int(np.searchsorted(self.cum, i, side="right"))
        return self.arrs[s][i - (int(self.cum[s - 1]) if s else 0)]


def main():
    """Run the whole pipeline (filter, remove duplicates, tokenise, pack, write shards), read
    the shards back to check them and save the statistics.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=3000)
    ap.add_argument("--source", choices=["cache", "stream"], default="cache")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--rows-per-shard", type=int, default=256)
    ap.add_argument("--tokenizer", default=env.SMOL_135M)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "data" / "shards"))
    ap.add_argument("--inject-dups", type=int, default=100, help="re-emit this many earlier docs verbatim to exercise dedup")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("shard_*.npy"):
        f.unlink()
    dd = StreamingDeduper()
    stats = {"seen": 0, "dropped_quality": 0, "dropped_duplicate": 0, "kept": 0, "tokens_kept": 0}
    fail_reasons: dict = {}
    first_docs = []

    def source_with_dups():
        """Yield the documents, and at the end repeat some early documents on purpose, to test
        the duplicate remover.
        """
        for i, t in enumerate(doc_iter(a.source, a.n_docs)):
            if i < a.inject_dups:
                first_docs.append(t)
            yield t
        for t in first_docs:  # planted exact duplicates of the first docs
            yield t

    def token_stream():
        """Yield token ids of every document that passes the quality filter and is not a
        duplicate, while counting what was dropped.
        """
        for text in source_with_dups():
            stats["seen"] += 1
            ok, fails = qf.check(text)
            if not ok:
                stats["dropped_quality"] += 1
                for f in fails:
                    fail_reasons[f] = fail_reasons.get(f, 0) + 1
                continue
            if dd.is_duplicate(text):
                stats["dropped_duplicate"] += 1
                continue
            ids = tok(text, add_special_tokens=False).input_ids + [tok.eos_token_id]
            stats["kept"] += 1; stats["tokens_kept"] += len(ids)
            yield ids

    t0 = time.perf_counter()
    rows, shard_id, n_rows = [], 0, 0
    for row in pack_tokens(token_stream(), a.seq_len):
        rows.append(row); n_rows += 1
        if len(rows) == a.rows_per_shard:
            np.save(out / f"shard_{shard_id:04d}.npy", np.stack(rows)); rows, shard_id = [], shard_id + 1
    if rows:
        np.save(out / f"shard_{shard_id:04d}.npy", np.stack(rows)); shard_id += 1
    elapsed = time.perf_counter() - t0
    ds = PackedDataset(out)
    assert len(ds) == n_rows and ds[0].shape == (a.seq_len,)
    res = {"environment": env.environment_info(), **stats, "fail_reasons": fail_reasons, "rows": n_rows,
           "seq_len": a.seq_len, "shards": shard_id, "seconds": round(elapsed, 1),
           "docs_per_s": round(stats["seen"] / elapsed, 1), "tokens_per_s": round(stats["tokens_kept"] / elapsed, 0),
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
           "planted_exact_dups": len(first_docs), "source": a.source,
           "roundtrip_ok": bool((ds[0] == np.load(sorted(out.glob("shard_*.npy"))[0])[0]).all())}
    print(res)
    env.save_json(res, Path(__file__).parent / "results" / "streaming_pipeline.json")


if __name__ == "__main__":
    main()
