"""This file downloads a small sample of real web pages from the FineWeb dataset and saves them on
this computer. It is part of area 6 and gives the other scripts real text to work with.

More technical notes:
Stream the first N documents of FineWeb (sample-10BT) and cache them locally as JSONL.

After this, every other script in this folder works offline from data/fineweb_sample.jsonl.
FineWeb: Penedo et al. 2024, arXiv:2406.17557, dataset card https://huggingface.co/datasets/HuggingFaceFW/fineweb
Run: python fetch_sample.py --n 3000
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import env  # noqa: E402,F401

DATA = Path(__file__).parent / "data" / "fineweb_sample.jsonl"


def stream_fineweb(config="sample-10BT"):
    """Open the FineWeb web-text dataset as a stream, which means that documents are downloaded
    one at a time instead of all at once.
    """
    from datasets import load_dataset
    return load_dataset("HuggingFaceFW/fineweb", name=config, split="train", streaming=True)


def load_sample(n=None):
    """Read the cached sample (fetches it first if missing)."""
    if not DATA.exists():
        fetch(3000)
    docs = []
    with open(DATA) as f:
        for line in f:
            docs.append(json.loads(line))
            if n and len(docs) >= n:
                break
    return docs


def fetch(n):
    """Download the first n documents of FineWeb and save them in a local file with one JSON
    object per line.
    """
    DATA.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA, "w") as f:
        for i, row in enumerate(stream_fineweb()):
            if i >= n:
                break
            f.write(json.dumps({"id": row["id"], "url": row["url"], "text": row["text"]}) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3000); a = ap.parse_args()
    fetch(a.n); print("wrote", DATA, "docs:", sum(1 for _ in open(DATA)), flush=True)
    # The background download threads of `datasets` streaming can abort ("terminate called without an active exception")
    # during interpreter shutdown on some builds; the file is complete and closed by now, so exit immediately.
    import os; os._exit(0)
