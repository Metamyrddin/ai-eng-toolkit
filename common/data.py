"""This file holds small helpers for loading datasets. Several areas use them so that the loading
code is written only once.

More technical notes:
Tiny dataset helpers used by several areas."""
from __future__ import annotations

from . import env  # noqa: F401  (sets HF_HOME)


def wikitext_text(n_chars: int = 20000, split: str = "test") -> str:
    """First `n_chars` of WikiText-2 (raw). Used for perplexity sanity checks.

    WikiText-2 is a standard LM benchmark (Merity et al. 2016, arXiv:1609.07843).
    """
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "".join(r["text"] for r in ds)
    return text[:n_chars]
