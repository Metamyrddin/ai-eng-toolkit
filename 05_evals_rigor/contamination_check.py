"""This file checks whether the questions of a benchmark also appear in a training text, which
would make the test unfair. It looks for runs of identical words. It is part of area 5.

More technical notes:
N-gram contamination check: does an evaluation item appear (verbatim-ish) in a training corpus?

Method (as in GPT-3, Brown et al. 2020, arXiv:2005.14165, App. C; and Llama-2-style reports): normalise text (lowercase,
strip punctuation), build word n-grams (default n=8 here; GPT-3 used 13 on larger text), and flag an eval item as
contaminated if >= `threshold` of its n-grams occur in the corpus n-gram set. Limits: only finds near-verbatim overlap,
not paraphrases or translated copies; absence of overlap is not proof of cleanliness.

Self-validation (--selftest): we plant K real eval items inside a clean corpus and verify detection recall,
and check false positives on untouched items. Real check: ARC-Easy test questions vs. (a) ARC-Easy TRAIN split,
(b) WikiText-2 train text.
Run: python contamination_check.py
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import env  # noqa: E402

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> list[str]:
    """Turn text into a list of lower-case words without punctuation, so that small formatting
    differences do not matter.
    """
    return _WS.sub(" ", _PUNCT.sub(" ", text.lower())).strip().split()


def ngrams(tokens: list[str], n: int):
    """Return the set of all runs of n neighbouring words in a list of words."""
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def build_corpus_index(docs, n: int):
    """Collect every n-gram of every document into one big set, which lets us look up an n-gram
    quickly.
    """
    idx = set()
    for d in docs:
        idx |= ngrams(normalize(d), n)
    return idx


def overlap_fraction(item_text: str, corpus_index: set, n: int):
    """Return what fraction of an item's n-grams also appear in the corpus. Returns None when
    the item is too short to judge.
    """
    g = ngrams(normalize(item_text), n)
    if not g:
        return None  # too short to judge
    return len(g & corpus_index) / len(g)


def contamination_report(items, corpus_index, n: int, threshold: float = 0.5):
    """Count how many items share at least the threshold fraction of their n-grams with the
    corpus, and return the summary.
    """
    fr = [overlap_fraction(t, corpus_index, n) for t in items]
    judged = [f for f in fr if f is not None]
    flagged = [i for i, f in enumerate(fr) if f is not None and f >= threshold]
    return {"n_items": len(items), "n_judged": len(judged), "n_too_short": len(items) - len(judged),
            "n_flagged": len(flagged), "flagged_pct": round(100 * len(flagged) / max(1, len(judged)), 2),
            "flagged_idx": flagged, "ngram": n, "threshold": threshold}


def item_text(it):
    """Join the question and the answer choices of a benchmark item into one piece of text.
    """
    return it["question"] + " " + " ".join(it["choices"])


def main():
    """Check the benchmark against two corpora, run the self-test with planted items, and save
    the results.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="n-gram size in words")
    ap.add_argument("--n-items", type=int, default=500)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "contamination.json"))
    a = ap.parse_args()
    from datasets import load_dataset
    arc_test = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    arc_train = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    items = [{"question": r["question"], "choices": r["choices"]["text"]} for r in arc_test.select(range(a.n_items))]
    texts = [item_text(i) for i in items]
    res = {"environment": env.environment_info(), "ngram": a.n}

    # (a) real: ARC test vs ARC train
    idx_train = build_corpus_index([item_text({"question": r["question"], "choices": r["choices"]["text"]}) for r in arc_train], a.n)
    res["arc_test_vs_arc_train"] = contamination_report(texts, idx_train, a.n)
    # (b) real: ARC test vs WikiText-2 train
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    wt_text = list(wt["text"])
    idx_wt = build_corpus_index(wt_text, a.n)
    res["arc_test_vs_wikitext2_train"] = contamination_report(texts, idx_wt, a.n)
    # (c) self-test: plant eval items in a clean corpus (wikitext) after corrupting a fraction of their words, measure
    #     recall (flagged / planted) as a function of the corruption rate, and the false-positive rate on un-planted items.
    import random as _random
    rng = _random.Random(0)
    planted = list(range(0, len(texts), 2))

    def corrupt(t, rate):
        """Replace a given fraction of the words of a text by nonsense words, to imitate a
        changed copy of the text.
        """
        w = t.split()
        for j in rng.sample(range(len(w)), k=int(round(rate * len(w)))):
            w[j] = "zzq" + str(rng.randint(0, 9999))
        return " ".join(w)

    selftest_by_edit = {}
    for rate in (0.0, 0.05, 0.1, 0.2, 0.3):
        docs = wt_text + [corrupt(texts[i], rate) for i in planted]
        idx_p = build_corpus_index(docs, a.n)
        rep_p = contamination_report(texts, idx_p, a.n)
        fl = set(rep_p["flagged_idx"])
        jp = [i for i in planted if overlap_fraction(texts[i], idx_p, a.n) is not None]
        jc = [i for i in range(len(texts)) if i not in set(planted) and overlap_fraction(texts[i], idx_p, a.n) is not None]
        selftest_by_edit[str(rate)] = {
            "planted_items_judgeable": len(jp), "recall": round(sum(i in fl for i in jp) / max(1, len(jp)), 4),
            "clean_items_judgeable": len(jc), "false_positive_rate": round(sum(i in fl for i in jc) / max(1, len(jc)), 4)}
    res["selftest_by_word_corruption_rate"] = selftest_by_edit
    for k, v in res.items():
        if isinstance(v, dict) and "flagged_idx" in v:
            v["flagged_idx"] = v["flagged_idx"][:20]
    print(res)
    env.save_json(res, a.out)


if __name__ == "__main__":
    main()
