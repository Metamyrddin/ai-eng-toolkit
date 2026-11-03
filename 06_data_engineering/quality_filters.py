"""This file contains simple rules that throw away low-quality text, such as spam or repeated
lines, plus a language-model score. It tests them on documents that we made bad on purpose. It
is part of area 6.

More technical notes:
Heuristic + model-based quality filtering for pre-training text, validated against planted bad documents.

Heuristics follow the rules of Gopher (Rae et al. 2021, arXiv:2112.11446, Appendix A "MassiveText quality filtering") and C4
(Raffel et al. 2020, arXiv:1910.10683): word-count bounds, mean word length, symbol-to-word ratio, fraction of lines
ending with terminal punctuation, fraction of repeated lines / repeated n-grams, presence of common stop-words.
Thresholds below are *our own defaults inspired by those papers*, not copies of their exact values - tune on your data.
Model-based signal: perplexity under a small LM (SmolLM2-135M) - the idea behind CCNet-style perplexity filtering
(Wenzek et al. 2020, arXiv:1911.00359, which used KenLM n-gram models).

Experiment (python quality_filters.py):
  * real documents: FineWeb sample (already filtered upstream, so we expect a HIGH pass rate)
  * planted bad documents: random symbols, heavy repetition, all-caps spam, keyword lists, tiny docs, with ground truth
  Report: heuristic pass rate on real docs, recall on each planted bad type, and ROC-AUC of LM perplexity for
  real-vs-garbage separation.
"""
import argparse
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

from common import env  # noqa: E402

STOP = {"the", "be", "to", "of", "and", "that", "have", "with", "in", "a", "is", "it", "for", "on", "as", "this"}
TERMINAL = tuple(".!?\"'”’")


def doc_stats(text: str) -> dict:
    """Measure simple facts about a document, such as the number of words, the average word
    length and how much of it is repeated.
    """
    words = text.split()
    lines = [l for l in text.split("\n") if l.strip()]
    n = max(1, len(words))
    sym = sum(w.count("#") + w.count("...") + w.count("…") for w in words)
    top = Counter(zip(words, words[1:], words[2:])).most_common(1)
    trigram_top_frac = (top[0][1] * 3 / n) if top else 0.0  # share of text covered by the most common word trigram
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    return {
        "n_words": len(words),
        "mean_word_len": sum(len(w) for w in words) / n,
        "symbol_word_ratio": sym / n,
        "frac_lines_terminal_punct": (sum(l.rstrip().endswith(TERMINAL) for l in lines) / len(lines)) if lines else 0.0,
        "frac_duplicate_lines": 1 - len(set(lines)) / len(lines) if lines else 0.0,
        "top_trigram_frac": trigram_top_frac,
        "frac_alpha_words": alpha_words / n,
        "n_stopwords_present": len(STOP & {w.lower().strip(string.punctuation) for w in words}),
        "frac_upper_words": sum(w.isupper() and len(w) > 1 for w in words) / n,
    }


DEFAULTS = dict(min_words=50, max_words=100000, mean_word_len=(3, 10), max_symbol_word_ratio=0.1,
                min_frac_lines_terminal=0.1, max_frac_duplicate_lines=0.3, max_top_trigram_frac=0.2,
                min_frac_alpha_words=0.8, min_stopwords=2, max_frac_upper=0.3)


def check(text: str, c=DEFAULTS):
    """Return (passes, [failed_rule_names])."""
    s = doc_stats(text)
    fails = []
    if not c["min_words"] <= s["n_words"] <= c["max_words"]: fails.append("word_count")
    if not c["mean_word_len"][0] <= s["mean_word_len"] <= c["mean_word_len"][1]: fails.append("mean_word_len")
    if s["symbol_word_ratio"] > c["max_symbol_word_ratio"]: fails.append("symbol_ratio")
    if s["frac_lines_terminal_punct"] < c["min_frac_lines_terminal"]: fails.append("terminal_punct")
    if s["frac_duplicate_lines"] > c["max_frac_duplicate_lines"]: fails.append("duplicate_lines")
    if s["top_trigram_frac"] > c["max_top_trigram_frac"]: fails.append("repeated_ngram")
    if s["frac_alpha_words"] < c["min_frac_alpha_words"]: fails.append("alpha_words")
    if s["n_stopwords_present"] < c["min_stopwords"]: fails.append("stopwords")
    if s["frac_upper_words"] > c["max_frac_upper"]: fails.append("all_caps")
    return not fails, fails


def plant_bad_docs(real_docs, n_each, rng):
    """Create several kinds of obviously bad documents (random symbols, repetition, shouting
    and so on), so that we know the right answer.
    """
    kinds = {}
    chars = string.ascii_letters + string.digits + "#@$%^&*~<>{}[]|\\/"
    kinds["random_symbols"] = [" ".join("".join(rng.choice(chars) for _ in range(rng.randint(3, 12))) for _ in range(150)) for _ in range(n_each)]
    kinds["repetition"] = [("Click here to buy now. " * 60).strip() for _ in range(n_each)]
    kinds["all_caps_spam"] = [" ".join(rng.choice(["BUY", "CHEAP", "FREE", "WINNER", "CLICK", "NOW", "BEST", "PRICE"]) for _ in range(120)) for _ in range(n_each)]
    kinds["keyword_list"] = [", ".join(rng.choice(["cheap", "pills", "casino", "loan", "watch", "replica", "bag", "online"]) + str(rng.randint(0, 99)) for _ in range(100)) for _ in range(n_each)]
    kinds["too_short"] = [" ".join(rng.choice(real_docs).split()[:rng.randint(3, 20)]) for _ in range(n_each)]
    kinds["shuffled_words"] = []  # real vocabulary but destroyed syntax: heuristics are BLIND to this, LM perplexity is not
    for _ in range(n_each):
        w = rng.choice(real_docs).split()[:200]; rng.shuffle(w); kinds["shuffled_words"].append(" ".join(w))
    return kinds


def lm_perplexities(texts, model_name, max_tokens=256, batch=8):
    """Return how surprised a small language model is by each text. Garbage text usually
    surprises it more.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name); tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=max_tokens)
        with torch.no_grad():
            logits = model(**enc).logits[:, :-1]
        tgt, m = enc.input_ids[:, 1:], enc.attention_mask[:, 1:].float()
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
        out += torch.exp((nll * m).sum(1) / m.sum(1).clamp_min(1)).tolist()
    return out


def auc(pos_scores, neg_scores):
    """P(score_pos > score_neg) (+0.5 ties) - the Mann-Whitney / ROC-AUC statistic."""
    p, n = np.asarray(pos_scores), np.asarray(neg_scores)
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def main():
    """Test the filters on real and planted documents, measure how well they separate them, and
    save the results.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=300)
    ap.add_argument("--n-bad-each", type=int, default=30)
    ap.add_argument("--model", default=env.SMOL_135M)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "quality.json"))
    a = ap.parse_args()
    from fetch_sample import load_sample
    real = [d["text"] for d in load_sample(a.n_real)][:a.n_real]
    rng = random.Random(a.seed)
    bad = plant_bad_docs(real, a.n_bad_each, rng)
    res = {"environment": env.environment_info(), "n_real": len(real), "n_bad_each": a.n_bad_each, "thresholds": DEFAULTS}
    passes = [check(t) for t in real]
    res["real_docs_heuristic_pass_rate"] = round(sum(p for p, _ in passes) / len(real), 4)
    res["real_docs_fail_reasons"] = dict(Counter(r for p, f in passes for r in f))
    res["planted_bad_recall_by_heuristics"] = {k: round(sum(not check(t)[0] for t in v) / len(v), 4) for k, v in bad.items()}
    ppl_real = lm_perplexities(real, a.model)
    res["lm_perplexity_real_median"] = round(float(np.median(ppl_real)), 1)
    res["lm_perplexity_auc_vs_real"] = {}
    for k, v in bad.items():
        if k == "too_short":
            continue  # perplexity of a handful of words is dominated by length, not quality
        ppl_bad = lm_perplexities(v, a.model)
        res["lm_perplexity_auc_vs_real"][k] = {"median_ppl_bad": round(float(np.median(ppl_bad)), 1), "auc": round(auc(ppl_bad, ppl_real), 4)}
    print(res)
    env.save_json(res, a.out)


if __name__ == "__main__":
    main()
