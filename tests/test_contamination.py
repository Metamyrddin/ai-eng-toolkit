"""These tests check the contamination detector of area 5: planted copies must be found and clean
text must not be flagged.
"""
import contamination_check as cc


def test_normalize_and_ngrams():
    """Check the text cleaning and the n-gram function on small examples."""
    assert cc.normalize("Hello,  World!! It's 5.") == ["hello", "world", "it", "s", "5"]
    assert cc.ngrams(["a", "b", "c", "d"], 2) == {("a", "b"), ("b", "c"), ("c", "d")}
    assert cc.ngrams(["a"], 2) == set()


def test_planted_detected_clean_not():
    """Check that a planted copy is flagged and an unrelated text is not."""
    corpus = ["the quick brown fox jumps over the lazy dog near the quiet river bank today",
              "completely different text about astronomy and the orbit of distant planets around stars"]
    idx = cc.build_corpus_index(corpus, 5)
    leaked = "Quick brown fox jumps over the lazy dog near the quiet river bank!"
    clean = "an unrelated sentence about cooking pasta with tomatoes and fresh basil leaves"
    assert cc.overlap_fraction(leaked, idx, 5) >= 0.9
    assert cc.overlap_fraction(clean, idx, 5) == 0.0
    assert cc.overlap_fraction("too short", idx, 5) is None
    rep = cc.contamination_report([leaked, clean, "too short"], idx, 5, 0.5)
    assert rep["flagged_idx"] == [0] and rep["n_too_short"] == 1


def test_partial_overlap_below_threshold():
    """Check that a text that only half overlaps gets a fraction below the threshold."""
    idx = cc.build_corpus_index(["alpha beta gamma delta epsilon zeta eta theta"], 4)
    half = "alpha beta gamma delta epsilon one two three four five"
    f = cc.overlap_fraction(half, idx, 4)
    assert 0 < f < 0.5
