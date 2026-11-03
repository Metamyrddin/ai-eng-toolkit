"""These tests check that area 1's speculative decoding gives exactly the probabilities of the big
model, even with a bad helper model.

More technical notes:
Exactness of speculative sampling on toy models: the output distribution must equal the target's (Leviathan et al. 2023)."""
import numpy as np
import pytest
import torch

import speculative as sp

V = 5


def make_markov(seed, sharp):
    """Build a pretend language model whose next-word probabilities depend only on the previous
    word.
    """
    g = torch.Generator().manual_seed(seed)
    T = torch.softmax(torch.randn(V, V, generator=g) * sharp, -1).double()

    def fn(seq):  # next-token distribution for every position, conditioned on the previous token only
        """Return the next-word probabilities for every position of a sequence."""
        return T[seq]
    return fn


def empirical(seqs, pos):
    """Count how often each word appears at one position over many generated sequences, as
    fractions.
    """
    return np.bincount([int(s[pos]) for s in seqs], minlength=V) / len(seqs)


@pytest.mark.parametrize("gamma", [1, 3])
def test_distribution_matches_target(gamma):
    """Check that speculative decoding produces the same word probabilities as the big model.
    """
    target, draft = make_markov(1, 2.0), make_markov(2, 2.0)   # different models -> nontrivial accept/reject
    prefix = torch.tensor([0])
    n, new = 6000, 3
    spec = [sp.speculative_generate(prefix, draft, target, new, gamma, seed=i)[0][1:] for i in range(n)]
    base = [sp.baseline_generate(prefix, target, new, seed=10_000 + i)[1:] for i in range(n)]
    for pos in range(new):
        tv = 0.5 * np.abs(empirical(spec, pos) - empirical(base, pos)).sum()
        # TV distance between two empirical distributions of n=6000 samples over 5 outcomes: noise ~ 0.02; 0.05 is ~2.5x margin
        assert tv < 0.05, (gamma, pos, tv)


def test_draft_equals_target_accepts_everything():
    """Check that identical helper and big models give an acceptance rate of 100 percent."""
    m = make_markov(3, 2.0)
    _, st = sp.speculative_generate(torch.tensor([0]), m, m, 24, gamma=4, seed=0)
    assert st.acceptance_rate == 1.0
    assert st.tokens_per_target_call == pytest.approx(5.0)  # gamma accepted + 1 bonus every round


def test_bad_draft_is_still_exact_and_lower_acceptance():
    """Check that a useless helper model lowers the acceptance rate but does not change the
    output probabilities.
    """
    target = make_markov(1, 3.0)
    bad = lambda seq: torch.full((len(seq), V), 1.0 / V, dtype=torch.float64)  # uniform draft
    prefix = torch.tensor([0])
    spec = [sp.speculative_generate(prefix, bad, target, 2, 4, seed=i)[0][1:] for i in range(5000)]
    base = [sp.baseline_generate(prefix, target, 2, seed=9000 + i)[1:] for i in range(5000)]
    for pos in range(2):
        assert 0.5 * np.abs(empirical(spec, pos) - empirical(base, pos)).sum() < 0.06
    _, st = sp.speculative_generate(prefix, bad, target, 50, 4, seed=0)
    assert st.acceptance_rate < 1.0


def test_greedy_matches_target_greedy_exactly():
    """Check that with temperature zero the output is identical to the big model's greedy
    output.
    """
    target, draft = make_markov(1, 4.0), make_markov(2, 4.0)
    t0 = lambda s: sp.adjust_probs(torch.log(target(s)), 0.0)
    d0 = lambda s: sp.adjust_probs(torch.log(draft(s)), 0.0)
    prefix = torch.tensor([2])
    ref = sp.baseline_generate(prefix, t0, 20)
    for gamma in (1, 2, 5):
        out, _ = sp.speculative_generate(prefix, d0, t0, 20, gamma)
        assert torch.equal(ref, out)


def test_adjust_probs_temperature():
    """Check that temperature zero gives a one-hot answer and that other temperatures give
    probabilities that add up to one.
    """
    lg = torch.tensor([[1.0, 2.0, 3.0]])
    assert sp.adjust_probs(lg, 0.0).tolist() == [[0.0, 0.0, 1.0]]
    assert torch.allclose(sp.adjust_probs(lg, 1.0).sum(-1), torch.tensor([1.0], dtype=torch.float64))
