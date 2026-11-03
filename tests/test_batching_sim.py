"""These tests check the batching simulator of area 1: continuous batching must win, no tokens may
be lost, and the result must not change between runs.
"""
import batching_sim as bs


def test_continuous_beats_static_and_conserves_tokens():
    """Check that continuous batching is faster than static batching and that both produce the
    same number of tokens.
    """
    reqs = bs.make_requests(100, seed=0, arrival_rate=0.01)
    cost = lambda b: 15 + 1.0 * b  # noqa: E731
    ls, ms, ts = bs.simulate_static(reqs, 8, cost)
    lc, mc, tc = bs.simulate_continuous(reqs, 8, cost)
    assert ts == tc == sum(r["out_len"] for r in reqs)
    assert len(ls) == len(lc) == 100
    assert sum(lc) / 100 < sum(ls) / 100
    assert mc <= ms


def test_deterministic_given_seed():
    """Check that the same seed always creates the same requests."""
    a = bs.make_requests(10, 5, 0.01); b = bs.make_requests(10, 5, 0.01)
    assert a == b


def test_batch_size_one_equal_modes():
    """Check that with a batch size of one both scheduling styles behave the same."""
    reqs = bs.make_requests(20, 1, 0.001)
    cost = lambda b: 10.0  # noqa: E731
    ls, ms, _ = bs.simulate_static(reqs, 1, cost)
    lc, mc, _ = bs.simulate_continuous(reqs, 1, cost)
    assert abs(ms - mc) < 1e-6 and abs(sum(ls) - sum(lc)) < 1e-6
