"""These tests check area 3's ZeRO memory calculator against the numbers printed in the original
ZeRO paper.
"""
import pytest

import zero_memory as z


def test_matches_zero_paper_figure1():
    """ZeRO paper (arXiv:1910.02054) Fig. 1: Psi=7.5B, N=64, K=12 -> 120GB, 31.4GB, 16.6GB, 1.9GB per GPU."""
    psi, n = 7.5e9, 64
    assert z.zero_model_state_gb(psi, n, 0) == pytest.approx(120.0)
    assert z.zero_model_state_gb(psi, n, 1) == pytest.approx(31.4, abs=0.05)
    assert z.zero_model_state_gb(psi, n, 2) == pytest.approx(16.6, abs=0.05)
    assert z.zero_model_state_gb(psi, n, 3) == pytest.approx(1.9, abs=0.05)


def test_monotone_and_single_gpu_equal():
    """Check that each ZeRO stage never needs more memory than the previous one and that one
    GPU gives equal numbers.
    """
    for st in (1, 2, 3):
        assert z.zero_model_state_gb(1e9, 8, st) < z.zero_model_state_gb(1e9, 8, st - 1)
        assert z.zero_model_state_gb(1e9, 1, st) == z.zero_model_state_gb(1e9, 1, 0)


def test_invalid():
    """Check that invalid inputs raise an error."""
    with pytest.raises(ValueError):
        z.zero_model_state_gb(1e9, 8, 4)
