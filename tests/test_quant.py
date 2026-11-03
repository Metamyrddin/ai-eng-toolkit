"""These tests check area 1's quantization code: the rounding error must stay small, more bits must
mean less error, and bad inputs must be rejected.
"""
import pytest
import torch

import quant_from_scratch as q


def test_error_bounded_by_half_step():
    """Check that rounding never changes a weight by more than half of a quantization step.
    """
    torch.manual_seed(0)
    w = torch.randn(16, 64)
    for bits in (8, 4, 3):
        wq = q.quantize_dequantize(w, bits, 0)
        step = w.abs().amax(-1, keepdim=True) / (2 ** (bits - 1) - 1)
        assert ((w - wq).abs() <= step / 2 + 1e-6).all()


def test_more_bits_less_error_and_smaller_groups_less_error():
    """Check that more bits and smaller groups both reduce the error."""
    torch.manual_seed(1)
    w = torch.randn(32, 128)
    e = lambda b, g: (w - q.quantize_dequantize(w, b, g)).pow(2).mean().item()  # noqa: E731
    assert e(8, 0) < e(4, 0) < e(3, 0) < e(2, 0)
    assert e(4, 32) < e(4, 0)


def test_idempotent_and_zero_rows_safe():
    """Check that quantizing twice changes nothing and that rows of zeros stay zero."""
    w = torch.randn(4, 32)
    w[1] = 0
    wq = q.quantize_dequantize(w, 4, 16)
    assert torch.allclose(q.quantize_dequantize(wq, 4, 16), wq, atol=1e-6)
    assert torch.isfinite(wq).all() and (wq[1] == 0).all()


def test_group_must_divide():
    """Check that a group size that does not divide the row length is rejected."""
    with pytest.raises(AssertionError):
        q.quantize_dequantize(torch.randn(4, 30), 4, 16)


def test_bits_per_weight():
    """Check the bits-per-weight formula with known values."""
    assert q.bits_per_weight(4, 32, 64) == pytest.approx(4.5)
    assert q.bits_per_weight(8, 0, 64) == pytest.approx(8.25)
