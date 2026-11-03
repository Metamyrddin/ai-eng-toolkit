"""These tests check that our tiled attention gives the same answers as the plain version and as
PyTorch's version, even for extreme numbers. They belong to area 2.
"""
import pytest
import torch
import torch.nn.functional as F

import flash_attention_from_scratch as fa


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("n,bq,bk", [(64, 16, 16), (100, 32, 24), (37, 64, 64), (130, 7, 50)])
def test_tiled_matches_naive_and_sdpa(causal, n, bq, bk):
    """Check that the tiled attention gives the same result as the plain version for several
    sizes.
    """
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, n, 16, dtype=torch.float64) for _ in range(3))
    ref = fa.naive_attention(q, k, v, causal)
    out = fa.flash_attention_tiled(q, k, v, causal, bq, bk)
    assert torch.allclose(out, ref, atol=1e-10)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v, is_causal=causal), atol=1e-9)


def test_extreme_scores_are_stable():
    """Check that huge attention scores do not produce infinite or not-a-number values."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 50, 8, dtype=torch.float32) * 30 for _ in range(3))  # huge logits: naive exp would overflow w/o max-subtraction
    out = fa.flash_attention_tiled(q, k, v, True, 16, 16)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, fa.naive_attention(q.double(), k.double(), v.double(), True).float(), atol=1e-3)
