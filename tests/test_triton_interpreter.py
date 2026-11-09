"""These tests run area 2's Triton kernels in interpreter mode on awkward sizes and compare them
with PyTorch. On native Windows the whole file is skipped, because Triton has no normal installer.
"""
import os
import sys

os.environ["TRITON_INTERPRET"] = "1"  # before triton is imported
import pytest  # noqa: E402
import torch  # noqa: E402

# A marker would still import Triton while collecting tests, and that import fails on Windows.
# This skip happens first, so the file is reported as skipped with a clear reason.
if sys.platform == "win32":
    pytest.skip("not supported on native Windows; use WSL2", allow_module_level=True)

pytest.importorskip("triton")
import triton_kernels as tk  # noqa: E402


def test_add_with_masking():
    """Check the vector addition kernel with a length that does not fit the block size."""
    x, y = torch.randn(1000), torch.randn(1000)
    assert torch.equal(tk.add(x, y), x + y)


def test_softmax_rows():
    """Check the softmax kernel against PyTorch."""
    s = torch.randn(5, 37) * 4
    assert torch.allclose(tk.softmax(s), torch.softmax(s, -1), atol=1e-6)
    assert torch.allclose(tk.softmax(s).sum(-1), torch.ones(5), atol=1e-6)


def test_matmul_odd_shapes():
    """Check the matrix multiplication kernel with sizes that do not fit the tiles."""
    a, b = torch.randn(21, 35), torch.randn(35, 19)
    assert torch.allclose(tk.matmul(a, b), a @ b, atol=1e-4)
