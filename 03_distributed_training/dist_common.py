"""This file holds the pieces that all area 3 scripts share: the tiny language model, the data
batches and the start-up code for several processes. It is part of area 3.

More technical notes:
Shared pieces for the distributed-training demos (CPU / gloo, 2 processes).

Launch any script with either
    torchrun --nproc_per_node=2 SCRIPT.py          (Linux/macOS/Windows; reads RANK/WORLD_SIZE env vars)
    python launch_spawn.py SCRIPT.py --world 2     (our tiny launcher, no torchrun needed)
"""
import os
import sys
from pathlib import Path

# Some Windows builds of PyTorch refuse to start the gloo backend unless this is turned off.
# Linux and macOS do not need the switch, so it is set only on native Windows.
if sys.platform == "win32":
    os.environ.setdefault("USE_LIBUV", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch import nn  # noqa: E402


def init_dist(backend="gloo"):
    """Initialise the default process group from torchrun-style env vars."""
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size()


class TinyLM(nn.Module):
    """Minimal causal transformer LM (no dropout -> deterministic, so DDP == single-process exactly)."""
    def __init__(self, vocab=64, d=32, layers=2, heads=4, seq=16):
        """Create the layers of a very small language model: token and position embeddings, a
        few transformer layers and an output layer.
        """
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        """Turn a batch of token ids into a score for every possible next token at every
        position.
        """
        n = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(n, device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(n, device=x.device)
        return self.head(self.enc(h, mask=mask, is_causal=True))


def make_model(seed=0):
    """Build a new tiny model whose starting weights are fixed by the seed, so every process
    starts with identical weights.
    """
    torch.manual_seed(seed)
    return TinyLM()


def make_batches(steps, global_batch, vocab=64, seq=16, seed=123):
    """Deterministic synthetic data: next token = (token + 1) % vocab. Identical on every rank."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(steps):
        x = torch.randint(0, vocab, (global_batch, seq), generator=g)
        out.append((x, (x + 1) % vocab))
    return out


def lm_loss(model, x, y):
    """Compute the next-token prediction loss (cross-entropy) of the model on one batch."""
    logits = model(x)
    return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
