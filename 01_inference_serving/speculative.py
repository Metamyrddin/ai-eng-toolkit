"""This file contains our own version of speculative decoding. A small fast model guesses several
next words, and the big model checks all the guesses in one step. The code is written so that
the final text has exactly the same probabilities as the big model alone. It is part of area 1
and is used by the demo and the tests.

More technical notes:
Speculative decoding from scratch (educational, cache-free) + exactness checks.

Algorithm = Algorithm 1 of Leviathan, Kalman & Matias 2023 (arXiv:2211.17192), equivalent to the
speculative sampling of Chen et al. 2023 (arXiv:2302.01318):
    draft model q proposes gamma tokens;  target model p scores all of them in ONE forward pass;
    token x_i is accepted with prob min(1, p_i(x_i)/q_i(x_i));
    on the first rejection, resample from norm(max(0, p_i - q_i)) and stop;
    if all gamma accepted, sample one bonus token from p_{gamma+1}.
The output distribution is *exactly* the target model's distribution (proved in the papers; we also
test it empirically in tests/test_speculative.py).

This implementation recomputes prefixes (no KV cache) to stay short and obviously correct. Therefore its
wall-clock time says nothing about real speedups; the meaningful measured quantities are
    acceptance rate  and  tokens produced per target forward pass.
Use HF `assistant_model=` (see speculative_demo.py) for real wall-clock comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

# A "model" is a function: 1-D LongTensor of token ids -> [len, V] next-token PROBABILITIES for every position.
ProbFn = Callable[[torch.Tensor], torch.Tensor]


def adjust_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Logits -> probabilities. temperature == 0 -> one-hot argmax (greedy)."""
    if temperature == 0:
        out = torch.zeros_like(logits, dtype=torch.float64)
        out.scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
        return out
    return torch.softmax(logits.double() / temperature, dim=-1)


@dataclass
class SpecStats:
    """A small counter that records how many tokens the draft model proposed, how many were
    accepted, and how many calls each model made.
    """
    target_calls: int = 0
    draft_calls: int = 0
    proposed: int = 0
    accepted: int = 0
    tokens_out: int = 0
    per_round_accepted: list = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        """Return the fraction of proposed tokens that the big model accepted."""
        return self.accepted / max(1, self.proposed)

    @property
    def tokens_per_target_call(self) -> float:
        """Return how many tokens were produced for each call of the big model. More than one
        means speculative decoding is helping.
        """
        return self.tokens_out / max(1, self.target_calls)


def speculative_round(prefix: torch.Tensor, draft_fn: ProbFn, target_fn: ProbFn, gamma: int,
                      gen: torch.Generator, stats: SpecStats) -> torch.Tensor:
    """One draft+verify round. Returns the 1..gamma+1 new tokens appended to the sequence."""
    seq = prefix
    q_list, drafted = [], []
    for _ in range(gamma):
        q = draft_fn(seq)[-1]
        stats.draft_calls += 1
        x = torch.multinomial(q.float(), 1, generator=gen)
        q_list.append(q); drafted.append(x)
        seq = torch.cat([seq, x])
    p_all = target_fn(seq)  # ONE target forward over prefix + gamma drafted tokens
    stats.target_calls += 1
    stats.proposed += gamma
    n_prefix = prefix.shape[0]
    out = []
    for i in range(gamma):
        p = p_all[n_prefix - 1 + i]  # target distribution for the position of drafted token i
        q, x = q_list[i], int(drafted[i])
        r = torch.rand((), generator=gen, dtype=torch.float64)
        if r < torch.clamp(p[x] / q[x].clamp_min(1e-300), max=1.0):
            out.append(x); stats.accepted += 1
        else:
            resid = torch.clamp(p - q, min=0)
            resid = resid / resid.sum()
            out.append(int(torch.multinomial(resid.float(), 1, generator=gen)))
            stats.per_round_accepted.append(i)
            stats.tokens_out += len(out)
            return torch.tensor(out, dtype=torch.long)
    # all accepted -> bonus token from target at the last position
    out.append(int(torch.multinomial(p_all[n_prefix - 1 + gamma].float(), 1, generator=gen)))
    stats.per_round_accepted.append(gamma)
    stats.tokens_out += len(out)
    return torch.tensor(out, dtype=torch.long)


def speculative_generate(prefix: torch.Tensor, draft_fn: ProbFn, target_fn: ProbFn, n_new: int, gamma: int = 4,
                         seed: int = 0):
    """Generate exactly n_new tokens. Returns (full sequence tensor, SpecStats)."""
    gen = torch.Generator().manual_seed(seed)
    stats = SpecStats()
    seq = prefix.clone()
    while seq.shape[0] - prefix.shape[0] < n_new:
        new = speculative_round(seq, draft_fn, target_fn, gamma, gen, stats)
        seq = torch.cat([seq, new])
    extra = seq.shape[0] - prefix.shape[0] - n_new
    if extra > 0:  # overshoot: trim (distribution-preserving since we only truncate a valid sample)
        seq = seq[:-extra]
    return seq, stats


def baseline_generate(prefix: torch.Tensor, target_fn: ProbFn, n_new: int, seed: int = 0) -> torch.Tensor:
    """Plain autoregressive sampling from the target (reference for exactness tests)."""
    gen = torch.Generator().manual_seed(seed)
    seq = prefix.clone()
    for _ in range(n_new):
        p = target_fn(seq)[-1]
        seq = torch.cat([seq, torch.multinomial(p.float(), 1, generator=gen)])
    return seq


def hf_prob_fn(model, temperature: float) -> ProbFn:
    """Wrap a HF causal LM as a ProbFn (full recompute, batch size 1)."""
    @torch.no_grad()
    def fn(ids: torch.Tensor) -> torch.Tensor:
        """Run the model on a list of token ids and return the probability of each possible
        next token at every position.
        """
        logits = model(ids.unsqueeze(0), use_cache=False).logits[0]
        return adjust_probs(logits, temperature)
    return fn
