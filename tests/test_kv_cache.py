"""These tests check area 1's KV cache code on a tiny random model: the cache must not change the
output, and the memory formula must match.
"""
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

import kv_cache_demo as kv
from common.hf_compat import cache_drop_last, cache_seq_len


def tiny_model():
    """Build a very small random Llama-style model that is quick to run."""
    cfg = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def test_cache_equals_no_cache_greedy():
    """Check that greedy generation gives identical tokens with and without the cache."""
    m = tiny_model()
    ids = torch.randint(0, 97, (1, 6))
    a, _ = kv.greedy_no_cache(m, ids, 12)
    b, _, _ = kv.greedy_with_cache(m, ids, 12)
    assert torch.equal(a, b)


def test_kv_formula_matches_measured_bytes():
    """Check that the memory formula equals the real size of the cache."""
    m = tiny_model()
    ids = torch.randint(0, 97, (1, 9))
    cache = DynamicCache()
    with torch.no_grad():
        m(ids, past_key_values=cache, use_cache=True)
    assert kv.cache_nbytes(cache) == kv.kv_formula_bytes(m.config, 9, bytes_per_elem=4)


def test_cache_drop_last():
    """Check that removing the last cache entries shortens the cache correctly."""
    m = tiny_model()
    cache = DynamicCache()
    with torch.no_grad():
        m(torch.randint(0, 97, (1, 10)), past_key_values=cache, use_cache=True)
    assert cache_seq_len(cache) == 10
    cache_drop_last(cache, 3)
    assert cache_seq_len(cache) == 7
