"""These tests check area 8's logit lens and activation patching on a tiny random model with a fake
tokenizer.
"""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import logit_lens as ll


class Tok:
    """Minimal char-level tokenizer stub: same-length clean/corrupt prompts."""
    def __call__(self, text, return_tensors=None):
        """Turn a text into token ids, returned in the same shape as a real tokenizer would.
        """
        ids = [ord(c) % 90 + 1 for c in text.lstrip()]  # lstrip: answers are passed as ' ' + word; keep the first token word-specific
        class O: pass
        o = O(); o.input_ids = torch.tensor([ids]) if return_tensors else ids  # HF returns a flat list for a single string
        return o

    def decode(self, ids): return "".join(chr(i + 31) for i in ids)


def tiny():
    """Build a very small random Llama-style model."""
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64)
    return LlamaForCausalLM(cfg).eval()


def test_last_layer_lens_equals_model_logits():
    """Check that the logit lens at the last layer equals the model's own output."""
    m = tiny()
    ids = torch.randint(0, 100, (1, 7))
    hs, logits = ll.residual_stream(m, ids)
    assert len(hs) == m.config.num_hidden_layers + 1
    assert torch.allclose(ll.lens_logits(m, hs[-1]), logits, atol=1e-5)


def test_hf_hidden_states_last_is_post_norm():
    """Documents why we use hooks: HF's last hidden state is already normalised (lens would normalise twice)."""
    m = tiny()
    ids = torch.randint(0, 100, (1, 7))
    hs_hf = m(ids, output_hidden_states=True).hidden_states
    hs, _ = ll.residual_stream(m, ids)
    assert torch.allclose(hs_hf[-2], hs[-2], atol=1e-6)
    assert torch.allclose(hs_hf[-1], m.model.norm(hs[-1]), atol=1e-5)


def test_patching_all_positions_all_layers_reproduces_clean():
    """Check that patching everything makes the corrupted run identical to the clean run."""
    m = tiny()
    tok = Tok()
    out = ll.patching_sweep(m, tok, "The capital of Fr", "The capital of It", "a", "b")
    assert out["identity_check_abs_error"] < 1e-4
    # patching the last position at the last layer should recover (almost) everything: it directly sets the final residual
    rec = out["recovery[layer][pos]"]
    assert rec[-1][-1] == pytest.approx(1.0, abs=1e-3)
    # patching a position that precedes any difference in the prompt changes nothing
    assert abs(rec[0][0]) < 1e-3
