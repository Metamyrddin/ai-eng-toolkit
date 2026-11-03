"""This file looks inside a small model. The logit lens shows what the model would guess after each
layer, and activation patching finds which layers and positions matter for an answer. It is part
of area 8.

More technical notes:
Logit lens + activation patching on a tiny open model (SmolLM2-135M by default; any Llama-style HF model works).

LOGIT LENS (nostalgebraist 2020, https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens):
decode the residual stream after layer l with the model's final norm + unembedding:   logits_l = lm_head( norm( h_l ) ).
It shows how the next-token prediction is refined across depth. Caveat (Belrose et al. 2023, "Eliciting Latent Predictions with
the Tuned Lens", arXiv:2303.08112): early layers are not trained to be decodable this way, so the lens can be misleading there.
CORRECTNESS CHECK: for the last layer, our lens logits must equal the model's own logits (tested, max abs diff reported).

ACTIVATION PATCHING (causal tracing; Meng et al. 2022 ROME arXiv:2202.05262, Wang et al. 2022 arXiv:2211.00593; overview:
Heimersheim & Nanda 2024, arXiv:2404.15255): run a CLEAN and a CORRUPTED prompt (same length). Replace the residual stream of
the corrupted run at (layer, position) with the clean one and measure how much of the clean answer's logit-difference is recovered.
CORRECTNESS CHECK: patching ALL layers at ALL positions must reproduce the clean logits exactly (recovery = 1.0).

Run: python logit_lens.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from common import env  # noqa: E402


@torch.no_grad()
def residual_stream(model, ids):
    """Residual stream BEFORE the final norm at every depth, captured with forward hooks:
    [output of embedding, output of layer 0, ..., output of layer L-1]  (L+1 tensors, each [batch, seq, d]).
    (HF's output_hidden_states returns the last entry AFTER the final norm, which would be normalised twice by the lens.)"""
    store, hooks = [], []
    hooks.append(model.model.embed_tokens.register_forward_hook(lambda m, i, o: store.append(o.detach().clone())))
    for layer in model.model.layers:
        hooks.append(layer.register_forward_hook(lambda m, i, o: store.append((o[0] if isinstance(o, tuple) else o).detach().clone())))
    try:
        logits = model(ids).logits
    finally:
        for h in hooks:
            h.remove()
    return store, logits


def lens_logits(model, h):
    """Apply final norm + lm_head to a residual-stream tensor [.., d]."""
    return model.lm_head(model.model.norm(h))


@torch.no_grad()
def logit_lens(model, tok, text, top_k=3):
    """Look at what the model would predict after every layer. For each layer we return the top
    guesses and how close they are to the final answer.
    """
    ids = tok(text, return_tensors="pt").input_ids
    hs, final_logits = residual_stream(model, ids)
    final_tok = final_logits[0, -1].argmax().item()
    final_probs = torch.softmax(final_logits[0, -1].float(), -1)
    rows = []
    for l, h in enumerate(hs):
        lg = lens_logits(model, h[0, -1]).float()
        p = torch.softmax(lg, -1)
        top = torch.topk(p, top_k)
        rank_of_final = int((lg > lg[final_tok]).sum().item()) + 1
        rows.append({"layer": l, "top": [(tok.decode([i]), round(v, 4)) for v, i in zip(top.values.tolist(), top.indices.tolist())],
                     "p_final_token": round(p[final_tok].item(), 4), "rank_of_final_token": rank_of_final,
                     "kl_to_final": round(float((final_probs * (final_probs.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum()), 4)})
    last_diff = float((lens_logits(model, hs[-1][0, -1]) - final_logits[0, -1]).abs().max())
    return {"text": text, "final_prediction": tok.decode([final_tok]), "layers": rows,
            "last_layer_lens_vs_model_logits_max_abs_diff": last_diff}


@torch.no_grad()
def patching_sweep(model, tok, clean, corrupt, answer_clean, answer_corrupt):
    """Residual-stream patching per (layer, position). Returns recovery matrix and the identity check."""
    ci, ki = tok(clean, return_tensors="pt").input_ids, tok(corrupt, return_tensors="pt").input_ids
    assert ci.shape == ki.shape, f"clean/corrupt must tokenise to the same length ({ci.shape} vs {ki.shape})"
    a, b = tok(" " + answer_clean).input_ids[0], tok(" " + answer_corrupt).input_ids[0]  # FIRST token of each answer
    layers = model.model.layers
    n_layers, n_pos = len(layers), ci.shape[1]
    clean_hs, _ = residual_stream(model, ci)  # clean_hs[l+1] = output of layer l (pre-final-norm for every l)

    def ld(logits):  # logit difference clean-answer minus corrupt-answer at the last position
        """Return the logit difference: the score of the clean answer minus the score of the
        corrupt answer at the last position.
        """
        return (logits[0, -1, a] - logits[0, -1, b]).item()

    base_clean, base_corrupt = ld(model(ci).logits), ld(model(ki).logits)

    def run_patched(patch):  # patch: dict {(layer, pos)}
        """Run the corrupted prompt, but copy the stored clean hidden states into the chosen
        (layer, position) places, and return the resulting scores.
        """
        hooks = []
        by_layer: dict = {}
        for (l, p) in patch:
            by_layer.setdefault(l, []).append(p)
        for l, ps in by_layer.items():
            def hook(mod, inp, out, l=l, ps=ps):
                """A hook that runs after a layer. It replaces the layer's output at the chosen
                positions with the stored clean values.
                """
                hs = out[0] if isinstance(out, tuple) else out
                hs = hs.clone()
                for p in ps:
                    hs[0, p] = clean_hs[l + 1][0, p]
                return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
            hooks.append(layers[l].register_forward_hook(hook))
        try:
            return ld(model(ki).logits)
        finally:
            for h in hooks:
                h.remove()

    full = run_patched({(l, p) for l in range(n_layers) for p in range(n_pos)})
    recovery = [[(run_patched({(l, p)}) - base_corrupt) / (base_clean - base_corrupt + 1e-9) for p in range(n_pos)] for l in range(n_layers)]
    return {"tokens": [tok.decode([t]) for t in ci[0]], "clean_logit_diff": base_clean, "corrupt_logit_diff": base_corrupt,
            "identity_check_patch_all_logit_diff": full, "identity_check_abs_error": abs(full - base_clean), "recovery[layer][pos]": recovery}


def main():
    """Run the logit lens and the activation patching experiments on the model and save the
    results.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=env.SMOL_135M)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "logit_lens.json"))
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    res = {"environment": env.environment_info(), "model": a.model, "lens": []}
    for text in ["The capital of France is", "The Eiffel Tower is located in the city of", "2 + 2 ="]:
        r = logit_lens(model, tok, text)
        res["lens"].append(r)
        print(f"\n{text!r} -> model predicts {r['final_prediction']!r}")
        for row in r["layers"][::3] + [r["layers"][-1]]:
            print(f"  layer {row['layer']:2d} top={row['top'][0]} rank_of_final={row['rank_of_final_token']:5d} KL={row['kl_to_final']}")
        print("  last-layer check:", {k: v for k, v in r.items() if "max_abs_diff" in k})
    res["patching"] = patching_sweep(model, tok, "The capital of France is", "The capital of Italy is", "Paris", "Rome")
    p = res["patching"]
    print("\npatching clean/corrupt logit diff:", round(p["clean_logit_diff"], 3), round(p["corrupt_logit_diff"], 3),
          "identity abs error:", p["identity_check_abs_error"])
    env.save_json(res, a.out)


if __name__ == "__main__":
    main()
