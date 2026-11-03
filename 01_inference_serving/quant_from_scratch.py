"""This file shrinks a real model by rounding every weight to fewer bits, using code that we wrote
ourselves. It then measures how much the model's predictions get worse. It is part of area 1 and
explains what quantization tools do inside.

More technical notes:
Weight-only round-to-nearest (RTN) quantization from scratch, measured on a real model.

For each bit-width and group size we fake-quantize ("quantize then dequantize") every nn.Linear weight
inside the transformer blocks, then measure
   * relative weight error  ||W - Wq||_F / ||W||_F
   * perplexity on a WikiText-2 excerpt (same 4x512-token protocol as bench_llamacpp.py)
   * ideal storage (bits per weight incl. one fp16 scale per group)
This is *simulated* quantization (weights are stored back in fp32), so it shows the QUALITY effect only;
real speed/memory gains need packed kernels (llama.cpp GGUF, bitsandbytes, GPTQ/AWQ kernels).
Embeddings / lm_head are left in full precision (common practice; SmolLM2 ties them).

Run:  python quant_from_scratch.py [--bits 8 4 3 2] [--groups 0 64 32]    (group 0 = per-output-channel)
"""
import argparse
import copy
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from common import env  # noqa: E402
from common.data import wikitext_text  # noqa: E402


def quantize_dequantize(w: torch.Tensor, bits: int, group_size: int = 0) -> torch.Tensor:
    """Symmetric absmax RTN. Scale per group of `group_size` input features (0 = whole row).

    q = clamp(round(w / s), -(2^(b-1)-1), 2^(b-1)-1),  s = max|w| / (2^(b-1)-1);  w_hat = q * s
    """
    out_f, in_f = w.shape
    g = in_f if group_size in (0, None) else group_size
    assert in_f % g == 0, f"in_features {in_f} not divisible by group {g}"
    wg = w.reshape(out_f, in_f // g, g)
    qmax = 2 ** (bits - 1) - 1
    scale = wg.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
    q = torch.clamp(torch.round(wg / scale), -qmax, qmax)
    return (q * scale).reshape(out_f, in_f)


def bits_per_weight(bits: int, group_size: int, in_f: int) -> float:
    """Return the real storage cost in bits per weight, including the small scale number that
    is stored for each group.
    """
    g = in_f if group_size == 0 else group_size
    return bits + 16 / g  # one fp16 scale per group


@torch.no_grad()
def quantize_model_(model, bits, group_size):
    """Round every weight matrix of the model to fewer bits, in place, and return statistics
    about the error.
    """
    num = den = 0.0
    n_params = 0
    bpw_total = 0.0
    for name, mod in model.model.layers.named_modules():
        if isinstance(mod, nn.Linear):
            w = mod.weight.data
            wq = quantize_dequantize(w, bits, group_size)
            num += (w - wq).pow(2).sum().item(); den += w.pow(2).sum().item()
            n_params += w.numel(); bpw_total += w.numel() * bits_per_weight(bits, group_size, w.shape[1])
            mod.weight.data = wq
    return math.sqrt(num / den), bpw_total / n_params


@torch.no_grad()
def ppl(model, toks, n_ctx=512, max_chunks=4):
    """Compute the perplexity of the model on a list of token ids. Lower perplexity means
    better predictions.
    """
    nll, cnt = 0.0, 0
    for c in range(max_chunks):
        chunk = toks[c * n_ctx:(c + 1) * n_ctx]
        ids = torch.tensor([chunk])
        lp = torch.log_softmax(model(ids).logits[0, :-1].double(), -1)
        nll += -lp[torch.arange(len(chunk) - 1), ids[0, 1:]].sum().item(); cnt += len(chunk) - 1
    return math.exp(nll / cnt)


def main():
    """Try every combination of bit width and group size, print a table and save the results as
    a JSON file.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=env.SMOL_135M)
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 4, 3, 2])
    ap.add_argument("--groups", type=int, nargs="+", default=[0, 64, 32])
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "quant_scratch.json"))
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    toks = tok(wikitext_text(20000), add_special_tokens=False).input_ids
    rows = [{"bits": 32, "group": None, "ppl": round(ppl(base, toks), 3), "rel_weight_err": 0.0, "bits_per_weight": 32.0}]
    print(rows[0], flush=True)
    for bits in a.bits:
        for g in a.groups:
            m = copy.deepcopy(base)
            err, bpw = quantize_model_(m, bits, g)
            rows.append({"bits": bits, "group": g, "ppl": round(ppl(m, toks), 3),
                         "rel_weight_err": round(err, 4), "bits_per_weight": round(bpw, 3)})
            print(rows[-1], flush=True)
            del m
    env.save_json({"environment": env.environment_info(), "model": a.model,
                   "protocol": "WikiText-2 raw test, first 20000 chars, 4 x 512 tokens", "rows": rows}, a.out)


if __name__ == "__main__":
    main()
