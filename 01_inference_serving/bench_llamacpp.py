"""This file tests a model that was shrunk with quantization (stored with fewer bits per number)
and run by llama.cpp. For each shrink level it measures speed and how much worse the predictions
get. It is part of area 1 and shows the trade-off between size and quality.

More technical notes:
llama.cpp (via llama-cpp-python) quantization benchmark on CPU.

For each GGUF quantization of Qwen2.5-0.5B-Instruct measure:
  file size, prompt-processing speed (prefill tok/s), TTFT, decode tok/s, and perplexity on a
  WikiText-2 excerpt (quality cost of quantization). Optionally the HF fp32 model as reference.

Run:  python bench_llamacpp.py --quants q8_0 q4_k_m --repeats 3
Everything printed is measured; results are saved to results/llamacpp_quant.json.
Perplexity here is a *sanity-check metric* on a ~2k-token excerpt, not a benchmark score.
"""
import argparse
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

from common import env  # noqa: E402
from common.data import wikitext_text  # noqa: E402

PROMPT_TEXT = ("Explain, in simple words, how a transformer language model generates text one token at a time. " * 40)


def load(path, n_ctx=1024, n_threads=None, logits_all=False, flash_attn=False):
    """Open a GGUF model file with llama.cpp and return the model object."""
    from llama_cpp import Llama
    return Llama(model_path=path, n_ctx=n_ctx, n_threads=n_threads or os.cpu_count(), logits_all=logits_all,
                 flash_attn=flash_attn, verbose=False, seed=0)


def speed(llm, repeats=3, gen_tokens=64):
    """Median prefill tok/s, TTFT (ms) and decode tok/s. Greedy decoding (temperature=0)."""
    ptoks = llm.tokenize(PROMPT_TEXT.encode())[:384]
    prompt_tokens = len(ptoks)
    pre, ttft, dec = [], [], []
    for r in range(repeats + 1):  # first iteration = warm-up
        llm.reset()
        t0 = time.perf_counter()
        first = None
        n = 0
        for _ in llm.create_completion(ptoks, max_tokens=gen_tokens, temperature=0.0, stream=True):
            now = time.perf_counter()
            if first is None:
                first = now - t0
            n += 1
        total = time.perf_counter() - t0
        if r == 0:
            continue
        pre.append(prompt_tokens / first)  # first token arrives after full prefill (+1 sampled token)
        ttft.append(first * 1000)
        if n > 1:
            dec.append((n - 1) / (total - first))
    return {"prompt_tokens": prompt_tokens,
            "prefill_tok_s_median": round(statistics.median(pre), 1),
            "ttft_ms_median": round(statistics.median(ttft), 1),
            "decode_tok_s_median": round(statistics.median(dec), 1),
            "decode_tok_s_all": [round(x, 1) for x in dec], "repeats": repeats}


def hf_token_ids(text):
    """Tokenize with the HF tokenizer so GGUF and HF models score the *identical* token sequence."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(env.QWEN_05B_INSTRUCT)(text, add_special_tokens=False).input_ids


def perplexity_gguf(path, toks, n_ctx=512, max_chunks=4):
    """Compute the perplexity of a GGUF model on a list of token ids. Lower perplexity means
    the model predicts the text better.
    """
    llm = load(path, n_ctx=n_ctx, logits_all=True)
    nll, cnt = 0.0, 0
    for c in range(max_chunks):
        chunk = toks[c * n_ctx:(c + 1) * n_ctx]
        if len(chunk) < 2:
            break
        llm.reset()
        llm.eval(chunk)
        logits = np.asarray(llm.scores[: len(chunk)], dtype=np.float64)
        m = logits.max(-1, keepdims=True)
        logp = logits - (m + np.log(np.exp(logits - m).sum(-1, keepdims=True)))
        tgt = np.asarray(chunk[1:])
        nll += -logp[np.arange(len(tgt)), tgt].sum()
        cnt += len(tgt)
    del llm
    return {"ppl": round(math.exp(nll / cnt), 3), "eval_tokens": cnt, "n_ctx": n_ctx}


def perplexity_hf(model_name, toks, n_ctx=512, max_chunks=4):
    """Compute the same perplexity with the original Hugging Face model, to use as the
    reference answer.
    """
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
    nll, cnt = 0.0, 0
    with torch.no_grad():
        for c in range(max_chunks):
            chunk = toks[c * n_ctx:(c + 1) * n_ctx]
            if len(chunk) < 2:
                break
            ids = torch.tensor([chunk])
            lp = torch.log_softmax(model(ids).logits[0, :-1].double(), -1)
            nll += -lp[torch.arange(len(chunk) - 1), ids[0, 1:]].sum().item()
            cnt += len(chunk) - 1
    return {"ppl": round(math.exp(nll / cnt), 3), "eval_tokens": cnt, "n_ctx": n_ctx}


def main():
    """Run the speed and perplexity tests for every quantization level, print a table and save
    the numbers as a JSON file.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--quants", nargs="+", default=["q2_k", "q4_k_m", "q5_k_m", "q8_0", "fp16"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--no-hf-ref", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "llamacpp_quant.json"))
    a = ap.parse_args()
    text = wikitext_text(20000)
    toks = hf_token_ids(text)
    rows = []
    for q in a.quants:
        path = env.gguf_path(q)
        llm = load(path, n_threads=a.threads)
        row = {"quant": q, "file_mb": round(os.path.getsize(path) / 2**20, 1)}
        row.update(speed(llm, a.repeats))
        del llm
        row["perplexity"] = perplexity_gguf(path, toks)
        rows.append(row)
        print(row, flush=True)
    res = {"environment": env.environment_info(), "model": "Qwen2.5-0.5B-Instruct GGUF (official Qwen repo)",
           "threads": a.threads or os.cpu_count(), "rows": rows}
    if not a.no_hf_ref:
        res["hf_fp32_reference"] = perplexity_hf(env.QWEN_05B_INSTRUCT, toks)
        print("HF fp32 reference:", res["hf_fp32_reference"])
    env.save_json(res, a.out)


if __name__ == "__main__":
    main()
