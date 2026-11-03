"""This file runs speculative decoding with two real models, a small one that guesses and a larger
one that checks. It reports how often the guesses are accepted and whether the process really
got faster. It is part of area 1.

More technical notes:
Speculative decoding on real models: acceptance statistics (own implementation) and wall-clock
(HF `assistant_model`, the production implementation in transformers).

Target: SmolLM2-360M-Instruct   Draft: SmolLM2-135M-Instruct  (same tokenizer family).
Run: python speculative_demo.py [--new-tokens 48] [--gammas 1 2 4 8]
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from common import env  # noqa: E402
from speculative import baseline_generate, hf_prob_fn, speculative_generate  # noqa: E402

PROMPTS = ["The three laws of motion were formulated by", "To make a cup of tea, first you",
           "In machine learning, overfitting happens when", "def fibonacci(n):\n    "]


def main():
    """Compare speculative decoding with normal decoding on real models, check that the outputs
    match, and save the results.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=env.SMOL_360M_INSTRUCT)
    ap.add_argument("--draft", default=env.SMOL_135M_INSTRUCT)
    ap.add_argument("--new-tokens", type=int, default=48)
    ap.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "speculative.json"))
    a = ap.parse_args()
    env.seed_everything(0)
    tok = AutoTokenizer.from_pretrained(a.target)
    target = AutoModelForCausalLM.from_pretrained(a.target, dtype=torch.float32).eval()
    draft = AutoModelForCausalLM.from_pretrained(a.draft, dtype=torch.float32).eval()
    assert tok.get_vocab() == AutoTokenizer.from_pretrained(a.draft).get_vocab(), "tokenizers differ"

    # ---- (1) own implementation, greedy (T=0): output must equal target-only greedy ----
    res = {"environment": env.environment_info(), "target": a.target, "draft": a.draft,
           "new_tokens": a.new_tokens, "own_impl_greedy": [], "own_impl_sampling_T1": []}
    tf0, df0 = hf_prob_fn(target, 0.0), hf_prob_fn(draft, 0.0)
    for gamma in a.gammas:
        eq, accs, tpc = [], [], []
        for pr in PROMPTS:
            ids = tok(pr, return_tensors="pt").input_ids[0]
            ref = baseline_generate(ids, tf0, a.new_tokens)
            out, st = speculative_generate(ids, df0, tf0, a.new_tokens, gamma)
            eq.append(bool(torch.equal(ref, out))); accs.append(st.acceptance_rate); tpc.append(st.tokens_per_target_call)
        row = {"gamma": gamma, "all_outputs_equal_target_greedy": all(eq),
               "acceptance_rate_mean": round(statistics.mean(accs), 3),
               "tokens_per_target_call_mean": round(statistics.mean(tpc), 3), "n_prompts": len(PROMPTS)}
        res["own_impl_greedy"].append(row); print("greedy", row, flush=True)
    # sampling at T=1 (cannot compare token-by-token; acceptance stats only)
    tf1, df1 = hf_prob_fn(target, 1.0), hf_prob_fn(draft, 1.0)
    for gamma in a.gammas:
        accs, tpc = [], []
        for i, pr in enumerate(PROMPTS):
            ids = tok(pr, return_tensors="pt").input_ids[0]
            _, st = speculative_generate(ids, df1, tf1, a.new_tokens, gamma, seed=i)
            accs.append(st.acceptance_rate); tpc.append(st.tokens_per_target_call)
        row = {"gamma": gamma, "acceptance_rate_mean": round(statistics.mean(accs), 3),
               "tokens_per_target_call_mean": round(statistics.mean(tpc), 3)}
        res["own_impl_sampling_T1"].append(row); print("T=1", row, flush=True)

    # ---- (2) wall-clock with HF assisted generation (production implementation) ----
    def run(assist, ids):
        """Generate text once, with or without the helper (draft) model, and return the time
        taken and the generated tokens.
        """
        t0 = time.perf_counter()
        out = target.generate(ids, max_new_tokens=a.new_tokens, min_new_tokens=a.new_tokens, do_sample=False,
                              assistant_model=draft if assist else None, pad_token_id=tok.eos_token_id)
        return time.perf_counter() - t0, out
    ids = tok(PROMPTS[0], return_tensors="pt").input_ids
    run(False, ids); run(True, ids)  # warm-up
    base_t, spec_t, same = [], [], True
    for pr in PROMPTS:
        ids = tok(pr, return_tensors="pt").input_ids
        for _ in range(a.repeats):
            tb, ob = run(False, ids); ts, os_ = run(True, ids)
            base_t.append(tb); spec_t.append(ts)
            same &= bool(torch.equal(ob, os_))
    res["hf_assisted_generation"] = {
        "baseline_s_median": round(statistics.median(base_t), 3), "assisted_s_median": round(statistics.median(spec_t), 3),
        "speedup_median_ratio": round(statistics.median(base_t) / statistics.median(spec_t), 2),
        "outputs_identical_to_baseline_greedy": same, "n_runs": len(base_t),
        "note": "CPU, fp32, 8 threads, batch 1. A speedup <1 means the draft overhead exceeded savings on this hardware."}
    print(res["hf_assisted_generation"])
    env.save_json(res, a.out)


if __name__ == "__main__":
    main()
