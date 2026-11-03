"""This file is the big-GPU version of the synthetic data generator in the same area. It has NOT
been run, because it needs a GPU and the vLLM library. It is part of area 4.

More technical notes:
UNTESTED (needs a GPU + `pip install vllm`). Rejection-sampling data generation at scale with vLLM.
Uses vLLM's offline `LLM.generate` / `SamplingParams(n=..., temperature=...)` API
(https://docs.vllm.ai/en/latest/serving/offline_inference/). Mirrors ../synthetic_data.py:rejection_sampling (tested on CPU):
few-shot prompt -> k samples -> keep one verified sample per prompt. The problem generator and verifier come from ../task.py.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import task  # noqa: E402
from synthetic_data import FEWSHOT  # noqa: E402


def main():
    """Use vLLM on a GPU to sample many answers per question, keep the verified ones and write
    them to a file (UNTESTED, needs a GPU).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="sft_rs_large.jsonl")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    test = task.make_split(200, seed=999)
    probs = task.make_split(a.n, seed=1, exclude={task.problem_key(p) for p in test})
    llm = LLM(model=a.model, dtype="bfloat16")
    sp = SamplingParams(n=a.k, temperature=0.7, top_p=0.95, max_tokens=6)
    outs = llm.generate([FEWSHOT + p["prompt"] for p in probs], sp)
    kept = 0
    with open(a.out, "w") as f:
        for p, o in zip(probs, outs):
            if any(task.parse_answer(c.text.split("\n")[0]) == p["answer"] for c in o.outputs):
                f.write(json.dumps({"prompt": p["prompt"], "completion": task.completion_for(p)}) + "\n"); kept += 1
    print(f"kept {kept} of {len(probs)} prompts")


if __name__ == "__main__":
    main()
