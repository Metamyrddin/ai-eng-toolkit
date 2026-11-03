"""GPU checks: one small, self-contained test per area that can run on a GPU (or, as a dry run, on the CPU).

How this file fits in the project:
    The rest of the toolkit was built and measured on a laptop-style CPU. This file is the bridge to free cloud GPUs
    such as Kaggle or Google Colab. The generated notebooks (see make_notebooks.py) call this file, and so does the
    Kaggle command-line script (kaggle_run.sh). Each check is short, so it fits easily inside free GPU time limits.

What a check does:
    Every check runs a little real work (for example, generating text, or training a few steps) on the chosen device,
    compares the answer with a known-correct answer, measures how long it took, and records PASS or FAIL.
    The results are saved as a JSON file in tools/gpu/results/ so that you can download them from the cloud.

Honest status:
    Every check was run on the CPU (--device cpu) to prove that the code works. No check has been run on a GPU yet,
    because the machine where this toolkit was built has no GPU. So GPU runs are UNTESTED until you run them, and
    the JSON file records which device was really used, so nobody can mistake a CPU dry run for a GPU result.

How to run it:
    python tools/gpu/gpu_checks.py --area 01              (area 01 .. 08, or "all")
    python tools/gpu/gpu_checks.py --area all --device cpu   (dry run on the CPU)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent      # the repository root
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TRITON_INTERPRET", "0")            # on a GPU we want the real Triton compiler, not the interpreter
from common import env  # noqa: E402  (this also points the Hugging Face cache at the repo folder)

RESULTS = Path(__file__).resolve().parent / "results"


def pick_device(requested):
    """Return the device to use: the requested one, or "cuda" when a GPU is visible, otherwise "cpu"."""
    import torch
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def sync(device):
    """Wait until the GPU has finished its queued work, so that timings are honest (a no-op on the CPU)."""
    import torch
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def run_script(args, timeout=900):
    """Run another script of the toolkit in a separate process and return (success, last lines of output)."""
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    tail = (r.stdout + r.stderr).strip().splitlines()[-5:]
    return r.returncode == 0, " | ".join(tail)[-400:]


# ----------------------------------------------------------------------------- one function per area
def area01(device):
    """Area 1: generate text with and without the KV cache and check that both give the same words; time both."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(env.SMOL_135M)
    model = AutoModelForCausalLM.from_pretrained(env.SMOL_135M, dtype=dtype).to(device).eval()
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(device)
    out, secs = {}, {}
    for use_cache in (True, False):
        sync(device); t = time.perf_counter()
        with torch.no_grad():
            out[use_cache] = model.generate(ids, max_new_tokens=64, do_sample=False, use_cache=use_cache)
        sync(device); secs[use_cache] = time.perf_counter() - t
    same = torch.equal(out[True], out[False])
    return same, {"same_tokens_with_and_without_cache": same, "seconds_with_cache": round(secs[True], 3),
                  "seconds_without_cache": round(secs[False], 3), "dtype": str(dtype)}


def area02(device):
    """Area 2: tiled attention must match PyTorch's built-in attention; then run the toolkit's SDPA benchmark script."""
    import torch
    sys.path.insert(0, str(ROOT / "02_gpu_systems_perf"))
    from flash_attention_from_scratch import flash_attention_tiled
    q, k, v = (torch.randn(1, 4, 512, 64, device=device) for _ in range(3))
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    error = (flash_attention_tiled(q, k, v, causal=True) - reference).abs().max().item()
    ok_script, tail = run_script([str(ROOT / "02_gpu_systems_perf" / "sdpa_backends.py"), "--device", device,
                                  "--out", str(RESULTS / f"sdpa_backends_{device}.json")])
    ok = error < 1e-3 and ok_script
    return ok, {"tiled_vs_builtin_max_error": error, "sdpa_script_ok": ok_script, "sdpa_script_output": tail}


def area03(device):
    """Area 3: with two or more GPUs, train with DDP over NCCL; on the CPU, train with DDP over gloo (a dry run)."""
    import torch
    spawn = str(ROOT / "03_distributed_training" / "launch_spawn.py")
    ddp = str(ROOT / "03_distributed_training" / "ddp_cpu.py")
    out = str(RESULTS / f"ddp_{device}.json")
    if device.startswith("cuda"):
        gpus = torch.cuda.device_count()
        if gpus < 2:
            return None, {"skipped": f"DDP over NCCL needs at least 2 GPUs, but only {gpus} is visible."}
        ok, tail = run_script(["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}", ddp,
                               "--backend", "nccl", "--steps", "20", "--out", out])
    else:
        ok, tail = run_script([spawn, ddp, "--world", "2", "--steps", "20", "--out", out])
    return ok, {"script_output": tail, "note": "ddp_cpu.py asserts internally that DDP matches single-process training."}


def area04(device):
    """Area 4: add LoRA adapters to the small model and train a few steps; the loss must go down."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(env.SMOL_135M)
    base = AutoModelForCausalLM.from_pretrained(env.SMOL_135M, dtype=dtype).to(device)
    model = get_peft_model(base, LoraConfig(r=8, target_modules="all-linear", task_type="CAUSAL_LM"))
    batch = tok(["Question: What is 12 plus 30?\nAnswer: 42"] * 8, return_tensors="pt").to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    losses = []
    sync(device); t = time.perf_counter()
    for _ in range(15):  # always the same batch, so the loss should fall quickly (this checks the training loop works)
        loss = model(**batch, labels=batch.input_ids).loss
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    sync(device)
    return losses[-1] < losses[0], {"first_loss": round(losses[0], 4), "last_loss": round(losses[-1], 4),
                                   "seconds_for_15_steps": round(time.perf_counter() - t, 3)}


def area05(device):
    """Area 5: score two answer choices with a language model on the device and check the statistics helper."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(ROOT / "05_evals_rigor"))
    from eval_harness import continuation_loglik
    from stats_utils import wilson_interval
    tok = AutoTokenizer.from_pretrained(env.SMOL_135M)
    model = AutoModelForCausalLM.from_pretrained(env.SMOL_135M, dtype=torch.float32).to(device).eval()
    scores, _token_counts = continuation_loglik(model, tok, "Question: What is the capital of France?\nAnswer:", [" Paris", " banana"], device=device)
    lo, hi = wilson_interval(50, 100)
    ok = float(scores[0]) > float(scores[1]) and 0.40 < lo < 0.41
    return ok, {"loglik_paris_vs_banana": [float(s) for s in scores], "wilson_50_of_100": [round(lo, 3), round(hi, 3)]}


def area06(device):
    """Area 6: this area is CPU work only (hashing text), so the check simply runs the MinHash test on the CPU."""
    sys.path.insert(0, str(ROOT / "06_data_engineering"))
    from minhash_dedup import dedup_minhash
    base = " ".join(f"word{i}" for i in range(200))
    keep, _ = dedup_minhash([base, base + " extra", " ".join(f"other{i}" for i in range(200))], threshold=0.8)
    return keep == [True, False, True], {"keep_mask": keep, "note": "CPU-only area; the device setting is ignored."}


def area07(device):
    """Area 7: embed sentences with the sentence-transformer on the device and check that similar sentences are closer."""
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer(env.MINILM, device=device)
    texts = ["cats purr and sleep", "a kitten is napping", "the stock market fell"] * 100
    sync(device); t = time.perf_counter()
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    sync(device)
    sim_close, sim_far = float(np.dot(emb[0], emb[1])), float(np.dot(emb[0], emb[2]))
    return sim_close > sim_far, {"similarity_related": round(sim_close, 3), "similarity_unrelated": round(sim_far, 3),
                                 "seconds_to_embed_300_sentences": round(time.perf_counter() - t, 3)}


def area08(device):
    """Area 8: the logit lens at the last layer must equal the model's own logits, also on the GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(ROOT / "08_safety_interp"))
    from logit_lens import residual_stream, lens_logits
    tok = AutoTokenizer.from_pretrained(env.SMOL_135M)
    model = AutoModelForCausalLM.from_pretrained(env.SMOL_135M, dtype=torch.float32).to(device).eval()
    ids = tok("Hello there", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        hs, logits = residual_stream(model, ids)
        diff = (lens_logits(model, hs[-1][0, -1]) - logits[0, -1]).abs().max().item()
    return diff < 1e-3, {"last_layer_lens_vs_model_logits_max_diff": diff}


CHECKS = {"01": area01, "02": area02, "03": area03, "04": area04, "05": area05, "06": area06, "07": area07, "08": area08}


def main():
    """Run the chosen area checks, print a PASS/FAIL line for each, save a JSON file, and return the exit code."""
    ap = argparse.ArgumentParser(description="Small GPU (or CPU dry-run) checks, one per toolkit area.")
    ap.add_argument("--area", default="all", help='"01" to "08", a comma-separated list, or "all"')
    ap.add_argument("--device", default="auto", help='"auto" (GPU if there is one), "cuda" or "cpu"')
    a = ap.parse_args()
    import torch
    device = pick_device(a.device)
    areas = sorted(CHECKS) if a.area == "all" else [x.strip().zfill(2) for x in a.area.split(",")]
    RESULTS.mkdir(parents=True, exist_ok=True)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print(f"Device used: {device}   GPU visible: {gpu_name}")
    if device == "cpu":
        print("NOTE: this is a CPU dry run. It does NOT count as a GPU test.")
    report = {"device": device, "gpu_name": gpu_name, "gpu_count": torch.cuda.device_count(), "torch": torch.__version__,
              "is_real_gpu_run": device.startswith("cuda"), "areas": {}}
    failed = 0
    for area in areas:
        start = time.time()
        try:
            ok, details = CHECKS[area](device)
        except Exception as error:  # report the problem instead of crashing, so the other areas still run
            ok, details = False, {"error": f"{type(error).__name__}: {error}"[:300]}
        status = "SKIPPED" if ok is None else ("PASS" if ok else "FAIL")
        failed += status == "FAIL"
        report["areas"][area] = {"status": status, "seconds": round(time.time() - start, 1), **details}
        print(f"area {area}: {status}  ({time.time() - start:.0f}s)  {json.dumps(details)[:200]}")
    out = RESULTS / f"gpu_checks_{device.replace(':', '')}_{'_'.join(areas) if len(areas) < 8 else 'all'}.json"
    out.write_text(json.dumps(report, indent=2))
    print("Saved", out)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
