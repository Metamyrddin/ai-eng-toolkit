"""Health check for the AI engineering toolkit: "is everything installed and does each area still work?"

How this file fits in the project:
    Run it after setting up (or after changing anything) to get one clear PASS/FAIL table.
    It does four kinds of checks, from cheapest to most expensive:
      1. The Python environment (is the virtual environment active, which Python version is it).
      2. Key Python packages (can they be imported, and which version is installed).
      3. Local files that the toolkit needs (cached models and the llama.cpp server program).
      4. One tiny "smoke test" per area. A smoke test runs a very small piece of real work, such as
         generating a few tokens, to prove that the area still runs from start to finish.
    The whole run is designed to finish in about one minute on a normal laptop CPU.

How to run it:
    python tools/healthcheck.py            (or:  bash tools/healthcheck.sh)
    python tools/healthcheck.py --fast     (skips the smoke tests that load a language model)

The exit code is 0 when every check passes and 1 when at least one check fails, so scripts can use it.
"""
import argparse
import concurrent.futures as cf
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

# The repository root is the folder that contains this "tools" folder.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import env  # noqa: E402  (importing it also points HF_HOME at the repo's own model cache)

# Each smoke test is a small Python program given as a string. It runs in its own process so a crash in one
# area can never break the others. Every program must print the word OK as its last line when it succeeds.
PRELUDE = f"""
import os, sys
os.environ.setdefault('TRITON_INTERPRET', '1')
os.environ['HF_HUB_OFFLINE'] = '1'          # the health check must work without internet
sys.path.insert(0, {str(ROOT)!r})
for d in sorted(__import__('pathlib').Path({str(ROOT)!r}).glob('0[1-8]_*')):
    sys.path.insert(0, str(d))              # lets us import the area scripts by their plain file names
from common import env
"""

SMOKE = {
    # Area 1: generate a few tokens with and without the KV cache; both must give exactly the same tokens.
    "1 inference": ("model", """
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(env.SMOL_135M)
model = AutoModelForCausalLM.from_pretrained(env.SMOL_135M, dtype=torch.float32).eval()
ids = tok('The capital of France is', return_tensors='pt').input_ids
with torch.no_grad():
    a = model.generate(ids, max_new_tokens=6, do_sample=False, use_cache=True)
    b = model.generate(ids, max_new_tokens=6, do_sample=False, use_cache=False)
assert torch.equal(a, b), 'cached and uncached generation differ'
print('OK')
"""),
    # Area 2: the tiled "FlashAttention-style" attention must equal the plain attention.
    "2 gpu-perf": ("fast", """
import torch
from flash_attention_from_scratch import naive_attention, flash_attention_tiled
q, k, v = (torch.randn(1, 2, 100, 16, dtype=torch.float64) for _ in range(3))
err = (naive_attention(q, k, v, causal=True) - flash_attention_tiled(q, k, v, causal=True, block_q=32, block_k=32)).abs().max().item()
assert err < 1e-9, err
print('OK')
"""),
    # Area 3: really start two worker processes that talk through the gloo backend, and train a few steps.
    "3 distributed": ("fast", """
import subprocess, sys, tempfile, os
d = os.path.join({root!r}, '03_distributed_training')
out = os.path.join(tempfile.mkdtemp(), 'ddp.json')
r = subprocess.run([sys.executable, os.path.join(d, 'launch_spawn.py'), os.path.join(d, 'ddp_cpu.py'), '--world', '2',
                    '--steps', '5', '--out', out], capture_output=True, text=True, timeout=50)
assert r.returncode == 0, r.stdout[-400:] + r.stderr[-400:]
print('OK')
""".replace("{root!r}", repr(str(ROOT)))),
    # Area 4: add LoRA adapters to the small model and take one training step; the loss must be a finite number.
    "4 post-training": ("model", """
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(env.SMOL_135M)
model = get_peft_model(AutoModelForCausalLM.from_pretrained(env.SMOL_135M, dtype=torch.float32),
                       LoraConfig(r=4, target_modules='all-linear', task_type='CAUSAL_LM'))
batch = tok(['Question: What is 2 plus 3?\\nAnswer: 5'], return_tensors='pt')
loss = model(**batch, labels=batch.input_ids).loss
loss.backward()
assert torch.isfinite(loss), loss
print('OK')
"""),
    # Area 5: the statistics helpers must give the textbook answer for a simple case.
    "5 evals": ("fast", """
from stats_utils import wilson_interval, mcnemar_exact
lo, hi = wilson_interval(50, 100)
assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60, (lo, hi)
mcnemar_exact([1, 1, 0, 0] * 10, [1, 0, 1, 0] * 10)
print('OK')
"""),
    # Area 6: two near-identical documents must be merged by MinHash, a different document must be kept.
    "6 data-eng": ("fast", """
from minhash_dedup import dedup_minhash
base = ' '.join(f'word{i}' for i in range(200))
docs = [base, base + ' extra', ' '.join(f'other{i}' for i in range(200))]
keep, _ = dedup_minhash(docs, threshold=0.8)
assert keep == [True, False, True], keep
print('OK')
"""),
    # Area 7: build a tiny BM25 index and a tiny FAISS index over three sentences and search both.
    "7 rag": ("model", """
from retrieval_lib import BM25Index, DenseIndex
ids = ['a', 'b', 'c']
texts = ['cats purr and sleep all day', 'the stock market fell today', 'dogs bark at strangers']
bm, de = BM25Index(ids, texts), DenseIndex(ids, texts, cache_name='healthcheck_tmp')
assert bm.search('stock market', 1)[0][0] == 'b'
assert de.search('a feline pet', 1)[0][0] == 'a'
import os; os.remove(os.path.join({cache!r}, 'healthcheck_tmp.npy'))
print('OK')
""".replace("{cache!r}", repr(str(ROOT / "07_rag_agents" / "cache")))),
    # Area 8: the refusal detector must work, and the logit lens at the last layer must equal the model's own logits.
    "8 safety": ("model", """
import torch
from redteam_harness import is_refusal
assert is_refusal("I'm sorry, but I can't help with that.") and not is_refusal('Sure, here is a cake recipe.')
from transformers import AutoModelForCausalLM, AutoTokenizer
from logit_lens import residual_stream, lens_logits
tok = AutoTokenizer.from_pretrained(env.SMOL_135M)
model = AutoModelForCausalLM.from_pretrained(env.SMOL_135M, dtype=torch.float32).eval()
ids = tok('Hello there', return_tensors='pt').input_ids
with torch.no_grad():
    hs, logits = residual_stream(model, ids)
    diff = (lens_logits(model, hs[-1][0, -1]) - logits[0, -1]).abs().max().item()
assert diff < 1e-3, diff
print('OK')
"""),
}

# Python packages that the toolkit needs, and the name used to import each of them.
IMPORTS = ["numpy", "scipy", "torch", "transformers", "datasets", "peft", "trl", "accelerate", "sentence_transformers",
           "faiss", "rank_bm25", "datasketch", "llama_cpp", "triton", "psutil", "pytest"]
# Models that must be in the local cache, as folder names inside the Hugging Face cache.
MODELS = ["HuggingFaceTB--SmolLM2-135M", "HuggingFaceTB--SmolLM2-360M", "HuggingFaceTB--SmolLM2-135M-Instruct",
          "Qwen--Qwen2.5-0.5B-Instruct", "Qwen--Qwen2.5-0.5B-Instruct-GGUF",
          "sentence-transformers--all-MiniLM-L6-v2", "cross-encoder--ms-marco-MiniLM-L-6-v2"]


def check_environment():
    """Return rows for the Python version and for whether a virtual environment is active."""
    rows = []
    ok = sys.version_info >= (3, 10)
    rows.append(("python version", ok, f"Python {sys.version.split()[0]} (need 3.10 or newer)"))
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)  # inside a venv these two paths differ
    rows.append(("virtual environment", in_venv, sys.prefix if in_venv else "not active: run `source .venv/bin/activate`"))
    return rows


def check_imports():
    """Return one row per required package, saying whether it can be imported and which version it is."""
    rows = []
    for name in IMPORTS:
        try:
            module = importlib.import_module(name)
            rows.append((f"import {name}", True, getattr(module, "__version__", "version unknown")))
        except Exception as error:  # a missing or broken package raises ImportError or something similar
            rows.append((f"import {name}", False, f"{type(error).__name__}: {str(error)[:70]}"))
    return rows


def check_files():
    """Return rows for the cached models, the GGUF model file and the llama-server program."""
    rows = []
    hub = Path(os.environ["HF_HOME"]) / "hub"
    for m in MODELS:
        found = (hub / f"models--{m}").exists()
        rows.append((f"cached model {m.replace('--', '/')}", found, "found" if found else "missing: run tools/download_models.py"))
    ggufs = list(hub.glob("models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/snapshots/*/*q4_k_m.gguf"))
    rows.append(("GGUF file q4_k_m", bool(ggufs), "found" if ggufs else "missing: run tools/download_models.py"))
    servers = list((ROOT / ".bin").glob("llama-*/llama-server"))
    if servers:
        # Ask the program for its version: this proves that it starts and that its libraries can be loaded.
        r = subprocess.run([str(servers[0]), "--version"], capture_output=True, text=True, timeout=20,
                           env={**os.environ, "LD_LIBRARY_PATH": str(servers[0].parent)})
        text = (r.stdout + r.stderr).strip().splitlines()
        rows.append(("llama-server binary", r.returncode == 0, text[-1][:70] if text else "no output"))
    else:
        rows.append(("llama-server binary", False, "missing: run `bash tools/get_llamacpp.sh`"))
    return rows


def run_smoke(name, code):
    """Run one smoke test in a separate Python process and return (name, passed, detail, seconds)."""
    start = time.time()
    try:
        r = subprocess.run([sys.executable, "-c", PRELUDE + code], capture_output=True, text=True, timeout=55)
        passed = r.returncode == 0 and r.stdout.strip().endswith("OK")
        detail = "ran correctly" if passed else (r.stderr.strip().splitlines() or ["no error text"])[-1][:90]
    except subprocess.TimeoutExpired:
        passed, detail = False, "took longer than 55 seconds"
    return (f"smoke test area {name}", passed, detail, time.time() - start)


def main():
    """Run every check, print the table, and exit with 0 (all passed) or 1 (something failed)."""
    ap = argparse.ArgumentParser(description="Quick health check of the AI engineering toolkit.")
    ap.add_argument("--fast", action="store_true", help="skip the smoke tests that load a language model")
    args = ap.parse_args()
    t0 = time.time()
    rows = [(n, ok, d, None) for n, ok, d in check_environment() + check_imports() + check_files()]
    tests = {k: v[1] for k, v in SMOKE.items() if not (args.fast and v[0] == "model")}
    # The smoke tests are independent, so four of them run at the same time to save wall-clock time.
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_smoke, k, c) for k, c in tests.items()]
        rows += sorted((f.result() for f in futures), key=lambda r: r[0])
    width = max(len(r[0]) for r in rows)
    print(f"{'CHECK':<{width}}  RESULT  DETAIL")
    for name, ok, detail, secs in rows:
        extra = f" ({secs:.0f}s)" if secs else ""
        print(f"{name:<{width}}  {'PASS' if ok else 'FAIL':<6}  {detail}{extra}")
    failed = [r for r in rows if not r[1]]
    print(f"\n{len(rows) - len(failed)} of {len(rows)} checks passed in {time.time() - t0:.0f} seconds.")
    if failed:
        print("FAILED:", ", ".join(r[0] for r in failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
