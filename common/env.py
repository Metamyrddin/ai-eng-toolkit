"""This file knows where things live (the repository folder, the model cache), sets random seeds so
that runs can be repeated, describes the computer for the results files and saves JSON files
safely. Every area imports it.

More technical notes:
Environment, paths, seeding and run-metadata helpers shared by every area.

Design rules:
  * Importing this module must be cheap (no torch import at top level).
  * Every benchmark/result JSON embeds `environment_info()` so numbers are
    traceable to the machine that produced them.
"""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Keep all model/dataset downloads inside the repo folder by default so the whole
# toolkit can be zipped / moved to an offline laptop. Override with HF_HOME.
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Default tiny models (all free / open weights).
SMOL_135M = "HuggingFaceTB/SmolLM2-135M"
SMOL_135M_INSTRUCT = "HuggingFaceTB/SmolLM2-135M-Instruct"
SMOL_360M_INSTRUCT = "HuggingFaceTB/SmolLM2-360M-Instruct"
QWEN_05B_INSTRUCT = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_05B_GGUF_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def set_offline(flag: bool = True) -> None:
    """Force HF libraries to use only the local cache (use after download_models.py)."""
    if flag:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (if installed)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _cpu_model() -> str:
    """Return the name of the CPU from the operating system, or 'unknown' if it cannot be read.
    """
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def environment_info() -> dict:
    """Machine + library versions, embedded in every results file."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    for mod in ("torch", "transformers", "peft", "trl", "datasets", "sentence_transformers",
                "faiss", "llama_cpp", "triton", "numpy", "datasketch"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            info[mod] = None
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
    except Exception:
        pass
    return info


def save_json(obj, path: str | Path) -> Path:
    """Write an object to a JSON file and return the path. In test mode the file goes to a
    temporary folder, so that tests never overwrite real results.
    """
    path = Path(path)
    if os.environ.get("AIETK_TEST_MODE") and "results" in path.parts:  # tests must never overwrite real measured results
        import tempfile
        path = Path(tempfile.gettempdir()) / "aietk_test_results" / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    def clean(x):  # NaN/inf are not valid JSON -> null
        """Replace values that JSON cannot store (not-a-number and infinity) by null, also
        inside lists and dictionaries.
        """
        if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
            return None
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return x
    with open(path, "w") as f:
        json.dump(clean(obj), f, indent=2, default=str)
    return path


def git_free_commit() -> str:
    """Return the short id of the current git commit, or 'no-git' if the code is not in a git
    repository.
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def gguf_path(quant: str) -> str:
    """Resolve a Qwen2.5-0.5B-Instruct GGUF file from the local HF cache (downloads if missing)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(QWEN_05B_GGUF_REPO, f"qwen2.5-0.5b-instruct-{quant}.gguf")
