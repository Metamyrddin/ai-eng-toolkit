"""This file downloads all the models that the toolkit needs into a local cache folder. You run it
once while you are online, and afterwards everything works offline.

More technical notes:
Download every model/dataset used by the toolkit into ./.hf_cache (needs internet once).

After this finishes, the whole toolkit runs OFFLINE:  export HF_HUB_OFFLINE=1
Usage:  python tools/download_models.py [--small]   (--small skips the GGUF fp16 + q2_k files)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import env  # noqa: E402,F401  (sets HF_HOME)
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

MODELS = [env.SMOL_135M, env.SMOL_135M_INSTRUCT, env.SMOL_360M_INSTRUCT, "HuggingFaceTB/SmolLM2-360M", env.QWEN_05B_INSTRUCT,
          env.MINILM, env.CROSS_ENCODER]
DATASETS = ["BeIR/scifact", "BeIR/scifact-qrels", "allenai/ai2_arc", "trl-lib/ultrafeedback_binarized",
            "trl-lib/Capybara", "Salesforce/wikitext"]


def main():
    """Download all models that the toolkit needs into the local Hugging Face cache."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    for m in MODELS:
        print("model", m, snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md"]))
    quants = ["q4_k_m", "q8_0"] if a.small else ["q2_k", "q4_k_m", "q5_k_m", "q8_0", "fp16"]
    for q in quants:
        print("gguf", q, hf_hub_download(env.QWEN_05B_GGUF_REPO, f"qwen2.5-0.5b-instruct-{q}.gguf"))
    from datasets import load_dataset
    load_dataset("BeIR/scifact", "corpus"); load_dataset("BeIR/scifact", "queries")
    load_dataset("BeIR/scifact-qrels")
    load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    load_dataset("trl-lib/ultrafeedback_binarized", split="train[:2000]")
    load_dataset("trl-lib/Capybara", split="train[:2000]")
    load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    print("done")


if __name__ == "__main__":
    main()
