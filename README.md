# AI Engineering Toolkit

This project teaches how modern AI language-model systems are built, served, trained and tested.
It is made for a beginner who has a normal laptop and wants to use only free and open-source tools.
You can read it, run it on your laptop today, and later move the same ideas onto a free or paid GPU.

If you are new here, read these files in this order:

1. This file explains what the project is and how to start it.
2. `STRUCTURE.md` shows every folder and file in a tree, with one plain sentence about each of them.
3. `GLOSSARY.md` explains the technical words, such as KV cache and LoRA, in everyday language.
4. The `README.md` file inside each numbered folder explains one area in more detail.

## What is in the box

The toolkit has eight areas, and each area lives in its own numbered folder.
Every area follows the same four rules:

- Each area has a README that explains the main ideas in plain language.
- Each area has a laptop path, which is a set of scripts that really ran from start to finish on a CPU.
- Each area has a cloud path in its `cloud/` folder, which is clearly marked UNTESTED because the build computer had no GPU.
- Each area has a `RESULTS.md` file that was generated from real runs, so no result number was typed in by hand.

The table below names the eight areas and gives the main measured results.
All timings were measured on one shared CPU machine, so use the ratios and not the exact seconds.

| # | Folder | What you learn | Main measured results |
|---|---|---|---|
| 1 | [`01_inference_serving`](01_inference_serving/README.md) | How a model is served quickly: KV cache, batching, speculative decoding, quantization and benchmarking. | Decoding with a KV cache gives exactly the same tokens and is up to about 7.7 times faster per step at 512 tokens of context. The llama.cpp server with 4 slots gave about 2.4 times the throughput of 1 slot with 4 clients at the same time, and the time to the first token was 28 ms instead of 1.6 s. Rounding weights to 4 bits without groups made perplexity worse, from 20.9 to 48.7, and groups of 64 weights gave 29.3. Speculative decoding in Hugging Face was slower (0.73 times) on this CPU. |
| 2 | [`02_gpu_systems_perf`](02_gpu_systems_perf/README.md) | How to find slow code and speed it up: the profiler, `torch.compile`, Triton and FlashAttention. | The matrix multiplication operation took 56 percent of the CPU time in a training step. Tiled attention was exact to about 1e-6 and used 18.6 MB of extra memory instead of 1072 MB at sequence length 4096. `torch.compile` gave no meaningful gain on the CPU (1.05 times with one thread). |
| 3 | [`03_distributed_training`](03_distributed_training/README.md) | How training is spread over many processes: DDP, FSDP2, collective operations, mixed precision, ZeRO memory math and DeepSpeed settings. | Training with two processes (DDP and FSDP) matched training with one process to about 2e-7. Our ZeRO memory calculator reproduced the numbers of the ZeRO paper: 120, 31.4, 16.6 and 1.9 GB. |
| 4 | [`04_post_training`](04_post_training/README.md) | How a model is tuned after pre-training: SFT with LoRA, DPO, a reward model and synthetic data. | LoRA fine-tuning of SmolLM2-135M raised the held-out accuracy on a toy task from 3 percent to 56.5 percent. DPO did not help, and we report that negative result. The reward model failed on the hard task and passed an easy sanity task (54.5 percent to 100 percent). |
| 5 | [`05_evals_rigor`](05_evals_rigor/README.md) | How to measure models honestly: confidence intervals, paired tests, random seeds and contamination checks. | The difference between the 135M and the 360M model on ARC-Easy (200 questions) was not significant (McNemar p between 0.13 and 0.23). The n-gram contamination detector found almost nothing once 20 percent of the words of a copied text were changed. |
| 6 | [`06_data_engineering`](06_data_engineering/README.md) | How training text is cleaned: near-duplicate removal with MinHash and LSH, quality filters, streaming and packing. | The MinHash similarity estimate had almost no bias (-0.0006). The LSH step found 74 percent of the duplicates at the 0.8 threshold, which is expected from its design. The perplexity filter flagged random text but did not flag repetition or spam. |
| 7 | [`07_rag_agents`](07_rag_agents/README.md) | How a model uses outside knowledge and tools: embeddings, FAISS, BM25, hybrid search, a reranker and a tool-calling agent. | On SciFact, nDCG@10 was 0.652 for BM25, 0.648 for dense search and 0.685 for the hybrid method, which was significantly better than both. The cross-encoder reranker gave no gain. The 0.5B agent wrote valid tool calls 31 times out of 31, but it called a tool in 4 of 5 questions that needed no tool. |
| 8 | [`08_safety_interp`](08_safety_interp/README.md) | How to test a model's safety and look inside it: a red-team harness, the logit lens and activation patching. | The harness reports refusal rates with confidence intervals for 3 small models. The logit lens at the last layer equals the model's own output. Patching showed the information about "France" reaching the last token at block 23. |

## Quick start on Linux or macOS

Run these commands from the project folder, one at a time.

```bash
./setup.sh                      # Creates the virtual environment, installs packages and downloads about 2.5 GB of small models.
source .venv/bin/activate       # Turns the virtual environment on for this terminal.
bash tools/healthcheck.sh       # Checks in about a minute that everything works and prints a PASS or FAIL table.
pytest -q -m "not slow"         # Runs the fast tests.
pytest -q                       # Runs all 102 tests, including the two-process jobs (needs the models downloaded).
export HF_HUB_OFFLINE=1         # After the first download, this lets everything run without internet.
cd 01_inference_serving && python kv_cache_demo.py   # Runs your first experiment.
```

On Windows you can run `powershell -ExecutionPolicy Bypass -File setup.ps1`.
This was not tested on Windows, because everything was built and run on Linux with an x86-64 CPU.
Triton and DeepSpeed have no normal Windows installers, so use WSL2 (Linux inside Windows) for those two.
You need Python 3.10 or newer, and we used Python 3.13.

## Using the toolkit offline

Run `python tools/download_models.py` once while you are online.
It downloads SmolLM2-135M and SmolLM2-360M (and their Instruct versions), Qwen2.5-0.5B-Instruct (also as a GGUF file), the MiniLM embedding model, a reranker and the datasets.
After that, set `HF_HUB_OFFLINE=1` and everything runs without internet.
The files are stored in the `.hf_cache` folder inside the project, so you can move the whole folder to another computer.
All results come from small models (at most 0.5 billion parameters), 8 CPU cores, and scripts that take at most about 10 minutes.

## Moving to a free GPU or a cloud machine

The repository lives at https://github.com/Metamyrddin/ai-eng-toolkit.
The folder `tools/gpu/` can run a short GPU test for each area on a free Kaggle or Colab GPU.
Its `README.md` explains every step for a beginner, including how to get a free Kaggle API token.
Those GPU runs are marked UNTESTED until someone really runs them, and a CPU dry run of the same code passed on 8 of 8 checks.

If you have your own GPU machine, follow these steps:

1. Install the CUDA build of PyTorch from https://pytorch.org/get-started/locally/ and then run `pip install -r requirements-gpu.txt`.
2. Open the `cloud/` folder of each area, which holds the launch scripts. Start with area 1, then area 3 and then area 4.
3. Remember that nobody has run these scripts yet. The flags were only compared with the official documentation, so expect to fix small things.
4. Run `run_all.sh` on the GPU machine to rebuild the results.

## What was tested and what was not

The following parts really ran on the build computer (a CPU only machine):

- All scripts in the laptop path of every area.
- All 102 unit and integration tests.
- Two-process DDP and FSDP training with the gloo backend.
- The Triton kernels in interpreter mode.
- The parsing of the DeepSpeed setting files.

The following parts are UNTESTED, because they need hardware or software that the build computer did not have:

- vLLM, NCCL, and training on several GPUs or several computers.
- Real DeepSpeed and FSDP training on GPUs, and QLoRA with bitsandbytes.
- The flash-attn package, Triton on a real GPU, and the CUDA profiler.
- Speed gains from FP8 and bfloat16 hardware.
- The setup scripts on Windows and macOS.

## One file that is not in the GitHub repository

The profiler trace `02_gpu_systems_perf/results/profile_trace.json` is a 38 MB file, and it could not be uploaded through the text-only GitHub connector that was used for publishing.
You can create it again on your own computer by running `python 02_gpu_systems_perf/profile_torch.py`, and you can open it at https://ui.perfetto.dev.
The table of the slowest operations from the same run is stored in `02_gpu_systems_perf/results/profile_top_ops.json`.

## Rules this project follows

- Every number in a `RESULTS.md` file was produced by a script in this repository, and the JSON file with the raw numbers is stored next to it.
- Timings on a shared virtual machine are noisy, so read the ratios and the confidence intervals instead of single absolute values.
- Negative and null results are kept on purpose. Examples are DPO, the reward model, `torch.compile`, the reranker and speculative decoding on a CPU.
- Simulations are always labelled as simulations. Examples are `batching_sim.py`, `quant_from_scratch.py` and `zero_memory.py`.
- Every web link and paper title is checked by `tools/check_urls.py`, and the latest report is `tools/url_check_report.json`.

## How the folders are organised

The file `STRUCTURE.md` has the complete tree. Here is the short version:

- The `common/` folder holds helper code that all areas share.
- The `tools/` folder holds helper programs for downloading models, checking the health of the project, building result pages, checking links, running on free GPUs and publishing to GitHub.
- The `tests/` folder holds the pytest tests.
- The numbered folders `01_` to `08_` hold the eight areas, and each has a README, a RESULTS file, a NOTES file, scripts, saved results and a `cloud/` folder.
- The `skills_drafts/` folder holds one short how-to guide per area, with the commands and the traps that we found in real runs.
- The file `requirements-cpu.txt` lists what to install, `requirements-lock-cpu-linux-py313.txt` lists the exact versions used for the results, and `requirements-gpu.txt` is for GPUs and is UNTESTED.

## Things to know before you trust the numbers

- The CPU run of `torch.compile` used an improvised C++ toolchain, because the build computer had no `g++`. Only the one-thread comparison is considered valid. On a normal computer, install `g++` (or the Visual Studio build tools) and run it again.
- The built-in server of `llama-cpp-python` returned empty streams when several requests came at once, so the serving benchmarks use the official `llama-server` program instead.
- A gap of about 1.4 percent between the perplexity of the fp16 GGUF file and the perplexity of the Hugging Face fp32 model is still unexplained.
- All models are tiny, so what works here may not work for models with billions of parameters.

## License

The code is released under the MIT license, which is in the file `LICENSE`.
