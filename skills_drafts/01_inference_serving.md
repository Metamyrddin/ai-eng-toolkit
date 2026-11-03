# Skill draft: area 1, inference and serving

## When to use this area

Use this area when someone asks how a language model is served quickly and cheaply.
It is the right place for questions about the KV cache, batching, speculative decoding, quantization, llama.cpp and the way latency and throughput are measured.

## Exact commands

Run these commands from the folder `01_inference_serving`.

```bash
python ../tools/download_models.py           # Downloads the models once, while you are online.
bash ../tools/get_llamacpp.sh                # Downloads the ready-made llama-server and llama-bench programs.
python kv_cache_demo.py                      # Checks exactness, speed and the memory formula of the KV cache.
python bench_hf.py --batch-sizes 1 2 4 8     # Measures static batching with Hugging Face.
python bench_llamacpp.py                     # Sweeps the quantization levels of a GGUF model.
python quant_from_scratch.py                 # Rounds weights to fewer bits and measures the damage.
python speculative_demo.py                   # Runs speculative decoding with two real models.
python batching_sim.py                       # Simulates static against continuous batching.
./serve_llamacpp.sh q4_k_m 8080 4 &          # Starts the server with 4 parallel slots.
python bench_serving.py --concurrency 1 2 4  # Measures the server with 1, 2 and 4 clients.
./run_area1.sh                               # Runs everything in the order used for RESULTS.md (6 to 10 minutes).
```

## Known traps and gotchas

- The built-in server of llama-cpp-python failed under concurrent streaming. With 4 clients, 7 of 8 requests returned no tokens in all 3 trials. Use the official `llama-server` program for serving tests, and run `run_llamacpp_python_server_check.sh` only to reproduce the problem.
- The llama-cpp-python wheel is much slower than the upstream `llama-bench` program for the same model, because the two use different llama.cpp builds. Never compare absolute numbers between the two tools.
- Speculative decoding in Hugging Face was slower than normal decoding on the CPU (0.73 times), because a 135M helper model is not cheap enough compared with a 360M target. Do not expect a speed-up unless the helper is at least about ten times cheaper.
- The perplexity of the fp16 GGUF file differs from the Hugging Face fp32 perplexity by about 1.4 percent, and we do not know why. The perplexity uses only 2,044 tokens, so small differences between neighbouring rows may be noise.
- The script `quant_from_scratch.py` only simulates quantization, which means it measures the loss of quality but saves no memory and gains no speed.
- The script `serve_llamacpp.sh` needs the programs in `.bin`, so run `tools/get_llamacpp.sh` first. The prebuilt programs are for Linux on x86-64 only.

## General traps that apply to every area

- Run every command inside the project's virtual environment, which you turn on with `source .venv/bin/activate` from the project folder.
- Run `python tools/download_models.py` once while you are online, and then set `export HF_HUB_OFFLINE=1` to work offline.
- The tests set the variable `AIETK_TEST_MODE`, which sends every results file to a temporary folder. This protects the real results from being overwritten, so do not be surprised that a test run leaves `results/` unchanged.
- Timings come from a shared virtual machine with 8 CPU cores, so they are noisy. Compare ratios and confidence intervals and do not quote single absolute numbers.
- Nothing has been run on a GPU, on Windows or on macOS. Treat the cloud scripts and the setup scripts for those systems as UNTESTED.
