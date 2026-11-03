# 1 · Inference & Serving

*How a trained language model is turned into fast, cheap text generation.*
All numbers quoted by scripts come from real runs on this repo's build machine — see [RESULTS.md](RESULTS.md).

## The mechanisms in plain words

**Generation is a loop.** A decoder-only model predicts *one next token* from all previous tokens, appends it, and repeats.
Two phases have very different costs:

| Phase | What happens | Bottleneck |
|---|---|---|
| **Prefill** | read the whole prompt at once, build the KV cache | compute (big matrix multiplies) |
| **Decode** | produce one token per step, re-using the cache | memory bandwidth (every step re-reads all weights + the cache) |

Metrics you will see: **TTFT** (time to first token ≈ prefill time), **TPOT / inter-token latency** (time per decoded token),
**decode tokens/s**, and **throughput** (total tokens/s over many requests).

**KV cache.** Attention needs the *keys* and *values* of every earlier token. Recomputing them each step costs O(n) work per token;
caching them makes each step O(1) extra work for the projections (attention itself still reads O(n) cached entries).
Cache size = `2 × layers × kv_heads × head_dim × seq_len × batch × bytes` — `kv_cache_demo.py` checks this formula against the
bytes really held by the Hugging Face cache object. The cache is *exact*: cached and uncached greedy decoding give identical tokens
(also checked). Long contexts and large batches make the KV cache — not the weights — the main memory consumer, which is why
vLLM's **PagedAttention** stores it in fixed-size blocks to avoid fragmentation ([Kwon et al. 2023](https://arxiv.org/abs/2309.06180)).

**Batching.** Decoding one sequence leaves the hardware mostly idle (it is bandwidth-bound), so serving several sequences per step
is almost free throughput. *Static batching* waits for the slowest sequence; *continuous (iteration-level) batching* — introduced
for transformers in [Orca (OSDI '22)](https://www.usenix.org/conference/osdi22/presentation/yu) and used by vLLM and llama.cpp's
server — lets finished sequences leave and new ones join at every step. `bench_hf.py` measures static batching in Hugging Face
Transformers; `batching_sim.py` simulates the scheduling effect; `serve_llamacpp.sh` + `bench_serving.py` measure it for real with
llama.cpp's `llama-server` (1 slot vs 4 slots).

**Speculative decoding.** A small *draft* model proposes γ tokens cheaply; the big *target* model checks all of them in **one**
forward pass and keeps the longest prefix it agrees with (plus one token of its own). With the accept/reject rule of
[Leviathan et al. 2023](https://arxiv.org/abs/2211.17192) / [Chen et al. 2023](https://arxiv.org/abs/2302.01318) the output
distribution is *exactly* the target's. `speculative.py` implements the algorithm from scratch; `tests/test_speculative.py`
checks exactness statistically on toy models; `speculative_demo.py` measures acceptance rate on SmolLM2-360M (target) /
SmolLM2-135M (draft) and wall-clock with Hugging Face's production implementation
(`assistant_model=`, [docs](https://huggingface.co/docs/transformers/generation_strategies#speculative-decoding)).
Speed-up requires the draft to be *much* cheaper than the target — see RESULTS.md for what happened on CPU.

**Quantization.** Store weights in fewer bits (8, 4, … ) to shrink memory and read less data per token. Simple *round-to-nearest*
(RTN) with a scale per group of weights (`quant_from_scratch.py`) works at 8 bits, degrades at 4 and collapses at ≤3 bits on a
small model; smarter methods use calibration data to fix the error: GPTQ ([Frantar et al. 2022](https://arxiv.org/abs/2210.17323)),
AWQ ([Lin et al. 2023](https://arxiv.org/abs/2306.00978)), LLM.int8() ([Dettmers et al. 2022](https://arxiv.org/abs/2208.07339)).
llama.cpp's GGUF "k-quants" (q2_k … q8_0) are ready-made; `bench_llamacpp.py` measures size, speed and perplexity for each.
Note `quant_from_scratch.py` is *simulated* (fake-quantize then run in fp32): it measures the quality effect only.

## Laptop / offline path (all tested here, CPU only)

```bash
python ../tools/download_models.py           # once, needs internet; afterwards export HF_HUB_OFFLINE=1
bash ../tools/get_llamacpp.sh                # (optional) prebuilt llama-server / llama-bench binaries (Linux x64)
python kv_cache_demo.py                      # KV cache: exactness, speed vs prefix length, memory formula
python bench_hf.py --batch-sizes 1 2 4 8     # TTFT / decode throughput vs batch size (HF transformers)
python bench_llamacpp.py                     # GGUF quant sweep: size, prefill tok/s, TTFT, decode tok/s, perplexity
python quant_from_scratch.py                 # RTN quantization quality vs bits and group size
python speculative_demo.py                   # speculative decoding: exactness + acceptance + wall-clock
python batching_sim.py                       # static vs continuous batching (simulation, labelled as such)
./serve_llamacpp.sh q4_k_m 8080 4 &          # OpenAI-compatible server with 4 parallel slots
python bench_serving.py --concurrency 1 2 4  # streaming client: TTFT / TPOT / throughput under concurrency
./run_area1.sh                               # everything above, in the order used for RESULTS.md
```

`llama-cpp-python` has its own OpenAI-compatible server (`python -m llama_cpp.server`), but in our test it returned **empty
streams for concurrent requests** (at 4 concurrent clients, 7 of 8 requests per trial yielded no tokens, in all 3 trials — `run_llamacpp_python_server_check.sh`, table in RESULTS.md), so the scripts use upstream `llama-server` instead —
see RESULTS.md "Caveats".

## Cloud-GPU path (in the `cloud/` folder, UNTESTED)

The `cloud/` folder holds the scripts `vllm_serve.sh`, `vllm_serve_speculative.sh`, `vllm_quantized.sh` and `bench_cloud.sh`.
There was no GPU on the build computer, so these scripts were never run.
Their flags were only compared with the vLLM documentation ([CLI](https://docs.vllm.ai/en/latest/cli/serve/) and [speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)).
The benchmark client `bench_serving.py` is tested, and only the server side is not.

## What each file does

- The file `kv_cache_demo.py` checks that the KV cache gives exact results, measures the cost of one step for different prompt lengths, and checks the memory formula.
- The file `bench_hf.py` measures the latency, the time to the first token and the throughput of Hugging Face generation with static batching.
- The file `bench_llamacpp.py` runs a sweep over quantization levels with llama.cpp and compares the perplexity with the Hugging Face fp32 model.
- The file `quant_from_scratch.py` rounds the weights to fewer bits and studies how the bit width and the group size change the quality.
- The files `speculative.py` and `speculative_demo.py` contain speculative sampling written from scratch and the measurements on real models.
- The file `batching_sim.py` simulates static and continuous batching.
- The file `bench_serving.py` is a load generator that uses only the Python standard library and talks to any OpenAI-compatible streaming server.
- The files `serve_llamacpp.sh` and `run_llama_bench.sh` start the local server and the official llama-bench program.
- The folder `cloud/` holds the vLLM templates, which are untested.

## Honest limitations

- CPU numbers from a shared virtual machine are noisy, so we report medians and, where it mattered, all trials. Comparisons between two rows, such as 4 slots against 1 slot, are meaningful, but the absolute tokens per second only describe this one computer.
- GPUs behave differently from CPUs, for example in where the limit switches between compute and memory bandwidth, and with FlashAttention and CUDA graphs. Do not copy conclusions from the CPU tables to GPUs.
- The file `batching_sim.py` simulates a scheduler with an assumed step-cost model, so it is not a measurement.
