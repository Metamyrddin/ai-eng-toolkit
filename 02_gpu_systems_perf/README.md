# 2 · GPU & Systems Performance

*Where time and memory go in a neural-network step, and the tools that reveal it.*
Everything marked "tested" ran on the CPU of the build computer, which had no GPU.
The GPU commands are marked **UNTESTED** and are listed separately.

## Mechanisms in plain words

**Profile before you optimise.** `torch.profiler` records every operator (and on GPUs every CUDA kernel). The table you get
shows *self time* per operator. In `profile_torch.py` we profile a real training step (forward + backward) of SmolLM2-135M and
save a Chrome-trace JSON you can open in [Perfetto](https://ui.perfetto.dev). Docs:
[PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html). On CPU, matrix multiplies (`aten::mm`) dominate, as expected
(see RESULTS.md for the measured percentage).

**`torch.compile`.** Python-level PyTorch runs one operator at a time ("eager"), writing every intermediate to memory.
`torch.compile` traces your code (TorchDynamo), then TorchInductor generates *fused* kernels (C++/OpenMP on CPU, Triton on GPU),
cutting memory traffic for chains of cheap elementwise operations
([docs](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)). You pay a one-off **compile time**; gains depend on how
memory-bound your model is. `compile_bench.py` reports compile time and steady-state speed-up *separately* and checks that outputs
match eager. **Important caveat in RESULTS.md:** on this box the C++ toolchain had to be improvised (no `g++`), which made the
multi-threaded compiled kernels slower than eager — we report the single-thread comparison as the fair one, and say why.

**Triton.** A Python DSL for writing GPU kernels in terms of *blocks* of data (`tl.load` / `tl.store` on pointer blocks, `tl.dot`
for tile matmuls). `triton_kernels.py` has vector-add, fused softmax and a tiled matmul. They run on **CPU through Triton's
interpreter** (`TRITON_INTERPRET=1`), which executes the kernel body in Python — good for learning and debugging, slow by design
([debugging docs](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html),
[tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)). The same kernels run on a GPU when
`TRITON_FORCE_GPU=1` (untested).

**Attention, FlashAttention and SDPA.** Standard attention builds an N×N score matrix: memory O(N²). FlashAttention
([Dao et al. 2022](https://arxiv.org/abs/2205.14135); [v2](https://arxiv.org/abs/2307.08691)) computes the *same* result by streaming
K/V in tiles and keeping a running max and sum (the "online softmax" trick of
[Milakov & Gimelshein 2018](https://arxiv.org/abs/1805.02867)), so extra memory is O(N) — and, in a fused GPU kernel, it keeps tiles in
fast on-chip SRAM, which is where the speed comes from. `flash_attention_from_scratch.py` implements the tiled algorithm in plain
PyTorch, verifies it against naive attention in float64, and measures peak memory in separate processes. (Pure-PyTorch tiling is *not*
faster than naive on CPU — it demonstrates the math and the memory property.) In practice you call
`torch.nn.functional.scaled_dot_product_attention` (SDPA), which dispatches to a fused kernel when available
([docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)); `sdpa_backends.py` forces
each backend and reports which exist here.

## Laptop / offline path (tested)

```bash
python profile_torch.py                       # operator table + Chrome trace
source ../tools/env_inductor_cpu_nocompiler.sh # ONLY if you have no g++ (see file header); otherwise install g++/MSVC build tools
OMP_NUM_THREADS=1 python compile_bench.py --threads 1 --steps 20
python triton_kernels.py                      # Triton interpreter, correctness self-check
python flash_attention_from_scratch.py        # exactness, timing, peak memory
python sdpa_backends.py                       # which SDPA backends are available
./run_area2.sh                                # all of the above as used for RESULTS.md
```

## Cloud-GPU path — [`cloud/`](cloud/) — UNTESTED

The cloud path uses the same scripts with the option `--device cuda`.
The cloud folder also has notes about FlashAttention-2 (`pip install flash-attn`, which needs the CUDA toolkit) and about Nsight Systems.

## Honest limitations

- There was no GPU, so nothing about CUDA kernel behaviour, occupancy, the memory hierarchy or FlashAttention speed-ups was measured. Please do not guess GPU speed from these CPU tables.
- The `torch.compile` results on the CPU come from an improvised toolchain, which is described in `tools/env_inductor_cpu_nocompiler.sh`.
