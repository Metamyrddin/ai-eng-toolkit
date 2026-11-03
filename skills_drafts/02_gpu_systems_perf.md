# Skill draft: area 2, GPU systems and performance

## When to use this area

Use this area when someone asks where the time and memory of a training step go, or wants to learn about the profiler, `torch.compile`, Triton kernels, FlashAttention and the attention functions of PyTorch.

## Exact commands

Run these commands from the folder `02_gpu_systems_perf`.

```bash
python profile_torch.py                        # Prints the slowest operations and saves a Chrome trace.
source ../tools/env_inductor_cpu_nocompiler.sh # Only needed if the computer has no g++ compiler.
OMP_NUM_THREADS=1 python compile_bench.py --threads 1 --steps 20   # Compares eager and compiled code.
python triton_kernels.py                       # Runs the Triton kernels in interpreter mode and checks them.
python flash_attention_from_scratch.py         # Checks exactness, speed and memory of the tiled attention.
python sdpa_backends.py                        # Lists which attention backends work here.
./run_area2.sh                                 # Runs everything as used for RESULTS.md.
```

## Known traps and gotchas

- The command `torch.compile` on the CPU needs a C++ compiler. The build computer had none, so we used an improvised toolchain based on Zig. On a normal computer, install `g++` (Linux) or the Visual Studio build tools (Windows) instead.
- The 8-thread compile result is not valid evidence, because the compiled code was about 40 times slower than eager mode, probably due to two different OpenMP runtimes in one process. Only the 1-thread result (about 1.05 times) counts.
- The Triton interpreter needs the environment variable `TRITON_INTERPRET=1` to be set before Triton is imported. The test file and `conftest.py` take care of this, but your own scripts must do it too.
- Our tiled attention is exact but slower than the plain version on a CPU. Its value is the memory saving (18.6 MB instead of 1072 MB extra at sequence length 4096), not the speed.
- The raw profiler trace is 38 MB, so it is not stored in the GitHub repository. Run `profile_torch.py` to create it again.
- On the CPU only the MATH and the CPU flash backends of PyTorch's attention function are available. The others report an error, which is expected.

## General traps that apply to every area

- Run every command inside the project's virtual environment, which you turn on with `source .venv/bin/activate` from the project folder.
- Run `python tools/download_models.py` once while you are online, and then set `export HF_HUB_OFFLINE=1` to work offline.
- The tests set the variable `AIETK_TEST_MODE`, which sends every results file to a temporary folder. This protects the real results from being overwritten, so do not be surprised that a test run leaves `results/` unchanged.
- Timings come from a shared virtual machine with 8 CPU cores, so they are noisy. Compare ratios and confidence intervals and do not quote single absolute numbers.
- Nothing has been run on a GPU, on Windows or on macOS. Treat the cloud scripts and the setup scripts for those systems as UNTESTED.
