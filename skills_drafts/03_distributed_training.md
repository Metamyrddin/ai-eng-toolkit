# Skill draft: area 3, distributed training

## When to use this area

Use this area when someone asks how training is split over several processes or GPUs, what DDP, FSDP and ZeRO do, why mixed precision needs loss scaling, or how much memory a model needs per GPU.

## Exact commands

Run these commands from the folder `03_distributed_training`.

```bash
torchrun --nproc_per_node=2 ddp_cpu.py          # Checks that two-process DDP equals one-process training.
torchrun --nproc_per_node=2 fsdp_cpu.py         # Checks FSDP2 sharding and exactness.
torchrun --nproc_per_node=2 collectives_demo.py # Checks and times the collective operations.
python launch_spawn.py ddp_cpu.py --world 2     # Does the same without torchrun, for example on Windows.
python mixed_precision_demo.py                  # Shows float16 underflow, loss scaling and bfloat16.
python zero_memory.py --params-b 7 --gpus 8     # Prints the memory per GPU for each ZeRO stage.
```

## Known traps and gotchas

- All distributed runs use two CPU processes with the gloo backend. Their timings say nothing about NCCL, NVLink or InfiniBand, and the model has only about 30 thousand parameters, so do not draw conclusions about scaling.
- Each script checks its own exactness claim and exits with an error if the claim is false, so a non-zero exit code means a real failure.
- On a CPU, bfloat16 autocast was slower than float32 for the tiny model, because the CPU has no bfloat16 hardware. This says nothing about GPUs.
- The ZeRO calculator counts only the model states. Activations, buffers and memory fragmentation come on top.
- The DeepSpeed files were only parsed by DeepSpeed's configuration class, and no GPU training was ever run with them. The file with `"auto"` values is meant for the Hugging Face Trainer, which fills the values in.
- DeepSpeed and Triton have no normal Windows installers, so use WSL2 on Windows.

## General traps that apply to every area

- Run every command inside the project's virtual environment, which you turn on with `source .venv/bin/activate` from the project folder.
- Run `python tools/download_models.py` once while you are online, and then set `export HF_HUB_OFFLINE=1` to work offline.
- The tests set the variable `AIETK_TEST_MODE`, which sends every results file to a temporary folder. This protects the real results from being overwritten, so do not be surprised that a test run leaves `results/` unchanged.
- Timings come from a shared virtual machine with 8 CPU cores, so they are noisy. Compare ratios and confidence intervals and do not quote single absolute numbers.
- Nothing has been run on a GPU, on Windows or on macOS. Treat the cloud scripts and the setup scripts for those systems as UNTESTED.
