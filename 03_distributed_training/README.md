# 3 · Distributed Training

This area explains how one model is trained on many processes or GPUs.
We practise it here with 2 CPU processes and the `gloo` communication backend.

## Mechanisms in plain words

**Data parallelism (DDP).** Every process keeps a full copy of the model and processes a different slice of the batch. During
backward, gradients are **averaged with an all-reduce**, so all copies take the same optimizer step and stay identical
([DDP docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)).
`ddp_cpu.py` proves this: after 60 SGD steps the 2-process result equals single-process training on the whole batch (max parameter
difference measured in RESULTS.md) and both replicas are bit-identical.

**Sharding (ZeRO / FSDP).** With plain DDP every GPU stores parameters, gradients *and* optimizer state. ZeRO
([Rajbhandari et al. 2019](https://arxiv.org/abs/1910.02054)) removes this redundancy in three stages: shard optimizer state (stage 1),
+ gradients (2), + parameters (3). Before a layer runs its parameters are **all-gathered**; afterwards gradients are
**reduce-scattered** so each rank keeps only its shard. PyTorch's implementation is FSDP
([paper](https://arxiv.org/abs/2304.11277); [`fully_shard` docs](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)).
`fsdp_cpu.py` verifies that each rank stores half the parameters and that training still matches the single-process reference.
`zero_memory.py` implements the paper's per-GPU memory formulas (unit-tested against the paper's Figure 1: 120 / 31.4 / 16.6 / 1.9 GB for a
7.5 B model on 64 GPUs).

**Collectives.** `collectives_demo.py` checks the semantics of `all_reduce`, `all_gather`, `reduce_scatter`, `broadcast`
([docs](https://docs.pytorch.org/docs/stable/distributed.html)) and times all-reduce on localhost.

**Mixed precision.** fp16 has a small range (max 65504, smallest normal 6.1e-5): tiny gradients underflow to 0 unless the loss is
scaled up first ("loss scaling", [Micikevicius et al. 2017](https://arxiv.org/abs/1710.03740)). bfloat16 has fp32's range but fewer mantissa
bits, so it needs no scaling. `mixed_precision_demo.py` shows both effects with real tensors and measures bf16-autocast loss error on CPU
([AMP docs](https://docs.pytorch.org/docs/stable/amp.html)).

## Laptop / offline path (tested: 2 processes, gloo, CPU)

```bash
torchrun --nproc_per_node=2 ddp_cpu.py          # DDP exactness check
torchrun --nproc_per_node=2 fsdp_cpu.py         # FSDP2 sharding + exactness check
torchrun --nproc_per_node=2 collectives_demo.py
python launch_spawn.py ddp_cpu.py --world 2     # same without torchrun (handy on Windows)
python mixed_precision_demo.py
python zero_memory.py --params-b 7 --gpus 8
```
Each distributed script checks its own exactness claim with an `assert` and stops with an error code if the claim is false.
The test file `tests/test_distributed.py` also runs these scripts.

## Cloud / multi-GPU path — [`cloud/`](cloud/) and [`deepspeed_configs/`](deepspeed_configs/) — **GPU runs UNTESTED**

- The scripts `cloud/torchrun_single_node.sh` and `cloud/torchrun_multi_node.sh` contain the launch commands. The option `ddp_cpu.py --backend nccl` assigns each process to one GPU, but we never ran it on a GPU.
- The files in `deepspeed_configs/` are settings for ZeRO-2, ZeRO-3, ZeRO-3 with CPU offload (in bf16), and a template for the Hugging Face Trainer that uses `"auto"` values.
  The key names were checked against the [DeepSpeed configuration documentation](https://www.deepspeed.ai/docs/config-json/), and DeepSpeed's own configuration class parses every file on the CPU in `tests/test_deepspeed_configs.py`.
  Parsing a file is not the same thing as a successful training run on a GPU.
- The file `cloud/accelerate_fsdp_config.yaml` is a template for [Accelerate FSDP](https://huggingface.co/docs/accelerate/usage_guides/fsdp). The field names change between versions, so create a fresh file with `accelerate config`.

## Honest limitations

- Timings of gloo on one computer say nothing about how NCCL, NVLink or InfiniBand would scale.
- The model is tiny (about 30 thousand parameters) on purpose, because that makes the exactness checks fast. We make no claim about scaling efficiency.
- Tensor parallelism and pipeline parallelism (Megatron-LM, [Shoeybi et al. 2019](https://arxiv.org/abs/1909.08053)) are described but not implemented.
