# Area 3 cloud path: training on several GPUs (UNTESTED)

The computer where the toolkit was built has no GPU, so nothing in this folder has been run.

The folder contains four files:

- The script `torchrun_single_node.sh` runs `ddp_cpu.py --backend nccl` on the 8 GPUs of one machine.
- The script `torchrun_multi_node.sh` runs the training on several machines with 8 GPUs each, and you run it once on every machine.
- The script `deepspeed_launch.sh` is a template for starting your own DeepSpeed training script together with one of the files in `../deepspeed_configs/`.
- The file `accelerate_fsdp_config.yaml` is a template for Accelerate with FSDP, and you should create a fresh one with `accelerate config` for your installed version.

Here is what was checked without a GPU:

- The logic of the Python scripts was run on 2 CPU processes with the gloo backend.
- The DeepSpeed JSON files were parsed with DeepSpeed's own configuration class in `tests/test_deepspeed_configs.py`.

Here is what was not checked:

- Any run with NCCL, any real DeepSpeed or FSDP training on GPUs, the multi-node start-up, and how well the speed scales.

These are the official documents:

- The torchrun documentation is at <https://docs.pytorch.org/docs/stable/elastic/run.html>.
- The DeepSpeed configuration reference is at <https://www.deepspeed.ai/docs/config-json/>.
- The Accelerate guide for FSDP is at <https://huggingface.co/docs/accelerate/usage_guides/fsdp>.
