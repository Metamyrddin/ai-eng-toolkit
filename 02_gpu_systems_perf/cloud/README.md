# Area 2 cloud path: running the speed experiments on a GPU (UNTESTED)

No GPU was available when this toolkit was built, so nothing in this folder has been run on a GPU.
Every script in the parent folder accepts the option `--device cuda`, so the same scripts are the cloud path.

Run these commands on a machine that has an NVIDIA GPU:

```bash
python profile_torch.py --device cuda          # Adds the CUDA kernel times to the profiler table.
python compile_bench.py --device cuda          # Lets Inductor write Triton kernels for the GPU.
python sdpa_backends.py --device cuda --dtype bfloat16 --seq 4096   # Tries each attention backend on the GPU.
TRITON_FORCE_GPU=1 python triton_kernels.py    # Runs the same kernels on the real GPU without the interpreter.
```

Please keep these points in mind:

- None of the commands above has been run on a GPU by us, so expect to fix small things such as data types and shapes.
- The FlashAttention-2 package (`pip install flash-attn`) compiles CUDA code and needs a matching CUDA toolkit. Its project page lists the supported GPUs: <https://github.com/Dao-AILab/flash-attention>.
- For most users, the PyTorch function `scaled_dot_product_attention` already chooses a flash kernel on supported GPUs, so the extra package is often not needed.
- For a full timeline of what the GPU does, use Nsight Systems with the command `nsys profile python script.py`. Its guide is at <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>.
- The folder `tools/gpu/` can run short GPU checks on a free Kaggle or Colab GPU. Those checks are also UNTESTED until someone runs them.
