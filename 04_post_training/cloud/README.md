# Area 4 cloud path: tuning bigger models on a GPU (UNTESTED)

The computer where the toolkit was built has no GPU, so nothing in this folder was trained.
The flags in the scripts were only checked by letting the `trl sft --help` and `trl dpo --help` commands of trl 1.17.1 parse them.
You will probably need to change the batch size or the sequence length to fit your GPU memory.
Gated models such as Llama also need you to log in with `huggingface-cli login`.

The folder contains five files:

- The script `sft_lora_trl.sh` fine-tunes a model with 7 to 8 billion parameters with LoRA, using the `trl sft` command and the bf16 number format.
- The script `sft_qlora_trl.sh` does the same with QLoRA, which keeps the base model frozen in 4 bits ([Dettmers et al. 2023](https://arxiv.org/abs/2305.14314)), and it needs the `bitsandbytes` package on a CUDA GPU.
- The script `dpo_trl.sh` runs DPO on top of the SFT result with the `trl dpo` command.
- The script `multi_gpu_accelerate.sh` runs the same SFT on several GPUs at once with `accelerate launch`.
- The file `vllm_synthetic_data.py` creates data by rejection sampling with vLLM and checks the answers with the verifier in `task.py`.

These are the official documents: the [trl command line guide](https://huggingface.co/docs/trl/clis), the [peft documentation](https://huggingface.co/docs/peft), the [accelerate documentation](https://huggingface.co/docs/accelerate) and the [vLLM offline inference guide](https://docs.vllm.ai/en/latest/serving/offline_inference/).
