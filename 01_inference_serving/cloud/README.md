# Area 1 cloud path: serving with vLLM (UNTESTED)

This folder shows how to run the serving experiments of area 1 on a computer with an NVIDIA GPU.
Nothing in this folder has been run, because the computer where the toolkit was built has no GPU.
The vLLM flags were compared with the official documentation, but they were never executed.

These are the documents we used to check the flags:

- The vLLM command line reference is at <https://docs.vllm.ai/en/latest/cli/serve/>.
- The page about speculative decoding is at <https://docs.vllm.ai/en/latest/features/speculative_decoding/>.
- The page about the OpenAI-compatible server is at <https://docs.vllm.ai/en/latest/serving/openai_compatible_server/>.

The folder contains four scripts:

- The script `vllm_serve.sh` starts a vLLM server that uses tensor parallelism and prefix caching.
- The script `vllm_serve_speculative.sh` starts a server that uses a draft model or an n-gram method for speculative decoding.
- The script `vllm_quantized.sh` starts a server for a model that was already quantized with AWQ, GPTQ or FP8.
- The script `bench_cloud.sh` runs the same `bench_serving.py` client that we used on the laptop, this time against vLLM.

If one of the scripts fails, trust the output of `vllm serve --help` on your installed version more than these templates.
