# Area 7 cloud path: stronger models and bigger indexes (UNTESTED)

There was no GPU when this toolkit was built, so nothing here has been run on a GPU.
The client code is the same code that ran on the CPU.

1. Serve a stronger model that can call tools, using vLLM's OpenAI-compatible server with tool calling turned on.
   ```bash
   vllm serve Qwen/Qwen2.5-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
   ```
   The flags come from vLLM's [tool calling documentation](https://docs.vllm.ai/en/latest/features/tool_calling/).
2. Point an OpenAI-compatible client at `http://localhost:8000/v1` and reuse the tools of `agent_loop.py`. The file `agent_loop.py` talks to llama-cpp-python directly, and moving it to the `openai` client takes only a few lines, but that change was not made or tested.
3. For retrieval, use a bigger embedding model such as `BAAI/bge-base-en-v1.5` by changing the model id in `retrieval_lib.py`, and use `faiss-gpu` for large indexes. Run `retrieval_eval.py` again and compare it with the tables in `RESULTS.md`. Do not assume that a bigger model is better on SciFact.
