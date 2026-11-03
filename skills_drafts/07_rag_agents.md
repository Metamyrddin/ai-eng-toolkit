# Skill draft: area 7, RAG and agents

## When to use this area

Use this area when someone wants to build or measure document search (keyword, embedding, hybrid and reranked), a question-answering system with citations, or an agent that calls tools with a small local model.

## Exact commands

Run these commands from the folder `07_rag_agents`.

```bash
python retrieval_eval.py                     # Compares BM25, dense, hybrid and reranked search on SciFact.
python rag_pipeline.py --n 30                # Answers 30 questions and checks the citations.
python agent_loop.py --eval                  # Runs the 29 agent tasks with Qwen2.5-0.5B-Instruct.
python agent_loop.py --ask "What is 17% of 2340?"   # Asks one question.
```

## Known traps and gotchas

- The cross-encoder reranker gave no gain on SciFact, and it made each query about 19 times slower. Do not add a reranker without measuring it on your own data.
- Reciprocal rank fusion was significantly better than BM25 alone and than dense search alone (nDCG@10 of 0.685 against 0.652 and 0.648). The weighted hybrid has an untuned weight, so trust it less.
- The 0.5B agent wrote valid tool-call JSON 31 times out of 31, but it called a tool in 4 of 5 questions that needed none, and it once dropped the parentheses in a calculation.
- The RAG script measures only things that code can check, namely whether the right document reached the prompt and whether the citations point to real passages. It makes no claim that the answers are correct.
- The calculator tool never uses `eval`. It walks the parsed expression and allows only basic arithmetic, so keep it that way when you add tools.
- The agent talks to llama-cpp-python directly. There is no option to point it at another server yet.

## General traps that apply to every area

- Run every command inside the project's virtual environment, which you turn on with `source .venv/bin/activate` from the project folder.
- Run `python tools/download_models.py` once while you are online, and then set `export HF_HUB_OFFLINE=1` to work offline.
- The tests set the variable `AIETK_TEST_MODE`, which sends every results file to a temporary folder. This protects the real results from being overwritten, so do not be surprised that a test run leaves `results/` unchanged.
- Timings come from a shared virtual machine with 8 CPU cores, so they are noisy. Compare ratios and confidence intervals and do not quote single absolute numbers.
- Nothing has been run on a GPU, on Windows or on macOS. Treat the cloud scripts and the setup scripts for those systems as UNTESTED.
