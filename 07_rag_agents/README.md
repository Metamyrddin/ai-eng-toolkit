# 7 · RAG & Agents

This area shows how to give a small language model a search engine and some tools.

## Mechanisms in plain words

**Retrieval.** There are four ideas for finding relevant text, and they work well together:
- **BM25** (sparse, keyword): scores a document by how many query words it contains, weighting rare words more and damping long documents
  ([Robertson & Zaragoza 2009](https://doi.org/10.1561/1500000019); we use the `rank_bm25` package).
- **Dense** (embeddings): a sentence-transformer maps text to vectors; relevance = cosine similarity, searched with **FAISS**
  ([FAISS](https://github.com/facebookresearch/faiss), [Johnson et al. 2017](https://arxiv.org/abs/1702.08734)). We use `sentence-transformers/all-MiniLM-L6-v2` ([docs](https://sbert.net)).
- **Hybrid** with **Reciprocal Rank Fusion** `score = Σ 1/(60 + rank_i)` ([Cormack et al. 2009](https://doi.org/10.1145/1571941.1572114)): needs no score calibration.
- **Reranker**: a cross-encoder reads (query, passage) *together* and scores them — slower but more accurate than the bi-encoder; applied only to the top-k candidates (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

**Evaluation.** `retrieval_eval.py` runs all variants on **SciFact** (BEIR, [Thakur et al. 2021](https://arxiv.org/abs/2104.08663)): 5,183 abstracts, 300 test claims. Metrics: nDCG@10, Recall@100, MRR@10 — with bootstrap
confidence intervals and *paired* tests, so you can see which differences are bigger than noise. The metric functions are unit-tested on hand-computed cases.

**RAG** ([Lewis et al. 2020](https://arxiv.org/abs/2005.11401)) = retrieve, put passages into the prompt, ask the LLM to answer *and cite*. `rag_pipeline.py` measures only what can be checked by code: whether a gold
document reached the prompt, and whether the citations the model wrote refer to passages that were really in its prompt. It does **not** claim answer correctness.

**Agents / tool calling.** The model is shown tool descriptions; when it wants a tool it emits a structured call (for Qwen2.5: a Hermes-style `<tool_call>{...}</tool_call>` block, [docs](https://qwen.readthedocs.io/en/latest/framework/function_call.html));
our loop parses it, runs the Python function, feeds the result back, and repeats until the model answers or `max_steps` is hit — the ReAct pattern ([Yao et al. 2022](https://arxiv.org/abs/2210.03629)).
Tools: a calculator built on a safe AST walker (never `eval`), unit conversion, and BM25 search over SciFact. `agent_loop.py --eval` runs 29 tasks with programmatic ground truth and saves every trace, failures included.

## Laptop / offline path (tested, CPU)
```bash
python retrieval_eval.py                    # BM25 / dense / hybrid / +rerank on SciFact
python rag_pipeline.py --n 30
python agent_loop.py --eval                 # tool-calling agent, Qwen2.5-0.5B-Instruct GGUF
python agent_loop.py --ask "What is 17% of 2340?"
```
First run downloads datasets and models (~0.6 GB); afterwards set `HF_HUB_OFFLINE=1`.

## Cloud-GPU path — [`cloud/README.md`](cloud/README.md) — UNTESTED
The cloud README describes how to put a model with 7 to 8 billion parameters behind vLLM's OpenAI-compatible server and how to use it from the agent.
It also suggests the `bge` or `e5` embedding models, a larger reranker, and FAISS-GPU or a vector database.
The current `agent_loop.py` talks to llama-cpp-python directly, so moving it to an OpenAI-style client is a small change that has not been made yet.

## Honest limitations
- A model with 0.5 billion parameters is a weak agent, so expect broken tool calls and wrong tool choices. The RESULTS file shows the real rates.
- We do not use Chroma and use only FAISS, to keep the list of packages short. The search interface in `retrieval_lib.py` needs only a few lines of change to use another store.
- SciFact is a collection of scientific claims, so the comparison of hybrid and dense search can come out the other way on other kinds of text. The BEIR paper reports exactly this kind of variation.
