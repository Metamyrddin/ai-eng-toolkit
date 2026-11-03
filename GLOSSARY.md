# Glossary

This glossary explains the technical words of the toolkit in everyday language.
The words are in alphabetical order, so you can use your browser's search to find one quickly.

Each entry is written for a reader who is new to AI engineering.

## Activation patching

A way to find which part of a model matters for an answer. You run the model on two prompts, copy a hidden value from one run into the other run, and see whether the answer changes.

## Agent

A language model that is allowed to use tools, such as a calculator or a search box. It reads the tool result and then decides what to do next.

## All-reduce

A step in which every process contributes a number or a list of numbers, and every process receives the sum (or the average) of all of them. DDP uses it to combine gradients.

## Attention

The part of a transformer that lets each word look at the other words and decide which ones matter for the next prediction.

## Batch

A group of examples that the computer handles at the same time, which is faster than handling them one by one.

## Batching (static and continuous)

Static batching waits until the slowest request in a group is finished. Continuous batching lets finished requests leave and new requests join at every step, so the hardware is never idle for long.

## Benchmark

A fixed test that is used to measure and compare speed or quality.

## BM25

A classic keyword search formula. It gives a higher score to documents that contain the rare words of your question.

## Bootstrap (confidence interval)

A method that repeats the evaluation many times on randomly re-picked copies of your data, to see how much the result could change by luck.

## Checkpoint

A saved copy of a model's weights, so that you can continue or reuse the model later.

## Confidence interval

A range of values that is likely to contain the true number. A narrow range means we are fairly sure, and a wide range means the result could be mostly luck.

## Contamination

The problem that questions of a test were also in the training text, so the model may have memorised the answers and the test is no longer fair.

## CPU and GPU

The CPU is the general-purpose processor of every computer. The GPU is a graphics card that can do huge numbers of simple calculations at once, which is why it is used to train and run AI models.

## Cross-encoder

A model that reads a question and a document together and gives one relevance score. It is slower but more accurate than comparing two separate embeddings.

## DDP (DistributedDataParallel)

A way to train with several processes. Each process holds a full copy of the model and works on a different part of the data, and they average their gradients after every step.

## Decode and prefill

Prefill is the first phase, in which the model reads the whole prompt at once. Decode is the second phase, in which the model writes one new token at a time.

## DeepSpeed

A library from Microsoft that makes it possible to train large models by sharing the memory load between GPUs. Its ZeRO method is the best known part.

## Dense retrieval

Searching by meaning. Texts are turned into embedding vectors, and the texts whose vectors are closest to the question are returned.

## DPO (Direct Preference Optimization)

A way to teach a model from pairs of answers, where one answer is better than the other. The model learns to make the better answer more likely than the worse one.

## Embedding

A list of numbers that stands for the meaning of a word or a text. Texts with similar meaning get similar lists.

## Epoch

One full pass over all the training examples.

## FAISS

A library from Meta that finds the closest vectors in a large collection very quickly.

## FlashAttention

A way to compute attention that never builds the huge table of all scores at once. It gives the same answer with much less memory.

## FSDP (Fully Sharded Data Parallel)

A PyTorch method in which each process stores only a slice (shard) of the model, and the slices are collected just in time for each layer. It saves memory compared with DDP.

## GGUF

A file format for models that llama.cpp uses. It can store weights in many shrunk (quantized) forms.

## Gloo

A communication backend that lets processes exchange data on a CPU. We use it instead of NCCL, because NCCL needs GPUs.

## Gradient

A list of numbers that tells the training how to change each weight to make the model a little less wrong.

## Hugging Face

A website and a set of Python libraries where models and datasets are shared. Most of the models in this toolkit come from there.

## Inference

Using a trained model to produce answers, as opposed to training it.

## Jaccard similarity

A score between 0 and 1 that tells how much two sets overlap. It is the size of the overlap divided by the size of everything that is in either set.

## Jailbreak

A trick in the prompt that tries to make a model ignore its safety rules.

## KV cache

A memory of the keys and values that the attention layers already computed for earlier words. With it, the model does not repeat that work for every new word, which makes writing text much faster, but the cache uses a lot of memory.

## llama.cpp

A fast program written in C++ that runs language models on ordinary computers. It includes llama-server and llama-bench.

## Logit lens

A trick to peek inside a model. We take the hidden values of a middle layer and ask what word the model would predict if it had to stop at that layer.

## LoRA (Low-Rank Adaptation)

A cheap way to fine-tune a model. The original weights stay frozen, and we train only two small extra matrices whose product is added to the original weights.

## Loss

A number that measures how wrong the model is on the training examples. Training tries to make it smaller.

## LSH (Locality-Sensitive Hashing)

A trick that puts similar items into the same bucket, so that we only compare items that share a bucket instead of comparing every pair.

## McNemar test

A significance test for comparing two models on the same questions. It looks only at the questions on which exactly one of the two models was right.

## MinHash

A trick that shrinks a large set of words into a short list of numbers, so that two such lists can be compared quickly to guess how similar the original texts are.

## Mixed precision

Doing most of the calculations with smaller number formats such as float16 or bfloat16, to save memory and time, while keeping the important parts in float32.

## NCCL

NVIDIA's communication library that lets GPUs exchange data very fast. Real multi-GPU training uses it.

## nDCG

A score for a ranked list of search results that gives more credit for putting relevant results near the top.

## Perplexity

A score for how surprised a language model is by a text. A lower number means that the model predicts the text better.

## Prompt

The text that you give to a model before it starts writing.

## Prompt injection

An attack in which instructions are hidden inside a document or web page that the model is asked to read, hoping that the model obeys them.

## Quantization

Storing the weights of a model with fewer bits, for example 4 bits instead of 32. The file gets much smaller and faster, and the quality gets a little worse.

## RAG (Retrieval-Augmented Generation)

A method in which the system first searches for documents that fit the question and then lets the model write the answer using those documents.

## Red-teaming

Deliberately trying to make a model behave badly, in order to find its weak points before real attackers do.

## Reranker

A second, more careful model that re-orders a short list of search results.

## Reward model

A model that gives a score to an answer to say how good people would find it. It is trained from pairs of a better and a worse answer.

## RRF (Reciprocal Rank Fusion)

A simple way to merge several ranked lists. Each document gets points that are larger for higher positions, and the points are added up.

## RTN (round to nearest)

The simplest form of quantization, in which every weight is just rounded to the closest value that the smaller format can store.

## Seed

A starting number for a random number generator. Using the same seed makes a run repeatable.

## SFT (Supervised Fine-Tuning)

Training a model further on examples of questions with the answers that we want it to give.

## Shard

One piece of a big file or model that has been split into several pieces.

## Slot

One of the parallel conversations that a llama.cpp server can handle at the same time.

## Speculative decoding

A speed-up in which a small fast model guesses several next words and the big model checks all the guesses in one go. The result is exactly what the big model alone would have produced.

## Streaming

Reading or sending data piece by piece instead of all at once. A streaming dataset never needs to fit into memory, and a streaming answer shows words as they are produced.

## Tokens and tokenizer

A token is a small piece of text, often a word or part of a word. The tokenizer is the program that cuts text into tokens and turns them into numbers.

## torch.compile

A PyTorch feature that rewrites your model into faster fused code the first time it runs.

## Transformer

The design of neural network that almost all modern language models use. Its key part is attention.

## Triton

A language for writing GPU programs (kernels) in Python. Its interpreter mode lets us run those programs on a CPU for learning and testing.

## TTFT (time to first token)

How long a user waits until the first word of the answer appears.

## vLLM

A popular open-source server for running language models fast on GPUs. It uses continuous batching and a memory trick called PagedAttention.

## Weights

The numbers inside a model that were learned during training. They are what makes the model behave the way it does.

## Wilson interval

A good way to compute a confidence interval for a percentage, such as accuracy. It behaves well even for small samples and for results near 0 or 100 percent.

## ZeRO

A method from DeepSpeed that stops every GPU from storing the same data. In its three stages it splits the optimizer state, then the gradients, then the weights across the GPUs.
