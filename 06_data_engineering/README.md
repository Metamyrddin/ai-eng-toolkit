# 6 · Data Engineering for LLMs

Most of a model's quality comes from its data. The tools in this area clean text, remove duplicates and stream large text files.

## Mechanisms in plain words

**Near-duplicate removal (MinHash + LSH).** Web text is full of copies; training on duplicates wastes compute and increases memorisation
([Lee et al. 2022](https://arxiv.org/abs/2107.06499)). Two documents are "near-duplicates" if their sets of word *shingles* (k consecutive words) have high **Jaccard
similarity** `|A∩B|/|A∪B|`. Comparing all pairs is O(n²), so:
1. **MinHash** ([Broder 1997](https://doi.org/10.1109/SEQUEN.1997.666900)) compresses each set to m numbers; the chance two documents agree at one position equals their Jaccard similarity.
2. **LSH banding** cuts the m numbers into b bands of r; documents sharing *any* whole band become candidates, with probability `1-(1-J^r)^b` (an S-curve).
`minhash_dedup.py` implements this from scratch, plants near-duplicates with known edit rates in real FineWeb text, checks the estimator against **exact** Jaccard,
cross-checks against the `datasketch` library, and measures recall/precision.

**Quality filtering.** `quality_filters.py` implements rule-based filters in the spirit of Gopher ([Rae et al. 2021](https://arxiv.org/abs/2112.11446)) and C4
([Raffel et al. 2020](https://arxiv.org/abs/1910.10683)) — length bounds, symbol ratios, repeated lines/n-grams, stop-word presence — plus a **language-model perplexity** signal (CCNet-style, [Wenzek et al. 2020](https://arxiv.org/abs/1911.00359), there with KenLM;
here a small neural LM). Thresholds are *our own defaults*, validated against planted garbage with known labels (recall per garbage type, AUC of perplexity) — they are not the papers' exact numbers.

**Streaming.** Real corpora (e.g. FineWeb, [Penedo et al. 2024](https://arxiv.org/abs/2406.17557)) are terabytes. `datasets.load_dataset(..., streaming=True)` yields documents lazily
([docs](https://huggingface.co/docs/datasets/stream)). `streaming_pipeline.py` chains generators: filter → dedup → tokenise → **pack** into fixed-length token rows → write `.npy` shards that can be
memory-mapped for training, and reports peak memory to demonstrate that it stays bounded. Only the dedup index grows with the data.

## Laptop / offline path (tested)
```bash
python fetch_sample.py --n 3000            # needs internet once; caches data/fineweb_sample.jsonl
python minhash_dedup.py
python quality_filters.py
python streaming_pipeline.py --n-docs 3000 --source cache
```
## Cloud / scale-up path — [`cloud/README.md`](cloud/README.md) — UNTESTED
The cloud README gives pointers to `datatrove` (the pipeline library behind FineWeb), to distributed MinHash and to ways of splitting data into shards.
It makes no claims about code beyond what was actually run.

## Honest limitations
- Our duplicate removal keeps everything in memory, using one Python dictionary per band. It is correct, but for more than ten million documents you should use a pipeline that works on disk or on a cluster.
- FineWeb was already filtered and deduplicated before we downloaded it, so most documents pass our filters. The test with planted bad documents is the one that really tells the filters apart.
