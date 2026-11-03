# Skill draft: area 6, data engineering

## When to use this area

Use this area when someone wants to clean text for model training: removing near-duplicates, filtering bad documents, streaming a large dataset and packing tokens into files.

## Exact commands

Run these commands from the folder `06_data_engineering`.

```bash
python fetch_sample.py --n 3000        # Downloads real FineWeb text once. Needs internet.
python minhash_dedup.py                # Plants near-duplicates, removes them and measures recall.
python quality_filters.py              # Tests the filters on planted bad documents.
python streaming_pipeline.py --n-docs 3000 --source cache   # Runs the whole pipeline from the cached file.
```

## Known traps and gotchas

- The script `fetch_sample.py` ends with `os._exit(0)` on purpose. The background download threads of the `datasets` library can crash with the message "terminate called without an active exception" while Python shuts down, and the file is already complete by then.
- A recall of 74 percent at the 0.8 threshold is expected and is not a bug. The LSH step follows an S-curve, so pairs just above the threshold are often missed and pairs above about 0.9 are almost always found. Change the numbers of bands and rows if you need a different balance.
- The perplexity filter only catches text that is less predictable than normal text, such as random symbols. Repetition, shouting and keyword lists get a lower perplexity than real text, so use the filter together with the rule-based checks.
- FineWeb has already been filtered upstream, so almost every real document passes. The test with planted bad documents is the one that tells the filters apart.
- Our duplicate remover keeps everything in memory. For more than ten million documents use a tool such as datatrove.
- The comparison of speed with the `datasketch` library is not a fair benchmark, because the two programs use different data structures.

## General traps that apply to every area

- Run every command inside the project's virtual environment, which you turn on with `source .venv/bin/activate` from the project folder.
- Run `python tools/download_models.py` once while you are online, and then set `export HF_HUB_OFFLINE=1` to work offline.
- The tests set the variable `AIETK_TEST_MODE`, which sends every results file to a temporary folder. This protects the real results from being overwritten, so do not be surprised that a test run leaves `results/` unchanged.
- Timings come from a shared virtual machine with 8 CPU cores, so they are noisy. Compare ratios and confidence intervals and do not quote single absolute numbers.
- Nothing has been run on a GPU, on Windows or on macOS. Treat the cloud scripts and the setup scripts for those systems as UNTESTED.
