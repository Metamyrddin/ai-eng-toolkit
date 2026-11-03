# Area 6 scale-up notes (UNTESTED)

This file contains notes only, and no code was run at this scale.

- The library datatrove ([GitHub](https://github.com/huggingface/datatrove)) is what Hugging Face used to build FineWeb. It offers building blocks for readers, filters, MinHash duplicate removal, tokenizers and writers, and it can run on one computer or on a Slurm cluster. Our single-machine `minhash_dedup.py` is the easy-to-read version of the same idea.
- For distributed MinHash, you compute the signatures for each shard in parallel, group the documents by their band hashes across all shards, and then join the candidate pairs with union-find. The FineWeb paper describes this ([Penedo et al. 2024](https://arxiv.org/abs/2406.17557)).
- For training on several GPUs, the script `streaming_pipeline.py` writes token shards of a fixed size as `.npy` files, and each process can read the shards `shards[rank::world_size]`. For Hugging Face streaming datasets, use `datasets.IterableDataset.shard` or `split_dataset_by_node` ([documentation](https://huggingface.co/docs/datasets/stream#split-dataset)).
- For cloud storage, write the shards to an object store and stream them back later. Keep a manifest file that lists the number of documents and tokens in each shard.
