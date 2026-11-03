"""This file hides small differences between versions of the transformers library, so the rest of
the code does not have to care which version is installed.

More technical notes:
Small compatibility shims for transformers API drift (verified against transformers 5.22.1)."""


def cache_seq_len(cache) -> int:
    """Return how many tokens are stored in a Hugging Face KV cache, which works with different
    versions of the transformers library.
    """
    return int(cache.get_seq_length())


def cache_drop_last(cache, n: int) -> None:
    """Remove the last `n` positions from a DynamicCache.

    In transformers 5.x `DynamicCache.crop(tokens_to_remove)` takes a NEGATIVE count (verified by reading
    the installed source); in 4.x `crop(max_length)` took the absolute length to keep.
    """
    if n <= 0:
        return
    try:
        cache.crop(-n)
    except ValueError:
        cache.crop(cache_seq_len(cache) - n)
