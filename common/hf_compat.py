"""This file hides small differences between versions of the transformers library, so the rest of
the code does not have to care which version is installed.

More technical notes:
Small compatibility shims for transformers API drift (verified against transformers 5.22.1).
The cache argument name was checked against the published source of transformers 5.23.0 and 5.24.0.
"""


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


def cache_argument_name() -> str:
    """Return the keyword name to use when you pass a KV cache into a model.

    A KV cache is the memory of earlier tokens. Transformers 5.24.0 renamed the
    argument that carries this memory from past_key_values to cache. The old
    name still runs on 5.24.0 and later, but Python prints a FutureWarning, and
    the old name will be removed in version 5.40.0. Older versions still need
    past_key_values. This function reads the installed version and returns the
    name that version accepts without a warning.

    Do not try the new name and catch an error. On an older version the
    Llama-style model accepts the extra name and then ignores it, so the cache
    you passed would stay empty and the answers would be wrong.

    Call a model like this. The two stars turn the dictionary into one keyword
    argument:

        model(input_ids, **{cache_argument_name(): cache}, use_cache=True)
    """
    import transformers
    from packaging.version import Version

    # 5.24.0 is the first release whose model code uses the new name.
    if Version(transformers.__version__) >= Version("5.24.0"):
        return "cache"
    return "past_key_values"
