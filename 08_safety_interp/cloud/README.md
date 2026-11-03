# Area 8 cloud path: bigger red-team tests and bigger models (UNTESTED)

There was no GPU when this toolkit was built, so nothing here has been run on a GPU.

- For red-teaming at a larger scale, run HarmBench ([paper](https://arxiv.org/abs/2402.04249), [code](https://github.com/centerforaisafety/HarmBench)) or AdvBench ([Zou et al. 2023](https://arxiv.org/abs/2307.15043)) against models with 7 to 8 billion parameters that are served by vLLM. Keep the design of our harness, which scores the answers with code, adds Wilson confidence intervals and compares the wrapped prompt with the direct prompt.
- For interpretability on larger models, TransformerLens ([GitHub](https://github.com/TransformerLensOrg/TransformerLens)) and nnsight ([website](https://nnsight.net/)) provide hooks for many model designs. The hooks in `logit_lens.py` only fit Llama-style models, because they use `model.model.layers`, `model.model.norm` and `lm_head`.
- For a more faithful look at the middle layers, use the Tuned Lens ([Belrose et al. 2023](https://arxiv.org/abs/2303.08112), [code](https://github.com/AlignmentResearch/tuned-lens)).
