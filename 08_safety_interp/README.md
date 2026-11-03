# 8 · Safety & Interpretability

This area shows how to test how models behave when someone attacks them, and how to look inside a model.

## Mechanisms in plain words

**Red-teaming.** Systematically try to make a model misbehave and *count* how often it does. `redteam_harness.py` has five probe families (50 hand-written prompts in `data/redteam_prompts.jsonl`):
direct harmful requests (does it refuse?), the same requests wrapped in role-play "jailbreak" prefixes ([Wei et al. 2023, "Jailbroken"](https://arxiv.org/abs/2307.02483)),
benign look-alikes (does it **over-refuse**?), **prompt injection** — an instruction hidden in a document the model was asked to summarise ([Greshake et al. 2023](https://arxiv.org/abs/2302.12173)) — and **secret leakage** using a canary string in the system prompt.
Everything is scored by code (regex / marker matching), so it is reproducible, but a refusal regex is only a proxy: *not refusing* ≠ *producing harmful content*, and tiny models often emit nonsense. The harness
prints those caveats. For serious evaluations use [HarmBench](https://arxiv.org/abs/2402.04249) or [AdvBench/GCG](https://arxiv.org/abs/2307.15043) on a GPU.
**The prompts are deliberately mild** (they test the *refusal behaviour*, not provide harmful instructions).

**Logit lens.** A transformer adds layer outputs into a "residual stream"; the last step multiplies it by the unembedding matrix to get next-token scores. The logit lens ([nostalgebraist 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)) applies that final step
to *intermediate* layers to see what the model "would predict if it stopped here". Caveat: early layers aren't trained for that, so it can mislead — the Tuned Lens ([Belrose et al. 2023](https://arxiv.org/abs/2303.08112)) fixes this with learned per-layer translators.
Our implementation is checked: the last-layer lens equals the model's own logits.

**Activation patching.** To find *where* a fact is computed: run a "clean" prompt and a "corrupted" one (e.g. different subject), copy the clean hidden state into the corrupted run at one (layer, position), and measure how much of the clean answer comes back
([Meng et al. 2022](https://arxiv.org/abs/2202.05262), [Wang et al. 2022](https://arxiv.org/abs/2211.00593), practical guide: [Heimersheim & Nanda 2024](https://arxiv.org/abs/2404.15255)). Sanity check built in: patching *everything* must give recovery 1.0.

## Laptop / offline path (tested, CPU)
```bash
python redteam_harness.py                      # 3 small instruct models, 50 prompts each
python logit_lens.py                           # lens + patching on SmolLM2-135M
```
## Cloud path — [`cloud/README.md`](cloud/README.md) — UNTESTED
The cloud README explains how to run the same harness against models with 7 to 8 billion parameters through vLLM and how to add HarmBench.
It also explains how to run the logit lens and patching on larger models with TransformerLens or nnsight.

## Honest limitations
- With only 50 prompts the confidence intervals are wide, and we report the Wilson intervals. This is a demonstration of a test harness and not a safety benchmark.
- The patching results come from a model with 135 million parameters and only a handful of prompts, so they are illustrations and not discoveries of real circuits.
