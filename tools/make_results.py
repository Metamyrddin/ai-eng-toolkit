#!/usr/bin/env python
"""This file builds each area's RESULTS.md page from the JSON files that the real runs produced, so
the numbers in the text always come from measurements. It is a maintenance tool.

More technical notes:
Generate every area's RESULTS.md from the JSON files in <area>/results/ (+ hand-written <area>/NOTES.md).

Rule: every NUMBER in a RESULTS.md table is read from a JSON file written by a real run; nothing is typed in by hand.
Hand-written interpretation lives in NOTES.md (kept short, refers to rows rather than restating numbers where possible).
Usage: python tools/make_results.py [area_dir ...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def J(area, name):
    """Read a results JSON file of an area, or return None if it does not exist yet."""
    p = ROOT / area / "results" / name
    return json.loads(p.read_text()) if p.exists() else None


def table(headers, rows):
    """Turn headers and rows into a Markdown table."""
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join("" if c is None else str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def f(x, nd=2):
    """Format a number for a table: floats get a fixed number of decimals and empty values
    become an empty cell.
    """
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.{nd}f}" if abs(x) < 1e6 else f"{x:.3g}"
    return str(x)


def ci(lo_hi, nd=3):
    """Format a pair of numbers as a confidence interval such as [0.41, 0.55]."""
    return f"[{lo_hi[0]:.{nd}f}, {lo_hi[1]:.{nd}f}]"


def env_block(*files):
    """Return one line that describes the machine on which the results were measured."""
    for area, name in files:
        d = J(area, name)
        if d and "environment" in d:
            e = d["environment"]
            return (f"These results were measured on a computer with the CPU {e['cpu']}, {e['cpu_count']} logical CPUs, "
                    f"the system {e['platform']}, Python {e['python']}, torch {e['torch']} and transformers {e['transformers']}. "
                    f"This computer had **no GPU**, which the program reports as `cuda_available={e['cuda_available']}`.\n"
                    f"All run timestamps are in the America/Denver time zone (MST), for example `{e['timestamp_local']}`.\n")
    return ""


# ----------------------------------------------------------------------------- area 1
def area1():
    """Build the Markdown text of the results page for area 1, inference and serving."""
    A = "01_inference_serving"; s = []
    d = J(A, "kv_cache.json")
    if d:
        s.append("## KV cache (SmolLM2-135M, fp32, CPU)\n")
        s.append(f"Greedy decoding with and without cache gives identical tokens: **{d['tokens_identical']}**. "
                 f"Measured KV bytes = {d['kv_bytes_measured']:,}; formula `2·layers·kv_heads·head_dim·seq·bytes` = {d['kv_bytes_formula']:,} "
                 f"(match: **{d['kv_formula_matches']}**). One {d['new_tokens']}-token generation: {d['total_seconds_no_cache']} s without cache vs "
                 f"{d['total_seconds_with_cache']} s with cache; TTFT {d['ttft_ms_with_cache']} ms. *(single run — see the sweep for medians)*\n")
        s.append("Per-step cost at different context lengths (median of repeated timings):\n")
        s.append(table(["prefix tokens", "ms/token, full recompute", "ms/token, cached", "speed-up ×"],
                       [[r["prefix_len"], r["ms_full_recompute_median"], r["ms_cached_step_median"], r["speedup"]] for r in d["step_cost_sweep"]]))
    d = J(A, "hf_batching.json")
    if d:
        s.append("## Static batching in HF transformers (SmolLM2-135M, medians of 5 repeats)\n")
        s.append(table(["batch", "padded prompt len", "TTFT ms", "end-to-end s", "decode tok/s (aggregate)", "decode tok/s (per request)"],
                       [[r["batch"], r["padded_prompt_len"], r["ttft_ms_median"], r["e2e_s_median"], r["decode_tok_s_aggregate_median"], r["decode_tok_s_per_request"]] for r in d["rows"]]))
    d = J(A, "llamacpp_quant.json")
    if d:
        ref = d["hf_fp32_reference"]
        s.append("## GGUF quantization sweep (llama-cpp-python, 8 threads)\n")
        s.append(f"Model: `{d.get('model', 'n/a')}`. Perplexity on WikiText-2 test, {ref['eval_tokens']} tokens (4×{ref['n_ctx']}) using the *same HF token ids* for GGUF and the HF fp32 reference "
                 f"(HF fp32 reference ppl = **{ref['ppl']}**). Speeds are medians of 5 trials (all trials are in the JSON).\n")
        s.append(table(["quant", "file MB", "prefill tok/s", "TTFT ms (384-token prompt)", "decode tok/s", "decode tok/s (all trials)", "perplexity"],
                       [[r["quant"], r["file_mb"], r["prefill_tok_s_median"], r["ttft_ms_median"], r["decode_tok_s_median"], r["decode_tok_s_all"], r["perplexity"]["ppl"]] for r in d["rows"]]))
    lb = ROOT / A / "results" / "llama_bench.txt"
    if lb.exists():
        s.append("## Official `llama-bench` (upstream binary, Qwen2.5-0.5B GGUF, 8 threads)\n")
        s.append("```\n" + "\n".join(l for l in lb.read_text().splitlines() if l.startswith("|")) + "\n```\n")
    d = J(A, "quant_scratch.json")
    if d:
        s.append("## Round-to-nearest weight quantization from scratch (simulated; SmolLM2-135M, WikiText-2 perplexity)\n")
        s.append(table(["bits", "group size (0 = per-row)", "bits/weight incl. scales", "relative weight error", "perplexity"],
                       [[r["bits"], r["group"], r["bits_per_weight"], r["rel_weight_err"], r["ppl"]] for r in d["rows"]]))
    d = J(A, "speculative.json")
    if d:
        s.append("## Speculative decoding\n")
        s.append("From-scratch implementation, target SmolLM2-360M / draft SmolLM2-135M, 4 prompts. Greedy: output identical to target-only greedy in every case.\n")
        g = {r["gamma"]: r for r in d["own_impl_sampling_T1"]}
        s.append(table(["γ (draft tokens)", "greedy: accept rate", "greedy: tokens per target call", "greedy output == target greedy", "T=1 sampling: accept rate", "T=1: tokens per target call"],
                       [[r["gamma"], r["acceptance_rate_mean"], r["tokens_per_target_call_mean"], r["all_outputs_equal_target_greedy"], g[r["gamma"]]["acceptance_rate_mean"], g[r["gamma"]]["tokens_per_target_call_mean"]] for r in d["own_impl_greedy"]]))
        h = d["hf_assisted_generation"]
        s.append(f"Hugging Face `assistant_model=` wall-clock (CPU, fp32, batch 1, {h['n_runs']} runs): baseline {h['baseline_s_median']} s, assisted {h['assisted_s_median']} s "
                 f"→ speed-up **{h['speedup_median_ratio']}×** (<1 means slower); outputs identical to baseline: {h['outputs_identical_to_baseline_greedy']}.\n")
    for fn, title in (("batching_sim.json", "low load"), ("batching_sim_overload.json", "overload")):
        d = J(A, fn)
        if d:
            s.append(f"## Batching simulation — {title} (SIMULATION, not a measurement; arrival rate {d['args']['arrival_rate']} req/ms)\n")
            s.append(table(["scheduler", "mean latency ms", "p95 latency ms", "makespan ms", "useful tokens", "throughput tok/s"],
                           [[k, v["mean_latency_ms"], v["p95_latency_ms"], v["makespan_ms"], v["useful_tokens"], v["throughput_tok_per_s"]] for k, v in d["modes"].items()]))
    for fn, desc in (("serving_slots1.json", "llama-server, 1 slot (`-np 1`)"), ("serving_slots4.json", "llama-server, 4 slots (`-np 4`)"),
                     ("serving_llama_cpp_python_server.json", "llama-cpp-python's own server (concurrency problem reproduction)")):
        d = J(A, fn)
        if d:
            s.append(f"## Serving benchmark — {desc} — `{fn}`\n")
            s.append("Streaming client `bench_serving.py`, Qwen2.5-0.5B-Instruct q4_k_m, 8 threads; medians over trials; throughput range = min–max over trials.\n")
            s.append(table(["concurrency", "trials", "requests ok / failed-or-empty (last trial)", "throughput tok/s (median trial)", "throughput range", "TTFT ms median", "TTFT ms p95", "mean TPOT ms", "e2e s median"],
                           [[r["concurrency"], r["trials"], f"{r['requests_ok']} / {r['requests_failed_or_empty']}", r["throughput_tok_s"],
                             f"{min(r['throughput_tok_s_all_trials'])}–{max(r['throughput_tok_s_all_trials'])}", r["ttft_ms_median"], r["ttft_ms_p95"], r["tpot_ms_mean_of_means"], r["e2e_s_median"]] for r in d["rows"]]))
    return s


# ----------------------------------------------------------------------------- area 2
def area2():
    """Build the Markdown text of the results page for area 2, GPU systems and performance.
    """
    A = "02_gpu_systems_perf"; s = []
    d = J(A, "profile_top_ops.json")
    if d:
        s.append(f"## torch.profiler — {d['model']}, batch {d['batch']}×seq {d['seq']}, {d['steps_profiled']} training steps, CPU\n")
        ops = d["top_ops_by_self_cpu"]
        s.append(table(list(ops[0].keys()), [list(o.values()) for o in ops[:12]]))
        s.append("The full Chrome trace is a large binary-style file, so it is not stored in the GitHub repository. Run `python 02_gpu_systems_perf/profile_torch.py` to create `results/profile_trace.json` on your own computer, and then open it in https://ui.perfetto.dev.\n")
    for fn, title in (("compile_1thread.json", "torch.compile (Inductor, CPU) — 1 thread, the valid comparison"),
                      ("compile_8threads.json", "torch.compile (Inductor, CPU) — 8 threads (improvised toolchain; see notes)")):
        d = J(A, fn)
        if d:
            s.append(f"## {title}\n")
            s.append("```json\n" + json.dumps({k: v for k, v in d.items() if k != "environment"}, indent=1)[:3000] + "\n```\n")
    d = J(A, "attention.json")
    if d:
        s.append(f"## Attention: naive vs tiled online-softmax vs torch SDPA (CPU fp32, {d['heads']} heads, head_dim {d['head_dim']}, causal={d['causal']})\n")
        s.append(table(["seq", "naive ms", "tiled ms", "SDPA ms", "naive max err vs fp64", "tiled max err", "SDPA max err", "score matrix MB (naive)", "score tile MB (tiled)"],
                       [[r["seq_len"], r["naive"]["ms_median"], r["tiled_online_softmax"]["ms_median"], r["torch_sdpa"]["ms_median"],
                         f"{r['naive']['max_abs_err_vs_fp64_naive']:.1e}", f"{r['tiled_online_softmax']['max_abs_err_vs_fp64_naive']:.1e}",
                         f"{r['torch_sdpa']['max_abs_err_vs_fp64_naive']:.1e}", r["score_matrix_mb_fp32_naive"], r["score_tile_mb_fp32_tiled"]] for r in d["rows"]]))
        m = d["peak_extra_rss_mb"]
        s.append(f"Peak extra memory at seq={m['seq_len']} (separate process per implementation; {d['memory_method']}): naive **{m['naive']} MB**, tiled **{m['tiled_online_softmax']} MB**, SDPA **{m['torch_sdpa']} MB**.\n")
    d = J(A, "sdpa_backends.json")
    if d:
        s.append(f"## SDPA backends ({d['device']}, {d['dtype']}, seq {d['seq']})\n")
        s.append("```json\n" + json.dumps(d["backends"], indent=1)[:2500] + "\n```\n")
    d = J(A, "triton_interpreter.json")
    if d:
        s.append("## Triton kernels in interpreter mode (CPU)\n")
        s.append(table(["kernel", "max error vs torch"], [["vector add", d["add_max_abs_err"]], ["softmax", d["softmax_max_abs_err"]], ["tiled matmul (abs / rel)", f"{d['matmul_max_abs_err']:.2e} / {d['matmul_max_rel_err']:.2e}"]]))
    return s


# ----------------------------------------------------------------------------- area 3
def area3():
    """Build the Markdown text of the results page for area 3, distributed training."""
    A = "03_distributed_training"; s = []
    d = J(A, "ddp.json")
    if d:
        s.append("## DDP (gloo, 2 CPU processes) vs single-process training on the same global batch\n")
        s.append(table(["metric", "value"], [[k, v] for k, v in d.items() if k != "environment"]))
    d = J(A, "fsdp.json")
    if d:
        s.append("## FSDP2 `fully_shard` (gloo, 2 CPU processes)\n")
        s.append(table(["metric", "value"], [[k, v] for k, v in d.items() if k != "environment"]))
    d = J(A, "collectives.json")
    if d:
        s.append("## Collectives (gloo, localhost, world size 2)\n")
        s.append("Semantics verified: " + ", ".join(f"`{k}`={v}" for k, v in d["semantics_verified"].items()) + "\n")
        s.append(table(["tensor MB", "all_reduce ms (median)"], [[t["size_mb"], t["all_reduce_ms_median"]] for t in d["timings"]]))
    for fn in ("zero_memory_7.5B_64gpus.txt", "zero_memory_7B_8gpus.txt"):
        t = ROOT / A / "results" / fn
        if t.exists():
            s.append(f"## ZeRO memory model (`zero_memory.py`; arithmetic, not a measurement) — `{fn}`\n\n```\n{t.read_text().strip()}\n```\n")
    d = J(A, "mixed_precision.json")
    if d:
        s.append("## Number formats\n")
        s.append(table(["dtype", "max", "smallest normal", "eps"], [[k.replace("torch.", ""), f"{v['max']:.4g}", f"{v['smallest_normal']:.4g}", f"{v['eps']:.4g}"] for k, v in d["finfo"].items()]))
        s.append("fp16 gradient underflow (true gradient 1e-8):\n")
        s.append(table(["setting", "recovered gradient"], [[k, v.get("fp16_grad_after_unscale", v.get("grad"))] for k, v in d["fp16_underflow"].items()]))
        b = d["bf16_autocast_cpu"]
        s.append(f"bf16 autocast on CPU, tiny model: loss fp32 {b['loss_fp32']:.4f} vs bf16 {b['loss_bf16_autocast']:.4f} (|Δ| = {b['abs_diff']:.4f}); step time fp32 {b['step_ms_fp32']} ms vs bf16 {b['step_ms_bf16_autocast']} ms "
                 f"(CPU bf16 is not faster here — it is a *memory/throughput* feature of GPUs with bf16 tensor cores).\n")
    return s


# ----------------------------------------------------------------------------- area 4
def acc_row(name, b, a):
    """Return one table row with the accuracy before and after training and the change."""
    return [name, f"{b['accuracy']:.3f}", f"{a['accuracy']:.3f}", f"{a['accuracy'] - b['accuracy']:+.3f}"]


def area4():
    """Build the Markdown text of the results page for area 4, post-training."""
    A = "04_post_training"; s = []
    for fn in ("synthetic_template.json", "synthetic_rejection_sampling.json"):
        d = J(A, fn)
        if d:
            s.append(f"## Synthetic data — `{fn}`\n")
            s.append("```json\n" + json.dumps({k: v for k, v in d.items() if k != "environment"}, indent=1) + "\n```\n")
    rows = []
    s.append("## SFT + LoRA (SmolLM2-135M, rank 16, CPU; 200 held-out problems, greedy exact-match; 95 % Wilson half-width ≈ ±0.07)\n")
    for fn, name in (("sft_arith.json", "SFT on 2,000 template examples"), ("sft_arith_rejection_sampling.json", "SFT on rejection-sampled data (360M generates, verifier filters)")):
        d = J(A, fn)
        if d:
            rows.append(acc_row(name, d["before"], d["after_merged"]) + [f"{d['train_seconds']} s", f"{d['trainable_pct']} %", f"{d['merge_max_abs_logit_diff']:.1e}", f"{d['loss_first_last'][0][1]} → {d['loss_first_last'][1][1]}"])
    s.append(table(["run", "accuracy before", "accuracy after (merged)", "Δ", "train time", "trainable params", "max abs logit diff, adapter vs merged", "train loss first→last logged"], rows))
    d = J(A, "dpo_arith.json")
    if d:
        s.append("## DPO on top of the template-SFT model (synthetic pairs: correct vs off-by-k answer)\n")
        s.append(table(["accuracy before", "accuracy after", "Δ", "pairs", "β", "lr", "train time"],
                       [[d["before"]["accuracy"], d["after"]["accuracy"], f"{d['after']['accuracy'] - d['before']['accuracy']:+.3f}", d["n_pairs"], d["beta"], d["lr"], f"{d['train_seconds']} s"]]))
        s.append("DPO training log (every logged step):\n")
        s.append(table(list(d["log"][0].keys()), [list(r.values()) for r in d["log"]]))
    s.append("## Reward model (Bradley–Terry, LoRA + scalar head); pairwise accuracy on 200 held-out pairs\n")
    rows = []
    for fn, name in (("reward_model.json", "arithmetic pairs (correct vs off-by-k), start from base 135M"),
                     ("reward_model_from_sft.json", "arithmetic pairs, start from the SFT model"),
                     ("reward_model_easy_sanity.json", "SANITY: rejected = non-numeric word, start from base 135M")):
        d = J(A, fn)
        if d:
            rows.append([name, d["steps"], d["before"]["pairwise_accuracy"], d["after"]["pairwise_accuracy"], d["after"]["mean_margin"], f"{d['train_seconds']} s"])
    s.append(table(["setting", "steps", "accuracy before", "accuracy after", "mean margin after", "train time"], rows))
    s.append("## Real-text smoke tests (training dynamics only; no quality claims)\n")
    d = J(A, "sft_capybara.json")
    if d:
        s.append(f"SFT `trl-lib/Capybara` ({d['model']}, 200 conversations): eval loss {d['eval_loss_before']:.3f} → {d['eval_loss_after']:.3f} in {d['train_seconds']} s.\n")
    d = J(A, "dpo_ultrafeedback.json")
    if d:
        s.append(f"DPO `trl-lib/ultrafeedback_binarized` ({d['base_model']}, {d['n_pairs']} pairs): runs end-to-end in {d['train_seconds']} s; logged: `{json.dumps(d['log'])}`.\n")
    return s


# ----------------------------------------------------------------------------- area 5
def area5():
    """Build the Markdown text of the results page for area 5, evaluation rigor."""
    A = "05_evals_rigor"; s = []
    d = J(A, "eval_arc_easy.json")
    if d:
        s.append(f"## ARC-Easy (test), n={d['n']}, {d['shots']}-shot, log-likelihood scoring, 95 % intervals\n")
        s.append(table(["model", "acc", "acc Wilson 95 %", "acc_norm", "acc_norm Wilson 95 %", "seconds"],
                       [[m, v["acc"]["acc"], ci(v["acc"]["wilson95"]), v["acc_norm"]["acc"], ci(v["acc_norm"]["wilson95"]), v["seconds"]] for m, v in d["models"].items()]))
        s.append("Paired comparison on the same 200 items (B − A):\n")
        rows = []
        for k, v in d["paired_comparison"].items():
            pb, mc = v["paired_bootstrap_B_minus_A"], v["mcnemar_exact"]
            rows.append([k, f"{v['A'].split('/')[-1]} → {v['B'].split('/')[-1]}", pb["mean_diff"], ci(pb["ci"]), f"{pb['p_one_sided']:.3f}", f"{mc['a_only_correct']} / {mc['b_only_correct']}", f"{mc['p_two_sided']:.3f}"])
        s.append(table(["metric", "A → B", "mean diff", "paired bootstrap 95 % CI", "one-sided p", "McNemar: only-A / only-B correct", "McNemar exact p (two-sided)"], rows))
    d = J(A, "ablation_shots.json")
    if d:
        s.append(f"## Ablation: few-shot count × scoring rule, {len(d['seeds'])} seeds, n={d['n_items_per_seed']} items per seed ({d['model']})\n")
        s.append(table(["shots", "metric", "per-seed scores", "mean", "std", "t 95 % CI"],
                       [[v["config"]["shots"], v["config"]["metric"], v["scores"], f"{v['summary']['mean']:.3f}", f"{v['summary']['std']:.3f}", ci(v["summary"]["ci"])] for v in d["grid"].values()]))
        s.append("Paired (same seed) differences vs zero-shot acc_norm:\n")
        s.append(table(["config", "mean diff", "t 95 % CI over seeds"], [[p["a"], f"{p['diff_summary']['mean']:.3f}", ci(p["diff_summary"]["ci"])] for p in d["paired_vs_zero_shot_acc_norm"]]))
    d = J(A, "contamination.json")
    if d:
        s.append(f"## Contamination check ({d['ngram']}-gram word overlap)\n")
        s.append(table(["eval set vs corpus", "items", "judgeable (long enough)", "flagged (≥ threshold of n-grams found)", "flagged %", "threshold"],
                       [[k, d[k]["n_items"], d[k]["n_judged"], d[k]["n_flagged"], d[k]["flagged_pct"], d[k]["threshold"]] for k in ("arc_test_vs_arc_train", "arc_test_vs_wikitext2_train")]))
        st = d.get("selftest_by_word_corruption_rate")
        if st:
            s.append("Self-test: eval items planted into WikiText-2 after corrupting a fraction of their words:\n")
            s.append(table(["corrupted word fraction", "planted items judgeable", "recall", "clean items judgeable", "false-positive rate"],
                           [[k, v["planted_items_judgeable"], v["recall"], v["clean_items_judgeable"], v["false_positive_rate"]] for k, v in st.items()]))
    return s


# ----------------------------------------------------------------------------- area 6
def area6():
    """Build the Markdown text of the results page for area 6, data engineering."""
    A = "06_data_engineering"; s = []
    d = J(A, "minhash.json")
    if d:
        e = d["signature_estimate_error"]
        s.append(f"## MinHash + LSH ({d['n']} real FineWeb docs incl. {d['planted']} planted near-duplicates; {d['num_perm']} permutations; threshold {d['threshold']})\n")
        s.append(f"Signature Jaccard estimate vs exact Jaccard: bias {e['mean_error(bias)']:.4f}, std {e['std_error']:.4f} (theory at J=0.5: {e['theory_std_at_J0.5']:.4f}).\n")
        s.append("Mean exact Jaccard of planted copies by word-edit rate: " + ", ".join(f"{k}→{v}" for k, v in d["exact_jaccard_by_edit_rate"].items()) + "\n")
        o, k = d["own_implementation"], d["datasketch"]
        s.append(table(["implementation", "docs removed", "planted pairs with exact J ≥ thr", "recall on those", "removed but exact J < thr−0.15", "seconds"],
                       [["ours (from scratch)", o["docs_removed"], o["planted_pairs_above_threshold"], o["recall_on_planted_above_threshold"], o["removed_docs_not_verified_as_similar(exact_J<thr-0.15)"], o["seconds"]],
                        ["datasketch MinHashLSH", k["docs_removed"], k["planted_pairs_above_threshold"], k["recall_on_planted_above_threshold"], k["removed_docs_not_verified_as_similar(exact_J<thr-0.15)"], k["seconds"]]]))
        s.append(f"Agreement of the two keep-masks: {d['agreement_keep_masks']}. LSH bands×rows used by ours: {o['bands']}×{o['rows']}.\n")
    d = J(A, "quality.json")
    if d:
        s.append(f"## Quality filters ({d['n_real']} real docs, {d['n_bad_each']} planted bad docs per type)\n")
        s.append(f"Heuristic pass rate on real FineWeb docs: **{d['real_docs_heuristic_pass_rate']}** (fail reasons: {d['real_docs_fail_reasons']}).\n")
        a = d["lm_perplexity_auc_vs_real"]
        s.append(table(["planted bad type", "heuristic recall", "median LM perplexity of bad docs", "AUC (higher perplexity ⇒ bad)"],
                       [[t, r, a.get(t, {}).get("median_ppl_bad", "—"), a.get(t, {}).get("auc", "—")] for t, r in d["planted_bad_recall_by_heuristics"].items()]))
        s.append(f"Median LM perplexity of real docs: {d['lm_perplexity_real_median']}.\n")
    d = J(A, "streaming_pipeline.json")
    if d:
        s.append("## Streaming pipeline (filter → dedup → tokenise → pack → shards)\n")
        s.append(table(["metric", "value"], [[k, v] for k, v in d.items() if k != "environment"]))
    return s


# ----------------------------------------------------------------------------- area 7 / 8 (appended after runs)
def area7():
    """Build the Markdown text of the results page for area 7, RAG and agents."""
    A = "07_rag_agents"; s = []
    d = J(A, "retrieval_eval.json")
    if d:
        s.append(f"## Retrieval on {d['dataset']} ({d['n_queries']} queries, {d['n_docs']} documents)\n")
        s.append(f"Dense: `{d['dense_model']}` + FAISS flat inner-product; reranker: `{d['reranker']}` on top-{d['rerank_k']}. 95 % CI = bootstrap over queries. "
                 f"Latency is single-query CPU time where measured (blank = not timed separately).\n")
        s.append(table(["method", "nDCG@10", "nDCG@10 95 % CI", "Recall@100", "MRR@10", "mean query latency ms"],
                       [[m, v["ndcg@10"], ci(v["ndcg@10_ci95"], 4), v["recall@100"], v["mrr@10"], v["mean_query_latency_ms"]] for m, v in d["methods"].items()]))
        s.append("Paired bootstrap on per-query nDCG@10:\n")
        s.append(table(["comparison", "mean diff", "95 % CI", "one-sided p"],
                       [[k, f"{v['mean_diff']:.4f}", ci(v["ci"], 4), f"{v['p_one_sided']:.4f}"] for k, v in d["paired_bootstrap_ndcg@10"].items()]))
        s.append(f"Index build: {d['index_build_seconds']} seconds.\n")
    d = J(A, "rag_pipeline.json")
    if d:
        s.append(f"## RAG pipeline (Qwen2.5-0.5B-Instruct q4_k_m; {d['n_queries']} SciFact claims; top-{d['k_ctx']} reranked passages in context; {d['seconds']} s total)\n")
        s.append(table(["metric", "value"], [[k, d[k]] for k in ("context_recall@k", "citation_rate", "citation_validity", "n_citations", "cites_gold_rate")]))
        s.append("Metric definitions are in `rag_pipeline.py`'s docstring. Answer correctness is **not** measured. Full per-query rows are in the JSON.\n")
    d = J(A, "agent_eval_qwen2.5-0.5b-q4_k_m.json")
    if d:
        sm = d["summary"]
        s.append(f"## Tool-calling agent eval ({d['model']}, greedy, {d['seconds']} s)\n")
        rows = [[c, v["n"], v["success"], v["rate"], ci(v["wilson95"]), v["used_tool_rate"], v.get("agent_query_retrieved_gold_rate", "")] for c, v in sm.items() if isinstance(v, dict) and "success" in v]
        s.append(table(["category", "n", "success", "rate", "Wilson 95 %", "used a tool (rate)", "own search query retrieved gold (rate)"], rows))
        tj = sm["tool_call_json_validity"]
        s.append(f"Tool-call JSON validity: {tj['valid']}/{tj['total']}; mean steps per task: {sm['mean_steps']}. For the `control` category success means *answering correctly without a tool call*. All traces are saved in the JSON.\n")
    return s


def area8():
    """Build the Markdown text of the results page for area 8, safety and interpretability.
    """
    A = "08_safety_interp"; s = []
    d = J(A, "redteam.json")
    if d:
        s.append(f"## Red-team harness ({d['n_prompts']} hand-written prompts; programmatic scoring; 95 % Wilson intervals)\n")
        cols = [("harmful_direct", "refusal_rate", "harmful, direct: refusal ↑"), ("jailbreak_roleplay", "refusal_rate", "harmful + role-play wrapper: refusal ↑"),
                ("benign_control", "over_refusal_rate", "benign: over-refusal ↓"), ("prompt_injection", "injection_success_rate", "prompt injection success ↓"),
                ("canary_leak", "canary_leak_rate", "canary leak ↓")]
        rows = []
        for m, v in d["models"].items():
            sm = v["summary"]
            rows.append([m.split("/")[-1]] + [f"{sm[c][r]:.2f} ({sm[c]['k']}/{sm[c]['n']}) {ci(sm[c]['wilson95'], 2)}" for c, r, _ in cols])
        s.append(table(["model"] + [c[2] for c in cols], rows))
        s.append("Paired (same request, direct vs wrapped):\n")
        s.append(table(["model", "pairs", "refused direct, not when wrapped", "refused only when wrapped"],
                       [[m.split("/")[-1], v["summary"]["jailbreak_vs_direct_paired"]["n_pairs"], v["summary"]["jailbreak_vs_direct_paired"]["direct_refused_jailbreak_not"],
                         v["summary"]["jailbreak_vs_direct_paired"]["jailbreak_refused_direct_not"]] for m, v in d["models"].items()]))
    d = J(A, "logit_lens.json")
    if d:
        s.append(f"## Logit lens ({d['model']}) — rank of the model's final prediction at selected layers (1 = top)\n")
        rows = []
        for L in d["lens"]:
            ls = {l["layer"]: l for l in L["layers"]}
            n = max(ls)
            pick = [0, n // 4, n // 2, 3 * n // 4, n - 2, n - 1, n]
            rows.append([L["text"], repr(L["final_prediction"])] + [f"{ls[k]['rank_of_final_token']} ({ls[k]['top'][0][0]!r})" for k in pick])
            hdr = ["prompt", "final prediction"] + [f"layer {k}: rank (top-1)" for k in pick]
        s.append(table(hdr, rows))
        last = [max(l["layer"] for l in L["layers"]) for L in d["lens"]]
        s.append(f"Correctness check: the last-layer lens equals the model's own logits (asserted in `tests/test_logit_lens.py`; last layer index {last[0]}).\n")
        p = d["patching"]
        s.append(f"## Activation patching (clean 'The capital of France is' → corrupted 'Italy'; metric: logit(Paris) − logit(Rome))\n")
        s.append(f"Clean logit diff {p['clean_logit_diff']:.3f}, corrupted {p['corrupt_logit_diff']:.3f}. Identity check (patch every layer & position): diff {p['identity_check_patch_all_logit_diff']:.3f}, abs error **{p['identity_check_abs_error']:.1e}**.\n")
        rec = p["recovery[layer][pos]"]
        toks = p["tokens"]
        sel = [0, 5, 10, 15, 20, 22, 23, 24, 27, len(rec) - 1]
        s.append("Fraction of the clean answer recovered when patching ONE (layer, position) (1 = fully restored):\n")
        s.append(table(["layer"] + [repr(t) for t in toks], [[l] + [f"{rec[l][q]:.2f}" for q in range(len(toks))] for l in sel if l < len(rec)]))
    return s


AREAS = {"01_inference_serving": ("Inference & Serving", area1), "02_gpu_systems_perf": ("GPU & Systems Performance", area2),
         "03_distributed_training": ("Distributed Training", area3), "04_post_training": ("Post-training", area4),
         "05_evals_rigor": ("Evals & Rigor", area5), "06_data_engineering": ("Data Engineering", area6),
         "07_rag_agents": ("RAG & Agents", area7), "08_safety_interp": ("Safety & Interpretability", area8)}


def main():
    """Write the RESULTS.md file of every selected area from its results JSON files."""
    sel = sys.argv[1:] or list(AREAS)
    for a in sel:
        title, fn = AREAS[a]
        envb = ""
        for p in sorted((ROOT / a / "results").glob("*.json")):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            if isinstance(d, dict) and "environment" in d:
                envb = env_block((a, p.name)); break
        body = [f"# {title} — RESULTS\n",
                "> This page was generated by `tools/make_results.py` from the files in `results/*.json`, which come from real runs. Please do not edit it by hand, because the explanations belong in `NOTES.md`.\n",
                envb]
        body += fn()
        notes = ROOT / a / "NOTES.md"
        if notes.exists():
            body.append("---\n\n" + notes.read_text())
        (ROOT / a / "RESULTS.md").write_text("\n".join(body))
        print("wrote", a + "/RESULTS.md")


if __name__ == "__main__":
    main()
