"""This file builds an agent, which is a language model that can use tools. Here the tools are a
calculator, a unit converter and a document search. It also contains a test set that measures
how well the agent picks and uses tools. It is part of area 7.

More technical notes:
A tool-calling agent loop with a small LOCAL model (Qwen2.5-0.5B-Instruct GGUF via llama-cpp-python), plus an evaluation.

Mechanism (the "ReAct"-style loop, Yao et al. 2022, arXiv:2210.03629):
   1. render the chat (with the tool JSON-schemas) using the model's own chat template (Qwen2.5 uses Hermes-style
      <tool_call>{json}</tool_call> blocks: https://qwen.readthedocs.io/en/latest/framework/function_call.html)
   2. generate; if the output contains <tool_call> blocks -> parse JSON, run the Python function, append the result as a
      `tool` message, go to 1 ; otherwise the text is the final answer.
   3. stop after `max_steps` (guards against infinite loops).
Tools: calculator (safe AST evaluator - never uses eval), unit_convert, search_docs (BM25 over SciFact abstracts).

Evaluation (python agent_loop.py --eval): 29 tasks with programmatic ground truth, greedy decoding:
   calc (8), convert (6), search (10: success = gold paper title appears in the final answer; we also report whether the
   agent's own search query retrieved the gold paper), control (5 questions that need NO tool: counts unnecessary tool use).
Reported: success per category with Wilson 95% CI, tool-call JSON validity rate, mean steps. A 0.5B model is weak - we
report what we measure and keep the failures in the saved traces.
"""
import argparse
import ast
import json
import operator
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "05_evals_rigor"))
import stats_utils as su  # noqa: E402

from common import env  # noqa: E402

# ------------------------------------------------------------------ tools
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos,
        ast.FloorDiv: operator.floordiv}


def safe_eval(expr: str):
    """Evaluate +,-,*,/,**,%,// on numbers only. Raises ValueError otherwise (no names, calls, attributes)."""
    def ev(n):
        """Calculate one node of the parsed expression. Only numbers and basic arithmetic are
        allowed, so nothing dangerous can run.
        """
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            l, r = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(r) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](l, r)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError(f"unsupported syntax: {type(n).__name__}")
    return ev(ast.parse(expr.strip(), mode="eval").body)


_UNITS = {("mi", "km"): 1.609344, ("km", "mi"): 1 / 1.609344, ("kg", "lb"): 2.2046226218, ("lb", "kg"): 1 / 2.2046226218,
          ("in", "cm"): 2.54, ("cm", "in"): 1 / 2.54}


def unit_convert(value, from_unit, to_unit):
    """Convert a number between a few supported units, such as Celsius and Fahrenheit or
    kilometres and miles.
    """
    f, t = str(from_unit).lower(), str(to_unit).lower()
    v = float(value)
    if (f, t) == ("c", "f"):
        return v * 9 / 5 + 32
    if (f, t) == ("f", "c"):
        return (v - 32) * 5 / 9
    if (f, t) in _UNITS:
        return v * _UNITS[(f, t)]
    raise ValueError(f"unsupported conversion {f}->{t}; supported: mi,km,kg,lb,in,cm,c,f")


TOOL_SPECS = [
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression such as '(45+55)/4'. Supports + - * / ** % //.",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "arithmetic expression"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_convert", "description": "Convert a value between units: mi<->km, kg<->lb, in<->cm, c<->f.",
     "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "search_docs", "description": "Search a corpus of scientific paper abstracts. Returns the top 3 results (id, title, snippet).",
     "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "search query"}}, "required": ["query"]}}},
]


class Toolbox:
    """Holds the tools that the agent may call (calculator, unit converter and document search)
    and runs them by name.
    """
    def __init__(self, bm25=None, id2title=None, id2text=None):
        """Remember the search index and the document titles that the search tool needs."""
        self.bm25, self.id2title, self.id2text = bm25, id2title or {}, id2text or {}
        self.last_search_ids: list = []

    def call(self, name, args):
        """Run the tool with the given name and arguments and return its result as text."""
        if name == "calculator":
            return str(safe_eval(str(args["expression"])))
        if name == "unit_convert":
            return f"{unit_convert(args['value'], args['from_unit'], args['to_unit']):.4f}"
        if name == "search_docs":
            if self.bm25 is None:
                raise ValueError("search backend not loaded")
            ids, _ = self.bm25.search(str(args["query"]), 3)
            self.last_search_ids = ids
            return json.dumps([{"id": i, "title": self.id2title[i], "snippet": self.id2text[i][:160]} for i in ids])
        raise ValueError(f"unknown tool {name!r}")


# ------------------------------------------------------------------ agent
TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_tool_calls(text: str):
    """Return list of (ok, payload). payload = dict(name, arguments) if ok else raw string."""
    out = []
    for raw in TOOL_RE.findall(text):
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict) or "name" not in obj or not isinstance(obj.get("arguments", {}), dict):
                raise ValueError("missing name/arguments")
            out.append((True, obj))
        except (json.JSONDecodeError, ValueError):
            out.append((False, raw))
    return out


class Agent:
    """The agent: it asks the model, runs any tool that the model requests, shows the result to
    the model and repeats until the model answers.
    """
    def __init__(self, llm, tokenizer, toolbox: Toolbox, tools=TOOL_SPECS, max_steps=4, max_tokens=200):
        """Store the model, the tokenizer, the tools and the limit on how many steps the agent
        may take.
        """
        self.llm, self.tok, self.tb, self.tools, self.max_steps, self.max_tokens = llm, tokenizer, toolbox, tools, max_steps, max_tokens

    def run(self, question: str, system="You are a helpful assistant. Use a tool only when it is needed."):
        """Answer one question and return a record (trace) of everything that happened,
        including every tool call.
        """
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        trace = {"question": question, "steps": [], "tool_calls_total": 0, "tool_calls_valid": 0, "final": None, "stopped": "final"}
        for step in range(self.max_steps):
            prompt = self.tok.apply_chat_template(msgs, tools=self.tools, tokenize=False, add_generation_prompt=True)
            text = self.llm.create_completion(prompt, max_tokens=self.max_tokens, temperature=0.0, stop=["<|im_end|>"])["choices"][0]["text"]
            calls = parse_tool_calls(text)
            trace["steps"].append({"model_output": text, "n_tool_calls": len(calls)})
            if not calls:
                trace["final"] = text.strip(); return trace
            good = [p for ok, p in calls if ok]
            trace["tool_calls_total"] += len(calls); trace["tool_calls_valid"] += len(good)
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": p["name"], "arguments": p.get("arguments", {})}} for p in good]}
                        if good else {"role": "assistant", "content": text})
            if not good:
                msgs.append({"role": "user", "content": "Your tool call was not valid JSON. Reply with a valid <tool_call> or answer directly."})
                continue
            for p in good:
                try:
                    result = self.tb.call(p["name"], p.get("arguments", {}))
                except Exception as e:  # tool errors are fed back to the model, like in production agents
                    result = f"ERROR: {type(e).__name__}: {e}"
                trace["steps"][-1].setdefault("tool_results", []).append({"name": p["name"], "arguments": p.get("arguments"), "result": result})
                msgs.append({"role": "tool", "content": result})
        trace["stopped"] = "max_steps"
        return trace


# ------------------------------------------------------------------ evaluation tasks
def numbers_in(text: str):
    """Find all numbers written in a text and return them as numbers."""
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text) if x.replace(",", "").replace("-", "").replace(".", "")]


def has_number(text, expected, rel=0.01):
    """Return True if the text contains the expected number, allowing a small relative
    difference.
    """
    return any(abs(n - expected) <= max(rel * abs(expected), 0.011) for n in numbers_in(text))


CALC = [("What is 1234 * 5678?", 1234 * 5678), ("Compute (45 + 55) / 4.", 25.0), ("What is 2 ** 10 - 24?", 1000),
        ("What is 987654 - 123456?", 864198), ("Calculate 17 * 23 + 5.", 396), ("What is 144 / 12 / 3?", 4.0),
        ("What is 99999 + 1?", 100000), ("What is 7 * 8 * 9?", 504)]
CONVERT = [("Convert 10 miles to kilometers.", 16.09344), ("How many pounds is 70 kg?", 154.32), ("Convert 100 F to Celsius.", 37.78),
           ("Convert 25 C to Fahrenheit.", 77.0), ("How many centimeters is 12 inches?", 30.48), ("How many miles is 42 km?", 26.10)]
CONTROL = [("What is the capital of France?", "paris"), ("What color is the sky on a clear day?", "blue"),
           ("Who wrote the play Romeo and Juliet?", "shakespeare"), ("What is the opposite of hot?", "cold"),
           ("How many days are in a week?", "7")]


def run_eval(agent, toolbox, qtext, qrels, id2title, n_search=10):
    """Run all test questions through the agent and return one result record for each."""
    results = []
    for q, exp in CALC:
        t = agent.run(q); ok = t["final"] is not None and has_number(t["final"], exp)
        results.append({"cat": "calc", "ok": ok, "used_tool": t["tool_calls_total"] > 0, "trace": t})
    for q, exp in CONVERT:
        t = agent.run(q); ok = t["final"] is not None and has_number(t["final"], exp, rel=0.01)
        results.append({"cat": "convert", "ok": ok, "used_tool": t["tool_calls_total"] > 0, "trace": t})
    for qid in sorted(qtext, key=int)[:n_search]:
        gold = [d for d, g in qrels[qid].items() if g > 0]
        toolbox.last_search_ids = []
        t = agent.run(f'Find the paper most relevant to this claim and tell me its exact title: "{qtext[qid]}"')
        retrieved_gold = any(g in toolbox.last_search_ids for g in gold)
        ok = t["final"] is not None and any(id2title[g].lower()[:60] in t["final"].lower() for g in gold)
        results.append({"cat": "search", "ok": ok, "retrieved_gold": retrieved_gold, "used_tool": t["tool_calls_total"] > 0, "trace": t, "gold_titles": [id2title[g] for g in gold]})
    for q, exp in CONTROL:
        t = agent.run(q); ok = t["final"] is not None and exp in t["final"].lower() and t["tool_calls_total"] == 0
        results.append({"cat": "control", "ok": ok, "used_tool": t["tool_calls_total"] > 0, "trace": t})
    return results


def summarize(results):
    """Count the successes per question category and add confidence intervals and tool-call
    statistics.
    """
    out = {}
    for cat in ("calc", "convert", "search", "control"):
        rs = [r for r in results if r["cat"] == cat]
        k = sum(r["ok"] for r in rs)
        out[cat] = {"n": len(rs), "success": k, "rate": round(k / len(rs), 3), "wilson95": [round(x, 3) for x in su.wilson_interval(k, len(rs))]}
        out[cat]["used_tool_rate"] = round(sum(r["used_tool"] for r in rs) / len(rs), 3)
    s = [r for r in results if r["cat"] == "search"]
    out["search"]["agent_query_retrieved_gold_rate"] = round(sum(r["retrieved_gold"] for r in s) / len(s), 3)
    tot = sum(r["trace"]["tool_calls_total"] for r in results); val = sum(r["trace"]["tool_calls_valid"] for r in results)
    out["tool_call_json_validity"] = {"valid": val, "total": tot, "rate": round(val / max(1, tot), 3)}
    out["mean_steps"] = round(sum(len(r["trace"]["steps"]) for r in results) / len(results), 2)
    return out


def load_llm(path, n_ctx=4096, threads=None):
    """Load the GGUF language model with llama.cpp."""
    from llama_cpp import Llama
    import os
    return Llama(model_path=path, n_ctx=n_ctx, n_threads=threads or os.cpu_count(), verbose=False, seed=0)


def main():
    """Either answer a single question or run the full evaluation, then print and save the
    results.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=None, help="path to a GGUF (default: Qwen2.5-0.5B-Instruct q4_k_m from the HF cache)")
    ap.add_argument("--hf-tokenizer", default=env.QWEN_05B_INSTRUCT, help="HF repo whose chat template matches the GGUF")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--ask", default=None, help="ask a single question and print the trace")
    ap.add_argument("--tag", default="qwen2.5-0.5b-q4_k_m")
    a = ap.parse_args()
    import retrieval_lib as rl
    from transformers import AutoTokenizer
    ids, texts, qtext, qrels = rl.load_scifact()
    id2title = {i: t.split(". ")[0] for i, t in zip(ids, texts)}
    tb = Toolbox(rl.BM25Index(ids, texts), id2title, dict(zip(ids, texts)))
    llm = load_llm(a.gguf or env.gguf_path("q4_k_m"))
    agent = Agent(llm, AutoTokenizer.from_pretrained(a.hf_tokenizer), tb)
    if a.ask:
        print(json.dumps(agent.run(a.ask), indent=1)); return
    t0 = time.perf_counter()
    results = run_eval(agent, tb, qtext, qrels, id2title)
    summ = summarize(results)
    print(json.dumps(summ, indent=1))
    env.save_json({"environment": env.environment_info(), "model": a.tag, "seconds": round(time.perf_counter() - t0, 1),
                   "summary": summ, "results": results}, Path(__file__).parent / "results" / f"agent_eval_{a.tag}.json")


if __name__ == "__main__":
    main()
