"""These tests check the calculator, the unit converter, the tool-call parser and the agent loop of
area 7. They use a fake model, so they run in a fraction of a second.
"""
import json

import pytest

import agent_loop as al


@pytest.mark.parametrize("expr,val", [("1+2*3", 7), ("(45+55)/4", 25.0), ("2**10-24", 1000), ("-3+5", 2), ("7%4", 3), ("9//2", 4), ("1_000+1", 1001)])
def test_safe_eval_ok(expr, val):
    """Check that the safe calculator gives the right answer for valid arithmetic."""
    assert al.safe_eval(expr) == val


@pytest.mark.parametrize("bad", ["__import__('os').system('echo hi')", "open('/etc/passwd')", "a+1", "2**10000", "[1,2]", "'a'*5", "print(1)", "(lambda: 1)()", "True+1"])
def test_safe_eval_rejects_non_arithmetic(bad):
    """Check that the safe calculator refuses anything that is not plain arithmetic, including
    dangerous code.
    """
    with pytest.raises((ValueError, SyntaxError, TypeError)):
        al.safe_eval(bad)


def test_unit_convert():
    """Check the unit converter with known values and check that unknown units are rejected.
    """
    assert al.unit_convert(10, "mi", "km") == pytest.approx(16.09344)
    assert al.unit_convert(100, "F", "C") == pytest.approx(37.7778, abs=1e-3)
    assert al.unit_convert(25, "c", "f") == pytest.approx(77.0)
    with pytest.raises(ValueError):
        al.unit_convert(1, "mi", "kg")


def test_parse_tool_calls():
    """Check that tool calls written by the model are found and that broken ones are marked as
    invalid.
    """
    ok = '<tool_call>\n{"name": "calculator", "arguments": {"expression": "1+1"}}\n</tool_call>'
    r = al.parse_tool_calls(ok + "\n" + ok)
    assert len(r) == 2 and all(x for x, _ in r) and r[0][1]["name"] == "calculator"
    assert al.parse_tool_calls("<tool_call>{bad json</tool_call>") == [(False, "{bad json")]
    assert al.parse_tool_calls('<tool_call>{"arguments": {}}</tool_call>')[0][0] is False   # missing name
    assert al.parse_tool_calls("plain answer") == []


def test_has_number_tolerance():
    """Check that the answer checker accepts numbers that are close enough and rejects wrong
    ones.
    """
    assert al.has_number("The answer is 7,006,652.", 7006652)
    assert al.has_number("about 16.09 km", 16.09344)
    assert not al.has_number("it is 17", 16.09344)


class FakeLLM:
    """Scripted model: first call emits a tool call, second call answers. Verifies the loop wiring without any model."""
    def __init__(self, outputs):
        """Store the list of prepared answers."""
        self.outputs = list(outputs); self.prompts = []

    def create_completion(self, prompt, **kw):
        """Record the prompt and return the next prepared answer."""
        self.prompts.append(prompt)
        return {"choices": [{"text": self.outputs.pop(0)}]}


class FakeTok:
    """A pretend tokenizer that only turns the chat messages into text."""
    def apply_chat_template(self, msgs, tools=None, tokenize=False, add_generation_prompt=True):
        """Turn the list of chat messages into a JSON text."""
        return json.dumps(msgs)


def test_agent_loop_executes_tool_and_returns_final():
    """Check that the agent runs the requested tool and then returns the final answer."""
    llm = FakeLLM(['<tool_call>{"name": "calculator", "arguments": {"expression": "12*13"}}</tool_call>', "It is 156."])
    t = al.Agent(llm, FakeTok(), al.Toolbox()).run("What is 12*13?")
    assert t["final"] == "It is 156." and t["tool_calls_total"] == 1 and t["tool_calls_valid"] == 1
    assert '"role": "tool"' in llm.prompts[1] and "156" in llm.prompts[1]


def test_agent_loop_recovers_from_invalid_json_and_unknown_tool_and_stops():
    """Check that the agent survives broken tool calls and unknown tools, and still stops."""
    llm = FakeLLM(["<tool_call>{oops</tool_call>", '<tool_call>{"name": "nope", "arguments": {}}</tool_call>', "done"])
    t = al.Agent(llm, FakeTok(), al.Toolbox()).run("x")
    assert t["tool_calls_total"] == 2 and t["tool_calls_valid"] == 1 and t["final"] == "done"
    assert "unknown tool" in llm.prompts[2]
    llm2 = FakeLLM(['<tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>'] * 10)
    t2 = al.Agent(llm2, FakeTok(), al.Toolbox(), max_steps=3).run("loop")
    assert t2["stopped"] == "max_steps" and len(t2["steps"]) == 3
