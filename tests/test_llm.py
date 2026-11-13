import json

import pytest

from agent.schemas import RewrittenQuery
from llm import client
from llm.client import LLM, load_prompt


def make_llm(tmp_path, monkeypatch, replies, backend="ollama"):
    monkeypatch.setattr(client, "LOG_PATH", tmp_path / "calls.jsonl")
    llm = LLM(backend)
    seen = []
    queue = iter(replies)

    def fake_call(system, user, schema):
        seen.append(user)
        return next(queue)

    llm._call = fake_call
    return llm, seen


def log_rows(tmp_path):
    return [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]


def test_load_prompt_splits_system_and_user_template():
    system, user = load_prompt("rewrite", 1)
    assert "standalone search query" in system
    assert "{question}" in user and "{history}" in user


def test_valid_reply_is_parsed_and_logged_without_prompt_text(tmp_path, monkeypatch):
    llm, _ = make_llm(tmp_path, monkeypatch, [('{"query": "what is HyDE"}', 100, 10)])
    result = llm.structured("rewrite", 1, RewrittenQuery, history="", question="what is HyDE?")
    assert result.query == "what is HyDE"
    (row,) = log_rows(tmp_path)
    assert (row["prompt"], row["prompt_version"], row["error"], row["cost_usd"]) == ("rewrite", 1, None, 0.0)
    assert (row["input_tokens"], row["output_tokens"]) == (100, 10)
    assert "HyDE" not in json.dumps(row)


def test_invalid_reply_gets_one_retry_with_error_feedback(tmp_path, monkeypatch):
    llm, seen = make_llm(tmp_path, monkeypatch, [("not json", 5, 5), ('{"query": "ok"}', 5, 5)])
    result = llm.structured("rewrite", 1, RewrittenQuery, history="", question="q")
    assert result.query == "ok"
    assert len(seen) == 2 and "previous answer was invalid" in seen[1]
    first, second = log_rows(tmp_path)
    assert first["error"] is not None and second["error"] is None


def test_gives_up_after_max_attempts(tmp_path, monkeypatch):
    llm, seen = make_llm(tmp_path, monkeypatch, [("nope", 1, 1), ("still nope", 1, 1)])
    with pytest.raises(ValueError):
        llm.structured("rewrite", 1, RewrittenQuery, history="", question="q")
    assert len(seen) == 2


def test_braces_in_user_text_do_not_break_prompt_formatting(tmp_path, monkeypatch):
    llm, seen = make_llm(tmp_path, monkeypatch, [('{"query": "x"}', 1, 1)])
    llm.structured("rewrite", 1, RewrittenQuery, history="{history}", question="{evil} {0}")
    assert "{evil} {0}" in seen[0]


def test_gemini_cost_uses_pinned_prices():
    assert LLM("gemini")._cost_usd(1_000_000, 1_000_000) == pytest.approx(2.80)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        LLM("openai")
