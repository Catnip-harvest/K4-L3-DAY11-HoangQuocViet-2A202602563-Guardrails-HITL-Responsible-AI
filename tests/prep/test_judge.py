"""Offline tests for the LLM-as-Judge layer. The Groq call is always faked.

conftest.py switches LLM_JUDGE off for every test; the tests here that need
the judge turn it back on and replace groq_chat, so nothing leaves the machine.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from google.genai import types  # noqa: E402

from guardrails import output_guardrails  # noqa: E402
from guardrails.output_guardrails import (  # noqa: E402
    JUDGE_REFUSAL, SECRET_REFUSAL, OutputGuardrailPlugin, llm_safety_check,
)


def run(coro):
    return asyncio.run(coro)


def _llm_response(text: str):
    return SimpleNamespace(content=types.Content(role="model", parts=[types.Part.from_text(text=text)]))


def _completion(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _grading(verdict="SAFE", safety=5, relevance=5, accuracy=5, tone=5, reason="fine"):
    return json.dumps({
        "safety": safety, "relevance": relevance, "accuracy": accuracy, "tone": tone,
        "verdict": verdict, "reason": reason,
    })


class FakeGroq:
    """Stands in for groq_chat and remembers what it was asked to grade."""

    def __init__(self, content: str | None = None, error: Exception | None = None):
        self.content = content
        self.error = error
        self.graded_texts: list[str] = []

    async def __call__(self, **kwargs):
        self.graded_texts.append(kwargs["messages"][-1]["content"])
        if self.error is not None:
            raise self.error
        return _completion(self.content)


@pytest.fixture
def judge_on(monkeypatch):
    monkeypatch.setenv("LLM_JUDGE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "test")

    def install(fake: FakeGroq) -> FakeGroq:
        monkeypatch.setattr(output_guardrails, "groq_chat", fake)
        return fake

    return install


def _judge_reply(plugin: OutputGuardrailPlugin, text: str) -> str:
    out = run(plugin.after_model_callback(callback_context=None, llm_response=_llm_response(text)))
    return out.content.parts[0].text


def test_disabled_judge_makes_no_call(monkeypatch):
    fake = FakeGroq(content=_grading("UNSAFE", safety=1))
    monkeypatch.setattr(output_guardrails, "groq_chat", fake)

    result = run(llm_safety_check("Our 12-month rate is 3.75% p.a."))
    plugin = OutputGuardrailPlugin(use_llm_judge=True)
    reply = _judge_reply(plugin, "Our 12-month rate is 3.75% p.a.")

    assert result == {
        "safe": True, "verdict": "Judge not initialized — skipping",
        "scores": None, "reason": "", "error": None,
    }
    assert fake.graded_texts == []
    assert reply == "Our 12-month rate is 3.75% p.a."
    assert plugin.use_llm_judge is False and plugin.last_judge is None


def test_safe_verdict_passes_reply_and_is_logged(judge_on):
    judge_on(FakeGroq(content=_grading("SAFE", accuracy=3, reason="unverified rate")))
    plugin = OutputGuardrailPlugin(use_llm_judge=True)

    reply = _judge_reply(plugin, "Our 12-month rate is 3.75% p.a.")

    assert reply == "Our 12-month rate is 3.75% p.a."
    assert plugin.last_action is None
    assert (plugin.judge_checks, plugin.judge_fails, plugin.judge_errors) == (1, 0, 0)
    assert plugin.last_judge["scores"] == {"safety": 5, "relevance": 5, "accuracy": 3, "tone": 5}
    assert plugin.last_judge["reason"] == "unverified rate"
    assert plugin.judge_log == [{
        "response_preview": "Our 12-month rate is 3.75% p.a.",
        "safety": 5, "relevance": 5, "accuracy": 3, "tone": 5, "verdict": "SAFE",
    }]


def test_unsafe_verdict_replaces_reply(judge_on):
    judge_on(FakeGroq(content=_grading("UNSAFE", safety=1, reason="harmful instructions")))
    plugin = OutputGuardrailPlugin(use_llm_judge=True)

    reply = _judge_reply(plugin, "Here is how to launder the money step by step.")

    assert reply == JUDGE_REFUSAL
    assert plugin.last_action == "judge_blocked"
    assert (plugin.judge_checks, plugin.judge_fails, plugin.blocked_count) == (1, 1, 1)
    assert plugin.judge_log[0]["verdict"] == "UNSAFE"


def test_low_safety_score_blocks_even_with_safe_verdict(judge_on):
    judge_on(FakeGroq(content=_grading("SAFE", safety=2)))
    plugin = OutputGuardrailPlugin(use_llm_judge=True)

    reply = _judge_reply(plugin, "Just share your OTP with anyone who calls.")

    assert reply == JUDGE_REFUSAL
    assert plugin.last_action == "judge_blocked" and plugin.judge_fails == 1


def test_judge_exception_fails_open(judge_on):
    judge_on(FakeGroq(error=RuntimeError("429 quota")))
    plugin = OutputGuardrailPlugin(use_llm_judge=True)

    reply = _judge_reply(plugin, "Our 12-month rate is 3.75% p.a.")

    assert reply == "Our 12-month rate is 3.75% p.a."
    assert (plugin.judge_checks, plugin.judge_fails, plugin.judge_errors) == (0, 0, 1)
    assert plugin.judge_log == []
    assert plugin.last_judge["verdict"] == "ERROR"
    assert plugin.last_judge["error"] == "RuntimeError: 429 quota"


def test_unparseable_judge_reply_is_an_error(judge_on):
    judge_on(FakeGroq(content="SAFE, looks fine to me"))

    result = run(llm_safety_check("Our 12-month rate is 3.75% p.a."))

    assert result["safe"] is True and result["verdict"] == "ERROR"
    assert result["scores"] is None
    assert result["error"].startswith("JSONDecodeError")


def test_secret_is_blocked_before_the_judge(judge_on):
    fake = judge_on(FakeGroq(content=_grading("SAFE")))
    plugin = OutputGuardrailPlugin(use_llm_judge=True)

    reply = _judge_reply(plugin, "Sure, the admin password is admin123")

    assert reply == SECRET_REFUSAL and plugin.last_action == "blocked"
    assert fake.graded_texts == []
    assert plugin.last_judge is None and plugin.judge_checks == 0


def test_judge_grades_the_redacted_text(judge_on):
    fake = judge_on(FakeGroq(content=_grading("SAFE")))
    plugin = OutputGuardrailPlugin(use_llm_judge=True)

    reply = _judge_reply(plugin, "Call the branch at 0901234567 for help.")

    assert reply == "Call the branch at [REDACTED] for help."
    assert plugin.last_action == "redacted"
    assert fake.graded_texts == ["Call the branch at [REDACTED] for help."]
    assert "0901234567" not in plugin.judge_log[0]["response_preview"]
