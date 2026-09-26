"""Groq client for the two ML guardrail layers (and optionally Blue).

Groq speaks the OpenAI API, so the plain OpenAI SDK works against it.

  Input classifier  meta-llama/llama-prompt-guard-2-86m   (Meta)     14,400 req/day free
  Output judge      qwen/qwen3.8-27b                       (Alibaba)   1,000 req/day free

Both layers are OPTIONAL. Without GROQ_API_KEY every caller must still work,
because the public tests and the grader run with no keys at all.

The client is synchronous and called through asyncio.to_thread(): an
AsyncOpenAI client binds to the event loop it was first used on, and the tests
and the Streamlit console each create their own loops.
"""
from __future__ import annotations

import asyncio
import os
from functools import lru_cache

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_PROMPT_GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"
DEFAULT_JUDGE_MODEL = "qwen/qwen3.8-27b"


def groq_api_key() -> str:
    return os.environ.get("GROQ_API_KEY", "").strip()


def groq_available() -> bool:
    return bool(groq_api_key())


def prompt_guard_model() -> str:
    return os.environ.get("PROMPT_GUARD_MODEL", "").strip() or DEFAULT_PROMPT_GUARD_MODEL


def judge_model() -> str:
    return os.environ.get("JUDGE_MODEL", "").strip() or DEFAULT_JUDGE_MODEL


def layer_enabled(env_flag: str) -> bool:
    """A Groq layer runs when a key exists and its flag is not "0"."""
    return groq_available() and os.environ.get(env_flag, "1").strip() != "0"


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    # Free tier is 30 requests/min per model; retries honour Groq's retry-after.
    return OpenAI(api_key=groq_api_key(), base_url=GROQ_BASE_URL, max_retries=6)


async def groq_chat(**kwargs):
    """chat.completions.create on Groq, without blocking the event loop."""
    return await asyncio.to_thread(_client().chat.completions.create, **kwargs)
