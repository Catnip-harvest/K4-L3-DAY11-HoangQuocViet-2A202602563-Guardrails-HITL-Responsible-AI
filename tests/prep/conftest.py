"""Keep tests/prep hermetic: .env holds live keys, and core.config loads it on import.

Every test starts with both Groq layers switched off. A test that exercises one
turns it on explicitly (monkeypatch.setenv) and replaces the network call.
"""
import pytest


@pytest.fixture(autouse=True)
def _network_layers_off(monkeypatch):
    monkeypatch.setenv("PROMPT_GUARD", "0")
    monkeypatch.setenv("LLM_JUDGE", "0")
