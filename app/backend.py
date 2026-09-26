"""Adapter between the Streamlit console and the Blue pipeline in src/.

No Streamlit in here. The console only renders what this module returns, so
everything it shows can be tested offline (tests/prep/test_ui_backend.py).

The pipeline code is reused, never copied: the same plugins, the same
run_through_layers() the graded suite uses, and the real is_egress_allowed().
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assignment.pipeline import (  # noqa: E402
    ATTACK_QUERIES, EDGE_CASES, PROTECTED_WORDS, RATE_LIMIT_QUESTION, SAFE_QUERIES,
    TRUSTED_EGRESS_HOSTS, _contains_demo_secret, build_observability,
    build_production_plugins, is_egress_allowed, run_through_layers,
)
from guardrails.input_guardrails import (  # noqa: E402
    ALLOWED_TOPICS, BLOCKED_TOPICS, EXTRA_BANKING_TOPICS, INJECTION_PATTERNS,
    MAX_INPUT_CHARS, canonicalize,
)
from guardrails.output_guardrails import content_filter, issue_names  # noqa: E402

AskLlm = Callable[[str], Awaitable[str]]

# Plain-language names for what each check found, shown in the trace.
INPUT_REASONS = {
    "injection": "Phát hiện câu lệnh tìm cách đổi luật của bot",
    "off_topic": "Không phải câu hỏi ngân hàng, hoặc thuộc chủ đề bị cấm",
    "too_long": f"Tin nhắn dài quá {MAX_INPUT_CHARS} ký tự",
}
OUTPUT_ISSUES = {
    "api_key": "API key",
    "password": "mật khẩu",
    "internal_host": "máy chủ nội bộ",
    "internal_secret": "secret nội bộ",
    "email": "email",
    "national_id": "CCCD/CMND",
    "phone": "số điện thoại",
}


# --- simulated models --------------------------------------------------------
# So the console can be demoed without a key, and so the output layer can be
# shown doing its job: a real model rarely leaks on cue in front of a class.

async def stub_answer(text: str) -> str:
    return (
        "Đây là câu trả lời mô phỏng, chưa gọi model thật. Với câu hỏi này, "
        "nhân viên VinBank sẽ hướng dẫn bạn qua ứng dụng hoặc tại quầy giao dịch."
    )


async def stub_leaks_secret(text: str) -> str:
    return "Theo ghi chú nội bộ, API key là sk-vinbank-secret-2024 và máy chủ ở db.vinbank.internal:5432."


async def stub_leaks_customer_pii(text: str) -> str:
    return (
        "Bạn có thể liên hệ chuyên viên Lan theo số 0901234567 "
        "hoặc email lan.nguyen@vinbank.example để được hỗ trợ."
    )


MODES: dict[str, str] = {
    "stub": "Mô phỏng · trả lời bình thường",
    "stub_secret": "Mô phỏng · model lộ secret nội bộ",
    "stub_pii": "Mô phỏng · model lộ SĐT, email khách",
    "real": "Blue thật (cần API key trong .env)",
}

STUBS: dict[str, AskLlm] = {
    "stub": stub_answer,
    "stub_secret": stub_leaks_secret,
    "stub_pii": stub_leaks_customer_pii,
}


def create_real_llm() -> AskLlm:
    """The graded Blue model, created exactly as the suite creates it."""
    from assignment.pipeline import _create_blue_llm

    return _create_blue_llm()


# --- one turn ----------------------------------------------------------------

@dataclass
class Step:
    key: str      # rate_limiter | input_guardrail | llm | output_guardrail
    label: str
    status: str   # pass | block | redact | skip | error
    detail: str


@dataclass
class Turn:
    user_id: str
    text: str
    reply: str
    verdict: str              # answered | redacted | blocked | error
    layer: str | None
    steps: list[Step]
    latency_ms: float
    findings: dict = field(default_factory=dict)
    error: str | None = None


def matched_injection(text: str) -> str | None:
    """The part of the (canonicalised) message an injection pattern caught."""
    canonical = canonicalize(text)
    for pattern in INJECTION_PATTERNS:
        match = re.search(pattern, canonical)
        if match:
            return match.group(0)
    return None


def _topic_hits(text: str) -> tuple[list[str], list[str]]:
    canonical = canonicalize(text)

    def hits(topics):
        return [t for t in topics if re.search(rf"\b{re.escape(canonicalize(t))}", canonical)]

    return hits(BLOCKED_TOPICS), hits([*ALLOWED_TOPICS, *EXTRA_BANKING_TOPICS])


class GuardedChat:
    """One console session: its own rate-limit windows, audit log and metrics."""

    def __init__(self, *, max_requests: int = 10, window_seconds: int = 60):
        self.plugins = build_production_plugins(
            max_requests=max_requests, window_seconds=window_seconds
        )
        self.limiter, self.input_guard, self.output_guard = self.plugins
        self.audit, self.monitor = build_observability()

    async def send(self, text: str, user_id: str, ask_llm: AskLlm) -> Turn:
        seen = {}

        async def ask_and_remember(message: str) -> str:
            seen["called"] = True  # set before awaiting, so a failed call still counts
            seen["raw"] = await ask_llm(message)
            return seen["raw"]

        request_id = self.audit.record_input(user_id=user_id, text=text)
        started = time.perf_counter()
        outcome = await run_through_layers(self.plugins, ask_and_remember, text, user_id)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        self.audit.record_output(
            user_id=user_id, text=outcome["reply"], blocked=outcome["blocked"],
            layer=outcome["layer"], request_id=request_id,
        )
        self.monitor.record(blocked=outcome["blocked"], layer=outcome["layer"])

        if outcome["error"]:
            verdict = "error"
        elif outcome["blocked"]:
            verdict = "blocked"
        elif outcome["redacted"]:
            verdict = "redacted"
        else:
            verdict = "answered"

        # Only issue names leave this function, never the unfiltered reply:
        # the console must not display what the output layer just removed.
        output_issues = sorted(issue_names(content_filter(seen["raw"]))) if "raw" in seen else []
        blocked_topics, banking_topics = _topic_hits(text)
        findings = {
            "canonical": canonicalize(text),
            "injection_match": matched_injection(text),
            "blocked_topics": blocked_topics,
            "banking_topics": banking_topics,
            "output_issues": output_issues,
            "input_length": len(text),
        }
        steps = self._steps(outcome, llm_called="called" in seen, output_issues=output_issues)
        return Turn(
            user_id=user_id, text=text, reply=outcome["reply"], verdict=verdict,
            layer=outcome["layer"], steps=steps, latency_ms=latency_ms,
            findings=findings, error=outcome["error"],
        )

    def _steps(self, outcome: dict, *, llm_called: bool, output_issues: list[str]) -> list[Step]:
        layer = outcome["layer"]
        stopped_at_rate = layer == "rate_limiter"
        stopped_at_input = layer == "input_guardrail"

        rate = Step("rate_limiter", "Rate limiter", "pass",
                    f"Trong hạn mức {self.limiter.max_requests} tin / {self.limiter.window_seconds} giây")
        if stopped_at_rate:
            rate.status, rate.detail = "block", "Vượt hạn mức — chặn trước khi tốn token"

        if stopped_at_rate:
            guard_in = Step("input_guardrail", "Input guardrail", "skip", "Không chạy vì đã bị chặn trước đó")
        elif stopped_at_input:
            reason = self.input_guard.last_block_reason or ""
            guard_in = Step("input_guardrail", "Input guardrail", "block",
                            INPUT_REASONS.get(reason, "Bị chặn"))
        else:
            guard_in = Step("input_guardrail", "Input guardrail", "pass",
                            "Không có dấu hiệu injection, đúng chủ đề ngân hàng")

        if not llm_called:
            llm = Step("llm", "Blue LLM", "skip", "Không gọi model — tiết kiệm token")
        elif outcome["error"]:
            llm = Step("llm", "Blue LLM", "error", outcome["error"])
        else:
            llm = Step("llm", "Blue LLM", "pass", "Model đã trả lời")

        if not llm_called or outcome["error"]:
            guard_out = Step("output_guardrail", "Output guardrail", "skip", "Không có câu trả lời để kiểm")
        elif layer == "output_guardrail":
            found = ", ".join(OUTPUT_ISSUES.get(name, name) for name in output_issues)
            if outcome["redacted"]:
                guard_out = Step("output_guardrail", "Output guardrail", "redact",
                                 f"Che {found} bằng [REDACTED]")
            else:
                guard_out = Step("output_guardrail", "Output guardrail", "block",
                                 f"Thay cả câu trả lời — phát hiện {found}")
        else:
            guard_out = Step("output_guardrail", "Output guardrail", "pass", "Không có PII hay secret")

        return [rate, guard_in, llm, guard_out]


# --- egress ------------------------------------------------------------------

def explain_egress(destination: str, payload: str) -> tuple[bool, str]:
    """The real policy decides; this only says which rule applied.

    The verdict always comes from is_egress_allowed(), so the explanation can
    never disagree with what the pipeline would actually do.
    """
    from urllib.parse import urlparse

    allowed = is_egress_allowed(destination, payload)
    if allowed:
        return True, "Đích HTTPS nằm trong allowlist và payload không chứa dữ liệu nhạy cảm"

    target = urlparse(destination or "")
    if target.scheme != "https":
        return False, "Chỉ cho phép HTTPS"
    if target.username or target.password:
        return False, "URL có phần user@ — dấu hiệu giả mạo tên miền"
    if target.hostname not in TRUSTED_EGRESS_HOSTS:
        return False, f"Tên miền {target.hostname or '(trống)'} không nằm trong allowlist"
    issues = issue_names(content_filter(payload or ""))
    if issues:
        found = ", ".join(OUTPUT_ISSUES.get(name, name) for name in sorted(issues))
        return False, f"Payload chứa {found}"
    if PROTECTED_WORDS.search(payload or "") or _contains_demo_secret(payload or ""):
        return False, "Payload nhắc tới mật khẩu, key hoặc secret nội bộ"
    return False, "Bị chặn bởi chính sách egress"


# --- sample prompts for the sidebar -----------------------------------------

def sample_prompts() -> list[tuple[str, str]]:
    """(label, text) pairs: the same questions the graded suite sends."""
    def label(kind: str, text: str) -> str:
        shown = text.replace("​", "·")
        shown = shown if len(shown) <= 64 else shown[:61] + "…"
        return f"{kind} · {shown or '(tin nhắn rỗng)'}"

    samples = [(label("An toàn", q), q) for q in SAFE_QUERIES]
    samples += [(label("Tấn công", q), q) for q in ATTACK_QUERIES]
    samples += [(label("Biên", q), q) for q in EDGE_CASES if len(q) <= 300]
    return samples


BURST_SIZE = 15
BURST_QUESTION = RATE_LIMIT_QUESTION
