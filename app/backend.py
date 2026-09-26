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
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Iterator

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assignment.pipeline import (  # noqa: E402
    ATTACK_QUERIES, EDGE_CASES, PROTECTED_WORDS, RATE_LIMIT_QUESTION, SAFE_QUERIES,
    TRUSTED_EGRESS_HOSTS, _contains_demo_secret, build_observability,
    build_production_plugins, is_egress_allowed, run_through_layers,
)
from guardrails import input_guardrails  # noqa: E402
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
    "prompt_guard": "Prompt Guard (Meta) chấm là injection / jailbreak",
}
# The judge's four criteria, in the order the trace and findings show them.
JUDGE_CRITERIA = {
    "safety": "an toàn",
    "relevance": "liên quan",
    "accuracy": "chính xác",
    "tone": "giọng điệu",
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


def real_mode_label() -> str:
    """The configured Blue model, e.g. "Blue thật · groq:openai/gpt-oss-120b"."""
    try:
        from core.config import blue_provider_label

        return f"Blue thật · {blue_provider_label()}"
    except Exception:  # noqa: BLE001 — a label must never break the page
        return "Blue thật"


class _Modes(Mapping):
    """Mode -> label. The "real" label is read from config on each access,
    so it follows .env without importing provider code at module load."""

    _fixed = {
        "stub": "Mô phỏng · trả lời bình thường",
        "stub_secret": "Mô phỏng · model lộ secret nội bộ",
        "stub_pii": "Mô phỏng · model lộ SĐT, email khách",
    }

    def __getitem__(self, mode: str) -> str:
        return real_mode_label() if mode == "real" else self._fixed[mode]

    def __iter__(self) -> Iterator[str]:
        return iter([*self._fixed, "real"])

    def __len__(self) -> int:
        return len(self._fixed) + 1


MODES: Mapping[str, str] = _Modes()


def real_blue_available() -> bool:
    """True when the key for the configured Blue provider is set (env only, no API call)."""
    try:
        from core.config import PROVIDER_GROQ, get_blue_provider, get_openrouter_api_key
        from core.groq_client import groq_available
    except Exception:  # noqa: BLE001 — the console must still open without src config
        return False
    return groq_available() if get_blue_provider() == PROVIDER_GROQ else bool(get_openrouter_api_key())


def default_mode() -> str:
    """Open on the real model when it can run; the simulations are for demos without a key."""
    return "real" if real_blue_available() else "stub"

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


def prompt_guard_threshold() -> float:
    return float(getattr(input_guardrails, "PROMPT_GUARD_THRESHOLD", 0.5))


def _redact(text: str) -> str:
    """Model-written text (the judge's reason) passes the same PII filter as replies."""
    return content_filter(text or "")["redacted"][:240]


def _judge_scores(judge: dict | None, criteria=("safety", "accuracy")) -> str:
    scores = (judge or {}).get("scores") or {}
    return " · ".join(
        f"{JUDGE_CRITERIA[name]} {scores[name]}/5" for name in criteria if scores.get(name) is not None
    )


def _judge_note(judge: dict | None) -> str:
    """Suffix for a step the judge let through: its verdict, or that it failed open."""
    if not judge:
        return ""
    if judge.get("error"):
        return " · Qwen judge lỗi — cho qua (fail open)"
    verdict = judge.get("verdict") or ("SAFE" if judge.get("safe") else "UNSAFE")
    scores = _judge_scores(judge)
    return f" · Qwen judge: {verdict}" + (f" · {scores}" if scores else "")


def _ml_layers() -> list[tuple[str, bool, str]]:
    """(short name, enabled, model) for each Groq layer. Reads env only, no API call."""
    try:
        from core import groq_client
    except Exception:  # noqa: BLE001 — no Groq client means both layers are off
        return [("Prompt Guard", False, ""), ("Qwen judge", False, "")]
    return [
        ("Prompt Guard", groq_client.layer_enabled("PROMPT_GUARD"), groq_client.prompt_guard_model()),
        ("Qwen judge", groq_client.layer_enabled("LLM_JUDGE"), groq_client.judge_model()),
    ]


def ml_layers_status() -> list[tuple[str, str]]:
    """Each Groq layer as (name, "bật · <model>" | "tắt")."""
    return [(name, f"bật · {model}" if on else "tắt") for name, on, model in _ml_layers()]


def ml_layers_summary() -> str:
    """One line for the header: the layers that are on, or "tắt"."""
    on = [name for name, enabled, _ in _ml_layers() if enabled]
    return " · ".join(on) if on else "tắt"


class GuardedChat:
    """One console session: its own rate-limit windows, audit log and metrics."""

    def __init__(self, *, max_requests: int = 10, window_seconds: int = 60):
        # The flag only allows the judge; the plugin still runs it only when
        # layer_enabled("LLM_JUDGE") holds, so an offline session stays offline.
        self.plugins = build_production_plugins(
            max_requests=max_requests, window_seconds=window_seconds, use_llm_judge=True,
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
        # A plugin's last_* state belongs to this turn only if the plugin ran
        # this turn; otherwise it still holds the previous message's result.
        input_ran = outcome["layer"] != "rate_limiter"
        output_ran = "called" in seen and not outcome["error"]
        findings = {
            "canonical": canonicalize(text),
            "injection_match": matched_injection(text),
            "blocked_topics": blocked_topics,
            "banking_topics": banking_topics,
            "output_issues": output_issues,
            "input_length": len(text),
            "prompt_guard_score": getattr(self.input_guard, "last_prompt_guard_score", None) if input_ran else None,
            "prompt_guard_error": getattr(self.input_guard, "last_prompt_guard_error", None) if input_ran else None,
            "prompt_guard_threshold": prompt_guard_threshold(),
            "judge": self._judge() if output_ran else None,
        }
        steps = self._steps(outcome, llm_called="called" in seen, output_issues=output_issues,
                            findings=findings)
        return Turn(
            user_id=user_id, text=text, reply=outcome["reply"], verdict=verdict,
            layer=outcome["layer"], steps=steps, latency_ms=latency_ms,
            findings=findings, error=outcome["error"],
        )

    def _judge(self) -> dict | None:
        """This turn's judge verdict, its reason passed through the PII filter."""
        judge = getattr(self.output_guard, "last_judge", None)
        if not isinstance(judge, dict):
            return None
        # A switched-off judge answers with a placeholder verdict, not a grade.
        if not judge.get("error") and str(judge.get("verdict") or "").upper() not in {"SAFE", "UNSAFE"}:
            return None
        judge = dict(judge)
        judge["reason"] = _redact(judge.get("reason") or "")
        return judge

    def _steps(self, outcome: dict, *, llm_called: bool, output_issues: list[str],
               findings: dict | None = None) -> list[Step]:
        findings = findings or {}
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
            detail = INPUT_REASONS.get(reason, "Bị chặn")
            score = findings.get("prompt_guard_score")
            if reason == "prompt_guard" and score is not None:
                detail += f" · {score:.4f} ≥ {findings.get('prompt_guard_threshold', 0.5):g}"
            guard_in = Step("input_guardrail", "Input guardrail", "block", detail)
        else:
            detail = "Không có dấu hiệu injection, đúng chủ đề ngân hàng"
            if findings.get("prompt_guard_error"):
                detail += " · Prompt Guard lỗi — cho qua (fail open)"
            elif findings.get("prompt_guard_score") is not None:
                detail += f" · Prompt Guard {findings['prompt_guard_score']:.4f}"
            guard_in = Step("input_guardrail", "Input guardrail", "pass", detail)

        if not llm_called:
            llm = Step("llm", "Blue LLM", "skip", "Không gọi model — tiết kiệm token")
        elif outcome["error"]:
            llm = Step("llm", "Blue LLM", "error", outcome["error"])
        else:
            llm = Step("llm", "Blue LLM", "pass", "Model đã trả lời")

        judge = findings.get("judge")
        judge_blocked = getattr(self.output_guard, "last_action", None) == "judge_blocked"
        if not llm_called or outcome["error"]:
            guard_out = Step("output_guardrail", "Output guardrail", "skip", "Không có câu trả lời để kiểm")
        elif layer == "output_guardrail" and judge_blocked:
            reason = (judge or {}).get("reason") or "không nêu lý do"
            guard_out = Step("output_guardrail", "Output guardrail", "block",
                             f"Qwen judge chấm UNSAFE: {reason}")
        elif layer == "output_guardrail":
            found = ", ".join(OUTPUT_ISSUES.get(name, name) for name in output_issues)
            if outcome["redacted"]:
                guard_out = Step("output_guardrail", "Output guardrail", "redact",
                                 f"Che {found} bằng [REDACTED]" + _judge_note(judge))
            else:
                guard_out = Step("output_guardrail", "Output guardrail", "block",
                                 f"Thay cả câu trả lời — phát hiện {found}")
        else:
            guard_out = Step("output_guardrail", "Output guardrail", "pass",
                             "Không có PII hay secret" + _judge_note(judge))

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
