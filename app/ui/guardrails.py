"""HTML building blocks for the guardrails console. Pure functions returning str.

THE KIT'S ONE RULE, APPLIED TO GUARDRAILS
-----------------------------------------
A hue appears only when it names the layer that decided the outcome:

    indigo  -> Blue LLM answered, nothing had to intervene
    teal    -> output guardrail redacted or replaced the answer
    amber   -> rate limiter stopped the request
    rose    -> input guardrail stopped the request

A layer that merely passed a message along stays grey. So in any trace at most
one step is coloured, and a glance at the colour answers "who handled this?".
The kit's token names (dense, bm25, hybrid, fallback) are reused rather than
renamed so the compiled stylesheet needs no rebuild.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .components import markdown_to_html
from .theme import COLORS

LAYER_COLORS: dict[str, str] = {
    "llm": COLORS["dense"],
    "output_guardrail": COLORS["bm25"],
    "rate_limiter": COLORS["hybrid"],
    "input_guardrail": COLORS["fallback"],
}

LAYER_NAMES: dict[str, str] = {
    "llm": "Blue LLM",
    "output_guardrail": "Output guardrail",
    "rate_limiter": "Rate limiter",
    "input_guardrail": "Input guardrail",
}

STATUS_LABELS: dict[str, str] = {
    "pass": "cho qua",
    "block": "chặn",
    "redact": "đã che",
    "skip": "bỏ qua",
    "error": "lỗi",
}

# The LLM judge's four criteria (keys as the output guardrail reports them).
JUDGE_CRITERIA_VI: dict[str, str] = {
    "safety": "an toàn",
    "relevance": "liên quan",
    "accuracy": "chính xác",
    "tone": "giọng điệu",
}

STYLESHEET = Path(__file__).with_name("guardrails.css")


def stylesheet() -> str:
    return STYLESHEET.read_text(encoding="utf-8")


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _reply_html(text: str) -> str:
    """Escape first (the reply is model output), then light markdown, then mark redactions."""
    body = markdown_to_html(html.escape(text or ""))
    return body.replace("[REDACTED]", '<span class="gr-redacted">[REDACTED]</span>')


def deciding_layer(turn) -> str:
    """The layer whose colour this turn carries."""
    if turn.verdict == "answered":
        return "llm"
    return turn.layer or "llm"


# --- chat --------------------------------------------------------------------

def reply_card(turn, model_label: str | None = None) -> str:
    """The assistant's side of a turn, headed by the layer that produced it.

    model_label names what wrote an answer: the Blue model, or None for a
    simulation, so a canned reply is never presented as the model's.
    """
    if turn.verdict == "error":
        return (
            '<div class="gr-reply gr-reply--error">'
            '<div class="gr-reply__h">Lỗi gọi model · không phải bị chặn</div>'
            f'<div class="gr-reply__n">{_esc(turn.error)}</div>'
            '<div class="gr-reply__n">Kiểm tra API key hoặc quota. Lượt này không được tính '
            "là một lần phòng thủ thành công.</div></div>"
        )

    layer = deciding_layer(turn)
    color = LAYER_COLORS[layer]
    headings = {
        "answered": f"Blue LLM trả lời · {model_label}" if model_label else "Mô phỏng · chưa gọi model thật",
        "redacted": "Output guardrail đã che dữ liệu",
        "blocked": f"Chặn ở {LAYER_NAMES.get(layer, layer)}",
    }
    note = next((s.detail for s in turn.steps if s.key == layer and s.status != "pass"), "")
    return (
        f'<div class="gr-reply" style="--gr-accent:{color}">'
        f'<div class="gr-reply__h"><i class="gr-dot"></i>{_esc(headings[turn.verdict])}'
        f'<span class="gr-reply__ms">{turn.latency_ms:.0f} ms</span></div>'
        f'<div class="gr-reply__b">{_reply_html(turn.reply)}</div>'
        + (f'<div class="gr-reply__n">{_esc(note)}</div>' if note else "")
        + "</div>"
    )


def thinking_bubble(simulated: bool = False) -> str:
    """Placeholder while a turn runs. Grey on purpose: no layer has decided yet.

    A simulation is not Blue, so it is not announced as Blue either.
    """
    label = "Mô phỏng đang trả lời…" if simulated else "Blue đang trả lời…"
    return (
        '<div class="gr-reply gr-thinking" role="status" aria-live="polite">'
        f'<span class="gr-thinking__t">{_esc(label)}</span>'
        '<span class="gr-thinking__dots" aria-hidden="true"><i></i><i></i><i></i></span>'
        "</div>"
    )


def pending_trace(burst_size: int | None = None) -> str:
    """Trace column while a turn runs, instead of the previous turn's trace."""
    title = f"Đang gửi {burst_size} tin qua 4 lớp…" if burst_size else "Đang chạy qua 4 lớp…"
    order = " → ".join(LAYER_NAMES[key] for key in ("rate_limiter", "input_guardrail", "llm", "output_guardrail"))
    return (
        '<div class="gr-pending" role="status">'
        f'<div class="gr-pending__t"><i class="gr-dot"></i>{_esc(title)}</div>'
        f'<div class="gr-pending__s">{_esc(order)}</div>'
        "</div>"
    )


def burst_card(passed: int, blocked: int, max_requests: int, window_seconds: int) -> str:
    color = LAYER_COLORS["rate_limiter"]
    return (
        f'<div class="gr-reply" style="--gr-accent:{color}">'
        f'<div class="gr-reply__h"><i class="gr-dot"></i>Thử spam · {passed + blocked} tin liên tiếp</div>'
        f'<div class="gr-reply__b"><p><strong>{passed}</strong> tin qua, '
        f"<strong>{blocked}</strong> tin bị rate limiter chặn.</p></div>"
        f'<div class="gr-reply__n">Hạn mức {max_requests} tin / {window_seconds} giây cho mỗi user, '
        "tính cả những tin user này vừa gửi trước đó trong cửa sổ. "
        "Các tin bị chặn không tốn token nào.</div></div>"
    )


# --- trace -------------------------------------------------------------------

def trace(turn) -> str:
    """The four layers in order, with only the deciding one coloured."""
    decider = deciding_layer(turn)
    rows = []
    for index, step in enumerate(turn.steps, start=1):
        acted = step.key == decider and step.status in {"pass", "block", "redact"} and turn.verdict != "error"
        classes = ["gr-step", f"is-{step.status}"] + (["is-acted"] if acted else [])
        rows.append(
            f'<div class="{" ".join(classes)}" style="--gr-accent:{LAYER_COLORS[step.key]}">'
            f'<div class="gr-step__n">{index}</div>'
            f'<div><div class="gr-step__t">{_esc(step.label)}</div>'
            f'<div class="gr-step__d">{_esc(step.detail)}</div></div>'
            f'<span class="gr-chip">{_esc(STATUS_LABELS.get(step.status, step.status))}</span>'
            "</div>"
        )
    return f'<div class="gr-trace">{"".join(rows)}</div>'


def findings(turn) -> str:
    """Technical detail behind the trace, for the expander."""
    f = turn.findings
    changed = f.get("canonical") != (turn.text or "").strip().casefold()

    def row(label: str, value: str) -> str:
        return f'<div class="gr-kv"><span>{_esc(label)}</span><div>{value}</div></div>'

    def code(value: str) -> str:
        return f"<code>{_esc(value)}</code>"

    def words(items) -> str:
        return ", ".join(code(i) for i in items) if items else '<span class="gr-none">không có</span>'

    not_run = '<span class="gr-none">không chạy</span>'

    def prompt_guard() -> str:
        score, error = f.get("prompt_guard_score"), f.get("prompt_guard_error")
        threshold = f.get("prompt_guard_threshold", 0.5)
        if error:
            return f'lỗi — cho qua (fail open) <span class="gr-none">{_esc(error)}</span>'
        if score is None:
            return not_run
        side = "≥ ngưỡng, chặn" if score >= threshold else "dưới ngưỡng"
        return f'{code(f"{score:.4f}")} <span class="gr-none">ngưỡng {threshold:g} · {side}</span>'

    def judge_verdict() -> str:
        judge = f.get("judge")
        if not judge:
            return not_run
        if judge.get("error"):
            return f'lỗi — cho qua (fail open) <span class="gr-none">{_esc(judge["error"])}</span>'
        verdict = judge.get("verdict") or ("SAFE" if judge.get("safe") else "UNSAFE")
        scores = judge.get("scores") or {}
        shown = " · ".join(
            f"{_esc(JUDGE_CRITERIA_VI.get(name, name))} {_esc(value)}/5"
            for name, value in scores.items() if value is not None
        )
        return code(verdict) + (f' <span class="gr-none">{shown}</span>' if shown else "")

    judge = f.get("judge") or {}
    rows = [
        row("Độ dài", f"{f.get('input_length', 0)} ký tự"),
        row("Sau chuẩn hoá", code(f.get("canonical", "")[:160]) + (
            ' <span class="gr-none">(khác bản gốc: ký tự ẩn, dấu hoặc chữ full-width đã được gỡ)</span>'
            if changed else "")),
        row("Khớp mẫu injection", code(f["injection_match"]) if f.get("injection_match")
            else '<span class="gr-none">không</span>'),
        row("Từ khoá ngân hàng", words(f.get("banking_topics"))),
        row("Chủ đề bị cấm", words(f.get("blocked_topics"))),
        row("Prompt Guard (Meta)", prompt_guard()),
        row("Output filter thấy", words(f.get("output_issues"))),
        row("Qwen judge", judge_verdict()),
    ]
    if judge.get("reason") and not judge.get("error"):
        rows.append(row("Lý do của judge", _esc(judge["reason"])))
    return f'<div class="gr-kvs">{"".join(rows)}</div>'


def legend() -> str:
    items = [
        ("llm", "model tự trả lời"),
        ("output_guardrail", "che / thay câu trả lời"),
        ("rate_limiter", "chặn vì spam"),
        ("input_guardrail", "chặn trước khi gọi model"),
    ]
    body = "".join(
        f'<div class="rag-legend__i"><i style="background:{LAYER_COLORS[key]}"></i>'
        f"<span>{_esc(LAYER_NAMES[key])}</span><em>{_esc(sub)}</em></div>"
        for key, sub in items
    )
    return f'<div class="rag-legend">{body}</div>'


# --- egress ------------------------------------------------------------------

def egress_verdict(allowed: bool, reason: str, destination: str) -> str:
    # Egress is a policy verdict, not a layer in the chat path, so it uses the
    # status colours rather than spending one of the four layer hues.
    color = COLORS["ok"] if allowed else COLORS["danger"]
    title = "Cho phép gửi" if allowed else "Chặn — dữ liệu không được rời hệ thống"
    return (
        f'<div class="gr-reply" style="--gr-accent:{color}">'
        f'<div class="gr-reply__h"><i class="gr-dot"></i>{_esc(title)}</div>'
        f'<div class="gr-reply__b"><p><code>{_esc(destination) or "(trống)"}</code></p></div>'
        f'<div class="gr-reply__n">{_esc(reason)}</div></div>'
    )


def egress_table(rows: list[dict]) -> str:
    """Preset egress cases with the policy's actual verdict next to the expected one."""
    body = "".join(
        "<tr>"
        f"<td><code>{_esc(r['destination'])}</code></td>"
        f"<td>{_esc(r['payload'])}</td>"
        f'<td class="{"gr-ok" if r["allowed"] else "gr-no"}">{"cho phép" if r["allowed"] else "chặn"}</td>'
        f'<td>{"✓" if r["allowed"] == r["expected"] else "✗ lệch kỳ vọng"}</td>'
        "</tr>"
        for r in rows
    )
    return (
        '<table class="gr-table"><thead><tr><th>Đích</th><th>Payload</th>'
        f"<th>Chính sách</th><th>Đúng?</th></tr></thead><tbody>{body}</tbody></table>"
    )
