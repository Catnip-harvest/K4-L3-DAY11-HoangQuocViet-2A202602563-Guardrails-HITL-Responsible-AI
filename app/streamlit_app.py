"""VinBank guardrails console — see which layer handled each message.

    .venv/Scripts/python.exe -m streamlit run app/streamlit_app.py   (from the repo root)

Runs on simulated models by default, so it needs no API key. "Blue thật" uses
the same Blue model the graded suite calls, configured by the repo's .env.
Not graded; the graded artifacts still come from `python src/main.py`.
"""
from __future__ import annotations

import asyncio

import streamlit as st

st.set_page_config(
    page_title="VinBank · Guardrails",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

import backend  # noqa: E402  (must follow set_page_config)
from ui import components, guardrails, theme  # noqa: E402

DEFAULT_MAX_REQUESTS = 10
DEFAULT_WINDOW_SECONDS = 60

# The monitoring module writes its alerts in English for metrics.json; the
# console shows them in the language of the rest of the page.
ALERT_TEXT = {
    "block_rate": "Hơn một nửa yêu cầu bị chặn — có thể đang bị tấn công, hoặc bộ lọc quá chặt.",
    "rate_limit_hits": "Nhiều yêu cầu chạm rate limit — có thể đang bị spam.",
    "judge_fail_rate": "LLM judge từ chối nhiều câu trả lời bất thường.",
}


# --- session -----------------------------------------------------------------

def run(coroutine):
    """One event loop per browser session, reused across reruns.

    asyncio.run() would close its loop after every message, and some model
    clients keep connections bound to the loop they were first used on.
    """
    if "loop" not in st.session_state:
        st.session_state.loop = asyncio.new_event_loop()
    return st.session_state.loop.run_until_complete(coroutine)


def chat_session(max_requests: int, window_seconds: int) -> backend.GuardedChat:
    """A fresh session whenever the limits change, so counters stay honest."""
    key = (max_requests, window_seconds)
    if st.session_state.get("chat_key") != key:
        st.session_state.chat = backend.GuardedChat(
            max_requests=max_requests, window_seconds=window_seconds
        )
        st.session_state.chat_key = key
        st.session_state.messages = []
    return st.session_state.chat


def model_for(mode: str) -> backend.AskLlm:
    if mode != "real":
        return backend.STUBS[mode]
    if "real_llm" not in st.session_state:
        try:
            st.session_state.real_llm = backend.create_real_llm()
        except Exception as error:  # noqa: BLE001 — shown in the trace as an LLM error
            message = f"{type(error).__name__}: {error}"

            async def unavailable(_text: str) -> str:
                raise RuntimeError(f"Không tạo được Blue model — {message}")

            return unavailable
    return st.session_state.real_llm


def send(chat: backend.GuardedChat, text: str, user_id: str, mode: str) -> None:
    turn = run(chat.send(text, user_id, model_for(mode)))
    st.session_state.messages.append({"kind": "turn", "turn": turn})


def send_burst(chat: backend.GuardedChat, user_id: str, mode: str) -> None:
    turns = [
        run(chat.send(backend.BURST_QUESTION, user_id, model_for(mode)))
        for _ in range(backend.BURST_SIZE)
    ]
    blocked = sum(1 for t in turns if t.layer == "rate_limiter")
    st.session_state.messages.append({
        "kind": "burst", "turn": turns[-1],
        "passed": len(turns) - blocked, "blocked": blocked,
    })


# --- page --------------------------------------------------------------------

theme.inject_css()
st.markdown(f"<style>{guardrails.stylesheet()}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="rag-panel-title">Bảng điều khiển</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Model phía sau", list(backend.MODES), format_func=backend.MODES.get,
        help="Các chế độ mô phỏng không gọi model nào, dùng để trình diễn lớp output.",
    )
    user_id = st.text_input("User ID", value="khach-hang-01",
                            help="Rate limit tính riêng cho từng user.")
    max_requests = st.slider("Hạn mức (tin)", 1, 20, DEFAULT_MAX_REQUESTS)
    window_seconds = st.slider("Cửa sổ (giây)", 10, 120, DEFAULT_WINDOW_SECONDS, step=5)
    chat = chat_session(max_requests, window_seconds)

    st.divider()
    samples = backend.sample_prompts()
    picked = st.selectbox("Câu mẫu", range(len(samples)), format_func=lambda i: samples[i][0])
    if st.button("Gửi câu mẫu", width="stretch"):
        send(chat, samples[picked][1], user_id, mode)
    if st.button(f"Thử spam {backend.BURST_SIZE} tin", width="stretch",
                 help="Gửi liên tiếp cùng một câu để thấy rate limiter hoạt động."):
        send_burst(chat, user_id, mode)

    st.divider()
    st.markdown('<div class="rag-label" style="margin-bottom:8px">Màu = lớp đã xử lý</div>',
                unsafe_allow_html=True)
    st.html(guardrails.legend())

    st.divider()
    if st.button("Xoá hội thoại", width="stretch"):
        del st.session_state["chat_key"]
        st.rerun()

snapshot = chat.monitor.snapshot()
st.html(components.header(
    [
        ("Model", backend.MODES[mode], theme.COLORS["ok"] if mode == "real" else None),
        ("Hạn mức", f"{max_requests} tin / {window_seconds} giây", None),
        ("Yêu cầu trong phiên", str(snapshot["total_requests"]), None),
        ("Bị chặn", f"{snapshot['blocked_requests']} ({snapshot['block_rate']:.0%})", None),
    ],
    mark="VB",
    title="VinBank · Defense-in-depth console",
    subtitle=(
        "Mỗi tin nhắn đi qua rate limiter, input guardrail, Blue LLM rồi output guardrail. "
        "Cột bên phải cho biết lớp nào đã xử lý tin nhắn gần nhất."
    ),
))

chat_tab, egress_tab, log_tab = st.tabs(["Trò chuyện", "Kiểm tra egress", "Nhật ký & metrics"])

with chat_tab:
    chat_col, trace_col = st.columns([1.6, 1], gap="large")

    with chat_col:
        for message in st.session_state.messages:
            turn = message["turn"]
            if message["kind"] == "burst":
                with st.chat_message("user", avatar=":material/person:"):
                    st.html(components.user_bubble(
                        f"{backend.BURST_SIZE} × “{backend.BURST_QUESTION}”"))
                with st.chat_message("assistant", avatar=":material/shield:"):
                    st.html(guardrails.burst_card(
                        message["passed"], message["blocked"],
                        chat.limiter.max_requests, chat.limiter.window_seconds))
                continue
            with st.chat_message("user", avatar=":material/person:"):
                shown = turn.text if len(turn.text) <= 400 else turn.text[:400] + f"… ({len(turn.text)} ký tự)"
                st.html(components.user_bubble(shown or "(tin nhắn rỗng)"))
            with st.chat_message("assistant", avatar=":material/shield:"):
                st.html(guardrails.reply_card(turn))

        if not st.session_state.messages:
            st.html(components.empty_state(
                "Gửi một tin nhắn",
                "Hỏi một câu ngân hàng, hoặc chọn một câu tấn công trong “Câu mẫu” ở thanh bên "
                "để xem lớp nào chặn nó.",
            ))

    with trace_col:
        if not st.session_state.messages:
            st.html(components.empty_state(
                "Chưa có lượt nào",
                "Sau mỗi tin nhắn, bốn lớp bảo vệ hiện ở đây theo đúng thứ tự chạy.",
            ))
        else:
            last = st.session_state.messages[-1]["turn"]
            st.markdown('<div class="rag-label" style="margin:2px 0 10px">Đường đi của tin nhắn gần nhất</div>',
                        unsafe_allow_html=True)
            st.html(guardrails.trace(last))
            with st.expander("Chi tiết kỹ thuật"):
                st.html(guardrails.findings(last))

with egress_tab:
    st.markdown(
        '<div class="rag-note" style="margin-bottom:12px">Trước khi dữ liệu rời hệ thống, '
        "chính sách egress kiểm tra đích đến và nội dung bằng luật cố định, không hỏi model.</div>",
        unsafe_allow_html=True,
    )
    form_col, result_col = st.columns([1, 1], gap="large")
    with form_col:
        destination = st.text_input("Đích đến", value="https://api.vinbank.example/v1/transfers")
        payload = st.text_area("Payload", value="approved transfer amount 500000", height=90)
        check = st.button("Kiểm tra", type="primary")
    with result_col:
        if check or "egress_last" in st.session_state:
            if check:
                st.session_state.egress_last = (destination, payload)
            dest, body = st.session_state.egress_last
            allowed, reason = backend.explain_egress(dest, body)
            st.html(guardrails.egress_verdict(allowed, reason, dest))
    with st.expander("Các ca mẫu mà bộ test chạy"):
        from assignment.pipeline import EGRESS_CASES, is_egress_allowed

        st.html(guardrails.egress_table([
            {"destination": d, "payload": p, "allowed": is_egress_allowed(d, p), "expected": e}
            for d, p, e in EGRESS_CASES
        ]))

with log_tab:
    alerts = chat.monitor.check_metrics()
    snapshot = chat.monitor.snapshot()
    st.html(components.metric_tiles([
        ("Tổng yêu cầu", str(snapshot["total_requests"]), "trong phiên này", theme.COLORS["border_strong"]),
        ("Bị chặn", str(snapshot["blocked_requests"]), f"tỉ lệ {snapshot['block_rate']:.0%}",
         theme.COLORS["border_strong"]),
        ("Rate limit", str(snapshot["rate_limit_hits"]), f"ngưỡng cảnh báo {chat.monitor.rate_limit_hit_threshold}",
         theme.COLORS["border_strong"]),
        ("Cảnh báo", str(len(alerts)), "đang bật", theme.COLORS["border_strong"]),
    ]))
    for alert in alerts:
        text = ALERT_TEXT.get(alert.metric, alert.message)
        st.html(components.banner(
            f"<b>{alert.metric}</b> · {text} (giá trị {alert.value:g}, ngưỡng {alert.threshold:g})",
            kind="warn"))

    logs = chat.audit.logs
    st.markdown(f'<div class="rag-label" style="margin:14px 0 8px">Nhật ký audit · {len(logs)} dòng</div>',
                unsafe_allow_html=True)
    if logs:
        st.dataframe(
            [
                {
                    "Thời điểm": row["responded_at"][11:19],
                    "User": row["user_id"],
                    "Tin nhắn": row["input"],
                    "Bị chặn": row["blocked"],
                    "Lớp": row["layer"] or "—",
                    "ms": row["latency_ms"],
                }
                for row in reversed(logs[-100:])
            ],
            width="stretch", hide_index=True,
        )
    else:
        st.html(components.empty_state("Nhật ký trống", "Mỗi tin nhắn sẽ được ghi lại ở đây."))

if text := st.chat_input("Hỏi VinBank một câu…"):
    send(chat, text, user_id, mode)
    st.rerun()
