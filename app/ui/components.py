"""HTML building blocks: header, status strip, source cards, badges, tiles.

Two decisions worth explaining.

**Everything returns a string, nothing calls Streamlit.** The caller decides how
to render. That lets the whole source list go out as *one* ``st.html`` node,
which is what makes the ``#src-…`` citation anchors and the ``:target``
highlight work — Streamlit puts every widget in its own DOM subtree, so cards
built from separate widgets could not be addressed as siblings of the answer.

**Answer markdown is converted here rather than handed to ``st.markdown``.**
CommonMark treats the inside of a raw HTML block as literal text, so an answer
wrapped in a styled ``<div>`` would show its ``**bold**`` markers. A small,
explicit converter keeps the card, the answer and the citation links in one
node and makes the output predictable.
"""

from __future__ import annotations

import html
import re
from typing import Any, Iterable

from .theme import COLORS, DOC_TYPE_LABELS, method_color, method_label

# Vietnamese function words carry no retrieval signal; highlighting them would
# paint half the chunk yellow and hide the terms that actually matched.
_STOPWORDS = {
    "là", "gì", "của", "và", "có", "không", "cho", "được", "các", "những",
    "một", "khi", "nào", "thì", "với", "để", "trong", "về", "như", "hay",
    "này", "đó", "ở", "ra", "vào", "bao", "nhiêu", "thế", "phải", "cần",
    "làm", "tôi", "em", "ạ", "ai", "mà", "nếu", "sẽ", "đang", "còn", "từ",
    "theo", "trên", "dưới", "sau", "trước", "đâu", "muốn",
}

_SCORE_LABELS = {
    "dense": ("COSINE", 4),
    "bm25": ("BM25", 2),
    "hybrid": ("RRF", 5),
    "pageindex": ("PAGEINDEX", 4),
}


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


# --- answer rendering --------------------------------------------------------
def _inline_md(text: str) -> str:
    """Bold, italic and inline code on already-escaped text."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text, flags=re.S)
    text = re.sub(r"(?<![\*\w])\*([^\*\n]+?)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"`([^`]+?)`", r"<code>\1</code>", text)
    return text


# Patterns live outside the f-strings: a backslash inside an f-string
# expression is a SyntaxError before Python 3.12, and labs pin 3.11.
_NUMBERED = re.compile(r"^\d+[\.\)]\s+")
_BULLETED = re.compile(r"^[-*•]\s+")
_HEADING = re.compile(r"^#{1,6}\s+(.*?)(?:\s+#+)?$")
_TABLE_RULE = re.compile(r"^[\s|:\-]+$")  # the |---|:---:| line under a header row
# Models put <br> inside table cells; after escaping it would print literally.
_ESCAPED_BR = re.compile(r"&lt;br\s*/?&gt;", re.IGNORECASE)


def _line_kind(line: str) -> str:
    if _HEADING.match(line):
        return "heading"
    return "table" if line.startswith("|") else "text"


def _segments(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Group a block's lines into headings, table runs and everything else.

    A model often writes "### Title" or a table straight under a sentence with
    no blank line between, so a block is split rather than rendered whole.
    """
    segments: list[tuple[str, list[str]]] = []
    for line in lines:
        kind = _line_kind(line)
        if kind != "heading" and segments and segments[-1][0] == kind:
            segments[-1][1].append(line)
        else:
            segments.append((kind, [line]))
    return segments


def _table_cells(line: str) -> list[str]:
    inner = line.strip()
    inner = inner[1:] if inner.startswith("|") else inner
    inner = inner[:-1] if inner.endswith("|") else inner
    return [_inline_md(_ESCAPED_BR.sub("<br>", cell.strip())) for cell in inner.split("|")]


def _table_html(lines: list[str]) -> str:
    """First row is the header; the dashed separator row is dropped."""
    rows = [
        _table_cells(line) for line in lines
        if not (_TABLE_RULE.match(line) and "-" in line)
    ]
    if not rows:
        return ""
    head = "".join(f"<th>{cell}</th>" for cell in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows[1:]
    )
    # The wrapper scrolls sideways so a wide table never widens the card.
    return (
        f'<div class="gr-mdtable"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _text_html(lines: list[str]) -> str:
    if all(_NUMBERED.match(line) for line in lines):
        items = "".join(f"<li>{_inline_md(_NUMBERED.sub('', line))}</li>" for line in lines)
        return f"<ol>{items}</ol>"
    if all(_BULLETED.match(line) for line in lines):
        items = "".join(f"<li>{_inline_md(_BULLETED.sub('', line))}</li>" for line in lines)
        return f"<ul>{items}</ul>"
    return "<p>" + _inline_md("<br>".join(lines)) + "</p>"


def markdown_to_html(text: str) -> str:
    """Convert the small markdown subset an LLM answer actually uses.

    Expects already-escaped text: headings, tables and lists are recognised by
    characters that escaping leaves alone (#, |, -, digits), so no markup the
    model wrote can reach the page.
    """
    out: list[str] = []
    for block in re.split(r"\n\s*\n", (text or "").strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        for kind, group in _segments(lines):
            if kind == "heading":
                title = _HEADING.match(group[0]).group(1)
                out.append(f'<p class="gr-h" role="heading" aria-level="4">{_inline_md(title)}</p>')
            elif kind == "table":
                out.append(_table_html(group))
            else:
                out.append(_text_html(group))
    return "".join(out)


def linkify_citations(
    answer_html: str,
    citations: Iterable[dict[str, Any]],
    turn: int,
    source_count: int,
) -> tuple[str, int]:
    """Turn every ``[n]`` marker into an anchor pointing at source card *n*.

    Returns the rewritten HTML and the number of markers linked. A marker whose
    number has no matching source is still rendered, but in the fallback colour
    and with a spoken-out title, because silently dropping an unverifiable
    citation would hide exactly the failure the audience should see.
    """
    by_number: dict[int, dict[str, Any]] = {}
    for citation in citations or []:
        match = re.search(r"\d+", str(citation.get("marker", "")))
        if match:
            by_number[int(match.group())] = citation

    linked = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal linked
        number = int(match.group(1))
        citation = by_number.get(number)
        valid = (
            citation.get("matched", True) if citation is not None
            else 1 <= number <= source_count
        )
        if not (1 <= number <= source_count):
            valid = False
        title = (citation or {}).get("title") or f"Nguồn {number}"
        tooltip = title if valid else f"{title} — không khớp nguồn nào"
        cls = "rag-cite" if valid else "rag-cite is-unmatched"
        linked += 1
        return (
            f'<a class="{cls}" href="#src-{turn}-{number}" '
            f'title="{_esc(tooltip)}">{number}</a>'
        )

    return re.sub(r"\[(\d+)\]", _replace, answer_html), linked


def answer_card(answer: str, citations: Iterable[dict[str, Any]], turn: int,
                source_count: int) -> str:
    """The styled answer block with clickable citation chips."""
    body, _ = linkify_citations(
        markdown_to_html(answer), citations, turn, source_count
    )
    return f'<div class="rag-answer">{body}</div>'


# --- source cards ------------------------------------------------------------
def _query_terms(query: str) -> list[str]:
    tokens = re.findall(r"[\wÀ-ỹ]+", (query or "").lower())
    terms = {t for t in tokens if len(t) >= 2 and t not in _STOPWORDS}
    return sorted(terms, key=len, reverse=True)


def highlight_terms(text: str, query: str) -> str:
    """Escape ``text`` and wrap query terms in ``<mark>``.

    Escaping happens first so a chunk containing ``<`` can never inject markup;
    the regex then runs over escaped text, which is safe because the terms are
    escaped with it.
    """
    escaped = _esc(text)
    terms = _query_terms(query)
    if not terms:
        return escaped
    pattern = "|".join(re.escape(_esc(t)) for t in terms)
    return re.sub(
        f"({pattern})", r"<mark>\1</mark>", escaped, flags=re.IGNORECASE
    )


def _pill(label: str, color: str, *, dot: bool = True, extra: str = "") -> str:
    inner = '<i class="d"></i>' if dot else ""
    return (
        f'<span class="rag-pill{extra}" style="color:{color}">{inner}'
        f"{_esc(label)}</span>"
    )


def source_card(
    index: int,
    result: dict[str, Any],
    *,
    turn: int,
    query: str = "",
    score_range: tuple[float, float] = (0.0, 1.0),
    dimmed: bool = False,
) -> str:
    """One numbered source card, addressable as ``#src-<turn>-<index>``.

    The score bar is min-max scaled across the cards on screen rather than drawn
    as an absolute fraction. RRF scores of a top-5 sit within ~5% of each other,
    so an absolute bar would render five identical full-width rails and say
    nothing; scaled, it shows the ordering the audience is being told about.
    """
    metadata = result.get("metadata") or {}
    method = str(result.get("retrieval_method") or "hybrid")
    accent = method_color(method)
    score = float(result.get("score") or 0.0)
    label, digits = _SCORE_LABELS.get(method, ("SCORE", 4))
    low, high = score_range
    span = high - low
    ratio = 0.16 + 0.84 * ((score - low) / span) if span > 1e-12 else 1.0
    ratio = max(0.08, min(1.0, ratio))

    doc_type = str(metadata.get("doc_type") or "")
    type_label = DOC_TYPE_LABELS.get(doc_type, doc_type.upper() or "TÀI LIỆU")
    url = metadata.get("url")
    link = (
        f'<a class="rag-src__link" href="{_esc(url)}" target="_blank" '
        f'rel="noopener">mở nguồn ↗</a>'
        if url
        else ""
    )

    return (
        f'<div class="rag-src" id="src-{turn}-{index}" '
        f'style="--rag-accent:{accent}'
        + (";opacity:.62" if dimmed else "")
        + '">'
        f'<div class="rag-src__head">'
        f'<div class="rag-src__n">{index}</div>'
        f'<div class="flex-1 min-w-0">'
        f'<p class="rag-src__title">{_esc(metadata.get("title") or result.get("id"))}</p>'
        f'<div class="rag-src__meta">'
        f'{_pill(type_label, COLORS["faint"], dot=False, extra=" rag-pill--type")}'
        f"{_pill(method_label(method), accent)}"
        f'<span class="rag-src__file">{_esc(metadata.get("source"))} · đoạn '
        f'{_esc(metadata.get("chunk_index"))}</span>{link}'
        f"</div></div>"
        f'<div class="rag-src__score"><b>{score:.{digits}f}</b>'
        f"<span>{label}</span></div>"
        f"</div>"
        f'<div class="rag-src__bar"><i style="width:{ratio * 100:.1f}%"></i></div>'
        f"<details>"
        f"<summary>Xem đoạn văn bản gốc</summary>"
        f'<div class="rag-chunk mt-2">'
        f"{highlight_terms(result.get('content', ''), query)}</div>"
        f"</details>"
        f"</div>"
    )


def source_list(
    results: list[dict[str, Any]],
    *,
    turn: int,
    query: str = "",
    dimmed: bool = False,
) -> str:
    """All source cards for a turn, in one node so anchors resolve."""
    if not results:
        return ""
    scores = [float(r.get("score") or 0.0) for r in results]
    score_range = (min(scores), max(scores)) if scores else (0.0, 1.0)
    cards = "".join(
        source_card(i, r, turn=turn, query=query, score_range=score_range,
                    dimmed=dimmed)
        for i, r in enumerate(results, start=1)
    )
    return f'<div class="rag-sources">{cards}</div>'


# --- chrome ------------------------------------------------------------------
def header(
    status: list[tuple[str, str, str | None]],
    *,
    mark: str = "RAG",
    title: str = "Trợ lý quy chế &amp; dịch vụ sinh viên UTC",
    subtitle: str = (
        "Hybrid retrieval (dense + BM25 → RRF) có trích dẫn kiểm chứng được — "
        "mọi câu trả lời đều chỉ rõ đoạn tài liệu đã dùng."
    ),
) -> str:
    """Product header plus the live status strip.

    ``status`` items are ``(label, value, dot_colour)``; the dot is what makes
    "index sẵn sàng" readable at a glance from the back of the room.
    ``mark``, ``title`` and ``subtitle`` are trusted HTML, not user input.
    """
    items = "".join(
        f'<div class="rag-strip__item"><div class="rag-strip__k">{_esc(k)}</div>'
        f'<div class="rag-strip__v">'
        + (f'<span class="dot" style="background:{dot}"></span>' if dot else "")
        + f"{_esc(v)}</div></div>"
        for k, v, dot in status
    )
    return (
        '<div class="rag-header"><div class="rag-header__row">'
        f'<div class="rag-mark">{mark}</div><div>'
        f'<p class="rag-title">{title}</p>'
        f'<p class="rag-sub">{subtitle}</p>'
        f'</div></div><div class="rag-strip">{items}</div></div>'
    )


def metric_tiles(tiles: list[tuple[str, str, str, str]]) -> str:
    """``(label, value, sub, accent)`` tiles — the numeric read-out of a turn."""
    body = "".join(
        f'<div class="rag-tile" style="--rag-accent:{accent}">'
        f'<div class="rag-tile__k">{_esc(k)}</div>'
        f'<div class="rag-tile__v">{_esc(v)}</div>'
        f'<div class="rag-tile__s">{_esc(sub)}</div></div>'
        for k, v, sub, accent in tiles
    )
    return f'<div class="rag-tiles">{body}</div>'


def refusal_card(trace: dict[str, Any], answer: str) -> str:
    """The safe-refusal state, styled as a deliberate outcome rather than an error."""
    best = float(trace.get("best_dense_score") or 0.0)
    threshold = float(trace.get("score_threshold") or 0.0)
    gap = threshold - best
    fallback = next(
        (s for s in (trace.get("stages") or []) if s.get("name") == "fallback"), {}
    )

    # A refusal can come from either of two gates, and the audience should be
    # told which one fired: the retrieval threshold, or the generator deciding
    # the retrieved context does not support an answer.
    if gap > 0:
        stage_label = "Chặn ở cổng ngưỡng truy hồi"
        note = (f"điểm dense cao nhất {best:.4f} &lt; ngưỡng {threshold:.3f} · "
                f"thiếu {gap:.4f}")
    else:
        stage_label = "Chặn ở bước sinh câu trả lời"
        note = (f"điểm dense cao nhất {best:.4f} ≥ ngưỡng {threshold:.3f}, "
                "nhưng ngữ cảnh lấy về không chứa căn cứ cho câu hỏi này")

    error = fallback.get("error")
    if fallback.get("triggered"):
        note += f" → fallback {_esc(fallback.get('provider') or 'pageindex')}"
        note += f" lỗi: {_esc(error)}" if error else " không trả về kết quả"

    return (
        '<div class="rag-refusal">'
        '<div class="rag-refusal__h"><span class="ic">!</span>'
        "Không đủ căn cứ trong tài liệu"
        f'<span class="rag-label" style="margin-left:auto">'
        f"{_esc(stage_label)}</span></div>"
        f'<div class="rag-refusal__b">{markdown_to_html(answer)}</div>'
        f'<div class="rag-refusal__n">{note}</div>'
        "</div>"
    )


def banner(message: str, *, kind: str = "info") -> str:
    """Inline notice. ``kind`` is ``info`` or ``warn``."""
    icon = "!" if kind == "warn" else "i"
    color = COLORS["warn"] if kind == "warn" else COLORS["muted"]
    return (
        f'<div class="rag-banner rag-banner--{kind}">'
        f'<span class="rag-banner__ic" style="color:{color}">'
        f"{icon}</span><div>{message}</div></div>"
    )


def color_legend() -> str:
    """Sidebar colour key — the same four colours as the SVG and the charts."""
    items = [
        (COLORS["dense"], "Dense", "tìm theo ngữ nghĩa"),
        (COLORS["bm25"], "BM25", "tìm theo từ khoá"),
        (COLORS["hybrid"], "Hybrid / RRF", "gộp hai bảng xếp hạng"),
        (COLORS["fallback"], "Fallback", "vectorless / từ chối"),
    ]
    body = "".join(
        f'<div class="rag-legend__i"><i style="background:{c}"></i>'
        f"<span>{_esc(name)}</span><em>{_esc(sub)}</em></div>"
        for c, name, sub in items
    )
    return f'<div class="rag-legend">{body}</div>'


def empty_state(title: str, subtitle: str) -> str:
    """A composed "nothing here yet" panel.

    Both empty columns used to render a bare dashed box with one grey sentence
    in it, which reads as a rendering failure rather than a starting point. A
    titled state tells the reader the panel is working and waiting.
    """
    return (
        f'<div class="rag-empty"><div><div class="rag-empty__t">{_esc(title)}</div>'
        f'<div class="rag-empty__s">{_esc(subtitle)}</div></div></div>'
    )


def user_bubble(text: str) -> str:
    """The user's question, as a chat bubble."""
    return f'<div class="rag-userq">{_esc(text)}</div>'


def reorder_table(trace: dict[str, Any]) -> str:
    """Compact before/after view of the lost-in-the-middle reordering."""
    rows = trace.get("reorder_map") or []
    if not rows:
        return ""
    cells = "".join(
        f'<span class="rag-reorder__c">#{r.get("from_rank")} '
        f'<em>&rarr;</em> vị trí {r.get("to_rank")}</span>'
        for r in rows
    )
    return (
        '<div class="rag-reorder">'
        '<span class="rag-label">Sắp xếp lại context</span>'
        f"{cells}</div>"
    )
