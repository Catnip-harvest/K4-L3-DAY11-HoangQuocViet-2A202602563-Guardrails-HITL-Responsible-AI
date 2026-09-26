"""Design tokens for the RAG demo UI, and the loader for the built stylesheet.

THE ONE RULE THIS FILE EXISTS TO ENFORCE
----------------------------------------
Colour is never decoration. A hue appears on screen only when it identifies
*which retrieval path* produced the thing you are looking at:

    indigo  -> dense / vector retrieval
    teal    -> BM25 / keyword retrieval
    amber   -> hybrid (RRF fusion), i.e. the path actually taken
    rose    -> the vectorless fallback / refusal path

Everything else — the header, the chat, buttons, sliders, borders, panels — is
drawn in a single cool-grey ramp plus one near-black chrome accent. The chrome
accent is deliberately NOT one of the four, so no interface furniture can ever
be mistaken for a pipeline signal.

The previous revision broke this rule: it painted the header, the answer card
and the chat bubbles in the same saturated violet/cyan/amber as the pipeline
stages, on a near-black canvas. The result read as a crypto dashboard and,
worse, made the colour code meaningless — if everything is amber, amber stops
telling you anything. It also projected badly: saturated accents on #080C16
collapse to mud under classroom lighting.

WHY LIGHT
---------
This UI answers questions about university regulations, and it is shown on a
projector in a lit room. Both point the same way. A light canvas holds contrast
under ambient light where a dark one washes out, and a white card reads as a
document, which is exactly what the audience is being shown.

WHERE THE CSS LIVES
-------------------
This module is the single source of truth for the palette, but it no longer
contains the stylesheet. The tokens below are compiled into Tailwind v4 theme
variables by ``ui/tailwind/build.py``, which emits ``ui/tailwind.build.css``.
That built file is committed, so a grader who clones the repo and runs
``streamlit run app.py`` needs no Node toolchain.

To change a colour: edit ``COLORS`` here, then run ``python ui/tailwind/build.py``.
Editing the built CSS by hand will be overwritten on the next build.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

import streamlit as st


# --- Palette -----------------------------------------------------------------
# Kept as a plain dict (not a dataclass) because it is interpolated into CSS,
# SVG and Plotly layouts as strings far more often than it is attribute-accessed.
#
# Every one of the four signal colours clears WCAG AA (4.5:1) as text on white,
# and they are tuned to roughly equal perceived weight so that none of them
# dominates the others when they sit side by side in a legend or a chart.
COLORS: Final[dict[str, str]] = {
    # canvas: a cool paper rather than pure white, so white cards read as raised
    # without needing a heavy shadow to prove it
    "bg": "#F4F6FA",
    "bg_soft": "#EBEFF6",
    "surface": "#FFFFFF",
    "surface_2": "#F8FAFC",
    "surface_3": "#EEF2F7",
    "border": "#E3E8F0",
    "border_soft": "#EDF1F6",
    "border_strong": "#CBD5E1",
    # type: one grey family only. Mixing a warm grey into a cool palette is the
    # fastest way to make a careful layout look accidental.
    "text": "#101828",
    "text_dim": "#344054",
    "muted": "#667085",
    "faint": "#98A2B3",
    # the four signal colours, and a tint of each for fills
    "dense": "#4F46E5",
    "dense_soft": "#EEF0FF",
    "bm25": "#0F766E",
    "bm25_soft": "#ECFDF6",
    "hybrid": "#B45309",
    "hybrid_soft": "#FEF6E7",
    "fallback": "#BE123C",
    "fallback_soft": "#FFF1F3",
    # status
    "ok": "#047857",
    "warn": "#A16207",
    "danger": "#DC2626",
    # interface chrome. Held apart from the four signal hues on purpose.
    "accent": "#1E293B",
    "accent_soft": "#F1F5F9",
    # highlighter for matched query terms inside a chunk. A background wash,
    # never a text colour, so it cannot be confused with amber = hybrid.
    "highlight": "#FDE68A",
}

# Be Vietnam Pro is chosen over Inter because its diacritic marks are drawn for
# Vietnamese specifically - stacked tone marks on chu hoa do not collide, which
# matters for a UI that is entirely in Vietnamese.
FONT_STACK: Final[str] = (
    '"Be Vietnam Pro", "Inter", -apple-system, BlinkMacSystemFont, '
    '"Segoe UI", Roboto, sans-serif'
)

# A serif, used only for the product title and panel headings. Regulations are
# documents, and a serif says so in a way no amount of layout can. Noto Serif is
# the pick because Google drew its Vietnamese coverage to the same standard as
# its Latin, so the tone marks do not degrade at display sizes. Scope is kept
# deliberately narrow - a serif running through the whole UI would fight the
# dense data tables.
SERIF_STACK: Final[str] = '"Noto Serif", "Source Serif 4", Georgia, serif'

MONO_STACK: Final[str] = '"JetBrains Mono", "SFMono-Regular", Consolas, monospace'

FONT_IMPORT: Final[str] = (
    "https://fonts.googleapis.com/css2"
    "?family=Be+Vietnam+Pro:wght@300;400;500;600;700"
    "&family=Noto+Serif:opsz,wght@8..144,500;8..144,600;8..144,700"
    "&family=JetBrains+Mono:wght@400;500;600"
    "&display=swap"
)

METHOD_COLORS: Final[dict[str, str]] = {
    "dense": COLORS["dense"],
    "bm25": COLORS["bm25"],
    "hybrid": COLORS["hybrid"],
    "pageindex": COLORS["fallback"],
}

METHOD_SOFT: Final[dict[str, str]] = {
    "dense": COLORS["dense_soft"],
    "bm25": COLORS["bm25_soft"],
    "hybrid": COLORS["hybrid_soft"],
    "pageindex": COLORS["fallback_soft"],
}

METHOD_LABELS: Final[dict[str, str]] = {
    "dense": "Dense",
    "bm25": "BM25",
    "hybrid": "Hybrid RRF",
    "pageindex": "PageIndex",
}

DOC_TYPE_LABELS: Final[dict[str, str]] = {
    "legal": "Pháp quy",
    "news": "Tin tức",
}


_BUILT_CSS: Final[Path] = Path(__file__).parent / "tailwind.build.css"

_FONT_LINKS: Final[str] = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    f'<link rel="stylesheet" href="{FONT_IMPORT}">'
)

_MISSING_CSS_HINT: Final[str] = (
    "ui/tailwind.build.css is missing. The stylesheet is a build artefact; "
    "regenerate it with:  python ui/tailwind/build.py"
)


@lru_cache(maxsize=1)
def _stylesheet() -> str:
    """Read the compiled stylesheet once per process.

    Cached because Streamlit re-runs the whole script on every widget
    interaction, and re-reading ~40 KB from disk on each keystroke is waste we
    can trivially avoid. The cache is per-process, so a rebuild during
    development needs a server restart to show up - which is the same thing
    Streamlit already requires for any other import-time change.
    """
    if not _BUILT_CSS.exists():
        return ""
    return _BUILT_CSS.read_text(encoding="utf-8")


def inject_css() -> None:
    """Inject webfonts and the compiled stylesheet once per rerun.

    MUST be ``st.markdown(unsafe_allow_html=True)``, not ``st.html``.

    Measured on Streamlit 1.64: ``st.html`` runs its payload through a
    sanitiser that drops ``<style>`` and ``<link>`` outright. The call then
    renders an empty node, the page comes up completely unstyled, and nothing
    is logged - so the failure looks like a broken stylesheet rather than a
    stripped one. This is the same sanitiser that deletes the ``<svg>`` subtree
    in ``render_pipeline_svg``; see ``render_svg`` in app.py, which documents
    the identical workaround for the identical reason.

    (Do not "verify" this with a hot reload. Streamlit leaves the previous
    render's ``<style>`` node in the DOM, so a hot-reloaded page keeps working
    on stale CSS and st.html appears to be fine. It only shows up on a cold
    start. This cost an hour once already.)

    Markdown cannot corrupt the payload: CommonMark treats the inside of a raw
    HTML block as literal text, so the ``*`` and ``_`` characters throughout
    minified CSS are never read as emphasis.

    A missing build artefact fails loudly rather than rendering an unstyled
    page, because an unstyled Streamlit app looks like a broken app and the
    actual cause - a skipped build step - is not guessable from the screen.
    """
    css = _stylesheet()
    if not css:
        st.error(_MISSING_CSS_HINT, icon=":material/style:")
        return
    st.markdown(f"{_FONT_LINKS}<style>{css}</style>", unsafe_allow_html=True)


def method_color(method: str) -> str:
    """Colour for a ``retrieval_method`` value, falling back to muted grey."""
    return METHOD_COLORS.get(method, COLORS["muted"])


def method_soft(method: str) -> str:
    """Tinted fill matching :func:`method_color`, for pill and badge grounds."""
    return METHOD_SOFT.get(method, COLORS["surface_3"])


def method_label(method: str) -> str:
    """Human label for a ``retrieval_method`` value."""
    return METHOD_LABELS.get(method, method or "-")
