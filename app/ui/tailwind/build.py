"""Compile ui/tailwind/input.css -> ui/tailwind.build.css.

Run this after editing COLORS in ui/theme.py or anything in input.css:

    python ui/tailwind/build.py

WHY THERE IS A BUILD STEP AT ALL
--------------------------------
Streamlit strips <script> tags out of anything passed to st.html, so the
Tailwind Play CDN - which is a script - cannot work here. The only way to get
real Tailwind into a Streamlit page is to compile a stylesheet ahead of time and
inject it as plain CSS. That is what this does.

WHY THE OUTPUT IS COMMITTED
---------------------------
ui/tailwind.build.css is checked in and node_modules is not. A grader clones the
repo, runs `pip install -r requirements.txt && streamlit run app.py`, and gets
the styled app with no Node toolchain anywhere in the picture. The build step is
for whoever changes the design, not for whoever runs it.

WHY THE TOKENS ARE GENERATED RATHER THAN HAND-WRITTEN
-----------------------------------------------------
The palette has to exist in two places at once: in CSS for the page, and in
Python for the Plotly figures and the hand-written SVG diagram. Keeping two
hand-maintained copies in sync is a promise nobody keeps. So ui/theme.py owns
the values and this script writes the CSS half, which means a colour can only
ever be changed in one place.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# Import the palette from the package rather than re-reading the file, so a
# typo in a token name fails here instead of silently emitting a broken var.
sys.path.insert(0, str(REPO))
from ui.theme import COLORS, FONT_STACK, MONO_STACK, SERIF_STACK  # noqa: E402


GENERATED = HERE / "tokens.generated.css"
INPUT = HERE / "input.css"
OUTPUT = REPO / "ui" / "tailwind.build.css"

BANNER = (
    "/* GENERATED FILE - do not edit.\n"
    "   Written by ui/tailwind/build.py from the COLORS dict in ui/theme.py.\n"
    "   Change a colour there, then re-run: python ui/tailwind/build.py */\n"
)


def _theme_block() -> str:
    """Tailwind v4 @theme: every token becomes a utility AND a CSS variable.

    Declaring --color-dense here is what makes `text-dense`, `bg-dense` and
    `border-dense` exist as utilities, so the palette is usable from markup and
    from @apply without a second declaration anywhere.
    """
    colors = "\n".join(
        f"  --color-{name.replace('_', '-')}: {value};"
        for name, value in COLORS.items()
    )
    return (
        "@theme {\n"
        f"{colors}\n"
        "\n"
        f"  --font-sans: {FONT_STACK};\n"
        f"  --font-serif: {SERIF_STACK};\n"
        f"  --font-mono: {MONO_STACK};\n"
        "\n"
        "  /* Radius varies by nesting depth on purpose: a container is softer\n"
        "     than the cards inside it, which are softer than the chips inside\n"
        "     those. One radius everywhere is the flattest tell of a template. */\n"
        "  --radius-chip: 5px;\n"
        "  --radius-inner: 8px;\n"
        "  --radius-card: 12px;\n"
        "  --radius-panel: 16px;\n"
        "\n"
        "  /* Shadows are tinted with the page's own slate rather than pure\n"
        "     black, and all cast downward from one light source. */\n"
        "  --shadow-card: 0 1px 2px rgba(16, 24, 40, .04),"
        " 0 1px 3px rgba(16, 24, 40, .06);\n"
        "  --shadow-raised: 0 4px 8px -2px rgba(16, 24, 40, .06),"
        " 0 2px 4px -2px rgba(16, 24, 40, .04);\n"
        "  --shadow-pop: 0 12px 24px -6px rgba(16, 24, 40, .10),"
        " 0 4px 8px -4px rgba(16, 24, 40, .06);\n"
        "}\n"
    )


def _alias_block() -> str:
    """Plain --rag-* aliases for code that writes inline styles.

    components.py sets `style="--rag-accent: …"` on a card to colour it by
    retrieval method, and reads var(--rag-mono) for numeric runs. Those are
    inline styles on generated HTML, not classes, so they need stable variable
    names that do not depend on Tailwind's naming.
    """
    colors = "\n".join(
        f"  --rag-{name.replace('_', '-')}: {value};"
        for name, value in COLORS.items()
    )
    return (
        ":root {\n"
        f"{colors}\n"
        f"  --rag-font: {FONT_STACK};\n"
        f"  --rag-serif: {SERIF_STACK};\n"
        f"  --rag-mono: {MONO_STACK};\n"
        "}\n"
    )


def main() -> int:
    GENERATED.write_text(
        f"{BANNER}\n{_theme_block()}\n{_alias_block()}", encoding="utf-8"
    )
    print(f"wrote {GENERATED.relative_to(REPO)} ({len(COLORS)} colour tokens)")

    npx = "npx.cmd" if sys.platform == "win32" else "npx"
    result = subprocess.run(
        [npx, "--no-install", "tailwindcss", "-i", str(INPUT), "-o", str(OUTPUT),
         "--minify"],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    # Tailwind writes its progress line to stderr even on success, so the exit
    # code is the only honest signal here.
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        print(
            "\nBuild failed. If this says the CLI is missing, run:\n"
            "  cd ui/tailwind && npm install",
            file=sys.stderr,
        )
        return result.returncode

    size = OUTPUT.stat().st_size
    print(f"wrote {OUTPUT.relative_to(REPO)} ({size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
