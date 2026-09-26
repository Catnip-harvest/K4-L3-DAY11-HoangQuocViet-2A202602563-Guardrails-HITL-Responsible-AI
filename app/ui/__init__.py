"""UI kit for Streamlit chatbots — design tokens and HTML building blocks.

Deliberately free of any retrieval, generation or model logic. Everything here
takes plain dicts and turns them into pixels, which is what lets an interface be
built and demoed before the backend exists.

Modules
-------
``theme``        colour tokens, fonts, and the compiled-stylesheet loader
``components``   header, source cards, badges, tiles, citation links, states

The one rule the whole kit enforces: colour is never decoration. See the module
docstring in ``theme`` and the README.
"""

from __future__ import annotations

__all__ = ["components", "theme"]
