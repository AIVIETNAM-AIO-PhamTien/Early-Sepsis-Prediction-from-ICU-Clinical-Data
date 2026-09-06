"""Load API keys from environment or Streamlit secrets."""

from __future__ import annotations

import os

_PLACEHOLDER_PREFIXES = ("PASTE_", "your-", "YOUR_")


def _clean_key(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if any(value.startswith(p) for p in _PLACEHOLDER_PREFIXES):
        return None
    return value


def get_api_keys() -> tuple[str | None, str | None]:
    """Return (gemini_api_key, groq_api_key) from env or st.secrets."""
    gemini = _clean_key(os.environ.get("GEMINI_API_KEY"))
    groq = _clean_key(os.environ.get("GROQ_API_KEY"))

    if gemini and groq:
        return gemini, groq

    try:
        import streamlit as st

        if not gemini:
            gemini = _clean_key(st.secrets.get("GEMINI_API_KEY"))  # type: ignore[attr-defined]
        if not groq:
            groq = _clean_key(st.secrets.get("GROQ_API_KEY"))  # type: ignore[attr-defined]
    except Exception:
        pass

    return gemini, groq
