"""LLM narrative panel — auto-generated after prediction."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.narrative.config import get_api_keys
from src.narrative.service import NarrativeError, generate_narrative


def narrative_cache_key(results: pd.DataFrame, patient_id: str | None) -> str:
    pid = patient_id or str(results["patient_id"].iloc[0])
    latest = float(results["risk_score"].iloc[-1])
    return f"{pid}:{len(results)}:{latest:.8f}"


def clear_narrative_cache() -> None:
    st.session_state.pop("narrative_text", None)
    st.session_state.pop("narrative_provider", None)
    st.session_state.pop("narrative_cache_key", None)


def generate_and_cache_narrative(
    results: pd.DataFrame,
    *,
    patient_id: str | None,
    threshold: float,
) -> None:
    """Generate narrative from results and store in session state."""
    cache_key = narrative_cache_key(results, patient_id)
    if (
        st.session_state.get("narrative_cache_key") == cache_key
        and st.session_state.get("narrative_text")
    ):
        return

    gemini_key, groq_key = get_api_keys()
    text, provider = generate_narrative(
        results,
        patient_id=patient_id,
        threshold=threshold,
        provider_name="auto",
        language="vi",
        gemini_api_key=gemini_key,
        groq_api_key=groq_key,
    )
    st.session_state["narrative_text"] = text
    st.session_state["narrative_provider"] = provider
    st.session_state["narrative_cache_key"] = cache_key


def render_narrative_panel(
    results: pd.DataFrame | None,
    *,
    threshold: float,
    patient_id: str | None,
) -> None:
    """Show auto-generated narrative for the current prediction."""
    if results is None or results.empty:
        return

    cache_key = narrative_cache_key(results, patient_id)
    needs_generate = (
        st.session_state.get("narrative_cache_key") != cache_key
        or not st.session_state.get("narrative_text")
    )

    if needs_generate:
        with st.spinner("Đang tạo tường thuật tình trạng bệnh nhân..."):
            try:
                generate_and_cache_narrative(
                    results, patient_id=patient_id, threshold=threshold
                )
            except NarrativeError as exc:
                st.warning(
                    f"Không tạo được tường thuật: {exc}",
                    icon=":material/warning:",
                )
                return

    narrative = st.session_state.get("narrative_text")
    if not narrative:
        return

    st.markdown("**Tường thuật tình trạng**")
    st.info(narrative, icon=":material/clinical_notes:")
    used = st.session_state.get("narrative_provider", "unknown")
    st.caption(
        f"Nguồn: `{used}` · Tự động từ kết quả dự đoán · "
        "Chỉ hỗ trợ quyết định, không thay thế đánh giá lâm sàng."
    )
