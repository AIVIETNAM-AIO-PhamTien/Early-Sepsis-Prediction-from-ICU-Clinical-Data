"""File upload and sample data selection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.narrative_panel import clear_narrative_cache
from src.inference.io import read_patient_file
from src.inference.validator import validate_patient_df
from src.shared.paths import SAMPLE_DATA_DIR


def render_upload_section() -> tuple[pd.DataFrame | None, str | None, str | None]:
    """Render upload controls and return (df, patient_id, source_name)."""
    col_upload, col_sample = st.columns([3, 2], gap="large")

    with col_upload:
        uploaded = st.file_uploader(
            "Upload patient file",
            type=["psv", "csv"],
            accept_multiple_files=False,
            help="PhysioNet Challenge 2019 format: hourly ICU vitals and labs (.psv or .csv).",
        )

    with col_sample:
        sample_files = sorted(SAMPLE_DATA_DIR.glob("*.psv"))
        sample_names = [f.name for f in sample_files]
        selected_sample = st.selectbox(
            "Or load a sample case",
            options=["Select sample…"] + sample_names,
            index=0,
            help="Demo patients from the PhysioNet 2019 training set.",
        )

    df: pd.DataFrame | None = None
    patient_id: str | None = None
    source_name: str | None = None

    if uploaded is not None:
        suffix = Path(uploaded.name).suffix
        tmp_path = Path(f"_tmp_upload{suffix}")
        tmp_path.write_bytes(uploaded.getvalue())
        try:
            df, patient_id = read_patient_file(tmp_path)
            source_name = uploaded.name
        finally:
            tmp_path.unlink(missing_ok=True)
    elif selected_sample != "Select sample…":
        sample_path = SAMPLE_DATA_DIR / selected_sample
        df, patient_id = read_patient_file(sample_path)
        source_name = selected_sample

    if df is not None and patient_id is not None:
        validation = validate_patient_df(df)
        clear_narrative_cache()
        st.session_state["last_validation"] = validation
        st.session_state["last_df"] = df
        st.session_state["last_patient_id"] = patient_id
        st.session_state["last_source_name"] = source_name

        meta = validation.metadata
        st.markdown("**Loaded patient summary**")
        with st.container(horizontal=True):
            st.metric("Patient ID", patient_id, border=True)
            st.metric("ICU length (hours)", meta.get("n_rows", 0), border=True)
            st.metric("Missing values", f"{meta.get('missing_pct', 0):.1f}%", border=True)
            st.metric("Source file", source_name or "—", border=True)

    return df, patient_id, source_name
