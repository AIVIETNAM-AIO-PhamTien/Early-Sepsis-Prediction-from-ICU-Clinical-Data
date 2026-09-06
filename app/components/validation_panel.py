"""Validation result display."""

from __future__ import annotations

import streamlit as st

from src.inference.validator import ValidationResult


def render_validation_panel(validation: ValidationResult | None) -> None:
    """Render validation errors, warnings, and pass/fail status."""
    if validation is None:
        st.info(
            "Upload or select a patient file to run schema and quality checks.",
            icon=":material/upload_file:",
        )
        return

    if validation.is_valid:
        st.success(
            "Data validation passed — the file is ready for model inference.",
            icon=":material/check_circle:",
        )
    else:
        st.error(
            "Validation failed. Resolve the issues below before running prediction.",
            icon=":material/error:",
        )

    if validation.errors:
        with st.container(border=True):
            st.markdown("**:material/cancel: Errors**")
            for err in validation.errors:
                st.markdown(f"- {err}")

    if validation.warnings:
        with st.container(border=True):
            st.markdown("**:material/warning: Warnings**")
            for warn in validation.warnings:
                st.markdown(f"- {warn}")

    if validation.metadata:
        with st.expander("Validation metadata", icon=":material/data_object:"):
            st.json(validation.metadata)
