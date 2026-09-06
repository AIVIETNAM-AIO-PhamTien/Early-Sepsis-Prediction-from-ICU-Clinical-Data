"""Patient prediction page."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.narrative_panel import clear_narrative_cache, generate_and_cache_narrative
from app.components.results_view import render_results_view
from app.components.upload import render_upload_section
from app.components.validation_panel import render_validation_panel
from src.inference.loader import ModelLoader
from src.inference.logger import log_prediction
from src.inference.predictor import PredictionError, run_prediction

st.session_state.setdefault("last_results", None)
st.session_state.setdefault("last_validation", None)
st.session_state.setdefault("last_df", None)
st.session_state.setdefault("last_patient_id", None)
st.session_state.setdefault("last_source_name", None)

loader = ModelLoader()
manifest = loader.get_manifest_info() or {}
threshold = float(manifest.get("threshold", 0.5))

st.title(":material/monitor_heart: Sepsis risk prediction")
st.caption(
    "Upload hourly ICU patient data to estimate sepsis onset risk across the ICU stay."
)

with st.container(border=True):
    st.markdown("#### :material/route: Workflow")
    step_cols = st.columns(4)
    step_cols[0].markdown(":material/upload_file: **1. Load data**")
    step_cols[1].markdown(":material/fact_check: **2. Validate**")
    step_cols[2].markdown(":material/play_circle: **3. Predict**")
    step_cols[3].markdown(":material/insights: **4. Review**")

with st.container(border=True):
    st.markdown("#### :material/upload_file: Step 1 · Patient data")
    df, patient_id, source_name = render_upload_section()

validation = st.session_state.get("last_validation")

with st.container(border=True):
    st.markdown("#### :material/fact_check: Step 2 · Data validation")
    render_validation_panel(validation)

can_predict = (
    validation is not None
    and validation.is_valid
    and st.session_state.get("last_df") is not None
)

with st.container(border=True):
    st.markdown("#### :material/play_circle: Step 3 · Run model")
    st.caption(
        "Scores each ICU hour using the Sepsyd pipeline. "
        f"Hours at or above **{threshold:.2f}** are flagged as elevated risk."
    )
    predict_col, info_col = st.columns([1, 2])
    with predict_col:
        run_clicked = st.button(
            "Run prediction",
            type="primary",
            disabled=not can_predict,
            icon=":material/play_arrow:",
            width="stretch",
        )
    with info_col:
        if not can_predict:
            st.info(
                "Load a valid patient file to enable prediction.",
                icon=":material/info:",
            )

if run_clicked:
    clear_narrative_cache()
    try:
        results, val, artifact = run_prediction(
            st.session_state["last_df"],
            st.session_state["last_patient_id"],
            loader,
        )
        if not val.is_valid:
            st.session_state["last_validation"] = val
            st.session_state["last_results"] = None
            log_prediction(
                input_file=st.session_state.get("last_source_name", "unknown"),
                patient_id=st.session_state["last_patient_id"],
                artifact=artifact,
                results=None,
                status="validation_failed",
                error="; ".join(val.errors),
            )
            st.rerun()
        else:
            st.session_state["last_results"] = results
            log_prediction(
                input_file=st.session_state.get("last_source_name", "unknown"),
                patient_id=st.session_state["last_patient_id"],
                artifact=artifact,
                results=results,
                status="success",
            )
            with st.spinner("Đang tạo tường thuật tình trạng bệnh nhân..."):
                generate_and_cache_narrative(
                    results,
                    patient_id=st.session_state["last_patient_id"],
                    threshold=threshold,
                )
            st.toast("Prediction complete", icon=":material/check_circle:")
            st.rerun()
    except PredictionError as exc:
        log_prediction(
            input_file=st.session_state.get("last_source_name", "unknown"),
            patient_id=st.session_state.get("last_patient_id", "unknown"),
            artifact=None,
            results=None,
            status="prediction_failed",
            error=str(exc),
        )
        st.error(f"Prediction failed: {exc}", icon=":material/error:")

with st.container(border=True):
    st.markdown("#### :material/insights: Step 4 · Clinical review")
    render_results_view(
        st.session_state.get("last_results"),
        threshold=threshold,
        patient_id=st.session_state.get("last_patient_id"),
    )
