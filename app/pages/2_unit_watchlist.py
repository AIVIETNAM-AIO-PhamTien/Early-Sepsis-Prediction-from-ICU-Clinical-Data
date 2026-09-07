"""Unit-level sepsis risk watchlist — multi-patient view."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.bundle_checklist import render_bundle_checklist
from app.components.results_view import build_risk_chart
from src.inference.escalation import get_escalation_tier
from src.inference.io import read_patient_file, read_patient_file_raw
from src.inference.loader import ModelLoader
from src.inference.logger import log_prediction
from src.inference.predictor import PredictionError, run_prediction
from src.shared.paths import SAMPLE_DATA_DIR

st.title(":material/groups: Unit watchlist")
st.caption(
    "View multiple patients across the unit at once, ranked by current risk — "
    "mirrors a nursing/Rapid Response Team worklist for monitoring the whole unit "
    "instead of one patient at a time."
)

loader = ModelLoader()
manifest = loader.get_manifest_info() or {}
threshold = float(manifest.get("threshold", 0.5))

uploaded_files = st.file_uploader(
    "Upload multiple patient files (.psv/.csv)",
    type=["psv", "csv"],
    accept_multiple_files=True,
)
use_samples = st.checkbox("Include sample data (sample_data)", value=not uploaded_files)

rows: list[dict] = []
patient_results: dict[str, pd.DataFrame] = {}
patient_raw_data: dict[str, pd.DataFrame] = {}
errors: list[str] = []

for idx, uploaded in enumerate(uploaded_files or []):
    suffix = Path(uploaded.name).suffix
    tmp_path = Path(f"_tmp_watchlist_{idx}{suffix}")
    tmp_path.write_bytes(uploaded.getvalue())
    try:
        df, patient_id = read_patient_file(tmp_path)
        raw_df = read_patient_file_raw(tmp_path)
        source_name = uploaded.name
    finally:
        tmp_path.unlink(missing_ok=True)

    try:
        results, validation, artifact = run_prediction(df, patient_id, loader)
        if not validation.is_valid:
            errors.append(f"{source_name}: {'; '.join(validation.errors)}")
            continue
        patient_results[patient_id] = results
        patient_raw_data[patient_id] = raw_df
        log_prediction(
            input_file=source_name,
            patient_id=patient_id,
            artifact=artifact,
            results=results,
            status="success",
        )
    except PredictionError as exc:
        errors.append(f"{source_name}: {exc}")

if use_samples:
    for sample_path in sorted(SAMPLE_DATA_DIR.glob("*.psv")):
        df, patient_id = read_patient_file(sample_path)
        raw_df = read_patient_file_raw(sample_path)
        try:
            results, validation, artifact = run_prediction(df, patient_id, loader)
            if not validation.is_valid:
                errors.append(f"{sample_path.name}: {'; '.join(validation.errors)}")
                continue
            patient_results[patient_id] = results
            patient_raw_data[patient_id] = raw_df
            log_prediction(
                input_file=sample_path.name,
                patient_id=patient_id,
                artifact=artifact,
                results=results,
                status="success",
            )
        except PredictionError as exc:
            errors.append(f"{sample_path.name}: {exc}")

for patient_id, results in patient_results.items():
    latest_score = float(results["risk_score"].iloc[-1])
    tier = get_escalation_tier(latest_score, threshold)
    rows.append(
        {
            "patient_id": patient_id,
            "n_hours": len(results),
            "latest_score": latest_score,
            "escalation": tier.name,
            "elevated_hours": int((results["prediction"] == 1).sum()),
            "model_version": results["model_version"].iloc[-1],
        }
    )

if errors:
    with st.expander(f":material/error: {len(errors)} file(s) failed", icon=":material/error:"):
        for err in errors:
            st.markdown(f"- {err}")

if not rows:
    st.info(
        "Upload multiple files or enable sample data to view the unit-wide watchlist.",
        icon=":material/info:",
    )
else:
    watchlist_df = pd.DataFrame(rows).sort_values("latest_score", ascending=False)
    n_elevated = int((watchlist_df["escalation"] == "Elevated").sum())
    n_watch = int((watchlist_df["escalation"] == "Watch").sum())

    with st.container(horizontal=True):
        st.metric("Patients on watchlist", len(watchlist_df), border=True)
        st.metric("Needs escalation (Elevated)", n_elevated, border=True)
        st.metric("Needs close monitoring (Watch)", n_watch, border=True)

    st.markdown("**Priority-ranked list**")
    st.dataframe(
        watchlist_df,
        hide_index=True,
        column_config={
            "patient_id": st.column_config.TextColumn("Patient ID"),
            "n_hours": st.column_config.NumberColumn("ICU hours"),
            "latest_score": st.column_config.ProgressColumn(
                "Latest risk", format="%.1f%%", min_value=0, max_value=1
            ),
            "escalation": st.column_config.TextColumn("Escalation tier"),
            "elevated_hours": st.column_config.NumberColumn("Elevated hours"),
            "model_version": st.column_config.TextColumn("Model"),
        },
    )

    st.markdown("**View raw data by patient**")
    selected_patient = st.selectbox(
        "Select patient",
        options=list(watchlist_df["patient_id"]),
    )
    if selected_patient:
        raw_df = patient_raw_data[selected_patient]
        with st.expander(
            f"Raw data — {selected_patient} ({raw_df.shape[0]} rows × {raw_df.shape[1]} columns)",
            icon=":material/table_view:",
            expanded=True,
        ):
            st.dataframe(raw_df, hide_index=True, width="stretch")

    elevated_ids = watchlist_df.loc[watchlist_df["escalation"] == "Elevated", "patient_id"]
    if len(elevated_ids):
        st.markdown("**:material/emergency: Patients needing escalation**")
        for patient_id in elevated_ids:
            with st.expander(f"Patient {patient_id}", icon=":material/person_search:"):
                results = patient_results[patient_id]
                st.altair_chart(build_risk_chart(results, threshold))
                render_bundle_checklist(patient_id)
