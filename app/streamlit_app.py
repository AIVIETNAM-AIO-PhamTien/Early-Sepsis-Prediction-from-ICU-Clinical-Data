"""Streamlit entry point."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.loader import ModelLoader
from src.shared.paths import PROJECT_ROOT

st.set_page_config(
    page_title="Sepsis Early Warning | ICU",
    page_icon=":material/monitor_heart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #F0F7FB !important;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #C5DFF0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### :material/monitor_heart: Sepsis Early Warning")
    st.caption("ICU sepsis risk scoring · PhysioNet 2019 · Sepsyd baseline")

    loader = ModelLoader()
    manifest = loader.get_manifest_info()
    artifact_ok = False
    artifact_error = loader.last_error

    if manifest:
        try:
            loader.get_artifact()
            artifact_ok = True
        except Exception:
            artifact_error = loader.last_error

    with st.container(border=True):
        st.markdown("**Model status**")
        if artifact_ok and manifest:
            st.badge("Ready", icon=":material/check_circle:", color="green")
            st.metric("Model version", manifest.get("model_version", "—"))
            st.metric("Decision threshold", manifest.get("threshold", "—"))
            st.caption(f"Pipeline: `{manifest.get('pipeline', '—')}`")
            st.caption(f"Updated: {manifest.get('updated_at', '—')}")
        elif manifest:
            st.badge("Load failed", icon=":material/error:", color="red")
            if artifact_error:
                st.error(artifact_error)
        else:
            st.badge("Unavailable", icon=":material/warning:", color="orange")
            if artifact_error:
                st.caption(artifact_error)

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "Hourly ICU vitals and labs are scored with the Sepsyd gradient-boosted "
        "model (CinC 2019). Upload one patient file per run."
    )
    st.caption(
        ":material/info: Research and decision-support only — not a regulated "
        "medical device."
    )
    st.caption(f"Project: `{PROJECT_ROOT.name}`")

pages = {
    "Clinical workflow": [
        st.Page(
            "pages/1_predict.py",
            title="Patient prediction",
            icon=":material/person_search:",
            default=True,
        ),
    ],
}

pg = st.navigation(pages)
pg.run()
