"""Hour-1 Sepsis Bundle checklist — shown once a patient reaches the Elevated tier."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

BUNDLE_ITEMS = [
    ("blood_cultures", "Obtain blood cultures (before giving antibiotics)"),
    ("lactate", "Measure blood lactate"),
    ("antibiotics", "Broad-spectrum antibiotics"),
    ("fluids", "30 mL/kg crystalloid fluid bolus for hypotension or lactate ≥ 4 mmol/L"),
]

BUNDLE_TARGET_MINUTES = 60


def _item_key(patient_id: str, item_key: str) -> str:
    return f"bundle::{patient_id}::{item_key}"


def _time_zero_key(patient_id: str) -> str:
    return f"bundle_time_zero::{patient_id}"


def render_bundle_checklist(patient_id: str) -> None:
    """Render the Surviving Sepsis Campaign Hour-1 bundle checklist for one patient."""
    time_zero_key = _time_zero_key(patient_id)
    if time_zero_key not in st.session_state:
        st.session_state[time_zero_key] = datetime.now()
    time_zero = st.session_state[time_zero_key]
    elapsed_min = (datetime.now() - time_zero).total_seconds() / 60

    st.markdown("**:material/emergency: Hour-1 Sepsis Bundle**")
    st.caption(
        f"Time zero (when Elevated was detected): {time_zero.strftime('%H:%M:%S')} · "
        f"{elapsed_min:.0f} min elapsed / {BUNDLE_TARGET_MINUTES} min target."
    )
    if elapsed_min > BUNDLE_TARGET_MINUTES:
        st.warning("Bundle target time has been exceeded.", icon=":material/schedule:")

    for item_key, label in BUNDLE_ITEMS:
        key = _item_key(patient_id, item_key)
        ts_key = f"{key}::ts"
        checked = st.checkbox(label, key=key)
        if checked and ts_key not in st.session_state:
            st.session_state[ts_key] = datetime.now()
        if not checked and ts_key in st.session_state:
            del st.session_state[ts_key]
        if checked:
            st.caption(f"✓ completed at {st.session_state[ts_key].strftime('%H:%M:%S')}")

    done = sum(1 for item_key, _ in BUNDLE_ITEMS if st.session_state.get(_item_key(patient_id, item_key)))
    st.progress(done / len(BUNDLE_ITEMS), text=f"{done}/{len(BUNDLE_ITEMS)} items completed")
    st.caption(
        "Illustrative checklist based on the Surviving Sepsis Campaign Hour-1 Bundle — "
        "decision support only, not a substitute for the unit's official protocol."
    )
