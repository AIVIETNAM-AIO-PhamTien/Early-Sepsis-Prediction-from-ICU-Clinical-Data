"""Prediction results display."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app.components.bundle_checklist import render_bundle_checklist
from app.components.narrative_panel import render_narrative_panel
from src.inference.escalation import ESCALATION_TIERS, get_escalation_tier


def build_risk_chart(results: pd.DataFrame, threshold: float) -> alt.Chart:
    plot_df = results[["ICULOS", "risk_score"]].copy()
    plot_df["risk_pct"] = plot_df["risk_score"] * 100

    area = (
        alt.Chart(plot_df)
        .mark_area(opacity=0.18, color="#0B6E99")
        .encode(
            x=alt.X("ICULOS:Q", title="ICU length of stay (hours)"),
            y=alt.Y("risk_pct:Q", title="Sepsis risk score (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=[
                alt.Tooltip("ICULOS:Q", title="Hour"),
                alt.Tooltip("risk_pct:Q", title="Risk (%)", format=".1f"),
            ],
        )
    )
    line = (
        alt.Chart(plot_df)
        .mark_line(color="#0B6E99", strokeWidth=2.5)
        .encode(x="ICULOS:Q", y="risk_pct:Q")
    )
    threshold_line = (
        alt.Chart(pd.DataFrame({"y": [threshold * 100]}))
        .mark_rule(color="#C62828", strokeDash=[6, 4], strokeWidth=2)
        .encode(y="y:Q")
    )
    threshold_label = (
        alt.Chart(pd.DataFrame({"y": [threshold * 100], "label": [f"Threshold {threshold:.2f}"]}))
        .mark_text(align="left", dx=6, dy=-6, color="#C62828", fontSize=12)
        .encode(y="y:Q", text="label:N")
    )
    return (area + line + threshold_line + threshold_label).properties(height=320)


def render_results_view(
    results: pd.DataFrame | None,
    *,
    threshold: float = 0.5,
    patient_id: str | None = None,
) -> None:
    """Render prediction KPIs, risk timeline, and hourly table."""
    if results is None or results.empty:
        st.info(
            "Run prediction to review hourly sepsis risk scores and flagged ICU hours.",
            icon=":material/insights:",
        )
        return

    latest_score = float(results["risk_score"].iloc[-1])
    max_score = float(results["risk_score"].max())
    n_positive = int((results["prediction"] == 1).sum())
    n_hours = len(results)
    latest_tier = get_escalation_tier(latest_score, threshold)

    header = f"Patient **{patient_id}**" if patient_id else "Prediction summary"
    st.markdown(header)

    with st.container(horizontal=True):
        st.metric(
            "Latest risk score",
            f"{latest_score:.1%}",
            border=True,
            help="Risk score at the final available ICU hour.",
        )
        st.metric(
            "Peak risk score",
            f"{max_score:.1%}",
            border=True,
            help="Highest hourly risk score across the stay.",
        )
        st.metric(
            "Elevated-risk hours",
            n_positive,
            f"{n_positive / n_hours:.0%} of stay" if n_hours else None,
            border=True,
            help=f"Hours with score ≥ {threshold:.2f}.",
        )
        st.metric(
            "ICU hours scored",
            n_hours,
            border=True,
        )

    badge_col, note_col = st.columns([1, 3])
    with badge_col:
        st.badge(
            f"Current status: {latest_tier.name}",
            icon=latest_tier.icon,
            color=latest_tier.color,
        )
    with note_col:
        st.caption(
            "Scores are model estimates for early warning — correlate with clinical "
            "context, labs, and bedside assessment before action."
        )

    with st.expander("Escalation policy", icon=":material/policy:"):
        for tier in ESCALATION_TIERS:
            current = " · **(current status)**" if tier.name == latest_tier.name else ""
            target = f"within {tier.target_minutes} min" if tier.target_minutes else "no limit"
            st.markdown(
                f"- **{tier.name}**{current} — {tier.action} "
                f"(responsible: {tier.responsible_role}, target: {target})"
            )

    st.markdown("**Risk trajectory**")
    st.altair_chart(build_risk_chart(results, threshold))

    if latest_tier.name == "Elevated":
        st.divider()
        render_bundle_checklist(patient_id or "unknown")

    display_df = results[
        ["patient_id", "ICULOS", "risk_score", "prediction", "model_version"]
    ].copy()

    st.markdown("**Hourly predictions**")
    st.dataframe(
        display_df,
        hide_index=True,
        column_config={
            "patient_id": st.column_config.TextColumn("Patient ID", width="small"),
            "ICULOS": st.column_config.NumberColumn(
                "ICU hour",
                help="Hour since ICU admission.",
                format="%d",
            ),
            "risk_score": st.column_config.ProgressColumn(
                "Risk score",
                format="%.1f%%",
                min_value=0,
                max_value=1,
            ),
            "prediction": st.column_config.CheckboxColumn(
                "Elevated risk",
                help=f"Score ≥ {threshold:.2f}",
                disabled=True,
            ),
            "model_version": st.column_config.TextColumn("Model", width="small"),
        },
    )

    render_narrative_panel(results, threshold=threshold, patient_id=patient_id)
