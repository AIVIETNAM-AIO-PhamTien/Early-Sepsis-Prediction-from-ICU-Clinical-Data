"""Model monitoring — post-deployment surveillance of the prediction log."""

import sys
from pathlib import Path

import altair as alt
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.monitoring import daily_summary, load_prediction_logs, model_version_breakdown

st.title(":material/monitoring: Model monitoring")
st.caption(
    "Post-deployment operational monitoring: alert volume by day and by model version. "
    "For tracking drift/performance over time, not real-time patient monitoring."
)

logs = load_prediction_logs()

if logs.empty:
    st.info(
        "No prediction logs yet. Run a prediction on the Patient prediction or Unit "
        "watchlist page first — logs will appear here automatically.",
        icon=":material/info:",
    )
else:
    total_runs = len(logs)
    total_elevated = int((logs["n_positive"] > 0).sum())
    failure_rate = float((logs["status"] != "success").mean() * 100)

    with st.container(horizontal=True):
        st.metric("Total prediction runs", total_runs, border=True)
        st.metric(
            "Cases with elevated hours",
            total_elevated,
            f"{total_elevated / total_runs:.0%} of total" if total_runs else None,
            border=True,
        )
        st.metric("Error / validation-fail rate", f"{failure_rate:.1f}%", border=True)

    st.markdown("**Daily trend**")
    daily = daily_summary(logs)
    trend_chart = (
        alt.Chart(daily)
        .transform_fold(["total_runs", "elevated_cases"], as_=["metric", "value"])
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Count"),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=["date:T", "metric:N", "value:Q"],
        )
        .properties(height=280)
    )
    st.altair_chart(trend_chart, width="stretch")

    st.markdown("**By model version**")
    st.dataframe(
        model_version_breakdown(logs),
        hide_index=True,
        column_config={
            "model_version": st.column_config.TextColumn("Model version"),
            "total_runs": st.column_config.NumberColumn("Total runs"),
            "elevated_cases": st.column_config.NumberColumn("Elevated cases"),
            "elevated_rate": st.column_config.ProgressColumn(
                "Elevated rate", format="%.1f%%", min_value=0, max_value=1
            ),
        },
    )

    with st.expander("Detailed log (raw)", icon=":material/data_object:"):
        st.dataframe(logs.sort_values("timestamp", ascending=False), hide_index=True)
