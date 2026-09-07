"""Prompt templates for narrative generation."""

from __future__ import annotations

from src.narrative.context import PredictionSummary


def _format_elevated_hours(summary: PredictionSummary) -> str:
    if not summary.elevated_iculos:
        return "none"
    hours = summary.elevated_iculos
    if len(hours) <= 8:
        return ", ".join(str(h) for h in hours)
    head = ", ".join(str(h) for h in hours[:4])
    tail = ", ".join(str(h) for h in hours[-2:])
    return f"{head}, ... , {tail} (total {len(hours)} hours)"


def build_user_prompt(summary: PredictionSummary, *, language: str) -> str:
    elevated_text = _format_elevated_hours(summary)
    lang_instruction = (
        "Write in clear, plain Vietnamese for clinicians."
        if language == "vi"
        else "Write in clear, plain English for clinicians."
    )

    return f"""Summarize this ICU sepsis risk prediction for a decision-support dashboard.

Patient ID: {summary.patient_id}
Model version: {summary.model_version}
Decision threshold: {summary.threshold:.2f}

ICU hours scored: {summary.n_hours}
Latest ICU hour: {summary.latest_iculos}
Latest risk score: {summary.latest_score:.1%} (status: {summary.current_status})
Peak risk score: {summary.peak_score:.1%} at ICU hour {summary.peak_iculos}
Elevated-risk hours (score >= threshold): {summary.elevated_hours} ({summary.elevated_pct:.0%} of stay)
Elevated ICU hours: {elevated_text}
Recent trend (last ~3 hours): {summary.trend}

Instructions:
- {lang_instruction}
- Use 3 to 5 short sentences in a warm but professional tone.
- Explain what the numbers mean in plain language (like the example explanations for Latest, Peak, Elevated hours).
- Mention whether the patient is currently below, near, or above the alert threshold.
- If there were elevated hours, note when they occurred and that most of the stay may still be below threshold.
- Do NOT diagnose sepsis. Say this is a model estimate for early warning only.
- Do NOT invent vitals, labs, treatments, or outcomes not listed above.
"""


SYSTEM_PROMPT = (
    "You are a clinical documentation assistant for an ICU sepsis early-warning "
    "dashboard. You translate model risk scores into concise, plain-language "
    "summaries. You never diagnose, prescribe, or claim certainty. You only "
    "describe the provided prediction metrics."
)
