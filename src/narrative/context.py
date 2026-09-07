"""Structured prediction summary for narrative generation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def _risk_status(score: float, threshold: float) -> str:
    if score >= threshold:
        return "Elevated"
    if score >= threshold * 0.6:
        return "Watch"
    return "Low"


@dataclass(frozen=True)
class PredictionSummary:
    patient_id: str
    model_version: str
    threshold: float
    n_hours: int
    latest_score: float
    peak_score: float
    elevated_hours: int
    elevated_pct: float
    current_status: str
    elevated_iculos: tuple[int, ...]
    latest_iculos: int
    peak_iculos: int
    trend: str


def build_prediction_summary(
    results: pd.DataFrame,
    *,
    patient_id: str | None,
    threshold: float,
) -> PredictionSummary:
    """Aggregate hourly prediction output into a narrative-friendly summary."""
    if results.empty:
        raise ValueError("Cannot build summary from empty results.")

    latest_score = float(results["risk_score"].iloc[-1])
    peak_idx = int(results["risk_score"].idxmax())
    peak_score = float(results["risk_score"].iloc[peak_idx])
    elevated_mask = results["prediction"] == 1
    elevated_iculos = tuple(int(x) for x in results.loc[elevated_mask, "ICULOS"].tolist())
    n_hours = len(results)
    elevated_hours = len(elevated_iculos)

    last_three = results["risk_score"].tail(3)
    if len(last_three) >= 2:
        delta = float(last_three.iloc[-1] - last_three.iloc[0])
        if delta > 0.05:
            trend = "rising"
        elif delta < -0.05:
            trend = "falling"
        else:
            trend = "stable"
    else:
        trend = "stable"

    pid = patient_id or str(results["patient_id"].iloc[0])
    model_version = str(results["model_version"].iloc[0])

    return PredictionSummary(
        patient_id=pid,
        model_version=model_version,
        threshold=threshold,
        n_hours=n_hours,
        latest_score=latest_score,
        peak_score=peak_score,
        elevated_hours=elevated_hours,
        elevated_pct=elevated_hours / n_hours if n_hours else 0.0,
        current_status=_risk_status(latest_score, threshold),
        elevated_iculos=elevated_iculos,
        latest_iculos=int(results["ICULOS"].iloc[-1]),
        peak_iculos=int(results.loc[peak_idx, "ICULOS"]),
        trend=trend,
    )
