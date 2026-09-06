"""Narrative generation service."""

from __future__ import annotations

import pandas as pd

from src.narrative.context import PredictionSummary, build_prediction_summary
from src.narrative.providers import NarrativeProvider, TemplateProvider, resolve_provider


class NarrativeError(Exception):
    """Raised when narrative generation fails."""


def generate_narrative(
    results: pd.DataFrame,
    *,
    patient_id: str | None,
    threshold: float,
    provider_name: str = "auto",
    language: str = "vi",
    gemini_api_key: str | None = None,
    groq_api_key: str | None = None,
) -> tuple[str, str]:
    """Generate a plain-language narrative.

    Returns:
        Tuple of (narrative_text, provider_used).
    """
    try:
        summary = build_prediction_summary(
            results, patient_id=patient_id, threshold=threshold
        )
    except ValueError as exc:
        raise NarrativeError(str(exc)) from exc

    try:
        provider = resolve_provider(
            provider_name,
            gemini_api_key=gemini_api_key,
            groq_api_key=groq_api_key,
        )
    except ValueError as exc:
        raise NarrativeError(str(exc)) from exc

    try:
        text = provider.generate(summary, language=language)
        return text, provider.name
    except Exception as exc:
        if provider_name == "auto" and not isinstance(provider, TemplateProvider):
            fallback = TemplateProvider()
            return fallback.generate(summary, language=language), fallback.name
        raise NarrativeError(str(exc)) from exc
