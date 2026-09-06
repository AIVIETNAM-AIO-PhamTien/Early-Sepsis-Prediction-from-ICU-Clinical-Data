"""LLM-assisted clinical narrative from prediction summaries."""

from src.narrative.context import PredictionSummary, build_prediction_summary
from src.narrative.service import NarrativeError, generate_narrative

__all__ = [
    "PredictionSummary",
    "build_prediction_summary",
    "generate_narrative",
    "NarrativeError",
]
