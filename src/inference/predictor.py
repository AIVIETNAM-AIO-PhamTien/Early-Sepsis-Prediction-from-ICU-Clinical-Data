"""Prediction orchestration."""

from __future__ import annotations

import pandas as pd

from src.inference.loader import ModelArtifact, ModelLoader, ModelLoadError
from src.inference.pipelines.sepsyd import SepsydPipeline
from src.inference.pipelines.team_v1 import TeamV1Pipeline
from src.inference.validator import ValidationResult, validate_patient_df

PIPELINE_REGISTRY = {
    "sepsyd": SepsydPipeline(),
    "team_v1": TeamV1Pipeline(),
}


class PredictionError(Exception):
    """Raised when prediction fails after validation."""


def run_prediction(
    df: pd.DataFrame,
    patient_id: str,
    loader: ModelLoader,
) -> tuple[pd.DataFrame, ValidationResult, ModelArtifact | None]:
    """Validate input, load model, and run the appropriate pipeline."""
    validation = validate_patient_df(df)
    if not validation.is_valid:
        return pd.DataFrame(), validation, None

    try:
        artifact = loader.get_artifact()
    except ModelLoadError as exc:
        validation.errors.append(str(exc))
        validation.is_valid = False
        return pd.DataFrame(), validation, None

    pipeline_name = artifact.pipeline
    if pipeline_name not in PIPELINE_REGISTRY:
        validation.errors.append(f"Unknown pipeline: {pipeline_name}")
        validation.is_valid = False
        return pd.DataFrame(), validation, artifact

    try:
        pipeline = PIPELINE_REGISTRY[pipeline_name]
        results = pipeline.predict_patient(df, artifact)
        results.insert(0, "patient_id", patient_id)
        results["model_version"] = artifact.version
        return results, validation, artifact
    except Exception as exc:
        raise PredictionError(str(exc)) from exc
