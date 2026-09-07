"""Inference pipeline for models produced by the Airflow retraining DAG."""

from __future__ import annotations

import pandas as pd
import xgboost as xgb

from src.inference.loader import ModelArtifact
from src.inference.pipelines.base import PredictionPipeline
from src.pipelines.processing.feature_builder import add_lookback
from src.pipelines.processing.preprocessor import FittedPreprocessor, transform_patients
from src.pipelines.training.objectives import sigmoid


class TeamV1Pipeline(PredictionPipeline):
    def predict_patient(
        self,
        df: pd.DataFrame,
        artifact: ModelArtifact,
    ) -> pd.DataFrame:
        frame = df.copy()
        frame["patient_id"] = "inference_patient"
        frame["source"] = "all"

        fitted = FittedPreprocessor.from_dict(artifact.preprocessor)
        stored_config = artifact.feature_config
        if "features" in stored_config:
            inference_config = stored_config
        else:
            inference_config = {
                "features": stored_config,
                "preprocessing": {"impute_columns": ["HR", "O2Sat", "Temp"]},
            }
        transformed = transform_patients(frame, fitted, inference_config)
        feature_config = inference_config["features"]
        matrix, feature_names = add_lookback(
            transformed,
            fitted.feature_columns,
            hours=int(feature_config["lookback_hours"]),
            padding=float(feature_config["lookback_padding"]),
        )
        if artifact.model.num_features() != len(feature_names):
            raise ValueError(
                "Inference feature count does not match model: "
                f"{len(feature_names)} != {artifact.model.num_features()}"
            )
        margins = artifact.model.predict(
            xgb.DMatrix(matrix, feature_names=feature_names),
            output_margin=True,
        )
        probabilities = sigmoid(margins)
        return pd.DataFrame(
            {
                "ICULOS": frame["ICULOS"].astype(int).to_numpy(),
                "risk_score": probabilities,
                "prediction": (probabilities >= artifact.threshold).astype(int),
            }
        )
