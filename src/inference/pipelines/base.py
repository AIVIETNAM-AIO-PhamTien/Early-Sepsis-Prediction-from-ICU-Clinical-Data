"""Abstract prediction pipeline interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.inference.loader import ModelArtifact


class PredictionPipeline(ABC):
    @abstractmethod
    def predict_patient(
        self,
        df: pd.DataFrame,
        artifact: ModelArtifact,
    ) -> pd.DataFrame:
        """Predict sepsis risk for each ICU hour.

        Returns DataFrame with columns: ICULOS, risk_score, prediction.
        """
