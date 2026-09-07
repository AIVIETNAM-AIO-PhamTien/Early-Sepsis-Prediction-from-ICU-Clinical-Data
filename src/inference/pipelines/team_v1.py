"""Team v1 pipeline stub - awaiting ML artifact v2."""

from __future__ import annotations

import pandas as pd

from src.inference.loader import ModelArtifact
from src.inference.pipelines.base import PredictionPipeline


class TeamV1Pipeline(PredictionPipeline):
    def predict_patient(
        self,
        df: pd.DataFrame,
        artifact: ModelArtifact,
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "The team_v1 pipeline is not ready yet. Requires the v2 artifact from the ML team."
        )
