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
            "Pipeline team_v1 chưa sẵn sàng. Cần artifact v2 từ nhóm ML."
        )
