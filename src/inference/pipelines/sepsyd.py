"""Sepsyd pipeline - port of vendor/sepsyd_original/get_sepsis_score.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src.inference.loader import ModelArtifact
from src.inference.pipelines.base import PredictionPipeline
from src.shared.schema import REQUIRED_COLUMNS


def compute_sepsis_score(
    data: np.ndarray,
    model,
    preprocessor: dict,
    threshold: float,
) -> tuple[float, int]:
    """Compute sepsis risk score for cumulative data up to current hour."""
    varmeans = np.array(preprocessor["varmeans"], dtype=float)
    varstds = np.array(preprocessor["varstds"], dtype=float)
    varlogmeans = np.array(preprocessor["varlogmeans"], dtype=float)
    varlogstds = np.array(preprocessor["varlogstds"], dtype=float)
    log_indices = set(preprocessor.get("log_transform_indices", []))
    n_preprocess_cols = int(preprocessor.get("n_preprocess_cols", 39))
    lookback = int(preprocessor.get("lookback_hours", 5))

    data = np.copy(data)
    nan_idx = np.where(np.isnan(data))
    delta = np.zeros(data.shape)
    mask = np.ones(data.shape)
    data[nan_idx] = np.take(varmeans, nan_idx[1])
    mask[nan_idx] = 0

    forward = np.copy(data[0, :])
    for t in range(data.shape[0]):
        for i in range(n_preprocess_cols):
            if mask[t, i] == 1:
                forward[i] = data[t, i]
            else:
                data[t, i] = forward[i]
        if t > 0:
            delta[t, :] = data[t, :] - data[t - 1, :]

    for i in range(n_preprocess_cols):
        if i in log_indices:
            data[:, i] = 10 * (np.log(data[:, i]) - varlogmeans[i]) / varlogstds[i]
        else:
            data[:, i] = 10 * (data[:, i] - varmeans[i]) / varstds[i]

    data = np.concatenate((data, delta), axis=1)
    data = np.concatenate((data, mask), axis=1)

    t = len(data) - 1
    row: list[float] = list(data[t, :])
    for j in range(1, lookback + 1):
        if t - j < 0:
            row.extend([0] * (data.shape[1]))
        else:
            row.extend(data[t - j, :])

    row_arr = np.asarray([row], dtype=np.float64)
    dtest = xgb.DMatrix(row_arr)
    pred_prob = float(model.predict(dtest)[0])
    disc = int(pred_prob >= threshold)
    return pred_prob, disc


class SepsydPipeline(PredictionPipeline):
    def predict_patient(
        self,
        df: pd.DataFrame,
        artifact: ModelArtifact,
    ) -> pd.DataFrame:
        data = df[REQUIRED_COLUMNS].to_numpy(dtype=float)
        rows = []
        for t in range(len(data)):
            prob, pred = compute_sepsis_score(
                data[: t + 1],
                artifact.model,
                artifact.preprocessor,
                artifact.threshold,
            )
            rows.append(
                {
                    "ICULOS": int(df.iloc[t]["ICULOS"]),
                    "risk_score": prob,
                    "prediction": pred,
                }
            )
        return pd.DataFrame(rows)
