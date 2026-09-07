from __future__ import annotations

import json

import numpy as np
import xgboost as xgb

from src.inference.loader import ModelLoader
from src.inference.predictor import run_prediction


def test_promoted_team_model_can_run_inference(tmp_path, sample_patient_df) -> None:
    model_dir = tmp_path / "model_v2"
    model_dir.mkdir()
    matrix = np.array([[60.0], [80.0], [100.0], [120.0]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 1, "eta": 0.5},
        xgb.DMatrix(matrix, label=labels, feature_names=["HR__t-0"]),
        num_boost_round=2,
    )
    booster.save_model(model_dir / "model.json")

    preprocessor = {
        "age_gender_values": {},
        "log_columns": [],
        "means": {"HR": 0.0},
        "stds": {"HR": 1.0},
        "fill_values": {"HR": 80.0},
        "feature_columns": ["HR"],
    }
    feature_config = {
        "pipeline": "team_v1",
        "preprocessing": {"impute_columns": ["HR"]},
        "features": {
            "columns": ["HR"],
            "moving_window_hours": 2,
            "delta_lag_hours": 1,
            "lookback_hours": 1,
            "lookback_padding": 0.0,
        },
    }
    (model_dir / "preprocessor.json").write_text(json.dumps(preprocessor), encoding="utf-8")
    (model_dir / "feature_config.json").write_text(json.dumps(feature_config), encoding="utf-8")
    manifest = {
        "model_version": "model_v2",
        "dataset_version": "dataset_v2",
        "model_path": str(model_dir / "model.json"),
        "preprocessor_path": str(model_dir / "preprocessor.json"),
        "feature_config_path": str(model_dir / "feature_config.json"),
        "pipeline": "team_v1",
        "model_format": "xgboost_json",
        "threshold": 0.5,
        "status": "current",
        "updated_at": "2026-09-07T00:00:00Z",
    }
    manifest_path = tmp_path / "current_model.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    results, validation, artifact = run_prediction(
        sample_patient_df,
        "sample_patient",
        ModelLoader(manifest_path),
    )

    assert validation.is_valid
    assert artifact is not None
    assert artifact.pipeline == "team_v1"
    assert len(results) == len(sample_patient_df)
    assert results["risk_score"].between(0.0, 1.0).all()
    assert set(results["prediction"].unique()).issubset({0, 1})
