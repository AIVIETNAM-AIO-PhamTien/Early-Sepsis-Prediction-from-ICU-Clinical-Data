from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import xgboost as xgb

from src.pipelines.processing.feature_builder import add_lookback
from src.pipelines.processing.preprocessor import fit_preprocessor, transform_patients
from src.pipelines.training.objectives import make_weighted_logloss_objective
from src.pipelines.training.utility import UTILITY_PARAMS


def train_final_model(
    frame: pd.DataFrame,
    config: dict[str, Any],
    schema: dict[str, Any],
    cv_result: dict[str, Any],
    output_dir: str | Path,
    dataset_version: str,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    fitted = fit_preprocessor(
        frame,
        config,
        schema,
        frozen_log_columns=list(cv_result.get("log_columns") or []),
    )
    transformed = transform_patients(frame, fitted, config)
    feature_config = config["features"]
    matrix, feature_names = add_lookback(
        transformed,
        fitted.feature_columns,
        hours=int(feature_config["lookback_hours"]),
        padding=float(feature_config["lookback_padding"]),
    )
    labels = transformed[schema["label_column"]].to_numpy(dtype="float32")
    dtrain = xgb.DMatrix(matrix, label=labels, feature_names=feature_names)
    booster = xgb.train(
        cv_result["params"],
        dtrain,
        num_boost_round=int(cv_result["mean_best_iteration"]),
        obj=make_weighted_logloss_objective(float(config["model"]["positive_weight"])),
    )
    booster.save_model(destination / "model.json")
    joblib.dump(booster, destination / "model.pkl")
    joblib.dump(fitted, destination / "preprocessor.pkl")
    (destination / "preprocessor.json").write_text(
        json.dumps(fitted.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (destination / "feature_config.json").write_text(
        json.dumps(feature_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    (destination / "feature_schema.json").write_text(
        json.dumps({"columns": feature_names, "n_features": len(feature_names)}, indent=2), encoding="utf-8"
    )
    (destination / "threshold.json").write_text(
        json.dumps({"threshold": float(cv_result["best_threshold"])}, indent=2), encoding="utf-8"
    )
    serializable_cv = {key: value for key, value in cv_result.items() if key != "oof_probabilities"}
    (destination / "cv_results.json").write_text(
        json.dumps(serializable_cv, indent=2, sort_keys=True), encoding="utf-8"
    )
    (destination / "utility_config.json").write_text(
        json.dumps(UTILITY_PARAMS, indent=2, sort_keys=True), encoding="utf-8"
    )
    metadata = {
        "dataset_version": dataset_version,
        "feature_schema_version": feature_config["feature_set"],
        "n_features": len(feature_names),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate",
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {**metadata, "candidate_dir": str(destination), "threshold": float(cv_result["best_threshold"])}
