from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)

from src.pipelines.processing.feature_builder import add_lookback
from src.pipelines.processing.preprocessor import transform_patients
from src.pipelines.training.ml_adapter import group_patient_outputs
from src.pipelines.training.objectives import sigmoid
from src.pipelines.training.utility import normalized_utility


def _metric_set(labels: np.ndarray, probabilities: np.ndarray, groups: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    labels_by_patient, probabilities_by_patient = group_patient_outputs(groups, labels, probabilities)
    return {
        "AUROC": float(roc_auc_score(labels, probabilities)),
        "AUPRC": float(average_precision_score(labels, probabilities)),
        "Accuracy": float(accuracy_score(labels, predictions)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "Precision": float(precision_score(labels, predictions, zero_division=0)),
        "Sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "Specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "FalsePositiveRate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "Utility": normalized_utility(labels_by_patient, probabilities_by_patient, threshold),
        "ConfusionMatrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate_model(
    model_dir: str | Path,
    frame: pd.DataFrame,
    config: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    root = Path(model_dir)
    fitted = joblib.load(root / "preprocessor.pkl")
    booster = joblib.load(root / "model.pkl")
    threshold = float(json.loads((root / "threshold.json").read_text(encoding="utf-8"))["threshold"])
    transformed = transform_patients(frame, fitted, config)
    feature_config = config["features"]
    matrix, feature_names = add_lookback(
        transformed,
        fitted.feature_columns,
        hours=int(feature_config["lookback_hours"]),
        padding=float(feature_config["lookback_padding"]),
    )
    expected = json.loads((root / "feature_schema.json").read_text(encoding="utf-8"))["columns"]
    if feature_names != expected:
        raise ValueError("Evaluation feature order does not match model artifact")
    margins = booster.predict(xgb.DMatrix(matrix, feature_names=feature_names), output_margin=True)
    probabilities = sigmoid(margins)
    labels = transformed[schema["label_column"]].to_numpy(dtype=int)
    groups = transformed["patient_id"].astype(str).to_numpy()
    results = {"Full": _metric_set(labels, probabilities, groups, threshold)}
    (root / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return results
