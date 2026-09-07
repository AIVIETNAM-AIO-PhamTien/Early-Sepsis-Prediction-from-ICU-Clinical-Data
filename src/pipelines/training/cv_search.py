from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.pipelines.training.ml_adapter import group_patient_outputs, prepare_fold
from src.pipelines.training.objectives import auprc_metric, make_weighted_logloss_objective, sigmoid
from src.pipelines.training.utility import select_threshold


def run_group_cv(
    frame: pd.DataFrame,
    config: dict[str, Any],
    schema: dict[str, Any],
    params: dict[str, Any] | None = None,
    n_splits: int | None = None,
) -> dict[str, Any]:
    cv_config = config["cross_validation"]
    model_config = config["model"]
    folds = int(n_splits or cv_config["search_n_splits"])
    patient_ids = frame["patient_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=folds)
    model_params = {
        "max_depth": int(model_config["baseline_max_depth"]),
        "eta": float(model_config["baseline_learning_rate"]),
        "tree_method": model_config["tree_method"],
        "seed": int(cv_config["random_state"]),
        **(params or {}),
    }
    oof = np.full(len(frame), np.nan, dtype=np.float32)
    fold_results: list[dict[str, Any]] = []
    best_iterations: list[int] = []
    feature_columns: list[str] | None = None
    frozen_log_columns: list[str] | None = None
    for fold, (train_index, valid_index) in enumerate(splitter.split(frame, groups=patient_ids)):
        prepared = prepare_fold(
            frame.iloc[train_index].copy(),
            frame.iloc[valid_index].copy(),
            config,
            schema,
            frozen_log_columns=frozen_log_columns,
        )
        frozen_log_columns = prepared["preprocessor"].log_columns
        feature_columns = prepared["feature_columns"]
        dtrain = xgb.DMatrix(prepared["X_train"], label=prepared["y_train"], feature_names=feature_columns)
        dvalid = xgb.DMatrix(prepared["X_valid"], label=prepared["y_valid"], feature_names=feature_columns)
        booster = xgb.train(
            model_params,
            dtrain,
            num_boost_round=int(cv_config["num_boost_round"]),
            obj=make_weighted_logloss_objective(float(model_config["positive_weight"])),
            custom_metric=auprc_metric,
            maximize=True,
            evals=[(dvalid, "validation")],
            early_stopping_rounds=int(cv_config["early_stopping_rounds"]),
            verbose_eval=False,
        )
        best_iteration = int(booster.best_iteration)
        margins = booster.predict(dvalid, iteration_range=(0, best_iteration + 1), output_margin=True)
        probabilities = sigmoid(margins).astype(np.float32)
        oof[valid_index] = probabilities
        best_iterations.append(best_iteration)
        fold_results.append(
            {
                "fold": fold,
                "best_iteration": best_iteration,
                "auroc": float(roc_auc_score(prepared["y_valid"], probabilities)),
                "auprc": float(average_precision_score(prepared["y_valid"], probabilities)),
            }
        )
    if np.isnan(oof).any():
        raise RuntimeError("OOF predictions are incomplete")
    labels_by_patient, probabilities_by_patient = group_patient_outputs(
        patient_ids,
        frame[schema["label_column"]].to_numpy(dtype=np.float32),
        oof,
    )
    threshold_config = config["threshold"]
    threshold, utility = select_threshold(
        labels_by_patient,
        probabilities_by_patient,
        float(threshold_config["start"]),
        float(threshold_config["stop"]),
        float(threshold_config["step"]),
    )
    return {
        "params": model_params,
        "folds": fold_results,
        "mean_auprc": float(np.mean([fold["auprc"] for fold in fold_results])),
        "std_auprc": float(np.std([fold["auprc"] for fold in fold_results])),
        "best_threshold": threshold,
        "mean_utility": utility,
        "mean_best_iteration": int(round(np.mean(best_iterations))) + 1,
        "feature_columns": feature_columns,
        "log_columns": frozen_log_columns,
        "oof_probabilities": oof,
    }


def run_training_workflow(
    frame: pd.DataFrame,
    config: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    cv_config = config["cross_validation"]
    candidates = list(cv_config.get("candidates") or [{}])
    search_results = [
        run_group_cv(
            frame,
            config,
            schema,
            params=candidate,
            n_splits=int(cv_config["search_n_splits"]),
        )
        for candidate in candidates
    ]
    best_search = max(
        search_results,
        key=lambda result: (result["mean_auprc"], result["mean_utility"]),
    )
    final_result = run_group_cv(
        frame,
        config,
        schema,
        params=best_search["params"],
        n_splits=int(cv_config["finalization_n_splits"]),
    )
    final_result["best_params"] = best_search["params"]
    final_result["search_results"] = [
        {key: value for key, value in result.items() if key != "oof_probabilities"}
        for result in search_results
    ]
    final_result["search_n_splits"] = int(cv_config["search_n_splits"])
    final_result["finalization_n_splits"] = int(cv_config["finalization_n_splits"])
    return final_result
