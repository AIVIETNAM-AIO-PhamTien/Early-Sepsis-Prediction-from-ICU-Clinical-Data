from __future__ import annotations

from collections.abc import Iterable

import numpy as np


UTILITY_PARAMS = {
    "dt_early": -12,
    "dt_optimal": -6,
    "dt_late": 3,
    "max_u_tp": 1.0,
    "min_u_fn": -2.0,
    "u_fp": -0.05,
    "u_tn": 0.0,
}


def prediction_utility(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    params = UTILITY_PARAMS
    is_septic = bool(np.any(labels))
    t_sepsis = int(np.argmax(labels) - params["dt_optimal"]) if is_septic else float("inf")
    m1 = params["max_u_tp"] / (params["dt_optimal"] - params["dt_early"])
    b1 = -m1 * params["dt_early"]
    m2 = -params["max_u_tp"] / (params["dt_late"] - params["dt_optimal"])
    b2 = -m2 * params["dt_late"]
    m3 = params["min_u_fn"] / (params["dt_late"] - params["dt_optimal"])
    b3 = -m3 * params["dt_optimal"]
    utility = np.zeros(len(labels), dtype=float)
    for time in range(len(labels)):
        if time > t_sepsis + params["dt_late"]:
            continue
        if is_septic and predictions[time]:
            if time <= t_sepsis + params["dt_optimal"]:
                utility[time] = max(m1 * (time - t_sepsis) + b1, params["u_fp"])
            else:
                utility[time] = m2 * (time - t_sepsis) + b2
        elif not is_septic and predictions[time]:
            utility[time] = params["u_fp"]
        elif is_septic and not predictions[time] and time > t_sepsis + params["dt_optimal"]:
            utility[time] = m3 * (time - t_sepsis) + b3
        else:
            utility[time] = params["u_tn"]
    return float(utility.sum())


def normalized_utility(
    labels_by_patient: Iterable[np.ndarray],
    probabilities_by_patient: Iterable[np.ndarray],
    threshold: float,
) -> float:
    observed = best = inaction = 0.0
    for labels_value, probabilities_value in zip(labels_by_patient, probabilities_by_patient):
        labels = np.asarray(labels_value, dtype=int)
        predictions = (np.asarray(probabilities_value) >= threshold).astype(int)
        observed += prediction_utility(labels, predictions)
        best_predictions = np.zeros(len(labels), dtype=int)
        if np.any(labels):
            t_sepsis = int(np.argmax(labels) - UTILITY_PARAMS["dt_optimal"])
            start = max(0, t_sepsis + UTILITY_PARAMS["dt_early"])
            stop = min(t_sepsis + UTILITY_PARAMS["dt_late"] + 1, len(labels))
            best_predictions[start:stop] = 1
        best += prediction_utility(labels, best_predictions)
        inaction += prediction_utility(labels, np.zeros(len(labels), dtype=int))
    denominator = best - inaction
    return float((observed - inaction) / denominator) if denominator else 0.0


def select_threshold(
    labels_by_patient: list[np.ndarray],
    probabilities_by_patient: list[np.ndarray],
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.05,
) -> tuple[float, float]:
    thresholds = np.arange(start, stop + step / 2.0, step)
    scored = [
        (float(threshold), normalized_utility(labels_by_patient, probabilities_by_patient, float(threshold)))
        for threshold in thresholds
    ]
    return max(scored, key=lambda item: item[1])

