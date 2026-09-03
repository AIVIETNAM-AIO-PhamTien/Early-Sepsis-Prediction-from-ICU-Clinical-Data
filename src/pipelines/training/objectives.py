from __future__ import annotations

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def make_weighted_logloss_objective(positive_weight: float = 40.0):
    def objective(predictions, dtrain):
        labels = dtrain.get_label()
        probabilities = sigmoid(predictions)
        weights = np.where(labels == 1, positive_weight, 1.0)
        gradient = weights * (probabilities - labels)
        hessian = weights * probabilities * (1.0 - probabilities)
        return gradient, hessian

    return objective


def auprc_metric(predictions, dtrain):
    from sklearn.metrics import average_precision_score

    return "auprc", float(average_precision_score(dtrain.get_label(), sigmoid(predictions)))

