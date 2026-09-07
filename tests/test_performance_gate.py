from __future__ import annotations

from src.pipelines.validation.performance_gate import evaluate_performance_gate


CONFIG = {"performance_gate": {"max_auprc_drop": 0.005, "max_utility_drop": 0.01}}


def test_performance_gate_accepts_candidate_within_tolerance() -> None:
    current = {"Full": {"AUPRC": 0.40, "Utility": 0.50}}
    candidate = {"Full": {"AUPRC": 0.396, "Utility": 0.491}}
    assert evaluate_performance_gate(candidate, current, CONFIG) == {"passed": True, "reasons": []}


def test_performance_gate_explains_regression() -> None:
    current = {"Full": {"AUPRC": 0.40, "Utility": 0.50}}
    candidate = {"Full": {"AUPRC": 0.39, "Utility": 0.48}}
    result = evaluate_performance_gate(candidate, current, CONFIG)
    assert result["passed"] is False
    assert len(result["reasons"]) == 2

