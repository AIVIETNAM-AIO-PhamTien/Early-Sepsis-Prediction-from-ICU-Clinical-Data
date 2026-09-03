from __future__ import annotations

from typing import Any


def evaluate_performance_gate(
    candidate: dict[str, Any],
    current: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    max_auprc_drop = float(config["performance_gate"]["max_auprc_drop"])
    max_utility_drop = float(config["performance_gate"]["max_utility_drop"])
    for scope in ("Full",):
        candidate_scope = candidate.get(scope, {})
        current_scope = current.get(scope, {})
        if not candidate_scope or not current_scope:
            reasons.append(f"Missing metrics for scope {scope}")
            continue
        if candidate_scope.get("AUPRC", float("-inf")) < current_scope.get("AUPRC", 0.0) - max_auprc_drop:
            reasons.append(f"{scope}: AUPRC drop exceeds {max_auprc_drop}")
        if candidate_scope.get("Utility", float("-inf")) < current_scope.get("Utility", 0.0) - max_utility_drop:
            reasons.append(f"{scope}: Utility drop exceeds {max_utility_drop}")
    return {"passed": not reasons, "reasons": reasons}
