"""Escalation policy mapping risk-score tiers to recommended clinical actions.

This is an illustrative, non-normative policy inspired by common ICU early-warning
workflows (e.g. Rapid Response Team triage, Surviving Sepsis Campaign Hour-1
Bundle). It is decision support only, not a validated hospital protocol.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EscalationTier:
    name: str
    color: str
    icon: str
    min_ratio: float
    action: str
    responsible_role: str
    target_minutes: int | None


ESCALATION_TIERS: list[EscalationTier] = [
    EscalationTier(
        name="Low",
        color="green",
        icon=":material/check_circle:",
        min_ratio=0.0,
        action="Routine monitoring per unit schedule.",
        responsible_role="Bedside nurse",
        target_minutes=None,
    ),
    EscalationTier(
        name="Watch",
        color="orange",
        icon=":material/visibility:",
        min_ratio=0.6,
        action="Increase vitals monitoring frequency; review the hourly risk score trend.",
        responsible_role="Bedside nurse",
        target_minutes=60,
    ),
    EscalationTier(
        name="Elevated",
        color="red",
        icon=":material/emergency:",
        min_ratio=1.0,
        action="Escalate immediately to the on-call physician / Rapid Response Team; consider activating the Hour-1 Sepsis Bundle.",
        responsible_role="On-call physician / RRT",
        target_minutes=30,
    ),
]


def get_escalation_tier(score: float, threshold: float) -> EscalationTier:
    """Return the escalation tier for a risk score given the model threshold."""
    ratio = score / threshold if threshold else float("inf")
    tier = ESCALATION_TIERS[0]
    for candidate in ESCALATION_TIERS:
        if ratio >= candidate.min_ratio:
            tier = candidate
    return tier
