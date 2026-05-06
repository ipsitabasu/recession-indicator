"""Composite scoring: aggregate per-indicator scores into one risk number."""
from __future__ import annotations

from dataclasses import dataclass


TIER_BANDS = [
    (25.0, "Low"),
    (50.0, "Moderate"),
    (70.0, "Elevated"),
    (101.0, "High"),
]


@dataclass
class IndicatorResult:
    key: str
    name: str
    tier: str
    cluster: str
    weight: float
    auc: float
    value: float | None
    score: float | None
    error: str | None = None
    as_of: str | None = None


def composite(results: list[IndicatorResult]) -> tuple[float, dict[str, float]]:
    """Returns (composite 0-100, contributions dict)."""
    used = [r for r in results if r.score is not None and r.weight > 0]
    if not used:
        return 0.0, {}
    total_weight = sum(r.weight for r in used)
    if total_weight == 0:
        return 0.0, {}
    contribs = {r.key: r.weight * r.score / total_weight for r in used}
    return sum(contribs.values()), contribs


def tier_for(score: float) -> str:
    for upper, label in TIER_BANDS:
        if score < upper:
            return label
    return "High"
