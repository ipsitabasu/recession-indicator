"""Indicator definition + scoring helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd

HERE = Path(__file__).parent

Tier = Literal["high", "medium", "low"]


def _load_json(name: str) -> dict[str, Any]:
    path = HERE / name
    if not path.exists():
        return {}
    return json.loads(path.read_text())


WEIGHTS = _load_json("weights.json")
THRESHOLDS = _load_json("thresholds.json")


@dataclass
class Indicator:
    key: str
    name: str
    tier: Tier
    cluster: str
    description: str
    rationale: str
    fetch: Callable[[], pd.Series]
    score_fn: Callable[[pd.Series, dict[str, float]], float]
    transform: Callable[[pd.Series], pd.Series] | None = None
    auc: float = field(default=0.0)
    weight: float = field(default=0.0)

    def __post_init__(self) -> None:
        meta = WEIGHTS.get(self.key, {})
        self.auc = float(meta.get("auc", self.auc))
        self.weight = float(meta.get("weight", self.weight))

    def thresholds(self) -> dict[str, float]:
        return THRESHOLDS.get(self.key, {})

    def score(self, series: pd.Series) -> float:
        return self.score_fn(series, self.thresholds())


def linear_score(value: float, low_anchor: float, high_anchor: float, *, inverted: bool = False) -> float:
    """Map `value` to a 0–100 risk score by linear interpolation.

    `low_anchor` = value at min risk (score 0), `high_anchor` = value at max risk (score 100).
    If `inverted=True`, low_anchor > high_anchor (e.g. yield curve: +150bps = safe, -50bps = risky).
    """
    if low_anchor == high_anchor:
        return 50.0
    if inverted:
        if value >= low_anchor:
            return 0.0
        if value <= high_anchor:
            return 100.0
        return 100.0 * (low_anchor - value) / (low_anchor - high_anchor)
    if value <= low_anchor:
        return 0.0
    if value >= high_anchor:
        return 100.0
    return 100.0 * (value - low_anchor) / (high_anchor - low_anchor)


def yoy_change(s: pd.Series, periods: int = 12) -> pd.Series:
    """Year-over-year percent change. Assumes monthly index; daily series should be resampled first."""
    return s.pct_change(periods=periods) * 100.0


def latest(s: pd.Series) -> float:
    if s is None or s.empty:
        raise ValueError("Cannot get latest of empty series")
    return float(s.iloc[-1])
