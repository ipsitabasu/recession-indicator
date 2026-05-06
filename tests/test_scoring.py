import math

import pytest

from indicators.base import linear_score, yoy_change
from scoring import IndicatorResult, composite, tier_for


class TestLinearScore:
    def test_direct_min_max(self):
        assert linear_score(0.0, 0.0, 0.5) == 0.0
        assert linear_score(0.5, 0.0, 0.5) == 100.0

    def test_direct_midpoint(self):
        assert linear_score(0.25, 0.0, 0.5) == pytest.approx(50.0)

    def test_direct_clamps(self):
        assert linear_score(-1.0, 0.0, 0.5) == 0.0
        assert linear_score(2.0, 0.0, 0.5) == 100.0

    def test_inverted_min_max(self):
        # yield curve: low (safe) = 1.5, high (risky) = -0.5
        assert linear_score(1.5, 1.5, -0.5, inverted=True) == 0.0
        assert linear_score(-0.5, 1.5, -0.5, inverted=True) == 100.0

    def test_inverted_midpoint(self):
        assert linear_score(0.5, 1.5, -0.5, inverted=True) == pytest.approx(50.0)

    def test_inverted_clamps(self):
        assert linear_score(2.0, 1.5, -0.5, inverted=True) == 0.0
        assert linear_score(-1.0, 1.5, -0.5, inverted=True) == 100.0

    def test_zero_range_returns_neutral(self):
        assert linear_score(0.0, 1.0, 1.0) == 50.0


class TestYoyChange:
    def test_basic(self):
        import pandas as pd
        idx = pd.date_range("2020-01-01", periods=24, freq="MS")
        s = pd.Series(range(100, 124), index=idx)
        out = yoy_change(s, periods=12).dropna()
        assert len(out) == 12
        # First valid value: (112/100 - 1)*100 = 12
        assert out.iloc[0] == pytest.approx(12.0)


class TestComposite:
    def _r(self, key, weight, score, error=None):
        return IndicatorResult(
            key=key, name=key, tier="high", cluster="x",
            weight=weight, auc=0.8, value=1.0, score=score, error=error,
        )

    def test_weighted_average(self):
        rs = [self._r("a", 0.5, 80), self._r("b", 0.3, 40), self._r("c", 0.2, 10)]
        s, contribs = composite(rs)
        assert s == pytest.approx(0.5*80 + 0.3*40 + 0.2*10)
        assert sum(contribs.values()) == pytest.approx(s)

    def test_drops_failed_and_renormalizes(self):
        rs = [
            self._r("a", 0.5, 80),
            IndicatorResult(key="b", name="b", tier="high", cluster="x",
                            weight=0.3, auc=0.7, value=None, score=None, error="x"),
            self._r("c", 0.2, 10),
        ]
        s, _ = composite(rs)
        expected = (0.5*80 + 0.2*10) / (0.5 + 0.2)
        assert s == pytest.approx(expected)

    def test_zero_weight_indicators_excluded(self):
        rs = [self._r("a", 0.5, 80), self._r("b", 0.0, 100)]
        s, contribs = composite(rs)
        assert s == pytest.approx(80.0)
        assert "b" not in contribs

    def test_all_failed_returns_zero(self):
        rs = [IndicatorResult(key="a", name="a", tier="high", cluster="x",
                              weight=0.5, auc=0.8, value=None, score=None, error="x")]
        s, contribs = composite(rs)
        assert s == 0.0
        assert contribs == {}


class TestTier:
    @pytest.mark.parametrize("score,tier", [
        (0, "Low"), (24.99, "Low"), (25, "Moderate"), (49.99, "Moderate"),
        (50, "Elevated"), (69.99, "Elevated"), (70, "High"), (100, "High"),
    ])
    def test_bands(self, score, tier):
        assert tier_for(score) == tier
