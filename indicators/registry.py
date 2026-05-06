"""Registry of all 17 recession indicators.

Each indicator declares: how to fetch its raw series, how to transform it
to the value we score on, and how to map that value to a 0-100 risk score.
Weights and scoring thresholds are loaded from weights.json / thresholds.json.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from .base import Indicator, latest, linear_score, yoy_change
from . import fred, stooq, pmi, trends, nyt


# ----- Score functions (value -> 0..100 risk) -----

def score_linear(s: pd.Series, t: dict[str, float], *, inverted: bool = False) -> float:
    return linear_score(latest(s), t["low"], t["high"], inverted=inverted)


def _score_inverted(s, t):  # pickled-friendly
    return score_linear(s, t, inverted=True)


# ----- Transforms (raw series -> series we score on) -----

def _identity(s: pd.Series) -> pd.Series:
    return s


def _yoy_monthly(s: pd.Series) -> pd.Series:
    monthly = s.resample("MS").last() if not s.index.is_monotonic_increasing or s.index.freqstr is None else s
    return yoy_change(monthly, periods=12).dropna()


def _claims_yoy(s: pd.Series) -> pd.Series:
    weekly = s.resample("W").last().dropna()
    return (weekly.pct_change(periods=52) * 100.0).dropna()


def _sp500_drawdown(s: pd.Series) -> pd.Series:
    rolling_max = s.rolling(window=252, min_periods=20).max()
    return ((s / rolling_max) - 1.0) * 100.0


def _copper_gold_yoy() -> pd.Series:
    copper = stooq.fetch_close("hg.f")
    gold = stooq.fetch_close("gc.f")
    df = pd.concat({"hg": copper, "gc": gold}, axis=1).dropna()
    ratio = df["hg"] / df["gc"]
    return (ratio.pct_change(periods=252) * 100.0).dropna()


def _real_m2_yoy() -> pd.Series:
    m2 = fred.fetch_series("M2SL")
    cpi = fred.fetch_series("CPIAUCSL")
    df = pd.concat({"m2": m2, "cpi": cpi}, axis=1).dropna()
    real = df["m2"] / df["cpi"]
    return yoy_change(real, periods=12).dropna()


def _lipstick_relative() -> pd.Series:
    el = stooq.fetch_close("el.us")
    spx = stooq.fetch_close("^spx")
    df = pd.concat({"el": el, "spx": spx}, axis=1).dropna()
    el_yoy = df["el"].pct_change(252)
    spx_yoy = df["spx"].pct_change(252)
    return ((el_yoy - spx_yoy) * 100.0).dropna()


# ----- Fetch wrappers (each returns a tuple (raw, scoring_series)) -----
# We wrap each so transforms pulling multiple series can do so internally.

def _fetch_and_transform(fetcher: Callable[[], pd.Series], transform: Callable[[pd.Series], pd.Series]):
    def f() -> pd.Series:
        return transform(fetcher())
    return f


# ===== Registry =====

INDICATORS: list[Indicator] = [
    # --- HIGH (yield curve cluster) ---
    Indicator(
        key="yield_curve_10y3m",
        name="10Y–3M Treasury Spread",
        tier="high",
        cluster="yield-curve",
        description="Difference between 10-year and 3-month Treasury yields (basis points implied; FRED reports in %).",
        rationale="Estrella & Mishkin (1998); NY Fed's preferred recession-probability input. Inversion has preceded every US recession since 1955 with ~12-month lead.",
        fetch=lambda: fred.fetch_series("T10Y3M"),
        score_fn=_score_inverted,
    ),
    Indicator(
        key="yield_curve_10y2y",
        name="10Y–2Y Treasury Spread",
        tier="high",
        cluster="yield-curve",
        description="Difference between 10-year and 2-year Treasury yields.",
        rationale="Bauer & Mertens (2018). Same yield-curve signal as 10Y-3M; popularized in financial press.",
        fetch=lambda: fred.fetch_series("T10Y2Y"),
        score_fn=_score_inverted,
    ),

    # --- HIGH (labor cluster) ---
    Indicator(
        key="sahm_rule",
        name="Sahm Rule",
        tier="high",
        cluster="labor",
        description="3-month avg unemployment rate minus its 12-month low. Triggers at ≥0.50pp.",
        rationale="Sahm (2019). Has signaled every US recession since 1970 in real time with zero false positives.",
        fetch=lambda: fred.fetch_series("SAHMREALTIME"),
        score_fn=score_linear,
    ),
    Indicator(
        key="initial_claims_yoy",
        name="Initial Jobless Claims (4-wk MA, YoY)",
        tier="high",
        cluster="labor",
        description="Year-over-year change in 4-week moving avg of initial unemployment claims.",
        rationale="Conference Board LEI component. Sharp YoY rises consistently precede labor-market downturns.",
        fetch=_fetch_and_transform(lambda: fred.fetch_series("IC4WSA"), _claims_yoy),
        score_fn=score_linear,
    ),

    # --- HIGH (credit) ---
    Indicator(
        key="hy_credit_spread",
        name="High-Yield Credit Spread",
        tier="high",
        cluster="credit",
        description="ICE BofA US High Yield option-adjusted spread (basis points).",
        rationale="Gilchrist & Zakrajšek (2012) Excess Bond Premium. Widening HY spreads precede recessions by 6–12 months.",
        fetch=lambda: fred.fetch_series("BAMLH0A0HYM2"),
        score_fn=score_linear,
    ),

    # --- HIGH (activity) ---
    Indicator(
        key="ism_pmi",
        name="ISM Manufacturing PMI",
        tier="high",
        cluster="activity",
        description="Diffusion index of US manufacturing activity. Below 50 = contraction.",
        rationale="Koenig (2002). Persistent readings below ~45 have a strong recession-coincident signal.",
        fetch=pmi.fetch_latest_pmi,
        score_fn=_score_inverted,
    ),

    # --- MEDIUM (housing) ---
    Indicator(
        key="building_permits_yoy",
        name="Building Permits (YoY)",
        tier="medium",
        cluster="housing",
        description="Year-over-year change in new privately-owned housing units authorized.",
        rationale="LEI component. Housing leads the cycle; sharp permit declines precede recessions historically.",
        fetch=_fetch_and_transform(lambda: fred.fetch_series("PERMIT"), _yoy_monthly),
        score_fn=_score_inverted,
    ),

    # --- MEDIUM (activity) ---
    Indicator(
        key="industrial_production_yoy",
        name="Industrial Production (YoY)",
        tier="medium",
        cluster="activity",
        description="Year-over-year change in Federal Reserve's Industrial Production index.",
        rationale="Coincident-to-leading with manufacturing-driven downturns; LEI-adjacent component.",
        fetch=_fetch_and_transform(lambda: fred.fetch_series("INDPRO"), _yoy_monthly),
        score_fn=_score_inverted,
    ),
    Indicator(
        key="weekly_hours_mfg",
        name="Avg Weekly Hours, Manufacturing",
        tier="medium",
        cluster="activity",
        description="Average weekly hours of production workers in manufacturing.",
        rationale="LEI component. Hours fall before layoffs; classic leading indicator since the 1950s.",
        fetch=lambda: fred.fetch_series("AWHMAN"),
        score_fn=_score_inverted,
    ),

    # --- MEDIUM (sentiment) ---
    Indicator(
        key="umich_sentiment",
        name="UMich Consumer Sentiment",
        tier="medium",
        cluster="sentiment",
        description="University of Michigan Index of Consumer Sentiment.",
        rationale="LEI component. Consumer pessimism precedes spending pullbacks but is noisy.",
        fetch=lambda: fred.fetch_series("UMCSENT"),
        score_fn=_score_inverted,
    ),
    Indicator(
        key="real_retail_sales_yoy",
        name="Real Retail Sales (YoY)",
        tier="medium",
        cluster="sentiment",
        description="Inflation-adjusted retail and food services sales, YoY change.",
        rationale="Coincident consumer demand; persistent negative YoY readings align with NBER recessions.",
        fetch=_fetch_and_transform(lambda: fred.fetch_series("RRSFS"), _yoy_monthly),
        score_fn=_score_inverted,
    ),

    # --- MEDIUM (markets) ---
    Indicator(
        key="sp500_drawdown",
        name="S&P 500 Drawdown from 52-Week High",
        tier="medium",
        cluster="markets",
        description="Percent decline from rolling 252-trading-day maximum.",
        rationale="Stocks predict recessions imperfectly (Samuelson: '9 of last 5'), but deep drawdowns coincide with downturns.",
        fetch=_fetch_and_transform(lambda: stooq.fetch_close("^spx"), _sp500_drawdown),
        score_fn=_score_inverted,
    ),

    # --- MEDIUM (credit/money) ---
    Indicator(
        key="real_m2_yoy",
        name="Real M2 Money Supply (YoY)",
        tier="medium",
        cluster="credit",
        description="M2 deflated by CPI, year-over-year change.",
        rationale="Former LEI component (deprecated 2012). Sharp contractions preceded 2008 and 2020.",
        fetch=_real_m2_yoy,
        score_fn=_score_inverted,
    ),

    # --- LOW / unconventional ---
    Indicator(
        key="copper_gold_ratio_yoy",
        name="Copper/Gold Ratio (YoY)",
        tier="low",
        cluster="markets",
        description="Ratio of copper to gold prices, YoY change. 'Dr. Copper' sees economic weakness.",
        rationale="Industrial-vs-safe-haven preference. Modest correlation with growth surprises; weak NBER signal alone.",
        fetch=_copper_gold_yoy,
        score_fn=_score_inverted,
    ),
    Indicator(
        key="google_trends_recession",
        name="Google Trends: 'recession'",
        tier="low",
        cluster="sentiment",
        description="Google Trends interest score (0-100) for 'recession' in the US.",
        rationale="Choi & Varian (2012) — search data nowcasts. Recession searches spike at/after the event, not before.",
        fetch=trends.fetch_recession_interest,
        score_fn=score_linear,
    ),
    Indicator(
        key="rword_index",
        name="R-Word Index (GDELT news mentions)",
        tier="low",
        cluster="sentiment",
        description="Volume of US English-language news articles mentioning 'recession'.",
        rationale="Economist magazine's heuristic. Near-coincident, sentiment-amplifying. Low standalone predictive value.",
        fetch=nyt.fetch_rword_index,
        score_fn=score_linear,
    ),
    Indicator(
        key="lipstick_proxy",
        name="Lipstick Proxy (Estée Lauder vs S&P 500, 1Y)",
        tier="low",
        cluster="markets",
        description="Trailing 1-year relative return of Estée Lauder vs S&P 500. EL outperformance (loosely) implies the 'lipstick effect'.",
        rationale="Hill et al. (2012) found mixed evidence. Included for cultural completeness; AUC ≈ 0.5 (no skill).",
        fetch=_lipstick_relative,
        score_fn=score_linear,
    ),
]


def by_key(key: str) -> Indicator:
    for ind in INDICATORS:
        if ind.key == key:
            return ind
    raise KeyError(key)
