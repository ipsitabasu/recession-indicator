"""Offline: compute AUC, redundancy-adjust, derive thresholds.

Run: python -m scripts.compute_weights

Pulls each indicator's full historical series (where backfillable from FRED;
some series like ISM PMI and Google Trends are short or unavailable
historically, in which case literature values are kept). Computes:

  1. AUC@12m for predicting NBER recession months from each indicator's
     monthly series, sign-flipped if the indicator scores risk in reverse.
  2. raw_weight = max(0, AUC - 0.5) * 2
  3. Cluster any pair with |corr(monthly diff)| > 0.7 in the post-1990
     sample; split each cluster's combined raw weight equally.
  4. Normalize so weights sum to 1.
  5. Threshold anchors: low = median of expansion-months value;
     high = 80th percentile (or 20th, if inverted) of values in the
     12-month window preceding each NBER peak.

Outputs indicators/weights.json and indicators/thresholds.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators import fred, registry  # noqa: E402

OUT_WEIGHTS = ROOT / "indicators" / "weights.json"
OUT_THRESHOLDS = ROOT / "indicators" / "thresholds.json"

# Literature-anchored AUCs used as fallback when historical data is too
# short to compute reliably (e.g. ISM PMI has no FRED-CSV history, Google
# Trends only goes back to 2004, lipstick proxy has too few cycles).
LITERATURE_AUC = {
    "yield_curve_10y3m": 0.92,    # Estrella & Mishkin 1998
    "yield_curve_10y2y": 0.89,    # Bauer & Mertens 2018
    "sahm_rule": 0.99,            # Sahm 2019
    "initial_claims_yoy": 0.82,   # LEI replication
    "hy_credit_spread": 0.85,     # Gilchrist–Zakrajšek 2012
    "ism_pmi": 0.78,              # Koenig 2002
    "building_permits_yoy": 0.76, # LEI replication
    "industrial_production_yoy": 0.72,
    "weekly_hours_mfg": 0.70,
    "umich_sentiment": 0.66,
    "real_retail_sales_yoy": 0.68,
    "sp500_drawdown": 0.62,
    "real_m2_yoy": 0.58,
    "copper_gold_ratio_yoy": 0.55,
    "google_trends_recession": 0.55,
    "rword_index": 0.52,
    "lipstick_proxy": 0.50,
}

# Direction: True if higher value = higher recession risk (e.g. claims YoY,
# HY spread, Sahm). False if lower value = higher risk (yield curve, PMI,
# permits, etc.).
RISK_DIRECTION_HIGHER_IS_RISKIER = {
    "yield_curve_10y3m": False,
    "yield_curve_10y2y": False,
    "sahm_rule": True,
    "initial_claims_yoy": True,
    "hy_credit_spread": True,
    "ism_pmi": False,
    "building_permits_yoy": False,
    "industrial_production_yoy": False,
    "weekly_hours_mfg": False,
    "umich_sentiment": False,
    "real_retail_sales_yoy": False,
    "sp500_drawdown": False,  # drawdown is negative; more negative = riskier
    "real_m2_yoy": False,
    "copper_gold_ratio_yoy": False,
    "google_trends_recession": True,
    "rword_index": True,
    "lipstick_proxy": True,
}


def _try_compute_auc(ind_key: str, ind_series: pd.Series, usrec: pd.Series) -> float | None:
    from sklearn.metrics import roc_auc_score

    monthly = ind_series.resample("MS").last().dropna()
    # 12-month lead: align indicator at t with USREC at t+12
    target = usrec.shift(-12).reindex(monthly.index).dropna()
    aligned = monthly.reindex(target.index).dropna()
    target = target.reindex(aligned.index)
    if len(aligned) < 60 or target.sum() < 5:
        return None
    score = aligned.values
    if not RISK_DIRECTION_HIGHER_IS_RISKIER[ind_key]:
        score = -score
    try:
        return float(roc_auc_score(target.values, score))
    except ValueError:
        return None


def _adjust_for_clusters(raw: dict[str, float], clusters: dict[str, str]) -> dict[str, float]:
    """A cluster of correlated indicators jointly counts as a single 'best'
    signal: the cluster's effective contribution is max(raw_in_cluster), and
    each member's share is proportional to its own raw weight. That prevents
    double-counting (10Y-3M and 10Y-2Y don't both count fully) while keeping
    higher-AUC members above lower-AUC ones within a cluster."""
    by_cluster: dict[str, list[str]] = {}
    for k, c in clusters.items():
        by_cluster.setdefault(c, []).append(k)
    adjusted: dict[str, float] = {}
    for members in by_cluster.values():
        cluster_max = max(raw[m] for m in members)
        cluster_sum = sum(raw[m] for m in members)
        if cluster_sum == 0:
            for m in members:
                adjusted[m] = 0.0
            continue
        for m in members:
            adjusted[m] = cluster_max * raw[m] / cluster_sum
    return adjusted


def _compute_thresholds(ind_key: str, ind_series: pd.Series, usrec: pd.Series) -> dict[str, float] | None:
    monthly = ind_series.resample("MS").last().dropna()
    target = usrec.reindex(monthly.index).fillna(0).astype(int)
    onsets = target.diff().fillna(0)
    onset_dates = onsets[onsets == 1].index
    onset_window_mask = pd.Series(False, index=monthly.index)
    for d in onset_dates:
        start = d - pd.DateOffset(months=12)
        onset_window_mask |= (monthly.index >= start) & (monthly.index < d)
    if not onset_window_mask.any():
        return None
    expansion_mask = (target == 0) & ~onset_window_mask
    if not expansion_mask.any():
        return None
    onset_vals = monthly[onset_window_mask]
    expansion_vals = monthly[expansion_mask]
    if RISK_DIRECTION_HIGHER_IS_RISKIER[ind_key]:
        low = float(np.nanmedian(expansion_vals))
        high = float(np.nanpercentile(onset_vals, 80))
    else:
        low = float(np.nanmedian(expansion_vals))
        high = float(np.nanpercentile(onset_vals, 20))
    return {"low": low, "high": high}


def main() -> None:
    print("Fetching NBER recession dates (USREC)…")
    usrec = fred.fetch_series("USREC")

    aucs: dict[str, float] = {}
    thresholds: dict[str, dict[str, float]] = {}
    fetched_series: dict[str, pd.Series] = {}

    for ind in registry.INDICATORS:
        print(f"  → {ind.key}")
        try:
            s = ind.fetch()
            fetched_series[ind.key] = s
        except Exception as e:
            print(f"    fetch failed ({e}); using literature AUC.")
            aucs[ind.key] = LITERATURE_AUC[ind.key]
            continue

        auc = _try_compute_auc(ind.key, s, usrec)
        if auc is None:
            print(f"    AUC not computable; using literature value {LITERATURE_AUC[ind.key]}")
            auc = LITERATURE_AUC[ind.key]
        else:
            print(f"    AUC = {auc:.3f}  (lit ≈ {LITERATURE_AUC[ind.key]})")
        aucs[ind.key] = auc

        t = _compute_thresholds(ind.key, s, usrec)
        if t is not None:
            thresholds[ind.key] = t

    # raw weights from AUC
    raw = {k: max(0.0, (v - 0.5) * 2.0) for k, v in aucs.items()}

    # cluster adjustment using registry-declared clusters (transparent + stable)
    clusters = {ind.key: ind.cluster for ind in registry.INDICATORS}
    adjusted = _adjust_for_clusters(raw, clusters)

    total = sum(adjusted.values())
    if total == 0:
        raise RuntimeError("All weights zero — cannot normalize.")
    final = {k: v / total for k, v in adjusted.items()}

    weights_out = {
        k: {"auc": round(aucs[k], 4), "weight": round(final[k], 4)}
        for k in aucs
    }
    OUT_WEIGHTS.write_text(json.dumps(weights_out, indent=2, sort_keys=True))
    print(f"Wrote {OUT_WEIGHTS}")

    # thresholds: keep what we computed, fall back to seeded values for the rest
    seed_path = ROOT / "indicators" / "thresholds.json"
    seeded = json.loads(seed_path.read_text()) if seed_path.exists() else {}
    seeded.update(thresholds)
    OUT_THRESHOLDS.write_text(json.dumps(seeded, indent=2, sort_keys=True))
    print(f"Wrote {OUT_THRESHOLDS}")

    print("\nFinal weights:")
    for k, v in sorted(final.items(), key=lambda x: -x[1]):
        print(f"  {k:32s} {v*100:5.2f}%   AUC={aucs[k]:.3f}")


if __name__ == "__main__":
    main()
