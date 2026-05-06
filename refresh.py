"""Daily refresh: fetch each indicator, compute composite, append history."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from indicators.registry import INDICATORS
from scoring import IndicatorResult, composite, tier_for

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
HISTORY_CSV = DATA_DIR / "history.csv"
LATEST_JSON = DATA_DIR / "latest.json"

HISTORY_COLUMNS = ["timestamp", "indicator", "value", "score", "weight", "as_of", "error"]


def _safe_float(x) -> float | None:
    try:
        if x is None:
            return None
        f = float(x)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def run_indicator(ind) -> IndicatorResult:
    try:
        series = ind.fetch()
        if series is None or series.empty:
            raise RuntimeError("fetch returned empty series")
        value = _safe_float(series.iloc[-1])
        score = ind.score(series)
        as_of = pd.Timestamp(series.index[-1]).strftime("%Y-%m-%d")
        return IndicatorResult(
            key=ind.key, name=ind.name, tier=ind.tier, cluster=ind.cluster,
            weight=ind.weight, auc=ind.auc, value=value, score=score, as_of=as_of,
        )
    except Exception as e:
        return IndicatorResult(
            key=ind.key, name=ind.name, tier=ind.tier, cluster=ind.cluster,
            weight=ind.weight, auc=ind.auc, value=None, score=None,
            error=f"{type(e).__name__}: {e}",
        )


def append_history(results: list[IndicatorResult], composite_score: float, ts: pd.Timestamp) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    rows = []
    for r in results:
        rows.append({
            "timestamp": ts.isoformat(),
            "indicator": r.key,
            "value": r.value,
            "score": r.score,
            "weight": r.weight,
            "as_of": r.as_of,
            "error": r.error,
        })
    rows.append({
        "timestamp": ts.isoformat(),
        "indicator": "_composite",
        "value": composite_score,
        "score": composite_score,
        "weight": 1.0,
        "as_of": ts.strftime("%Y-%m-%d"),
        "error": None,
    })
    new_df = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    if HISTORY_CSV.exists():
        old = pd.read_csv(HISTORY_CSV)
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(HISTORY_CSV, index=False)


def write_latest(results: list[IndicatorResult], composite_score: float, contribs: dict[str, float], ts: pd.Timestamp) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    payload = {
        "timestamp": ts.isoformat(),
        "composite": round(composite_score, 2),
        "tier": tier_for(composite_score),
        "indicators": [
            {
                "key": r.key, "name": r.name, "tier": r.tier, "cluster": r.cluster,
                "weight": round(r.weight, 4), "auc": round(r.auc, 3),
                "value": r.value, "score": None if r.score is None else round(r.score, 2),
                "contribution": round(contribs.get(r.key, 0.0), 2),
                "as_of": r.as_of, "error": r.error,
            }
            for r in results
        ],
    }
    LATEST_JSON.write_text(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    ts = pd.Timestamp.now(tz=timezone.utc)
    results: list[IndicatorResult] = []
    print(f"Refreshing {len(INDICATORS)} indicators @ {ts.isoformat()}")
    for ind in INDICATORS:
        r = run_indicator(ind)
        results.append(r)
        if not args.quiet:
            if r.error:
                print(f"  ✗ {ind.key:32s} ERROR: {r.error}")
            else:
                print(f"  ✓ {ind.key:32s} value={r.value:>10.3f}  score={r.score:>6.2f}  w={r.weight*100:5.2f}%  as_of={r.as_of}")

    score, contribs = composite(results)
    print(f"\nComposite recession risk: {score:.1f} / 100  ({tier_for(score)})")

    append_history(results, score, ts)
    write_latest(results, score, contribs, ts)
    print(f"Wrote {HISTORY_CSV} and {LATEST_JSON}")

    sahm = next((r for r in results if r.key == "sahm_rule"), None)
    if sahm and sahm.value is not None and sahm.value >= 0.5:
        print(f"⚠️  Sahm Rule triggered ({sahm.value:.2f}) — historically a near-perfect recession signal.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
