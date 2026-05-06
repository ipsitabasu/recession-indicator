# Recession Indicator Tracker

A self-hosted dashboard that pulls 17 US recession indicators every day,
weights each by its historical predictive skill (AUC against NBER
recessions at a 12-month horizon), and rolls them up into a single 0–100
composite recession-risk score.

The point: ignore pundits, read the data yourself.

## Quick start

```bash
pip install -r requirements.txt
python refresh.py             # pull all indicators, write data/
streamlit run app.py          # open the dashboard
```

For automated daily updates, push the repo to GitHub — the workflow at
`.github/workflows/daily-refresh.yml` runs at 06:00 UTC every day and
commits the latest data snapshot.

## What it tracks

Indicators are grouped into three tiers based on the strength of their
historical signal (AUC@12-month-ahead NBER prediction):

### High-tier (combined weight ~64%)
- **Sahm Rule** — has signaled every recession since 1970, zero false positives.
- **HY credit spread** — Gilchrist–Zakrajšek "excess bond premium" tradition.
- **10Y–3M Treasury spread** — NY Fed's preferred recession-probability input.
- **10Y–2Y Treasury spread** — same yield-curve signal, popular in media.
- **Initial jobless claims (4-wk MA, YoY)** — Conference Board LEI input.
- **ISM Manufacturing PMI** — diffusion index of factory activity.

### Medium-tier (~31%)
Building permits YoY · Industrial production YoY · Avg weekly hours mfg ·
UMich consumer sentiment · Real retail sales YoY · S&P 500 drawdown ·
Real M2 money supply YoY.

### Low / unconventional (~3%)
Copper/Gold ratio (Dr. Copper) · Google Trends "recession" · GDELT R-word
news mentions · Lipstick proxy (Estée Lauder vs S&P 500).

The unconventional indicators are kept for completeness and color, but the
math correctly down-weights them — a no-skill indicator (AUC ≈ 0.5) gets
a weight of zero by construction.

## Methodology

### Why AUC?

AUC (area under the ROC curve) is the standard metric in the recession-
prediction literature. It answers: *"if I draw one recession month and one
expansion month at random, how often does this indicator rank them
correctly?"* — 0.5 = coin flip, 1.0 = perfect.

This is the same family of methodology used by:
- Estrella & Hardouvelis (1991), Estrella & Mishkin (1996, 1998) — yield-curve probit models
- NY Fed's monthly *Probability of Recession* model
- Gilchrist & Zakrajšek (2012) — Excess Bond Premium
- Sahm (2019) — real-time unemployment rule
- Conference Board — Leading Economic Index methodology

### Step 1 — Per-indicator weight

```
raw_weight = max(0, AUC - 0.5) × 2
```

A no-skill indicator (AUC = 0.5) gets weight 0; a perfect indicator
(AUC = 1.0) gets weight 1. Linear in skill above chance.

### Step 2 — Redundancy adjustment

Indicators that measure the same underlying signal (e.g. 10Y–3M and 10Y–2Y
yield spreads) are grouped into clusters. Within a cluster:
- The cluster's combined contribution equals the strongest member's raw weight.
- Each member's share is proportional to its own AUC.

This prevents double-counting without flattening the AUC ranking inside
the cluster.

Clusters: yield-curve · labor · credit · activity · housing · sentiment ·
markets.

### Step 3 — Normalize

Final weights sum to 1. Composite = weighted average of indicator scores
on the 0–100 scale.

### Per-indicator scoring

Each indicator value maps to a 0–100 risk score by piecewise-linear
interpolation between two anchors derived from its historical distribution
at NBER recession onsets vs expansion months:
- low anchor (score 0) = expansion median
- high anchor (score 100) = 80th percentile in the 12-month window
  preceding NBER peaks

For "inverted" indicators where lower values mean higher risk (yield
curve, PMI, permits, etc.), the scoring direction is flipped at score time.

### Tiers

| Composite | Tier |
|-----------|------|
| 0–25 | Low |
| 25–50 | Moderate |
| 50–70 | Elevated |
| 70–100 | High |

## Recomputing weights from real data

`indicators/weights.json` and `indicators/thresholds.json` are seeded
from published AUC values. To recompute them from current FRED data:

```bash
python -m scripts.compute_weights
```

This pulls each indicator's full history, replicates the AUCs against
`USREC` (the NBER recession indicator series), redoes the cluster
adjustment, and overwrites the JSONs. Re-run when you add a new indicator.

## Data sources

All free and key-less:

| Source | Used for |
|--------|----------|
| FRED public CSV (`fred.stlouisfed.org/graph/fredgraph.csv?id=…`) | Most macro series |
| Stooq CSVs (`stooq.com/q/d/l/`) | S&P 500, copper, gold, Estée Lauder |
| pytrends | Google Trends "recession" search interest |
| GDELT 2.0 Doc API | News-mention R-word index |
| Public ISM mirrors | Latest ISM Manufacturing PMI |

If a source fails on a given day, that indicator is excluded from the
day's composite (its weight pulled out of the denominator) and the
dashboard shows a "stale" badge — no silent fallbacks to fake values.

## File layout

```
indicators/        — fetchers, registry, weights.json, thresholds.json
scripts/           — compute_weights.py (offline AUC + clustering)
scoring.py         — composite math
refresh.py         — daily runner; writes data/history.csv + data/latest.json
app.py             — Streamlit dashboard
data/              — committed daily snapshots
tests/             — scoring, registry, weight-derivation tests
.github/workflows/ — daily-refresh.yml
```

## Contributing / extending

Add a new indicator by appending to `INDICATORS` in
`indicators/registry.py` with its `fetch` and `score_fn`. Add an entry in
`LITERATURE_AUC` (in `scripts/compute_weights.py`) and re-run the script
to regenerate weights/thresholds.

## License

MIT.
