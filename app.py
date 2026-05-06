"""Streamlit dashboard for the recession tracker."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
LATEST = DATA_DIR / "latest.json"
HISTORY = DATA_DIR / "history.csv"

# NBER recession dates (US, post-1950) for chart shading
NBER_RECESSIONS = [
    ("1953-07-01", "1954-05-31"),
    ("1957-08-01", "1958-04-30"),
    ("1960-04-01", "1961-02-28"),
    ("1969-12-01", "1970-11-30"),
    ("1973-11-01", "1975-03-31"),
    ("1980-01-01", "1980-07-31"),
    ("1981-07-01", "1982-11-30"),
    ("1990-07-01", "1991-03-31"),
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]


@st.cache_data(ttl=600)
def load_latest() -> dict | None:
    if not LATEST.exists():
        return None
    return json.loads(LATEST.read_text())


@st.cache_data(ttl=600)
def load_history() -> pd.DataFrame:
    if not HISTORY.exists():
        return pd.DataFrame(columns=["timestamp", "indicator", "value", "score", "weight", "as_of", "error"])
    df = pd.read_csv(HISTORY, parse_dates=["timestamp"])
    return df


def gauge(score: float, tier: str) -> go.Figure:
    color = {"Low": "#16a34a", "Moderate": "#eab308", "Elevated": "#f97316", "High": "#dc2626"}[tier]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"<b>Recession Risk: {tier}</b>", "font": {"size": 22}},
        number={"suffix": " / 100", "font": {"size": 44}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.7},
            "steps": [
                {"range": [0, 25], "color": "#dcfce7"},
                {"range": [25, 50], "color": "#fef9c3"},
                {"range": [50, 70], "color": "#fed7aa"},
                {"range": [70, 100], "color": "#fecaca"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": score},
        },
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def composite_history_chart(history: pd.DataFrame) -> go.Figure:
    comp = history[history["indicator"] == "_composite"].copy()
    comp = comp.sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=comp["timestamp"], y=comp["score"],
        mode="lines+markers", line=dict(color="#2563eb", width=2),
        name="Composite", hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.1f}<extra></extra>",
    ))
    if not comp.empty:
        xmin, xmax = comp["timestamp"].min(), comp["timestamp"].max()
        for start, end in NBER_RECESSIONS:
            s, e = pd.Timestamp(start), pd.Timestamp(end)
            if e >= xmin and s <= xmax:
                fig.add_vrect(x0=max(s, xmin), x1=min(e, xmax),
                              fillcolor="lightgray", opacity=0.4, line_width=0)
    fig.update_layout(
        height=300,
        yaxis=dict(title="Composite (0-100)", range=[0, 100]),
        xaxis=dict(title=""),
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False,
    )
    fig.add_hline(y=70, line_dash="dot", line_color="#dc2626", annotation_text="High")
    fig.add_hline(y=50, line_dash="dot", line_color="#f97316", annotation_text="Elevated")
    fig.add_hline(y=25, line_dash="dot", line_color="#16a34a", annotation_text="Low/Mod")
    return fig


def indicator_history_chart(history: pd.DataFrame, key: str, name: str) -> go.Figure:
    sub = history[history["indicator"] == key].copy().sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["timestamp"], y=sub["value"], name="value",
        mode="lines+markers", line=dict(color="#0ea5e9"),
    ))
    fig.add_trace(go.Scatter(
        x=sub["timestamp"], y=sub["score"], name="risk score (0-100)",
        mode="lines+markers", line=dict(color="#dc2626", dash="dot"), yaxis="y2",
    ))
    fig.update_layout(
        height=260, title=name,
        yaxis=dict(title="value"),
        yaxis2=dict(title="score", overlaying="y", side="right", range=[0, 100]),
        margin=dict(l=40, r=40, t=40, b=30),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def render_indicator_table(indicators: list[dict]) -> None:
    rows = []
    for ind in indicators:
        rows.append({
            "Indicator": ind["name"],
            "Tier": ind["tier"],
            "Cluster": ind["cluster"],
            "Value": ind["value"] if ind["value"] is not None else "—",
            "Score": ind["score"] if ind["score"] is not None else "—",
            "Weight": f"{ind['weight']*100:.2f}%",
            "AUC@12m": f"{ind['auc']:.2f}",
            "Contribution": ind["contribution"] if ind["contribution"] is not None else 0.0,
            "As of": ind["as_of"] or "—",
            "Status": "✗ stale" if ind["error"] else "✓",
            "_key": ind["key"],
        })
    df = pd.DataFrame(rows).sort_values("Contribution", ascending=False).drop(columns=["_key"])
    st.dataframe(df, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Recession Tracker", page_icon="📉", layout="wide")
    st.title("📉 Recession Indicator Tracker")
    st.caption(
        "Composite recession-risk score (0-100) from 17 leading indicators. "
        "Each indicator is weighted by its historical AUC for predicting NBER recessions, "
        "with redundant indicators (e.g. yield-curve spreads) sharing weight to avoid double-counting. "
        "See README for methodology."
    )

    latest = load_latest()
    history = load_history()

    if latest is None:
        st.warning("No data yet. Run `python refresh.py` to generate the first snapshot.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(gauge(latest["composite"], latest["tier"]), use_container_width=True)
        st.caption(f"As of {latest['timestamp']}")
        sahm = next((i for i in latest["indicators"] if i["key"] == "sahm_rule"), None)
        if sahm and sahm["value"] is not None and sahm["value"] >= 0.5:
            st.error(f"⚠️ **Sahm Rule triggered** ({sahm['value']:.2f}). Has signaled every recession since 1970 with zero false positives.")

    with col2:
        st.subheader("Composite history")
        if history.empty or history[history["indicator"] == "_composite"].empty:
            st.info("History will appear after a few daily refreshes accumulate.")
        else:
            st.plotly_chart(composite_history_chart(history), use_container_width=True)

    st.subheader("All indicators")
    tab_all, tab_high, tab_med, tab_low = st.tabs(["All", "High-tier", "Medium-tier", "Low-tier (unconventional)"])
    with tab_all:
        render_indicator_table(latest["indicators"])
    with tab_high:
        render_indicator_table([i for i in latest["indicators"] if i["tier"] == "high"])
    with tab_med:
        render_indicator_table([i for i in latest["indicators"] if i["tier"] == "medium"])
    with tab_low:
        render_indicator_table([i for i in latest["indicators"] if i["tier"] == "low"])

    st.subheader("Indicator details")
    options = {f"{i['name']}  ({i['weight']*100:.1f}%)": i for i in latest["indicators"]}
    pick = st.selectbox("Pick an indicator", list(options.keys()))
    chosen = options[pick]
    cleft, cright = st.columns([2, 3])
    with cleft:
        st.markdown(f"**Tier**: {chosen['tier']}  \n**Cluster**: {chosen['cluster']}  \n"
                    f"**Weight**: {chosen['weight']*100:.2f}%  \n**AUC@12m**: {chosen['auc']:.2f}  \n"
                    f"**Latest value**: {chosen['value']}  \n**Score**: {chosen['score']}  \n"
                    f"**As of**: {chosen['as_of']}")
        if chosen["error"]:
            st.error(f"Last fetch failed: {chosen['error']}")
    with cright:
        if not history.empty and (history["indicator"] == chosen["key"]).any():
            st.plotly_chart(indicator_history_chart(history, chosen["key"], chosen["name"]),
                            use_container_width=True)
        else:
            st.info("Not enough history yet.")

    with st.expander("Methodology"):
        st.markdown("""
**Weighting** — each indicator's weight comes from its **AUC** (area under the ROC curve)
for predicting NBER recessions at a 12-month-ahead horizon. AUC is the standard
metric in the recession-prediction literature (Estrella & Mishkin 1998; Sahm 2019;
Gilchrist & Zakrajšek 2012).

`raw_weight = max(0, AUC - 0.5) × 2` so a no-skill indicator (AUC = 0.5) has weight
zero and a perfect indicator (AUC = 1.0) has weight one.

**Redundancy adjustment** — correlated indicators are grouped into clusters
(e.g. 10Y-3M and 10Y-2Y are both yield-curve signals). Within a cluster, the
combined effective weight equals the strongest member's raw weight; each
member's share is proportional to its own AUC. This prevents double-counting
without flattening the AUC ranking.

**Scoring** — each indicator value maps to a 0-100 risk score by piecewise-linear
interpolation between two anchors derived from its historical distribution at
NBER recession onsets vs expansion months.

**Composite** — weighted average of indicator scores, with weights renormalized to
the sum of indicators that returned data (so a missed fetch doesn't poison the score).

Run `python scripts/compute_weights.py` to recompute weights and thresholds from
current FRED data.
        """)


if __name__ == "__main__":
    main()
