"""Google Trends search interest via pytrends."""
from __future__ import annotations

import time

import pandas as pd


class TrendsFetchError(RuntimeError):
    pass


def fetch_recession_interest(*, geo: str = "US", timeframe: str = "today 5-y", retries: int = 3) -> pd.Series:
    """Weekly Google search interest for 'recession' (0–100 normalized by Google)."""
    from pytrends.request import TrendReq

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            pytrends = TrendReq(hl="en-US", tz=0, retries=2, backoff_factor=0.5)
            pytrends.build_payload(["recession"], timeframe=timeframe, geo=geo)
            df = pytrends.interest_over_time()
            if df is None or df.empty:
                raise TrendsFetchError("Google Trends returned empty")
            s = df["recession"].astype(float)
            s.name = "google_trends_recession"
            return s
        except Exception as e:  # pytrends raises a wide range of errors
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise TrendsFetchError(f"Failed Google Trends fetch: {last_err}")
