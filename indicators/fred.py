"""FRED public CSV loader. No API key required."""
from __future__ import annotations

import io
import time
from functools import lru_cache

import pandas as pd
import requests

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
USER_AGENT = "recession-indicator/1.0 (+https://github.com/ipsitabasu/recession-indicator)"


class FredFetchError(RuntimeError):
    pass


@lru_cache(maxsize=64)
def fetch_series(series_id: str, *, timeout: float = 30.0, retries: int = 3) -> pd.Series:
    url = FRED_CSV_URL.format(series_id=series_id)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            df.columns = [c.strip().lower() for c in df.columns]
            date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
            value_col = series_id.lower() if series_id.lower() in df.columns else df.columns[1]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
            s = df.set_index(date_col)[value_col].dropna().sort_index()
            s.name = series_id
            if s.empty:
                raise FredFetchError(f"FRED series {series_id} returned empty data")
            return s
        except (requests.RequestException, ValueError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise FredFetchError(f"Failed to fetch {series_id}: {last_err}")
