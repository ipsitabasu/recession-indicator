"""Stooq public CSV loader for market data."""
from __future__ import annotations

import io
import time
from functools import lru_cache

import pandas as pd
import requests

STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"
USER_AGENT = "recession-indicator/1.0"


class StooqFetchError(RuntimeError):
    pass


@lru_cache(maxsize=32)
def fetch_close(symbol: str, *, timeout: float = 30.0, retries: int = 3) -> pd.Series:
    """Daily close series for `symbol` (e.g. '^spx', 'hg.f', 'gc.f', 'el.us')."""
    url = STOOQ_URL.format(symbol=symbol)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            text = resp.text.strip()
            if not text or text.lower().startswith("no data"):
                raise StooqFetchError(f"Stooq returned no data for {symbol}")
            df = pd.read_csv(io.StringIO(text))
            df.columns = [c.strip().lower() for c in df.columns]
            if "date" not in df.columns or "close" not in df.columns:
                raise StooqFetchError(f"Unexpected Stooq schema for {symbol}: {df.columns.tolist()}")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            s = df.set_index("date")["close"].dropna().sort_index()
            s.name = symbol
            if s.empty:
                raise StooqFetchError(f"Empty close series for {symbol}")
            return s
        except (requests.RequestException, ValueError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise StooqFetchError(f"Failed to fetch {symbol}: {last_err}")
