"""R-word index via GDELT 2.0 (no API key required).

Counts daily news articles mentioning 'recession' in US English-language
sources. GDELT's free Doc API returns a timeline of article counts.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

USER_AGENT = "recession-indicator/1.0"
GDELT_URL = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    "?query=recession%20sourcecountry:US%20sourcelang:english"
    "&mode=timelinevolraw&format=json&timespan=24m"
)


class RwordFetchError(RuntimeError):
    pass


def fetch_rword_index(*, retries: int = 3, timeout: float = 30.0) -> pd.Series:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(GDELT_URL, timeout=timeout, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            data = resp.json()
            tl = data.get("timeline", [])
            if not tl or "data" not in tl[0]:
                raise RwordFetchError("GDELT timeline empty")
            rows = tl[0]["data"]
            idx = pd.to_datetime([r["date"] for r in rows], format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
            vals = pd.to_numeric([r["value"] for r in rows], errors="coerce")
            s = pd.Series(vals, index=idx, name="rword_index").dropna().sort_index()
            s.index = s.index.tz_convert(None)  # naive datetimes
            if s.empty:
                raise RwordFetchError("Parsed empty R-word series")
            return s
        except (requests.RequestException, ValueError, KeyError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RwordFetchError(f"Failed GDELT fetch: {last_err}")
