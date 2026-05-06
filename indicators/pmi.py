"""ISM Manufacturing PMI fetcher.

ISM publishes a monthly headline PMI (the 'Manufacturing PMI' value).
There is no free, structured API. We try a sequence of public mirrors and
fall back to None if none succeed — refresh.py will mark the indicator as
'stale' rather than poison the composite.
"""
from __future__ import annotations

import re
import time

import pandas as pd
import requests

USER_AGENT = "Mozilla/5.0 (compatible; recession-indicator/1.0)"


class PmiFetchError(RuntimeError):
    pass


def _try_ycharts_mirror() -> pd.Series | None:
    """YCharts public series page exposes the latest value in HTML."""
    url = "https://ycharts.com/indicators/us_pmi"
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
    except requests.RequestException:
        return None
    m = re.search(r"Last Value[^0-9]*([0-9]{2}\.[0-9])", resp.text)
    d = re.search(r"Last Value Date[^A-Z]*([A-Z][a-z]+ \d{1,2}, \d{4})", resp.text)
    if not m:
        return None
    value = float(m.group(1))
    date = pd.to_datetime(d.group(1)) if d else pd.Timestamp.today().normalize()
    return pd.Series({date: value}, name="ISM_PMI")


def fetch_latest_pmi(*, retries: int = 2) -> pd.Series:
    """Returns a single-value series with the most recent PMI reading.

    History is not provided here (ISM doesn't expose a free CSV);
    compute_weights.py uses NAPM from FRED's deprecated archive when
    available for the AUC backtest.
    """
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            s = _try_ycharts_mirror()
            if s is not None and not s.empty:
                return s
            raise PmiFetchError("PMI mirror returned no data")
        except (requests.RequestException, PmiFetchError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise PmiFetchError(f"Failed to fetch ISM PMI: {last_err}")
