"""
fetch_ercot_rtm_full_year.py — backfill a trailing year of real ERCOT RTM
(HB_WEST, 15-minute) data, cached to data/ercot_rtm_west.parquet, for
building a real-time forecaster/controller (RTM is the genuinely
uncertain signal — unlike DAM, which is known a day ahead by
construction, so there's nothing to forecast there).

Uses fetch_rtm_prices_range() in 9-day chunks: RTM's 15-minute grid means
ERCOT's 1000-record page limit is reached at ~10 days of one hub, not
DAM's ~41 — so a full year needs 41 chunked requests here, not 12. Paced
via fetch_with_retry() to stay under ERCOT's ~20 requests/minute limit
and back off on a 429.

Run with your own ERCOT credentials set as environment variables —
never paste them into any file or chat:

    export ERCOT_USERNAME="you@example.com"
    export ERCOT_PASSWORD="..."
    export ERCOT_SUBSCRIPTION_KEY="..."
    python3 scripts/fetch_ercot_rtm_full_year.py
"""

import os
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from _ercot_backfill_utils import fetch_with_retry

from bess.pipeline import _merge_with_cache, compute_quality_report
from bess.sources_ercot import CHICAGO, SOURCE_RTM, fetch_rtm_prices_range, get_token

CACHE_PATH = Path("data/ercot_rtm_west.parquet")
CHUNK_DAYS = 9

end_date = date.today() - timedelta(days=1)
start_date = end_date - timedelta(days=364)

token = get_token(
    username=os.environ["ERCOT_USERNAME"],
    password=os.environ["ERCOT_PASSWORD"],
    subscription_key=os.environ["ERCOT_SUBSCRIPTION_KEY"],
)

chunks = []
d = start_date
while d <= end_date:
    chunk_end = min(d + timedelta(days=CHUNK_DAYS - 1), end_date)
    chunks.append((d, chunk_end))
    d = chunk_end + timedelta(days=1)

frames = []
for chunk_start, chunk_end in chunks:
    try:
        frames.append(fetch_with_retry(fetch_rtm_prices_range, chunk_start, chunk_end, token=token, session=requests))
        print(f"  fetched {chunk_start} to {chunk_end}")
    except (requests.exceptions.RequestException, ValueError) as exc:
        warnings.warn(f"chunk {chunk_start} to {chunk_end} failed: {exc}")

combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
report = compute_quality_report(combined, start_date, end_date, source=SOURCE_RTM, tz=CHICAGO, period_minutes=15)

CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
merged = _merge_with_cache(combined, CACHE_PATH, tz=CHICAGO)
merged.to_parquet(CACHE_PATH, index=False)

print()
print(report.summary())
print(f"\ncached {len(merged)} rows to {CACHE_PATH}")
