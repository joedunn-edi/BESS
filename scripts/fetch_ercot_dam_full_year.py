"""
fetch_ercot_dam_full_year.py — backfill a trailing year of real ERCOT DAM
(HB_WEST) data, cached to data/ercot_dam_west.parquet, sized to compare
against GB's own full-year Tier 1 result.

Uses fetch_dam_prices_range() rather than the pilot script's one-day-per-
request loop: ERCOT pages responses at 1000 records, so the trailing year
is split into 12 chunks (~30 days each, ~720-744 records, safely under
that limit) — 12 requests total instead of 365.

Run with your own ERCOT credentials set as environment variables —
never paste them into any file or chat:

    export ERCOT_USERNAME="you@example.com"
    export ERCOT_PASSWORD="..."
    export ERCOT_SUBSCRIPTION_KEY="..."
    python3 scripts/fetch_ercot_dam_full_year.py
"""

import os
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from bess.pipeline import _merge_with_cache, compute_quality_report
from bess.sources_ercot import CHICAGO, SOURCE_DAM, fetch_dam_prices_range, get_token

CACHE_PATH = Path("data/ercot_dam_west.parquet")
N_CHUNKS = 12

end_date = date.today() - timedelta(days=1)
start_date = end_date - timedelta(days=364)

token = get_token(
    username=os.environ["ERCOT_USERNAME"],
    password=os.environ["ERCOT_PASSWORD"],
    subscription_key=os.environ["ERCOT_SUBSCRIPTION_KEY"],
)

total_days = (end_date - start_date).days + 1
base, extra = divmod(total_days, N_CHUNKS)
chunks = []
d = start_date
for i in range(N_CHUNKS):
    length = base + (1 if i < extra else 0)
    chunk_end = d + timedelta(days=length - 1)
    chunks.append((d, chunk_end))
    d = chunk_end + timedelta(days=1)

frames = []
for chunk_start, chunk_end in chunks:
    try:
        frames.append(fetch_dam_prices_range(chunk_start, chunk_end, token=token, session=requests))
        print(f"  fetched {chunk_start} to {chunk_end}")
    except (requests.exceptions.RequestException, ValueError) as exc:
        warnings.warn(f"chunk {chunk_start} to {chunk_end} failed: {exc}")

combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
report = compute_quality_report(combined, start_date, end_date, source=SOURCE_DAM, tz=CHICAGO, period_minutes=60)

CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
merged = _merge_with_cache(combined, CACHE_PATH, tz=CHICAGO)
merged.to_parquet(CACHE_PATH, index=False)

print()
print(report.summary())
print(f"\ncached {len(merged)} rows to {CACHE_PATH}")
