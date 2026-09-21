"""
fetch_ercot_solar_full_year.py — backfill a trailing year of real ERCOT
solar generation (actual + ERCOT's own forecasts, FarWest and system-
wide), cached to data/ercot_solar_farwest.parquet, matching the same
window as data/ercot_rtm_west.parquet for the Stage A/B analysis.

Unlike DAM/RTM, this report can't be range-fetched — the postedDatetime
window needed to catch a day's data grows with the date range, which
would reintroduce the ~200+ duplicate-repost-per-hour problem the narrow
per-day window was specifically built to avoid (see
sources_ercot_solar.py's module docstring). So this is one request per
day, 364 total, paced the same way as the RTM full-year backfill.

Saves after every successful day, not just at the end, and skips any day
already present in the cache on startup — safe to interrupt or re-run;
a crash partway through only costs the days after the last save, and a
rerun picks up exactly where it left off rather than starting over.

Run with your own ERCOT credentials set as environment variables —
never paste them into any file or chat:

    export ERCOT_USERNAME="you@example.com"
    export ERCOT_PASSWORD="..."
    export ERCOT_SUBSCRIPTION_KEY="..."
    python3 scripts/fetch_ercot_solar_full_year.py
"""

import os
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from _ercot_backfill_utils import fetch_with_retry

from bess.sources_ercot import get_token
from bess.sources_ercot_solar import fetch_solar_generation

CACHE_PATH = Path("data/ercot_solar_farwest.parquet")

# matches data/ercot_rtm_west.parquet's window exactly, for Stage A/B
START_DATE = date(2025, 9, 8)
END_DATE = date(2026, 9, 6)  # solar needs delivery_date+2 <= today to be fetchable; 9-7 isn't yet postable

CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
already_cached: set[date] = set()
if CACHE_PATH.exists():
    cached = pd.read_parquet(CACHE_PATH)
    already_cached = set(cached["delivery_date"].dt.date)
    print(f"{len(already_cached)} day(s) already cached — skipping those")

token = get_token(
    username=os.environ["ERCOT_USERNAME"],
    password=os.environ["ERCOT_PASSWORD"],
    subscription_key=os.environ["ERCOT_SUBSCRIPTION_KEY"],
)


def _save_day(day_df: pd.DataFrame) -> None:
    if CACHE_PATH.exists():
        existing = pd.read_parquet(CACHE_PATH)
        combined = pd.concat([existing, day_df], ignore_index=True)
    else:
        combined = day_df
    combined = combined.drop_duplicates(subset=["delivery_date", "hour_ending"], keep="last").sort_values("timestamp_utc")
    combined.to_parquet(CACHE_PATH, index=False)


d = START_DATE
n_fetched, n_failed = 0, 0
while d <= END_DATE:
    if d in already_cached:
        d += timedelta(days=1)
        continue
    try:
        day_df = fetch_with_retry(fetch_solar_generation, d, token=token, session=requests)
        _save_day(day_df)
        print(f"  fetched and saved {d}")
        n_fetched += 1
    except (requests.exceptions.RequestException, ValueError) as exc:
        warnings.warn(f"day {d} failed: {exc}")
        n_failed += 1
    d += timedelta(days=1)

combined = pd.read_parquet(CACHE_PATH)
expected_days = (END_DATE - START_DATE).days + 1
actual_days = combined["delivery_date"].nunique()

print(f"\nthis run: {n_fetched} fetched, {n_failed} failed, {len(already_cached)} already cached")
print(f"{len(combined)} rows cached to {CACHE_PATH}")
print(f"date range: {combined['delivery_date'].min()} to {combined['delivery_date'].max()}")
print(f"days: {actual_days}/{expected_days}")
