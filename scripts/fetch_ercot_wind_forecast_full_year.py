"""
fetch_ercot_wind_forecast_full_year.py — backfill a trailing year of the
genuinely leakage-safe 48h-ahead wind forecast (STWPF, WGRPP as they
stood 2 days before each delivery day, not the post-delivery, possibly-
revised value data/ercot_wind_west.parquet's forecast columns carry),
cached to data/ercot_wind_forecast_48h.parquet.

Added after a 3-day pilot showed the post-delivery-based Stage 2 check
may have been misleading: on a real, dramatic wind-lull day (2026-08-26),
WGRPP tracked actual generation meaningfully more closely than STWPF at
this genuine lead time — the opposite of what the full-year, post-
delivery-vintage check found. Three days isn't enough to settle which
forecast is actually better here; this gets the full year at the correct
vintage before drawing that conclusion.

Same one-request-per-day constraint and crash-hardened design (save after
every day, skip whatever's already cached) as the other ERCOT backfills
in this project.

Run with your own ERCOT credentials set as environment variables —
never paste them into any file or chat:

    export ERCOT_USERNAME="you@example.com"
    export ERCOT_PASSWORD="..."
    export ERCOT_SUBSCRIPTION_KEY="..."
    python3 scripts/fetch_ercot_wind_forecast_full_year.py
"""

import os
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from _ercot_backfill_utils import fetch_with_retry

from bess.sources_ercot import get_token
from bess.sources_ercot_wind import fetch_wind_forecast_snapshot

CACHE_PATH = Path("data/ercot_wind_forecast_48h.parquet")
LEAD_DAYS = 2

# same window as data/ercot_wind_west.parquet, so the two line up for
# forecast-vs-actual comparison
START_DATE = date(2025, 9, 8)
END_DATE = date(2026, 9, 6)

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
        day_df = fetch_with_retry(fetch_wind_forecast_snapshot, d, lead_days=LEAD_DAYS, token=token, session=requests)
        _save_day(day_df)
        print(f"  fetched and saved {d}")
        n_fetched += 1
    except (requests.exceptions.RequestException, ValueError, TypeError, KeyError) as exc:
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
