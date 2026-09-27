"""
fetch_ercot_wind_forecast_pilot.py — one-off pilot: fetch a few real days
of the 48h-ahead wind forecast snapshot (STWPF/WGRPP as they stood 2 days
before delivery, not the post-delivery revised value), and compare
against the already-cached actual generation to sanity-check accuracy at
this specific, genuinely leakage-safe lead time.

Run with your own ERCOT credentials set as environment variables —
never paste them into any file or chat:

    export ERCOT_USERNAME="you@example.com"
    export ERCOT_PASSWORD="..."
    export ERCOT_SUBSCRIPTION_KEY="..."
    python3 scripts/fetch_ercot_wind_forecast_pilot.py
"""

import os
from datetime import date

import pandas as pd

from bess.sources_ercot import get_token
from bess.sources_ercot_wind import fetch_wind_forecast_snapshot

DAYS = [date(2026, 8, 24), date(2026, 8, 25), date(2026, 8, 26)]
LEAD_DAYS = 2

token = get_token(
    username=os.environ["ERCOT_USERNAME"],
    password=os.environ["ERCOT_PASSWORD"],
    subscription_key=os.environ["ERCOT_SUBSCRIPTION_KEY"],
)

actual_df = pd.read_parquet("data/ercot_wind_west.parquet")

for d in DAYS:
    snapshot = fetch_wind_forecast_snapshot(d, lead_days=LEAD_DAYS, token=token)
    actual = actual_df[actual_df["delivery_date"] == pd.Timestamp(d)][["hour_ending", "wind_gen_west_mw"]]
    merged = snapshot.merge(actual, on="hour_ending", how="left")
    print(f"--- {d} (forecast as it stood {LEAD_DAYS} days before) ---")
    print(merged[["hour_ending", "wind_stwpf_west_mw", "wind_wgrpp_west_mw", "wind_gen_west_mw"]].to_string(index=False))
    print()
