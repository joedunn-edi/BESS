"""
fetch_ercot_solar_pilot.py — one-off pilot fetch: a few real days of
ERCOT solar generation (actual + ERCOT's own forecasts, FarWest and
system-wide), to confirm fetch_solar_generation() works end-to-end
against the real API before committing to a longer backfill.

Run with your own ERCOT credentials set as environment variables —
never paste them into any file or chat:

    export ERCOT_USERNAME="you@example.com"
    export ERCOT_PASSWORD="..."
    export ERCOT_SUBSCRIPTION_KEY="..."
    python3 scripts/fetch_ercot_solar_pilot.py
"""

import os
from datetime import date, timedelta

from bess.sources_ercot import get_token
from bess.sources_ercot_solar import fetch_solar_generation

DAYS = [date(2026, 8, 24), date(2026, 8, 25), date(2026, 8, 26)]

token = get_token(
    username=os.environ["ERCOT_USERNAME"],
    password=os.environ["ERCOT_PASSWORD"],
    subscription_key=os.environ["ERCOT_SUBSCRIPTION_KEY"],
)

for d in DAYS:
    df = fetch_solar_generation(d, token=token)
    print(f"--- {d} ---")
    print(f"{len(df)} rows")
    print(df[["hour_ending", "solar_gen_farwest_mw", "solar_stppf_farwest_mw", "solar_pvgrpp_farwest_mw"]].to_string(index=False))
    print()
