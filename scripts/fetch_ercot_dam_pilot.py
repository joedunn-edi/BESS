"""
fetch_ercot_dam_pilot.py — one-off pilot fetch: 2 weeks of real ERCOT DAM
(HB_WEST) data, cached to data/ercot_dam_west.parquet, to confirm the
whole real pipeline works end-to-end before committing to a longer fetch.

Run with your own ERCOT credentials set as environment variables —
never paste them into any file or chat:

    export ERCOT_USERNAME="you@example.com"
    export ERCOT_PASSWORD="..."
    export ERCOT_SUBSCRIPTION_KEY="..."
    python3 scripts/fetch_ercot_dam_pilot.py
"""

import os
from datetime import date
from pathlib import Path

from bess.pipeline import GapThresholdExceededError, run_ercot_dam_pipeline
from bess.sources_ercot import get_token

START_DATE = date(2026, 8, 24)
END_DATE = date(2026, 9, 6)

token = get_token(
    username=os.environ["ERCOT_USERNAME"],
    password=os.environ["ERCOT_PASSWORD"],
    subscription_key=os.environ["ERCOT_SUBSCRIPTION_KEY"],
)

try:
    combined, report = run_ercot_dam_pipeline(
        START_DATE,
        END_DATE,
        token=token,
        cache_path=Path("data/ercot_dam_west.parquet"),
    )
    print(report.summary())
    print(f"\ncached {len(combined)} rows to data/ercot_dam_west.parquet")
except GapThresholdExceededError as exc:
    # per ADR-008, any good days were still cached before this raised —
    # only the gappy day(s) named below are the problem
    print(f"gaps found, but good days are still cached: {exc}")
