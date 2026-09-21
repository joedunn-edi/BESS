"""
sources_ercot_solar.py — fetcher for ERCOT's solar generation report
(NP4-745-CD, "Solar Power Production - Hourly Averaged Actual and
Forecasted Values by Geographical Region"), for the weather/solar-supply
exploration (see DECISIONS.md).

Confirmed live (2026-09-21): this report re-posts roughly hourly, each
time restating a rolling ~48-52 hour window of recent delivery hours,
whether or not anything actually changed — a single delivery day's 24
hours, once they've fully entered that window, get reposted identically
on every subsequent cycle (verified: the same hour's `genFarWest` etc.
was byte-identical across three consecutive hourly reposts). Querying
with `postedDatetimeFrom`/`postedDatetimeTo` set to [delivery_date+1,
delivery_date+2] reliably catches every hour of that day at least once —
confirmed to return exactly 576 rows for one day (24 hourly postings x 24
identical hours each), safely inside the retention window. Any single
posting per hour is used below, not specifically the latest, since
values don't change within that window.

`hourEnding` here is a plain integer (1-24), unlike DAM's "HH:00" string
— confirmed from the real response's `fields` metadata (`dataType:
INTEGER`). `DSTFlag` is present (a real boolean, same as DAM/RTM), so the
same position-in-sorted-order assignment discipline applies for the
autumn clock-change day.

Responsible for:
    * fetch_solar_generation(): one day of actual + ERCOT's own two
      forecast horizons (STPPF, PVGRPP), for FarWest (where most of
      ERCOT's solar capacity sits — confirmed live: FarWest was ~84% of
      system-wide generation in a real sample) and system-wide.

Deliberately NOT responsible for:
    * the other five named regions or the COPHSL field — COPHSL's exact
      meaning is unconfirmed (a real sample showed COPHSL < actual
      generation at the same hour, which doesn't match "capacity ceiling"
      the way its name suggests — not needed for this exploration, so
      left unresolved rather than guessed at)
    * any leakage-safety modelling for `STPPFFarWest`/`PVGRPPFarWest` as
      live trading features — this fetch gets whatever value was posted
      1-2 days after the fact, which is fine for a historical ceiling/
      accuracy check but is NOT the value that would have been available
      at the time a real trading decision needed it
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from bess.schema import settlement_day_utc_bounds
from bess.sources_ercot import CHICAGO, BASE_URL, ErcotToken, _auth_headers, _rows_as_dicts

SOLAR_PRODUCT_PATH = "/np4-745-cd/spp_hrly_actual_fcast_geo"
SOURCE_SOLAR = "ercot_solar_farwest"

_TIMEOUT_S = 30


def fetch_solar_generation(delivery_date: date, token: ErcotToken, session: requests.Session = requests) -> pd.DataFrame:
    """
    Fetch one day of ERCOT solar generation (actual + two forecast
    horizons), FarWest and system-wide, deduplicated to one row per hour.
    See the module docstring for the repost/retention-window reasoning
    behind the postedDatetime window used here.
    """
    posted_from = delivery_date + timedelta(days=1)
    posted_to = delivery_date + timedelta(days=2)
    response = session.get(
        BASE_URL + SOLAR_PRODUCT_PATH,
        headers=_auth_headers(token),
        params={
            "deliveryDateFrom": delivery_date.isoformat(),
            "deliveryDateTo": delivery_date.isoformat(),
            "postedDatetimeFrom": posted_from.isoformat(),
            "postedDatetimeTo": posted_to.isoformat(),
        },
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()
    total_pages = payload.get("_meta", {}).get("totalPages", 1)
    if total_pages > 1:
        raise ValueError(
            f"{SOURCE_SOLAR}: {delivery_date} came back paginated ({total_pages} pages) — "
            "the postedDatetime window may need narrowing"
        )

    records = _rows_as_dicts(payload)
    if not records:
        raise ValueError(f"{SOURCE_SOLAR}: no records returned for {delivery_date}")

    # first occurrence wins — values are stable across reposts within
    # this window, so which specific posting we keep doesn't matter.
    # Keyed on (hourEnding, DSTFlag), not hourEnding alone — on the
    # autumn clock-change day, hourEnding repeats once at the same label
    # for a genuinely different real hour, distinguished only by DSTFlag;
    # keying on hourEnding alone would wrongly collapse that repeated
    # hour into a single row instead of dropping actual repost duplicates.
    by_hour: dict[tuple[int, bool], dict] = {}
    for r in records:
        by_hour.setdefault((int(r["hourEnding"]), bool(r.get("DSTFlag", False))), r)

    ordered = sorted(by_hour.values(), key=lambda r: (int(r["hourEnding"]), bool(r.get("DSTFlag", False))))
    start_utc, _ = settlement_day_utc_bounds(delivery_date, tz=CHICAGO)
    n = len(ordered)

    return pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime([start_utc + timedelta(hours=i) for i in range(n)], utc=True),
            "delivery_date": pd.Series([pd.Timestamp(delivery_date)] * n, dtype="datetime64[ns]"),
            "hour_ending": np.array([int(r["hourEnding"]) for r in ordered], dtype="int64"),
            "solar_gen_farwest_mw": np.array([float(r["genFarWest"]) for r in ordered], dtype="float64"),
            "solar_gen_systemwide_mw": np.array([float(r["genSystemWide"]) for r in ordered], dtype="float64"),
            "solar_stppf_farwest_mw": np.array([float(r["STPPFFarWest"]) for r in ordered], dtype="float64"),
            "solar_pvgrpp_farwest_mw": np.array([float(r["PVGRPPFarWest"]) for r in ordered], dtype="float64"),
        }
    )
