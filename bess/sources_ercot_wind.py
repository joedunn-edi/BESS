"""
sources_ercot_wind.py — fetcher for ERCOT's wind generation report
(NP4-742-CD, "Wind Power Production - Hourly Averaged Actual and
Forecasted Values by Geographical Region"), for the same weather/supply
exploration as sources_ercot_solar.py (see DECISIONS.md) — added after
the solar Stage 1 check showed price spikes cluster at evening hours
when solar is already zero, pointing at wind (Texas's evening peak being
a well-known wind-ramp-down story) as the more promising candidate.

Structurally identical to sources_ercot_solar.py — same report family,
same repost/retention-window behaviour (confirmed live, 2026-09-23: an
identical query shape returned exactly 576 rows for one day, matching
solar's pattern exactly), same DSTFlag/null-value handling already
learned there. Kept as a separate, parallel module rather than a shared
abstraction — the two reports use different regional taxonomies (solar's
"FarWest" isn't wind's "West"), and this project's own precedent
(sources_ercot.py's DAM/RTM fetchers) already favours parallel, similarly
-shaped fetchers over factoring out a shared "actual+forecast by geo"
abstraction for two call sites.

`WGRPP`/`STWPF` are wind's forecast fields, confirmed live to parallel
solar's `PVGRPP`/`STPPF` naming exactly (Wind/Short-Term Wind Generation
Resource Power Potential vs Photovoltaic equivalents). Confirmed against
ERCOT's own published methodology (2026-09-27, not just inferred from
field names): both are published for the next 48 hours, but at different
quantiles of ERCOT's probabilistic wind forecast — STWPF is a P50
forecast (actual production expected to exceed it 50% of the time, a
median/"typical" estimate), WGRPP is a P80 forecast (expected to exceed
it 80% of the time — deliberately conservative/low-side, built for grid-
reliability planning: "how much wind can be safely counted on," not
"what will most likely happen"). This directly explains a real finding
in DECISIONS.md's ADR-022: WGRPP tracked actual generation far more
accurately than STWPF specifically during genuine wind lulls, at the
correct 48h-ahead vintage — the same asymmetric-quantile reasoning this
project applied to its own price forecaster (the quantile_alpha
experiments in ADR-021), mirrored in the opposite direction (ERCOT's P80
avoids overestimating available wind; this project's high-alpha
experiments avoided underestimating price peaks) — not a coincidence,
the same statistical tool applied to two different asymmetric-cost
decisions. Source: https://www.ercot.com/mp/data-products/data-product-details?id=NP4-732-CD

Responsible for:
    * fetch_wind_generation(): one day of actual + ERCOT's own two
      forecast horizons (STWPF, WGRPP), for West (confirmed live: the
      region sums (Panhandle + Coastal + South + West + North) reproduce
      genSystemWide exactly, and West alone is ~62% of system-wide
      generation in a real sample — the McCamey/Sweetwater wind corridor,
      the single largest of ERCOT's five wind regions) and system-wide.
    * fetch_wind_forecast_snapshot(): the forecast (STWPF, WGRPP) ONLY,
      as it stood at a specific lead time before delivery — confirmed
      live (2026-09-24): querying for a delivery day 2 days ahead of
      "now" already returns real, non-null STWPF/WGRPP (while genWest/
      genSystemWide were null, since that day hadn't happened yet),
      proving these are genuine forecasts published at least 48h ahead,
      not live observations. This is the leakage-safe one — the value a
      real trading decision would actually have had in hand at that lead
      time — unlike fetch_wind_generation()'s forecast columns, which are
      deliberately fetched AFTER delivery (for the most accurate actual-
      generation reading) and so may reflect a later, closer-to-delivery
      revision of the forecast, not what was truly known in advance.

Deliberately NOT responsible for:
    * the other four named regions (Panhandle, Coastal, South, North) or
      COPHSL — same reasoning as sources_ercot_solar.py
    * fetch_wind_generation()'s STWPF/WGRPP columns being leakage-safe —
      see fetch_wind_forecast_snapshot() above for the one that is
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from bess.schema import settlement_day_utc_bounds
from bess.sources_ercot import CHICAGO, BASE_URL, ErcotToken, _auth_headers, _rows_as_dicts

WIND_PRODUCT_PATH = "/np4-742-cd/wpp_hrly_actual_fcast_geo"
SOURCE_WIND = "ercot_wind_west"

_TIMEOUT_S = 30


def _to_float_or_nan(value: float | None) -> float:
    """Same real, confirmed reason as sources_ercot_solar.py's own
    _to_float_or_nan(): ERCOT genuinely returns JSON null for some
    numeric fields on some hours — reported as NaN, not a crash."""
    return float(value) if value is not None else float("nan")


def fetch_wind_generation(delivery_date: date, token: ErcotToken, session: requests.Session = requests) -> pd.DataFrame:
    """
    Fetch one day of ERCOT wind generation (actual + two forecast
    horizons), West and system-wide, deduplicated to one row per hour.
    See the module docstring (and sources_ercot_solar.py's, which this
    mirrors) for the repost/retention-window reasoning behind the
    postedDatetime window used here.
    """
    posted_from = delivery_date + timedelta(days=1)
    posted_to = delivery_date + timedelta(days=2)
    response = session.get(
        BASE_URL + WIND_PRODUCT_PATH,
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
            f"{SOURCE_WIND}: {delivery_date} came back paginated ({total_pages} pages) — "
            "the postedDatetime window may need narrowing"
        )

    records = _rows_as_dicts(payload)
    if not records:
        raise ValueError(f"{SOURCE_WIND}: no records returned for {delivery_date}")

    # first occurrence wins; keyed on (hourEnding, DSTFlag) — see
    # sources_ercot_solar.py's identical comment for why hourEnding alone
    # would wrongly collapse the US autumn clock-change day's repeated hour
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
            "wind_gen_west_mw": np.array([_to_float_or_nan(r["genWest"]) for r in ordered], dtype="float64"),
            "wind_gen_systemwide_mw": np.array(
                [_to_float_or_nan(r["genSystemWide"]) for r in ordered], dtype="float64"
            ),
            "wind_stwpf_west_mw": np.array([_to_float_or_nan(r["STWPFWest"]) for r in ordered], dtype="float64"),
            "wind_wgrpp_west_mw": np.array([_to_float_or_nan(r["WGRPPWest"]) for r in ordered], dtype="float64"),
        }
    )


def fetch_wind_forecast_snapshot(
    delivery_date: date, lead_days: int, token: ErcotToken, session: requests.Session = requests
) -> pd.DataFrame:
    """
    Fetch the West wind FORECAST (STWPF, WGRPP) for one delivery day, as
    it stood at the EARLIEST posting within a one-day window starting
    `lead_days` before delivery — i.e. as close as this window allows to
    the forecast a real trading decision `lead_days` ahead would actually
    have had in hand. See the module docstring for why this differs from
    fetch_wind_generation()'s forecast columns (which are fetched after
    delivery and may reflect a later revision, not what was truly known
    in advance).

    No actual-generation columns here at all — that's fetch_wind_
    generation()'s job; this function is deliberately forecast-only.
    """
    posted_from = delivery_date - timedelta(days=lead_days)
    posted_to = posted_from + timedelta(days=1)
    response = session.get(
        BASE_URL + WIND_PRODUCT_PATH,
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
            f"{SOURCE_WIND}: forecast snapshot for {delivery_date} (lead_days={lead_days}) came back "
            f"paginated ({total_pages} pages) — the postedDatetime window may need narrowing"
        )

    records = _rows_as_dicts(payload)
    if not records:
        raise ValueError(f"{SOURCE_WIND}: no forecast records returned for {delivery_date} at lead_days={lead_days}")

    # earliest posting wins — the forecast is genuinely revised over
    # time (unlike fetch_wind_generation()'s actual-generation value,
    # which is stable across reposts), so we deliberately want the
    # earliest snapshot in the window, not whichever happens to be
    # returned first (the API's own default sort is postedDatetime DESC —
    # latest first — so this must be sorted explicitly, not just deduped
    # in response order the way fetch_wind_generation() safely can be)
    records_by_posted_asc = sorted(records, key=lambda r: r["postedDatetime"])
    by_hour: dict[tuple[int, bool], dict] = {}
    for r in records_by_posted_asc:
        by_hour.setdefault((int(r["hourEnding"]), bool(r.get("DSTFlag", False))), r)

    ordered = sorted(by_hour.values(), key=lambda r: (int(r["hourEnding"]), bool(r.get("DSTFlag", False))))
    start_utc, _ = settlement_day_utc_bounds(delivery_date, tz=CHICAGO)
    n = len(ordered)

    return pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime([start_utc + timedelta(hours=i) for i in range(n)], utc=True),
            "delivery_date": pd.Series([pd.Timestamp(delivery_date)] * n, dtype="datetime64[ns]"),
            "hour_ending": np.array([int(r["hourEnding"]) for r in ordered], dtype="int64"),
            "wind_stwpf_west_mw": np.array([_to_float_or_nan(r["STWPFWest"]) for r in ordered], dtype="float64"),
            "wind_wgrpp_west_mw": np.array([_to_float_or_nan(r["WGRPPWest"]) for r in ordered], dtype="float64"),
        }
    )
