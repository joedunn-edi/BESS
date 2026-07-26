"""
sources_elexon.py — fetchers for GB electricity prices from the Elexon
BMRS API (base https://data.elexon.co.uk/bmrs/api/v1, no API key needed).

Responsible for:
    * fetch_imbalance_prices(): /balancing/settlement/system-prices/{date}
    * fetch_day_ahead_prices(): /balancing/pricing/market-index (APXMIDP
      provider — N2EX returns all-zero prices for the same dates, verified
      empirically; see ADR-007)
    * converting each raw response into the canonical schema (bess.schema),
      including the £/MWh -> £/kWh conversion
    * a guard against an empty or all-zero price series (ADR-007)

Deliberately NOT responsible for:
    * fetching more than one settlement day per call, caching, gap repair,
      or data-quality reporting across many days — see pipeline.py
    * deciding which series (imbalance vs day-ahead) feeds the optimiser —
      that's a stage-4 decision
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import requests

from bess.schema import settlement_day_utc_bounds, validate

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
SOURCE_IMBALANCE = "elexon_imbalance"
SOURCE_DAY_AHEAD = "elexon_apxmidp"

_TIMEOUT_S = 30


class AllZeroPriceSeriesError(ValueError):
    """Raised when a fetched price series is empty or entirely zero (ADR-007)."""


@dataclass(frozen=True)
class ImbalanceFetchResult:
    """prices: canonical schema. quality_flags: per-period BSAD/derivation
    flags, kept separately since they aren't part of the 5 canonical columns
    (see ADR-007) — settlement_date/settlement_period join back to prices."""

    prices: pd.DataFrame
    quality_flags: pd.DataFrame


def _raise_if_empty(records: list[dict], settlement_date: date, source: str) -> None:
    if not records:
        raise AllZeroPriceSeriesError(f"{source}: no records returned for {settlement_date}")


def _raise_if_all_zero(prices: pd.DataFrame, settlement_date: date, source: str) -> None:
    if (prices["price_gbp_per_kwh"] == 0).all():
        raise AllZeroPriceSeriesError(f"{source}: all-zero price series for {settlement_date}")


def fetch_imbalance_prices(settlement_date: date, session: requests.Session = requests) -> ImbalanceFetchResult:
    """
    Fetch one settlement day of GB imbalance (system) prices.

    Uses systemSellPrice as the canonical price; raises if it ever differs
    from systemBuyPrice (they're identical under GB's post-2015 single
    cash-out pricing — a difference would mean either older dual-price data
    or a market-design change this fetcher hasn't been updated for; see
    ADR-007). `session` defaults to the `requests` module itself (its `.get`
    has the same signature as `requests.Session.get`) — pass a fake session
    in tests to avoid hitting the live API.
    """
    url = f"{BASE_URL}/balancing/settlement/system-prices/{settlement_date.isoformat()}"
    response = session.get(url, timeout=_TIMEOUT_S)
    response.raise_for_status()
    records = response.json()["data"]
    _raise_if_empty(records, settlement_date, SOURCE_IMBALANCE)

    mismatched = [r for r in records if r["systemSellPrice"] != r["systemBuyPrice"]]
    if mismatched:
        raise ValueError(
            f"{SOURCE_IMBALANCE}: systemSellPrice != systemBuyPrice for "
            f"{len(mismatched)} period(s) on {settlement_date} — the single-price "
            f"assumption behind using systemSellPrice alone (ADR-007) no longer holds"
        )

    prices = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime([r["startTime"] for r in records], utc=True),
            "settlement_date": pd.to_datetime([r["settlementDate"] for r in records]),
            "settlement_period": np.array([r["settlementPeriod"] for r in records], dtype="int64"),
            "price_gbp_per_kwh": np.array([r["systemSellPrice"] for r in records], dtype="float64") / 1000,
            "source": SOURCE_IMBALANCE,
        }
    )
    prices = validate(prices)
    _raise_if_all_zero(prices, settlement_date, SOURCE_IMBALANCE)

    quality_flags = pd.DataFrame(
        {
            "settlement_date": pd.to_datetime([r["settlementDate"] for r in records]),
            "settlement_period": np.array([r["settlementPeriod"] for r in records], dtype="int64"),
            "bsad_defaulted": [r["bsadDefaulted"] for r in records],
            "price_derivation_code": [r["priceDerivationCode"] for r in records],
        }
    )

    return ImbalanceFetchResult(prices=prices, quality_flags=quality_flags)


def fetch_day_ahead_prices(settlement_date: date, session: requests.Session = requests) -> pd.DataFrame:
    """
    Fetch one settlement day of GB day-ahead prices from APXMIDP.

    The market-index endpoint takes a UTC from/to window rather than a
    settlement date, and a naive UTC-calendar-day window doesn't line up
    with the London settlement day under BST (verified empirically — see
    ADR-007). We query the DST-aware window from
    schema.settlement_day_utc_bounds() and then filter the response to rows
    whose settlementDate matches, rather than trusting the API's from/to
    boundary behaviour to return exactly one day's periods.
    """
    start_utc, end_utc = settlement_day_utc_bounds(settlement_date)
    response = session.get(
        f"{BASE_URL}/balancing/pricing/market-index",
        params={
            "from": start_utc.isoformat(),
            "to": end_utc.isoformat(),
            "dataProviders": "APXMIDP",
        },
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    records = [r for r in response.json()["data"] if r["settlementDate"] == settlement_date.isoformat()]
    _raise_if_empty(records, settlement_date, SOURCE_DAY_AHEAD)

    prices = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime([r["startTime"] for r in records], utc=True),
            "settlement_date": pd.to_datetime([r["settlementDate"] for r in records]),
            "settlement_period": np.array([r["settlementPeriod"] for r in records], dtype="int64"),
            "price_gbp_per_kwh": np.array([r["price"] for r in records], dtype="float64") / 1000,
            "source": SOURCE_DAY_AHEAD,
        }
    )
    prices = validate(prices)
    _raise_if_all_zero(prices, settlement_date, SOURCE_DAY_AHEAD)
    return prices
