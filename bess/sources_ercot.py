"""
sources_ercot.py — fetchers for ERCOT day-ahead (NP4-190-CD, hourly) and
real-time (NP6-905-CD, 15-minute) settlement point prices, for the West
trading hub (HB_WEST).

Field names and endpoint behaviour below were cross-checked against the
`gridstatus` open-source library (github.com/gridstatus/gridstatus), which
already parses these same two endpoints in production. Confirmed this
way: the base URL, both endpoint paths, and the query parameter names
(deliveryDateFrom/deliveryDateTo — there is no settlement-point query
filter; you fetch every settlement point for the date and filter locally,
same pattern as sources_elexon.py's day-ahead fetcher). The token
*request format* was NOT correctly inferred this way, and was only fixed
after a live 400 Bad Request against a real account (2026-09-07) and
ERCOT's own official example code: credentials go as URL query
parameters, not a POST body, and the Bearer token is `access_token`, not
`id_token` — see get_token()'s docstring. Still genuinely unverified: the
*exact casing* of the JSON field names in the DAM/RTM response bodies
themselves (gridstatus's rename map handles several historical casing
variants across both its CSV-report and JSON-API code paths, so which one
these specific endpoints use isn't 100% pinned down without a live
response) — marked VERIFY below.

The one load-bearing, non-obvious fact this research surfaced: ERCOT
reports hours in "Hour Ending" form (the label marks the END of the hour,
not the start) and publishes an explicit DSTFlag ("Y"/"N") to disambiguate
the repeated hour on the US autumn clock-change day — GB's Elexon data
never needed anything like this, because settlement_period there is
already a real elapsed-time position, not a wall-clock label that can
repeat. Handled here by sorting each day's records into true chronological
order (DSTFlag breaks the tie on the repeated hour) and assigning
settlement_period by *position* in that sorted order — the same
"position-in-sequence, not label arithmetic" approach schema.full_grid()
already uses — rather than trying to compute an elapsed-hours offset
directly from the hourEnding label, which would get the repeated hour
silently wrong.

Responsible for:
    * ERCOT's two-part auth (a subscription key + an hourly-expiring
      OAuth ID token — genuinely more involved than Elexon's no-auth API)
    * fetch_dam_prices() / fetch_rtm_prices(): one day of HB_WEST prices,
      parsed into the canonical schema (currency="USD", not converted to GBP)

Deliberately NOT responsible for:
    * caching/gap-reporting (pipeline.py) or anything about which price
      series feeds an optimiser (that's still undecided for ERCOT)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from bess.schema import settlement_day_utc_bounds, validate

CHICAGO = ZoneInfo("America/Chicago")

BASE_URL = "https://api.ercot.com/api/public-reports"
TOKEN_URL = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"
SCOPE = "openid+fec253ea-0d06-4272-a5e6-b478baeecd70+offline_access"

DAM_PRODUCT_PATH = "/np4-190-cd/dam_stlmnt_pnt_prices"
RTM_PRODUCT_PATH = "/np6-905-cd/spp_node_zone_hub"

HUB = "HB_WEST"
SOURCE_DAM = "ercot_dam_west"
SOURCE_RTM = "ercot_rtm_west"

_TIMEOUT_S = 30


class AllZeroPriceSeriesError(ValueError):
    """Same guard as sources_elexon.py's — raised on an empty or all-zero price series."""


@dataclass
class ErcotToken:
    access_token: str
    subscription_key: str
    obtained_at: float | None = None

    def __post_init__(self):
        # None (not 0.0) is the "unset" sentinel — 0.0 is a legitimate
        # timestamp (e.g. for tests constructing a deliberately-expired
        # token) and must not be silently overwritten
        if self.obtained_at is None:
            self.obtained_at = time.time()

    @property
    def expired(self) -> bool:
        # tokens are valid for 1 hour with no refresh mechanism — treat
        # anything over 50 minutes old as expired, leaving margin
        return time.time() - self.obtained_at > 50 * 60


def get_token(username: str, password: str, subscription_key: str) -> ErcotToken:
    """
    Exchange ERCOT account credentials for a short-lived access token. Call
    again whenever token.expired is True — there is no refresh-token
    shortcut used here (the response does include one, but a fresh
    password grant is simpler than managing refresh-token state for a
    batch fetcher that runs occasionally, not continuously).

    Requires a real account registered at apiexplorer.ercot.com, plus a
    subscription (for the subscription_key) to the DAM/RTM settlement
    point price products — this project's own tooling has no way to
    create that account for you; it needs a person to register.

    Confirmed against a real account (2026-09-07, two live 400 Bad Request
    responses diagnosed and fixed in turn):

    1. ERCOT's B2C token endpoint expects these credentials as URL query
       parameters, not a POST body — passing them as `data=` (standard
       OAuth2 ROPC form-encoding, and what the endpoint's own written
       documentation implied) gets a 400. ERCOT's own official example
       code confirms query parameters, built via raw string formatting.
       Their example also extracts `access_token` from the response (not
       `id_token`, despite `response_type=id_token` in the request) and
       uses that as the Bearer token — matched here.

    2. SCOPE contains literal `+` characters used, per the old
       application/x-www-form-urlencoded convention, as separators between
       three distinct scope values (not literal plus signs) — ERCOT's
       server expects them completely unescaped. Passing SCOPE through
       `params=` like the other fields makes `requests` "correctly"
       percent-encode `+` to `%2B`, which silently changes its meaning
       into one scope value containing literal plus signs, and the server
       rejects it. Fixed by appending SCOPE to the URL unescaped, exactly
       as ERCOT's own example does, and passing only the genuinely
       variable fields (username, password, etc.) through `params=` so
       *those* still get properly encoded — a password containing `&`,
       `%`, or `+` would otherwise corrupt the query string the same way
       ERCOT's own raw-string example is vulnerable to.

    Any failure here is re-raised with the credential-bearing request URL
    stripped out of the error message — `requests.HTTPError`'s default
    message includes the full failed URL, which for this API means the
    password in plaintext. That's genuinely dangerous: it can end up in
    logs, terminal scrollback, or get pasted into a bug report or chat
    without anyone noticing it's in there (this happened once already,
    while developing this fetcher against a real account — no lasting
    exposure, since it never left a local, uncommitted file, but it
    should never be possible to repeat by accident).
    """
    url_with_scope = f"{TOKEN_URL}?scope={SCOPE}"
    try:
        response = requests.post(
            url_with_scope,
            params={
                "username": username,
                "password": password,
                "grant_type": "password",
                "client_id": CLIENT_ID,
                "response_type": "id_token",
            },
            timeout=_TIMEOUT_S,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise requests.exceptions.HTTPError(
            f"ERCOT token request failed with status {status} "
            "(URL and credentials deliberately omitted from this message — "
            "see ERCOT's API Explorer for request/response details if needed)"
        ) from None
    payload = response.json()
    return ErcotToken(access_token=payload["access_token"], subscription_key=subscription_key)


def _auth_headers(token: ErcotToken) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token.access_token}",
        "Ocp-Apim-Subscription-Key": token.subscription_key,
    }


def _raise_if_empty(records: list[dict], settlement_date: date, source: str) -> None:
    if not records:
        raise AllZeroPriceSeriesError(f"{source}: no records returned for {settlement_date}")


def _raise_if_all_zero(prices: pd.DataFrame, settlement_date: date, source: str) -> None:
    if (prices["price_per_kwh"] == 0).all():
        raise AllZeroPriceSeriesError(f"{source}: all-zero price series for {settlement_date}")


def _dst_sort_key(record: dict) -> tuple:
    """
    Order records into true chronological order for one day. hourEnding
    marks the END of each hour (VERIFY exact field casing) and repeats
    once, at the same label, during the US autumn clock-change day —
    DSTFlag ("Y" during the repeated/second pass, "N" otherwise, per
    ERCOT's own convention — VERIFY this against a real response, some
    ERCOT datasets use a bool instead of "Y"/"N") breaks that tie so the
    first real occurrence sorts before the repeated one.
    """
    hour_ending = int(record["hourEnding"])  # VERIFY field name/format — may be "01:00" style, not a bare int
    interval = int(record.get("deliveryInterval", 1))  # VERIFY — RTM only, 1-4 within the hour
    dst_flag = record.get("DSTFlag", "N")  # VERIFY field name; "Y" = repeated hour, sorts second
    return (hour_ending, interval, dst_flag == "Y")


def _hub_filter(records: list[dict], hub: str) -> list[dict]:
    """ERCOT's SPP endpoints return every settlement point for the requested
    date range — there is no server-side settlement-point filter, so we
    filter client-side, same pattern as sources_elexon.py's day-ahead
    fetcher filtering on settlementDate."""
    return [r for r in records if r["settlementPoint"] == hub]  # VERIFY field name/casing


def fetch_dam_prices(
    settlement_date: date, token: ErcotToken, hub: str = HUB, session: requests.Session = requests
) -> pd.DataFrame:
    """
    Fetch one day of ERCOT Day-Ahead Market settlement point prices for
    `hub` (hourly, USD). See the module docstring for what's confirmed vs
    still VERIFY-flagged.
    """
    start_utc, _ = settlement_day_utc_bounds(settlement_date, tz=CHICAGO)
    response = session.get(
        BASE_URL + DAM_PRODUCT_PATH,
        headers=_auth_headers(token),
        params={
            "deliveryDateFrom": settlement_date.isoformat(),
            "deliveryDateTo": (settlement_date + timedelta(days=1)).isoformat(),
        },
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    all_records = response.json().get("data", [])  # VERIFY top-level response shape (may be paginated)
    records = _hub_filter(all_records, hub)
    records = [r for r in records if r["deliveryDate"] == settlement_date.isoformat()]  # VERIFY field name
    _raise_if_empty(records, settlement_date, SOURCE_DAM)

    records.sort(key=_dst_sort_key)
    n = len(records)
    prices = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime([start_utc + timedelta(hours=i) for i in range(n)], utc=True),
            "settlement_date": pd.Series([pd.Timestamp(settlement_date)] * n, dtype="datetime64[ns]"),
            "settlement_period": np.arange(1, n + 1, dtype="int64"),
            "period_minutes": np.full(n, 60, dtype="int64"),
            "price_per_kwh": np.array([r["settlementPointPrice"] for r in records], dtype="float64") / 1000,  # VERIFY
            "currency": "USD",
            "source": SOURCE_DAM,
        }
    )
    prices = validate(prices, tz=CHICAGO)
    _raise_if_all_zero(prices, settlement_date, SOURCE_DAM)
    return prices


def fetch_rtm_prices(
    settlement_date: date, token: ErcotToken, hub: str = HUB, session: requests.Session = requests
) -> pd.DataFrame:
    """
    Fetch one day of ERCOT Real-Time Market settlement point prices for
    `hub` (15-minute, USD). Same VERIFY caveats as fetch_dam_prices(), plus
    the DST tie-break described in _dst_sort_key() matters here too — a
    15-minute grid has 4x as many periods where the ordering must be right.
    """
    start_utc, _ = settlement_day_utc_bounds(settlement_date, tz=CHICAGO)
    response = session.get(
        BASE_URL + RTM_PRODUCT_PATH,
        headers=_auth_headers(token),
        params={
            "deliveryDateFrom": settlement_date.isoformat(),
            "deliveryDateTo": (settlement_date + timedelta(days=1)).isoformat(),
        },
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    all_records = response.json().get("data", [])
    records = _hub_filter(all_records, hub)
    records = [r for r in records if r["deliveryDate"] == settlement_date.isoformat()]
    _raise_if_empty(records, settlement_date, SOURCE_RTM)

    records.sort(key=_dst_sort_key)
    n = len(records)
    prices = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime([start_utc + timedelta(minutes=15 * i) for i in range(n)], utc=True),
            "settlement_date": pd.Series([pd.Timestamp(settlement_date)] * n, dtype="datetime64[ns]"),
            "settlement_period": np.arange(1, n + 1, dtype="int64"),
            "period_minutes": np.full(n, 15, dtype="int64"),
            "price_per_kwh": np.array([r["settlementPointPrice"] for r in records], dtype="float64") / 1000,  # VERIFY
            "currency": "USD",
            "source": SOURCE_RTM,
        }
    )
    prices = validate(prices, tz=CHICAGO)
    _raise_if_all_zero(prices, settlement_date, SOURCE_RTM)
    return prices
