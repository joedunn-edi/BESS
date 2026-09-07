"""
sources_ercot.py — fetchers for ERCOT day-ahead (NP4-190-CD, hourly) and
real-time (NP6-905-CD, 15-minute) settlement point prices, for the West
trading hub (HB_WEST).

Confirmed against live responses from a real account (2026-09-07):

    * Both endpoints paginate (1000 records/page) and return *every*
      settlement point in ERCOT (over a thousand resource nodes, load
      zones and hubs) unless filtered. `settlementPoint` is a genuine
      server-side query filter (confirmed: adding it dropped a
      53,664-record/54-page DAM response to a single page) — so we filter
      by hub server-side, never paginate, and never filter client-side.
    * `deliveryDateFrom`/`deliveryDateTo` are both *inclusive* (confirmed:
      a `08-24`→`08-25` range returned both days in full — 192 RTM records
      for one hub = 2 days × 96 intervals). A single day's fetch therefore
      sets both to the same date, and needs no post-fetch date filter.
    * The response body is `{"fields": [...], "data": [[...], ...]}` —
      each row is a *positional* array, not an object; field names/order
      come from the separate `fields` list. Parsed here by zipping the
      two rather than assuming fixed dict keys, since nothing guarantees
      the position order is stable across ERCOT report versions.
    * DAM's hour field (`hourEnding`) is a string like `"01:00"`, not a
      bare int — parsed via `int(x.split(":")[0])`.
    * RTM has no `hourEnding` at all: it splits into separate integer
      `deliveryHour` and `deliveryInterval` fields instead.
    * `DSTFlag` is a real JSON boolean (`true`/`false`), not the `"Y"`/`"N"`
      strings `gridstatus`'s CSV-report code path uses — this one would
      have been a silent bug, not a crash: `dst_flag == "Y"` is simply
      always False against a real bool, so the repeated-hour tie-break
      would never have fired.

The token *request format* was separately fixed after a live 400 Bad
Request (also 2026-09-07): credentials go as URL query parameters, not a
POST body, and the Bearer token is `access_token`, not `id_token` — see
get_token()'s docstring.

The one load-bearing, non-obvious fact this surfaced: ERCOT reports hours
in "Hour Ending" form (the label marks the END of the hour, not the
start) and publishes DSTFlag to disambiguate the repeated hour on the US
autumn clock-change day — GB's Elexon data never needed anything like
this, because settlement_period there is already a real elapsed-time
position, not a wall-clock label that can repeat. Handled here by
normalising each record to an (hour, interval, is_dst_repeat) tuple,
sorting each day's records into true chronological order on that tuple,
and assigning settlement_period by *position* in that sorted order — the
same "position-in-sequence, not label arithmetic" approach
schema.full_grid() already uses — rather than computing an elapsed-hours
offset directly from the hour label, which would get the repeated hour
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
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable
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


def _build_day_frames_isolating_failures(
    records_by_date: dict[str, list[dict]], build_one_day: Callable[[list[dict], date], pd.DataFrame], source: str
) -> list[pd.DataFrame]:
    """Build one frame per date in records_by_date via build_one_day(),
    warning and skipping any single date that fails (e.g. a genuine
    upstream data anomaly like an extra, duplicate-looking record for one
    day) rather than letting one bad day discard every other — otherwise-
    good — day in the same multi-day request, matching pipeline.py's own
    "cache the good, flag the bad" discipline (ADR-008) for the day-by-day
    fetch path."""
    frames = []
    for delivery_date, day_records in sorted(records_by_date.items()):
        d = date.fromisoformat(delivery_date)
        try:
            frames.append(build_one_day(day_records, d))
        except (ValueError, AllZeroPriceSeriesError) as exc:
            warnings.warn(f"{source}: {d} failed to parse, skipped: {exc}")
    if not frames:
        raise AllZeroPriceSeriesError(f"{source}: every day in this range failed to parse")
    return frames


def _raise_if_all_zero(prices: pd.DataFrame, settlement_date: date, source: str) -> None:
    if (prices["price_per_kwh"] == 0).all():
        raise AllZeroPriceSeriesError(f"{source}: all-zero price series for {settlement_date}")


def _rows_as_dicts(payload: dict) -> list[dict]:
    """ERCOT returns {"fields": [{"name": ...}, ...], "data": [[...], ...]}
    — each data row is positional, matching `fields` order. Zipped here
    rather than assumed fixed, since nothing guarantees that order is
    stable across report versions."""
    field_names = [f["name"] for f in payload["fields"]]
    return [dict(zip(field_names, row)) for row in payload["data"]]


def _dam_sort_key(record: dict) -> tuple:
    """Order one day's DAM records into true chronological order.
    hourEnding is Hour-Ending, marking the END of the hour, as a string
    like "01:00" — repeats once, at the same label, on the US autumn
    clock-change day. DSTFlag (a real bool) breaks that tie so the first
    real occurrence sorts before the repeated one."""
    hour_ending = int(record["hourEnding"].split(":")[0])
    return (hour_ending, bool(record.get("DSTFlag", False)))


def _rtm_sort_key(record: dict) -> tuple:
    """Same ordering as _dam_sort_key, but RTM has no hourEnding field —
    it splits into separate deliveryHour and deliveryInterval (1-4,
    within the hour) integer fields instead."""
    return (int(record["deliveryHour"]), int(record["deliveryInterval"]), bool(record.get("DSTFlag", False)))


def _build_dam_frame(records: list[dict], settlement_date: date) -> pd.DataFrame:
    """Shared by fetch_dam_prices() and fetch_dam_prices_range(): turn one
    day's already-fetched, already-hub-filtered records into a canonical
    schema frame."""
    start_utc, _ = settlement_day_utc_bounds(settlement_date, tz=CHICAGO)
    records = sorted(records, key=_dam_sort_key)
    n = len(records)
    prices = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime([start_utc + timedelta(hours=i) for i in range(n)], utc=True),
            "settlement_date": pd.Series([pd.Timestamp(settlement_date)] * n, dtype="datetime64[ns]"),
            "settlement_period": np.arange(1, n + 1, dtype="int64"),
            "period_minutes": np.full(n, 60, dtype="int64"),
            "price_per_kwh": np.array([r["settlementPointPrice"] for r in records], dtype="float64") / 1000,
            "currency": "USD",
            "source": SOURCE_DAM,
        }
    )
    prices = validate(prices, tz=CHICAGO)
    _raise_if_all_zero(prices, settlement_date, SOURCE_DAM)
    return prices


def fetch_dam_prices(
    settlement_date: date, token: ErcotToken, hub: str = HUB, session: requests.Session = requests
) -> pd.DataFrame:
    """
    Fetch one day of ERCOT Day-Ahead Market settlement point prices for
    `hub` (hourly, USD).
    """
    response = session.get(
        BASE_URL + DAM_PRODUCT_PATH,
        headers=_auth_headers(token),
        params={
            "deliveryDateFrom": settlement_date.isoformat(),
            "deliveryDateTo": settlement_date.isoformat(),
            "settlementPoint": hub,
        },
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    records = _rows_as_dicts(response.json())
    _raise_if_empty(records, settlement_date, SOURCE_DAM)
    return _build_dam_frame(records, settlement_date)


def fetch_dam_prices_range(
    start_date: date, end_date: date, token: ErcotToken, hub: str = HUB, session: requests.Session = requests
) -> pd.DataFrame:
    """
    Fetch multiple days of ERCOT DAM prices for `hub` in a single request,
    rather than one request per day — exploits the same two confirmed-live
    facts fetch_dam_prices() does (settlementPoint filters server-side,
    deliveryDateFrom/deliveryDateTo accept a genuine multi-day range), just
    without collapsing the range down to one day.

    Raises ValueError if the range would come back paginated (ERCOT pages
    at 1000 records — roughly 41 days of one hub's hourly DAM prices):
    this function deliberately does not follow pagination itself, so a
    caller backfilling a long history should chunk their own calls (e.g.
    one call per ~30-day window) rather than risk silently missing pages.

    A single day within the range that fails to parse (e.g. a genuine
    upstream anomaly — one real day came back with 97 fifteen-minute RTM
    records instead of 96, tripping schema.validate()'s period-count
    check) is warned about and skipped, not allowed to discard every
    other, otherwise-good day in the same request.
    """
    response = session.get(
        BASE_URL + DAM_PRODUCT_PATH,
        headers=_auth_headers(token),
        params={
            "deliveryDateFrom": start_date.isoformat(),
            "deliveryDateTo": end_date.isoformat(),
            "settlementPoint": hub,
        },
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()
    total_pages = payload.get("_meta", {}).get("totalPages", 1)
    if total_pages > 1:
        raise ValueError(
            f"{SOURCE_DAM}: {start_date} to {end_date} came back paginated ({total_pages} pages) — "
            "request a shorter range per call instead of relying on this function to paginate"
        )
    records = _rows_as_dicts(payload)
    _raise_if_empty(records, start_date, SOURCE_DAM)

    records_by_date: dict[str, list[dict]] = {}
    for r in records:
        records_by_date.setdefault(r["deliveryDate"], []).append(r)

    frames = _build_day_frames_isolating_failures(records_by_date, _build_dam_frame, SOURCE_DAM)
    return pd.concat(frames, ignore_index=True)


def _build_rtm_frame(records: list[dict], settlement_date: date) -> pd.DataFrame:
    """Shared by fetch_rtm_prices() and fetch_rtm_prices_range(): turn one
    day's already-fetched, already-hub-filtered records into a canonical
    schema frame."""
    start_utc, _ = settlement_day_utc_bounds(settlement_date, tz=CHICAGO)
    records = sorted(records, key=_rtm_sort_key)
    n = len(records)
    prices = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime([start_utc + timedelta(minutes=15 * i) for i in range(n)], utc=True),
            "settlement_date": pd.Series([pd.Timestamp(settlement_date)] * n, dtype="datetime64[ns]"),
            "settlement_period": np.arange(1, n + 1, dtype="int64"),
            "period_minutes": np.full(n, 15, dtype="int64"),
            "price_per_kwh": np.array([r["settlementPointPrice"] for r in records], dtype="float64") / 1000,
            "currency": "USD",
            "source": SOURCE_RTM,
        }
    )
    prices = validate(prices, tz=CHICAGO)
    _raise_if_all_zero(prices, settlement_date, SOURCE_RTM)
    return prices


def fetch_rtm_prices(
    settlement_date: date, token: ErcotToken, hub: str = HUB, session: requests.Session = requests
) -> pd.DataFrame:
    """
    Fetch one day of ERCOT Real-Time Market settlement point prices for
    `hub` (15-minute, USD). The DST tie-break described in _rtm_sort_key()
    matters more here than for DAM — a 15-minute grid has 4x as many
    periods where the ordering must be right.
    """
    response = session.get(
        BASE_URL + RTM_PRODUCT_PATH,
        headers=_auth_headers(token),
        params={
            "deliveryDateFrom": settlement_date.isoformat(),
            "deliveryDateTo": settlement_date.isoformat(),
            "settlementPoint": hub,
        },
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    records = _rows_as_dicts(response.json())
    _raise_if_empty(records, settlement_date, SOURCE_RTM)
    return _build_rtm_frame(records, settlement_date)


def fetch_rtm_prices_range(
    start_date: date, end_date: date, token: ErcotToken, hub: str = HUB, session: requests.Session = requests
) -> pd.DataFrame:
    """
    Fetch multiple days of ERCOT RTM prices for `hub` in a single request —
    see fetch_dam_prices_range()'s docstring for the reasoning, identical
    here except RTM's 15-minute grid means ERCOT's 1000-record page limit
    is reached far sooner: ~10 days of one hub, not ~41. Raises rather than
    silently paginating if a request comes back split across more than one
    page, same as fetch_dam_prices_range().
    """
    response = session.get(
        BASE_URL + RTM_PRODUCT_PATH,
        headers=_auth_headers(token),
        params={
            "deliveryDateFrom": start_date.isoformat(),
            "deliveryDateTo": end_date.isoformat(),
            "settlementPoint": hub,
        },
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()
    total_pages = payload.get("_meta", {}).get("totalPages", 1)
    if total_pages > 1:
        raise ValueError(
            f"{SOURCE_RTM}: {start_date} to {end_date} came back paginated ({total_pages} pages) — "
            "request a shorter range per call instead of relying on this function to paginate"
        )
    records = _rows_as_dicts(payload)
    _raise_if_empty(records, start_date, SOURCE_RTM)

    records_by_date: dict[str, list[dict]] = {}
    for r in records:
        records_by_date.setdefault(r["deliveryDate"], []).append(r)

    frames = _build_day_frames_isolating_failures(records_by_date, _build_rtm_frame, SOURCE_RTM)
    return pd.concat(frames, ignore_index=True)
