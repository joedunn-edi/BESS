"""
test_sources_ercot.py — tests for bess/sources_ercot.py.

Fixtures here are shaped to match real ERCOT API responses, confirmed
live against a real account on 2026-09-07 (see sources_ercot.py's module
docstring): the {"fields": [...], "data": [[...], ...]} positional
envelope, DAM's string "HH:00" hourEnding, RTM's separate integer
deliveryHour/deliveryInterval fields, and DSTFlag as a real JSON boolean.
"""

from datetime import date

import pytest
import requests

import bess.sources_ercot as sources_ercot
from bess.sources_ercot import (
    HUB,
    SOURCE_DAM,
    SOURCE_RTM,
    AllZeroPriceSeriesError,
    ErcotToken,
    _dam_sort_key,
    _rtm_sort_key,
    fetch_dam_prices,
    fetch_rtm_prices,
    get_token,
)

DAM_FIELDS = ["deliveryDate", "hourEnding", "settlementPoint", "settlementPointPrice", "DSTFlag"]
RTM_FIELDS = [
    "deliveryDate",
    "deliveryHour",
    "deliveryInterval",
    "settlementPoint",
    "settlementPointType",
    "settlementPointPrice",
    "DSTFlag",
]


def _payload(records: list[dict], field_order: list[str]) -> dict:
    """Build ERCOT's real positional-array envelope from a list of
    field-named dicts, the shape confirmed against a live response."""
    return {
        "fields": [{"name": f} for f in field_order],
        "data": [[r[f] for f in field_order] for r in records],
    }


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self._payload = payload

    def get(self, *args, **kwargs):
        return _FakeResponse(self._payload)


class _CapturingSession:
    """Records the params it was called with, so tests can confirm the
    server-side settlementPoint filter and single-day date range are
    actually sent — the mechanism the real fetch now relies on entirely,
    since ERCOT paginates at 1000 records/page across every settlement
    point when unfiltered."""

    def __init__(self, payload):
        self._payload = payload
        self.captured_params = None

    def get(self, *args, **kwargs):
        self.captured_params = kwargs.get("params")
        return _FakeResponse(self._payload)


def _token() -> ErcotToken:
    return ErcotToken(access_token="fake", subscription_key="fake")


def _dam_record(hour_ending: int, price: float, hub: str = HUB, delivery_date: str = "2026-07-15", dst: bool = False) -> dict:
    return {
        "deliveryDate": delivery_date,
        "hourEnding": f"{hour_ending:02d}:00",
        "settlementPoint": hub,
        "settlementPointPrice": price,
        "DSTFlag": dst,
    }


def _rtm_record(
    hour: int, interval: int, price: float, hub: str = HUB, delivery_date: str = "2026-07-15", dst: bool = False
) -> dict:
    return {
        "deliveryDate": delivery_date,
        "deliveryHour": hour,
        "deliveryInterval": interval,
        "settlementPoint": hub,
        "settlementPointType": "HU",
        "settlementPointPrice": price,
        "DSTFlag": dst,
    }


# --- _dam_sort_key / _rtm_sort_key (pure logic, no fixture needed) ------------------


def test_dam_sort_key_orders_repeated_hour_before_the_dst_flagged_one():
    first_pass = _dam_record(2, 0.10, dst=False)
    second_pass = _dam_record(2, 0.05, dst=True)  # the repeated hour, sorts after
    later_hour = _dam_record(3, 0.20)

    ordered = sorted([later_hour, second_pass, first_pass], key=_dam_sort_key)

    assert ordered == [first_pass, second_pass, later_hour]


def test_rtm_sort_key_orders_by_hour_then_interval_then_dst_flag():
    first_pass = _rtm_record(2, 4, 0.10, dst=False)
    second_pass = _rtm_record(2, 4, 0.05, dst=True)  # the repeated hour's last interval, sorts after
    next_hour = _rtm_record(3, 1, 0.20)

    ordered = sorted([next_hour, second_pass, first_pass], key=_rtm_sort_key)

    assert ordered == [first_pass, second_pass, next_hour]


# --- fetch_dam_prices (against a fixture shaped like the real response) ------------


def test_fetch_dam_prices_parses_a_normal_24_hour_day():
    records = [_dam_record(h, 0.10 + h * 0.001) for h in range(1, 25)]
    session = _FakeSession(_payload(records, DAM_FIELDS))
    prices = fetch_dam_prices(date(2026, 7, 15), token=_token(), session=session)

    assert len(prices) == 24
    assert (prices["source"] == SOURCE_DAM).all()
    assert (prices["currency"] == "USD").all()
    assert (prices["period_minutes"] == 60).all()
    assert list(prices["settlement_period"]) == list(range(1, 25))
    assert prices["price_per_kwh"].iloc[0] == pytest.approx((0.10 + 1 * 0.001) / 1000)


def test_fetch_dam_prices_sends_the_settlement_point_filter_and_a_single_day_range():
    records = [_dam_record(h, 0.10) for h in range(1, 25)]
    session = _CapturingSession(_payload(records, DAM_FIELDS))
    fetch_dam_prices(date(2026, 7, 15), token=_token(), session=session)

    # confirmed live: settlementPoint filters server-side, and
    # deliveryDateFrom/deliveryDateTo are both inclusive, so a single
    # day's fetch uses the same date for both
    assert session.captured_params["settlementPoint"] == HUB
    assert session.captured_params["deliveryDateFrom"] == "2026-07-15"
    assert session.captured_params["deliveryDateTo"] == "2026-07-15"


def test_fetch_dam_prices_handles_the_repeated_fall_back_hour():
    # 2026-11-01 is the US autumn clock-change day: 25 hours, with
    # hourEnding="02:00" appearing twice (DSTFlag False then True)
    records = [_dam_record(h, 0.10, delivery_date="2026-11-01") for h in range(1, 3)]
    records.append(_dam_record(2, 0.09, delivery_date="2026-11-01", dst=True))  # the repeated hour
    records += [_dam_record(h, 0.10, delivery_date="2026-11-01") for h in range(3, 25)]

    session = _FakeSession(_payload(records, DAM_FIELDS))
    prices = fetch_dam_prices(date(2026, 11, 1), token=_token(), session=session)

    assert len(prices) == 25
    assert list(prices["settlement_period"]) == list(range(1, 26))


def test_fetch_dam_prices_handles_the_spring_forward_short_day():
    # 2026-03-08 is the US spring-forward day: hour 3 is skipped
    # entirely, so only 23 hourEnding labels appear, not 24
    records = [_dam_record(h, 0.10, delivery_date="2026-03-08") for h in range(1, 25) if h != 3]

    session = _FakeSession(_payload(records, DAM_FIELDS))
    prices = fetch_dam_prices(date(2026, 3, 8), token=_token(), session=session)

    assert len(prices) == 23
    assert list(prices["settlement_period"]) == list(range(1, 24))


def test_fetch_dam_prices_allows_negative_prices():
    # ERCOT clears well below zero during renewable oversupply — more
    # extreme than GB ever sees — must not be rejected or miscounted as
    # the all-zero guard
    prices_list = [-50.0] * 4 + [10.0] * 20
    records = [_dam_record(h, p) for h, p in zip(range(1, 25), prices_list)]

    session = _FakeSession(_payload(records, DAM_FIELDS))
    prices = fetch_dam_prices(date(2026, 7, 15), token=_token(), session=session)

    assert (prices["price_per_kwh"].iloc[:4] < 0).all()
    assert len(prices) == 24


def test_rows_as_dicts_is_independent_of_field_order():
    # nothing in ERCOT's contract guarantees `fields` stays in this exact
    # order across report versions — the parser must key off `name`, not
    # position, to survive a reorder
    shuffled_payload = {
        "fields": [{"name": "settlementPointPrice"}, {"name": "DSTFlag"}, {"name": "hourEnding"}, {"name": "deliveryDate"}],
        "data": [[23.5, False, "01:00", "2026-07-15"]],
    }
    rows = sources_ercot._rows_as_dicts(shuffled_payload)

    assert rows == [
        {"settlementPointPrice": 23.5, "DSTFlag": False, "hourEnding": "01:00", "deliveryDate": "2026-07-15"}
    ]


def test_fetch_dam_prices_raises_on_empty_response():
    session = _FakeSession(_payload([], DAM_FIELDS))
    with pytest.raises(AllZeroPriceSeriesError, match="no records returned"):
        fetch_dam_prices(date(2026, 7, 15), token=_token(), session=session)


def test_fetch_dam_prices_raises_on_all_zero():
    records = [_dam_record(h, 0.0) for h in range(1, 25)]
    session = _FakeSession(_payload(records, DAM_FIELDS))
    with pytest.raises(AllZeroPriceSeriesError, match="all-zero"):
        fetch_dam_prices(date(2026, 7, 15), token=_token(), session=session)


# --- fetch_rtm_prices ----------------------------------------------------------------


def test_fetch_rtm_prices_parses_a_normal_96_period_day():
    records = [_rtm_record(h, i, 0.10) for h in range(1, 25) for i in range(1, 5)]
    session = _FakeSession(_payload(records, RTM_FIELDS))
    prices = fetch_rtm_prices(date(2026, 7, 15), token=_token(), session=session)

    assert len(prices) == 96
    assert (prices["source"] == SOURCE_RTM).all()
    assert (prices["period_minutes"] == 15).all()
    assert list(prices["settlement_period"]) == list(range(1, 97))


def test_fetch_rtm_prices_handles_the_spring_forward_short_day():
    # same 2026-03-08 skip as DAM, but 4 intervals wide: hour 3 never
    # appears at all, so 92 records instead of 96
    records = [_rtm_record(h, i, 0.10, delivery_date="2026-03-08") for h in range(1, 25) if h != 3 for i in range(1, 5)]

    session = _FakeSession(_payload(records, RTM_FIELDS))
    prices = fetch_rtm_prices(date(2026, 3, 8), token=_token(), session=session)

    assert len(prices) == 92
    assert list(prices["settlement_period"]) == list(range(1, 93))


def test_fetch_rtm_prices_sends_the_settlement_point_filter_and_a_single_day_range():
    records = [_rtm_record(h, i, 0.10) for h in range(1, 25) for i in range(1, 5)]
    session = _CapturingSession(_payload(records, RTM_FIELDS))
    fetch_rtm_prices(date(2026, 7, 15), token=_token(), session=session)

    assert session.captured_params["settlementPoint"] == HUB
    assert session.captured_params["deliveryDateFrom"] == "2026-07-15"
    assert session.captured_params["deliveryDateTo"] == "2026-07-15"


def test_fetch_rtm_prices_raises_on_empty_response():
    session = _FakeSession(_payload([], RTM_FIELDS))
    with pytest.raises(AllZeroPriceSeriesError, match="no records returned"):
        fetch_rtm_prices(date(2026, 7, 15), token=_token(), session=session)


# --- ErcotToken ------------------------------------------------------------------------


def test_token_not_expired_when_fresh():
    assert not _token().expired


def test_token_expired_when_old():
    old_token = ErcotToken(access_token="fake", subscription_key="fake", obtained_at=0.0)
    assert old_token.expired


# --- get_token() — scope encoding and credential-safe error handling --------------


class _FakePostResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.exceptions.HTTPError(f"{self.status_code} Client Error: Bad Request for url: FAKE-URL")
            err.response = self
            raise err

    def json(self):
        return self._payload


def test_get_token_appends_scope_unescaped_not_via_params(monkeypatch):
    captured = {}

    def fake_post(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _FakePostResponse(200, {"access_token": "tok"})

    monkeypatch.setattr(sources_ercot.requests, "post", fake_post)

    get_token(username="user@example.com", password="pw", subscription_key="sub")

    # the literal '+' separators in SCOPE must survive unescaped in the URL
    # itself — they must NOT be passed through params (which would get them
    # percent-encoded to %2B, silently changing their meaning for ERCOT's server)
    assert f"scope={sources_ercot.SCOPE}" in captured["url"]
    assert "scope" not in captured["params"]
    assert captured["params"]["username"] == "user@example.com"


def test_get_token_uses_access_token_field(monkeypatch):
    def fake_post(url, params=None, timeout=None):
        return _FakePostResponse(200, {"access_token": "the-real-token", "id_token": "not-this-one"})

    monkeypatch.setattr(sources_ercot.requests, "post", fake_post)

    token = get_token(username="u", password="p", subscription_key="s")

    assert token.access_token == "the-real-token"


def test_get_token_failure_never_leaks_url_or_credentials(monkeypatch):
    def fake_post(url, params=None, timeout=None):
        return _FakePostResponse(400, {})

    monkeypatch.setattr(sources_ercot.requests, "post", fake_post)

    real_password = "super-secret-password-do-not-leak"
    with pytest.raises(requests.exceptions.HTTPError) as exc_info:
        get_token(username="user@example.com", password=real_password, subscription_key="sub")

    message = str(exc_info.value)
    assert real_password not in message
    assert "FAKE-URL" not in message  # the underlying HTTPError's own message must not survive
    assert "400" in message  # the status code itself is fine to keep — it's not sensitive
