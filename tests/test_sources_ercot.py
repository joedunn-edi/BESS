"""
test_sources_ercot.py — tests for bess/sources_ercot.py.

Unlike test_sources_elexon.py, these fixtures are NOT recorded real API
responses — ERCOT requires a registered account we don't have, so these
are synthetic payloads shaped according to our best understanding (see
sources_ercot.py's module docstring for what's confirmed vs still
unverified). These tests prove the *logic* (DST tie-breaking, hub
filtering, guards, schema construction) is correct given data shaped the
way we assume — they cannot prove the assumption itself is correct. That
still needs a live call once a real account exists.
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
    _dst_sort_key,
    _hub_filter,
    fetch_dam_prices,
    fetch_rtm_prices,
    get_token,
)


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


def _token() -> ErcotToken:
    return ErcotToken(access_token="fake", subscription_key="fake")


def _dam_record(hour_ending: int, price: float, hub: str = HUB, delivery_date: str = "2026-07-15", dst_flag: str = "N"):
    return {
        "deliveryDate": delivery_date,
        "hourEnding": hour_ending,
        "settlementPoint": hub,
        "settlementPointPrice": price,
        "DSTFlag": dst_flag,
    }


def _rtm_record(hour_ending: int, interval: int, price: float, hub: str = HUB, delivery_date: str = "2026-07-15", dst_flag: str = "N"):
    return {
        "deliveryDate": delivery_date,
        "hourEnding": hour_ending,
        "deliveryInterval": interval,
        "settlementPoint": hub,
        "settlementPointPrice": price,
        "DSTFlag": dst_flag,
    }


# --- _dst_sort_key / _hub_filter (pure logic, no fixture needed) --------------------


def test_dst_sort_key_orders_repeated_hour_with_n_before_y():
    first_pass = _dam_record(2, 0.10, dst_flag="N")
    second_pass = _dam_record(2, 0.05, dst_flag="Y")  # the repeated hour, sorts after
    later_hour = _dam_record(3, 0.20)

    ordered = sorted([later_hour, second_pass, first_pass], key=_dst_sort_key)

    assert ordered == [first_pass, second_pass, later_hour]


def test_hub_filter_keeps_only_the_requested_hub():
    records = [_dam_record(1, 0.10, hub="HB_WEST"), _dam_record(1, 0.20, hub="HB_HOUSTON")]
    filtered = _hub_filter(records, "HB_WEST")
    assert len(filtered) == 1
    assert filtered[0]["settlementPoint"] == "HB_WEST"


# --- fetch_dam_prices (against a synthetic, assumption-shaped fixture) --------------


def test_fetch_dam_prices_parses_a_normal_24_hour_day():
    records = [_dam_record(h, 0.10 + h * 0.001) for h in range(1, 25)]
    prices = fetch_dam_prices(date(2026, 7, 15), token=_token(), session=_FakeSession({"data": records}))

    assert len(prices) == 24
    assert (prices["source"] == SOURCE_DAM).all()
    assert (prices["currency"] == "USD").all()
    assert (prices["period_minutes"] == 60).all()
    assert list(prices["settlement_period"]) == list(range(1, 25))
    assert prices["price_per_kwh"].iloc[0] == pytest.approx((0.10 + 1 * 0.001) / 1000)


def test_fetch_dam_prices_filters_other_hubs_and_other_dates():
    records = [_dam_record(h, 0.10) for h in range(1, 25)]
    records += [_dam_record(h, 0.30, hub="HB_HOUSTON") for h in range(1, 25)]  # wrong hub
    records += [_dam_record(h, 0.50, delivery_date="2026-07-16") for h in range(1, 25)]  # wrong day

    prices = fetch_dam_prices(date(2026, 7, 15), token=_token(), session=_FakeSession({"data": records}))

    assert len(prices) == 24
    assert (prices["price_per_kwh"] < 0.02).all()  # only the 0.10-ish HB_WEST/correct-day rows survive


def test_fetch_dam_prices_handles_the_repeated_fall_back_hour():
    # 2026-11-01 is the US autumn clock-change day: 25 hours, with
    # hourEnding=2 appearing twice (DSTFlag N then Y)
    records = [_dam_record(h, 0.10, delivery_date="2026-11-01") for h in range(1, 3)]
    records.append(_dam_record(2, 0.09, delivery_date="2026-11-01", dst_flag="Y"))  # the repeated hour
    records += [_dam_record(h, 0.10, delivery_date="2026-11-01") for h in range(3, 25)]

    prices = fetch_dam_prices(date(2026, 11, 1), token=_token(), session=_FakeSession({"data": records}))

    assert len(prices) == 25
    assert list(prices["settlement_period"]) == list(range(1, 26))


def test_fetch_dam_prices_raises_on_empty_response():
    with pytest.raises(AllZeroPriceSeriesError, match="no records returned"):
        fetch_dam_prices(date(2026, 7, 15), token=_token(), session=_FakeSession({"data": []}))


def test_fetch_dam_prices_raises_on_all_zero():
    records = [_dam_record(h, 0.0) for h in range(1, 25)]
    with pytest.raises(AllZeroPriceSeriesError, match="all-zero"):
        fetch_dam_prices(date(2026, 7, 15), token=_token(), session=_FakeSession({"data": records}))


# --- fetch_rtm_prices ----------------------------------------------------------------


def test_fetch_rtm_prices_parses_a_normal_96_period_day():
    records = [_rtm_record(h, i, 0.10) for h in range(1, 25) for i in range(1, 5)]
    prices = fetch_rtm_prices(date(2026, 7, 15), token=_token(), session=_FakeSession({"data": records}))

    assert len(prices) == 96
    assert (prices["source"] == SOURCE_RTM).all()
    assert (prices["period_minutes"] == 15).all()
    assert list(prices["settlement_period"]) == list(range(1, 97))


def test_fetch_rtm_prices_raises_on_empty_response():
    with pytest.raises(AllZeroPriceSeriesError, match="no records returned"):
        fetch_rtm_prices(date(2026, 7, 15), token=_token(), session=_FakeSession({"data": []}))


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
