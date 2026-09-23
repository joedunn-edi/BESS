"""
test_sources_ercot_wind.py — tests for bess/sources_ercot_wind.py.

Fixtures shaped to match the real NP4-742-CD response, confirmed live on
2026-09-23 — structurally identical to solar's NP4-745-CD (see
sources_ercot_wind.py's module docstring): positional {"fields": [...],
"data": [[...], ...]} envelope, hourEnding as a plain integer, DSTFlag as
a real boolean, and the same rolling-repost pattern (confirmed: an
identical query returned exactly 576 rows for one day, matching solar).
"""

from datetime import date

import pytest

from bess.sources_ercot import ErcotToken
from bess.sources_ercot_wind import SOURCE_WIND, fetch_wind_generation

FIELDS = [
    "deliveryDate",
    "hourEnding",
    "genWest",
    "genSystemWide",
    "STWPFWest",
    "WGRPPWest",
    "DSTFlag",
]


def _payload(records: list[dict]) -> dict:
    return {
        "fields": [{"name": f} for f in FIELDS],
        "data": [[r[f] for f in FIELDS] for r in records],
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
        self.captured_params = None

    def get(self, url, headers=None, params=None, timeout=None):
        self.captured_params = params
        return _FakeResponse(self._payload)


def _record(hour: int, gen: float, delivery_date: str = "2026-08-24", dst: bool = False) -> dict:
    return {
        "deliveryDate": delivery_date,
        "hourEnding": hour,
        "genWest": gen,
        "genSystemWide": gen * 10,
        "STWPFWest": gen + 1.0,
        "WGRPPWest": gen + 2.0,
        "DSTFlag": dst,
    }


def _token() -> ErcotToken:
    return ErcotToken(access_token="fake", subscription_key="fake")


def test_fetch_wind_generation_parses_a_clean_24_hour_day():
    records = [_record(h, float(h)) for h in range(1, 25)]
    session = _FakeSession(_payload(records))
    df = fetch_wind_generation(date(2026, 8, 24), token=_token(), session=session)

    assert len(df) == 24
    assert list(df["hour_ending"]) == list(range(1, 25))
    assert df["wind_gen_west_mw"].iloc[0] == pytest.approx(1.0)
    assert df["wind_gen_systemwide_mw"].iloc[0] == pytest.approx(10.0)
    assert df["wind_stwpf_west_mw"].iloc[0] == pytest.approx(2.0)
    assert df["wind_wgrpp_west_mw"].iloc[0] == pytest.approx(3.0)


def test_fetch_wind_generation_deduplicates_many_reposts_of_the_same_hour():
    records = []
    for _ in range(24):
        records += [_record(h, float(h)) for h in range(1, 25)]
    assert len(records) == 576

    session = _FakeSession(_payload(records))
    df = fetch_wind_generation(date(2026, 8, 24), token=_token(), session=session)

    assert len(df) == 24
    assert list(df["hour_ending"]) == list(range(1, 25))


def test_fetch_wind_generation_sends_the_expected_posted_datetime_window():
    records = [_record(h, float(h)) for h in range(1, 25)]
    session = _FakeSession(_payload(records))
    fetch_wind_generation(date(2026, 8, 24), token=_token(), session=session)

    assert session.captured_params["deliveryDateFrom"] == "2026-08-24"
    assert session.captured_params["deliveryDateTo"] == "2026-08-24"
    assert session.captured_params["postedDatetimeFrom"] == "2026-08-25"
    assert session.captured_params["postedDatetimeTo"] == "2026-08-26"


def test_fetch_wind_generation_handles_the_repeated_fall_back_hour():
    records = [_record(h, float(h), delivery_date="2026-11-01") for h in range(1, 3)]
    records.append(_record(2, 0.5, delivery_date="2026-11-01", dst=True))
    records += [_record(h, float(h), delivery_date="2026-11-01") for h in range(3, 25)]

    session = _FakeSession(_payload(records))
    df = fetch_wind_generation(date(2026, 11, 1), token=_token(), session=session)

    assert len(df) == 25


def test_fetch_wind_generation_raises_on_empty_response():
    session = _FakeSession(_payload([]))
    with pytest.raises(ValueError, match="no records returned"):
        fetch_wind_generation(date(2026, 8, 24), token=_token(), session=session)


def test_fetch_wind_generation_raises_when_response_is_paginated():
    payload = _payload([_record(h, float(h)) for h in range(1, 25)])
    payload["_meta"] = {"totalPages": 2}
    session = _FakeSession(payload)

    with pytest.raises(ValueError, match="paginated"):
        fetch_wind_generation(date(2026, 8, 24), token=_token(), session=session)


def test_source_wind_constant_is_stable():
    assert SOURCE_WIND == "ercot_wind_west"


def test_fetch_wind_generation_turns_a_null_value_into_nan_not_a_crash():
    import math

    records = [_record(h, float(h)) for h in range(1, 25)]
    records[9]["genWest"] = None  # hour_ending 10

    session = _FakeSession(_payload(records))
    df = fetch_wind_generation(date(2026, 8, 24), token=_token(), session=session)

    assert len(df) == 24
    null_row = df[df["hour_ending"] == 10].iloc[0]
    assert math.isnan(null_row["wind_gen_west_mw"])
    assert null_row["wind_gen_systemwide_mw"] == pytest.approx(100.0)
