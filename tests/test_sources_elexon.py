"""
test_sources_elexon.py — tests for bess/sources_elexon.py.

Uses real recorded Elexon API responses (tests/fixtures/) captured against
the live API for 2026-07-15, via a fake session, rather than hitting the
network in CI. Covers the price-field parsing, the SSP/SBP mismatch guard,
the empty/all-zero guard, and the day-ahead boundary-period filtering.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from bess.sources_elexon import (
    SOURCE_DAY_AHEAD,
    SOURCE_IMBALANCE,
    AllZeroPriceSeriesError,
    fetch_day_ahead_prices,
    fetch_imbalance_prices,
)

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    """Returns a fixed payload regardless of request args — one endpoint per fixture per test."""

    def __init__(self, payload):
        self._payload = payload

    def get(self, *args, **kwargs):
        return _FakeResponse(self._payload)


def _load_fixture(name):
    with open(FIXTURES / name) as f:
        return json.load(f)


def _deep_copy(payload):
    return json.loads(json.dumps(payload))


# --- fetch_imbalance_prices --------------------------------------------------


def test_fetch_imbalance_prices_parses_real_fixture():
    payload = _load_fixture("elexon_system_prices_2026-07-15.json")
    result = fetch_imbalance_prices(date(2026, 7, 15), session=_FakeSession(payload))

    assert len(result.prices) == 48
    assert (result.prices["source"] == SOURCE_IMBALANCE).all()
    assert list(result.prices["settlement_period"]) == list(range(1, 49))
    first_raw = payload["data"][0]
    assert result.prices.loc[0, "price_gbp_per_kwh"] == pytest.approx(first_raw["systemSellPrice"] / 1000)


def test_fetch_imbalance_prices_quality_flags_align_with_prices():
    payload = _load_fixture("elexon_system_prices_2026-07-15.json")
    result = fetch_imbalance_prices(date(2026, 7, 15), session=_FakeSession(payload))

    assert len(result.quality_flags) == len(result.prices)
    assert list(result.quality_flags.columns) == [
        "settlement_date",
        "settlement_period",
        "bsad_defaulted",
        "price_derivation_code",
    ]


def test_fetch_imbalance_prices_raises_on_ssp_sbp_mismatch():
    payload = _deep_copy(_load_fixture("elexon_system_prices_2026-07-15.json"))
    payload["data"][0]["systemBuyPrice"] += 1.0
    with pytest.raises(ValueError, match="systemSellPrice != systemBuyPrice"):
        fetch_imbalance_prices(date(2026, 7, 15), session=_FakeSession(payload))


def test_fetch_imbalance_prices_raises_on_empty_response():
    with pytest.raises(AllZeroPriceSeriesError, match="no records returned"):
        fetch_imbalance_prices(date(2026, 7, 15), session=_FakeSession({"data": []}))


def test_fetch_imbalance_prices_raises_on_all_zero():
    payload = _deep_copy(_load_fixture("elexon_system_prices_2026-07-15.json"))
    for row in payload["data"]:
        row["systemSellPrice"] = 0.0
        row["systemBuyPrice"] = 0.0
    with pytest.raises(AllZeroPriceSeriesError, match="all-zero"):
        fetch_imbalance_prices(date(2026, 7, 15), session=_FakeSession(payload))


# --- fetch_day_ahead_prices ---------------------------------------------------


def test_fetch_day_ahead_prices_parses_real_fixture_and_filters_boundary():
    payload = _load_fixture("elexon_market_index_2026-07-15.json")
    assert len(payload["data"]) == 49  # raw fixture includes one boundary period from the next day

    prices = fetch_day_ahead_prices(date(2026, 7, 15), session=_FakeSession(payload))

    assert len(prices) == 48
    assert (prices["source"] == SOURCE_DAY_AHEAD).all()
    assert (prices["settlement_date"] == pd.Timestamp("2026-07-15")).all()


def test_fetch_day_ahead_prices_raises_on_empty_response():
    with pytest.raises(AllZeroPriceSeriesError, match="no records returned"):
        fetch_day_ahead_prices(date(2026, 7, 15), session=_FakeSession({"data": []}))


def test_fetch_day_ahead_prices_raises_on_all_zero():
    payload = _deep_copy(_load_fixture("elexon_market_index_2026-07-15.json"))
    for row in payload["data"]:
        row["price"] = 0.0
    with pytest.raises(AllZeroPriceSeriesError, match="all-zero"):
        fetch_day_ahead_prices(date(2026, 7, 15), session=_FakeSession(payload))
