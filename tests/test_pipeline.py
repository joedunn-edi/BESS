"""
test_pipeline.py — tests for bess/pipeline.py.

Uses small synthetic fetch_one_day() callables (not the real Elexon
fetchers — pipeline.py is fetcher-agnostic) to exercise: gap accounting
against the DST-aware grid, the "cache first, raise after" ordering
(ADR-008), the per-day-fraction and max-consecutive-missing thresholds,
failed-fetch handling, and cache-revision merging.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from bess.pipeline import (
    DataQualityReport,
    GapThresholdExceededError,
    run_ercot_dam_pipeline,
    run_ercot_rtm_pipeline,
    run_pipeline,
)
from bess.schema import expected_period_count, settlement_day_utc_bounds, validate
from bess.sources_ercot import ErcotToken


def _day_frame(
    d: date, periods: list[int] | None = None, price: float | list[float] = 0.10, source: str = "test"
) -> pd.DataFrame:
    """A minimal valid canonical frame for one day, with only `periods` present."""
    if periods is None:
        periods = list(range(1, expected_period_count(d) + 1))
    start_utc, _ = settlement_day_utc_bounds(d)
    timestamps = [start_utc + timedelta(minutes=30 * (p - 1)) for p in periods]
    prices = np.array(price, dtype="float64") if isinstance(price, list) else np.full(len(periods), price, dtype="float64")
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(timestamps, utc=True),
            "settlement_date": pd.Series([pd.Timestamp(d)] * len(periods), dtype="datetime64[ns]"),
            "settlement_period": np.array(periods, dtype="int64"),
            "period_minutes": np.full(len(periods), 30, dtype="int64"),
            "price_per_kwh": prices,
            "currency": "GBP",
            "source": source,
        }
    )
    return validate(df)


# --- no gaps ------------------------------------------------------------------


def test_run_pipeline_no_gaps_caches_and_reports_zero_missing(tmp_path):
    days = [date(2026, 7, 10), date(2026, 7, 11), date(2026, 7, 12)]

    def fetch_one_day(d):
        return _day_frame(d)

    combined, report = run_pipeline(
        fetch_one_day, days[0], days[-1], source="test", cache_path=tmp_path / "cache.parquet"
    )

    assert report.n_missing_periods == 0
    assert report.max_consecutive_missing == 0
    assert len(combined) == 48 * 3
    assert (tmp_path / "cache.parquet").exists()
    assert len(pd.read_parquet(tmp_path / "cache.parquet")) == 48 * 3


# --- cache-before-raise (ADR-008) ---------------------------------------------


def test_raises_when_a_day_is_gappy_but_still_caches_the_good_days(tmp_path):
    good_day = date(2026, 7, 10)
    bad_day = date(2026, 7, 11)

    def fetch_one_day(d):
        if d == bad_day:
            return _day_frame(d, periods=[1, 2, 3])  # most of the day missing
        return _day_frame(d)

    cache_path = tmp_path / "cache.parquet"
    with pytest.raises(GapThresholdExceededError, match="missing-fraction threshold"):
        run_pipeline(fetch_one_day, good_day, bad_day, source="test", cache_path=cache_path)

    # the good day must still have been cached despite the raise
    cached = pd.read_parquet(cache_path)
    assert len(cached[cached["settlement_date"] == pd.Timestamp(good_day)]) == 48
    assert len(cached[cached["settlement_date"] == pd.Timestamp(bad_day)]) == 3


def test_no_raise_when_missing_fraction_within_threshold(tmp_path):
    d = date(2026, 7, 10)

    def fetch_one_day(_d):
        return _day_frame(_d, periods=list(range(1, 48)))  # missing period 48 only

    combined, report = run_pipeline(
        fetch_one_day,
        d,
        d,
        source="test",
        cache_path=tmp_path / "cache.parquet",
        max_missing_fraction_per_day=0.1,  # 1/48 ~= 2.1%, under 10%
        max_consecutive_missing=1,
    )
    assert report.n_missing_periods == 1
    assert len(combined) == 47


# --- failed fetches -------------------------------------------------------------


def test_failed_fetch_day_is_treated_as_fully_missing_not_a_crash(tmp_path):
    ok_day = date(2026, 7, 10)
    failing_day = date(2026, 7, 11)

    def fetch_one_day(d):
        if d == failing_day:
            raise ValueError("simulated network failure")
        return _day_frame(d)

    with pytest.raises(GapThresholdExceededError):
        run_pipeline(fetch_one_day, ok_day, failing_day, source="test", cache_path=tmp_path / "cache.parquet")


def test_report_records_failed_fetch_dates(tmp_path):
    ok_day = date(2026, 7, 10)
    failing_day = date(2026, 7, 11)

    def fetch_one_day(d):
        if d == failing_day:
            raise ValueError("simulated network failure")
        return _day_frame(d)

    _, report = run_pipeline(
        fetch_one_day,
        ok_day,
        failing_day,
        source="test",
        cache_path=tmp_path / "cache.parquet",
        max_missing_fraction_per_day=1.0,  # tolerate it so we can inspect the report
        max_consecutive_missing=48,
    )
    assert report.failed_fetch_dates == [failing_day]
    assert report.missing_fraction_by_day[failing_day] == 1.0
    assert report.missing_fraction_by_day[ok_day] == 0.0


# --- max consecutive missing ---------------------------------------------------


def test_max_consecutive_missing_spans_day_boundary(tmp_path):
    day1, day2 = date(2026, 7, 10), date(2026, 7, 11)

    def fetch_one_day(d):
        if d == day1:
            return _day_frame(d, periods=list(range(1, 47)))  # missing 47, 48
        return _day_frame(d, periods=list(range(3, 49)))  # missing 1, 2

    # missing run: day1 periods 47-48 + day2 periods 1-2 = 4 consecutive
    with pytest.raises(GapThresholdExceededError, match="consecutive"):
        run_pipeline(
            fetch_one_day,
            day1,
            day2,
            source="test",
            cache_path=tmp_path / "cache.parquet",
            max_missing_fraction_per_day=1.0,  # isolate the consecutive-run check
            max_consecutive_missing=3,
        )

    # exactly 4 is fine when the threshold is 4
    _, report = run_pipeline(
        fetch_one_day,
        day1,
        day2,
        source="test",
        cache_path=tmp_path / "cache2.parquet",
        max_missing_fraction_per_day=1.0,
        max_consecutive_missing=4,
    )
    assert report.max_consecutive_missing == 4


# --- negative prices and distribution ------------------------------------------


def test_negative_price_frequency_and_distribution(tmp_path):
    d = date(2026, 7, 10)
    prices = [-0.01] * 4 + [0.10] * 44  # 4 negative periods out of 48

    def fetch_one_day(_d):
        return _day_frame(_d, price=prices)

    _, report = run_pipeline(fetch_one_day, d, d, source="test", cache_path=tmp_path / "cache.parquet")

    assert report.n_negative_price_periods == 4
    assert report.negative_price_fraction == pytest.approx(4 / 48)
    assert "mean" in report.price_distribution
    assert "min" in report.price_distribution


def test_report_summary_runs_without_error(tmp_path):
    d = date(2026, 7, 10)
    _, report = run_pipeline(lambda _d: _day_frame(_d), d, d, source="test", cache_path=tmp_path / "cache.parquet")
    assert isinstance(report.summary(), str)
    assert "test" in report.summary()


# --- cache revision merging -----------------------------------------------------


def test_rerun_with_revised_prices_overwrites_cached_values(tmp_path):
    d = date(2026, 7, 10)
    cache_path = tmp_path / "cache.parquet"

    run_pipeline(lambda _d: _day_frame(_d, price=0.10), d, d, source="test", cache_path=cache_path)
    combined, _ = run_pipeline(lambda _d: _day_frame(_d, price=0.20), d, d, source="test", cache_path=cache_path)

    assert len(combined) == 48  # no duplicate rows after the re-fetch
    assert (combined["price_per_kwh"] == 0.20).all()  # freshest fetch wins


# --- ERCOT wrappers (fake session, since we have no registered account) -----------


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


def _ercot_payload(records: list[dict], field_order: list[str]) -> dict:
    """ERCOT's real response envelope: positional rows, field names/order
    given separately — confirmed live 2026-09-07, see sources_ercot.py."""
    return {
        "fields": [{"name": f} for f in field_order],
        "data": [[r[f] for f in field_order] for r in records],
    }


class _FakeErcotResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeErcotSession:
    """Returns one full day's worth of records regardless of request args."""

    def __init__(self, payload: dict):
        self._payload = payload

    def get(self, *args, **kwargs):
        return _FakeErcotResponse(self._payload)


class _FakeErcotSessionByDate:
    """Returns a different payload depending on the requested
    deliveryDateFrom — needed to exercise a gappy day alongside a good one
    over a multi-day run_ercot_dam_pipeline/run_ercot_rtm_pipeline range."""

    def __init__(self, payload_by_date: dict[str, dict]):
        self._payload_by_date = payload_by_date

    def get(self, *args, **kwargs):
        requested_date = kwargs["params"]["deliveryDateFrom"]
        return _FakeErcotResponse(self._payload_by_date[requested_date])


def _dam_day_records(delivery_date: str, price: float = 0.10) -> list[dict]:
    return [
        {
            "deliveryDate": delivery_date,
            "hourEnding": f"{h:02d}:00",
            "settlementPoint": "HB_WEST",
            "settlementPointPrice": price,
            "DSTFlag": False,
        }
        for h in range(1, 25)
    ]


def test_run_ercot_dam_pipeline_caches_hourly_chicago_data(tmp_path):
    d = date(2026, 7, 15)
    token = ErcotToken(access_token="fake", subscription_key="fake")
    session = _FakeErcotSession(_ercot_payload(_dam_day_records(d.isoformat()), DAM_FIELDS))

    combined, report = run_ercot_dam_pipeline(
        d, d, token=token, cache_path=tmp_path / "ercot_dam.parquet", session=session
    )

    assert len(combined) == 24
    assert (combined["period_minutes"] == 60).all()
    assert (combined["currency"] == "USD").all()
    assert report.n_missing_periods == 0


def test_run_ercot_dam_pipeline_reports_a_gappy_day_alongside_a_good_one(tmp_path):
    good_day, gappy_day = date(2026, 7, 15), date(2026, 7, 16)
    token = ErcotToken(access_token="fake", subscription_key="fake")
    gappy_records = _dam_day_records(gappy_day.isoformat())[:20]  # 4 hours missing
    session = _FakeErcotSessionByDate(
        {
            good_day.isoformat(): _ercot_payload(_dam_day_records(good_day.isoformat()), DAM_FIELDS),
            gappy_day.isoformat(): _ercot_payload(gappy_records, DAM_FIELDS),
        }
    )

    with pytest.raises(GapThresholdExceededError, match="missing-fraction threshold"):
        run_ercot_dam_pipeline(good_day, gappy_day, token=token, cache_path=tmp_path / "ercot_dam.parquet", session=session)

    # the good day must still have been cached despite the raise (ADR-008)
    cached = pd.read_parquet(tmp_path / "ercot_dam.parquet")
    assert len(cached[cached["settlement_date"] == pd.Timestamp(good_day)]) == 24
    assert len(cached[cached["settlement_date"] == pd.Timestamp(gappy_day)]) == 20


def test_run_ercot_rtm_pipeline_caches_15_minute_chicago_data(tmp_path):
    d = date(2026, 7, 15)
    token = ErcotToken(access_token="fake", subscription_key="fake")
    records = [
        {
            "deliveryDate": d.isoformat(),
            "deliveryHour": h,
            "deliveryInterval": i,
            "settlementPoint": "HB_WEST",
            "settlementPointType": "HU",
            "settlementPointPrice": 0.10,
            "DSTFlag": False,
        }
        for h in range(1, 25)
        for i in range(1, 5)
    ]
    session = _FakeErcotSession(_ercot_payload(records, RTM_FIELDS))

    combined, report = run_ercot_rtm_pipeline(
        d, d, token=token, cache_path=tmp_path / "ercot_rtm.parquet", session=session
    )

    assert len(combined) == 96
    assert (combined["period_minutes"] == 15).all()
    assert report.n_missing_periods == 0
