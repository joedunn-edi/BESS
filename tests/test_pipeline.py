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
    run_pipeline,
)
from bess.schema import expected_period_count, settlement_day_utc_bounds, validate


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
            "price_gbp_per_kwh": prices,
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
    assert (combined["price_gbp_per_kwh"] == 0.20).all()  # freshest fetch wins
