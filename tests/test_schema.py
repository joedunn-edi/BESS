"""
test_schema.py — tests for the canonical data contract in bess/schema.py.

Covers: the DST-aware period-count helper (normal / spring-forward /
autumn-back days), and validate()'s rejection of each contract violation
named in the brief (naive timestamps, NaN prices, duplicate periods),
plus the sort-into-canonical-order behaviour on success.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from bess.schema import CANONICAL_COLUMNS, SchemaValidationError, expected_period_count, validate


def _valid_frame(n_periods: int = 4, settlement_date: str = "2026-07-22") -> pd.DataFrame:
    """A minimal, contract-satisfying frame with n_periods half-hours."""
    start = pd.Timestamp(f"{settlement_date}T00:00:00", tz="UTC")
    timestamps = pd.date_range(start=start, periods=n_periods, freq="30min")
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            # explicit dtype="datetime64[ns]": pandas 2.x infers second-resolution
            # for a bare date-only string otherwise, which fails the ns-resolution
            # contract in EXPECTED_DTYPES — see schema.py's dtype checks.
            "settlement_date": pd.Series([pd.Timestamp(settlement_date)] * n_periods, dtype="datetime64[ns]"),
            "settlement_period": np.arange(1, n_periods + 1, dtype="int64"),
            "price_gbp_per_kwh": np.linspace(0.10, 0.20, n_periods),
            "source": "test",
        }
    )


# --- expected_period_count -------------------------------------------------


def test_period_count_normal_day():
    # an ordinary July day: no clock change
    assert expected_period_count(date(2026, 7, 22)) == 48


def test_period_count_spring_forward_day():
    # 2026-03-29: UK clocks go forward 01:00 -> 02:00, day is 23h long
    assert expected_period_count(date(2026, 3, 29)) == 46


def test_period_count_autumn_back_day():
    # 2026-10-25: UK clocks go back 02:00 -> 01:00, day is 25h long
    assert expected_period_count(date(2026, 10, 25)) == 50


# --- validate(): happy path --------------------------------------------------


def test_validate_accepts_well_formed_frame():
    df = _valid_frame()
    out = validate(df)
    assert list(out.columns) == CANONICAL_COLUMNS
    assert out["timestamp_utc"].is_monotonic_increasing


def test_validate_sorts_into_canonical_order():
    df = _valid_frame().iloc[::-1].reset_index(drop=True)  # shuffle to reverse order
    out = validate(df)
    assert out["timestamp_utc"].is_monotonic_increasing
    assert out["settlement_period"].tolist() == sorted(out["settlement_period"].tolist())


# --- validate(): rejections named explicitly in the brief -------------------


def test_validate_rejects_naive_timestamp():
    df = _valid_frame()
    df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize(None)
    with pytest.raises(SchemaValidationError, match="dtype"):
        validate(df)


def test_validate_rejects_non_utc_timezone():
    df = _valid_frame()
    df["timestamp_utc"] = df["timestamp_utc"].dt.tz_convert("Europe/London")
    with pytest.raises(SchemaValidationError):
        validate(df)


def test_validate_rejects_nan_price():
    df = _valid_frame()
    df.loc[1, "price_gbp_per_kwh"] = np.nan
    with pytest.raises(SchemaValidationError, match="price_gbp_per_kwh"):
        validate(df)


def test_validate_rejects_duplicate_periods():
    df = _valid_frame()
    df.loc[1, "settlement_period"] = df.loc[0, "settlement_period"]
    with pytest.raises(SchemaValidationError, match="duplicate"):
        validate(df)


# --- validate(): structural / other checks ----------------------------------


def test_validate_rejects_missing_column():
    df = _valid_frame().drop(columns=["source"])
    with pytest.raises(SchemaValidationError, match="missing"):
        validate(df)


def test_validate_rejects_extra_column():
    df = _valid_frame()
    df["unexpected"] = 1
    with pytest.raises(SchemaValidationError, match="extra"):
        validate(df)


def test_validate_rejects_period_beyond_days_maximum():
    df = _valid_frame(n_periods=1)
    df.loc[0, "settlement_period"] = 49  # normal day only allows 1..48
    with pytest.raises(SchemaValidationError, match="allows periods"):
        validate(df)


def test_validate_accepts_50_periods_on_autumn_back_day():
    df = _valid_frame(n_periods=50, settlement_date="2026-10-25")
    out = validate(df)
    assert len(out) == 50
