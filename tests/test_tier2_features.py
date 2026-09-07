"""
test_tier2_features.py — tests for bess/features.py.

The leakage guard is the priority test here, built the same way as this
project's other correctness anchors (Stage 5's LP/backtest cross-check):
not by reading the code and checking shift() signs look right, but by a
black-box check that would catch a leakage bug regardless of how it was
introduced — corrupt only the *future* portion of a price series and
assert every feature row unaffected by that corruption is byte-identical.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from bess.features import LAG_PERIODS, build_features, build_features_with_dam, longest_lookback_periods

ROLLING_WINDOW = 48  # GB default: 24h at 30-min periods, matches build_features()'s own periods_per_day
from bess.schema import settlement_day_utc_bounds, validate


def _synthetic_price_history(n_days: int, price_fn) -> pd.DataFrame:
    """A canonical multi-day price history where price_fn(row_index) gives
    each period's price, in chronological order — lets tests hand-verify
    lag/rolling values against a known, predictable sequence."""
    start_date = date(2026, 1, 1)
    rows = []
    idx = 0
    for day_offset in range(n_days):
        d = start_date + timedelta(days=day_offset)
        start_utc, _ = settlement_day_utc_bounds(d)
        for period in range(1, 49):  # fixed 48-period days, no DST days in this range
            rows.append(
                {
                    "timestamp_utc": start_utc + timedelta(minutes=30 * (period - 1)),
                    "settlement_date": pd.Timestamp(d),
                    "settlement_period": period,
                    "period_minutes": 30,
                    "price_per_kwh": price_fn(idx),
                    "currency": "GBP",
                    "source": "test",
                }
            )
            idx += 1

    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["settlement_period"] = df["settlement_period"].astype("int64")
    df["period_minutes"] = df["period_minutes"].astype("int64")
    df["price_per_kwh"] = df["price_per_kwh"].astype("float64")
    return validate(df)


# --- the leakage guard -----------------------------------------------------------


def test_features_at_row_t_are_unaffected_by_changes_to_future_prices():
    n_days = 16  # 768 periods: room for a cutoff well past the 336-period warm-up
    rng = np.random.default_rng(0)
    base_prices = rng.uniform(0.05, 0.30, n_days * 48)

    df = _synthetic_price_history(n_days, lambda i: base_prices[i])
    features_original = build_features(df)

    # corrupt only prices from this point on — must be well past the
    # warm-up drop (336), or every surviving feature row would already be
    # "at or after" the cutoff and the test would be vacuous
    cutoff_row = longest_lookback_periods(30) + 200
    corrupted_prices = base_prices.copy()
    corrupted_prices[cutoff_row:] = rng.uniform(0.05, 0.30, len(corrupted_prices) - cutoff_row)
    df_corrupted = _synthetic_price_history(n_days, lambda i: corrupted_prices[i])
    features_corrupted = build_features(df_corrupted)

    # both feature frames dropped the same number of warm-up rows (the
    # corruption starts well after the warm-up period, so row counts and
    # timestamps line up); every row from a period strictly before the
    # corrupted region must be byte-identical between the two runs
    assert len(features_original) == len(features_corrupted)
    unaffected = features_original["timestamp_utc"] < df.loc[cutoff_row, "timestamp_utc"]
    assert unaffected.sum() > 0  # sanity: the assertion below isn't vacuous

    pd.testing.assert_frame_equal(
        features_original.loc[unaffected].reset_index(drop=True),
        features_corrupted.loc[unaffected].reset_index(drop=True),
    )


def test_leakage_guard_actually_catches_a_broken_implementation():
    # a leakage guard that would pass no matter what isn't a guard — prove
    # this one actually fires, by rebuilding a deliberately-leaky feature
    # (a rolling mean that includes today's own price) and confirming the
    # same corruption-comparison technique catches it.
    n_days = 16
    rng = np.random.default_rng(1)
    base_prices = rng.uniform(0.05, 0.30, n_days * 48)
    df = _synthetic_price_history(n_days, lambda i: base_prices[i])

    def leaky_rolling_mean(frame: pd.DataFrame) -> pd.Series:
        # BUG: no shift(1) — includes price at t itself in the window
        return frame["price_per_kwh"].rolling(ROLLING_WINDOW).mean()

    cutoff_row = longest_lookback_periods(30) + 200
    corrupted_prices = base_prices.copy()
    corrupted_prices[cutoff_row:] = rng.uniform(0.05, 0.30, len(corrupted_prices) - cutoff_row)
    df_corrupted = _synthetic_price_history(n_days, lambda i: corrupted_prices[i])

    leaky_original = leaky_rolling_mean(df).iloc[cutoff_row - 1]
    leaky_corrupted = leaky_rolling_mean(df_corrupted).iloc[cutoff_row - 1]
    # the row just before the cutoff still shouldn't be affected — but a
    # non-shifted rolling window reaching row cutoff-1 only touches past
    # rows too, so use a row where the *window itself* would only be
    # leak-visible via a boundary one step later
    leaky_original_at_cutoff = leaky_rolling_mean(df).iloc[cutoff_row]
    leaky_corrupted_at_cutoff = leaky_rolling_mean(df_corrupted).iloc[cutoff_row]
    assert leaky_original_at_cutoff != leaky_corrupted_at_cutoff  # confirms the bug is real
    # and confirms build_features()'s own (correct) rolling mean at the
    # same row is NOT affected, i.e. the guard technique discriminates
    # between the two implementations
    correct_original = build_features(df)
    correct_corrupted = build_features(df_corrupted)
    ts_at_cutoff = df.loc[cutoff_row, "timestamp_utc"]
    row_original = correct_original.loc[correct_original["timestamp_utc"] == ts_at_cutoff]
    row_corrupted = correct_corrupted.loc[correct_corrupted["timestamp_utc"] == ts_at_cutoff]
    assert row_original[f"rolling_mean_{ROLLING_WINDOW}"].item() == pytest.approx(
        row_corrupted[f"rolling_mean_{ROLLING_WINDOW}"].item()
    )


# --- structural / hand-computed checks --------------------------------------------


def test_drops_rows_without_full_lag_history():
    n_days = 10  # 480 periods
    df = _synthetic_price_history(n_days, lambda i: float(i))

    features = build_features(df)

    # lag_1_week is the binding constraint (the largest lookback) — the
    # first 336 periods can never have a valid "336 periods ago" value
    assert len(features) == n_days * 48 - longest_lookback_periods(30)
    assert not features.isna().any().any()


def test_lag_values_match_hand_computed_sequence():
    n_days = 10
    df = _synthetic_price_history(n_days, lambda i: float(i))  # price == row index

    features = build_features(df)

    # price[i] = i, so at the first surviving row (original index =
    # longest_lookback_periods(30) = 336), lag_k should equal (336 - k)
    # exactly for every lookback this module computes
    first_row = features.iloc[0]
    first_row_index = longest_lookback_periods(30)
    for lag in LAG_PERIODS:
        assert first_row[f"lag_{lag}"] == pytest.approx(first_row_index - lag)
    assert first_row["lag_1_day"] == pytest.approx(first_row_index - 48)
    assert first_row["lag_1_week"] == pytest.approx(first_row_index - 336)


def test_rolling_mean_excludes_current_period():
    n_days = 10
    df = _synthetic_price_history(n_days, lambda i: float(i))  # price == row index

    features = build_features(df)
    first_row_index = longest_lookback_periods(30)  # original row index of the first surviving feature row
    expected_mean = np.mean(range(first_row_index - ROLLING_WINDOW, first_row_index))

    assert features.iloc[0][f"rolling_mean_{ROLLING_WINDOW}"] == pytest.approx(expected_mean)


def test_calendar_features_are_correct():
    # independently recompute the expected day-of-week/weekend/hour for
    # every surviving row from its own timestamp/settlement_period, rather
    # than hardcoding which calendar dates survive the warm-up drop
    n_days = 14
    df = _synthetic_price_history(n_days, lambda i: 0.10)

    features = build_features(df)
    df_by_timestamp = df.set_index("timestamp_utc")

    for _, row in features.iterrows():
        expected_dow = row["timestamp_utc"].weekday()  # Monday=0 ... Sunday=6
        assert row["day_of_week"] == expected_dow
        assert row["is_weekend"] == (expected_dow in (5, 6))

        period = df_by_timestamp.loc[row["timestamp_utc"], "settlement_period"]
        assert row["hour_of_day"] == (period - 1) // 2


# --- period_minutes generalisation (ADR-020) ---------------------------------------


def _synthetic_15min_price_history(n_days: int, price_fn) -> pd.DataFrame:
    """Same shape as _synthetic_price_history(), but ERCOT-RTM-like:
    period_minutes=15, 96 periods/day, USD."""
    start_date = date(2026, 1, 1)
    rows = []
    idx = 0
    for day_offset in range(n_days):
        d = start_date + timedelta(days=day_offset)
        start_utc, _ = settlement_day_utc_bounds(d)
        for period in range(1, 97):
            rows.append(
                {
                    "timestamp_utc": start_utc + timedelta(minutes=15 * (period - 1)),
                    "settlement_date": pd.Timestamp(d),
                    "settlement_period": period,
                    "period_minutes": 15,
                    "price_per_kwh": price_fn(idx),
                    "currency": "USD",
                    "source": "test",
                }
            )
            idx += 1
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["settlement_period"] = df["settlement_period"].astype("int64")
    df["period_minutes"] = df["period_minutes"].astype("int64")
    df["price_per_kwh"] = df["price_per_kwh"].astype("float64")
    return validate(df)


def test_lag_1_day_and_1_week_and_hour_of_day_generalise_to_15_minute_periods():
    # the real bug this guards against: lag_1_day/lag_1_week and
    # hour_of_day were originally hardcoded to GB's 48-periods/day, which
    # would silently mean something different (or crash) on ERCOT RTM's
    # 96-periods/day data
    n_days = 10
    df = _synthetic_15min_price_history(n_days, lambda i: float(i))  # price == row index

    features = build_features(df)

    periods_per_day_15min = 96
    first_row_index = longest_lookback_periods(15)
    assert first_row_index == periods_per_day_15min * 7

    first_row = features.iloc[0]
    assert first_row["lag_1_day"] == pytest.approx(first_row_index - periods_per_day_15min)
    assert first_row["lag_1_week"] == pytest.approx(first_row_index - periods_per_day_15min * 7)
    assert f"rolling_mean_{periods_per_day_15min}" in features.columns

    df_by_timestamp = df.set_index("timestamp_utc")
    for _, row in features.iterrows():
        period = df_by_timestamp.loc[row["timestamp_utc"], "settlement_period"]
        assert row["hour_of_day"] == (period - 1) // 4  # 4 fifteen-minute periods per hour


# --- build_features_with_dam (ADR-020) ----------------------------------------------


def _synthetic_dam_price_history(n_days: int, price_fn) -> pd.DataFrame:
    """Same shape as _synthetic_15min_price_history(), but DAM-like:
    period_minutes=60, 24 periods/day. price_fn(day_offset, hour) gives
    each hour's price, so tests can build a DAM series distinguishable
    from the RTM one it's merged against."""
    start_date = date(2026, 1, 1)
    rows = []
    for day_offset in range(n_days):
        d = start_date + timedelta(days=day_offset)
        start_utc, _ = settlement_day_utc_bounds(d)
        for period in range(1, 25):
            hour = period - 1
            rows.append(
                {
                    "timestamp_utc": start_utc + timedelta(hours=hour),
                    "settlement_date": pd.Timestamp(d),
                    "settlement_period": period,
                    "period_minutes": 60,
                    "price_per_kwh": price_fn(day_offset, hour),
                    "currency": "USD",
                    "source": "test",
                }
            )
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["settlement_period"] = df["settlement_period"].astype("int64")
    df["period_minutes"] = df["period_minutes"].astype("int64")
    df["price_per_kwh"] = df["price_per_kwh"].astype("float64")
    return validate(df)


def test_build_features_with_dam_merges_the_correct_hours_dam_price():
    n_days = 10
    rtm_df = _synthetic_15min_price_history(n_days, lambda i: 1.0)  # RTM price irrelevant here
    dam_df = _synthetic_dam_price_history(n_days, lambda day, hour: 100.0 + day * 10 + hour)

    features = build_features_with_dam(rtm_df, dam_df)

    dam_by_key = dam_df.set_index(["settlement_date", "settlement_period"])["price_per_kwh"]
    rtm_by_timestamp = rtm_df.set_index("timestamp_utc")

    for _, row in features.iterrows():
        settlement_date = rtm_by_timestamp.loc[row["timestamp_utc"], "settlement_date"]
        period = rtm_by_timestamp.loc[row["timestamp_utc"], "settlement_period"]
        hour_period = (period - 1) // 4 + 1  # RTM period -> matching DAM (hourly) period
        expected_dam_price = dam_by_key.loc[(settlement_date, hour_period)]
        assert row["dam_price_this_hour"] == pytest.approx(expected_dam_price)


def test_rtm_minus_dam_lag_1_is_lag_1_minus_dam_price():
    n_days = 10
    rng = np.random.default_rng(3)
    rtm_prices = rng.uniform(0.01, 0.05, n_days * 96)
    rtm_df = _synthetic_15min_price_history(n_days, lambda i: rtm_prices[i])
    dam_df = _synthetic_dam_price_history(n_days, lambda day, hour: 0.03)

    features = build_features_with_dam(rtm_df, dam_df)

    pd.testing.assert_series_equal(
        features["rtm_minus_dam_lag_1"],
        features["lag_1"] - features["dam_price_this_hour"],
        check_names=False,
    )


def test_build_features_with_dam_drops_rows_with_no_matching_dam_data():
    n_days = 10
    rtm_df = _synthetic_15min_price_history(n_days, lambda i: 1.0)
    # DAM data only for the first 9 of the 10 days — day 10 has no match at all
    dam_df = _synthetic_dam_price_history(n_days - 1, lambda day, hour: 0.03)

    features = build_features_with_dam(rtm_df, dam_df)

    assert not features.isna().any().any()
    last_day = date(2026, 1, 1) + timedelta(days=n_days - 1)
    assert (features["timestamp_utc"].dt.date < last_day).all()
