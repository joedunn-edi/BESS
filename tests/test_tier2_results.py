"""
test_tier2_results.py — tests for bess/results_tier2.py.

Uses a synthetic weekday/weekend price regime (same pattern proven in
test_tier2_forecaster.py to give the model a genuine, learnable edge over
naive persistence) sized just large enough for a real train/test split
and a real MPC backtest, kept small so the suite stays fast. The
corrupted-forecast check is the priority test — it's the one the brief
calls "the sanity check that matters most."
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from bess.config import Battery
from bess.features import build_features
from bess.results_tier2 import chronological_train_test_split, corrupted_forecast_check, run_tier2_comparison
from bess.schema import settlement_day_utc_bounds, validate


def _synthetic_price_history(n_days: int, price_fn) -> pd.DataFrame:
    start_date = date(2026, 1, 1)
    rows = []
    idx = 0
    for day_offset in range(n_days):
        d = start_date + timedelta(days=day_offset)
        start_utc, _ = settlement_day_utc_bounds(d)
        for period in range(1, 49):
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


def _battery(**overrides) -> Battery:
    defaults = dict(capacity_kwh=10.0, power_kw=10.0, round_trip_eff=0.9, soc_min=0.0, soc_max=1.0, degradation_cost_per_kwh=0.0)
    defaults.update(overrides)
    return Battery(**defaults)


# a weekday/weekend regime, small daily noise on top so there's something
# for lag/rolling features to do beyond the pure calendar signal
_START = date(2026, 1, 1)


def _price_fn(i: int) -> float:
    d = _START + timedelta(days=i // 48)
    base = 0.25 if d.weekday() >= 5 else 0.10
    hour_wiggle = 0.02 * np.sin(2 * np.pi * (i % 48) / 48)
    return base + hour_wiggle


# --- chronological split -----------------------------------------------------------


def test_chronological_split_respects_fraction_and_order():
    df = _synthetic_price_history(20, lambda i: 0.10)
    features = build_features(df)

    train, test = chronological_train_test_split(features, train_fraction=0.8)

    assert len(train) + len(test) == len(features)
    assert len(train) == pytest.approx(0.8 * len(features), abs=1)
    assert train["timestamp_utc"].max() < test["timestamp_utc"].min()


# --- the comparison itself ----------------------------------------------------------


def test_mpc_sits_between_naive_and_tier1_ceiling():
    df = _synthetic_price_history(40, _price_fn)
    features = build_features(df)
    battery = _battery()

    comparison = run_tier2_comparison(df, features, battery, horizon=4, train_fraction=0.8)

    assert comparison.naive_total_profit <= comparison.mpc_total_profit + 1e-6
    assert comparison.mpc_total_profit <= comparison.tier1_total_profit + 1e-6
    assert comparison.mpc_mean_cycles_per_day >= 0
    assert comparison.n_test_days > 0


# --- the sanity check that matters most ---------------------------------------------


def test_corrupted_forecast_collapses_toward_or_below_naive_floor():
    df = _synthetic_price_history(40, _price_fn)
    features = build_features(df)
    battery = _battery()

    comparison = run_tier2_comparison(df, features, battery, horizon=4, train_fraction=0.8)
    _, test_features = chronological_train_test_split(features, train_fraction=0.8)

    from bess.forecaster import train_forecaster

    train_features, _ = chronological_train_test_split(features, train_fraction=0.8)
    forecaster = train_forecaster(train_features, horizons=range(1, 5))

    corrupted_result = corrupted_forecast_check(
        test_features, forecaster, battery, horizon=4, initial_soc_kwh=0.5 * battery.capacity_kwh, seed=0
    )

    # the real, working forecaster should clearly beat a decoupled one —
    # if it doesn't, MPC's good performance isn't coming from the
    # forecast, which means it's coming from a leak somewhere else
    assert corrupted_result.cashflow < comparison.mpc_total_profit
    # and should collapse toward/below the naive floor specifically, not
    # just "worse than MPC" — a small allowance since a corrupted-but-real
    # model can still occasionally do a little better than naive by chance
    assert corrupted_result.cashflow <= comparison.naive_total_profit + 0.5 * abs(comparison.naive_total_profit) + 1e-6
