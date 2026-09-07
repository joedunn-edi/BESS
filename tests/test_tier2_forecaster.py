"""
test_tier2_forecaster.py — tests for bess/forecaster.py.

Covers: naive_forecast()'s horizon generalisation against hand-computed
cases, that TimeSeriesSplit itself never trains on data later than what
it validates on (the CV-level analogue of features.py's leakage guard),
and — on a synthetic series with a genuine learnable trend a fixed-lag
naive baseline can't track — that the trained model actually beats naive,
which is "the one job" this module has to prove before Tier 2 means
anything.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from bess.features import build_features
from bess.forecaster import Tier2Forecaster, evaluate_forecaster, naive_forecast, train_forecaster
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


# --- naive_forecast's horizon generalisation --------------------------------------


def test_naive_forecast_h1_matches_lag_1_day():
    n_days = 12
    rng = np.random.default_rng(0)
    prices = rng.uniform(0.05, 0.30, n_days * 48)
    df = _synthetic_price_history(n_days, lambda i: prices[i])
    features = build_features(df)

    naive_h1 = naive_forecast(features, horizon=1)

    pd.testing.assert_series_equal(naive_h1, features["lag_1_day"], check_names=False)


def test_naive_forecast_h48_degenerates_to_most_recent_price():
    # hand-verified against the original (pre-drop) price sequence at a
    # few interior rows directly, rather than compared against a whole
    # -series proxy expression — a plain shift(1) of the truncated frame
    # has different (and less complete) edge behaviour at *both* ends:
    # naive_forecast() can reach further back at the leading edge (via
    # lag_1_day, computed before warm-up rows were dropped) and correctly
    # runs out 47 rows earlier at the trailing edge (it needs to see 47
    # periods ahead, same as _target_for_horizon(h=48) would) — neither
    # edge matches a same-length shift(1) of the already-truncated series.
    n_days = 12
    rng = np.random.default_rng(1)
    prices = rng.uniform(0.05, 0.30, n_days * 48)
    df = _synthetic_price_history(n_days, lambda i: prices[i])
    features = build_features(df)

    naive_h48 = naive_forecast(features, horizon=48)

    warm_up = 336
    for row in (0, 50, 150, 192):  # spans the leading edge through the last valid row
        original_index = row + warm_up
        assert naive_h48.iloc[row] == pytest.approx(prices[original_index - 1])
    for row in (193, 220, 239):  # the trailing (h-1)=47 rows correctly have no valid target
        assert pd.isna(naive_h48.iloc[row])


# --- TimeSeriesSplit itself never trains on the future -----------------------------


def test_time_series_split_never_trains_on_future_rows():
    X = pd.DataFrame({"x": range(500)})
    for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(X):
        assert train_idx.max() < test_idx.min()


# --- Tier2Forecaster / train_forecaster --------------------------------------------


def test_train_forecaster_produces_one_model_per_horizon():
    n_days = 16
    rng = np.random.default_rng(2)
    prices = rng.uniform(0.05, 0.30, n_days * 48)
    df = _synthetic_price_history(n_days, lambda i: prices[i])
    features = build_features(df)

    forecaster = train_forecaster(features, horizons=[1, 2, 6])

    assert set(forecaster.models.keys()) == {1, 2, 6}
    preds = forecaster.predict(features.iloc[:5], horizons=[1, 2])
    assert list(preds.columns) == [1, 2]
    assert len(preds) == 5


def test_predict_raises_on_untrained_horizon():
    n_days = 16
    df = _synthetic_price_history(n_days, lambda i: 0.10)
    features = build_features(df)
    forecaster = train_forecaster(features, horizons=[1])

    with pytest.raises(ValueError, match="horizon"):
        forecaster.predict(features.iloc[:5], horizons=[12])


# --- the one job: beats naive on data naive can't track -----------------------------


def test_model_beats_naive_when_naive_cannot_track_a_weekday_weekend_regime():
    # "yesterday same period" naive persistence gets every weekday/weekend
    # *transition* day wrong (predicting Monday from Sunday's price, or
    # Saturday from Friday's) — a large, systematic error a model with an
    # explicit is_weekend feature has no excuse to make. Deliberately a
    # bounded, stationary two-level pattern, not an unbounded trend: a
    # plain gradient-boosted tree model genuinely cannot extrapolate
    # beyond the value range it was trained on (splits only ever predict
    # values seen in training), so a trending series would unfairly
    # penalise the model for a real, separate limitation this test isn't
    # about.
    start = date(2026, 1, 1)
    n_days = 30

    def price_fn(i):
        d = start + timedelta(days=i // 48)
        return 0.25 if d.weekday() >= 5 else 0.10

    df = _synthetic_price_history(n_days, price_fn)
    features = build_features(df)

    report = evaluate_forecaster(features, horizons=[1], n_splits=3)

    row = report.iloc[0]
    assert row["model_mae"] < row["naive_mae"]
    assert row["model_rmse"] < row["naive_rmse"]


def test_evaluate_forecaster_returns_one_row_per_horizon_with_expected_columns():
    n_days = 16
    rng = np.random.default_rng(3)
    prices = rng.uniform(0.05, 0.30, n_days * 48)
    df = _synthetic_price_history(n_days, lambda i: prices[i])
    features = build_features(df)

    report = evaluate_forecaster(features, horizons=[1, 6], n_splits=3)

    assert list(report["horizon"]) == [1, 6]
    assert set(report.columns) == {"horizon", "model_mae", "model_rmse", "naive_mae", "naive_rmse"}
    assert (report[["model_mae", "model_rmse", "naive_mae", "naive_rmse"]] >= 0).all().all()
