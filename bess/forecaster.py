"""
forecaster.py — Tier 2 price forecasting: predicts price h periods ahead
for each horizon h needed by mpc.py, plus a naive persistence baseline to
confirm the learned model is actually earning its keep.

Responsible for:
    * naive_forecast(): "same period yesterday" persistence, generalised
      to arbitrary horizons (ADR-015)
    * train_forecaster(): one LightGBM model per horizon, trained via
      direct multi-horizon forecasting — a separate model per horizon
      rather than recursive 1-step iteration, so a model never has to
      treat its own earlier prediction as if it were a real lag feature
      (ADR-015)
    * evaluate_forecaster(): time-series cross-validated MAE/RMSE per
      horizon (expanding window — train on the past, validate on a later
      unseen chunk, never the reverse), model vs naive_forecast()

Deliberately NOT responsible for:
    * the MPC control loop (mpc.py) — this module only produces price
      predictions from a feature row; it has no notion of a battery,
      a schedule, or state of charge
    * the feature matrix itself (features.py) — takes its already
      leakage-checked output as input and trusts it
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

TARGET_COLUMN = "price_per_kwh"
NON_FEATURE_COLUMNS = ("timestamp_utc", TARGET_COLUMN)


def _feature_columns(features_df: pd.DataFrame) -> list[str]:
    return [c for c in features_df.columns if c not in NON_FEATURE_COLUMNS]


def _target_for_horizon(features_df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    target_h[i] = price[i + horizon - 1] — the price `horizon` periods
    beyond the base (h=1) target, using the SAME row's features (which
    already only reflect data up to i-1, per features.py's leakage guard).
    horizon=1 reduces to price[i] itself, the unmodified base target.
    Rows that would need a future row past the end of the series get NaN
    here and must be dropped before training/evaluating at that horizon —
    the trailing-side mirror of features.py dropping leading warm-up rows.
    """
    return features_df[TARGET_COLUMN].shift(-(horizon - 1))


def naive_forecast(features_df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    "Same period yesterday" persistence, generalised to any horizon: the
    naive guess for the target `horizon` periods out is whatever the price
    was one day before *that target period*, not one day before now.
    At horizon=1 this is exactly lag_1_day (price one day ago) unshifted,
    as specified; at horizon=(periods per day) it degenerates to price[i-1]
    (the most recent known price) — both are honest "persistence" guesses
    for their own horizon, not the same fixed reference reused regardless
    of how far out the target is.

    Built from the already-computed `lag_1_day` column (features.py derives
    this from the data's own period_minutes, so it means "yesterday" on
    any market's granularity, not just GB's), not by re-shifting
    `price_per_kwh` on this (already warm-up-truncated) frame: the
    latter would needlessly reproduce NaNs for the first (horizon-1) rows
    that `lag_1_day` doesn't have, since `lag_1_day` was computed before
    the warm-up rows were dropped and every surviving row already has one.
    """
    return features_df["lag_1_day"].shift(-(horizon - 1))


@dataclass(frozen=True)
class Tier2Forecaster:
    """One trained LightGBM model per horizon. predict() returns a
    DataFrame with one column per requested horizon."""

    models: dict[int, lgb.LGBMRegressor]
    feature_columns: list[str]

    def predict(self, X: pd.DataFrame, horizons: Iterable[int] | None = None) -> pd.DataFrame:
        horizons = list(horizons) if horizons is not None else sorted(self.models)
        missing = set(horizons) - set(self.models)
        if missing:
            raise ValueError(f"no trained model for horizon(s) {sorted(missing)}")
        return pd.DataFrame({h: self.models[h].predict(X[self.feature_columns]) for h in horizons}, index=X.index)


def train_forecaster(features_df: pd.DataFrame, horizons: Iterable[int]) -> Tier2Forecaster:
    """Train one LightGBM model per horizon on all rows in `features_df`
    that have a valid (non-NaN) target for that horizon."""
    feature_columns = _feature_columns(features_df)
    models: dict[int, lgb.LGBMRegressor] = {}

    for horizon in horizons:
        target = _target_for_horizon(features_df, horizon)
        valid = target.notna()
        model = lgb.LGBMRegressor(random_state=0, verbosity=-1)
        model.fit(features_df.loc[valid, feature_columns], target.loc[valid])
        models[horizon] = model

    return Tier2Forecaster(models=models, feature_columns=feature_columns)


def evaluate_forecaster(features_df: pd.DataFrame, horizons: Iterable[int], n_splits: int = 5) -> pd.DataFrame:
    """
    Time-series cross-validated MAE/RMSE per horizon, model vs naive,
    averaged across folds. Uses an expanding window (TimeSeriesSplit's
    default): each fold trains on all data before a cutoff and validates
    on a later, entirely unseen chunk — the split never trains on data
    later than what it validates on.
    """
    feature_columns = _feature_columns(features_df)
    rows = []

    for horizon in horizons:
        target = _target_for_horizon(features_df, horizon)
        naive = naive_forecast(features_df, horizon)
        valid = target.notna() & naive.notna()

        X = features_df.loc[valid, feature_columns].reset_index(drop=True)
        y = target.loc[valid].reset_index(drop=True)
        naive_y = naive.loc[valid].reset_index(drop=True)

        model_mae, model_rmse, naive_mae, naive_rmse = [], [], [], []
        for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
            model = lgb.LGBMRegressor(random_state=0, verbosity=-1)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            y_test = y.iloc[test_idx]

            model_mae.append(mean_absolute_error(y_test, pred))
            model_rmse.append(root_mean_squared_error(y_test, pred))
            naive_mae.append(mean_absolute_error(y_test, naive_y.iloc[test_idx]))
            naive_rmse.append(root_mean_squared_error(y_test, naive_y.iloc[test_idx]))

        rows.append(
            {
                "horizon": horizon,
                "model_mae": float(np.mean(model_mae)),
                "model_rmse": float(np.mean(model_rmse)),
                "naive_mae": float(np.mean(naive_mae)),
                "naive_rmse": float(np.mean(naive_rmse)),
            }
        )

    return pd.DataFrame(rows)
