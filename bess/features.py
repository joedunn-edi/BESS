"""
features.py — feature engineering for the Tier 2 (MPC) price forecaster.

Responsible for:
    * build_features(): lag, calendar, and rolling-statistic features for
      each settlement period, constructed so that every feature at row t
      is computable using only data known at or before t (see
      tests/test_tier2_features.py's leakage guard, and ADR-014). This is
      the Tier 2 equivalent of the Stage 5 LP/backtest cross-check: the
      one test that has to be trusted before anything built on top of it
      (the forecaster, the MPC controller) means anything.

Deliberately NOT responsible for:
    * the forecasting model itself (forecaster.py)
    * imputing missing history — rows at the start of the series where a
      lag/rolling window isn't yet fully available are dropped, not
      filled in, since a plausible-looking invented value for history
      that doesn't exist is a silent, undetectable assumption (ADR-014)
"""

from __future__ import annotations

import pandas as pd

from bess.schema import validate

LAG_PERIODS = (1, 2, 48, 336)  # t-1, t-2, yesterday same period, week-ago same period
ROLLING_WINDOW = 48


def build_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Tier 2 feature matrix from a canonical price DataFrame.
    Returns one row per period with a fully-populated feature set, plus
    the actual price at t as `price_gbp_per_kwh` (the forecasting target,
    not a feature — callers building X/y for training must exclude it
    from X). Rows without enough history for every lag/rolling feature
    are dropped entirely.
    """
    df = validate(price_df).reset_index(drop=True)
    price = df["price_gbp_per_kwh"]

    features = pd.DataFrame(index=df.index)
    features["timestamp_utc"] = df["timestamp_utc"]

    for lag in LAG_PERIODS:
        features[f"lag_{lag}"] = price.shift(lag)

    # shift(1) before rolling: the window covers [t-ROLLING_WINDOW, t-1],
    # deliberately excluding price at t itself — a rolling stat that
    # included the current row would leak the very value being predicted.
    trailing = price.shift(1).rolling(ROLLING_WINDOW)
    features[f"rolling_mean_{ROLLING_WINDOW}"] = trailing.mean()
    features[f"rolling_std_{ROLLING_WINDOW}"] = trailing.std()

    # derived from settlement_period/settlement_date (local-calendar
    # fields already on the canonical frame), not timestamp_utc — sidesteps
    # the UTC/London-offset conversion this project has been careful about
    # everywhere else, since these are calendar-of-record fields already.
    features["hour_of_day"] = (df["settlement_period"] - 1) // 2
    day_of_week = df["settlement_date"].dt.dayofweek  # Monday=0 ... Sunday=6
    features["day_of_week"] = day_of_week
    features["is_weekend"] = day_of_week.isin([5, 6])

    features["price_gbp_per_kwh"] = price  # target, carried through — not a feature

    return features.dropna().reset_index(drop=True)
