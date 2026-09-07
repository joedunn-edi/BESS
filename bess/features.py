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
    * build_features_with_dam(): the above, plus ERCOT-specific exogenous
      features from that hour's already-known DAM clearing price — see
      ADR-020. Not applicable to GB, which has no equivalent second,
      known-in-advance price series for its imbalance-price forecast.

Deliberately NOT responsible for:
    * the forecasting model itself (forecaster.py)
    * imputing missing history — rows at the start of the series where a
      lag/rolling window isn't yet fully available are dropped, not
      filled in, since a plausible-looking invented value for history
      that doesn't exist is a silent, undetectable assumption (ADR-014)

period_minutes generalisation (ADR-020): LAG_PERIODS/ROLLING_WINDOW were
originally hardcoded assuming GB's 48-periods/day (lag_48 = "yesterday",
lag_336 = "a week ago"). ERCOT RTM has 96 periods/day, so those numbers
meant something different there. Generalised by deriving "a day ago"/"a
week ago" from each frame's own `period_minutes` column, and renaming the
resulting columns to say what they mean directly (`lag_1_day`,
`lag_1_week`) rather than a period count that varies by market — LAG_PERIODS
itself now only covers the two short, market-agnostic lags (1, 2 periods
back), since "1/2 periods ago" means the same thing structurally
regardless of granularity.
"""

from __future__ import annotations

import pandas as pd

from bess.schema import validate

LAG_PERIODS = (1, 2)  # t-1, t-2 — market-agnostic; day/week lags are separate, see below


def _periods_per_day(period_minutes: int) -> int:
    return 24 * 60 // period_minutes


def longest_lookback_periods(period_minutes: int) -> int:
    """The largest lookback build_features() needs (the week-ago lag) —
    exposed so callers/tests can compute the warm-up row count without
    duplicating this arithmetic."""
    return _periods_per_day(period_minutes) * 7


def build_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Tier 2 feature matrix from a canonical price DataFrame.
    Returns one row per period with a fully-populated feature set, plus
    the actual price at t as `price_per_kwh` (the forecasting target,
    not a feature — callers building X/y for training must exclude it
    from X). Rows without enough history for every lag/rolling feature
    are dropped entirely.
    """
    df = validate(price_df).reset_index(drop=True)
    price = df["price_per_kwh"]
    period_minutes = int(df["period_minutes"].iloc[0])
    periods_per_day = _periods_per_day(period_minutes)
    periods_per_hour = 60 // period_minutes

    features = pd.DataFrame(index=df.index)
    features["timestamp_utc"] = df["timestamp_utc"]

    for lag in LAG_PERIODS:
        features[f"lag_{lag}"] = price.shift(lag)
    features["lag_1_day"] = price.shift(periods_per_day)
    features["lag_1_week"] = price.shift(periods_per_day * 7)

    # shift(1) before rolling: the window covers [t-periods_per_day, t-1],
    # deliberately excluding price at t itself — a rolling stat that
    # included the current row would leak the very value being predicted.
    rolling_window = periods_per_day
    trailing = price.shift(1).rolling(rolling_window)
    features[f"rolling_mean_{rolling_window}"] = trailing.mean()
    features[f"rolling_std_{rolling_window}"] = trailing.std()

    # derived from settlement_period/settlement_date (local-calendar
    # fields already on the canonical frame), not timestamp_utc — sidesteps
    # the UTC/London-offset conversion this project has been careful about
    # everywhere else, since these are calendar-of-record fields already.
    features["hour_of_day"] = (df["settlement_period"] - 1) // periods_per_hour
    day_of_week = df["settlement_date"].dt.dayofweek  # Monday=0 ... Sunday=6
    features["day_of_week"] = day_of_week
    features["is_weekend"] = day_of_week.isin([5, 6])

    features["price_per_kwh"] = price  # target, carried through — not a feature

    return features.dropna().reset_index(drop=True)


def build_features_with_dam(rtm_df: pd.DataFrame, dam_df: pd.DataFrame) -> pd.DataFrame:
    """
    build_features() for ERCOT RTM, plus two exogenous features from that
    hour's DAM clearing price — genuinely known in advance, since DAM
    settles the day before RTM trades (ADR-020):

        dam_price_this_hour  — the DAM price for the hour this RTM period
                                falls in, no leakage (published a day ahead)
        rtm_minus_dam_lag_1  — lag_1 (last period's actual RTM price) minus
                                dam_price_this_hour: the DAM/RTM spread as
                                of the most recent known period, a classic
                                real-time-trading feature

    A period whose hour has no matching DAM row (e.g. DAM had a gap that
    day) is dropped, same "don't invent it" policy as a missing lag.
    """
    features = build_features(rtm_df)

    dam = dam_df.copy()
    dam["hour_of_day"] = dam["settlement_period"] - 1  # DAM is hourly: period IS the hour
    dam_by_hour = dam.set_index(["settlement_date", "hour_of_day"])["price_per_kwh"]

    rtm = validate(rtm_df).reset_index(drop=True)
    rtm_settlement_date = rtm.set_index("timestamp_utc")["settlement_date"].reindex(features["timestamp_utc"]).to_numpy()
    key = pd.MultiIndex.from_arrays([rtm_settlement_date, features["hour_of_day"].to_numpy()])
    features["dam_price_this_hour"] = dam_by_hour.reindex(key).to_numpy()
    features["rtm_minus_dam_lag_1"] = features["lag_1"] - features["dam_price_this_hour"]

    return features.dropna().reset_index(drop=True)
