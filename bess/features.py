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

from zoneinfo import ZoneInfo

import pandas as pd

from bess.schema import LONDON, validate

LAG_PERIODS = (1, 2)  # t-1, t-2 — market-agnostic; day/week lags are separate, see below


def _periods_per_day(period_minutes: int) -> int:
    return 24 * 60 // period_minutes


def longest_lookback_periods(period_minutes: int) -> int:
    """The largest lookback build_features() needs (the week-ago lag) —
    exposed so callers/tests can compute the warm-up row count without
    duplicating this arithmetic."""
    return _periods_per_day(period_minutes) * 7


def build_features(price_df: pd.DataFrame, tz: ZoneInfo = LONDON) -> pd.DataFrame:
    """
    Build the Tier 2 feature matrix from a canonical price DataFrame.
    Returns one row per period with a fully-populated feature set, plus
    the actual price at t as `price_per_kwh` (the forecasting target,
    not a feature — callers building X/y for training must exclude it
    from X). Rows without enough history for every lag/rolling feature
    are dropped entirely.

    tz must match whatever market `price_df` actually came from (default
    Europe/London — GB) — schema.validate()'s own DST-aware period-count
    check depends on it; pass tz=CHICAGO for ERCOT data, same as
    pipeline.py/results.py already do elsewhere.
    """
    df = validate(price_df, tz=tz).reset_index(drop=True)
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


def build_features_with_dam(rtm_df: pd.DataFrame, dam_df: pd.DataFrame, tz: ZoneInfo = LONDON) -> pd.DataFrame:
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

    tz must match rtm_df's actual market (pass tz=CHICAGO for real ERCOT
    data) — see build_features()'s docstring for why.
    """
    features = build_features(rtm_df, tz=tz)

    dam = dam_df.copy()
    dam["hour_of_day"] = dam["settlement_period"] - 1  # DAM is hourly: period IS the hour
    dam_by_hour = dam.set_index(["settlement_date", "hour_of_day"])["price_per_kwh"]

    rtm = validate(rtm_df, tz=tz).reset_index(drop=True)
    rtm_settlement_date = rtm.set_index("timestamp_utc")["settlement_date"].reindex(features["timestamp_utc"]).to_numpy()
    key = pd.MultiIndex.from_arrays([rtm_settlement_date, features["hour_of_day"].to_numpy()])
    features["dam_price_this_hour"] = dam_by_hour.reindex(key).to_numpy()
    features["rtm_minus_dam_lag_1"] = features["lag_1"] - features["dam_price_this_hour"]

    return features.dropna().reset_index(drop=True)


def build_features_with_weather_ceiling(rtm_df: pd.DataFrame, weather_df: pd.DataFrame, tz: ZoneInfo = LONDON) -> pd.DataFrame:
    """
    build_features() for ERCOT RTM, plus PERFECT-FORESIGHT weather features
    (temperature/wind/cloud-cover/solar-radiation columns from
    sources_weather.py) — the actual, real weather during the exact hour
    each RTM period falls in, not a live, latency-delayed observation.

    This is a CEILING TEST ONLY: "if weather were known with certainty,
    would it carry any signal about RTM prices at all?" It is deliberately
    NOT leakage-safe for real trading — in reality you only ever observe
    weather with reporting latency, never the true value at the instant a
    trading decision needs it (see sources_weather.py's module docstring).
    Do not use this function's output to claim a realistic trading result;
    it exists to decide whether the harder, latency-realistic version is
    worth building at all.

    A period whose hour has no matching weather row is dropped, same
    "don't invent it" policy as a missing lag or DAM match.

    tz must match rtm_df's actual market (pass tz=CHICAGO for real ERCOT
    data) — see build_features()'s docstring for why.
    """
    features = build_features(rtm_df, tz=tz)

    weather_by_hour = weather_df.set_index("timestamp_utc")
    weather_columns = [c for c in weather_df.columns if c != "timestamp_utc"]

    hour = features["timestamp_utc"].dt.floor("h")
    matched = weather_by_hour.reindex(hour)
    for col in weather_columns:
        features[col] = matched[col].to_numpy()

    return features.dropna().reset_index(drop=True)


def build_features_with_wind(rtm_df: pd.DataFrame, wind_forecast_df: pd.DataFrame, tz: ZoneInfo = LONDON) -> pd.DataFrame:
    """
    build_features() for ERCOT RTM, plus the genuinely leakage-safe
    48h-ahead wind forecast (WGRPP, West) as an exogenous feature.

    Unlike build_features_with_weather_ceiling(), this one IS safe to use
    as a real trading feature: `wind_forecast_df` must come from
    sources_ercot_wind.fetch_wind_forecast_snapshot(), confirmed
    (DECISIONS.md's ADR-022) to be published at least 48h ahead of
    delivery — not fetch_wind_generation()'s post-delivery forecast
    columns, which may reflect a later, closer-to-delivery revision.

    Deliberately WGRPP only, not also STWPF: confirmed (ADR-022) WGRPP is
    dramatically more accurate specifically in low-wind hours — the
    regime Stage 1 tied to RTM's price spikes — while STWPF badly
    overpredicts available wind during a genuine lull. That's a real P50
    (median) vs P80 (deliberately conservative) distinction in what each
    forecast is built to represent, not noise, so there's no reason to
    also include the one already shown to be worse for this purpose.

    A period whose hour has no matching wind-forecast row is dropped,
    same "don't invent it" policy as a missing lag or DAM match.

    tz must match rtm_df's actual market (pass tz=CHICAGO for real ERCOT
    data) — see build_features()'s docstring for why.
    """
    features = build_features(rtm_df, tz=tz)

    wind_by_hour = wind_forecast_df.set_index("timestamp_utc")["wind_wgrpp_west_mw"]
    hour = features["timestamp_utc"].dt.floor("h")
    features["wind_wgrpp_west_mw"] = wind_by_hour.reindex(hour).to_numpy()

    return features.dropna().reset_index(drop=True)


def build_features_with_actual_generation(
    rtm_df: pd.DataFrame, solar_df: pd.DataFrame, wind_df: pd.DataFrame, tz: ZoneInfo = LONDON
) -> pd.DataFrame:
    """
    build_features() for ERCOT RTM, plus the ACTUAL (perfect-foresight)
    solar and wind generation for the exact hour each RTM period falls in
    — `solar_gen_farwest_mw` from sources_ercot_solar.fetch_solar_generation()
    and `wind_gen_west_mw` from sources_ercot_wind.fetch_wind_generation().

    This is a CEILING TEST, same spirit as build_features_with_weather_
    ceiling(): NOT leakage-safe (you would never actually know real-time
    generation in advance of the delivery hour; build_features_with_wind()'s
    WGRPP forecast is the realistic one). It answers a different, sharper
    question than the realistic feature tests do: if the model had PERFECT
    knowledge of generation — not just an accurate forecast of it — would
    MPC's profit improve at all? If even perfect information doesn't help,
    that's strong evidence the problem found in ADR-022 (real, verified
    accuracy improvements failing to improve, or worsening, real MPC
    profit) is structural — something about the forecast-then-optimize
    architecture itself, or how MPC's LP uses these features — rather
    than simply "the forecasts aren't accurate enough yet."

    A period whose hour has no matching solar or wind row is dropped,
    same "don't invent it" policy as a missing lag or DAM match.

    tz must match rtm_df's actual market (pass tz=CHICAGO for real ERCOT
    data) — see build_features()'s docstring for why.
    """
    features = build_features(rtm_df, tz=tz)

    solar_by_hour = solar_df.set_index("timestamp_utc")["solar_gen_farwest_mw"]
    wind_by_hour = wind_df.set_index("timestamp_utc")["wind_gen_west_mw"]
    hour = features["timestamp_utc"].dt.floor("h")
    features["solar_gen_farwest_mw"] = solar_by_hour.reindex(hour).to_numpy()
    features["wind_gen_west_mw"] = wind_by_hour.reindex(hour).to_numpy()

    return features.dropna().reset_index(drop=True)
