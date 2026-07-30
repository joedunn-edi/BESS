"""
schema.py — the canonical data contract for price series in this project.

Responsible for:
    * defining the five canonical columns every price DataFrame must have,
      in a fixed order, with fixed dtypes;
    * `validate()`, a hard gate that every DataFrame must pass before it is
      allowed further into the pipeline (fetchers -> validate -> pipeline ->
      optimiser). It raises rather than repairs.
    * DST-aware knowledge of how many half-hour settlement periods exist on
      a given Europe/London calendar day (48 normally, 46 on the spring
      clock-change day, 50 on the autumn one).

Deliberately NOT responsible for:
    * fetching data from any source (see sources_elexon.py);
    * deciding what to do about gaps (flag vs fill, thresholds) — that's a
      policy decision, made in pipeline.py. schema.py only knows the pure
      calendar fact of which periods *should* exist (full_grid(),
      expected_period_count()); it doesn't diff that against real data;
    * anything about the battery or optimisation (see config.py).

Conventions locked here: energy in kWh, power in kW, price in £/kWh, 0.5 h
steps. timestamp_utc and settlement_date are both stored rather than one
derived from the other — see ADR-001 in DECISIONS.md.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

LONDON = ZoneInfo("Europe/London")

#: Every validated DataFrame has exactly these columns, in this order.
CANONICAL_COLUMNS: list[str] = [
    "timestamp_utc",
    "settlement_date",
    "settlement_period",
    "price_gbp_per_kwh",
    "source",
]

#: settlement_date is a tz-naive midnight Timestamp, not a tz-aware one —
#: it must never be compared directly against timestamp_utc.
EXPECTED_DTYPES: dict[str, str] = {
    "timestamp_utc": "datetime64[ns, UTC]",
    "settlement_date": "datetime64[ns]",
    "settlement_period": "int64",
    "price_gbp_per_kwh": "float64",
    "source": "object",
}


class SchemaValidationError(ValueError):
    """Raised by validate() when a DataFrame violates the canonical contract."""


def settlement_day_utc_bounds(settlement_date: date) -> tuple[datetime, datetime]:
    """
    UTC instants marking the start and end of a Europe/London settlement day
    (local midnight to next local midnight). Used both by
    expected_period_count() below and by fetchers that need to query an API
    with UTC from/to bounds for a given settlement date (see sources_elexon.py).
    """
    start = datetime(settlement_date.year, settlement_date.month, settlement_date.day, tzinfo=LONDON)
    end = start + timedelta(days=1)
    # must convert via UTC: subtracting two aware datetimes with the *same*
    # tzinfo object skips the offset change across a DST boundary (ADR-002)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def expected_period_count(settlement_date: date) -> int:
    """
    Number of half-hour settlement periods in one Europe/London calendar day:
    48 normally, 46 on the spring clock-change day, 50 on the autumn one.
    Computed from actual elapsed time, not hardcoded DST dates. See ADR-002.
    """
    start_utc, end_utc = settlement_day_utc_bounds(settlement_date)
    elapsed_hours = (end_utc - start_utc).total_seconds() / 3600
    periods = elapsed_hours * 2
    assert periods == int(periods), f"non-half-hour-aligned day length: {elapsed_hours}h"
    return int(periods)


def full_grid(start_date: date, end_date: date) -> pd.DataFrame:
    """
    The complete expected half-hourly grid for [start_date, end_date]
    inclusive: one row per period that *should* exist, DST-aware (46/48/50
    per day). Columns: timestamp_utc, settlement_date, settlement_period —
    no price/source, since this is a pure calendar skeleton, not real data.

    This is what pipeline.py diffs fetched data against to find gaps; it
    intentionally has no opinion on what to do about a gap (see the module
    docstring) — it just states what "complete" means.
    """
    rows = []
    d = start_date
    while d <= end_date:
        start_utc, _ = settlement_day_utc_bounds(d)
        for period in range(1, expected_period_count(d) + 1):
            rows.append((start_utc + timedelta(minutes=30 * (period - 1)), d, period))
        d += timedelta(days=1)

    grid = pd.DataFrame(rows, columns=["timestamp_utc", "settlement_date", "settlement_period"])
    grid["timestamp_utc"] = pd.to_datetime(grid["timestamp_utc"], utc=True)
    grid["settlement_date"] = pd.to_datetime(grid["settlement_date"])
    grid["settlement_period"] = grid["settlement_period"].astype("int64")
    return grid


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a price DataFrame against the canonical contract and return it
    sorted into canonical order (by timestamp_utc, ascending).

    Raises SchemaValidationError — collecting every problem found rather
    than stopping at the first — on wrong/missing/extra columns, naive or
    non-UTC timestamps, NaNs, duplicate periods, or an out-of-range
    settlement_period. Never coerces or repairs; see ADR-005.
    """
    errors: list[str] = []

    if not isinstance(df, pd.DataFrame):
        raise SchemaValidationError(f"expected a pandas DataFrame, got {type(df)}")

    actual_cols = list(df.columns)
    missing = [c for c in CANONICAL_COLUMNS if c not in actual_cols]
    extra = [c for c in actual_cols if c not in CANONICAL_COLUMNS]
    if missing:
        errors.append(f"missing required column(s): {missing}")
    if extra:
        errors.append(f"unexpected extra column(s): {extra} (strip these before calling validate())")
    if missing:
        # can't check dtypes/values on columns that don't exist
        raise SchemaValidationError("; ".join(errors))

    for col, expected_dtype in EXPECTED_DTYPES.items():
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            errors.append(f"column '{col}' has dtype {actual_dtype!r}, expected {expected_dtype!r}")

    if errors:
        # bail before value checks — e.g. .dt.tz on a wrong-dtype column
        # would raise an unrelated AttributeError instead of this error
        raise SchemaValidationError("; ".join(errors))

    if df["timestamp_utc"].dt.tz is None:
        errors.append("timestamp_utc is timezone-naive; canonical timestamps must be tz-aware UTC")

    if df["timestamp_utc"].isna().any():
        errors.append(f"{df['timestamp_utc'].isna().sum()} row(s) have NaT timestamp_utc")
    if df["settlement_date"].isna().any():
        errors.append(f"{df['settlement_date'].isna().sum()} row(s) have NaT settlement_date")
    if df["settlement_period"].isna().any():
        errors.append(f"{df['settlement_period'].isna().sum()} row(s) have NaN settlement_period")
    if df["price_gbp_per_kwh"].isna().any():
        errors.append(f"{df['price_gbp_per_kwh'].isna().sum()} row(s) have NaN price_gbp_per_kwh")

    if errors:
        raise SchemaValidationError("; ".join(errors))

    dup_period_mask = df.duplicated(subset=["settlement_date", "settlement_period"], keep=False)
    if dup_period_mask.any():
        dupes = df.loc[dup_period_mask, ["settlement_date", "settlement_period"]].drop_duplicates()
        errors.append(f"duplicate (settlement_date, settlement_period) pairs found:\n{dupes.to_string(index=False)}")

    dup_ts_mask = df["timestamp_utc"].duplicated(keep=False)
    if dup_ts_mask.any():
        dupes = df.loc[dup_ts_mask, "timestamp_utc"].drop_duplicates()
        errors.append(f"duplicate timestamp_utc values found: {list(dupes)}")

    if (df["settlement_period"] <= 0).any():
        errors.append("settlement_period must be >= 1")

    for sdate, group in df.groupby("settlement_date"):
        max_allowed = expected_period_count(sdate.date())
        bad = group.loc[group["settlement_period"] > max_allowed, "settlement_period"]
        if not bad.empty:
            errors.append(
                f"settlement_date {sdate.date()} allows periods 1..{max_allowed}, "
                f"but found: {sorted(bad.unique().tolist())}"
            )

    if errors:
        raise SchemaValidationError("; ".join(errors))

    return df.sort_values("timestamp_utc").reset_index(drop=True)[CANONICAL_COLUMNS]
