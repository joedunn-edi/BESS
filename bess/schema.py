"""
schema.py — the canonical data contract for price series in this project.

Responsible for:
    * defining the canonical columns every price DataFrame must have,
      in a fixed order, with fixed dtypes;
    * `validate()`, a hard gate that every DataFrame must pass before it is
      allowed further into the pipeline. It raises rather than repairs.
    * DST-aware, timezone-parameterised knowledge of how many settlement
      periods of a given length exist on a given local calendar day —
      generalised beyond GB/half-hourly so the same, already-tested logic
      serves any market (see the tz/period_minutes parameters below).

Deliberately NOT responsible for:
    * fetching data from any source (see sources_elexon.py, sources_ercot.py);
    * deciding what to do about gaps (flag vs fill, thresholds) — pipeline.py;
    * anything about the battery or optimisation (see config.py).

Conventions locked here: energy in kWh, power in kW, 0.5 h steps for GB
specifically (other markets use their own native period length — see
period_minutes). timestamp_utc and settlement_date are both stored rather
than one derived from the other — see ADR-001 in DECISIONS.md.

Multi-market note: price is stored in each source's native currency
(price_per_kwh + an explicit currency column), not converted to GBP at
ingest — avoids introducing FX rate data as a confound in what's supposed
to be a signal about the market's own price dynamics. Each cached dataset
is expected to be internally homogeneous — one market, one currency, one
period length — validate() checks this rather than assuming it.
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
    "period_minutes",
    "price_per_kwh",
    "currency",
    "source",
]

#: settlement_date is a tz-naive midnight Timestamp, not a tz-aware one —
#: it must never be compared directly against timestamp_utc.
EXPECTED_DTYPES: dict[str, str] = {
    "timestamp_utc": "datetime64[ns, UTC]",
    "settlement_date": "datetime64[ns]",
    "settlement_period": "int64",
    "period_minutes": "int64",
    "price_per_kwh": "float64",
    "currency": "object",
    "source": "object",
}


class SchemaValidationError(ValueError):
    """Raised by validate() when a DataFrame violates the canonical contract."""


def settlement_day_utc_bounds(settlement_date: date, tz: ZoneInfo = LONDON) -> tuple[datetime, datetime]:
    """
    UTC instants marking the start and end of a settlement day in the
    given local calendar (local midnight to next local midnight).
    Defaults to Europe/London for backward compatibility with existing GB
    call sites; pass a different ZoneInfo (e.g. America/Chicago) for
    another market's calendar — it has its own, different DST transition
    dates, which is exactly why this needed to become a parameter rather
    than a hardcoded constant.
    """
    start = datetime(settlement_date.year, settlement_date.month, settlement_date.day, tzinfo=tz)
    end = start + timedelta(days=1)
    # must convert via UTC: subtracting two aware datetimes with the *same*
    # tzinfo object skips the offset change across a DST boundary (ADR-002)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def expected_period_count(settlement_date: date, tz: ZoneInfo = LONDON, period_minutes: int = 30) -> int:
    """
    Number of settlement periods of length `period_minutes` in one local
    calendar day for timezone `tz`: normally 24h * (60/period_minutes),
    fewer/more on that calendar's own clock-change days. Defaults
    reproduce the original GB half-hourly behaviour exactly (48 normally,
    46 on the spring clock-change day, 50 on the autumn one).
    """
    start_utc, end_utc = settlement_day_utc_bounds(settlement_date, tz=tz)
    elapsed_hours = (end_utc - start_utc).total_seconds() / 3600
    periods = elapsed_hours * (60 / period_minutes)
    assert periods == int(periods), (
        f"day length {elapsed_hours}h is not aligned to {period_minutes}-minute periods"
    )
    return int(periods)


def full_grid(start_date: date, end_date: date, tz: ZoneInfo = LONDON, period_minutes: int = 30) -> pd.DataFrame:
    """
    The complete expected grid for [start_date, end_date] inclusive: one
    row per period that *should* exist, DST-aware for the given tz/period
    length. Columns: timestamp_utc, settlement_date, settlement_period,
    period_minutes — no price/source/currency, since this is a pure
    calendar skeleton, not real data.
    """
    rows = []
    d = start_date
    while d <= end_date:
        start_utc, _ = settlement_day_utc_bounds(d, tz=tz)
        for period in range(1, expected_period_count(d, tz=tz, period_minutes=period_minutes) + 1):
            rows.append((start_utc + timedelta(minutes=period_minutes * (period - 1)), d, period))
        d += timedelta(days=1)

    grid = pd.DataFrame(rows, columns=["timestamp_utc", "settlement_date", "settlement_period"])
    grid["timestamp_utc"] = pd.to_datetime(grid["timestamp_utc"], utc=True)
    grid["settlement_date"] = pd.to_datetime(grid["settlement_date"])
    grid["settlement_period"] = grid["settlement_period"].astype("int64")
    grid["period_minutes"] = pd.array([period_minutes] * len(grid), dtype="int64")
    return grid


def validate(df: pd.DataFrame, tz: ZoneInfo = LONDON) -> pd.DataFrame:
    """
    Validate a price DataFrame against the canonical contract and return it
    sorted into canonical order (by timestamp_utc, ascending).

    tz identifies which local calendar's DST rules apply when checking
    settlement_period ranges (default Europe/London for GB data) — pass
    the correct tz explicitly for a different market (e.g. ERCOT's
    America/Chicago).

    Raises SchemaValidationError — collecting every problem found rather
    than stopping at the first — on wrong/missing/extra columns, naive or
    non-UTC timestamps, NaNs, duplicate periods, mixed currency/period
    -length within one dataset, or an out-of-range settlement_period.
    Never coerces or repairs; see ADR-005.
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
        raise SchemaValidationError("; ".join(errors))

    for col, expected_dtype in EXPECTED_DTYPES.items():
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            errors.append(f"column '{col}' has dtype {actual_dtype!r}, expected {expected_dtype!r}")

    if errors:
        raise SchemaValidationError("; ".join(errors))

    if df["timestamp_utc"].dt.tz is None:
        errors.append("timestamp_utc is timezone-naive; canonical timestamps must be tz-aware UTC")

    for col in ("timestamp_utc", "settlement_date", "settlement_period", "period_minutes", "price_per_kwh", "currency"):
        if df[col].isna().any():
            errors.append(f"{df[col].isna().sum()} row(s) have NaN/NaT {col}")

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

    if df["currency"].nunique() > 1:
        errors.append(
            f"mixed currencies in one dataset: {sorted(df['currency'].unique())} "
            "— each cached dataset should hold one market/currency"
        )
    if df["period_minutes"].nunique() > 1:
        errors.append(
            f"mixed period_minutes in one dataset: {sorted(df['period_minutes'].unique())} "
            "— each cached dataset should hold one granularity"
        )

    if not errors:
        period_minutes = int(df["period_minutes"].iloc[0])
        for sdate, group in df.groupby("settlement_date"):
            max_allowed = expected_period_count(sdate.date(), tz=tz, period_minutes=period_minutes)
            bad = group.loc[group["settlement_period"] > max_allowed, "settlement_period"]
            if not bad.empty:
                errors.append(
                    f"settlement_date {sdate.date()} allows periods 1..{max_allowed}, "
                    f"but found: {sorted(bad.unique().tolist())}"
                )

    if errors:
        raise SchemaValidationError("; ".join(errors))

    return df.sort_values("timestamp_utc").reset_index(drop=True)[CANONICAL_COLUMNS]
