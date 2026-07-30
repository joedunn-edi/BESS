"""
pipeline.py — fetch a date range, account for gaps against the DST-aware
grid, report data quality, and cache to parquet.

Responsible for:
    * looping a single-day fetcher (sources_elexon.py) over a date range,
      tolerating individual day failures rather than aborting the whole run
    * diffing what was actually fetched against schema.full_grid() to find
      gaps, and computing a DataQualityReport (missing periods, negative
      price frequency, price distribution)
    * merging with, and updating, a parquet cache on disk
    * raising GapThresholdExceededError if gaps are bad enough — see
      ADR-008 for the exact thresholds and why. Raised *after* caching
      (see run_pipeline): a bad day is still reported and still stops the
      caller from silently proceeding, but doesn't cost the good days
      fetched alongside it.

Deliberately NOT responsible for:
    * ever inventing a price value for a missing period. The returned/cached
      DataFrame contains only real, already-validated periods — a gappy day
      just has fewer rows than schema.expected_period_count() for that day.
      Gaps are only ever visible through DataQualityReport, never through a
      placeholder row (ADR-008). Deciding what to do with an incomplete day
      (e.g. exclude it) is left to whoever consumes the cache next.
    * choosing which series (imbalance vs day-ahead) feeds the optimiser —
      still a stage-4 decision (see sources_elexon.py).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from bess.schema import CANONICAL_COLUMNS, EXPECTED_DTYPES, full_grid, validate
from bess.sources_elexon import SOURCE_DAY_AHEAD, SOURCE_IMBALANCE, fetch_day_ahead_prices, fetch_imbalance_prices

_EMPTY_CANONICAL = pd.DataFrame({col: pd.Series([], dtype=dtype) for col, dtype in EXPECTED_DTYPES.items()})[
    CANONICAL_COLUMNS
]


class GapThresholdExceededError(ValueError):
    """Raised when fetched data is too gappy to proceed silently (ADR-008)."""


@dataclass(frozen=True)
class DataQualityReport:
    """Summary of one pipeline run over [start_date, end_date] for one source."""

    source: str
    start_date: date
    end_date: date
    n_expected_periods: int
    n_present_periods: int
    missing_periods: list[tuple[date, int]]
    missing_fraction_by_day: dict[date, float]
    max_consecutive_missing: int
    n_negative_price_periods: int
    negative_price_fraction: float
    price_distribution: dict[str, float]
    failed_fetch_dates: list[date] = field(default_factory=list)

    @property
    def n_missing_periods(self) -> int:
        return len(self.missing_periods)

    def summary(self) -> str:
        lines = [
            f"Data quality report — {self.source}, {self.start_date} to {self.end_date}",
            f"  periods: {self.n_present_periods}/{self.n_expected_periods} present "
            f"({self.n_missing_periods} missing)",
            f"  max consecutive missing: {self.max_consecutive_missing}",
            f"  negative prices: {self.n_negative_price_periods} "
            f"({self.negative_price_fraction:.2%} of present periods)",
            f"  price distribution (£/kWh): {self.price_distribution}",
        ]
        if self.failed_fetch_dates:
            lines.append(f"  failed fetches (treated as fully missing): {self.failed_fetch_dates}")
        worst_days = [(d, f) for d, f in sorted(self.missing_fraction_by_day.items(), key=lambda kv: -kv[1]) if f > 0][
            :5
        ]
        if worst_days:
            lines.append(f"  worst days by missing fraction: {worst_days}")
        return "\n".join(lines)


def _fetch_range(
    fetch_one_day: Callable[[date], pd.DataFrame],
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, list[date]]:
    """
    Fetch every day in [start_date, end_date], concatenating successes.
    A single day's fetch failure (network error, AllZeroPriceSeriesError,
    etc.) doesn't abort the run — that day is recorded as failed and treated
    as fully missing by the gap accounting below, uniformly with a day that
    partially failed inside a successful response.
    """
    frames = []
    failed_dates: list[date] = []
    d = start_date
    while d <= end_date:
        try:
            frames.append(fetch_one_day(d))
        except (requests.exceptions.RequestException, ValueError) as exc:
            warnings.warn(f"fetch failed for {d}: {exc}")
            failed_dates.append(d)
        d += timedelta(days=1)

    fetched = pd.concat(frames, ignore_index=True) if frames else _EMPTY_CANONICAL.copy()
    return fetched, failed_dates


def _missing_mask(grid: pd.DataFrame, present_keys: set[tuple[date, int]]) -> pd.Series:
    keys = zip(grid["settlement_date"].dt.date, grid["settlement_period"])
    return pd.Series([k not in present_keys for k in keys], index=grid.index)


def _max_consecutive_missing(missing_mask: pd.Series) -> int:
    """Longest run of consecutive missing periods, in the chronological grid
    order already carried by missing_mask — a run can span a day boundary,
    so this is computed over the whole range at once, not per day."""
    if not missing_mask.any():
        return 0
    # a new run starts wherever a value differs from the row before it;
    # cumsum of those start-flags gives every run a distinct id, then
    # summing True within each id gives that run's length.
    run_id = (missing_mask != missing_mask.shift(fill_value=False)).cumsum()
    return int(missing_mask.groupby(run_id).sum().max())


def compute_quality_report(
    fetched: pd.DataFrame,
    start_date: date,
    end_date: date,
    source: str,
    failed_fetch_dates: list[date] | None = None,
) -> DataQualityReport:
    """Diff `fetched` against the complete expected grid and summarise gaps,
    negative-price frequency, and price distribution. Never mutates `fetched`."""
    grid = full_grid(start_date, end_date)
    present_keys = set(zip(fetched["settlement_date"].dt.date, fetched["settlement_period"]))
    missing_mask = _missing_mask(grid, present_keys)

    missing_rows = grid.loc[missing_mask]
    missing_periods = list(zip(missing_rows["settlement_date"].dt.date, missing_rows["settlement_period"]))

    missing_fraction_by_day = {
        d: missing_mask.loc[day_grid.index].mean() for d, day_grid in grid.groupby(grid["settlement_date"].dt.date)
    }

    negative_mask = fetched["price_gbp_per_kwh"] < 0
    price_distribution = fetched["price_gbp_per_kwh"].describe().to_dict() if not fetched.empty else {}

    return DataQualityReport(
        source=source,
        start_date=start_date,
        end_date=end_date,
        n_expected_periods=len(grid),
        n_present_periods=len(fetched),
        missing_periods=missing_periods,
        missing_fraction_by_day=missing_fraction_by_day,
        max_consecutive_missing=_max_consecutive_missing(missing_mask),
        n_negative_price_periods=int(negative_mask.sum()),
        negative_price_fraction=float(negative_mask.mean()) if len(fetched) else 0.0,
        price_distribution=price_distribution,
        failed_fetch_dates=failed_fetch_dates or [],
    )


def _check_thresholds(
    report: DataQualityReport,
    max_missing_fraction_per_day: float,
    max_consecutive_missing: int,
) -> None:
    bad_days = {d: f for d, f in report.missing_fraction_by_day.items() if f > max_missing_fraction_per_day}
    if bad_days:
        raise GapThresholdExceededError(
            f"{report.source}: {len(bad_days)} day(s) exceed the "
            f"{max_missing_fraction_per_day:.0%} missing-fraction threshold: {bad_days}"
        )
    if report.max_consecutive_missing > max_consecutive_missing:
        raise GapThresholdExceededError(
            f"{report.source}: {report.max_consecutive_missing} consecutive missing periods "
            f"exceeds the threshold of {max_consecutive_missing}"
        )


def _merge_with_cache(new_data: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """Load an existing parquet cache (if any), merge in new_data, and keep
    the freshest value for any (settlement_date, settlement_period) that
    appears in both — Elexon can revise recent settlement prices after
    initial publication, so a re-fetch should supersede what's cached."""
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        combined = pd.concat([cached, new_data], ignore_index=True)
    else:
        combined = new_data
    combined = combined.drop_duplicates(subset=["settlement_date", "settlement_period"], keep="last")
    return validate(combined)


def run_pipeline(
    fetch_one_day: Callable[[date], pd.DataFrame],
    start_date: date,
    end_date: date,
    source: str,
    cache_path: Path,
    max_missing_fraction_per_day: float = 0.0,
    max_consecutive_missing: int = 0,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Fetch [start_date, end_date] one day at a time, merge whatever real data
    comes back into the parquet cache at cache_path, then raise
    GapThresholdExceededError if gaps exceed the configured thresholds.

    Caching happens *before* the threshold check on purpose: the point of a
    per-day threshold (rather than a whole-range one) is that one bad day
    shouldn't cost you the good days fetched alongside it — so they're saved
    regardless, and the exception is still raised afterwards to make sure a
    bad day can't be silently missed by the caller.
    """
    fetched, failed_dates = _fetch_range(fetch_one_day, start_date, end_date)
    report = compute_quality_report(fetched, start_date, end_date, source, failed_dates)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined = _merge_with_cache(fetched, cache_path)
    combined.to_parquet(cache_path, index=False)

    _check_thresholds(report, max_missing_fraction_per_day, max_consecutive_missing)

    return combined, report


def run_imbalance_pipeline(
    start_date: date,
    end_date: date,
    cache_path: Path = Path("data/imbalance.parquet"),
    max_missing_fraction_per_day: float = 0.0,
    max_consecutive_missing: int = 0,
    session: requests.Session = requests,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """run_pipeline() wired to fetch_imbalance_prices(). Quality flags
    (bsad_defaulted etc.) aren't threaded through here — see ADR-007;
    call fetch_imbalance_prices() directly for a given day if needed."""
    return run_pipeline(
        fetch_one_day=lambda d: fetch_imbalance_prices(d, session=session).prices,
        start_date=start_date,
        end_date=end_date,
        source=SOURCE_IMBALANCE,
        cache_path=cache_path,
        max_missing_fraction_per_day=max_missing_fraction_per_day,
        max_consecutive_missing=max_consecutive_missing,
    )


def run_day_ahead_pipeline(
    start_date: date,
    end_date: date,
    cache_path: Path = Path("data/day_ahead.parquet"),
    max_missing_fraction_per_day: float = 0.0,
    max_consecutive_missing: int = 0,
    session: requests.Session = requests,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """run_pipeline() wired to fetch_day_ahead_prices()."""
    return run_pipeline(
        fetch_one_day=lambda d: fetch_day_ahead_prices(d, session=session),
        start_date=start_date,
        end_date=end_date,
        source=SOURCE_DAY_AHEAD,
        cache_path=cache_path,
        max_missing_fraction_per_day=max_missing_fraction_per_day,
        max_consecutive_missing=max_consecutive_missing,
    )
