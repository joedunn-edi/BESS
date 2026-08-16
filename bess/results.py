"""
results.py — stage 7: run Tier 1 over the full cached price history,
compute headline metrics, verify every day against the independent
backtester, and plot one example day.

Responsible for:
    * run_tier1_over_history(): solves every cached day independently,
      collecting cross-check failures rather than raising on the first one
      (see ADR-010's Q9 discussion — a batch check needs full visibility
      across the whole history, not a stop at day 1)
    * cumulative P&L, cycles/day (discharge-based, consistent with the
      discharge-referenced degradation convention in ADR-009), and an
      annualised £/kWh-of-capacity/year figure
    * boundary_soc_sensitivity(): the sensitivity sweep over the fixed
      cyclic boundary value that ADR-009 explicitly deferred to this stage
    * plot_example_day(): price, SoC, and charge/discharge for one day

Deliberately NOT responsible for:
    * fetching or caching data (pipeline.py) — takes an already-loaded
      price history DataFrame, not a file path, so it can be tested
      against small synthetic histories without touching disk
    * re-implementing the LP or the cashflow/SoC arithmetic — reuses
      optimiser_tier1.solve_day(), naive_baseline.solve_day_naive(), and
      backtest.simulate()/assert_matches_lp() unchanged
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bess.backtest import assert_matches_lp, simulate
from bess.config import Battery
from bess.naive_baseline import solve_day_naive
from bess.optimiser_tier1 import DT_HOURS, Tier1Schedule, solve_day


@dataclass(frozen=True)
class DayResult:
    settlement_date: date
    prices: np.ndarray
    tier1_schedule: Tier1Schedule
    tier1_profit: float
    naive_profit: float
    discharged_kwh: float  # Tier 1's discharged energy this day, for cycles/day


@dataclass(frozen=True)
class HistoryResults:
    day_results: list[DayResult]
    failed_days: list[tuple[date, str]]  # (date, reason) for days excluded from the metrics below
    cumulative_pnl: pd.Series  # Tier 1, indexed by settlement_date
    cumulative_pnl_naive: pd.Series
    mean_cycles_per_day: float
    annualised_gbp_per_kwh_capacity: float


def run_tier1_over_history(price_history: pd.DataFrame, battery: Battery, boundary_soc: float = 0.5) -> HistoryResults:
    """
    Solve every settlement day in `price_history` independently with Tier 1,
    cross-check each against the backtester, and aggregate. A day that
    fails to solve to optimality or fails the LP/backtester cross-check is
    excluded from the metrics and recorded in `failed_days` — the whole run
    doesn't abort on one bad day, matching the same "cache the good, flag
    the bad" spirit as pipeline.py's gap handling (ADR-008).
    """
    usable_capacity_kwh = (battery.soc_max - battery.soc_min) * battery.capacity_kwh

    day_results: list[DayResult] = []
    failed_days: list[tuple[date, str]] = []

    for settlement_date, day_df in price_history.groupby("settlement_date"):
        d = settlement_date.date()
        day_df = day_df.sort_values("settlement_period")
        prices = day_df["price_gbp_per_kwh"].to_numpy()

        try:
            schedule = solve_day(prices, battery, boundary_soc=boundary_soc)
            assert_matches_lp(schedule, prices, battery)
        except (RuntimeError, AssertionError) as exc:
            failed_days.append((d, str(exc)))
            continue

        naive_schedule = solve_day_naive(prices, battery, boundary_soc=boundary_soc)
        naive_result = simulate(naive_schedule.charge_kw, naive_schedule.discharge_kw, prices, battery, schedule.soc_kwh[0])

        day_results.append(
            DayResult(
                settlement_date=d,
                prices=prices,
                tier1_schedule=schedule,
                tier1_profit=schedule.objective_value,
                naive_profit=naive_result.cashflow,
                discharged_kwh=float(schedule.discharge_kw.sum() * DT_HOURS),
            )
        )

    day_results.sort(key=lambda r: r.settlement_date)
    dates = [r.settlement_date for r in day_results]
    cumulative_pnl = pd.Series([r.tier1_profit for r in day_results], index=dates).cumsum()
    cumulative_pnl_naive = pd.Series([r.naive_profit for r in day_results], index=dates).cumsum()

    n_days = len(day_results)
    mean_cycles_per_day = (
        float(np.mean([r.discharged_kwh / usable_capacity_kwh for r in day_results])) if n_days else 0.0
    )
    total_profit = sum(r.tier1_profit for r in day_results)
    # scale to a 365-day year if the sample isn't exactly a year (e.g. some
    # days were excluded above) — an approximation, not a projection
    annualised = (total_profit / battery.capacity_kwh) * (365 / n_days) if n_days else 0.0

    return HistoryResults(
        day_results=day_results,
        failed_days=failed_days,
        cumulative_pnl=cumulative_pnl,
        cumulative_pnl_naive=cumulative_pnl_naive,
        mean_cycles_per_day=mean_cycles_per_day,
        annualised_gbp_per_kwh_capacity=annualised,
    )


def boundary_soc_sensitivity(
    price_history: pd.DataFrame, battery: Battery, values: tuple[float, ...] = (0.25, 0.5, 0.75)
) -> dict[float, float]:
    """
    Total profit over the full history at each candidate boundary_soc,
    fulfilling the sensitivity check ADR-009 deferred to this stage rather
    than settling the 50% choice by assertion alone.
    """
    return {v: sum(r.tier1_profit for r in run_tier1_over_history(price_history, battery, v).day_results) for v in values}


def plot_example_day(day_result: DayResult, battery: Battery, boundary_soc: float, output_path: str) -> None:
    """Price, charge/discharge, and SoC for one settled day."""
    schedule = day_result.tier1_schedule
    T = len(day_result.prices)
    hours = np.arange(T) * DT_HOURS

    fig, (ax_price, ax_soc) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax_price.plot(hours, day_result.prices * 1000, color="black", label="price (£/MWh)", linewidth=1.5)
    ax_price.set_ylabel("price (£/MWh)")
    ax_power = ax_price.twinx()
    ax_power.bar(hours, schedule.charge_kw, width=DT_HOURS, color="tab:blue", alpha=0.5, label="charge (kW)")
    ax_power.bar(hours, -schedule.discharge_kw, width=DT_HOURS, color="tab:red", alpha=0.5, label="discharge (kW)")
    ax_power.set_ylabel("charge (+) / discharge (-) kW")
    ax_price.set_title(f"Tier 1 schedule — {day_result.settlement_date} (profit £{day_result.tier1_profit:.2f})")
    lines1, labels1 = ax_price.get_legend_handles_labels()
    lines2, labels2 = ax_power.get_legend_handles_labels()
    ax_price.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    soc_hours = np.arange(T + 1) * DT_HOURS
    ax_soc.plot(soc_hours, schedule.soc_kwh, color="tab:green", linewidth=1.5, label="SoC (kWh)")
    ax_soc.axhline(battery.soc_min * battery.capacity_kwh, color="grey", linestyle="--", linewidth=1, label="soc_min/max")
    ax_soc.axhline(battery.soc_max * battery.capacity_kwh, color="grey", linestyle="--", linewidth=1)
    ax_soc.set_ylabel("SoC (kWh)")
    ax_soc.set_xlabel("hour of day")
    ax_soc.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
