"""
backtest.py — an independent simulator that recomputes SoC and cashflow
from a charge/discharge schedule, without relying on any of the LP's own
internal arithmetic.

Responsible for:
    * simulate(): given a schedule (charge_kw/discharge_kw per period),
      prices, a Battery, and an initial SoC, independently recomputes the
      SoC trajectory, total cashflow, and whether the schedule actually
      respects SoC bounds, power limits, and mutual exclusivity
    * assert_matches_lp(): the correctness anchor — does the independent
      recomputation agree with a solved Tier1Schedule's own reported SoC
      and objective value, to within a tiny tolerance?

The arithmetic in simulate() is written fresh here, not shared with or
called from optimiser_tier1.py (both do pull Battery.eta_charge/
eta_discharge from config.py, since that's the intentional single source
of truth for efficiency — ADR-003 — not a shortcut around independence).
The point of a cross-check is that a bug in one implementation is unlikely
to be reproduced identically in a separately-written one; reusing the same
constraint-building code in both places would make agreement close to
tautological.

Deliberately NOT responsible for:
    * deciding what a *good* schedule looks like — that's
      optimiser_tier1.py, and (stage 6) the naive baseline. This module
      only asks "given this schedule, what actually happens?" — it has no
      optimisation logic of its own, which is also why simulate() takes
      plain charge/discharge arrays rather than a Tier1Schedule: stage 6's
      naive baseline needs backtesting too, and shouldn't have to
      construct a Tier1Schedule just to be checked.
    * repairing a physically-invalid schedule — violations are reported
      (soc_bounds_violated, power_limits_violated,
      simultaneous_charge_discharge), consistent with the fail-loud
      precedent elsewhere in this project (ADR-005, ADR-008).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bess.config import Battery
from bess.optimiser_tier1 import Tier1Schedule

DT_HOURS = 0.5
DEFAULT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class BacktestResult:
    soc_kwh: np.ndarray  # length T + 1, independently recomputed
    cashflow: float  # independently recomputed total £ (same convention as the Tier 1 objective)
    soc_bounds_violated: bool
    power_limits_violated: bool
    simultaneous_charge_discharge: bool


def simulate(
    charge_kw: np.ndarray,
    discharge_kw: np.ndarray,
    prices: np.ndarray,
    battery: Battery,
    initial_soc_kwh: float,
) -> BacktestResult:
    """
    Independently recompute the SoC trajectory and cashflow for a given
    schedule, and flag whether it actually respects SoC bounds, power
    limits, and charge/discharge mutual exclusivity. Uses the same
    degradation convention as optimiser_tier1.py (discharge-only, ADR-009)
    since that's a modelling decision already made, not something this
    module re-litigates — only the arithmetic implementing it is separate.
    """
    if not (len(charge_kw) == len(discharge_kw) == len(prices)):
        raise ValueError(
            f"charge_kw ({len(charge_kw)}), discharge_kw ({len(discharge_kw)}), and "
            f"prices ({len(prices)}) must be the same length"
        )

    T = len(charge_kw)
    soc_kwh = np.empty(T + 1)
    soc_kwh[0] = initial_soc_kwh
    cashflow = 0.0

    for t in range(T):
        soc_kwh[t + 1] = (
            soc_kwh[t]
            + battery.eta_charge * charge_kw[t] * DT_HOURS
            - discharge_kw[t] * DT_HOURS / battery.eta_discharge
        )
        cashflow += (
            prices[t] * discharge_kw[t] * DT_HOURS
            - prices[t] * charge_kw[t] * DT_HOURS
            - battery.degradation_cost_per_kwh * discharge_kw[t] * DT_HOURS
        )

    soc_min_kwh = battery.soc_min * battery.capacity_kwh
    soc_max_kwh = battery.soc_max * battery.capacity_kwh
    charge_kw, discharge_kw = np.asarray(charge_kw), np.asarray(discharge_kw)

    return BacktestResult(
        soc_kwh=soc_kwh,
        cashflow=cashflow,
        soc_bounds_violated=bool(
            ((soc_kwh < soc_min_kwh - DEFAULT_TOLERANCE) | (soc_kwh > soc_max_kwh + DEFAULT_TOLERANCE)).any()
        ),
        power_limits_violated=bool(
            (charge_kw > battery.power_kw + DEFAULT_TOLERANCE).any()
            or (discharge_kw > battery.power_kw + DEFAULT_TOLERANCE).any()
        ),
        simultaneous_charge_discharge=bool(
            ((charge_kw > DEFAULT_TOLERANCE) & (discharge_kw > DEFAULT_TOLERANCE)).any()
        ),
    )


def assert_matches_lp(
    schedule: Tier1Schedule,
    prices: np.ndarray,
    battery: Battery,
    tolerance: float = DEFAULT_TOLERANCE,
) -> BacktestResult:
    """
    The correctness anchor: independently recompute `schedule`'s cashflow
    and SoC trajectory and assert they agree with the LP's own reported
    values to within `tolerance`. Raises AssertionError with a specific,
    descriptive message on any mismatch or constraint violation, rather
    than returning a bool a caller could silently ignore.
    """
    result = simulate(schedule.charge_kw, schedule.discharge_kw, prices, battery, schedule.soc_kwh[0])

    if result.soc_bounds_violated:
        raise AssertionError("backtest: SoC bounds violated in the given schedule")
    if result.power_limits_violated:
        raise AssertionError("backtest: power limits violated in the given schedule")
    if result.simultaneous_charge_discharge:
        raise AssertionError("backtest: simultaneous charge and discharge found in the given schedule")

    if not np.allclose(result.soc_kwh, schedule.soc_kwh, atol=tolerance):
        max_diff = float(np.max(np.abs(result.soc_kwh - schedule.soc_kwh)))
        raise AssertionError(f"backtest SoC trajectory disagrees with the LP's own SoC by up to {max_diff:.2e} kWh")

    if abs(result.cashflow - schedule.objective_value) > tolerance:
        raise AssertionError(
            f"backtest cashflow (£{result.cashflow:.6f}) disagrees with the LP's own "
            f"objective value (£{schedule.objective_value:.6f}) by more than £{tolerance}"
        )

    return result
