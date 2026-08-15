"""
naive_baseline.py — the naive arbitrage baseline: charge during the
cheapest periods of the day, discharge during the priciest, sized to
exactly one full daily cycle for this battery (boundary SoC -> soc_max ->
boundary SoC again). Exists to give Tier 1's profit a floor to beat — the
brief's own framing is "confirm it sits below the Tier 1 ceiling."

Responsible for:
    * solve_day_naive(): builds a charge_kw/discharge_kw schedule via this
      simple price-sorting rule — no optimisation, no solver involved

Deliberately NOT responsible for:
    * evaluating the schedule's cashflow/SoC trajectory — that's
      backtest.simulate(), reused unchanged (it was built in stage 5 to
      take plain arrays specifically so this stage wouldn't need its own
      duplicate evaluation logic)
    * being a good schedule — it's deliberately naive, a floor for
      comparison, not a competitor to Tier 1

Formulation:
    N_charge and N_discharge are derived independently from the battery's
    own physical limits, not a single shared "N" (see ADR-011): round-trip
    efficiency means moving a given amount of *stored* energy takes a
    different number of full-power periods depending on direction —
    charging loses energy on the way in, so adding X kWh of stored energy
    takes MORE full-power periods than removing the same X kWh via
    discharging, which loses energy on the way out instead:

        headroom_kwh  = soc_max_kwh - boundary_soc_kwh
        N_charge      = ceil(headroom_kwh / (eta_charge * power_kw * dt))
        N_discharge   = ceil(headroom_kwh / (power_kw * dt / eta_discharge))

    The N_charge cheapest periods are used for charging: full power on all
    but the priciest period within that set, which takes whatever partial
    power exactly fills the remaining headroom — landing exactly back on
    the boundary SoC, matching Tier 1's own cyclic constraint. Symmetric
    logic (full power on all but the cheapest-within-the-set) for the
    N_discharge priciest periods on the discharge side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bess.config import Battery

DT_HOURS = 0.5


@dataclass(frozen=True)
class NaiveSchedule:
    charge_kw: np.ndarray
    discharge_kw: np.ndarray


def solve_day_naive(prices: np.ndarray, battery: Battery, boundary_soc: float = 0.5) -> NaiveSchedule:
    """
    Build the naive charge-cheapest/discharge-priciest schedule for one day.
    See the module docstring for how N_charge/N_discharge are derived.
    """
    T = len(prices)
    prices = np.asarray(prices)
    soc_min_kwh = battery.soc_min * battery.capacity_kwh
    soc_max_kwh = battery.soc_max * battery.capacity_kwh
    boundary_soc_kwh = boundary_soc * battery.capacity_kwh
    if not (soc_min_kwh <= boundary_soc_kwh <= soc_max_kwh):
        raise ValueError(
            f"boundary_soc={boundary_soc} ({boundary_soc_kwh:.2f} kWh) is outside "
            f"[soc_min, soc_max] = [{soc_min_kwh:.2f}, {soc_max_kwh:.2f}] kWh"
        )

    headroom_kwh = soc_max_kwh - boundary_soc_kwh
    charge_full_step = battery.eta_charge * battery.power_kw * DT_HOURS
    discharge_full_step = battery.power_kw * DT_HOURS / battery.eta_discharge

    n_charge = math.ceil(headroom_kwh / charge_full_step) if headroom_kwh > 0 else 0
    n_discharge = math.ceil(headroom_kwh / discharge_full_step) if headroom_kwh > 0 else 0

    if n_charge + n_discharge > T:
        raise ValueError(
            f"battery needs {n_charge} charge + {n_discharge} discharge periods to complete "
            f"one full cycle, but the day only has {T} periods — reduce capacity_kwh or "
            f"increase power_kw"
        )

    charge_kw = np.zeros(T)
    discharge_kw = np.zeros(T)

    if n_charge > 0:
        cheapest = np.argsort(prices)[:n_charge]  # ascending: cheapest first
        remaining_kwh = headroom_kwh
        for i, t in enumerate(cheapest):
            if i < n_charge - 1:
                charge_kw[t] = battery.power_kw
                remaining_kwh -= charge_full_step
            else:
                charge_kw[t] = remaining_kwh / (battery.eta_charge * DT_HOURS)

    if n_discharge > 0:
        priciest = np.argsort(-prices)[:n_discharge]  # descending: priciest first
        remaining_kwh = headroom_kwh
        for i, t in enumerate(priciest):
            if i < n_discharge - 1:
                discharge_kw[t] = battery.power_kw
                remaining_kwh -= discharge_full_step
            else:
                discharge_kw[t] = remaining_kwh * battery.eta_discharge / DT_HOURS

    return NaiveSchedule(charge_kw=charge_kw, discharge_kw=discharge_kw)
