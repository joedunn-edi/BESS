"""
optimiser_tier1.py — Tier 1 optimiser: a linear program that schedules one
day of battery charge/discharge against a known price vector, assuming
perfect foresight of that day's prices.

Responsible for:
    * solve_day(): builds and solves the LP/MILP for one settlement day
    * the SoC dynamics, power limits, mutual-exclusivity, and cyclic
      end-of-day constraints that define what "a valid schedule" means here
    * Tier1Schedule, the per-period result (charge/discharge/SoC) that
      backtest.py independently re-derives cashflow from, as a correctness
      check (stage 5)

Deliberately NOT responsible for:
    * fetching/caching price data (pipeline.py), or choosing which series
      to run against — day-ahead (APXMIDP) is used for the main results,
      since it's realistically knowable in advance, unlike imbalance price
      (see ADR-009)
    * looping this over many days or aggregating results — that's stage 7
    * verifying its own answer — a solver checking its own arithmetic isn't
      an independent check; that's the whole point of backtest.py

Formulation (see ADR-009 for the reasoning behind each modelling choice):

    Indices:      t = 1..T                    (periods in the day: 46/48/50)
    Decisions:    charge_kw[t]      >= 0       (continuous, kW)
                  discharge_kw[t]   >= 0       (continuous, kW)
                  is_charging[t]    in {0, 1}  (binary — mutual exclusivity)
                  soc[t]                       (kWh, t = 0..T)

    Objective:    maximise
        sum_t [ price[t] * discharge_kw[t] * dt
                - price[t] * charge_kw[t] * dt
                - degradation_cost_per_kwh * discharge_kw[t] * dt ]

    Degradation is charged on discharged energy only (ADR-009): published
    cycle-life figures are conventionally stated in terms of discharged
    throughput ("rated for N full cycles"), so a degradation_cost_per_kwh
    calibrated from such a figure and then also charged on the charge leg
    would double-count wear relative to its own source, unless
    independently halved to compensate.

    Subject to, for each t = 1..T:
        charge_kw[t]    <= power_kw * is_charging[t]
        discharge_kw[t] <= power_kw * (1 - is_charging[t])
        soc[t] = soc[t-1] + eta_charge * charge_kw[t] * dt
                           - discharge_kw[t] * dt / eta_discharge
        soc_min_kwh <= soc[t] <= soc_max_kwh

    Boundary condition:
        soc[0] = boundary_soc * capacity_kwh   (fixed, not a decision — a
                                                 free boundary would let the
                                                 LP quietly inflate profit by
                                                 idealising a condition a
                                                 real operator can't choose
                                                 for free each night)
        soc[T] = soc[0]                        (cyclic, per the brief)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pulp

from bess.config import Battery

DT_HOURS = 0.5


@dataclass(frozen=True)
class Tier1Schedule:
    """One day's solved schedule. charge_kw/discharge_kw/soc_kwh are the
    LP's own decision-variable values, and objective_value is its own
    reported optimum — backtest.py recomputes both independently rather
    than trusting these at face value."""

    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    soc_kwh: np.ndarray  # length T + 1: soc_kwh[0] is the boundary condition
    objective_value: float
    status: str


def solve_day(
    prices: np.ndarray,
    battery: Battery,
    boundary_soc: float = 0.5,
    cyclic: bool = True,
    initial_soc_kwh: float | None = None,
    dt_hours: float = DT_HOURS,
) -> Tier1Schedule:
    """
    Solve the Tier 1 MILP for a price vector (£/kWh, one entry per
    settlement period, in chronological order). See the module docstring
    for the full formulation.

    dt_hours (default 0.5, GB's half-hourly settlement period) is each
    period's real duration — pass the period's own period_minutes/60 for a
    market with a different native granularity (e.g. 1.0 for ERCOT's
    hourly DAM). Every existing call site leaves this at the default, so
    this parameter changes no prior GB behaviour.

    boundary_soc (fraction of capacity_kwh, default 0.5) is the shared,
    fixed start-of-day/end-of-day SoC — not a free decision variable. 0.5
    is both the midpoint of the usable SoC range and a commonly cited
    healthy resting charge level for lithium-ion cells (minimising both
    over-charge and over-discharge stress). This is flagged in ADR-009 for
    a sensitivity check (e.g. 0.25/0.5/0.75) over the full cached history in
    stage 7, rather than treated as beyond question.

    cyclic (default True) requires soc[T] == soc[0] — Tier 1's whole-day
    assumption. mpc.py (Tier 2) solves short, partial-day windows against
    forecasted prices and must NOT force a return to any particular SoC
    mid-window; only a genuine day boundary would justify that, and MPC
    doesn't currently impose one. Every existing call site leaves this at
    the default, so this parameter changes no prior behaviour (ADR-016).

    initial_soc_kwh, if given, overrides boundary_soc for the STARTING
    condition only, in absolute kWh rather than a fraction of capacity —
    used by mpc.py to hand the LP the battery's real, simulator-computed
    current SoC directly, without a needless fraction<->kWh round trip
    (config.py's fraction-based SoC vs the LP's own kWh-based decision
    variables have already been a real source of confusion once — see the
    stage-4 quiz). If cyclic=True and initial_soc_kwh is given, the end
    condition targets *this* value, not boundary_soc*capacity_kwh.
    """
    T = len(prices)
    soc_min_kwh = battery.soc_min * battery.capacity_kwh
    soc_max_kwh = battery.soc_max * battery.capacity_kwh
    start_soc_kwh = boundary_soc * battery.capacity_kwh if initial_soc_kwh is None else initial_soc_kwh

    # a small tolerance, not a strict bounds check: mpc.py chains many
    # solve_day() calls, each seeded from the previous call's real,
    # simulator-computed SoC (see ADR-016) — floating-point drift across
    # many iterations can produce a value like -1e-15 that is genuinely
    # at the boundary but not bit-exact. Tier 1's own single-solve-per-day
    # usage never chains calls this way, so this never surfaced there.
    _BOUNDARY_TOLERANCE = 1e-6
    if not (soc_min_kwh - _BOUNDARY_TOLERANCE <= start_soc_kwh <= soc_max_kwh + _BOUNDARY_TOLERANCE):
        raise ValueError(
            f"starting SoC {start_soc_kwh:.2f} kWh is outside "
            f"[soc_min, soc_max] = [{soc_min_kwh:.2f}, {soc_max_kwh:.2f}] kWh"
        )
    # clip to the exact bounds before handing to the LP: soc[0]'s own
    # declared bounds are [soc_min_kwh, soc_max_kwh] exactly, and a
    # within-tolerance-but-marginally-outside value (e.g. -1e-15) pinned
    # via an equality constraint could make the LP itself infeasible even
    # though the tolerance check above accepted it.
    start_soc_kwh = min(max(start_soc_kwh, soc_min_kwh), soc_max_kwh)

    problem = pulp.LpProblem("tier1_arbitrage", pulp.LpMaximize)

    charge = pulp.LpVariable.dicts("charge_kw", range(T), lowBound=0, upBound=battery.power_kw)
    discharge = pulp.LpVariable.dicts("discharge_kw", range(T), lowBound=0, upBound=battery.power_kw)
    is_charging = pulp.LpVariable.dicts("is_charging", range(T), cat="Binary")
    soc = pulp.LpVariable.dicts("soc_kwh", range(T + 1), lowBound=soc_min_kwh, upBound=soc_max_kwh)

    problem += soc[0] == start_soc_kwh
    if cyclic:
        problem += soc[T] == soc[0]

    for t in range(T):
        problem += charge[t] <= battery.power_kw * is_charging[t]
        problem += discharge[t] <= battery.power_kw * (1 - is_charging[t])
        problem += (
            soc[t + 1]
            == soc[t] + battery.eta_charge * charge[t] * dt_hours - discharge[t] * dt_hours / battery.eta_discharge
        )

    problem += pulp.lpSum(
        prices[t] * discharge[t] * dt_hours
        - prices[t] * charge[t] * dt_hours
        - battery.degradation_cost_per_kwh * discharge[t] * dt_hours
        for t in range(T)
    )

    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[problem.status]
    if status != "Optimal":
        raise RuntimeError(f"Tier 1 LP did not solve to optimality (solver status: {status})")

    return Tier1Schedule(
        charge_kw=np.array([charge[t].value() for t in range(T)]),
        discharge_kw=np.array([discharge[t].value() for t in range(T)]),
        soc_kwh=np.array([soc[t].value() for t in range(T + 1)]),
        objective_value=pulp.value(problem.objective),
        status=status,
    )
