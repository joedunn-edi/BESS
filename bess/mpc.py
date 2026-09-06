"""
mpc.py — Tier 2's centrepiece: a rolling-horizon (model predictive
control) controller. At every period it re-solves a short-horizon version
of the Tier 1 LP using FORECASTED prices, takes only the first period's
decision, and advances using the REAL realised price and the REAL,
simulator-computed SoC — never the LP's own forecast-based SoC estimate.

Responsible for:
    * run_mpc(): the rolling-horizon loop over a full leakage-safe
      feature history (features.py's output)
    * the SoC handoff discipline: the SoC carried from one solve to the
      next is always backtest.simulate()'s own recomputed value from the
      REAL action taken against the REAL realised price, never read back
      from the LP's internal soc_kwh trajectory (which reflects the
      FORECAST, not reality). Getting this backwards is the single most
      likely way real foresight could quietly leak into the result — see
      ADR-016.

Deliberately NOT responsible for:
    * producing the forecasts themselves (forecaster.py) — this module
      only calls forecaster.predict() and trusts its leakage guarantee
    * evaluating the resulting schedule's overall profitability or
      running the corrupted-forecast sanity check — that's
      results_tier2.py, which is the actual trust check on this module
    * the inner LP formulation — reuses optimiser_tier1.solve_day()
      unchanged (with cyclic=False and an explicit initial_soc_kwh, both
      added to that function specifically for this reuse, ADR-016)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bess.backtest import simulate
from bess.config import Battery
from bess.forecaster import Tier2Forecaster
from bess.optimiser_tier1 import solve_day


@dataclass(frozen=True)
class MPCResult:
    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    soc_kwh: np.ndarray  # length T + 1 — the REAL trajectory, from simulate(), never the LP's own
    cashflow: float  # against REAL realised prices, never the forecast


def run_mpc(
    features_df: pd.DataFrame,
    forecaster: Tier2Forecaster,
    battery: Battery,
    horizon: int = 12,
    initial_soc_kwh: float | None = None,
) -> MPCResult:
    """
    Roll forward one period at a time over features_df (features.py's
    leakage-safe output). At each row t:

      1. predict the next `horizon` periods' prices using only features
         known as of t (forecaster.predict(), leakage-safe by ADR-014)
      2. solve a non-cyclic, `horizon`-period Tier 1 LP over that
         FORECAST, starting from the battery's REAL current SoC
      3. take only the first period's charge/discharge decision — the
         rest of the horizon's plan is discarded, re-solved fresh next
         period against an updated forecast
      4. advance the REAL SoC and REAL cashflow via backtest.simulate(),
         against the REAL realised price at t — never the forecast price,
         and never the LP's own soc_kwh[1] from step 2

    `forecaster` must have a trained model for every horizon 1..`horizon`
    (Tier2Forecaster.predict() raises if one is missing, rather than
    silently falling back to a shorter horizon).
    """
    T = len(features_df)
    if initial_soc_kwh is None:
        initial_soc_kwh = 0.5 * battery.capacity_kwh

    charge_kw = np.zeros(T)
    discharge_kw = np.zeros(T)
    soc_kwh = np.empty(T + 1)
    soc_kwh[0] = initial_soc_kwh
    cashflow = 0.0

    horizons = list(range(1, horizon + 1))
    real_prices = features_df["price_per_kwh"].to_numpy()

    for t in range(T):
        forecast_prices = forecaster.predict(features_df.iloc[[t]], horizons=horizons).iloc[0].to_numpy()

        schedule = solve_day(
            forecast_prices,
            battery,
            cyclic=False,
            initial_soc_kwh=soc_kwh[t],
        )

        charge_kw[t] = schedule.charge_kw[0]
        discharge_kw[t] = schedule.discharge_kw[0]

        # advance using the REAL realised price and the REAL simulator —
        # never schedule.soc_kwh[1], which reflects the FORECAST
        step_result = simulate(
            charge_kw[t : t + 1],
            discharge_kw[t : t + 1],
            real_prices[t : t + 1],
            battery,
            soc_kwh[t],
        )
        soc_kwh[t + 1] = step_result.soc_kwh[1]
        cashflow += step_result.cashflow

    return MPCResult(charge_kw=charge_kw, discharge_kw=discharge_kw, soc_kwh=soc_kwh, cashflow=cashflow)
