"""
results_tier2.py — Tier 2, part 4: backtest MPC over a held-out test
period, compare against the naive floor and Tier 1 ceiling computed on
that SAME period, and run the corrupted-forecast sanity check that is the
real trust check on everything built in Tier 2 so far.

Responsible for:
    * chronological_train_test_split(): splits a leakage-safe feature
      history into an earlier training chunk and a later, held-out test
      chunk — the forecaster only ever trains on the earlier chunk,
      avoiding the in-sample leakage of training and backtesting a
      forecaster on the exact period it's then scored against (ADR-017)
    * run_tier2_comparison(): trains the forecaster on the training
      chunk, backtests MPC on the held-out chunk, and computes the SAME
      metrics (cumulative profit, cycles/day) that Tier 1 (results.py)
      and naive (naive_baseline.py) already report — on the SAME
      held-out window, so the comparison is genuinely apples-to-apples
    * corrupted_forecast_check(): the trust check that matters most —
      does MPC's performance collapse toward/below the naive floor when
      the forecast is decoupled from the actual situation? If not,
      there's a leakage bug in the SoC handoff or the feature matrix.
    * plot_comparison(): naive <= MPC <= Tier 1 ceiling, as a bar chart

Deliberately NOT responsible for:
    * the MPC loop itself (mpc.py), the forecaster (forecaster.py), or
      Tier 1/naive (optimiser_tier1.py, naive_baseline.py, results.py) —
      this module only orchestrates and compares
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bess.config import Battery
from bess.forecaster import Tier2Forecaster, train_forecaster
from bess.mpc import MPCResult, run_mpc
from bess.optimiser_tier1 import DT_HOURS, solve_day
from bess.results import run_tier1_over_history


def chronological_train_test_split(
    features_df: pd.DataFrame, train_fraction: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a leakage-safe feature history (already sorted chronologically
    by build_features()) into an earlier training chunk and a later,
    held-out test chunk, by row position. The forecaster trains only on
    the earlier chunk; MPC is backtested only on the later one, so the
    headline comparison reflects genuine out-of-sample performance, not a
    model that has already seen the exact period it's being scored
    against (ADR-017).
    """
    split_idx = int(len(features_df) * train_fraction)
    return features_df.iloc[:split_idx].reset_index(drop=True), features_df.iloc[split_idx:].reset_index(drop=True)


@dataclass(frozen=True)
class Tier2Comparison:
    tier1_total_profit: float
    naive_total_profit: float
    mpc_total_profit: float
    mpc_pct_of_ceiling: float
    mpc_mean_cycles_per_day: float
    n_test_days: float
    mpc_result: MPCResult


def run_tier2_comparison(
    price_history: pd.DataFrame,
    features_df: pd.DataFrame,
    battery: Battery,
    horizon: int = 12,
    train_fraction: float = 0.8,
    boundary_soc: float = 0.5,
) -> Tier2Comparison:
    """
    Train the forecaster on the earlier `train_fraction` of the feature
    history, backtest MPC on the later, held-out remainder, and compare
    against Tier 1/naive computed on that SAME held-out date range.
    `price_history` is the raw canonical price data (results.py's Tier 1
    runner needs this shape, not the feature matrix).
    """
    train_features, test_features = chronological_train_test_split(features_df, train_fraction)
    forecaster = train_forecaster(train_features, horizons=range(1, horizon + 1))

    initial_soc_kwh = boundary_soc * battery.capacity_kwh
    mpc_result = run_mpc(test_features, forecaster, battery, horizon=horizon, initial_soc_kwh=initial_soc_kwh)

    test_start = test_features["timestamp_utc"].iloc[0]
    test_end = test_features["timestamp_utc"].iloc[-1]
    test_price_history = price_history[
        (price_history["timestamp_utc"] >= test_start) & (price_history["timestamp_utc"] <= test_end)
    ].sort_values("timestamp_utc")
    if len(test_price_history) != len(test_features):
        raise ValueError(
            f"test_price_history ({len(test_price_history)} rows) and test_features "
            f"({len(test_features)} rows) must cover exactly the same periods"
        )

    # naive floor: the existing per-day-cyclic naive strategy — a simple
    # reference, not a tight bound, so its own daily structure doesn't
    # need to match MPC's continuous one the way the ceiling does below.
    tier1_results = run_tier1_over_history(test_price_history, battery, boundary_soc=boundary_soc)
    naive_total = sum(r.naive_profit for r in tier1_results.day_results)

    # TRUE ceiling: one non-cyclic solve over the WHOLE continuous test
    # window, perfect foresight, same starting SoC and same structural
    # freedom (no forced daily resets) as MPC. Comparing MPC against a
    # per-day-cyclic Tier 1 total (as an earlier draft did) understates
    # the true ceiling whenever there's value in carrying SoC across a day
    # boundary — MPC can do that, a cyclic-per-day Tier 1 cannot, so it
    # isn't actually an upper bound on MPC. Found empirically (a synthetic
    # test genuinely produced MPC > that "ceiling"), not assumed — ADR-017.
    ceiling_schedule = solve_day(
        test_price_history["price_gbp_per_kwh"].to_numpy(), battery, cyclic=False, initial_soc_kwh=initial_soc_kwh
    )
    tier1_total = ceiling_schedule.objective_value

    usable_capacity_kwh = (battery.soc_max - battery.soc_min) * battery.capacity_kwh
    n_test_days = len(test_features) * DT_HOURS / 24
    mpc_discharged_kwh = float(mpc_result.discharge_kw.sum() * DT_HOURS)
    mpc_mean_cycles_per_day = (mpc_discharged_kwh / usable_capacity_kwh) / n_test_days if n_test_days else 0.0

    return Tier2Comparison(
        tier1_total_profit=tier1_total,
        naive_total_profit=naive_total,
        mpc_total_profit=mpc_result.cashflow,
        mpc_pct_of_ceiling=mpc_result.cashflow / tier1_total if tier1_total else float("nan"),
        mpc_mean_cycles_per_day=mpc_mean_cycles_per_day,
        n_test_days=n_test_days,
        mpc_result=mpc_result,
    )


class _ShuffledForecaster:
    """
    Wraps a real, trained Tier2Forecaster, substituting a randomly-chosen
    OTHER row's features for whatever row is actually asked about — the
    corrupted-forecast sanity check (ADR-017). Real, same-distribution
    predictions from the real trained model, just conditioned on the
    wrong moment in time, so a suspiciously good MPC result here can't be
    explained away as "unrealistic corrupted inputs" — if it stays good,
    there's a leakage bug elsewhere (the SoC handoff, or the feature
    matrix), not a working forecaster earning its keep.
    """

    def __init__(self, real_forecaster: Tier2Forecaster, reference_features: pd.DataFrame, seed: int = 0):
        self.real_forecaster = real_forecaster
        self.reference_features = reference_features
        self._rng = np.random.default_rng(seed)

    def predict(self, X: pd.DataFrame, horizons=None) -> pd.DataFrame:
        random_rows = self.reference_features.sample(n=len(X), random_state=self._rng)
        return self.real_forecaster.predict(random_rows, horizons=horizons)


def corrupted_forecast_check(
    test_features: pd.DataFrame,
    forecaster: Tier2Forecaster,
    battery: Battery,
    horizon: int = 12,
    initial_soc_kwh: float | None = None,
    seed: int = 0,
) -> MPCResult:
    """Run MPC with predictions decoupled from the actual situation and
    return the result — see _ShuffledForecaster."""
    corrupted = _ShuffledForecaster(real_forecaster=forecaster, reference_features=test_features, seed=seed)
    return run_mpc(test_features, corrupted, battery, horizon=horizon, initial_soc_kwh=initial_soc_kwh)


def plot_comparison(naive_total: float, mpc_total: float, tier1_total: float, output_path: str) -> None:
    """naive <= MPC <= Tier 1 ceiling, as a bar chart, with % of ceiling captured as the punchline."""
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Naive", "MPC (Tier 2)", "Tier 1\n(perfect foresight)"]
    values = [naive_total, mpc_total, tier1_total]
    bars = ax.bar(labels, values, color=["tab:gray", "tab:blue", "tab:green"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"£{value:.2f}", ha="center", va="bottom")
    pct = 100 * mpc_total / tier1_total if tier1_total else float("nan")
    ax.set_ylabel("total profit over held-out test period (£)")
    ax.set_title(f"MPC captures {pct:.0f}% of the Tier 1 ceiling")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
