"""
evaluate_rtm_mpc.py — Tier 2, part 3 for ERCOT: backtest MPC against real
RTM+DAM data, compare against the RTM perfect-foresight ceiling and naive
baseline on the same held-out window, and run the corrupted-forecast
sanity check (ADR-017's trust check, repeated here since it's a new
market). Mirrors GB's own Tier 2 comparison (results_tier2.py), with
horizon=96 (24h at RTM's 15-minute periods, chosen because the forecaster
stayed accurate to 24h out — see ADR-020's Part 2) and dt_hours=0.25.
"""

import pandas as pd

from bess.config import Battery
from bess.features import build_features_with_dam
from bess.forecaster import train_forecaster
from bess.results_tier2 import chronological_train_test_split, corrupted_forecast_check, plot_comparison, run_tier2_comparison
from bess.sources_ercot import CHICAGO

HORIZON = 96  # 24h at RTM's 15-minute periods
DT_HOURS = 0.25

rtm_df = pd.read_parquet("data/ercot_rtm_west.parquet")
dam_df = pd.read_parquet("data/ercot_dam_west.parquet")
features = build_features_with_dam(rtm_df, dam_df, tz=CHICAGO)

battery = Battery(
    capacity_kwh=100, power_kw=50, round_trip_eff=0.9, soc_min=0.05, soc_max=0.95, degradation_cost_per_kwh=0.01
)

comparison = run_tier2_comparison(
    rtm_df, features, battery, horizon=HORIZON, train_fraction=0.8, boundary_soc=0.5, dt_hours=DT_HOURS
)

print(f"Tier 1 ceiling (RTM, perfect foresight): ${comparison.tier1_total_profit:.2f}")
print(f"MPC (Tier 2): ${comparison.mpc_total_profit:.2f} ({comparison.mpc_pct_of_ceiling * 100:.1f}% of ceiling)")
print(f"Naive: ${comparison.naive_total_profit:.2f}")
print(f"Mean cycles/day: {comparison.mpc_mean_cycles_per_day:.3f}")
print(f"n_test_days: {comparison.n_test_days:.1f}")

plot_comparison(
    comparison.naive_total_profit,
    comparison.mpc_total_profit,
    comparison.tier1_total_profit,
    "results/ercot_rtm_tier2_comparison.png",
    currency_symbol="$",
)
print("saved results/ercot_rtm_tier2_comparison.png")

train_features, test_features = chronological_train_test_split(features, train_fraction=0.8)
forecaster = train_forecaster(train_features, horizons=range(1, HORIZON + 1))
initial_soc_kwh = 0.5 * battery.capacity_kwh

corrupted_result = corrupted_forecast_check(
    test_features, forecaster, battery, horizon=HORIZON, initial_soc_kwh=initial_soc_kwh, dt_hours=DT_HOURS
)
print(f"\ncorrupted-forecast MPC cashflow: ${corrupted_result.cashflow:.2f} (naive: ${comparison.naive_total_profit:.2f})")
if corrupted_result.cashflow < comparison.naive_total_profit:
    print("collapses below naive when the forecast is decoupled from reality, as expected")
else:
    print("WARNING: did not collapse below naive — investigate before trusting the MPC result above")
