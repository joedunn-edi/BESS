"""
test_naive_baseline.py — tests for bess/naive_baseline.py.

Covers: a hand-computed example (the same scenario used in the stage-4
Tier 1 test, so the two can be directly compared), the exact-cyclic
boundary landing, the "sits below the Tier 1 ceiling" claim itself both
by hand and against real cached data, and the guard against a battery too
large to cycle within one day.
"""

import numpy as np
import pandas as pd
import pytest

from bess.backtest import simulate
from bess.config import Battery
from bess.naive_baseline import solve_day_naive
from bess.optimiser_tier1 import solve_day


def _battery(**overrides) -> Battery:
    defaults = dict(capacity_kwh=10.0, power_kw=10.0, round_trip_eff=1.0, soc_min=0.0, soc_max=1.0)
    defaults.update(overrides)
    return Battery(**defaults)


# --- hand-computed example, same scenario as the Tier 1 anchor test -----------


def test_matches_hand_computed_four_period_example():
    # same setup as test_optimiser_tier1's hand-computed test: cheap,
    # expensive, cheap, expensive. One full cycle only needs 1 charge + 1
    # discharge period here (capacity=10, power=10 -> 5kWh headroom moved
    # in exactly one full-power period each way) — so the naive baseline
    # can only capture ONE of the two available price swings, unlike Tier 1.
    prices = np.array([0.05, 0.20, 0.05, 0.20])
    battery = _battery(degradation_cost_per_kwh=0.0)

    schedule = solve_day_naive(prices, battery, boundary_soc=0.5)
    result = simulate(schedule.charge_kw, schedule.discharge_kw, prices, battery, initial_soc_kwh=5.0)

    np.testing.assert_allclose(schedule.charge_kw, [10, 0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(schedule.discharge_kw, [0, 10, 0, 0], atol=1e-9)
    assert result.cashflow == pytest.approx(0.75, abs=1e-9)  # 0.20*5 - 0.05*5
    assert result.soc_kwh[0] == pytest.approx(5.0)
    assert result.soc_kwh[-1] == pytest.approx(5.0)


def test_sits_strictly_below_tier1_ceiling_on_hand_computed_example():
    prices = np.array([0.05, 0.20, 0.05, 0.20])
    battery = _battery(degradation_cost_per_kwh=0.0)

    naive_schedule = solve_day_naive(prices, battery, boundary_soc=0.5)
    naive_result = simulate(naive_schedule.charge_kw, naive_schedule.discharge_kw, prices, battery, 5.0)

    tier1_schedule = solve_day(prices, battery, boundary_soc=0.5)

    # Tier 1 (£1.50) captures both cheap/expensive pairs; naive (£0.75)
    # only has budget for one full cycle and misses the second swing.
    assert naive_result.cashflow == pytest.approx(0.75, abs=1e-9)
    assert tier1_schedule.objective_value == pytest.approx(1.50, abs=1e-9)
    assert naive_result.cashflow < tier1_schedule.objective_value


# --- cyclic boundary and physical validity --------------------------------------


def test_naive_schedule_is_exactly_cyclic():
    prices = np.array([0.10, 0.30, 0.05, 0.25, 0.15, 0.35, 0.02, 0.40])
    battery = _battery(round_trip_eff=0.85)

    schedule = solve_day_naive(prices, battery, boundary_soc=0.4)
    boundary_kwh = 0.4 * battery.capacity_kwh
    result = simulate(schedule.charge_kw, schedule.discharge_kw, prices, battery, boundary_kwh)

    assert result.soc_kwh[0] == pytest.approx(boundary_kwh)
    assert result.soc_kwh[-1] == pytest.approx(boundary_kwh, abs=1e-6)


def test_naive_never_charges_and_discharges_same_period():
    prices = np.array([0.10, 0.30, 0.05, 0.25, 0.15, 0.35, 0.02, 0.40])
    battery = _battery(power_kw=5.0, round_trip_eff=0.85)

    schedule = solve_day_naive(prices, battery, boundary_soc=0.5)
    result = simulate(schedule.charge_kw, schedule.discharge_kw, prices, battery, 5.0)

    assert not result.simultaneous_charge_discharge
    assert not result.soc_bounds_violated
    assert not result.power_limits_violated


def test_raises_when_battery_cannot_cycle_within_one_day():
    battery = _battery(capacity_kwh=1000.0, power_kw=1.0)  # would need >>48 periods
    with pytest.raises(ValueError, match="full cycle"):
        solve_day_naive(np.random.default_rng(0).uniform(0.05, 0.3, 48), battery)


# --- against real cached data ----------------------------------------------------


def test_naive_sits_below_tier1_on_real_data():
    df = pd.read_parquet("data/day_ahead.parquet")
    battery = _battery(
        capacity_kwh=100.0, power_kw=50.0, round_trip_eff=0.9, soc_min=0.05, soc_max=0.95, degradation_cost_per_kwh=0.01
    )

    for _, day in df.groupby("settlement_date"):
        day = day.sort_values("settlement_period")
        prices = day["price_gbp_per_kwh"].to_numpy()

        tier1_schedule = solve_day(prices, battery, boundary_soc=0.5)
        naive_schedule = solve_day_naive(prices, battery, boundary_soc=0.5)
        naive_result = simulate(
            naive_schedule.charge_kw, naive_schedule.discharge_kw, prices, battery, tier1_schedule.soc_kwh[0]
        )

        assert naive_result.cashflow <= tier1_schedule.objective_value + 1e-6
