"""
test_backtest.py — tests for bess/backtest.py.

Covers: a hand-computed simulate() check that never touches solve_day()
(true independence), the actual LP-vs-simulator cross-check across several
battery/price scenarios (the correctness anchor the brief asks for), and
each constraint-violation detector on deliberately-broken schedules.
"""

import numpy as np
import pytest

from bess.backtest import assert_matches_lp, simulate
from bess.config import Battery
from bess.optimiser_tier1 import solve_day


def _battery(**overrides) -> Battery:
    defaults = dict(capacity_kwh=10.0, power_kw=10.0, round_trip_eff=1.0, soc_min=0.0, soc_max=1.0)
    defaults.update(overrides)
    return Battery(**defaults)


# --- simulate() in isolation, no dependency on solve_day() ---------------------


def test_simulate_matches_hand_computed_example():
    # same scenario as the stage-4 hand-computed LP test, but here the
    # schedule is supplied directly — simulate() never calls solve_day().
    charge_kw = np.array([10.0, 0.0, 10.0, 0.0])
    discharge_kw = np.array([0.0, 10.0, 0.0, 10.0])
    prices = np.array([0.05, 0.20, 0.05, 0.20])
    battery = _battery(degradation_cost_per_kwh=0.0)

    result = simulate(charge_kw, discharge_kw, prices, battery, initial_soc_kwh=5.0)

    np.testing.assert_allclose(result.soc_kwh, [5, 10, 5, 10, 5], atol=1e-9)
    assert result.cashflow == pytest.approx(1.50, abs=1e-9)
    assert not result.soc_bounds_violated
    assert not result.power_limits_violated
    assert not result.simultaneous_charge_discharge


def test_simulate_raises_on_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        simulate(np.array([1.0]), np.array([1.0, 2.0]), np.array([0.1, 0.1]), _battery(), 5.0)


# --- constraint-violation detection ---------------------------------------------


def test_detects_soc_bounds_violation():
    # charging far beyond capacity
    charge_kw = np.array([10.0, 10.0, 10.0])
    discharge_kw = np.zeros(3)
    prices = np.full(3, 0.10)
    battery = _battery(capacity_kwh=10.0, soc_min=0.0, soc_max=1.0)

    result = simulate(charge_kw, discharge_kw, prices, battery, initial_soc_kwh=5.0)
    assert result.soc_bounds_violated


def test_detects_power_limit_violation():
    charge_kw = np.array([20.0])  # battery's power_kw is 10
    discharge_kw = np.array([0.0])
    result = simulate(charge_kw, discharge_kw, np.array([0.1]), _battery(power_kw=10.0), initial_soc_kwh=5.0)
    assert result.power_limits_violated


def test_detects_simultaneous_charge_discharge():
    charge_kw = np.array([5.0])
    discharge_kw = np.array([5.0])
    result = simulate(charge_kw, discharge_kw, np.array([0.1]), _battery(), initial_soc_kwh=5.0)
    assert result.simultaneous_charge_discharge


# --- the correctness anchor: LP vs independent simulator ------------------------


@pytest.mark.parametrize(
    "prices,battery_kwargs,boundary_soc",
    [
        (np.array([0.10, 0.30, 0.05, 0.25, 0.15, 0.35, 0.02, 0.40]), dict(degradation_cost_per_kwh=0.01), 0.5),
        (np.array([0.05, 0.40, 0.05, 0.40]), dict(power_kw=3.0), 0.5),
        (np.full(8, 0.15), dict(round_trip_eff=0.9, degradation_cost_per_kwh=0.0), 0.5),
        (np.array([0.10, 0.11, 0.10, 0.11]), dict(degradation_cost_per_kwh=1.0), 0.5),
        (np.array([0.05, 0.30, 0.05, 0.30, 0.05, 0.30]), dict(round_trip_eff=0.81), 0.3),
        (np.array([0.12, 0.09, 0.20, 0.07, 0.18, 0.11]), dict(soc_min=0.1, soc_max=0.9), 0.7),
    ],
)
def test_assert_matches_lp_across_scenarios(prices, battery_kwargs, boundary_soc):
    battery = _battery(**battery_kwargs)
    schedule = solve_day(prices, battery, boundary_soc=boundary_soc)

    result = assert_matches_lp(schedule, prices, battery)  # raises on any disagreement

    assert result.cashflow == pytest.approx(schedule.objective_value, abs=1e-6)
    np.testing.assert_allclose(result.soc_kwh, schedule.soc_kwh, atol=1e-6)


def test_assert_matches_lp_raises_on_genuine_disagreement():
    prices = np.array([0.05, 0.20, 0.05, 0.20])
    battery = _battery(degradation_cost_per_kwh=0.0)
    schedule = solve_day(prices, battery, boundary_soc=0.5)

    # corrupt the LP's own reported objective so the two genuinely disagree
    tampered = schedule.__class__(
        charge_kw=schedule.charge_kw,
        discharge_kw=schedule.discharge_kw,
        soc_kwh=schedule.soc_kwh,
        objective_value=schedule.objective_value + 100.0,
        status=schedule.status,
    )
    with pytest.raises(AssertionError, match="disagrees"):
        assert_matches_lp(tampered, prices, battery)
