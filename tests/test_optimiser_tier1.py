"""
test_optimiser_tier1.py — tests for bess/optimiser_tier1.py.

Covers: a fully hand-computed example (the strongest correctness anchor
until backtest.py exists in stage 5), the mutual-exclusivity guarantee,
the cyclic boundary condition, SoC/power bounds, and basic economic sanity
checks (no free lunch from cycling at a flat price, degradation suppresses
marginal cycling, efficiency loss actually costs something).
"""

import numpy as np
import pytest

from bess.config import Battery
from bess.optimiser_tier1 import solve_day


def _battery(**overrides) -> Battery:
    defaults = dict(capacity_kwh=10.0, power_kw=10.0, round_trip_eff=1.0, soc_min=0.0, soc_max=1.0)
    defaults.update(overrides)
    return Battery(**defaults)


# --- hand-computed correctness anchor -----------------------------------------


def test_matches_hand_computed_four_period_example():
    # perfect efficiency, no degradation, power/capacity generous enough to
    # hit the price pattern exactly: cheap, expensive, cheap, expensive.
    # Optimal: charge fully (5kWh) each cheap period, discharge fully (5kWh)
    # each expensive period, ending back at the 5kWh boundary.
    prices = np.array([0.05, 0.20, 0.05, 0.20])
    battery = _battery(degradation_cost_per_kwh=0.0)

    result = solve_day(prices, battery, boundary_soc=0.5)

    assert result.status == "Optimal"
    assert result.objective_value == pytest.approx(1.50, abs=1e-6)
    np.testing.assert_allclose(result.charge_kw, [10, 0, 10, 0], atol=1e-6)
    np.testing.assert_allclose(result.discharge_kw, [0, 10, 0, 10], atol=1e-6)
    np.testing.assert_allclose(result.soc_kwh, [5, 10, 5, 10, 5], atol=1e-6)


# --- mutual exclusivity ---------------------------------------------------------


def test_never_charges_and_discharges_in_the_same_period():
    prices = np.array([0.10, 0.30, 0.05, 0.25, 0.15, 0.35, 0.02, 0.40])
    battery = _battery(degradation_cost_per_kwh=0.01)

    result = solve_day(prices, battery)

    both_active = (result.charge_kw > 1e-6) & (result.discharge_kw > 1e-6)
    assert not both_active.any()


# --- cyclic boundary and bounds --------------------------------------------------


def test_cyclic_boundary_condition_holds():
    prices = np.array([0.10, 0.30, 0.05, 0.25])
    battery = _battery()

    result = solve_day(prices, battery, boundary_soc=0.3)

    assert result.soc_kwh[0] == pytest.approx(0.3 * battery.capacity_kwh)
    assert result.soc_kwh[-1] == pytest.approx(result.soc_kwh[0], abs=1e-6)


def test_soc_stays_within_bounds():
    prices = np.array([0.10, 0.30, 0.05, 0.25, 0.15, 0.35])
    battery = _battery(soc_min=0.1, soc_max=0.9)

    result = solve_day(prices, battery, boundary_soc=0.5)

    soc_min_kwh = 0.1 * battery.capacity_kwh
    soc_max_kwh = 0.9 * battery.capacity_kwh
    assert (result.soc_kwh >= soc_min_kwh - 1e-6).all()
    assert (result.soc_kwh <= soc_max_kwh + 1e-6).all()


def test_respects_power_limits():
    prices = np.array([0.05, 0.40, 0.05, 0.40])
    battery = _battery(power_kw=3.0)

    result = solve_day(prices, battery)

    assert (result.charge_kw <= 3.0 + 1e-6).all()
    assert (result.discharge_kw <= 3.0 + 1e-6).all()


def test_raises_when_boundary_soc_outside_bounds():
    battery = _battery(soc_min=0.2, soc_max=0.8)
    with pytest.raises(ValueError, match="boundary_soc"):
        solve_day(np.array([0.1, 0.2]), battery, boundary_soc=0.9)


# --- economic sanity checks ------------------------------------------------------


def test_flat_price_with_efficiency_loss_means_no_cycling():
    # cycling at a single flat price with round-trip losses is a pure loss
    # (you pay for lost energy with nothing to show for it) — optimal is to
    # do nothing.
    prices = np.full(8, 0.15)
    battery = _battery(round_trip_eff=0.9, degradation_cost_per_kwh=0.0)

    result = solve_day(prices, battery)

    assert result.objective_value == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(result.charge_kw, np.zeros(8), atol=1e-6)
    np.testing.assert_allclose(result.discharge_kw, np.zeros(8), atol=1e-6)


def test_degradation_cost_can_suppress_marginal_cycling():
    # a small price spread that's barely profitable with zero degradation
    # cost should stop being worth cycling for once degradation cost is
    # large enough to exceed the spread.
    prices = np.array([0.10, 0.11, 0.10, 0.11])
    battery_no_deg = _battery(degradation_cost_per_kwh=0.0)
    battery_high_deg = _battery(degradation_cost_per_kwh=1.0)

    cheap_case = solve_day(prices, battery_no_deg)
    expensive_case = solve_day(prices, battery_high_deg)

    assert cheap_case.objective_value > 0
    assert expensive_case.objective_value == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(expensive_case.charge_kw, np.zeros(4), atol=1e-6)


def test_perfect_efficiency_never_worse_than_imperfect():
    prices = np.array([0.05, 0.30, 0.05, 0.30, 0.05, 0.30])
    perfect = _battery(round_trip_eff=1.0)
    lossy = _battery(round_trip_eff=0.81)  # eta_charge = eta_discharge = 0.9

    perfect_result = solve_day(prices, perfect)
    lossy_result = solve_day(prices, lossy)

    assert perfect_result.objective_value >= lossy_result.objective_value - 1e-6
