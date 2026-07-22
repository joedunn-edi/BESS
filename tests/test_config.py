"""
test_config.py — tests for bess/config.py's Battery dataclass.

Covers: the eta_charge/eta_discharge derivation (symmetric sqrt split,
and that the two legs multiply back to round_trip_eff), and the
__post_init__ validation guards on each parameter.
"""

import math

import pytest

from bess.config import Battery


def test_eta_split_multiplies_back_to_round_trip_eff():
    b = Battery(capacity_kwh=100, power_kw=50, round_trip_eff=0.81)
    assert b.eta_charge == pytest.approx(0.9)
    assert b.eta_discharge == pytest.approx(0.9)
    assert b.eta_charge * b.eta_discharge == pytest.approx(0.81)


def test_eta_split_is_symmetric():
    b = Battery(capacity_kwh=100, power_kw=50, round_trip_eff=0.86)
    assert b.eta_charge == b.eta_discharge == pytest.approx(math.sqrt(0.86))


@pytest.mark.parametrize("bad_rte", [0.0, -0.1, 1.1])
def test_rejects_out_of_range_round_trip_eff(bad_rte):
    with pytest.raises(ValueError, match="round_trip_eff"):
        Battery(capacity_kwh=100, power_kw=50, round_trip_eff=bad_rte)


@pytest.mark.parametrize("cap", [0, -10])
def test_rejects_non_positive_capacity(cap):
    with pytest.raises(ValueError, match="capacity_kwh"):
        Battery(capacity_kwh=cap, power_kw=50, round_trip_eff=0.9)


@pytest.mark.parametrize("power", [0, -5])
def test_rejects_non_positive_power(power):
    with pytest.raises(ValueError, match="power_kw"):
        Battery(capacity_kwh=100, power_kw=power, round_trip_eff=0.9)


@pytest.mark.parametrize("soc_min,soc_max", [(0.5, 0.5), (0.6, 0.4), (-0.1, 1.0), (0.0, 1.1)])
def test_rejects_invalid_soc_bounds(soc_min, soc_max):
    with pytest.raises(ValueError, match="soc_min"):
        Battery(capacity_kwh=100, power_kw=50, round_trip_eff=0.9, soc_min=soc_min, soc_max=soc_max)


def test_rejects_negative_degradation_cost():
    with pytest.raises(ValueError, match="degradation_cost_per_kwh"):
        Battery(capacity_kwh=100, power_kw=50, round_trip_eff=0.9, degradation_cost_per_kwh=-0.01)


def test_battery_is_frozen():
    b = Battery(capacity_kwh=100, power_kw=50, round_trip_eff=0.9)
    with pytest.raises(AttributeError):
        b.capacity_kwh = 200
