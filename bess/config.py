"""
config.py — the battery asset being optimised: hardware and economic
parameters, as a single frozen dataclass.

Responsible for:
    * `Battery`, describing capacity, power, efficiency, SoC bounds and
      degradation cost, plus the derived per-leg efficiencies used by the
      SoC update equation.

Deliberately NOT responsible for:
    * market/price data (see schema.py);
    * the SoC update equation itself, or how degradation cost enters the
      objective function — those live in optimiser_tier1.py / backtest.py,
      which both import eta_charge/eta_discharge from here so the two
      independent implementations can never drift apart on efficiency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Battery:
    """
    A single battery asset. All energy in kWh, power in kW, cost in £.

    soc_min / soc_max are fractions of capacity_kwh in [0, 1], not absolute
    kWh (ADR-004), so they stay meaningful across different capacities.

    degradation_cost_per_kwh is a £/kWh cost applied in the optimiser's
    objective; which energy flow it's charged against is decided in
    optimiser_tier1.py (not yet built) — this field just carries the number.
    """

    capacity_kwh: float
    power_kw: float
    round_trip_eff: float
    soc_min: float = 0.0
    soc_max: float = 1.0
    degradation_cost_per_kwh: float = 0.0

    def __post_init__(self) -> None:
        if self.capacity_kwh <= 0:
            raise ValueError(f"capacity_kwh must be > 0, got {self.capacity_kwh}")
        if self.power_kw <= 0:
            raise ValueError(f"power_kw must be > 0, got {self.power_kw}")
        if not (0 < self.round_trip_eff <= 1):
            raise ValueError(f"round_trip_eff must be in (0, 1], got {self.round_trip_eff}")
        if not (0 <= self.soc_min < self.soc_max <= 1):
            raise ValueError(
                f"require 0 <= soc_min < soc_max <= 1, got soc_min={self.soc_min}, soc_max={self.soc_max}"
            )
        if self.degradation_cost_per_kwh < 0:
            raise ValueError(f"degradation_cost_per_kwh must be >= 0, got {self.degradation_cost_per_kwh}")

    @property
    def eta_charge(self) -> float:
        """
        Charging efficiency, eta_c = sqrt(round_trip_eff) — symmetric split
        chosen because RTE alone doesn't tell us the per-leg split; see
        ADR-003. Used in the SoC update (optimiser_tier1.py / backtest.py):

            soc[t+1] = soc[t]
                       + eta_charge * charge_kw[t] * dt / capacity_kwh
                       - discharge_kw[t] * dt / (eta_discharge * capacity_kwh)

        eta_charge shrinks energy going INTO storage (grid kWh drawn is
        more than what ends up stored); eta_discharge shrinks energy
        coming OUT (storage gives up more than reaches the grid).
        """
        return math.sqrt(self.round_trip_eff)

    @property
    def eta_discharge(self) -> float:
        """See eta_charge — same value, other leg of the cycle."""
        return math.sqrt(self.round_trip_eff)
