"""
test_tier2_mpc.py — tests for bess/mpc.py.

The hand-computable scenario is the priority test here (same "verify by
hand before trusting real data" discipline as Stage 4's hand-computed LP
test): a fake forecaster with fixed, known predictions makes the inner
LP's decisions fully predictable, isolating the rolling-horizon loop's
own mechanics from forecaster accuracy, which is already covered
separately in test_tier2_forecaster.py.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from bess.backtest import simulate
from bess.config import Battery
from bess.features import build_features
from bess.forecaster import Tier2Forecaster
from bess.mpc import run_mpc
from bess.schema import settlement_day_utc_bounds, validate


def _battery(**overrides) -> Battery:
    defaults = dict(capacity_kwh=10.0, power_kw=10.0, round_trip_eff=1.0, soc_min=0.0, soc_max=1.0, degradation_cost_per_kwh=0.0)
    defaults.update(overrides)
    return Battery(**defaults)


class _FixedModel:
    """A fake per-horizon model that always predicts the same value,
    regardless of the input row — isolates MPC's loop mechanics from
    forecaster accuracy for hand-computable scenarios."""

    def __init__(self, value: float):
        self.value = value

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value)


def _synthetic_price_history(n_days: int, price_fn) -> pd.DataFrame:
    start_date = date(2026, 1, 1)
    rows = []
    idx = 0
    for day_offset in range(n_days):
        d = start_date + timedelta(days=day_offset)
        start_utc, _ = settlement_day_utc_bounds(d)
        for period in range(1, 49):
            rows.append(
                {
                    "timestamp_utc": start_utc + timedelta(minutes=30 * (period - 1)),
                    "settlement_date": pd.Timestamp(d),
                    "settlement_period": period,
                    "period_minutes": 30,
                    "price_per_kwh": price_fn(idx),
                    "currency": "GBP",
                    "source": "test",
                }
            )
            idx += 1
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["settlement_period"] = df["settlement_period"].astype("int64")
    df["period_minutes"] = df["period_minutes"].astype("int64")
    df["price_per_kwh"] = df["price_per_kwh"].astype("float64")
    return validate(df)


def _tiny_features(prices: list[float]) -> pd.DataFrame:
    """A minimal 2-column-of-interest feature frame for MPC tests — real
    features.py output, but only price_per_kwh's real values matter
    here since the fake forecaster ignores every feature column."""
    n_days = 8  # enough real days that build_features' warm-up drop still leaves rows
    rng = np.random.default_rng(0)
    filler = rng.uniform(0.05, 0.30, n_days * 48)

    def price_fn(i):
        # overwrite the final len(prices) periods with the caller's exact
        # hand-chosen sequence; everything before is warm-up filler
        offset = n_days * 48 - len(prices)
        return prices[i - offset] if i >= offset else filler[i]

    df = _synthetic_price_history(n_days, price_fn)
    return build_features(df).tail(len(prices)).reset_index(drop=True)


# --- the hand-computable scenario --------------------------------------------------


def test_hand_computable_two_period_scenario_reveals_static_forecast_deferral():
    # forecaster always predicts [cheap, expensive] regardless of when
    # it's asked — a fixed, unchanging 2-period-ahead outlook.
    #
    # The battery starts at SoC=5, already holding sellable energy. Given
    # a non-cyclic 2-period window forecast [0.05, 0.20], the optimal plan
    # is NOT "charge now, discharge next" (net profit 0.20*5 - 0.05*5 =
    # 0.75) — it's "don't charge at all, just sell the 5 kWh already held
    # at the expensive period" (net profit 0.20*5 = 1.00, strictly better).
    # Charging first is pure wasted cost: discharge power (10 kW) already
    # caps how much can be resold at 5 kWh regardless of buying more, so
    # the extra purchase could never be sold within this window anyway.
    # Verified directly against solve_day() before writing this test,
    # exactly the "verify by hand before trusting real data" discipline
    # used for the Tier 1 hand-computed test.
    forecaster = Tier2Forecaster(models={1: _FixedModel(0.05), 2: _FixedModel(0.20)}, feature_columns=["lag_1"])
    battery = _battery()  # capacity=10, power=10, perfect efficiency, no degradation
    features = _tiny_features([0.05, 0.20])  # the REAL realised prices for our 2 real periods

    result = run_mpc(features, forecaster, battery, horizon=2, initial_soc_kwh=5.0)

    # Because the plan is "do nothing in period 0 of the window, sell in
    # period 1," and MPC only ever executes period 0's action, it defers
    # the sale every single step — and since the forecaster never changes
    # its prediction, the real SoC never changes either, so t=1's solve is
    # IDENTICAL to t=0's and defers again. The profitable sale never
    # actually happens within this 2-real-period dataset: a genuine
    # property of rolling-horizon control fed a STATIC forecast, not a
    # bug. A real, time-varying forecaster (as trained in forecaster.py)
    # doesn't exhibit this, because its prediction genuinely changes as
    # real information arrives each step — confirmed separately by the
    # live full-year run producing real cycling activity, not paralysis.
    np.testing.assert_allclose(result.charge_kw, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result.discharge_kw, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result.soc_kwh, [5.0, 5.0, 5.0], atol=1e-6)
    assert result.cashflow == pytest.approx(0.0, abs=1e-6)


def test_liquidates_fully_within_a_non_cyclic_window_regardless_of_price_level():
    # non-cyclic means any energy still held at the END of the window is
    # simply worth nothing in that solve's objective — so the LP always
    # prefers to sell everything it physically can before the window
    # closes, even at a "cheap" forecast price, rather than hold stock
    # whose value isn't credited past the window edge. With SoC=10,
    # power=10 (5 kWh/period max) and forecast [0.20, 0.05], that means
    # discharging in BOTH periods (5+5=10 kWh total) — not just the
    # pricier one. Verified directly against solve_day() before writing
    # this assertion, not derived from intuition alone (an earlier draft
    # of this exact test got that intuition wrong on the first attempt).
    forecaster = Tier2Forecaster(models={1: _FixedModel(0.20), 2: _FixedModel(0.05)}, feature_columns=["lag_1"])
    battery = _battery()
    features = _tiny_features([0.20, 0.05])

    result = run_mpc(features, forecaster, battery, horizon=2, initial_soc_kwh=10.0)  # start full

    # t=0: sell 5 kWh at the REAL price 0.20 (matches this step's forecast)
    # t=1: SoC is now 5, forecast is STILL [0.20, 0.05] (static forecaster)
    #      — its own "period 0" now corresponds to real t=1, so it sells
    #      the remaining 5 kWh here too. The REAL price at t=1 is 0.05,
    #      not the 0.20 the LP's stale forecast assumed for "period 0 of
    #      its window" — real revenue reflects the REAL price, exactly
    #      the forecast-vs-reality gap MPC exists to be tested against.
    np.testing.assert_allclose(result.charge_kw, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result.discharge_kw, [10.0, 10.0], atol=1e-6)
    np.testing.assert_allclose(result.soc_kwh, [10.0, 5.0, 0.0], atol=1e-6)
    assert result.cashflow == pytest.approx(0.20 * 5 + 0.05 * 5, abs=1e-6)  # 1.25


# --- SoC handoff discipline: cross-check against an independent recomputation -------


def test_result_matches_independent_simulation_of_its_own_executed_actions():
    # the strongest available check on the SoC-handoff discipline itself:
    # feed a forecaster that's WRONG relative to the real prices (so the
    # LP's internal, forecast-based soc_kwh would differ from reality if
    # it were ever used), then confirm MPCResult's reported soc_kwh/
    # cashflow are recoverable by independently re-simulating its own
    # *executed* charge/discharge against the REAL prices — proving
    # run_mpc reported what actually happened, not what the LP predicted.
    forecaster = Tier2Forecaster(models={1: _FixedModel(0.30), 2: _FixedModel(0.30)}, feature_columns=["lag_1"])
    battery = _battery(round_trip_eff=0.9, degradation_cost_per_kwh=0.01)
    real_prices_list = [0.10, 0.05, 0.25, 0.08, 0.30, 0.02]
    features = _tiny_features(real_prices_list)

    result = run_mpc(features, forecaster, battery, horizon=2, initial_soc_kwh=5.0)

    independent = simulate(
        result.charge_kw, result.discharge_kw, np.array(real_prices_list), battery, initial_soc_kwh=5.0
    )
    np.testing.assert_allclose(result.soc_kwh, independent.soc_kwh, atol=1e-9)
    assert result.cashflow == pytest.approx(independent.cashflow, abs=1e-9)


def test_never_charges_and_discharges_in_the_same_period():
    forecaster = Tier2Forecaster(models={1: _FixedModel(0.15), 2: _FixedModel(0.15)}, feature_columns=["lag_1"])
    battery = _battery(round_trip_eff=0.85)
    real_prices_list = [0.10, 0.25, 0.05, 0.30, 0.12, 0.08]
    features = _tiny_features(real_prices_list)

    result = run_mpc(features, forecaster, battery, horizon=2, initial_soc_kwh=5.0)

    both_active = (result.charge_kw > 1e-6) & (result.discharge_kw > 1e-6)
    assert not both_active.any()


def test_defaults_initial_soc_to_half_capacity():
    forecaster = Tier2Forecaster(models={1: _FixedModel(0.10)}, feature_columns=["lag_1"])
    battery = _battery(capacity_kwh=20.0)
    features = _tiny_features([0.10, 0.10])

    result = run_mpc(features, forecaster, battery, horizon=1)

    assert result.soc_kwh[0] == pytest.approx(10.0)  # 0.5 * 20
