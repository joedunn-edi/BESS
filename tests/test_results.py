"""
test_results.py — tests for bess/results.py.

Uses small synthetic multi-day price histories (built the same way as
test_pipeline.py's _day_frame helper) rather than the full real cache, so
these run fast and offline. Covers: cumulative P&L aggregation, cycles/day,
the annualised £/kWh figure, that a bad day is excluded rather than
crashing the whole run, and the boundary_soc sensitivity sweep.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from bess.config import Battery
from bess.results import boundary_soc_sensitivity, plot_example_day, run_tier1_over_history
from bess.schema import settlement_day_utc_bounds, validate


def _day_frame(d: date, prices: list[float], source: str = "test") -> pd.DataFrame:
    start_utc, _ = settlement_day_utc_bounds(d)
    n = len(prices)
    timestamps = [start_utc + timedelta(minutes=30 * i) for i in range(n)]
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(timestamps, utc=True),
            "settlement_date": pd.Series([pd.Timestamp(d)] * n, dtype="datetime64[ns]"),
            "settlement_period": np.arange(1, n + 1, dtype="int64"),
            "price_gbp_per_kwh": np.array(prices, dtype="float64"),
            "source": source,
        }
    )
    return validate(df)


def _battery(**overrides) -> Battery:
    defaults = dict(capacity_kwh=10.0, power_kw=10.0, round_trip_eff=0.9, soc_min=0.0, soc_max=1.0, degradation_cost_per_kwh=0.01)
    defaults.update(overrides)
    return Battery(**defaults)


def _multi_day_history(prices_by_day: dict[date, list[float]]) -> pd.DataFrame:
    return pd.concat([_day_frame(d, prices) for d, prices in prices_by_day.items()], ignore_index=True)


# --- aggregation -----------------------------------------------------------------


def test_cumulative_pnl_is_running_total_in_date_order():
    history = _multi_day_history(
        {
            date(2026, 1, 1): [0.05, 0.20, 0.05, 0.20],
            date(2026, 1, 2): [0.10, 0.10, 0.10, 0.10],  # flat price -> zero profit
        }
    )
    battery = _battery(round_trip_eff=1.0, degradation_cost_per_kwh=0.0)

    results = run_tier1_over_history(history, battery, boundary_soc=0.5)

    assert list(results.cumulative_pnl.index) == [date(2026, 1, 1), date(2026, 1, 2)]
    assert results.cumulative_pnl.iloc[0] == pytest.approx(1.50, abs=1e-6)
    # flat day adds ~0 profit, so cumulative barely moves
    assert results.cumulative_pnl.iloc[1] == pytest.approx(1.50, abs=1e-6)
    assert not results.failed_days


def test_cycles_per_day_and_annualised_figure_are_positive_and_sane():
    history = _multi_day_history(
        {
            date(2026, 1, 1): [0.05, 0.20, 0.05, 0.20],
            date(2026, 1, 2): [0.02, 0.30, 0.02, 0.30],
        }
    )
    battery = _battery()

    results = run_tier1_over_history(history, battery, boundary_soc=0.5)

    assert results.mean_cycles_per_day > 0
    # 2 days sampled -> annualised figure scales profit-per-kWh-capacity by 365/2
    total_profit = sum(r.tier1_profit for r in results.day_results)
    expected = (total_profit / battery.capacity_kwh) * (365 / 2)
    assert results.annualised_gbp_per_kwh_capacity == pytest.approx(expected, rel=1e-9)


def test_naive_cumulative_pnl_never_exceeds_tier1():
    history = _multi_day_history(
        {
            date(2026, 1, 1): [0.05, 0.20, 0.05, 0.20],
            date(2026, 1, 2): [0.02, 0.30, 0.10, 0.25],
        }
    )
    battery = _battery()

    results = run_tier1_over_history(history, battery, boundary_soc=0.5)

    assert (results.cumulative_pnl_naive.to_numpy() <= results.cumulative_pnl.to_numpy() + 1e-6).all()


# --- failure isolation -------------------------------------------------------------


def test_one_failing_day_is_excluded_and_recorded_not_fatal(monkeypatch):
    # force solve_day to fail for exactly one of three days, to prove
    # run_tier1_over_history isolates the failure instead of crashing the
    # whole run — mirrors how pipeline.py isolates a bad day (ADR-008).
    good_days = [date(2026, 1, 1), date(2026, 1, 3)]
    bad_day = date(2026, 1, 2)
    history = _multi_day_history(
        {
            good_days[0]: [0.05, 0.20, 0.05, 0.20],
            bad_day: [0.10, 0.15, 0.05, 0.25],
            good_days[1]: [0.02, 0.30, 0.10, 0.25],
        }
    )
    battery = _battery()

    import bess.results as results_module

    real_solve_day = results_module.solve_day

    def flaky_solve_day(prices, battery, boundary_soc=0.5):
        if prices[0] == 0.10:  # only the bad day starts at this price
            raise RuntimeError("simulated solver failure")
        return real_solve_day(prices, battery, boundary_soc=boundary_soc)

    monkeypatch.setattr(results_module, "solve_day", flaky_solve_day)

    results = run_tier1_over_history(history, battery, boundary_soc=0.5)

    assert [r.settlement_date for r in results.day_results] == good_days
    assert len(results.failed_days) == 1
    assert results.failed_days[0][0] == bad_day
    assert "simulated solver failure" in results.failed_days[0][1]


# --- sensitivity sweep -------------------------------------------------------------


def test_boundary_soc_sensitivity_returns_all_requested_values():
    history = _multi_day_history(
        {
            date(2026, 1, 1): [0.05, 0.20, 0.05, 0.20],
            date(2026, 1, 2): [0.02, 0.30, 0.10, 0.25],
        }
    )
    battery = _battery()

    sensitivity = boundary_soc_sensitivity(history, battery, values=(0.25, 0.5, 0.75))

    assert set(sensitivity.keys()) == {0.25, 0.5, 0.75}
    assert all(isinstance(v, float) for v in sensitivity.values())


# --- plotting (smoke test) --------------------------------------------------------


def test_plot_example_day_produces_a_file(tmp_path):
    history = _day_frame(date(2026, 1, 1), [0.05, 0.20, 0.05, 0.20])
    battery = _battery(degradation_cost_per_kwh=0.0)
    results = run_tier1_over_history(history, battery, boundary_soc=0.5)

    output_path = tmp_path / "example_day.png"
    plot_example_day(results.day_results[0], battery, boundary_soc=0.5, output_path=str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0
