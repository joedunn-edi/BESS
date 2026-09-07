"""
compare_gb_ercot.py — comparative analysis of GB day-ahead vs ERCOT DAM
(HB_WEST) prices, both real cached data (no fetching). Prints distribution
stats and saves two plots to results/ (gitignored, regenerable).

Not a live comparison of the same calendar window (see ADR-019) — this is
about each market's own price *shape* and volatility, not a claim that
one is more profitable than the other in the same period.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def _stats(prices_per_mwh: pd.Series) -> dict:
    return {
        "mean": prices_per_mwh.mean(),
        "std": prices_per_mwh.std(),
        "cov": prices_per_mwh.std() / prices_per_mwh.mean(),
        "min": prices_per_mwh.min(),
        "max": prices_per_mwh.max(),
        "skew": prices_per_mwh.skew(),
        "kurtosis": prices_per_mwh.kurt(),
        "negative_pct": (prices_per_mwh < 0).mean() * 100,
        "p99": prices_per_mwh.quantile(0.99),
        "p999": prices_per_mwh.quantile(0.999),
    }


def main() -> None:
    gb = pd.read_parquet("data/day_ahead.parquet")
    ercot = pd.read_parquet("data/ercot_dam_west.parquet")

    gb_mwh = gb["price_per_kwh"] * 1000
    ercot_mwh = ercot["price_per_kwh"] * 1000

    for name, s in [("GB day-ahead", _stats(gb_mwh)), ("ERCOT DAM (HB_WEST)", _stats(ercot_mwh))]:
        print(f"--- {name} ---")
        for k, v in s.items():
            print(f"  {k}: {v:.3f}")
        print()

    gb = gb.copy()
    gb["hour"] = (gb["settlement_period"] - 1) // 2
    gb_by_hour = gb.groupby("hour")["price_per_kwh"].mean() * 1000

    ercot = ercot.copy()
    ercot["hour"] = ercot["settlement_period"] - 1
    ercot_by_hour = ercot.groupby("hour")["price_per_kwh"].mean() * 1000

    fig, (ax_hour, ax_dist) = plt.subplots(1, 2, figsize=(13, 5))

    ax_hour.plot(gb_by_hour.index, gb_by_hour.values, label="GB (£/MWh)", color="tab:blue")
    ax_hour_r = ax_hour.twinx()
    ax_hour_r.plot(ercot_by_hour.index, ercot_by_hour.values, label="ERCOT HB_WEST ($/MWh)", color="tab:red")
    ax_hour.set_xlabel("hour of day")
    ax_hour.set_ylabel("GB price (£/MWh)", color="tab:blue")
    ax_hour_r.set_ylabel("ERCOT price ($/MWh)", color="tab:red")
    ax_hour.set_title("Mean price by hour of day")
    lines1, labels1 = ax_hour.get_legend_handles_labels()
    lines2, labels2 = ax_hour_r.get_legend_handles_labels()
    ax_hour.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax_dist.hist(gb_mwh.clip(upper=300), bins=80, alpha=0.5, density=True, label="GB (£/MWh)", color="tab:blue")
    ax_dist.hist(ercot_mwh.clip(upper=300), bins=80, alpha=0.5, density=True, label="ERCOT ($/MWh)", color="tab:red")
    ax_dist.set_xlabel("price (clipped at 300, native currency)")
    ax_dist.set_ylabel("density")
    ax_dist.set_title("Price distribution (extreme tail clipped for readability)")
    ax_dist.legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "gb_ercot_comparison.png", dpi=120)
    plt.close(fig)
    print(f"saved {RESULTS_DIR / 'gb_ercot_comparison.png'}")


if __name__ == "__main__":
    main()
