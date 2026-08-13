# BESS — battery energy-storage arbitrage

Fetch GB electricity price data, optimise a battery's charge/discharge
schedule against it, and backtest the result against an independent
simulator. Built stage by stage, with every modelling judgement recorded
in [DECISIONS.md](DECISIONS.md).

## Conventions (locked, never deviate)

| Quantity | Unit |
|---|---|
| Energy | kWh |
| Power | kW |
| Price | £/kWh |
| Time step | 0.5 h (half-hourly) |
| Profit | £ |

Elexon publishes prices in £/MWh — fetchers divide by 1000 on ingest so
nothing downstream ever sees £/MWh.

## Architecture / data flow

```
sources_elexon.py   (Elexon BMRS API: imbalance + day-ahead prices)
        |
        v
   [ raw records ]
        |
        v
   schema.validate()   <-- hard gate: canonical columns, dtypes, no NaNs,
        |                   no naive timestamps, no duplicate periods
        v
   pipeline.py          fetch each day -> diff against the DST-aware
        |                expected grid (46/48/50 periods/day) -> cache
        v                (parquet) -> data-quality report (missing periods,
   [ cached parquet ]      negative price frequency, price distribution).
                          Gaps are only ever reported, never fabricated —
                          see ADR-008.
        |
        v
   optimiser_tier1.py   MILP (PuLP/CBC) on day-ahead (APXMIDP) prices,
        |                one day at a time, perfect foresight of that day
        v                -> maximises discharge revenue - charge cost -
   [ charge/discharge        discharge-side degradation cost, subject to
     schedule for 1 day]     SoC dynamics, power limits, no simultaneous
        |                    charge+discharge, and a fixed cyclic
        |                    start/end SoC (50%, see ADR-009)
        v
        |
        v
   backtest.py          independent SoC/cashflow simulator
        |                -> cross-checked against the LP's own objective
        v                   value as a correctness anchor
   [ P&L, cycles/day, £/kWh-capacity/year, example-day plot ]
```

`config.py` (the `Battery` dataclass) is read by both `optimiser_tier1.py`
and `backtest.py`, so hardware assumptions (capacity, power, efficiency
split, degradation cost) can never drift apart between the two independent
implementations.

## Canonical dataset

Every price DataFrame that has passed `schema.validate()` has exactly
these columns, in this order:

| Column | Type | Meaning |
|---|---|---|
| `timestamp_utc` | tz-aware UTC | instant marking the *start* of the half-hour |
| `settlement_date` | date (Europe/London) | the settlement day this period belongs to — **not** derivable from `timestamp_utc` alone, see [ADR-001](DECISIONS.md#adr-001-utc-as-the-canonical-instant-europelondon-as-the-canonical-calendar-day) |
| `settlement_period` | int | 1–48 normally; 46 on the spring clock-change day, 50 on the autumn one |
| `price_gbp_per_kwh` | float | already converted from Elexon's £/MWh |
| `source` | str | which fetcher/API produced this row |

## Project layout

```
bess/
    schema.py           canonical data contract + validate() + full_grid()
    config.py           Battery dataclass (hardware/economic parameters)
    sources_elexon.py   Elexon BMRS fetchers (imbalance + day-ahead)
    pipeline.py         fetch -> gap-report -> cache (parquet)
    optimiser_tier1.py  MILP scheduler, one day, perfect foresight
    backtest.py         independent SoC/cashflow simulator      [stage 5]
tests/
    test_schema.py
    test_config.py
    test_sources_elexon.py
    test_pipeline.py
    test_optimiser_tier1.py
    fixtures/           recorded real API responses used by test_sources_elexon.py
data/                   parquet cache (gitignored — regenerable via pipeline.py)
DECISIONS.md            ADR log — every modelling choice, alternatives weighed
README.md               this file
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Python is pinned to 3.12 (not the system's 3.14) for stable wheel
availability across pandas/PuLP/pyarrow. pandas is pinned `<3.0` — see
[ADR-006](DECISIONS.md#adr-006-pandas-pinned-to-30-installed-233). PuLP is
pinned `<4.0` for the same reason — see
[ADR-009](DECISIONS.md#adr-009-tier-1-lp--day-ahead-prices-forbid-simultaneous-chargedischarge-fixed-cyclic-soc-discharge-only-degradation).

## Status

- [x] Stage 1 — contracts (`schema.py`, `config.py`)
- [x] Stage 2 — fetchers (`sources_elexon.py`)
- [x] Stage 3 — pipeline (`pipeline.py`)
- [x] Stage 4 — Tier 1 optimiser (`optimiser_tier1.py`)
- [ ] Stage 5 — backtester (`backtest.py`)
- [ ] Stage 6 — naive baseline
- [ ] Stage 7 — results (P&L, cycles/day, plot)
- [ ] Stage 8 — tests + CI
