# BESS — battery energy-storage arbitrage

A final-year dissertation project: fetch GB electricity price data,
optimise a battery's charge/discharge schedule against it, and backtest
the result against an independent simulator. Built stage by stage, with
every modelling judgement recorded in [DECISIONS.md](DECISIONS.md).

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
   pipeline.py          fetch -> validate -> repair half-hourly grid
        |                (DST-aware: 46/48/50 periods/day) -> cache (parquet)
        v                -> data-quality report (missing periods, negative
   [ cached parquet ]      price frequency, price distribution)
        |
        v
   optimiser_tier1.py   LP (PuLP/CBC), perfect-foresight price vector
        |                -> maximises arbitrage profit minus degradation
        v
   [ charge/discharge schedule ]
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
    schema.py           canonical data contract + validate()
    config.py           Battery dataclass (hardware/economic parameters)
    sources_elexon.py   Elexon BMRS fetchers (imbalance + day-ahead)
    pipeline.py         fetch -> validate -> repair -> cache    [stage 3]
    optimiser_tier1.py  LP scheduler, perfect foresight         [stage 4]
    backtest.py         independent SoC/cashflow simulator      [stage 5]
tests/
    test_schema.py
    test_config.py
    test_sources_elexon.py
    fixtures/           recorded real API responses used by the tests above
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
[ADR-006](DECISIONS.md#adr-006-pandas-pinned-to-30-installed-233).

## Status

- [x] Stage 1 — contracts (`schema.py`, `config.py`)
- [x] Stage 2 — fetchers (`sources_elexon.py`)
- [ ] Stage 3 — pipeline (`pipeline.py`)
- [ ] Stage 4 — Tier 1 optimiser (`optimiser_tier1.py`)
- [ ] Stage 5 — backtester (`backtest.py`)
- [ ] Stage 6 — naive baseline
- [ ] Stage 7 — results (P&L, cycles/day, plot)
- [ ] Stage 8 — tests + CI
