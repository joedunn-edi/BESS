# BESS — battery energy-storage arbitrage

[![CI](https://github.com/joedunn-edi/BESS/actions/workflows/ci.yml/badge.svg)](https://github.com/joedunn-edi/BESS/actions/workflows/ci.yml)

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
        v                    start/end SoC (50%, see ADR-009)
   backtest.py          independently recomputes SoC + cashflow from the
        |                schedule with fresh arithmetic (not shared with
        v                the LP) -> cross-checked to a 1e-6 tolerance as
   [ agreement verified]    the correctness anchor, see ADR-010
        |
        v
   naive_baseline.py    charge-cheapest/discharge-priciest floor, one full
        |                cycle sized to this battery -> also run through
        v                backtest.simulate() for a comparable £ figure
   [ Tier1 vs naive £ ]     see ADR-011
        |
        v
   results.py           runs Tier 1 (+ naive, + the LP/backtest cross
                          -check) over the full cached history, per day,
                          isolating any failing day rather than aborting
                          the batch -> cumulative P&L, cycles/day,
                          annualised £/kWh-capacity/year, one example-day
                          plot. See ADR-012 and the Results section below.
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

## Results

Tier 1 run over a full year of real day-ahead prices (2025-08-14 to
2026-08-13, 365/365 days solved, every day cross-checked against the
independent backtester with zero failures). Battery: 100 kWh / 50 kW /
90% round-trip efficiency / SoC 5-95% / £0.01 per kWh degradation
(discharge-referenced), boundary_soc=0.5.

| Metric | Tier 1 | Naive baseline |
|---|---|---|
| Cumulative annual profit | £1563.72 | £880.90 (56% of Tier 1) |
| Mean cycles/day | 1.368 | (capped at 1 cycle/day by construction) |
| £ per kWh of capacity per year | £15.64 | — |

The boundary_soc=50% assumption (ADR-009) turned out **not** to be
insensitive: sweeping 25/50/75% gave £1650.32 / £1563.72 / £1420.55 — a
16.2% spread, with lower boundary values winning because they leave more
headroom before `soc_max`. Recorded as a genuine finding in ADR-009, not
smoothed over.

Example day (2026-06-23, the most profitable day found — a summer day
with a large evening price spike over £550/MWh): regenerate with
`bess.results.plot_example_day()`, saved to `results/example_day.png`
(gitignored, regenerable — not committed).

Full reasoning for every metric definition and the failure-isolation
approach: [ADR-012](DECISIONS.md#adr-012-resultspy--discharge-based-cyclesday-per-day-failure-isolation-a-full-year-fetched-for-credibility).

### GB vs ERCOT — same Tier 1 optimiser, same battery, two real markets

Same `solve_day()`/`backtest.py`, same battery, run against a trailing
year of real ERCOT DAM (HB_WEST) prices instead of GB day-ahead —
`dt_hours` generalised from each row's own `period_minutes` (1.0 for
ERCOT's hourly settlement vs 0.5 for GB's half-hourly), see ADR-019:

| Metric | GB (2025-08-14 to 2026-08-13) | ERCOT HB_WEST (2025-09-07 to 2026-09-06) |
|---|---|---|
| Days solved | 365/365 | 364/365 (one day, 2026-03-07, entirely missing from ERCOT's own API — not a bug here) |
| Cumulative annual profit | £1563.72 | $1412.91 |
| Naive baseline | £880.90 | $860.15 |
| Uplift over naive | 77.5% | 64.3% |
| Mean cycles/day | 1.368 | 1.299 |
| Per kWh of capacity per year | £15.64 | $14.17 |
| Negative-price frequency | rare | 7.38% of periods |

**Not a currency-adjusted comparison** — both runs use the identical
`degradation_cost_per_kwh=0.01` figure with no GBP/USD conversion, and the
two windows don't overlap in calendar time, so the closeness of £15.64 vs
$14.17 is not evidence the two markets are equally profitable once FX is
accounted for. What the comparison *does* show cleanly: ERCOT is
genuinely more volatile (real negative prices 7.38% of the time vs GB's
rarity, and a real scarcity spike to $2000.02/MWh — GB never approaches
that), yet Tier 1's uplift over naive is *lower* on ERCOT (64.3% vs
77.5%), suggesting HB_WEST's price shape rewards simple
charge-cheapest/discharge-priciest timing more than GB's does, relative
to the extra room a perfect-foresight optimiser has to exploit.

### Tier 2 — MPC with a learned forecast, no perfect foresight

Same battery, same real data, but the optimiser only ever sees a
LightGBM-forecasted price path, never the real future. An 80/20
chronological split trains the forecaster on the earlier ~80% of the year
and backtests MPC only on the later, held-out ~72 days it never saw
during training:

| Strategy | Total profit (held-out 71.6 days) |
|---|---|
| Tier 1 ceiling (perfect foresight, same window) | £485.43 |
| **MPC (Tier 2)** | **£335.62 (69.1% of ceiling)** |
| Naive baseline | £243.04 |
| MPC with a deliberately corrupted forecast | **-£308.53** |

![Tier 1 vs MPC vs naive](results/tier2_comparison.png)

MPC clearly beats naive (~38% more profit) while giving up about 31% of
the theoretical ceiling to the fact that it can't actually see the
future — a believable, honest result for a realistic controller. The
corrupted-forecast row is the sanity check that matters most: feeding MPC
a forecast deliberately decoupled from reality doesn't just make it worse
than naive, it goes sharply negative — a broken forecast actively costs
money rather than mysteriously still working, which is the strongest
available evidence against a leak in the SoC handoff or feature matrix.

The Tier 1 "ceiling" here is *not* the per-day-cyclic number from the
table above — it's a single non-cyclic solve over the whole continuous
test window, matching MPC's own structural freedom to carry SoC across a
day boundary. An earlier version of this comparison used the per-day
number directly and a test caught MPC legitimately exceeding it — full
story in
[ADR-017](DECISIONS.md#adr-017-results_tier2py--chronological-traintest-split-and-a-same-structure-ceiling).

## Project layout

```
bess/
    schema.py           canonical data contract + validate() + full_grid()
    config.py           Battery dataclass (hardware/economic parameters)
    sources_elexon.py   Elexon BMRS fetchers (imbalance + day-ahead), GBP
    sources_ercot.py    ERCOT DAM/RTM fetchers (HB_WEST), USD          [new — see ADR-018]
    pipeline.py         fetch -> gap-report -> cache (parquet), multi-market
    optimiser_tier1.py  MILP scheduler, one day, perfect foresight
    backtest.py         independent SoC/cashflow simulator, cross-checks the LP
    naive_baseline.py   charge-cheapest/discharge-priciest floor for comparison
    results.py          runs Tier 1 over cached history, metrics, example-day plot
    features.py          Tier 2: leakage-safe lag/calendar/rolling feature matrix
    forecaster.py         Tier 2: LightGBM price forecaster, one model per horizon
    mpc.py                 Tier 2: rolling-horizon controller, forecast-driven
    results_tier2.py       Tier 2: MPC backtest, ceiling/floor comparison, plot
tests/
    test_schema.py
    test_config.py
    test_sources_elexon.py
    test_sources_ercot.py
    test_pipeline.py
    test_optimiser_tier1.py
    test_backtest.py
    test_naive_baseline.py
    test_results.py
    test_tier2_features.py
    test_tier2_forecaster.py
    test_tier2_mpc.py
    test_tier2_results.py
    fixtures/           recorded real API responses used by test_sources_elexon.py
data/                   parquet cache (gitignored — regenerable via pipeline.py)
results/                generated plots (gitignored — regenerable via results.py)
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

**macOS only:** LightGBM (Tier 2) needs the OpenMP runtime, which isn't
installed by default with Homebrew Python — `pip install` succeeds but
`import lightgbm` fails with a `dlopen`/`Library not loaded` error until
you run:

```bash
brew install libomp
```

## Status

- [x] Stage 1 — contracts (`schema.py`, `config.py`)
- [x] Stage 2 — fetchers (`sources_elexon.py`)
- [x] Stage 3 — pipeline (`pipeline.py`)
- [x] Stage 4 — Tier 1 optimiser (`optimiser_tier1.py`)
- [x] Stage 5 — backtester (`backtest.py`)
- [x] Stage 6 — naive baseline (`naive_baseline.py`)
- [x] Stage 7 — results (`results.py`)
- [x] Stage 8 — tests + CI (`.github/workflows/ci.yml`)

**Tier 2 (rolling-horizon MPC controller, using a learned forecast instead of perfect foresight):**

- [x] Part 1 — features (`features.py`) — see [ADR-014](DECISIONS.md#adr-014-featurespy--drop-warm-up-rows-rather-than-impute-and-a-black-box-leakage-guard)
- [x] Part 2 — forecaster (`forecaster.py`) — see [ADR-015](DECISIONS.md#adr-015-forecasterpy--direct-multi-horizon-models-and-a-real-degradation-finding)
- [x] Part 3 — MPC controller (`mpc.py`) — see [ADR-016](DECISIONS.md#adr-016-mpcpy--extending-solve_day-for-reuse-the-soc-handoff-discipline-and-a-static-forecast-deferral-finding)
- [x] Part 4 — results + corrupted-forecast sanity check (`results_tier2.py`) — see [ADR-017](DECISIONS.md#adr-017-results_tier2py--chronological-traintest-split-and-a-same-structure-ceiling)

**Multi-market extension — ERCOT (Texas), West Hub:**

- [x] Contracts + fetchers (`schema.py` generalised, `sources_ercot.py`, `pipeline.py` wrappers) — see [ADR-018](DECISIONS.md#adr-018-schemapy-multi-market-generalisation-and-choosing-hb_west-over-a-system-wide-average)
- [x] `sources_ercot.py` field names confirmed against live DAM/RTM responses from a real account (2026-09-07) — no `VERIFY` tags remain; see ADR-018's 2026-09-07 update
- [x] Tier 1 pointed at ERCOT DAM (HB_WEST), full real trailing year — see the GB vs ERCOT results table above and [ADR-019](DECISIONS.md#adr-019-tier-1-generalised-to-ercot-dt_hours-from-period_minutes-and-a-full-real-year-backfilled)
- [x] `features.py` generalised to any `period_minutes` (`lag_1_day`/`lag_1_week` replace GB-hardcoded `lag_48`/`lag_336`), plus `build_features_with_dam()` for ERCOT's DAM-price exogenous features — see [ADR-020](DECISIONS.md#adr-020-featurespy-generalised-for-ercot-rtm-plus-a-dam-price-exogenous-feature). Full real RTM year backfilled (`data/ercot_rtm_west.parquet`, 34798/35040 periods).
- [ ] Forecaster + MPC trained/backtested on real ERCOT RTM data — not started; `forecaster.py`'s training logic needs no changes, untested against real RTM data yet

**Tier 2 is now feature-complete: features → forecaster → MPC → backtest & sanity check, all built and verified against real data.**
