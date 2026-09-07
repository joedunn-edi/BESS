# Decisions

Architecture Decision Records for modelling and engineering choices made in
this project. Each entry: context, the decision, alternatives weighed, and
consequences. Entries are added in the order decisions were made, not
renumbered later — if a later stage reverses one, a new entry supersedes it
rather than editing history.

---

## ADR-001: UTC as the canonical instant, Europe/London as the canonical calendar day

**Status:** Accepted (stage 1)

**Context:** GB electricity settlement runs on "settlement days" that follow
the Europe/London calendar, including its clock changes — not on UTC days.
A settlement day has 48 half-hour periods normally, but 46 on the day
clocks go forward (23h) and 50 on the day clocks go back (25h). We need one
representation that is unambiguous for arithmetic (sorting, joining,
duration) and one that matches how the market actually organises a day.

**Decision:** Store both, as separate columns. `timestamp_utc` is the
tz-aware UTC instant marking the start of the half-hour — unambiguous,
sortable, DST-proof, good for arithmetic. `settlement_date` is the
Europe/London calendar date the period belongs to — needed because it is
*not* recoverable by truncating `timestamp_utc` to a date (a period
starting 23:30 UTC in British Summer Time is 00:30 the next London day).

**Alternatives considered:**
- *Store only UTC, derive settlement_date on demand.* Rejected: correct
  derivation requires a timezone conversion at every use site, which is
  exactly the kind of repeated, easy-to-get-wrong logic a canonical schema
  should eliminate once, not push onto every downstream module.
- *Store only local (Europe/London) time.* Rejected: naive/local timestamps
  are ambiguous across the clock-change hour (on the autumn day, 01:30
  local time occurs twice) and don't sort correctly/uniquely without extra
  disambiguation — exactly the naive-timestamp problem the contract is
  designed to forbid.

**Consequences:** Every fetcher must supply both fields explicitly (no
derive-one-from-the-other convenience function is provided on purpose,
per ADR-005). Downstream code that needs "what calendar day is this" reads
`settlement_date` directly rather than reimplementing the London conversion.

---

## ADR-002: settlement_period range is checked via a DST-aware day-length calculation, not hardcoded

**Status:** Accepted (stage 1)

**Context:** `validate()` needs to reject an impossible `settlement_period`
(e.g. period 49 on a normal 48-period day) as a defence against off-by-one
and DST bugs in fetchers. The number of valid periods depends on whether
the given `settlement_date` is a clock-change day.

**Decision:** Compute the day length from first principles:
`Europe/London midnight -> next Europe/London midnight`, converted to true
UTC-instant duration, `* 2` for half-hours (`expected_period_count()` in
`schema.py`). This is correct for any year without hardcoding specific DST
switchover dates (which move a little year to year — last Sunday of March
/ October).

**A bug this surfaced during development, worth recording:** the first
implementation subtracted two `datetime` objects that shared the same
`ZoneInfo` instance directly (`end - start` where both had
`tzinfo=ZoneInfo("Europe/London")`). This is wrong: CPython's `datetime`
subtraction special-cases "both operands have the *same* tzinfo object" and
falls back to subtracting the naive wall-clock fields, skipping the
UTC-offset adjustment — on the (here false) assumption that identical
tzinfo means a constant offset. Across a DST boundary the offset isn't
constant, so this silently returned 24h on both clock-change days instead
of 23h/25h. The tests written alongside this function (`test_period_count_*`
in `test_schema.py`) caught it immediately. The fix: explicitly
`.astimezone(timezone.utc)` both datetimes before subtracting, forcing a
genuine UTC-instant diff. Kept here as a documented gotcha because it is
a very easy mistake to reintroduce.

**Alternatives considered:**
- *Hardcode a lookup table of UK clock-change dates.* Rejected: correct but
  needs yearly maintenance and is exactly the kind of magic-constant table
  that silently goes stale.
- *Use `pytz` instead of `zoneinfo`.* Rejected: `zoneinfo` is stdlib
  (Python 3.9+), needs no extra dependency, and is the currently
  recommended approach; `pytz`'s "localize" API is easier to misuse.

**Consequences:** `expected_period_count()` is a small, independently
testable pure function, exercised by three tests: a normal day, the spring
day (46), and the autumn day (50).

---

## ADR-003: symmetric efficiency split — eta_charge = eta_discharge = sqrt(round_trip_eff)

**Status:** Accepted (stage 1)

**Context:** A battery's round-trip efficiency (RTE) is normally quoted as
one number (energy out / energy in over a full cycle). The SoC update
needs two numbers — how much of the energy drawn from the grid while
charging actually gets stored, and how much of the stored energy delivered
while discharging actually reaches the grid.

**Decision:** Split the loss symmetrically: `eta_charge = eta_discharge =
sqrt(round_trip_eff)`, implemented as properties on `Battery`
(`bess/config.py`).

**Alternatives considered:**
- *Put all the loss on one leg* (e.g. `eta_charge = 1`,
  `eta_discharge = RTE`), which some simplified models do. Rejected: it's
  an arbitrary, unmotivated asymmetry with no physical basis when only an
  RTE figure is available — real losses occur in both directions
  (inverter, transformer, internal resistance both ways).
- *Use manufacturer per-leg datasheet figures* if available. Preferred
  *when available* — this project only has an aggregate RTE input, so this
  isn't currently exercised, but `Battery` could be extended to accept
  `eta_charge`/`eta_discharge` directly instead of deriving both from one
  number, without changing anything downstream (both consumers read the
  properties, not the raw field).

**Consequences:** `eta_charge * eta_discharge == round_trip_eff` exactly,
so a full charge-then-discharge cycle at rated power reproduces the quoted
RTE — a useful invariant, tested in `test_config.py`. Both
`optimiser_tier1.py` and `backtest.py` will import these properties (not
reimplement the sqrt) so the two independent implementations cannot drift
apart on this convention.

---

## ADR-004: SoC bounds (soc_min/soc_max) as fractions of capacity, not absolute kWh

**Status:** Accepted (stage 1)

**Context:** `Battery` needs operating SoC bounds (e.g. "never discharge
below 10% to protect cell life").

**Decision:** `soc_min`/`soc_max` are dimensionless fractions in `[0, 1]`,
not absolute kWh values.

**Alternatives considered:**
- *Absolute kWh bounds* (e.g. `soc_min_kwh = 10`). Rejected: couples the
  bound to a specific `capacity_kwh`, so changing capacity (e.g. comparing
  a 100 kWh vs 200 kWh asset) silently changes the *relative* operating
  window unless the absolute bound is remembered and rescaled by hand —
  fragile and easy to forget in a sensitivity sweep.

**Consequences:** Any battery sizing sweep can vary `capacity_kwh` while
keeping the same operating policy (e.g. "5%–95%") unchanged. Absolute kWh
bounds, where needed, are computed on demand as `soc_min * capacity_kwh`.

---

## ADR-005: schema.py fails loud — no dtype coercion or silent repair

**Status:** Accepted (stage 1)

**Context:** `validate()` could either accept "close enough" input and fix
it up (cast types, drop unknown columns, fill NaNs), or reject anything
that doesn't already match the contract exactly.

**Decision:** Strict, non-coercing validation: wrong dtype, an extra
column, a NaN, a naive timestamp — all raise `SchemaValidationError`
rather than being silently corrected. `validate()` accumulates every
problem it can find in a DataFrame and raises them together, rather than
stopping at the first one, so a caller fixing a broken fetcher output
doesn't have to run validate() -> fix one thing -> re-run -> fix the next
thing in a loop.

**Alternatives considered:**
- *Permissive/coercing validation* (auto-cast int to float, auto-drop
  extra columns, etc.). Rejected as the default: a fetcher that produces
  the wrong dtype has a bug, and silently coercing it here would hide that
  bug rather than surface it at the boundary where it's cheapest to find.
  This is the same philosophy the brief asks for explicitly at the pipeline
  stage ("flag gaps beyond a threshold rather than silently filling") —
  applied one layer earlier, at the schema boundary.

**Consequences:** Fetchers are responsible for producing exactly the right
dtypes; there is no `coerce()` convenience function. If this proves too
strict in practice (e.g. a legitimate source needs a documented, deliberate
type quirk), the fix belongs in the fetcher's own type handling, not by
loosening the shared contract everyone else relies on.

---

## ADR-006: pandas pinned to `<3.0` (installed: 2.3.3)

**Status:** Accepted (stage 1) — engineering note, not a modelling choice

**Context:** `pip install` initially resolved pandas 3.0.3 (released very
recently). Under it, `pd.Timestamp` columns defaulted to microsecond
resolution (`datetime64[us, ...]`) rather than nanosecond
(`datetime64[ns, ...]`), and plain string columns got a new backend
string dtype rather than `object` — both changes broke the exact-dtype
checks in `EXPECTED_DTYPES`.

**Decision:** Pin `pandas>=2.2,<3.0` in `pyproject.toml` (currently
resolves to 2.3.3).

**Alternatives considered:**
- *Make `EXPECTED_DTYPES` resolution-agnostic* (accept any of ns/us/ms for
  datetimes, any string-like dtype for `source`). Rejected for now: it
  weakens a contract that's supposed to be exact, to accommodate a pandas
  major-version bump that is only weeks old at the time of writing. Worth
  revisiting once pandas 3.x is the ecosystem default and its behaviour is
  stable/well-documented.

**Consequences:** Reproducible dtype behaviour matching the vast majority
of current pandas documentation and tooling. Revisit this pin later in the
project rather than fighting a moving target now.

---

## ADR-007: Elexon fetchers — APXMIDP over N2EX, SSP as the imbalance price, quality flags kept out-of-band

**Status:** Accepted (stage 2)

**Context:** `sources_elexon.py` needs to fetch and canonicalise two GB
price series: imbalance (system) prices and day-ahead prices. Several
forks came up once the real API responses were inspected (per the brief's
instruction to validate one real day and print the first raw record before
writing the parser).

**Decision, part 1 — APXMIDP over N2EX for day-ahead prices:** use
`dataProviders=APXMIDP` on the `market-index` endpoint. Verified
empirically (not just assumed) against 2026-07-15: N2EX returned 49
records with every `price` exactly `0.0`; APXMIDP returned real,
non-zero prices for the same window. Consequence: the all-zero/empty
guard (part 3 below) exists specifically because this failure mode is
real and silent otherwise, not a hypothetical.

**Decision, part 2 — imbalance price field:** the `system-prices` endpoint
returns both `systemSellPrice` and `systemBuyPrice`. Checked all 48
periods of a real day — identical throughout, consistent with GB's
post-2015 (BSC P305) single cash-out price. Use `systemSellPrice` as the
canonical `price_gbp_per_kwh`, and raise if the two ever differ, rather
than silently averaging or picking one — same fail-loud philosophy as
ADR-005. A mismatch would mean either historical dual-price data (pre
Nov 2015) or a market-design change this fetcher doesn't account for;
either way it should stop the pipeline, not blend into a number.

*Alternatives considered:* average the two fields (rejected — masks a
mismatch instead of surfacing it, and does nothing extra while they're
equal); use `systemBuyPrice` instead (no reason to prefer one over the
other while both are always equal, so the choice is arbitrary — flagged
for the user rather than picked silently).

**Decision, part 3 — all-zero/empty guard:** both fetchers raise
`AllZeroPriceSeriesError` if the parsed series is empty or every price is
exactly zero, per the brief. This is the guard that would have caught an
accidental N2EX/APXMIDP mix-up automatically, rather than relying on a
human noticing a suspiciously flat price column downstream.

**Decision, part 4 — BSAD/derivation quality flags kept out-of-band:** the
raw imbalance record includes `bsadDefaulted` and `priceDerivationCode`
(whether a period's price was an estimate/default rather than normally
derived) — not part of the 5 canonical columns. Rather than drop these,
`fetch_imbalance_prices()` returns an `ImbalanceFetchResult` with `.prices`
(canonical) and `.quality_flags` (a side table, joinable on
settlement_date/settlement_period), so stage 3's data-quality report can
use them without having to re-fetch or re-derive them later.

*Alternatives considered:* drop entirely now, revisit only if stage 3
analysis turns up something odd (simpler, but would require re-fetching
historical data later just to recover a flag we already had in hand).

**Decision, part 5 — DST-aware query window for day-ahead prices:** the
`market-index` endpoint takes a UTC `from`/`to` window, not a settlement
date. A naive UTC-calendar-day window does not line up with the London
settlement day under BST — verified empirically: querying
`2026-07-15T00:00Z`–`2026-07-16T00:00Z` (UTC calendar day) returns a
mismatched mix of periods from two different settlement dates, whereas
the correct window (from `schema.settlement_day_utc_bounds()`, i.e. local
midnight to local midnight) returns the right 48 periods once filtered to
the target `settlementDate`. The fetcher queries the correct DST-aware
window and then filters client-side on `settlementDate`, rather than
trusting the endpoint's `from`/`to` boundary inclusivity to hand back
exactly one day.

**Consequences:** `sources_elexon.py` shares `settlement_day_utc_bounds()`
with `schema.py` (extracted from `expected_period_count()` during this
stage) rather than reimplementing the DST-window logic a second time.
Both fetchers accept an injectable `session` (defaults to the `requests`
module) so tests run against real recorded fixtures
(`tests/fixtures/elexon_*_2026-07-15.json`) without hitting the network.

---

## ADR-008: pipeline.py — gaps never fabricate values, thresholds are per-day + run-length, cache before raise

**Status:** Accepted (stage 3)

**Context:** `pipeline.py` fetches a date range day-by-day and needs a
policy for what happens when periods are missing — the brief asks to
"repair" the half-hourly grid but also to "flag gaps beyond a configurable
threshold rather than silently filling," which pull in different
directions unless the exact behaviour is pinned down.

**Decision, part 1 — the cache never contains a fabricated value.**
`schema.full_grid()` (added this stage) generates the complete DST-aware
expected grid purely so `pipeline.py` can diff real fetched data against
it and count/locate gaps. The returned/cached DataFrame itself contains
only real, already-validated periods — a gappy day simply has fewer rows
than `expected_period_count()` for that day. Gaps are only ever visible
through `DataQualityReport`, never as a placeholder/NaN row in the data
itself. This keeps the cache always able to pass `schema.validate()`
as-is (which still rejects NaN prices, unchanged since ADR-005) and pushes
the "what do I do with an incomplete day" decision to whichever stage
consumes the cache next (e.g. stage 4 can choose to exclude any day that
isn't exactly 46/48/50 rows).

*Alternative considered:* reindex onto the full grid and let missing
periods be real NaN rows in the cached table. Rejected — it's more
immediately visible ("this day has a hole at period 23" without cross
-referencing the report), but the cached table would then violate the
no-NaN canonical contract, and something downstream would have to filter
it before treating it as validated data. Rejected in favour of keeping
exactly one definition of "valid data" in the codebase.

**Decision, part 2 — threshold is per-day missing-fraction plus a
whole-range max-consecutive-missing check.** Each day is judged against
the same missing-fraction threshold independently (so one bad day's
fraction doesn't get diluted into, or contaminate, a multi-month average),
*and* separately, the longest run of consecutive missing periods across
the entire fetched range (which can span a day boundary — a gap of periods
47-48 on one day plus 1-2 the next is a 4-period outage, not two
unrelated 2-period ones) is checked against its own threshold. Two
distinct failure modes, since "5% of periods missing, scattered as
isolated blips" and "5% of periods missing, as one contiguous outage" are
different data-quality problems and can warrant different tolerances.

**Decision, part 3 — cache first, raise after.** `run_pipeline()`
deliberately writes the merged cache to disk *before* calling
`_check_thresholds()`. The whole point of a per-day (rather than
whole-range) threshold, per part 2, is that one bad day shouldn't cost you
the good days fetched alongside it — so caching is unconditional on the
threshold check passing. `GapThresholdExceededError` is still raised
afterwards, so a caller can never silently miss that a day was bad; they
just don't lose the 89 good days out of 90 finding that out. A failed
day's fetch (network error, `AllZeroPriceSeriesError`, etc.) is folded
into the same accounting as a partial in-response gap — both just mean
"this day has fewer present periods than expected" — so there's one gap
-handling code path, not two.

**Decision, part 4 — cache merge keeps the freshest value per period.**
When merging newly-fetched data into an existing parquet cache,
`drop_duplicates(..., keep="last")` is used with new data concatenated
after old, so a re-fetched period always overwrites what was cached
before. This matters because Elexon settlement prices for very recent
periods can be revised after initial publication; without this, an early
fetch could permanently freeze a preliminary price in the cache even after
a corrected value becomes available from a later re-fetch.

**Consequences:** `pipeline.py`'s public entry points
(`run_imbalance_pipeline`, `run_day_ahead_pipeline`) are thin wrappers
around a fetcher-agnostic `run_pipeline(fetch_one_day, ...)`, so the gap
-accounting/caching/threshold logic is written and tested once, not once
per source.

---

## ADR-009: Tier 1 LP — day-ahead prices, forbid simultaneous charge/discharge, fixed cyclic SoC, discharge-only degradation

**Status:** Accepted (stage 4)

**Context:** `optimiser_tier1.py` needed four modelling decisions pinned
down before the LP could be written: which price series it represents,
whether to structurally forbid simultaneous charge/discharge (the brief
calls this out directly), what the cyclic end-of-day SoC constraint should
actually equal, and where `degradation_cost_per_kwh` enters the objective.

**Decision, part 1 — day-ahead (APXMIDP), not imbalance, for the main
results.** Day-ahead prices are published the day before delivery, so a
real operator could genuinely see the whole day's prices in advance —
"perfect foresight" is a realistic idealisation of an actually-knowable
quantity. Imbalance/system prices are only determined after real-time
balancing occurs; optimising against them with perfect foresight would
require a crystal ball for something that fundamentally isn't knowable in
advance, making it a more artificial, backtest-only construct. `solve_day()`
itself is price-series-agnostic (it just takes a price vector) — this
decision is about which cached series stage 7 actually feeds it for the
headline results, not a constraint baked into the LP.

**Decision, part 2 — forbid simultaneous charge/discharge via a binary
variable per period.** A binary `is_charging[t]` makes `charge_kw[t]` and
`discharge_kw[t]` mutually exclusive each period
(`charge_kw[t] <= power_kw * is_charging[t]`,
`discharge_kw[t] <= power_kw * (1 - is_charging[t])`), matching how a real
inverter physically operates (one direction at a time). This turns the LP
into a small MILP, solved instantly by CBC at 46-50 periods/day.

*Alternative considered:* a pure LP relying on economics (efficiency loss
+ degradation cost make simultaneous charge/discharge strictly wasteful,
so a well-posed LP "shouldn't" choose it). Rejected: this argument breaks
down exactly when `degradation_cost_per_kwh = 0` combined with
`round_trip_eff = 1.0` (tested explicitly — see
`test_flat_price_with_efficiency_loss_means_no_cycling` for the
lossy case), and more generally relies on an argument that would need
verifying after every solve rather than being structurally guaranteed.
Given the problem size makes the MILP free to solve, there's no real cost
to just forbidding it outright.

**Decision, part 3 — the cyclic start/end SoC is fixed at 50% of capacity,
not a free decision variable, pending a sensitivity check.** A free
boundary would let the LP quietly inflate profit by idealising a starting
condition a real operator doesn't get to choose for free each night. 50%
is used specifically (not an arbitrary round number): it's the midpoint of
the usable SoC range, and separately a commonly cited healthy resting
charge level for lithium-ion cells, minimising both over-charge and
over-discharge stress. This is flagged as a decision to stress-test, not
settle by assertion: stage 7 should re-run Tier 1 over the full cached
history at a few boundary values (e.g. 25/50/75%) and report how much the
profit ceiling actually moves. If it's insensitive, that's a strong,
cheap sentence for the write-up ("results are robust to the boundary SoC
assumption to within X%"); if it isn't, that's a real finding to report
rather than a footnote.

**Update (stage 7, real full-year sweep):** it isn't insensitive. Over the
full cached year (2025-08-14 to 2026-08-13, 365/365 days solved, battery:
100 kWh / 50 kW / 90% RTE / SoC 5-95% / £0.01/kWh degradation):

| boundary_soc | total annual profit |
|---|---|
| 0.25 | £1650.32 |
| 0.50 | £1563.72 |
| 0.75 | £1420.55 |

A 16.2% spread between the extremes tested — not a rounding footnote. The
direction makes sense once you see it: with `soc_max=0.95`, a *lower*
boundary leaves more headroom between the boundary and `soc_max`, so each
day's "charge up" leg can absorb more energy before hitting the ceiling —
a bigger achievable cycle amplitude, not a numerical artefact. This means
the 50% choice is a genuine, stated modelling assumption that measurably
shapes the headline profit figure, not a detail that washes out — worth
saying plainly in any write-up of these results, not glossed over.

**Decision, part 4 — degradation cost applies to discharged energy only,
not both legs of throughput.** Published cycle-life figures for
lithium-ion cells are conventionally stated in terms of discharged
throughput or full-cycle-equivalents ("rated for N full cycles," counted
via discharged kWh/Ah) — that convention is what a real
`degradation_cost_per_kwh` figure would be calibrated from. Charging the
cost on both legs while using a discharge-referenced figure would
double-count wear relative to its own source, unless the per-kWh figure
were independently halved to compensate — an easy detail to get subtly
wrong. Applying it to discharge only keeps the cost basis consistent with
how the number would actually be sourced and cited.

*Alternative considered:* both legs of throughput. More "complete" in
pure physical-stress terms (charging does stress a cell too), but
rejected specifically because it would silently misrepresent a
literature-sourced discharge-referenced figure rather than because the
physical argument for it is wrong.

**Consequences:** `solve_day()` operates on one day (one price vector) at
a time, consistent with the cyclic-per-day constraint — multi-day runs
(stage 7) loop over days and aggregate outside this module.
`degradation_cost_per_kwh`'s meaning is now fully pinned down (previously
deferred in `config.py`, stage 1): it is a cost per kWh of *discharged*
energy specifically, not throughput in general.

**Note (engineering, not modelling):** PuLP 3.x deprecates
`LpVariable.dicts` and `PULP_CBC_CMD` ahead of a 4.0 release that removes
them in favour of `problem.add_variable_dicts(...)` and `COIN_CMD`. The
replacement solver requires an extra ~60MB `cbcbox` binary dependency
(via the `pulp[cbc]` extra) for a release that isn't out yet. Pinned
`pulp<4.0` instead (same approach as the pandas pin in ADR-006) — cheaper
than adding a large binary dependency to silence a warning with no present
functional effect; revisit when 4.0 is actually released and stable.

---

## ADR-010: backtest.py — genuinely independent arithmetic, a generic array interface, fail-loud on violations

**Status:** Accepted (stage 5)

**Context:** the brief specifies what `backtest.py` needs to do fairly
precisely (independently recompute SoC/cashflow, cross-check the LP to a
tiny tolerance), so this stage was mostly implementation judgement rather
than a modelling fork — recorded here for the reasoning rather than as a
question that was asked.

**Decision, part 1 — the SoC/cashflow arithmetic is written fresh in
`backtest.py`, not called from or shared with `optimiser_tier1.py`.** Both
modules do read `Battery.eta_charge`/`eta_discharge` from `config.py`
(the intentional single source of truth for efficiency, ADR-003) — that's
not a violation of independence, it's the one place drift is supposed to
be structurally impossible. But the SoC-update loop and the
revenue-minus-cost-minus-degradation calculation are separate code in each
module. The value of a cross-check comes specifically from two
independently-derived computations agreeing; if `backtest.py` just called
back into the LP's own constraint-building code, agreement would be close
to tautological — a bug in that shared code would reproduce identically in
"both" places and the check would never catch it.

**Decision, part 2 — `simulate()` takes plain `charge_kw`/`discharge_kw`
arrays, not a `Tier1Schedule`.** Stage 6's naive baseline will produce a
schedule that needs backtesting too, and it isn't a `Tier1Schedule` (it
has no LP objective value, no solver status) — coupling the simulator to
that type would mean rewriting or wrapping it for stage 6. `simulate()` is
the reusable, schedule-agnostic core; `assert_matches_lp()` is a thin
Tier-1-specific layer on top that also checks agreement against the LP's
own reported numbers specifically.

**Decision, part 3 — violations are reported as flags on
`BacktestResult`, not silently repaired, and `assert_matches_lp()` raises
with a specific message identifying which check failed.** Consistent with
the fail-loud precedent already set in `schema.validate()` (ADR-005) and
`pipeline.py`'s gap handling (ADR-008): a schedule that violates SoC
bounds, power limits, or mutual exclusivity is a bug somewhere upstream,
and should be surfaced precisely, not smoothed over.

**Decision, part 4 — tolerance is `1e-6` (absolute), on both the £
cashflow comparison and the kWh SoC-trajectory comparison.** Chosen to be
tight enough to catch a genuine formulation bug (which would typically
show up as a difference of a meaningful fraction of a kWh or a penny, not
a rounding artefact) while comfortably clearing ordinary floating-point
noise from the solver and from independently-accumulated floating-point
sums over 46-50 periods. Verified in practice against 10 real cached
day-ahead days (2026-07-10 to 2026-07-19): the LP's own objective and the
backtester's independently-recomputed cashflow agreed exactly to displayed
precision on every day, with no bounds or mutual-exclusivity violations.

**Consequences:** stage 6 (naive baseline) can call `simulate()` directly
to get a comparable cashflow figure without needing anything LP-specific,
and stage 7's results can call `assert_matches_lp()` as a standing sanity
check over the full cached history, not just in the test suite.

---

## ADR-011: naive baseline — N derived from the battery, exact cyclic match, two different N values per direction

**Status:** Accepted (stage 6)

**Context:** the brief describes the naive baseline in plain English
("charge the N cheapest periods, discharge the N dearest"), which leaves
two things unstated: how N is actually chosen, and whether the naive
schedule has to play by the same start=end SoC rule Tier 1 does.

**Decision, part 1 — N is derived from the battery's own physical limits
(one full daily cycle: boundary SoC -> soc_max -> boundary SoC), not a
separately-configurable number.** This makes the baseline's scale
principled and battery-specific rather than arbitrary, and gives a
natural interpretation to "the naive strategy" — it does the single
largest cycle this battery can physically complete in a day.

**Decision, part 2 — the naive schedule enforces the same exact cyclic
boundary as Tier 1**, via a partial-power period on whichever chosen
period is marginal (the priciest of the charging set, the cheapest of the
discharging set) rather than whole full-power periods only. This keeps
the comparison to Tier 1 fair: both strategies start and end the day at
the same SoC, so any profit gap reflects scheduling skill, not one
strategy quietly banking extra stored value the other didn't.

**Decision, part 3 (a consequence of part 1, not separately asked) — N is
actually two different numbers, N_charge and N_discharge, not one shared
N.** Round-trip efficiency is asymmetric in its effect on period counts:
charging loses energy on the way *in*, so adding a given amount of stored
energy takes *more* full-power periods than removing the same amount via
discharging, which loses energy on the way *out* instead — the "N cheapest
/ N dearest" framing in the brief is a simplification of what's actually
two related-but-different counts once efficiency losses are real (< 1).
Implemented as two derivations from the same headroom (`soc_max_kwh -
boundary_soc_kwh`), one per direction; flagged here rather than silently
picked, since it's a legitimate reading of an ambiguous plain-English
description.

**Consequences:** verified against real cached day-ahead data over the
same 10 real days used in stages 4-5: the naive baseline sits strictly
below the Tier 1 ceiling on every single day, capturing about 60% of
Tier 1's total profit over that window — consistent with a baseline
capped at one cycle competing against an optimiser free to exploit
multiple price swings per day where the data supports it.

---

## ADR-012: results.py — discharge-based cycles/day, per-day failure isolation, a full year fetched for credibility

**Status:** Accepted (stage 7)

**Context:** stage 7 aggregates Tier 1 over the entire cached history into
a few headline numbers, which meant deciding what "cycles/day" actually
counts, how a bad day should be handled in a 365-day batch run, and how
much history was worth fetching before the numbers meant anything.

**Decision, part 1 — cycles/day is discharge-based**
(`discharged_kwh / usable_capacity_kwh`, averaged over all solved days),
consistent with the discharge-referenced degradation convention already
adopted in ADR-009. Manufacturer cycle-life figures ("N full cycles") are
themselves discharge-referenced, so measuring cycles the same way keeps
this number directly comparable to a real datasheet figure — not a fresh
fork, just carrying an already-made decision through consistently.

**Decision, part 2 — a failing day is excluded and recorded, not fatal to
the whole run**, directly acting on the Q9 discussion from ADR-010: a
batch check over hundreds of days needs full visibility in one pass, not
a stop at the first failure. `run_tier1_over_history()` catches
`RuntimeError` (non-optimal solve) and `AssertionError` (a
backtest-cross-check failure) per day, continues, and returns
`failed_days` as data the caller can inspect — the correctness anchor from
stage 5 still runs on every single day, it just doesn't abort the batch.

**Decision, part 3 — fetched a full year of real day-ahead data (365
days, 2025-08-14 to 2026-08-13) rather than running results on the 10
days already cached.** Chosen over the user's own stated preference for
credibility: cumulative P&L, cycles/day, and especially an *annualised*
£/kWh-capacity/year figure are far more defensible computed from an
actual year (capturing real seasonal variation — the best day found,
2026-06-23, both a summer day and by far the most profitable, at £34.69
vs a typical day around £1-10) than extrapolated from 10 days in one
month.

**Consequences (results over the full year, battery: 100 kWh / 50 kW /
90% RTE / SoC 5-95% / £0.01/kWh degradation, boundary_soc=0.5):**
- 365/365 days solved to optimality and passed the LP/backtester
  cross-check — zero failures, the strongest evidence yet that the
  formulation is correct across genuinely varied real price conditions,
  not just the 10 days used during development.
- Cumulative Tier 1 profit: £1563.72/year. Naive baseline: £880.90/year
  (56% of Tier 1) — consistent with the ~60% seen on the smaller 10-day
  sample in ADR-011, not a fluke of that smaller window.
- Mean cycles/day: 1.368 — Tier 1 usually completes just over one full
  cycle per day, sometimes more when the day's price pattern supports a
  second profitable swing (see the example-day plot, which shows two
  cycles on 2026-06-23).
- Annualised: £15.64 per kWh of installed capacity per year.
- The boundary_soc sensitivity check promised in ADR-009 was run here
  and found to be a genuine, non-negligible effect (16.2% spread) — see
  the update appended to ADR-009 rather than duplicated here.

---

## ADR-013: CI — a committed real-data fixture instead of the gitignored local cache

**Status:** Accepted (stage 8)

**Context:** wiring up CI (GitHub Actions, `.github/workflows/ci.yml`)
surfaced a real bug rather than a hypothetical one:
`test_naive_baseline.py::test_naive_sits_below_tier1_on_real_data` read
`data/day_ahead.parquet` directly — a file that only exists locally
because it was fetched during earlier stages, and is gitignored on
purpose (ADR-006/ADR-008 precedent: cached data is regenerable, not
source-controlled). On a fresh checkout, or in CI, that file simply isn't
there, and the test would fail on a missing-file error that has nothing
to do with the code being wrong. Verified by temporarily moving `data/`
aside locally and re-running the full suite before treating this as
fixed, not just asserting it.

**Decision:** extracted a small (10-day, 480-row) real slice of the
already-fetched day-ahead data into a committed fixture
(`tests/fixtures/day_ahead_sample_2026-07-10_to_2026-07-19.parquet`),
matching the pattern already established in stage 2
(`tests/fixtures/elexon_*.json` — real recorded API responses, committed,
used via dependency injection instead of live calls). The test now reads
from this fixture instead of the local cache.

**Alternatives considered:**
- *Skip the test if the cache file doesn't exist* (`pytest.mark.skipif`).
  Rejected: silently skipping a real-data correctness check in CI is
  exactly the kind of "looks green but isn't actually checking anything"
  outcome this project has avoided elsewhere (see ADR-008's "no silent
  gap-filling").
- *Fetch fresh data at test time.* Rejected: makes the test suite's
  runtime and pass/fail status depend on Elexon's API being up and fast,
  every time CI runs — flaky and slow for something that only needs to
  confirm a comparison holds on data that doesn't change.

**Consequences:** the full test suite (83 tests, 96% coverage) now passes
identically whether or not `pipeline.py` has ever been run locally —
verified directly, not assumed. CI (`.github/workflows/ci.yml`) runs
`pytest -v --cov=bess --cov-report=term-missing` on every push/PR to
`master` on a clean `ubuntu-latest` checkout, so this class of bug (a test
that only passes because of leftover local state) can't silently
reappear.

---

## Tier 2 — a rolling-horizon (MPC) controller using a learned price forecast

Everything from here on is a second phase beyond the original 8-stage
plan: Tier 1 assumes perfect foresight of a day's prices, which is a
useful ceiling but not something a real deployed battery has access to.
Tier 2 asks a different question: using only a forecast of future prices
(learned from historical patterns, not a crystal ball), how much of that
ceiling can a realistic, causally-valid controller actually capture?

## ADR-014: features.py — drop warm-up rows rather than impute, and a black-box leakage guard

**Status:** Accepted (Tier 2, part 1)

**Context:** `build_features()` needs lag (t-1, t-2, t-48, t-336) and
rolling-statistic (trailing 48-period mean/std) features. Both leave the
first `max(lag)` = 336 rows of any series without a valid value, and
getting the leakage boundary wrong here would silently invalidate every
result built on top of it later (the forecaster's accuracy, and the MPC
controller's profit figure).

**Decision, part 1 — drop rows without full history, don't impute.**
Filling in a plausible-looking value for history that doesn't exist (e.g.
backfilling, or a global mean) would be a silent, undetectable assumption
baked into the training data — precisely the kind of "coerce rather than
reject" choice this project rejected for the canonical schema in ADR-005.
Dropping is simpler and honest: it costs 336 periods (a week) at the
start of the cached history, which is a rounding error against a
full-year sample.

**Decision, part 2 — the leakage guard is a black-box behavioural test,
not a check that the code reads correctly.** Rather than asserting shift
amounts are positive (a check that only catches a bug the same way it was
introduced — reading the code and confirming it says what you think it
says), `tests/test_tier2_features.py` corrupts only the *future* portion
of a price series and asserts every feature row computed from data
strictly before the corruption point is byte-identical to the
uncorrupted run. This is the direct Tier 2 analogue of the Stage 5
LP/backtest cross-check: a property-level check that would catch a
leakage bug regardless of *how* it was introduced (a forgotten `.shift()`,
an off-by-one, a rolling window that isn't pre-shifted), not just the one
bug pattern the author happened to think to check for. Verified the test
actually discriminates real leakage from correct code by rebuilding a
deliberately-leaky rolling mean (no `shift(1)` before `.rolling()`) and
confirming the same corruption-comparison technique catches it, rather
than trusting the guard's design without evidence it can fail.

**Decision, part 3 — calendar features derived from `settlement_period`/
`settlement_date`, not `timestamp_utc`.** These are already
local-calendar-of-record fields on the canonical schema (ADR-001) — using
them sidesteps the UTC/London-offset conversion this project has handled
carefully everywhere else (ADR-002), rather than reintroducing it for a
feature that doesn't need it. Consequence: `hour_of_day` can read as `24`
on the rare autumn clock-change day (the genuine extra repeated half-hour
pair) rather than staying confined to `0-23` — a faithful reflection of
that day actually having an extra half-hour, not a bug to paper over.

**Consequences:** verified against the real full-year cache: 17,520 raw
periods produce exactly 17,184 feature rows (336 dropped, matching
`max(LAG_PERIODS)` exactly), zero NaNs in the output. `forecaster.py` and
`mpc.py` can build on this feature matrix without re-deriving or
re-verifying the leakage boundary themselves.

## ADR-015: forecaster.py — direct multi-horizon models, and a real degradation finding

**Status:** Accepted (Tier 2, part 2)

**Context:** `mpc.py` needs a full price-path forecast (periods t+1
through t+H) at every decision point, and the brief asks for an accuracy
comparison across horizons (1 period ahead vs 48) to sanity-check that
accuracy degrades as expected — a flat accuracy curve across horizons
would be a leakage red flag.

**Decision, part 1 — direct multi-horizon forecasting: one LightGBM model
per horizon, not recursive 1-step iteration.** Recursive forecasting
(predict t+1, then feed that prediction back in as if it were a real
`lag_1` value to predict t+2, and so on) compounds error at every step and
means a lag feature sometimes holds real data and sometimes the model's
own guess — a semantic inconsistency that's also a plausible route for
foresight to quietly leak in. Direct per-horizon models (`target_h[i] =
price[i+h-1]`, same features, a fresh shifted target per horizon) avoid
both problems and are standard practice for tree-based models, which have
no native sequence-generation mechanism the way an RNN would. Training
cost is negligible at this data size (a few seconds per horizon).

**Decision, part 2 — a real bug found and fixed: `naive_forecast()` must
be built from `lag_48`, not by re-shifting the already-truncated feature
frame.** The first implementation computed `price_gbp_per_kwh.shift(49 -
horizon)` directly on `features_df` — but that frame has already had its
first 336 rows dropped (ADR-014), so re-shifting it from scratch throws
away validity that `lag_48` (computed *before* the drop) already has.
Concretely, at horizon=1 this made the naive baseline disagree with its
own literal definition (`lag_48`) on the first 48 rows for no reason.
Fixed by deriving `naive_forecast` from the existing `lag_48` column
(`lag_48.shift(-(horizon-1))`), which reaches back into pre-drop history
correctly and only loses rows at the genuinely-unavoidable trailing edge
(the last `horizon-1` rows, which have no future target to compare
against at all — the same edge `_target_for_horizon` has). Found via a
test written to confirm the horizon=1 case exactly matches `lag_48` per
the brief's own definition, not by inspection.

**Consequences — real evaluation results, full cached year, 5-fold
expanding-window time-series CV:**

| horizon | model MAE | naive MAE | model beats naive |
|---|---|---|---|
| 1 (30 min) | 0.0056 | 0.0231 | yes, by ~4x |
| 6 (3 h) | 0.0161 | 0.0231 | yes |
| 12 (6 h, MPC's default H) | 0.0215 | 0.0231 | yes, narrowly (~7%) |
| 24 (12 h) | 0.0245 | 0.0231 | **no** |
| 48 (24 h) | 0.0252 | 0.0231 | **no** |

Accuracy degrades monotonically with horizon as expected (a flat curve
would have been a leakage red flag) — reassuring evidence the leakage
guard in ADR-014 is doing its job, not just passing its own tests. The
model clearly beats naive across the horizons MPC actually uses (up to
H=12), which is what this stage needed to prove; per the brief, stopped
here rather than tuning further. Flagged for later: the model's edge
narrows sharply approaching h=12 and is gone by h=24 — pushing MPC's
horizon much past its current default without reconsidering the
forecaster would likely stop helping.

**Note (engineering, not modelling):** LightGBM requires the OpenMP
runtime (`libomp`) on macOS, not installed by default with Homebrew
Python — `pip install` succeeds but the import fails with a
`dlopen`/`Library not loaded` error until `brew install libomp` is run
separately. Verified directly (not just assumed) that this does NOT
affect Linux: ran the full suite inside a genuine `ubuntu` Docker
container with no extra system packages beyond `python3`/`pip`, and all
100 tests passed — confirming CI (`ubuntu-latest`) needs no changes.

## ADR-016: mpc.py — extending solve_day() for reuse, the SoC-handoff discipline, and a static-forecast deferral finding

**Status:** Accepted (Tier 2, part 3)

**Context:** `mpc.py` needs to solve a short, partial-window LP at every
period using forecasted prices, starting from the battery's real current
SoC — but `solve_day()` (Tier 1) always requires the whole-day cyclic
boundary (`soc[T] == soc[0]`) and only accepts a starting SoC expressed as
a fraction of capacity. Reusing it required extending it, not
duplicating it.

**Decision, part 1 — extended `solve_day()` with two new optional
parameters, both defaulting to Tier 1's existing behaviour exactly.**
`cyclic: bool = True` makes the `soc[T] == soc[0]` constraint conditional
— MPC passes `cyclic=False` for its non-cyclic partial windows.
`initial_soc_kwh: float | None = None` lets a caller hand the LP a
starting SoC directly in kWh, overriding `boundary_soc` for the starting
condition only — MPC's real current SoC is a genuine kWh figure produced
by the simulator, not a clean fraction of capacity, and forcing it
through a fraction round-trip would reintroduce exactly the kind of
fraction-vs-kWh confusion already flagged as a real risk in the stage-4
quiz. Every existing Tier 1 call site (results.py, naive_baseline.py, the
whole Tier 1 test suite) leaves both parameters at their defaults and is
unaffected — verified by running the full pre-existing suite unchanged
after this edit, not just by inspection.

**Decision, part 2 — the SoC handed from one MPC step to the next is
always `backtest.simulate()`'s own recomputed value, never
`schedule.soc_kwh[1]` from the LP's internal solve.** This is the
single load-bearing discipline of the whole module, exactly as flagged in
the brief: the LP's `soc_kwh` trajectory reflects the *forecast*, not
reality, so reading it back as the "current" SoC would silently let
forecast-based (not real) information flow into the next decision —
foresight leaking in through the back door of the state variable rather
than the feature matrix. `run_mpc()` only ever reads `schedule.charge_kw[0]`
and `schedule.discharge_kw[0]` (the decision) from each solve, and
recomputes what actually happened via `simulate()` against the real
realised price, independently.

**Decision, part 3 (a bug found while testing, not a modelling choice) —
`solve_day()`'s starting-SoC bounds check needed a small tolerance.**
Chaining many `solve_day()` calls, each seeded from the previous call's
real, simulator-computed SoC, accumulates floating-point drift that can
produce a value like `-1e-15` at a boundary that is genuinely zero but
not bit-exact. Tier 1's single-solve-per-day usage never chains calls
this way, so this never surfaced before MPC existed. Fixed with the same
`1e-6` tolerance pattern already used in `backtest.py`, plus clipping the
value to the exact bounds before it reaches the LP's own declared
variable bounds (a within-tolerance-but-marginally-outside value pinned
via an equality constraint could otherwise make the LP itself infeasible
even after the tolerance check accepts it).

**Finding — a static (non-time-varying) forecast causes MPC to defer a
profitable action forever, and that's correct, not a bug.** Discovered
while building the hand-computable test: if the forecaster's prediction
never changes from step to step, and the LP's optimal plan for "period 0
of the window" is to wait for a better price at "period 1," then MPC —
which only ever executes period 0's decision — re-derives the identical
"wait" decision every single step, since the real SoC never changes
either. The profitable sale never happens. This is a property of
rolling-horizon control fed an unchanging forecast, not a defect in the
loop: a real, time-varying forecaster (forecaster.py) doesn't exhibit it,
because its prediction genuinely updates as real information arrives each
step. Confirmed the loop isn't inherently paralysis-prone with a second
hand-computed case (forecast favouring the *immediate* period) that does
act immediately, and separately confirmed real full-year MPC runs produce
active cycling, not paralysis (results_tier2.py).

**Also verified directly rather than assumed:** the non-cyclic window's
"liquidate everything by the window's end regardless of price level"
behaviour — since unsold energy at a window's edge is credited zero value
in that solve's objective, the LP prefers selling at even a "cheap"
forecast price over holding stock past the window boundary. An earlier
draft of the hand-computable test got this wrong by intuition alone and
was corrected against `solve_day()`'s actual output before being written
down — the same "verify by hand against the real function before trusting
the number" discipline as the Tier 1 hand-computed test in ADR post
stage-4.

**Consequences:** end-to-end wiring verified on real data — training 12
horizon models takes ~6s, and MPC over a 2-day real slice runs at
~22ms/period, projecting to roughly 6 minutes for the full cached year
(run in the background in results_tier2.py, not synchronously).

## ADR-017: results_tier2.py — chronological train/test split, and a same-structure ceiling

**Status:** Accepted (Tier 2, part 4)

**Context:** the final comparison needed two things pinned down beyond
what the brief specified directly: whether the forecaster backtesting MPC
should ever see the period it's scored on, and — discovered only once
real testing began — whether Tier 1's existing per-day-cyclic total is
actually a valid ceiling to compare a continuous MPC run against.

**Decision, part 1 — an 80/20 chronological train/test split, not
in-sample.** `chronological_train_test_split()` trains the forecaster
only on the earlier 80% of the feature history and backtests MPC only on
the later, held-out 20%. Training and scoring on the same year would let
the model effectively memorise that year's specific patterns, inflating
MPC's apparent performance in a way a genuinely new deployment wouldn't
reproduce — the same leakage concern as ADR-014, one level up (at the
training-set boundary rather than the per-row feature boundary).

**Decision, part 2 — a real bug found while testing, not assumed away:
the "ceiling" must be a single non-cyclic solve over the whole continuous
test window, not the existing per-day-cyclic Tier 1 total.** The first
implementation reused `run_tier1_over_history()` (per-calendar-day,
forced back to `boundary_soc` every day, ADR-009) directly as "the
ceiling." A synthetic test caught this immediately: MPC's continuous,
non-cyclic run genuinely beat that number, because MPC can carry SoC
across a day boundary when profitable and a cyclic-per-day Tier 1
structurally cannot — meaning the per-day total was never a true upper
bound on what MPC could achieve, only on what a cyclic-per-day strategy
could. Fixed by computing the ceiling as one `solve_day(..., cyclic=False)`
call over the entire test window at once, with the same starting SoC MPC
uses — perfect foresight and the same structural freedom (no forced
resets) as MPC, differing from it only in whether the price vector is
real or forecasted. The naive floor keeps its existing per-day-cyclic
structure unchanged: it's a simple reference strategy, not a bound that
must never be violated, so it doesn't need matching MPC's freedom the way
a true ceiling does.

**Decision, part 3 — the corrupted-forecast check substitutes a
randomly-chosen OTHER row's real features, rather than injecting noise.**
`_ShuffledForecaster` still calls the real, trained model — just on a
random different row's genuine historical features instead of the actual
current one. This keeps predictions realistic in scale and distribution
(never an out-of-range value that might trip some unrelated effect) while
completely destroying any correlation between the forecast and the actual
situation. If MPC's performance doesn't collapse toward the naive floor
under this corruption, the good performance isn't coming from the
forecast — meaning a leak exists in the SoC handoff (ADR-016) or the
feature matrix (ADR-014) that this check is specifically positioned to
catch.

**Consequences — real results, 80/20 chronological split, held-out test
period of 71.6 days, battery: 100 kWh / 50 kW / 90% RTE / SoC 5-95% /
£0.01/kWh degradation, horizon=12:**

| Strategy | Total profit (held-out period) |
|---|---|
| Tier 1 ceiling (perfect foresight, non-cyclic, same window) | £485.43 |
| MPC (Tier 2, learned forecast) | £335.62 (69.1% of ceiling) |
| Naive baseline | £243.04 |
| MPC, corrupted forecast (sanity check) | **-£308.53** |

The ordering naive < MPC < ceiling holds exactly as expected once the
ceiling was fixed to match MPC's own structural freedom (part 2). The
corrupted-forecast result is the important one: performance doesn't just
drop below naive, it goes sharply negative — a forecast decoupled from
reality doesn't merely fail to help, it actively costs money (paying to
charge/discharge on wrong information), which is exactly the signature
you want from a controller that's genuinely using its forecast rather
than getting lucky some other way. No indication of a leak in the SoC
handoff or feature matrix.

---

## Multi-market extension — ERCOT (Texas), West Hub

Everything before this point was GB-only. This section generalises the
canonical schema to support a genuinely different market — different
currency, different settlement grain, different timezone/DST calendar —
and adds ERCOT's West trading hub (HB_WEST) as the first non-GB source.
Tier 1/Tier 2 (the optimiser, forecaster, MPC) are not yet pointed at
ERCOT data — this covers contracts and fetching only.

## ADR-018: schema.py multi-market generalisation, and choosing HB_WEST over a system-wide average

**Status:** Accepted

**Context:** ERCOT differs from GB in four structural ways that the
original schema, built GB-only, didn't need to represent: native currency
(USD vs GBP), settlement grain (ERCOT's Day-Ahead Market is hourly, Real
-Time Market is 15-minute — GB is uniformly half-hourly), timezone/DST
calendar (America/Chicago, on US clock-change dates, not UK ones), and
locational pricing (GB has one national price; ERCOT prices differ by
settlement point/node).

**Decision, part 1 — native currency, not FX-converted to GBP.** Added an
explicit `currency` column rather than converting USD to GBP at ingest.
Converting would require trusting a new external FX-rate data source and
would bake currency fluctuation into what's supposed to be a signal about
ERCOT's own price dynamics — a confound this project didn't need to
accept. `price_gbp_per_kwh` renamed to `price_per_kwh` throughout the
codebase to reflect this (a mechanical but wide rename — every module
from `sources_elexon.py` through `results_tier2.py`, and every test).

**Decision, part 2 — model each market at its own native grain.**
`schema.py`'s timezone (`tz`) and period length (`period_minutes`) are now
parameters on `settlement_day_utc_bounds()`, `expected_period_count()`,
`full_grid()`, and `validate()`, defaulting to Europe/London and 30
minutes so every existing GB call site is unchanged. The DST-window
arithmetic itself (convert to UTC before subtracting, ADR-002) didn't need
to change at all — only the hardcoded constants did. `period_minutes` is
also now a column on the canonical schema itself (added to `full_grid()`'s
skeleton, since it's a calendar/structural fact like `settlement_period`,
not a data value like price) — added specifically so a row is
self-describing about its own granularity, and `validate()` can catch a
dataset that accidentally mixes granularities or currencies, rather than
assuming each cached file stays internally homogeneous by convention alone.

**Decision, part 3 — HB_WEST (a real trading hub), not a system-wide
average.** The recommendation going in was a system-wide/hub-average
price, as the closest single-number analogue to GB's one national price.
Rejected on the same grounds this project has applied elsewhere (day
-ahead over imbalance for Tier 1's realism, ADR-009): an average across
the whole system is a synthetic number nobody actually settles at — a
real battery sits at a real physical location and is paid that location's
real price. West was chosen specifically over North/South/Houston for its
heavy wind penetration and the resulting price volatility (lowest average
prices in ERCOT, frequent negative pricing from local oversupply) — a
more interesting arbitrage signal than North's closer-to-average, more
GB-like profile, which was the alternative on the table.

**Decision, part 4 (discovered during implementation, not decided in
advance) — a genuinely different DST-handling approach was needed for
ERCOT versus GB.** GB's `settlement_period` is already a real
elapsed-time position (period 46 vs 48 on the spring clock-change day
tells you directly how long the day was) — it never needed anything extra
to handle DST. ERCOT reports hours in "Hour Ending" form and — cross
-checked against the `gridstatus` open-source library, which already
parses these same two endpoints in production, since we have no ERCOT
account of our own to verify against live — publishes an explicit
`DSTFlag` ("Y"/"N") specifically because its wall-clock hour label
*repeats* on the US autumn clock-change day (the same "hourEnding" value
appears twice). Computing an elapsed-hours offset directly from that label
would silently get the repeated hour wrong. Fixed by sorting each day's
raw records into true chronological order first (`DSTFlag` breaks the tie
on the repeated hour) and assigning `settlement_period` by *position* in
that sorted sequence — reusing the same "position-in-sequence, not label
arithmetic" principle `schema.full_grid()` already relies on — rather than
computing the period number from the hour label directly.

**Consequences / what's confirmed vs. still open:** the base URL, both
endpoint paths (`/np4-190-cd/dam_stlmnt_pnt_prices`,
`/np6-905-cd/spp_node_zone_hub`), the token URL/OAuth flow, and the query
parameters (`deliveryDateFrom`/`deliveryDateTo`, no settlement-point
filter — every settlement point is returned and filtered client-side, the
same pattern as `sources_elexon.py`'s day-ahead fetcher) are all
cross-checked against `gridstatus`'s working source and are trusted. The
*exact casing* of individual JSON field names in ERCOT's specific REST API
responses is not independently verified — `sources_ercot.py` marks every
such assumption `VERIFY` in its docstrings, and needs a live call against
a real registered ERCOT account before being trusted in production, per
the same "validate one real day before writing the parser" discipline
`sources_elexon.py` was built with (ADR-007). The entire existing GB test
suite (125 tests) passes unchanged after the schema generalisation — the
new parameters' defaults exactly reproduce prior GB behaviour, confirmed
directly by running the full suite before and after, not assumed.

**Update (2026-09-07) — `get_token()` corrected against a real account.**
The first version of `get_token()` sent credentials as a POST body
(`data=`), matching the general shape described in ERCOT's own written
API documentation. Against a real registered account this returned `400
Bad Request`. ERCOT's own official example code (not just their prose
documentation) showed the actual, working shape: credentials go as URL
query parameters, not a body — some Azure B2C custom-policy token
endpoints are configured to read the request this way, which written
documentation describing "POST parameters" doesn't distinguish from a
form body. Their example also reads `access_token` from the response and
uses that as the Bearer token, not `id_token` (both are present in the
response; only one is the one their own working flow actually uses) —
`ErcotToken`'s field renamed to match. Fixed to use `requests`' `params=`
rather than copying their example's raw string formatting verbatim, since
the latter doesn't URL-encode the password and would break on one
containing `&`, `%`, or `+`. This is the second time in this ERCOT
integration that *written* API documentation described something subtly
differently from what the API actually does (the first being the
Hour-Ending/DSTFlag behaviour in ADR-018's part 4) — reinforcing the same
lesson this project has applied to itself from the start: verify against
the real thing before trusting a description of it, however official the
source.

**Update (2026-09-07) — DAM/RTM field names confirmed live; the
`gridstatus` cross-check on filtering was wrong.** A real account finally
got past the earlier "socket hang up" issue and returned live DAM and RTM
responses, resolving every remaining `VERIFY` tag:

* `settlementPoint` **is** a working server-side query filter — confirmed
  by adding `settlementPoint=HB_WEST`, which dropped a 53,664-record,
  54-page unfiltered DAM response down to a single page. The
  `gridstatus`-based assumption in this ADR's "Consequences" section above
  (no server-side filter, fetch-everything-then-filter-client-side) was
  wrong; without this, `sources_ercot.py` would have paginated through
  every resource node, load zone and hub in Texas just to find one hub's
  prices. Fixed by filtering server-side and deleting the client-side
  `_hub_filter()` entirely — trusting the confirmed API behaviour rather
  than keeping a redundant client-side check for a case the server no
  longer allows through.
* `deliveryDateFrom`/`deliveryDateTo` are both *inclusive* (a
  `08-24`→`08-25` range returned both days in full). A single day's fetch
  now sets both to the same date — no post-fetch date filter needed.
* The response body is `{"fields": [...], "data": [[...], ...]}` —
  positional rows, not objects; field names/order come from the separate
  `fields` list. Parsed by zipping the two, since nothing guarantees that
  position order is stable across ERCOT report versions.
* DAM's `hourEnding` is a string like `"01:00"`, not a bare int.
* RTM has no `hourEnding` field at all — it splits into separate integer
  `deliveryHour` and `deliveryInterval` fields.
* `DSTFlag` is a real JSON boolean (`true`/`false`), not the `"Y"`/`"N"`
  strings `gridstatus`'s CSV-report code path uses. This was the most
  dangerous of the wrong assumptions: `dst_flag == "Y"` against a real
  bool is simply always `False`, so the repeated-hour DST tie-break would
  never have fired — not a crash, a silent misordering of the one day a
  year it matters most.

All fixed in `sources_ercot.py`, with `test_sources_ercot.py` and
`test_pipeline.py`'s ERCOT fixtures rebuilt to match the real response
shape. Full suite (129 tests) still green. No `VERIFY` tags remain in
`sources_ercot.py`.
