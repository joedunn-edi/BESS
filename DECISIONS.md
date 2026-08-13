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
