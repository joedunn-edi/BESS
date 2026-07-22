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
