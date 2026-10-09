"""The readings ledger (BLUEPRINT §14.1): the leash as architecture.

Recognition's leash (BLUEPRINT §14) says a recognizer may be wrong, but a wrong recognition may
never silently change a number. The third intelligence gate showed it cannot be enforced reader by
reader: after the roles were leashed, seven other heuristics still turned a name or a value into a
number-changing default without asking (an ``*_id`` read as the grouping identifier, ``day1_day2``
read as one day, a weight in pounds read as kilograms, an ``assessment`` column's occasions read as
repeats, ``bmi_kg`` given the unit kg, counts combined as codes, a survey design read by name over
the user's confirmed roles). So the invariant lives here, once:

* **Every interpretation the engine makes about the data is a** :class:`Reading`: its subject
  (a column), its kind (:data:`KINDS`), its value, its confidence, its evidence, whether the values
  corroborate it, and its state (proposed, or confirmed by the user).
* **A reading is settled** (:func:`settled`) when its values corroborate it at high confidence or
  the user confirmed it. Confirmation is one reading at a time: ``confirm_reading`` (one column,
  one kind, one value per record; ``confirm_role`` records stay valid as the role kind's), or the
  kind's own answer (``set_column_unit``, ``set_outcome_unit``, ``set_categorical``,
  ``set_repeat_kind``, ``set_grain``, ``set_survey``), never a bulk confirm of uncertain readings.
* **Every number-changing consumer reads only settled readings** (:data:`CONSUMERS` names each
  one and its conservative path). When a reading it needs is unsettled it asks, with one exit per
  reading (:func:`confirm_exit`), or takes the conservative path: no unit in a sentence, no
  clustering without a confirmed unit, no screen until the unit and the days are settled, no
  combining rule for a column whose code-or-count reading is unsettled, no fit while a role the
  predictor set reads waits for its confirmation.

**Corroboration must discriminate** (BLUEPRINT §14.3). Each kind of reading declares its plausible
alternatives in :data:`KIND_RULES`, and the values settle it only through a test that rejects every
one of them (a household's line number repeats as a subject's identifier does; a measurement varies
within units as a time does; FIPS states hold 51 codes and a blank writes codes 1–5 as 1.0–5.0; a
country of birth has 70 labels; ``uL`` is no U/L; studies code sex 1/2 either way). A kind with no
such test is settled only by the user. Names never count as corroboration. The consumer sets the
question's scope, not the reader (:func:`code_question`: every whole-valued predictor, any type, any
count of values), and every confirmation is honored wherever it is stored (:func:`confirmation`).
The ask stays light (:func:`ask_exits`): the unsettled readings a consumer needs are listed once,
ordered by consequence, a homogeneous family grouped, each led by its best guess and its evidence,
and one block confirmation (``confirm_readings``) settles exactly the readings it lists, each with
the value it shows.

**The codebook is an evidence source** (BLUEPRINT §14.2, "let the codebook answer"; V2 definition
of done §1). An imported codebook's structured fields (a unit, a value-code table, the variable
type) are the user's own documentation: ``import_codebook`` records them through the confirmation
path ``confirm_readings`` writes, so every consumer honors them as it honors a confirmation, and the
settle rules are unchanged. The ledger names the codebook as their evidence
(:func:`codebook_source`, :func:`recorded_evidence`). Its free-text labels are names: they settle
nothing, and only strengthen the guess an ask leads with (:func:`labeled`).

The structural acceptance test enumerates :data:`CONSUMERS` against the independent census of
readers and consumers, and checks that each number-changing consumer calls into this module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

ATTENTION = ("medium", "low")
CONFIDENCES = ("high", "medium", "low")

# What each kind of reading says about the data (the subject is a column).
KINDS: dict[str, str] = {
    "role": "the column's role in the analysis",
    "cluster": "whether the column's repeating values mark rows that belong together",
    "unit": "the unit the column's values are in",
    "day_count": "how many days each total-energy value spans",
    "code_or_count": "whether the column's whole numbers are codes for categories or amounts",
    "outcome_unit": "the unit the outcome is stated in",
    "repeat_kind": "whether a unit's rows are repeats of one measurement or time points",
    "time_column": "the column that orders a unit's rows",
    "task": "what kind of outcome the column is",
    "design": "the part of a survey design the column is",
    "nested_in": "the total the column is a part of",
    "orientation": "whether the table holds one row per sample",
    "sex_coding": "which level of a sex column is female",
    "detection_limit": "the value a result below a detection limit is read at",
    "injection_order": "the column that orders an assay run's injections",
    "batch": "the column naming each injection's analytical batch",
    "time_invariant": "whether the column holds one value for each unit of clustered rows",
}
# The kinds ``confirm_reading`` records (the others are answered by their own decisions).
CONFIRMABLE = ("role", "cluster", "unit", "day_count", "code_or_count", "time_column",
               "nested_in", "sex_coding", "time_invariant")
# The values a confirmation may record, per kind (``day_count``: a whole number of days).
UNIT_VALUES = ("kcal", "kj", "g", "kg", "lb", "cm", "m", "in", "years", "months", "weeks", "days",
               "pct_energy")
VALUES: dict[str, tuple[str, ...]] = {
    "role": ("identifier", "exposure", "energy", "covariate", "design", "flag", "time",
             "excluded", "cluster"),
    "cluster": ("yes", "no"),
    "unit": UNIT_VALUES,
    "code_or_count": ("code", "amount"),
    "time_column": ("orders",),
    "time_invariant": ("yes", "no"),
}

PREDICTOR_ROLES = ("exposure", "covariate", "energy")
# Whole numbers with this many distinct values or fewer may be codes or amounts (working.py's
# CODE_LEVELS: smoking 1/2/3, education 1–5, cups of coffee 0–6); two values give one line either
# way (an indicator is its own slope), and more than this are read as amounts by their values.
CODE_LEVELS = 10


@dataclass(frozen=True)
class Reading:
    """One interpretation of the data. ``corroborated``: the values agree with it (a name alone
    never does). ``state``: ``proposed``, or ``confirmed`` when the user recorded it."""

    subject: tuple[str, ...]
    kind: str
    value: Any
    confidence: str = "medium"
    evidence: str = ""
    corroborated: bool = False
    state: str = "proposed"

    @property
    def column(self) -> str:
        return self.subject[0]

    @property
    def settled(self) -> bool:
        return settled(self)

    def to_dict(self) -> dict[str, Any]:
        return {"subject": list(self.subject), "kind": self.kind, "value": self.value,
                "confidence": self.confidence, "evidence": self.evidence,
                "corroborated": self.corroborated, "state": self.state,
                "settled": self.settled}


def settled(reading: Reading | None) -> bool:
    """BLUEPRINT §14.1: confirmed by the user, or corroborated by its values at high confidence."""
    if reading is None:
        return False
    return reading.state == "confirmed" or (reading.confidence == "high" and reading.corroborated)


def reading(kind: str, column: str, value: Any, *, confidence: str = "medium",
            evidence: str = "", corroborated: bool = False, state: Any = None) -> Reading:
    """A reading as the ledger holds it: confirmed when ``state`` records this value for it
    (:func:`confirmation`), else as proposed."""
    if state is not None:
        recorded = confirmation(state, kind, column)
        if recorded is not None:
            return Reading((str(column),), kind, recorded, "high",
                           recorded_evidence(state, kind, column), corroborated, "confirmed")
    return Reading((str(column),), kind, value, confidence, evidence, corroborated, "proposed")


def key(kind: str, column: str) -> str:
    return f"{kind}:{column}"


def _confirmations(state: Any) -> Mapping[str, Any]:
    """Every reading the user confirmed on its own but a role (``confirm_reading``): the
    table-shaping ones (``shape_confirmations``) and the rest (``reading_confirmations``)."""
    return {**(_get(state, "reading_confirmations") or {}),
            **(_get(state, "shape_confirmations") or {})}


def _get(obj: Any, name: str) -> Any:
    """A slot or a spec's field, from a model or its dict form (a state copied with an update
    holds the dict until it is validated)."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def confirmation(state: Any, kind: str, column: str | None) -> Any:
    """The value the user recorded for this reading, or None: its own ``confirm_reading``, or the
    kind's own answer (a role recorded by the user's own change, ``set_column_unit``,
    ``set_outcome_unit``, ``set_categorical``, the aggregation's rule for the column, the repeat
    kind, the time column named with it or with the temporal answer, the grain's unit)."""
    if state is None or column is None:
        return None
    if kind in ("unit", "day_count"):
        # One store (BLUEPRINT §14.3): ``set_column_unit`` and a unit's or a day count's own
        # confirmation write the same spec, so every consumer reads the latest answer here.
        spec = unit_record(state, column)
        if spec is None:
            return None
        if kind == "unit":
            unit = _get(spec, "unit")
            if unit == DRINKS and _get(spec, "grams_per_drink") is not None:
                return drinks_value(float(_get(spec, "grams_per_drink")))
            return unit
        days = _get(spec, "days")
        return None if days is None else int(days)
    own = _confirmations(state).get(key(kind, column))
    if own is not None:
        return own
    if kind == "sex_coding":
        return (_get(state, "sex_codings") or {}).get(column)
    if kind == "role":
        legacy = (getattr(state, "role_confirmations", None) or {}).get(column)  # confirm_role
        return legacy
    if kind == "outcome_unit":
        unit = getattr(state, "outcome_unit", None)
        if unit and getattr(state, "target", None) == column:
            return unit
        return codebook_unit(state, column)
    if kind == "code_or_count":
        if column in (_get(state, "categorical") or []):
            return "code"
        agg = _get(state, "aggregation")
        rule = (_get(agg, "columns") or {}).get(column) if agg is not None else None
        if rule is not None:
            return "code" if rule == "mode" else "amount"
        return None
    if kind == "repeat_kind":
        spec = _get(state, "repeat_kind")
        return str(_get(spec, "repeat_kind")) if spec is not None else None
    if kind == "time_column":
        for spec in (_get(state, "repeat_kind"), _get(state, "temporal")):
            if spec is not None and _get(spec, "time_column") == column:
                return "orders"
        return None
    if kind == "cluster":
        grain = _get(state, "grain")
        if grain is not None and _get(grain, "id_column") == column:
            return "yes" if _get(grain, "grain") == "repeated" else None
        return None
    return None


def unit_record(state: Any, column: str | None, *, units: Mapping[str, Any] | None = None) -> Any:
    """The one accessor of a column's recorded unit and day count (BLUEPRINT §14.3, every
    confirmation is honored): the spec ``set_column_unit``, ``confirm_reading`` and
    ``confirm_readings`` write together (``ProjectState.column_units``), its ``unit`` and ``days``
    each None while unrecorded; None when nothing is recorded. A stage that holds the state's
    units mapping alone passes it as ``units``."""
    if column is None:
        return None
    held = units if units is not None else _get(state, "column_units")
    return held.get(column) if isinstance(held, Mapping) else None


# ── the codebook as an evidence source (BLUEPRINT §14.2) ──────────────────────


CODEBOOK_EVIDENCE = "from the codebook"


def _codebooks(state: Any) -> list[Any]:
    """The imported codebooks, in import order (``ProjectState.codebooks``)."""
    books = _get(state, "codebooks") or {}
    return list(books.values()) if isinstance(books, Mapping) else []


def codebook_source(state: Any, kind: str, column: str | None) -> str | None:
    """The name of the codebook whose structured field recorded this reading's current value
    (the latest import that wrote it), or None: the value was recorded by the user, or nothing
    is recorded."""
    if column is None:
        return None
    books = _codebooks(state)
    if not books:
        return None
    recorded = confirmation(state, kind, column)
    if recorded is None:
        return None
    for spec in reversed(books):
        written = (_get(spec, "settled") or {}).get(key(kind, column))
        if written is not None and str(written) == str(recorded):
            return str(_get(spec, "name"))
    return None


def recorded_evidence(state: Any, kind: str, column: str | None) -> str:
    """A recorded reading's evidence: the codebook that documents it (``from the codebook
    `DEMO_J.htm```), else the user's own answer."""
    name = codebook_source(state, kind, column)
    return f"{CODEBOOK_EVIDENCE} `{name}`, the user's own documentation" if name else \
        "recorded by the user"


def codebook_unit(state: Any, column: str | None) -> str | None:
    """The unit the latest imported codebook documents for ``column`` (any unit: ``mg/dL`` as
    well as the reading kinds' ``kg``), or None. A documented unit is the user's own; an outcome
    whose unit it documents is stated in it (:func:`confirmation`, kind ``outcome_unit``).

    A recorded answer stands (BLUEPRINT §14.2): when the user's own answer to the column's unit
    reading says otherwise, given before the import or after it, the codebook documents no unit a
    sentence states (None), never its own in place of the user's."""
    if column is None:
        return None
    for spec in reversed(_codebooks(state)):
        unit = (_get(spec, "units") or {}).get(column) or \
            (_get(spec, "settled") or {}).get(key("unit", column))
        if unit:
            return None if _unit_answered_otherwise(state, column, str(unit)) else str(unit)
    return None


def _unit_answered_otherwise(state: Any, column: str, documented: str) -> bool:
    """Whether the column's unit reading holds the user's own answer (not a codebook's), and it
    differs from ``documented``."""
    recorded = confirmation(state, "unit", column)
    if recorded is None or codebook_source(state, "unit", column) is not None:
        return False
    from turbotab.core.codebook import unit_value

    return str(recorded) != (unit_value(documented) or documented)


def codebook_label(state: Any, column: str | None) -> str | None:
    """The free-text label the latest imported codebook gives ``column``, or None. A label is a
    name: it never settles a reading (BLUEPRINT §14.3), and only strengthens a guess
    (:func:`labeled`)."""
    if column is None:
        return None
    for spec in reversed(_codebooks(state)):
        label = (_get(spec, "labels") or {}).get(column)
        if label:
            return str(label)
    return None


def recorded_energy(state: Any, column: str | None, *,
                    units: Mapping[str, Any] | None = None) -> tuple[str | None, int | None]:
    """``(unit, days)`` recorded for a total-energy column (:func:`unit_record`), each None while
    unrecorded. A recorded unit that is no energy unit (``g``) is returned as given, so the
    consumer refuses it rather than reading it as kcal."""
    spec = unit_record(state, column, units=units)
    if spec is None:
        return None, None
    unit = _get(spec, "unit")
    days = _get(spec, "days")
    return (None if unit is None else str(unit)), (None if days is None else int(days))


class Unsettled(ValueError):
    """A number-changing consumer met a reading that is not settled: it asks, never guesses.
    ``exits`` are one confirmation per reading (never one for all)."""

    def __init__(self, message: str, readings: Sequence[Reading] = (),
                 exits: Sequence[Mapping[str, Any]] = ()):
        super().__init__(message)
        self.readings = list(readings)
        self.exits = [dict(e) for e in exits]


# ── the kind registry (BLUEPRINT §14.3: corroboration must discriminate) ─────
#
# The fourth gate found the ledger held, but six readers settled themselves on evidence that fits
# the alternatives as well: repeating values read as a subject's identifier (a household's line
# number repeats too), a value that varies within units read as time (so does any measurement),
# "integer, at most 10 levels" read as the only codes worth asking about (FIPS states have 51; a
# blank turns 1–5 into 1.0–5.0), a label count read as free text (country of birth has 70), `uL`
# read as U/L, and `sex` coded 1/2 read as the CDC's coding. So each kind of reading declares its
# plausible alternatives, and the values settle it only through a test that rejects every one of
# them; a kind with no such test is settled by the user (or the user's codebook, §14.2). Names
# are never corroboration: a name chooses what is proposed and what is asked, never what is
# settled.


@dataclass(frozen=True)
class Verdict:
    """A value test's answer: whether the values settle the reading, the value they settle it at,
    and the evidence a question or a sentence shows."""

    settles: bool
    value: Any = None
    evidence: str = ""
    # The readings the values leave open when they settle nothing, each a value the question
    # offers (the energy unit: ``(unit, days)`` pairs a reconstruction ratio fits).
    candidates: tuple[Any, ...] = ()
    # What the test read, for a consumer's words and its exits, never for a settlement of its
    # own: a text column's numbers and marks, a nutrient's intake check, the Atwater reading.
    detail: Any = None


@dataclass(frozen=True)
class KindRule:
    """One kind of reading as the registry declares it. ``alternatives``: every plausible other
    meaning of what the reader saw. ``test``: the value test that rejects each of them
    (``rejects``: how, one line per alternative), or None when no honest test exists and the
    reading is settled only by ``settled_by``. ``table``: the same test over a whole table, where
    a column's verdict reads the others (duplicates of one nutrient, the identity they sum to).
    ``helpers``: the functions (``module:function``) the test, or a kind without one its guess,
    reads its values with; outside this module they are called only through :func:`by_values` and
    :func:`by_values_table`, or where :data:`EVIDENCE_ONLY` says no number follows (BLUEPRINT
    §14.3: one test per kind, no private settler; the structural test holds it)."""

    kind: str
    reads: str
    alternatives: tuple[str, ...]
    test: Any = None
    rejects: tuple[tuple[str, str], ...] = ()
    settled_by: str = "the user's own answer"
    table: Any = None
    helpers: tuple[str, ...] = ()

    @property
    def value_settleable(self) -> bool:
        return self.test is not None


def by_values(kind: str, *args: Any, **kwargs: Any) -> Verdict:
    """The one way a consumer settles a kind by its values (BLUEPRINT §14.3): the registry's test
    for it (:data:`KIND_RULES`), never a test of the consumer's own. A kind with no value test is
    settled only by the user, so asking it here is an error."""
    rule = KIND_RULES[kind]
    if rule.test is None:
        raise ValueError(f"no value test settles the {kind} reading: {rule.settled_by}")
    return rule.test(*args, **kwargs)


def by_values_table(kind: str, *args: Any, **kwargs: Any) -> dict[str, Verdict]:
    """The registry's test for ``kind`` over a whole table: each column's verdict
    (:attr:`KindRule.table`), the same test :func:`by_values` runs on one column."""
    rule = KIND_RULES[kind]
    if rule.table is None:
        raise ValueError(f"the {kind} reading has no test over a whole table")
    return rule.table(*args, **kwargs)


# Whole numbers all different on n rows within a span this many times narrower than n² / 2: a
# measurement spread as widely as any distribution can be over that span (uniformly) would repeat
# this many times on average, so P(no repeat) ≤ e^-10 (the birthday bound).
ROWS_COLLISIONS = 10.0


def _present(values: Any) -> Any:
    import pandas as pd

    return pd.Series(values).dropna()


def _numbers(s: Any) -> Any:
    import numpy as np
    import pandas as pd

    if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
        return None
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    return x[np.isfinite(x)]


def _fmt_value(v: float) -> str:
    # Six significant digits: a code's decimals are its meaning (ICD-9-CM 250.02 is no `250`).
    return f"{int(v):,}" if float(v).is_integer() else f"{v:,.6g}"


def names_rows(values: Any) -> Verdict:
    """The identifier role of a column that names rows: one value per row, and values no
    measurement would take. Rejects a repeating code (a household's line number 1…k, a stratum, an
    interviewer, a subject's identifier in a long table: whether rows sharing one belong together is
    the cluster reading, the user's to settle) and a measurement whose values happen to differ (text
    labels never are one; whole numbers all different within a span narrow enough that a
    measurement would repeat :data:`ROWS_COLLISIONS` times)."""
    import numpy as np
    import pandas as pd

    s = _present(values)
    n, k = int(len(s)), int(s.nunique())
    if n < 3:
        return Verdict(False, None, f"only `{n}` values")
    if k < n:
        most = int(s.value_counts().max())
        return Verdict(False, None, f"`{k:,}` values on `{n:,}` rows, up to `{most:,}` rows each: "
                                    f"a value that repeats may name a unit, or number people "
                                    f"within a household, a stratum or an interviewer")
    x = _numbers(s)
    if x is None:
        if pd.api.types.is_bool_dtype(s):
            return Verdict(False, None, "yes/no values")
        return Verdict(True, "identifier", f"`{n:,}` labels, one per row")
    if len(x) < n or not np.all(x == np.floor(x)):
        return Verdict(False, None, "fractional values, as a measurement's are")
    span = float(x.max() - x.min() + 1)
    expected = n * (n - 1) / (2.0 * span)
    if expected >= ROWS_COLLISIONS:
        return Verdict(True, "identifier",
                       f"`{n:,}` whole numbers, each on one row, within a span of `{span:,.0f}`: a "
                       f"measurement spread over so narrow a span would repeat about "
                       f"`{expected:,.0f}` times")
    return Verdict(False, None, f"`{n:,}` whole numbers, each on one row, over a span of "
                                f"`{span:,.0f}`, wide enough for a measurement's values to differ")


def dates_order_rows(values: Any, units: Any = None, *, varies: float = 0.5) -> Verdict:
    """The time role: dates that change within units (a date is no amount, and it changes within
    a unit's rows only as the rows' own times do). Numbers and labels never settle it: a
    measurement in a time unit (hours slept, days active) or a crossover's treatment varies within
    units as readily as a visit index does; a date constant within units (a birth or randomization
    date) orders nothing."""
    import pandas as pd

    from turbotab.core.recognizers import within_unit_variation

    s = pd.Series(values)
    if not pd.api.types.is_datetime64_any_dtype(s):
        return Verdict(False, None, "numbers or labels, which a measurement in a time unit or a "
                                    "within-unit treatment changes as a time does")
    if units is None:
        return Verdict(False, None, "no unit repeats here, so nothing shows it orders a unit's rows")
    share = within_unit_variation(s, units)
    if share is None:
        return Verdict(False, None, "no unit repeats here, so nothing shows it orders a unit's rows")
    if share >= varies:
        return Verdict(True, "time", f"dates that change within `{share:.0%}` of units")
    return Verdict(False, None, f"dates that change within only `{share:.0%}` of units: a date "
                                f"of an event (birth, randomization), not of each row")


# Values with decimals settle "amount" only where they fill their grid as a measurement's do
# (BLUEPRINT §14.3, amendment after the fifth gate: ICD-9-CM 307.1, 307.51, 250.02, 401.9 are codes
# written with a decimal point). A measurement recorded at a resolution (0.1 kg, 0.01 %) spreads
# its rows over the grid of that resolution, so the middle half of its distinct values occupies
# about as many grid points as its rows can: ``G · (1 − e^(−n/G))`` for ``n`` rows over ``G``
# points (the occupancy expectation). A code list's values sit apart on that grid (307.1, 307.5,
# 307.51, 307.59 leave 38 of 42 hundredths empty), and a short list (ten values or fewer) is
# asked whatever its spacing. One definition serves a column read here and one read by the store
# (``DataStore.whole_numbers``): :func:`grid_reading` over the distinct values and their counts.
GRID_FILL = 0.75  # the share of the occupancy expectation a measurement's middle half reaches
GRID_MAX_VALUES = 200_000  # more distinct values than any code list: read as a measurement's
GRID_DECIMALS = 6  # finer than this, values are read as continuous (no grid repeats them)
GRID_COPIES = 0.8  # the share of values' counts one copy count divides (a long table's repeats)


def _on_grid(x: Any, d: int) -> Any:
    import numpy as np

    scaled = np.asarray(x, dtype=float) * 10.0 ** d
    return np.abs(scaled - np.round(scaled)) <= 1e-6 * np.maximum(1.0, np.abs(scaled))


def decimals_of(values: Any) -> int | None:
    """The fewest decimals that write every value exactly, or None beyond :data:`GRID_DECIMALS`."""
    import numpy as np

    x = np.asarray(values, dtype=float)
    return next((d for d in range(GRID_DECIMALS + 1) if bool(np.all(_on_grid(x, d)))), None)


def _needed_decimals(values: Any) -> Any:
    """Each value's own decimals (``GRID_DECIMALS + 1`` beyond the grid's finest)."""
    import numpy as np

    x = np.asarray(values, dtype=float)
    out = np.full(len(x), GRID_DECIMALS + 1)
    for d in range(GRID_DECIMALS, -1, -1):
        out[_on_grid(x, d)] = d
    return out


def grid_reading(distinct: Any, counts: Any) -> dict[str, Any]:
    """Whether a column's values fill their grid as a measurement's do (see :data:`GRID_FILL`):
    ``distinct`` values and how many rows hold each.

    * The grid is the precision most rows were recorded at: the values that need exactly that
      many decimals (NHANES writes most glucose values whole and some to 0.1; an export's imputed
      or averaged values, 70.0333, sit off the resolution the rest were recorded at), stepped by
      the greatest common difference among them (halves for a 0.5-step scale).
    * Its band is the middle half of those distinct values by rank, so a value most rows share (an
      abstainer's 0 g) neither widens nor narrows it.
    * Rows that copy one another (a long table repeating a person's height on each of three
      visits: the counts 3, 6, 9) count once per copy: when the most common count is two or more
      and divides :data:`GRID_COPIES` of the values' counts, each count is divided by it. A code
      list's counts follow its codes' shares and share no divisor."""
    import math

    import numpy as np

    u = np.asarray(distinct, dtype=float)
    c = np.asarray(counts, dtype=float)
    order = np.argsort(u)
    u, c = u[order], c[order]
    k, n = int(len(u)), int(c.sum())
    out: dict[str, Any] = {"k": k, "n": n, "fills": False, "k_band": 0, "n_band": 0,
                           "grid": None, "step": None, "expected": None}
    if k < 2:
        return out
    if k >= GRID_MAX_VALUES:
        out.update(fills=True, expected=float(k), k_band=k, n_band=n)
        return out
    whole_counts = np.round(c).astype(np.int64)
    sizes, freq = np.unique(whole_counts, return_counts=True)
    copies = int(sizes[np.argmax(freq)])
    if copies >= 2 and float(np.mean(whole_counts % copies == 0)) >= GRID_COPIES:
        c = np.maximum(c / copies, 1.0)
    need = _needed_decimals(u)
    rows = {int(dd): float(c[need == dd].sum()) for dd in np.unique(need)}
    d = max(rows, key=lambda dd: (rows[dd], -dd))
    u, c = u[need == d], c[need == d]
    if d > GRID_DECIMALS:
        d = None
    m = int(len(u))
    if m < 2:
        return out
    lo_i = m // 4
    hi_i = min(max(lo_i + 1, (3 * m - 1) // 4), m - 1)
    k_band = hi_i - lo_i + 1
    n_band = int(c[lo_i:hi_i + 1].sum())
    if d is None:
        expected, grid, step = float(n_band), None, None
    else:
        scaled = np.round(u * 10.0 ** d).astype(np.int64)
        g = int(np.gcd.reduce(np.diff(scaled))) or 1
        step = g / 10.0 ** d
        grid = int(round((scaled[hi_i] - scaled[lo_i]) / g)) + 1
        expected = grid * (1.0 - math.exp(-n_band / grid))
    out.update(k_band=k_band, n_band=n_band, grid=grid, step=step, expected=expected,
               fills=bool(k_band >= GRID_FILL * expected))
    return out


def _grid_of(x: Any) -> dict[str, Any]:
    import numpy as np

    u, c = np.unique(np.asarray(x, dtype=float), return_counts=True)
    return grid_reading(u, c)


def fractional_verdict(grid: Mapping[str, Any], first: float | None = None) -> Verdict:
    """The code-or-amount verdict for values with decimals, from :func:`grid_reading`: an amount
    only with more than :data:`CODE_LEVELS` values that fill their grid; else asked."""
    k = int(grid.get("k") or 0)
    shown = f" (`{_fmt_value(float(first))}`)" if first is not None else ""
    if k <= CODE_LEVELS:
        return Verdict(False, None,
                       f"`{k:,}` values with decimals{shown}, each on many rows: codes written with "
                       f"a decimal point (ICD-9-CM 307.1, 250.02) look like this")
    if not grid.get("fills"):
        step = grid.get("step")
        on = f" of {_fmt_value(float(step))}" if step else ""
        return Verdict(False, None,
                       f"`{k:,}` values with decimals{shown} that sit apart on their grid{on} "
                       f"(the middle half holds `{int(grid.get('k_band') or 0):,}` where a "
                       f"measurement's rows would fill about `{float(grid.get('expected') or 0):,.0f}`), "
                       f"as a code list's do")
    return Verdict(True, "amount",
                   f"`{k:,}` values with decimals that fill their grid as a measurement's do (a "
                   f"code list's sit apart)")


def amounts_by_values(values: Any, counts: Any = None) -> Verdict:
    """Codes or amounts. ``counts``: how many rows hold each of ``values`` when they are a
    column's distinct values (the store reads them so, without materializing the column).

    * Text settles "code" only as labels: a text column whose values are mostly numbers once its
      missing marks (``.``, ``.A``–``.Z``, ``NA``, blanks) and censoring marks (``<0.20``,
      ``>200``, ``<LOD``) are set aside is numbers written as text (:func:`text_numbers`: the
      gate's SAS-exported BMI with ten ``.`` read as 175 categories), so it settles nothing.
    * Values with decimals are amounts only where there are more than :data:`CODE_LEVELS` of them
      and they fill their grid as a measurement's do (:func:`grid_reading`): ICD-9-CM 307.1, 307.51
      and 250.02 are codes written with a decimal point.
    * Whole numbers settle nothing, whatever their count or type: FIPS states hold 51 codes, UK
      Biobank's ethnic background 22 from -3 to 4003, and a blank writes codes 1–5 as 1.0–5.0."""
    import numpy as np
    import pandas as pd

    if counts is not None:
        distinct = pd.Series(values)
        weights = np.asarray(counts, dtype=float)
        keep = distinct.notna().to_numpy()
        distinct, weights = distinct[keep].reset_index(drop=True), weights[keep]
        if distinct.empty:
            return Verdict(False, None, "no values")
        s = distinct
    else:
        s = _present(values)
        weights = None
        if s.empty:
            return Verdict(False, None, "no values")
    if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
        found = text_numbers(s, weights)
        if found is not None:
            return Verdict(False, None, found["evidence"], candidates=("amount", "code"),
                           detail=found)
        return Verdict(True, "code", "labels, not numbers", detail={"text_numbers": False})
    if weights is None:
        x = _numbers(s)
        fractional = x[x != np.floor(x)] if x is not None else x
        if fractional is not None and len(fractional):
            return fractional_verdict(_grid_of(x), float(fractional[0]))
    else:
        u = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(u)
        u, w = u[finite], weights[finite]
        fractions = u[u != np.floor(u)]
        if len(fractions):
            if len(u) >= GRID_MAX_VALUES:
                return fractional_verdict({"k": len(u), "fills": True}, None)
            return fractional_verdict(grid_reading(u, w), float(np.sort(fractions)[0]))
        x = u
    k = int(len(np.unique(x))) if x is not None else 0
    lo = _fmt_value(float(x.min())) if x is not None and len(x) else "?"
    hi = _fmt_value(float(x.max())) if x is not None and len(x) else "?"
    return Verdict(False, None, f"`{k:,}` whole-number values from {lo} to {hi}")


# A text column is numbers written as text when its numbers are this share of the values left once
# the missing and censoring marks are set aside: the repair registry's own threshold for "numbers
# stored as text" (``repairs.NUMBER_SHARE``), so the finding, its "Read as numbers" repair and this
# reading are one definition.
TEXT_NUMBER_SHARE = 0.8


def text_numbers(values: Any, counts: Any = None) -> dict[str, Any] | None:
    """Numbers written as text (BLUEPRINT §14.3, the sixth gate: a BMI exported from SAS with ten
    ``.`` for missing, a CRP with fifteen ``<0.20``): what :func:`repairs.parse_text_numbers
    <turbotab.core.repairs.parse_text_numbers>` reads, when the numbers are at least
    :data:`TEXT_NUMBER_SHARE` of the values left once the missing marks and the censoring marks are
    set aside; else None (labels). With ``text_numbers`` True, the counts, and the ``evidence`` a
    question shows."""
    from turbotab.core.repairs import parse_text_numbers

    found = parse_text_numbers(values, counts)
    if found is None or not found["numbers"]:
        return None
    marks = sum(found["marks"].values())
    below, above = sum(found["below"].values()), sum(found["above"].values())
    censored = below + above + sum(found["censored"].values())
    rest = found["n_values"] - marks - censored
    if rest <= 0 or found["numbers"] < TEXT_NUMBER_SHARE * rest:
        return None
    return {**found, "text_numbers": True, "n_marks": marks, "n_below": below, "n_above": above,
            "n_censored": censored, "evidence": text_numbers_evidence(found)}


def _spellings(found: Mapping[str, int], limit: int = 3) -> str:
    items = list(found.items())
    shown = ", ".join(f"`{s or '(blank)'}` ×{c:,}" for s, c in items[:limit])
    return shown + (f" and {len(items) - limit:,} more" if len(items) > limit else "")


def text_numbers_evidence(found: Mapping[str, Any]) -> str:
    """What a text column that holds numbers shows a question: its numbers, its missing marks,
    the values censored at a limit, and what reading it as amounts makes of each."""
    n, numbers = int(found["n_values"]), int(found["numbers"])
    parts = [f"`{numbers:,}` of `{n:,}` values are numbers written as text"]
    if found.get("marks"):
        parts.append(f"missing marks {_spellings(found['marks'])} (SAS and Stata write a missing "
                     f"value as `.`), blank as amounts")
    below = found.get("below") or {}
    if below:
        first = next(iter(below))
        parts.append(f"`{sum(below.values()):,}` below a detection limit (`<{first}`), read as "
                     f"amounts at the detection-limit answer")
    above = found.get("above") or {}
    if above:
        first = next(iter(above))
        parts.append(f"`{sum(above.values()):,}` above a limit (`>{first}`), blank as amounts")
    if found.get("censored"):
        parts.append(f"censored with no limit {_spellings(found['censored'])}, blank as amounts")
    if found.get("other"):
        parts.append(f"other text {_spellings(found['other'])}, blank as amounts")
    return "; ".join(parts) + ": numbers with missing marks, or labels"


def labels_spell_sex(values: Any) -> Verdict:
    """Which level of a sex column is female: settled when the values are the words themselves
    (``F``/``M``, ``female``/``male``), never from numeric codes, which studies write either way
    (NHANES RIAGENDR codes 1 male, 2 female; others code 1 female, or 0/1)."""
    import pandas as pd

    s = _present(values)
    if s.empty or (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)):
        levels = ", ".join(f"`{_level(v)}`" for v in sorted(pd.unique(s))[:4]) if len(s) else ""
        return Verdict(False, None, f"numeric codes ({levels}), which studies assign to female "
                                    f"and male either way")
    found: dict[str, str] = {}
    for v in pd.unique(s):
        word = str(v).strip().lower()
        if word in FEMALE_WORDS:
            found[_level(v)] = "female"
        elif word in MALE_WORDS:
            found[_level(v)] = "male"
        else:
            return Verdict(False, None, f"`{v}` is no word for female or male")
    if sorted(found.values()) != ["female", "male"]:
        return Verdict(False, None, "the labels do not hold both sexes")
    female = next(k for k, v in found.items() if v == "female")
    return Verdict(True, f"female={female}", f"the labels spell the sexes (`{female}` female)")


FEMALE_WORDS = frozenset({"f", "female", "woman", "women", "w", "girl", "fem"})
MALE_WORDS = frozenset({"m", "male", "man", "men", "boy", "masc"})


def _level(value: Any) -> str:
    """A level as a rule names it: ``1.0`` and ``1`` are one level; text is stripped."""
    import math

    if isinstance(value, bool):
        return str(value)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if math.isnan(f):
        return ""
    return str(int(f)) if f.is_integer() else repr(f)


def energy_follows_macronutrients(frame: Any, column: str,
                                  candidates: Mapping[str, Sequence[str]] | None = None) -> Verdict:
    """Total energy intake: its values follow the energy the macronutrients in grams carry (FAO
    factors 4/4/9/7) at r ≥ 0.7 (``recognizers.energy_against_macros``). A device's energy
    expenditure (Fitbit ``Calories``) or an energy requirement computed from body size does not.
    ``candidates``: every macronutrient total to choose among (default: every one but ``column``),
    so the values choose among duplicates (an InBody body ``Protein`` beside a recall's
    ``protein_g``)."""
    from turbotab.core.recognizers import energy_against_macros, macro_candidates

    if candidates is None:
        candidates = macro_candidates(frame, exclude=[column])
    check = energy_against_macros(frame, column, candidates=candidates)
    if check is None:
        return Verdict(False, None, "no macronutrients to read it against")
    return Verdict(bool(check.by_values), "energy" if check.by_values else None, check.why,
                   detail=check)


def characteristic_predictor(name: Any, values: Any, units: Any = None) -> Verdict:
    """A person's characteristic kept as a predictor: values that fit what the name says
    (``stages.rows.characteristic_fits``: a sex with 2–3 levels, an age from 0 to 120 at its
    median, a BMI's median within 10–80), which no identifier or constant has, and that do not
    change within most units (an age at each visit may be the time axis)."""
    from turbotab.core.recognizers import within_unit_variation
    from turbotab.core.stages.rows import characteristic_fits

    if not characteristic_fits(str(name), values):
        return Verdict(False, None, "its values do not fit the characteristic its name says")
    if units is not None:
        share = within_unit_variation(values, units)
        if share is not None and share >= 0.5:
            return Verdict(False, None, f"it changes within `{share:.0%}` of units: the time axis?")
    return Verdict(True, "covariate", "its values fit the characteristic")


def constant_values(values: Any) -> Verdict:
    """Left out because every row holds one value: no reading of a constant explains anything."""
    s = _present(values)
    if int(s.nunique()) <= 1:
        return Verdict(True, "excluded", "every row holds the same value")
    return Verdict(False, None, f"`{int(s.nunique()):,}` different values")


def height_in_band(values: Any) -> Verdict:
    """A height's unit by its median: 100–230 is a human height only in cm, 1.0–2.3 only in m (no
    one is 100–230 inches or 1–2.3 cm tall). A child's height in cm, or anyone's in inches, falls
    in neither band and is asked."""
    import numpy as np

    x = _numbers(_present(values))
    x = x[x > 0] if x is not None else None
    if x is None or not len(x):
        return Verdict(False, None, "no values")
    median = float(np.median(x))
    for unit, (lo, hi) in HEIGHT_BANDS.items():
        if lo <= median <= hi:
            return Verdict(True, unit, f"its median, {median:,.1f}, is a human height only in {unit}")
    return Verdict(False, None, f"its median, {median:,.1f}, is no adult height in cm or m")


# The energy each unit carries per kcal, and the days a total may span: a reconstruction ratio
# near ``factor × days`` reads as that unit over that many of the macronutrients' days.
ENERGY_UNIT_FACTORS = {"kcal": 1.0, "kj": 4.184}
RATIO_TOLERANCE = 0.05  # the pack's band around the kJ factor (nutrition._reconstruct)
MAX_TOTAL_DAYS = 31


def unit_day_candidates(ratio: float) -> list[tuple[str, int]]:
    """The ``(unit, days)`` readings a reconstruction ratio fits: kcal over the macronutrients'
    own days within the pack's pass band (0.90–1.10), and any unit's total over ``days`` times
    their days within :data:`RATIO_TOLERANCE` (a 4-day kcal total reads 4.00, inside the band
    around kJ's 4.184: both are offered)."""
    import math

    out: list[tuple[str, int]] = []
    if not (isinstance(ratio, (int, float)) and math.isfinite(ratio) and ratio > 0):
        return out
    for unit, factor in ENERGY_UNIT_FACTORS.items():
        for days in range(1, MAX_TOTAL_DAYS + 1):
            target = factor * days
            if unit == "kcal" and days == 1:
                fits = 0.90 <= ratio <= 1.10
            else:
                fits = abs(ratio - target) / target < RATIO_TOLERANCE
            if fits:
                out.append((unit, days))
    return out


def atwater_unit(frame: Any, energy: str) -> Verdict:
    """Total energy's unit by the Atwater identity (NUTRITION_PACK §01): its values against the
    energy its macronutrients in grams carry. Settled only where the ratio admits one reading:
    near 1, kcal over the macronutrients' own days (how many those are stays the user's). A ratio
    near 4.184 is kJ or a 4-day kcal total beside daily means (the fifth gate's 4.00), and a ratio
    near any whole number N ≥ 2 an N-day total: the values cannot tell, so the unit and the days
    are asked together, the fitting readings offered (:func:`unit_day_candidates`). With the
    macronutrients in another unit, or too few, it settles nothing."""
    from turbotab.core.methods.energy import atwater_check

    try:
        check = atwater_check(frame, energy)
    except Exception:  # noqa: BLE001 - a check that cannot run settles nothing
        check = None
    if check is None or check.verdict in ("mixed_units", "macros_not_grams"):
        return Verdict(False, None, "the macronutrients do not reconstruct it", detail=check)
    ratio = float(getattr(check, "ratio", float("nan")))
    fits = unit_day_candidates(ratio)
    if fits == [("kcal", 1)]:
        return Verdict(True, "kcal", "matches the energy its macronutrients carry: kcal, over the "
                                     "same days as they span", candidates=tuple(fits),
                       detail=check)
    if not fits:
        return Verdict(False, None, f"is {ratio:.2f}× the energy its macronutrients carry, which "
                                    f"no unit or day count explains", detail=check)
    words = " or ".join(f"{'kJ' if u == 'kj' else 'kcal'}" + (f" over {d} days" if d > 1 else "")
                        for u, d in fits)
    return Verdict(False, None, f"is about {ratio:.2f}× the energy its macronutrients carry: "
                                f"{words}, which the values cannot tell apart",
                   candidates=tuple(fits), detail=check)


# The share of total energy below which the Atwater identity cannot see a source's unit: the pack's
# pass band (0.90–1.10) holds a reconstruction whose every other source is exact even with this
# source's energy missing altogether (1 / (1 − s) ≤ 1.10 for s ≤ 1 − 1/1.10, about 9%), so the
# source could be in kilograms, in standard drinks or absent and the identity would still pass.
MINOR_SHARE = 1.0 - 1.0 / 1.10
# Alcohol's own unit: research exports record it in grams, in kcal, or in standard drinks, whose
# grams of ethanol are set by each country ("the modal standard drink size was 10 g pure ethanol,
# but variation was wide (8-20 g)", Kalinowski & Humphreys 2016, Addiction 111:1293). A drink of
# 14 g carries 98 kcal at alcohol's 7 kcal/g, a gram 7: a 14-fold difference the identity sees only
# where alcohol is a large share of energy, so alcohol's unit is always the user's (the sixth gate).
DRINK_GRAMS = (8.0, 20.0)


def _factor_alternatives(role: str, factor: float) -> tuple[tuple[str, float], ...]:
    """The other units an energy source in grams might be in, each as how many grams one of its
    units holds: kilograms 1,000; kcal 1/f; kJ 1/(4.184 f), ``f`` the source's kcal per gram."""
    from turbotab.core.methods.energy import KCAL_PER_KJ

    return (("kg", 1000.0), ("kcal", 1.0 / factor), ("kJ", 1.0 / (KCAL_PER_KJ * factor)))


def factor_in_grams(frame: Any, energy: str, column: str) -> Verdict:
    """An energy source's unit for its kcal per unit: grams, settled by the Atwater identity only
    where it excludes every other unit (BLUEPRINT §14.3; the sixth gate's US standard drinks of
    alcohol, which reconstruct at a ratio of 1.02 read as grams):

    * the identity holds between total energy and the macronutrients with this column among them
      (the pack's pass band, ratio 0.90–1.10);
    * the column is no alcohol: its standard drinks hold 8–20 g by country (:data:`DRINK_GRAMS`),
      and alcohol is a minor source, so its unit is the user's;
    * the column carries at least :data:`MINOR_SHARE` of the reconstructed energy on the median
      row: below that the band cannot see its unit at all;
    * read in each other unit it could be in (kg, kcal, kJ: :func:`_factor_alternatives`), the
      reconstruction leaves the band, so that unit is rejected.

    In kilograms (an InBody export's body protein), kcal, kJ or percent of energy the
    reconstruction misses by a factor of 1,000, about 4, 4.184 × that, or the percentages' sum."""
    import numpy as np
    import pandas as pd

    from turbotab.core.methods.energy import atwater_check
    from turbotab.nutrition import ATWATER, PASS_HIGH, PASS_LOW

    try:
        check = atwater_check(frame, energy)
    except Exception:  # noqa: BLE001 - a check that cannot run settles nothing
        check = None
    macros = dict(getattr(check, "macro_columns", None) or {})
    if check is None or check.verdict != "pass" or column not in set(macros.values()):
        return Verdict(False, None, "the Atwater identity does not read it in grams", detail=check)
    role = next(r for r, c in macros.items() if c == column)
    if role == "alcohol":
        return Verdict(False, None,
                       f"the Atwater identity holds with `{column}` read in grams, but alcohol is "
                       f"also recorded in kcal or in standard drinks (8–20 g a drink by country), "
                       f"and a minor source's unit is one the identity cannot see", detail=check)
    declared = pd.to_numeric(frame[energy], errors="coerce")
    parts = {c: pd.to_numeric(frame[c], errors="coerce").fillna(0.0) * ATWATER[r]
             for r, c in macros.items()}
    recon = sum(parts.values())
    usable = declared.notna() & (recon > 0) & (declared > 0)
    e, r = declared[usable].to_numpy(float), recon[usable].to_numpy(float)
    mine = parts[column][usable].to_numpy(float)
    share = float(np.median(mine / r)) if len(r) else 0.0
    if share < MINOR_SHARE:
        return Verdict(False, None,
                       f"`{column}` carries about {share:.0%} of the energy its macronutrients "
                       f"reconstruct, below the {MINOR_SHARE:.0%} the identity's 0.90–1.10 band can "
                       f"see: in another unit, or missing, the identity would hold as well",
                       detail=check)
    ratio = float(np.median(e / r))
    seen = []
    for unit, grams_per_unit in _factor_alternatives(role, float(ATWATER[role])):
        other = float(np.median(e / (r + mine * (grams_per_unit - 1.0))))
        if PASS_LOW <= other <= PASS_HIGH:
            return Verdict(False, None,
                           f"the Atwater identity holds with `{column}` read in grams (ratio "
                           f"{ratio:.2f}) and as well were it in {unit} (ratio {other:.2f})",
                           detail=check)
        seen.append(f"{unit} ({other:.2f})")
    return Verdict(True, "g", f"the Atwater identity holds with total energy in grams (ratio "
                              f"{ratio:.2f}) and fails were it in {', '.join(seen)}",
                   detail=check)


def flag_marks_blanks(values: Any, base: Any = None) -> Verdict:
    """A flag on another column: two values, one of which marks exactly where that column is blank
    (``recognizers.flag_values``). A yes/no characteristic marks no column's blanks; continuous
    values (an imputed copy) are no marker."""
    from turbotab.core.recognizers import flag_values

    check = flag_values(values, base)
    return Verdict(check.verdict == "flag", "flag" if check.verdict == "flag" else None, check.why)


def outcome_task_by_values(values: Any) -> Verdict:
    """The outcome's task by its values: two values (or two labels) are a binary outcome under any
    reading (an ordinal outcome with two levels is one); values with decimals that fill their grid
    as a measurement's do, more than :data:`CODE_LEVELS` of them, are a regression outcome
    (:func:`grid_reading`). Whole numbers settle nothing, whatever their type (a PHQ-9 total
    written 0.0–27.0 because one value is blank is the same score as 0–27, and an ordinal score, a
    count and a measurement all look like it); nor do three or more labels (ordered or not)."""
    import numpy as np
    import pandas as pd

    s = _present(values)
    k = int(s.nunique())
    if k < 2:
        return Verdict(False, None, f"`{k}` value: nothing to tell apart")
    if k == 2:
        return Verdict(True, "binary", "two values: a binary outcome under any reading")
    if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
        return Verdict(False, None, f"`{k:,}` labels, which may be ordered (an ordinal outcome) or "
                                    f"not", candidates=("multiclass", "ordinal"))
    x = _numbers(s)
    if x is not None and len(x) and np.any(x != np.floor(x)):
        verdict = fractional_verdict(_grid_of(x))
        if verdict.settles:
            return Verdict(True, "regression", verdict.evidence)
        return Verdict(False, None, verdict.evidence,
                       candidates=("regression", "ordinal", "multiclass"))
    lo, hi = _fmt_value(float(np.min(x))), _fmt_value(float(np.max(x)))
    return Verdict(False, None,
                   f"`{k:,}` whole-number values from {lo} to {hi}: a measurement, a count and an "
                   f"ordinal score (a summed scale, a 0–27 PHQ-9) all look like this",
                   candidates=("regression", "ordinal") + (("multiclass",) if k <= CODE_LEVELS
                                                           else ()))


def nutrients_by_values(frame: Any, *, energy: str | None = None, energy_unit: str | None = None,
                        skip: Iterable[str] = (), proposed_unit: str | None = None
                        ) -> dict[str, Verdict]:
    """A nutrient intake, for every column of ``frame`` the name reads as a nutrient: the one test
    the roles, the energy card and the energy finding read (``recognizers.corroborated_nutrients``):

    * amounts plausible as a day's intake of the nutrient the name proposes, rising with total
      energy at r ≥ 0.3; or a macronutrient total in a passing Atwater reconstruction of total
      energy (r ≥ 0.7, ratio 0.90–1.10) carrying at least 5% of it on the median row;
    * of two columns read as one nutrient, only the one whose energy totals with ``energy`` by a
      clear margin; neither when nothing chooses.

    A lab count named like a nutrient (``ALC``, lymphocytes), a body measure (BIA ``Fat%``) or a
    yes/no does not rise with energy, is no day's intake, or is too small a member to ride with the
    identity. Each verdict's ``detail`` is the column's intake check, for the words."""
    from turbotab.core.recognizers import corroborated_nutrients

    checks = corroborated_nutrients(frame, energy=energy, energy_unit=energy_unit, skip=skip,
                                    proposed_unit=proposed_unit)
    return {str(c): Verdict(bool(chk.by_values), "exposure" if chk.by_values else None, chk.why,
                            detail=chk) for c, chk in checks.items()}


def intake_rises_with_energy(name: Any, values: Any, energy: Any = None, *,
                             energy_unit: str | None = None) -> Verdict:
    """:func:`nutrients_by_values` on one column beside total energy (``energy``: its values, in
    ``energy_unit`` when it is settled)."""
    import pandas as pd

    frame = pd.DataFrame({str(name): pd.Series(values).reset_index(drop=True)})
    total = None
    if energy is not None:
        total = "__total_energy__"
        frame[total] = pd.Series(energy).reset_index(drop=True)
    found = nutrients_by_values(frame, energy=total, energy_unit=energy_unit).get(str(name))
    if found is None:
        return Verdict(False, None, "its name reads as no energy-bearing nutrient")
    return found


KIND_RULES: dict[str, KindRule] = {r.kind: r for r in (
    KindRule("role:identifier", "the column names each row",
             ("a code that repeats: a household's line number, a stratum, an interviewer",
              "a measurement whose values happen to all differ"),
             names_rows,
             (("a code that repeats: a household's line number, a stratum, an interviewer",
               "any value on two rows fails it"),
              ("a measurement whose values happen to all differ",
               "text never is one; whole numbers pass only within a span a measurement would repeat "
               "in; fractional values fail"))),
    KindRule("cluster", "rows sharing the column's value belong together (the unit, or a group of "
                        "units the intervals must keep together)",
             ("each unit's identifier", "a line number within a household (MEPS PID, a roster's "
              "person number)", "a stratum, a sampling unit or an interviewer",
              "a code for a group the intervals need not keep together"),
             settled_by="the grain answer naming the unit, the user's confirmation, or an "
                        "identifier or cluster role the user confirmed"),
    # The fifth gate: an assay's run date changes within units exactly as a visit date does (each
    # visit's sample assayed on its batch's date), so no value test tells them apart; the user
    # does (BLUEPRINT §14.3, amendment). ``dates_order_rows`` stays the guess's evidence.
    KindRule("role:time", "the column is when each row was measured, so it leaves the predictors",
             ("a measurement in a time unit that changes within units (hours slept, days active)",
              "a crossover's treatment, which changes within units",
              "a date constant within units (a birth or randomization date)",
              "an assay's run or batch date, which changes within units as a visit date does"),
             settled_by="the user's confirmation of the role, or the repeats or temporal answer "
                        "naming it as the column that orders the rows",
             helpers=("turbotab.core.readings:dates_order_rows",)),
    KindRule("role:free_text", "the column is free text, so it leaves the predictors",
             ("a category with many labels (country of birth, occupation)",
              "an identifier written as text"),
             settled_by="the user's confirmation of the role"),
    KindRule("role:design", "the column is part of a survey design, so it leaves the predictors",
             ("a measurement with positive values", "a code for groups that is a predictor"),
             settled_by="the user's confirmation of the role, and the survey answer"),
    # The fifth gate: a survey's skip-pattern gate marks its follow-up's blanks exactly as a
    # missingness flag does (NHANES ALQ111 "No" skips ALQ130, drinks a day), yet it is a yes/no
    # characteristic the model needs; the values cannot tell them apart, so the user does.
    KindRule("role:flag", "the column marks another column's blanks, so it leaves the predictors",
             ("a yes/no characteristic", "an imputed copy of a measurement",
              "a survey's skip-pattern gate (NHANES ALQ111, whose 'No' skips ALQ130), a "
              "characteristic that marks its follow-up's blanks exactly as a flag does"),
             settled_by="the user's confirmation of the role (a flag, or a characteristic)",
             helpers=("turbotab.core.recognizers:flag_values",)),
    KindRule("role:exposure", "the column is a nutrient intake",
             ("a lab count named like a nutrient (ALC: lymphocytes)",
              "a body measure named like a nutrient (BIA Fat%)", "a yes/no named like a nutrient"),
             intake_rises_with_energy,
             (("a lab count named like a nutrient (ALC: lymphocytes)",
               "it does not rise with total energy (r < 0.3), and a member of the Atwater identity "
               "must carry at least 5% of energy"),
              ("a body measure named like a nutrient (BIA Fat%)",
               "no day's intake, or no rise with energy"),
              ("a yes/no named like a nutrient", "two values are no intake")),
             table=nutrients_by_values,
             helpers=("turbotab.core.recognizers:corroborated_nutrients",
                      "turbotab.core.recognizers:intake_check",
                      "turbotab.core.recognizers:nutrient_check",
                      "turbotab.core.recognizers:identity_members",
                      "turbotab.core.recognizers:resolve_duplicates")),
    KindRule("role:energy", "the column is total energy intake",
             ("a device's energy expenditure (Fitbit Calories)",
              "an energy requirement computed from body size"),
             energy_follows_macronutrients,
             (("a device's energy expenditure (Fitbit Calories)",
               "it does not follow the macronutrients' energy (r < 0.7)"),
              ("an energy requirement computed from body size",
               "it follows them only loosely (r < 0.7)")),
             helpers=("turbotab.core.recognizers:energy_against_macros",)),
    KindRule("role:covariate", "the column is a person's characteristic in the model (a sex, an "
                               "age, a BMI)",
             ("a column left out of the model (an identifier, a constant)",
              "the time axis of repeated rows (an age at each visit)"),
             characteristic_predictor,
             (("a column left out of the model (an identifier, a constant)",
               "a sex holds 2–3 levels, an age's or a BMI's median sits in a human range"),
              ("the time axis of repeated rows (an age at each visit)",
               "it must not change within most units")),
             helpers=("turbotab.core.stages.rows:characteristic_fits",)),
    KindRule("role:excluded", "the column holds one value, so it explains nothing",
             ("a predictor",), constant_values,
             (("a predictor", "a constant has no variation for any model to use"),)),
    # The sixth gate: a BMI exported from SAS with ten "." for missing, and a CRP with fifteen
    # "<0.20", arrived as text; "labels, not numbers" settled them codes, and the fit entered 175
    # and 300 indicators where one slope belonged. Numbers written as text are an alternative to
    # labels, so text settles codes only where its values are not mostly numbers once the missing
    # and censoring marks are set aside (:func:`text_numbers`).
    KindRule("code_or_count", "the column's numbers are codes for categories, or amounts",
             ("codes for categories (one indicator per level; a unit's rows take the most "
              "frequent)", "amounts (one slope; a unit's rows averaged)",
              "codes written with a decimal point (ICD-9-CM 307.1 anorexia nervosa, 307.51 "
              "bulimia nervosa, 250.02, 401.9)",
              "numbers written as text, a few values missing marks (SAS and Stata '.', '.A'–'.Z', "
              "'NA', blank) or censored at a limit ('<0.20', '>200', '<LOD')"),
             amounts_by_values,
             (("codes for categories (one indicator per level; a unit's rows take the most "
               "frequent)", "whole numbers, of any count or type, settle nothing"),
              ("amounts (one slope; a unit's rows averaged)",
               "labels settle codes; whole numbers, of any count or type, settle nothing"),
              ("codes written with a decimal point (ICD-9-CM 307.1 anorexia nervosa, 307.51 "
               "bulimia nervosa, 250.02, 401.9)",
               "values with decimals settle amounts only beyond ten values and filling their grid "
               "as a measurement's do; a code list's sit apart"),
              ("numbers written as text, a few values missing marks (SAS and Stata '.', '.A'–'.Z', "
               "'NA', blank) or censored at a limit ('<0.20', '>200', '<LOD')",
               "text settles codes only where numbers are under 80% of its values once the missing "
               "and censoring marks are set aside (the repair registry's own threshold); numbers "
               "written with leading zeros (FIPS '01') read as codes")),
             helpers=("turbotab.core.readings:grid_reading", "turbotab.core.readings:fractional_verdict",
                      "turbotab.core.readings:text_numbers",
                      "turbotab.core.repairs:parse_text_numbers")),
    KindRule("outcome_unit", "the unit the outcome's values are in",
             ("another unit the header's letters spell (`uL`: per microlitre, not U/L)",
              "a unit with a denominator the header leaves out (`IU` for IU/L)",
              "a multiplier (x10^3)"),
             settled_by="the user's recorded unit (until then a sentence quotes the header)"),
    KindRule("unit:height", "the unit a height's values are in (cm or m)",
             ("inches", "metres, against centimetres"),
             height_in_band,
             (("inches", "no one is 100–230 or 1.0–2.3 inches tall: an inch median falls in no band"),
              ("metres, against centimetres", "the two bands do not overlap")),
             settled_by="the user's recorded unit when the median is in neither band (a child's "
                        "height, inches)"),
    KindRule("unit:weight", "the unit a body weight's values are in",
             ("kg", "lb (a heavy cohort's kg is a light one's lb)"),
             settled_by="the user's recorded unit (lb is converted exactly)"),
    KindRule("unit:age", "the unit an age's values are in",
             ("years", "months (a child's age in months)"),
             settled_by="the user's recorded unit"),
    KindRule("unit:energy", "the unit total energy's values are in (kcal or kJ)",
             ("kJ, against kcal", "macronutrients in another unit than grams",
              "a 4-day kcal total beside daily-mean macronutrients (ratio 4.00, next to kJ's 4.184)",
              "an N-day total beside daily-mean macronutrients (a ratio near a whole number N ≥ 2)"),
             atwater_unit,
             (("kJ, against kcal", "only a ratio near 1 settles, and only kcal; kJ's 4.184 is asked"),
              ("macronutrients in another unit than grams",
               "the identity fails, so it settles nothing"),
              ("a 4-day kcal total beside daily-mean macronutrients (ratio 4.00, next to kJ's "
               "4.184)", "a ratio that fits more than one unit and day count is asked, both offered"),
              ("an N-day total beside daily-mean macronutrients (a ratio near a whole number N ≥ 2)",
               "only a ratio near 1 settles")),
             settled_by="the user's recorded unit and days in one answer (a name's kcal or kJ only "
                        "proposes it)",
             helpers=("turbotab.core.methods.energy:atwater_check",)),
    # The sixth gate: alcohol recorded in US standard drinks reconstructed total energy at 1.02
    # read as grams (about 4% of energy: the band cannot see it), and a private settler beside the
    # registry settled even an unmarked `alcohol` the identity had left out; a 20 kcal swap moved
    # it at 7 kcal a drink, not 98.
    KindRule("unit:factor", "the unit an energy source's amounts are in, which sets the kcal each "
                            "unit carries (g: its Atwater factor; kg: 1,000 × it; kcal: 1; kJ: "
                            "1/4.184; alcohol's standard drinks: 7 × the drink's grams)",
             ("kilograms (an InBody export's body protein)", "kcal", "kJ", "percent of energy",
              "standard drinks of alcohol (8–20 g a drink, by country)",
              "a minor source the identity cannot see (under about 9% of energy: any unit, or none, "
              "keeps the ratio in its band)"),
             factor_in_grams,
             (("kilograms (an InBody export's body protein)",
               "read in kilograms the reconstruction leaves the band: rejected"),
              ("kcal", "read in kcal the reconstruction leaves the band: rejected"),
              ("kJ", "read in kJ the reconstruction leaves the band: rejected"),
              ("percent of energy", "percentages are no grams: the pack reads them apart"),
              ("standard drinks of alcohol (8–20 g a drink, by country)",
               "alcohol is never settled by its values"),
              ("a minor source the identity cannot see (under about 9% of energy: any unit, or none, "
               "keeps the ratio in its band)",
               "a source under 1 − 1/1.10 of the reconstructed energy is never settled by its "
               "values")),
             settled_by="the user's recorded unit (a name's _g or _kcal never settles it)",
             helpers=("turbotab.core.methods.energy:atwater_check",)),
    KindRule("sex_coding", "which level of a sex column is female",
             ("1 male, 2 female (NHANES, the CDC growth charts)", "1 female, 2 male",
              "0/1 either way"),
             labels_spell_sex,
             (("1 male, 2 female (NHANES, the CDC growth charts)", "numeric codes never settle it"),
              ("1 female, 2 male", "numeric codes never settle it"),
              ("0/1 either way", "numeric codes never settle it")),
             settled_by="the user's confirmation (text labels settle it by their words)"),
    KindRule("day_count", "how many days each total-energy value spans",
             ("a total over two or more days", "a mean over several days (one day's intake)",
              "a child's total over several days, inside an adult's one-day band"),
             settled_by="the user's recorded day count"),
    KindRule("time_column", "the column that orders a unit's records",
             ("an index of the records in file order", "another column that orders them"),
             settled_by="the user naming it (the repeats answer, or its own confirmation)"),
    # The sixth gate: a confirmation naming a total other than the one the design found was
    # accepted and read nowhere. A part never exceeds its total, which the values check: necessary,
    # not sufficient, so the nesting is the user's, whichever column they name.
    KindRule("nested_in", "the total the column is a part of (saturated fat of total fat)",
             ("a part of another total than the names suggest (sugars of carbohydrate)",
              "no part of any total: a nutrient of its own"),
             settled_by="the user's confirmation, naming any column of the table or none "
                        "(the names and the values, a part never above its total, only propose)",
             helpers=("turbotab.core.methods.nesting:nested_components",)),
    # How a value below a detection limit (``<0.20``) is read once its column is read as numbers:
    # half the limit, or the limit over √2; the values hold no answer (each is somewhere below).
    KindRule("detection_limit", "the value a result below a detection limit is read at",
             ("half the limit", "the limit over √2"),
             settled_by="the user's below-detection repair for the column (``apply_repair``)"),
    # The fifth gate: a PHQ-9 total written 0.0–27.0 after one blank was read "regression (high)"
    # and the task question skipped, while the same values as integers were asked; the dtype
    # decided. The app offers an ordinal task, so the alternative changes the model.
    KindRule("task", "what kind of outcome the column is (regression, binary, multiclass, ordinal)",
             ("an ordinal score (a summed scale, a PHQ-9 total 0–27)",
              "codes for unordered classes", "a count",
              "whole numbers written with a decimal point after a blank (0.0–27.0)"),
             outcome_task_by_values,
             (("an ordinal score (a summed scale, a PHQ-9 total 0–27)",
               "whole numbers, of any type, settle nothing"),
              ("codes for unordered classes", "three or more values or labels settle nothing"),
              ("a count", "whole numbers settle nothing"),
              ("whole numbers written with a decimal point after a blank (0.0–27.0)",
               "regression needs values with decimals filling their grid, more than ten of them"))),
    # MS7 repair: QC-RLSC fits each feature's drift along the injection order, one curve per
    # analytical batch. Both are guesses from names (or a permutation of the rows), so the QC-RLSC
    # options name them, offer the other readings beside them, and the user's choice settles them.
    KindRule("injection_order", "the column that orders an assay run's injections, along which "
                                "QC-RLSC fits each feature's drift",
             ("a sample or participant number, which is also all-distinct",
              "the order samples were prepared or stored in, not injected",
              "a time that is not the acquisition time"),
             settled_by="the user's choice of the QC-RLSC option, which names it"),
    # MI repair (ruling 12): clustered multiple imputation imputes a time-invariant column once per
    # unit and carries a unit's recorded value to its blank rows. A covariate recorded only at
    # baseline (education) agrees within every unit because each unit records it once, exactly as
    # a characteristic that changes between visits but was asked once would; and few recorded rows
    # agree by chance. No value test excludes either, so the user settles it. A column whose
    # recorded values differ within a unit is out of the question's scope (no one value per unit
    # holds its records; it is imputed row by row), unless the user confirms it anyway.
    KindRule("time_invariant", "the column holds one value for each unit of clustered rows, so "
                               "the clustered imputation carries a unit's recorded value to its "
                               "blank rows and imputes it once per unit where no row records it",
             ("a characteristic that can change between a unit's rows but was recorded once (an "
              "education or an income asked only at baseline)",
              "a measure whose few recorded values per unit agree by chance"),
             settled_by="the user's confirmation, one column at a time or listed in one block; "
                        "a column whose recorded values differ within a unit is imputed row by "
                        "row without asking, since no one value per unit holds its records"),
    KindRule("batch", "the column naming each injection's analytical batch, within which QC-RLSC "
                      "fits one curve",
             ("a sample-preparation plate or box that does not split the analytical run",
              "a study group run in blocks of injections",
              "no batch: one run, one curve (a batch column named but not a batch)",
              "a batch column the name does not say (`run`, `sequence`, a site code)"),
             settled_by="the user's choice among the QC-RLSC options, each naming its batch column "
                        "or none"),
)}


# ── exits: one confirmation per reading ───────────────────────────────────────


def confirm_exit(kind: str, column: str, value: Any, label: str | None = None) -> dict[str, Any]:
    """The exit that confirms one reading: a ``confirm_reading`` decision in its JSON form, so it
    travels in a refusal and in a stage's artifact alike."""
    from turbotab.core.decisions import ConfirmReading

    decision = ConfirmReading(reading=kind, column=str(column), value=str(value))
    return {"label": label or f"Confirm `{column}`: {value}",
            "decision": decision.model_dump(mode="json")}


# ── the role readings (BLUEPRINT §14 rule 2) ──────────────────────────────────

_ROLE_WORDS = {
    "exposure": "an exposure", "covariate": "a covariate", "energy": "total energy intake",
    "identifier": "the unit's identifier", "cluster": "a cluster of units",
    "design": "part of the survey design", "time": "the time of each row",
    "flag": "a flag on another column", "excluded": "left out of the models",
}


def role_words(role: str | None) -> str:
    return _ROLE_WORDS.get(str(role), str(role))


def proposals_of(artifact: Any) -> list[dict[str, Any]]:
    """The roles stage's proposals from its artifact (a Bundle, a dict, or None)."""
    data = getattr(artifact, "data", artifact)
    if not isinstance(data, Mapping):
        return []
    return [dict(p) for p in data.get("columns") or [] if isinstance(p, Mapping)]


def attention_columns(proposals: Iterable[Mapping[str, Any]]) -> list[str]:
    """The proposals that need their own confirmation: every one below high."""
    return [str(p["column"]) for p in proposals if p.get("confidence") in ATTENTION]


def rode_along(roles: Mapping[str, str], proposals: Iterable[Mapping[str, Any]]) -> list[str]:
    """The attention proposals a ``set_roles`` records exactly as proposed: what a bulk confirm
    carried without the user's own look. A role the user changed is the user's answer."""
    out = []
    for p in proposals:
        column = str(p.get("column"))
        if p.get("confidence") in ATTENTION and column in roles \
                and roles[column] == p.get("proposed"):
            out.append(column)
    return out


def roles_completion(state: Any, artifact: Any) -> dict[str, Any] | None:
    """The roles answer TurboTab records itself once every predictor's role is settled through the
    person's own confirmations in Your data (``confirm_role``, ``confirm_reading`` or
    ``confirm_readings``; crosswalk disagreement 1), so its line, "Column roles recorded as you
    confirmed them in Your data", is true of every column. A reading the values made high is a
    Confirm line until it is confirmed, never recorded unseen. None while any column waits: the
    Router then holds at the roles and Who's in waits for that column. The person's own roles
    answer is never replaced: this is only for the step the Router reaches with no roles
    recorded."""
    if getattr(state, "roles", None):
        return None
    proposals = proposals_of(artifact)
    if not proposals:
        return None
    target = getattr(state, "target", None)
    roles: dict[str, str] = {}
    for p in proposals:
        column = str(p.get("column"))
        if column == target:
            continue
        own = confirmation(state, "role", column)
        if own is None:
            return None
        roles[column] = str(own)
    return {"kind": "set_roles", "roles": roles} if roles else None


def role_reading(state: Any, column: str) -> Reading | None:
    """``column``'s role as the ledger holds it: settled when the answer recorded it as the user's
    own (changed from the proposal, or proposed high from its values) or a confirmation since
    recorded the same role; proposed while it rode along unconfirmed. None with no role."""
    roles = getattr(state, "roles", None) or {}
    if column not in roles:
        return None
    role = roles[column]
    waiting = column in set(getattr(state, "roles_unconfirmed", None) or [])
    if not waiting or confirmation(state, "role", column) == role or _answered_role(state, column, role):
        return Reading((column,), "role", role, "high", "recorded", True, "confirmed")
    return Reading((column,), "role", role, "medium",
                   "proposed below high confidence and recorded with the other roles", False,
                   "proposed")


def _answered_role(state: Any, column: str, role: str) -> bool:
    """A role another of the user's own answers states: the grain answer names the column as the
    unit (its identifier), or the repeats or temporal answer (or a confirmation of its own) names it
    as the column that orders a unit's rows (its time)."""
    if role == "identifier":
        grain = _get(state, "grain")
        return grain is not None and _get(grain, "id_column") == column
    if role == "time":
        return confirmation(state, "time_column", column) == "orders"
    return False


def unsettled(state: Any, columns: Iterable[str] | None = None) -> list[str]:
    """The recorded roles a number-changing default may not read yet: carried by a bulk confirm
    below high, and not confirmed one by one since (for the same role)."""
    roles = getattr(state, "roles", None) or {}
    names = list(roles) if columns is None else [str(c) for c in columns]
    out = []
    for c in names:
        r = role_reading(state, c) if c in roles else None
        if r is not None and not r.settled:
            out.append(c)
    return out


unsettled_roles = unsettled


def is_settled(state: Any, column: str | None) -> bool:
    return column is None or not unsettled(state, [column])


def settled_roles(state: Any) -> dict[str, str]:
    """The recorded roles a number-changing consumer may read: every settled one."""
    roles = getattr(state, "roles", None) or {}
    waiting = set(unsettled(state))
    return {c: r for c, r in roles.items() if c not in waiting}


def settled_columns(state: Any, artifact: Any = None) -> set[str] | None:
    """The columns whose role a number-changing default may read: the recorded roles less the
    unsettled ones; before any roles are recorded, the proposals the values made high. None when
    neither is known (every caller then reads the roles as given)."""
    roles = getattr(state, "roles", None) or {}
    if roles:
        waiting = set(unsettled(state))
        return {c for c in roles if c not in waiting}
    proposals = proposals_of(artifact)
    if not proposals:
        return None
    return {str(p["column"]) for p in proposals if p.get("confidence") == "high"}


def predictor_columns(state: Any, order: Sequence[str] | None = None,
                      drop: Iterable[str] = ()) -> list[str]:
    """The columns whose settled role puts them in the model (exposure, covariate or energy), in
    table order. A role that rode along unconfirmed is in no fit (the fit asks first:
    :func:`predictors_or_ask`)."""
    roles = settled_roles(state)
    gone = set(drop)
    names = [c for c, r in roles.items() if r in PREDICTOR_ROLES and c not in gone]
    if order is None:
        return names
    rank = {c: i for i, c in enumerate(order)}
    return sorted(names, key=lambda c: rank.get(c, len(rank)))


# The roles whose plausible alternative the values cannot exclude (BLUEPRINT §14.3, amendment
# after the fifth gate), offered beside the guess: a flag may be a skip-pattern gate, a
# characteristic in its own right; a date that changes within units may be an assay's run date.
ROLE_ALTERNATIVES: dict[str, tuple[tuple[str, str], ...]] = {
    "flag": (("covariate", "a characteristic in its own right, kept in the model (a survey's "
                           "skip-pattern gate, such as NHANES ALQ111)"),),
    "time": (("covariate", "a characteristic of each row, kept in the model (an assay's run or "
                           "batch date, not when the row was measured)"),),
}


def confirm_exits(state: Any, columns: Sequence[str]) -> list[dict[str, Any]]:
    """One exit per unsettled role: its own confirmation, never one for all of them, and beside
    it each alternative the values cannot exclude (:data:`ROLE_ALTERNATIVES`)."""
    roles = getattr(state, "roles", None) or {}
    out = []
    for c in columns:
        if c not in roles:
            continue
        out.append(confirm_exit("role", c, roles[c], f"Confirm `{c}` as {role_words(roles.get(c))}"))
        for other, words in ROLE_ALTERNATIVES.get(str(roles[c]), ()):
            out.append(confirm_exit("role", c, other, f"`{c}` is {words}"))
    return out


def unsettled_message(columns: Sequence[str], what: str) -> str:
    listed = listing(columns)
    one = len(columns) == 1
    return (f"{listed} {'was' if one else 'were'} proposed below high confidence and recorded "
            f"with the other roles, not confirmed on {'its' if one else 'their'} own, so "
            f"{'it' if one else 'they'} cannot set {what} yet. Confirm "
            f"{'it' if one else 'each'} after reading why it was proposed, or leave "
            f"{'it' if one else 'them'} out.")


def listing(columns: Sequence[str]) -> str:
    quoted = [f"`{c}`" for c in columns]
    if not quoted:
        return "no column"
    if len(quoted) == 1:
        return quoted[0]
    return f"{', '.join(quoted[:-1])} and {quoted[-1]}"


# ── whole numbers: codes or amounts (the fit's predictors, a unit's combined rows) ──
#
# BLUEPRINT §14.3: the consumer sets the question's scope, not the reader. The values settle the
# reading only where they leave no doubt (:func:`amounts_by_values`: labels are codes, fractional
# values amounts); whole numbers never do, whatever their type or count (FIPS states hold 51 codes;
# UK Biobank's ethnic background 22 from -3 to 4003; NHANES ``DR1_030Z`` 21 eating occasions; a
# blank turns codes 1–5 into 1.0–5.0). So every whole-valued column a number-changing consumer reads
# as codes or as amounts is asked, unless its alternatives give the consumer the same numbers.


def whole_facts(columns: Iterable[str], info: Mapping[str, Any] | None = None,
                store: Any = None) -> dict[str, dict[str, Any]]:
    """What the values say about each numeric column's whole numbers (``whole``, ``zero_one``,
    ``n_values``, ``min``, ``max``): the store's reading (``DataStore.whole_numbers``) where a
    store is at hand; else an integer column's type and distinct count (whether its two values are
    0 and 1 unknown, so it is asked). A float column with no store to read is left out: its values
    are not known here, and the stage that reads them asks."""
    names = [str(c) for c in columns]
    out: dict[str, dict[str, Any]] = {}
    if store is not None:
        try:
            out.update(store.whole_numbers(names))
        except Exception:  # noqa: BLE001 - a store that cannot answer leaves the info's reading
            out = {}
        # Text that holds numbers (BLUEPRINT §14.3, the sixth gate): asked too, as codes or as
        # numbers with missing marks; text the values read as labels is codes by its values.
        try:
            out.update({c: f for c, f in store.text_numbers([c for c in names if c not in out]).items()
                        if f.get("text_numbers")})
        except Exception:  # noqa: BLE001 - no text reading: the stage that reads the values asks
            pass
    for c in names:
        if c in out:
            continue
        entry = (info or {}).get(c) or {}
        get = entry.get if isinstance(entry, Mapping) else (lambda k, e=entry: getattr(e, k, None))
        if get("whole") is not None:
            out[c] = {"whole": bool(get("whole")), "zero_one": bool(get("zero_one")),
                      "n_values": int(get("n_values") or get("n_unique") or 0),
                      "min": get("min"), "max": get("max")}
        elif get("dtype") == "integer":
            out[c] = {"whole": True, "zero_one": None, "n_values": int(get("n_unique") or 0),
                      "min": None, "max": None}
    return out


def code_question(facts: Mapping[str, Any] | None, *, scope: str = "fit") -> bool:
    """Whether a consumer needs the column's code-or-amount reading settled.

    * Every whole-valued numeric column with two or more values.
    * Every column with decimals whose values do not settle "amount" (:func:`amounts_by_values`:
      ten values or fewer, or values that sit apart on their grid, as ICD-9-CM 307.1 and 250.02
      do).
    * Under the fit and its imputation model, a column of exactly two values is exempt: it is one
      indicator either way (a code's one indicator is the amount moved to start at zero, so the
      slope, its interval and every prediction are the same, the intercept's reference aside), and
      both imputation models fill it as a yes/no (``methods.imputation.column_kind``; the single
      fill takes its most frequent value).
    * Combining a unit's rows takes a two-valued column's mean (a share of the records) or its most
      frequent value, which differ, so there it asks."""
    if not facts:
        return False
    if facts.get("text_numbers"):
        # Numbers written as text: codes (one indicator per spelling, the marks among them) or
        # numbers with missing marks (one slope, the marks blank) differ whatever their count.
        return True
    if not facts.get("whole") and facts.get("amount_by_values") is not False:
        return False  # decimals that settle "amount" (or no reading of them at all)
    k = int(facts.get("n_values") or 0)
    if k < 2:
        return False
    return not (scope == "fit" and k == 2)


def code_guess(facts: Mapping[str, Any] | None) -> tuple[str, str]:
    """The best guess a code-or-amount question leads with, and its evidence: codes when the
    values are few (at most :data:`CODE_LEVELS`), sit far apart for their count (UK Biobank's
    1001–4003), include negative sentinels, or carry decimals that did not settle amounts (ICD-9-CM
    307.1); else amounts. A guess the user confirms or changes, never a settlement."""
    f = dict(facts or {})
    if f.get("text_numbers"):
        # BLUEPRINT §14.3 (the sixth gate): numbers written as text are led by "numbers with
        # missing marks", unless they are a few whole values (codes 1/2/3 with a SAS ".").
        evidence = str(f.get("evidence") or "numbers written as text")
        few = bool(f.get("whole")) and int(f.get("distinct") or 0) <= CODE_LEVELS
        return ("code", f"{evidence}; few whole values, as codes") if few else ("amount", evidence)
    if f.get("whole") is False:
        return "code", str(f.get("evidence") or "values with decimals that sit apart, as codes do")
    k = int(f.get("n_values") or 0)
    lo, hi = f.get("min"), f.get("max")
    shown = (f"from {_fmt_value(float(lo))} to {_fmt_value(float(hi))}"
             if lo is not None and hi is not None else "")
    evidence = f"`{k:,}` whole-number values {shown}".strip()
    if f.get("zero_one"):
        return "amount", f"{evidence}: a yes/no, whose mean is a share"
    # Far apart for their count, or negative beside positive, among a code list's few dozen values
    # (UK Biobank's 1001–4003, sentinels -1 and -3); a measurement's hundreds of whole values sit
    # far apart too, and are guessed amounts.
    few = k <= GUESS_CODES_UP_TO
    sparse = few and lo is not None and hi is not None and (float(hi) - float(lo) + 1) / k >= 10
    negative = few and lo is not None and float(lo) < 0 < float(hi or 0)
    if k <= CODE_LEVELS or sparse or negative:
        why = ("few of them" if k <= CODE_LEVELS else "far apart for their count" if sparse
               else "negative beside positive")
        return "code", f"{evidence}, {why}"
    return "amount", evidence


GUESS_CODES_UP_TO = 60  # a code list's size the guess considers (codes, never a settlement)


def code_or_count_reading(state: Any, column: str, facts: Mapping[str, Any] | None = None, *,
                          scope: str = "fit", dtype: str | None = None,
                          n_unique: int | None = None) -> Reading | None:
    """Whether ``column``'s numbers are codes for categories or amounts, as the ledger holds it for
    a consumer (``scope``: ``fit`` or ``combine``); None when that consumer needs no answer
    (:func:`code_question`). Never settled by the values: whole numbers fit both. ``dtype`` and
    ``n_unique`` stand in for ``facts`` where only a column's summary is known."""
    if facts is None and dtype is not None:
        facts = whole_facts([column], {column: {"dtype": dtype, "n_unique": n_unique}}).get(column)
    if not code_question(facts, scope=scope):
        return None
    guess, evidence = code_guess(facts)
    return reading("code_or_count", column, guess, confidence="medium", evidence=evidence,
                   corroborated=False, state=state)


def code_or_count_exits(column: str) -> list[dict[str, Any]]:
    return [confirm_exit("code_or_count", column, "amount",
                         f"`{column}` is a count or an amount (a slope; a unit's rows averaged)"),
            confirm_exit("code_or_count", column, "code",
                         f"`{column}` holds codes for categories (one indicator per level; a "
                         f"unit's rows take their most frequent value)")]


ASSAY_LENSES = ("metabolomics", "genomics")


def unsettled_codes(state: Any, columns: Iterable[str],
                    facts: Mapping[str, Mapping[str, Any]] | None, *,
                    scope: str = "fit") -> list[Reading]:
    """The code-or-amount readings the fit (or combining) may not read yet, as readings with their
    best guesses. Under an assay lens (the user's answer that the table is an assay), an exposure
    is a measured feature (a count or an intensity), an amount by that answer; its other columns
    are read as any table's."""
    assay = any(k in ASSAY_LENSES for k in (_get(state, "lens") or []))
    roles = settled_roles(state) if assay else {}
    out = []
    for c in columns:
        if assay and roles.get(c) == "exposure":
            continue
        r = code_or_count_reading(state, c, (facts or {}).get(c), scope=scope)
        if r is not None and not r.settled:
            out.append(r)
    return out


# The energy methods that compute with total energy and the nutrients they adjust (a residual, a
# density, a partition's kcal): those columns are amounts by the energy answer (BLUEPRINT §13,
# "implies": the app states it rather than asks), and none of them reaches the model as its raw
# column but total energy kept as a covariate (``residual``, ``density_multivariate``).
COMPUTING_METHODS = ("residual", "residual_energy_dropped", "density", "density_multivariate",
                     "partition", "all_components")
# The energy models that split total energy into kcal from each named source: each source's kcal
# per unit is read from its settled unit (:func:`energy_sources_or_ask`).
PARTITION_METHODS = ("partition", "all_components")


def energy_plan(state: Any, predictors: Iterable[str]) -> tuple[set[str], set[str]]:
    """What the recorded energy answer does with the predictors, ``(left_out, amounts)``: under
    "none" every total-energy column leaves the model (``models.steps.energy_step``), so no reading
    of it is asked for the fit; under a method that computes with total energy and the adjusted
    nutrients, those are amounts by that answer (a code has no residual, density or kcal), so
    their code-or-amount reading is not asked either (BLUEPRINT §14.2: ask only where it matters)."""
    adj = _get(state, "energy_adjustment")
    names = set(predictors)
    if adj is None:
        return set(), set()
    method = str(_get(adj, "method"))
    energy = _get(adj, "energy_column")
    # The recorded roles: a column whose energy role waits for its confirmation is asked that
    # first, and leaves with total energy once it is confirmed (the fit asks again otherwise).
    roles = getattr(state, "roles", None) or {}
    if method == "none":
        totals = {c for c in (energy, *(c for c, r in roles.items() if r == "energy")) if c}
        return totals & names, set()
    if method in COMPUTING_METHODS:
        return set(), {c for c in (energy, *(_get(adj, "nutrients") or ())) if c} & names
    return set(), set()


def scale_items(state: Any) -> set[str]:
    """The items of the declared scales (``set_scales``; MS8): answers on each scale's response
    scale by that answer, scored into one column, never fit as their own values."""
    return {str(c) for s in (_get(state, "scales") or []) for c in (_get(s, "items") or [])}


def predictors_or_ask(state: Any, info: Mapping[str, Any] | None = None,
                      order: Sequence[str] | None = None, drop: Iterable[str] = (), *,
                      store: Any = None) -> list[str]:
    """The fit's predictor set, once every reading it rests on is settled; else :class:`Unsettled`
    with the ask (:func:`ask_exits`): every recorded role that rode along (it decides whether its
    column is in the model or out), and each predictor's code-or-amount reading the fit uses
    (:func:`code_question`: one indicator per level or one slope), the roles first. A column whose
    role is waiting is asked its code-or-amount reading too when its proposed role is a predictor,
    so one answer settles both. A column the plan does not fit as its own values is not asked
    (:func:`energy_plan`: total energy the energy answer "none" leaves out; the columns an energy
    method computes with, amounts by that answer); one the user confirmed as codes there is
    refused, since that answer would be ignored."""
    gone = set(drop)
    waiting = unsettled(state)
    roles = getattr(state, "roles", None) or {}
    role_readings = [role_reading(state, c) for c in waiting]
    preds = predictor_columns(state, order, drop)
    pending = [c for c in waiting if roles.get(c) in PREDICTOR_ROLES and c not in gone]
    left, amounts = energy_plan(state, [*preds, *pending])
    clash = [c for c in sorted(amounts) if confirmation(state, "code_or_count", c) == "code"]
    if clash and not role_readings:
        method = _get(_get(state, "energy_adjustment"), "method")
        raise Unsettled(
            f"{listing(clash)} {'is' if len(clash) == 1 else 'are'} recorded as codes for "
            f"categories, but the energy answer ({method}) computes with "
            f"{'it' if len(clash) == 1 else 'them'} as amounts. Say which holds: the codes answer "
            f"or the energy answer.",
            [reading("code_or_count", c, "code", state=state) for c in clash],
            [*(confirm_exit("code_or_count", c, "amount",
                            f"`{c}` is an amount, as the energy answer reads it") for c in clash),
             {"label": "Change the energy answer (the energy question)", "decision": None}])
    # MS8: a declared scale's items are answers on its response scale by the scale's answer, summed
    # into its score, so their code-or-amount reading is not asked; one recorded as codes is refused
    # here, since the score would ignore that answer.
    scored = scale_items(state) & {*preds, *pending}
    coded = [c for c in sorted(scored) if confirmation(state, "code_or_count", c) == "code"]
    if coded and not role_readings:
        raise Unsettled(
            f"{listing(coded)} {'is' if len(coded) == 1 else 'are'} recorded as codes for "
            f"categories, but the scales answer sums {'it' if len(coded) == 1 else 'them'} as "
            f"answers into a score. Say which holds: the codes answer or the scales answer.",
            [reading("code_or_count", c, "code", state=state) for c in coded],
            [*(confirm_exit("code_or_count", c, "amount",
                            f"`{c}` holds answers, as the scales answer reads it") for c in coded),
             {"label": "Change the scales answer", "decision": None}])
    asked_for = [c for c in [*preds, *pending]
                 if c not in left and c not in amounts and c not in scored]
    if any(k in ASSAY_LENSES for k in (_get(state, "lens") or [])):
        # Under an assay lens an exposure is a measured feature, an amount by that answer
        # (:func:`unsettled_codes`); its values are not read for the question (20,000 genes).
        exposures = {c for c, r in settled_roles(state).items() if r == "exposure"}
        asked_for = [c for c in asked_for if c not in exposures]
    facts = whole_facts(asked_for, info, store)
    codes = unsettled_codes(state, asked_for, facts)
    if not role_readings and not codes:
        # Numbers written as text read as amounts once the user said so (the working table reads
        # them: marks blank), but a value below a detection limit has no number until the
        # detection-limit answer gives it one (BLUEPRINT §14.3: censored marks are routed there).
        commas = [c for c in asked_for if (facts.get(c) or {}).get("text_numbers")
                  and (facts.get(c) or {}).get("ambiguous_comma")
                  and confirmation(state, "code_or_count", c) == "amount"]
        if commas:
            c = commas[0]
            raise Unsettled(
                f"`{c}` is read as numbers, as you said, but every comma in it could split "
                f"thousands (`1,234`) or mark decimals (`1,234` as 1.234); the values cannot say "
                f"which. Choose how its commas read.",
                [Reading((c,), "code_or_count", "amount", "medium",
                         (facts.get(c) or {}).get("evidence") or "", False, "proposed")],
                [e for c in commas for e in comma_exits(c)])
        limits = [c for c in asked_for if (facts.get(c) or {}).get("text_numbers")
                  and (facts.get(c) or {}).get("below")
                  and confirmation(state, "code_or_count", c) == "amount"
                  and detection_limit(state, c) is None]
        if limits:
            c = limits[0]
            f = facts[c]
            first = next(iter(f["below"]))
            raise Unsettled(
                f"`{c}` is read as numbers, as you said, but `{sum(f['below'].values()):,}` of its "
                f"values lie below a detection limit (such as `<{first}`): each is somewhere below "
                f"its limit, so its number is your answer. Choose how they are read.",
                [Reading((c,), "detection_limit", "half_limit", "medium", f.get("evidence") or "",
                         False, "proposed")],
                [e for c in limits for e in detection_exits(c, facts[c])])
    if role_readings or codes:
        needed = [r for r in role_readings if r is not None] + codes
        if role_readings:
            message = (unsettled_message(waiting, "the model's predictors") + " The fit waits for "
                       "them: a column whose role nobody confirmed would enter or leave the model "
                       "on a guess.")
        else:
            names = [r.column for r in codes]
            text = [c for c in names if (facts.get(c) or {}).get("text_numbers")]
            what = ("numbers written as text" if len(text) == len(names) else
                    "whole numbers" if not text else "whole numbers or numbers written as text")
            message = (f"{listing(names)} {'holds' if len(names) == 1 else 'hold'} {what}, "
                       f"which may be codes for categories (one indicator per level) or amounts "
                       f"(one slope); the fit waits for the answer for each.")
        raise Unsettled(f"{message} {ask_text(needed, state)}", labeled(needed, state),
                        ask_exits(needed, state))
    # The energy models that convert each source to kcal (the partition, the all-components model)
    # read each source's kcal per unit from its settled unit, never from its name.
    adj = _get(state, "energy_adjustment")
    if adj is not None and str(_get(adj, "method")) in PARTITION_METHODS:
        energy_sources_or_ask(state, [c for c in (_get(adj, "nutrients") or ()) if c in preds],
                              store=store)
    return preds


# ── readings a stage makes once and states with its confidence ───────────────


def stated_reading(kind: str, column: str, value: Any, confidence: Any,
                   evidence: str = "") -> Reading:
    """A reading a stage made and stated with its confidence (the orientation of the table, the
    outcome's task, total energy by its values): settled only when the reader made it high from
    the values, never from a name or a shape's grammar alone."""
    level = str(confidence or "low")
    return Reading((str(column),), kind, value, level, evidence, level == "high", "proposed")


def task_reading(column: str, detected: str, confidence: Any, evidence: str = "",
                 values: Any = None) -> Reading:
    """The outcome's task. Settled only by its registry value test (:data:`KIND_RULES`
    ``task``, :func:`outcome_task_by_values`): two values are binary, values with decimals that
    fill their grid a regression outcome; the detection's dtype never settles it (the fifth gate:
    a PHQ-9 total read "regression (high)" because one blank made it float). Anything else is the
    detection's guess at medium at most, and asked among the tasks the app offers, ordinal
    included. Without ``values`` (a caller holding only the detection), three or more labels are
    never settled."""
    level = str(confidence or "low")
    if values is not None:
        verdict = outcome_task_by_values(values)
        if verdict.settles:
            return Reading((str(column),), "task", verdict.value, "high", verdict.evidence, True,
                           "proposed")
        level = "medium" if level == "high" else level
        return Reading((str(column),), "task", detected, level, verdict.evidence or evidence,
                       False, "proposed")
    if detected == "multiclass" and level == "high":
        level = "medium"
    return stated_reading("task", column, detected, level, evidence)


# ── the cluster reading: which rows belong together ──────────────────────────


def cluster_reading(state: Any, column: str) -> Reading:
    """Whether ``column``'s repeating values mark rows the intervals (and the seal, the folds, a
    mixed or GEE model's units) must keep together. No value test settles it (BLUEPRINT §14.3: a
    household's line number, a stratum or an interviewer repeats exactly as a subject's identifier
    in a long table does), so only the user does:

    * the column's own confirmation (``confirm_reading`` cluster yes/no);
    * the grain answer naming it as the unit;
    * a cluster role the user confirmed (a household, a site: a group of units);
    * an identifier role the user confirmed, unless the grain answer names another column as the
      unit: the grain answer always wins over any other reading of which rows belong together
      (the gate: MEPS ``PID`` clustered the intervals over the grain's ``DUPERSID``)."""
    recorded = _confirmations(state).get(key("cluster", column))
    if recorded is not None:
        return Reading((column,), "cluster", recorded, "high", "recorded by the user", True,
                       "confirmed")
    grain = _get(state, "grain")
    named = _get(grain, "id_column") if grain is not None else None
    if named == column:
        return Reading((column,), "cluster", "yes", "high", "the grain answer names it as the unit",
                       True, "confirmed")
    role = role_reading(state, column)
    if role is not None and role.settled and role.value == "cluster":
        return Reading((column,), "cluster", "yes", "high", "its cluster role is the user's", True,
                       "confirmed")
    if role is not None and role.settled and role.value == "identifier":
        if named:
            return Reading((column,), "cluster", "yes", "medium",
                           f"the grain answer names `{named}` as the unit; whether rows sharing "
                           f"`{column}` belong together too is yours to say", False, "proposed")
        return Reading((column,), "cluster", "yes", "high", "its identifier role is the user's",
                       True, "confirmed")
    return Reading((column,), "cluster", "yes", "medium",
                   "its values repeat, as a unit's identifier, a household's line number, a stratum "
                   "or an interviewer's do", False, "proposed")


CLUSTER_PRIORITY = {"grain": 0, "cluster": 1, "identifier": 2}


def cluster_rank(state: Any, column: str) -> int:
    """Which settled grouping wins when several repeat: the grain's unit, then a group of units the
    user named (a cluster role or confirmation), then an identifier the user confirmed."""
    grain = _get(state, "grain")
    if grain is not None and _get(grain, "id_column") == column:
        return CLUSTER_PRIORITY["grain"]
    if _confirmations(state).get(key("cluster", column)) == "yes":
        return CLUSTER_PRIORITY["cluster"]
    role = role_reading(state, column)
    if role is not None and role.value == "cluster":
        return CLUSTER_PRIORITY["cluster"]
    return CLUSTER_PRIORITY["identifier"]


def cluster_exits(state: Any, column: str) -> list[dict[str, Any]]:
    """The ways to settle whether ``column``'s rows belong together, one reading each."""
    return [confirm_exit("cluster", column, "yes",
                         f"Rows sharing a `{column}` belong together: cluster the intervals by it"),
            confirm_exit("cluster", column, "no",
                         f"`{column}` groups nothing the intervals must keep together (a line "
                         f"number within a household, a stratum, an interviewer)"),
            {"label": "Name the column that identifies the unit (the grain question)",
             "decision": None}]


# ── a time column, a repeat kind ─────────────────────────────────────────────


def repeat_kind_reading(state: Any, structure: Mapping[str, Any] | None) -> Reading | None:
    """The repeat kind: the user's answer, or the structure stage's reading as a proposal. A stated
    reading is medium at most (its evidence is spacing, an index or a name), so it never stands in
    for the answer (the gate: recalls numbered across an ``assessment``'s occasions)."""
    spec = _get(state, "repeat_kind")
    if spec is not None:
        return Reading(("__rows__",), "repeat_kind", str(_get(spec, "repeat_kind")), "high",
                       "answered", True, "confirmed")
    found = (structure or {}).get("repeats") or {}
    if found.get("stated") and found.get("reading"):
        return Reading(("__rows__",), "repeat_kind", str(found["reading"]),
                       str(found.get("confidence") or "medium"), str(found.get("sentence") or ""),
                       False, "proposed")
    return None


def time_column_reading(state: Any, structure: Mapping[str, Any] | None) -> Reading | None:
    """The column that orders a unit's rows: named by the user (with the repeat kind or the
    temporal answer, or confirmed on its own), else the structure reading's spacing column or
    index as a proposal."""
    for spec in (_get(state, "repeat_kind"), _get(state, "temporal")):
        column = _get(spec, "time_column") if spec is not None else None
        if column:
            return Reading((str(column),), "time_column", "orders", "high", "named by the user",
                           True, "confirmed")
    for name, value in _confirmations(state).items():
        if name.startswith("time_column:") and value == "orders":
            column = name.split(":", 1)[1]
            return Reading((column,), "time_column", "orders", "high", "confirmed by the user",
                           True, "confirmed")
    found = (structure or {}).get("repeats") or {}
    spacing = found.get("spacing") or {}
    column = spacing.get("column") or found.get("replicate_index")
    if not column:
        return None
    return reading("time_column", str(column), "orders", confidence="medium",
                   evidence=f"the repeats reading orders the rows by `{column}`", state=state)


def confirmed_codes(state: Any) -> list[str]:
    """The columns the user said hold codes: declared categorical, given the mode as their
    combining rule, or confirmed as codes on their own."""
    out = list(_get(state, "categorical") or [])
    agg = _get(state, "aggregation")
    out += [c for c, r in (_get(agg, "columns") or {}).items() if r == "mode"]
    out += [name.split(":", 1)[1] for name, value in _confirmations(state).items()
            if name.startswith("code_or_count:") and value == "code"]
    # Each answer is read as the ledger holds it: a later "amount" stands over an earlier
    # declaration (BLUEPRINT §14.3, every confirmation is honored).
    return [c for c in dict.fromkeys(out) if confirmation(state, "code_or_count", c) == "code"]


# ── units ─────────────────────────────────────────────────────────────────────

# Bare amounts: a mass that says neither per what nor of what. On a quantity the clinical pack
# reads as a concentration or an index (``ldl_mg`` is mg/dL; ``bmi_kg`` is kg/m²), it is part of a
# unit, not one; elsewhere it is still only the name's (the gate: ``hb_g`` stated "in g").
BARE_AMOUNTS = ("mg", "g", "kg", "µg", "lb")


def outcome_unit_reading(column: str, recorded: str | None = None) -> Reading | None:
    """The outcome's unit (audit IN-05; BLUEPRINT §14.3). Settled only when the user recorded it
    (``set_outcome_unit``): a header's letters are a name, and a name never settles a unit (the
    fourth gate: ``WBC (x10^3/uL)`` stated "in U/L" where NHANES gives 1000 cells/uL; ``ALT (IU)``
    stated "in IU" where NHANES gives U/L). The unit the letters spell, else the clinical pack's
    proposal, is the best guess a question leads with; until it is recorded a sentence quotes the
    header verbatim, which reads nothing into it. None when nothing proposes a unit."""
    from turbotab.core.units import from_name, proposed_unit

    if recorded:
        return Reading((column,), "outcome_unit", str(recorded), "high", "recorded by the user",
                       True, "confirmed")
    unit = from_name(column)
    if unit is not None:
        return Reading((column,), "outcome_unit", unit, "medium",
                       f"the header's letters read as {unit}, and a header is never checked "
                       f"against the values", False, "proposed")
    pack = proposed_unit(column)
    if pack and pack.get("unit"):
        return Reading((column,), "outcome_unit", pack["unit"], "low",
                       f"the clinical pack reads this quantity in "
                       f"{' or '.join(pack.get('candidates') or [pack['unit']])}", False, "proposed")
    return None


def stated_outcome_unit(column: str, recorded: str | None = None) -> tuple[str | None, str | None]:
    """``(unit, source)`` a sentence may state: only a settled outcome-unit reading."""
    r = outcome_unit_reading(column, recorded)
    if r is None or not r.settled:
        return None, None
    return str(r.value), ("decision" if r.state == "confirmed" else "name")


# Body measures the Goldberg screen reads: the unit each must be in, and the suffixes and codebook
# names that spell it out (NHANES BMXWT "Weight (kg)", BMXHT "Standing Height (cm)", RIDAGEYR "Age
# in years at screening").
BODY_UNITS: dict[str, tuple[str, ...]] = {"weight": ("kg",), "height": ("cm", "m"),
                                          "age": ("years",)}
_BODY_SPELLED = {
    "weight": {"kg": ("kg", "kgs", "kilograms", "bmxwt")},
    "height": {"cm": ("cm", "bmxht"), "m": ("m", "meters")},
    "age": {"years": ("years", "yrs", "yr", "ridageyr", "ageyr", "ageyears")},
}


# Height's units sit far enough apart that a human median places them (units.py: MIN_FACTOR 3 is
# the floor for telling units by magnitude; cm against inches is 2.54×, but no adult or child is
# 100–230 inches or 100–230 m tall, and 1.0–2.3 is meters only): a median in the band is the
# values' own corroboration. Weight's kg and lb overlap there (a heavy cohort's kg is a light
# one's lb), as an age's years and a child's months do, so their medians only propose.
HEIGHT_BANDS = {"cm": (100.0, 230.0), "m": (1.0, 2.3)}


def body_unit_reading(column: str, measure: str, state: Any = None, values: Any = None) -> Reading:
    """The unit of a body measure the Goldberg screen reads (weight in kg or lb, height in cm or
    m, age in years). Settled when the user recorded it, or (height only) when its median sits in
    one unit's human band (:func:`height_in_band`). A header that spells a unit (``weight_kg``,
    ``BMXWT``) proposes it and settles nothing (BLUEPRINT §14.3: names never corroborate); a
    weight's or an age's median tells kg from lb, or years from a child's months, no better than a
    heavy cohort from a light one (the gate: a US cohort's ``weight`` in pounds, median 154, read as
    kg)."""
    from turbotab.core.recognizers import tokens

    recorded = confirmation(state, "unit", column)
    if recorded is not None:
        return Reading((column,), "unit", str(recorded), "high",
                       recorded_evidence(state, "unit", column), True, "confirmed")
    if measure == "height" and values is not None:
        verdict = height_in_band(values)
        if verdict.settles:
            return Reading((column,), "unit", verdict.value, "high", verdict.evidence, True,
                           "proposed")
    words = tokens(column)
    joined = "".join(words)
    for unit, spelled in _BODY_SPELLED.get(measure, {}).items():
        if (words and words[-1] in spelled) or joined in spelled:
            return Reading((column,), "unit", unit, "medium",
                           f"the header says {unit}, and a header is never checked against the "
                           f"values", False, "proposed")
    default = BODY_UNITS[measure][0]
    return Reading((column,), "unit", default, "medium",
                   f"only its median says {default}", False, "proposed")


# The units a body measure may be recorded in, and their exact factor to the unit the Goldberg
# screen reads (1 lb = 0.45359237 kg exactly: the international yard and pound agreement, 1959;
# 1 in = 2.54 cm exactly).
BODY_CONVERSIONS: dict[str, dict[str, float]] = {
    "weight": {"kg": 1.0, "lb": 0.45359237},
    "height": {"cm": 1.0, "m": 100.0, "in": 2.54},
    "age": {"years": 1.0},
}


# ── energy: the day count ────────────────────────────────────────────────────


def names_several_days(name: Any) -> bool:
    """The name says its value spans or numbers more than one day, in a form the day-count parser
    does not read as a count (``energy_kcal_day1_day2``, ``kcal_sum_d1_d2``, ``kcal_d1d2``,
    ``Energy (kcal) - 2 recalls``, ``kcal_2rec``, ``kcal_2x24h``, ``energy_3dfr``, ``kcal_7dd``,
    ``kcal_both_days``): it is asked, never read as one day's intake."""
    import re

    from turbotab.core.recognizers import day_count, tokens

    if day_count(name) is not None:
        return True
    words = tokens(name)
    days = [w for w in words if re.fullmatch(r"(?:d|day|dy)\d+", w)]
    if len(set(days)) >= 2:
        return True
    for w in words:
        if re.fullmatch(r"d\d+d\d+", w) or re.fullmatch(r"\d+(?:x24h|x24hr|x24|rec|recs|recall|"
                                                      r"recalls|dr|dfr|dd|dwr)", w):
            return True
    if {"recalls", "days", "both"} & set(words):
        return True
    if "sum" in words and days:
        return True
    return False


def day_count_reading(column: str, values: Any = None, *, unit: str = "kcal",
                      state: Any = None) -> Reading:
    """How many days each value of a total-energy column spans. Settled only when recorded
    (``set_column_unit``, or its own confirmation; BLUEPRINT §14.3). The values propose: a median
    inside the field's one-day band for the unit (NUTRITION_PACK §01's prior for adults:
    1,600–2,600 kcal, 7,000–11,000 kJ) under a name that says nothing of several days is the best
    guess, one day; but a young child's total over two days sits in an adult's one-day band too (the
    fourth gate's ``kcal_total``, median 2,227), so no band rejects every alternative. The Atwater
    identity settles kcal against kJ and says nothing about days (the third gate: 2-day totals
    beside 2-day macronutrients passed it and were read as one day)."""
    import numpy as np
    import pandas as pd

    from turbotab.core.recognizers import KCAL_PRIOR, KJ_PRIOR, day_count

    recorded = confirmation(state, "day_count", column)
    if recorded is not None:
        return Reading((column,), "day_count", int(recorded), "high", "recorded by the user",
                       True, "confirmed")
    spanned = day_count(column)
    if spanned is not None:
        return Reading((column,), "day_count", spanned, "medium",
                       f"its name carries `{spanned}` days, a total or a mean", False, "proposed")
    if names_several_days(column):
        return Reading((column,), "day_count", None, "medium",
                       "its name numbers more than one day or recall", False, "proposed")
    median = None
    if values is not None:
        x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x) & (x > 0)]
        median = float(np.median(x)) if len(x) else None
    band = KJ_PRIOR if unit == "kj" else KCAL_PRIOR
    if median is not None and band[0] <= median <= band[1]:
        return Reading((column,), "day_count", 1, "medium",
                       f"its median, {median:,.0f}, is an adult's day's intake (a child's total "
                       f"over two days sits there too)", False, "proposed")
    word = "kJ" if unit == "kj" else "kcal"
    said = (f"its median, {median:,.0f}, is outside a day's {band[0]:,.0f}–{band[1]:,.0f} {word}"
            if median is not None else "its values were not read")
    return Reading((column,), "day_count", 1, "medium", said, False, "proposed")


def day_count_candidates(name: Any, values: Any = None, *, unit: str = "kcal") -> list[int]:
    """The day counts an exit offers for a total-energy column whose days are not settled: one day,
    the count the name numbers (``day1_day2``: 2; ``2 recalls``: 2; ``7dd``: 7), and the count its
    median points to against a day's band (a median of 4,160 kcal: 2)."""
    import re

    import numpy as np
    import pandas as pd

    from turbotab.core.recognizers import KCAL_PRIOR, KJ_PRIOR, day_count, tokens

    out = [1]
    spanned = day_count(name)
    if spanned:
        out.append(int(spanned))
    words = tokens(name)
    days = {w for w in words if re.fullmatch(r"(?:d|day|dy)\d+", w)}
    if len(days) >= 2:
        out.append(len(days))
    for w in words:
        m = re.fullmatch(r"(\d+)(?:x24h|x24hr|x24|rec|recs|recall|recalls|dr|dfr|dd|dwr)", w)
        if m:
            out.append(int(m.group(1)))
        m = re.fullmatch(r"d(\d+)d(\d+)", w)
        if m:
            out.append(2)
    for i, w in enumerate(words[:-1]):
        if w.isdigit() and words[i + 1] in ("recalls", "recall", "records", "days"):
            out.append(int(w))
    if values is not None:
        x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x) & (x > 0)]
        if len(x):
            band = KJ_PRIOR if unit == "kj" else KCAL_PRIOR
            ratio = float(np.median(x)) / ((band[0] + band[1]) / 2)
            if ratio >= 1.5:
                out.append(int(round(ratio)))
    return [d for d in dict.fromkeys(out) if 1 <= d <= 366]


# ── an energy source's kcal per unit ────────────────────────────────────────


@dataclass(frozen=True)
class KcalPerUnit:
    """An energy source's kcal per unit as the ledger holds it: ``factor`` once settled (by the
    user's recorded unit, or by the registry's Atwater test reading its grams), else None with
    ``why``; ``route`` names the way forward when the recorded unit carries no constant kcal per
    unit (a share of energy moves on the share-of-energy scale)."""

    column: str
    factor: float | None
    settled: bool
    why: str
    route: str | None = None
    # The unit the factor is per, as recorded (``g``, ``kg``, ``kcal``, ``kj``, ``drinks:14``) or
    # read by the values (``g``); None while unsettled.
    unit: str | None = None


# The kcal one unit of an energy source carries, by its recorded unit (BLUEPRINT §14.3: derived
# from the confirmation, never from the name): grams at the nutrient's Atwater factor
# (NUTRITION_PACK §01: 4/4/9/7), kilograms 1,000 times that, kcal itself, kJ at 1/4.184 (4.184 kJ
# per thermochemical kcal), and alcohol's standard drinks at 7 kcal per gram of the drink's ethanol
# (:data:`DRINK_GRAMS`). A share of energy has no constant kcal per unit; the other units are no
# amount of food at all.
FACTOR_UNITS = ("g", "kg", "kcal", "kj")
DRINKS = "drinks"
# Standard drinks the substitution's question offers, each with its source: the US (NIAAA: "In the
# United States, one standard drink contains about 14 grams, or about 0.6 fluid ounces, of pure
# alcohol"), the UK unit (NHS: "One unit equals 10ml or 8g of pure alcohol"), and the most common
# governmental size (Kalinowski & Humphreys 2016: "the modal standard drink size was 10 g pure
# ethanol"). Any size from 8 to 20 g is recorded the same way (``drinks:<grams>``).
OFFERED_DRINKS: tuple[tuple[float, str], ...] = (
    (14.0, "a US standard drink, 14 g of alcohol (NIAAA)"),
    (8.0, "a UK unit, 8 g of alcohol (NHS)"),
    (10.0, "a 10 g standard drink, the most common governmental size (Australia, much of Europe)"),
)


def _grams_text(grams: float) -> str:
    return f"{grams:.2f}".rstrip("0").rstrip(".")


def drinks_value(grams: float) -> str:
    """``drinks:<grams>``: a unit of standard drinks of ``grams`` g of ethanol each, written with
    no trailing zeros (``drinks:14``, ``drinks:13.45``)."""
    return f"{DRINKS}:{_grams_text(float(grams))}"


def parse_drinks(value: Any) -> float | None:
    """The grams of ethanol a ``drinks:<grams>`` unit's drink holds, or None when ``value`` is no
    such unit: grams from 8 to 20 (:data:`DRINK_GRAMS`), to the hundredth of a gram, written as
    :func:`drinks_value` writes them."""
    import re

    m = re.fullmatch(r"drinks:(\d{1,2}(?:\.\d{1,2})?)", str(value or ""))
    if not m:
        return None
    grams = float(m.group(1))
    if not DRINK_GRAMS[0] <= grams <= DRINK_GRAMS[1] or drinks_value(grams) != str(value):
        return None
    return grams


def drink_units() -> tuple[str, ...]:
    """Every ``drinks:<grams>`` unit a confirmation may record, in order."""
    lo, hi = (int(round(g * 100)) for g in DRINK_GRAMS)
    return tuple(drinks_value(i / 100) for i in range(lo, hi + 1))


def per_unit_words(value: Any) -> str:
    """One recorded unit, as what an estimate is "per": g, kg, kcal, kJ, standard drink of 14 g."""
    grams = parse_drinks(value)
    if grams is not None:
        return f"standard drink of {_grams_text(grams)} g"
    return {"kj": "kJ", "pct_energy": "% of energy"}.get(str(value), str(value))


def unit_words(value: Any) -> str:
    """A recorded unit in words: kJ, kcal, standard drinks of 14 g."""
    grams = parse_drinks(value)
    if grams is not None:
        return f"standard drinks of {_grams_text(grams)} g"
    return {"kj": "kJ", "pct_energy": "% of energy"}.get(str(value), str(value))


def kcal_per_unit(state: Any, column: str, store: Any = None, *,
                  verdict: Verdict | None = None) -> KcalPerUnit:
    """``column``'s kcal per unit for a substitution (or anything that moves energy through it).

    Settled by the user's recorded unit (:func:`unit_record`, the one store): ``g`` → the
    nutrient's Atwater factor, ``kg`` → 1,000 × it, ``kcal`` → 1, ``kj`` → 1/4.184, alcohol's
    ``drinks:<grams>`` → 7 × the drink's grams; ``pct_energy`` is refused with the share-of-energy
    route, and any other unit (lb, cm, years …) refused as no amount of food. Unrecorded, it is
    settled only by the registry's value test (``KIND_RULES["unit:factor"]``, :func:`factor_in_grams`:
    grams, where the Atwater identity excludes every other unit), read through
    :func:`factor_verdicts` (``verdict``, else read from ``store``). A name's ``_g``, ``_kcal`` or
    codebook never settles it (the fifth gate: an InBody ``Protein`` confirmed in kg moved energy at
    4 kcal per unit), nor do the values for alcohol or a minor source (the sixth gate: drinks moved
    at 7 kcal per drink)."""
    from turbotab.core.methods.energy import KCAL_PER_KJ, default_atwater, nutrient_role

    unit = confirmation(state, "unit", column)

    def atwater() -> tuple[float | None, str, str | None]:
        try:
            role = nutrient_role(column)
        except ValueError as err:
            return None, f"{err}, so its Atwater factor is ambiguous", None
        factors = default_atwater()
        if role is None or role not in factors:
            return None, (f"`{column}` reads as no nutrient the Atwater factors cover, so its kcal "
                          f"per gram must be declared"), role
        return float(factors[role]), f"{role} at {factors[role]:g} kcal/g", role

    if unit is not None:
        unit = str(unit)
        grams = parse_drinks(unit)
        if grams is not None:
            per_gram, why, role = atwater()
            if role != "alcohol" or per_gram is None:
                return KcalPerUnit(column, None, False,
                                   f"`{column}` is recorded in standard drinks, which measure "
                                   f"alcohol only")
            return KcalPerUnit(column, per_gram * grams, True,
                               f"recorded in {unit_words(unit)}: {why} × {_grams_text(grams)} g "
                               f"= {per_gram * grams:g} kcal per drink", unit=unit)
        if unit in ("g", "kg"):
            per_gram, why, _ = atwater()
            if per_gram is None:
                return KcalPerUnit(column, None, False, why)
            scale = 1000.0 if unit == "kg" else 1.0
            return KcalPerUnit(column, per_gram * scale, True,
                               f"recorded in {unit}: {why}" + (" × 1,000 per kg" if scale > 1
                                                               else ""), unit=unit)
        if unit == "kcal":
            return KcalPerUnit(column, 1.0, True, "recorded in kcal", unit=unit)
        if unit == "kj":
            return KcalPerUnit(column, 1.0 / KCAL_PER_KJ, True,
                               f"recorded in kJ: 1/{KCAL_PER_KJ} kcal per kJ", unit=unit)
        if unit == "pct_energy":
            return KcalPerUnit(column, None, False,
                               f"`{column}` is recorded as a share of energy, which carries no "
                               f"constant kcal per unit",
                               route="Move a share of energy instead (the substitution's share-"
                                     "of-energy scale)")
        return KcalPerUnit(column, None, False,
                           f"`{column}` is recorded in {unit}, which is no amount of food energy")
    if verdict is None and store is not None:
        verdict = factor_verdicts(state, store, [column]).get(column)
    if verdict is not None and verdict.settles:
        per_gram, why, _ = atwater()
        if per_gram is not None:
            return KcalPerUnit(column, per_gram, True, f"in grams by its values ({verdict.evidence}): "
                                                       f"{why}", unit="g")
    said = f": {verdict.evidence}" if verdict is not None and verdict.evidence else ""
    return KcalPerUnit(column, None, False,
                       f"`{column}`'s unit is not recorded, and a name never says it (an InBody "
                       f"`Protein` is kilograms of body protein; alcohol is kept in drinks){said}")


def factor_verdicts(state: Any, store: Any, columns: Iterable[str],
                    row_ids: Any = None) -> dict[str, Verdict]:
    """``unit:factor`` by the values for each of ``columns``: the registry's one test
    (:func:`by_values`, :func:`factor_in_grams`) against the settled total-energy column, over the
    columns that name a macronutrient total beside it (NUTRITION_PACK §01). Empty with no store or
    no settled total-energy column: only a recorded unit settles then."""
    columns = [str(c) for c in columns]
    if store is None:
        return {}
    names = set(getattr(store, "columns", ()) or ())
    energy = next((c for c, r in settled_roles(state).items() if r == "energy"), None)
    if energy is None or energy not in names:
        return {}
    from turbotab.core.decisions import _names_a_macro_total

    totals = [c for c in store.columns if c != energy and _names_a_macro_total(c)]
    wanted = list(dict.fromkeys([energy, *totals, *(c for c in columns if c in names)]))
    try:
        # ``row_ids``: the rows a check before the seal may read (never a held-out one).
        frame = store.materialize(wanted) if row_ids is None else store.materialize(wanted,
                                                                                    row_ids)
    except Exception:  # noqa: BLE001 - no values to read: only a recorded unit settles
        return {}
    return {c: by_values("unit:factor", frame, energy, c) for c in columns if c in frame.columns}


def frame_factor_verdicts(state: Any, frame: Any, columns: Iterable[str]) -> dict[str, Verdict]:
    """``unit:factor`` by the values of ``frame`` alone (:func:`factor_verdicts` reads the store):
    for a consumer that holds the rows it fits on and no store. Empty without a settled total-energy
    column among them: only a recorded unit settles then."""
    names = {str(c) for c in (frame.columns if frame is not None else ())}
    energy = next((c for c, r in settled_roles(state).items() if r == "energy"), None)
    if energy is None or energy not in names:
        return {}
    return {str(c): by_values("unit:factor", frame, energy, str(c)) for c in columns
            if str(c) in names}


def energy_source_factors(state: Any, columns: Iterable[str], *, store: Any = None,
                          frame: Any = None, row_ids: Any = None) -> dict[str, KcalPerUnit]:
    """Each energy source's kcal per unit for an energy model that converts it to kcal (the
    partition and the all-components model), read as :func:`kcal_per_unit` reads it: the user's
    recorded unit, else grams the registry's Atwater test reads from the values of ``store`` (every
    row) or ``frame`` (the rows at hand); never the name (BLUEPRINT §14.3: the ledger's repair 3
    residue, `alcohol_g` holding US drinks moved at 7 kcal per unit by its suffix)."""
    columns = [str(c) for c in columns]
    if store is not None:
        verdicts = factor_verdicts(state, store, columns, row_ids)
    elif frame is not None:
        verdicts = frame_factor_verdicts(state, frame, columns)
    else:
        verdicts = {}
    return {c: kcal_per_unit(state, c, verdict=verdicts.get(c)) for c in columns}


def energy_sources_or_ask(state: Any, columns: Iterable[str], *, store: Any = None,
                          frame: Any = None, row_ids: Any = None,
                          found: Mapping[str, KcalPerUnit] | None = None) -> dict[str, KcalPerUnit]:
    """:func:`energy_source_factors` (or ``found``, already read), once every one is settled; else
    :class:`Unsettled`, asking each unsettled source's unit (:func:`factor_exits`), or naming the
    recorded unit that carries no constant kcal per unit."""
    found = dict(found) if found is not None else energy_source_factors(
        state, columns, store=store, frame=frame, row_ids=row_ids)
    refused = [f for f in found.values() if not f.settled and confirmation(state, "unit", f.column)]
    if refused:
        f = refused[0]
        exits = [e for e in factor_exits(f.column) if e.get("decision")]
        raise Unsettled(f"{f.why}, so an energy model that converts it to kcal has no kcal per unit "
                        f"to read. Record its unit as an amount, or adjust for energy another way.",
                        [Reading((f.column,), "unit", None, "low", f.why, False, "proposed")],
                        exits)
    waiting = [f for f in found.values() if not f.settled]
    if waiting:
        names = [f.column for f in waiting]
        one = len(names) == 1
        raise Unsettled(
            f"{listing(names)} {'is' if one else 'are'} split into {'its' if one else 'their'} kcal "
            f"by the energy model, but {'its' if one else 'their'} unit is not recorded, so the kcal "
            f"each unit carries would be a guess (a name never says it: an `alcohol_g` may count "
            f"standard drinks). Record {'its' if one else 'each'} unit.",
            [Reading((f.column,), "unit", "g", "medium", f.why, False, "proposed")
             for f in waiting],
            [e for f in waiting for e in factor_exits(f.column)])
    return found


def unsettled_factors(state: Any, columns: Iterable[str], store: Any = None) -> list[str]:
    """The columns whose kcal per unit is not settled (:func:`kcal_per_unit`)."""
    columns = list(columns)
    verdicts = factor_verdicts(state, store, columns)
    return [c for c in columns
            if not kcal_per_unit(state, c, verdict=verdicts.get(c)).settled]


def factor_exits(column: str) -> list[dict[str, Any]]:
    """The ways to record an energy source's unit, each of which sets its kcal per unit; for
    alcohol, the standard drinks too (:data:`OFFERED_DRINKS`, any other size named the same way)."""
    from turbotab.core.methods.energy import nutrient_role

    words = {"g": "grams", "kg": "kilograms", "kcal": "kcal", "kj": "kJ"}
    out = [confirm_exit("unit", column, u, f"`{column}` is in {words[u]}") for u in FACTOR_UNITS]
    try:
        alcohol = nutrient_role(column) == "alcohol"
    except ValueError:
        alcohol = False
    if alcohol:
        out += [confirm_exit("unit", column, drinks_value(g), f"`{column}` counts drinks: {said}")
                for g, said in OFFERED_DRINKS]
        out.append({"label": f"`{column}` counts another country's standard drinks: record "
                             f"`drinks:<grams of alcohol>` (8–20 g)", "decision": None})
    out.append({"label": f"`{column}` is a share of energy: move a share of energy instead (the "
                         f"substitution's share-of-energy scale)", "decision": None})
    return out


# ── parts of totals (nested_in) ──────────────────────────────────────────────

# The answer that a column is no part of any total (a nutrient of its own), beside naming a column.
NOT_NESTED = "not_nested"
# The confirmable kinds whose value names a column of the table: confirming any eligible column
# must produce that column's behavior (the property test enumerates every one).
COLUMN_VALUED = ("nested_in",)


def nesting_parents(column: str, columns: Iterable[str], target: str | None = None) -> list[str]:
    """The values a ``nested_in`` confirmation of ``column`` may record: every other column of the
    table but the outcome, and :data:`NOT_NESTED`."""
    from turbotab.core.decisions import ROW_ID

    return [str(c) for c in columns if str(c) not in (str(column), str(target), ROW_ID)] \
        + [NOT_NESTED]


def nesting(state: Any, found: Mapping[str, str] | None = None, *, frame: Any = None,
            columns: Iterable[str] | None = None) -> dict[str, str]:
    """Child -> the total it is part of, as the ledger holds it (BLUEPRINT §14.3, every confirmation
    is honored; the sixth gate's ``sfa_g`` confirmed as part of `carbohydrate_g` and read nowhere):
    the guess (``found``, else the names and values of ``frame``'s ``columns``:
    ``methods.nesting.nested_components``), each column's own confirmation standing over it: the
    total the user named, any column of the table, or no total (:data:`NOT_NESTED`). Restricted to
    ``columns`` when given (a part and its total both among them)."""
    names = None if columns is None else [str(c) for c in columns]
    if found is None and frame is not None:
        from turbotab.core.methods.nesting import nested_components

        found = nested_components(frame, names)
    out = {str(c): str(p) for c, p in dict(found or {}).items()}
    for name, value in _confirmations(state).items():
        if not name.startswith("nested_in:"):
            continue
        child = name.split(":", 1)[1]
        if str(value) == NOT_NESTED:
            out.pop(child, None)
        else:
            out[child] = str(value)
    if names is not None:
        keep = set(names)
        out = {c: p for c, p in out.items() if c in keep and p in keep}
    return {c: p for c, p in out.items() if c != p}


def nested_exits(column: str, parent: str) -> list[dict[str, Any]]:
    """The ways to settle what ``column`` is part of: the total the guess names, none, or another
    column the user names (``confirm_reading`` nested_in, any column of the table)."""
    return [confirm_exit("nested_in", column, parent, f"`{column}` is part of `{parent}`"),
            confirm_exit("nested_in", column, NOT_NESTED,
                         f"`{column}` is no part of `{parent}` or any other total"),
            {"label": f"`{column}` is part of another column: name it (confirm `{column}` nested in "
                      f"that column)", "decision": None}]


# ── time-invariance under clustered imputation (MODELING_SEQUENCE §0 ruling 12) ─

TIME_INVARIANT_WORDS = {"yes": "one value for each unit (carried to the unit's blank rows; "
                               "imputed once per unit where no row records it)",
                        "no": "can change between a unit's rows (imputed row by row)"}


def time_invariance(state: Any, column: str, values: Any, units: Any) -> Reading:
    """Whether ``column`` holds one value for each unit (``units``: a code per row), as the ledger
    holds it: the user's confirmation, else a guess from the values that is never settled
    (:data:`KIND_RULES` ``time_invariant``: a covariate asked only at baseline agrees within every
    unit as a characteristic that does not change would, and so do few recorded values by chance).
    The guess is "yes" where the recorded values agree within every unit, its evidence how many
    units record it on two or more rows, on one, and on none; "no" where they differ within a unit
    (no one value per unit holds those records)."""
    import numpy as np
    import pandas as pd

    recorded = confirmation(state, "time_invariant", column)
    if recorded is not None:
        return Reading((str(column),), "time_invariant", str(recorded), "high",
                       recorded_evidence(state, "time_invariant", column), False, "confirmed")
    col = pd.Series(np.asarray(values, dtype=object)).reset_index(drop=True)
    codes = pd.Series(np.asarray(units)).reset_index(drop=True)
    rec = col.notna().to_numpy()
    g = col[rec].astype(str).groupby(codes[rec].to_numpy())
    distinct = g.nunique()
    sizes = g.size()
    n_units = int(codes.nunique())
    differ = int((distinct > 1).sum())
    if differ:
        return Reading((str(column),), "time_invariant", "no", "medium",
                       f"its recorded values differ within {differ:,} of the {n_units:,} units",
                       False, "proposed")
    several, once = int((sizes >= 2).sum()), int((sizes == 1).sum())
    none = n_units - int(len(sizes))
    return Reading((str(column),), "time_invariant", "yes", "medium" if several else "low",
                   f"one recorded value in each unit that records it: {several:,} units record it "
                   f"on two or more rows, {once:,} on one row, {none:,} on none", False, "proposed")


def time_invariance_exit(column: str, value: str) -> dict[str, Any]:
    """The confirmation of ``column``'s time-invariance as ``value`` (``yes`` or ``no``)."""
    return confirm_exit("time_invariant", column, value,
                        f"`{column}` holds {TIME_INVARIANT_WORDS[value]}" if value == "yes"
                        else f"`{column}` {TIME_INVARIANT_WORDS[value]}")


# ── values below a detection limit ───────────────────────────────────────────


def detection_limit(state: Any, column: str) -> float | None:
    """The fraction of its limit a value below a detection limit (``<0.20``) is read at, as the
    user answered for ``column`` (the below-detection repair: half the limit, 0.5; the limit over
    √2, 1/√2), or None while unanswered. The values hold no answer: each such value is somewhere
    below its limit."""
    dispositions = _get(state, "findings") or {}
    found = dispositions.get(f"below_detection__{column}") if isinstance(dispositions, Mapping) \
        else None
    if found is None or _get(found, "action") != "applied":
        return None
    spec = ((_get(found, "params") or {}).get("columns") or {}).get(column) or {}
    factor = spec.get("factor") if isinstance(spec, Mapping) else None
    return None if factor is None else float(factor)


def comma_exits(column: str) -> list[dict[str, Any]]:
    """The two readings of a text column whose every comma could split thousands or mark decimals:
    the text-number repair's two offers (``repairs``), and codes instead."""
    from turbotab.core.decisions import ApplyRepair

    out = []
    for key, decimal, thousands, words in (("thousands", ".", True, "commas split thousands"),
                                           ("decimal_comma", ",", False, "commas mark decimals")):
        decision = ApplyRepair(finding_id=f"text_numbers__{column}", option=key,
                               params={"columns": {column: {"decimal": decimal,
                                                            "thousands": thousands}}})
        out.append({"label": f"`{column}` is numbers whose {words}",
                    "decision": decision.model_dump(mode="json")})
    out.append(code_or_count_exits(column)[1])
    return out


def detection_exits(column: str, facts: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The answers to how ``column``'s values below a detection limit read once it is read as
    numbers: the below-detection repair's two offers (``repairs``: half the limit, the limit over
    √2), each with the parameters the finding offers, and codes instead."""
    from turbotab.core.decisions import ApplyRepair
    from turbotab.core.repairs import LOD_FACTORS

    below = dict(facts.get("below") or {})
    first = next(iter(below), None)
    spec = {"decimal": str(facts.get("decimal") or "."), "thousands": bool(facts.get("thousands"))}
    out = []
    for key, words in (("half_limit", "half its limit"), ("limit_root2", "its limit over √2")):
        factor = LOD_FACTORS[key]
        example = f" (`<{first}` is `{float(first) * factor:.4g}`)" if first is not None else ""
        decision = ApplyRepair(finding_id=f"below_detection__{column}", option=key,
                               params={"columns": {column: {**spec, "factor": factor}}})
        out.append({"label": f"`{column}` is numbers: read each value below a detection limit as "
                             f"{words}{example}", "decision": decision.model_dump(mode="json")})
    out.append(code_or_count_exits(column)[1])
    return out


# ── a sex column's coding ────────────────────────────────────────────────────


def sex_coding_value(female: Any, male: Any) -> str:
    return f"female={_level(female)},male={_level(male)}"


def parse_sex_coding(value: Any) -> dict[str, str] | None:
    """``"female=2,male=1"`` -> ``{"2": "female", "1": "male"}``; None when it is no coding."""
    import re

    m = re.fullmatch(r"female=([^,]+),male=([^,]+)", str(value or "").strip())
    if not m or m.group(1) == m.group(2):
        return None
    return {m.group(1).strip(): "female", m.group(2).strip(): "male"}


def sex_coding_reading(state: Any, column: str, values: Any = None) -> Reading:
    """Which level of a sex column is female and which male. Settled when the user confirmed it
    (``confirm_reading`` sex_coding ``female=2,male=1``), or when the values are the words
    themselves (:func:`labels_spell_sex`). Numeric codes never settle it: NHANES and the CDC growth
    charts code 1 male and 2 female, other studies 1 female, or 0/1 either way. The best guess for
    two numeric codes is the CDC's order (the lower code male)."""
    import pandas as pd

    recorded = confirmation(state, "sex_coding", column)
    if recorded is not None and parse_sex_coding(recorded):
        return Reading((column,), "sex_coding", str(recorded), "high",
                       recorded_evidence(state, "sex_coding", column), True, "confirmed")
    if values is None:
        return Reading((column,), "sex_coding", None, "low", "its values were not read", False,
                       "proposed")
    verdict = labels_spell_sex(values)
    if verdict.settles:
        mapping = {}
        for v in pd.unique(_present(values)):
            word = str(v).strip().lower()
            mapping[_level(v)] = "female" if word in FEMALE_WORDS else "male"
        female = next(k for k, v in mapping.items() if v == "female")
        male = next(k for k, v in mapping.items() if v == "male")
        return Reading((column,), "sex_coding", sex_coding_value(female, male), "high",
                       verdict.evidence, True, "proposed")
    levels = sorted({_level(v) for v in pd.unique(_present(values))}, key=_sort_key)
    if len(levels) == 2:
        guess = sex_coding_value(levels[1], levels[0])
        return Reading((column,), "sex_coding", guess, "medium",
                       f"{verdict.evidence}; NHANES and the CDC growth charts code "
                       f"`{levels[0]}` male and `{levels[1]}` female", False, "proposed")
    return Reading((column,), "sex_coding", None, "low", verdict.evidence, False, "proposed")


def _sort_key(level: str) -> tuple[int, Any]:
    try:
        return (0, float(level))
    except ValueError:
        return (1, level)


def sex_levels(state: Any, column: str, values: Any = None) -> dict[str, str]:
    """``{level: "female" | "male"}`` from a settled sex coding; empty while it is not settled."""
    r = sex_coding_reading(state, column, values)
    if not r.settled:
        return {}
    return parse_sex_coding(r.value) or {}


def sex_coding_exits(column: str, values: Any) -> list[dict[str, Any]]:
    """One confirmation per way the two levels may be coded."""
    import pandas as pd

    levels = sorted({_level(v) for v in pd.unique(_present(values))}, key=_sort_key)
    if len(levels) != 2:
        return [{"label": f"Say which level of `{column}` is female (the roles question)",
                 "decision": None}]
    a, b = levels
    return [confirm_exit("sex_coding", column, sex_coding_value(b, a),
                         f"`{column}`: {b} female, {a} male"),
            confirm_exit("sex_coding", column, sex_coding_value(a, b),
                         f"`{column}`: {a} female, {b} male")]


# ── settlement is visible: what the values settled (BLUEPRINT §14.3, amendment) ──

# Each role read from the values may be changed to these (one confirmation each; a confirmed role
# is the column's role, ``decisions._confirmed_role_is_the_role``).
ROLE_CHANGES: dict[str, tuple[str, ...]] = {
    "identifier": ("covariate", "excluded"), "exposure": ("covariate", "excluded"),
    "covariate": ("exposure", "excluded"), "energy": ("covariate", "excluded"),
    "excluded": ("covariate", "exposure"), "time": ("covariate",), "flag": ("covariate",),
    "design": ("covariate",), "cluster": ("covariate",),
}
TASKS = ("regression", "binary", "multiclass", "ordinal")
_SEX_WORDS = ("sex", "gender")


def read_from_data(state: Any, *, roles: Any = None, target_info: Any = None,
                   proposals: Any = None, store: Any = None,
                   info: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """The readings the values settled with no question asked, each with its evidence and the
    exits that change it (BLUEPRINT §14.3, amendment: "Readings settled by their values appear on
    the card under 'read from your data', each with its evidence and a way to change it, and in the
    methods record"). Not a question: nothing waits for them. Listed only while the user has not
    answered the reading themselves:

    * a role proposed high (its value test rejected every alternative) and recorded as proposed;
    * the outcome's task, settled by its values (two values; decimals filling their grid);
    * total energy's unit by the Atwater identity (kcal, ratio near 1);
    * a height's unit by its human band;
    * a predictor's values with decimals that fill their grid: an amount;
    * a text predictor's labels (three or more): codes;
    * an energy source's grams, by the Atwater identity where it excludes every other unit;
    * a sex column's coding, spelled by its labels.

    ``roles``, ``target_info`` and ``proposals`` are those stages' artifacts (or None); ``store``
    the working table's data, read for the predictors' values."""
    from turbotab.core.decisions import SetColumnUnit, SetTask

    items: list[dict[str, Any]] = []
    recorded = getattr(state, "roles", None) or {}
    own = getattr(state, "role_confirmations", None) or {}

    def add(kind: str, column: str, value: Any, words: str, evidence: str,
            change: list[dict[str, Any]]) -> None:
        items.append({"kind": kind, "column": str(column), "value": value, "words": words,
                      "evidence": str(evidence or ""), "change": change})

    for p in proposals_of(roles):
        column, role = str(p.get("column")), str(p.get("proposed"))
        if p.get("confidence") != "high" or column in own:
            continue
        if recorded and recorded.get(column) != role:
            continue  # the user's own answer, not the values'
        add("role", column, role, f"is {role_words(role)}", str(p.get("reason") or ""),
            [confirm_exit("role", column, alt, f"`{column}` is {role_words(alt)}")
             for alt in ROLE_CHANGES.get(role, ("covariate",))])
    ti = getattr(target_info, "data", target_info)
    target = _get(state, "target")
    if isinstance(ti, Mapping) and target and ti.get("column") == target \
            and ti.get("confidence") == "high" and _get(state, "task") is None:
        task = str(ti.get("task"))
        add("task", target, task, f"is a {task} outcome", str(ti.get("reason") or ""),
            [{"label": f"`{target}` is a {t} outcome",
              "decision": SetTask(column=target, task=t).model_dump(mode="json")}
             for t in TASKS if t != task])
    props = getattr(proposals, "data", proposals)
    unit = (props or {}).get("energy_unit") if isinstance(props, Mapping) else None
    energy = ((props or {}).get("energy") or {}).get("energy_column") \
        if isinstance(props, Mapping) else None
    if isinstance(unit, Mapping) and unit.get("basis") == "atwater" and energy \
            and confirmation(state, "unit", energy) is None:
        add("unit", energy, unit.get("unit"), "is in kcal", str(unit.get("sentence") or ""),
            [{"label": f"`{energy}` is in kJ, one day's intake",
              "decision": SetColumnUnit(column=energy, unit="kj", days=1).model_dump(mode="json")},
             {"label": f"`{energy}` is a total over 2 days, in kcal",
              "decision": SetColumnUnit(column=energy, unit="kcal", days=2).model_dump(mode="json")}])
    if store is not None:
        columns = set(getattr(store, "columns", ()) or ())
        preds = [c for c in predictor_columns(state) if c in columns]
        if any(k in ASSAY_LENSES for k in (_get(state, "lens") or [])):
            # An assay's features are amounts by the lens answer, not by their values.
            preds = [c for c in preds if (getattr(state, "roles", None) or {}).get(c) != "exposure"]
        try:
            facts = whole_facts(preds, info, store)
        except Exception:  # noqa: BLE001 - a store that cannot answer lists nothing
            facts = {}
        left, amounts = energy_plan(state, preds)
        for c in preds:
            f = facts.get(c) or {}
            if c in left or c in amounts or f.get("whole") or not f.get("amount_by_values"):
                continue
            if confirmation(state, "code_or_count", c) is not None:
                continue
            add("code_or_count", c, "amount", "is an amount (one slope)", str(f.get("evidence") or ""),
                [confirm_exit("code_or_count", c, "code",
                              f"`{c}` holds codes for categories (one indicator per level)")])
        # Text read as labels: codes by their values (the sixth gate: labels settled as codes were
        # never listed). Two labels are one indicator either way, so they are not listed.
        try:
            text = store.text_numbers([c for c in preds if c not in left | amounts])
        except Exception:  # noqa: BLE001 - a store that cannot answer lists nothing
            text = {}
        for c, f in text.items():
            if not f.get("labels") or int(f.get("n_values") or 0) < 3 \
                    or confirmation(state, "code_or_count", c) is not None:
                continue
            add("code_or_count", c, "code", "holds codes for categories (one indicator per label)",
                f"`{int(f.get('n_values') or 0):,}` labels, not numbers",
                [confirm_exit("role", c, "excluded", f"Leave `{c}` out of the model")])
        # An energy source's grams, read by the Atwater identity (its kcal per unit: the Atwater
        # factor), where it excludes every other unit (``factor_in_grams``).
        from turbotab.core.stages.rows import energy_bearing

        sources = [c for c in preds if (getattr(state, "roles", None) or {}).get(c) == "exposure"
                   and energy_bearing(c) and confirmation(state, "unit", c) is None]
        try:
            verdicts = factor_verdicts(state, store, sources) if sources else {}
        except Exception:  # noqa: BLE001 - no values to read: nothing is listed
            verdicts = {}
        for c, v in verdicts.items():
            if v.settles:
                add("unit", c, "g", "is in grams (its kcal per unit is its Atwater factor)",
                    v.evidence, [e for e in factor_exits(c) if e.get("decision")
                                 and e["decision"].get("value") != "g"])
        settled_now = settled_roles(state)
        from turbotab.core.recognizers import tokens

        for c in [c for c in settled_now if c in columns]:
            words = set(tokens(c))
            if words & {"height", "stature", "bmxht"} or str(c).lower() in ("height", "bmxht"):
                try:
                    values = store.materialize([c])[c]
                except Exception:  # noqa: BLE001
                    continue
                r = body_unit_reading(c, "height", state, values)
                if r.settled and r.state != "confirmed":
                    add("unit", c, r.value, f"is in {r.value}", r.evidence,
                        [confirm_exit("unit", c, u, f"`{c}` is in {u}")
                         for u in ("cm", "m", "in") if u != r.value])
            if words & set(_SEX_WORDS):
                try:
                    values = store.materialize([c])[c]
                except Exception:  # noqa: BLE001
                    continue
                r = sex_coding_reading(state, c, values)
                if r.settled and r.state != "confirmed":
                    coding = parse_sex_coding(r.value) or {}
                    female = next((k for k, v in coding.items() if v == "female"), None)
                    male = next((k for k, v in coding.items() if v == "male"), None)
                    add("sex_coding", c, r.value, f"is coded {guess_words(r)}", r.evidence,
                        [confirm_exit("sex_coding", c, sex_coding_value(male, female),
                                      f"`{c}`: {male} female, {female} male")]
                        if female is not None and male is not None else [])
    return items


def read_from_values_sentence(items: Sequence[Mapping[str, Any]]) -> str:
    """The methods record's line for the readings the values settled: "Read from the values: …",
    each with its evidence; empty when there are none."""
    if not items:
        return ""
    parts = []
    for it in items:
        evidence = str(it.get("evidence") or "").strip().rstrip(".")
        parts.append(f"`{it['column']}` {it['words']}" + (f" ({evidence})" if evidence else ""))
    return "Read from the values, no question asked: " + "; ".join(parts) + "."


# ── the ask: what a consumer needs, listed once, by consequence (BLUEPRINT §14.2) ──

# How much a reading changes, by kind (lower first): which rows belong together moves every
# interval; a role puts a column in or out of the model; codes or amounts reshape the design matrix
# (one slope or an indicator per level); a unit or a day count moves a screen's bounds; a sex
# coding the sex-specific references; the order of records the values first, last and change take.
CONSEQUENCE = {"cluster": 0, "role": 1, "code_or_count": 2, "unit": 3, "day_count": 3,
               "time_invariant": 3, "sex_coding": 4, "time_column": 5, "outcome_unit": 6,
               "nested_in": 6}
FAMILY_MIN = 5  # this many readings of one kind, one guess and one name pattern are one family


def _weight(r: Reading) -> int:
    """Within a kind, the readings that change more come first (more levels: more indicators)."""
    import re

    if r.kind == "code_or_count":
        m = re.search(r"`([\d,]+)` whole-number values", r.evidence or "")
        return -int(m.group(1).replace(",", "")) if m else 0
    return 0


def _pattern(column: str) -> str:
    """A family's name pattern: the name's first word, digits as ``#`` (``ENSG00000141510`` →
    ``ENSG#``, ``imputed_bmi`` → ``imputed``, ``item_07`` → ``item``)."""
    import re

    first = re.split(r"[^0-9A-Za-z]+", str(column).strip())[0] or str(column)
    return re.sub(r"\d+", "#", first)


def ordered(readings: Iterable[Reading]) -> list[Reading]:
    """Each reading once (by kind and column), ordered by consequence."""
    seen: dict[tuple[str, str], Reading] = {}
    for r in readings:
        if r is None:
            continue
        seen.setdefault((r.kind, r.column), r)
    return sorted(seen.values(), key=lambda r: (CONSEQUENCE.get(r.kind, 9), _weight(r), r.column))


def families(readings: Iterable[Reading]) -> list[list[Reading]]:
    """The readings in order, a homogeneous family (one kind, one guess, one name pattern, at
    least :data:`FAMILY_MIN` of them: 20,000 genes proposed as exposures) as one group."""
    groups: dict[tuple[str, str, str], list[Reading]] = {}
    order: list[tuple[str, str, str]] = []
    for r in ordered(readings):
        k = (r.kind, str(r.value), _pattern(r.column))
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(r)
    out: list[list[Reading]] = []
    for k in order:
        g = groups[k]
        if len(g) >= FAMILY_MIN:
            out.append(g)
        else:
            out.extend([r] for r in g)
    return out


_GUESS_WORDS = {"code": "codes for categories", "amount": "an amount", "yes": "rows belong together",
                "no": "groups nothing", "orders": "orders the records",
                # a value below a detection limit, read as numbers (``detection_exits``)
                "half_limit": "below its detection limit, read at half the limit",
                "limit_root2": "below its detection limit, read at the limit over √2"}


def guess_words(r: Reading) -> str:
    if r.kind == "role":
        return role_words(r.value)
    if r.kind == "day_count":
        return f"{r.value} day{'s' if str(r.value) != '1' else ''}" if r.value else "how many days?"
    if r.kind == "sex_coding" and r.value:
        coding = parse_sex_coding(r.value) or {}
        return ", ".join(f"{k} {v}" for k, v in coding.items())
    if r.kind == "time_invariant":
        return TIME_INVARIANT_WORDS.get(str(r.value), str(r.value))
    return _GUESS_WORDS.get(str(r.value), str(r.value))


# A codebook label's words that say what a column's numbers are (BLUEPRINT §14.2: a label only
# strengthens the guess the card leads with; the user confirms it in one tap).
_LABEL_CODES = re.compile(
    r"\b(code|codes|coded|status|category|categories|type|group|race|ethnicity|hispanic|origin|"
    r"gender|sex|education|marital|stratum|strata|psu|region|language|interpreter|proxy|"
    r"yes/no|indicator|flag|level|class|comment|cycle|release|period)\b", re.I)
_LABEL_AMOUNTS = re.compile(
    r"\b(age in|number of|count of|total number|amount|weight|height|length|circumference|"
    r"intake|ratio|index|score|minutes|hours|days|years|months|per day|per week)\b|#|"
    r"\((?:g|mg|mcg|µg|kg|kcal|kj|cm|mm|m|ml|l|mmhg|mg/dl|mmol/l|%)\)", re.I)
# A unit a label spells in parentheses or as "in <unit>" (``Weight (kg)``, ``Age in years``).
_LABEL_UNIT = re.compile(r"\(([^()]{1,24})\)\s*$|\bin (years|months|weeks|days)\b", re.I)


def label_guess(r: Reading, label: str) -> Any:
    """The value a codebook label points a reading's guess to, or None (no word in it says)."""
    if r.kind == "code_or_count":
        codes, amounts = bool(_LABEL_CODES.search(label)), bool(_LABEL_AMOUNTS.search(label))
        if codes != amounts:
            return "code" if codes else "amount"
        return None
    if r.kind == "unit":
        from turbotab.core.codebook import unit_value

        m = _LABEL_UNIT.search(label)
        if m:
            unit = unit_value(m.group(1) or m.group(2))
            return unit if unit in UNIT_VALUES else None
    return None


def labeled(readings: Iterable[Reading], state: Any = None) -> list[Reading]:
    """Each unsettled reading with the label an imported codebook gives its column beside its
    evidence, and its guess moved to what the label's words point to (a unit it spells, codes or
    an amount). Still proposed and still medium: a label is a name, and names never settle a
    reading (BLUEPRINT §14.3); the user confirms the guess in one tap (§14.2)."""
    out = []
    for r in readings:
        label = codebook_label(state, r.column) if state is not None and r is not None else None
        if r is None or label is None or r.settled:
            out.append(r)
            continue
        guess = label_guess(r, label)
        said = f"your codebook labels it \"{label}\""
        evidence = f"{r.evidence}; {said}" if r.evidence else said
        value = guess if guess is not None else r.value
        out.append(Reading(r.subject, r.kind, value, r.confidence, evidence, r.corroborated,
                           r.state))
    return out


def ask_text(readings: Iterable[Reading], state: Any = None) -> str:
    """The question in words: each reading (or family) once, by consequence, its best guess and its
    evidence (a codebook's label beside it, :func:`labeled`, when ``state`` holds one)."""
    if state is not None:
        readings = labeled(readings, state)
    lines = []
    for group in families(readings):
        r = group[0]
        if len(group) > 1:
            lines.append(f"{len(group):,} columns like `{r.column}`: {guess_words(r)}?")
        else:
            why = f" ({r.evidence})" if r.evidence else ""
            lines.append(f"`{r.column}`: {guess_words(r)}?{why}")
    return ("Tell me about " + ("this column" if len(lines) == 1 else "these columns") + ": "
            + "; ".join(lines) + ".") if lines else ""


def _alternatives(r: Reading, state: Any = None) -> list[str]:
    if r.kind == "code_or_count":
        return ["amount", "code"] if r.value != "code" else ["code", "amount"]
    if r.kind == "cluster":
        return ["yes", "no"]
    if r.kind == "role":
        return [str(r.value)]
    if r.kind == "time_column":
        return ["orders"]
    if r.kind == "time_invariant":
        return ["yes", "no"] if r.value != "no" else ["no", "yes"]
    return [str(r.value)] if r.value is not None else []


def block_exit(readings: Iterable[Reading], label: str | None = None) -> dict[str, Any] | None:
    """One ``confirm_readings`` decision that settles exactly the readings listed, each with the
    value it shows (BLUEPRINT §14.2): never a reading it does not list. None when no reading has a
    guess to confirm."""
    from turbotab.core.decisions import ConfirmReadings

    items = [{"reading": r.kind, "column": r.column, "value": str(r.value)}
             for r in ordered(readings) if r.value is not None and r.kind in CONFIRMABLE]
    if not items:
        return None
    n = len(items)
    decision = ConfirmReadings(items=items)
    return {"label": label or (f"Confirm {'it' if n == 1 else f'each of the {n:,}'} as shown"),
            "decision": decision.model_dump(mode="json")}


def ask_exits(readings: Iterable[Reading], state: Any = None) -> list[dict[str, Any]]:
    """The ask's ways forward: one block confirmation of every best guess as shown (it settles
    exactly those readings), then each reading's own alternatives, one confirmation each, in the
    ask's order. A codebook label moves a guess as :func:`ask_text` shows it (:func:`labeled`)."""
    listed = ordered(labeled(readings, state) if state is not None else readings)
    out: list[dict[str, Any]] = []
    block = block_exit(listed)
    if block is not None and len(listed) > 1:
        out.append(block)
    for r in listed:
        if r.kind == "code_or_count":
            for value in _alternatives(r):
                out.append(code_or_count_exits(r.column)[0 if value == "amount" else 1])
        elif r.kind == "role":
            roles = getattr(state, "roles", None) or {}
            out += confirm_exits(state, [r.column]) if r.column in roles else [
                confirm_exit("role", r.column, r.value, f"Confirm `{r.column}` as "
                                                        f"{role_words(r.value)}")]
        elif r.kind == "cluster":
            out += cluster_exits(state, r.column)[:2]
        elif r.kind == "time_invariant":
            out += [time_invariance_exit(r.column, value) for value in _alternatives(r)]
        else:
            for value in _alternatives(r, state):
                out.append(confirm_exit(r.kind, r.column, value,
                                        f"`{r.column}`: {guess_words(r)}"))
    return out


# ── acquisition: the injection order and the batch QC-RLSC reads (MS7) ─────────────

# Acquisition words a batch column goes by: an analytical batch, a run, a plate (a 96-well plate is
# often the batch an untargeted run is injected in), in that order of trust.
BATCH_KINDS = ("batch", "run", "plate")
MAX_BATCH_CANDIDATES = 2  # columns offered besides "one curve": the QC-RLSC options stay few


def _whole_permutation(values: Any) -> bool:
    import numpy as np
    import pandas as pd

    x = pd.to_numeric(values, errors="coerce")
    if x.isna().any() or not len(x):
        return False
    v = np.sort(x.to_numpy(dtype=float))
    return bool(np.all(np.mod(v, 1) == 0) and (np.array_equal(v, np.arange(1, len(v) + 1))
                                                 or np.array_equal(v, np.arange(len(v)))))


def injection_order_reading(frame: Any, exclude: Sequence[str] = ()) -> Reading | None:
    """The column that orders the injections, as a guess: a column named as a run order
    (``injection_order``, ``run_order``; the recognizer's acquisition phrases) whose values are
    numbers, complete and all distinct; else a whole-number column that is a permutation of the row
    positions (the metabolomics pack's name-blind reading). Never settled by its values: a sample
    number is all-distinct too. The QC-RLSC options name it, and the user's choice settles it."""
    import pandas as pd

    from turbotab.core.recognizers import acquisition_kind

    gone = {str(c) for c in exclude}
    for c in frame.columns:
        name = str(c)
        if name in gone or acquisition_kind(name) != "run_order":
            continue
        x = pd.to_numeric(frame[c], errors="coerce")
        if x.notna().all() and x.nunique() == len(x):
            return Reading((name,), "injection_order", name, "medium",
                           f"`{name}` is named as a run order, and its {len(x):,} values are "
                           f"distinct numbers")
    for c in frame.columns:
        name = str(c)
        if name not in gone and _whole_permutation(frame[c]):
            return Reading((name,), "injection_order", name, "low",
                           f"`{name}` numbers the {len(frame):,} rows once each, as a run order "
                           f"does; its name does not say so")
    return None


def _contiguous_in(order: Any, labels: Any) -> bool:
    """Every level of ``labels`` occupies one unbroken stretch of the injection order."""
    import numpy as np
    import pandas as pd

    s = pd.DataFrame({"o": pd.to_numeric(pd.Series(order).reset_index(drop=True), errors="coerce"),
                      "b": pd.Series(labels).reset_index(drop=True).astype(str)})
    s = s.dropna().sort_values("o", kind="stable")
    changes = int(np.sum(s["b"].to_numpy()[1:] != s["b"].to_numpy()[:-1]))
    return changes == s["b"].nunique() - 1


def batch_readings(frame: Any, exclude: Sequence[str] = (), order: str | None = None) -> list[Reading]:
    """The columns that may name each injection's analytical batch, best guess first, each a
    guess (:data:`KIND_RULES` ``batch``): columns named as a batch, a run or a plate
    (:data:`BATCH_KINDS`, the recognizer's acquisition phrases) with two or more levels, each level
    on more than one row; then, name-blind, a column of two or more such levels each of which
    occupies one unbroken stretch of the injection order (batches are injected in blocks). At most
    :data:`MAX_BATCH_CANDIDATES`. "No batch: one curve over the whole run" is always the other
    reading; the QC-RLSC options name each, and the user's choice settles it."""
    import pandas as pd

    from turbotab.core.recognizers import acquisition_kind

    gone = {str(c) for c in exclude} | ({order} if order else set())
    n = len(frame)
    named: list[tuple[int, Reading]] = []
    blocks: list[Reading] = []
    for c in frame.columns:
        name = str(c)
        if name in gone:
            continue
        s = frame[c]
        counts = s.value_counts(dropna=True)
        if len(counts) < 2 or int(counts.min()) < 2 or len(counts) > max(2, n // 4):
            continue
        kind = acquisition_kind(name)
        if kind in BATCH_KINDS:
            together = order is not None and _contiguous_in(frame[order], s)
            named.append((BATCH_KINDS.index(kind), Reading(
                (name,), "batch", name, "medium",
                f"`{name}` is named as a{'n' if kind == 'run' else ''} {kind} and holds "
                f"{len(counts):,} levels" + (", each one unbroken stretch of the injection order"
                                            if together else ""))))
        elif (order is not None and not pd.api.types.is_float_dtype(s)
              and _contiguous_in(frame[order], s)):
            blocks.append(Reading(
                (name,), "batch", name, "low",
                f"`{name}`'s {len(counts):,} levels each occupy one unbroken stretch of the "
                f"injection order, as batches do; its name does not say batch"))
    ranked = [r for _, r in sorted(named, key=lambda t: t[0])] + blocks
    return ranked[:MAX_BATCH_CANDIDATES]


# ── the registry of consumers (BLUEPRINT §14.1: the gate checks the invariant structurally) ──


@dataclass(frozen=True)
class Consumer:
    """A consumer of readings, as the census of readers and consumers names it.

    ``where``: ``module:function`` (``module:Class.method`` for a method). ``census``: the ids of
    the census entries it covers (``turbotab/core/tests/acceptance/readings_census.json``).
    ``changes``: whether it can change a number (a fit, an interval, a screen, a weight, a
    combining rule, a unit in a sentence, a default) rather than only words, offers or hints.
    ``path``: what it does when a reading it needs is unsettled. ``via``: for a consumer handed
    its readings already settled, the function (``module:function``) that settles them and calls
    it. The structural acceptance test checks that every consumer that changes a number reads
    through this module, itself or through ``via``."""

    where: str
    census: tuple[str, ...]
    changes: bool
    path: str
    via: str | None = None
    # The registry kinds (:data:`KIND_RULES`, BLUEPRINT §14.3) it reads: confirming each
    # alternative of each must produce that alternative's behavior (the property test).
    kinds: tuple[str, ...] = ()


ASK = "asks: a refusal with one confirmation exit per reading"
SETTLED_ONLY = "reads settled readings only; the rest wait, named"
NO_UNIT = "states no unit until the reading is settled"
VALUES_SETTLE = "settled by its own values (high and corroborated), else asked"
USER_APPLIED = "acts only on the user's own recorded answer, its columns and values shown"
WORDS_ONLY = "words, offers and hints only; no number follows from it"

_C = "turbotab.core."
CONSUMERS: tuple[Consumer, ...] = (
    # ── roles: the fit's predictor set, the energy card, the screens, the survey, the clusters ──
    Consumer(_C + "models.pipeline:model_predictors", ("roles", "covariates"), True,
             SETTLED_ONLY,
             kinds=("role",)),
    Consumer(_C + "models.pipeline:design_spec", ("roles", "predictor_codes", "energy_fill"),
             True, SETTLED_ONLY, kinds=("code_or_count",)),
    # The sixth gate: a text column confirmed "amount" reaches every consumer as numbers, through
    # the working table (marks blank; values below a detection limit at the user's answer).
    Consumer(_C + "stages.working:text_amounts", ("predictor_codes", "combine_codes"), True,
             SETTLED_ONLY, kinds=("code_or_count",)),
    Consumer(_C + "stages.modeling:design_stage",
             ("roles", "predictor_codes", "design_role", "acquisition", "flag", "time_role",
              "covariates", "nesting", "free_text", "total_energy_names"), True, ASK),
    Consumer(_C + "decisions:_models_read_settled_readings",
             ("roles", "predictor_codes", "design_role", "acquisition", "flag", "time_role",
              "covariates"), True, ASK,
             kinds=("code_or_count", "role")),
    Consumer(_C + "stages.rows:cohort_inputs", ("roles", "missing_not_asked"), True,
             SETTLED_ONLY, kinds=("role",)),
    # FORM (repair round): a column's declared form is a consumer of whether its numbers are codes
    # (indicators, no form) or amounts (a form): the form question's card waits for the reading
    # and asks it, and a declaration on an unsettled or coded column is refused.
    Consumer(_C + "methods.exposure_form:form_plan", ("predictor_codes",), True, ASK,
             kinds=("code_or_count",)),
    Consumer(_C + "methods.exposure_form:_form_reads_a_settled_reading", ("predictor_codes",),
             True, ASK, kinds=("code_or_count",)),
    Consumer(_C + "stages.modeling:shelf_stage", ("roles",), True, SETTLED_ONLY),
    Consumer(_C + "methods.missing:energy_fill", ("energy_fill",), True, SETTLED_ONLY,
             via=_C + "models.pipeline:design_spec"),
    Consumer(_C + "methods.omics:design_normalization", ("roles", "assay_scale"), True,
             SETTLED_ONLY),
    # BLUEPRINT §14.3: the imputation model reads the codes the design settled (``spec.categorical``).
    Consumer(_C + "methods.missing:imputation_frame", ("predictor_codes",), True, SETTLED_ONLY,
             via=_C + "stages.modeling:fit_stage", kinds=("code_or_count",)),
    # MI repair (ruling 12): the clustered imputation carries and imputes once per unit only the
    # columns whose time-invariance the user confirmed; a column whose records agree within every
    # unit and is not yet confirmed either way is asked (``_missing_for_table`` holds the table).
    Consumer(_C + "methods.missing:time_invariant_columns", (), True, ASK,
             kinds=("time_invariant",)),
    Consumer(_C + "decisions:_roles_record_what_rode_along", ("roles",), True, ASK),
    Consumer(_C + "decisions:_answers_keep_settled_roles", ("roles",), True, ASK),
    Consumer(_C + "decisions:_answers_keep_settled_readings",
             ("body_columns", "time_column", "predictor_codes"), True, ASK),
    Consumer(_C + "decisions:_energy_reads_settled_roles", ("roles", "nutrients"), True, ASK),
    Consumer(_C + "decisions:_screens_read_a_settled_energy_column",
             ("roles", "energy_column"), True, ASK),
    Consumer(_C + "stages.proposals:proposals_stage",
             ("roles", "nutrients", "energy_column", "sex_column", "energy_unit_days"), True,
             SETTLED_ONLY, kinds=("role",)),
    Consumer(_C + "stages.proposals:exclusion_proposals", ("sex_column", "energy_unit_days"),
             True, ASK, via=_C + "stages.proposals:proposals_stage",
             kinds=("unit:energy", "day_count")),
    Consumer(_C + "stages.proposals:nutrient_candidates", ("nutrients",), True, SETTLED_ONLY,
             via=_C + "stages.proposals:proposals_stage"),
    Consumer(_C + "stages.modeling:_measurement_error_line", ("roles", "total_energy_names"),
             True, SETTLED_ONLY),
    # MS8: the scales answer reads its items' roles and codes through the ledger (the scales stage
    # reads its covariates off the design, ``design_spec``, which reads settled roles only).
    Consumer(_C + "scales:_scale_items_are_settled_predictors", ("roles",), True, ASK),
    Consumer(_C + "scales:_answers_fit_the_response_scale", ("predictor_codes",), True, ASK),
    Consumer(_C + "models.inference:resolve_clusters", ("identifier", "roles"), True, ASK,
             kinds=("cluster", "role")),
    Consumer(_C + "stages.modeling:fit_stage", ("identifier",), True, ASK,
             via=_C + "models.inference:resolve_clusters"),
    Consumer(_C + "methods.survey:for_fit", ("identifier", "survey_design"), True, ASK),
    Consumer(_C + "seal:seal_inputs", ("identifier", "roles", "grain_stated"), True,
             SETTLED_ONLY,
             kinds=("cluster",)),
    Consumer(_C + "stages.rows:split_inputs", ("split_inputs",), True, SETTLED_ONLY,
             kinds=("cluster",)),
    Consumer(_C + "stages.rows:value_facts", ("identifier", "time_role"), False, WORDS_ONLY),
    # ── the grain, stated by values; its suggestions offered ──
    Consumer(_C + "stages.working:stated_grain", ("grain_stated",), True, VALUES_SETTLE),
    Consumer(_C + "sequence:_grain_is_consistent", ("grain_suggestion",), False, USER_APPLIED),
    # ── the repeat kind and the time column ──
    Consumer(_C + "stages.working:effective_repeat_kind", ("repeat_kind",), True, SETTLED_ONLY),
    Consumer(_C + "interview:_repeat_kind_gate", ("repeat_kind",), True, ASK),
    Consumer(_C + "interview:_temporal_gate", ("repeat_kind",), True, ASK,
             via=_C + "stages.working:effective_repeat_kind"),
    Consumer(_C + "coach:aggregation_coach", ("repeat_kind",), False, WORDS_ONLY),
    Consumer(_C + "stages.calibration:calibration_stage", ("repeat_kind",), True, SETTLED_ONLY,
             via=_C + "stages.working:effective_repeat_kind"),
    # the NCI usual-intake method: recalls only where the repeats are settled as repeats, ordered
    # only by a settled time column or the user's own; an EAR refused only for a settled energy role
    Consumer(_C + "stages.usual_intake:gate", ("repeat_kind",), True, SETTLED_ONLY,
             via=_C + "stages.working:effective_repeat_kind"),
    Consumer(_C + "stages.usual_intake:recall_order", ("time_column",), True, SETTLED_ONLY,
             via=_C + "stages.working:time_column", kinds=("time_column",)),
    Consumer(_C + "usual_intake:_cutoff_fits_the_reference", ("roles",), True, SETTLED_ONLY),
    Consumer(_C + "stages.proposals:recall_days", ("recall_days", "repeat_kind"), True,
             SETTLED_ONLY),
    Consumer(_C + "stages.working:time_column", ("time_column",), True, SETTLED_ONLY,
             kinds=("time_column",)),
    Consumer(_C + "stages.working:aggregation_plan", ("time_column", "combine_codes"), True,
             ASK),
    Consumer(_C + "sequence:_aggregation_reads_settled_readings",
             ("time_column", "combine_codes"), True, ASK),
    Consumer(_C + "sequence:_temporal_needs_time_points_as_rows", ("time_column",), True, ASK,
             via=_C + "stages.working:time_column"),
    Consumer(_C + "sequence:_temporal_names_its_time_column", ("time_column",), True,
             SETTLED_ONLY, via=_C + "stages.working:time_column"),
    Consumer(_C + "voice:_time_order", ("time_column",), True, SETTLED_ONLY),
    Consumer(_C + "voice:_set_aggregation", ("time_column", "combine_codes"), True,
             SETTLED_ONLY),
    Consumer(_C + "stages.working:prepare_combine", ("combine_codes", "time_column"), True, ASK),
    # ── the outcome: its kind, its event, its unit ──
    Consumer(_C + "stages.target:target_info_stage", ("task", "outcome_unit"), True, ASK),
    Consumer(_C + "sequence:_event_is_a_level_of_the_outcome", ("event_level",), False,
             USER_APPLIED),
    Consumer(_C + "units:outcome_unit", ("outcome_unit",), True, NO_UNIT,
             kinds=("outcome_unit",)),
    Consumer(_C + "voice:_outcome_unit", ("outcome_unit",), True, NO_UNIT,
             via=_C + "units:outcome_unit",
             kinds=("outcome_unit",)),
    # Wave 2, EXPLAIN: the fitted equation states the outcome's unit and each column's unit only
    # as recorded or confirmed; until then it quotes the header verbatim.
    Consumer(_C + "models.explain:equation_units", ("outcome_unit",), True, NO_UNIT,
             kinds=("outcome_unit", "unit:column")),
    # ── total energy, its unit, its day count, the body measures and sex the screens read ──
    Consumer(_C + "stages.proposals:energy_unit_reading", ("energy_unit_days",), True, ASK,
             kinds=("unit:energy", "day_count")),
    Consumer(_C + "decisions:_screens_wait_for_the_unit", ("energy_unit_days",), True, ASK,
             via=_C + "stages.proposals:energy_unit_reading", kinds=("unit:energy", "day_count")),
    Consumer(_C + "stages.finding_words:restate_implausible",
             ("energy_unit_days", "energy_column"), True, SETTLED_ONLY,
             kinds=("unit:energy", "day_count")),
    Consumer(_C + "stages.finding_words:FindingContext.energy_settled", ("energy_column",), True,
             VALUES_SETTLE),
    Consumer(_C + "stages.finding_words:restate_energy", ("energy_column",), True, SETTLED_ONLY),
    Consumer(_C + "coach:card_lines", ("energy_column", "energy_unit_days"), True, NO_UNIT,
             via=_C + "stages.proposals:build_proposals"),
    Consumer(_C + "stages.proposals:goldberg_proposal", ("body_columns", "sex_column"), True,
             ASK,
             kinds=("unit:weight", "unit:height", "unit:age", "unit:energy", "day_count")),
    Consumer(_C + "stages.proposals:body_refusal", ("body_columns", "sex_column"), True, ASK),
    Consumer(_C + "stages.proposals:sex_column", ("sex_column",), True, SETTLED_ONLY,
             kinds=("sex_coding",)),
    Consumer(_C + "detectors.plausibility:sex_codes", ("plausibility",), True, SETTLED_ONLY,
             kinds=("sex_coding",)),
    Consumer(_C + "decisions:_screens_read_settled_body_measures",
             ("body_columns", "sex_column"), True, ASK),
    Consumer(_C + "coach:range_notes", ("coach_names", "energy_column"), True, NO_UNIT),
    Consumer(_C + "coach:_unit_suffix", ("coach_names",), True, NO_UNIT,
             kinds=("unit:energy",)),
    Consumer(_C + "coach:exclusions_coach", ("coach_names",), False, WORDS_ONLY),
    # BLUEPRINT §14.2 (audit WP18): the one ask card, on the question whose consumer reads these
    # readings. It lists the unsettled ones its consumer's refusal would ask, with that refusal's
    # ways forward, and the ones the values settled ("read from your data"); it only asks.
    Consumer(_C + "ask:card", ("roles", "predictor_codes", "combine_codes", "energy_column",
                               "energy_unit_days", "survey_design"), False, WORDS_ONLY),
    # ── the survey design ──
    Consumer(_C + "survey:reading_of", ("survey_design", "design_role"), True, SETTLED_ONLY),
    Consumer(_C + "survey:not_applicable_reason", ("survey_design",), True, ASK,
             via=_C + "survey:reading_of"),
    Consumer(_C + "survey:offered", ("survey_design", "design_role", "survey_weight_rank",
                                     "survey_cycle"), True, SETTLED_ONLY),
    Consumer(_C + "survey:_design_is_settled", ("survey_design", "design_role"), True, ASK),
    Consumer(_C + "methods.survey:proposal", ("survey_design", "survey_cycle"), True,
             SETTLED_ONLY, via=_C + "survey:reading_of"),
    # ── energy sources: factors, shares of energy, parts of totals ──
    Consumer(_C + "decisions:_substitution_reads_settled_readings",
             ("energy_factor", "percent_energy", "nesting", "substitution_validators"), True,
             ASK, kinds=("unit:factor", "nested_in")),
    Consumer(_C + "stages.modeling:substitution_stage", ("energy_factor",), True, ASK,
             kinds=("unit:factor",)),
    # MS1: the imputation's energy identity computes with each source's kcal per unit, settled only.
    Consumer(_C + "stages.modeling:_settled_factors", ("energy_factor",), True, SETTLED_ONLY,
             kinds=("unit:factor",)),
    # The routing gate's ledger residue: the partition methods split each source into kcal by its
    # settled kcal per unit, asked at the energy question and read by the design, never the name.
    Consumer(_C + "decisions:_energy_adjustment_fits_the_roles", ("energy_factor",), True, ASK,
             kinds=("unit:factor",)),
    Consumer(_C + "models.pipeline:_energy_factors", ("energy_factor",), True, ASK,
             kinds=("unit:factor",)),
    # The sixth gate: the curve's Shift and the design read the nesting the user confirmed,
    # whichever column it names (``readings.nesting``).
    Consumer(_C + "stages.modeling:shift_for", ("nesting",), True, SETTLED_ONLY,
             kinds=("nested_in",)),
    Consumer(_C + "stages.modeling:design_nesting", ("nesting",), True, SETTLED_ONLY,
             kinds=("nested_in",)),
    Consumer(_C + "decisions:_substitution_has_every_energy_source",
             ("substitution_validators", "energy_factor"), False, ASK),
    Consumer(_C + "methods.energy:total_energy_columns", ("total_energy_names",), True,
             SETTLED_ONLY, via=_C + "stages.modeling:design_stage"),
    # ── questions gated on a reading ──
    Consumer(_C + "interview:_energy_applicability", ("energy_applicability",), True, ASK),
    Consumer(_C + "interview:_orientation_gate", ("orientation",), True, ASK),
    # ── missing values and their codes ──
    Consumer(_C + "stages.proposals:missing_reading", ("missing_not_asked",), False,
             USER_APPLIED),
    Consumer(_C + "decisions:_non_detections_are_not_filled_by_the_median",
             ("below_detection",), True, USER_APPLIED),
    Consumer(_C + "methods.imputation:column_kind", ("imputation_kind",), True, VALUES_SETTLE),
    Consumer(_C + "models.pipeline:level_columns", ("missing_level",), True, VALUES_SETTLE),
    Consumer(_C + "repairs:_offer_impossible", ("plausibility",), True, USER_APPLIED),
    Consumer(_C + "repairs:_offer_sentinels", ("sentinels",), True, USER_APPLIED),
    Consumer(_C + "repairs:column_expressions", ("repairs", "ingest_values"), True,
             USER_APPLIED),
    Consumer(_C + "stages.working:read_date_columns", ("date_format",), True, VALUES_SETTLE),
    # ── acquisition: the injection order and batch QC-RLSC fits along (MS7 repair) ──
    # The offer reads them as guesses (``injection_order_reading``, ``batch_readings``) and names
    # each in its options, one option per batch reading; the working table runs only the option
    # the user chose, with the columns it named.
    Consumer(_C + "methods.qc_drift:rlsc_offer", ("acquisition",), False, WORDS_ONLY),
    Consumer(_C + "methods.qc_drift:reference_plan", ("acquisition",), True, USER_APPLIED),
    # ── sentences, hints and findings only ──
    Consumer(_C + "methods.dietary_caveats:energy_related", ("energy_related_outcome",), False,
             WORDS_ONLY),
    Consumer(_C + "detectors.lenses:hints", ("lens_hints",), False, WORDS_ONLY),
    Consumer(_C + "detectors.scales:findings", ("scales",), False, WORDS_ONLY),
    Consumer(_C + "detectors.genomics:card", ("assay_type",), False, WORDS_ONLY),
    Consumer(_C + "stages.working:unit_suggestions", ("measurement", "grain_suggestion"), False,
             WORDS_ONLY),
    # ── V2 causal row: a time-varying exposure by g-methods (turbotab/core/time_varying.py) ──
    # The lane reads each unit's history by the settled time column, and its models read each
    # covariate's role and code-or-amount reading as the fit does; while one waits, it asks.
    Consumer(_C + "stages.time_varying:read_setting", (), True, SETTLED_ONLY),
    Consumer(_C + "time_varying:_lane_needs_a_settled_time_column", (), True, ASK),
    Consumer(_C + "time_varying:_lane_reads_settled_readings", (), True, ASK),
)


# Calls to a kind's helpers (:attr:`KindRule.helpers`) outside this module that decide no reading:
# ``(caller, helper)`` -> what the call reads and why no number follows from it. The structural
# test (BLUEPRINT §14.3, the sixth gate: one test per kind, no private settler) fails on any other
# call, and on any entry whose caller is a number-changing consumer.
EVIDENCE_ONLY: dict[tuple[str, str], str] = {
    ("turbotab.core.stages.rows:value_facts", "turbotab.core.recognizers:flag_values"):
        "the flag role's guess and its evidence (no value test settles a flag: a skip-pattern "
        "gate marks its follow-up's blanks the same way); value_facts changes no number",
}


__all__ = [
    "ATTENTION", "ASK", "CODE_LEVELS", "CONFIRMABLE", "CONSUMERS", "Consumer", "KINDS", "NO_UNIT",
    "PREDICTOR_ROLES", "Reading", "SETTLED_ONLY", "USER_APPLIED", "UNIT_VALUES", "Unsettled",
    "VALUES", "VALUES_SETTLE", "WORDS_ONLY", "attention_columns", "body_unit_reading",
    "cluster_reading",
    "code_or_count_exits", "code_or_count_reading", "confirm_exit", "confirm_exits",
    "confirmation", "confirmed_codes", "day_count_candidates", "day_count_reading",
    "is_settled", "key", "listing", "names_several_days",
    "outcome_unit_reading", "predictor_columns", "predictors_or_ask", "proposals_of", "reading",
    "repeat_kind_reading", "rode_along", "stated_reading", "task_reading", "role_reading", "role_words", "settled",
    "settled_columns", "settled_roles", "stated_outcome_unit", "time_column_reading",
    "unsettled", "unsettled_codes", "unsettled_factors", "unsettled_message", "unsettled_roles",
    "KcalPerUnit", "ROLE_ALTERNATIVES", "energy_plan", "factor_exits", "factor_in_grams",
    "fractional_verdict", "grid_reading", "kcal_per_unit", "outcome_task_by_values",
    "read_from_data", "read_from_values_sentence", "recorded_energy", "unit_day_candidates",
    "unit_record", "COLUMN_VALUED", "DRINK_GRAMS", "EVIDENCE_ONLY", "MINOR_SHARE", "NOT_NESTED",
    "TEXT_NUMBER_SHARE", "by_values", "by_values_table", "comma_exits", "detection_exits",
    "detection_limit", "drink_units", "drinks_value", "factor_verdicts", "nested_exits", "nesting",
    "nesting_parents", "nutrients_by_values", "parse_drinks", "text_numbers", "unit_words",
    "unit_record", "CODEBOOK_EVIDENCE", "codebook_label", "codebook_source", "codebook_unit",
    "label_guess", "labeled", "recorded_evidence", "BATCH_KINDS", "batch_readings",
    "injection_order_reading", "TIME_INVARIANT_WORDS", "time_invariance", "time_invariance_exit",
]
