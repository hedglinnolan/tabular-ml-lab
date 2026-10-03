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

The structural acceptance test enumerates :data:`CONSUMERS` against the independent census of
readers and consumers, and checks that each number-changing consumer calls into this module.
"""
from __future__ import annotations

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
}
# The kinds ``confirm_reading`` records (the others are answered by their own decisions).
CONFIRMABLE = ("role", "cluster", "unit", "day_count", "code_or_count", "time_column",
               "nested_in")
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
            return Reading((str(column),), kind, recorded, "high", "recorded by the user",
                           corroborated, "confirmed")
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
    own = _confirmations(state).get(key(kind, column))
    if own is not None:
        return own
    if kind == "role":
        legacy = (getattr(state, "role_confirmations", None) or {}).get(column)  # confirm_role
        return legacy
    if kind in ("unit", "day_count"):
        spec = (_get(state, "column_units") or {}).get(column)
        if spec is None:
            return None
        return _get(spec, "unit") if kind == "unit" else int(_get(spec, "days") or 1)
    if kind == "outcome_unit":
        unit = getattr(state, "outcome_unit", None)
        return unit if unit and getattr(state, "target", None) == column else None
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


class Unsettled(ValueError):
    """A number-changing consumer met a reading that is not settled: it asks, never guesses.
    ``exits`` are one confirmation per reading (never one for all)."""

    def __init__(self, message: str, readings: Sequence[Reading] = (),
                 exits: Sequence[Mapping[str, Any]] = ()):
        super().__init__(message)
        self.readings = list(readings)
        self.exits = [dict(e) for e in exits]


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


def role_reading(state: Any, column: str) -> Reading | None:
    """``column``'s role as the ledger holds it: settled when the answer recorded it as the user's
    own (changed from the proposal, or proposed high from its values) or a confirmation since
    recorded the same role; proposed while it rode along unconfirmed. None with no role."""
    roles = getattr(state, "roles", None) or {}
    if column not in roles:
        return None
    role = roles[column]
    waiting = column in set(getattr(state, "roles_unconfirmed", None) or [])
    if not waiting or confirmation(state, "role", column) == role:
        return Reading((column,), "role", role, "high", "recorded", True, "confirmed")
    return Reading((column,), "role", role, "medium",
                   "proposed below high confidence and recorded with the other roles", False,
                   "proposed")


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


def confirm_exits(state: Any, columns: Sequence[str]) -> list[dict[str, Any]]:
    """One exit per unsettled role: its own confirmation, never one for all of them."""
    roles = getattr(state, "roles", None) or {}
    return [confirm_exit("role", c, roles[c], f"Confirm `{c}` as {role_words(roles.get(c))}")
            for c in columns if c in roles]


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


def code_or_count_reading(state: Any, column: str, *, dtype: str | None, n_unique: int | None,
                          fewest: int = 3) -> Reading | None:
    """Whether ``column``'s numbers are codes for categories or amounts, for a fit or for
    combining a unit's rows. Values settle it only where they leave no doubt: text and booleans
    are categories, and many distinct values (or a constant, or two values: one indicator is its
    own slope) are amounts. Whole numbers with ``fewest``–:data:`CODE_LEVELS` values may be either
    (smoking 1/2/3 or cups of coffee 0–6), so the user says which; None when it is no question."""
    if dtype not in ("integer",):
        return None
    k = int(n_unique or 0)
    if k < fewest or k > CODE_LEVELS:
        return None
    return reading("code_or_count", column, "amount", confidence="medium",
                   evidence=f"`{k}` whole-number values: codes for categories, or a count",
                   corroborated=False, state=state)


def code_or_count_exits(column: str) -> list[dict[str, Any]]:
    return [confirm_exit("code_or_count", column, "amount",
                         f"`{column}` is a count or an amount (a slope; a unit's rows averaged)"),
            confirm_exit("code_or_count", column, "code",
                         f"`{column}` holds codes for categories (one indicator per level; a "
                         f"unit's rows take their most frequent value)")]


ASSAY_LENSES = ("metabolomics", "genomics")


def unsettled_codes(state: Any, columns: Iterable[str],
                    info: Mapping[str, Mapping[str, Any]] | None) -> list[str]:
    """The predictors whose code-or-amount reading the fit may not read yet. Under an assay lens
    (the user's answer that the table is an assay), an exposure is a measured feature (a count or
    an intensity), an amount by that answer; its other columns are read as any table's."""
    assay = any(k in ASSAY_LENSES for k in (_get(state, "lens") or []))
    roles = settled_roles(state) if assay else {}
    out = []
    for c in columns:
        if assay and roles.get(c) == "exposure":
            continue
        entry = (info or {}).get(c) or {}
        dtype = entry.get("dtype") if isinstance(entry, Mapping) else getattr(entry, "dtype", None)
        n_unique = (entry.get("n_unique") if isinstance(entry, Mapping)
                    else getattr(entry, "n_unique", None))
        r = code_or_count_reading(state, c, dtype=dtype, n_unique=n_unique)
        if r is not None and not r.settled:
            out.append(c)
    return out


def predictors_or_ask(state: Any, info: Mapping[str, Mapping[str, Any]] | None = None,
                      order: Sequence[str] | None = None,
                      drop: Iterable[str] = ()) -> list[str]:
    """The fit's predictor set, once every reading it rests on is settled; else :class:`Unsettled`
    with one exit per reading: every recorded role (a role that rode along decides whether its
    column is in the model or out), and each whole-number predictor's code-or-amount reading."""
    waiting = unsettled(state)
    if waiting:
        raise Unsettled(
            unsettled_message(waiting, "the model's predictors") + " The fit waits for them: a "
            "column whose role nobody confirmed would enter or leave the model on a guess.",
            [role_reading(state, c) for c in waiting], confirm_exits(state, waiting))
    preds = predictor_columns(state, order, drop)
    codes = unsettled_codes(state, preds, info)
    if codes:
        exits = [e for c in codes for e in code_or_count_exits(c)]
        raise Unsettled(
            f"{listing(codes)} {'holds' if len(codes) == 1 else 'hold'} a few whole-number values, "
            f"which may be codes for categories (one indicator per level) or amounts (one slope); "
            f"the fit waits for the answer for each.",
            [], exits)
    return preds


# ── readings a stage makes once and states with its confidence ───────────────


def stated_reading(kind: str, column: str, value: Any, confidence: Any,
                   evidence: str = "") -> Reading:
    """A reading a stage made and stated with its confidence (the orientation of the table, the
    outcome's task, total energy by its values): settled only when the reader made it high from
    the values, never from a name or a shape's grammar alone."""
    level = str(confidence or "low")
    return Reading((str(column),), kind, value, level, evidence, level == "high", "proposed")


def task_reading(column: str, detected: str, confidence: Any, evidence: str = "") -> Reading:
    """The outcome's task as detected (``ml.triage``): two levels or a many-valued number settle
    it by its values; three or more labels never do, since no dtype says whether they are ordered
    (an ordinal outcome) or not, so the task is asked."""
    level = str(confidence or "low")
    if detected == "multiclass" and level == "high":
        level = "medium"
    return stated_reading("task", column, detected, level, evidence)


# ── the cluster reading: which rows belong together ──────────────────────────


def cluster_reading(state: Any, column: str) -> Reading:
    """Whether ``column``'s repeating values mark rows the intervals must keep together. Settled
    only by the user: the grain's named unit, its own ``confirm_reading``, or a settled identifier
    or cluster role. A repeating ``*_id`` read by name (a stratum, a PSU, an interviewer) is no
    unit until then (BLUEPRINT §14.1, the gate's ``stratum_id``)."""
    recorded = confirmation(state, "cluster", column)
    if recorded is not None:
        return Reading((column,), "cluster", recorded, "high", "recorded by the user", True,
                       "confirmed")
    role = role_reading(state, column)
    if role is not None and role.settled and role.value in ("identifier", "cluster"):
        return Reading((column,), "cluster", "yes", "high", f"its {role.value} role is settled",
                       True, "confirmed")
    return Reading((column,), "cluster", "yes", "medium",
                   "its role was proposed below high confidence", False, "proposed")


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
    out = list(getattr(state, "categorical", None) or [])
    agg = getattr(state, "aggregation", None)
    out += [c for c, r in (getattr(agg, "columns", None) or {}).items() if r == "mode"]
    out += [name.split(":", 1)[1] for name, value in _confirmations(state).items()
            if name.startswith("code_or_count:") and value == "code"]
    return list(dict.fromkeys(out))


# ── units ─────────────────────────────────────────────────────────────────────

# Bare amounts: a mass that says neither per what nor of what. On a quantity the clinical pack
# reads as a concentration or an index (``ldl_mg`` is mg/dL; ``bmi_kg`` is kg/m²), it is part of a
# unit, not one; elsewhere it is still only the name's (the gate: ``hb_g`` stated "in g").
BARE_AMOUNTS = ("mg", "g", "kg", "µg", "lb")


def outcome_unit_reading(column: str, recorded: str | None = None) -> Reading | None:
    """The outcome's unit (audit IN-05; BLUEPRINT §14.1). Settled when recorded
    (``set_outcome_unit``), or when the name spells out a whole unit the quantity can carry: a
    unit with its denominator (``glucose_mg_dl``), a unit of its own (``sbp_mmhg``,
    ``energy_kcal``, ``age_years``), or a bare amount the clinical pack lists for that quantity
    (``weight_kg``). A bare amount anywhere else is the name's proposal: no sentence carries it."""
    from turbotab.core.units import from_name, proposed_unit

    if recorded:
        return Reading((column,), "outcome_unit", str(recorded), "high", "recorded by the user",
                       True, "confirmed")
    unit = from_name(column)
    if unit is None:
        return None
    if unit not in BARE_AMOUNTS:
        return Reading((column,), "outcome_unit", unit, "high", "the name spells out a whole unit",
                       True, "proposed")
    pack = proposed_unit(column)
    candidates = list((pack or {}).get("candidates") or [])
    if unit in candidates:
        return Reading((column,), "outcome_unit", unit, "high",
                       f"the name's {unit} is one of the clinical pack's units for it", True,
                       "proposed")
    why = (f"the name ends in {unit}, but the clinical pack reads this quantity in "
           f"{' or '.join(candidates)}" if candidates else
           f"the name ends in {unit}, a bare amount that says neither per what nor of what")
    return Reading((column,), "outcome_unit", unit, "medium", why, False, "proposed")


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
    """The unit of a body measure the Goldberg screen reads (weight in kg, height in cm or m, age
    in years). Settled when the name spells the unit (``weight_kg``, ``BMXWT``, ``age_years``),
    the user recorded it, or (height only) its median sits in the one unit's human band; a weight's
    or an age's median is a magnitude, which tells kg from lb, or years from a child's months, no
    better than a heavy cohort from a light one, so it only proposes (the gate: a US cohort's
    ``weight`` in pounds, median 154, read as kg)."""
    import numpy as np
    import pandas as pd

    from turbotab.core.recognizers import tokens

    recorded = confirmation(state, "unit", column)
    if recorded is not None:
        return Reading((column,), "unit", str(recorded), "high", "recorded by the user", True,
                       "confirmed")
    words = tokens(column)
    joined = "".join(words)
    for unit, spelled in _BODY_SPELLED.get(measure, {}).items():
        if (words and words[-1] in spelled) or joined in spelled:
            return Reading((column,), "unit", unit, "high", "the name spells out the unit", True,
                           "proposed")
    if measure == "height" and values is not None:
        x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x) & (x > 0)]
        median = float(np.median(x)) if len(x) else None
        for unit, (lo, hi) in HEIGHT_BANDS.items():
            if median is not None and lo <= median <= hi:
                return Reading((column,), "unit", unit, "high",
                               f"its median, {median:,.1f}, is a human height only in {unit}",
                               True, "proposed")
    default = BODY_UNITS[measure][0]
    return Reading((column,), "unit", default, "medium",
                   f"only its median says {default}", False, "proposed")


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
    """How many days each value of a total-energy column spans. Recorded (``set_column_unit``),
    or one day corroborated by the values: a median inside the field's one-day band for the unit
    (NUTRITION_PACK §01's prior: 1,600–2,600 kcal, 7,000–11,000 kJ) and a name that says nothing
    of several days. The Atwater identity settles kcal against kJ and says nothing about days
    (the gate: 2-day totals beside 2-day macronutrients passed it and were read as one day)."""
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
        return Reading((column,), "day_count", 1, "high",
                       f"its median, {median:,.0f}, is a day's intake", True, "proposed")
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


def unsettled_factors(state: Any, columns: Iterable[str], store: Any = None) -> list[str]:
    """The columns whose kcal per unit a substitution or a partition may not read yet: an Atwater
    factor is per gram, so the column must be in grams by its name or codebook (``protein_g``,
    ``DR1TPROT``), by the values (the Atwater identity agreeing with total energy), or by the
    user's confirmation of its unit. An unmarked ``Protein`` beside an InBody export is kilograms
    of body protein (BLUEPRINT §14)."""
    from turbotab.core.methods.energy import energy_factor

    waiting = []
    for c in columns:
        if confirmation(state, "unit", c) is not None:
            continue
        found = energy_factor(c)
        if found.factor is not None and not found.declared:
            waiting.append(c)
    if not waiting or store is None:
        return waiting
    energy = next((c for c, r in settled_roles(state).items() if r == "energy"), None)
    if energy is None or energy not in set(getattr(store, "columns", ()) or ()):
        return waiting
    from turbotab.core.decisions import _names_a_macro_total
    from turbotab.core.methods.energy import atwater_check
    from turbotab.core.recognizers import macro_totals

    names = [c for c in store.columns if c != energy and _names_a_macro_total(c)]
    frame = store.materialize([energy, *names])
    macros = macro_totals(frame, exclude=[energy])
    try:
        check = atwater_check(frame[[energy, *macros.values()]], energy)
    except Exception:  # noqa: BLE001 - a check that cannot run settles nothing
        check = None
    if check is not None and check.verdict == "pass":
        return [c for c in waiting if c not in set(macros.values())]
    return waiting


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
             SETTLED_ONLY),
    Consumer(_C + "models.pipeline:design_spec", ("roles", "predictor_codes", "energy_fill"),
             True, SETTLED_ONLY),
    Consumer(_C + "stages.modeling:design_stage",
             ("roles", "predictor_codes", "design_role", "acquisition", "flag", "time_role",
              "covariates", "nesting", "free_text", "total_energy_names"), True, ASK),
    Consumer(_C + "decisions:_models_read_settled_readings",
             ("roles", "predictor_codes", "design_role", "acquisition", "flag", "time_role",
              "covariates"), True, ASK),
    Consumer(_C + "stages.rows:cohort_inputs", ("roles", "missing_not_asked"), True,
             SETTLED_ONLY),
    Consumer(_C + "stages.modeling:shelf_stage", ("roles",), True, SETTLED_ONLY),
    Consumer(_C + "methods.missing:energy_fill", ("energy_fill",), True, SETTLED_ONLY,
             via=_C + "models.pipeline:design_spec"),
    Consumer(_C + "methods.omics:design_normalization", ("roles", "assay_scale"), True,
             SETTLED_ONLY),
    Consumer(_C + "decisions:_roles_record_what_rode_along", ("roles",), True, ASK),
    Consumer(_C + "decisions:_answers_keep_settled_roles", ("roles",), True, ASK),
    Consumer(_C + "decisions:_answers_keep_settled_readings",
             ("body_columns", "time_column", "predictor_codes"), True, ASK),
    Consumer(_C + "decisions:_energy_reads_settled_roles", ("roles", "nutrients"), True, ASK),
    Consumer(_C + "decisions:_screens_read_a_settled_energy_column",
             ("roles", "energy_column"), True, ASK),
    Consumer(_C + "stages.proposals:proposals_stage",
             ("roles", "nutrients", "energy_column", "sex_column", "energy_unit_days"), True,
             SETTLED_ONLY),
    Consumer(_C + "stages.proposals:exclusion_proposals", ("sex_column", "energy_unit_days"),
             True, ASK, via=_C + "stages.proposals:proposals_stage"),
    Consumer(_C + "stages.proposals:nutrient_candidates", ("nutrients",), True, SETTLED_ONLY,
             via=_C + "stages.proposals:proposals_stage"),
    Consumer(_C + "stages.modeling:_measurement_error_line", ("roles", "total_energy_names"),
             True, SETTLED_ONLY),
    Consumer(_C + "models.inference:resolve_clusters", ("identifier", "roles"), True, ASK),
    Consumer(_C + "stages.modeling:fit_stage", ("identifier",), True, ASK,
             via=_C + "models.inference:resolve_clusters"),
    Consumer(_C + "methods.survey:for_fit", ("identifier", "survey_design"), True, ASK),
    Consumer(_C + "seal:seal_inputs", ("identifier", "roles", "grain_stated"), True,
             SETTLED_ONLY),
    Consumer(_C + "stages.rows:split_inputs", ("split_inputs",), True, SETTLED_ONLY),
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
    Consumer(_C + "stages.proposals:recall_days", ("recall_days", "repeat_kind"), True,
             SETTLED_ONLY),
    Consumer(_C + "stages.working:time_column", ("time_column",), True, SETTLED_ONLY),
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
    Consumer(_C + "units:outcome_unit", ("outcome_unit",), True, NO_UNIT),
    Consumer(_C + "voice:_outcome_unit", ("outcome_unit",), True, NO_UNIT,
             via=_C + "units:outcome_unit"),
    # ── total energy, its unit, its day count, the body measures and sex the screens read ──
    Consumer(_C + "stages.proposals:energy_unit_reading", ("energy_unit_days",), True, ASK),
    Consumer(_C + "decisions:_screens_wait_for_the_unit", ("energy_unit_days",), True, ASK,
             via=_C + "stages.proposals:energy_unit_reading"),
    Consumer(_C + "stages.finding_words:restate_implausible",
             ("energy_unit_days", "energy_column"), True, SETTLED_ONLY),
    Consumer(_C + "stages.finding_words:FindingContext.energy_settled", ("energy_column",), True,
             VALUES_SETTLE),
    Consumer(_C + "stages.finding_words:restate_energy", ("energy_column",), True, SETTLED_ONLY),
    Consumer(_C + "coach:card_lines", ("energy_column", "energy_unit_days"), True, NO_UNIT,
             via=_C + "stages.proposals:build_proposals"),
    Consumer(_C + "stages.proposals:goldberg_proposal", ("body_columns", "sex_column"), True,
             ASK),
    Consumer(_C + "stages.proposals:body_refusal", ("body_columns", "sex_column"), True, ASK),
    Consumer(_C + "decisions:_screens_read_settled_body_measures",
             ("body_columns", "sex_column"), True, ASK),
    Consumer(_C + "coach:range_notes", ("coach_names", "energy_column"), True, NO_UNIT),
    Consumer(_C + "coach:_unit_suffix", ("coach_names",), True, NO_UNIT),
    Consumer(_C + "coach:exclusions_coach", ("coach_names",), False, WORDS_ONLY),
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
             ASK),
    Consumer(_C + "stages.modeling:substitution_stage", ("energy_factor",), True, ASK),
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
    # ── sentences, hints and findings only ──
    Consumer(_C + "methods.dietary_caveats:energy_related", ("energy_related_outcome",), False,
             WORDS_ONLY),
    Consumer(_C + "detectors.lenses:hints", ("lens_hints",), False, WORDS_ONLY),
    Consumer(_C + "detectors.scales:findings", ("scales",), False, WORDS_ONLY),
    Consumer(_C + "detectors.genomics:card", ("assay_type",), False, WORDS_ONLY),
    Consumer(_C + "stages.working:unit_suggestions", ("measurement", "grain_suggestion"), False,
             WORDS_ONLY),
)


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
]
