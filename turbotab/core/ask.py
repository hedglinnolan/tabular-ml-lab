"""The one ask card, placed where the first consumer needs it (BLUEPRINT §14.2; audit WP18).

The readings ledger (:mod:`turbotab.core.readings`) knows which readings feed a number-changing
consumer, and each consumer asks for its unsettled ones by refusing an answer, with one
confirmation per reading. That put the question after the user had already answered: the card
arrived as a refusal. §14.2 asks for the opposite: *one* card, "Tell me about these columns",
holding each field's best guess with its evidence, ordered by how much the field changes, a
homogeneous family confirmed as one block, and shown **where the first consumer needs it, not as a
wall at upload**.

So the Router (:func:`turbotab.core.interview.route`) asks this module for the card of the question
it holds open, and nothing else:

* :data:`CONSUMERS` maps a question to the consumer its answer feeds and the readings that
  consumer reads: combining a unit's rows (codes or counts), the screens (total energy's role,
  unit and days), the energy adjustment (the roles of total energy and the nutrients), the survey
  design (its roles) and the fit (every predictor's role, and each whole-valued predictor's codes
  or amounts). A question whose answer reads no reading has no card: the lens, the outcome, the
  purpose, and the questions whose own answer *is* the reading (roles, grain, repeats).
* Each reading comes from the ledger's own functions, the same ones the consumer's refusal calls,
  so the card asks exactly what the refusal would. Only unsettled readings are listed; a reading
  the values settled is not asked (the ledger's "read from your data" section shows those).
* The card is :func:`turbotab.core.readings.ordered` and :func:`~turbotab.core.readings.families`,
  worded by :func:`~turbotab.core.readings.ask_text`, and answered by
  :func:`~turbotab.core.readings.ask_exits`: one block ``confirm_readings`` that settles exactly
  the readings it lists, then each reading's alternatives. Where the consumer's refusal names its
  own ways forward (the fit's values below a detection limit, its commas that read two ways), the
  card offers those, never a confirmation the reading cannot take. An energy column's unit and days
  are recorded by ``set_column_unit``, as the screens' refusal records them.
* **Read from your data** (BLUEPRINT §14.3, amendment after the fifth gate: "Readings settled by
  their values appear on the card under 'read from your data', each with its evidence and a way to
  change it"). The card carries the ledger's own section, :func:`turbotab.core.readings.
  read_from_data`, the one ``GET /api/projects/{pid}/readings`` serves, cut to the readings the
  same consumer reads, so the readings it settled sit beside the ones it asks. A consumer with
  nothing to ask shows no card (§14.2: ask only where it matters); its settled readings stay listed
  by that endpoint and in the methods record.

Nothing here reads a held-out row or changes a number: the card only asks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from pydantic import BaseModel, ConfigDict


class AskGroup(BaseModel):
    """One line of the card: a reading, or a homogeneous family of them confirmed as one block."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    kind: str  # the reading's kind (``turbotab.core.readings.KINDS``)
    columns: list[str]  # one column, or every column of the family
    guess: str | None  # the best guess the line leads with (``None``: nothing to propose)
    guess_words: str  # the guess in words ("codes for categories")
    evidence: str  # why it is guessed, from the values or the name ("" when none)


class AskExit(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    label: str
    decision: dict[str, Any] | None  # a Decision's JSON (``confirm_readings``, ``set_column_unit``…)


class AskSettled(BaseModel):
    """One reading the values settled for this consumer, no question asked
    (``readings.read_from_data``, the readings endpoint's own item)."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    kind: str
    column: str
    value: Any
    words: str  # what was read, as a predicate of the column ("is an amount (one slope)")
    evidence: str
    change: list[AskExit]  # the answers that change it


class AskCard(BaseModel):
    """"Tell me about these columns", in the question whose answer feeds the consumer."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    question: str  # the Router question it sits in
    consumer: str  # what reads these readings, in words ("the fit")
    text: str  # the question, each reading once by consequence (``readings.ask_text``)
    groups: list[AskGroup]
    exits: list[AskExit]  # the block confirmation first, then each reading's alternatives
    # BLUEPRINT §14.3: the readings this consumer reads that the values settled ("read from your
    # data"), each with its evidence and the answers that change it; never a required tap.
    read_from_data: list[AskSettled] = []


@dataclass
class AskContext:
    """What the card may read: the state, the Router's fresh artifacts, the table's column
    summaries and a store (each read lazily; the fit's card reads whole numbers from it)."""

    state: Any
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    column_info: Callable[[], Mapping[str, Any] | None] | Mapping[str, Any] | None = None
    store: Callable[[], Any] | Any = None

    def artifact(self, stage: str) -> Mapping[str, Any] | None:
        value = self.artifacts.get(stage)
        value = getattr(value, "data", value)
        return value if isinstance(value, Mapping) else None

    def info(self) -> Mapping[str, Any] | None:
        value = self.column_info() if callable(self.column_info) else self.column_info
        return value if isinstance(value, Mapping) else None

    def table(self) -> Any:
        try:
            return self.store() if callable(self.store) else self.store
        except Exception:  # noqa: BLE001 - no store: integer columns are read from their type
            return None


@dataclass
class Needed:
    """What a consumer asks: its unsettled readings and, where its refusal names them, its own
    ways forward (``readings.Unsettled.exits``)."""

    readings: list[Any]
    exits: list[dict[str, Any]] | None = None


# ── the readings each consumer reads, through the ledger's own functions ─────


def _combining(ctx: AskContext) -> list[Any]:
    """``sequence._aggregation_reads_settled_readings``: whole numbers that change within units,
    each a code (its most frequent value) or a count (its mean, first, last or change)."""
    from turbotab.core.readings import code_or_count_reading

    structure = ctx.artifact("structure") or {}
    facts = structure.get("code_or_count_facts") or {}
    own = dict(getattr(getattr(ctx.state, "aggregation", None), "columns", None) or {})
    out = []
    for c in structure.get("code_or_count") or []:
        if c in own:
            continue
        r = code_or_count_reading(ctx.state, c, facts.get(c) or {"whole": True, "n_values": 2},
                                  scope="combine")
        if r is not None and not r.settled:
            out.append(r)
    return out


def _roles_of(ctx: AskContext, columns: list[str]) -> list[Any]:
    from turbotab.core.readings import role_reading

    out = []
    for c in dict.fromkeys(columns):
        r = role_reading(ctx.state, c)
        if r is not None and not r.settled:
            out.append(r)
    return out


def _energy(ctx: AskContext) -> list[Any]:
    """``decisions._energy_reads_settled_roles``: total energy and every exposure that carries
    energy are the adjustment's columns, each a settled role before it is read."""
    from turbotab.core.stages.rows import energy_bearing

    roles = getattr(ctx.state, "roles", None) or {}
    named = [c for c, r in roles.items() if r == "energy"]
    named += [c for c, r in roles.items() if r == "exposure" and energy_bearing(c)]
    return _roles_of(ctx, named)


def _survey(ctx: AskContext) -> list[Any]:
    """``survey._design_is_settled``: the weight, strata and PSU a design-based answer reads."""
    roles = getattr(ctx.state, "roles", None) or {}
    return _roles_of(ctx, [c for c, r in roles.items() if r == "design"])


def _screens(ctx: AskContext) -> list[Any]:
    """``decisions._screens_read_a_settled_energy_column`` and ``_screens_wait_for_the_unit``:
    total energy's role, and its unit and days, which every offered intake screen reads."""
    roles = getattr(ctx.state, "roles", None) or {}
    return _roles_of(ctx, [c for c, r in roles.items() if r == "energy"])


def _fit(ctx: AskContext) -> Needed:
    """``readings.predictors_or_ask``: every recorded role that rode along, and each whole-valued
    predictor's codes or amounts; then, for text the user said holds amounts, how its values below
    a detection limit or its commas read, with the refusal's own ways forward."""
    from turbotab.core.decisions import left_out
    from turbotab.core.readings import Unsettled, predictors_or_ask

    if not getattr(ctx.state, "roles", None):
        return Needed([])
    try:
        predictors_or_ask(ctx.state, ctx.info(), drop=left_out(ctx.state), store=ctx.table())
    except Unsettled as waiting:
        return Needed(list(waiting.readings), list(waiting.exits))
    return Needed([])


def _energy_unit_lines(ctx: AskContext) -> tuple[list[AskGroup], list[AskExit]]:
    """Total energy's unit and days while only proposed: one line, answered by ``set_column_unit``
    as the screens' own refusal answers it (``stages.proposals.unit_refusal``)."""
    from turbotab.core.decisions import SetColumnUnit

    proposals = ctx.artifact("proposals") or {}
    reading = proposals.get("energy_unit") or {}
    energy = (reading.get("column") or (proposals.get("energy") or {}).get("energy_column")
              or next((c for c, r in (getattr(ctx.state, "roles", None) or {}).items()
                       if r == "energy"), None))
    if not reading or reading.get("confirmed", True) or not energy:
        return [], []
    unit = "kj" if reading.get("unit") == "kj" else "kcal"
    days = int((reading.get("days_reading") or {}).get("value") or reading.get("days") or 1)
    word = "kJ" if unit == "kj" else "kcal"
    span = "one day's intake" if days == 1 else f"a total over {days} days"
    group = AskGroup(kind="unit", columns=[str(energy)], guess=f"{unit}:{days}",
                     guess_words=f"{word}, {span}", evidence=str(reading.get("sentence") or ""))
    exits = [AskExit(label=f"`{energy}` is {word}, {span}",
                     decision=SetColumnUnit(column=str(energy), unit=unit, days=days)
                     .model_dump(mode="json"))]
    return [group], exits


def _plan(ctx: AskContext) -> list[Any]:
    """``estimand._estimand_names_a_predictor`` and the adjustment question: under inference the
    exposure is chosen among the settled predictors, and every recorded role decides whether its
    column is a covariate the adjustment set is asked about, so every role that rode along is
    asked here, before the exposure, rather than at the models question, where confirming it would
    reopen the adjustment set behind them (the routing gate's p16; BLUEPRINT §14.2: where the first
    consumer needs it)."""
    from turbotab.core.readings import unsettled

    if getattr(ctx.state, "purpose", None) != "inference":
        return []
    return _roles_of(ctx, unsettled(ctx.state))


#: question -> (the consumer its answer feeds, in words; the readings that consumer reads)
CONSUMERS: dict[str, tuple[str, Callable[[AskContext], list[Any] | Needed]]] = {
    "aggregation": ("combining each unit's rows", _combining),
    "survey": ("the survey design", _survey),
    "exclusions": ("the screens", _screens),
    "estimand": ("what you study and its effect", _plan),
    "adjustment": ("the adjustment set", _plan),
    "energy_adjustment": ("the energy adjustment", _energy),
    "models": ("the fit", _fit),
}


def _needed(question: str, ctx: AskContext) -> Needed:
    entry = CONSUMERS.get(question)
    if entry is None or ctx.state is None:
        return Needed([])
    try:
        found = entry[1](ctx)
    except Exception:  # noqa: BLE001 - a card that cannot be read asks nothing; the refusal asks
        return Needed([])
    needed = found if isinstance(found, Needed) else Needed(list(found))
    needed.readings = [r for r in needed.readings if r is not None and not r.settled]
    return needed


def readings_for(question: str, ctx: AskContext) -> list[Any]:
    """The unsettled readings the consumer behind ``question`` reads (none for a question whose
    answer reads no reading)."""
    return _needed(question, ctx).readings


# ── read from your data: the settled readings the same consumer reads ────────

_BODY_UNITS = ("cm", "m", "in")


def _energy_columns(state: Any) -> set[str]:
    return {c for c, r in (getattr(state, "roles", None) or {}).items() if r == "energy"}


def _reads(question: str, ctx: AskContext) -> Callable[[Mapping[str, Any]], bool] | None:
    """Which of the ledger's "read from your data" items the consumer behind ``question`` reads:
    the fit, its predictors' roles and codes or amounts and the outcome's task; combining, the
    codes or counts that change within units; the survey, its design's roles; the screens, total
    energy's role and unit, a height's unit and a sex column's coding; the energy adjustment,
    total energy's and the energy sources' roles and units."""
    from turbotab.core.readings import predictor_columns
    from turbotab.core.stages.rows import energy_bearing

    state = ctx.state
    roles = getattr(state, "roles", None) or {}
    if question == "models":
        columns = {*predictor_columns(state), *roles}
        target = getattr(state, "target", None)
        return lambda i: ((i["kind"] in ("role", "code_or_count") and i["column"] in columns)
                          or (i["kind"] == "task" and i["column"] == target))
    if question == "aggregation":
        structure = ctx.artifact("structure") or {}
        combined = set(structure.get("code_or_count") or [])
        return lambda i: i["kind"] == "code_or_count" and i["column"] in combined
    if question == "survey":
        design = {c for c, r in roles.items() if r == "design"}
        return lambda i: i["kind"] == "role" and i["column"] in design
    if question == "exclusions":
        energy = _energy_columns(state)
        return lambda i: ((i["kind"] in ("role", "unit") and i["column"] in energy)
                          or i["kind"] == "sex_coding"
                          or (i["kind"] == "unit" and i["value"] in _BODY_UNITS))
    if question == "energy_adjustment":
        sources = _energy_columns(state) | {c for c, r in roles.items()
                                            if r == "exposure" and energy_bearing(c)}
        return lambda i: i["kind"] in ("role", "unit") and i["column"] in sources
    if question in ("estimand", "adjustment"):
        return lambda i: i["kind"] == "role" and i["column"] in roles
    return None


def settled_for(question: str, ctx: AskContext) -> list[AskSettled]:
    """The "read from your data" items (``readings.read_from_data``, as the readings endpoint
    serves them) that the consumer behind ``question`` reads."""
    from turbotab.core.readings import read_from_data

    keep = _reads(question, ctx)
    if keep is None or ctx.state is None:
        return []
    try:
        items = read_from_data(ctx.state, roles=ctx.artifact("roles"),
                               target_info=ctx.artifact("target_info"),
                               proposals=ctx.artifact("proposals"), store=ctx.table(),
                               info=ctx.info())
    except Exception:  # noqa: BLE001 - nothing read: the card still asks what it must
        return []
    return [AskSettled(kind=str(i["kind"]), column=str(i["column"]), value=i["value"],
                       words=str(i["words"]), evidence=str(i["evidence"]),
                       change=[AskExit(label=str(e["label"]), decision=e.get("decision"))
                               for e in i.get("change") or []])
            for i in items if keep(i)]


def card(question: str, ctx: AskContext) -> AskCard | None:
    """The card for ``question``: its consumer's unsettled readings, ordered by consequence, a
    homogeneous family as one line, each with its guess and evidence, and the ways to answer
    them. None when nothing it reads waits."""
    from turbotab.core.readings import ask_exits, ask_text, families, guess_words

    entry = CONSUMERS.get(question)
    if entry is None:
        return None
    needed = _needed(question, ctx)
    readings = needed.readings
    extra_groups, extra_exits = (_energy_unit_lines(ctx) if question == "exclusions"
                                 else ([], []))
    if not readings and not extra_groups:
        return None
    groups = [AskGroup(kind=g[0].kind, columns=[r.column for r in g],
                       guess=None if g[0].value is None else str(g[0].value),
                       guess_words=guess_words(g[0]), evidence=g[0].evidence)
              for g in families(readings)]
    # The consumer's own ways forward where its refusal names them, else the ledger's ask.
    own = needed.exits or (ask_exits(readings, ctx.state) if readings else [])
    exits = [AskExit(label=str(e["label"]), decision=e.get("decision")) for e in own]
    text = ask_text(readings) if readings else (
        f"Tell me about this column: `{extra_groups[0].columns[0]}`: "
        f"{extra_groups[0].guess_words}? ({extra_groups[0].evidence})")
    return AskCard(question=question, consumer=entry[0], text=text,
                   groups=[*groups, *extra_groups], exits=[*exits, *extra_exits],
                   read_from_data=settled_for(question, ctx))


__all__ = ["AskCard", "AskContext", "AskExit", "AskGroup", "AskSettled", "CONSUMERS", "Needed",
           "card", "readings_for", "settled_for"]
