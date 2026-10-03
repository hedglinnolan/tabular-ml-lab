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
  the readings it lists, then each reading's alternatives. An energy column's unit and days are
  recorded by ``set_column_unit``, as the screens' refusal records them.

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


class AskCard(BaseModel):
    """"Tell me about these columns", in the question whose answer feeds the consumer."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    question: str  # the Router question it sits in
    consumer: str  # what reads these readings, in words ("the fit")
    text: str  # the question, each reading once by consequence (``readings.ask_text``)
    groups: list[AskGroup]
    exits: list[AskExit]  # the block confirmation first, then each reading's alternatives


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


def _fit(ctx: AskContext) -> list[Any]:
    """``readings.predictors_or_ask``: every recorded role that rode along, and each whole-valued
    predictor's codes or amounts."""
    from turbotab.core.decisions import left_out
    from turbotab.core.readings import Unsettled, predictors_or_ask

    if not getattr(ctx.state, "roles", None):
        return []
    try:
        predictors_or_ask(ctx.state, ctx.info(), drop=left_out(ctx.state), store=ctx.table())
    except Unsettled as waiting:
        return list(waiting.readings)
    return []


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


#: question -> (the consumer its answer feeds, in words; the readings that consumer reads)
CONSUMERS: dict[str, tuple[str, Callable[[AskContext], list[Any]]]] = {
    "aggregation": ("combining each unit's rows", _combining),
    "survey": ("the survey design", _survey),
    "exclusions": ("the screens", _screens),
    "energy_adjustment": ("the energy adjustment", _energy),
    "models": ("the fit", _fit),
}


def readings_for(question: str, ctx: AskContext) -> list[Any]:
    """The unsettled readings the consumer behind ``question`` reads (none for a question whose
    answer reads no reading)."""
    entry = CONSUMERS.get(question)
    if entry is None or ctx.state is None:
        return []
    try:
        return [r for r in entry[1](ctx) if r is not None and not r.settled]
    except Exception:  # noqa: BLE001 - a card that cannot be read asks nothing; the refusal asks
        return []


def card(question: str, ctx: AskContext) -> AskCard | None:
    """The card for ``question``: its consumer's unsettled readings, ordered by consequence, a
    homogeneous family as one line, each with its guess and evidence, and the ways to answer
    them. None when nothing it reads waits."""
    from turbotab.core.readings import ask_exits, ask_text, families, guess_words

    entry = CONSUMERS.get(question)
    if entry is None:
        return None
    readings = readings_for(question, ctx)
    extra_groups, extra_exits = (_energy_unit_lines(ctx) if question == "exclusions"
                                 else ([], []))
    if not readings and not extra_groups:
        return None
    groups = [AskGroup(kind=g[0].kind, columns=[r.column for r in g],
                       guess=None if g[0].value is None else str(g[0].value),
                       guess_words=guess_words(g[0]), evidence=g[0].evidence)
              for g in families(readings)]
    exits = [AskExit(label=str(e["label"]), decision=e.get("decision"))
             for e in ask_exits(readings, ctx.state)] if readings else []
    text = ask_text(readings) if readings else (
        f"Tell me about this column: `{extra_groups[0].columns[0]}`: "
        f"{extra_groups[0].guess_words}? ({extra_groups[0].evidence})")
    return AskCard(question=question, consumer=entry[0], text=text,
                   groups=[*groups, *extra_groups], exits=[*exits, *extra_exits])


__all__ = ["AskCard", "AskContext", "AskExit", "AskGroup", "CONSUMERS", "card", "readings_for"]
