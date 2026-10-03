"""The opening sequence's refusals (M2_CONTRACT §1; OPENING_SEQUENCE.md §01 and §03).

Validators for the structural questions — orientation, the event level, grain, repeats or time
points, the unit of analysis, aggregation and temporal prediction. Each reads only what ``ctx``
names (the server's ``DecisionContext``: ``columns``, ``column_info``, ``state``, ``task``,
``target``, and ``artifact(stage)`` for a fresh public artifact) and checks nothing it is not told.
Every refusal carries exits, and a contradiction between the user's answer and the data carries an
attestation exit as well: the user is the authority, and the disagreement is recorded rather than
blocked (``turbotab/grain.py``; DESIGN_LANGUAGE §09: resolve or attest, never a dead end).

Decision A (structural answers refused after the seal) is the seal's, not this module's.

Imported by ``turbotab.core.decisions``, which registers these on import.
"""
from __future__ import annotations

from typing import Any, Mapping

from turbotab.core.decisions import (
    _UNKNOWN,
    NUMERIC_DTYPES,
    ROW_ID,
    Refusal,
    SetAggregation,
    SetEvent,
    SetGrain,
    SetOrientation,
    SetTask,
    SetTemporal,
    SetUnit,
    _and,
    _columns_of,
    _ctx,
    _state,
    _target_of,
    register_completion,
    register_validator,
)

ATTEST = "My answer is right; the data is like this"


def artifact(ctx: Any, stage: str) -> Mapping[str, Any] | None:
    """A stage's fresh public artifact from ``ctx`` (its ``artifact`` callable), else None."""
    fn = _ctx(ctx, "artifact")
    if not callable(fn):
        return None
    try:
        value = fn(stage)
    except Exception:  # noqa: BLE001 - a missing artifact checks nothing
        return None
    value = getattr(value, "data", value)
    return value if isinstance(value, Mapping) else None


def _repeated(state: Any) -> bool:
    spec = getattr(state, "grain", None)
    return spec is not None and spec.grain == "repeated"


def _unknown(column: str | None, ctx: Any) -> bool:
    if column is None:
        return False
    if column == ROW_ID:
        return True
    columns = _columns_of(ctx)
    return columns is not None and column not in columns


def _no_such_column(column: str, exits: list[dict[str, Any]] | None = None) -> Refusal:
    return Refusal("unknown_column", f"This dataset has no column named `{column}`.",
                   exits=exits or [{"label": "Choose one of the dataset's columns", "decision": None}])


# ── 1.5 · which way round ─────────────────────────────────────────────────────


def _orientation_turns_before_the_target(decision: SetOrientation, ctx: Any) -> None:
    state = _state(ctx)
    if state is None or state.target is None:
        return
    turned = state.orientation == "feature_major"
    if (decision.orientation == "feature_major") == turned:
        return  # nothing about the table changes
    keep = "feature_major" if turned else "sample_major"
    raise Refusal(
        "target_exists",
        f"The outcome `{state.target}` is chosen, and turning the table around would make it a "
        f"row. The table is turned before the outcome is chosen.",
        exits=[{"label": "Keep the table as it is", "decision": SetOrientation(orientation=keep)}],
    )


def _orientation_can_turn(decision: SetOrientation, ctx: Any) -> None:
    if decision.orientation != "feature_major":
        return
    turn = (artifact(ctx, "oriented") or {}).get("turn") or {}
    if turn.get("refusal"):
        raise Refusal(
            str(turn.get("code") or "cannot_turn"), str(turn["refusal"]),
            exits=[{"label": "Keep the table as it is",
                    "decision": SetOrientation(orientation="sample_major")}],
        )


# ── 2 · which level is the event ──────────────────────────────────────────────


def _levels(ctx: Any, target: str) -> list[Any]:
    info = artifact(ctx, "target_info") or {}
    if info.get("column") != target:
        return []
    return [c.get("value") for c in info.get("classes") or [] if c.get("value") is not None]


def _event_is_a_level_of_the_outcome(decision: SetEvent, ctx: Any) -> None:
    from turbotab.core.stages.rows import _level_key

    target = _target_of(ctx)
    if target is _UNKNOWN:
        return
    if target is None:
        raise Refusal("no_target", "Choose the outcome first; the event is one of its levels.",
                      exits=[{"label": "Choose the outcome", "decision": None}])
    if decision.column != target:
        raise Refusal(
            "not_the_target",
            f"The outcome is `{target}`, not `{decision.column}`; the event is a level of the outcome.",
            exits=[{"label": f"Choose the event level of `{target}`", "decision": None}],
        )
    levels = _levels(ctx, target)
    task = _ctx(ctx, "task")
    if task is not None and task not in ("binary", "time_to_event"):
        exits = ([{"label": "Treat it as binary", "decision": SetTask(column=target, task="binary")}]
                 if len(levels) == 2 else [])
        raise Refusal(
            "not_binary",
            f"`{target}` is read as a {task} outcome; an event level belongs to a two-level outcome.",
            exits=exits + [{"label": "Keep it as it is", "decision": None}],
        )
    if levels and _level_key(decision.level) not in {_level_key(v) for v in levels}:
        raise Refusal(
            "unknown_level",
            f"`{target}` has no level `{decision.level}`; its levels are {_and([str(v) for v in levels])}.",
            exits=[{"label": f"The event is `{v}`", "decision": SetEvent(column=target, level=str(v))}
                   for v in levels[:2]],
        )


# ── 3 · can one unit appear in more than one row ──────────────────────────────


def _oriented_column(ctx: Any, column: str) -> tuple[Mapping[str, Any] | None, int | None]:
    """The column's record in the oriented table (before any rows are combined), and its rows."""
    oriented = artifact(ctx, "oriented")
    if not oriented:
        return None, None
    for c in oriented.get("columns") or []:
        if c.get("name") == column:
            return c, int(oriented.get("n_rows") or 0)
    return None, int(oriented.get("n_rows") or 0)


def _grain_is_consistent(decision: SetGrain, ctx: Any) -> None:
    structure = artifact(ctx, "structure") or {}
    reading = structure.get("grain") or {}
    target = _target_of(ctx)
    target = None if target is _UNKNOWN else target
    suggested = [c for c in reading.get("suggested") or [] if c != target]
    column = decision.id_column

    def repeats_by(c: str) -> dict[str, Any]:
        return {"label": f"Rows repeat per `{c}`", "decision": SetGrain(grain="repeated", id_column=c)}

    if decision.grain == "unknown":
        return  # "I don't know" claims nothing the data could contradict; the seal says so
    if column is not None and _unknown(column, ctx):
        raise _no_such_column(column, [repeats_by(c) for c in suggested[:3]] or None)
    if decision.grain == "repeated":
        if not column:
            raise Refusal(
                "no_id_column",
                "Name the column that says which unit a row belongs to; the held-out rows keep "
                "each unit's rows together by it.",
                exits=[repeats_by(c) for c in suggested[:3]] + [{"label": "Name the column", "decision": None}],
            )
        if column == target:
            raise Refusal("target_is_id", f"`{column}` is the outcome, so it cannot name the units.",
                          exits=[repeats_by(c) for c in suggested[:3]])
        if decision.acknowledged:
            return
        info, n_rows = _oriented_column(ctx, column)
        if info is not None and n_rows:
            present = n_rows - int(info.get("n_missing") or 0)
            if present and int(info.get("n_unique") or 0) >= present:
                raise Refusal(
                    "id_column_unique",
                    f"`{column}` has a different value on every one of its {present:,} rows, so no "
                    f"unit repeats in it; grouping by it would hold out single rows.",
                    exits=[*(repeats_by(c) for c in suggested[:2] if c != column),
                           {"label": "One row per unit", "decision": SetGrain(grain="one_row_per_unit")},
                           {"label": ATTEST, "decision": SetGrain(grain="repeated", id_column=column,
                                                                  acknowledged=True)}],
                )
        return
    if decision.acknowledged:
        return
    found = reading.get("if_one_row")  # turbotab/grain.py's contradiction, read off the table
    if found:
        raise Refusal(
            "data_repeats", str(found["message"]),
            exits=[repeats_by(str(found["columns"][0])),
                   {"label": ATTEST, "decision": SetGrain(grain="one_row_per_unit", id_column=column,
                                                          acknowledged=True)}],
        )


# ── 4–7 · repeats, the unit, combining, temporal ──────────────────────────────


def _needs_repeats(ctx: Any, question: str) -> None:
    from turbotab.core.stages.working import effective_grain

    state = _state(ctx)
    if state is None or _repeated(state):
        return
    grain = effective_grain(state, artifact(ctx, "structure"))
    if grain is None:
        message = f"Say first whether a unit can appear in more than one row; {question} follows from it."
    elif grain.grain == "unknown":
        message = (f"Whether a unit can appear in more than one row was answered as not known, so "
                   f"{question} does not arise.")
    else:
        message = f"Each unit appears once, so {question} does not arise."
    raise Refusal("not_repeated", message, exits=[{"label": "Answer the grain question", "decision": None}])


def _repeat_kind_follows_the_grain(decision: Any, ctx: Any) -> None:
    _needs_repeats(ctx, "whether rows are repeats or time points")
    if decision.time_column is not None and _unknown(decision.time_column, ctx):
        raise _no_such_column(decision.time_column)


def _unit_follows_the_grain(decision: SetUnit, ctx: Any) -> None:
    _needs_repeats(ctx, "what one row of the analysis is")


def _aggregation_knows_the_outcome(decision: SetAggregation, ctx: Any) -> None:
    _needs_repeats(ctx, "how a unit's rows are combined")
    state = _state(ctx)
    if state is None:
        return
    if state.unit != "unit":
        said = ("The answer was one row per record, so no rows are combined." if state.unit == "row"
                else "What one row of the analysis is has not been answered yet; combining "
                     "follows it.")
        raise Refusal(
            "rows_stay",
            said,
            exits=[{"label": "Combine each unit's rows", "decision": SetUnit(unit="unit")}],
        )
    if state.target is None:
        raise Refusal("no_target",
                      "Choose the outcome first: combining rows has to know which outcome to keep.",
                      exits=[{"label": "Choose the outcome", "decision": None}])
    outcome = (artifact(ctx, "structure") or {}).get("outcome") or {}
    if outcome.get("column") != state.target or not outcome.get("varies"):
        return
    task = _ctx(ctx, "task")
    info = (_ctx(ctx, "column_info") or {}).get(state.target) or {}
    numeric = bool(outcome.get("numeric", info.get("dtype") in NUMERIC_DTYPES))
    allowed = ["first", "last"] + (["mean"] if numeric and task not in ("binary", "multiclass",
                                                                         "ordinal", "time_to_event")
                                   else [])
    label = {"first": "Keep the first outcome", "last": "Keep the last outcome",
             "mean": "Average the outcome"}
    exits = [{"label": label[o], "decision": SetAggregation(method=decision.method, outcome=o)}
             for o in allowed]
    n = int(outcome.get("n_units_varying") or 0)
    if decision.outcome is None:
        raise Refusal(
            "which_outcome",
            f"`{state.target}` changes within {n:,} units, so combining their rows needs to know "
            f"which outcome to keep.",
            exits=exits,
        )
    if decision.outcome not in allowed:
        raise Refusal(
            "outcome_not_numeric",
            f"`{state.target}` is a {task or 'categorical'} outcome, so its mean is not one of its "
            f"values; keep the first or the last.",
            exits=exits,
        )


def _aggregation_can_order_the_records(decision: SetAggregation, ctx: Any) -> None:
    """Refuse first, last or change when the time column cannot put a unit's records in order.

    The structure stage reads the column (``time_order``, stages.working.time_order): text visit
    labels, dates that read month-first and day-first alike, or a column that places fewer than
    half of its values. Combining by it once fell back to file order while the Record said "in
    order of `visit_date`" (audit MA-03). Exits: declare the levels' order (the natural order is
    offered), combine by the mean, or choose another column.
    """
    from turbotab.core.decisions import SetRepeatKind
    from turbotab.core.stages.working import needs_order, order_refusal

    for column in decision.columns:
        if _unknown(column, ctx):
            raise _no_such_column(column)
    state = _state(ctx)
    if state is None or not needs_order(decision.method, decision.outcome, decision.columns):
        return
    order = (artifact(ctx, "structure") or {}).get("time_order")
    if not order or order.get("orderable"):
        return
    exits: list[dict[str, Any]] = []
    spec = state.repeat_kind
    kind = spec.repeat_kind if spec is not None else "time_points"
    if order.get("kind") in ("none", "levels") and order.get("proposed"):
        levels = list(order["proposed"])
        shown = ", ".join(f"`{v}`" for v in levels[:4]) + (" …" if len(levels) > 4 else "")
        exits.append({"label": f"Order them {shown}",
                      "decision": SetRepeatKind(repeat_kind=kind, time_column=order["column"],
                                                levels=levels)})
    if order.get("kind") == "ambiguous":
        exits.append({"label": "Say how the dates are written (the dates finding)", "decision": None})
    if decision.outcome != "first" and decision.outcome != "last":
        exits.append({"label": "Combine by the mean", "decision": SetAggregation(
            method="mean", outcome=decision.outcome)})
    exits.append({"label": "Choose another time column", "decision": None})
    raise Refusal("cannot_order", order_refusal(order, decision.method), exits=exits)


def _aggregation_reads_settled_readings(decision: SetAggregation, ctx: Any) -> None:
    """BLUEPRINT §14.1 (the readings ledger): combining a unit's rows is a number-changing
    consumer of two readings, each settled before it is read.

    * **Code or count.** Whole numbers with a few values each seen several times fit a count (cups
      of coffee per recall) as well as a code (smoking 1/2/3); under "mean" one is averaged and the
      other takes its most frequent value. The gate's ``coffee_cups`` and ``eating_occasions`` were
      combined by their mode unasked, changing 112 and 130 of 150 people's values. Asked per
      column (a code or a count), unless the answer gives the column its own rule.
    * **The time column.** First, last and change take a value from a particular record; the order
      is the user's (named, or the reading's column confirmed on its own), never the reading's
      alone."""
    from turbotab.core.readings import code_or_count_exits, confirm_exit, confirmation, listing
    from turbotab.core.stages.working import needs_order, proposed_time_column, time_column

    state = _state(ctx)
    structure = artifact(ctx, "structure") or {}
    if state is None or not structure:
        return
    waiting = [c for c in structure.get("code_or_count") or []
               if c not in decision.columns and confirmation(state, "code_or_count", c) is None]
    if waiting:
        one = len(waiting) == 1
        raise Refusal(
            "reading_unsettled",
            f"{listing(waiting)} {'holds' if one else 'hold'} a few whole-number values that "
            f"change within units, which may be codes for categories (combined by the most "
            f"frequent value) or counts (combined by the {decision.method}). Say which for "
            f"{'it' if one else 'each'}, one at a time, then combine.",
            exits=[e for c in waiting for e in code_or_count_exits(c)])
    if not needs_order(decision.method, decision.outcome, decision.columns):
        return
    if time_column(state, structure) is not None:
        return
    proposed = proposed_time_column(state, structure)
    if proposed is None:
        return  # no reading proposes an order: the records keep the file's, as the record says
    exits: list[dict[str, Any]] = [
        confirm_exit("time_column", proposed, "orders",
                     f"`{proposed}` orders each unit's records")]
    if decision.method != "mean" and decision.outcome != "first" and decision.outcome != "last":
        exits.append({"label": "Combine by the mean",
                      "decision": SetAggregation(method="mean", outcome=decision.outcome)})
    exits.append({"label": "Choose another time column (the repeats question)", "decision": None})
    raise Refusal(
        "reading_unsettled",
        f"Combining by the first, last or change takes each unit's values from a particular record, "
        f"and only the repeats reading orders the records, by `{proposed}`. Confirm that it orders "
        f"them, or name the column that does.",
        exits=exits)


def _temporal_needs_time_points_as_rows(decision: SetTemporal, ctx: Any) -> None:
    from turbotab.core.stages.working import time_column

    _needs_repeats(ctx, "temporal prediction")
    state = _state(ctx)
    if state is None:
        return
    if state.unit == "unit":
        raise Refusal(
            "rows_combined",
            "Each unit's rows are combined into one, so there is no later row to predict from an earlier one.",
            exits=[{"label": "Keep one row per record", "decision": SetUnit(unit="row")}],
        )
    if not decision.temporal:
        return
    if decision.time_column is not None and _unknown(decision.time_column, ctx):
        raise _no_such_column(decision.time_column)
    structure = artifact(ctx, "structure")
    if decision.time_column or time_column(state, structure):
        return
    # BLUEPRINT §14.1: a bare "yes" is never completed by the repeats reading's column alone; the
    # reading's column is offered first, to be named.
    from turbotab.core.stages.working import proposed_time_column

    proposed = proposed_time_column(state, structure)
    candidates = list(dict.fromkeys([*([proposed] if proposed else []),
                                     *((structure or {}).get("time_columns") or [])]))[:3]
    raise Refusal(
        "no_time_column",
        "A chronological split needs a column that says when each row was taken, and none is known.",
        exits=[{"label": f"Order by `{c}`", "decision": SetTemporal(temporal=True, time_column=c)}
               for c in candidates]
        + [{"label": "Not a temporal prediction", "decision": SetTemporal(temporal=False)}],
    )


def _temporal_names_its_time_column(decision: SetTemporal, ctx: Any) -> SetTemporal:
    """A "yes" that names no column takes the one the validator accepted it by (the stated
    reading's): the seal draws by the recorded column only, so the record must name it, or the
    sentence would say "the latest" while the seal drew at random."""
    from turbotab.core.stages.working import time_column

    if not decision.temporal or decision.time_column:
        return decision
    state = _state(ctx)
    column = time_column(state, artifact(ctx, "structure")) if state is not None else None
    return decision.model_copy(update={"time_column": column}) if column else decision


# ── answers in the Router's order (M2_CONTRACT §12.2) ─────────────────────────


def question_of(kind: str) -> str | None:
    """The Router question a decision kind answers (``set_grain`` → ``grain``), or None."""
    from turbotab.core.decisions import SLOTS
    from turbotab.core.interview import QUESTION_KEYS, SLOT_OF

    by_slot = {SLOT_OF.get(k, k): k for k in QUESTION_KEYS}
    slot = SLOTS.get(kind)
    return by_slot.get(slot) if slot is not None else None


def _steps(ctx: Any) -> list[Any] | None:
    """The Router's steps for the project as it stands (``ctx.interview()``), else None."""
    fn = _ctx(ctx, "interview")
    if not callable(fn):
        return None
    try:
        return list(fn())
    except Exception:  # noqa: BLE001 - no Router: nothing is checked against it
        return None


def _answers_in_order(decision: Any, ctx: Any) -> None:
    """Refuse an answer to a question still waiting behind an earlier unanswered one.

    Changing an answered question is always allowed, as is overturning a stated skip ("Ask me
    anyway"); only a question the Router has not reached yet is refused, with the way forward.
    """
    from turbotab.core.interview import first_unanswered
    from turbotab.core.voice import question_name

    steps = _steps(ctx)
    question = question_of(decision.kind)
    if not steps or question is None:
        return
    step = next((s for s in steps if s.key == question), None)
    if step is None or step.status not in ("open", "waiting"):
        return  # answered, stated, or not applicable: the Router is not holding it back
    first = first_unanswered(steps)
    if first is None or first.key == question:
        return
    name = question_name(first.key)
    raise Refusal(
        "not_yet",
        f"{name[:1].upper()}{name[1:]} comes before this one and is not answered yet; the "
        f"questions are asked in order because each later one depends on the earlier answers.",
        exits=[{"label": f"Answer {name} first", "decision": None}],
    )


def _seal_needs_grain(decision: Any, ctx: Any) -> None:
    """The seal is drawn by the grain answer: answered, or stated from a unique identifier. An
    undetermined basis comes only from answering "I don't know", never from skipping the question."""
    from turbotab.core.stages.working import effective_grain
    from turbotab.core.voice import question_name

    state = _state(ctx)
    if state is None or state.grain is not None or not callable(_ctx(ctx, "artifact")):
        return
    if effective_grain(state, artifact(ctx, "structure")) is not None:
        return
    name = question_name("grain")
    raise Refusal(
        "not_yet",
        "The held-out rows are drawn by the grain answer, and whether a unit can appear in more "
        "than one row is not answered yet.",
        exits=[{"label": f"Answer {name} first", "decision": None}],
    )


def _register_order() -> None:
    from turbotab.core.decisions import SLOTS

    # first: an answer the Router has not reached is refused for that reason before any other
    # check reads it (the seal's Decision A, registered later and first, still outranks it)
    for kind in list(SLOTS):
        if question_of(kind) is not None:
            register_validator(kind, _answers_in_order, first=True)
    register_validator("set_split", _seal_needs_grain)


_register_order()
register_completion("set_temporal", _temporal_names_its_time_column)
register_validator("set_orientation", _orientation_turns_before_the_target)
register_validator("set_orientation", _orientation_can_turn)
register_validator("set_event", _event_is_a_level_of_the_outcome)
register_validator("set_grain", _grain_is_consistent)
register_validator("set_repeat_kind", _repeat_kind_follows_the_grain)
register_validator("set_unit", _unit_follows_the_grain)
register_validator("set_aggregation", _aggregation_knows_the_outcome)
register_validator("set_aggregation", _aggregation_can_order_the_records)
register_validator("set_aggregation", _aggregation_reads_settled_readings)
register_validator("set_temporal", _temporal_needs_time_points_as_rows)

__all__ = ["ATTEST", "artifact", "question_of"]
