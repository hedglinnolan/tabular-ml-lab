"""The methods text, built from the append-only decision log (audit WP16).

The Record keeps every answer ever given, in order; a methods section states the analysis. Built
naively from the log, it kept sentences no longer true: after a time-to-event outcome was answered
as binary, "`cvd_event` was analyzed as a time to event" stayed beside the corrective sentence
while the fit was logistic (the intelligence fix run's observation). So the text folds superseded
decisions out, but only those superseded before anything was seen:

* **In force.** A record is in force when it is the latest live write of what it writes (its slot,
  or its key of a keyed slot), its condition holds (a task answer is for the current outcome), and
  it still applies to the analysis: a follow-up only while the outcome is answered as a time to
  event, an event level only for a binary or time-to-event outcome, a declared order only for an
  ordinal one (:data:`APPLIES`). A revert is never in force: the answer it restores is.
* **Events.** An opening of the held-out rows and a re-seal are in force while they are about the
  current outcome; the analysis-plan lock always is.
* **Nothing is folded out once something was seen.** From the first opening or the plan lock on,
  the records in force at that moment stay (the analysis as it was opened, or as it was declared
  before any estimate was displayed), and so does every record made after it, each led by what had
  been seen ("After the estimates were seen, …"; ``decisions.disclose``). A change tried after the
  estimates were seen and then withdrawn is a fork in Gelman & Loken's (2013) sense, so it is
  stated, not folded away. An answer held when it was recorded and held still (the energy model or
  the substitution posted under prediction; :func:`never_applied`) changed nothing, so it is no
  fork and writes no sentence, nor does a revert of it; nor does one recorded after that moment
  that the goal holds now, whatever the goal was when it was recorded, since the methods never
  state a held answer as applied (W1-R round 2, ruling R-d). An answer in force at that moment and
  held by the goal now stays in the plan as declared, which is what the lock fixed, with one added
  sentence: it is not applied under the goal as it stands (:func:`not_applied_sentence`).
* **Restated.** An in-force sentence that says what another answer does to it (under the surveyed
  population, each chosen family's estimator or block, the substitution band, the calibration;
  MS4) is restated on the answers as they stand (:func:`restated`; ``voice.restate``), so the
  sample-only exit, or a population answer given after the models were chosen, never leaves a
  sentence that contradicts the analysis. A sentence that counts rows (the exclusions', the
  complete cases') is restated with the counts the participant flow has now, given in the context
  (``counts``; EXPORT), so the adjustment set answered after the missing-values question never
  leaves its count stale. The Record keeps each sentence as said.
"""
from __future__ import annotations

import typing
from typing import Any, Callable, Sequence

from pydantic import BaseModel, ConfigDict

from turbotab.core import decisions
from turbotab.core.decisions import Refusal, Revert

# kind -> whether its answer still applies under the state (beside its own slot's condition).
APPLIES: dict[str, Callable[[Any, Any], bool]] = {
    "set_follow_up": lambda d, state: state.task == "time_to_event",
    "set_event": lambda d, state: state.task in (None, "binary", "time_to_event"),
    "set_outcome_order": lambda d, state: state.task == "ordinal",
}
EVENTS = ("open_seal", "reseal", "lock_plan")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class MethodsLine(_Model):
    """One record's sentence in the methods text."""

    record_id: str
    seq: int
    kind: str
    sentence: str
    in_force: bool  # what the analysis is now; False: kept because it was seen (see the module)
    post_seal: bool
    after_estimates: bool


class MethodsText(_Model):
    """The methods text: the sentences in record order, and the same joined as one paragraph."""

    lines: list[MethodsLine]
    # The record from which nothing is folded out (the first opening or the plan lock), if any.
    seen_from: int | None = None
    text: str


def _writes(decision: Any) -> list[tuple[str, str | None]]:
    """What a decision writes: ``(slot, key)`` pairs, key None for an unkeyed slot."""
    kind = decision.kind
    if kind in decisions._ENTRIES:
        return [(slot, key) for slot, key, _ in decisions._ENTRIES[kind](decision)]
    if kind in decisions._KEYS:
        slot = decisions._SLOT_FOR[kind](decision) if kind in decisions._SLOT_FOR \
            else decisions.SLOTS[kind]
        return [(slot, decisions._KEYS[kind](decision))]
    return [(decisions.SLOTS[kind], None)]


def in_force(records: Sequence[Any]) -> set[str]:
    """Ids of the records in force in ``records`` (see the module)."""
    ordered = sorted(records, key=lambda r: r.seq)
    try:
        cancelled = decisions.reverted(ordered)
        state = decisions.fold(ordered)
    except Refusal:
        return set()
    slots = state.model_dump()
    latest: dict[tuple[str, str | None], str] = {}
    out: set[str] = set()
    for record in ordered:
        d = record.decision
        if record.id in cancelled or isinstance(d, Revert) or d.kind not in decisions.SLOTS:
            continue
        if d.kind in EVENTS:
            target = getattr(d, "target", None)
            if d.kind == "lock_plan" or target is None or target == state.target:
                out.add(record.id)
            continue
        holds = decisions._HOLDS.get(d.kind)
        if holds is not None and not holds(d, slots):
            continue
        for written in _writes(d):
            latest[written] = record.id
    for record_id in latest.values():
        out.add(record_id)
    by_id = {r.id: r for r in ordered}
    return {i for i in out
            if (fn := APPLIES.get(by_id[i].decision.kind)) is None or fn(by_id[i].decision, state)}


def held_by_goal(decision: Any, slots: dict[str, Any]) -> bool:
    """Whether the goal holds ``decision``: its condition (``decisions.register_kind(...,
    holds=)``) fails on ``slots`` and holds under another goal (the energy model and the
    substitution while the goal is prediction, BEATS_PREDICT §9 ruling 5; a survey answer given for
    another goal). A condition the goal does not decide (a task answered for an earlier outcome) is
    not held by it."""
    holds = decisions._HOLDS.get(decision.kind)
    if holds is None or holds(decision, slots):
        return False
    goals = (*typing.get_args(decisions.Purpose), None)
    return any(holds(decision, {**slots, "purpose": g}) for g in goals
               if g != slots.get("purpose"))


def never_applied(records: Sequence[Any], since: int) -> set[str]:
    """Ids of the records after ``since`` (a seq) that applied to nothing the analysis is now: an
    answer whose condition (``decisions.register_kind(..., holds=)``) failed when it was recorded
    and fails on the answers as they stand, an answer the goal holds now (:func:`held_by_goal`)
    whatever the goal was when it was recorded (W1-R round 2, ruling R-d: the energy model recorded
    under Estimate after the opening, the goal then prediction again), and a revert of one. Held,
    such an answer is no change made to the analysis now, so it writes no methods sentence; it
    does once its condition holds. A revert of an answer recorded by ``since`` that applied when
    it was recorded withdraws an answer of the plan as declared, which is a change made whatever
    the goal is now, so it is not one (W1-R round 3, SE-m5). An opening and a re-seal are events,
    never one."""
    ordered = sorted(records, key=lambda r: r.seq)
    try:
        now = decisions.fold(ordered).model_dump()
    except Refusal:
        return set()
    by_id = {r.id: r for r in ordered}
    at = {r.id: i for i, r in enumerate(ordered)}
    out: set[str] = set()
    for record in ordered:
        if record.seq <= since:
            continue
        answer = record
        while answer is not None and isinstance(answer.decision, Revert):
            answer = by_id.get(answer.decision.decision_id)
        if answer is None or answer.decision.kind in EVENTS:
            continue
        holds = decisions._HOLDS.get(answer.decision.kind)
        if holds is None or holds(answer.decision, now):
            continue
        declared = answer is not record and answer.seq <= since
        if held_by_goal(answer.decision, now) and not declared:
            out.add(record.id)
            continue
        try:
            then = decisions.fold(ordered[:at[answer.id] + 1]).model_dump()
        except Refusal:
            continue
        if not holds(answer.decision, then):
            out.add(record.id)
    return out


def not_applied_sentence(slots: dict[str, Any]) -> str:
    """What a kept answer the goal holds adds (:func:`methods_text`): the goal it is not applied
    under, as the purpose answer names it."""
    from turbotab.core.voice import tick

    return f"It is not applied while the analysis is declared for {tick(slots.get('purpose'))}."


def seen_from(records: Sequence[Any]) -> int | None:
    """The seq of the first record after which something had been seen: the first opening of
    held-out rows or the analysis-plan lock (None: nothing seen yet)."""
    ordered = sorted(records, key=lambda r: r.seq)
    try:
        cancelled = decisions.reverted(ordered)
    except Refusal:
        cancelled = {}
    first = next((r for r in ordered if r.id not in cancelled
                  and r.decision.kind in ("open_seal", "lock_plan")), None)
    if first is None:
        return None
    # A lock recorded when the purpose became inference after estimates were displayed under
    # prediction: they were seen from the record before they were shown (the routing gate's p02).
    at = getattr(first.decision, "seen_at", None) if first.decision.kind == "lock_plan" else None
    return min(first.seq, int(at)) if at else first.seq


def restated(records: Sequence[Any], now: set[str], ctx: Any = None) -> dict[str, str]:
    """The in-force sentences that read differently on the answers as they stand: record id → the
    sentence restated (``voice.restate``; MS4). Such a sentence says what another answer does to its
    own; for example, the families' estimators under the survey answer. It was authored on the
    state just before its record (``DecisionLog.append``) and is restated on the state of the
    whole log. ``ctx``: what a restated clause may read besides the state (the detected task). A
    record kept only because it was seen is left as said."""
    from turbotab.core import voice

    ordered = sorted(records, key=lambda r: r.seq)
    todo = [i for i, r in enumerate(ordered)
            if r.id in now and r.sentence and voice.restates(r.decision.kind)]
    if not todo:
        return {}
    try:
        state = decisions.fold(ordered)
    except Refusal:
        return {}
    out: dict[str, str] = {}
    for i in todo:
        r = ordered[i]
        try:
            then = decisions.fold(ordered[:i])
        except Refusal:
            continue
        said = voice.restate(r.decision, r.sentence, then, state, ctx,
                             post_seal=bool(r.post_seal),
                             after_estimates=bool(getattr(r, "after_estimates", False)))
        if said != r.sentence:
            out[r.id] = said
    return out


def methods_text(records: Sequence[Any], ctx: Any = None) -> MethodsText:
    """The methods text of a decision log (see the module). ``ctx``: what a restated sentence may
    read besides the answers (``detected_task``, the task the target stage detected)."""
    ordered = sorted(records, key=lambda r: r.seq)
    now = in_force(ordered)
    seen = seen_from(ordered)
    kept = set(now)
    declared: set[str] = set()
    if seen is not None:
        declared = in_force([r for r in ordered if r.seq <= seen])
        kept |= declared
        kept |= {r.id for r in ordered if r.seq > seen} - never_applied(ordered, seen)
    again = restated(ordered, now, ctx)
    # Ruling R-d: an answer in force when the plan was locked (or the held-out rows opened) stays
    # in the plan as declared, which is what the lock fixed; one the goal holds now says so.
    held = _held_now(ordered, declared - now)
    lines = [MethodsLine(record_id=r.id, seq=r.seq, kind=r.decision.kind,
                         sentence=_with(again.get(r.id, r.sentence), held.get(r.id)),
                         in_force=r.id in now, post_seal=bool(r.post_seal),
                         after_estimates=bool(getattr(r, "after_estimates", False)))
             for r in ordered if r.id in kept and r.sentence]
    return MethodsText(lines=lines, seen_from=seen, text=" ".join(line.sentence for line in lines))


def _held_now(ordered: Sequence[Any], ids: set[str]) -> dict[str, str]:
    """Of the records ``ids``, each the goal holds now (:func:`held_by_goal`), with the sentence
    that says so; not one reverted since, which the revert withdrew (W1-R round 3, SE-m5)."""
    if not ids:
        return {}
    try:
        slots = decisions.fold(ordered).model_dump()
        withdrawn = decisions.reverted(ordered)
    except Refusal:
        return {}
    return {r.id: not_applied_sentence(slots) for r in ordered
            if r.id in ids and r.id not in withdrawn and held_by_goal(r.decision, slots)}


def _with(sentence: str, added: str | None) -> str:
    return f"{sentence} {added}" if added and sentence else sentence


__all__ = ["APPLIES", "MethodsLine", "MethodsText", "held_by_goal", "in_force", "methods_text",
           "never_applied", "not_applied_sentence", "restated", "seen_from"]
