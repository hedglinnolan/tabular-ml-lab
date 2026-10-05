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
  stated, not folded away.
"""
from __future__ import annotations

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


def methods_text(records: Sequence[Any]) -> MethodsText:
    """The methods text of a decision log (see the module)."""
    ordered = sorted(records, key=lambda r: r.seq)
    now = in_force(ordered)
    seen = seen_from(ordered)
    kept = set(now)
    if seen is not None:
        kept |= in_force([r for r in ordered if r.seq <= seen])
        kept |= {r.id for r in ordered if r.seq > seen}
    lines = [MethodsLine(record_id=r.id, seq=r.seq, kind=r.decision.kind, sentence=r.sentence,
                         in_force=r.id in now, post_seal=bool(r.post_seal),
                         after_estimates=bool(getattr(r, "after_estimates", False)))
             for r in ordered if r.id in kept and r.sentence]
    return MethodsText(lines=lines, seen_from=seen, text=" ".join(line.sentence for line in lines))


__all__ = ["APPLIES", "MethodsLine", "MethodsText", "in_force", "methods_text", "seen_from"]
