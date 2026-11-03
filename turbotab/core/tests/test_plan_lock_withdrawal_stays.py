"""A withdrawn analysis-plan lock stays withdrawn (INBOX 1044; calm/FOUNDATION §7; plan_lock).

TurboTab withdraws a lock no estimate was shown under, on Cancel or a change to the plan, by
recording a revert of it. A revert of that withdrawal would bring the lock back over the plan as it
stands, which may have changed since, so estimates would be served under the old lock's plan and
fingerprint: it is refused, with pressing Fit as the way forward where a press would lock the plan
(under inference, while it is unlocked). A revert of any other revert is not the lock's business
and passes. Further down the chain the parity says what a revert does: one that would withdraw a
lock again (a reinstatement an older engine accepted) is refused as a revert of the lock itself is.

Each expectation is read from the log's own fold (``decisions.fold``, ``decisions.reverted``),
never from the check under test: a revert is refused here only where folding the log with it would
change whether a lock is in force or bring a withdrawn lock back into the record, and passes where
the fold's lock stays as it is.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from turbotab.core import decisions, fit_press, plan_lock
from turbotab.core.decisions import DecisionRecord, Refusal, parse_decision

T0 = datetime(2026, 11, 3, 9, 0, tzinfo=timezone.utc)
PLAN = {"purpose": "inference", "models": ["linear"]}


def record(seq: int, decision: dict, by: str = "you") -> DecisionRecord:
    return DecisionRecord(id=f"r{seq}", seq=seq, at=T0 + timedelta(minutes=seq),
                          decision=parse_decision(decision), recorded_by=by)


def withdrawn_log(purpose: str = "inference") -> list[DecisionRecord]:
    """A lock (r3), TurboTab's withdrawal of it (r4) when the models changed (r5)."""
    return [record(1, {"kind": "set_purpose", "purpose": "inference"}),
            record(2, {"kind": "select_models", "models": ["linear"]}),
            record(3, {"kind": "lock_plan", "plan": PLAN, "digest": plan_lock.digest(PLAN)}),
            record(4, {"kind": "revert", "decision_id": "r3"}, by="turbotab"),
            record(5, {"kind": "select_models", "models": ["linear", "elastic_net"]}),
            *([record(6, {"kind": "set_purpose", "purpose": purpose})]
              if purpose != "inference" else [])]


def locked_with(records: list[DecisionRecord], target: str) -> bool:
    """Whether folding the log with a revert of ``target`` leaves a lock in force (the fold's own
    reading of the log, independent of the check)."""
    probe = DecisionRecord(id="probe", seq=max(r.seq for r in records) + 1, at=T0 + timedelta(1),
                           decision=parse_decision({"kind": "revert", "decision_id": target}))
    return bool(decisions.fold([*records, probe]).plan_locked)


def check(records: list[DecisionRecord], target: str) -> None:
    decisions.validate({"kind": "revert", "decision_id": target},
                       {"records": records, "state": decisions.fold(records)})


def test_a_revert_of_turbotabs_withdrawal_of_a_lock_is_refused():
    records = withdrawn_log()
    assert decisions.fold(records).plan_locked is None  # withdrawn
    assert locked_with(records, "r4")  # the revert would bring the lock back
    assert plan_lock.current_lock(records) is None
    with pytest.raises(Refusal) as refused:
        check(records, "r4")
    assert refused.value.code == "lock_stays_withdrawn"
    message = refused.value.message
    assert message.startswith("This revert would bring back an analysis-plan lock that was "
                              "withdrawn, and a withdrawn lock is not brought back")
    assert "the record keeps it, with its time and fingerprint" in message
    assert ("Under inference, a press of Fit that is accepted while the plan is unlocked locks "
            "the plan as it then stands.") in message
    assert [e["label"] for e in refused.value.exits] == [fit_press.PRESS_FIT]
    assert refused.value.exits[0]["decision"] is None  # a job command, not a decision
    assert not any(w in message.lower() for w in plan_lock.NEVER_SAID)


def test_under_prediction_the_withdrawal_stays_and_no_press_is_offered_as_a_lock():
    """The goal changed to prediction after the withdrawal: the revert is refused just the same
    (it would bring a lock back under prediction), and pressing Fit is not offered as the way to
    lock, since under prediction a press locks nothing (``press_fit``)."""
    records = withdrawn_log("prediction")
    assert decisions.fold(records).purpose == "prediction"
    assert locked_with(records, "r4")
    with pytest.raises(Refusal) as refused:
        check(records, "r4")
    assert refused.value.code == "lock_stays_withdrawn"
    assert refused.value.exits == []


def test_with_a_newer_lock_in_force_the_withdrawal_stays_and_no_press_is_offered_as_a_lock():
    """Fit pressed again after the withdrawal recorded a newer lock (r6), so the plan is locked. The
    revert of the withdrawal (r4) is refused just the same: the lock in force stays r6, but the
    revert would bring the withdrawn lock (r3) back into the record as never withdrawn. Pressing
    Fit is not offered as the way forward, since a press records a lock only while the plan is
    unlocked (``press_fit``); here the fold holds it locked."""
    newer = {"purpose": "inference", "models": ["linear", "elastic_net"]}
    records = [*withdrawn_log(),
               record(6, {"kind": "lock_plan", "plan": newer, "digest": plan_lock.digest(newer)})]
    state = decisions.fold(records)
    assert state.purpose == "inference" and state.plan_locked is True
    assert locked_with(records, "r4")
    probe = DecisionRecord(id="probe", seq=7, at=T0 + timedelta(1),
                           decision=parse_decision({"kind": "revert", "decision_id": "r4"}))
    assert decisions.reverted(records)["r3"] == "r4"  # withdrawn now
    assert "r3" not in decisions.reverted([*records, probe])  # the revert would bring it back
    with pytest.raises(Refusal) as refused:
        check(records, "r4")
    assert refused.value.code == "lock_stays_withdrawn"
    assert refused.value.exits == []


def test_a_revert_of_any_other_revert_is_not_the_locks_business():
    """The guard stays silent where the lock in force does not move: the revert of a revert of a
    models answer (r7 of r6 of r5), made under a withdrawn lock, leaves it withdrawn."""
    records = [*withdrawn_log(), record(6, {"kind": "revert", "decision_id": "r5"})]
    assert not locked_with(records, "r6")
    check(records, "r6")  # no refusal
    records.append(record(7, {"kind": "revert", "decision_id": "r6"}))
    assert not locked_with(records, "r7")
    check(records, "r7")


def test_a_revert_of_a_lock_itself_is_refused_as_before():
    records = withdrawn_log()[:3]
    assert decisions.fold(records).plan_locked is True and not locked_with(records, "r3")
    with pytest.raises(Refusal) as refused:
        check(records, "r3")
    assert refused.value.code == "plan_stays_locked"


def test_a_lock_an_older_engine_brought_back_is_not_withdrawn_by_a_client_either():
    """An older engine accepted the revert of a withdrawal (r6 of r4), so its log holds the lock in
    force again. A client's revert of that reinstatement would withdraw the lock: it is refused as
    a revert of the lock itself is, since only TurboTab withdraws a lock. Its counterpart one step
    further down a models answer's chain (r8 of r7 of r6' of r5) passes."""
    records = [*withdrawn_log(), record(6, {"kind": "revert", "decision_id": "r4"})]
    assert decisions.fold(records).plan_locked is True  # back in force
    assert not locked_with(records, "r6")  # this revert would withdraw it again
    with pytest.raises(Refusal) as refused:
        check(records, "r6")
    assert refused.value.code == "plan_stays_locked"

    plain = [*withdrawn_log(), record(6, {"kind": "revert", "decision_id": "r5"}),
             record(7, {"kind": "revert", "decision_id": "r6"})]
    assert decisions.fold(plain).models == ["linear", "elastic_net"]
    assert decisions.fold([*plain, record(8, {"kind": "revert", "decision_id": "r7"})]
                          ).models == ["linear"]
    check(plain, "r7")  # no refusal: no lock is reached down this chain
