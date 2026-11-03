"""A withdrawn analysis-plan lock stays withdrawn (INBOX 1044; calm/FOUNDATION §7; plan_lock).

TurboTab withdraws a lock no estimate was shown under, on Cancel or a change to the plan, by
recording a revert of it. A revert of that withdrawal would bring the lock back over the plan as it
stands, which may have changed since, so estimates would be served under the old lock's plan and
fingerprint: it is refused, with pressing Fit as the way forward where a press would lock the plan
(under inference, while it is unlocked). A revert of any other revert is not the lock's business
and passes. A revert of a lock itself is refused while the log holds it unreverted, whether it is
the lock in force or not. Further down a chain of reverts the fold says what a revert does, not
the chain's length: one that would withdraw the lock in force (a revert of a reinstatement an
older engine accepted) is refused in words that do not say whether an estimate was shown under
it, and one that leaves the lock it reaches as it was (withdrawn by another revert too, or a lock
not in force) passes.
Where the log refuses the revert itself (a decision already reverted), its own refusal is said,
with only the exits the check accepts.

Each expectation is read from the log's own fold (``decisions.fold``, ``decisions.reverted``) and
its own refusal (``decisions._check_revert``, which ``DecisionLog.append`` runs), never from the
check under test: down a chain, a revert is refused here only where folding the log with it would
withdraw the lock in force (the newest unreverted lock) or bring a withdrawn lock back into the
record, and passes where the lock it reaches stays as it is.
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
    force again. A client's revert of that reinstatement would withdraw the lock in force: it is
    refused, since only TurboTab withdraws a lock, in words that do not say whether an estimate
    was shown under it (the log does not record it). Its counterpart one step further down a
    models answer's chain (r8 of r7 of r6' of r5) passes."""
    records = reinstated_log()
    assert decisions.fold(records).plan_locked is True  # back in force
    assert lock_in_force(records) == "r3"
    assert not locked_with(records, "r6")  # this revert would withdraw it again
    with pytest.raises(Refusal) as refused:
        check(records, "r6")
    assert refused.value.code == "lock_stays_in_force"
    message = refused.value.message
    assert message == ("This revert would withdraw the analysis-plan lock in force, which only "
                       "TurboTab does, and only before any estimate is shown under it: when Fit is "
                       "canceled or the plan is changed.")
    assert "displayed" not in message and "were seen" not in message
    assert refused.value.exits == []
    assert not any(w in message.lower() for w in plan_lock.NEVER_SAID)

    plain = [*withdrawn_log(), record(6, {"kind": "revert", "decision_id": "r5"}),
             record(7, {"kind": "revert", "decision_id": "r6"})]
    assert decisions.fold(plain).models == ["linear", "elastic_net"]
    assert decisions.fold([*plain, record(8, {"kind": "revert", "decision_id": "r7"})]
                          ).models == ["linear"]
    check(plain, "r7")  # no refusal: no lock is reached down this chain


# ── decided by the fold, not by the chain's length (verifier pp2-plan-lock-V2) ──


def reinstated_log() -> list[DecisionRecord]:
    """:func:`withdrawn_log`, then an older engine's acceptance of a client's revert of the
    withdrawal (r6 of r4): the lock r3 is in force again over the models changed since (r5)."""
    return [*withdrawn_log(), record(6, {"kind": "revert", "decision_id": "r4"})]


def withdrawn_again_log() -> list[DecisionRecord]:
    """:func:`reinstated_log`, then a change of the models (r8) before which TurboTab withdrew the
    reinstated lock again (r7 of r3): two reverts of the lock are live, r4 behind r6 and r7."""
    return [*reinstated_log(), record(7, {"kind": "revert", "decision_id": "r3"}, by="turbotab"),
            record(8, {"kind": "select_models", "models": ["elastic_net"]})]


def with_revert(records: list[DecisionRecord], target: str) -> list[DecisionRecord]:
    return [*records, record(max(r.seq for r in records) + 1,
                             {"kind": "revert", "decision_id": target})]


def lock_in_force(records: list[DecisionRecord]) -> str | None:
    """The newest lock the log's own fold leaves unreverted (``decisions.reverted``), by id."""
    withdrawn = decisions.reverted(records)
    live = [r.id for r in sorted(records, key=lambda r: r.seq)
            if r.decision.kind == "lock_plan" and r.id not in withdrawn]
    return live[-1] if live else None


def logs_refusal(records: list[DecisionRecord], target: str) -> Refusal:
    """The log's own refusal of a revert of ``target`` (``DecisionLog.append``'s check)."""
    with pytest.raises(Refusal) as refused:
        decisions._check_revert(records, decisions.Revert(decision_id=target))
    return refused.value


def test_a_revert_that_moves_no_lock_is_not_refused_whatever_its_chain():
    """The reinstated lock withdrawn again by TurboTab (r7) before a change: the client's revert of
    the reinstatement (r6) is three reverts above the lock, but r7 withdraws the lock either way,
    so no lock is in force before or after it, and the check is silent (C2). Its counterpart,
    with no second withdrawal, would withdraw the lock in force and is refused (C1, above)."""
    records = withdrawn_again_log()
    assert decisions.fold(records).plan_locked is None
    assert lock_in_force(records) is None and lock_in_force(with_revert(records, "r6")) is None
    assert decisions.fold(with_revert(records, "r6")).plan_locked is None
    check(records, "r6")  # no refusal

    records = reinstated_log()
    assert lock_in_force(records) == "r3" and lock_in_force(with_revert(records, "r6")) is None
    with pytest.raises(Refusal) as refused:
        check(records, "r6")
    assert refused.value.code == "lock_stays_in_force"


def test_a_chain_that_withdraws_a_lock_not_in_force_is_not_refused():
    """An older engine's log with two live locks: the first (r3) brought back by a reinstatement
    (r7 of r4) after a press recorded a newer one (r6). A client's revert of that reinstatement
    withdraws r3 again, and the lock in force, the newest, stays r6: silent. A revert of the newer
    lock's own withdrawal chain that would withdraw the lock in force is refused."""
    newer = {"purpose": "inference", "models": ["linear", "elastic_net"]}
    records = [*withdrawn_log(),
               record(6, {"kind": "lock_plan", "plan": newer, "digest": plan_lock.digest(newer)}),
               record(7, {"kind": "revert", "decision_id": "r4"})]
    assert lock_in_force(records) == "r6" and "r3" not in decisions.reverted(records)
    after = with_revert(records, "r7")
    assert lock_in_force(after) == "r6" and "r3" in decisions.reverted(after)
    check(records, "r7")  # no refusal: the lock in force does not move

    moved = [*records, record(8, {"kind": "revert", "decision_id": "r6"}, by="turbotab"),
             record(9, {"kind": "revert", "decision_id": "r8"})]  # r6 back by an older engine
    assert lock_in_force(moved) == "r6" and lock_in_force(with_revert(moved, "r9")) == "r3"
    with pytest.raises(Refusal) as refused:
        check(moved, "r9")
    assert refused.value.code == "lock_stays_in_force"


def test_a_revert_the_log_refuses_hears_the_logs_refusal_with_only_the_exits_accepted():
    """A second revert of the withdrawal (r4), already reverted by the reinstatement (r6): the log
    refuses it (``already_reverted``), and that refusal is said, not a lock's (C4). Its exit, the
    revert of r6, would withdraw the lock in force, which is refused, so it is not offered. Where
    the lock was withdrawn again (r7) the same exit moves no lock and is offered; where r4 is not
    yet reverted the lock's own refusal is said (``lock_stays_withdrawn``, above)."""
    records = reinstated_log()
    logs = logs_refusal(records, "r4")
    assert logs.code == "already_reverted"
    assert [e["decision"] for e in logs.exits] == [{"kind": "revert", "decision_id": "r6"}]
    assert lock_in_force(records) == "r3" and lock_in_force(with_revert(records, "r6")) is None
    with pytest.raises(Refusal) as refused:
        check(records, "r4")
    assert (refused.value.code, refused.value.message) == (logs.code, logs.message)
    assert refused.value.exits == []

    records = withdrawn_again_log()
    logs = logs_refusal(records, "r4")
    assert logs.code == "already_reverted"
    assert lock_in_force(with_revert(records, "r6")) is None  # its exit moves no lock
    check(records, "r6")
    with pytest.raises(Refusal) as refused:
        check(records, "r4")
    assert (refused.value.code, refused.value.message) == (logs.code, logs.message)
    assert refused.value.exits == logs.exits

    records = withdrawn_log()
    assert "r4" not in decisions.reverted(records)  # the log takes this one
    with pytest.raises(Refusal) as refused:
        check(records, "r4")
    assert refused.value.code == "lock_stays_withdrawn"


def test_a_revert_of_a_withdrawn_lock_hears_the_logs_refusal():
    """A revert of the lock TurboTab already withdrew (r3, by r4) is the log's to refuse
    (``already_reverted``); its exit, the revert of the withdrawal, would bring the lock back, which
    is refused, so it is not offered. A revert of a lock the log holds unreverted is refused as a
    revert of the lock, whether it is the lock in force or not (``plan_stays_locked``)."""
    records = withdrawn_log()
    logs = logs_refusal(records, "r3")
    assert logs.code == "already_reverted"
    assert [e["decision"] for e in logs.exits] == [{"kind": "revert", "decision_id": "r4"}]
    assert lock_in_force(records) is None and lock_in_force(with_revert(records, "r4")) == "r3"
    with pytest.raises(Refusal) as refused:
        check(records, "r3")
    assert (refused.value.code, refused.value.message) == (logs.code, logs.message)
    assert refused.value.exits == []

    newer = {"purpose": "inference", "models": ["linear", "elastic_net"]}
    records = [*withdrawn_log(),
               record(6, {"kind": "lock_plan", "plan": newer, "digest": plan_lock.digest(newer)}),
               record(7, {"kind": "revert", "decision_id": "r4"})]
    assert lock_in_force(records) == "r6" and "r3" not in decisions.reverted(records)
    for target in ("r3", "r6"):
        with pytest.raises(Refusal) as refused:
            check(records, target)
        assert refused.value.code == "plan_stays_locked", target
