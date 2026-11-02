"""A revisit refused because a recorded answer reads the reading it would unsettle
(``decisions._answers_keep_settled_readings``, ``role_unconfirmed``) offers each route its words
name, as an exit (INBOX: run_30, run_33 and run_34 met "Change the models answer first, or keep
the confirmation" with "Keep the answers as they are" as its only exit, and left the codes-versus-
amount test of charlson_index untested).

The routes, by construction: the reading's other value confirmed in place of the one revisited
(whole numbers are codes or amounts, so the other of "code" is "amount"), which keeps the reading
settled, so the models answer still validates; the answer that reads it withdrawn first, a revert
of the record in force for its slot, after which the revisit is accepted; and the answers kept.
With no answer reading the column, the revisit is accepted and nothing is offered.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from turbotab.core import decisions as d

ROLES = {"charlson_index": "covariate", "age": "exposure"}
# Seven whole-number values: the fit asks whether they are codes or amounts (``code_question``).
INFO = {"charlson_index": {"dtype": "integer", "n_unique": 7},
        "age": {"dtype": "float", "n_unique": 80}}
CODES = {"kind": "confirm_reading", "reading": "code_or_count", "column": "charlson_index",
         "value": "code"}
AMOUNT = {**CODES, "value": "amount"}
LINEAR = {"kind": "select_models", "models": ["linear"]}
KEEP = {"label": "Keep the answers as they are", "decision": None}


def rec(seq: int, decision: dict[str, Any]) -> d.DecisionRecord:
    return d.DecisionRecord(id=f"r{seq}", seq=seq, at=datetime(2026, 11, 1, tzinfo=timezone.utc),
                            decision=d.parse_decision(decision))


def ctx(log: list[d.DecisionRecord]) -> dict[str, Any]:
    return {"state": d.fold(log), "records": lambda: list(log), "columns": [*ROLES, "y"],
            "target": "y", "column_info": INFO}


def recorded(log: list[d.DecisionRecord], decision: dict[str, Any]) -> list[d.DecisionRecord]:
    """``log`` with ``decision`` recorded after it, checked as any answer is."""
    done = d.validate(decision, ctx(log))
    return [*log, rec(log[-1].seq + 1, done.model_dump(mode="json"))]


def plan(*models: dict[str, Any]) -> list[d.DecisionRecord]:
    """A prediction plan reading `charlson_index` as codes (record 5), then ``models``."""
    head = [{"kind": "set_target", "column": "y"},
            {"kind": "set_task", "column": "y", "task": "regression"},
            {"kind": "set_purpose", "purpose": "prediction"},
            {"kind": "set_roles", "roles": ROLES}, CODES, *models]
    return [rec(i, x) for i, x in enumerate(head, start=1)]


def refused(log: list[d.DecisionRecord], decision: dict[str, Any]) -> d.Refusal:
    with pytest.raises(d.Refusal) as caught:
        d.validate(decision, ctx(log))
    return caught.value


REVISIT = {"kind": "revert", "decision_id": "r5"}


def test_the_refusal_offers_the_other_reading_the_withdrawal_and_keeping_and_says_each():
    refusal = refused(plan(LINEAR), REVISIT)
    assert refusal.code == "role_unconfirmed"
    assert refusal.message == (
        "The recorded models answer reads `charlson_index`, which would then rest on a reading "
        "nobody settled. Confirm the other reading of `charlson_index` in place of this one, "
        "withdraw the models answer first, or keep the confirmation.")
    other, withdraw, keep = refusal.exits
    # the ask's own exit for the reading (its label and decision), as the fit would ask it
    from turbotab.core.readings import code_or_count_exits

    assert other["decision"] == AMOUNT and other == code_or_count_exits("charlson_index")[0]
    assert withdraw == {"label": "Withdraw the models answer first",
                        "decision": {"kind": "revert", "decision_id": "r6"}}
    assert keep == KEEP


def test_the_other_reading_keeps_it_settled_and_the_models_answer_still_validates():
    log = plan(LINEAR)
    other = refused(log, REVISIT).exits[0]["decision"]
    after = recorded(log, other)
    assert d.fold(after).shape_confirmations == {"code_or_count:charlson_index": "amount"}
    assert d.fold(after).models == ["linear"]
    d.validate(LINEAR, ctx(after))
    # The check bites: with the reading never confirmed the models answer is refused.
    unread = [r for r in plan() if r.id != "r5"]
    assert refused(unread, LINEAR).code == "reading_unsettled"


def test_withdrawing_the_models_answer_leaves_none_and_the_revisit_is_then_accepted():
    log = plan(LINEAR)
    withdraw = refused(log, REVISIT).exits[1]["decision"]
    after = recorded(log, withdraw)
    assert d.fold(after).models is None
    d.validate(REVISIT, ctx(after))


def test_the_withdrawal_names_the_models_record_in_force():
    """Two models answers, the earlier reverted then answered again: the record in force is the
    latest no revert cancels (``decisions.fold``)."""
    two = {"kind": "select_models", "models": ["linear", "elastic_net"]}
    log = plan(LINEAR, two, {"kind": "revert", "decision_id": "r7"})
    assert d.fold(log).models == ["linear"]
    assert refused(log, REVISIT).exits[1]["decision"] == {"kind": "revert", "decision_id": "r6"}
    log = plan(LINEAR, two)
    assert refused(log, REVISIT).exits[1]["decision"] == {"kind": "revert", "decision_id": "r7"}


def test_with_no_answer_reading_it_the_revisit_is_accepted():
    d.validate(REVISIT, ctx(plan()))


GOLD = {"kind": "goldberg", "column": "energy_kcal", "days": 1, "sex": "sex", "female": ["F"],
        "male": ["M"], "age": "age", "weight": "weight", "equation": "schofield", "pal": 1.55,
        "reason": "implausible energy reports"}
BODY_ROLES = {"energy_kcal": "energy", "age": "covariate", "weight": "covariate",
              "sex": "covariate"}


def screened(answer: dict[str, Any]) -> list[d.DecisionRecord]:
    """The weight read in kg (record 2) and a Goldberg screen reading it, by ``answer``."""
    return [rec(1, {"kind": "set_roles", "roles": BODY_ROLES}),
            rec(2, {"kind": "confirm_reading", "reading": "unit", "column": "weight",
                    "value": "kg"}),
            rec(3, {"kind": "confirm_reading", "reading": "unit", "column": "age",
                    "value": "years"}),
            rec(4, answer)]


def bare(log: list[d.DecisionRecord]) -> dict[str, Any]:
    return {"state": d.fold(log), "records": lambda: list(log)}


@pytest.mark.parametrize("answer, named", [
    ({"kind": "set_exclusions", "rules": [GOLD]}, "exclusion rules"),
    ({"kind": "set_sensitivity", "analyses": [{"label": "screened", "rules": [GOLD]}]},
     "sensitivity analyses")], ids=["exclusions", "sensitivity"])
def test_a_screen_reading_the_unit_is_withdrawn_by_the_record_that_holds_it(answer, named):
    log = screened(answer)
    undo = {"kind": "revert", "decision_id": "r2"}
    with pytest.raises(d.Refusal) as caught:
        d.validate(undo, bare(log))
    assert caught.value.message == (
        f"The recorded exclusion rules reads `weight`, which would then rest on a reading nobody "
        f"settled. Withdraw the {named} first, or keep the confirmation.")
    withdraw, keep = caught.value.exits
    assert withdraw == {"label": f"Withdraw the {named} first",
                        "decision": {"kind": "revert", "decision_id": "r4"}}
    assert keep == KEEP
    d.validate(withdraw["decision"], bare(log))
    log = [*log, rec(5, withdraw["decision"])]
    d.validate(undo, bare(log))


def test_the_combining_answer_reading_the_order_is_withdrawn_by_its_record():
    log = [rec(1, {"kind": "set_grain", "grain": "repeated", "id_column": "pid"}),
           rec(2, {"kind": "set_repeat_kind", "repeat_kind": "time_points"}),
           rec(3, {"kind": "set_unit", "unit": "unit"}),
           rec(4, {"kind": "confirm_reading", "reading": "time_column", "column": "visit_date",
                   "value": "orders"}),
           rec(5, {"kind": "set_aggregation", "method": "last", "outcome": "last"})]
    undo = {"kind": "revert", "decision_id": "r4"}
    with pytest.raises(d.Refusal) as caught:
        d.validate(undo, bare(log))
    assert caught.value.message == (
        "The recorded combining answer reads `visit_date`, which would then rest on a reading "
        "nobody settled. Withdraw the combining answer first, or keep the confirmation.")
    withdraw, keep = caught.value.exits
    assert withdraw["decision"] == {"kind": "revert", "decision_id": "r5"} and keep == KEEP
    d.validate(withdraw["decision"], bare(log))
    log = [*log, rec(6, withdraw["decision"])]
    d.validate(undo, bare(log))
