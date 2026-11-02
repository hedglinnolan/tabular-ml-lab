"""A revisit refused because a recorded answer reads the reading it would unsettle
(``decisions._answers_keep_settled_readings``, ``role_unconfirmed``) offers each route its words
name, as an exit (INBOX: run_30, run_33 and run_34 met "Change the models answer first, or keep
the confirmation" with "Keep the answers as they are" as its only exit, and left the codes-versus-
amount test of charlson_index untested).

The routes, by construction: the reading's other value confirmed in place of the one revisited
(whole numbers are codes or amounts, so the other of "code" is "amount"), which keeps the reading
settled, so the models answer still validates; the answer that reads it withdrawn first, a revert
of the record in force for its slot, after which the revisit is accepted; and the answers kept.
Only a route the record accepts is offered and named: where the other reading is itself refused
(a column of labels is no amount) or the withdrawal is (the seal keeps how each unit's rows are
combined), the words leave it out. And the withdrawal only where it withdraws: where an earlier
answer of the same slot is still live, reverting the one in force returns that earlier answer,
which reads the column as well (the pilot answered the models question twice in run_30 and three
times in run_35), so no withdrawal is offered or named. One confirmation that covered two columns
names both, and offers each column's other reading, each in place of that column's reading (where
only one column's is accepted, that one alone, still in place of its own reading: "in place of this
one", the confirmation revisited, only where the revisit unsettles one column). With
no answer reading the column, the revisit is accepted and nothing is offered.
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


def ctx(log: list[d.DecisionRecord], roles: dict[str, str] = ROLES,
        info: dict[str, Any] = INFO, **extra: Any) -> dict[str, Any]:
    return {"state": d.fold(log), "records": lambda: list(log), "columns": [*roles, "y"],
            "target": "y", "column_info": info, **extra}


def recorded(log: list[d.DecisionRecord], decision: dict[str, Any]) -> list[d.DecisionRecord]:
    """``log`` with ``decision`` recorded after it, checked as any answer is."""
    done = d.validate(decision, ctx(log))
    return [*log, rec(log[-1].seq + 1, done.model_dump(mode="json"))]


def plan(*models: dict[str, Any], roles: dict[str, str] = ROLES,
         reading: dict[str, Any] = CODES) -> list[d.DecisionRecord]:
    """A prediction plan reading `charlson_index` as codes (record 5, ``reading``), then
    ``models``."""
    head = [{"kind": "set_target", "column": "y"},
            {"kind": "set_task", "column": "y", "task": "regression"},
            {"kind": "set_purpose", "purpose": "prediction"},
            {"kind": "set_roles", "roles": roles}, reading, *models]
    return [rec(i, x) for i, x in enumerate(head, start=1)]


def refused(log: list[d.DecisionRecord], decision: dict[str, Any], **kw: Any) -> d.Refusal:
    with pytest.raises(d.Refusal) as caught:
        d.validate(decision, ctx(log, **kw))
    return caught.value


def offered(refusal: d.Refusal, given: dict[str, Any]) -> None:
    """Every way forward the refusal offers is accepted where it is offered."""
    for way in refusal.exits:
        if way["decision"] is not None:
            d.validate(way["decision"], given)


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
    """Two models answers, the later reverted: the record in force is the one no revert cancels
    (``decisions.fold``), the earlier (record 6), and the only one live, so reverting it leaves no
    models answer, and the revisit is then accepted."""
    two = {"kind": "select_models", "models": ["linear", "elastic_net"]}
    log = plan(LINEAR, two, {"kind": "revert", "decision_id": "r7"})
    assert d.fold(log).models == ["linear"]
    withdraw = refused(log, REVISIT).exits[1]
    assert withdraw == {"label": "Withdraw the models answer first",
                        "decision": {"kind": "revert", "decision_id": "r6"}}
    after = recorded(log, withdraw["decision"])
    assert d.fold(after).models is None
    d.validate(REVISIT, ctx(after))


TWO = {"kind": "select_models", "models": ["linear", "elastic_net"]}
NO_HOLDOUT = {"kind": "set_split", "holdout": 0.0}  # the readmission plan's (prereg.json)
SEEN = {"y": ["linear", "elastic_net"]}  # both families' cross-validated scores were shown


@pytest.mark.parametrize("models, extra", [
    ((LINEAR, TWO), {}),  # the verifier's server probe: `linear`, then the two families
    ((TWO, TWO, TWO), {}),  # run_35: the same answer three times
    ((NO_HOLDOUT, LINEAR, TWO), {"seen": SEEN}),  # run_30: twice, no rows held out, scores shown
], ids=["linear-then-two", "the-same-three-times", "no-holdout-scores-shown"])
def test_with_an_earlier_models_answer_live_no_withdrawal_is_offered_or_named(models, extra):
    """Reverting the models record in force where an earlier one is live returns that earlier
    answer (``decisions.fold`` keeps the latest live word), which reads `charlson_index` as well, so
    the revisit is refused again: by construction it withdraws nothing. The refusal offers no exit
    labeled as a withdrawal, and names none."""
    log = plan(*models)
    latest = log[-1].id
    returned = [*log, rec(log[-1].seq + 1, {"kind": "revert", "decision_id": latest})]
    assert d.fold(returned).models == d.fold(log[:-1]).models is not None
    with pytest.raises(d.Refusal):
        d.validate(REVISIT, ctx(returned, **extra))
    refusal = refused(log, REVISIT, **extra)
    assert refusal.message == (
        "The recorded models answer reads `charlson_index`, which would then rest on a reading "
        "nobody settled. Confirm the other reading of `charlson_index` in place of this one, or "
        "keep the confirmation.")
    from turbotab.core.readings import code_or_count_exits

    assert refusal.exits == [code_or_count_exits("charlson_index")[0], KEEP]
    for way in refusal.exits:
        if way["label"].startswith("Withdraw"):
            assert d.fold(recorded(log, way["decision"])).models is None
    offered(refusal, ctx(log, **extra))


def test_with_no_answer_reading_it_the_revisit_is_accepted():
    d.validate(REVISIT, ctx(plan()))


def test_a_withdrawal_the_engine_accepts_is_offered_under_the_readmission_plan():
    """One models answer (record 7) under the readmission plan, no rows held out and both
    families' scores shown: with no family selected after its revert none is left out
    (``selection._compared_families_stay`` is silent), so the revert is accepted, offered, and
    leaves no models answer."""
    log = plan(NO_HOLDOUT, TWO)
    given = ctx(log, seen=SEEN)
    refusal = refused(log, REVISIT, seen=SEEN)
    assert "withdraw the models answer first" in refusal.message
    assert refusal.exits[1] == {"label": "Withdraw the models answer first",
                                "decision": {"kind": "revert", "decision_id": "r7"}}
    offered(refusal, given)
    assert d.fold(recorded(log, refusal.exits[1]["decision"])).models is None


# run_34: one confirmation read two whole-number predictors as codes.
TWO_ROLES = {**ROLES, "prior_adm": "covariate"}
TWO_INFO = {**INFO, "prior_adm": {"dtype": "integer", "n_unique": 9}}
BOTH_CODES = {"kind": "confirm_readings", "items": [
    {"reading": "code_or_count", "column": "charlson_index", "value": "code"},
    {"reading": "code_or_count", "column": "prior_adm", "value": "code"}]}


def test_one_confirmation_of_two_columns_names_both_and_offers_each_other_reading():
    """Undoing one confirmation that read `charlson_index` and `prior_adm` as codes unsettles
    both: the refusal names both, and offers each column's other reading (whole numbers are
    codes or amounts, so the other of "code" is "amount"), each accepted, and each keeping the
    models answer valid, since the other column's reading still stands. Each replaces its
    column's code reading, not the confirmation revisited: after `charlson_index`'s, that
    confirmation still holds `prior_adm`'s, and the revisit is refused again naming it alone."""
    from turbotab.core.readings import code_or_count_exits

    log = plan(LINEAR, roles=TWO_ROLES, reading=BOTH_CODES)
    given = ctx(log, TWO_ROLES, TWO_INFO)
    refusal = refused(log, REVISIT, roles=TWO_ROLES, info=TWO_INFO)
    assert refusal.message == (
        "The recorded models answer reads `charlson_index` and `prior_adm`, which would then rest "
        "on readings nobody settled. Confirm the other reading of `charlson_index` or of "
        "`prior_adm` in place of its code reading, withdraw the models answer first, or keep the "
        "confirmation.")
    assert refusal.exits == [code_or_count_exits("charlson_index")[0],
                             code_or_count_exits("prior_adm")[0],
                             {"label": "Withdraw the models answer first",
                              "decision": {"kind": "revert", "decision_id": "r6"}}, KEEP]
    offered(refusal, given)
    for way in refusal.exits[:2]:
        after = [*log, rec(7, d.validate(way["decision"], given).model_dump(mode="json"))]
        d.validate(LINEAR, ctx(after, TWO_ROLES, TWO_INFO))
    after = [*log, rec(7, refusal.exits[0]["decision"])]
    again = refused(after, REVISIT, roles=TWO_ROLES, info=TWO_INFO)
    assert again.message.startswith("The recorded models answer reads `prior_adm`, which would "
                                    "then rest on a reading nobody settled.")


def test_two_columns_read_differently_are_each_named_in_place_of_their_own_reading():
    """One confirmation read `charlson_index` as codes and `prior_adm` as an amount: each column's
    other reading is the other value (an amount, codes), and the words name each in place of its
    own reading."""
    from turbotab.core.readings import code_or_count_exits

    mixed = {"kind": "confirm_readings", "items": [
        {"reading": "code_or_count", "column": "charlson_index", "value": "code"},
        {"reading": "code_or_count", "column": "prior_adm", "value": "amount"}]}
    log = plan(LINEAR, roles=TWO_ROLES, reading=mixed)
    refusal = refused(log, REVISIT, roles=TWO_ROLES, info=TWO_INFO)
    assert refusal.message == (
        "The recorded models answer reads `charlson_index` and `prior_adm`, which would then rest "
        "on readings nobody settled. Confirm the other reading of `charlson_index` in place of its "
        "code reading or of `prior_adm` in place of its amount reading, withdraw the models answer "
        "first, or keep the confirmation.")
    assert refusal.exits[:2] == [code_or_count_exits("charlson_index")[0],
                                 code_or_count_exits("prior_adm")[1]]
    offered(refusal, ctx(log, TWO_ROLES, TWO_INFO))


class Labels:
    """A table whose `charlson_index` holds labels (``DataStore.text_numbers``): read as amounts,
    every value would be blank, so the engine refuses that reading (``not_numbers``)."""

    def text_numbers(self, columns: list[str]) -> dict[str, Any]:
        return {c: {"labels": True, "evidence": "such as `mild`"} for c in columns
                if c == "charlson_index"}


def test_an_other_reading_the_engine_refuses_is_neither_offered_nor_named():
    """Where the other reading is itself refused (a column of labels is no amount), the refusal
    neither offers nor names it: the withdrawal and keeping remain, each accepted where offered.
    Its other direction is the first test (the other reading accepted, offered and named)."""
    log = plan(LINEAR)
    given = ctx(log, store=Labels)
    with pytest.raises(d.Refusal) as amount:
        d.validate(AMOUNT, given)
    assert amount.value.code == "not_numbers"
    refusal = refused(log, REVISIT, store=Labels)
    assert refusal.message == (
        "The recorded models answer reads `charlson_index`, which would then rest on a reading "
        "nobody settled. Withdraw the models answer first, or keep the confirmation.")
    assert refusal.exits == [{"label": "Withdraw the models answer first",
                              "decision": {"kind": "revert", "decision_id": "r6"}}, KEEP]
    offered(refusal, given)


def test_two_columns_named_with_one_other_reading_offered_name_it_in_place_of_its_own():
    """V3-2: one confirmation read `charlson_index` (labels, so no amount: :class:`Labels`) and
    `prior_adm` as codes. The revisit unsettles both, and only `prior_adm`'s other reading is
    accepted, so it alone is offered, and named in place of its own code reading, not in place of
    "this one", the confirmation revisited: by construction that confirmation still holds
    `charlson_index`'s reading after it, and the revisit is refused again naming that column
    alone. "In place of this one" is the words only where the revisit unsettles one column (the
    first test)."""
    from turbotab.core.readings import code_or_count_exits

    log = plan(LINEAR, roles=TWO_ROLES, reading=BOTH_CODES)
    given = ctx(log, TWO_ROLES, TWO_INFO, store=Labels)
    refusal = refused(log, REVISIT, roles=TWO_ROLES, info=TWO_INFO, store=Labels)
    assert refusal.message == (
        "The recorded models answer reads `charlson_index` and `prior_adm`, which would then rest "
        "on readings nobody settled. Confirm the other reading of `prior_adm` in place of its "
        "code reading, withdraw the models answer first, or keep the confirmation.")
    assert refusal.exits == [code_or_count_exits("prior_adm")[0],
                             {"label": "Withdraw the models answer first",
                              "decision": {"kind": "revert", "decision_id": "r6"}}, KEEP]
    offered(refusal, given)
    after = [*log, rec(7, d.validate(refusal.exits[0]["decision"], given).model_dump(mode="json"))]
    again = refused(after, REVISIT, roles=TWO_ROLES, info=TWO_INFO, store=Labels)
    assert again.message.startswith("The recorded models answer reads `charlson_index`, which "
                                    "would then rest on a reading nobody settled.")


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


SCREEN = {"kind": "set_exclusions", "rules": [GOLD]}
SCREENED = {"kind": "set_sensitivity", "analyses": [{"label": "screened", "rules": [GOLD]}]}


@pytest.mark.parametrize("answer, named", [(SCREEN, "exclusion rules"),
                                           (SCREENED, "sensitivity analyses")],
                         ids=["exclusions", "sensitivity"])
def test_a_screen_reading_the_unit_is_withdrawn_by_the_record_that_holds_it(answer, named):
    """The words name the answer that holds the screen, in both sentences (no exclusions answer is
    recorded when the sensitivity analyses hold it)."""
    log = screened(answer)
    undo = {"kind": "revert", "decision_id": "r2"}
    with pytest.raises(d.Refusal) as caught:
        d.validate(undo, bare(log))
    assert caught.value.message == (
        f"The recorded {named} read `weight`, which would then rest on a reading nobody "
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


TIMED = [rec(1, {"kind": "set_grain", "grain": "repeated", "id_column": "pid"}),
         rec(2, {"kind": "set_repeat_kind", "repeat_kind": "time_points"}),
         rec(3, {"kind": "set_unit", "unit": "unit"}),
         rec(4, {"kind": "confirm_reading", "reading": "time_column", "column": "visit_date",
                 "value": "orders"}),
         rec(5, {"kind": "set_aggregation", "method": "last", "outcome": "last"})]


def test_a_withdrawal_the_engine_refuses_is_neither_offered_nor_named():
    """The combining answer's one record, after the held-out rows were drawn (record 6): its
    revert changes what a row is, so the seal refuses it (``seal._revert_keeps_the_seal``), and the
    revisit's refusal neither offers nor names it. Its other direction, with no split drawn, is
    the test above."""
    from turbotab.core import seal  # noqa: F401 - registers the seal's checks on a revert

    log = [*TIMED, rec(6, {"kind": "set_split", "holdout": 0.2})]
    with pytest.raises(d.Refusal) as withdrawal:
        d.validate({"kind": "revert", "decision_id": "r5"}, bare(log))
    assert withdrawal.value.code == "sealed"
    with pytest.raises(d.Refusal) as caught:
        d.validate({"kind": "revert", "decision_id": "r4"}, bare(log))
    assert caught.value.message == (
        "The recorded combining answer reads `visit_date`, which would then rest on a reading "
        "nobody settled. Keep the confirmation.")
    assert caught.value.exits == [KEEP]


@pytest.mark.parametrize("log, revisited, named", [
    (screened(SCREEN)[:3] + [rec(4, SCREEN), rec(5, SCREEN)], "r2", "exclusion rules"),
    (screened(SCREENED)[:3] + [rec(4, SCREENED), rec(5, SCREENED)], "r2", "sensitivity analyses"),
    ([*TIMED, rec(6, TIMED[-1].decision.model_dump(mode="json"))], "r4", "combining answer"),
], ids=["exclusions", "sensitivity", "aggregation"])
def test_a_screen_or_combining_answer_given_twice_is_not_offered_as_withdrawn(log, revisited,
                                                                               named):
    """The same answer recorded twice: reverting the later returns the earlier, which reads the
    column as well (by construction the revisit is refused again after it), so no withdrawal is
    offered or named, for each slot the refusal can name beside the models answer."""
    undo = {"kind": "revert", "decision_id": revisited}
    returned = [*log, rec(log[-1].seq + 1, {"kind": "revert", "decision_id": log[-1].id})]
    with pytest.raises(d.Refusal) as again:
        d.validate(undo, bare(returned))
    assert again.value.code == "role_unconfirmed"
    with pytest.raises(d.Refusal) as caught:
        d.validate(undo, bare(log))
    assert caught.value.message.endswith("nobody settled. Keep the confirmation.")
    assert f"recorded {named} " in caught.value.message
    assert caught.value.exits == [KEEP]
