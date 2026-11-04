"""A revisit refused because a recorded answer reads the role it would unsettle
(``decisions._answers_keep_settled_roles``, ``role_unconfirmed``) offers each route its words name,
as an exit, as its readings sibling does (``test_revisit_offers_the_routes_it_names``). INBOX
11.1039: it said "Change the <answer> first, or keep its confirmation" with "Keep the answers as
they are" its only exit, and "The recorded exclusion rules reads `x`".

The routes, by construction: the answer that reads the role withdrawn first, a revert of the record
in force for its slot, after which the revisit is accepted; and the answers kept. That revert is
offered and named only where, after it, no answer of the slot reads a role the revisit unsettles,
and the record accepts it (the seal keeps the exclusion rules once the held-out rows are drawn).
Where it returns an earlier answer of the slot that reads none of them (the Router records each
slot's first answer, the card's adjustment, no rules, the sample's estimate, before any change), it
is offered as a return to that earlier answer; where the earlier answer reads one as well, it is
not offered. The words agree with the answer's name and the columns' count ("The recorded
exclusion rules read"; "roles nobody confirmed"), and keeping names what the
revisit undoes: a confirmation, or (reverting the user's own roles answer, which settled the role
without a confirmation, or a bulk roles answer that would carry a settled role along) the answers
as they are. With no answer reading the role, the revisit is accepted and nothing is offered.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from turbotab.core import decisions as d

ROLES = {"kcal": "energy", "fat_g": "exposure", "ALC": "exposure", "age": "covariate"}
ADJ = {"kind": "set_energy_adjustment", "method": "standard", "energy_column": "kcal",
       "nutrients": ["fat_g", "ALC"]}
SCREEN = {"kind": "set_exclusions", "rules": [{"column": "kcal", "low": 500, "high": 5000,
                                               "reason": "implausible energy reports"}]}
KEEP = {"label": "Keep the answers as they are", "decision": None}
REVISIT = {"kind": "revert", "decision_id": "r3"}


def rec(seq: int, decision: dict[str, Any]) -> d.DecisionRecord:
    return d.DecisionRecord(id=f"r{seq}", seq=seq, at=datetime(2026, 11, 1, tzinfo=timezone.utc),
                            decision=d.parse_decision(decision))


def ctx(log: list[d.DecisionRecord], roles: dict[str, str] = ROLES, **extra: Any
        ) -> dict[str, Any]:
    return {"state": d.fold(log), "records": lambda: list(log), "columns": [*roles, "y"],
            "target": "y", **extra}


def confirmed(*answers: dict[str, Any], rode: tuple[str, ...] = ("ALC",),
              roles: dict[str, str] = ROLES,
              confirmation: dict[str, Any] | None = None) -> list[d.DecisionRecord]:
    """A bulk roles answer that carried ``rode`` below high (record 2), their confirmation on its
    own (record 3), then ``answers``."""
    one = {"kind": "confirm_reading", "reading": "role", "column": rode[0],
           "value": roles[rode[0]]}
    head = [{"kind": "set_target", "column": "y"},
            {"kind": "set_roles", "roles": roles, "unconfirmed": list(rode)},
            confirmation or one, *answers]
    return [rec(i, x) for i, x in enumerate(head, start=1)]


def refused(log: list[d.DecisionRecord], decision: dict[str, Any], **kw: Any) -> d.Refusal:
    with pytest.raises(d.Refusal) as caught:
        d.validate(decision, ctx(log, **kw))
    return caught.value


def taken(log: list[d.DecisionRecord], decision: Any, **kw: Any) -> list[d.DecisionRecord]:
    """``log`` with ``decision`` recorded after it, checked as any answer is."""
    done = d.validate(decision, ctx(log, **kw))
    return [*log, rec(log[-1].seq + 1, done.model_dump(mode="json"))]


def test_the_refusal_offers_the_withdrawal_and_keeping_and_says_each():
    log = confirmed(ADJ)
    refusal = refused(log, REVISIT)
    assert refusal.code == "role_unconfirmed"
    assert refusal.message == (
        "The recorded energy adjustment reads `ALC`, which would then be a role nobody "
        "confirmed. Withdraw the energy adjustment first, or keep its confirmation.")
    withdraw, keep = refusal.exits
    assert withdraw == {"label": "Withdraw the energy adjustment first",
                        "decision": {"kind": "revert", "decision_id": "r4"}}
    assert keep == KEEP


def test_withdrawing_the_energy_adjustment_leaves_none_and_the_revisit_is_then_accepted():
    log = confirmed(ADJ)
    withdraw = refused(log, REVISIT).exits[0]
    assert withdraw["label"] == "Withdraw the energy adjustment first"
    after = taken(log, withdraw["decision"])
    assert d.fold(after).energy_adjustment is None
    d.validate(REVISIT, ctx(after))
    # The role is unsettled once the revisit is recorded: by construction the answer read it.
    from turbotab.core.readings import unsettled

    assert unsettled(d.fold(taken(after, REVISIT)), ["ALC"]) == ["ALC"]


def test_two_columns_agree_in_number():
    both = {"kind": "confirm_readings", "items": [
        {"reading": "role", "column": "fat_g", "value": "exposure"},
        {"reading": "role", "column": "ALC", "value": "exposure"}]}
    log = confirmed(ADJ, rode=("fat_g", "ALC"), confirmation=both)
    refusal = refused(log, REVISIT)
    assert refusal.message == (
        "The recorded energy adjustment reads `fat_g` and `ALC`, which would then be roles "
        "nobody confirmed. Withdraw the energy adjustment first, or keep their confirmation.")
    assert refusal.exits == [{"label": "Withdraw the energy adjustment first",
                              "decision": {"kind": "revert", "decision_id": "r4"}}, KEEP]


def test_the_exclusion_rules_read_the_energy_column_and_are_withdrawn_by_their_record():
    log = confirmed(SCREEN, rode=("kcal",))
    refusal = refused(log, REVISIT)
    assert refusal.message == (
        "The recorded exclusion rules read `kcal`, which would then be a role nobody confirmed. "
        "Withdraw the exclusion rules first, or keep its confirmation.")
    assert refusal.exits == [{"label": "Withdraw the exclusion rules first",
                              "decision": {"kind": "revert", "decision_id": "r4"}}, KEEP]
    after = taken(log, refusal.exits[0]["decision"])
    assert d.fold(after).exclusions is None
    d.validate(REVISIT, ctx(after))


SURVEY_ROLES = {"wt": "design", "age": "covariate"}
SURVEY = {"kind": "set_survey", "estimand": "population", "weight": "wt"}


def test_the_survey_answer_reads_its_weight_and_is_withdrawn_by_its_record():
    log = confirmed(SURVEY, rode=("wt",), roles=SURVEY_ROLES)
    refusal = refused(log, REVISIT, roles=SURVEY_ROLES)
    assert refusal.message == (
        "The recorded survey answer reads `wt`, which would then be a role nobody confirmed. "
        "Withdraw the survey answer first, or keep its confirmation.")
    assert refusal.exits == [{"label": "Withdraw the survey answer first",
                              "decision": {"kind": "revert", "decision_id": "r4"}}, KEEP]
    after = taken(log, refusal.exits[0]["decision"], roles=SURVEY_ROLES)
    assert d.fold(after).survey is None
    d.validate(REVISIT, ctx(after, SURVEY_ROLES))


@pytest.mark.parametrize("answer, slot", [(ADJ, "energy_adjustment"), (SCREEN, "exclusions")],
                         ids=["energy-adjustment", "exclusions"])
def test_an_answer_given_twice_is_not_offered_as_withdrawn(answer, slot):
    """The same answer recorded twice: reverting the later returns the earlier, which reads the
    column as well (by construction the revisit is refused again after it), so no withdrawal is
    offered or named."""
    rode = ("ALC",) if slot == "energy_adjustment" else ("kcal",)
    log = confirmed(answer, answer, rode=rode)
    returned = [*log, rec(6, {"kind": "revert", "decision_id": "r5"})]
    assert getattr(d.fold(returned), slot) is not None
    assert refused(returned, REVISIT).code == "role_unconfirmed"
    refusal = refused(log, REVISIT)
    assert refusal.message.endswith("would then be a role nobody confirmed. Keep its "
                                    "confirmation.")
    assert refusal.exits == [KEEP]


# An earlier answer of the slot that reads no role the revisit unsettles, then the answer that
# reads it (record 5): the A2 InBody card's adjustment without `ALC`, then one adding it; no
# adjustment, then one; the sample's estimate, then the population's; no exclusion rules, then a
# rule on `kcal`; a rule on `age`, then one on `kcal` beside it (the Router records each slot's
# first answer before any change). The return names the earlier answer by what it does where it
# records no adjustment or no rule (R1-P2, repair round 2: "Keep every row" records no rules, and
# "Do not adjust for energy" the method "none", so no earlier rules or adjustment exist to return
# to), and as the earlier answer of its name where it records one.
WITHOUT_ALC = {"kind": "set_energy_adjustment", "method": "standard", "energy_column": "kcal",
               "nutrients": ["fat_g"]}
ADULTS = {"column": "age", "low": 18, "high": 90, "reason": "adults"}
EARLIER = {
    "an-adjustment-without-the-role": (
        ROLES, "ALC", WITHOUT_ALC, ADJ, "energy_adjustment",
        "The recorded energy adjustment reads `ALC`, which would then be a role nobody confirmed. "
        "Return to the earlier energy adjustment first, or keep its confirmation.",
        "Return to the earlier energy adjustment first"),
    "no-adjustment": (
        ROLES, "ALC", {"kind": "set_energy_adjustment", "method": "none"}, ADJ,
        "energy_adjustment",
        "The recorded energy adjustment reads `ALC`, which would then be a role nobody confirmed. "
        "Return to no energy adjustment first, or keep its confirmation.",
        "Return to no energy adjustment first"),
    "the-sample-s-estimate": (
        SURVEY_ROLES, "wt", {"kind": "set_survey", "estimand": "sample"}, SURVEY, "survey",
        "The recorded survey answer reads `wt`, which would then be a role nobody confirmed. "
        "Return to the earlier survey answer first, or keep its confirmation.",
        "Return to the earlier survey answer first"),
    "no-rules": (
        ROLES, "kcal", {"kind": "set_exclusions", "rules": []}, SCREEN, "exclusions",
        "The recorded exclusion rules read `kcal`, which would then be a role nobody confirmed. "
        "Return to keeping every row first, or keep its confirmation.",
        "Return to keeping every row first"),
    "a-rule-on-another-column": (
        ROLES, "kcal", {"kind": "set_exclusions", "rules": [ADULTS]},
        {"kind": "set_exclusions", "rules": [ADULTS, *SCREEN["rules"]]}, "exclusions",
        "The recorded exclusion rules read `kcal`, which would then be a role nobody confirmed. "
        "Return to the earlier exclusion rules first, or keep its confirmation.",
        "Return to the earlier exclusion rules first"),
}


@pytest.mark.parametrize("roles, column, earlier, later, slot, message, label",
                         list(EARLIER.values()), ids=list(EARLIER))
def test_an_earlier_answer_that_reads_no_unsettled_role_is_offered_to_return_to(
        roles, column, earlier, later, slot, message, label):
    """Reverting the answer in force (record 5) returns the earlier one (record 4), which reads no
    role the revisit unsettles, so the revert is offered and named for what it does: a return to
    that earlier answer, not a withdrawal (the slot still holds an answer), named by what it does
    where it records no rule or no adjustment, and as the earlier answer where it records one. By
    construction, the state after it holds the earlier answer as if the later had never been
    recorded, and the revisit is then accepted, leaving the role unsettled. Its other direction,
    an earlier answer that reads the role as well, is the test above (nothing offered)."""
    from turbotab.core.readings import unsettled

    log = confirmed(earlier, later, rode=(column,), roles=roles)
    held = getattr(d.fold(confirmed(earlier, rode=(column,), roles=roles)), slot)
    nothing = held == [] or getattr(held, "method", None) == "none"
    assert label.startswith("Return to the earlier ") != nothing
    refusal = refused(log, REVISIT, roles=roles)
    assert refusal.code == "role_unconfirmed"
    assert refusal.message == message
    assert refusal.exits == [{"label": label, "decision": {"kind": "revert", "decision_id": "r5"}},
                             KEEP]
    after = taken(log, refusal.exits[0]["decision"], roles=roles)
    never = confirmed(earlier, rode=(column,), roles=roles)
    assert getattr(d.fold(after), slot) is not None
    assert getattr(d.fold(after), slot) == getattr(d.fold(never), slot)
    revisited = taken(after, REVISIT, roles=roles)
    assert unsettled(d.fold(revisited), [column]) == [column]


def test_an_earlier_answer_that_reads_another_unsettled_role_is_not_offered():
    """One confirmation settled `fat_g` and `ALC` (record 3); the earlier adjustment (record 4)
    reads `fat_g`, the later (record 5) both. Returning to the earlier still leaves `fat_g` a role
    nobody confirmed (by construction the revisit is refused again after it), so no return is
    offered or named."""
    both = {"kind": "confirm_readings", "items": [
        {"reading": "role", "column": "fat_g", "value": "exposure"},
        {"reading": "role", "column": "ALC", "value": "exposure"}]}
    log = confirmed(WITHOUT_ALC, ADJ, rode=("fat_g", "ALC"), confirmation=both)
    returned = [*log, rec(6, {"kind": "revert", "decision_id": "r5"})]
    again = refused(returned, REVISIT)
    assert again.code == "role_unconfirmed" and "reads `fat_g`," in again.message
    refusal = refused(log, REVISIT)
    assert refusal.message == (
        "The recorded energy adjustment reads `fat_g` and `ALC`, which would then be roles "
        "nobody confirmed. Keep their confirmation.")
    assert refusal.exits == [KEEP]


def test_a_withdrawal_the_record_refuses_is_neither_offered_nor_named(monkeypatch):
    """Where the record refuses the withdrawal (no rule refuses reverting these answers today, so
    a stand-in rule on a revert refuses record 4's), the revisit's refusal neither offers nor names
    it. Its other direction, the withdrawal accepted, is the exclusion rules' test above."""
    def refuse_the_screens_revert(decision: Any, _: Any) -> None:
        if decision.decision_id == "r4":
            raise d.Refusal("stand_in", "A stand-in rule keeps record 4.")

    monkeypatch.setitem(d._VALIDATORS, "revert",
                        [*d._VALIDATORS["revert"], refuse_the_screens_revert])
    log = confirmed(SCREEN, rode=("kcal",))
    with pytest.raises(d.Refusal) as withdrawal:
        d.validate({"kind": "revert", "decision_id": "r4"}, ctx(log))
    assert withdrawal.value.code == "stand_in"
    refusal = refused(log, REVISIT)
    assert refusal.message == (
        "The recorded exclusion rules read `kcal`, which would then be a role nobody confirmed. "
        "Keep its confirmation.")
    assert refusal.exits == [KEEP]


def test_reverting_the_users_own_roles_answer_keeps_the_answers_not_a_confirmation():
    """The user's own roles answer (record 3, nothing riding along) settled `ALC` without any
    confirmation; reverting it returns the bulk answer that carried `ALC` below high (record 2),
    so the words keep the answers as they are: no confirmation is undone."""
    own = {"kind": "set_roles", "roles": ROLES, "unconfirmed": []}
    log = confirmed(ADJ, confirmation=own)
    assert d.fold(log).role_confirmations in (None, {})
    refusal = refused(log, REVISIT)
    assert refusal.message == (
        "The recorded energy adjustment reads `ALC`, which would then be a role nobody "
        "confirmed. Withdraw the energy adjustment first, or keep the answers as they are.")
    assert refusal.exits == [{"label": "Withdraw the energy adjustment first",
                              "decision": {"kind": "revert", "decision_id": "r4"}}, KEEP]
    after = taken(log, refusal.exits[0]["decision"])
    d.validate(REVISIT, ctx(after))


def shown(name: str) -> Any:
    """The roles the server showed: `ALC` proposed as a covariate, below high."""
    return {"columns": [{"column": "ALC", "proposed": "covariate", "confidence": "medium"}]} \
        if name == "roles" else None


def test_a_bulk_roles_answer_that_carries_a_settled_role_off_is_refused_with_its_routes():
    """The user's own roles answer settled `ALC` as a study factor (record 2) and the energy
    adjustment reads it (record 3); a bulk answer recording `ALC` as its proposal, a covariate
    below high, would carry it along unconfirmed. No confirmation is undone, so the words keep
    the answers as they are; the withdrawal is accepted, and the bulk answer then is."""
    own = {"kind": "set_roles", "roles": ROLES, "unconfirmed": []}
    log = [rec(1, {"kind": "set_target", "column": "y"}), rec(2, own), rec(3, ADJ)]
    bulk = {"kind": "set_roles", "roles": {**ROLES, "ALC": "covariate"}}
    refusal = refused(log, bulk, artifact=shown)
    assert refusal.message == (
        "The recorded energy adjustment reads `ALC`, which would then be a role nobody "
        "confirmed. Withdraw the energy adjustment first, or keep the answers as they are.")
    assert refusal.exits == [{"label": "Withdraw the energy adjustment first",
                              "decision": {"kind": "revert", "decision_id": "r3"}}, KEEP]
    after = taken(log, refusal.exits[0]["decision"], artifact=shown)
    assert d.validate(bulk, ctx(after, artifact=shown)).unconfirmed == ["ALC"]


def test_with_no_answer_reading_the_role_the_revisit_is_accepted():
    d.validate(REVISIT, ctx(confirmed()))
    d.validate(REVISIT, ctx(confirmed({"kind": "set_exclusions", "rules": [
        {"column": "age", "low": 18, "high": 90, "reason": "adults"}]})))
