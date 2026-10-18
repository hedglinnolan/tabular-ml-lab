"""W1-R round 2, ruling R-b · no press confirms several roles proposed below high confidence
(BLUEPRINT §14 rule 2; the walkthrough of 2026-10-17, finding 1; W1-R R1's verifier, P1).

A proposal below high "needs its own confirmation; it never rides along in a bulk 'confirm all'".
The readings ledger served such roles as a family and confirmed them, with the codes they ride
with, in one block: the Models ask's block exit, its refusal's exits and the shell's confirm-all
each settled six flags and a covariate in one press. Now:

* the ask never groups a role that waits into a family, and no block exit lists a role
  (:func:`readings.families`, :func:`readings.block_exit`, :func:`readings.ask_exits`), so each
  role keeps one line and one confirmation of its own; a family of codes or amounts keeps its
  block;
* the engine refuses a ``confirm_readings`` block that lists a role proposed below high
  confidence, whoever posts it, with an exit that confirms each such role on its own; a block of
  other readings, or one role on its own, is accepted.

Every expectation is from construction: which columns the roles answer recorded as proposed below
high (``roles_unconfirmed``), or which proposals the roles stage made below high, is written here.
"""
from __future__ import annotations

import pytest

from turbotab.core import ask, decisions, quest, readings
from turbotab.core.decisions import ProjectState, Refusal, validate
from turbotab.core.readings import Reading
from turbotab.core.tests.test_stage_registry import record, steps_until

FLAGS = ["imputed_bp", "imputed_hdl", "imputed_height", "imputed_tg", "imputed_waist",
         "imputed_weight"]  # six of one kind, one guess, one name pattern: once a family
ROLES = {"SEQN": "identifier", "age": "covariate", "bp_sys": "covariate",
         **{f: "flag" for f in FLAGS}}
WAITING = ["bp_sys", *FLAGS]  # recorded exactly as proposed, below high confidence


def _state(purpose: str | None = "prediction", **slots) -> ProjectState:
    base = dict(lens=["clinical"], target="glucose", purpose=purpose, roles=dict(ROLES),
                roles_unconfirmed=list(WAITING))
    return ProjectState(**{**base, **slots})


def _decisions(exits) -> list[dict]:
    out = []
    for e in exits:
        d = e["decision"] if isinstance(e, dict) else e.decision
        if d is not None:
            out.append(d if isinstance(d, dict) else d.model_dump(mode="json"))
    return out


def _roles_in(decision: dict) -> list[str]:
    """The role readings one press settles."""
    if decision.get("kind") == "confirm_readings":
        return [i["column"] for i in decision["items"] if i["reading"] == "role"]
    if decision.get("kind") == "confirm_role" or (decision.get("kind") == "confirm_reading"
                                                  and decision.get("reading") == "role"):
        return [decision["column"]]
    return []


def _waiting_readings(state: ProjectState) -> list[Reading]:
    return [readings.role_reading(state, c) for c in readings.unsettled(state)]


# ── the ask: one line and one press per waiting role ──────────────────────────


def test_a_waiting_role_is_never_grouped_into_a_family():
    state = _state()
    groups = readings.families(_waiting_readings(state))
    assert [[r.column for r in g] for g in groups] == [[c] for c in sorted(WAITING)]


def test_a_family_of_codes_or_amounts_is_still_one_line():
    codes = [Reading((f"item_{i:02d}",), "code_or_count", "amount", "medium", "whole numbers")
             for i in range(readings.FAMILY_MIN)]
    groups = readings.families(codes)
    assert [[r.column for r in g] for g in groups] == [[r.column for r in codes]]


def test_no_exit_of_the_ask_confirms_more_than_one_role():
    state = _state()
    exits = readings.ask_exits(_waiting_readings(state), state)
    found = _decisions(exits)
    assert found and all(len(_roles_in(d)) <= 1 for d in found)
    # each waiting role keeps its own confirmation, as proposed
    own = {d["column"]: d["value"] for d in found if d.get("kind") == "confirm_reading"
           and d.get("reading") == "role" and d["value"] == ROLES[d["column"]]}
    assert own == {c: ROLES[c] for c in WAITING}
    assert readings.block_exit(_waiting_readings(state)) is None


def test_a_block_of_codes_rides_beside_the_roles_without_them():
    state = _state()
    codes = [Reading((c,), "code_or_count", "amount", "medium", "whole numbers")
             for c in ("visits", "children")]
    exits = readings.ask_exits([*_waiting_readings(state), *codes], state)
    blocks = [d for d in _decisions(exits) if d.get("kind") == "confirm_readings"]
    assert [[(i["reading"], i["column"]) for i in b["items"]] for b in blocks] == [
        [("code_or_count", "children"), ("code_or_count", "visits")]]


def test_the_models_card_asks_each_waiting_role_on_its_own_line():
    state = _state()
    card = ask.card("models", ask.AskContext(state, {}))
    assert card is not None
    assert [(g.kind, g.columns) for g in card.groups] == [("role", [c]) for c in sorted(WAITING)]
    found = _decisions(card.exits)
    assert found and all(len(_roles_in(d)) <= 1 for d in found)


def test_the_fits_refusal_offers_no_press_that_confirms_several_roles():
    state = _state()
    with pytest.raises(Refusal) as refused:
        decisions._models_read_settled_readings(decisions.SelectModels(models=["linear"]),
                                                {"state": state})
    found = _decisions(refused.value.exits)
    assert found and all(len(_roles_in(d)) <= 1 for d in found)
    assert {c for d in found for c in _roles_in(d)} == set(WAITING)


def test_the_quest_logs_ask_line_offers_one_column_per_press():
    records = [record(1, {"kind": "set_lens", "lenses": ["clinical"]}),
               record(2, {"kind": "set_target", "column": "glucose"}),
               record(3, {"kind": "set_purpose", "purpose": "prediction"}),
               record(4, {"kind": "set_roles", "roles": dict(ROLES), "unconfirmed": WAITING})]
    state = decisions.fold(records)
    asked = quest.fit_waits(state)
    log = quest.quest_log(state, records, steps_until(None), asked=[asked])
    line = next(l for s in log.stages for l in s.lines if l.key == "ask:models")
    found = [o.decision for i in line.items for o in i.options if o.decision is not None]
    assert found and all(len(_roles_in(d)) <= 1 for d in found)


# ── the engine: a block that lists a role below high is refused ───────────────


def _block(*items: tuple[str, str, str]) -> dict:
    return {"kind": "confirm_readings",
            "items": [{"reading": k, "column": c, "value": v} for k, c, v in items]}


def test_a_block_of_waiting_roles_is_refused_with_one_confirmation_each():
    state = _state()
    body = _block(*(("role", f, "flag") for f in FLAGS))
    with pytest.raises(Refusal) as refused:
        validate(body, {"state": state})
    assert refused.value.code == "roles_one_at_a_time"
    found = _decisions(refused.value.exits)
    assert [(d["kind"], d["reading"], d["column"], d["value"]) for d in found] == [
        ("confirm_reading", "role", f, "flag") for f in FLAGS]
    for d in found:  # each way forward is accepted on its own
        validate(d, {"state": state})


def test_a_block_that_lists_one_waiting_role_beside_codes_is_refused_and_keeps_the_codes():
    state = _state()
    body = _block(("role", "bp_sys", "covariate"), ("code_or_count", "visits", "amount"),
                  ("code_or_count", "children", "amount"))
    with pytest.raises(Refusal) as refused:
        validate(body, {"state": state})
    found = _decisions(refused.value.exits)
    assert found[0] == {"kind": "confirm_reading", "reading": "role", "column": "bp_sys",
                        "value": "covariate"}
    # the other readings stay one block, exactly as listed
    assert found[1] == _block(("code_or_count", "visits", "amount"),
                              ("code_or_count", "children", "amount"))


def test_before_the_roles_answer_a_block_of_proposals_below_high_is_refused():
    state = ProjectState(lens=["clinical"], target="glucose")
    proposals = {"columns": [
        {"column": "age", "proposed": "covariate", "confidence": "high", "reason": "an age"},
        {"column": "SEQN", "proposed": "identifier", "confidence": "high", "reason": "unique"},
        {"column": "bp_sys", "proposed": "covariate", "confidence": "medium", "reason": "a name"}]}
    ctx = {"state": state, "artifact": lambda stage: proposals if stage == "roles" else None}
    with pytest.raises(Refusal) as refused:
        validate(_block(("role", "age", "covariate"), ("role", "bp_sys", "covariate")), ctx)
    assert [(d["kind"], d["column"]) for d in _decisions(refused.value.exits)] == [
        ("confirm_reading", "bp_sys"), ("confirm_reading", "age")]
    # silent: the proposals the values made high may be confirmed together
    validate(_block(("role", "age", "covariate"), ("role", "SEQN", "identifier")), ctx)


def test_silent_a_block_with_no_role_below_high_and_a_role_on_its_own_are_accepted():
    state = _state()
    validate(_block(("code_or_count", "visits", "amount"), ("code_or_count", "children", "amount")),
             {"state": state})
    validate(_block(("role", "bp_sys", "covariate")), {"state": state})
    # settled roles (the values made them high, or confirmed since) may ride together
    settled = _state(roles_unconfirmed=[])
    validate(_block(("role", "age", "covariate"), ("role", "bp_sys", "covariate")),
             {"state": settled})


# ── a refusal's own block: the batch as a covariate (MS7) ─────────────────────


def _batch_exits(state: ProjectState) -> list[dict]:
    from turbotab.core.decisions import SetBatch

    ctx = {"state": state, "columns": ["g00", "g01", "batch", "y"],
           "column_info": {"batch": {"dtype": "numeric", "n_unique": 3, "n_missing": 0}},
           "artifact": lambda name: {"findings": []} if name == "findings" else None}
    with pytest.raises(Refusal) as refused:
        validate(SetBatch(column="batch", method="covariate"), ctx)
    assert refused.value.code == "batch_covariate_unsettled"
    return _decisions(refused.value.exits)


def test_a_batch_whose_role_waits_is_confirmed_as_a_covariate_on_its_own_first():
    roles = {"g00": "exposure", "g01": "exposure", "batch": "excluded"}
    waiting = ProjectState(target="y", purpose="inference", roles=roles,
                           roles_unconfirmed=["batch"])
    assert _batch_exits(waiting) == [{"kind": "confirm_reading", "reading": "role",
                                      "column": "batch", "value": "covariate"}]
    # silent: with its role settled (left out, on the person's own answer), the role and the
    # codes are one block, as before
    settled = ProjectState(target="y", purpose="inference", roles=roles)
    assert _batch_exits(settled) == [_block(("role", "batch", "covariate"),
                                            ("code_or_count", "batch", "code"))]
