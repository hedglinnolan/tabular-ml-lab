"""Ruling R-g (2026-10-17) · one judgment of the recognizer is confirmed in one press.

BLUEPRINT §14 rule 2 and the gate's criterion: a misrecognition proposed at medium or low "with its
evidence visible and confirmed individually" is caught by the user; the danger is a proposal below
high riding along in a bulk "confirm all". Ruling R-b read that as one press per column, so a
researcher pressed 300 times on the omics depth table (20,000 on a real expression table). Under
R-g the roles proposed below high that share one proposal, the same role, the same confidence and
the same reason word for word, are one group:

* Your data's role lines list each group as one line with one press ("Confirm these 300 as assay
  features"), its columns a click away, each with its own presses to move it; a group of one is a
  column's own line, as before;
* the engine accepts a ``confirm_readings`` that lists exactly one group's waiting columns with
  its proposed role, and refuses any other block of roles below high (two groups, a group beside
  another reading, part of a group), naming each group and offering each group's own press;
* every column a group press confirms is settled where its own confirmation would be, so the
  ledger, ``unsettled`` and the predictors read it unchanged;
* the record and the methods say a group's press in one sentence.

Every expectation is from construction: the proposals are written here or are the roles stage's own
on a table built here, and the counts are read from those proposals, never retyped.
"""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions, provenance, quest, readings, voice
from turbotab.core.decisions import Refusal, validate
from turbotab.core.stages.rows import roles_stage
from turbotab.core.tests.stage_harness import Ingested
from turbotab.core.tests.test_stage_registry import record, stage, steps_until

GENES = [f"g{i:02d}" for i in range(5)]
FEATURE = "A measured feature: what you study under the omics lens."
TRAIT = "A person's characteristic, usually adjusted for rather than studied."
PROPOSALS = {"columns": [
    {"column": "sid", "proposed": "identifier", "confidence": "high", "reason": "one per row"},
    {"column": "age", "proposed": "covariate", "confidence": "high", "reason": "ages in years"},
    *({"column": g, "proposed": "exposure", "confidence": "medium", "reason": FEATURE}
      for g in GENES),
    {"column": "bp_sys", "proposed": "covariate", "confidence": "medium", "reason": TRAIT},
    {"column": "weight", "proposed": "covariate", "confidence": "medium", "reason": TRAIT},
    # one word differs: a judgment of its own
    {"column": "waist", "proposed": "covariate", "confidence": "medium",
     "reason": "A person's characteristic, read from its name alone."},
]}
ROLES = {p["column"]: p["proposed"] for p in PROPOSALS["columns"]}
WAITING = [p["column"] for p in PROPOSALS["columns"] if p["confidence"] != "high"]
# The W1-U verifier's P4: under Predict a group's shared reason is quoted without its study clause,
# as a role's evidence is (``readings.role_evidence``); under Estimate as the roles stage said it.
# The words are written out here.
UNDER_PREDICT = {FEATURE: "A measured feature under the omics lens.",
                 TRAIT: "A person's characteristic."}


def _said(reason: str, purpose: str | None) -> str:
    return UNDER_PREDICT.get(reason, reason) if purpose == "prediction" else reason


def _records(lenses=("genomics",), purpose="prediction", *later: dict) -> list:
    out = [record(1, {"kind": "set_lens", "lenses": list(lenses)}),
           record(2, {"kind": "set_target", "column": "y"}),
           record(3, {"kind": "set_purpose", "purpose": purpose}),
           record(4, {"kind": "set_roles", "roles": dict(ROLES), "unconfirmed": list(WAITING)})]
    return out + [record(5 + i, d) for i, d in enumerate(later)]


def _ctx(state, artifact=PROPOSALS) -> dict:
    return {"state": state, "artifact": lambda name: artifact if name == "roles" else None}


def _block(*items: tuple[str, str, str]) -> dict:
    return {"kind": "confirm_readings",
            "items": [{"reading": k, "column": c, "value": v} for k, c, v in items]}


def _press(columns, role) -> dict:
    return _block(*(("role", c, role) for c in columns))


def _judgments(proposals: dict) -> dict[tuple[str, str, str], list[str]]:
    """The proposals below high grouped by role, confidence and reason, read from the artifact."""
    out: dict[tuple[str, str, str], list[str]] = collections.OrderedDict()
    for p in proposals["columns"]:
        if p["confidence"] != "high":
            out.setdefault((p["proposed"], p["confidence"], p["reason"]), []).append(p["column"])
    return out


def _role_lines(log) -> list:
    return [l for l in stage(log, "data").lines if l.id == "decision:confirm_role"]


def _decided(decision: dict) -> dict:
    d = decision.model_dump(mode="json") if hasattr(decision, "model_dump") else dict(decision)
    return d


# ── the ledger's groups ───────────────────────────────────────────────────────


def test_roles_proposed_alike_word_for_word_are_one_group_and_one_differing_word_is_another():
    state = decisions.fold(_records())
    groups = readings.role_groups(state, PROPOSALS)
    found = {(g.role, g.confidence, g.reason): list(g.waiting) for g in groups}
    assert found == dict(_judgments(PROPOSALS))
    assert [len(g.waiting) for g in groups] == [5, 2, 1]


def test_a_group_is_named_by_its_first_columns_and_how_many_more():
    many = [f"g{i}" for i in range(20_000)]
    assert readings.first_columns(many) == "`g0`, `g1` and 19,998 more"
    assert readings.first_columns(GENES) == "`g00`, `g01` and 3 more"
    assert readings.first_columns(["a", "b", "c"]) == "`a`, `b` and `c`"
    assert readings.first_columns(["a", "b"]) == "`a` and `b`"


# ── the engine: one group's press accepted, any other block of them refused ──


def test_a_group_of_identical_proposals_is_confirmed_in_one_press_and_every_column_settles():
    state = decisions.fold(_records())
    press = _press(GENES, "exposure")
    validate(press, _ctx(state))  # accepted
    after = decisions.fold(_records(("genomics",), "prediction", press))
    # read from the ledger: each column's own confirmation is recorded, as one press of its own
    # would record it, so every consumer reads it unchanged
    for g in GENES:
        assert readings.confirmation(after, "role", g) == "exposure"
        assert readings.role_reading(after, g).settled
    assert after.role_confirmations == {g: "exposure" for g in GENES}
    assert set(readings.unsettled(after)) == set(WAITING) - set(GENES)
    assert set(GENES) <= set(readings.predictor_columns(after))
    one_each = decisions.fold(_records(("genomics",), "prediction", *(
        {"kind": "confirm_reading", "reading": "role", "column": g, "value": "exposure"}
        for g in GENES)))
    assert readings.settled_roles(after) == readings.settled_roles(one_each)


def test_a_block_of_two_groups_is_refused_and_names_each_with_its_own_press():
    state = decisions.fold(_records())
    traits = ["bp_sys", "weight"]
    with pytest.raises(Refusal) as refused:
        validate(_block(*(("role", c, "exposure") for c in GENES),
                        *(("role", c, "covariate") for c in traits)), _ctx(state))
    assert refused.value.code == "roles_one_at_a_time"
    exits = [_decided(e["decision"]) for e in refused.value.exits]
    assert exits == [_press(GENES, "exposure"), _press(traits, "covariate")]
    for e in exits:  # each group's own press is accepted on its own
        validate(e, _ctx(state))
    message = refused.value.message
    assert f"the {len(GENES)} columns `g00`, `g01` and 3 more" in message
    assert "the 2 columns `bp_sys` and `weight`" in message
    # each group's shared reason, in the goal's words (Predict here; the W1-U verifier's P4)
    assert UNDER_PREDICT[FEATURE] in message and UNDER_PREDICT[TRAIT] in message


TRAITS = ["bp_sys", "weight"]
WAIST = next(p["reason"] for p in PROPOSALS["columns"] if p["column"] == "waist")


@pytest.mark.parametrize("purpose", ["prediction", "inference"])
def test_two_judgments_alike_but_for_their_reason_are_refused_in_one_block(purpose):
    """The M1 run's block: `bp_sys` and `weight` proposed as covariates at medium confidence for one
    reason, and `waist` proposed so for another, the same role and confidence. Each reason is a
    judgment of its own (ruling R-g), so the block of all three is refused, offering each group's
    own press (the pair's ``confirm_readings``, `waist`'s own confirmation) and naming each group;
    each press is accepted on its own. A guard that read only the role and the confidence would
    accept the block (the mutation this test is run against)."""
    assert [p["confidence"] for p in PROPOSALS["columns"] if p["column"] in (*TRAITS, "waist")] \
        == ["medium"] * 3 and WAIST != TRAIT
    state = decisions.fold(_records(("dietary",), purpose))
    with pytest.raises(Refusal) as refused:
        validate(_press([*TRAITS, "waist"], "covariate"), _ctx(state))
    assert refused.value.code == "roles_one_at_a_time"
    exits = [_decided(e["decision"]) for e in refused.value.exits]
    assert exits == [_press(TRAITS, "covariate"),
                     {"kind": "confirm_reading", "reading": "role", "column": "waist",
                      "value": "covariate"}]
    for e in exits:  # the other way: each group's own press is accepted on its own
        validate(e, _ctx(state))
    message = refused.value.message
    many, one = (("characteristics", "a characteristic") if purpose == "prediction"
                 else ("covariates", "a covariate"))
    assert (f"the 2 columns `bp_sys` and `weight`, proposed as {many} at medium confidence, below "
            f"high, for one reason (\"{_said(TRAIT, purpose)}\")") in message
    assert f"`waist`, proposed as {one} at medium confidence, below high" in message
    labels = [e["label"] for e in refused.value.exits]
    assert labels == [f"Confirm these 2 as {many}", f"Confirm `waist` as {one}"]


def test_a_group_beside_another_reading_is_refused_and_the_other_reading_stays_its_own():
    state = decisions.fold(_records())
    with pytest.raises(Refusal) as refused:
        validate(_block(*(("role", c, "exposure") for c in GENES),
                        ("code_or_count", "visits", "amount")), _ctx(state))
    exits = [_decided(e["decision"]) for e in refused.value.exits]
    assert exits == [_press(GENES, "exposure"),
                     {"kind": "confirm_reading", "reading": "code_or_count", "column": "visits",
                      "value": "amount"}]
    assert "1 other reading" in refused.value.message


def test_a_group_beside_a_role_proposed_high_is_refused():
    state = decisions.fold(_records())
    with pytest.raises(Refusal) as refused:
        validate(_block(*(("role", c, "exposure") for c in GENES), ("role", "age", "covariate")),
                 _ctx(state))
    exits = [_decided(e["decision"]) for e in refused.value.exits]
    assert exits == [_press(GENES, "exposure"),
                     {"kind": "confirm_reading", "reading": "role", "column": "age",
                      "value": "covariate"}]


def test_part_of_a_group_is_refused_with_the_whole_groups_press():
    state = decisions.fold(_records())
    with pytest.raises(Refusal) as refused:
        validate(_press(GENES[:3], "exposure"), _ctx(state))
    assert [_decided(e["decision"]) for e in refused.value.exits] == [_press(GENES, "exposure")]
    assert refused.value.message.count(f"3 of the {len(GENES)} columns") == 1


def test_a_group_listed_with_another_role_is_refused_one_column_at_a_time():
    state = decisions.fold(_records())
    with pytest.raises(Refusal) as refused:
        validate(_press(GENES, "covariate"), _ctx(state))
    assert [_decided(e["decision"]) for e in refused.value.exits] == [
        {"kind": "confirm_reading", "reading": "role", "column": g, "value": "covariate"}
        for g in GENES]


def test_silent_a_groups_press_one_column_and_roles_settled_by_their_values_are_accepted():
    state = decisions.fold(_records())
    validate(_press(["bp_sys", "weight"], "covariate"), _ctx(state))
    validate(_press(["waist"], "covariate"), _ctx(state))  # a group of one, on its own
    validate(_block(("role", "age", "covariate"), ("role", "sid", "identifier")), _ctx(state))


def test_without_the_proposals_each_waiting_column_is_a_group_of_its_own():
    state = decisions.fold(_records())
    with pytest.raises(Refusal) as refused:
        validate(_press(GENES, "exposure"), _ctx(state, artifact=None))
    assert [_decided(e["decision"]) for e in refused.value.exits] == [
        {"kind": "confirm_reading", "reading": "role", "column": g, "value": "exposure"}
        for g in GENES]


def test_before_the_roles_answer_a_groups_press_is_accepted_and_part_of_it_refused():
    state = decisions.fold(_records()[:3])
    validate(_press(GENES, "exposure"), _ctx(state))
    with pytest.raises(Refusal):
        validate(_press(GENES[:2], "exposure"), _ctx(state))


# ── Your data's role lines: one line and one press per group ─────────────────


def _log(records, artifact=PROPOSALS):
    return quest.quest_log(decisions.fold(records), records, steps_until("exclusions"),
                           artifacts={"roles": artifact} if artifact is not None else {})


def test_each_judgment_is_one_line_and_a_group_of_one_is_a_columns_own_line():
    lines = _role_lines(_log(_records()))
    judged = list(_judgments(PROPOSALS).values())
    assert len(lines) == len(judged)
    for line, columns in zip(lines, judged):
        listed = [i.column for i in line.items if i.kind == "role"]
        assert listed == columns
        assert (line.status, line.counted, line.label) == ("open", True, "Decide")
    genes, traits, waist = lines
    assert genes.key == quest.role_group_key("g00") and waist.key == quest.role_line_key("waist")
    head = genes.items[0]
    assert head.kind == quest.ROLE_GROUP_ITEM and head.evidence == UNDER_PREDICT[FEATURE]
    # the line says how many, which role in the goal's words, the confidence and the reason
    assert head.words == (f"{len(GENES)} columns proposed as assay features at medium confidence, "
                          f"below high, for one reason")
    assert genes.name == "the roles of `g00`, `g01` and 3 more"
    [press] = head.options
    assert press.label == f"Confirm these {len(GENES)} as assay features"
    assert press.decision == _press(GENES, "exposure")
    # each column keeps only its own moves, one column per press; the group of one is today's line
    for item in genes.items[1:]:
        assert {(o.decision["kind"], o.decision["column"]) for o in item.options} == {
            ("confirm_role", item.column)}
    assert [o.label for o in waist.items[0].options][0] == "Confirm `waist` as a characteristic"


@pytest.mark.parametrize("purpose, words", [("prediction", "assay features"),
                                            ("inference", "what you study")])
def test_a_groups_press_is_worded_by_the_goal(purpose, words):
    genes = _role_lines(_log(_records(("genomics",), purpose)))[0]
    assert genes.items[0].options[0].label == f"Confirm these {len(GENES)} as {words}"
    traits = _role_lines(_log(_records(("genomics",), purpose)))[1]
    many = {"prediction": "characteristics", "inference": "covariates"}[purpose]
    assert traits.items[0].options[0].label == f"Confirm these 2 as {many}"


def test_a_groups_press_answers_its_line_and_your_data_completes():
    records = _records()
    first = stage(_log(records), "data").progress
    for line in _role_lines(_log(records)):
        press = line.items[0].options[0].decision
        validate(press, _ctx(decisions.fold(records)))
        records = [*records, record(len(records) + 1, press)]
    log = _log(records)
    assert [l.status for l in _role_lines(log)] == ["answered"] * 3
    data = stage(log, "data").progress
    assert data.required == first.required and data.complete
    assert _role_lines(log)[0].decision_id == records[4].id


def test_moving_one_column_is_one_press_and_the_rest_stay_one_group():
    move = {"kind": "confirm_role", "column": "g02", "role": "covariate"}
    records = _records(("genomics",), "prediction", move)
    state = decisions.fold(records)
    assert state.roles["g02"] == "covariate" and readings.role_reading(state, "g02").settled
    genes, *_ = lines = _role_lines(_log(records))
    rest = [g for g in GENES if g != "g02"]
    assert [i.column for i in genes.items if i.kind == "role"] == rest
    assert genes.items[0].options[0].decision == _press(rest, "exposure")
    validate(_press(rest, "exposure"), _ctx(state))
    own = next(l for l in lines if l.key == quest.role_line_key("g02"))
    assert own.status == "answered" and own.decision_id == records[-1].id


def test_a_group_keeps_its_line_when_its_first_column_moves():
    records = _records(("genomics",), "prediction",
                       {"kind": "confirm_role", "column": "g00", "role": "excluded"})
    lines = _role_lines(_log(records))
    genes = next(l for l in lines if l.key == quest.role_group_key("g00"))
    assert [i.column for i in genes.items if i.kind == "role"] == GENES[1:]
    assert genes.name == "the roles of `g01`, `g02` and 2 more"
    own = next(l for l in lines if l.key == quest.role_line_key("g00"))
    assert (own.status, own.items[0].value) == ("answered", "excluded")


def test_without_the_proposals_each_waiting_column_is_a_line_of_its_own():
    assert len(_role_lines(_log(_records(), artifact=None))) == len(WAITING)


# ── the ledger's ask card asks a whole group in one line ─────────────────────


def test_the_models_ask_card_lists_a_whole_group_as_one_line_with_one_press():
    from turbotab.core import ask

    state = decisions.fold(_records())
    card = ask.card("models", ask.AskContext(state, {"roles": PROPOSALS}))
    roles = [g for g in card.groups if g.kind == "role"]
    assert sorted(len(g.columns) for g in roles) == [1, 2, 5]
    genes = next(g for g in roles if len(g.columns) == 5)
    assert genes.guess_words == "assay features" and genes.evidence == UNDER_PREDICT[FEATURE]
    presses = [e.decision for e in card.exits if e.decision and e.decision["kind"] ==
               "confirm_readings" and len(e.decision["items"]) > 1]
    assert sorted(len(d["items"]) for d in presses) == [2, 5]
    for d in presses:
        validate(d, _ctx(state))
    # silent: without the proposals, one line per column, as before
    alone = ask.card("models", ask.AskContext(state, {}))
    assert all(len(g.columns) == 1 for g in alone.groups if g.kind == "role")


# ── the fit's refusal: one press per judgment where the proposals are at hand ──


def _confirms(exits) -> list[dict]:
    """The refusal's ways forward that record a decision, as plain dicts."""
    out = []
    for e in exits:
        d = e.get("decision") if isinstance(e, dict) else getattr(e, "decision", None)
        if d is not None:
            out.append(_decided(d))
    return out


def _check_one_press_per_judgment(found: list[dict], state, proposals=PROPOSALS) -> None:
    """Each judgment of two or more is confirmed by its one press, exactly its columns as proposed,
    and by no press of one column; a group of one by its own confirmation; each press accepted."""
    for (role, _confidence, _reason), columns in _judgments(proposals).items():
        own = [d for d in found if d["kind"] == "confirm_reading" and d["column"] in columns
               and d["value"] == role]
        if len(columns) > 1:
            assert found.count(_press(columns, role)) == 1, columns
            assert not own, own
        else:
            assert len(own) == 1, columns
    blocks = [d for d in found if d["kind"] == "confirm_readings"]
    assert len(blocks) == len([v for v in _judgments(proposals).values() if len(v) > 1])
    for d in [*blocks, *(d for d in found if d["kind"] == "confirm_reading")]:
        validate(d, _ctx(state, proposals))


def test_the_fits_refusal_confirms_each_whole_judgment_in_one_press():
    """The W1-U verifier's P4: the fit's refusal offered one confirmation per waiting column (300
    on the depth table). With the roles stage's proposals at hand it offers each judgment's one
    press, as select_models's refusal does (the validator's context holds them)."""
    state = decisions.fold(_records())
    with pytest.raises(readings.Unsettled) as waiting:
        readings.predictors_or_ask(state, artifact=PROPOSALS)
    _check_one_press_per_judgment(_confirms(waiting.value.exits), state)
    with pytest.raises(Refusal) as refused:
        decisions._models_read_settled_readings(decisions.SelectModels(models=["linear"]),
                                                _ctx(state))
    assert refused.value.code == "reading_unsettled"
    _check_one_press_per_judgment(_confirms(refused.value.exits), state)


def test_the_quest_logs_ask_line_lists_each_judgment_as_one_item_with_its_one_press():
    """The quest log's ask:models line (from ``quest.fit_waits``, the server's roles artifact
    passed in): each judgment of two or more is one group item, how many and the first columns
    named, carrying its one press; its columns follow, each with only its own presses; a group of
    one keeps its column's own line."""
    records = _records()
    state = decisions.fold(records)
    asked = quest.fit_waits(state, artifact=PROPOSALS)
    log = quest.quest_log(state, records, steps_until(None), asked=[asked])
    line = next(l for s in log.stages for l in s.lines if l.key == "ask:models")
    heads = [i for i in line.items if i.kind == quest.ROLE_GROUP_ITEM]
    judged = [(k, v) for k, v in _judgments(PROPOSALS).items() if len(v) > 1]
    assert len(heads) == len(judged)
    for head in heads:
        (role, confidence, reason), columns = next((k, v) for k, v in judged
                                                   if v[0] == head.column)
        [press] = head.options
        assert press.decision == _press(columns, role)
        assert press.label == readings.group_press_label(len(columns), role, "prediction",
                                                         ["genomics"])
        assert head.evidence == UNDER_PREDICT[reason]
        assert head.words.startswith(readings.first_columns(columns) + ": ")
        assert f"{len(columns)} columns proposed as" in head.words and confidence in head.words
        at = line.items.index(head)
        members = line.items[at + 1:at + 1 + len(columns)]
        assert [(m.kind, m.column) for m in members] == [("role", c) for c in columns]
        assert not [o for m in members for o in m.options
                    if o.decision and o.decision["kind"] == "confirm_readings"]
    presses = [o.decision for i in line.items for o in i.options if o.decision]
    _check_one_press_per_judgment(presses, state)


def test_silent_a_judgment_asked_only_in_part_or_without_the_proposals_keeps_a_press_per_column():
    """Both ways: a refusal that asks only some of a judgment's columns, or that has no proposals
    at hand, offers each column its own confirmation and no block of roles."""
    state = decisions.fold(_records())
    part = [readings.role_reading(state, g) for g in GENES[:3]]
    listed = readings.with_proposals(part, state, PROPOSALS)
    assert all(r.judgment is None for r in listed)
    exits = readings.judged_exits(readings.ask_exits(listed, state), listed, state)
    asked = quest.Asked("models", "waits", tuple(listed), tuple(exits))
    for found in (_confirms(exits),
                  [o.decision for i in quest._ask_items(asked, state) for o in i.options]):
        assert sorted((d["kind"], d["column"]) for d in found) == [
            ("confirm_reading", g) for g in GENES[:3]]
    alone = quest.fit_waits(state)
    found = _confirms(alone.exits)
    assert not [d for d in found if d["kind"] == "confirm_readings"]
    assert {d["column"] for d in found} == set(WAITING)
    items = quest._ask_items(alone, state)
    assert not [i for i in items if i.kind == quest.ROLE_GROUP_ITEM]


# ── the shared reason in the goal's words (the W1-U verifier's P4) ────────────


@pytest.mark.parametrize("purpose", ["prediction", "inference"])
def test_a_groups_reason_is_quoted_in_the_goals_words_on_every_surface(purpose):
    """The verifier's journeys: under Predict a group's line, the fit's ask and the record quoted
    the roles stage's reason in Estimate's words ("what you study under the omics lens", "usually
    adjusted for rather than studied"), and the ask joined it as "…studied.; one reason". Each
    surface quotes it once, in the goal's words, and the ask's clause follows it without its
    period."""
    from turbotab.core import ask

    records = _records(("genomics",), purpose)
    state = decisions.fold(records)
    log = _log(records)
    heads = {l.items[0].column: l.items[0] for l in _role_lines(log)
             if l.items[0].kind == quest.ROLE_GROUP_ITEM}
    asked = quest.fit_waits(state, artifact=PROPOSALS)
    items = {i.column: i for i in quest._ask_items(asked, state)
             if i.kind == quest.ROLE_GROUP_ITEM}
    card = ask.card("models", ask.AskContext(state, {"roles": PROPOSALS}))
    groups = {g.columns[0]: g for g in card.groups if g.kind == "role" and len(g.columns) > 1}
    assert ".;" not in asked.message
    for (role, confidence, reason), columns in _judgments(PROPOSALS).items():
        if len(columns) < 2:
            continue
        said = _said(reason, purpose)
        assert heads[columns[0]].evidence == said
        assert items[columns[0]].evidence == said
        assert groups[columns[0]].evidence == said
        assert (f"({said.rstrip('.')}; one reason for all {len(columns)}, at {confidence} "
                f"confidence)") in asked.message
        sentence = voice.sentence_for(_press(columns, role), state, {"roles": PROPOSALS})
        assert f"for one reason (\"{said}\")" in sentence
    if purpose == "prediction":
        for text in (asked.message, *(h.evidence for h in heads.values())):
            for word in ("what you study", "studied", "adjusted for"):
                assert word not in text, (word, text)


# The researcher's own data dictionary, imported: a label for the first column of each group and
# for the column proposed alone.
CODEBOOK = {"kind": "import_codebook", "codebook": "c0123456789", "name": "dictionary.csv",
            "form": "table", "labels": {"g00": "Gene zero, read counts",
                                        "bp_sys": "Systolic blood pressure (mmHg)",
                                        "waist": "Waist circumference (cm)"}}


@pytest.mark.parametrize("purpose", ["prediction", "inference"])
def test_a_groups_clause_quotes_only_its_shared_reason_beside_a_codebook(purpose):
    """The ruling of 2026-10-18 (m2): with a codebook imported, the ask's clause for a judged group
    quoted its first column's label inside "one reason for all N" ("…; your codebook labels it
    "…"; one reason for all 5"), though the label is that column's alone. Under either goal the
    clause, the fit's ask line's group item and the ask card quote the shared reason alone; the
    column proposed alone keeps its own label beside its evidence (the other way)."""
    from turbotab.core import ask

    records = _records(("genomics",), purpose, CODEBOOK)
    state = decisions.fold(records)
    for column, label in CODEBOOK["labels"].items():
        assert readings.codebook_label(state, column) == label
    asked = quest.fit_waits(state, artifact=PROPOSALS)
    items = {i.column: i for i in quest._ask_items(asked, state)
             if i.kind == quest.ROLE_GROUP_ITEM}
    card = ask.card("models", ask.AskContext(state, {"roles": PROPOSALS}))
    judged = [(k, v) for k, v in _judgments(PROPOSALS).items() if len(v) > 1]
    assert [v[0] for _k, v in judged] == ["g00", "bp_sys"]
    for text in (asked.message, card.text):
        for (role, confidence, reason), columns in judged:
            said = _said(reason, purpose)
            many = readings.role_group_words(role, purpose, ["genomics"])
            assert (f"{len(columns)} columns like `{columns[0]}`: {many}? ({said.rstrip('.')}; one "
                    f"reason for all {len(columns)}, at {confidence} confidence)") in text, text
            assert items[columns[0]].evidence == said
            assert CODEBOOK["labels"][columns[0]] not in text, text
        assert f"your codebook labels it \"{CODEBOOK['labels']['waist']}\"" in text, text


def test_an_ask_of_one_group_of_several_columns_says_these_columns():
    """The ruling of 2026-10-18 (m2): one group asked alone is one line of the ask, and the ask said
    "Tell me about this column: 5 columns like `g00`…". A line holding several columns is said of
    "these columns"; a column asked on its own of "this column" (the other way)."""
    genes = {"columns": [p for p in PROPOSALS["columns"]
                         if p["confidence"] == "high" or p["column"] in GENES]}
    lone = {"columns": [p for p in PROPOSALS["columns"]
                        if p["confidence"] == "high" or p["column"] == "waist"]}
    for proposals, waiting, lead in ((genes, list(GENES), "these columns: 5 columns like `g00`"),
                                     (lone, ["waist"], "this column: `waist`")):
        roles = {p["column"]: p["proposed"] for p in proposals["columns"]}
        records = [record(1, {"kind": "set_lens", "lenses": ["genomics"]}),
                   record(2, {"kind": "set_target", "column": "y"}),
                   record(3, {"kind": "set_purpose", "purpose": "prediction"}),
                   record(4, {"kind": "set_roles", "roles": roles, "unconfirmed": waiting})]
        with pytest.raises(readings.Unsettled) as asked:
            readings.predictors_or_ask(decisions.fold(records), artifact=proposals)
        assert f" Tell me about {lead}" in str(asked.value), str(asked.value)


# ── the fit's refusal names each group by its first columns (the W1-U verifier's P5) ──


def test_the_fits_refusal_names_each_group_by_its_first_columns_and_its_one_press():
    """The fit's refusal named every waiting column and said "Confirm each", though each group has
    one press (300 columns named on the depth table, 20,000 on an expression table). It names each
    group by how many and its first columns, the lone column by name, and says that a group is
    confirmed in its one press; select_models's refusal says the same."""
    state = decisions.fold(_records())
    with pytest.raises(readings.Unsettled) as waiting:
        readings.predictors_or_ask(state, artifact=PROPOSALS)
    lead = str(waiting.value).split(" The fit waits for them", 1)[0]
    assert lead == (
        f"{len(WAITING)} columns were proposed below high confidence and recorded with the other "
        f"roles, not confirmed on their own, so they cannot set the model's predictors yet: 5 for "
        f"one reason (`g00`, `g01` and 3 more), 2 for another (`bp_sys` and `weight`) and "
        f"`waist`. Confirm each group in its one press and the other column on its own, after "
        f"reading why each was proposed, or leave them out.")
    for g in GENES[2:]:
        assert f"`{g}`" not in lead, g
    with pytest.raises(Refusal) as refused:
        decisions._models_read_settled_readings(decisions.SelectModels(models=["linear"]),
                                                _ctx(state))
    assert refused.value.message.startswith(lead)
    asked = quest.fit_waits(state, artifact=PROPOSALS)
    assert asked.message.startswith(lead)


def test_a_refusal_that_asks_one_group_alone_names_it_once():
    """One judgment waiting alone: the refusal names it by its count and first columns, and its
    one press."""
    proposals = {"columns": [p for p in PROPOSALS["columns"]
                             if p["confidence"] == "high" or p["column"] in GENES]}
    roles = {p["column"]: p["proposed"] for p in proposals["columns"]}
    records = [record(1, {"kind": "set_lens", "lenses": ["genomics"]}),
               record(2, {"kind": "set_target", "column": "y"}),
               record(3, {"kind": "set_purpose", "purpose": "prediction"}),
               record(4, {"kind": "set_roles", "roles": roles, "unconfirmed": list(GENES)})]
    with pytest.raises(readings.Unsettled) as waiting:
        readings.predictors_or_ask(decisions.fold(records), artifact=proposals)
    assert str(waiting.value).startswith(
        f"The {len(GENES)} columns `g00`, `g01` and 3 more were proposed below high confidence for "
        f"one reason and recorded with the other roles, not confirmed on their own, so they cannot "
        f"set the model's predictors yet. Confirm them in their one press after reading why they "
        f"were proposed, or leave them out. The fit waits for them")


def test_silent_a_refusal_without_a_group_names_each_column_as_before():
    """The other way: without the roles stage's proposals each waiting column is asked on its own,
    so the refusal names each and asks to confirm each, as before."""
    state = decisions.fold(_records())
    with pytest.raises(readings.Unsettled) as waiting:
        readings.predictors_or_ask(state)
    listed = ", ".join(f"`{c}`" for c in WAITING[:-1]) + f" and `{WAITING[-1]}`"
    assert str(waiting.value).startswith(
        f"{listed} were proposed below high confidence and recorded with the other roles, not "
        f"confirmed on their own, so they cannot set the model's predictors yet. Confirm each "
        f"after reading why it was proposed, or leave them out. The fit waits for them")


# ── the record and the methods: one sentence per group ───────────────────────


@pytest.mark.parametrize("purpose, words", [("prediction", "assay features"),
                                            ("inference", "what you study")])
def test_a_groups_press_is_recorded_and_stated_in_one_sentence(purpose, words):
    records = _records(("genomics",), purpose)
    before = decisions.fold(records)
    press = _press(GENES, "exposure")
    said = voice.sentence_for(press, before, {"roles": PROPOSALS})
    reason = _said(FEATURE, purpose)
    assert said == (f"The {len(GENES)} columns `g00`, `g01` and 3 more, proposed as {words} at "
                    f"medium confidence, below high, for one reason (\"{reason}\"), were "
                    f"confirmed together as one group, after the evidence for their proposal was "
                    f"read.")
    # silent: another block is said as a block, reading by reading
    other = voice.sentence_for(_block(("code_or_count", "visits", "amount"),
                                      ("code_or_count", "kids", "amount")), before,
                               {"roles": PROPOSALS})
    assert other.startswith("Confirmed together, each as the question showed it")
    # the methods hold it as recorded, and the roles answer's own sentence says only what waits
    from turbotab.core.decisions import DecisionRecord  # noqa: F401 - the log's records

    log = [*records[:3], records[3].model_copy(update={"sentence": voice.sentence_for(
        records[3].decision, decisions.fold(records[:3]))}),
        record(5, press).model_copy(update={"sentence": said})]
    text = provenance.methods_text(log, {"roles": PROPOSALS}).text
    assert said in text
    roles_sentence = voice._set_roles_text(records[3].decision, decisions.fold(records[:3]),
                                           decisions.fold(log))
    for g in GENES:
        assert f"`{g}`" not in roles_sentence.split(";", 1)[-1]
    assert "`bp_sys`" in roles_sentence.split(";", 1)[-1]


# ── on the roles stage's own proposals ───────────────────────────────────────


def _proposed(path: Path, folder: Path, lens: str, target: str) -> tuple[dict, list]:
    out = Ingested(path, folder).run(roles_stage, decisions.ProjectState(
        lens=[lens], target=target, purpose="prediction"))
    roles = {p["column"]: p["proposed"] for p in out["columns"]}
    records = [record(1, {"kind": "set_lens", "lenses": [lens]}),
               record(2, {"kind": "set_target", "column": target}),
               record(3, {"kind": "set_purpose", "purpose": "prediction"}),
               record(4, {"kind": "set_roles", "roles": roles,
                          "unconfirmed": readings.rode_along(roles, out["columns"])})]
    return out, records


def test_on_the_omics_depth_table_the_genes_are_one_line_and_one_press(tmp_path):
    from turbotab.core.tests.acceptance.test_wp11_omics import depth_table

    out, records = _proposed(depth_table(tmp_path), tmp_path, "genomics", "case")
    below = [p["column"] for p in out["columns"] if p["confidence"] != "high"]
    assert len(below) == len([c for c in out["columns"] if str(c["column"]).startswith("ENSG")])
    [line] = _role_lines(_log(records, out))
    [press] = line.items[0].options
    assert press.decision == _press(below, "exposure")
    assert press.label == f"Confirm these {len(below):,} as assay features"
    validate(press.decision, _ctx(decisions.fold(records), out))
    # the fit's refusal (the quest log's ask:models line) offers the same one press for the genes
    asked = quest.fit_waits(decisions.fold(records), artifact=out)
    offered = _confirms(asked.exits)
    assert offered.count(_press(below, "exposure")) == 1
    # and it names the genes by how many and the first two, never each (the W1-U verifier's P5)
    lead = asked.message.split(" The fit waits for them", 1)[0]
    assert lead.startswith(f"The {len(below):,} columns `{below[0]}`, `{below[1]}` and "
                           f"{len(below) - 2:,} more were proposed below high confidence for one "
                           f"reason"), lead
    assert not [c for c in below[2:] if f"`{c}`" in lead]
    assert not [d for d in offered if d["kind"] == "confirm_reading" and d["column"] in below]
    [head] = [i for i in quest._ask_items(asked, decisions.fold(records))
              if i.kind == quest.ROLE_GROUP_ITEM]
    assert [o.decision for o in head.options] == [_press(below, "exposure")]
    after = [*records, record(5, press.decision)]
    assert not readings.unsettled(decisions.fold(after))
    assert stage(_log(after, out), "data").progress.complete


def test_waiting_columns_with_different_reasons_are_each_their_own_line(tmp_path):
    """A table built here: three measures with a flag each on its blanks (the flags' reasons name
    their own column, so they differ), the three measures read alike by name, and three lab values
    read alike as measured columns."""
    rng = np.random.default_rng(5)
    n = 200
    frame = pd.DataFrame({"pid": np.arange(n), "y": rng.normal(100, 15, n)})
    for c in ("bp_sys", "weight", "waist"):
        x = rng.normal(120, 10, n)
        blank = rng.random(n) < 0.15
        frame[f"imputed_{c}"] = blank.astype(int)
        x[blank] = np.nan
        frame[c] = x
    for c in ("lab_a", "lab_b", "lab_c"):
        frame[c] = rng.normal(5, 1, n)
    frame.to_csv(tmp_path / "t.csv", index=False)
    out, records = _proposed(tmp_path / "t.csv", tmp_path, "clinical", "y")
    judged = _judgments(out)
    assert sorted(len(v) for v in judged.values()) == [1, 1, 1, 3, 3]  # what the table holds
    lines = _role_lines(_log(records, out))
    assert len(lines) == len(judged)
    found = sorted(tuple(i.column for i in l.items if i.kind == "role") for l in lines)
    assert found == sorted(tuple(v) for v in judged.values())
    for line in lines:
        presses = [o.decision for i in line.items for o in i.options
                   if o.decision and o.decision["kind"] == "confirm_readings"]
        # one press per line, and it confirms that line's columns alone
        assert len(presses) == (1 if len(line.items) > 1 else 0)
    # the fit's refusal offers each judgment's one press, and each lone column its own
    state = decisions.fold(records)
    _check_one_press_per_judgment(_confirms(quest.fit_waits(state, artifact=out).exits), state,
                                  out)
