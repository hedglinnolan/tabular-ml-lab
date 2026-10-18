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
    assert FEATURE in message and TRAIT in message


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
    assert head.kind == quest.ROLE_GROUP_ITEM and head.evidence == FEATURE
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
    assert genes.guess_words == "assay features" and genes.evidence == FEATURE
    presses = [e.decision for e in card.exits if e.decision and e.decision["kind"] ==
               "confirm_readings" and len(e.decision["items"]) > 1]
    assert sorted(len(d["items"]) for d in presses) == [2, 5]
    for d in presses:
        validate(d, _ctx(state))
    # silent: without the proposals, one line per column, as before
    alone = ask.card("models", ask.AskContext(state, {}))
    assert all(len(g.columns) == 1 for g in alone.groups if g.kind == "role")


# ── the record and the methods: one sentence per group ───────────────────────


@pytest.mark.parametrize("purpose, words", [("prediction", "assay features"),
                                            ("inference", "what you study")])
def test_a_groups_press_is_recorded_and_stated_in_one_sentence(purpose, words):
    records = _records(("genomics",), purpose)
    before = decisions.fold(records)
    press = _press(GENES, "exposure")
    said = voice.sentence_for(press, before, {"roles": PROPOSALS})
    assert said == (f"The {len(GENES)} columns `g00`, `g01` and 3 more, proposed as {words} at "
                    f"medium confidence, below high, for one reason (\"{FEATURE}\"), were "
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
