"""W1-R finding 1 · a role proposed below high confidence waits in Your data, one line per column
(``turbotab/core/quest.py``; the walkthrough of 2026-10-17, finding 1).

BLUEPRINT §14 rule 2: a medium or low proposal "needs its own confirmation; it never rides along in
a bulk 'confirm all'". A roles answer that records such a proposal exactly as proposed keeps it in
``roles_unconfirmed``, and no number-changing default reads it until a ``confirm_role`` of its own
(``readings.unsettled``). The expectations come from the construction of each log: which columns
the roles answer recorded unconfirmed, and which were confirmed since. The predictor set is the
ledger's (``readings.predictor_columns``), the consumer the cohort and the fit read.
"""
from __future__ import annotations

import json
import typing
from pathlib import Path

from turbotab.core import decisions, quest, readings, sweep
from turbotab.core.readings import Reading
from turbotab.core.tests.test_stage_registry import record, stage, steps_until

# What the shell's role card and the mock read from the engine, so neither retypes its words
# (``turbotab/frontend/src/quest/cards/roles``; written by the last test here).
FIXTURE = (Path(__file__).resolve().parents[2] / "frontend" / "src" / "quest" / "cards" / "roles"
           / "fixture.json")

ROLES = {"SEQN": "identifier", "age": "covariate", "bp_sys": "covariate", "waist": "covariate",
         "cycle": "excluded"}
WAITING = ["bp_sys", "waist", "cycle"]  # recorded as proposed, below high confidence
PROPOSALS = {"columns": [
    {"column": "SEQN", "proposed": "identifier", "confidence": "high", "reason": "unique per row"},
    {"column": "age", "proposed": "covariate", "confidence": "high", "reason": "ages in years"},
    {"column": "bp_sys", "proposed": "covariate", "confidence": "medium",
     "reason": "the name reads as systolic blood pressure; nothing in the values confirms it"},
    {"column": "waist", "proposed": "covariate", "confidence": "low",
     "reason": "the name alone suggests a waist circumference"},
    {"column": "cycle", "proposed": "excluded", "confidence": "medium",
     "reason": "a few repeating values that may only label the survey cycle"},
]}


def _records(unconfirmed: list[str], *later: dict) -> list:
    out = [
        record(1, {"kind": "set_lens", "lenses": ["clinical"]}),
        record(2, {"kind": "set_target", "column": "glucose"}),
        record(3, {"kind": "set_purpose", "purpose": "prediction"}),
        record(4, {"kind": "set_roles", "roles": dict(ROLES), "unconfirmed": list(unconfirmed)}),
    ]
    return out + [record(5 + i, d) for i, d in enumerate(later)]


def _log(records: list, **kw) -> quest.QuestLog:
    return quest.quest_log(decisions.fold(records), records, steps_until("exclusions"), **kw)


def _role_lines(log: quest.QuestLog) -> list[quest.QuestLine]:
    return [l for l in stage(log, "data").lines if l.id == "decision:confirm_role"]


def _confirm(column: str, role: str) -> dict:
    return {"kind": "confirm_role", "column": column, "role": role}


# ── the line: one per waiting column, open and counted ───────────────────────


def test_each_role_recorded_below_high_is_an_open_counted_line_of_your_data():
    log = _log(_records(WAITING), artifacts={"roles": PROPOSALS})
    lines = _role_lines(log)
    assert [l.items[0].column for l in lines] == WAITING  # the table's order
    for line, column in zip(lines, WAITING):
        assert (line.source, line.label, line.status, line.counted) == (
            "reading", "Decide", "open", True)
        assert line.key == f"confirm_role:{column}"
        [item] = line.items
        assert (item.kind, item.column, item.value) == ("role", column, ROLES[column])
        # the evidence the roles stage served for its proposal, word for word
        reason = next(p["reason"] for p in PROPOSALS["columns"] if p["column"] == column)
        assert item.evidence == reason
        # the ledger's own words for a role that waits, the fit's refusal's where a press leaves the
        # column out (an exclusion's names its own presses: the tests of P4 below)
        if "excluded" in quest.role_choices(ROLES[column]):
            assert line.reason == readings.unsettled_message([column], "the model's predictors")
        else:
            assert "leave it out" not in line.reason
    data = stage(log, "data")
    settled = stage(_log(_records([])), "data")
    # Three more objectives than the same answers with nothing waiting, none of them answered.
    assert data.progress.required == settled.progress.required + len(WAITING)
    assert data.progress.answered == settled.progress.answered
    assert settled.progress.complete and not data.progress.complete


def test_a_roles_answer_with_nothing_below_high_adds_no_line():
    log = _log(_records([]), artifacts={"roles": PROPOSALS})
    assert _role_lines(log) == []
    assert stage(log, "data").progress.complete


def test_a_waiting_column_is_no_predictor_until_its_own_confirmation_and_then_it_enters():
    state = decisions.fold(_records(WAITING))
    # Of the recorded covariates only `age` is settled; the identifier and the exclusion enter no
    # model whatever their confirmation.
    assert readings.predictor_columns(state) == ["age"]
    records = _records(WAITING, _confirm("bp_sys", "covariate"))
    state = decisions.fold(records)
    assert readings.predictor_columns(state) == ["age", "bp_sys"]
    log = _log(records)
    bp, waist, cycle = _role_lines(log)
    # The confirmed line is answered by its own record and still counted, so the bar moves forward
    # rather than losing the objective.
    assert (bp.status, bp.counted, bp.decision_id) == ("answered", True, records[-1].id)
    assert (waist.status, cycle.status) == ("open", "open")
    first = stage(_log(_records(WAITING)), "data").progress
    assert stage(log, "data").progress.answered == first.answered + 1
    assert stage(log, "data").progress.required == first.required


def test_a_changed_role_settles_its_line_with_the_role_the_person_chose():
    records = _records(WAITING, _confirm("bp_sys", "covariate"), _confirm("waist", "excluded"),
                       _confirm("cycle", "excluded"))
    state = decisions.fold(records)
    assert readings.predictor_columns(state) == ["age", "bp_sys"]
    assert state.roles["waist"] == "excluded"
    log = _log(records)
    assert [l.status for l in _role_lines(log)] == ["answered"] * 3
    assert _role_lines(log)[1].items[0].value == "excluded"
    assert stage(log, "data").progress.complete


def test_the_lines_wait_for_the_roles_question_while_it_is_asked_again():
    # A roles answer that no longer holds is asked again by the Router: the columns it left
    # waiting are asked with it, not beside it.
    records = _records(WAITING)
    log = quest.quest_log(decisions.fold(records), records, steps_until("roles"))
    assert _role_lines(log) == []


# ── one column per press, never one press for several ────────────────────────


def _decisions(line: quest.QuestLine) -> list[dict]:
    return [o.decision for i in line.items for o in i.options if o.decision is not None]


def test_each_way_forward_confirms_exactly_one_column_with_its_own_confirm_role():
    log = _log(_records(WAITING), artifacts={"roles": PROPOSALS})
    assert len(_role_lines(log)) == len(WAITING)
    for line in _role_lines(log):
        column = line.items[0].column
        found = _decisions(line)
        assert found, column
        assert all(d == {"kind": "confirm_role", "column": column, "role": d["role"]}
                   for d in found), found
        # the proposal first, then the roles the ledger offers instead, each once
        roles = [d["role"] for d in found]
        assert roles[0] == ROLES[column] and len(set(roles)) == len(roles)
        assert set(roles[1:]) == set(readings.ROLE_CHANGES[ROLES[column]])


# ── why it waits names only what its presses do (the verifier's P4) ───────────

# One waiting column per kind of proposal: two whose changes leave it out of the models, a flag, a
# time, a design column and a cluster whose only change keeps it in as a covariate, and one proposed
# as left out of the models.
EVERY_KIND = {"SEQN": "identifier", "bp_sys": "covariate", "diet": "exposure",
              "imputed_waist": "flag", "exam_date": "time", "sdmvstra": "design", "sdmvpsu": "cluster",
              "cycle": "excluded"}


def _kinds_log() -> quest.QuestLog:
    waiting = [c for c in EVERY_KIND if c != "SEQN"]
    records = [
        record(1, {"kind": "set_lens", "lenses": ["clinical"]}),
        record(2, {"kind": "set_target", "column": "glucose"}),
        record(3, {"kind": "set_purpose", "purpose": "prediction"}),
        record(4, {"kind": "set_roles", "roles": dict(EVERY_KIND), "unconfirmed": waiting}),
    ]
    return quest.quest_log(decisions.fold(records), records, steps_until("exclusions"))


def _instead(line: quest.QuestLine) -> list[str]:
    """The roles a line's presses record other than its proposal, read from its options."""
    item = line.items[0]
    return [d["role"] for d in _decisions(line) if d["role"] != item.value]


def test_a_role_lines_reason_says_leave_it_out_only_where_a_press_leaves_it_out():
    lines = {l.items[0].column: l for l in _role_lines(_kinds_log())}
    assert set(lines) == set(EVERY_KIND) - {"SEQN"}
    said = {c: "leave it out" in (l.reason or "") for c, l in lines.items()}
    offered = {c: "excluded" in _instead(l) for c, l in lines.items()}
    # It fires where a press records the column as left out of the models, and is silent where none
    # does (the flag, the time, the design column and the cluster offer only a covariate; the
    # exclusion is the proposal itself).
    assert said == offered
    assert {c for c, v in offered.items() if v} == {"bp_sys", "diet"}
    for column in ("bp_sys", "diet"):
        # where it holds, the line keeps the fit's own words (the refusal says the same)
        assert lines[column].reason == readings.unsettled_message([column], quest.ROLE_WAITS_FOR)


def test_a_role_lines_reason_names_the_role_its_only_change_records():
    for column in ("imputed_waist", "exam_date", "sdmvstra", "sdmvpsu"):
        line = next(l for l in _role_lines(_kinds_log()) if l.items[0].column == column)
        others = _instead(line)
        assert others == ["covariate"], (column, others)
        first, _, last = line.reason.partition(" Confirm it after reading why it was proposed, ")
        # The first sentence is the fit's for the same column; the last names each press's role in
        # the words its option label uses.
        fit = readings.unsettled_message([column], quest.ROLE_WAITS_FOR)
        assert fit.startswith(first + " ")
        labels = [o.label for o in line.items[0].options]
        for role in others:
            words = readings.role_words(role)
            assert f"`{column}` is {words}" in labels
            assert words in last, (column, line.reason)


def test_an_exclusion_lines_reason_offers_keeping_it_in_the_model():
    [line] = [l for l in _role_lines(_kinds_log()) if l.items[0].column == "cycle"]
    # Every press but the confirmation puts the column in the model as a predictor.
    assert _instead(line) and set(_instead(line)) <= set(readings.PREDICTOR_ROLES)
    assert "leave it out" not in line.reason
    assert line.reason.endswith("or keep it in the model.")


def test_confirm_all_holds_none_of_them():
    log = _log(_records(WAITING), artifacts={"roles": PROPOSALS})
    data = stage(log, "data")
    assert [l.label for l in _role_lines(log)] == ["Decide"] * len(WAITING)
    assert not [s for s in sweep.sweep_lines(data) if s.key.startswith("confirm_role:")]


def _ask_line(log: quest.QuestLog) -> quest.QuestLine:
    return next(l for s in log.stages for l in s.lines if l.key == "ask:models")


def _several(decision: dict) -> bool:
    return decision.get("kind") == "confirm_readings" and len(decision.get("items") or []) > 1


def test_the_fits_ask_offers_no_block_that_confirms_several_roles_in_one_press():
    records = _records(WAITING)
    state = decisions.fold(records)
    asked = quest.fit_waits(state)
    assert asked is not None and {r.column for r in asked.readings} == set(WAITING)
    log = quest.quest_log(state, records, steps_until(None), asked=[asked])
    offered = _decisions(_ask_line(log))
    assert offered and not [d for d in offered if _several(d)]
    for item in _ask_line(log).items:
        assert all(o.decision.get("column") == item.column for o in item.options if o.decision)


def test_the_fits_ask_keeps_a_block_of_several_codes_or_amounts():
    # §14.2: a block of readings that are not roles settles exactly what it lists, and stays.
    state = decisions.fold(_records([]))
    codes = [Reading((c,), "code_or_count", "amount", "medium", "whole numbers") for c in
             ("visits", "children")]
    asked = quest.Asked("models", "Tell me about these columns: …", tuple(codes),
                        tuple(readings.ask_exits(codes, state)))
    log = quest.quest_log(state, _records([]), steps_until(None), asked=[asked])
    blocks = [d for d in _decisions(_ask_line(log)) if _several(d)]
    assert blocks and {i["column"] for i in blocks[0]["items"]} == {"visits", "children"}


# ── the shell's fixture ──────────────────────────────────────────────────────


def _fixture() -> dict:
    """Your data's role lines on the NHANES columns the capture under Predict left waiting (two
    open, one confirmed since, a flag), and the words and choices they are made of, by role."""
    roles = {"SEQN": "identifier", "age": "covariate", "bp_sys": "covariate",
             "waist": "covariate", "imputed_waist": "flag", "cycle_begin_year": "excluded"}
    waiting = ["bp_sys", "waist", "imputed_waist", "cycle_begin_year"]
    proposals = {"columns": [
        {"column": c, "proposed": r, "confidence": "medium" if c in waiting else "high",
         "reason": f"the reason the roles stage gives for `{c}`"} for c, r in roles.items()]}
    records = [
        record(1, {"kind": "set_lens", "lenses": ["clinical"]}),
        record(2, {"kind": "set_target", "column": "glucose"}),
        record(3, {"kind": "set_purpose", "purpose": "prediction"}),
        record(4, {"kind": "set_roles", "roles": roles, "unconfirmed": waiting}),
        record(5, _confirm("waist", "covariate")),
    ]
    log = quest.quest_log(decisions.fold(records), records, steps_until("exclusions"),
                          artifacts={"roles": proposals})
    # Without the roles stage's artifact at hand, the evidence is the ledger's own reading's.
    [unread, *_] = _role_lines(quest.quest_log(decisions.fold(records), records,
                                               steps_until("exclusions")))
    every = typing.get_args(decisions.Role)
    # Why a line waits names what its presses do, which differs by role: one template per role.
    templates = quest.role_line_text("{column}", "{role}", "{other}")
    del templates["reason"]
    return {
        "about": "Written by turbotab/core/tests/test_quest_unconfirmed_roles.py from core/quest.py "
                 "(_role_lines): never edit by hand.",
        "place": {"stage": quest.ROLE_LINE.stage, "id": quest.ROLE_LINE.item,
                  "order": quest.ROLE_LINE.order},
        "key": quest.role_line_key("{column}"),
        "unread_evidence": unread.items[0].evidence,
        "templates": templates,
        "reasons": {r: quest.role_line_reason("{column}", r) for r in every},
        "words": {r: readings.role_words(r) for r in every},
        "choices": {r: quest.role_choices(r) for r in every},
        "lines": [l.model_dump(mode="json") for l in _role_lines(log)],
    }


def test_the_shells_role_fixture_is_the_engines():
    want = _fixture()
    have = json.loads(FIXTURE.read_text("utf-8")) if FIXTURE.is_file() else None
    if have != want:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(want, indent=2, ensure_ascii=False) + "\n", "utf-8")
    assert have == want, f"{FIXTURE} was out of date and has been written anew: commit it"
