"""The re-walk's N7 (2026-10-18): "Your change to Your data reopened 2 questions in Who's in" stood
beside the bar while Who's in read done (``turbotab/core/quest.py``).

The re-walk's own log (its decisions 19 to 27) showed why. The eligibility question was answered
with no rule, which settles neither noticing it decides (the out-of-range intakes, the impossible
diastolic pressures): ``repairs.matching_answer`` needs a rule that names one of a noticing's
columns. Both stayed open from the start, and under Predict they travel to the triage, so Who's in
counts neither. Then the families card confirmed five code-or-amount readings, a change in Your
data to a slot the eligibility question reads, and the quest log named it as what reopened them:
a reopening that never happened. A noticing is reopened by its question only where an answer to
that question settled it when given; and a reason the stage gives counts only the lines it still
holds open, so it clears once they are answered, whether by their own answer, by the triage's
record or by "Confirm all" (W1-W round 2, P3: the two records answered the line and left its
reopening standing).

The logs are built here by hand; the expectations come from what each log holds.
"""
from __future__ import annotations

import pytest

from turbotab.core import decisions, quest, sweep
from turbotab.core.tests.test_stage_registry import line, record, stage, steps_until

ROLES = {"SEQN": "identifier", "kcal": "energy", "sugar": "exposure", "age": "covariate"}
INTAKE = {"id": "pack::dietary::implausible_intake", "severity": "warning",
          "summary": "Some energy reports are outside 500 to 5,000 kcal",
          "affected_columns": ["kcal"], "routes_to": "exclusions", "repairs": []}
KCAL_RULE = {"column": "kcal", "low": 500.0, "high": 5000.0, "reason": "implausible intake"}


def _start(purpose: str = "prediction") -> list:
    return [
        record(1, {"kind": "set_lens", "lenses": ["dietary"]}),
        record(2, {"kind": "set_target", "column": "glucose"}),
        record(3, {"kind": "set_purpose", "purpose": purpose}),
        record(4, {"kind": "set_roles", "roles": dict(ROLES), "unconfirmed": []}),
    ]


def _log(records: list, answered_by: str | None = None) -> quest.QuestLog:
    findings = {"findings": [{**INTAKE, "answered_by": answered_by}]}
    return quest.quest_log(decisions.fold(records), records, steps_until(None), findings=findings,
                           columns=["SEQN", "kcal", "sugar", "age", "glucose"])


def test_a_noticing_no_answer_ever_settled_is_never_said_to_be_reopened():
    records = [*_start(),
               record(5, {"kind": "set_exclusions", "rules": []}),
               # Your data: a reading the eligibility question reads (``question_reads``)
               record(6, {"kind": "confirm_reading", "reading": "code_or_count", "column": "age",
                          "value": "amount"})]
    log = _log(records)
    noticing = line(log, INTAKE["id"])
    assert noticing.status == "open" and noticing.reopened_by is None
    # No reason Who's in gives names it, nor the reading as its cause.
    assert not any(noticing.id in r.questions or r.decision_id == "r6"
                   for r in stage(log, "whos_in").reopened)


def test_a_noticing_its_withdrawn_answer_had_settled_is_reopened_and_says_so():
    # The guard's other side: the rule on `kcal` settled the noticing; the person withdrew it.
    settled = [*_start(), record(5, {"kind": "set_exclusions", "rules": [KCAL_RULE]})]
    assert line(_log(settled, answered_by="r5"), INTAKE["id"]).status == "answered"
    records = [*settled, record(6, {"kind": "revert", "decision_id": "r5"})]
    log = _log(records)
    noticing = line(log, INTAKE["id"])
    assert noticing.status == "open"
    assert (noticing.reopened_by.kind, noticing.reopened_by.decision_id) == ("revert", "r6")
    assert any(noticing.id in r.questions and r.decision_id == "r6"
               for r in stage(log, "whos_in").reopened)


def test_the_reason_clears_once_the_reopened_line_is_answered_again():
    # Under Estimate: the design is stated in Your question; "Confirm all" confirms it; the person
    # withdraws that confirmation, so the design is set for them again and the stage says why;
    # confirming it again answers it, and the reason clears.
    records = [*_start("inference")]
    state = decisions.fold(records)
    question = stage(quest.quest_log(state, records, steps_until(None, stated=("design",))),
                     "question")
    lines = [l.model_dump() for l in sweep.sweep_lines(question)]
    assert [l["key"] for l in lines] == ["design"]
    records += [record(5, {"kind": "confirm_sweep", "stage": "question", "lines": lines}),
                record(6, {"kind": "revert", "decision_id": "r5"})]
    reopened = stage(quest.quest_log(decisions.fold(records), records,
                                     steps_until(None, stated=("design",))), "question")
    assert line_of(reopened, "design").status == "set_for_you" and reopened.reopened
    records.append(record(7, {"kind": "confirm_sweep", "stage": "question", "lines": lines}))
    again = stage(quest.quest_log(decisions.fold(records), records,
                                  steps_until(None, stated=("design",))), "question")
    assert line_of(again, "design").status == "answered" and again.progress.complete
    assert again.reopened == []


def line_of(s: quest.QuestStage, key: str) -> quest.QuestLine:
    return next(l for l in s.lines if l.key == key)


@pytest.mark.parametrize("purpose", ["prediction", "inference"])
def test_a_reopened_noticing_the_triage_answers_clears_its_reason(purpose):
    # W1-W round 2, P3: the rule on `kcal` settled the noticing, the person withdrew it (so it is
    # reopened, and Who's in says why), and the triage then records it as a limitation. Its line is
    # answered by that record, so nothing holds it open: the line names no reopening, and Who's in's
    # reason no longer counts it.
    records = [*_start(purpose), record(5, {"kind": "set_exclusions", "rules": [KCAL_RULE]}),
               record(6, {"kind": "revert", "decision_id": "r5"})]
    findings = {"findings": [{**INTAKE, "answered_by": None}]}
    log = _log(records)
    noticing = line(log, INTAKE["id"])
    assert (noticing.reopened_by.kind, noticing.reopened_by.decision_id) == ("revert", "r6")
    assert any(noticing.id in r.questions for r in stage(log, "whos_in").reopened)
    t = sweep.triage(decisions.fold(records), log, findings, records)
    [item] = [i for i in t.items if i.id == INTAKE["id"]]
    assert item.recommended == "could_bias"
    d = {"kind": "confirm_sweep", "stage": t.stage, "sweep": "noticings",
         "lines": [{"id": item.line, "key": item.id, "value": "could_bias",
                    "basis": sweep.basis(item.recommended, item.reason)}]}
    after = _log([*records, record(7, d)])
    noticing = line(after, INTAKE["id"])
    assert (noticing.status, noticing.decision_id, noticing.reopened_by) == ("answered", "r7", None)
    assert not any(noticing.id in r.questions for r in stage(after, "whos_in").reopened)


def test_a_reopened_default_that_confirm_all_answers_clears_its_reason():
    # P3's other setter, under Estimate: the design was answered on its own and the answer
    # withdrawn, so it is set for the person again and Your question says why; "Confirm all" then
    # answers it, and nothing holds it open.
    records = [*_start("inference"), record(5, {"kind": "set_design", "design": "observational"}),
               record(6, {"kind": "revert", "decision_id": "r5"})]

    def question(recs: list) -> quest.QuestStage:
        return stage(quest.quest_log(decisions.fold(recs), recs,
                                     steps_until(None, stated=("design",))), "question")

    withdrawn = question(records)
    design = line_of(withdrawn, "design")
    assert design.status == "set_for_you" and design.reopened_by.decision_id == "r6"
    assert any(design.id in r.questions for r in withdrawn.reopened)
    lines = [l.model_dump() for l in sweep.sweep_lines(withdrawn)]
    confirmed = question([*records, record(7, {"kind": "confirm_sweep", "stage": "question",
                                               "lines": lines})])
    design = line_of(confirmed, "design")
    assert (design.status, design.decision_id, design.reopened_by) == ("answered", "r7", None)
    assert confirmed.reopened == []
