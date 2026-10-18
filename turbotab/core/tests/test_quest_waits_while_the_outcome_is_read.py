"""The re-walk's N5 (2026-10-18): while the outcome's reading recomputes (a change in Your data put
``target_info`` out of date as the shelf re-ranked), the Router cannot say yet whether the event
question is asked at all, so every later question waited "for the event question", which a numeric
outcome does not have (``turbotab/core/quest.py``, ``_question_lines``).

The steps are laid out here as ``interview.route`` lays them out in that moment, by construction:
the event question is the first unanswered one and waits only on the outcome's reading
(``target_info``, the stage its gate reads, ``quest.READ_BY_GATE``), and each later unanswered
question waits on it. Nothing answers the event question: glucose is numeric, and once the reading
is back the Router finds the event question not applicable. A line then waits for the result still
computing, never for a question whose asking that result decides; the event and follow-up lines are
not listed until the reading says whether they are asked. Where the event question is asked (open,
its reading fresh), the lines after it wait for it by name, as before.
"""
from __future__ import annotations

from turbotab.core import decisions, quest
from turbotab.core.interview import QUESTION_KEYS, InterviewStep
from turbotab.core.tests.test_stage_registry import line, record, stage

ROLES = {"SEQN": "identifier", "age": "covariate", "bp_sys": "covariate"}
RECORDS = [
    record(1, {"kind": "set_lens", "lenses": ["clinical"]}),
    record(2, {"kind": "set_target", "column": "glucose"}),
    record(3, {"kind": "set_purpose", "purpose": "prediction"}),
    record(4, {"kind": "set_roles", "roles": dict(ROLES), "unconfirmed": []}),
]
# Answered before the outcome's reading went out of date; the rest wait (the models question was
# being asked), and the questions that do not apply to this table are not applicable.
ANSWERED = {"lens", "target", "purpose", "roles", "exclusions", "missing", "split"}
NOT_APPLICABLE = {"orientation", "grain", "repeat_kind", "unit", "aggregation", "temporal",
                  "clusters", "survey", "estimand", "adjustment", "time_varying",
                  "energy_adjustment", "form", "modification", "causal", "substitution", "design"}


def _steps(first: str, first_waits_on: list[str]) -> list[InterviewStep]:
    out = []
    for key in QUESTION_KEYS:
        if key in ANSWERED:
            out.append(InterviewStep(key=key, status="answered", decision_id=f"answer-{key}"))
        elif key in NOT_APPLICABLE:
            out.append(InterviewStep(key=key, status="not_applicable", reason="does not apply"))
        elif key == first:
            out.append(InterviewStep(key=key, status="waiting" if first_waits_on else "open",
                                     waiting_on=list(first_waits_on)))
        else:
            own = ["target_info"] if key in ("task", "follow_up") and first_waits_on else []
            if key == "open_seal":
                own = ["fit"]
            out.append(InterviewStep(key=key, status="waiting", waiting_on=[first, *own]))
    return out


def _log(steps: list[InterviewStep]) -> quest.QuestLog:
    return quest.quest_log(decisions.fold(RECORDS), RECORDS, steps)


def _named(log: quest.QuestLog) -> dict[str, list[str]]:
    return {l.key: [w.key for w in l.waiting_for] for s in log.stages for l in s.lines
            if l.waiting_for}


def test_while_the_outcome_is_read_no_line_waits_for_the_event_question():
    log = _log(_steps("event", ["target_info"]))
    assert not any("event" in keys for keys in _named(log).values()), _named(log)
    models = line(log, "models")
    assert models.status == "waiting" and "target_info" in models.computing
    # The task is asked or stated for every outcome: its line waits for the reading too.
    task = line(log, "task")
    assert task.waiting_for == [] and "target_info" in task.computing
    # Whether the event and follow-up questions are asked at all is the reading's: not listed yet.
    listed = {l.key for s in log.stages for l in s.lines}
    assert "event" not in listed and "follow_up" not in listed
    assert all(l.status != "waiting" or "event" not in l.name for l in stage(log, "question").lines)


def test_an_event_question_that_is_asked_is_waited_for_by_name():
    # The guard's other side: the reading is fresh and the outcome is two-level, so the Router asks
    # the event question (open), and the questions after it wait for it by name.
    log = _log(_steps("event", []))
    assert line(log, "event").status == "open"
    assert "event" in _named(log)["models"]
    assert _named(log)["task"] == ["event"]
