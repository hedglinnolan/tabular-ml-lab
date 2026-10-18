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

from turbotab.core import decisions, quest, sweep
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


# ── N9 beside N5: a noticing the unlisted event question decides (W1-W round 2, P1) ──────────
#
# The triage proposes a noticing routed to a question as a limitation only where that question is
# already answered and its answer stands (the re-walk's N9). While the outcome's reading recomputes
# the event question is not listed (N5), but it is not answered: a noticing it decides is still to
# come, as it was while the event line was listed as waiting. Built by construction: a finding
# routed to the event question, on the steps above.
POSITIVE_CLASS = {"findings": [
    {"id": "positive_class", "severity": "warning", "summary": "Which class is the event?",
     "affected_columns": ["glucose"], "routes_to": "event", "repairs": [], "answered_by": None}]}
STILL_TO_COME = "It is decided at the event question in Your question, still to come."


def _event_answered() -> list[InterviewStep]:
    return [InterviewStep(key="event", status="answered", decision_id="answer-event")
            if s.key == "event" else s for s in _steps("models", [])]


def _triaged(steps: list[InterviewStep], records: list = RECORDS):
    state = decisions.fold(records)
    log = quest.quest_log(state, records, steps, findings=POSITIVE_CLASS)
    [item] = sweep.triage(state, log, POSITIVE_CLASS, records).items
    return log, item


def test_a_noticing_the_unlisted_event_question_decides_is_still_to_come():
    log, item = _triaged(_steps("event", ["target_info"]))
    assert "event" not in {l.key for s in log.stages for l in s.lines}  # N5: not listed yet
    assert (item.recommended, item.blocker, item.limitation) == ("act_on_it", False, False)
    assert item.reason == STILL_TO_COME
    # The same while the event question is listed and open.
    _log_open, listed = _triaged(_steps("event", []))
    assert (listed.recommended, listed.reason) == ("act_on_it", STILL_TO_COME)


def test_a_noticing_the_answered_event_question_decides_is_left_as_a_limitation():
    # The guard's other side: the event question answered, its answer standing.
    log, item = _triaged(_event_answered())
    assert line(log, "event").status == "answered"
    assert (item.recommended, item.limitation) == ("could_bias", True)
    assert item.reason.startswith("It is decided at the event question in Your question, already "
                                  "answered, and that answer stands.")


def test_the_quest_log_and_the_triage_read_the_unlisted_question_alike():
    # A disposition the person chose against the recommendation while the event question is not
    # listed (a limitation, though it is still to come) stands while that recommendation does: the
    # triage reads it as recorded, and the quest log's noticing line is answered by the same record.
    steps = _steps("event", ["target_info"])
    log, item = _triaged(steps)
    d = {"kind": "confirm_sweep", "stage": "results", "sweep": "noticings",
         "lines": [{"id": item.line, "key": item.id, "value": "could_bias",
                    "basis": sweep.basis(item.recommended, item.reason)}]}
    records = [*RECORDS, record(5, d)]
    after_log, after = _triaged(steps, records)
    assert after.recorded == "could_bias"
    noticing = line(after_log, "positive_class")
    assert (noticing.status, noticing.decision_id) == ("answered", "r5")
