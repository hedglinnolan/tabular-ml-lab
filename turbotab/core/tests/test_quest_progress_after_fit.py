"""Q-c · quest progress after Fit, and SURFACING_POLICY §7.2's I11 as the orchestrator ruled it
(WAVE_C6A_PLAN §7, ruling 3).

Under Estimate and Describe, once Fit is pressed, Results reads 0 of N (each exhibit's wording and
placement is one of its objectives, calm/FOUNDATION §3 and §8), not "0 of 0, complete"; Write-up is
not reached until Results is placed. The references are written by hand here: the design's state
(``design/quest-models-stage``, ``screens.ts``: after Fit, ``results = {answered: 0, required:
EXHIBITS.length, complete: false}`` and ``writeup = null``), and the exhibits the NHANES fixture is
served, listed one by one from each stage's ``requires`` (``stages/__init__.py``) against the
fixture's answers.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from turbotab.core import fit_press, quest
from turbotab.core.quest import Progress, QuestLine, QuestLog, QuestStage, Reopened
from turbotab.core.tests import path_fuzzer
from turbotab.core.tests.test_path_fuzzer import _decide, _log_of

# Every compute stage whose card is an exhibit (CROSSWALK, "Engine stages and quest stages"),
# listed by hand. The substitution curve has no card of its own yet, so it is not one.
EXHIBIT_MAP = ["usual_intake", "fit", "sensitivity", "calibration", "secondary", "scales",
               "effects", "causal", "time_varying", "modification", "explain", "evaluation"]

# The ones the engine serves on the NHANES fixture answered under Estimate (models ["linear"],
# the dietary lens, an adjustment set and an estimand, nothing else declared), by each stage's
# ``requires``:
#   usual_intake  lens, purpose                      served
#   fit           models, purpose                    served
#   secondary     models, adjustment, purpose        served
#   effects       models, estimand, purpose          served
#   evaluation    models, purpose                    served
#   sensitivity   sensitivity                        no sensitivity answer
#   calibration   measurement_error, models, purpose no regression-calibration answer
#   scales        scales, models, purpose            no scale declared
#   causal        causal, models, purpose            no causal-lane answer
#   time_varying  estimand, unit, purpose            one row per participant: no unit answer
#   modification  modifications, models, purpose     no effect modifier declared
#   explain       explain, models, purpose           no explain answer
NHANES_ESTIMATE = ["usual_intake", "fit", "secondary", "effects", "evaluation"]
NOT_ANSWERED = ["sensitivity", "measurement_error", "scales", "causal", "unit", "modifications",
                "explain"]


def _after_fit(purpose: str, extra: list[dict] | None = None) -> dict[str, QuestStage]:
    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, purpose)
    for payload in extra or []:
        state = _decide(records, payload)
    fit = fit_press.fit_lock(state, records, pressed=True, held=False, estimate=None, opened=True)
    log = _log_of(state, records, stages={"fit": {"status": "fresh"}}, fit=fit)
    return {s.key: s for s in log.stages}


def test_the_exhibit_map_and_the_version():
    assert sorted(quest.EXHIBIT_STAGES) == sorted(EXHIBIT_MAP)
    assert quest.QUEST_VERSION == 4


def test_the_fixture_answers_none_of_the_exhibits_it_is_not_served():
    state, _artifacts, _records = path_fuzzer.answered_through(path_fuzzer.NHANES, "inference")
    assert [slot for slot in NOT_ANSWERED if getattr(state, slot) is not None] == []
    assert state.models == ["linear"] and state.lens == ["dietary"]
    assert state.adjustment and state.estimand is not None
    assert sorted(quest.served_exhibits(state)) == sorted(NHANES_ESTIMATE)


def test_results_after_fit_under_estimate_reads_0_of_n_served_and_is_not_complete():
    results = _after_fit("inference")["results"]
    assert results.reached
    # nothing in Results is a counted Decide yet: the whole count is the served exhibits
    assert [l for l in results.lines if l.label == "Decide" and l.counted] == []
    assert results.progress == Progress(answered=0, required=5, complete=False)


def test_an_explain_answer_serves_one_more_exhibit():
    # explain requires explain, models and purpose: answered, it is served, 0 of 6
    results = _after_fit("inference", [{"kind": "set_explain", "reseeds": 0}])["results"]
    assert results.progress == Progress(answered=0, required=6, complete=False)


def test_write_up_is_not_reached_until_results_is_placed():
    writeup = _after_fit("inference")["writeup"]
    assert writeup.progress is None and not writeup.reached


def test_before_fit_results_has_no_progress_at_all():
    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, "inference")
    log = _log_of(state, records)  # Fit not pressed: nothing opened
    stages = {s.key: s for s in log.stages}
    assert not stages["results"].reached and stages["results"].progress is None


def test_predict_is_unchanged():
    # As on the base (553f6f27): Results asks the updating question and the seal (waiting), the
    # explain question uncounted; 0 of 2. Write-up is reached, asks nothing: 0 of 0, complete.
    stages = _after_fit("prediction")
    results = stages["results"]
    assert [(l.id, l.counted) for l in results.lines if l.label == "Decide"] == [
        ("decision:set_updating", True), ("q:open_seal", True), ("decision:set_explain", False)]
    assert results.progress == Progress(answered=0, required=2, complete=False)
    assert results.reopened == []
    assert stages["writeup"].reached
    assert stages["writeup"].progress == Progress(answered=0, required=0, complete=True)


def test_a_goal_changed_to_estimate_says_why_results_counts_its_exhibits():
    # Under Predict Results was open; the goal becomes Estimate. On the Predict path no
    # adjustment set or estimand was answered, so the served exhibits are usual_intake, fit and
    # evaluation; Results says the change to Your question made them out of date.
    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, "prediction")
    state = _decide(records, {"kind": "set_purpose", "purpose": "inference"})
    fit = fit_press.fit_lock(state, records, pressed=True, held=False, estimate=None, opened=True)
    log = _log_of(state, records, stages={"fit": {"status": "fresh"}}, fit=fit)
    results = next(s for s in log.stages if s.key == "results")
    assert results.reopened == [Reopened(
        changed_in="question", decision_id=records[-1].id, kind="set_purpose",
        results=["usual_intake", "fit", "evaluation"],
        sentence="Your change to Your question made 3 results in Results out of date.")]
    # once the plan is locked for the new goal, the drop-back is behind it
    state = _decide(records, {"kind": "lock_plan", "seen_target": state.target})
    fit = fit_press.fit_lock(state, records, pressed=True, held=False, estimate=None, opened=True)
    log = _log_of(state, records, stages={"fit": {"status": "fresh"}}, fit=fit)
    assert next(s for s in log.stages if s.key == "results").reopened == []


def test_a_goal_changed_to_predict_says_why_who_s_in_asks_the_split():
    # Under Estimate the split is For the record, not asked (0 of it counted). The goal changed
    # to Predict asks it in Who's in, which names the change: the path fuzzer's 2,000-journey tier
    # found Who's in, complete, gaining the line with no record (seed 968).
    from turbotab.core.quest import ReopenedBy
    from turbotab.core.tests.test_path_fuzzer import _line as line_of

    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, "inference",
                                                              stop="split")
    stage, split = line_of(_log_of(state, records), "split")
    assert (stage.key, split.label, split.counted) == ("whos_in", "For the record", False)
    state = _decide(records, {"kind": "set_purpose", "purpose": "prediction"})
    log = _log_of(state, records)
    stage, split = line_of(log, "split")
    assert (split.label, split.counted, split.status) == ("Decide", True, "open")
    assert split.reopened_by == ReopenedBy(decision_id=records[-1].id, kind="set_purpose",
                                           stage="question")
    assert any("q:split" in r.questions and r.changed_in == "question" for r in stage.reopened)
    # a first answer of Predict is the journey going forward: it reopens nothing
    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, "prediction",
                                                              stop="split")
    assert line_of(_log_of(state, records), "split")[1].reopened_by is None


# ── I11, as §7.2 states it (ruling 3) ────────────────────────────────────────


def _line(key: str, status: str) -> QuestLine:
    return QuestLine(id=f"q:{key}", key=key, source="question", label="Decide", name=key,
                     status=status, counted=True, order=1.0)


def _stage(key: str, lines: list[QuestLine], reopened: list[Reopened] | None = None,
           exhibits: int = 0, reached: bool = True) -> QuestStage:
    answered = sum(l.status == "answered" for l in lines)
    required = len(lines) + exhibits
    return QuestStage(key=key, name=key, reached=reached,
                      progress=Progress(answered=answered, required=required,
                                        complete=answered == required) if reached else None,
                      lines=lines, reopened=reopened or [])


def _snap(action: str, kind: str, reopening: bool, stages: list[QuestStage],
          purpose: str | None = None) -> SimpleNamespace:
    # I11 reads the log, the action and its kind, and the goal
    return SimpleNamespace(action=action, kind=kind, reopening=reopening,
                           log=QuestLog(stages=stages, kinds=quest.kind_stages()),
                           steps=[], state=SimpleNamespace(purpose=purpose), artifacts={})


def _i11(prev: SimpleNamespace, snap: SimpleNamespace) -> list[str]:
    return path_fuzzer.i11_monotone_progress(prev, snap)


def _reopened(stage: str, kind: str = "set_purpose") -> Reopened:
    return Reopened(changed_in=stage, decision_id="d9", kind=kind, questions=[],
                    sentence="Your change reopened it.")


def test_required_grows_only_within_the_stage_being_worked():
    # The energy answer, decided in Models, makes the form question apply there: 1 of 2 becomes
    # 2 of 3, within the stage being worked, with no record. It passes.
    assert quest.kind_stages()["set_energy_adjustment"] == "models"
    prev = _snap("answer", "select_models", False, [
        _stage("models", [_line("models", "answered"), _line("energy_adjustment", "open")])])
    models = [_line("models", "answered"), _line("energy_adjustment", "answered")]
    snap = _snap("answer", "set_energy_adjustment", False, [
        _stage("models", [*models, _line("form", "open")])])
    assert _i11(prev, snap) == []
    # Models complete at 2 of 2, and an answer decided there (the model sequence) finds the form
    # question: still the stage being worked. It passes.
    assert quest.kind_stages()["set_model_sequence"] == "models"
    prev = _snap("answer", "set_energy_adjustment", False, [_stage("models", models)])
    snap = _snap("answer", "set_model_sequence", False, [
        _stage("models", [*models, _line("form", "open")])])
    assert _i11(prev, snap) == []
    # The same line found by an answer decided in Who's in: Models is not the stage being
    # worked, so the complete stage was reopened silently.
    assert quest.kind_stages()["set_aggregation"] == "whos_in"
    prev = _snap("answer", "set_exclusions", False, [
        _stage("whos_in", [_line("aggregation", "open")]), _stage("models", models)])
    snap = _snap("answer", "set_aggregation", False, [
        _stage("whos_in", [_line("aggregation", "answered")]),
        _stage("models", [*models, _line("form", "open")])])
    [message] = _i11(prev, snap)
    assert message.startswith("models was complete and gained 1 line (2/2 to 2/3)")


def test_a_forward_answer_that_reopens_a_complete_stage_while_an_earlier_one_asks_fails():
    # Your question still asks; Who's in is complete; an answer decided in Models makes a line
    # apply in Who's in. A stage still asking earlier on does not make Who's in the stage worked.
    assert quest.kind_stages()["select_models"] == "models"
    prev = _snap("answer", "set_energy_adjustment", False, [
        _stage("question", [_line("purpose", "answered"), _line("design", "open")]),
        _stage("whos_in", [_line("unit", "answered")]),
        _stage("models", [_line("models", "open")])])
    snap = _snap("answer", "select_models", False, [
        _stage("question", [_line("purpose", "answered"), _line("design", "open")]),
        _stage("whos_in", [_line("unit", "answered"), _line("aggregation", "open")]),
        _stage("models", [_line("models", "answered")])])
    [message] = _i11(prev, snap)
    assert message.startswith("whos_in was complete and gained 1 line")
    assert message.endswith("with no Reopened record (after answer select_models)")


def test_a_change_that_adds_to_a_complete_stage_without_a_reopened_record_fails():
    # The goal changes to Estimate (decided in Your question). Models, complete, gains the form
    # question; Results, complete under Predict at 2 of 2, gains its three exhibits, objectives
    # that are no line. Neither says why: both were reopened silently.
    assert quest.kind_stages()["set_purpose"] == "question"
    done = [_line("updating", "answered"), _line("open_seal", "answered")]
    prev = _snap("answer", "select_models", False, [
        _stage("models", [_line("models", "answered")]), _stage("results", done)], "prediction")
    stages = [_stage("models", [_line("models", "answered"), _line("form", "open")]),
              _stage("results", done, exhibits=3)]
    snap = _snap("change", "set_purpose", True, stages, "inference")
    assert _i11(prev, snap) == [
        "models was complete and gained 1 line (1/1 to 1/2) with no Reopened record "
        "(after change set_purpose)",
        "results was complete and gained 3 lines (2/2 to 2/5) with no Reopened record "
        "(after change set_purpose)"]
    # each with its Reopened record, it passes
    stages = [_stage("models", [_line("models", "answered"), _line("form", "open")],
                     [_reopened("question")]),
              _stage("results", done, [_reopened("question")], exhibits=3)]
    assert _i11(prev, _snap("change", "set_purpose", True, stages, "inference")) == []
    # and an answered count that falls says why too
    prev = _snap("answer", "select_models", False, [
        _stage("models", [_line("models", "answered"), _line("form", "answered")])])
    snap = _snap("change", "set_purpose", True, [
        _stage("models", [_line("models", "answered"), _line("form", "open")])])
    assert any("answered count fell from 2 to 1" in m for m in _i11(prev, snap))


def test_write_up_closes_without_a_reason_only_when_the_goal_became_estimate():
    writeup = _stage("writeup", [])
    closed = _stage("writeup", [], reached=False)
    results = _stage("results", [_line("open_seal", "answered")])
    # Predict to Estimate: Write-up waits for Results to be placed, by definition
    prev = _snap("answer", "select_models", False, [results, writeup], "prediction")
    snap = _snap("change", "set_purpose", True, [results, closed], "inference")
    assert _i11(prev, snap) == []
    # already under Estimate, Write-up closing with nothing saying why is still caught
    prev = _snap("answer", "select_models", False, [results, writeup], "inference")
    snap = _snap("answer", "set_explain", False, [results, closed], "inference")
    [message] = _i11(prev, snap)
    assert message == ("writeup stopped being reached with nothing saying why "
                       "(after answer set_explain)")


@pytest.mark.xfail(strict=True, reason=(
    "Engine conflict with ruling 3, reported, not fixed in Q-c: after the unit is changed, the "
    "aggregation answer (a first answer, decided in Who's in) makes the regression-calibration "
    "declaration apply in a complete Models, and quest._declaration_asked_again names a cause "
    "only for a record that changes an earlier answer, so Models says nothing."))
def test_the_aggregation_answer_that_makes_calibration_apply_says_why_in_models():
    journey = path_fuzzer.run_journey(26)
    prev, snap = journey.snapshots[38], journey.snapshots[39]
    assert (snap.action, snap.kind) == ("answer", "set_aggregation")
    assert path_fuzzer.i11_monotone_progress(prev, snap) == []
