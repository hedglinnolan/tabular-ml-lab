"""Q-c · quest progress after Fit, and SURFACING_POLICY §7.2's I11 as the orchestrator ruled it
(WAVE_C6A_PLAN §7, ruling 3).

Under Estimate and Describe, once Fit is pressed, Results reads 0 of N (each exhibit's wording and
placement is one of its objectives, calm/FOUNDATION §3 and §8), not "0 of 0, complete"; Write-up is
not reached until Results is placed. The references are written by hand here: the design's state
(``design/quest-models-stage``, ``screens.ts``: after Fit, ``results = {answered: 0, required:
EXHIBITS.length, complete: false}`` and ``writeup = null``) and the exhibits the engine's fixture
serves, listed one by one.
"""
from __future__ import annotations

from types import SimpleNamespace

from turbotab.core import fit_press, quest
from turbotab.core.quest import Progress, QuestLine, QuestLog, QuestStage, Reopened
from turbotab.core.tests import path_fuzzer
from turbotab.core.tests.test_path_fuzzer import _log_of

# The exhibits the engine serves in Results, by the compute stage that makes each (CROSSWALK,
# "Engine stages and quest stages": the stages whose card is an exhibit), listed by hand. The
# substitution curve has no card of its own yet, so it is not one.
EXHIBITS = ["usual_intake", "fit", "sensitivity", "calibration", "secondary", "scales", "effects",
            "causal", "time_varying", "modification", "explain", "evaluation"]


def _after_fit(purpose: str) -> dict[str, QuestStage]:
    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, purpose)
    fit = fit_press.fit_lock(state, records, pressed=True, held=False, estimate=None, opened=True)
    log = _log_of(state, records, stages={"fit": {"status": "fresh"}}, fit=fit)
    return {s.key: s for s in log.stages}


def test_the_exhibits_are_the_ones_listed_by_hand():
    assert sorted(quest.EXHIBIT_STAGES) == sorted(EXHIBITS) and len(EXHIBITS) == 12
    assert quest.QUEST_VERSION >= 4


def test_results_after_fit_under_estimate_reads_0_of_n_and_is_not_complete():
    stages = _after_fit("inference")
    results = stages["results"]
    assert results.reached
    # nothing in Results is a counted Decide yet: the whole count is the exhibits, none decided
    assert [l for l in results.lines if l.label == "Decide" and l.counted] == []
    assert results.progress == Progress(answered=0, required=len(EXHIBITS), complete=False)


def test_write_up_is_not_reached_until_results_is_placed():
    writeup = _after_fit("inference")["writeup"]
    assert writeup.progress is None and not writeup.reached


def test_before_fit_results_has_no_progress_at_all():
    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, "inference")
    log = _log_of(state, records)  # Fit not pressed: nothing opened
    stages = {s.key: s for s in log.stages}
    assert not stages["results"].reached and stages["results"].progress is None


def test_predict_is_unchanged():
    stages = _after_fit("prediction")
    results = stages["results"]
    # what Results asks under Predict: its counted Decides (the seal's questions), counted by hand
    decides = [l for l in results.lines if l.label == "Decide" and l.counted]
    assert results.progress == Progress(
        answered=sum(l.status == "answered" for l in decides), required=len(decides),
        complete=all(l.status == "answered" for l in decides))
    assert results.progress.required < len(EXHIBITS)
    # Write-up asks nothing under Predict: reached, 0 of 0, complete
    assert stages["writeup"].reached
    assert stages["writeup"].progress == Progress(answered=0, required=0, complete=True)


# ── I11, as §7.2 states it (ruling 3) ────────────────────────────────────────


def _line(key: str, status: str) -> QuestLine:
    return QuestLine(id=f"q:{key}", key=key, source="question", label="Decide", name=key,
                     status=status, counted=True, order=1.0)


def _stage(key: str, lines: list[QuestLine], reopened: list[Reopened] | None = None) -> QuestStage:
    answered = sum(l.status == "answered" for l in lines)
    return QuestStage(key=key, name=key, reached=True,
                      progress=Progress(answered=answered, required=len(lines),
                                        complete=answered == len(lines)),
                      lines=lines, reopened=reopened or [])


def _snap(action: str, kind: str, reopening: bool, stages: list[QuestStage]) -> SimpleNamespace:
    # I11 reads the log, the action and its kind only
    return SimpleNamespace(action=action, kind=kind, reopening=reopening,
                           log=QuestLog(stages=stages, kinds=quest.kind_stages()),
                           steps=[], state=None, artifacts={})


def _i11(prev: SimpleNamespace, snap: SimpleNamespace) -> list[str]:
    return path_fuzzer.i11_monotone_progress(prev, snap)


def test_a_forward_answer_that_adds_a_line_to_the_stage_being_worked_passes():
    # Models: the energy answer is given (a forward step, decided in Models) and, with it,
    # a new line applies in the same stage: 1 of 2 becomes 2 of 3. Required grew, answered did not
    # fall, and the stage is the one being worked.
    assert quest.kind_stages()["set_energy_adjustment"] == "models"
    prev = _snap("answer", "select_models", False, [
        _stage("models", [_line("models", "answered"), _line("energy_adjustment", "open")])])
    snap = _snap("answer", "set_energy_adjustment", False, [
        _stage("models", [_line("models", "answered"), _line("energy_adjustment", "answered"),
                          _line("form", "open")])])
    assert _i11(prev, snap) == []
    # The same when the stage worked was complete and a later one still asks: a forward answer
    # decided in Your question finds one more line there.
    prev = _snap("answer", "set_target", False, [
        _stage("question", [_line("purpose", "answered")]),
        _stage("models", [_line("models", "open")])])
    snap = _snap("answer", "set_purpose", False, [
        _stage("question", [_line("purpose", "answered"), _line("design", "open")]),
        _stage("models", [_line("models", "open")])])
    assert _i11(prev, snap) == []


def test_a_change_that_adds_a_line_to_a_complete_stage_with_no_reopened_record_fails():
    # Models is complete; changing the goal (decided in Your question) makes a line apply in
    # Models, and Models says nothing: a complete stage that gained a line has been reopened, and
    # must say why.
    assert quest.kind_stages()["set_purpose"] == "question"
    prev = _snap("answer", "select_models", False, [
        _stage("models", [_line("models", "answered")])])
    snap = _snap("change", "set_purpose", True, [
        _stage("models", [_line("models", "answered"), _line("form", "open")])])
    [message] = _i11(prev, snap)
    assert "models" in message and "no Reopened record" in message and "after change" in message


def test_the_same_change_with_its_reopened_record_passes():
    reopened = Reopened(changed_in="question", decision_id="d2", kind="set_purpose",
                        questions=["q:form"], sentence="Your change to Your question reopened 1 question.")
    prev = _snap("answer", "select_models", False, [
        _stage("models", [_line("models", "answered")])])
    snap = _snap("change", "set_purpose", True, [
        _stage("models", [_line("models", "answered"), _line("form", "open")], [reopened])])
    assert _i11(prev, snap) == []


def test_a_forward_answer_that_adds_a_line_to_an_earlier_complete_stage_fails():
    # Your question is complete and the person is working in Models; the answer given there
    # makes a line apply in Your question, which says nothing: that stage was reopened silently.
    prev = _snap("answer", "select_models", False, [
        _stage("question", [_line("purpose", "answered")]),
        _stage("models", [_line("models", "open")])])
    snap = _snap("answer", "select_models", False, [
        _stage("question", [_line("purpose", "answered"), _line("design", "open")]),
        _stage("models", [_line("models", "answered")])])
    [message] = _i11(prev, snap)
    assert "question" in message and "no Reopened record" in message


def test_a_forward_answer_that_adds_a_line_to_a_later_complete_stage_passes():
    # The person is working in Who's in; the answer makes a line apply in Models ahead of it,
    # which is not behind the work: required grew in a stage the work has not left.
    prev = _snap("answer", "set_exclusions", False, [
        _stage("whos_in", [_line("aggregation", "open")]),
        _stage("models", [_line("models", "answered")])])
    snap = _snap("answer", "set_aggregation", False, [
        _stage("whos_in", [_line("aggregation", "answered")]),
        _stage("models", [_line("models", "answered"), _line("form", "open")])])
    assert _i11(prev, snap) == []


def test_an_answered_count_that_falls_with_no_reopened_record_fails():
    prev = _snap("answer", "select_models", False, [
        _stage("models", [_line("models", "answered"), _line("form", "answered")])])
    snap = _snap("change", "set_purpose", True, [
        _stage("models", [_line("models", "answered"), _line("form", "open")])])
    assert any("no Reopened record" in m for m in _i11(prev, snap))
