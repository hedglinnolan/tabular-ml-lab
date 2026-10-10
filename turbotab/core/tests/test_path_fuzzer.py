"""U16 · the offline path fuzzer (SURFACING_POLICY §7.1): seeded journeys over ``decisions.fold_onto``
and ``interview.route``, headless, holding the invariants of §7.2 that are implementable now
(``path_fuzzer.py`` names which, and how each is checked).

300 journeys run on every push; 2,000 in the full tier. Each invariant is shown to bite: with the
gate it checks broken on purpose, the fuzzer reports it. The two places the fuzzer found the
engine breaking an invariant are held here directly, each on the smallest state that showed it.
"""
from __future__ import annotations

import pytest

from turbotab.core import fit_press, quest, surfacing
from turbotab.core.decisions import ProjectState
from turbotab.core.interview import QUESTION_KEYS, InterviewStep, route
from turbotab.core.tests import path_fuzzer
from turbotab.core.tests.path_fuzzer import fuzz


def _held(report: path_fuzzer.Report) -> None:
    assert not report.violations, report.summary() + "\n" + "\n".join(
        str(v) for v in report.violations[:25])


def test_300_seeded_journeys_hold_the_invariants():
    report = fuzz(300)
    _held(report)
    # The journeys reach where the invariants matter: both goals, Fit and the lock, Results, and
    # the stated reopenings.
    assert report.goals.get("inference", 0) >= 50 and report.goals.get("prediction", 0) >= 50
    assert report.fits >= 50 and report.locked >= 200
    assert report.reached.get("results", 0) >= 200 and report.reopenings >= 300
    assert report.refused == 0


@pytest.mark.slow
def test_2000_seeded_journeys_hold_the_invariants():
    report = fuzz(2000, seed=300)
    _held(report)
    assert report.fits >= 400


def test_a_journey_is_its_seed():
    a, b = path_fuzzer.run_journey(7), path_fuzzer.run_journey(7)
    assert [(s.action, s.kind) for s in a.snapshots] == [(s.action, s.kind) for s in b.snapshots]
    assert a.snapshots[-1].log == b.snapshots[-1].log


# ── each invariant bites ─────────────────────────────────────────────────────


def _found(report: path_fuzzer.Report, invariant: str) -> bool:
    return any(v.invariant == invariant for v in report.violations)


def test_an_estimate_served_before_the_lock_is_caught(monkeypatch):
    monkeypatch.setattr(fit_press, "serving_gate", lambda state, pressed: None)
    assert _found(fuzz(10), "I1")


def test_the_outcome_beside_a_column_before_the_lock_is_caught(monkeypatch):
    monkeypatch.setattr(fit_press, "relationships_served", lambda artifact, state, pressed: artifact)
    assert _found(fuzz(10), "I6")


def test_a_mode_that_changes_a_tier_is_caught(monkeypatch):
    honest = surfacing.disclosure

    def relabels(log, mode="standard", remembered=()):
        shown = honest(log, mode, remembered)
        if mode == "knows_the_field":
            shown = [s for s in shown if s.label != "Confirm"]
        return shown

    monkeypatch.setattr(surfacing, "disclosure", relabels)
    assert _found(fuzz(10), "I10")


def test_a_registry_that_disagrees_with_the_router_is_caught(monkeypatch):
    real = surfacing.question_fires
    monkeypatch.setattr(surfacing, "question_fires",
                        lambda key, state, facts: key != "split" and real(key, state, facts))
    surfacing.registry.cache_clear()
    try:
        assert _found(fuzz(10), "I2")
    finally:
        monkeypatch.undo()
        surfacing.registry.cache_clear()


def test_an_open_line_that_waits_is_caught(monkeypatch):
    real = quest.quest_log

    def opens_the_waiting(*args, **kwargs):
        log = real(*args, **kwargs)
        for stage in log.stages:
            for line in stage.lines:
                if line.status == "waiting" and line.waiting_for:
                    line.status = "open"
        return log

    monkeypatch.setattr(quest, "quest_log", opens_the_waiting)
    assert _found(fuzz(10), "I4")


def test_progress_that_falls_silently_is_caught(monkeypatch):
    real = quest.quest_log
    calls = {"n": 0}

    def forgets(*args, **kwargs):
        log = real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            for stage in log.stages:
                for line in stage.lines:
                    if line.status == "answered" and line.source == "question":
                        line.status = "open"
                        line.reopened_by = None
                stage.reopened = []
        return log

    monkeypatch.setattr(quest, "quest_log", forgets)
    assert _found(fuzz(5), "I11")


# ── what the fuzzer found ────────────────────────────────────────────────────


def test_under_predict_the_seal_question_does_not_open_results_before_fit():
    # I1: with Models answered under Predict and rows held out, the Router's next question is the
    # seal's, which sits in Results and waits for the fit. Results opened on it, before Fit was
    # pressed, while the lock's own words say "Pressing Fit opens Results".
    state = ProjectState(lens=["dietary"], target="glucose", purpose="prediction")
    steps = [InterviewStep(key=k, status="waiting", waiting_on=["fit"]) if k == "open_seal"
             else InterviewStep(key=k, status="answered", decision_id=f"answer-{k}")
             for k in QUESTION_KEYS]
    for pressed in (False, True):
        fit = fit_press.fit_lock(state, [], pressed=pressed, held=False, estimate=None,
                                 opened=pressed)
        log = quest.quest_log(state, [], steps, {"fit": {"status": "idle"}}, fit=fit)
        reached = {s.key: s.reached for s in log.stages}
        assert (reached["models"], reached["results"], reached["writeup"]) == (
            True, pressed, pressed), pressed


def test_a_question_reread_while_its_answer_holds_keeps_the_later_stages_reached():
    # I11: dismissing a finding re-reads the outcome (the target_info stage reads the findings'
    # dispositions). While it recomputes, the recorded task holds, and the quest log says so on
    # its line; but the frontier stopped at the task, and First look and Who's in fell out of
    # reach with nothing saying why.
    state = ProjectState(lens=["dietary"], target="glucose", task="regression",
                         purpose="inference")
    artifacts = {"target_info": {"column": "glucose", "task": "regression", "confidence": "medium"}}
    rereading = {"target_info": {"status": "running"}}
    steps = route(state, rereading, artifacts)
    assert next(s for s in steps if s.key == "task").status == "waiting"
    log = quest.quest_log(state, [], steps, rereading, artifacts=artifacts)
    settled = quest.quest_log(state, [], route(state, {}, artifacts), {}, artifacts=artifacts)
    assert [s.reached for s in log.stages] == [s.reached for s in settled.stages]
    assert next(s for s in log.stages if s.key == "whos_in").reached
