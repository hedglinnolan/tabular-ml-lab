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


# The one engine conflict with §7.2's I11 as ruling 3 states it, reported and held by its own
# strict xfail (test_quest_progress_after_fit): an aggregation answer, given after the unit was
# changed, makes the regression-calibration declaration apply in a complete Models, which names
# no cause. Exactly that message is set aside here; any other I11 finding fails.
ENGINE_CONFLICTS = ("models was complete and gained 1 line",
                    "with no Reopened record (after answer set_aggregation)")


def _conflict(v: path_fuzzer.Violation) -> bool:
    return (v.invariant == "I11" and v.message.startswith(ENGINE_CONFLICTS[0])
            and v.message.endswith(ENGINE_CONFLICTS[1]))


def _held(report: path_fuzzer.Report) -> None:
    found = [v for v in report.violations if not _conflict(v)]
    assert not found, report.summary() + "\n" + "\n".join(str(v) for v in found[:25])


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


@pytest.mark.parametrize("fit", ["fresh", "error"])
def test_under_predict_the_seal_question_waits_for_fit_itself_not_for_a_fit_that_computed(fit):
    # I1, I4: a fit under the 2-minute hold computes before any press (the server's fit runs with
    # no hold), and a failed fit holds nothing back. The seal question opened on either, the one
    # open Decide sitting in a Results the person had not reached, with Fit not pressed.
    state, artifacts, _records = path_fuzzer.answered_through(path_fuzzer.NHANES, "prediction")
    stages = {"fit": {"status": fit}}
    for pressed in (None, False, True):
        steps = route(state, stages, artifacts, pressed=pressed)
        seal = next(s for s in steps if s.key == "open_seal")
        assert [s.key for s in steps if s.status in ("open", "waiting")] == ["open_seal"]
        assert (seal.status == "open") is bool(pressed), (pressed, seal)
        if not pressed:
            assert seal.waiting_on == ["fit"]
        lock = fit_press.fit_lock(state, [], pressed=bool(pressed), held=False, estimate=None,
                                  opened=bool(pressed))
        log = quest.quest_log(state, [], steps, stages, artifacts=artifacts, fit=lock)
        results = next(s for s in log.stages if s.key == "results")
        line = next(l for l in results.lines if l.key == "open_seal")
        assert results.reached is bool(pressed)
        assert (line.status == "open") is bool(pressed)


def test_under_estimate_the_seal_question_waits_for_the_plan_lock():
    state, artifacts, _records = path_fuzzer.answered_through(path_fuzzer.NHANES, "inference")
    state = state.model_copy(update={"split": state.split.model_copy(update={"holdout": 0.2})})
    stages = {"fit": {"status": "fresh"}}
    seal = next(s for s in route(state, stages, artifacts, pressed=True) if s.key == "open_seal")
    assert seal.status == "waiting" and seal.waiting_on == ["fit"]
    locked = state.model_copy(update={"plan_locked": True})
    assert next(s for s in route(locked, stages, artifacts) if s.key == "open_seal").status == "open"


def test_a_seal_that_opens_on_a_fit_computed_before_the_press_is_caught(monkeypatch):
    # The fuzzer's fit computes before the press as a short fit does live; with the Router's
    # wait for the press removed, the seal opens before Fit and I1 says so.
    from turbotab.core import interview

    monkeypatch.setattr(interview, "AFTER_FIT", frozenset())
    assert _found(fuzz(40), "I1")


# ── what the stronger invariants found (I1 numbers, I2 sweeps and defaults, I6, I11) ──────────


def _decide(records, payload):
    from turbotab.core import decisions

    records.append(path_fuzzer._record(records, decisions.parse_decision(payload)))
    return decisions.fold(records)


def _log_of(state, records, fx=path_fuzzer.NHANES, stages=None, fit=None, shown_at=None,
            confidence="high"):
    artifacts = path_fuzzer.artifacts_for(state, fx, records, confidence=confidence)
    stages = stages if stages is not None else {"fit": {"status": "idle"}}
    steps = route(state, stages, artifacts, records)
    return quest.quest_log(state, records, steps, stages, findings=artifacts.get("findings"),
                           columns=fx.columns, artifacts=artifacts, fit=fit, shown_at=shown_at)


def _line(log, key):
    return next((stage, line) for stage in log.stages for line in stage.lines if line.key == key)


def test_an_answer_the_person_withdrew_says_so_on_its_line_and_its_stage():
    records = []
    _decide(records, {"kind": "set_lens", "lenses": ["dietary"]})
    _decide(records, {"kind": "set_target", "column": "glucose"})
    state = _decide(records, {"kind": "set_purpose", "purpose": "prediction"})
    state = _decide(records, {"kind": "revert", "decision_id": records[-1].id})
    stage, line = _line(_log_of(state, records), "purpose")
    assert line.status == "open" and line.reopened_by is not None
    assert line.reopened_by.kind == "revert" and line.reopened_by.decision_id == records[-1].id
    assert any(line.id in r.questions for r in stage.reopened)


def test_a_withdrawn_confirm_all_sets_its_defaults_for_the_person_again_and_says_so():
    from turbotab.core.sweep import sweep_lines

    records = []
    _decide(records, {"kind": "set_lens", "lenses": ["dietary"]})
    state = _decide(records, {"kind": "set_target", "column": "glucose"})
    stage, _ = _line(_log_of(state, records), "design")
    lines = [l.model_dump() for l in sweep_lines(stage)]
    state = _decide(records, {"kind": "confirm_sweep", "stage": "question", "lines": lines})
    assert _line(_log_of(state, records), "design")[1].status == "answered"
    state = _decide(records, {"kind": "revert", "decision_id": records[-1].id})
    stage, line = _line(_log_of(state, records), "design")
    assert line.status == "set_for_you" and line.reopened_by.kind == "revert"
    assert stage.reopened and line.id in stage.reopened[0].questions


def test_a_withdrawn_dismissal_opens_the_finding_with_its_reason():
    records = []
    _decide(records, {"kind": "set_lens", "lenses": ["dietary"]})
    state = _decide(records, {"kind": "dismiss_finding", "finding_id": "unnamed_columns__x"})
    assert _line(_log_of(state, records), "unnamed_columns__x")[1].status == "answered"
    state = _decide(records, {"kind": "revert", "decision_id": records[-1].id})
    _stage, line = _line(_log_of(state, records), "unnamed_columns__x")
    assert line.status == "open" and line.reopened_by.kind == "revert"


def test_a_change_that_makes_a_declaration_apply_is_named_as_its_cause():
    records = []
    _decide(records, {"kind": "set_lens", "lenses": ["dietary"]})
    _decide(records, {"kind": "set_target", "column": "glucose"})
    state = _decide(records, {"kind": "set_purpose", "purpose": "inference"})
    assert not any(l.key == "set_intended_use" for s in _log_of(state, records).stages
                   for l in s.lines)
    state = _decide(records, {"kind": "set_purpose", "purpose": "prediction"})
    _stage, line = _line(_log_of(state, records), "set_intended_use")
    assert line.status == "open"
    assert (line.reopened_by.kind, line.reopened_by.decision_id) == ("set_purpose", records[-1].id)
    # a first answer is the journey going forward: it reopens nothing
    first = []
    _decide(first, {"kind": "set_lens", "lenses": ["dietary"]})
    _decide(first, {"kind": "set_target", "column": "glucose"})
    fresh = _decide(first, {"kind": "set_purpose", "purpose": "prediction"})
    assert _line(_log_of(fresh, first), "set_intended_use")[1].reopened_by is None


def test_models_chosen_for_another_outcome_say_its_kind_is_asked_again():
    # I6: the outcome changed, its kind is asked again, and the models chosen for the old one
    # stay answered (the Router never reopens a later answer); the line said nothing.
    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, "prediction")
    assert _line(_log_of(state, records), "models")[1].changed_since is None
    state = _decide(records, {"kind": "set_target", "column": "meds_hbp"})
    log = _log_of(state, records, confidence="medium")  # the new outcome's kind is asked
    assert _line(log, "task")[1].status in ("open", "waiting")
    line = _line(log, "models")[1]
    assert line.status == "answered" and line.changed_since is not None
    assert line.changed_since.decision_id == records[-1].id


def test_results_seen_under_predict_stay_reached_when_the_goal_becomes_estimate():
    # I11: Fit pressed under Predict, then the goal changed to Estimate; once the fit recomputed
    # for the new goal (fresh, withheld until the plan locks) Results emptied with nothing saying
    # why. It stays reached, its estimates withheld with the lock's reason.
    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, "prediction")
    state = _decide(records, {"kind": "set_purpose", "purpose": "inference"})
    fit = fit_press.fit_lock(state, records, pressed=True, held=False, estimate=None, opened=True)
    log = _log_of(state, records, stages={"fit": {"status": "fresh"}}, fit=fit)
    assert next(s for s in log.stages if s.key == "results").reached
    assert fit_press.serving_gate(state, True) is not None  # nothing estimated is shown


def test_confirm_all_fires_in_its_own_stage_only_where_a_default_is_stated():
    registry = surfacing.registry()
    assert "decision:confirm_sweep:first_look" not in registry
    state = ProjectState(lens=["dietary"], target="glucose")
    facts = quest.Facts(columns=path_fuzzer.NHANES.columns,
                        artifacts=path_fuzzer.artifacts_for(state, path_fuzzer.NHANES, []))
    fires = {k.rsplit(":", 1)[1]: i.fires(state, facts) for k, i in registry.items()
             if k.startswith("decision:confirm_sweep:")}
    # the design is stated observational in Your question; Results and Write-up state nothing yet
    assert fires["question"] and not fires["results"] and not fires["writeup"]
    designed = state.model_copy(update={"design": "observational"})
    assert not registry["decision:confirm_sweep:question"].fires(designed, facts)


# ── each new check bites ─────────────────────────────────────────────────────


def test_an_estimate_number_on_a_quest_line_is_caught(monkeypatch):
    real = quest.quest_log

    def leaks(*args, **kwargs):
        log = real(*args, **kwargs)
        if "fit" in (kwargs.get("artifacts") or {}):
            log.stages[0].lines[0].name += f" ({path_fuzzer.ESTIMATE_NUMBERS[0]})"
        return log

    monkeypatch.setattr(quest, "quest_log", leaks)
    assert _found(fuzz(20), "I1")


def test_an_open_question_in_an_unreached_stage_is_caught(monkeypatch):
    from turbotab.core import interview

    monkeypatch.setattr(interview, "AFTER_FIT", frozenset())
    assert any(v.invariant == "I4" and "not reached" in v.message for v in fuzz(40).violations)


def test_a_sweep_item_that_ignores_its_stage_is_caught(monkeypatch):
    monkeypatch.setattr(surfacing, "_sweep_fires", lambda stage: lambda state, facts: (
        state.target is not None or state.purpose is not None))
    surfacing.registry.cache_clear()
    try:
        assert _found(fuzz(10), "I2")
    finally:
        monkeypatch.undo()
        surfacing.registry.cache_clear()


def test_a_default_that_fires_where_the_log_does_not_show_it_is_caught(monkeypatch):
    monkeypatch.setitem(surfacing.DEFAULT_LINES, "default:design_observational",
                        ("question", "purpose"))
    assert any(v.invariant == "I2" and "default:design_observational" in v.message
               for v in fuzz(10).violations)


def test_an_answer_resting_silently_on_a_question_asked_again_is_caught(monkeypatch):
    monkeypatch.setattr(quest, "_rests_on_asked", lambda *args: None)
    assert _found(fuzz(70), "I6")


def test_a_reopening_that_names_no_cause_is_caught(monkeypatch):
    monkeypatch.setattr(quest, "_declaration_asked_again", lambda *args: None)
    monkeypatch.setattr(quest, "_asked_again", lambda *args: None)
    assert _found(fuzz(40), "I11")
