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
    # Every I11 finding fails: the one engine conflict ruling 3 named (an aggregation answer
    # reopening a complete Models without saying so) is fixed (P1-FU).
    found = list(report.violations)
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


# ── what the full tier found on the W1 part 2 landing (CI run 39169596231) ────────────────────

ACQUISITION = "pack::metabolomics::acquisition_design#1"


def _goal_left_predict(journey: path_fuzzer.Journey) -> list[int]:
    """The steps at which the goal changed from Predict while a stage before them was complete
    with an open noticing in it: the case the three seeds below met."""
    out = []
    for prev, snap in zip(journey.snapshots, journey.snapshots[1:]):
        if (snap.reopening and prev.state.purpose == "prediction"
                and snap.state.purpose != "prediction"
                and any(s.progress is not None and s.progress.complete and any(
                    l.source == "finding" and l.status == "open" for l in s.lines)
                    for s in prev.log.stages if s.key != "results")):
            out.append(snap.index)
    return out


@pytest.mark.parametrize("seed,step", [(475, 26), (1181, 15), (1221, 18)])
def test_the_goal_changed_from_predict_in_the_full_tiers_journeys_holds_the_invariants(seed, step):
    # I11 at each seed's step: the goal changed from Predict to Estimate under the metabolomics
    # lens, and Your data, complete at 3 of 3 with the acquisition-design noticing open, read 3 of
    # 4 with no Reopened record.
    journey = path_fuzzer.run_journey(seed)
    assert step in _goal_left_predict(journey), "the seed no longer meets the case it pins"
    found = path_fuzzer.check(journey)
    assert not found, "\n".join(str(v) for v in found)


def test_a_goal_changed_from_predict_says_why_its_noticings_count_in_their_stage_again():
    # I11, on the smallest state the three seeds showed: under Predict an open noticing counts
    # toward the triage in Results (ruling R-a), so Your data is complete with the
    # acquisition-design noticing open in it; under Estimate it counts in Your data again, and
    # Your data, complete before, said nothing of why it asks one more.
    fx = path_fuzzer.METABOLOMICS
    state, _artifacts, records = path_fuzzer.answered_through(fx, "prediction", stop="exclusions")
    data, line = _line(_log_of(state, records, fx=fx), ACQUISITION)
    assert data.key == "data" and line.status == "open" and line.reopened_by is None
    assert data.progress.complete and not data.reopened
    answered, required = data.progress.answered, data.progress.required
    state = _decide(records, {"kind": "set_purpose", "purpose": "inference"})
    data, line = _line(_log_of(state, records, fx=fx), ACQUISITION)
    assert (data.progress.answered, data.progress.required) == (answered, required + 1)
    assert (line.reopened_by.kind, line.reopened_by.decision_id, line.reopened_by.stage) == (
        "set_purpose", records[-1].id, "question")
    assert [(r.decision_id, r.questions) for r in data.reopened] == [(records[-1].id, [line.id])]
    assert data.reopened[0].sentence == "Your change to Your question reopened 1 question in Your data."


def test_a_withdrawn_predict_says_why_its_noticings_count_in_their_stage_again():
    # The same with the goal withdrawn rather than changed: with no goal the triage is Models'
    # again, and the revert is the cause.
    fx = path_fuzzer.METABOLOMICS
    state, _artifacts, records = path_fuzzer.answered_through(fx, "prediction", stop="exclusions")
    goal = next(r for r in records if r.decision.kind == "set_purpose")
    state = _decide(records, {"kind": "revert", "decision_id": goal.id})
    assert state.purpose is None
    data, line = _line(_log_of(state, records, fx=fx), ACQUISITION)
    assert (line.reopened_by.kind, line.reopened_by.decision_id) == ("revert", records[-1].id)
    assert any(line.id in r.questions for r in data.reopened)


def test_a_goal_that_was_never_predict_or_is_predict_now_reopens_no_noticing():
    fx = path_fuzzer.METABOLOMICS
    # a first answer of Estimate is the journey going forward: Your data counts the noticing from
    # the start, and nothing reopened it
    state, _artifacts, records = path_fuzzer.answered_through(fx, "inference", stop="exclusions")
    data, line = _line(_log_of(state, records, fx=fx), ACQUISITION)
    assert line.status == "open" and line.reopened_by is None and not data.reopened
    assert not data.progress.complete
    # changed to Predict: the noticing counts toward Results' triage, and Your data is complete
    state = _decide(records, {"kind": "set_purpose", "purpose": "prediction"})
    data, line = _line(_log_of(state, records, fx=fx), ACQUISITION)
    assert line.reopened_by is None and not data.reopened and data.progress.complete
    # Predict, then Estimate, then Predict again: the goal standing is Predict's
    state = _decide(records, {"kind": "set_purpose", "purpose": "inference"})
    assert _line(_log_of(state, records, fx=fx), ACQUISITION)[1].reopened_by is not None
    state = _decide(records, {"kind": "set_purpose", "purpose": "prediction"})
    data, line = _line(_log_of(state, records, fx=fx), ACQUISITION)
    assert line.reopened_by is None and not data.reopened and data.progress.complete


def test_a_noticing_disposed_of_before_the_goal_changed_from_predict_is_not_reopened():
    fx = path_fuzzer.METABOLOMICS
    state, _artifacts, records = path_fuzzer.answered_through(fx, "prediction", stop="exclusions")
    _decide(records, {"kind": "dismiss_finding", "finding_id": ACQUISITION})
    state = _decide(records, {"kind": "set_purpose", "purpose": "inference"})
    data, line = _line(_log_of(state, records, fx=fx), ACQUISITION)
    assert line.status == "answered" and line.reopened_by is None
    assert data.progress.complete and not data.reopened


# The goal's change from Predict names itself only on a noticing its stage held open then, in a
# stage reached and complete then (§7.2 I11's case): judged from the log as it stood just before
# the change, never from how the stage stands now.

IMPLAUSIBLE = "pack::dietary::implausible_intake#1"
ENERGY = "pack::dietary::energy_adjustment#1"
DRIFT = {"id": "pack::metabolomics::run_order_drift#1", "routes_to": "roles",
         "affected_columns": ["run_order"], "repairs": [],
         "summary": "Intensity drifts with run order"}


def _log_serving(state, records, fx, noticings):
    # the quest log as _log_of draws it, the findings stage serving ``noticings`` beside the
    # fixture's own (a noticing it computes once a slot it reads has changed)
    from turbotab.core.repairs import annotate

    artifacts = path_fuzzer.artifacts_for(state, fx, records)
    artifacts["findings"] = annotate(
        {"findings": [*(dict(f) for f in fx.findings), *(dict(n) for n in noticings)]},
        state, records)
    stages = {"fit": {"status": "idle"}}
    steps = route(state, stages, artifacts, records)
    return quest.quest_log(state, records, steps, stages, findings=artifacts["findings"],
                           columns=fx.columns, artifacts=artifacts)


def _forward(fx, records, stop):
    # answered_through's walk from ``records`` on: each Router question before ``stop`` answered
    # by its first option on the fixture
    from turbotab.core import decisions

    state = decisions.fold(records)
    for _ in range(3 * len(QUESTION_KEYS)):
        artifacts = path_fuzzer.artifacts_for(state, fx, records)
        steps = route(state, {"fit": {"status": "fresh"}}, artifacts, records, pressed=True)
        first = next((s for s in steps if s.status == "open"), None)
        if first is None or first.key == stop:
            return state
        options = path_fuzzer.question_options(first.key, state, fx)
        assert options, first.key
        records.append(path_fuzzer._record(records, path_fuzzer._validate(options[0], state, fx)))
        state = decisions.fold(records)
    return state


def _named(stage, line, record):
    # the line names ``record`` as why it is open, and its stage's Reopened record lists it
    return (line.reopened_by is not None and line.reopened_by.decision_id == record.id
            and any(r.decision_id == record.id and line.id in r.questions
                    for r in stage.reopened))


def test_a_noticing_served_after_the_goal_left_predict_names_no_goal_change():
    # P1: with the acquisition-design noticing dismissed, the goal's change leaves Your data
    # complete at 4 of 4; a reading then confirmed in Your data changes a slot the findings stage
    # reads, and the run-order noticing it serves for that was not open when the goal changed.
    # Your data counts it within the stage being worked (I11), and no Reopened record is owed.
    fx = path_fuzzer.METABOLOMICS
    state, _artifacts, records = path_fuzzer.answered_through(fx, "prediction", stop="exclusions")
    _decide(records, {"kind": "dismiss_finding", "finding_id": ACQUISITION})
    state = _decide(records, {"kind": "set_purpose", "purpose": "inference"})
    data, _open = _line(_log_of(state, records, fx=fx), ACQUISITION)
    assert data.progress.complete and not data.reopened
    answered, required = data.progress.answered, data.progress.required
    state = _decide(records, {"kind": "confirm_reading", "reading": "code_or_count",
                              "column": "age", "value": "amount"})
    assert quest.written_slots(records[-1].decision) & quest.stage_reads("findings")
    data, line = _line(_log_serving(state, records, fx, [DRIFT]), DRIFT["id"])
    assert data.key == "data" and line.status == "open" and line.counted
    assert (data.progress.answered, data.progress.required) == (answered, required + 1)
    assert line.reopened_by is None and not data.reopened


def test_a_noticing_open_when_the_goal_left_predict_keeps_that_cause_past_another_answer():
    # P1's other side: an answer after the change that writes no slot the findings stage reads
    # (the exclusions, in Who's in) leaves the noticing the one that was open then, and Your data,
    # complete before the change, still says why it asks one more.
    fx = path_fuzzer.METABOLOMICS
    state, _artifacts, records = path_fuzzer.answered_through(fx, "prediction", stop="exclusions")
    state = _decide(records, {"kind": "set_purpose", "purpose": "inference"})
    goal = records[-1]
    state = _decide(records, {"kind": "set_exclusions", "rules": []})
    assert not quest.written_slots(records[-1].decision) & quest.stage_reads("findings")
    data, line = _line(_log_of(state, records, fx=fx), ACQUISITION)
    assert data.key == "data" and line.status == "open" and _named(data, line, goal)


@pytest.mark.parametrize("fx,stop,key,home,waiting", [
    # Your data still asked the roles: 2 of 3 under Predict, 2 of 4 after
    (path_fuzzer.METABOLOMICS, "roles", ACQUISITION, "data", "roles"),
    # Who's in still asked the held-out rows, a line Predict alone asks: 3 of 4 before and after
    (path_fuzzer.NHANES, "split", IMPLAUSIBLE, "whos_in", "split"),
])
def test_a_stage_still_asking_when_the_goal_left_predict_is_not_reopened_by_it(fx, stop, key, home,
                                                                                waiting):
    # P2: I11 asks a Reopened record only of a stage complete before; one still asking when the
    # goal changed counts the noticing as the journey going forward.
    state, _artifacts, records = path_fuzzer.answered_through(fx, "prediction", stop=stop)
    stage, line = _line(_log_of(state, records, fx=fx), key)
    assert stage.key == home and line.status == "open"
    assert not stage.progress.complete
    assert next(l for l in stage.lines if l.key == waiting).status == "open"
    state = _decide(records, {"kind": "set_purpose", "purpose": "inference"})
    stage, line = _line(_log_of(state, records, fx=fx), key)
    assert stage.key == home and line.status == "open" and line.counted
    assert line.reopened_by is None and not stage.reopened


def test_a_stage_not_reached_when_the_goal_left_predict_is_not_reopened_by_it():
    # P2: the goal changed to Estimate right after it was first answered, with Who's in reached
    # at its first question (the grain, open) and Models not reached; the journey then goes
    # forward through both, and neither was complete when the goal changed.
    fx = path_fuzzer.NHANES
    state, _artifacts, records = path_fuzzer.answered_through(fx, "prediction", stop="grain")
    state = _decide(records, {"kind": "set_purpose", "purpose": "inference"})
    log = _log_of(state, records, fx=fx)
    stages = {s.key: s for s in log.stages}
    assert stages["models"].progress is None and not stages["models"].reached
    assert not stages["whos_in"].progress.complete
    assert next(l for l in stages["whos_in"].lines if l.key == "grain").status == "open"
    state = _forward(fx, records, "energy_adjustment")
    log = _log_of(state, records, fx=fx)
    for key, home in ((IMPLAUSIBLE, "whos_in"), (ENERGY, "models")):
        stage, line = _line(log, key)
        assert stage.key == home and stage.reached and line.status == "open" and line.counted
        assert line.reopened_by is None, key
        assert not any(line.id in r.questions for r in stage.reopened), key


def test_a_drawn_split_that_leaves_with_predict_leaves_the_goal_change_the_cause():
    # P2's other side: Who's in complete under Predict (the held-out rows drawn), the split leaves
    # its count with the goal's change and the noticing counts there: the change is why.
    fx = path_fuzzer.NHANES
    state, _artifacts, records = path_fuzzer.answered_through(fx, "prediction", stop="split")
    state = _decide(records, {"kind": "set_split", "holdout": 0.2, "seed": 0})
    whos_in, line = _line(_log_of(state, records, fx=fx), IMPLAUSIBLE)
    assert whos_in.progress.complete and line.status == "open"
    state = _decide(records, {"kind": "set_purpose", "purpose": "inference"})
    whos_in, line = _line(_log_of(state, records, fx=fx), IMPLAUSIBLE)
    assert not whos_in.progress.complete and _named(whos_in, line, records[-1])
    assert [r.sentence for r in whos_in.reopened] == [
        "Your change to Your question reopened 1 question in Who's in."]


# The name stays while the cause holds: the noticing open still, the goal the one the change set,
# and the findings the ones the change met (repair round 2's M1 and M2).

READING = {"kind": "confirm_reading", "reading": "code_or_count", "column": "age",
           "value": "amount"}


def _findings_inputs(state):
    # what the findings stage reads, as the answers hold it
    return {slot: getattr(state, slot, None) for slot in quest.stage_reads("findings")}


def _withdraw_the_last(records):
    return {"kind": "revert", "decision_id": records[-1].id}


@pytest.mark.parametrize("answers", [
    pytest.param(lambda s: [{"kind": "set_target", "column": s.target}], id="the outcome again"),
    pytest.param(lambda s: [{"kind": "set_orientation", "orientation": s.orientation}],
                 id="the orientation again"),
    pytest.param(lambda s: [{"kind": "set_lens", "lenses": list(s.lens)}], id="the lens again"),
    pytest.param(lambda s: [READING, _withdraw_the_last], id="a reading confirmed and withdrawn"),
])
def test_a_noticing_open_when_the_goal_left_predict_keeps_that_cause_while_the_findings_stand(
        answers):
    # M1: an answer since the change that writes what the findings stage reads, but leaves it as
    # it was (the same value again, or a change withdrawn), leaves the findings the ones the change
    # met. The acquisition-design noticing open then is open still, and Your data still says why.
    fx = path_fuzzer.METABOLOMICS
    state, _artifacts, records = path_fuzzer.answered_through(fx, "prediction", stop="exclusions")
    state = _decide(records, {"kind": "set_purpose", "purpose": "inference"})
    goal, then, first = records[-1], _findings_inputs(state), len(records)
    for payload in answers(state):
        state = _decide(records, payload(records) if callable(payload) else payload)
    assert quest.written_slots(records[first].decision) & quest.stage_reads("findings")
    assert _findings_inputs(state) == then and state.purpose == "inference"
    data, line = _line(_log_of(state, records, fx=fx), ACQUISITION)
    assert data.key == "data" and line.status == "open" and line.counted
    assert _named(data, line, goal)
    assert [r.sentence for r in data.reopened] == [
        "Your change to Your question reopened 1 question in Your data."]


def test_a_noticing_open_when_the_goal_left_predict_names_no_cause_once_the_findings_are_new():
    # M1's other side, with P1's: a reading confirmed since the change is read by the findings
    # stage, which computes the findings again, and the quest log holds only those. One may be new
    # (P1's run-order noticing), and nothing it is given says which were open when the goal
    # changed, so none names the change; Your data still counts the one open (it was not complete).
    fx = path_fuzzer.METABOLOMICS
    state, _artifacts, records = path_fuzzer.answered_through(fx, "prediction", stop="exclusions")
    state = _decide(records, {"kind": "set_purpose", "purpose": "inference"})
    then = _findings_inputs(state)
    state = _decide(records, READING)
    assert _findings_inputs(state) != then
    data, line = _line(_log_of(state, records, fx=fx), ACQUISITION)
    assert line.status == "open" and line.counted and not data.progress.complete
    assert line.reopened_by is None and not data.reopened


def _models_complete_under_predict():
    # NHANES under Predict with every Router question, the selection and Models' "Confirm all"
    # answered: Models complete, as the path fuzzer's seed 98 had it at step 38
    from turbotab.core.sweep import sweep_lines

    fx = path_fuzzer.NHANES
    state, _artifacts, records = path_fuzzer.answered_through(fx, "prediction", stop="open_seal")
    state = _decide(records, path_fuzzer.declaration_options("set_selection", state, fx)[0])
    models = next(s for s in _log_of(state, records, fx=fx).stages if s.key == "models")
    state = _decide(records, {"kind": "confirm_sweep", "stage": "models",
                              "lines": [l.model_dump() for l in sweep_lines(models)]})
    return state, records


def _asking(stage):
    return {l.id for l in stage.lines if l.status in ("open", "waiting")}


def test_a_noticing_the_goal_change_made_apply_keeps_that_cause_past_an_answer_it_opened():
    # M2 (the path fuzzer's seed 98, steps 38 to 40, built by hand): Models complete under
    # Predict, where the energy noticing does not fire (ruling 5). The change to Estimate makes it
    # apply, with the estimand and the questions that follow, and names each. Answering the
    # estimand, a line the change made apply, leaves Models complete just before the change, so
    # the noticing, open still, keeps the change as its cause, and the Reopened record lists each
    # line the change opened that still asks: one answer takes one line off it, not two.
    fx = path_fuzzer.NHANES
    state, records = _models_complete_under_predict()
    models = next(s for s in _log_of(state, records, fx=fx).stages if s.key == "models")
    assert models.progress.complete and not _asking(models)
    assert ENERGY not in {l.key for l in models.lines}
    state = _decide(records, {"kind": "set_purpose", "purpose": "inference"})
    goal = records[-1]
    models, line = _line(_log_of(state, records, fx=fx), ENERGY)
    assert models.key == "models" and _named(models, line, goal)
    opened = set(models.reopened[0].questions)
    assert opened == _asking(models) and len(models.reopened) == 1
    state = _decide(records, path_fuzzer.question_options("estimand", state, fx)[0])
    models, line = _line(_log_of(state, records, fx=fx), ENERGY)
    assert next(l for l in models.lines if l.key == "estimand").status == "answered"
    assert line.status == "open" and line.counted and _named(models, line, goal)
    still = opened & _asking(models)
    assert [(r.decision_id, set(r.questions)) for r in models.reopened] == [(goal.id, still)]
    assert models.reopened[0].sentence == (
        f"Your change to Your question reopened {len(still)} questions in Models.")


def test_the_goal_change_in_the_fuzzers_seed_98_keeps_naming_what_it_opened_past_the_estimand():
    # M2 as the path fuzzer met it: at step 40 the estimand is answered, and every line the goal's
    # change at step 39 named that still asks is named and listed still, the energy noticing too.
    journey = path_fuzzer.run_journey(98)
    s38, s39, s40 = journey.snapshots[38:41]
    assert (s39.kind, s40.kind) == ("set_purpose", "set_estimand"), "the seed no longer meets it"
    assert s38.state.purpose == "prediction" and s39.state.purpose == "inference"
    old = next(s for s in s38.log.stages if s.key == "models")
    assert old.progress.complete and ENERGY not in {l.key for l in old.lines}
    at39 = next(s for s in s39.log.stages if s.key == "models")
    at40 = next(s for s in s40.log.stages if s.key == "models")
    goal = s39.records[-1]
    named = {r.decision_id: set(r.questions) for r in at39.reopened}[goal.id]
    assert f"finding:{ENERGY}" in named
    still = named & _asking(at40)
    assert f"finding:{ENERGY}" in still
    assert {r.decision_id: set(r.questions) for r in at40.reopened}.get(goal.id) == still


def test_a_line_asked_when_the_goal_left_predict_and_answered_since_leaves_its_stage_open_then():
    # M2's other side: Your data still asked the roles under Predict, so it was not complete when
    # the goal changed; the roles answered after the change do not make it so, and the
    # acquisition-design noticing it counts names no goal change.
    fx = path_fuzzer.METABOLOMICS
    state, _artifacts, records = path_fuzzer.answered_through(fx, "prediction", stop="roles")
    data, _open = _line(_log_of(state, records, fx=fx), ACQUISITION)
    assert next(l for l in data.lines if l.key == "roles").status == "open"
    state = _decide(records, {"kind": "set_purpose", "purpose": "inference"})
    state = _decide(records, path_fuzzer.question_options("roles", state, fx)[0])
    data, line = _line(_log_of(state, records, fx=fx), ACQUISITION)
    assert next(l for l in data.lines if l.key == "roles").status == "answered"
    assert data.key == "data" and line.status == "open" and line.counted
    assert line.reopened_by is None and not data.reopened


def _noticings(stage):
    return {l.key: l for l in stage.lines if l.source == "finding"}


def test_a_noticing_open_when_the_goal_left_predict_keeps_that_cause_past_another_ones_disposal():
    # LF-1's last verifier: Your data complete at 3 of 3 under Predict with the acquisition-design
    # and run-order noticings open in it (Predict counts both toward Results' triage, ruling R-a);
    # the change to Estimate makes both count there, 3 of 5, and names itself on each. Dismissing
    # the run-order noticing answers it now, but it did not count under Predict, so its answer
    # says nothing of how Your data stood then: the acquisition-design noticing, open still, the
    # goal standing and the findings the ones the change met, keeps the change as its cause.
    fx = path_fuzzer.METABOLOMICS
    state, _artifacts, records = path_fuzzer.answered_through(fx, "prediction", stop="exclusions")
    data, _line_ = _line(_log_serving(state, records, fx, [DRIFT]), ACQUISITION)
    assert data.key == "data" and data.progress.complete
    assert (data.progress.answered, data.progress.required) == (3, 3)
    assert {k for k, l in _noticings(data).items() if l.status == "open"} == {ACQUISITION,
                                                                               DRIFT["id"]}
    state = _decide(records, {"kind": "set_purpose", "purpose": "inference"})
    goal = records[-1]
    data, line = _line(_log_serving(state, records, fx, [DRIFT]), ACQUISITION)
    assert (data.progress.answered, data.progress.required) == (3, 5)
    assert _named(data, line, goal) and _named(data, _noticings(data)[DRIFT["id"]], goal)
    assert [r.sentence for r in data.reopened] == [
        "Your change to Your question reopened 2 questions in Your data."]
    then = _findings_inputs(state)
    state = _decide(records, {"kind": "dismiss_finding", "finding_id": DRIFT["id"]})
    assert _findings_inputs(state) == then and state.purpose == "inference"
    data, line = _line(_log_serving(state, records, fx, [DRIFT]), ACQUISITION)
    drift = _noticings(data)[DRIFT["id"]]
    assert drift.status == "answered" and drift.reopened_by is None
    assert (data.progress.answered, data.progress.required) == (4, 5)
    assert line.status == "open" and line.counted and _named(data, line, goal)
    assert [(r.decision_id, r.questions) for r in data.reopened] == [(goal.id, [line.id])]
    assert data.reopened[0].sentence == (
        "Your change to Your question reopened 1 question in Your data.")


def test_a_stage_still_asking_when_the_goal_left_predict_stays_unnamed_past_a_noticings_disposal():
    # Its other side: Your data still asked the roles under Predict, so it was not complete when
    # the goal changed. Dismissing the run-order noticing after the change leaves the roles open
    # then, and the acquisition-design noticing names no goal change.
    fx = path_fuzzer.METABOLOMICS
    state, _artifacts, records = path_fuzzer.answered_through(fx, "prediction", stop="roles")
    data, _line_ = _line(_log_serving(state, records, fx, [DRIFT]), ACQUISITION)
    assert next(l for l in data.lines if l.key == "roles").status == "open"
    assert not data.progress.complete
    state = _decide(records, {"kind": "set_purpose", "purpose": "inference"})
    state = _decide(records, {"kind": "dismiss_finding", "finding_id": DRIFT["id"]})
    data, line = _line(_log_serving(state, records, fx, [DRIFT]), ACQUISITION)
    assert _noticings(data)[DRIFT["id"]].status == "answered"
    assert data.key == "data" and line.status == "open" and line.counted
    assert line.reopened_by is None and not data.reopened


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


def test_a_goal_change_from_predict_that_names_no_cause_is_caught(monkeypatch):
    monkeypatch.setattr(quest, "_goal_moved_the_triage", lambda *args: None)
    found = path_fuzzer.check(path_fuzzer.run_journey(1181))
    assert any(v.invariant == "I11" and v.index == 15 and "data was complete" in v.message
               for v in found), found
