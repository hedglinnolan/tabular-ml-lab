"""Q-c · quest progress after Fit, and SURFACING_POLICY §7.2's I11 as the orchestrator ruled it
(WAVE_C6A_PLAN §7, ruling 3).

Under Estimate and Describe, once Fit is pressed, Results reads 0 of N (each exhibit's wording and
placement is one of its objectives, calm/FOUNDATION §3 and §8), not "0 of 0, complete"; Write-up is
not reached until Results is placed. The references are written by hand here: the design's state
(``design/quest-models-stage``, ``screens.ts``: after Fit, ``results = {answered: 0, required:
EXHIBITS.length, complete: false}`` and ``writeup = null``), and the exhibits the NHANES fixture is
served, listed one by one from each stage's ``requires`` (``stages/__init__.py``) against the
fixture's answers.

B1 (WO1; the vision audit's V2; the owner's ruling (3) of 2026-11-04): until C7a adds placement,
each exhibit arrives placed in Results by default (FOUNDATION §8: "Each exhibit arrives with a
default place") once it is shown for the answers now: the plan locked, no question the estimates
rest on open, its stage fresh. Results then counts it answered, and Write-up is reached once every
one is placed, so the shell offers the download and the checklist. Before the lock, while an
exhibit recomputes or failed, or while the estimates are withheld, it is not placed. The counts
expected here are the hand-listed exhibits below, each fresh or not by construction.
"""
from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from turbotab.core import fit_press, quest
from turbotab.core.quest import Progress, QuestLine, QuestLog, QuestStage, Reopened
from turbotab.core.tests import path_fuzzer
from turbotab.core.tests.test_path_fuzzer import _decide, _log_of

# Every compute stage whose card is an exhibit (CROSSWALK, "Engine stages and quest stages"),
# listed by hand, and the substitution curve (TRUST: the second comparison is an exhibit of
# Results, with its question's goals while the crosswalk gives it no card of its own).
EXHIBIT_MAP = ["usual_intake", "fit", "substitution", "sensitivity", "calibration", "secondary",
               "scales", "effects", "causal", "time_varying", "modification", "explain",
               "evaluation"]

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
# TRUST: only the exhibits Estimate serves count (the crosswalk's goals): usual_intake is a
# Describe exhibit, fit (the performance table) and evaluation (the decision curve) Predict ones.
NHANES_SERVED = ["usual_intake", "fit", "secondary", "effects", "evaluation"]
NHANES_ESTIMATE = ["secondary", "effects"]
NOT_ANSWERED = ["sensitivity", "measurement_error", "scales", "causal", "unit", "modifications",
                "explain"]


def _after_fit(purpose: str, extra: list[dict] | None = None,
               stages: dict | None = None) -> dict[str, QuestStage]:
    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, purpose)
    for payload in extra or []:
        state = _decide(records, payload)
    fit = fit_press.fit_lock(state, records, pressed=True, held=False, estimate=None, opened=True)
    log = _log_of(state, records, stages=stages or {"fit": {"status": "fresh"}}, fit=fit)
    return {s.key: s for s in log.stages}


def _fresh(*names: str, **others: str) -> dict[str, dict[str, str]]:
    """Each named compute stage fresh (computed for the answers now), the others at their status."""
    return {**{n: {"status": "fresh"} for n in names},
            **{n: {"status": status} for n, status in others.items()}}


# The fit and both Estimate exhibits the NHANES fixture is served, computed for the answers now.
ALL_FRESH = _fresh("fit", *NHANES_ESTIMATE)


def _locked(stages: dict, extra: list[dict] | None = None) -> dict[str, QuestStage]:
    """The NHANES fixture answered under Estimate with the plan locked as pressing Fit records it
    (``lock_plan``), then ``extra`` recorded, each compute stage at its status in ``stages``."""
    return _after_fit("inference", [{"kind": "lock_plan", "seen_target": "glucose"},
                                    *(extra or [])], stages)


def test_the_exhibit_map_and_the_version():
    assert sorted(quest.EXHIBIT_STAGES) == sorted(EXHIBIT_MAP)
    assert quest.QUEST_VERSION == 6


def test_the_fixture_answers_none_of_the_exhibits_it_is_not_served():
    state, _artifacts, _records = path_fuzzer.answered_through(path_fuzzer.NHANES, "inference")
    assert [slot for slot in NOT_ANSWERED if getattr(state, slot) is not None] == []
    assert state.models == ["linear"] and state.lens == ["dietary"]
    assert state.adjustment and state.estimand is not None
    assert sorted(quest.served_exhibits(state)) == sorted(NHANES_ESTIMATE)
    from turbotab.core.surfacing import computable

    assert sorted(n for n in quest.EXHIBIT_STAGES if n in computable(state)) == sorted(
        NHANES_SERVED)


def test_results_after_fit_under_estimate_reads_0_of_n_served_and_is_not_complete():
    results = _after_fit("inference")["results"]
    assert results.reached
    # nothing in Results is a counted Decide yet: the whole count is the served exhibits
    assert [l for l in results.lines if l.label == "Decide" and l.counted] == []
    assert results.progress == Progress(answered=0, required=2, complete=False)


def test_an_explain_answer_serves_one_more_exhibit():
    # explain requires explain, models and purpose: answered, it is served (an Estimate exhibit
    # too), 0 of 3
    results = _after_fit("inference", [{"kind": "set_explain", "reseeds": 0}])["results"]
    assert results.progress == Progress(answered=0, required=3, complete=False)


def test_write_up_is_not_reached_until_results_is_placed():
    writeup = _after_fit("inference")["writeup"]
    assert writeup.progress is None and not writeup.reached


# ── B1: each exhibit placed in Results by default, once shown (WO1) ─────────


def test_locked_with_every_exhibit_fresh_results_is_placed_and_write_up_is_reached():
    # Both Estimate exhibits (secondary, effects) are computed for the answers now after the
    # lock: each is placed in Results by default, so Results reads 2 of 2, complete, and Write-up
    # is reached, asking nothing: 0 of 0, complete, as under Predict.
    stages = _locked(ALL_FRESH)
    assert stages["results"].reached
    assert stages["results"].progress == Progress(answered=2, required=2, complete=True)
    assert stages["writeup"].reached
    assert stages["writeup"].progress == Progress(answered=0, required=0, complete=True)


def test_before_the_lock_a_fresh_exhibit_is_not_placed():
    # Results reached by an earlier press (``fit.opened``) and every exhibit fresh, but the plan
    # is not locked: none is shown, so none is placed, and Write-up waits.
    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, "inference")
    assert not state.plan_locked
    stages = _after_fit("inference", stages=ALL_FRESH)
    assert stages["results"].reached
    assert stages["results"].progress == Progress(answered=0, required=2, complete=False)
    assert not stages["writeup"].reached and stages["writeup"].progress is None


@pytest.mark.parametrize("status", ["queued", "running", "stale", "idle", "blocked", "error"])
def test_an_exhibit_not_computed_for_the_answers_now_is_not_placed(status):
    # After the lock, Table 2 recomputes (or waits, or failed): the further-adjusted model, fresh,
    # is placed; Table 2 is not. Results reads 1 of 2, and Write-up waits for it.
    stages = _locked(_fresh("fit", "secondary", effects=status))
    assert stages["results"].progress == Progress(answered=1, required=2, complete=False)
    assert not stages["writeup"].reached and stages["writeup"].progress is None
    # and neither computed for the answers now: 0 of 2
    stages = _locked(_fresh("fit", secondary=status, effects=status))
    assert stages["results"].progress == Progress(answered=0, required=2, complete=False)
    assert not stages["writeup"].reached


def test_while_the_estimates_are_withheld_no_exhibit_is_placed():
    # After the lock a new covariate (waist) asks the adjustment set again, and no estimate is
    # shown until it is answered (``estimand.served_gate``, WP17), though each exhibit is fresh:
    # none is placed, and Write-up waits.
    from turbotab.core import estimand
    from turbotab.core.interview import route

    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, "inference")
    covariate = {"kind": "set_roles", "roles": {**state.roles, "waist": "covariate"}}
    state = _decide(records, {"kind": "lock_plan", "seen_target": state.target})
    state = _decide(records, covariate)
    steps = route(state, ALL_FRESH, path_fuzzer.artifacts_for(state, path_fuzzer.NHANES, records),
                  records)
    assert estimand.served_gate(state, steps)["question"] == "adjustment"
    stages = _locked(ALL_FRESH, [covariate])
    assert stages["results"].progress == Progress(answered=0, required=2, complete=False)
    assert not stages["writeup"].reached
    # the same lock with no new covariate: the estimates are shown, and both are placed
    assert _locked(ALL_FRESH)["results"].progress == Progress(answered=2, required=2,
                                                               complete=True)


def _describe(usual_intake: str, lens: bool | str = True) -> dict[str, QuestStage]:
    """Describe (TRUST: inference with no effect declared, ``goal_of``): the dietary lens (unless
    not ``lens``; ``"after"``: after the lock), the outcome and the goal answered, the plan locked,
    every other question of Estimate answered, and the usual-intake stage at ``usual_intake``."""
    return _describe_log(usual_intake, lens)[0]


def _describe_log(usual_intake: str, lens: bool | str = True
                  ) -> tuple[dict[str, QuestStage], list]:
    from turbotab.core.interview import QUESTION_KEYS, InterviewStep

    records: list = []
    dietary = {"kind": "set_lens", "lenses": ["dietary"]}
    for payload in (dietary if lens is True else None,
                    {"kind": "set_target", "column": "glucose"},
                    {"kind": "set_purpose", "purpose": "inference"},
                    {"kind": "lock_plan", "seen_target": "glucose"},
                    dietary if lens == "after" else None):
        if payload is not None:
            state = _decide(records, payload)
    assert quest.served_exhibits(state) == (("usual_intake",) if lens else ())
    steps = [InterviewStep(key=k, status="answered", decision_id=f"answer-{k}")
             for k in QUESTION_KEYS if k != "open_seal"]  # the seal is Predict's
    fit = fit_press.fit_lock(state, records, pressed=True, held=False, estimate=None, opened=True)
    log = quest.quest_log(state, records, steps, _fresh(usual_intake=usual_intake), fit=fit)
    return {s.key: s for s in log.stages}, records


def test_under_describe_usual_intake_is_placed_once_fresh_after_the_lock():
    # usual_intake requires the lens and the goal: its one exhibit (no models answered, so no
    # other is computed). Fresh after the lock: 1 of 1, and Write-up is reached; recomputing,
    # 0 of 1, and it waits.
    state = _decide([], {"kind": "set_purpose", "purpose": "inference"})
    assert quest.goal_of(state) == "describe"
    stages = _describe("fresh")
    assert stages["results"].progress == Progress(answered=1, required=1, complete=True)
    assert stages["writeup"].reached
    stages = _describe("running")
    assert stages["results"].progress == Progress(answered=0, required=1, complete=False)
    assert not stages["writeup"].reached


def test_with_no_exhibit_served_nothing_is_placed_and_write_up_waits():
    # No lens: usual_intake (lens, purpose) is not computed, and with no models nothing else is.
    # Results counts no exhibit (0 of 0), nothing is placed, and Write-up waits, as before B1.
    stages = _describe("fresh", lens=False)
    assert stages["results"].reached and stages["results"].progress.required == 0
    assert not stages["writeup"].reached and stages["writeup"].progress is None


# ── B1 round 1: an exhibit served after the lock says why (V1; §7.2 I11) ────
# Results, complete once both exhibits are placed, has one more objective when an answer given
# after the lock serves another exhibit: it reads 2 of 3 and Write-up waits for it, so Results
# names the answer and the exhibit it served, until that exhibit is computed and placed. Each
# expected record is written by hand: the answer is the last record built here, its stage the
# one its kind is decided in (``kind_stages``: set_sensitivity, set_substitution and set_scales in
# Models, set_explain in Results, set_lens in Your data), and its sentence the reason's template
# with the counts constructed (one exhibit served; two left out of date).

SENSITIVITY = {"kind": "set_sensitivity", "analyses": [{"label": "Every row", "rules": []}]}
SUBSTITUTION = {"kind": "set_substitution", "donor": "carb", "recipient": "fat_total"}
EXPLAIN = {"kind": "set_explain", "reseeds": 0}
LOCK = {"kind": "lock_plan", "seen_target": "glucose"}


def _journey(before: list[dict], after: list[dict], stages: dict,
             shown: tuple[str, ...] = ()) -> tuple[Any, list, QuestLog]:
    """The NHANES fixture answered under Estimate, ``before`` recorded, the plan locked, then
    ``after`` recorded; each compute stage at its status in ``stages``, and each of ``shown``
    computed just after the lock (``shown_at``: shown for earlier answers). Its state, records
    and quest log."""
    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, "inference")
    for payload in [*before, LOCK]:
        state = _decide(records, payload)
    at = records[-1].at + timedelta(seconds=30)
    for payload in after:
        state = _decide(records, payload)
    fit = fit_press.fit_lock(state, records, pressed=True, held=False, estimate=None, opened=True)
    log = _log_of(state, records, stages=stages, fit=fit, shown_at={n: at for n in shown})
    return state, records, log


def _by_key(log: QuestLog) -> dict[str, QuestStage]:
    return {s.key: s for s in log.stages}


@pytest.mark.parametrize("status", ["queued", "running", "error"])
@pytest.mark.parametrize("answer,exhibit", [(SENSITIVITY, "sensitivity"),
                                            (SUBSTITUTION, "substitution")])
def test_an_exhibit_a_models_answer_serves_after_the_lock_says_why_results_dropped_back(
        answer, exhibit, status):
    # sensitivity requires its answer, the substitution curve its pair (the substitution question
    # opens once the fit is fresh: with a held fit, after the press that locks the plan)
    _state, records, log = _journey([], [answer], {**ALL_FRESH, exhibit: {"status": status}})
    stages = _by_key(log)
    assert stages["results"].progress == Progress(answered=2, required=3, complete=False)
    assert not stages["writeup"].reached
    assert stages["results"].reopened == [Reopened(
        changed_in="models", decision_id=records[-1].id, kind=answer["kind"], added=[exhibit],
        sentence="Your change to Models added 1 result to Results.")]


def test_an_exhibit_a_results_answer_serves_after_the_lock_says_so_within_results():
    # the explain declaration is decided in Results itself: the reason is within it
    _state, records, log = _journey([], [EXPLAIN], {**ALL_FRESH, "explain": {"status": "queued"}})
    stages = _by_key(log)
    assert stages["results"].progress == Progress(answered=2, required=3, complete=False)
    assert not stages["writeup"].reached
    assert stages["results"].reopened == [Reopened(
        changed_in="results", decision_id=records[-1].id, kind="set_explain", added=["explain"],
        within=True, sentence="Your change to another answer in Results added 1 result.")]


@pytest.mark.parametrize("answer,exhibit", [(SENSITIVITY, "sensitivity"),
                                            (SUBSTITUTION, "substitution"), (EXPLAIN, "explain")])
def test_an_exhibit_served_after_the_lock_and_placed_since_names_nothing(answer, exhibit):
    # computed for the answers now, it is placed: 3 of 3, Write-up reached again, nothing to say
    _state, _records, log = _journey([], [answer], {**ALL_FRESH, exhibit: {"status": "fresh"}})
    stages = _by_key(log)
    assert stages["results"].progress == Progress(answered=3, required=3, complete=True)
    assert stages["writeup"].reached
    assert stages["results"].reopened == []


def test_the_lock_and_an_exhibit_served_before_it_name_nothing():
    # the plain lock, every exhibit placed
    assert _locked(ALL_FRESH)["results"].reopened == []
    # an exhibit served at the lock and not computed yet was never one more: Results was not
    # complete since the lock, and nothing reopened it
    stages = _locked(_fresh("fit", "secondary", effects="queued"))
    assert stages["results"].progress == Progress(answered=1, required=2, complete=False)
    assert stages["results"].reopened == []
    _state, _records, log = _journey([SENSITIVITY], [],
                                     {**ALL_FRESH, "sensitivity": {"status": "queued"}})
    stages = _by_key(log)
    assert stages["results"].progress == Progress(answered=2, required=3, complete=False)
    assert stages["results"].reopened == []
    # before any lock nothing is placed, so nothing is one more
    stages = _after_fit("inference", [SENSITIVITY],
                        {**ALL_FRESH, "sensitivity": {"status": "queued"}})
    assert stages["results"].progress == Progress(answered=0, required=3, complete=False)
    assert stages["results"].reopened == []


def test_one_answer_that_leaves_two_results_out_of_date_and_serves_a_third_says_both():
    # A scale declared after the lock: Table 2 and Model 3 read the scales (computed just after
    # the lock, out of date now), and the scales exhibit requires them (served now). One record
    # says both, in one sentence.
    scale = {"kind": "set_scales", "scales": [{"name": "fat_score", "low": 0, "high": 100,
                                              "items": ["fat_sat", "fat_mon", "fat_poly"],
                                              "kind": "formative"}]}
    queued = {"status": "queued"}
    _state, records, log = _journey([], [scale], {**ALL_FRESH, "secondary": queued,
                                                  "effects": queued, "scales": queued},
                                    shown=("secondary", "effects"))
    [reason] = _by_key(log)["results"].reopened
    assert (reason.decision_id, reason.changed_in, reason.added) == (records[-1].id, "models",
                                                                     ["scales"])
    assert sorted(reason.results) == ["effects", "secondary"]
    assert reason.sentence == ("Your change to Models made 2 results in Results out of date and "
                               "added 1 result there.")


def test_under_describe_a_lens_given_after_the_lock_says_why_results_counts_its_exhibit():
    # No lens at the lock: Results counts nothing yet, and waits for usual intake (0 of 0, not
    # complete: ``exhibits_waiting``). The dietary lens after the lock serves it: 0 of 1, and Your
    # data's answer is named as the one that served it.
    stages = _describe("fresh", lens=False)
    assert stages["results"].progress == Progress(answered=0, required=0, complete=False)
    assert stages["results"].reopened == []
    stages, records = _describe_log("queued", lens="after")
    assert stages["results"].progress == Progress(answered=0, required=1, complete=False)
    assert stages["results"].reopened == [Reopened(
        changed_in="data", decision_id=records[-1].id, kind="set_lens", added=["usual_intake"],
        sentence="Your change to Your data added 1 result to Results.")]
    # computed since: placed, Write-up reached, nothing said
    stages, _records = _describe_log("fresh", lens="after")
    assert stages["results"].progress == Progress(answered=1, required=1, complete=True)
    assert stages["writeup"].reached and stages["results"].reopened == []


@pytest.mark.parametrize("answer,exhibit,action", [(SENSITIVITY, "sensitivity", "declare"),
                                                   (EXPLAIN, "explain", "declare"),
                                                   (SUBSTITUTION, "substitution", "answer")])
def test_i11_holds_on_the_engines_own_logs_when_an_exhibit_is_served_after_the_lock(
        answer, exhibit, action):
    # The verifier's probe (B1-WO1-V1): the lock with both exhibits placed, then the answer with
    # its exhibit not computed yet. I11 asks a complete stage that gains a line, and a stage that
    # stops being reached, to say why; and once the exhibit is placed, nothing is left to say.
    state, records, before = _journey([], [], ALL_FRESH)
    prev = SimpleNamespace(log=before, state=state)
    for status, expected in (("queued", (2, 3)), ("fresh", (3, 3))):
        state, records, log = _journey([], [answer], {**ALL_FRESH, exhibit: {"status": status}})
        snap = SimpleNamespace(action=action, kind=answer["kind"], reopening=False, log=log,
                               steps=[], state=state, artifacts={})
        results = _by_key(log)["results"]
        assert (results.progress.answered, results.progress.required) == expected
        assert _by_key(log)["writeup"].reached is (status == "fresh")
        assert _i11(prev, snap) == []


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
    # adjustment set or estimand was answered, so the engine serves usual_intake, fit and
    # evaluation, and with nothing declared to estimate the goal is Describe (TRUST): of them only
    # usual_intake is its exhibit, and Results says the change to Your question made it out of
    # date.
    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, "prediction")
    state = _decide(records, {"kind": "set_purpose", "purpose": "inference"})
    fit = fit_press.fit_lock(state, records, pressed=True, held=False, estimate=None, opened=True)
    log = _log_of(state, records, stages={"fit": {"status": "fresh"}}, fit=fit)
    results = next(s for s in log.stages if s.key == "results")
    assert results.reopened == [Reopened(
        changed_in="question", decision_id=records[-1].id, kind="set_purpose",
        results=["usual_intake"],
        sentence="Your change to Your question made 1 result in Results out of date.")]
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


def test_write_up_closes_without_a_reason_only_when_the_goal_became_estimate_or_results_says_why():
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
    # B1: under Estimate an eligibility rule leaves Results' exhibits out of date, so none is
    # placed and Write-up closes; Results, holding them, says why, and that is enough
    dropped = _stage("results", [_line("open_seal", "answered")], [_reopened("whos_in",
                                                                             "set_exclusions")])
    snap = _snap("answer", "set_exclusions", False, [dropped, closed], "inference")
    assert _i11(prev, snap) == []
    # under Predict Write-up opens with Results, so Results' reason does not excuse it
    prev = _snap("answer", "select_models", False, [results, writeup], "prediction")
    snap = _snap("answer", "set_exclusions", False, [dropped, closed], "prediction")
    [message] = _i11(prev, snap)
    assert message.startswith("writeup stopped being reached with nothing saying why")


def test_the_aggregation_answer_that_makes_calibration_apply_says_why_in_models():
    """Ruling 3 (P1-FU): after the unit is changed, the aggregation answer (a first answer, decided
    in Who's in) makes the regression-calibration question apply in a Models already worked and
    complete. Models has been reopened, and says why: by that answer, in Who's in. Journey 758:
    journey 26 reached this through the energy question under Predict, which BEATS_PREDICT §9
    ruling 5 no longer asks."""
    journey = path_fuzzer.run_journey(758)
    prev, snap = journey.snapshots[37], journey.snapshots[38]
    assert (snap.action, snap.kind) == ("answer", "set_aggregation")
    assert path_fuzzer.i11_monotone_progress(prev, snap) == []
    before = next(s for s in prev.log.stages if s.key == "models")
    models = next(s for s in snap.log.stages if s.key == "models")
    assert before.progress.complete and not models.progress.complete
    [line] = [l for l in models.lines if l.key == "set_measurement_error"]
    answer = snap.records[-1]
    assert answer.decision.kind == "set_aggregation"
    assert (line.status, line.reopened_by.decision_id, line.reopened_by.stage) == (
        "open", answer.id, "whos_in")
    [reason] = [r for r in models.reopened if r.decision_id == answer.id]
    assert reason.changed_in == "whos_in" and reason.questions == [line.id]
    assert reason.sentence == "Your change to Who's in reopened 1 question in Models."


def test_a_first_answer_into_a_stage_still_asking_reopens_nothing():
    """A first answer that makes a declaration apply in a stage that was not complete is the
    journey going forward: the exclusion rule (Who's in) makes the every-row sensitivity question
    apply while Models still asks its adjustment set again; the new line is open, with no cause,
    and Models names only the confirmation that asked the adjustment set again."""
    from turbotab.core import decisions
    from turbotab.core.tests.test_stage_registry import (EXCLUSION, TG_COVARIATE, _plan, record,
                                                         stage, steps_until)

    _state, records = _plan(TG_COVARIATE)
    records = [*records, record(8, EXCLUSION)]
    log = quest.quest_log(decisions.fold(records), records, steps_until("adjustment"))
    models = stage(log, "models")
    [sensitivity] = [l for l in models.lines if l.key == "set_sensitivity"]
    assert (sensitivity.status, sensitivity.reopened_by) == ("open", None)
    assert [r.kind for r in models.reopened] == ["confirm_role"]
    # the same answer given after a complete Models reaches back into it (``_reaches_back``)
    assert quest._reaches_back(quest._Log(records), records[-1], "models")


def _models_after(snap, records):
    """Models in the quest log the path fuzzer computes for ``records`` on ``snap``'s table."""
    from turbotab.core import decisions

    state = decisions.fold(records)
    arts = path_fuzzer.artifacts_for(state, snap.fixture, records, confidence="high")
    steps = path_fuzzer.route(state, snap.stages, arts, records, pressed=snap.pressed)
    log = path_fuzzer.quest_of(state, records, steps, snap.stages, arts, snap.fixture,
                               snap.pressed, snap.pressed_ever, snap.shown_at)
    return next(s for s in log.stages if s.key == "models")


def test_a_first_answer_given_while_the_stage_still_asked_is_not_named_later():
    """Completeness is judged as the log stood when the answer was given (verifier, P1-FU item 6).
    Journey 758 (journey 26's scenario before BEATS_PREDICT §9 ruling 5) with the model sequence
    answered only after the aggregation answer: Models was 7/9 when the aggregation made the
    calibration question apply, so answering the sequence later (8/9) does not turn that answer
    into a reopening of Models."""
    journey = path_fuzzer.run_journey(758)
    snap = journey.snapshots[38]
    records = list(snap.records)
    [sequence] = [i for i, r in enumerate(records) if r.decision.kind == "set_model_sequence"]
    assert records[-1].decision.kind == "set_aggregation" and sequence < len(records) - 1
    moved = [*records[:sequence], *records[sequence + 1:], records[sequence]]
    moved = [r.model_copy(update={"seq": i + 1, "id": f"r{i + 1}",
                                  "at": path_fuzzer.T0 + path_fuzzer.timedelta(minutes=i + 1)})
             for i, r in enumerate(moved)]
    models = _models_after(snap, moved)
    assert (models.progress.answered, models.progress.required) == (8, 9)
    [line] = [l for l in models.lines if l.key == "set_measurement_error"]
    assert (line.status, line.reopened_by) == ("open", None)
    assert models.reopened == []


def test_no_declaration_is_named_reopened_by_an_earlier_answer_after_the_fact():
    """Seeds where a first answer was given while Models was unreached or still asking, and Models'
    other questions were answered later (17: the exclusions answered at step 20, the selection
    declared at step 28): no open declaration gains a cause that is not the action just taken."""
    for seed in (17, 19, 20):
        journey = path_fuzzer.run_journey(seed)
        for prev, snap in zip(journey.snapshots, journey.snapshots[1:]):
            newest = snap.records[-1].id if snap.records else None
            for old, new in zip(prev.log.stages, snap.log.stages):
                was = {l.key: l for l in old.lines}
                for line in new.lines:
                    o = was.get(line.key)
                    if (line.source != "declaration" or o is None or o.reopened_by is not None
                            or line.reopened_by is None or o.status != "open"
                            or line.status != "open"):
                        continue
                    assert line.reopened_by.decision_id == newest, (
                        seed, snap.index, new.key, line.key, line.reopened_by.kind)


# ── W1-R round 3 (SE-M1): ruling R-a's Results leg ───────────────────────────────
#
# Under Predict an open noticing placed outside Results waits for the triage in Results, so it
# leaves its own stage's count (FOUNDATION §7). One placed in Results is Results' own line: it
# counts there, open or answered. Here a noticing decided at the seal's question (``open_seal``,
# a Results question) is added to the findings the NHANES fixture serves, open, then answered.

AT_THE_SEAL = {"id": "pack::probe::at_the_seal", "severity": "warning",
               "summary": "A noticing decided at the held-out rows", "affected_columns": ["sugar"],
               "routes_to": "open_seal"}


def _results_with(purpose: str, finding: dict | None) -> QuestStage:
    from turbotab.core.interview import route

    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, purpose)
    fit = fit_press.fit_lock(state, records, pressed=True, held=False, estimate=None, opened=True)
    artifacts = path_fuzzer.artifacts_for(state, path_fuzzer.NHANES, records)
    served = artifacts.get("findings") or {}
    findings = {**served, "findings": [*(served.get("findings") or []),
                                       *([finding] if finding else [])]}
    stages = {"fit": {"status": "fresh"}}
    log = quest.quest_log(state, records, route(state, stages, artifacts, records), stages,
                          findings=findings, columns=path_fuzzer.NHANES.columns,
                          artifacts=artifacts, fit=fit)
    return next(s for s in log.stages if s.key == "results")


@pytest.mark.parametrize("purpose", ["prediction", "inference"])
def test_a_noticing_placed_in_results_counts_in_results_open_or_answered(purpose):
    without = _results_with(purpose, None).progress
    opened = _results_with(purpose, AT_THE_SEAL)
    [line] = [l for l in opened.lines if l.key == AT_THE_SEAL["id"]]
    assert (line.status, line.counted) == ("open", True)
    assert opened.progress == Progress(answered=without.answered, required=without.required + 1,
                                       complete=False)
    answered = _results_with(purpose, {**AT_THE_SEAL, "answered_by": "r-dismissed"})
    assert answered.progress.required == without.required + 1
    assert answered.progress.answered == without.answered + 1
