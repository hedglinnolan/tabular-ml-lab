"""P0.8 · Fit, the hold and a visible lock (SIZING P0.8; RECIPES_AND_TUNING §4.4; CROSSWALK
"Settled here"), with seam guard 7 (V2X_SEAMS) and guarantee test 2 (EXTERNAL_AUDIT_2026-10-09
recommendation 4 and §3.2).

The expectations come from outside the module under test: which artifacts put an estimate in view
is ``plan_lock.shows_estimates`` (and, for scores, ``selection.scored_in``); the plan's SHA-256 is
computed here with ``hashlib`` over the canonical JSON the export documents; the hold's numbers are
the ruling's ("about 2 minutes", "1.5 times the last one confirmed").
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from turbotab.core import decisions, fit_press
from turbotab.core.decisions import DecisionRecord, ProjectState, parse_decision
from turbotab.core.plan_lock import shows_estimates

T0 = datetime(2026, 10, 9, 14, 2, tzinfo=timezone.utc)

# One artifact per estimate stage that puts an estimate in view, by ``shows_estimates``' reading.
SHOWING = {
    "fit": {"models": [{"family": "linear", "coefficients": [{"feature": "sugar", "estimate": 0.4}],
                        "cv": {"r2": 0.31}}]},
    "substitution": {"models": [{"family": "linear", "delta": [0.1, 0.2]}]},
    "sensitivity": {"families": [{"fits": [{"coefficients": [{"feature": "sugar"}]}]}]},
    "secondary": {"families": [{"fits": [{"exposure_tests": [{"feature": "sugar"}]}]}]},
    "calibration": {"exposures": [{"column": "sugar", "estimate": 0.5, "naive": 0.4}],
                    "contrasts": [{"estimate": 0.5}]},
    "scales": {"scales": [{"name": "stress", "alpha": 0.8, "correction": {"estimate": 1.2}}]},
    "effects": {"families": [{"sequence": [{"effects": [{"estimate": 1.1}]}]}]},
    "causal": {"estimates": [{"estimate": 0.2}]},
    "time_varying": {"estimates": {"rows": [{"estimate": 0.3}]}},
    "explain": {"families": [{"family": "linear", "explained": True}]},
    "modification": {"modifications": [{"families": [{"effects": [{"estimate": 1.4}]}]}]},
    "evaluation": {"estimates": {"path": [{"estimate": 0.1}]}},
    "usual_intake": {"offer": {"offered": True},
                     "analyses": [{"nutrient": "iron", "mean": {"value": 12.0},
                                   "percentiles": {"50": {"value": 11.0}}}]},
}


def record(seq: int, decision: dict) -> DecisionRecord:
    return DecisionRecord(id=f"r{seq}", seq=seq, at=T0 + timedelta(minutes=seq),
                          decision=parse_decision(decision))


def canonical_sha(plan: dict) -> str:
    text = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── seam guard 7: the estimate stages are declared, Describe's among them ────


def test_describe_s_usual_intake_distribution_is_declared_an_estimate():
    from turbotab.core.estimand import ESTIMATE_STAGES
    from turbotab.core.stages import ESTIMATE, build_graph

    assert build_graph()["usual_intake"].serves == ESTIMATE
    assert "usual_intake" in ESTIMATE_STAGES
    # Every declared estimate stage has an artifact here that shows an estimate, so a new one
    # cannot join the list without a check below.
    assert set(ESTIMATE_STAGES) == set(SHOWING)


def test_the_usual_intake_distribution_is_read_as_an_estimate():
    assert shows_estimates("usual_intake", SHOWING["usual_intake"])
    assert not shows_estimates("usual_intake", {"offer": {"offered": True}, "analyses": []})


# ── guarantee test 2: nothing estimated computes or is served without a purpose ──


def test_every_estimate_stage_waits_for_the_purpose_before_it_computes():
    # The engine computes no stage while a slot it requires is unanswered (``graph.Engine``: its
    # key is None and its status ``blocked``), so requiring the purpose is what keeps every
    # estimate stage from computing without one, whatever else it requires.
    from turbotab.core.estimand import ESTIMATE_STAGES
    from turbotab.core.stages import build_graph

    graph = build_graph()
    for name in ESTIMATE_STAGES:
        assert "purpose" in graph[name].requires, name


def test_a_stage_that_serves_an_estimate_without_waiting_for_the_purpose_is_refused():
    from turbotab.core.graph import Graph, GraphError, Stage
    from turbotab.core.stages import ESTIMATE, estimates_wait_for_the_purpose

    loose = Graph([Stage("late_notices", 1, (), ("target",), lambda ctx: {}, serves=ESTIMATE,
                         requires=("target",))])
    with pytest.raises(GraphError, match="late_notices"):
        estimates_wait_for_the_purpose(loose)
    held = Graph([Stage("late_notices", 1, (), ("target",), lambda ctx: {}, serves=ESTIMATE,
                        requires=("target", "purpose"))])
    assert estimates_wait_for_the_purpose(held) is held


@pytest.mark.parametrize("stage", sorted(SHOWING))
def test_no_estimate_is_served_while_the_purpose_is_unanswered(stage):
    artifact = json.loads(json.dumps(SHOWING[stage]))
    assert shows_estimates(stage, artifact)
    for pressed in (False, True):
        served = fit_press.served(stage, artifact, ProjectState(), pressed=pressed)
        assert not shows_estimates(stage, served), (stage, pressed)
        assert served["withheld"] and served["withheld_for"] == "purpose"
    assert shows_estimates(stage, artifact)  # the artifact itself is left as computed


# ── (a) under Estimate and Describe, nothing before the lock ─────────────────


@pytest.mark.parametrize("stage", sorted(SHOWING))
def test_under_inference_no_estimate_is_served_before_the_plan_is_locked(stage):
    open_plan = ProjectState(purpose="inference")
    served = fit_press.served(stage, SHOWING[stage], open_plan, pressed=True)
    assert not shows_estimates(stage, served)
    assert served["withheld_for"] == "fit"
    locked = open_plan.model_copy(update={"plan_locked": True})
    assert fit_press.served(stage, SHOWING[stage], locked, pressed=False) == SHOWING[stage]


# ── (b) under Predict, no estimate and no score before Fit ───────────────────


def test_under_prediction_no_score_is_served_before_fit():
    from turbotab.core.models.selection import scored_in

    state = ProjectState(purpose="prediction", target="glucose")
    fit = SHOWING["fit"]
    assert scored_in(fit) == ["linear"]
    before = fit_press.served("fit", fit, state, pressed=False)
    assert scored_in(before) == [] and not shows_estimates("fit", before)
    assert before["withheld_for"] == "fit"
    assert fit_press.served("fit", fit, state, pressed=True) == fit
    for stage in SHOWING:
        assert not shows_estimates(stage, fit_press.served(stage, SHOWING[stage], state,
                                                           pressed=False)), stage


def test_under_prediction_the_evaluation_s_scores_wait_for_fit_too():
    # The benchmark, the decision curve, the subgroups' and the internal-external scores are the
    # prediction track's results: none is served before Fit.
    evaluation = {"purpose": "prediction", "task": "binary", "scores_shown": True, "metric": "auc",
                  "benchmark": {"metric": "auc", "estimate": 0.71, "sentence": "AUC 0.71"},
                  "decision_curve": {"net_benefit": [0.12]}, "subgroups": [{"auc": 0.69}],
                  "internal_external": [{"site": "a", "auc": 0.66}], "shrinkage": {"slope": 0.9},
                  "design_based": {"families": [{"auc": 0.7}]}, "estimates": None,
                  "sentences": ["The benchmark's AUC was 0.71."]}
    state = ProjectState(purpose="prediction", target="event")
    before = fit_press.served("evaluation", evaluation, state, pressed=False)
    numbers = json.dumps({k: v for k, v in before.items() if k != "withheld"})
    assert not any(str(v) in numbers for v in (0.71, 0.12, 0.69, 0.66, 0.9, 0.7))
    assert before["scores_shown"] is False and before["withheld_for"] == "fit"
    assert fit_press.served("evaluation", evaluation, state, pressed=True) == evaluation


def test_a_press_counts_for_the_outcome_it_was_pressed_for():
    press = {"target": "glucose", "seconds": 30.0, "seq": 12}
    assert fit_press.pressed_for(press, "glucose")
    assert not fit_press.pressed_for(press, "insulin")
    assert not fit_press.pressed_for(None, "glucose")


# ── (c) the scheduler's hold ─────────────────────────────────────────────────


def test_the_fit_estimate_is_the_chosen_families_measured_times_summed():
    shelf = {"families": [{"key": "linear", "estimate_seconds": 30.0},
                          {"key": "elastic_net", "estimate_seconds": 100.0},
                          {"key": "boosted_trees", "estimate_seconds": 900.0},
                          {"key": "ridge", "estimate_seconds": None}]}
    assert fit_press.fit_estimate(shelf, ["linear", "elastic_net"]) == 130.0
    assert fit_press.fit_estimate(shelf, ["ridge"]) is None
    assert fit_press.fit_estimate(None, ["linear"]) is None
    assert fit_press.fit_estimate({"data": shelf}, ["boosted_trees"]) == 900.0


def test_a_fit_expected_to_take_over_about_two_minutes_waits_for_fit():
    assert fit_press.HOLD_SECONDS == 120.0
    assert not fit_press.holds(None, None, "glucose")  # nothing measured: computing stays live
    assert not fit_press.holds(120.0, None, "glucose")
    assert fit_press.holds(121.0, None, "glucose")
    press = {"target": "glucose", "seconds": 200.0, "seq": 3}
    assert not fit_press.holds(200.0, press, "glucose")
    assert fit_press.holds(200.0, press, "insulin")  # pressed for another outcome
    # The hold re-arms when a new estimate exceeds 1.5 times the last one confirmed.
    assert not fit_press.holds(300.0, press, "glucose")
    assert fit_press.holds(300.5, press, "glucose")
    assert not fit_press.holds(900.0, {"target": "glucose", "seconds": None, "seq": 3}, "glucose")


def test_a_press_is_kept_beside_the_project(tmp_path):
    assert fit_press.read_press(tmp_path) is None
    fit_press.record_press(tmp_path, target="glucose", seconds=140.0, seq=9)
    assert fit_press.read_press(tmp_path) == {"target": "glucose", "seconds": 140.0, "seq": 9}
    (tmp_path / fit_press.PRESS_FILE).write_text("{not json", "utf-8")
    assert fit_press.read_press(tmp_path) is None


# ── (d) the lock is visible ──────────────────────────────────────────────────


def test_the_lock_is_reported_with_its_time_and_fingerprint():
    plan = {"purpose": "inference", "target": "glucose", "models": ["linear"]}
    sha = canonical_sha(plan)
    records = [record(1, {"kind": "set_purpose", "purpose": "inference"}),
               record(2, {"kind": "lock_plan", "plan": plan, "digest": sha})]
    state = decisions.fold(records)
    report = fit_press.fit_lock(state, records, pressed=True, held=False, estimate=None)
    assert report.locks and report.locked
    assert report.at == T0 + timedelta(minutes=2)
    assert report.sha256 == sha
    assert "14:04" in report.reason and "before any estimate was shown" in report.reason

    unlocked = fit_press.fit_lock(decisions.fold(records[:1]), records[:1], pressed=False,
                                  held=True, estimate=600.0)
    assert unlocked.locks and not unlocked.locked and unlocked.held
    assert unlocked.at is None and unlocked.sha256 is None
    assert "Fit" in unlocked.reason and "about 10 minutes" in unlocked.reason


def test_a_lock_made_after_estimates_were_shown_under_prediction_says_so():
    plan = {"purpose": "inference", "target": "glucose"}
    records = [record(1, {"kind": "set_purpose", "purpose": "inference"}),
               record(2, {"kind": "lock_plan", "plan": plan, "digest": canonical_sha(plan),
                          "seen": "prediction", "seen_target": "glucose", "seen_at": 1})]
    report = fit_press.fit_lock(decisions.fold(records), records, pressed=False, held=False,
                                estimate=None)
    assert report.locked and "under prediction" in report.reason


def test_under_prediction_nothing_locks_and_the_report_says_why():
    records = [record(1, {"kind": "set_purpose", "purpose": "prediction"})]
    state = decisions.fold(records)
    before = fit_press.fit_lock(state, records, pressed=False, held=False, estimate=None)
    assert not before.locks and not before.locked and not before.pressed
    assert "nothing locks" in before.reason and "Fit" in before.reason
    after = fit_press.fit_lock(state, records, pressed=True, held=False, estimate=None)
    assert after.pressed and not after.locked


def test_with_no_purpose_nothing_is_locked_or_shown():
    report = fit_press.fit_lock(ProjectState(), [], pressed=False, held=False, estimate=None)
    assert report.purpose is None and not report.locked and not report.pressed
    assert "goal" in report.reason
