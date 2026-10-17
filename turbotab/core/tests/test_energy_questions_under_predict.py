"""BEATS_PREDICT §9 ruling 5 (W1-P11): under Predict the energy model and the substitution are not
asked, and the dietary energy noticing does not fire; under Estimate each behaves as before.

The energy model decides what a nutrient's coefficient means, and the substitution compares two
diets, an effect: neither is a question a prediction model reads (Willett, Howe & Kushi 1997;
Shmueli 2010). The energy noticing is the same reading twice: the dietary pack's finding that
total energy sits beside candidate nutrients (``pack::dietary::energy_adjustment``, routed to the
energy model), and the measured thread that the exposure moves with total energy
(``diet-energy-carries-the-nutrient``).

Every expectation is from construction: the table and the answers are built here, so which
question a goal asks is the ruling's, and the correlation the noticing measures is NumPy's on the
columns this file draws. Each guard is checked both ways: silent under Predict, firing under
Estimate on the same table and answers.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from turbotab.core import materiality, quest, seal, surfacing, sweep, voice
from turbotab.core.decisions import (
    DecisionLog,
    DismissFinding,
    ExclusionRule,
    OpenSeal,
    ProjectState,
    Revert,
    SelectModels,
    SetEnergyAdjustment,
    SetExclusions,
    SetLens,
    SetMissing,
    SetPurpose,
    SetRoles,
    SetSplit,
    SetSubstitution,
    SetTarget,
    SetTask,
)
from turbotab.core.export.methods import methods_section
from turbotab.core.interview import (
    ENERGY_UNDER_PREDICTION,
    NOT_ASKED,
    SUBSTITUTION_UNDER_PREDICTION,
    _live_writer,
    applicability,
    route,
)
from turbotab.core.models.pipeline import (
    design_spec,
    predictors_from_roles,
    shared_steps,
    transformer,
)
from turbotab.core.provenance import methods_text
from turbotab.core.quest import Facts
from turbotab.core.repairs import annotate
from turbotab.core.stages import build_graph
from turbotab.core.tests import modeling_fixtures as mf

ONE_ROW = {"grain": "one_row_per_unit"}
# Total energy beside three nutrients that carry it, as the dietary lens proposes them today
# (BEATS_PREDICT C1: the nutrients as what you study, under either goal).
ROLES = {"participant_id": "identifier", "age": "covariate", "energy_kcal": "energy",
         "protein_g": "exposure", "fat_g": "exposure", "carbohydrate_g": "exposure"}
ENERGY_FINDING = "pack::dietary::energy_adjustment"
ENERGY_THREAD = "diet-energy-carries-the-nutrient"
FRESH = {s: {"status": "fresh"} for s in (
    "ingest", "oriented", "profile", "findings", "structure", "working", "target_info", "roles",
    "proposals", "cohort", "split", "shelf", "design", "fit", "substitution")}


def _state(purpose: str, **slots) -> ProjectState:
    base = dict(lens=["dietary"], target="hba1c", task="regression", purpose=purpose,
                grain=ONE_ROW, roles=ROLES, exclusions=[], missing="impute",
                split={"holdout": 0.2 if purpose == "prediction" else 0.0})
    return ProjectState(**{**base, **slots})


def _steps(state: ProjectState) -> dict:
    return {s.key: s for s in route(state, FRESH)}


# ── the two questions ────────────────────────────────────────────────────────


def test_under_predict_neither_energy_question_is_asked_and_the_next_question_opens():
    state = _state("prediction")
    steps = _steps(state)
    for key, reason in (("energy_adjustment", ENERGY_UNDER_PREDICTION),
                        ("substitution", SUBSTITUTION_UNDER_PREDICTION)):
        assert steps[key].status == "not_applicable", key
        assert steps[key].reason == reason
        # the surfacing registry's ``fires`` reads the same gate
        assert applicability(key, state) == reason
        # the reason is the clause after the client's own "Not asked:" label
        assert not reason.startswith(NOT_ASKED) and "under prediction" in reason
    # nothing waits behind them: the models question opens
    assert steps["models"].status == "open"
    # with the energy model already recorded and the fit fresh, still not asked
    answered = _steps(_state("prediction", energy_adjustment={"method": "standard",
                                                              "energy_column": "energy_kcal"},
                             models=["linear"]))
    assert answered["energy_adjustment"].status == "not_applicable"
    assert answered["substitution"].status == "not_applicable"


def test_under_estimate_the_same_table_asks_both_as_before():
    state = _state("inference")
    steps = _steps(state)
    for key in ("energy_adjustment", "substitution"):
        assert steps[key].status in ("open", "waiting"), key
        assert applicability(key, state) is None, key
    # each one's own reasons stand under Estimate, as they read before the ruling
    no_energy = {c: r for c, r in ROLES.items() if r != "energy"}
    assert "energy role" in applicability("energy_adjustment", _state("inference", roles=no_energy))
    one = {"energy_kcal": "energy", "protein_g": "exposure", "age": "covariate"}
    assert applicability("substitution", _state("inference", roles=one)) == (
        "Only 1 study factor carries energy; a substitution swaps kcal between two.")


def test_under_predict_the_lens_reason_still_comes_first():
    state = _state("prediction", lens=["genomics"], orientation="sample_major")
    assert applicability("energy_adjustment", state) == (
        "The dietary lens is off, so energy adjustment does not apply.")


def test_a_question_reads_the_goal_it_now_depends_on():
    # clearing every answer a question does not read leaves its gate where it was
    # (test_surfaceable's rule): the goal is now one of them for both questions
    reads = surfacing.QUESTION_READS
    assert "purpose" in reads["energy_adjustment"] and "purpose" in reads["substitution"]
    bare = _state("prediction").model_copy(update={"purpose": None})
    assert applicability("energy_adjustment", bare) is None  # the goal is what removed it


# ── the measured noticing ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    kcal = rng.normal(2000, 450, 400)
    return pd.DataFrame({
        "participant_id": np.arange(400), "age": rng.integers(20, 80, 400).astype(float),
        "energy_kcal": kcal,
        "protein_g": 0.04 * kcal + rng.normal(0, 18, 400),
        "fat_g": 0.035 * kcal + rng.normal(0, 15, 400),
        "carbohydrate_g": 0.12 * kcal + rng.normal(0, 40, 400),
    })


def test_the_energy_noticing_fires_under_estimate_with_numpys_correlation(frame):
    r = float(np.corrcoef(frame["protein_g"], frame["energy_kcal"])[0, 1])
    assert abs(r) >= materiality.ENERGY_R  # the construction is above the noticing's threshold
    found = materiality.energy_noticing(_state("inference"), frame)
    assert found is not None and found.thread == ENERGY_THREAD
    assert found.measure == pytest.approx(r, abs=1e-12)
    assert ENERGY_THREAD in {n.thread for n in materiality.dietary_noticings(_state("inference"),
                                                                             frame)}


def test_the_energy_noticing_is_silent_under_predict_and_the_others_still_fire(frame):
    state = _state("prediction")
    assert materiality.energy_noticing(state, frame) is None
    threads = {n.thread for n in materiality.dietary_noticings(state, frame)}
    assert ENERGY_THREAD not in threads
    # the guard is the energy noticing's alone: day-to-day variance still fires under Predict
    assert "diet-day-to-day-variance" in threads


# ── the pack's finding, as served and as the quest log and the triage carry it ────────────


FINDINGS = {"findings": [
    {"id": ENERGY_FINDING, "severity": "warning", "routes_to": "energy_adjustment",
     "affected_columns": ["energy_kcal", "protein_g", "fat_g", "carbohydrate_g"], "repairs": [],
     "summary": "`fat_g` correlates 0.88 with `energy_kcal`: nutrient effects are tangled with "
                "total energy."},
    {"id": "pack::dietary::implausible_intake", "severity": "info", "routes_to": "exclusions",
     "affected_columns": ["energy_kcal"], "repairs": [],
     "summary": "12 records report an implausible daily intake"},
]}


def _served(state: ProjectState) -> dict:
    return annotate(FINDINGS, state, [])


def _ids(served: dict) -> set[str]:
    return {f["id"] for f in served["findings"]}


@pytest.mark.parametrize("purpose", ["inference", "prediction"])
def test_the_packs_energy_finding_is_served_only_under_estimate(purpose):
    state = _state(purpose)
    served = _served(state)
    item = surfacing.registry()[f"noticing:{ENERGY_FINDING}"]
    fires = item.fires(state, Facts(artifacts={"findings": served}))
    if purpose == "inference":
        assert ENERGY_FINDING in _ids(served) and fires
    else:
        assert ENERGY_FINDING not in _ids(served) and not fires
    # the guard is the energy finding's alone: the implausible intakes are served under both
    assert "pack::dietary::implausible_intake" in _ids(served)


@pytest.mark.parametrize("purpose", ["inference", "prediction"])
def test_the_quest_log_and_the_triage_carry_the_energy_finding_only_under_estimate(purpose):
    state = _state(purpose)
    served = _served(state)
    steps = route(state, FRESH)
    log = quest.quest_log(state, [], steps, findings=served, columns=list(ROLES))
    lines = {l.key: (s.key, l) for s in log.stages for l in s.lines if l.source == "finding"}
    triaged = {i.id for i in sweep.triage(state, log, served).items}
    if purpose == "inference":
        where, line = lines[ENERGY_FINDING]
        assert (where, line.label, line.status) == ("models", quest.DECIDE, "open")
        assert ENERGY_FINDING in triaged
    else:
        # no Decide in Models routed to a question Predict does not ask, and nothing in the
        # triage to act on there, so the held-out rows do not wait on it
        assert ENERGY_FINDING not in lines
        assert ENERGY_FINDING not in triaged


# ── an answer recorded under Estimate, then the goal changed to Predict (option a') ──────────
#
# A direct post can still record the energy model or the substitution, and an answer recorded
# under Estimate stays in the record when the goal changes. Under Predict such an answer is held,
# as an answer whose premise has lapsed is (``decisions.register_kind(..., holds=)``): it stays in
# the record, shapes no fit and writes no methods sentence, and applies again once the goal is
# Estimate. The truth is a second project built here with the same answers but that one: under
# Predict the two are the same project.

PROJECT = (SetLens(lenses=["dietary"]), SetTarget(column="glucose"),
           SetTask(column="glucose", task="regression"), SetRoles(roles=mf.NHANES_ROLES),
           SetMissing(strategy="complete_case"), SetSplit(holdout=0.2, seed=0, folds=5),
           SelectModels(models=["linear"]))
ENERGY = SetEnergyAdjustment(method="residual", energy_column="kcal",
                             nutrients=["protein", "carb", "fat_total"])
SWAP = SetSubstitution(donor="fat_total", recipient="carb", step_kcal=100.0)
ESTIMATE, PREDICT = SetPurpose(purpose="inference"), SetPurpose(purpose="prediction")


def _project(folder, *answers) -> DecisionLog:
    log = DecisionLog(folder / "decisions.jsonl")
    for decision in (*PROJECT, *answers):
        _say(log, decision)
    return log


def _say(log: DecisionLog, decision):
    return log.append(decision, sentence=lambda d, before: voice.sentence_for(d, before, None))


@pytest.fixture(scope="module")
def nhanes() -> pd.DataFrame:
    return mf.nhanes_like(300, seed=3)


def _matrix(state: ProjectState, frame: pd.DataFrame) -> pd.DataFrame:
    """The columns the fit's models are given: the design's shared steps (the fill, the energy
    model, the forms, the levels) on the roles' predictors, as the design stage builds them."""
    spec = design_spec(state, frame, predictors_from_roles(state.roles, state.target))
    return transformer(shared_steps(spec)).fit_transform(frame[spec.inputs])


def _methods(log: DecisionLog) -> list[tuple[str, str, bool]]:
    return [(line.kind, line.sentence, line.in_force)
            for line in methods_text(log.records()).lines]


def _routed(log: DecisionLog) -> dict:
    return {s.key: s for s in route(log.state(), FRESH, records=log.records())}


def _writer(log: DecisionLog, slot: str) -> str | None:
    return _live_writer(log.records(), log.state()).get(slot)


def test_an_energy_answer_recorded_under_estimate_is_held_under_predict_and_applies_again(
        tmp_path, nhanes):
    switched = _project(tmp_path / "switched", ESTIMATE, ENERGY, PREDICT)
    never = _project(tmp_path / "never", ESTIMATE, PREDICT)
    recorded = next(r for r in switched.records() if r.decision.kind == "set_energy_adjustment")
    # the answer stays in the record, and holds nothing under Predict
    state = switched.state()
    assert state.purpose == "prediction" and state.energy_adjustment is None
    assert _writer(switched, "energy_adjustment") is None
    step = _routed(switched)["energy_adjustment"]
    assert (step.status, step.decision_id) == ("not_applicable", None)
    # the fit's predictors are the roles': total energy beside the nutrients, each as recorded,
    # as in the project that never had an energy answer
    predict = _matrix(state, nhanes)
    pd.testing.assert_frame_equal(predict, _matrix(never.state(), nhanes))
    for column in ("kcal", "protein", "carb", "fat_total"):
        np.testing.assert_array_equal(predict[column].to_numpy(), nhanes[column].to_numpy())
    # the methods are the other project's, word for word: no energy model is stated
    assert _methods(switched) == _methods(never)
    assert "set_energy_adjustment" not in {kind for kind, _, _ in _methods(switched)}

    # back under Estimate the answer applies again: as in the project whose goal never changed
    _say(switched, ESTIMATE)
    before = _project(tmp_path / "before", ESTIMATE, ENERGY)
    back = switched.state()
    assert back.energy_adjustment is not None
    assert back.energy_adjustment == before.state().energy_adjustment
    assert _writer(switched, "energy_adjustment") == recorded.id
    step = _routed(switched)["energy_adjustment"]
    assert (step.status, step.decision_id) == ("answered", recorded.id)
    estimate = _matrix(back, nhanes)
    pd.testing.assert_frame_equal(estimate, _matrix(before.state(), nhanes))
    # the guard is silent here: each nutrient reaches the models as its energy residual, in
    # place of its recorded values, with total energy beside it
    assert {"protein_adj", "carb_adj", "fat_total_adj", "kcal"} <= set(estimate.columns)
    assert not {"protein", "carb", "fat_total"} & set(estimate.columns)
    assert not np.allclose(estimate["protein_adj"].to_numpy(), nhanes["protein"].to_numpy())
    said = [(sentence, in_force) for kind, sentence, in_force in _methods(switched)
            if kind == "set_energy_adjustment"]
    assert said == [(recorded.sentence, True)] and recorded.sentence
    assert said == [(sentence, in_force) for kind, sentence, in_force in _methods(before)
                    if kind == "set_energy_adjustment"]


def test_a_substitution_recorded_under_estimate_is_held_under_predict_and_applies_again(
        tmp_path, nhanes):
    switched = _project(tmp_path / "switched", ESTIMATE, SWAP, PREDICT)
    never = _project(tmp_path / "never", ESTIMATE, PREDICT)
    recorded = next(r for r in switched.records() if r.decision.kind == "set_substitution")
    state = switched.state()
    assert state.purpose == "prediction" and state.substitution is None
    assert _writer(switched, "substitution") is None
    # no curve is drawn: the substitution stage waits on its answer, as with none recorded
    assert surfacing.blocked(state)["substitution"] == ("substitution",)
    assert surfacing.blocked(state) == surfacing.blocked(never.state())
    pd.testing.assert_frame_equal(_matrix(state, nhanes), _matrix(never.state(), nhanes))
    assert _methods(switched) == _methods(never)
    assert "set_substitution" not in {kind for kind, _, _ in _methods(switched)}

    _say(switched, ESTIMATE)
    before = _project(tmp_path / "before", ESTIMATE, SWAP)
    back = switched.state()
    assert back.substitution is not None and back.substitution == before.state().substitution
    assert _writer(switched, "substitution") == recorded.id
    # the guard is silent here: the curve's stage can compute again
    assert surfacing.blocked(back)["substitution"] == ()
    said = [(sentence, in_force) for kind, sentence, in_force in _methods(switched)
            if kind == "set_substitution"]
    assert said == [(recorded.sentence, True)] and recorded.sentence


def test_an_answer_recorded_under_predict_holds_nothing_and_applies_once_the_goal_is_estimate(
        tmp_path):
    # a direct post under Predict is recorded, held from the start, and stands under Estimate
    log = _project(tmp_path, PREDICT, ENERGY, SWAP)
    assert len(log.records()) == len(PROJECT) + 3
    state = log.state()
    assert state.energy_adjustment is None and state.substitution is None
    _say(log, ESTIMATE)
    state = log.state()
    assert state.energy_adjustment.method == "residual" and state.substitution.donor == "fat_total"
    # with no goal answered yet, an answer stands, as before the ruling
    bare = _project(tmp_path / "bare", ENERGY, SWAP).state()
    assert bare.purpose is None
    assert bare.energy_adjustment is not None and bare.substitution is not None


# ── an answer posted after the held-out rows were opened (W1-P11c) ───────────────────────────
#
# Under Predict a direct post still records the energy model or the substitution after the opening.
# Held, it shapes no fit, so it is no change made after the opening: the methods, and the exported
# methods, state neither, and the seal lists neither among the post-seal changes (what the fit, or
# the substitution stage, reads). The truth is the project with the same answers but those: under
# Predict the two are the same project. A noticing dismissed after the opening repairs nothing, so
# it is stated as dismissed and not listed either. Silent: a missing-values change and a new
# exclusion after the opening change what the fit reads, so each is stated and listed; and once
# the goal is Estimate the energy answer applies, so it is stated and listed as a change after the
# opening.

OPENED = OpenSeal(family="linear", target="glucose")


def _changes(log: DecisionLog, stage: str) -> list[str]:
    return seal.post_seal_changes(log.records(), seal.slots_read_by(build_graph(), stage))


def _exported(log: DecisionLog) -> str:
    """The bundle's methods.md for the log, the analysis's own paragraphs aside."""
    source = SimpleNamespace(name="diet", purpose=log.state().purpose,
                             methods=methods_text(log.records()), artifact=lambda stage: None)
    return methods_section(source, {"engine": {}, "inputs": [], "decisions": {"n": 1},
                                    "analysis_plan": {}}).markdown


def test_an_energy_or_substitution_answer_posted_after_the_opening_under_predict_is_no_change(
        tmp_path, nhanes):
    posted = _project(tmp_path / "posted", PREDICT, OPENED, ENERGY, SWAP)
    never = _project(tmp_path / "never", PREDICT, OPENED)
    energy, swap = (next(r for r in posted.records() if r.decision.kind == kind)
                    for kind in ("set_energy_adjustment", "set_substitution"))
    # recorded after the opening, and marked so in the record
    assert energy.post_seal and swap.post_seal
    assert energy.sentence.startswith("After the held-out rows were opened, energy was adjusted")
    # held: the fit's predictors are the other project's
    pd.testing.assert_frame_equal(_matrix(posted.state(), nhanes), _matrix(never.state(), nhanes))
    # the methods, and the exported methods, are the other project's word for word
    assert _methods(posted) == _methods(never)
    assert _exported(posted) == _exported(never)
    assert "energy was adjusted" not in _exported(posted)
    assert "Decisions made after the held-out rows were opened" not in _exported(posted)
    assert _changes(posted, "fit") == [] and _changes(posted, "substitution") == []
    # a revert of the held answer withdraws nothing that stood
    withdrawn = _project(tmp_path / "withdrawn", PREDICT, OPENED, ENERGY)
    _say(withdrawn, Revert(decision_id=withdrawn.records()[-1].id))
    assert _methods(withdrawn) == _methods(never) and _changes(withdrawn, "fit") == []
    # a noticing dismissed after the opening repairs nothing: stated as dismissed, never listed
    dismissed = _say(posted, DismissFinding(finding_id=ENERGY_FINDING))
    assert dismissed.sentence.startswith("After the held-out rows were opened, ")
    assert "was kept as is, with nothing changed" in dismissed.sentence  # R-f 18
    assert _changes(posted, "fit") == []

    # silent: a missing-values change and a new exclusion after the opening are stated and listed
    missing = _say(posted, SetMissing(strategy="impute"))
    excluded = _say(posted, SetExclusions(rules=[ExclusionRule(column="age", low=20,
                                                               reason="adults")]))
    assert _changes(posted, "fit") == [missing.id, excluded.id]
    after = [(line.record_id, line.sentence) for line in methods_text(posted.records()).lines
             if line.post_seal]
    assert after == [(dismissed.id, dismissed.sentence), (missing.id, missing.sentence),
                     (excluded.id, excluded.sentence)]
    assert missing.sentence.startswith("After the held-out rows were opened, ")
    exported = _exported(posted)
    assert "## Decisions made after the held-out rows were opened" in exported
    assert missing.sentence in exported and "energy was adjusted" not in exported

    # silent: under Estimate the energy answer applies, so it is a change after the opening
    goal = _say(posted, ESTIMATE)
    assert posted.state().energy_adjustment is not None
    assert _changes(posted, "fit") == [energy.id, missing.id, excluded.id, goal.id]
    assert swap.id in _changes(posted, "substitution")
    said = {line.record_id: (line.sentence, line.in_force)
            for line in methods_text(posted.records()).lines}
    assert said[energy.id] == (energy.sentence, True) and swap.id in said


# ── W1-R round 2, ruling R-d: what the lock means, and what a held answer never says ──────────
#
# An answer recorded after the opening and held now applied to nothing the analysis is now, so
# the methods never state it as applied, whatever the goal was when it was recorded: the project
# is the one with the same answers but those, word for word. An answer in force at the lock (or
# the opening) and held now stays in the plan as declared, since that is what the lock fixed, and
# one added sentence says it is not applied under the goal now. The truth is built here: the same
# log without the held answers, and the goal the state declares.

NOT_APPLIED = "It is not applied while the analysis is declared for `prediction`."


def test_an_answer_recorded_under_estimate_after_the_opening_and_held_now_is_never_stated(
        tmp_path):
    switched = _project(tmp_path / "switched", PREDICT, OPENED, ESTIMATE, ENERGY, SWAP, PREDICT)
    never = _project(tmp_path / "never", PREDICT, OPENED, ESTIMATE, PREDICT)
    state = switched.state()
    assert state.energy_adjustment is None and state.substitution is None
    assert _methods(switched) == _methods(never)
    assert _exported(switched) == _exported(never)
    for words in ("energy was adjusted", "substitution studied", NOT_APPLIED):
        assert words not in _exported(switched), words
    # the methods and the seal's listing agree: neither answer is a change made after the opening
    held = {r.id for r in switched.records()
            if r.decision.kind in ("set_energy_adjustment", "set_substitution")}
    assert not held & set(_changes(switched, "fit")) and not held & set(
        _changes(switched, "substitution"))
    # silent: once the goal is Estimate again each applies, so each is stated
    _say(switched, ESTIMATE)
    kinds = {line.kind: line for line in methods_text(switched.records()).lines}
    assert {"set_energy_adjustment", "set_substitution"} <= set(kinds)
    assert kinds["set_energy_adjustment"].in_force
    assert NOT_APPLIED not in kinds["set_energy_adjustment"].sentence


def test_an_answer_in_force_at_the_lock_and_held_now_stays_with_its_goal_said(tmp_path):
    from turbotab.core.decisions import LockPlan

    log = _project(tmp_path / "locked", ESTIMATE, ENERGY, LockPlan(), PREDICT)
    energy = next(r for r in log.records() if r.decision.kind == "set_energy_adjustment")
    assert log.state().energy_adjustment is None
    lines = {line.record_id: line for line in methods_text(log.records()).lines}
    # the plan as declared keeps it, not in force, with one sentence on the goal now
    assert energy.id in lines and not lines[energy.id].in_force
    assert lines[energy.id].sentence == f"{energy.sentence} {NOT_APPLIED}"
    assert lines[energy.id].sentence.count("It is not applied") == 1
    # silent: still under Estimate, the plan's answer applies and is said as recorded
    kept = _project(tmp_path / "kept", ESTIMATE, ENERGY, LockPlan())
    [said] = [line for line in methods_text(kept.records()).lines
              if line.kind == "set_energy_adjustment"]
    assert said.in_force and said.sentence == energy.sentence
    # silent: back under Estimate, the sentence on the goal is gone again
    _say(log, ESTIMATE)
    lines = {line.record_id: line for line in methods_text(log.records()).lines}
    assert lines[energy.id].in_force and lines[energy.id].sentence == energy.sentence
