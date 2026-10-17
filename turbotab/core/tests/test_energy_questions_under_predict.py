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

import numpy as np
import pandas as pd
import pytest

from turbotab.core import materiality, quest, surfacing, sweep
from turbotab.core.decisions import ProjectState
from turbotab.core.interview import (
    ENERGY_UNDER_PREDICTION,
    NOT_ASKED,
    SUBSTITUTION_UNDER_PREDICTION,
    applicability,
    route,
)
from turbotab.core.quest import Facts
from turbotab.core.repairs import annotate

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
