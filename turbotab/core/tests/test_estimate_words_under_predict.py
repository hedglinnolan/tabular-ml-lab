"""W1-R (the walkthrough of 2026-10-17, findings 3a, 3b and 3c): under Predict the engine says
nothing that only an Estimate says, and under Estimate it says what it said before.

* **3a, day-to-day noise.** Regression dilution weakens a slope on a noisy measure by ``1 − λ``
  (Hutcheon et al. 2010), which biases an estimate. A prediction model is fit and scored on that same
  noisy measure, so the rows it is scored on carry the noise the model meets in use and the score is
  honest. The sentence names them "the held-out rows" where the split holds some out, else "the
  rows the model is scored on" (cross-validated with none held out). The noise limits how much
  the column can predict. Under Predict the triage therefore records it For the record: it changes
  no number, and no limitation is drafted. Under Estimate it could bias the estimate, as before.
* **3b, the design's concerns.** What a total's coefficient means beside its parts, a substitution
  through it, two columns of one nutrient that may overlap in a swap, and fiber's energy counted
  twice are concerns about coefficients and substitutions. Under Predict no coefficient is
  interpreted and nothing is substituted (BEATS_PREDICT §9 ruling 5), so none is said. The shares
  that sum to 100% stay: no linear model can estimate all of them under either goal.
* **3c, the role words.** Under Predict no column is studied and nothing is adjusted for, so an
  exposure and a covariate are each "a predictor" (BEATS_PREDICT C1: a predictor is the engine's
  covariate), a reading the values settled carries the roles stage's own evidence without its study
  clause, and the roles card names no study factor. The engine's roles stay as they are, and the
  words under Estimate are unchanged.

Every expectation is from construction: the tables, the answers and the nesting are built here. The
reliability λ is the balanced one-way random-effects estimate (method of moments), written out here.
Each guard is checked both ways, on the same table and answers under each goal.
"""
from __future__ import annotations

import re
import typing

import numpy as np
import pandas as pd
import pytest

from turbotab.core import materiality as M
from turbotab.core import quest, readings, sweep
from turbotab.core.decisions import ProjectState, SplitSpec
from turbotab.core.models.pipeline import design_spec, warnings_for
from turbotab.core.stages.rows import roles_stage
from turbotab.core.teaching.content import entries_now
from turbotab.core.tests.stage_harness import Ingested
from turbotab.core.tests.test_stage_registry import steps_until

DAYS = "diet-day-to-day-variance"
ROLES = {"participant_id": "identifier", "age": "covariate", "energy_kcal": "energy",
         "protein_g": "exposure", "fat_g": "exposure"}
STUDY_WORDS = ("what you study", "study factor", "exposure")


def _state(purpose: str | None, **slots) -> ProjectState:
    base = dict(lens=["dietary"], target="hba1c", task="regression", purpose=purpose,
                grain={"grain": "one_row_per_unit"}, roles=ROLES, exclusions=[])
    return ProjectState(**{**base, **slots})


def _one_recall() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 300
    kcal = rng.normal(2000, 450, n)
    return pd.DataFrame({"participant_id": np.arange(n), "age": rng.integers(20, 80, n).astype(float),
                         "energy_kcal": kcal, "protein_g": 0.04 * kcal + rng.normal(0, 18, n),
                         "fat_g": 0.035 * kcal + rng.normal(0, 15, n)})


def _two_recalls(sd_between: float, sd_within: float, people: int = 200, k: int = 2):
    """``k`` recalls a person: a usual intake plus day-to-day noise, the same count each."""
    rng = np.random.default_rng(17)
    usual = rng.normal(70, sd_between, people)
    rows = pd.DataFrame({"participant_id": np.repeat(np.arange(people), k),
                         "protein_g": np.repeat(usual, k) + rng.normal(0, sd_within, people * k)})
    rows["age"] = 50.0
    rows["energy_kcal"] = 2000.0
    rows["fat_g"] = 60.0
    return rows


def _balanced_lambda(frame: pd.DataFrame, k: int) -> float:
    """λ of a person's mean over ``k`` recalls, balanced: σ²b = (MSB − MSW) / k, σ²w = MSW."""
    groups = frame.groupby("participant_id")["protein_g"]
    g = groups.ngroups
    means = groups.mean()
    msw = float(((frame["protein_g"] - frame["participant_id"].map(means)) ** 2).sum()) / (g * (k - 1))
    msb = k * float(((means - frame["protein_g"].mean()) ** 2).sum()) / (g - 1)
    var_b = max((msb - msw) / k, 0.0)
    return var_b / (var_b + msw / k)


def _triaged(state: ProjectState, noticing: M.Noticing) -> sweep.TriageItem:
    log = quest.quest_log(state, [], steps_until(None))
    t = sweep.triage(state, log, {"findings": []}, noticed=[noticing])
    return next(i for i in t.items if i.id == DAYS)


# ── 3a · day-to-day noise ────────────────────────────────────────────────────


HELD_OUT = SplitSpec(holdout=0.2, seed=0)


def test_under_predict_day_to_day_noise_is_for_the_record_and_biases_no_score():
    state = _state("prediction", split=HELD_OUT)
    noticing = M.variance_noticing(state, _one_recall())
    assert noticing is not None and noticing.predicted.not_measurable  # one recall day
    item = _triaged(state, noticing)
    assert (item.recommended, item.label, item.limitation) == ("no_change", "For the record", False)
    assert item.reason == ("The held-out rows carry the same day-to-day noise in `protein_g` that "
                           "the model meets in use, where `protein_g` is measured the same way, so "
                           "the score is honest; the noise limits how much `protein_g` can "
                           "predict.")
    assert "bias" not in item.reason and "slope" not in item.reason
    # no movement of the score is measured: the attenuation bounds a slope, not a score
    assert (item.band, item.measure, item.calibrated) == (None, None, None)
    assert item.summary == "Part of `protein_g`'s spread is day-to-day noise"


@pytest.mark.parametrize("split", [SplitSpec(holdout=0.0, seed=0, folds=5), None],
                         ids=["no rows held out", "the split unanswered"])
def test_under_predict_with_no_rows_held_out_the_noise_line_names_the_rows_the_model_is_scored_on(
        split):
    """With no rows held out the score is cross-validated (the BBC-CV case), and with the split
    unanswered no row is drawn yet: either way no held-out row exists to carry the noise, so the
    line names the rows the model is scored on, which carry it under any validation."""
    state = _state("prediction", split=split)
    item = _triaged(state, M.variance_noticing(state, _one_recall()))
    assert (item.recommended, item.label, item.limitation) == ("no_change", "For the record", False)
    assert item.reason == ("The rows the model is scored on carry the same day-to-day noise in "
                           "`protein_g` that the model meets in use, where `protein_g` is "
                           "measured the same way, so the score is honest; the noise limits how "
                           "much `protein_g` can predict.")
    assert "held-out" not in item.reason


def test_under_predict_measured_noise_says_its_reliability_and_is_never_acted_on():
    """λ measured from two recalls a person, with 1 − λ past the band-2 cut-off (0.5): under
    Estimate it is acted on; under Predict it is still For the record, with λ as measured."""
    k = 2
    frame = _two_recalls(sd_between=4.0, sd_within=12.0, k=k)
    lam = _balanced_lambda(frame, k)
    # by construction σ²w / k = 72 against σ²b = 16, a population λ of 0.18; the drawn rows' λ is
    # the one computed by hand above
    assert 1 - lam > 0.5
    repeats = dict(grain={"grain": "repeated", "id_column": "participant_id"},
                   repeat_kind={"repeat_kind": "repeats"})
    for purpose, expected in (("inference", "act_on_it"), ("prediction", "no_change")):
        state = _state(purpose, **repeats)
        noticing = M.variance_noticing(state, frame)
        assert noticing.measure == pytest.approx(lam, abs=1e-12)
        item = _triaged(state, noticing)
        assert item.recommended == expected, purpose
    assert f"(its reliability λ = {lam:.2f})" in item.reason
    assert item.label == "For the record" and not item.limitation


def test_under_estimate_day_to_day_noise_still_could_bias_the_estimate():
    """The guard is Predict's alone: the same table and answers under Estimate (and with the goal
    unanswered, the strictest case) recommend as the materiality ledger does."""
    for purpose in ("inference", None):
        state = _state(purpose)
        noticing = M.variance_noticing(state, _one_recall())
        item = _triaged(state, noticing)
        disposition, reason, owed = M.recommend(noticing, purpose=purpose)
        assert (item.recommended, item.reason, item.limitation) == (disposition, reason, owed)
        assert (disposition, owed) == ("could_bias", True)
        assert item.label == "Could bias the estimate (not measurable here)"
        assert "a slope could be weakened" in item.reason
        assert item.band == 1 and item.measure == noticing.predicted.label


def test_under_predict_a_score_noticing_that_is_not_noise_still_could_bias_the_score():
    """The rule is the noise's alone: under Predict a screen whose leavers differ (rows × SMD)
    still could bias the score."""
    state = _state("prediction")
    screen = M.Noticing(thread="diet-implausible-reporters", family="S5", stage="results",
                        subject=["energy_kcal"], summary="Some energy reports are implausible",
                        decides_by="meaning", alternative="keeping every row",
                        predicted=M.movement("rows_smd", 0.05, "The screen moves rows."),
                        question="exclusions", answered=True)
    log = quest.quest_log(state, [], steps_until(None))
    item = sweep.triage(state, log, {"findings": []}, noticed=[screen]).items[0]
    assert (item.recommended, item.label) == ("could_bias", "Could bias the score")


# ── 3b · the design's concerns ───────────────────────────────────────────────

DESIGN_ROLES = {"carb_g": "exposure", "sugar_g": "exposure", "fiber_g": "exposure",
                "fat_a_g": "exposure", "fat_b_g": "exposure", "age": "covariate"}
NESTED = {"sugar_g": "carb_g"}  # constructed: sugar is part of carbohydrate
MEANING = (  # today's words under Estimate, each a concern about a coefficient or a substitution
    "fiber_g sits beside carb_g: if carbohydrate is by difference (total carbohydrate, as in USDA "
    "and NHANES tables), it already holds fiber, so fiber's energy counts twice in kcal accounting "
    "and a swap through fiber also moves carbohydrate.",
    "sugar_g is a part of carb_g: carb_g's coefficient is the remainder (holding sugar_g fixed), not "
    "total carb_g; leave it out to estimate the total. A substitution through carb_g moves it in "
    "proportion, and never pairs carb_g with its own part.",
    "fat_a_g, fat_b_g all read as fat, and none is part of another. If some overlap, moving energy "
    "through one while the rest stay fixed is not a coherent substitution.",
)


def _diet(shares: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 200
    carb = rng.gamma(9.0, 27.0, n)
    frame = pd.DataFrame({"carb_g": carb, "sugar_g": carb * 0.3, "fiber_g": rng.normal(18, 4, n),
                          "fat_a_g": rng.gamma(9, 4, n), "fat_b_g": rng.gamma(9, 4, n),
                          "age": rng.integers(20, 80, n).astype(float), "y": rng.normal(size=n)})
    if shares:  # three shares of energy that sum to 100% on every row
        p = rng.dirichlet([4, 5, 3], n) * 100
        frame = frame.assign(protein_pct_kcal=p[:, 0], fat_pct_kcal=p[:, 1],
                             carbohydrate_pct_kcal=p[:, 2])
    return frame


def _concerns(purpose: str | None, frame: pd.DataFrame, roles: dict[str, str]) -> list[str]:
    state = ProjectState(lens=["dietary"], target="y", roles=roles, purpose=purpose)
    spec = design_spec(state, frame, list(roles))
    return warnings_for(spec, frame, [], {}, nested=NESTED)


def test_under_predict_the_concerns_about_coefficients_and_substitutions_are_silent():
    said = _concerns("prediction", _diet(), DESIGN_ROLES)
    assert said == []
    for word in ("coefficient", "substitution", "swap", "remainder"):
        assert not any(word in w for w in said), word


@pytest.mark.parametrize("purpose", ["inference", None])
def test_under_estimate_the_same_design_says_each_concern_as_before(purpose):
    assert _concerns(purpose, _diet(), DESIGN_ROLES) == list(MEANING)


@pytest.mark.parametrize("purpose", ["prediction", "inference"])
def test_shares_that_sum_to_one_hundred_are_said_under_either_goal(purpose):
    """Not a concern about meaning: beside the intercept one share is fixed by the others, so a
    linear model cannot estimate them all, whatever the goal."""
    shares = ["protein_pct_kcal", "fat_pct_kcal", "carbohydrate_pct_kcal"]
    roles = {**DESIGN_ROLES, **{c: "exposure" for c in shares}}
    said = _concerns(purpose, _diet(shares=True), roles)
    assert any(w.startswith(", ".join(shares) + " sum to 100% on every row") for w in said), said


TOTAL_AS_COVARIATE = (  # audit ME-02's words under Estimate, for the constructed table below
    "energy_kcal reads as total energy and is in the model as a covariate, so each nutrient's "
    "coefficient is at fixed total energy (the standard model), not an absolute intake. Give "
    "energy_kcal the energy role to choose the energy model, or leave it out for the unadjusted "
    "one.")


def _design_warnings(tmp_path, purpose: str | None) -> list[str]:
    """The design stage's concerns on a table where total energy is a covariate beside four
    nutrients (W1-R R2's verifier, V2: the design concerns the shell draws)."""
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.tests import modeling_fixtures as mf

    rng = np.random.default_rng(11)
    n = 160
    kcal = rng.normal(2000, 400, n)
    frame = pd.DataFrame({"protein_g": 0.04 * kcal + rng.normal(0, 10, n),
                          "carb_g": 0.12 * kcal + rng.normal(0, 20, n),
                          "sugar_g": 0.05 * kcal + rng.normal(0, 8, n),
                          "fiber_g": rng.normal(18, 4, n), "energy_kcal": kcal,
                          "age": rng.uniform(20, 80, n), "y": rng.normal(size=n)})
    roles = {"protein_g": "exposure", "carb_g": "exposure", "sugar_g": "exposure",
             "fiber_g": "exposure", "energy_kcal": "covariate", "age": "covariate"}
    paths = mf.ingest_frame(frame, tmp_path / str(purpose))
    st = mf.state(roles=roles, target="y", task="regression", models=["linear"], purpose=purpose,
                  split=SplitSpec(holdout=0.0, seed=0, folds=5), shape_confirmations={})
    design = design_stage(mf.context(st, {"split": mf.split_bundle(np.arange(n), holdout=0.0),
                                          "target_info": mf.target_info("regression", "y")},
                                     paths))
    return list(design.data["warnings"])


def test_under_predict_the_design_says_nothing_of_total_energy_as_a_covariate(tmp_path):
    """Under Predict no coefficient is read and the energy model is never asked (BEATS_PREDICT §9
    ruling 5), so audit ME-02's concern, what each nutrient's coefficient means beside total energy
    and how to choose the energy model, is not said."""
    said = _design_warnings(tmp_path, "prediction")
    assert not [w for w in said if "reads as total energy" in w], said


@pytest.mark.parametrize("purpose", ["inference", None])
def test_under_estimate_the_design_says_total_energy_is_a_covariate_as_before(tmp_path, purpose):
    assert TOTAL_AS_COVARIATE in _design_warnings(tmp_path, purpose)


def test_the_design_and_fit_stages_compute_again_rather_than_serve_a_cache_from_before():
    """Ruling R-e: the design's concerns under Predict changed (3b), and the fit serves the
    calibration's support (finding 10), so neither stage's cached artifact from before is served:
    each stage's version is the one after the change. The design changed again (W1-T: no estimand
    and no coefficient's meaning under Predict), and so did the SAS-zero finding's summary (R-f
    14), which the findings stage computes. The fit changed again (C-110: its headline sentences and
    chain say what the drawn folds did to a unit's rows), and again under the large-table ruling
    (fewer comparison repeats from 5,000 training units); the design again when its impute line
    came to state the imputations the coefficient table draws, and again when it came to state
    them only under inference, where the fit draws the copies; the fit again when a table on no
    more rows than its coefficients came to be refused, and again when a table whose fit gives a
    coefficient no standard error came to be refused and the mixed model's boundary to be decided
    by the profile's slope, and when SMC-FCS came to draw its outcome model on the identified,
    unit-scaled design (the Sullivan 2021 crash sweep); and the design again when its feature-wise
    model step came to test an energy-adjusted study factor under the energy step's name and to
    refuse one in a declared form, which re-keys the fit and every stage after the design; and the
    fit again when feature-wise regression with no study factor came to refuse its table rather
    than the fit, and SMC-FCS's Cox refusal to name the reference stratum and offer only the ways
    forward that run (the swarm's tail, T2), with the effects stage, whose feature-wise tables came
    to refuse a formed study factor, and a category the estimand names, and whose family sentence
    came to count the tests feature-wise regression runs; and the fit again when a model's methods
    paragraph beside a refused feature-wise table came to say it was refused rather than that 0
    features were tested (T3); and the fit again when a feature-wise study factor with nothing left
    after the covariates came to be refused by row and the detection sentence beside a refused
    table to claim no interval, with the effects stage, whose family sentence came to count the
    tests its feature-wise table made and whose models came to name the category they adjust for
    (T4a), and the sensitivity and secondary stages, which refit the feature-wise table, and the
    proposals stage, whose estimand card came to say what feature-wise regression would test of a
    family (T4a's repair); and, where the tail tables and T4a both moved a stage, one
    past both at their merge; and the fit, effects, sensitivity, calibration, secondary and scales
    stages again when their tables came to be told the study factor's columns, so a dependency
    among the covariates is refused by row (with the effects stage's methods text naming each
    family wherever two report from different models), and the proposals stage when it came to
    hand the estimand card the table's columns (T5b)."""
    from turbotab.core.stages import build_graph

    versions = {s.name: s.version for s in build_graph().stages()}
    assert (versions["design"], versions["fit"], versions["findings"]) == (35, 38, 21)
    assert (versions["effects"], versions["sensitivity"], versions["calibration"],
            versions["secondary"], versions["scales"]) == (12, 21, 18, 12, 9)
    assert versions["proposals"] == 22


def _design_data(tmp_path, purpose: str | None) -> dict:
    """The design stage's artifact on :func:`_design_warnings`'s table: four nutrients beside total
    energy as a covariate, which under Estimate is the standard model, its coefficients each a
    substitution in place of the energy sources not in the model."""
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.tests import modeling_fixtures as mf

    rng = np.random.default_rng(11)
    n = 160
    kcal = rng.normal(2000, 400, n)
    frame = pd.DataFrame({"protein_g": 0.04 * kcal + rng.normal(0, 10, n),
                          "carb_g": 0.12 * kcal + rng.normal(0, 20, n),
                          "fat_g": 0.03 * kcal + rng.normal(0, 6, n),
                          "fiber_g": rng.normal(18, 4, n), "energy_kcal": kcal,
                          "age": rng.uniform(20, 80, n), "y": rng.normal(size=n)})
    roles = {"protein_g": "exposure", "carb_g": "exposure", "fat_g": "exposure",
             "fiber_g": "exposure", "energy_kcal": "covariate", "age": "covariate"}
    paths = mf.ingest_frame(frame, tmp_path / f"terms-{purpose}")
    st = mf.state(roles=roles, target="y", task="regression", models=["linear"], purpose=purpose,
                  split=SplitSpec(holdout=0.0, seed=0, folds=5), shape_confirmations={})
    design = design_stage(mf.context(st, {"split": mf.split_bundle(np.arange(n), holdout=0.0),
                                          "target_info": mf.target_info("regression", "y")},
                                     paths))
    return dict(design.data)


def test_under_predict_the_design_serves_no_estimand_and_no_coefficients_meaning(tmp_path):
    """Ruling 5 extended (W1-S se): under Predict no coefficient is read, so the design serves no
    estimand and no term's meaning, a substitution among them, and Results' saved provenance
    (``energyProvenance``) has none to carry."""
    data = _design_data(tmp_path, "prediction")
    assert data["estimand"] is None
    assert data["terms"] == {}


@pytest.mark.parametrize("purpose", ["inference", None])
def test_under_estimate_the_design_serves_its_estimand_and_each_terms_substitution(tmp_path,
                                                                                    purpose):
    data = _design_data(tmp_path, purpose)
    assert data["estimand"] and "substitution" in data["estimand"]
    assert set(data["terms"]) >= {"protein_g", "carb_g", "fat_g"}
    assert all("in place of" in data["terms"][c] for c in ("protein_g", "carb_g", "fat_g"))


def test_the_design_spec_carries_the_goal_its_concerns_read():
    frame = _diet()
    for purpose in ("prediction", "inference", None):
        state = ProjectState(lens=["dietary"], target="y", roles=DESIGN_ROLES, purpose=purpose)
        assert design_spec(state, frame, list(DESIGN_ROLES)).purpose == purpose


# ── 3c · the role words ──────────────────────────────────────────────────────


def _proposals() -> dict:
    """The roles stage's artifact: both proposed high, so the values settled them."""
    return {"columns": [
        {"column": "protein_g", "proposed": "exposure", "confidence": "high",
         "reason": "rises with total energy (r = 0.62)"},
        {"column": "age", "proposed": "covariate", "confidence": "high", "reason": "an age"},
    ]}


def test_under_predict_a_role_read_from_the_values_says_its_group():
    """W1-T's V1 (U8a): under Predict a role the values settled reads as the role lines say it, by
    the roles card's groups (the roles ruling of 2026-10-17: under the dietary lens a nutrient and a
    characteristic), and its presses are the role lines' own: a move between the groups, or out of
    the models. Written out here from the ruling's words."""
    state = _state("prediction", roles={"protein_g": "exposure", "age": "covariate"})
    items = {i["column"]: i for i in readings.read_from_data(state, roles=_proposals())}
    assert items["protein_g"]["words"] == "is a nutrient"
    assert items["age"]["words"] == "is a characteristic"
    assert [e["label"] for e in items["protein_g"]["change"]] == [
        "Move `protein_g` to the characteristics", "`protein_g` is left out of the models"]
    assert [e["label"] for e in items["age"]["change"]] == [
        "Move `age` to the nutrients", "`age` is left out of the models"]
    said = readings.read_from_values_sentence(list(items.values()))
    assert "`protein_g` is a nutrient (rises with total energy" in said
    assert not any(w in said for w in STUDY_WORDS)


def test_under_estimate_a_role_read_from_the_values_says_what_you_study():
    for purpose in ("inference", None):
        state = _state(purpose, roles={"protein_g": "exposure", "age": "covariate"})
        items = {i["column"]: i for i in readings.read_from_data(state, roles=_proposals())}
        assert items["protein_g"]["words"] == "is what you study"
        assert items["age"]["words"] == "is a covariate"
        assert [e["label"] for e in items["protein_g"]["change"]] == [
            "`protein_g` is a covariate", "`protein_g` is left out of the models"]
        assert [e["label"] for e in items["age"]["change"]] == [
            "`age` is what you study", "`age` is left out of the models"]


def _diet_table(tmp_path) -> tuple[pd.DataFrame, object]:
    """A dietary table built here: three macronutrients whose Atwater energy (FAO: 4/4/9 kcal/g)
    plus noise is the energy column, an age from 20 to 79 and one identifier a row."""
    rng = np.random.default_rng(11)
    n = 300
    p, c, f = rng.gamma(9, 9, n), rng.gamma(9, 30, n), rng.gamma(9, 8, n)
    frame = pd.DataFrame({"participant_id": np.arange(1, n + 1), "age": rng.integers(20, 80, n),
                          "protein_g": p.round(1), "carbohydrate_g": c.round(1),
                          "fat_g": f.round(1),
                          "energy_kcal": (4 * p + 4 * c + 9 * f + rng.normal(0, 40, n)).round(0),
                          "hba1c": rng.normal(5.6, 0.5, n).round(2)})
    path = tmp_path / "diet.csv"
    frame.to_csv(path, index=False)
    return frame, path


def _values_readings(tmp_path, purpose: str | None) -> tuple[pd.DataFrame, dict, dict]:
    """The roles stage's own proposals on :func:`_diet_table` under ``purpose``, recorded as
    proposed, and the readings the values settled from them: column → proposal, column → item."""
    frame, path = _diet_table(tmp_path)
    out = Ingested(path, tmp_path / "work").run(
        roles_stage, ProjectState(lens=["dietary"], target="hba1c", purpose=purpose))
    proposals = {p["column"]: p for p in out["columns"]}
    state = ProjectState(lens=["dietary"], target="hba1c", purpose=purpose,
                         roles={c: p["proposed"] for c, p in proposals.items()})
    items = {i["column"]: i for i in readings.read_from_data(state, roles=out)}
    return frame, proposals, items


def test_under_predict_the_values_readings_say_the_roles_stages_evidence_with_no_study_word(
        tmp_path):
    """W1-R finding 3c on the engine's own reasons: the roles stage says a nutrient is "what you
    study" and an age is "usually adjusted for rather than studied". Under Predict the reading
    keeps what the values showed (the r is numpy's, on the table built here) and leaves the study
    clause out."""
    frame, proposals, items = _values_readings(tmp_path, "prediction")
    assert {c: proposals[c]["confidence"] for c in ("age", "carbohydrate_g", "fat_g")} == \
        {"age": "high", "carbohydrate_g": "high", "fat_g": "high"}
    assert items["age"]["evidence"] == "A person's characteristic."
    for c in ("carbohydrate_g", "fat_g"):
        r = float(np.corrcoef(frame[c], frame["energy_kcal"])[0, 1])
        assert items[c]["evidence"] == (f"A nutrient that carries energy; it rises with total "
                                        f"energy (r = {r:.2f})."), c
    assert items["protein_g"]["evidence"].startswith("A nutrient that carries energy; its energy "
                                                     "and the others' add up to `energy_kcal`")
    assert items["age"]["words"] == "is a characteristic"
    for c in ("protein_g", "carbohydrate_g", "fat_g"):
        assert items[c]["words"] == "is a nutrient", c
    said = readings.read_from_values_sentence(list(items.values())).lower()
    for word in (*STUDY_WORDS, "studied", "adjusted for", "covariate"):
        assert word not in said, word


@pytest.mark.parametrize("purpose", ["inference", None])
def test_under_estimate_the_values_readings_say_the_roles_stages_evidence_as_it_is(tmp_path,
                                                                                   purpose):
    frame, proposals, items = _values_readings(tmp_path, purpose)
    for c in ("age", "protein_g", "carbohydrate_g", "fat_g", "energy_kcal", "participant_id"):
        assert items[c]["evidence"] == proposals[c]["reason"], c
    # today's words, as the roles stage says them
    assert items["age"]["evidence"] == "A person's characteristic, usually adjusted for rather " \
                                       "than studied."
    r = float(np.corrcoef(frame["fat_g"], frame["energy_kcal"])[0, 1])
    assert items["fat_g"]["evidence"] == (f"A nutrient that carries energy: what you study; it "
                                          f"rises with total energy (r = {r:.2f}).")


# The W1-U verifier's P4: the reasons the roles stage gives below high confidence, which a group's
# line, the fit's ask and the record quote (ruling R-g), say Estimate's study clause too. Under
# Predict each is quoted without it; the words are written out here.
BELOW_HIGH_UNDER_PREDICT = {
    "A measured feature: what you study under the omics lens.":
        "A measured feature under the omics lens.",
    "A measured column: read as what you study until you say otherwise.":
        "A measured column: read as a predictor until you say otherwise.",
    "An amount per day or week: a rate, read as what you study for now.":
        "An amount per day or week: a rate, read as a predictor for now.",
    "Fasting status: a known influence on other measures, adjusted for rather than studied.":
        "Fasting status: a known influence on other measures.",
    "A nutrient intake per kg of body weight: what you study, not a day's amount.":
        "A nutrient intake per kg of body weight, not a day's amount.",
    "An intake by its unit, not a nutrient: what you study under the dietary lens.":
        "An intake by its unit, not a nutrient, under the dietary lens.",
    # The ruling of 2026-10-18 (m3): the stage's two low-confidence covariate defaults read a
    # column "as a covariate"; under Predict a covariate is a predictor, as P4 quotes "what you
    # study"
    "Not a nutrient or feature: read as a covariate until you say otherwise.":
        "Not a nutrient or feature: read as a predictor until you say otherwise.",
    "A category that describes the row: read as a covariate for now.":
        "A category that describes the row: read as a predictor for now.",
}


def _below_high_reasons(tmp_path) -> dict[str, str]:
    """The roles stage's own reasons below high on three tables built here, by column: two
    measured features under the genomics lens; a score, a rate, a fasting flag and a yes/no
    category under the clinical lens; a nutrient per kg of body weight, a food in grams and a
    measure that is no nutrient under the dietary lens."""
    rng = np.random.default_rng(3)
    n = 120
    tables = {
        "genomics": {"gene_a": rng.gamma(4, 50, n).round(1), "gene_b": rng.gamma(4, 50, n).round(1)},
        "clinical": {"x_score": rng.normal(10, 2, n), "steps_per_day": rng.normal(7000, 900, n),
                     "fasting": rng.integers(0, 2, n), "rainy": rng.choice(["yes", "no"], n)},
        "dietary": {"fat_g_kg": rng.normal(1.2, 0.2, n), "fatty_fish_g": rng.gamma(2, 20, n),
                    "zq_value": rng.normal(10, 2, n)},
    }
    out: dict[str, str] = {}
    for lens, columns in tables.items():
        path = tmp_path / f"{lens}.csv"
        pd.DataFrame({**columns, "y": rng.normal(size=n)}).to_csv(path, index=False)
        found = Ingested(path, tmp_path / f"work_{lens}").run(
            roles_stage, ProjectState(lens=[lens], target="y", purpose="prediction"))
        out.update({p["column"]: p["reason"] for p in found["columns"]
                    if p["confidence"] != "high"})
    return out


def test_under_predict_each_reason_below_high_is_quoted_without_its_study_clause(tmp_path):
    """Each reason the roles stage gave on the tables built here is one this test words, so a
    reworded reason fails; under Predict ``readings.role_evidence`` quotes each without a study
    word, under Estimate as the stage said it."""
    reasons = _below_high_reasons(tmp_path)
    assert set(reasons.values()) == set(BELOW_HIGH_UNDER_PREDICT)
    for reason, said in BELOW_HIGH_UNDER_PREDICT.items():
        assert readings.role_evidence(reason, "prediction") == said
        for word in (*STUDY_WORDS, "studied", "adjusted for", "covariate"):
            assert word not in said.lower(), (word, said)
        for purpose in ("inference", None):
            assert readings.role_evidence(reason, purpose) == reason


def test_under_predict_a_left_out_column_offers_one_way_into_the_models():
    proposals = {"columns": [{"column": "note", "proposed": "excluded", "confidence": "high",
                              "reason": "free text"}]}
    labels = {p: [e["label"] for e in readings.read_from_data(
        _state(p, roles={"note": "excluded"}), roles=proposals)[0]["change"]]
        for p in ("prediction", "inference")}
    # under Predict each way in is a move into one of the roles card's groups (U8a)
    assert labels == {"prediction": ["Move `note` to the characteristics",
                                     "Move `note` to the nutrients"],
                      "inference": ["`note` is a covariate", "`note` is what you study"]}


@pytest.mark.parametrize("purpose, words", [("prediction", "a nutrient"),
                                            ("inference", "what you study"),
                                            (None, "what you study")])
def test_an_unsettled_roles_exits_and_ask_name_it_by_the_goals_word(purpose, words):
    """A role that rode along in a bulk confirm: its own confirmation, and the ask's line."""
    state = _state(purpose, roles={"protein_g": "exposure", "age": "covariate"},
                   roles_unconfirmed=["protein_g"])
    exits = readings.confirm_exits(state, ["protein_g"])
    assert exits[0]["label"] == f"Confirm `protein_g` as {words}"
    r = readings.role_reading(state, "protein_g")
    assert not r.settled
    assert readings.ask_text([r], state) == f"Tell me about this column: `protein_g`: {words}? (" \
                                            f"proposed below high confidence and recorded with " \
                                            f"the other roles)."
    assert readings.ask_exits([r], state)[0]["label"] == f"Confirm `protein_g` as {words}"
    assert readings.guess_words(r, state) == words


ASKED = {"protein_g": "exposure", "bp_sys": "covariate"}  # recorded below high, waiting


def _models_card(purpose: str | None):
    from turbotab.core import ask

    state = ProjectState(lens=["dietary"], target="y", task="regression", purpose=purpose,
                         roles={**ASKED, "age": "covariate", "energy_kcal": "energy",
                                "participant_id": "identifier"},
                         roles_unconfirmed=list(ASKED))
    return ask.card("models", ask.AskContext(state, {}))


@pytest.mark.parametrize("purpose, words", [
    ("prediction", {"protein_g": "a nutrient", "bp_sys": "a characteristic"}),
    ("inference", {"protein_g": "what you study", "bp_sys": "a covariate"}),
    (None, {"protein_g": "what you study", "bp_sys": "a covariate"})])
def test_the_models_ask_card_asks_and_answers_in_one_goals_words(purpose, words):
    """W1-R round 2 (R2's P4): the card's lines, its question and its buttons word each role the
    same way, the goal's (under Predict the roles card's group, as the role lines say it: U8a;
    under Estimate as before)."""
    card = _models_card(purpose)
    said = {g.columns[0]: g.guess_words for g in card.groups}
    assert said == words
    for column, word in words.items():
        assert f"`{column}`: {word}?" in card.text
        assert f"Confirm `{column}` as {word}" in [e.label for e in card.exits]


def test_the_quest_logs_ask_items_word_each_role_by_the_goal():
    from turbotab.core.tests.test_stage_registry import record

    for purpose, word in (("prediction", "a characteristic"), ("inference", "a covariate")):
        records = [record(1, {"kind": "set_lens", "lenses": ["dietary"]}),
                   record(2, {"kind": "set_target", "column": "y"}),
                   record(3, {"kind": "set_purpose", "purpose": purpose}),
                   record(4, {"kind": "set_roles", "roles": {"bp_sys": "covariate",
                                                             "age": "covariate"},
                              "unconfirmed": ["bp_sys"]})]
        from turbotab.core import decisions

        state = decisions.fold(records)
        log = quest.quest_log(state, records, steps_until(None), asked=[quest.fit_waits(state)])
        line = next(l for s in log.stages for l in s.lines if l.key == "ask:models")
        assert [(i.column, i.words) for i in line.items] == [("bp_sys", word)], purpose


def test_the_role_words_are_the_estimates_without_a_state():
    """A caller that passes no goal reads today's words: the strictest case, as before."""
    assert readings.role_words("exposure") == "what you study"
    assert readings.role_words("covariate") == "a covariate"
    for role in ("exposure", "covariate"):
        assert readings.role_words(role, "prediction") == "a predictor"
        assert readings.role_words(role, "inference") == readings.role_words(role)
    for role in ("energy", "identifier", "excluded", "flag", "time", "design", "cluster"):
        assert readings.role_words(role, "prediction") == readings.role_words(role)


def _roles_entry(purpose: str | None) -> dict:
    return next(e for e in entries_now(purpose) if e["key"] == "roles")


def test_under_predict_the_roles_card_names_no_study_factor():
    entry = _roles_entry("prediction")
    texts = [entry["one_liner"], entry["why"], entry["consumer"],
             *(f"{o['label']} {o['consequence']}" for o in entry["options"]),
             *(f"{t['term']} {t['definition']}" for t in entry["terms"])]
    for text in texts:
        assert not any(w in text.lower() for w in STUDY_WORDS), text
    # the roles ruling of 2026-10-17: every predictor role is a predictor; the one-liner says which
    # roles are predictor roles, never that they enter the models (the ruling at W1-X's cap)
    assert entry["one_liner"] == ONE_LINER
    options = {o["value"]: o for o in entry["options"]}
    assert options["exposure"]["label"] == "Predictor"
    # each role keeps a label of its own: the card's buttons are named by them
    assert len({o["label"] for o in entry["options"]}) == len(entry["options"])
    # the engine's roles stay as they are: the same values, in the same order
    assert [o["value"] for o in entry["options"]] == [o["value"] for o in _roles_entry(None)["options"]]
    # no energy model is asked under Predict, so the consumer line does not name one
    assert "energy adjustment" not in entry["consumer"]


# An example of each group the Predictor role is named by (``readings.predict_kind``): a nutrient
# under the dietary lens, an assay feature under an assay lens, a measure (such as a lab value)
# under any other; a project with both lenses names both.
GROUP_EXAMPLES = {"dietary": ("nutrient intake",), "assay": ("assay feature",),
                  "neither": ("lab value",), "both": ("nutrient intake", "assay feature")}
GROUP_LENSES = (("dietary",), ("clinical",), ("genomics",), ("metabolomics",),
                ("dietary", "metabolomics"), ("dietary", "clinical"), ())


@pytest.mark.parametrize("lenses", GROUP_LENSES)
def test_under_predict_the_predictor_options_examples_cover_the_projects_group(lenses):
    """The ruling at W1-V's cap: the Predictor role's option is served to every project under
    Predict, so its examples name each group the role is named by, whatever the lenses: a clinical
    project's measures among them (a lab value), not a nutrient intake and an assay feature alone."""
    entry = _roles_entry("prediction")
    consequence = next(o["consequence"] for o in entry["options"] if o["value"] == "exposure")
    assert set(GROUP_EXAMPLES) == {readings.predict_kind(k) for k in GROUP_LENSES}
    for example in GROUP_EXAMPLES[readings.predict_kind(lenses)]:
        assert example in consequence, (lenses, example, consequence)
    # under Estimate the option is as before
    estimate = next(o["consequence"] for o in _roles_entry("inference")["options"]
                    if o["value"] == "exposure")
    assert "lab value" not in estimate


def test_under_estimate_the_roles_card_is_as_before():
    from turbotab.core.teaching.content import ROLES as TODAY

    for purpose in ("inference", None):
        assert _roles_entry(purpose) == TODAY
    # the roles ruling at the landing's cap: the Estimate one-liner says which roles are the
    # predictor roles, never what reaches the models (tests/test_rf1_sentences.py)
    assert _roles_entry(None)["one_liner"].startswith("Study factors, covariates and the energy "
                                                      "column hold the predictor roles")


def test_the_dietary_lens_offers_the_energy_model_and_substitution_for_estimating_effects():
    """R-f 13: the lens is chosen before the goal, and neither the energy model nor the
    substitution is asked under Predict (BEATS_PREDICT §9 ruling 5), so the option says what they
    are for. The other lenses' options say what they did."""
    lens = next(e for e in entries_now() if e["key"] == "lens")
    options = {o["value"]: o["consequence"] for o in lens["options"]}
    assert options["dietary"] == ("Checks energy units and implausible intakes; for estimating "
                                  "effects, offers energy adjustment and substitution curves.")
    assert "effects" not in options["clinical"]


def test_under_predict_only_the_roles_card_changes():
    before = {e["key"]: e for e in entries_now()}
    after = {e["key"]: e for e in entries_now("prediction")}
    assert list(before) == list(after)
    assert [k for k in before if before[k] != after[k]] == ["roles"]


def test_the_predict_roles_card_fits_the_teaching_budgets():
    from turbotab.core import teaching, voice
    from turbotab.core.teaching import TeachingEntry

    entry = TeachingEntry.model_validate(_roles_entry("prediction"))
    B = teaching.BUDGETS
    for f in ("title", "question", "one_liner", "why", "consumer"):
        assert voice.words(getattr(entry, f)) <= B[f], f
    for o in entry.options:
        assert voice.words(o.label) <= B["option_label"] and \
            voice.words(o.consequence) <= B["option_consequence"], o.value
    for t in entry.terms:
        assert voice.words(t.definition) <= B["term_definition"], t.term
        assert t.definition.endswith(".")
    assert not voice.machinery(entry.one_liner)


# ── the predictor roles under Predict: their groups, no step named (the ruling of 2026-10-18) ──
#
# Every predictor role is a predictor, and the roles card names each of the three by its group in
# the project, by the lenses (nutrients, measures or assay features; characteristics; total energy),
# with its count. Four rounds named the steps that treat a group apart, and each produced a false
# sentence: the omics-only steps, "never normalized", "never screened out, assay-normalized or
# batch-corrected", and the energy-aware fill named under the complete-cases answer, where
# ``models.pipeline`` builds no fill. So no Predict card, option, term, served group or press names
# a step; the card says one sentence that holds in every project
# (``readings.predictors_sentence``): how many columns hold a predictor role, whichever group holds
# them. W1-V's sentence added that a group changes only how some preparation and screening steps
# treat a column, a claim about steps the ruling at W1-V's cap took out with its "only". W1-X's
# sentence said its columns enter the models, which a missing-values answer or a repair that marks
# a column unusable makes false (``decisions.left_out``), so the ruling at W1-X's cap (its P3) has
# the card say what the roles hold, never what the models get.

# The steps the rounds named, the QC chain's words, the energy model's two steps, which never run
# under Predict (the W1-V verifier's P2: the drawer named them), and W1-V's generic "preparation
# and screening steps": none is said under Predict. "Pooled" is matched as the QC chain's "pooled
# QC", since pooled cycles are a survey's, which no step on a predictor role reads.
STEP_NAMES = re.compile(r"energy-aware|\bscreen(ed|ing)?\b|normaliz|\bbatch|combat|\bqc\b|"
                        r"pooled[- ]qc|assay-process|\bfill|-filled|energy adjust|substitut|"
                        r"\bsteps?\b|preparation", re.I)
# What the four rounds said (the brief's words, the base's served words): the guard fires on each.
FOUR_ROUNDS = (
    "The omics screen, normalization and batch correction skip them.",
    "A covariate is never normalized.",
    "A covariate is never screened out, assay-normalized or batch-corrected.",
    "The energy-aware fill fills them without total energy.",
    "Total energy, a predictor: median-filled, fills nutrients; kept by the screen; "
    "assay-processed only by pooled-QC steps.",
)
SENTENCE = "12 columns hold a predictor role, whichever group holds them."
# W1-X's sentence, re-ruled at its cap: it says what the models get.
MODELS_SENTENCE = "All 12 enter the models as predictors, whichever group holds them."
# W1-V's sentence, re-ruled at its cap: the guard fires on it.
STEP_SENTENCE = ("All 12 enter the models as predictors; a column's group changes only how some "
                 "preparation and screening steps treat it.")
# The lenses the card's names follow (``readings.predict_kind``), a pooled-QC lens among them.
PREDICT_LENSES = (("dietary",), ("clinical",), ("genomics",), ("metabolomics",),
                  ("dietary", "metabolomics"), ("dietary", "clinical"))
ENERGY = {"kcal": "energy", "carb_g": "exposure", "age": "covariate"}
PLAIN = {"carb_g": "exposure", "age": "covariate"}


def _names_a_step(text: str) -> bool:
    return STEP_NAMES.search(text) is not None


def _served_words(entry: object) -> list[str]:
    """Every string a teaching entry serves, at any depth: its question, one-liner, why and
    consumer, each option and term, and each drawer section's heading, body and evidence."""
    if isinstance(entry, str):
        return [entry]
    if isinstance(entry, dict):
        return [w for v in entry.values() for w in _served_words(v)]
    if isinstance(entry, list):
        return [w for v in entry for w in _served_words(v)]
    return []


def _predict_presses(lenses: tuple[str, ...]) -> list[str]:
    """Every press and line word a role line, a group line and a values' reading say under Predict
    in a project with these lenses, for every role and every change the ledger offers."""
    from turbotab.core.decisions import Role

    every = typing.get_args(Role)
    out: list[str] = []
    for role in every:
        for other in ("", *quest.role_choices(role, "prediction", lenses)):
            out += quest.role_line_text("x", role, other, "prediction", lenses).values()
        out += [quest.role_group_reason(7, role, "prediction", lenses),
                readings.group_press_label(7, role, "prediction", lenses),
                readings.role_group_words(role, "prediction", lenses)]
    for roles in (ENERGY, {**ENERGY, "bp_sys": "covariate", "note": "excluded"}):
        proposals = {"columns": [{"column": c, "proposed": r, "confidence": "high",
                                  "reason": "read"} for c, r in roles.items()]}
        state = ProjectState(lens=list(lenses), target="y", purpose="prediction", roles=roles)
        for item in readings.read_from_data(state, roles=proposals):
            out += [item["words"], *(e["label"] for e in item["change"])]
    return out


def test_under_predict_no_roles_card_option_term_group_or_press_names_a_step():
    """The ruling of 2026-10-18: the Predict roles card's served words (options, terms, one-liner
    and consumer line), the predictor groups the server serves beside the proposals for every lens,
    task and set of roles, and every role press name no step (the fill, the screen, normalization,
    batch correction or the QC chain). The guard fires on each sentence the four rounds wrote."""
    from turbotab.core.decisions import Task

    for old in (*FOUR_ROUNDS, STEP_SENTENCE):
        assert _names_a_step(old), old
    assert not _names_a_step(SENTENCE)
    entry = _roles_entry("prediction")
    texts = [entry["one_liner"], entry["consumer"],
             *(f"{o['label']} {o['consequence']}" for o in entry["options"]),
             *(f"{t['term']} {t['definition']}" for t in entry["terms"])]
    for text in texts:
        assert not _names_a_step(text), text
    for lenses in PREDICT_LENSES:
        for task in (*typing.get_args(Task), None):
            for columns in ((), ("carb_g", "fat_g")):
                served = readings.served_groups(lenses, task, columns)
                for shelf in ("fill", "no_fill"):
                    for role, group in served[shelf].items():
                        assert group["note"] == "", (lenses, task, role, group["note"])
                        assert not _names_a_step(" ".join((group["one"], *group["head"]))), group
        for said in _predict_presses(lenses):
            assert not _names_a_step(said), (lenses, said)


def test_under_predict_the_whole_roles_entry_names_no_step_its_drawer_and_why_included():
    """The W1-V verifier's P2: the Predict roles entry inherited the Estimate drawer's "Units decide
    everything downstream", which says every implausibility screen, energy adjustment and
    substitution is a function of total energy; neither of the last two runs under Predict
    (BEATS_PREDICT §9 ruling 5), and the screen is a step on a role. The guard reads every string
    the served Predict entry holds, so it reaches the drawer and the why: it fires on that sentence
    in the Estimate entry, which keeps its words, and is silent on the whole Predict entry. Pooled
    survey cycles are no QC chain, so a section on them is not a step."""
    from turbotab.core.teaching.content import ROLES as TODAY

    estimate = _roles_entry("inference")
    assert estimate == TODAY
    units = next(s for s in estimate["drawer"]["sections"]
                 if s["heading"] == "Units decide everything downstream")
    assert _names_a_step(units["body"])
    assert any(_names_a_step(w) for w in _served_words(estimate))
    assert not _names_a_step("Pooled cycles")
    entry = _roles_entry("prediction")
    words = _served_words(entry)
    assert units["body"] not in words
    assert any(entry["drawer"]["sections"]) and entry["why"] in words
    for said in words:
        assert not _names_a_step(said), said
    for said in words:
        for step in ("energy adjustment", "substitution", "implausib"):
            assert step not in said.lower(), (step, said)


def test_under_predict_the_roles_card_says_one_sentence_for_every_project():
    """The card keeps its groups and its count and says one sentence that holds in every project,
    its count interpolated (``readings.predictors_sentence``); one column is said as one. The ruling
    at W1-V's cap: no "only" and no claim about steps; at W1-X's cap: the sentence says how many
    columns hold a predictor role, never that they enter the models."""
    assert readings.predictors_sentence(12) == SENTENCE
    assert readings.predictors_sentence(1_234) == SENTENCE.replace("12 columns", "1,234 columns")
    assert readings.predictors_sentence(1) == "One column holds a predictor role."
    assert readings.predictors_sentence("{n}") == SENTENCE.replace("12", "{n}")
    for n in (1, 12, "{n}"):
        said = readings.predictors_sentence(n)
        assert not _names_a_step(said) and not re.search(r"\bonly\b", said, re.I), said
        assert not SAYS_WHAT_THE_MODELS_GET.search(said), said
    assert re.search(r"\bonly\b", STEP_SENTENCE)


# What the models get, said of a role (the ruling at W1-X's cap, its P3): a role "enters the
# models", or a column "becomes a predictor". The card says neither, since what the models get is
# the predictor-role columns less those an answer left out (``models.pipeline.model_predictors``).
SAYS_WHAT_THE_MODELS_GET = re.compile(r"\benters? the models?\b|\bbecomes? (?:a )?predictors?\b",
                                      re.I)
# A project's recorded roles: four columns hold a predictor role, two hold none.
HELD = {"carb_g": "exposure", "kcal": "energy", "age": "covariate", "sbp": "covariate",
        "seqn": "identifier", "wtint2yr": "design"}
HOLD_A_PREDICTOR_ROLE = ["carb_g", "kcal", "age", "sbp"]
IMPOSSIBLE = "pack::clinical::impossible_vs_extreme"


def _left_out_by(how: str) -> ProjectState:
    """``HELD`` under Predict with `sbp` left out of the models by ``how``: the missing-values
    answer's drop, or a repair that marks it unusable for its impossible values; "no drop" leaves
    every column in."""
    from turbotab.core.decisions import FindingDisposition, MissingSpec

    state = ProjectState(target="glucose", purpose="prediction", roles=HELD,
                         missing=MissingSpec(strategy="complete_case"))
    if how == "the missing-values answer":
        return state.model_copy(update={"missing": MissingSpec(strategy="complete_case",
                                                               drop_columns=["sbp"])})
    if how == "an unusable repair":
        return state.model_copy(update={"findings": {IMPOSSIBLE: FindingDisposition(
            action="applied", option="unusable", params={"bands": {"sbp": [30.0, 300.0]}})}})
    return state


@pytest.mark.parametrize("how, gone", [("no drop", []), ("the missing-values answer", ["sbp"]),
                                       ("an unusable repair", ["sbp"])])
def test_under_predict_the_roles_card_says_what_the_roles_hold_not_what_the_models_get(how, gone):
    """The ruling at W1-X's cap (its P3). Four columns are built to hold a predictor role. With no
    drop the models get all four; the missing-values answer's drop, or a repair that marks `sbp`
    unusable, leaves it out, so the models get three while the four columns still hold their
    roles. The card's sentence is the same in all three, says the four that hold a role, and claims
    nothing about the models; W1-X's sentence ("All 4 enter the models") was true only with no
    drop. The guard on what the models get fires on W1-X's sentences and is silent on the card's."""
    from turbotab.core.decisions import left_out
    from turbotab.core.models.pipeline import model_predictors

    state = _left_out_by(how)
    assert left_out(state) == gone
    got = model_predictors(state)
    assert got == [c for c in HOLD_A_PREDICTOR_ROLE if c not in gone]
    held = [c for c, r in state.roles.items() if r in readings.PREDICTOR_ROLES]
    assert held == HOLD_A_PREDICTOR_ROLE
    said = readings.predictors_sentence(len(held))
    assert said == "4 columns hold a predictor role, whichever group holds them."
    assert not SAYS_WHAT_THE_MODELS_GET.search(said)
    # W1-X's sentence for these four: it said what the models get, which a drop makes false.
    before = MODELS_SENTENCE.replace("12", "4")
    assert SAYS_WHAT_THE_MODELS_GET.search(before)
    assert (len(got) == len(held)) is (not gone)
    for old in ("It enters the models as a predictor, whichever group holds it.",
                "`4` columns become predictors: 1 exposure, energy `kcal`, 2 covariates.",
                "Records every column's role. Only exposures, covariates and energy become "
                "predictors."):
        assert SAYS_WHAT_THE_MODELS_GET.search(old), old


# The Predict roles card's one-liner, under the ruling at W1-X's cap as its line is (the land-1
# verifier's P1): how many roles are predictor roles, never that they enter the models, since the
# missing-values answer can leave a role's only column out of them.
ONE_LINER = "Three roles are predictor roles; every other role stays out of the predictors."
# The one-liner of the roles ruling of 2026-10-17: true only while each of the three keeps a column
# in the models.
MODELS_ONE_LINER = "Three roles enter the models as predictors; every other role stays out of them."


@pytest.mark.parametrize("drop", [[], ["kcal"]], ids=["no drop", "energy's only column dropped"])
def test_under_predict_the_roles_one_liner_says_which_roles_are_predictor_roles(drop):
    """The land-1 verifier's P1, under the ruling at W1-X's cap. ``HELD`` is built with a column in
    each of the three predictor roles, energy's only one `kcal`. With no drop the models get a
    column of each of the three; the missing-values answer that drops `kcal` leaves energy none, so
    two roles reach the models while three are still predictor roles. The card's one-liner is the
    same in both and true in both; the 2026-10-17 one-liner was true only with no drop, and the
    guard on what the models get fires on it and is silent on the card's. With or without the
    drop, every column the models get holds one of the three."""
    from turbotab.core.decisions import MissingSpec, left_out
    from turbotab.core.models.pipeline import model_predictors

    state = ProjectState(target="glucose", purpose="prediction", roles=HELD,
                         missing=MissingSpec(strategy="complete_case", drop_columns=drop))
    assert left_out(state) == drop
    reaching = {HELD[c] for c in model_predictors(state)}
    built = {"exposure", "energy", "covariate"}
    assert reaching == (built - {"energy"} if drop else built)
    assert set(readings.PREDICTOR_ROLES) == built
    one_liner = _roles_entry("prediction")["one_liner"]
    assert one_liner == ONE_LINER
    assert not SAYS_WHAT_THE_MODELS_GET.search(one_liner)
    assert SAYS_WHAT_THE_MODELS_GET.search(MODELS_ONE_LINER)
    assert (len(reaching) == len(built)) is (not drop)


@pytest.mark.parametrize("lenses, names", [
    (["dietary", "clinical"], ("a nutrient", "Nutrients")),
    (["genomics"], ("an assay feature", "Assay features")),
    (["metabolomics"], ("an assay feature", "Assay features")),
    (["dietary", "metabolomics"], ("a nutrient or assay feature", "Nutrients and assay features")),
    (["clinical"], ("a measure", "Measures")),
    (["dietary"], ("a nutrient", "Nutrients")),
])
def test_under_predict_each_group_is_named_by_the_lenses_alone(lenses, names):
    """The groups' names follow the lenses alone: the task and the roles change none of them."""
    from turbotab.core.decisions import Task

    groups = readings.predictor_groups(lenses)
    assert (groups["exposure"].one, groups["exposure"].head[1]) == names
    assert (groups["covariate"].one, groups["covariate"].head) == (
        "a characteristic", ("Characteristic", "Characteristics"))
    assert (groups["energy"].one, groups["energy"].head) == ("total energy",
                                                             ("Total energy", "Total energy"))
    drawn = {r: {"one": g.one, "head": list(g.head), "note": ""} for r, g in groups.items()}
    for task in (*typing.get_args(Task), None):
        for columns in ((), ("carb_g",)):
            served = readings.served_groups(lenses, task, columns)
            assert served["fill"] == served["no_fill"] == drawn, (task, columns)
            assert served["energy_bearing"] == list(columns)


def test_under_estimate_no_group_name_reaches_a_role_line():
    for purpose in ("inference", None):
        for lenses in (["dietary"], ["genomics"], ["clinical"]):
            for role in readings.PREDICTOR_ROLES:
                assert readings.role_line_words(role, purpose, lenses) == \
                    readings.role_words(role, purpose)


def test_the_energy_aware_fill_reads_the_predictor_role_apart_under_a_single_fill():
    """A preparation step that reads a group apart, read from the code: under a single fill a
    Predictor nutrient that carries energy is filled from its least-squares line on the energy
    column, and a nutrient given the covariate role is filled by its median. With total energy given
    another predictor role, no energy-aware fill runs. The values are constructed: carbohydrate and
    fat are exact lines on energy, so the line gives 0.1 and 0.04 of 2,800."""
    from turbotab.core.methods.missing import EnergyAwareImputer, energy_fill

    roles = {"carb_g": "exposure", "fat_g": "covariate", "age": "covariate", "kcal": "energy"}
    predictors = list(roles)
    fill = energy_fill(None, predictors, roles, predictors)
    assert fill == {"energy": "kcal", "nutrients": ["carb_g"]}
    assert energy_fill(None, predictors, {**roles, "fat_g": "exposure"}, predictors) == {
        "energy": "kcal", "nutrients": ["carb_g", "fat_g"]}
    assert energy_fill(None, predictors, {**roles, "kcal": "covariate"}, predictors) is None
    kcal = np.array([1000.0, 1200.0, 1400.0, 2800.0, 3000.0])
    frame = pd.DataFrame({"carb_g": 0.1 * kcal, "fat_g": 0.04 * kcal,
                          "age": [30.0, 40.0, 50.0, 60.0, 70.0], "kcal": kcal})
    frame.loc[3, ["carb_g", "fat_g"]] = np.nan
    out = np.asarray(EnergyAwareImputer(energy="kcal", nutrients=fill["nutrients"])
                     .fit(frame).transform(frame), dtype=float)
    assert out[3, 0] == pytest.approx(280.0)  # carb_g: its line on kcal, 0.1 × 2,800
    assert out[3, 1] == pytest.approx(float(np.median([40.0, 48.0, 56.0, 120.0])))  # fat_g: median
    as_predictor = np.asarray(EnergyAwareImputer(energy="kcal", nutrients=["carb_g", "fat_g"])
                              .fit(frame).transform(frame), dtype=float)
    assert as_predictor[3, 1] == pytest.approx(112.0)  # fat_g as a Predictor: 0.04 × 2,800


def test_normalization_and_batch_correction_act_on_predictor_columns_alone():
    """Two more preparation steps that read a group apart, read from the code on one constructed
    state: the omics normalization an applied ``omics_scale`` repair records and reference ComBat
    each take the Predictor columns (role ``exposure``) and skip a covariate and the energy
    column."""
    from turbotab.core.decisions import BatchSpec, FindingDisposition
    from turbotab.core.methods.batch import design_batch
    from turbotab.core.methods.omics import FINDING, design_normalization

    roles = {"g1": "exposure", "g2": "exposure", "age": "covariate", "kcal": "energy",
             "plate": "design"}
    predictors = ["g1", "g2", "age", "kcal"]
    scale = {FINDING: FindingDisposition(action="applied", option="log_cpm",
                                         params={"kind": "counts", "columns": predictors})}
    state = ProjectState(lens=["genomics"], target="y", purpose="prediction", roles=roles,
                         findings=scale, batch=BatchSpec(column="plate", method="reference_combat"))
    inputs = [*predictors, "plate"]
    assert design_normalization(state, inputs)["columns"] == ["g1", "g2"]
    assert design_batch(state, inputs, predictors, None)["columns"] == ["g1", "g2"]


@pytest.mark.parametrize("roles, runs", [
    # a clinical project with kcal as total energy and no predictor that carries energy
    ({"kcal": "energy", "bp_sys": "exposure", "age": "covariate"}, False),
    # the same beside a nutrient that does
    ({"kcal": "energy", "fat_g": "exposure", "age": "covariate"}, True)])
def test_the_energy_columns_move_names_no_step_whether_or_not_the_fill_would_run(roles, runs):
    """The energy column's move named the energy-aware fill where it would run (W1-T's V3), though
    under the complete-cases answer no fill is built. Under the ruling of 2026-10-18 its role line
    and its values' reading say the move alone, both where ``methods.missing.energy_fill`` would
    run and where it would not (read from the code here)."""
    from turbotab.core import decisions
    from turbotab.core.methods.missing import energy_fill
    from turbotab.core.tests.test_stage_registry import record

    predictors = list(roles)
    assert (energy_fill(None, predictors, roles, predictors) is not None) == runs
    move = "Move `kcal` to the characteristics"
    records = [record(1, {"kind": "set_lens", "lenses": ["clinical"]}),
               record(2, {"kind": "set_target", "column": "y"}),
               record(3, {"kind": "set_purpose", "purpose": "prediction"}),
               record(4, {"kind": "set_roles", "roles": roles, "unconfirmed": ["kcal"]})]
    log = quest.quest_log(decisions.fold(records), records, steps_until("exclusions"))
    [line] = [l for s in log.stages for l in s.lines if l.key == quest.role_line_key("kcal")]
    assert [o.label for o in line.items[0].options][1] == move
    proposals = {"columns": [{"column": c, "proposed": r, "confidence": "high", "reason": "read"}
                             for c, r in roles.items()]}
    state = ProjectState(lens=["clinical"], target="y", purpose="prediction", roles=roles)
    kcal = next(i for i in readings.read_from_data(state, roles=proposals) if i["column"] == "kcal")
    assert kcal["words"] == "is total energy"
    assert [e["label"] for e in kcal["change"]][0] == move


def test_under_estimate_the_energy_columns_values_reading_is_as_before():
    roles = {"kcal": "energy", "fat_g": "exposure", "age": "covariate"}
    proposals = {"columns": [{"column": c, "proposed": r, "confidence": "high", "reason": "read"}
                             for c, r in roles.items()]}
    for purpose in ("inference", None):
        state = ProjectState(lens=["clinical"], target="y", purpose=purpose, roles=roles)
        kcal = next(i for i in readings.read_from_data(state, roles=proposals)
                    if i["column"] == "kcal")
        assert kcal["words"] == "is total energy intake"
        assert [e["label"] for e in kcal["change"]] == ["`kcal` is a covariate",
                                                        "`kcal` is left out of the models"]


# ── W1-T's V2 (U8b): a role line's press is recorded as the press says it ────


@pytest.mark.parametrize("column, role, said", [
    ("bp_sys", "covariate", "`bp_sys` was confirmed as a characteristic"),
    ("bp_sys", "exposure", "`bp_sys` moved from the Characteristic role to the Nutrient role"),
    ("kcal", "covariate", "`kcal` left the energy role for the Characteristic role"),
    ("cycle", "covariate", "`cycle` moved to the Characteristic role"),
    ("bp_sys", "excluded", "`bp_sys` was recorded as left out of the models")])
def test_under_predict_a_role_lines_press_is_recorded_as_its_label_says_it(column, role, said):
    """The press "Move `bp_sys` to the nutrients" was recorded as "`bp_sys` was confirmed as a
    predictor on its own", the words of confirming the proposal. Under Predict the record names the
    roles card's groups (dietary lens: nutrients and characteristics): a confirmation as the group,
    a move between groups as a move, the energy column's as leaving the energy role."""
    from turbotab.core import voice

    state = ProjectState(lens=["dietary"], target="y", purpose="prediction",
                         roles={"bp_sys": "covariate", "kcal": "energy", "fat_g": "exposure",
                                "cycle": "excluded"},
                         roles_unconfirmed=["bp_sys", "kcal", "cycle"])
    sentence = voice.sentence_for({"kind": "confirm_role", "column": column, "role": role}, state)
    assert sentence == f"{said} on its own, after the evidence for its proposal was read."


def test_under_estimate_a_role_lines_press_is_recorded_as_before():
    from turbotab.core import voice

    state = ProjectState(lens=["dietary"], target="y", purpose="inference",
                         roles={"bp_sys": "covariate"}, roles_unconfirmed=["bp_sys"])
    for role, words in (("covariate", "a covariate"), ("exposure", "an exposure")):
        assert voice.sentence_for({"kind": "confirm_role", "column": "bp_sys", "role": role},
                                  state) == (f"`bp_sys` was confirmed as {words} on its own, after "
                                             f"the evidence for its proposal was read.")


# ── the W1-U verifier's P3 (U8a): a values' reading's press is recorded as it says it ──

VALUE_ROLES = {"kcal": "energy", "fat_g": "exposure", "age": "covariate"}


def _value_presses(purpose: str | None) -> tuple[ProjectState, dict[tuple[str, str], dict]]:
    """The presses the readings the values settled offer (``readings.read_from_data``, each a
    ``confirm_reading`` of a role), by column and role, on roles every one of which was proposed
    high: total energy, a nutrient that carries energy and an age, under the dietary lens."""
    proposals = {"columns": [{"column": c, "proposed": r, "confidence": "high", "reason": "read"}
                             for c, r in VALUE_ROLES.items()]}
    state = ProjectState(lens=["dietary"], target="y", purpose=purpose, roles=dict(VALUE_ROLES))
    presses = {(e["decision"]["column"], e["decision"]["value"]): e
               for i in readings.read_from_data(state, roles=proposals) for e in i["change"]}
    return state, presses


@pytest.mark.parametrize("column, role, label, said", [
    ("age", "exposure", "Move `age` to the nutrients",
     "`age` moved from the Characteristic role to the Nutrient role"),
    ("fat_g", "covariate", "Move `fat_g` to the characteristics",
     "`fat_g` moved from the Nutrient role to the Characteristic role"),
    ("kcal", "covariate", "Move `kcal` to the characteristics",
     "`kcal` left the energy role for the Characteristic role"),
    ("age", "excluded", "`age` is left out of the models",
     "`age` was recorded as left out of the models")])
def test_under_predict_a_values_readings_press_is_recorded_as_its_label_says_it(column, role,
                                                                               label, said):
    """The verifier's probe: the press "Move `age` to the nutrients" on a reading the values
    settled was recorded as "`age` was confirmed as a nutrient", and the energy column's press as a
    confirmation as a characteristic, never as leaving the energy role. Under Predict the record
    says the press as the label says it, as a role line's press is said (the test above): a move
    between the roles card's groups, the energy column's as leaving the energy role."""
    from turbotab.core import voice

    state, presses = _value_presses("prediction")
    press = presses[(column, role)]
    assert press["label"] == label
    assert press["decision"]["kind"] == "confirm_reading"
    assert voice.sentence_for(press["decision"], state) == (
        f"{said}, on its own, after the evidence for its reading was read.")


def test_silent_under_predict_a_values_reading_confirmed_as_recorded_is_a_confirmation():
    """The other way: a confirmation of the role the answer recorded is still said as one, in the
    roles card's group's words."""
    from turbotab.core import voice

    state, _presses = _value_presses("prediction")
    for column, words in (("age", "a characteristic"), ("fat_g", "a nutrient"),
                          ("kcal", "total energy")):
        decision = {"kind": "confirm_reading", "reading": "role", "column": column,
                    "value": VALUE_ROLES[column]}
        assert voice.sentence_for(decision, state) == (
            f"`{column}` was confirmed as {words}, on its own, after the evidence for its reading "
            f"was read.")


@pytest.mark.parametrize("purpose", ["inference", None])
def test_under_estimate_a_values_readings_press_is_recorded_as_before(purpose):
    """Under Estimate a values' reading's press names what the column is ("`age` is what you
    study"), and the record says it so, as before."""
    from turbotab.core import voice

    state, presses = _value_presses(purpose)
    for (column, role), words in ((("age", "exposure"), "an exposure"),
                                  (("kcal", "covariate"), "a covariate"),
                                  (("fat_g", "covariate"), "a covariate")):
        assert voice.sentence_for(presses[(column, role)]["decision"], state) == (
            f"`{column}` was confirmed as {words}, on its own, after the evidence for its reading "
            f"was read.")


# ── U7: the explanations' default curves under Predict ────────────────────────


def _default_curves(purpose: str | None) -> tuple[list[str], list[str]]:
    """A linear model on four predictors, one per predictor role and two covariates, ``age``
    weighing most: the default curves' inputs under ``purpose`` and the order numpy gives the
    candidates by mean |SHAP| (a linear model's is |β| times the mean absolute deviation)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    from turbotab.core.models import explain as E
    from turbotab.core.stages.explain import curve_candidates

    rng = np.random.default_rng(31)
    n = 300
    X = pd.DataFrame({"protein": rng.normal(70, 15, n), "age": rng.uniform(20, 80, n),
                      "kcal": rng.normal(2000, 400, n), "bmi": rng.normal(27, 4, n)})
    y = (0.05 * X["protein"] + 2.0 * X["age"] + 0.01 * X["kcal"] + 0.3 * X["bmi"]
         + rng.normal(0, 5, n)).to_numpy()
    roles = {"protein": "exposure", "age": "covariate", "kcal": "energy", "bmi": "covariate",
             "sid": "identifier"}
    candidates = curve_candidates(list(X.columns), roles, purpose)
    linear = Pipeline([("model", LinearRegression())]).fit(X, y)
    fam = E.FamilyFit(key="linear", label="Linear model", fitted=linear, unfitted=linear,
                      versus={"verdict": "better"}, score=25.0, baseline=900.0)
    art = E.explain([fam], E.Setting(task="regression", purpose=purpose, target="y", event=None,
                                     X=X, y=y, reseeds=0, candidates=candidates,
                                     metric_label="MSE", baseline_label="the outcome's average"),
                    lambda *a: None)
    beta = dict(zip(X.columns, linear[-1].coef_))
    weight = {c: abs(beta[c]) * float(np.mean(np.abs(X[c] - X[c].mean()))) for c in candidates}
    return [c.input for c in art.curves], sorted(candidates, key=lambda c: -weight[c])


def test_under_predict_the_default_curves_choose_among_every_predictor_role_column():
    from turbotab.core.models import explain as E
    from turbotab.core.stages.explain import curve_candidates

    roles = {"protein": "exposure", "age": "covariate", "kcal": "energy", "sid": "identifier",
             "imputed_bmi": "flag"}
    assert curve_candidates(["protein", "age", "kcal"], roles, "prediction") == [
        "protein", "age", "kcal"]
    shown, want = _default_curves("prediction")
    # the same rule as among the Predictor columns (the best family's importance), the cap kept
    assert shown == want[:E.TOP_EXPOSURES] and len(want) > E.TOP_EXPOSURES
    assert shown[0] == "age"


@pytest.mark.parametrize("purpose", ["inference", None])
def test_under_estimate_the_default_curves_choose_among_the_predictor_columns_as_before(purpose):
    from turbotab.core.stages.explain import curve_candidates

    roles = {"protein": "exposure", "age": "covariate", "kcal": "energy"}
    assert curve_candidates(["protein", "age", "kcal"], roles, purpose) == ["protein"]
    if purpose is None:
        shown, want = _default_curves(purpose)
        assert shown == want == ["protein"]
