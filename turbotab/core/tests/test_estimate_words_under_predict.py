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
    14), which the findings stage computes."""
    from turbotab.core.stages import build_graph

    versions = {s.name: s.version for s in build_graph().stages()}
    assert (versions["design"], versions["fit"], versions["findings"]) == (32, 29, 21)


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


def test_under_predict_a_role_read_from_the_values_says_predictor():
    """An exposure and a covariate are both predictors under Predict (BEATS_PREDICT C1: a
    predictor is the engine's covariate), so neither offers the other: the change would read
    "is a predictor" beside "is a predictor"."""
    state = _state("prediction", roles={"protein_g": "exposure", "age": "covariate"})
    items = {i["column"]: i for i in readings.read_from_data(state, roles=_proposals())}
    assert items["protein_g"]["words"] == items["age"]["words"] == "is a predictor"
    assert [e["label"] for e in items["protein_g"]["change"]] == [
        "`protein_g` is left out of the models"]
    assert [e["label"] for e in items["age"]["change"]] == ["`age` is left out of the models"]
    said = readings.read_from_values_sentence(list(items.values()))
    assert "`protein_g` is a predictor (rises with total energy" in said
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
    for c in ("age", "protein_g", "carbohydrate_g", "fat_g"):
        assert items[c]["words"] == "is a predictor", c
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


def test_under_predict_a_left_out_column_offers_one_way_into_the_models():
    proposals = {"columns": [{"column": "note", "proposed": "excluded", "confidence": "high",
                              "reason": "free text"}]}
    labels = {p: [e["label"] for e in readings.read_from_data(
        _state(p, roles={"note": "excluded"}), roles=proposals)[0]["change"]]
        for p in ("prediction", "inference")}
    assert labels == {"prediction": ["`note` is a predictor"],
                      "inference": ["`note` is a covariate", "`note` is what you study"]}


@pytest.mark.parametrize("purpose, words", [("prediction", "a predictor"),
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
    ("prediction", {"protein_g": "a predictor", "bp_sys": "a predictor"}),
    ("inference", {"protein_g": "what you study", "bp_sys": "a covariate"}),
    (None, {"protein_g": "what you study", "bp_sys": "a covariate"})])
def test_the_models_ask_card_asks_and_answers_in_one_goals_words(purpose, words):
    """W1-R round 2 (R2's P4): the card's lines, its question and its buttons word each role the
    same way, the goal's (under Predict a predictor, under Estimate as before)."""
    card = _models_card(purpose)
    said = {g.columns[0]: g.guess_words for g in card.groups}
    assert said == words
    for column, word in words.items():
        assert f"`{column}`: {word}?" in card.text
        assert f"Confirm `{column}` as {word}" in [e.label for e in card.exits]


def test_the_quest_logs_ask_items_word_each_role_by_the_goal():
    from turbotab.core.tests.test_stage_registry import record

    for purpose, word in (("prediction", "a predictor"), ("inference", "a covariate")):
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
    # the roles ruling of 2026-10-17: every predictor role is a predictor
    assert entry["one_liner"] == ("Three roles enter the models as predictors; every other role "
                                  "stays out of them.")
    options = {o["value"]: o for o in entry["options"]}
    assert options["exposure"]["label"] == "Predictor"
    # each role keeps a label of its own: the card's buttons are named by them
    assert len({o["label"] for o in entry["options"]}) == len(entry["options"])
    # the engine's roles stay as they are: the same values, in the same order
    assert [o["value"] for o in entry["options"]] == [o["value"] for o in _roles_entry(None)["options"]]
    # no energy model is asked under Predict, so the consumer line does not name one
    assert "energy adjustment" not in entry["consumer"]


def test_under_estimate_the_roles_card_is_as_before():
    from turbotab.core.teaching.content import ROLES as TODAY

    for purpose in ("inference", None):
        assert _roles_entry(purpose) == TODAY
    assert _roles_entry(None)["one_liner"].startswith("Only study factors, covariates and the energy")


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


# ── the predictor roles under Predict, named by what differs (the roles ruling of 2026-10-17) ──
#
# Every predictor role is a predictor, and the roles card names each of the three by the steps that
# treat it differently in the project (``readings.predictor_groups``), each only where the project
# can run it: the energy-aware fill wherever a column holds the energy role, under any lens; the
# screened elastic net's screen where that family is on the shelf of the project's task (a
# regression or a yes/no outcome's); and normalization and batch correction under an assay lens.
# The names follow the lenses. The words are the ruling's; each step they name is read from the
# code below.

STEPS = (readings.FILL, readings.SCREEN, readings.NORMALIZATION, readings.BATCH)
FILL, SCREEN, NORM, BATCH = STEPS
ENERGY = {"kcal": "energy", "carb_g": "exposure", "age": "covariate"}
PLAIN = {"carb_g": "exposure", "age": "covariate"}
# The Predict shelf by task (``models.base.rank``): the screen correlates with a number or a 0/1
# event, so the screened elastic net models a regression and a yes/no outcome alone.
SHELF_HAS_SCREEN = {"regression": True, "binary": True, "multiclass": False, "ordinal": False,
                    "time_to_event": False}


def test_under_predict_the_served_roles_card_names_every_step_that_reads_the_predictor_role():
    """The served card is the same in every project, so each predictor option names all four steps
    (W1-S se: the energy-aware fill was left out), within the teaching budgets (above)."""
    entry = _roles_entry("prediction")
    options = {o["value"]: o["consequence"] for o in entry["options"]}
    terms = {t["term"]: t["definition"] for t in entry["terms"]}
    steps = "the energy-aware fill, screen, normalization and batch correction"
    assert options["exposure"] == f"A predictor that {steps} act on."
    assert options["covariate"] == f"A predictor such as age, which {steps} skip."
    assert options["energy"] == ("Total energy, a predictor the energy-aware fill reads; screen, "
                                 "normalization and batch correction skip it.")
    assert terms["covariate"].endswith(f"; {steps} skip it.")
    for text in (*options.values(), *terms.values()):
        assert "omics screen" not in text and "like any other" not in text, text


@pytest.mark.parametrize("lenses, task, roles, names, steps", [
    (["dietary", "clinical"], "regression", ENERGY, ("a nutrient", "Nutrients"), (FILL, SCREEN)),
    (["genomics"], "binary", PLAIN, ("an assay feature", "Assay features"), (SCREEN, NORM, BATCH)),
    (["dietary", "metabolomics"], "regression", ENERGY,
     ("a nutrient or assay feature", "Nutrients and assay features"), (FILL, SCREEN, NORM, BATCH)),
    (["clinical"], "regression", PLAIN, ("a measure", "Measures"), (SCREEN,)),
    # the W1-T verifier's P2: total energy beside a clinical lens runs the energy-aware fill, and a
    # dietary project with no column in the energy role has no such fill to name
    (["clinical"], "binary", ENERGY, ("a measure", "Measures"), (FILL, SCREEN)),
    (["dietary"], "regression", PLAIN, ("a nutrient", "Nutrients"), (SCREEN,)),
    # its P1: a multiclass, ordinal or time-to-event shelf holds no screened elastic net
    (["clinical"], "multiclass", PLAIN, ("a measure", "Measures"), ()),
    (["dietary"], "ordinal", ENERGY, ("a nutrient", "Nutrients"), (FILL,)),
    (["genomics"], "time_to_event", PLAIN, ("an assay feature", "Assay features"), (NORM, BATCH)),
])
def test_under_predict_each_group_is_named_by_the_steps_that_treat_it_apart(lenses, task, roles,
                                                                            names, steps):
    groups = readings.predictor_groups(lenses, task, roles)
    assert readings.predict_steps(lenses, task, roles) == steps
    assert (groups["exposure"].one, groups["exposure"].head[1]) == names
    assert (groups["covariate"].one, groups["covariate"].head) == (
        "a characteristic", ("Characteristic", "Characteristics"))
    assert (groups["energy"].one, groups["energy"].head) == ("total energy",
                                                             ("Total energy", "Total energy"))
    for group in groups.values():
        # each named step fires where the project has it, and is silent where it has not
        for step in STEPS:
            assert (step in group.note.lower()) == (step in steps), (lenses, group.role, step)
        # a project that runs none of them says nothing that sets the group apart
        assert bool(group.note) == bool(steps), (lenses, task, group.role)
    if FILL in steps:
        assert "fills a missing amount of one that carries energy from total energy" in \
            groups["exposure"].note
        assert f"fills the {names[1].lower()} that carry energy from it" in groups["energy"].note
        assert "fills them without total energy" in groups["covariate"].note
    if SCREEN in steps:
        assert "may leave any of them out" in groups["exposure"].note
        assert "keeps every one" in groups["covariate"].note and "keeps it" in groups["energy"].note


@pytest.mark.parametrize("task", sorted(SHELF_HAS_SCREEN))
def test_the_screen_is_named_only_where_the_screened_elastic_net_is_on_the_predict_shelf(task):
    """The W1-T verifier's P1: the card named the screen in every project, though a multiclass,
    ordinal or time-to-event shelf holds no screened elastic net. The shelf is read from
    ``models.base.rank`` under each lens (it is never shortened), and the screen is named where it
    holds that family and nowhere else."""
    import turbotab.core.methods.omics  # noqa: F401 - registers the screened family
    from turbotab.core.models.base import Situation, rank

    on = SHELF_HAS_SCREEN[task]
    for lenses in (("dietary",), ("clinical",), ("genomics",)):
        shelf = [f.key for f, _ in rank(Situation(task=task, purpose="prediction", n_rows=200,
                                                  n_features=12, lenses=lenses))]
        assert ("screened_elastic_net" in shelf) == on, (task, lenses)
        assert readings.screen_on_shelf(task) == on
        groups = readings.predictor_groups(lenses, task, PLAIN)
        for role, group in groups.items():
            assert (readings.SCREEN in group.note.lower()) == on, (task, lenses, role, group.note)


def test_no_screen_is_named_before_the_task_is_read():
    assert not readings.screen_on_shelf(None)
    assert readings.SCREEN not in readings.predict_steps(["clinical"], None, PLAIN)


def test_the_energy_aware_fill_is_named_wherever_a_column_holds_the_energy_role(tmp_path):
    """The W1-T verifier's P2: the fill reads no lens, and the roles stage proposes total energy
    under any lens, so a clinical project whose table holds total energy runs the energy-aware fill
    while its card named only the screen. On the table built here (energy the macronutrients'
    Atwater sum, 4/4/9 kcal/g) the roles stage under the clinical lens proposes ``energy_kcal`` as
    total energy and the macronutrients as Predictor columns; with those roles the fill runs and the
    card names it in each group, and with total energy given another role neither holds."""
    from turbotab.core.methods.missing import energy_fill

    _frame, path = _diet_table(tmp_path)
    out = Ingested(path, tmp_path / "work").run(
        roles_stage, ProjectState(lens=["clinical"], target="hba1c", purpose="prediction"))
    roles = {p["column"]: p["proposed"] for p in out["columns"]}
    assert roles["energy_kcal"] == "energy"
    assert {roles[c] for c in ("protein_g", "carbohydrate_g", "fat_g")} == {"exposure"}
    predictors = [c for c, r in roles.items() if r in readings.PREDICTOR_ROLES]
    without = {**roles, "energy_kcal": "covariate"}
    for task in ("regression", "multiclass"):
        for project, runs in ((roles, True), (without, False)):
            assert (energy_fill(None, predictors, project, predictors) is not None) == runs
            groups = readings.predictor_groups(["clinical"], task, project)
            for role, group in groups.items():
                assert (readings.FILL in group.note.lower()) == runs, (task, role, group.note)
    assert readings.predictor_groups(["clinical"], "regression", roles)["energy"].note == (
        "The energy-aware fill fills the measures that carry energy from it; the screened elastic "
        "net's screen keeps it.")


def test_under_estimate_no_group_name_reaches_a_role_line():
    for purpose in ("inference", None):
        for lenses in (["dietary"], ["genomics"], ["clinical"]):
            for role in readings.PREDICTOR_ROLES:
                assert readings.role_line_words(role, purpose, lenses) == \
                    readings.role_words(role, purpose)


def test_the_energy_aware_fill_the_card_names_reads_the_predictor_role_alone():
    """The energy-aware fill, read from the code: under a single fill a Predictor nutrient that
    carries energy is filled from its least-squares line on the energy column, and a nutrient given
    the covariate role is filled by its median (W1-S se: on a dietary Predict project). With total
    energy given another predictor role, no energy-aware fill runs. The values are constructed:
    carbohydrate and fat are exact lines on energy, so the line gives 0.1 and 0.04 of 2,800."""
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


def test_the_screen_is_on_a_regression_shelf_under_every_lens_and_screens_the_predictor_role():
    """The screened elastic net's screen, read from the code: on the Predict shelf of a regression
    outcome under a dietary, a clinical and an assay lens alike (the shelf is never shortened; the
    shelves of the other tasks are the test above's), it takes the Predictor columns and passes
    every other."""
    from types import SimpleNamespace

    import turbotab.core.methods.omics  # noqa: F401 - registers the screened family
    from turbotab.core.models.base import Situation, get_family, rank

    for lenses in (("dietary",), ("clinical",), ("genomics",)):
        shelf = [f.key for f, _ in rank(Situation(task="regression", purpose="prediction",
                                                  n_rows=200, n_features=12, lenses=lenses))]
        assert "screened_elastic_net" in shelf, lenses
    roles = {"carb_g": "exposure", "fat_g": "exposure", "age": "covariate", "kcal": "energy"}
    [(_name, screen)] = get_family("screened_elastic_net").preprocess(
        SimpleNamespace(predictors=list(roles), roles=roles))
    assert list(screen.columns) == ["carb_g", "fat_g"]


def test_normalization_and_batch_correction_the_assay_card_names_act_on_predictor_columns_alone():
    """The omics normalization an applied ``omics_scale`` repair records and reference ComBat, read
    from the code on one constructed state, each take the Predictor columns (role ``exposure``) and
    skip a covariate and the energy column."""
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
