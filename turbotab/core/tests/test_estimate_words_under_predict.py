"""W1-R (the walkthrough of 2026-10-17, findings 3a, 3b and 3c): under Predict the engine says
nothing that only an Estimate says, and under Estimate it says what it said before.

* **3a, day-to-day noise.** Regression dilution weakens a slope on a noisy measure by ``1 − λ``
  (Hutcheon et al. 2010), which biases an estimate. A prediction model is fit and scored on that same
  noisy measure, so the held-out rows carry the noise the model meets in use and the score is
  honest. The noise limits how much the column can predict. Under Predict the triage therefore
  records it For the record: it changes no number, and no limitation is drafted. Under Estimate it
  could bias the estimate, as before.
* **3b, the design's concerns.** What a total's coefficient means beside its parts, a substitution
  through it, two columns of one nutrient that may overlap in a swap, and fiber's energy counted
  twice are concerns about coefficients and substitutions. Under Predict no coefficient is
  interpreted and nothing is substituted (BEATS_PREDICT §9 ruling 5), so none is said. The shares
  that sum to 100% stay: no linear model can estimate all of them under either goal.
* **3c, the role words.** Under Predict no column is studied and nothing is adjusted for, so an
  exposure and a covariate are each "a predictor" (BEATS_PREDICT C1: a predictor is the engine's
  covariate), and the roles card names no study factor. The engine's roles stay as they are, and
  the words under Estimate are unchanged.

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
from turbotab.core.decisions import ProjectState
from turbotab.core.models.pipeline import design_spec, warnings_for
from turbotab.core.teaching.content import entries_now
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


def test_under_predict_day_to_day_noise_is_for_the_record_and_biases_no_score():
    state = _state("prediction")
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


def test_under_predict_measured_noise_says_its_reliability_and_is_never_acted_on():
    """λ measured from two recalls a person, with 1 − λ past the band-2 cut-off (0.5): under
    Estimate it is acted on; under Predict it is still For the record, with λ as measured."""
    k = 2
    frame = _two_recalls(sd_between=4.0, sd_within=12.0, k=k)
    lam = _balanced_lambda(frame, k)
    assert 1 - lam > 0.5  # by construction: σ²w / k = 72 against σ²b = 16, so λ ≈ 0.18
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
    assert entry["one_liner"] == ("Only predictors, covariates and the energy column enter the "
                                  "models; every other role stays out of them.")
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
