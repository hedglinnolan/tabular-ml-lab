"""Q-b repair (WAVE_C6A_PLAN §3 row Q-b, §5 (b)): the substitution's wording holds on every surface
in the order the answers are really given, and says no unit its name does not declare.

* The methods sentence of ``set_estimand`` is written on the state before it: before the
  adjustment set and the energy method, which come after it. The methods text restates it on the
  answers as they stand, so it never contradicts the caption and Table 2.
* ``sugar_tsp`` declares no gram: the swap is "more sugar ...", never "1 g more sugar".
* Table 2 says each swap as the other surfaces do, and the preview's line keeps the measure.

References, by hand: the wordings below are written from the models' algebra, proved by the
statsmodels identities in ``test_qb_substitution_sentence`` (other = carb − sugar gives
β′_sugar − β′_other = β_sugar; rest = kcal − 4·sugar − 4·protein − 9·fat gives
β″_sugar − 4·β″_rest = β_sugar), and here for the total beside its part (other = carb − sugar gives
β′_other = β_carb: the carbohydrate that is not sugar).
"""
from __future__ import annotations

import pytest
import statsmodels.api as sm

from turbotab.core import estimand as est
from turbotab.core import voice
from turbotab.core.consequences import CAPTION_WORDS, words
from turbotab.core.decisions import (CovariateAnswers, DecisionLog, EnergyAdjustment, EstimandSpec,
                                     ProjectState, SetAdjustment, SetEnergyAdjustment, SetEstimand,
                                     SetLens, SetPurpose, SetRoles, SetTarget, SetTask)
from turbotab.core.plan_previews import estimand_line
from turbotab.core.provenance import methods_text
from turbotab.core.tests.acceptance.test_qb_substitution_sentence import _diet
from turbotab.core.tests.acceptance.test_wp6_energy_estimands import _stages

ROLES = {"sugar_g": "exposure", "carb_g": "covariate", "protein_g": "covariate",
         "fat_g": "covariate", "kcal": "energy", "age": "covariate"}
NUTRIENTS = ["sugar_g", "carb_g", "protein_g", "fat_g"]
CONFOUNDER = CovariateAnswers(causes_exposure="yes", causes_outcome="yes", after_exposure="no")
# Changed by the exposure and a cause of the outcome: a mediator, left out of a total effect.
CONSEQUENCE = CovariateAnswers(causes_exposure="no", causes_outcome="yes", after_exposure="yes")

HELD = "1 g more sugar in place of other carbohydrate (total carbohydrate and energy fixed)"
CARB_OUT = ("1 g more sugar in place of other carbohydrate, alcohol and other energy "
            "(total energy fixed)")


def _record(tmp_path, *later) -> tuple[str, str, ProjectState]:
    """The methods text of a log answered in the Router's order: the estimand first, then the
    adjustment set and the energy method (``later``), each sentence written on the state before it."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    log = DecisionLog(tmp_path / "decisions.jsonl")
    for d in (SetLens(lenses=["dietary"]), SetTarget(column="y"),
              SetTask(column="y", task="regression"), SetPurpose(purpose="inference"),
              SetRoles(roles=ROLES),
              SetEstimand(exposure="sugar_g", contrast="substitution", measure="mean_difference"),
              *later):
        log.append(d, sentence=lambda d, before: voice.sentence_for(d, before, None))
    [said] = [r.sentence for r in log.records() if r.decision.kind == "set_estimand"]
    [line] = [m.sentence for m in methods_text(log.records()).lines
              if "estimates the total effect" in m.sentence]
    return said, line, log.state()


def test_the_methods_text_restates_the_swap_once_the_adjustment_leaves_carbohydrate_out(tmp_path):
    answers = {"carb_g": CONSEQUENCE, "protein_g": CONFOUNDER, "fat_g": CONFOUNDER,
               "age": CONFOUNDER}
    said, line, state = _record(
        tmp_path, SetAdjustment(exposure="sugar_g", answers=answers),
        SetEnergyAdjustment(method="standard", energy_column="kcal", nutrients=NUTRIENTS))
    # As said, before the adjustment answers: every covariate held, carbohydrate with it.
    assert HELD in said, said
    # As the analysis now stands: carbohydrate is left out, so sugar displaces it with the rest.
    assert CARB_OUT in line, line
    assert "total carbohydrate" not in line, line
    assert CARB_OUT in est.caption(state, "regression")


def test_the_methods_text_follows_the_energy_method_given_after_the_estimand(tmp_path):
    answers = {c: CONFOUNDER for c in ("carb_g", "protein_g", "fat_g", "age")}
    said, line, _ = _record(
        tmp_path / "standard", SetAdjustment(exposure="sugar_g", answers=answers),
        SetEnergyAdjustment(method="standard", energy_column="kcal", nutrients=NUTRIENTS))
    assert HELD in said and HELD in line, (said, line)  # the answers hold carbohydrate
    said, line, _ = _record(
        tmp_path / "density", SetAdjustment(exposure="sugar_g", answers=answers),
        SetEnergyAdjustment(method="density", energy_column="kcal", nutrients=NUTRIENTS))
    assert HELD in said, said
    # A density's coefficient is no swap at fixed energy (methods.energy.substitution_swap): the
    # restated sentence claims no "1 g more ... fixed".
    assert "1 g more" not in line and "carbohydrate and energy fixed" not in line, line


def _state(roles: dict[str, str], exposure: str, nutrients: list[str]) -> ProjectState:
    return ProjectState(
        lens=["dietary"], target="y", task="regression", purpose="inference", roles=roles,
        estimand=EstimandSpec(exposure=exposure, contrast="substitution", measure="mean_difference"),
        energy_adjustment=EnergyAdjustment(method="standard", energy_column="kcal",
                                           nutrients=nutrients))


def test_a_name_that_declares_no_gram_is_not_said_in_grams():
    roles = {"sugar_tsp": "exposure", "protein_g": "covariate", "fat_g": "covariate",
             "kcal": "energy", "age": "covariate"}
    st = _state(roles, "sugar_tsp", ["sugar_tsp", "protein_g", "fat_g"])
    text = est.caption(st, "regression")
    assert "more sugar in place of" in text and "1 g" not in text, text
    assert "per unit of `sugar_tsp`" in text, text


def test_table_2_says_the_total_beside_its_part_as_the_caption_does(tmp_path):
    frame = _diet(seed=5)
    X = ["sugar_g", "carb_g", "protein_g", "fat_g", "kcal", "age"]
    original = sm.OLS(frame["y"], sm.add_constant(frame[X])).fit()
    other = frame.assign(other_g=frame["carb_g"] - frame["sugar_g"])
    reparam = sm.OLS(other["y"], sm.add_constant(
        other[["sugar_g", "other_g", "protein_g", "fat_g", "kcal", "age"]])).fit()
    # the reference: carb's coefficient is the carbohydrate that is not sugar
    assert reparam.params["other_g"] == pytest.approx(original.params["carb_g"], abs=1e-10)

    roles = {"carb_g": "exposure", "sugar_g": "covariate", "protein_g": "covariate",
             "fat_g": "covariate", "kcal": "energy", "age": "covariate"}
    run = _stages(frame, tmp_path / "total", roles, EnergyAdjustment(
        method="standard", energy_column="kcal", nutrients=NUTRIENTS))
    row = run["rows"]["carb_g"]
    assert row["estimate"] == pytest.approx(reparam.params["other_g"], rel=1e-8)
    # written by hand: protein, carbohydrate and fat are in the model, alcohol is left out
    phrase = ("remaining carbohydrate (holding sugar fixed) in place of alcohol and other energy "
              "(total energy fixed)")
    assert row["meaning"] == phrase
    assert f"1 g more {phrase}" in est.caption(_state(roles, "carb_g", NUTRIENTS), "regression")


def test_the_preview_line_keeps_the_measure_beside_the_whole_swap():
    line = estimand_line(_state(ROLES, "sugar_g", NUTRIENTS))
    assert words(line) <= CAPTION_WORDS, line
    assert "sugar in place of other carbohydrate (total carbohydrate and energy fixed)" in line
    assert "mean difference" in line or "difference in the mean outcome" in line, line
