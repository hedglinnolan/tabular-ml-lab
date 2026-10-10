"""Q-a · the triage's dispositions point to where a finding is fixed (WAVE_C6A_PLAN §3 row Q-a,
§5 quest-log item (a); ``turbotab/core/sweep.py:recommend``).

The quest-log design review found the triage filing SAS zeros in what you study, and an
out-of-range intake row, as "could bias the estimate, kept as a limitation". A nutrition researcher
would not sign that: the first has a repair in Your data, the second is decided at the eligibility
question in Who's in. The rules, in order: a blocker; a reparameterization of a column that is not
focal (not what you study, not the outcome, not a declared modifier) changes no number here; a
finding with a repair, or routed to a question (open or answered), is acted on where it is
decided, naming the stage and the question; then the rules as they were.

The reparameterization rule rests on a theorem, not on the code: recoding a two-valued covariate
(which value is 1) is an invertible linear map of the model matrix's column space, so the fitted
values, the residuals and every other coefficient are unchanged (Frisch–Waugh–Lovell). The test
shows it with statsmodels on data drawn here, and shows the exception: with gender × what you
study, the coefficient of what you study is the effect where gender is 0, so recoding changes it.
"""
from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from turbotab.core import decisions, repairs, sweep
from turbotab.core.tests.test_confirm_sweep import log_of, planned

# The dietary design fixture's three findings (the quest-log design review's Models triage).
SAS = {"id": "sas_zeros", "severity": "warning",
       "summary": "152 cells hold 5.4e-79, a misread SAS zero",
       "affected_columns": ["sugar", "kcal"], "routes_to": None,
       "repairs": [{"key": "zero"}], "answered_by": None}
INTAKE = {"id": "pack::dietary::implausible_intake", "severity": "warning",
          "summary": "Some energy reports are outside 500 to 5,000 kcal",
          "affected_columns": ["kcal"], "routes_to": "exclusions", "repairs": [],
          "answered_by": None}
GENDER = {"id": "binary_text__gender", "severity": "warning",
          "summary": "gender is two values written as text",
          "affected_columns": ["gender"], "routes_to": None,
          "repairs": [{"key": "level:male"}, {"key": "level:female"}], "answered_by": None}
FINDINGS = {"findings": [SAS, INTAKE, GENDER]}
ROLES = {"SEQN": "identifier", "sugar": "exposure", "kcal": "energy", "age": "covariate",
         "gender": "covariate"}
COLUMNS = ["SEQN", "sugar", "kcal", "age", "gender", "glucose"]


def design(**update):
    """Sugar → fasting glucose under Estimate, every question answered (the eligibility question
    included, with a rule on age only, so the intake finding is still open)."""
    update.setdefault("exclusions", [decisions.ExclusionRule(column="age", low=18.0,
                                                              reason="adults")])
    update.setdefault("roles", dict(ROLES))
    return planned(**update)


def items(state, findings=FINDINGS):
    log = log_of(state, findings=findings, columns=COLUMNS)
    return {i.id: i for i in sweep.triage(state, log, findings).items}


# ── the design fixture's three items ─────────────────────────────────────────


def test_sas_zeros_are_acted_on_by_their_repair_in_your_data():
    got = items(design())["sas_zeros"]
    assert (got.recommended, got.blocker) == ("act_on_it", False)
    assert "Your data" in got.reason and "repair" in got.reason
    assert got.stage == "data" and not got.limitation


def test_an_implausible_intake_routed_to_an_answered_question_is_acted_on_there():
    got = items(design())["pack::dietary::implausible_intake"]
    assert (got.recommended, got.blocker) == ("act_on_it", False)
    assert "Who's in" in got.reason and "the eligibility question" in got.reason
    assert got.question == "exclusions" and got.stage == "whos_in" and not got.limitation
    # Still to come, it is decided there too, and says so.
    state = design()
    still = sweep.recommend(state, INTAKE, {"exclusions"})
    assert still[:2] == ("act_on_it", False) and "still to come" in still[2]


def test_recoding_a_covariate_that_is_not_focal_changes_no_number_here():
    got = items(design())["binary_text__gender"]
    assert (got.recommended, got.blocker) == ("no_change", False)
    assert got.label == "Doesn't change your numbers here"
    assert "`gender`" in got.reason and "`sugar`" in got.reason
    # Exact by theorem: band 0, not floored at 1, and the theorem is its quiet label.
    assert got.band == 0 and got.calibrated and "Frisch–Waugh–Lovell" in (got.measure or "")
    assert not got.limitation


def test_the_rule_does_not_hold_when_gender_is_focal():
    # What you study.
    roles = {**ROLES, "gender": "exposure", "sugar": "covariate"}
    state = design(roles=roles, estimand=decisions.EstimandSpec(exposure="gender",
                                                                  measure="mean_difference"))
    assert items(state)["binary_text__gender"].recommended != "no_change"
    # A declared modifier: the effect of sugar is reported in each of gender's groups.
    modified = design(modifications={"gender": decisions.ModificationSpec(
        kind="effect_modification", exposure="sugar")})
    assert items(modified)["binary_text__gender"].recommended != "no_change"
    # A modifier withdrawn before any estimate is no longer focal.
    assert items(design(modifications={"gender": None}))["binary_text__gender"].recommended \
        == "no_change"
    # Under Predict there is no focal estimate the theorem speaks to: the repair is acted on.
    assert items(design(purpose="prediction"))["binary_text__gender"].recommended == "act_on_it"


def test_the_repair_families_declare_which_reparameterize():
    assert repairs.FAMILIES["binary_text"].reparameterizes
    assert not any(f.reparameterizes for k, f in repairs.FAMILIES.items() if k != "binary_text")


def test_a_finding_with_no_repair_and_no_route_keeps_the_rules_as_they_were():
    plain = {"id": "outliers__age", "severity": "warning", "summary": "Extreme ages",
             "affected_columns": ["age"], "routes_to": None, "repairs": [], "answered_by": None}
    assert sweep.recommend(design(), plain, set())[0] == "could_bias"
    critical = {**GENDER, "severity": "critical"}
    assert sweep.recommend(design(), critical, set())[:2] == ("act_on_it", True)


# ── the independent reference: the theorem, by statsmodels ──────────────────


def _draw(n=500, seed=7):
    rng = np.random.default_rng(seed)
    male = rng.integers(0, 2, n).astype(float)
    age = rng.uniform(20, 80, n)
    sugar = rng.gamma(4.0, 20.0, n) + 15 * male
    glucose = 90 + 0.05 * sugar + 0.2 * age + 4 * male + 0.03 * sugar * male + rng.normal(0, 8, n)
    return sugar, age, male, glucose


def test_gender_coded_either_way_gives_the_same_estimate_and_interval_for_sugar():
    sugar, age, male, glucose = _draw()
    female = 1.0 - male
    a = sm.OLS(glucose, np.column_stack([np.ones_like(sugar), sugar, age, male])).fit()
    b = sm.OLS(glucose, np.column_stack([np.ones_like(sugar), sugar, age, female])).fit()
    assert abs(a.params[1] - b.params[1]) < 1e-12
    assert abs(a.bse[1] - b.bse[1]) < 1e-12
    assert a.params[3] == pytest.approx(-b.params[3], abs=1e-10)  # gender's own sign flips
    # With gender × sugar, sugar's coefficient is its effect where gender is 0: recoding moves it.
    ai = sm.OLS(glucose, np.column_stack([np.ones_like(sugar), sugar, age, male, sugar * male])).fit()
    bi = sm.OLS(glucose, np.column_stack([np.ones_like(sugar), sugar, age, female,
                                          sugar * female])).fit()
    assert abs(ai.params[1] - bi.params[1]) > 1e-3
    assert ai.params[1] + ai.params[4] == pytest.approx(bi.params[1], abs=1e-10)
