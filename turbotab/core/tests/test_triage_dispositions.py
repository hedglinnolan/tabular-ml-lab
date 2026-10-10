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

import pandas as pd

from ml import binary_text
from turbotab import engine
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


def _level(one, zero, relabels=True):
    """A binary repair option as ``repairs._offer_binary`` serves it: ``relabels`` says the column
    as written is two spellings and no blank, so the repair only renames its two values."""
    return {"key": "level", "decision": {"params": {"column": "gender", "one": one, "zero": zero,
                                                    "relabels": relabels}}}


GENDER = {"id": "binary_text__gender", "severity": "warning",
          "summary": "gender is two values written as text",
          "affected_columns": ["gender"], "routes_to": None,
          "repairs": [_level("male", "female"), _level("female", "male")], "answered_by": None}
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


# ── a pure relabeling only: the column as written is two spellings and no blank ────────────────
#
# The theorem covers recoding one indicator into another. Left alone, a text column enters the
# model one-hot encoded on its raw spellings (``models.pipeline``: OneHotEncoder(drop="first")), so
# "Male", "male " and "MALE" are three indicators and "n/a" a fourth: the repair then changes the
# column space, and the estimate moves. A blank is filled one way as text (its most frequent value)
# and another as a 0/1 number (with a missing indicator under that answer): not a relabeling either.

SPELLINGS = ["Male", "male ", "MALE", "Female", "female", "n/a"]


def _frame(gender, n=600, seed=5):
    rng = np.random.default_rng(seed)
    g = np.resize(np.asarray(gender, dtype=object), n)
    rng.shuffle(g)
    male = np.array([str(v).strip().lower() == "male" for v in g], dtype=float)
    sugar = rng.gamma(4.0, 20.0, n) + 12 * male + 6 * (g == "MALE")
    age = rng.uniform(20, 80, n)
    glucose = 90 + 0.05 * sugar + 0.2 * age + 4 * male + 3 * (g == "n/a") + rng.normal(0, 8, n)
    return pd.DataFrame({"SEQN": np.arange(n), "sugar": sugar, "kcal": rng.gamma(9, 220, n),
                         "age": age, "gender": g, "glucose": glucose})


def _served(frame):
    """The binary finding as the findings stage serves it, with the options its repair offers."""
    raw = engine.shape_finding_to_dict(binary_text.binary_text_finding("gender", frame["gender"]))
    served = {"id": raw["id"], "severity": raw["severity"], "summary": raw["title"],
              "affected_columns": ["gender"], "routes_to": None, "answered_by": None}
    repairs.attach([(raw, served)], frame, "glucose")
    return served


def test_variant_spellings_are_not_a_relabeling_so_the_repair_is_acted_on_in_your_data():
    frame = _frame(SPELLINGS)
    served = _served(frame)
    assert served["repairs"] and not any(o["decision"]["params"]["relabels"]
                                         for o in served["repairs"])
    got = sweep.recommend(design(), served, set())
    assert got[:2] == ("act_on_it", False) and "Your data" in got[2] and "exactly" not in got[2]
    # The reference: as written it enters as one indicator per raw spelling after the first (four
    # here), repaired as one 0/1 column; sugar's coefficient differs between the two.
    y = frame["glucose"].to_numpy(float)
    base = frame[["sugar", "age"]].to_numpy(float)
    raw = pd.get_dummies(frame["gender"], drop_first=True).to_numpy(float)
    assert raw.shape[1] == len(set(SPELLINGS)) - 1
    coded = frame["gender"].map(lambda v: float(str(v).strip().lower() == "male"))
    keep = frame["gender"] != "n/a"
    a = sm.OLS(y, sm.add_constant(np.column_stack([base, raw]))).fit()
    b = sm.OLS(y[keep], sm.add_constant(np.column_stack([base, coded])[keep])).fit()
    assert abs(a.params[1] - b.params[1]) > 1e-4


@pytest.mark.parametrize("values", [["Male", "male", "Female"], ["male", "female", None]],
                         ids=["two-spellings", "a-blank"])
def test_a_second_spelling_or_a_blank_is_not_a_relabeling(values):
    served = _served(_frame(values))
    assert served["repairs"] and not any(o["decision"]["params"]["relabels"]
                                         for o in served["repairs"])
    assert sweep.recommend(design(), served, set())[0] == "act_on_it"


def test_two_spellings_and_no_blank_are_a_relabeling_and_change_no_number():
    frame = _frame(["Male", "Female"])
    served = _served(frame)
    assert all(o["decision"]["params"]["relabels"] for o in served["repairs"])
    got = sweep.recommend(design(), served, set())
    assert got[0] == "no_change" and "exactly" in got[2]
    # The reference: as written, one indicator after the first ("Female" dropped); repaired, `male`
    # as 1 or `female` as 1. All three span the same space: sugar is identical to 1e-12.
    y = frame["glucose"].to_numpy(float)
    base = frame[["sugar", "age"]].to_numpy(float)
    raw = pd.get_dummies(frame["gender"], drop_first=True).to_numpy(float)
    fits = [sm.OLS(y, sm.add_constant(np.column_stack([base, g]))).fit()
            for g in (raw, (frame["gender"] == "Male").to_numpy(float),
                      (frame["gender"] == "Female").to_numpy(float))]
    for f in fits[1:]:
        assert abs(f.params[1] - fits[0].params[1]) < 1e-12
        assert abs(f.bse[1] - fits[0].bse[1]) < 1e-12


@pytest.mark.parametrize("role", ["design", "time", "cluster", "flag", "excluded", "identifier"])
def test_the_rule_holds_only_for_an_adjustment_term(role):
    """A column with no coefficient (a design, time or flag column, or one left out) has no sign to
    flip, and a time column is not guaranteed to enter as one indicator: the repair is acted on."""
    state = design(roles={**ROLES, "gender": role})
    got = sweep.recommend(state, GENDER, set())
    assert got[0] != "no_change" and "coefficient" not in got[2]
    assert sweep.recommend(design(), GENDER, set())[0] == "no_change"
