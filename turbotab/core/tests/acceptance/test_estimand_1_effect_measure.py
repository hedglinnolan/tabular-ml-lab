"""ESTIMAND · 1 · the effect measure is part of the estimand (MODELING_SEQUENCE §0 ruling 9: "The
estimand question asks: difference or ratio; conditional or marginal. ORs and HRs are labeled
conditional and non-collapsible"; §2: "adding a precision covariate under an OR or HR *changes* the
conditional estimand, and the app says so").

Sources (read 2026-10-05, the review record's quotations): Daniel R, Zhang J, Farewell D. *Biom J*
2021;63:528: "to promote the idea that the words conditional and adjusted (likewise marginal and
unadjusted) should not be used interchangeably"; VanderWeele TJ. *Eur J Epidemiol* 2019;34:211:
"the change-in-estimate approach is relative to the effect measure and it is inappropriate for
non-collapsible measures such as the odds ratio or hazard ratio if the outcome is common."

The relation the app states is first shown to hold, by simulation with a known truth: with the
exposure randomized (independent of everything) and a cause of the outcome only beside it, the
logistic odds ratio moves when that cause is adjusted for (statsmodels), while the marginal risk
difference does not move beyond sampling noise (standardization written out with NumPy).
"""
from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from turbotab.core import decisions as d, estimand, voice
from turbotab.core.tests.acceptance import estimand_fixtures as ef

TASKS = {"regression": ("mean_difference", "difference", "conditional and marginal", True),
         "binary": ("odds_ratio", "ratio", "conditional", False),
         "time_to_event": ("hazard_ratio", "ratio", "conditional", False),
         "ordinal": ("cumulative_odds_ratio", "ratio", "conditional", False),
         "multiclass": ("relative_risk_ratio", "ratio", "conditional", False)}


@pytest.mark.parametrize("task", list(TASKS))
def test_1_every_measure_is_labeled_difference_or_ratio_and_conditional_or_marginal(task):
    measure, scale, conditioning, collapsible = TASKS[task]
    offered = {m["measure"]: m for m in estimand.measures_offered(task)}
    own = offered[measure]
    assert own["fitted"] and own["scale"] == scale and own["conditioning"] == conditioning
    assert own["collapsible"] is collapsible
    if not collapsible:
        assert own["reason"].startswith("conditional and non-collapsible")
        assert estimand.MEASURE_WORDS[measure].startswith("conditional")
    for marginal in ("risk_difference", "risk_ratio"):
        m = offered[marginal]
        assert m["conditioning"] == "marginal"
        assert m["scale"] == ("difference" if marginal == "risk_difference" else "ratio")
        assert m["fitted"] is (task == "binary"), (task, marginal)
        if task != "binary":
            assert m["reason"].startswith("a marginal risk")  # named, and refused with why


def test_1_a_cause_of_the_outcome_only_changes_the_odds_ratio_not_the_risk_difference():
    """Non-collapsibility, by simulation with a known truth: the conditional odds ratio of a
    randomized exposure moves when a strong cause of the outcome only is adjusted for; the marginal
    risk difference stays where it was (its two estimates differ by sampling noise only)."""
    rng = np.random.default_rng(11)
    n = 200_000
    x = rng.binomial(1, 0.5, n).astype(float)
    z = rng.normal(0, 1.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-0.5 + 1.0 * x + 1.5 * z)))).astype(float)
    alone = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
    both = sm.Logit(y, sm.add_constant(np.column_stack([x, z]))).fit(disp=0)
    assert both.params[1] - alone.params[1] > 0.3  # the conditional log-odds ratio moves
    X = sm.add_constant(np.column_stack([x, z]))
    X1, X0 = X.copy(), X.copy()
    X1[:, 1], X0[:, 1] = 1.0, 0.0
    expit = lambda e: 1 / (1 + np.exp(-e))  # noqa: E731
    rd_adjusted = float(np.mean(expit(X1 @ both.params)) - np.mean(expit(X0 @ both.params)))
    rd_crude = float(y[x == 1].mean() - y[x == 0].mean())
    assert rd_adjusted == pytest.approx(rd_crude, abs=0.005)  # the marginal difference does not


def _state(measure: str, task: str) -> d.ProjectState:
    return ef.state(target="dm" if task == "binary" else "glucose", task=task, measure=measure,
                    event="yes" if task == "binary" else None)


@pytest.mark.parametrize("measure,task,said", [("odds_ratio", "binary", True),
                                                ("hazard_ratio", "time_to_event", True),
                                                ("mean_difference", "regression", False),
                                                ("risk_difference", "binary", False)])
def test_1_the_app_says_a_precision_covariate_changes_a_conditional_ratio(measure, task, said):
    """``activity`` is answered a cause of the outcome only (precision). Under an odds or hazard
    ratio the adjustment card, the record's sentence and the caption say that adjusting for it
    changes the conditional estimand, not only its precision; under a collapsible measure (a mean
    difference, a marginal risk difference) none of them does."""
    st = _state(measure, task)
    card = estimand.adjustment_card(st)
    note = card["answered"]["activity"]["estimand_note"]
    words = estimand.MEASURE_WORDS[measure]
    methods = (f"Under a {words}, adjusting for `activity`, a cause of the outcome only, changes "
               f"the conditional estimand, not only its precision: the ratio is non-collapsible "
               f"(Daniel, Zhang & Farewell 2021, Biom J 63:528).")
    # the card says the comparison you want; the record's sentence keeps the term
    expected = methods.replace("conditional estimand", "conditional comparison you want")
    assert (note == expected) is said and (note is None) is (not said)
    assert (card["estimand_note"] == expected) is said
    assert all(card["answered"][c]["estimand_note"] is None for c in ("age", "sex", "smoking"))
    decision = d.SetAdjustment(exposure="fiber", answers={"activity": d.CovariateAnswers(
        **ef.PRECISION)})
    sentence = voice.sentence_for(decision, st, None)
    assert (sentence == "For the effect of `fiber`, by the disjunctive cause criterion: `activity` "
                        "is a cause of the outcome only, adjusted for. " + methods) is said
    caption = estimand.caption(st)
    assert ("; a conditional ratio, which changes with the covariates even without confounding: "
            "adjusting for `activity`, a cause of the outcome only, changes the conditional "
            "estimand, not only its precision" in caption) is said


def test_1_a_group_the_pack_guesses_a_cause_of_the_outcome_only_carries_the_note(monkeypatch):
    """On the one-tap card a group whose guess derives "precision" says it before the tap."""
    st = _state("odds_ratio", "binary").model_copy(update={"adjustment": None})
    # LEASH: smoking is read as lifestyle, whose guess the card blocks with the demographics' when
    # the two guesses agree (the same answers, one tap); both are set to "precision" here.
    for key in ("demographic", "lifestyle"):
        monkeypatch.setitem(estimand.GUESSES, key,
                            {**estimand.GUESSES[key], "answers": dict(ef.PRECISION)})
    card = estimand.adjustment_card(st)
    group = next(g for g in card["groups"] if g["key"] == "demographic")
    assert group["derived"] == "precision"
    assert group["estimand_note"].startswith("Under a conditional odds ratio, adjusting for `age`, "
                                             "`sex` and `smoking`, causes of the outcome only")
