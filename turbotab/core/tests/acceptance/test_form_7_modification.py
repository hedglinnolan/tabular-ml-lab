"""FORM · 7 · effect modification and interaction as two declared objects (MODELING_SEQUENCE §1
row 7; Knol & VanderWeele 2012, Int J Epidemiol 41:514).

The cohort is ESTIMAND's (``estimand_fixtures.cohort``): a yes/no outcome ``dm``, the exposure
``fiber``, ``sex`` as the declared modifier, ``supplement`` as a second exposure.

The independent references:

* **R ``glm``** (binomial) on the same design written out (``fiber * sex`` with the covariates
  the adjustment answers keep), its coefficients and ``vcov``, to 1e-6;
* **the hand computation** of every reported quantity from R's coefficients, with NumPy: each
  combination against the single reference ``(a0, female)``, the stratum-specific odds ratios, the
  ratio of odds ratios, and the RERI with Hosmer & Lemeshow's delta-method interval (its gradient
  in the three coefficients written out), to 1e-6 against the stage, and to 1e-8 against the
  engine's own arithmetic fed the same coefficients;
* the interaction's second adjustment set, the Router's re-ask and the post hoc label from the
  fixture's declared answers.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from turbotab.core import decisions as d
from turbotab.core.methods import interaction as ix
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.acceptance import estimand_fixtures as est_f
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r

COVARIATES = ["age", "smoking", "activity"]  # the primary set beside sex (bmi: unknown timing)


def _run(frame: pd.DataFrame, folder: Any, st: d.ProjectState) -> dict[str, Any]:
    from turbotab.core.stages.modeling import design_stage

    paths = mf.ingest_frame(frame, folder)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
    info = mf.target_info(st.task, st.target)
    design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
    return ix.modification_stage(mf.context(st, {"design": design, "split": split,
                                                 "target_info": info}, paths)).data


@pytest.fixture(scope="module")
def em(tmp_path_factory) -> dict[str, Any]:
    frame = est_f.cohort(n=1500, seed=17)
    st = est_f.state(target="dm", task="binary", event="yes", measure="odds_ratio",
                     modifications={"sex": d.ModificationSpec(kind="effect_modification",
                                                              exposure="fiber")})
    return {"frame": frame, "state": st,
            "out": _run(frame, tmp_path_factory.mktemp("em"), st)}


def _hand(beta: dict[str, float], cov: np.ndarray, a0: float, a1: float) -> dict[str, Any]:
    """Every quantity from (β_fiber, β_male, β_fiber×male) and their covariance, written out."""
    bf, bm, bi = beta["fiber"], beta["sexmale"], beta["fiber:sexmale"]
    delta = a1 - a0
    L10 = bf * delta
    L01 = bm + bi * a0
    L11 = bf * delta + bm + bi * a1
    c10, c01, c11 = np.array([delta, 0, 0]), np.array([0, 1, a0]), np.array([delta, 1, a1])
    z = stats.norm.ppf(0.975)

    def ci(c: np.ndarray) -> tuple[float, float, float]:
        est = float(c @ [bf, bm, bi])
        se = math.sqrt(float(c @ cov @ c))
        return math.exp(est), math.exp(est - z * se), math.exp(est + z * se)

    reri = math.exp(L11) - math.exp(L10) - math.exp(L01) + 1
    g = math.exp(L11) * c11 - math.exp(L10) * c10 - math.exp(L01) * c01
    se = math.sqrt(float(g @ cov @ g))
    return {"effect_female": ci(c10), "effect_male": ci(c11 - c01), "joint_a0_male": ci(c01),
            "joint_a1_male": ci(c11), "mult": ci(c11 - c10 - c01),
            "reri": (reri, reri - z * se, reri + z * se),
            "chi2": bi ** 2 / cov[2, 2]}


@needs_r
def test_7_the_reported_effects_reri_and_ratio_of_ratios_agree_with_r_glm_by_hand(em, tmp_path):
    frame = em["frame"]
    found = run_r("""
d <- read.csv(frame_csv)
d$y <- as.integer(d$dm == "yes")
d$sex <- factor(d$sex, levels = c("female", "male"))
f <- glm(y ~ fiber * sex + age + smoking + activity, family = binomial, data = d,
         control = glm.control(epsilon = 1e-14, maxit = 100))
keep <- c("fiber", "sexmale", "fiber:sexmale")
out(list(beta = as.list(coef(f)[keep]), cov = unname(vcov(f)[keep, keep])))
""", {"frame": frame}, tmp_path)
    cov = np.asarray(found["cov"], dtype=float)
    a0, a1 = (float(v) for v in np.quantile(frame["fiber"].to_numpy(float), [0.25, 0.75],
                                            method="linear"))
    hand = _hand(found["beta"], cov, a0, a1)
    result = em["out"]["modifications"][0]
    assert (result["low"], result["high"]) == (pytest.approx(a0, abs=1e-12),
                                               pytest.approx(a1, abs=1e-12))
    assert result["strata"] == ["female", "male"] and result["reference"] == "female"
    fit = result["families"][0]
    assert fit["measure"] == "odds ratio" and fit["scale"] == "ratio"
    effects = {q["stratum"]: q for q in fit["effects"]}
    joint = {(q["a"], q["stratum"]): q for q in fit["joint"]}

    def same(q: dict[str, Any], ref: tuple[float, float, float]) -> None:
        assert (q["estimate"], q["ci_low"], q["ci_high"]) == (
            pytest.approx(ref[0], rel=1e-6), pytest.approx(ref[1], rel=1e-6),
            pytest.approx(ref[2], rel=1e-6))

    same(effects["female"], hand["effect_female"])
    same(effects["male"], hand["effect_male"])
    same(joint[("a1", "female")], hand["effect_female"])  # against the single reference
    same(joint[("a0", "male")], hand["joint_a0_male"])
    same(joint[("a1", "male")], hand["joint_a1_male"])
    same(fit["multiplicative"][0], hand["mult"])
    same(fit["reri"][0], hand["reri"])
    het = fit["heterogeneity"]
    assert het["distribution"] == "chi2" and het["df_num"] == 1
    assert het["statistic"] == pytest.approx(hand["chi2"], rel=1e-6)
    assert "The RERI is computed from odds ratios: a RERI from odds ratios approximates the RERI " \
           "of risk ratios only when the outcome is rare (VanderWeele & Knol 2014)." in \
           fit["concerns"]


def test_7_the_arithmetic_is_the_hand_computation_to_1e_8():
    """The engine's own arithmetic (``measures``, ``reri``) fed a coefficient vector and covariance,
    against the formulas written out in :func:`_hand`."""
    rng = np.random.default_rng(3)
    names = ["(intercept)", "fiber", "sex_male", "age", "fiber×sex_male"]
    beta = np.array([-1.2, 0.031, 0.42, 0.018, -0.022])
    A = rng.normal(size=(5, 5))
    cov = A @ A.T / 400 + np.eye(5) * 1e-4
    a0, a1 = 16.4, 23.9
    found = ix.measures(names, beta, cov,
                        a_values={"a0": {"fiber": a0}, "a1": {"fiber": a1}},
                        m_values={"female": {"sex_male": 0.0}, "male": {"sex_male": 1.0}},
                        reference="female", products=[("fiber", "sex_male", "fiber×sex_male")],
                        ratio=True)
    sub = cov[np.ix_([1, 2, 4], [1, 2, 4])]
    hand = _hand({"fiber": beta[1], "sexmale": beta[2], "fiber:sexmale": beta[4]}, sub, a0, a1)
    assert math.exp(found["effect|female"]["estimate"]) == pytest.approx(hand["effect_female"][0],
                                                                        rel=1e-8)
    assert math.exp(found["effect|male"]["estimate"]) == pytest.approx(hand["effect_male"][0],
                                                                      rel=1e-8)
    assert math.exp(found["mult|male"]["estimate"]) == pytest.approx(hand["mult"][0], rel=1e-8)
    for got, want in zip((found["reri|male"]["estimate"], found["reri|male"]["ci_low"],
                          found["reri|male"]["ci_high"]), hand["reri"]):
        assert got == pytest.approx(want, rel=1e-8)


def test_7_the_sentence_and_the_family_are_written_as_declared(em):
    result = em["out"]["modifications"][0]
    assert result["status"] == "declared" and result["kind"] == "effect_modification"
    assert result["adjusted_for"] == COVARIATES
    low, high = result["low"], result["high"]
    assert result["sentence"] == (
        f"Effect modification of `fiber`'s effect on `dm` by `sex` was declared before the "
        f"estimates were seen: the odds ratio for `fiber` from {low:.4g} to {high:.4g} (its 25th "
        f"to 75th percentile) within each level of `sex`, and each combination against a single "
        f"reference (`fiber` at {low:.4g}, `sex` at female); interaction on the additive scale as "
        f"the relative excess risk due to interaction (RERI) with a delta-method interval (Hosmer "
        f"& Lemeshow 1992, Epidemiology 3:452) and on the multiplicative scale as the ratio of "
        f"odds ratios (Knol & VanderWeele 2012, Int J Epidemiol 41:514), adjusted for `age`, "
        f"`smoking` and `activity`; the heterogeneity test is the Wald test that the 1 product "
        f"term is zero.")
    assert em["out"]["family_count"]["statement"] == (
        "2 tests in the family: the exposure's effect and 1 declared heterogeneity test (`sex`); "
        "the p-values are reported unadjusted with the number of tests stated")


# ── interaction: the adjustment set asked again for the second exposure ──────

ROLES = {**est_f.ROLES, "supplement": "covariate"}
SECOND = {"age": est_f.CONFOUNDER, "smoking": est_f.CONFOUNDER,
          "sex": {"causes_exposure": "no", "causes_outcome": "no", "after_exposure": "no"},
          "activity": est_f.PRECISION,
          "bmi": {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no"}}


def _interaction_state(answers: dict[str, Any] | None = None, **slots: Any) -> d.ProjectState:
    first = {**est_f.ANSWERS, "supplement": est_f.PRECISION}
    spec = d.ModificationSpec(kind="interaction", exposure="fiber",
                              answers={c: d.CovariateAnswers(**a) for c, a in
                                       (answers or {}).items()})
    return est_f.state(target="dm", task="binary", event="yes", measure="odds_ratio",
                       roles=ROLES, answers=first, modifications={"supplement": spec}, **slots)


def test_7_an_interaction_asks_the_adjustment_set_again_for_the_second_exposure():
    from turbotab.core.interview import route

    st = _interaction_state()
    spec = st.modifications["supplement"]
    assert ix.missing_answers(st, "supplement", spec) == ["age", "sex", "smoking", "activity",
                                                          "bmi"]
    step = next(s for s in route(st, {}, {}, []) if s.key == "modification")
    assert step.status in ("open", "waiting") and step.followup == "adjustment"
    done = _interaction_state(SECOND)
    step = next(s for s in route(done, {}, {}, []) if s.key == "modification")
    assert step.status == "answered" and step.followup is None
    a_set, extra, _ = ix.adjusted_sets(done, "supplement", done.modifications["supplement"])
    assert a_set == ["age", "sex", "smoking", "activity"] and extra == ["bmi"]
    # answers about a column that is not a covariate of the model are refused
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_modification", "modifier": "supplement",
                    "modification": "interaction",
                    "answers": {"glucose": est_f.CONFOUNDER, **SECOND}},
                   {"state": done, "columns": list(est_f.cohort(n=20).columns)})
    assert refused.value.code == "not_a_covariate"


def test_7_the_interaction_model_adjusts_for_both_exposures_confounders(tmp_path):
    frame = est_f.cohort(n=1200, seed=19)
    st = _interaction_state(SECOND)
    out = _run(frame, tmp_path, st)
    result = out["modifications"][0]
    assert result["kind"] == "interaction" and result["second_adjusted_for"] == ["bmi"]
    assert result["strata"] == ["0", "1"] and result["reference"] == "0"
    fit = result["families"][0]
    assert fit["reri"] and fit["multiplicative"]
    low = result["low"]
    assert result["sentence"] == (
        f"The interaction of `fiber` and `supplement` on `dm` was declared before the estimates "
        f"were seen: the joint effects of {result['contrast']} and of `supplement` against a "
        f"single reference (`fiber` at {low:.4g}, `supplement` at 0), and `fiber`'s effect within "
        f"each level of `supplement`; interaction on the additive scale as the relative excess "
        f"risk due to interaction (RERI) with a delta-method interval (Hosmer & Lemeshow 1992, "
        f"Epidemiology 3:452) and on the multiplicative scale as the ratio of odds ratios (Knol & "
        f"VanderWeele 2012, Int J Epidemiol 41:514), adjusted for the confounders of `fiber` "
        f"(`age`, `sex`, `smoking` and `activity`) and of `supplement` as its own answers derive "
        f"them (`bmi`); the heterogeneity test is the Wald test that the 1 product term is zero.")


# ── under multiple imputation: the products are in the imputation model ──────


def test_7_under_multiple_imputation_the_products_are_in_the_imputation_model(tmp_path):
    """MODELING_SEQUENCE §1.1: interactions use SMC-FCS. The modification's copies are drawn with
    the product terms in the substantive model (never the primary fit's copies, which lack them);
    every quantity is pooled by Rubin's rules and the heterogeneity test by D1."""
    frame = est_f.cohort(n=600, seed=23)
    rng = np.random.default_rng(230)
    frame.loc[rng.random(len(frame)) < 0.2, "age"] = np.nan
    st = est_f.state(target="glucose", task="regression", measure="mean_difference",
                     missing=d.MissingSpec(strategy="multiple_imputation"),
                     modifications={"sex": d.ModificationSpec(kind="effect_modification",
                                                              exposure="fiber")})
    out = _run(frame, tmp_path, st)
    fit = out["modifications"][0]["families"][0]
    assert fit["pooled"].startswith("Multiple imputation compatible with the product terms "
                                    "(SMC-FCS with them in the substantive model), m = ")
    assert fit["pooled"].endswith("each quantity pooled by Rubin's rules, the heterogeneity test "
                                  "by D1")
    assert fit["heterogeneity"]["caption"].startswith("Pooled over ")
    assert "Li, Raghunathan & Rubin's D1" in fit["heterogeneity"]["caption"]
    assert out["modifications"][0]["sentence"].endswith(
        "; multiple imputation compatible with the product terms (SMC-FCS with them in the "
        f"substantive model), m = {fit['pooled'].split('m = ')[1].split(':')[0]}: each quantity "
        "pooled by Rubin's rules, the heterogeneity test by D1.")


# ── declared before the estimates, or suggested by data inspection ───────────


def test_7_a_modifier_declared_after_the_estimates_is_suggested_by_data_inspection():
    st = est_f.state(target="dm", task="binary", event="yes", measure="odds_ratio",
                     plan_locked=True)
    ctx = {"state": st, "columns": list(est_f.cohort(n=20).columns)}
    done = d.validate({"kind": "set_modification", "modifier": "smoking"}, ctx)
    assert done.post_hoc is True and done.exposure == "fiber"
    from turbotab.core.voice import sentence_for

    assert sentence_for(done, st) == (
        "Effect modification of `fiber`'s effect by `smoking` was suggested by data inspection, "
        "reported within each level of `smoking` against a single reference on the additive and "
        "multiplicative scales (Knol & VanderWeele 2012, Int J Epidemiol 41:514).")
    later = st.model_copy(update={"modifications": {
        "sex": d.ModificationSpec(kind="effect_modification", exposure="fiber"),
        "smoking": done.spec()}})
    count = ix.family_count(later)
    assert count["n_tests"] == 3 and count["post_hoc"] == ["smoking"]
    assert count["statement"] == (
        "3 tests in the family: the exposure's effect, 1 declared heterogeneity test (`sex`) and "
        "1 suggested by data inspection (`smoking`); the p-values are reported unadjusted with the "
        "number of tests stated")


def test_7_modification_is_an_inference_question_about_one_declared_exposure():
    pred = est_f.state(target="dm", task="binary", event="yes",
                       measure="odds_ratio").model_copy(update={"purpose": "prediction"})
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_modification", "modifier": "sex"}, {"state": pred})
    assert refused.value.code == "modification_for_inference"
    st = est_f.state(target="dm", task="binary", event="yes", measure="odds_ratio")
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_modification", "modifier": "fiber"}, {"state": st})
    assert refused.value.code == "modifier_is_exposure_or_outcome"
    assert ix.modification_gate(st) == (
        "skipped", "No effect modifier or second study factor is declared; one can be, before the "
                   "estimates are seen (afterwards it is labeled suggested by data inspection).")
    assert ix.modification_gate(pred)[0] == "not_applicable"
