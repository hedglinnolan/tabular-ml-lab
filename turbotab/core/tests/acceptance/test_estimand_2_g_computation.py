"""ESTIMAND · 2 · marginal standardization (g-computation) to a risk difference and a risk ratio
(MODELING_SEQUENCE §0 ruling 9: "Marginal standardization (g-computation) to an RD or RR is
offered, ranked first for a common binary outcome").

On a cohort whose yes/no outcome is common (about 30%), declared with the marginal risk difference,
the ``effects`` stage standardizes the logistic model over the analyzed rows:

* the point estimates agree with a NumPy computation written out here (Newton–Raphson on the
  design built by hand, the mean predicted risk under each setting) to 1e-8, and with R (``glm``
  with a binomial family, then ``mean(predict(…, type = "response"))`` on the data with the
  exposure set each way) to 1e-8: a continuous exposure one unit higher than observed, and a
  two-valued exposure set to each value;
* its 95% interval is the percentile interval of the bootstrap that refits the whole chain on each
  resample: recomputed here on the same resamples (``numpy.random.default_rng(seed)``, n rows drawn
  with replacement each time, the convention the stage documents) by statsmodels' logistic fit and
  NumPy's quantile, to 1e-8;
* the marginal measures rank first on the estimand card when the event's share is above 10% (Zhang
  & Yu 1998) and second below it.

Source check (Zhang J, Yu KF. What's the relative risk? JAMA 1998;280:1690–1691, read as its
abstract on 2026-10-05): the odds ratio "overestimates" the relative risk when the incidence of the
outcome is common, and they flag an incidence above 10%. The rule here ranks; it refuses nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core import decisions as d, estimand
from turbotab.core.models.effects import BOOT
from turbotab.core.tests.acceptance import estimand_fixtures as ef
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r

COVARIATES = ["age", "sex", "smoking", "activity"]  # the primary set (bmi: unknown timing, beside)


def _newton(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Logistic maximum likelihood, Newton–Raphson written out (no step halving needed here)."""
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ beta))
        step = np.linalg.solve((X.T * (p * (1 - p))) @ X, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-14:
            break
    return beta


def _risks(X0: np.ndarray, X1: np.ndarray, beta: np.ndarray) -> tuple[float, float]:
    return float(np.mean(1 / (1 + np.exp(-X0 @ beta)))), float(np.mean(1 / (1 + np.exp(-X1 @ beta))))


@pytest.fixture(scope="module")
def common(tmp_path_factory):
    frame = ef.cohort(800, seed=7, prevalence=0.3)
    st = ef.state(target="dm", task="binary", event="yes", measure="risk_difference",
                  model_1=["age", "sex"])
    return frame, ef.run(frame, tmp_path_factory.mktemp("gcomp"), st, fit=False)


R_GLM = """
d <- read.csv(rows_csv)
d$sex <- factor(d$sex, levels = c("female", "male"))
d$y <- as.numeric(d$dm == "yes")
m <- glm(y ~ fiber + age + sex + smoking + activity, data = d, family = binomial,
         control = glm.control(epsilon = 1e-15, maxit = 100))
d1 <- d; d1$fiber <- d1$fiber + 1
r0 <- mean(predict(m, newdata = d, type = "response"))
r1 <- mean(predict(m, newdata = d1, type = "response"))
m2 <- glm(y ~ supplement + age + sex + smoking + activity, data = d, family = binomial,
          control = glm.control(epsilon = 1e-15, maxit = 100))
e0 <- d; e0$supplement <- 0; e1 <- d; e1$supplement <- 1
s0 <- mean(predict(m2, newdata = e0, type = "response"))
s1 <- mean(predict(m2, newdata = e1, type = "response"))
out(list(r0 = r0, r1 = r1, s0 = s0, s1 = s1))
"""


def test_2_the_point_estimates_agree_with_numpy_by_hand(common):
    frame, run = common
    [contrast] = run["effects"]["families"][0]["marginal"]["contrasts"]
    assert contrast["setting"] == "one unit higher than observed"
    y = (frame["dm"] == "yes").astype(float).to_numpy()
    X = ef.design_matrix(frame, ["fiber", *COVARIATES])
    beta = _newton(X.to_numpy(), y)
    X1 = X.copy()
    X1["fiber"] += 1
    r0, r1 = _risks(X.to_numpy(), X1.to_numpy(), beta)
    assert contrast["risk_low"] == pytest.approx(r0, rel=1e-8)
    assert contrast["risk_high"] == pytest.approx(r1, rel=1e-8)
    assert contrast["rd"] == pytest.approx(r1 - r0, rel=1e-8)
    assert contrast["rr"] == pytest.approx(r1 / r0, rel=1e-8)
    # the logistic model's maximum likelihood with an intercept reproduces the event's share
    assert r0 == pytest.approx(y.mean(), rel=1e-10)


@needs_r
def test_2_the_point_estimates_agree_with_r_glm(common, tmp_path):
    frame, run = common
    r = run_r(R_GLM, {"rows": frame}, tmp_path)
    [contrast] = run["effects"]["families"][0]["marginal"]["contrasts"]
    assert contrast["rd"] == pytest.approx(r["r1"] - r["r0"], rel=1e-8)
    assert contrast["rr"] == pytest.approx(r["r1"] / r["r0"], rel=1e-8)
    # a two-valued exposure is set to each of its values
    roles = {**ef.ROLES, "supplement": "exposure"}
    roles.pop("fiber")
    st = ef.state(target="dm", task="binary", event="yes", exposure="supplement",
                  measure="risk_ratio", roles=roles,
                  amounts={**ef.AMOUNTS, "code_or_count:supplement": "amount"})
    found = ef.run(frame, tmp_path / "supplement", st, fit=False)
    marginal = found["effects"]["families"][0]["marginal"]
    assert marginal["declared"] == "risk_ratio"
    [two] = marginal["contrasts"]
    assert two["setting"] == "`1` against `0`"
    assert two["risk_low"] == pytest.approx(r["s0"], rel=1e-8)
    assert two["risk_high"] == pytest.approx(r["s1"], rel=1e-8)
    assert two["rd"] == pytest.approx(r["s1"] - r["s0"], rel=1e-8)
    assert two["rr"] == pytest.approx(r["s1"] / r["s0"], rel=1e-8)


def test_2_the_interval_is_the_whole_chain_bootstrap_percentile(common):
    """The stage's 95% interval is the percentile interval over its resamples, each refitting the
    chain; recomputed here on the same resamples with statsmodels' logistic fit."""
    frame, run = common
    [contrast] = run["effects"]["families"][0]["marginal"]["contrasts"]
    assert contrast["n_boot"] == BOOT and contrast["n_failed"] == 0 and contrast["by_unit"] is None
    y = (frame["dm"] == "yes").astype(float).to_numpy()
    X = ef.design_matrix(frame, ["fiber", *COVARIATES]).to_numpy()
    X1 = X.copy()
    X1[:, 1] += 1
    rng = np.random.default_rng(run["state"].split.seed)
    rds, rrs = [], []
    for _ in range(BOOT):
        idx = rng.integers(0, len(y), len(y))
        fit = sm.Logit(y[idx], X[idx]).fit(disp=0, method="newton", tol=1e-14, maxiter=100)
        r0, r1 = _risks(X[idx], X1[idx], np.asarray(fit.params))
        rds.append(r1 - r0)
        rrs.append(r1 / r0)
    lo, hi = np.percentile(rds, [2.5, 97.5])
    assert contrast["rd_low"] == pytest.approx(lo, rel=1e-8)
    assert contrast["rd_high"] == pytest.approx(hi, rel=1e-8)
    lo, hi = np.percentile(rrs, [2.5, 97.5])
    assert contrast["rr_low"] == pytest.approx(lo, rel=1e-8)
    assert contrast["rr_high"] == pytest.approx(hi, rel=1e-8)
    assert contrast["rd_low"] < contrast["rd"] < contrast["rd_high"]


def test_2_the_marginal_measures_rank_first_for_a_common_outcome():
    """Above 10% the marginal risk difference and ratio lead the card; below, the conditional odds
    ratio leads and the marginal ones follow, still offered (a ranking, never a refusal)."""
    common = ef.cohort(4000, seed=1, prevalence=0.3)
    rare = ef.cohort(4000, seed=1, prevalence=0.04)
    st = ef.state(target="dm", task="binary", event="yes", measure="odds_ratio")
    share_common = estimand.event_share(common["dm"], "yes")
    share_rare = estimand.event_share(rare["dm"], "yes")
    assert share_common == pytest.approx((common["dm"] == "yes").mean())
    assert share_common > 0.10 > share_rare
    first = estimand.estimand_card(st, "binary", prevalence=share_common)["measures"]
    second = estimand.estimand_card(st, "binary", prevalence=share_rare)["measures"]
    assert [m["measure"] for m in first] == ["risk_difference", "risk_ratio", "odds_ratio"]
    assert [m["measure"] for m in second] == ["odds_ratio", "risk_difference", "risk_ratio"]
    assert all(m["fitted"] for m in first + second)
    assert "Zhang & Yu 1998" in first[0]["reason"] and "Zhang" not in second[1]["reason"]
    # the event's share is the event's, not the minority class's
    assert estimand.event_share(common["dm"], "no") == pytest.approx(1 - share_common)


def test_2_rows_that_repeat_are_resampled_by_whole_unit(tmp_path):
    """When a person's rows repeat, the bootstrap draws whole people (every row of a drawn person,
    as often as the person is drawn) and refits the chain; recomputed here by drawing the people
    with the same stream and refitting with statsmodels."""
    one = ef.cohort(300, seed=13, prevalence=0.3)
    frame = pd.concat([one, one.assign(age=one["age"] + 1.0)], ignore_index=True)
    frame["dm"] = np.where(np.random.default_rng(2).random(len(frame)) < 0.33, "yes", "no")
    st = ef.state(target="dm", task="binary", event="yes", measure="risk_difference",
                  grain=d.GrainSpec(grain="repeated", id_column="pid"))
    run = ef.run(frame, tmp_path, st, fit=False)
    [c] = run["effects"]["families"][0]["marginal"]["contrasts"]
    assert c["by_unit"] == "pid"
    y = (frame["dm"] == "yes").astype(float).to_numpy()
    X = ef.design_matrix(frame, ["fiber", *COVARIATES]).to_numpy()
    X1 = X.copy()
    X1[:, 1] += 1
    codes, uniques = pd.factorize(frame["pid"])
    people = [np.flatnonzero(codes == g) for g in range(len(uniques))]
    rng = np.random.default_rng(st.split.seed)
    rds = []
    for _ in range(BOOT):
        idx = np.concatenate([people[g] for g in rng.integers(0, len(people), len(people))])
        fit = sm.Logit(y[idx], X[idx]).fit(disp=0, method="newton", tol=1e-14, maxiter=100)
        r0, r1 = _risks(X[idx], X1[idx], np.asarray(fit.params))
        rds.append(r1 - r0)
    lo, hi = np.percentile(rds, [2.5, 97.5])
    assert (c["rd_low"], c["rd_high"]) == pytest.approx((lo, hi), rel=1e-8)


def test_2_a_surveyed_population_blocks_the_marginal_risks_and_records_the_way_out(tmp_path):
    """No design-based standardization is built, so under the surveyed-population answer the
    marginal risks are not served as if they were the population's (MODELING_SEQUENCE §4:
    block and record, exit the sample-only attestation); the conditional odds ratio stays."""
    frame = ef.cohort(400, seed=17, prevalence=0.3)
    rng = np.random.default_rng(5)
    frame["sampling_weight"] = rng.uniform(0.5, 3.0, len(frame))
    frame["stratum"] = rng.integers(1, 5, len(frame))
    frame["psu"] = frame["stratum"] * 10 + rng.integers(1, 3, len(frame))
    roles = {**ef.ROLES, "sampling_weight": "design", "stratum": "design", "psu": "design"}
    survey = d.SurveySpec(estimand="population", weight="sampling_weight", strata="stratum",
                          psu="psu")
    st = ef.state(target="dm", task="binary", event="yes", measure="risk_difference", roles=roles,
                  survey=survey, amounts={**ef.AMOUNTS, "code_or_count:stratum": "code",
                                          "code_or_count:psu": "code"})
    run = ef.run(frame, tmp_path, st, fit=False)
    marginal = run["effects"]["families"][0]["marginal"]
    assert marginal["contrasts"] == [] and marginal["refused"].startswith(
        "The marginal risks are standardized over these participants")
    sample, conditional = (e["decision"] for e in marginal["exits"])
    assert sample["kind"] == "set_survey" and sample["estimand"] == "sample"
    assert conditional["kind"] == "set_estimand" and conditional["measure"] == "odds_ratio"
    primary = ef.sequence(run["effects"])["model_2"]
    assert primary["effects"][0]["ratio"] is not None
    assert primary["inference"]["covariance"] == "design"
