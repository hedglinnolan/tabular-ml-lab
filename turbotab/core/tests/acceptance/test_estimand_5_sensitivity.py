"""ESTIMAND · 5 · sensitivity to unmeasured confounding (MODELING_SEQUENCE §0 ruling 10).

"Unmeasured-confounding sensitivity is in v2, … offered for every inference result and required in
the causal lane: the E-value for ratio measures, never with a pass/fail threshold; the
Cinelli–Hazlett robustness value for linear outcomes, benchmarked against a named measured
confounder, ranked first."

References, each R's own package on the same numbers or rows (``r_reference.run_r``):

* ``EValue::evalues.OR`` (rare and common), ``evalues.HR`` (rare and common), ``evalues.RR`` and
  ``evalues.OLS``: the E-value of the estimate and of the confidence limit nearer the null, to
  1e-8, including an interval that crosses the null (its E-value is 1);
* ``sensemakr::sensemakr(lm(...), treatment, benchmark_covariates, kd = 1)``: the robustness
  values (q = 1, α = 1 and α = 0.05), the exposure's partial R², and each benchmark's bounds and
  adjusted estimate and interval, to 1e-8, with a single-column benchmark (``smoking``) and a
  three-level one (its two indicators as one group, as ``benchmark_covariates = list(...)`` takes
  it).

Source check (read 2026-10-05 in the R packages' own code, ``EValue`` 4.1.4 and ``sensemakr``
0.1.6, which implement VanderWeele & Ding 2017 and Cinelli & Hazlett 2020): the conversions an
E-value applies to an odds ratio (``sqrt(OR)`` when the outcome is common), to a hazard ratio
(``(1 − 0.5^√HR)/(1 − 0.5^√(1/HR))``) and to a linear coefficient (``exp(0.91 · β δ / sd)``, the
interval ``exp(0.91 d ± 1.78 se_d)``), and sensemakr's bound ``r2dz.x = kd · r2dxj / (1 − r2dxj)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core.models import effects
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r

CASES = [  # (measure, estimate, low, high, rare)
    ("OR", 1.8, 1.3, 2.5, True),
    ("OR", 1.8, 1.3, 2.5, False),
    ("OR", 0.62, 0.45, 0.85, False),
    ("OR", 1.25, 0.9, 1.7, False),  # the interval crosses the null: its E-value is 1
    ("HR", 1.45, 1.12, 1.88, True),
    ("HR", 1.45, 1.12, 1.88, False),
    ("HR", 0.7, 0.55, 0.9, False),
    ("RR", 2.1, 1.5, 2.9, None),
    ("RR", 0.8, 0.66, 0.97, None),
]

EVALUE_R = """
library(EValue)
cases <- read.csv(cases_csv, stringsAsFactors = FALSE)
res <- lapply(seq_len(nrow(cases)), function(i) {
  c <- cases[i, ]
  e <- suppressMessages(switch(c$measure,
    OR = evalues.OR(c$est, c$lo, c$hi, rare = as.logical(c$rare)),
    HR = evalues.HR(c$est, c$lo, c$hi, rare = as.logical(c$rare)),
    RR = evalues.RR(c$est, c$lo, c$hi)))
  lim <- if (is.na(e[2, 2])) e[2, 3] else e[2, 2]
  list(point = e[2, 1], limit = lim)
})
ols <- suppressMessages(evalues.OLS(est = 0.42, se = 0.11, sd = 2.3))
out(list(cases = res, ols_point = ols[2, 1], ols_limit = if (is.na(ols[2, 2])) ols[2, 3] else ols[2, 2],
         ols_rr = ols[1, 1], ols_lo = ols[1, 2], ols_hi = ols[1, 3]))
"""


@needs_r
def test_5_the_e_value_agrees_with_r_evalue(tmp_path):
    """The E-value of each estimate and of the limit nearer the null agrees with R's ``EValue`` to
    1e-8; an interval that crosses the null has E-value 1; and nothing is a pass or a fail."""
    frame = pd.DataFrame([{"measure": m, "est": e, "lo": lo, "hi": hi,
                           "rare": "NA" if r is None else str(r).upper()}
                          for m, e, lo, hi, r in CASES])
    r = run_r(EVALUE_R, {"cases": frame}, tmp_path)
    for (measure, est, lo, hi, rare), ref in zip(CASES, r["cases"]):
        found = effects.e_values(est, lo, hi, measure=measure, rare=rare)
        assert found["point"] == pytest.approx(ref["point"], rel=1e-8), (measure, est, rare)
        assert found["limit"] == pytest.approx(ref["limit"], rel=1e-8), (measure, est, rare)
        assert not {"pass", "fail", "verdict", "threshold"} & set(found)
    crossing = effects.e_values(1.25, 0.9, 1.7, measure="OR", rare=False)
    assert crossing["limit"] == 1.0 and crossing["interval_includes_null"] is True
    ols = effects.e_values(0.42, measure="OLS", sd=2.3, se=0.11)
    assert ols["point"] == pytest.approx(r["ols_point"], rel=1e-8)
    assert ols["limit"] == pytest.approx(r["ols_limit"], rel=1e-8)
    assert (ols["rr"], ols["rr_low"], ols["rr_high"]) == pytest.approx(
        (r["ols_rr"], r["ols_lo"], r["ols_hi"]), rel=1e-8)


def _cohort(n: int = 800, seed: int = 21) -> pd.DataFrame:
    """A linear outcome confounded by smoking and a three-level income band."""
    rng = np.random.default_rng(seed)
    smoking = rng.binomial(1, 0.3, n).astype(float)
    income = rng.choice(["low", "middle", "high"], n, p=[0.3, 0.4, 0.3])
    age = rng.normal(50, 10, n)
    inc = (income == "middle") * 0.5 + (income == "high") * 1.0
    fiber = 20 - 3 * smoking + 2.5 * inc + 0.05 * (age - 50) + rng.normal(0, 4, n)
    glucose = 100 - 0.4 * fiber + 6 * smoking - 2 * inc + 0.3 * (age - 50) + rng.normal(0, 8, n)
    return pd.DataFrame({"glucose": glucose, "fiber": fiber, "smoking": smoking, "age": age,
                         "income": income})


SENSEMAKR_R = """
suppressPackageStartupMessages(library(sensemakr))
d <- read.csv(cohort_csv)
d$income <- factor(d$income, levels = c("high", "low", "middle"))
m <- lm(glucose ~ fiber + smoking + age + income, data = d)
s <- sensemakr(m, treatment = "fiber", benchmark_covariates = "smoking", kd = 1)
g <- sensemakr(m, treatment = "fiber",
               benchmark_covariates = list(income = c("incomelow", "incomemiddle")), kd = 1)
st <- s$sensitivity_stats
b <- s$bounds; bg <- g$bounds
out(list(estimate = st$estimate, se = st$se, t = st$t_statistic, dof = st$dof,
         r2yd = st$r2yd.x, rv_q = st$rv_q, rv_qa = st$rv_qa,
         b_r2dz = b$r2dz.x, b_r2yz = b$r2yz.dx, b_est = b$adjusted_estimate, b_se = b$adjusted_se,
         b_lo = b$adjusted_lower_CI, b_hi = b$adjusted_upper_CI,
         g_r2dz = bg$r2dz.x, g_r2yz = bg$r2yz.dx, g_est = bg$adjusted_estimate,
         g_lo = bg$adjusted_lower_CI, g_hi = bg$adjusted_upper_CI))
"""


@needs_r
def test_5_the_robustness_value_and_its_benchmarks_agree_with_r_sensemakr(tmp_path):
    """For a linear outcome the robustness values, the exposure's partial R² and each benchmark's
    bounds and adjusted estimate agree with R's ``sensemakr`` to 1e-8: ``smoking`` (one column) and
    ``income`` (two indicators, one group). The reading names the benchmark: "an unmeasured
    confounder as strong as `smoking`"."""
    frame = _cohort()
    matrix = pd.DataFrame({"fiber": frame["fiber"], "smoking": frame["smoking"], "age": frame["age"],
                           "income_low": (frame["income"] == "low").astype(float),
                           "income_middle": (frame["income"] == "middle").astype(float)})
    found = effects.linear_sensitivity(
        matrix, frame["glucose"].to_numpy(), "fiber",
        {"smoking": ["smoking"], "income": ["income_low", "income_middle"]})
    r = run_r(SENSEMAKR_R, {"cohort": frame}, tmp_path)
    assert found.estimate == pytest.approx(r["estimate"], rel=1e-8)
    assert found.se == pytest.approx(r["se"], rel=1e-8)
    assert found.t == pytest.approx(r["t"], rel=1e-8)
    assert found.dof == r["dof"]
    assert found.partial_r2 == pytest.approx(r["r2yd"], rel=1e-8)
    assert found.rv == pytest.approx(r["rv_q"], rel=1e-8)
    assert found.rv_alpha == pytest.approx(r["rv_qa"], rel=1e-8)
    smoking, income = found.benchmarks
    assert smoking.covariate == "smoking" and income.covariate == "income"
    for mine, key in ((smoking, "b"), (income, "g")):
        assert mine.r2dz == pytest.approx(r[f"{key}_r2dz"], rel=1e-8)
        assert mine.r2yz == pytest.approx(r[f"{key}_r2yz"], rel=1e-8)
        assert mine.estimate == pytest.approx(r[f"{key}_est"], rel=1e-8)
        assert mine.ci_low == pytest.approx(r[f"{key}_lo"], rel=1e-8)
        assert mine.ci_high == pytest.approx(r[f"{key}_hi"], rel=1e-8)
    assert smoking.se == pytest.approx(r["b_se"], rel=1e-8)


STAGE_R = """
suppressPackageStartupMessages(library(sensemakr)); library(EValue)
d <- read.csv(rows_csv)
d$sex <- factor(d$sex, levels = c("female", "male"))
m <- lm(glucose ~ fiber + age + sex + smoking + activity, data = d)
s <- sensemakr(m, treatment = "fiber", kd = 1,
               benchmark_covariates = list(age = "age", sex = "sexmale", smoking = "smoking",
                                           activity = "activity"))
st <- s$sensitivity_stats; b <- s$bounds
out(list(rv_q = st$rv_q, rv_qa = st$rv_qa, r2yd = st$r2yd.x, labels = b$bound_label,
         r2dz = b$r2dz.x, r2yz = b$r2yz.dx, est = b$adjusted_estimate, lo = b$adjusted_lower_CI,
         hi = b$adjusted_upper_CI))
"""


@needs_r
def test_5_the_stage_offers_it_for_a_linear_outcome_robustness_value_first(tmp_path):
    """Through the effects stage on a linear outcome: the robustness value leads, with every
    adjusted covariate as a named benchmark, each agreeing with ``sensemakr`` to 1e-8; the reading
    names the benchmark that moves the estimate most ("one as strong as `smoking`"), and the E-value
    follows (``evalues.OLS`` with the outcome's standard deviation)."""
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    frame = ef.cohort(700, seed=5)
    roles = {c: r for c, r in ef.ROLES.items() if c != "bmi"}
    answers = {c: a for c, a in ef.ANSWERS.items() if c != "bmi"}
    st = ef.state(target="glucose", task="regression", measure="mean_difference", roles=roles,
                  answers=answers)
    run = ef.run(frame, tmp_path / "stage", st, fit=False)
    [sens] = run["effects"]["families"][0]["sensitivity"]
    assert sens["methods"] == ["robustness_value", "e_value"]
    r = run_r(STAGE_R, {"rows": frame}, tmp_path / "r")
    rob = sens["robustness"]
    assert rob["rv"] == pytest.approx(r["rv_q"], rel=1e-8)
    assert rob["rv_alpha"] == pytest.approx(r["rv_qa"], rel=1e-8)
    assert rob["partial_r2"] == pytest.approx(r["r2yd"], rel=1e-8)
    names = {"1x age": "age", "1x sex": "sex", "1x smoking": "smoking", "1x activity": "activity"}
    ours = {b["covariate"]: b for b in rob["benchmarks"]}
    assert set(ours) == set(names.values())
    for label, r2dz, r2yz, est, lo, hi in zip(r["labels"], r["r2dz"], r["r2yz"], r["est"], r["lo"],
                                              r["hi"]):
        mine = ours[names[label]]
        assert (mine["r2dz"], mine["r2yz"]) == pytest.approx((r2dz, r2yz), rel=1e-8), label
        assert (mine["estimate"], mine["ci_low"], mine["ci_high"]) == pytest.approx(
            (est, lo, hi), rel=1e-8), label
    strongest = max(rob["benchmarks"], key=lambda b: abs(rob["estimate"] - b["estimate"]))
    assert strongest["covariate"] == "smoking"
    assert f"one as strong as `smoking` would move it to {strongest['estimate']:.4g}" in sens["reading"]
    assert "pass or a fail" in sens["reading"] and sens["e_value"]["measure"] == "OLS"


EVALUE_STAGE_R = """
library(EValue)
v <- read.csv(values_csv)
or <- suppressMessages(evalues.OR(v$or, v$or_lo, v$or_hi, rare = FALSE))
rr <- suppressMessages(evalues.RR(v$rr, v$rr_lo, v$rr_hi))
lim <- function(e) if (is.na(e[2, 2])) e[2, 3] else e[2, 2]
out(list(or_point = or[2, 1], or_limit = lim(or), rr_point = rr[2, 1], rr_limit = lim(rr)))
"""


@needs_r
def test_5_the_stage_reports_the_e_value_of_the_ratio_it_shows(tmp_path):
    """A common yes/no outcome: under the conditional odds ratio the E-value converts it (the
    outcome is not rare, ``rare = FALSE``); under the declared marginal measure it is the marginal
    risk ratio's own. Both agree with ``EValue`` on the stage's own estimates to 1e-8."""
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    frame = ef.cohort(800, seed=7, prevalence=0.3)
    by_or = ef.run(frame, tmp_path / "or", ef.state(target="dm", task="binary", event="yes",
                                                    measure="odds_ratio"), fit=False)
    by_rd = ef.run(frame, tmp_path / "rd", ef.state(target="dm", task="binary", event="yes",
                                                    measure="risk_difference"), fit=False)
    [s_or] = by_or["effects"]["families"][0]["sensitivity"]
    [s_rd] = by_rd["effects"]["families"][0]["sensitivity"]
    row = ef.sequence(by_or["effects"])["model_2"]["effects"][0]
    [c] = by_rd["effects"]["families"][0]["marginal"]["contrasts"]
    values = pd.DataFrame([{"or": row["ratio"], "or_lo": row["ratio_low"], "or_hi": row["ratio_high"],
                            "rr": c["rr"], "rr_lo": c["rr_low"], "rr_hi": c["rr_high"]}])
    r = run_r(EVALUE_STAGE_R, {"values": values}, tmp_path / "r")
    assert s_or["e_value"]["rare"] is False and s_or["e_value"]["converted"] is True
    assert s_or["e_value"]["point"] == pytest.approx(r["or_point"], rel=1e-8)
    assert s_or["e_value"]["limit"] == pytest.approx(r["or_limit"], rel=1e-8)
    assert s_rd["e_value"]["measure"] == "RR" and s_rd["e_value"]["converted"] is False
    assert s_rd["e_value"]["point"] == pytest.approx(r["rr_point"], rel=1e-8)
    assert s_rd["e_value"]["limit"] == pytest.approx(r["rr_limit"], rel=1e-8)
    assert s_rd["reading"].startswith("E-value for the marginal risk ratio:")


def test_5_the_function_the_causal_lane_calls_ranks_the_robustness_value_first():
    """``unmeasured_confounding`` is the exported entry (the causal package calls it): for a linear
    outcome the robustness value leads and the E-value follows; for a ratio the E-value alone; an
    odds ratio of a common outcome is converted, a rare one read as a risk ratio; a risk
    difference has no E-value without its risks."""
    frame = _cohort(seed=4)
    matrix = frame[["fiber", "smoking", "age"]]
    linear = effects.unmeasured_confounding(
        measure="mean_difference", estimate=-0.4, ci_low=-0.5, ci_high=-0.3, se=0.05,
        outcome_sd=float(frame["glucose"].std()), matrix=matrix,
        y=frame["glucose"].to_numpy(), exposure_column="fiber", benchmarks={"smoking": ["smoking"]})
    assert linear["methods"] == ["robustness_value", "e_value"]
    assert linear["robustness"]["benchmarks"][0]["covariate"] == "smoking"
    common = effects.unmeasured_confounding(measure="odds_ratio", estimate=1.8, ci_low=1.3,
                                            ci_high=2.5, outcome_share=0.32)
    rare = effects.unmeasured_confounding(measure="odds_ratio", estimate=1.8, ci_low=1.3,
                                          ci_high=2.5, outcome_share=0.05)
    assert common["methods"] == ["e_value"] and common["e_value"]["converted"] is True
    assert common["e_value"]["rr"] == pytest.approx(np.sqrt(1.8))
    assert rare["e_value"]["converted"] is False and rare["e_value"]["rr"] == 1.8
    with pytest.raises(ValueError):
        effects.unmeasured_confounding(measure="risk_difference", estimate=0.05, ci_low=0.01,
                                       ci_high=0.09, outcome_share=0.3)
