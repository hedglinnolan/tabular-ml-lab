"""D6: design-based trends across stacked NHANES cycles (`turbotab.core.methods.cycle_trends`).

Every number is held to R survey on a seeded four-cycle NHANES-shaped stack (2011–2012 to 2017–March
2020, the last 3.2 years long): ``svyby(…, svymean, covmat = TRUE)`` and ``degf`` of each cycle's
subset for the cycle estimates; ``svyby(…, svyciprop, method = "beta")`` for the Korn–Graubard
intervals; ``svycontrast`` with ``contr.poly(T, scores = times)`` for the orthogonal polynomial
contrasts; ``svyglm`` (gaussian and quasibinomial) with the cycle's midpoint year as a predictor for
the polynomial and joinpoint regressions, and ``svycontrast`` for a later segment's slope. P-values
are R's ``pt`` on ``degf`` (the design's degrees of freedom, as NCHS asks), not ``svyglm``'s
residual degrees of freedom. The contrast coefficients for equally spaced cycles are also held to
the classical table (−3, −1, 1, 3)/√20, (1, −1, −1, 1)/2, (−1, 3, −3, 1)/√20.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from turbotab.core.contracts import contract
from turbotab.core.methods import cycle_trends as T
from turbotab.core.methods import cycles as C
from turbotab.core.tests.acceptance.cycle_fixtures import four_cycles, nhanes_cycle
from turbotab.core.tests.acceptance.survey_r import needs_r, run_r

TIMES = [2012.0, 2014.0, 2016.0, 2018.6]
LABELS = ["2011–2012", "2013–2014", "2015–2016", "2017–March 2020"]

R_STACK = """
f <- read.csv("stacked.csv")
f$w <- ifelse(f$cycle_release == 66, 3.2 / 9.2, 2 / 9.2) * f$cycle_weight
f$time <- c(`7` = 2012, `8` = 2014, `9` = 2016, `66` = 2018.6)[as.character(f$cycle_release)]
f$tc <- f$time - mean(c(2012, 2014, 2016, 2018.6))
f$y <- ifelse(f$RIDAGEYR >= 20, f$BMXBMI, NA)
f$ob <- as.numeric(f$y >= 30)
f$female <- as.numeric(f$RIAGENDR == 2)
d <- svydesign(ids = ~SDMVPSU, strata = ~interaction(cycle_release, SDMVSTRA), weights = ~w,
               nest = TRUE, data = f)
s <- subset(d, !is.na(y) & w > 0)
df <- degf(s)
p <- function(est, se) 2 * pt(-abs(est / se), df)
fit <- function(g) list(coef = unname(coef(g)), se = unname(SE(g)),
                        p = unname(p(coef(g), SE(g))))
"""


@pytest.fixture(scope="module")
def stack() -> C.Stacked:
    return C.stack_cycles(four_cycles(), weight="WTMEC2YR")


def _adult_bmi(s: C.Stacked) -> pd.Series:
    return s.frame["BMXBMI"].where(s.frame["RIDAGEYR"] >= 20)


def _obese(s: C.Stacked) -> pd.Series:
    y = _adult_bmi(s)
    return (y >= 30).astype(float).where(y.notna())


def _close(got: T.Term, coef: float, se: float, p: float, rel: float = 1e-8) -> None:
    assert got.estimate == pytest.approx(coef, rel=rel, abs=1e-12)
    assert got.se == pytest.approx(se, rel=rel)
    assert got.p == pytest.approx(p, rel=1e-6, abs=1e-12)


# ── the cycle estimates and the contrasts ────────────────────────────────────


@needs_r
def test_the_cycle_means_their_intervals_and_the_contrasts_match_svyby_and_svycontrast(stack,
                                                                                    tmp_path):
    r = T.cycle_trend(_adult_bmi(stack), stack.frame[C.CYCLE], design=stack.design(),
                      name="body mass index")
    ref = run_r(R_STACK + """
by <- svyby(~y, ~time, s, svymean, covmat = TRUE)
dfs <- sapply(sort(unique(f$time)), function(t) degf(subset(s, time == t)))
# By hand: confint(by, df = dfs) recycles a vector df and returns wrong limits when the dfs differ.
half <- qt(0.975, dfs) * SE(by)
ci <- cbind(coef(by) - half, coef(by) + half)
Z <- contr.poly(4, scores = c(2012, 2014, 2016, 2018.6))
cc <- svycontrast(by, list(lin = Z[, 1], quad = Z[, 2], cub = Z[, 3]))
out(list(mean = unname(coef(by)), se = unname(SE(by)), vcov = unname(vcov(by)), dfs = dfs,
         lo = unname(ci[, 1]), hi = unname(ci[, 2]), Z = unname(Z), df = df,
         c = unname(coef(cc)), cse = unname(as.vector(SE(cc))), cp = unname(p(coef(cc), as.vector(SE(cc))))))
""", {"stacked": stack.frame}, tmp_path)
    assert [e.cycle for e in r.estimates] == LABELS
    assert [e.time for e in r.estimates] == pytest.approx(TIMES)
    np.testing.assert_allclose([e.estimate for e in r.estimates], ref["mean"], rtol=1e-12)
    np.testing.assert_allclose([e.se for e in r.estimates], ref["se"], rtol=1e-10)
    assert [e.df for e in r.estimates] == ref["dfs"] == [9, 9, 9, 9]
    np.testing.assert_allclose([e.lower for e in r.estimates], ref["lo"], rtol=1e-10)
    np.testing.assert_allclose([e.upper for e in r.estimates], ref["hi"], rtol=1e-10)
    np.testing.assert_allclose(r.covariance, ref["vcov"], rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(T.poly_contrasts(TIMES, 3), np.array(ref["Z"]), rtol=1e-12,
                               atol=1e-14)
    assert r.df == ref["df"] == 68 - 32
    for got, c, se, pv in zip(r.terms, ref["c"], ref["cse"], ref["cp"]):
        _close(got, c, se, pv)
    assert [t.name for t in r.terms] == ["linear", "quadratic", "cubic"]
    assert r.shape == "increasing" and "rose" in r.says


@needs_r
def test_each_cycle_interval_takes_its_own_degrees_of_freedom_as_computed_by_hand_in_r(tmp_path):
    """Cycles with different numbers of strata have different design degrees of freedom (PSUs minus
    strata, Ingram et al. 2018); each cycle's t interval uses its own, mean ± qt(0.975, df_c)·SE,
    computed in R from svyby's means and standard errors and each cycle's degf."""
    files = {7: nhanes_cycle(7, strata=8), 8: nhanes_cycle(8, strata=6),
             9: nhanes_cycle(9, strata=5)}
    s = C.stack_cycles(files, weight="WTMEC2YR")
    r = T.cycle_trend(_adult_bmi(s), s.frame[C.CYCLE], design=s.design())
    ref = run_r("""
f <- read.csv("stacked.csv")
f$y <- ifelse(f$RIDAGEYR >= 20, f$BMXBMI, NA)
d <- svydesign(ids = ~SDMVPSU, strata = ~interaction(cycle_release, SDMVSTRA),
               weights = ~pooled_weight, nest = TRUE, data = f)
s <- subset(d, !is.na(y) & pooled_weight > 0)
by <- svyby(~y, ~cycle_midpoint, s, svymean)
dfs <- sapply(sort(unique(f$cycle_midpoint)), function(t) degf(subset(s, cycle_midpoint == t)))
half <- qt(0.975, dfs) * SE(by)
out(list(mean = unname(coef(by)), se = unname(SE(by)), dfs = dfs,
         lo = unname(coef(by) - half), hi = unname(coef(by) + half)))
""", {"stacked": s.frame}, tmp_path)
    assert [e.df for e in r.estimates] == ref["dfs"] == [9, 7, 6]
    np.testing.assert_allclose([e.estimate for e in r.estimates], ref["mean"], rtol=1e-12)
    np.testing.assert_allclose([e.se for e in r.estimates], ref["se"], rtol=1e-10)
    np.testing.assert_allclose([e.lower for e in r.estimates], ref["lo"], rtol=1e-10)
    np.testing.assert_allclose([e.upper for e in r.estimates], ref["hi"], rtol=1e-10)


def test_equally_spaced_contrasts_are_the_classical_orthogonal_polynomials():
    Z = T.poly_contrasts([2000, 2002, 2004, 2006], 3)
    np.testing.assert_allclose(Z[:, 0], np.array([-3, -1, 1, 3]) / math.sqrt(20), atol=1e-15)
    np.testing.assert_allclose(Z[:, 1], np.array([1, -1, -1, 1]) / 2, atol=1e-15)
    np.testing.assert_allclose(Z[:, 2], np.array([-1, 3, -3, 1]) / math.sqrt(20), atol=1e-15)


@needs_r
def test_a_prevalence_takes_korn_graubard_intervals_as_svyciprop_computes_them(stack, tmp_path):
    r = T.cycle_trend(_obese(stack), stack.frame[C.CYCLE], design=stack.design(),
                      kind="prevalence", name="obesity")
    ref = run_r(R_STACK + """
kg <- svyby(~ob, ~time, s, svyciprop, method = "beta", vartype = "ci")
out(list(p = unname(kg$ob), lo = unname(kg$ci_l), hi = unname(kg$ci_u)))
""", {"stacked": stack.frame}, tmp_path)
    np.testing.assert_allclose([e.estimate for e in r.estimates], ref["p"], rtol=1e-12)
    np.testing.assert_allclose([e.lower for e in r.estimates], ref["lo"], rtol=1e-8)
    np.testing.assert_allclose([e.upper for e in r.estimates], ref["hi"], rtol=1e-8)
    assert {e.interval for e in r.estimates} == {"korn_graubard"}
    assert "Korn–Graubard" in T.trend_sentence(r)


# ── regressions on time ──────────────────────────────────────────────────────


@needs_r
def test_the_polynomial_regression_matches_svyglm_degree_by_degree(stack, tmp_path):
    r = T.cycle_trend(_adult_bmi(stack), stack.frame[C.CYCLE], design=stack.design(),
                      method="regression")
    ref = run_r(R_STACK + """
out(list(g1 = fit(svyglm(y ~ tc, s)), g2 = fit(svyglm(y ~ tc + I(tc^2), s)),
         g3 = fit(svyglm(y ~ tc + I(tc^2) + I(tc^3), s))))
""", {"stacked": stack.frame}, tmp_path)
    for degree in (1, 2, 3):
        want = ref[f"g{degree}"]
        for got, c, se, pv in zip(r.models[degree], want["coef"], want["se"], want["p"]):
            _close(got, c, se, pv)
    assert [t.name for t in r.terms] == ["cubic", "quadratic", "linear"]
    assert r.degree == 1 and r.shape == "increasing"
    assert "per year" in r.says
    # The cubic contrast and the cubic power test the same thing with four cycles
    contrasts = T.cycle_trend(_adult_bmi(stack), stack.frame[C.CYCLE], design=stack.design())
    assert r.terms[0].t == pytest.approx(contrasts.terms[2].t, rel=1e-9)


@needs_r
def test_a_logistic_trend_with_covariates_matches_svyglm_quasibinomial(stack, tmp_path):
    cov = pd.DataFrame({"RIDAGEYR": stack.frame["RIDAGEYR"],
                        "female": (stack.frame["RIAGENDR"] == 2).astype(float)})
    r = T.cycle_trend(_obese(stack), stack.frame[C.CYCLE], design=stack.design(),
                      kind="prevalence", method="regression", scale="logit", covariates=cov,
                      degree=2)
    ref = run_r(R_STACK + """
q <- quasibinomial()
ctl <- glm.control(epsilon = 1e-13, maxit = 100)
out(list(l1 = fit(svyglm(ob ~ tc + RIDAGEYR + female, s, family = q, control = ctl)),
         l2 = fit(svyglm(ob ~ tc + I(tc^2) + RIDAGEYR + female, s, family = q, control = ctl))))
""", {"stacked": stack.frame}, tmp_path)
    for degree in (1, 2):
        want = ref[f"l{degree}"]
        for got, c, se, pv in zip(r.models[degree], want["coef"], want["se"], want["p"]):
            _close(got, c, se, pv, rel=1e-6)
    assert [t.name for t in r.models[1]] == ["intercept", "linear", "RIDAGEYR", "female"]
    assert "adjusted for RIDAGEYR, female" in T.trend_sentence(r)
    assert "log-odds" in r.says


@needs_r
def test_the_joinpoint_model_matches_svyglm_and_the_later_slope_svycontrast(stack, tmp_path):
    r = T.cycle_trend(_adult_bmi(stack), stack.frame[C.CYCLE], design=stack.design(),
                      method="joinpoint", joins=["2015–2016"])
    ref = run_r(R_STACK + """
j <- svyglm(y ~ time + pmax(time - 2016, 0), s)
later <- svycontrast(j, c(0, 1, 1))
out(list(j = fit(j), later = unname(coef(later)), later_se = unname(as.vector(SE(later)))))
""", {"stacked": stack.frame}, tmp_path)
    slope1, change, slope2 = r.terms
    _close(slope1, ref["j"]["coef"][1], ref["j"]["se"][1], ref["j"]["p"][1])
    _close(change, ref["j"]["coef"][2], ref["j"]["se"][2], ref["j"]["p"][2])
    assert slope2.estimate == pytest.approx(ref["later"], rel=1e-9)
    assert slope2.se == pytest.approx(ref["later_se"], rel=1e-9)
    assert r.joins == (2016.0,) and [t.name for t in r.terms] == ["slope 1", "change at 2016",
                                                                   "slope 2"]
    assert r.joins == T.cycle_trend(_adult_bmi(stack), stack.frame[C.CYCLE],
                                    design=stack.design(), method="joinpoint", joins=[2016]).joins


@needs_r
def test_a_bend_is_found_by_the_contrasts_and_by_the_regression(tmp_path):
    files = {code: nhanes_cycle(code, shift=s, n_per_psu=45) for code, s in
             ((5, 0.0), (6, 2.5), (7, 3.5), (8, 3.6), (9, 2.6))}
    s = C.stack_cycles(files, weight="WTMEC2YR")
    y = _adult_bmi(s)
    contrasts = T.cycle_trend(y, s.frame[C.CYCLE], design=s.design())
    regression = T.cycle_trend(y, s.frame[C.CYCLE], design=s.design(), method="regression")
    assert contrasts.shape == regression.shape == "nonlinear"
    assert regression.degree == 2
    assert "not a straight line (the quadratic term" in contrasts.says
    ref = run_r("""
f <- read.csv("stacked.csv")
f$w <- f$cycle_weight / 5
f$y <- ifelse(f$RIDAGEYR >= 20, f$BMXBMI, NA)
f$tc <- f$cycle_midpoint - 2012
d <- subset(svydesign(ids = ~SDMVPSU, strata = ~interaction(cycle_release, SDMVSTRA),
                      weights = ~w, nest = TRUE, data = f), !is.na(y) & w > 0)
by <- svyby(~y, ~cycle_midpoint, d, svymean, covmat = TRUE)
cc <- svycontrast(by, list(q = contr.poly(5)[, 2]))
g <- svyglm(y ~ tc + I(tc^2), d)
out(list(q = unname(coef(cc)), qse = unname(as.vector(SE(cc))), g = unname(coef(g))[3],
         gse = unname(SE(g))[3]))
""", {"stacked": s.frame}, tmp_path)
    assert contrasts.terms[1].estimate == pytest.approx(ref["q"], rel=1e-9)
    assert contrasts.terms[1].se == pytest.approx(ref["qse"], rel=1e-9)
    assert regression.models[2][2].estimate == pytest.approx(ref["g"], rel=1e-9)
    assert regression.models[2][2].se == pytest.approx(ref["gse"], rel=1e-9)


@needs_r
def test_a_single_cluster_stratum_and_no_design_match_r(tmp_path):
    files = four_cycles()
    files[8] = nhanes_cycle(8, lonely=True)
    s = C.stack_cycles(files, weight="WTMEC2YR")
    y = _adult_bmi(s)
    r = T.cycle_trend(y, s.frame[C.CYCLE], design=s.design())
    assert any("single sampled cluster" in c for c in r.concerns)
    plain = T.cycle_trend(y, s.frame[C.CYCLE])
    assert not plain.weighted and "these participants" in plain.estimator
    ref = run_r(R_STACK + """
by <- svyby(~y, ~time, s, svymean, covmat = TRUE)
Z <- contr.poly(4, scores = c(2012, 2014, 2016, 2018.6))
cc <- svycontrast(by, list(lin = Z[, 1]))
u <- svydesign(ids = ~1, data = f[!is.na(f$y), ])
ub <- svyby(~y, ~time, u, svymean, covmat = TRUE)
uc <- svycontrast(ub, list(lin = Z[, 1]))
out(list(se = unname(SE(by)), c = unname(coef(cc)), cse = unname(as.vector(SE(cc))),
         use = unname(SE(ub)), uc = unname(coef(uc)), ucse = unname(as.vector(SE(uc))), udf = degf(u)))
""", {"stacked": s.frame}, tmp_path)
    np.testing.assert_allclose([e.se for e in r.estimates], ref["se"], rtol=1e-10)
    assert r.terms[0].estimate == pytest.approx(ref["c"], rel=1e-10)
    assert r.terms[0].se == pytest.approx(ref["cse"], rel=1e-10)
    np.testing.assert_allclose([e.se for e in plain.estimates], ref["use"], rtol=1e-10)
    assert plain.terms[0].estimate == pytest.approx(ref["uc"], rel=1e-10)
    assert plain.terms[0].se == pytest.approx(ref["ucse"], rel=1e-10)
    assert plain.df == ref["udf"]


# ── refusals, each with exits that run ───────────────────────────────────────


def test_requests_the_guidelines_rule_out_are_refused_with_exits_that_run(stack):
    y, cyc, d = _adult_bmi(stack), stack.frame[C.CYCLE], stack.design()
    cases = [
        (dict(scale="logit"), "for a prevalence"),
        (dict(kind="prevalence", y=_obese(stack), scale="logit"), "not on the log-odds"),
        (dict(covariates=pd.DataFrame({"age": stack.frame["RIDAGEYR"]})), "cannot take other"),
        (dict(method="joinpoint", joins="search"), "not searched for here"),
        (dict(method="joinpoint", joins=["2011–2012"]), "first or the last cycle"),
        (dict(method="joinpoint", joins=[2015]), "not one of the cycles studied"),
        (dict(method="joinpoint"), "named in advance"),
        (dict(kind="prevalence"), "must be 0 or 1"),
    ]
    for kw, says in cases:
        values = kw.pop("y", y)
        with pytest.raises(T.TrendRefused) as refused:
            T.cycle_trend(values, cyc, design=d, **kw)
        assert says in str(refused.value), kw
        assert refused.value.exits, kw
        for exit_ in refused.value.exits:
            change = {k: v for k, v in exit_.items() if k in ("method", "scale", "kind",
                                                             "covariates")}
            if not change:
                continue
            retry = {**kw, **change}
            if retry.get("kind") == "mean":
                retry.pop("scale", None)
            if retry.get("covariates") is None:
                retry.pop("covariates", None)
            if retry.get("method") != "joinpoint":
                retry.pop("joins", None)
            T.cycle_trend(_obese(stack) if retry.get("kind") == "prevalence" else values, cyc,
                          design=d, **retry)


def test_a_straight_line_prevalence_that_leaves_0_to_1_is_refused_for_the_log_odds():
    rng = np.random.default_rng(5)
    cycles = np.repeat(["2011-2012", "2013-2014", "2015-2016", "2017-2018"], 400)
    p = np.repeat([0.004, 0.01, 0.15, 0.6], 400)
    y = pd.Series((rng.random(len(p)) < p).astype(float))
    with pytest.raises(T.TrendRefused) as refused:
        T.cycle_trend(y, pd.Series(cycles), kind="prevalence", method="regression", degree=1)
    assert "below 0 or above 1" in str(refused.value)
    assert refused.value.exits == [T.LOGIT_EXIT]
    T.cycle_trend(y, pd.Series(cycles), kind="prevalence", method="regression", scale="logit",
                  degree=1)


def test_one_cycle_or_no_design_degrees_of_freedom_is_refused(stack):
    one = stack.frame[C.CYCLE] == "2013–2014"
    with pytest.raises(T.TrendRefused, match="at least two cycles"):
        T.cycle_trend(_adult_bmi(stack).where(one), stack.frame[C.CYCLE], design=stack.design())
    files = {code: nhanes_cycle(code, strata=3, n_per_psu=20) for code in (8, 9)}
    for f in files.values():
        f["SDMVPSU"] = 1  # one PSU per stratum
    s = C.stack_cycles(files, weight="WTMEC2YR")
    with pytest.raises(T.TrendRefused) as refused:
        T.cycle_trend(_adult_bmi(s), s.frame[C.CYCLE], design=s.design())
    assert "no design degrees of freedom" in str(refused.value)
    assert refused.value.exits == [T.SAMPLE_EXIT]
    T.cycle_trend(_adult_bmi(s), s.frame[C.CYCLE])  # the sample-only answer


def test_a_cycle_without_a_known_time_is_refused_until_it_is_given():
    y = pd.Series(np.arange(40, dtype=float))
    cyc = pd.Series(np.repeat(["wave A", "wave B"], 20))
    with pytest.raises(T.TrendRefused) as refused:
        T.cycle_trend(y, cyc)
    assert "No time is known" in str(refused.value)
    r = T.cycle_trend(y, cyc, times={"wave A": 1.0, "wave B": 3.0})
    assert [e.time for e in r.estimates] == [1.0, 3.0] and r.terms[0].name == "linear"


# ── the goal and the contract ────────────────────────────────────────────────


def test_only_describe_offers_a_trend():
    assert T.eligible("describe").offered
    for goal in ("estimate", "predict"):
        e = T.eligible(goal)
        assert not e.offered and e.exits[0]["goal"] == "describe"


# Each relation the contract declares, and the test here that holds the code to it.
RELATION_TESTS = {
    "design_based": "test_the_cycle_means_their_intervals_and_the_contrasts_match_svyby_and_"
                    "svycontrast",
    "korn_graubard": "test_a_prevalence_takes_korn_graubard_intervals_as_svyciprop_computes_them",
    "highest_first": "test_a_bend_is_found_by_the_contrasts_and_by_the_regression",
    "contrasts_unadjusted": "test_requests_the_guidelines_rule_out_are_refused_with_exits_that_run",
    "contrasts_linear": "test_requests_the_guidelines_rule_out_are_refused_with_exits_that_run",
    "unit_interval": "test_a_straight_line_prevalence_that_leaves_0_to_1_is_refused_for_the_log_"
                     "odds",
    "no_search": "test_requests_the_guidelines_rule_out_are_refused_with_exits_that_run",
    "design_df": "test_one_cycle_or_no_design_degrees_of_freedom_is_refused",
    "describe_only": "test_only_describe_offers_a_trend",
}


def test_the_contract_declares_the_rules_the_code_enforces():
    import importlib

    c = contract("cycle_trends")
    assert c.scope == "descriptive" and c.package == "D6"
    assert {r.name for r in c.relations} == set(RELATION_TESTS)
    assert set(RELATION_TESTS.values()) <= {n for n in globals() if n.startswith("test_")}
    assert {o.key for o in c.options} == set(T.DESCRIBE_LABELS) == set(T.METHODS)
    for o in c.options:
        assert set(o.rung.values()) == {"not_offered"}
    for r in c.relations:
        module, fn = r.enforced_by.split(":")
        assert callable(getattr(importlib.import_module(module), fn))
        if r.kind == "conflicts":
            assert r.rung == "refused" and r.exits
    assert contract("stack_cycles").relation("trends").target == "cycle_trends"
    assert T.trend_sentence().startswith("The mean in each NHANES cycle")
