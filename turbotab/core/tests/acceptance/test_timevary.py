"""TIMEVARY · A time-varying exposure by g-methods (V2_DEFINITION_OF_DONE §2, the causal row).

The package's acceptance, numbered as its spec:

1. A marginal structural model with stabilized inverse-probability-of-treatment weights (and
   censoring weights) for a time-varying exposure on long data. The weights agree with R
   ``ipw::ipwtm`` to 1e-8, and the MSM estimate and its robust SE with R ``geepack::geeglm``
   (independence working correlation) to 1e-6. This holds on ``ipw``'s own ``haartdat`` (an
   exposure that, once started, stays, and loss to follow-up), on ``gfoRmula``'s
   ``basicdata_nocomp`` (an exposure that switches), on a repeated continuous outcome, and through
   the app's own stage on a cohort table.
2. The parametric g-formula. On ``gfoRmula``'s example data, the risks under "always", "never" and
   the natural course agree with R ``gfoRmula`` within Monte Carlo error, with **50,000 simulated
   units on each side**, and every fitted model's coefficients agree to 1e-6. On a simulated cohort
   whose truth is computed exactly by enumeration, the risks fall within four bootstrap standard
   errors of it.
3. Diagnostics before estimates: the weight distribution (stabilized mean near 1), the truncation
   options with their stated trade-off, and positivity at each time point. These are computed and
   shown first, and no estimate is computed, served or locked until the truncation is declared.
4. Routing. The lane needs repeated measures with a settled time column and a declared time
   ordering. Time-varying confounders affected by prior exposure require g-methods; standard
   regression adjustment for them is block and record, with an exit to this lane. A simulation
   with a known null shows why.
5. The methods sentence, asserted verbatim.

Also: the §13 contract with the chain test that every relation it declares fires, and the E-values
against R ``EValue``.

Every expected value comes from an independent path: R (``timevary_fixtures.run_r``; skipped
without ``Rscript``), statsmodels or NumPy written out here, an exact enumeration, or a simulation's
known truth. The app's values come through its own path: ``models/time_varying.py``, the
``time_varying`` stage run by the real graph, the Router, the validators and the voice.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from turbotab.core import decisions as d
from turbotab.core import plan_lock
from turbotab.core.decisions import ProjectState, Refusal
from turbotab.core.interview import route
from turbotab.core.models import time_varying as tv
from turbotab.core.tests.acceptance.timevary_fixtures import (exact_risks, feedback_cohort,
                                                              gformula_cohort, needs_r,
                                                              r_dataset,
                                                              repeated_measure_cohort, run_r)
from turbotab.core.tests.graph_runner import GraphRun

A = d.CovariateAnswers
ALL_FRESH = {s: {"status": "fresh"} for s in (
    "ingest", "oriented", "profile", "findings", "structure", "working", "target_info", "roles",
    "proposals", "cohort", "split", "shelf", "design", "fit", "substitution", "seal_plan",
    "time_varying")}


# ── the shared states ────────────────────────────────────────────────────────

# The questions before the lane that these projects answer plainly: each interval is one period
# for everyone (the follow-up of a yes/no outcome per row), no temporal holdout, no exclusions.
EARLIER = {"censoring": "same", "temporal": d.TemporalSpec(temporal=False), "exclusions": []}


def feedback_state(**update: Any) -> ProjectState:
    """The DASH cohort as an inference project: visits kept as rows, the time column named with the
    repeats answer, the diet declared the exposure, and blood pressure answered as a cause of the
    diet that the diet could have changed (a confounder affected by prior exposure)."""
    state = ProjectState(
        lens=["clinical"], target="cvd", task="binary", event="1", purpose="inference",
        grain=d.GrainSpec(grain="repeated", id_column="pid"),
        repeat_kind=d.RepeatSpec(repeat_kind="time_points", time_column="visit"), unit="row",
        roles={"pid": "identifier", "visit": "time", "dash": "exposure", "sbp": "covariate",
               "female": "covariate", "age": "covariate", "lost": "excluded"},
        missing=d.MissingSpec(strategy="complete_case"),
        split=d.SplitSpec(holdout=0.0, seed=0, folds=5),
        estimand=d.EstimandSpec(exposure="dash", measure="odds_ratio"),
        adjustment={"sbp": d.AdjustmentAnswer(exposure="dash", causes_exposure="yes",
                                              causes_outcome="yes", after_exposure="yes"),
                    "female": d.AdjustmentAnswer(exposure="dash", causes_exposure="yes",
                                                 causes_outcome="yes", after_exposure="no"),
                    "age": d.AdjustmentAnswer(exposure="dash", causes_exposure="yes",
                                              causes_outcome="yes", after_exposure="no")},
        shape_confirmations={"code_or_count:age": "amount"}, **EARLIER)
    return state.model_copy(update=update)


MSM_LANE = d.TimeVaryingSpec(exposure="dash", method="msm_iptw",
                             ordering="exposure_precedes_outcome", confounders=["sbp"],
                             baseline=["female", "age"], censoring="lost")


def gformula_state(**update: Any) -> ProjectState:
    state = ProjectState(
        lens=["clinical"], target="event", task="binary", event="1", purpose="inference",
        grain=d.GrainSpec(grain="repeated", id_column="pid"),
        repeat_kind=d.RepeatSpec(repeat_kind="time_points", time_column="visit"), unit="row",
        roles={"pid": "identifier", "visit": "time", "diet": "exposure", "htn": "covariate",
               "female": "covariate"},
        missing=d.MissingSpec(strategy="complete_case"),
        split=d.SplitSpec(holdout=0.0, seed=0, folds=5),
        estimand=d.EstimandSpec(exposure="diet", measure="odds_ratio"),
        adjustment={"htn": d.AdjustmentAnswer(exposure="diet", causes_exposure="yes",
                                              causes_outcome="yes", after_exposure="yes"),
                    "female": d.AdjustmentAnswer(exposure="diet", causes_exposure="yes",
                                                 causes_outcome="yes", after_exposure="no")},
        **EARLIER)
    return state.model_copy(update=update)


G_LANE = d.TimeVaryingSpec(exposure="diet", method="gformula", ordering="exposure_precedes_outcome",
                           confounders=["htn"], baseline=["female"], simulations=20_000,
                           bootstrap=100)


def run_stage(frame: pd.DataFrame, state: ProjectState, folder: Path, name: str) -> dict[str, Any]:
    """The ``time_varying`` stage's public artifact, run by the real graph on ``frame``."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.csv"
    frame.to_csv(path, index=False)
    graph = GraphRun(path, folder / name)
    try:
        out = graph.run(state, upto=["time_varying"])
        return graph.public(out)["time_varying"]
    finally:
        graph.close()


def lagged_within(values: pd.Series, units: pd.Series, first: float) -> np.ndarray:
    """Each row's value at its unit's previous row (``first`` at the unit's first row)."""
    return values.groupby(units, sort=False).shift(1).fillna(first).to_numpy(float)


# ── 1 · the weights and the marginal structural model, against R ──────────────


def _haart(folder: Path) -> pd.DataFrame:
    h = r_dataset("haartdat", "ipw", folder)
    h = h.sort_values(["patient", "fuptime"], kind="mergesort").reset_index(drop=True)
    h["tindex"] = tv.time_index(h["fuptime"]).astype(float)
    h["tindex2"] = h["tindex"] ** 2
    return h


HAART_R = """
suppressMessages(library(ipw))
d <- read.csv("haart.csv")
d <- d[order(d$patient, d$fuptime), ]
wa <- ipwtm(exposure = haartind, family = "binomial", link = "logit",
            numerator = ~ sex + age + tindex + I(tindex^2),
            denominator = ~ sex + age + cd4.sqrt + tindex + I(tindex^2),
            id = patient, timevar = fuptime, type = "first", data = d, control = ctl, trunc = 0.01)
risk <- d$event == 0
d0 <- d[risk, ]
wc <- ipwtm(exposure = dropout, family = "binomial", link = "logit",
            numerator = ~ sex + age + haartind + tindex + I(tindex^2),
            denominator = ~ sex + age + cd4.sqrt + haartind + tindex + I(tindex^2),
            id = patient, timevar = fuptime, type = "cens", data = d0, control = ctl)
out(list(patient = d$patient, fuptime = d$fuptime, wa = wa$ipw.weights, wat = wa$weights.trunc,
         wc = wc$ipw.weights))
"""


@needs_r
def test_1_weights_match_ipwtm_for_initiation_truncation_and_loss_to_follow_up(tmp_path):
    """``haartdat`` (1,200 patients, 19,175 rows; HAART, once started, stays; 690 lost): the
    exposure weights (``type = "first"``), their 1st/99th-percentile truncation (``trunc = 0.01``)
    and the loss-to-follow-up weights agree with ``ipwtm`` to 1e-8. A lost unit's last row is kept
    (its outcome was seen), so each row's censoring weight is ipwtm's ``type = "cens"`` product,
    fit on the rows at risk of being lost, read at the unit's previous row. That lag is computed
    here with pandas from R's weights."""
    h = _haart(tmp_path)
    r = run_r(HAART_R, {"haart": h}, tmp_path / "r")
    assert r["patient"] == h["patient"].tolist() and r["fuptime"] == h["fuptime"].tolist()
    exposure = tv.ipw_weights(h, id="patient", time="fuptime", indicator="haartind",
                              numerator=["sex", "age", "tindex", "tindex2"],
                              denominator=["sex", "age", "cd4.sqrt", "tindex", "tindex2"],
                              kind="first")
    np.testing.assert_allclose(exposure.weights, r["wa"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(tv.truncate(exposure.weights, 0.01), r["wat"], rtol=0, atol=1e-8)
    risk = (h["event"] == 0).to_numpy()
    censoring = tv.censoring_weights(
        h, id="patient", time="fuptime", indicator="dropout",
        numerator=["sex", "age", "haartind", "tindex", "tindex2"],
        denominator=["sex", "age", "cd4.sqrt", "haartind", "tindex", "tindex2"], at_risk=risk)
    full = pd.Series(np.nan, index=h.index)
    full[risk] = r["wc"]
    expected = lagged_within(full, h["patient"], 1.0)
    np.testing.assert_allclose(censoring.weights, expected, rtol=0, atol=1e-8)
    assert (h.groupby("patient")["dropout"].sum() <= 1).all()  # lost once, on the last row


BASIC_R = """
suppressMessages(library(ipw))
d <- read.csv("basic.csv")
d <- d[order(d$id, d$t0), ]
wa <- ipwtm(exposure = A, family = "binomial", link = "logit",
            numerator = ~ L3 + lagA + t0 + I(t0^2),
            denominator = ~ L3 + L1 + L2 + lagA + t0 + I(t0^2),
            id = id, timevar = t0, type = "all", data = d, control = ctl)
out(list(wa = wa$ipw.weights))
"""


def _basic(folder: Path) -> pd.DataFrame:
    b = r_dataset("basicdata_nocomp", "gfoRmula", folder)
    b = b.sort_values(["id", "t0"], kind="mergesort").reset_index(drop=True)
    b["lagA"] = lagged_within(b["A"].astype(float), b["id"], 0.0)
    b["t02"] = b["t0"].astype(float) ** 2
    b["cumA"] = b.groupby("id")["A"].cumsum().astype(float)
    return b


@needs_r
def test_1_weights_match_ipwtm_for_an_exposure_that_switches(tmp_path):
    """``basicdata_nocomp`` (2,500 units, 13,170 rows; ``A`` switches on and off): the stabilized
    weights with the previous exposure in both models (``type = "all"``) agree with ``ipwtm`` to
    1e-8."""
    b = _basic(tmp_path)
    r = run_r(BASIC_R, {"basic": b}, tmp_path / "r")
    w = tv.ipw_weights(b, id="id", time="t0", indicator="A",
                       numerator=["L3", "lagA", "t0", "t02"],
                       denominator=["L3", "L1", "L2", "lagA", "t0", "t02"], kind="all")
    np.testing.assert_allclose(w.weights, r["wa"], rtol=0, atol=1e-8)
    assert abs(w.weights.mean() - 1) < 0.05  # Cole & Hernán's necessary condition


MSM_R = {
    "haart": HAART_R.replace("out(list(patient", "invisible(list(patient") + """
suppressMessages(library(geepack))
wcf <- rep(NA, nrow(d)); wcf[risk] <- wc$ipw.weights
wcl <- ave(wcf, d$patient, FUN = function(x) c(1, head(x, -1)))
d$w <- wa$ipw.weights * wcl
m <- geeglm(event ~ haartind + sex + age + tindex + I(tindex^2), family = binomial, data = d,
            id = patient, weights = w, corstr = "independence",
            control = geese.control(epsilon = 1e-14, maxit = 100))
s <- summary(m)$coefficients
out(list(coef = unname(s[, "Estimate"]), se = unname(s[, "Std.err"])))
""",
    "basic": BASIC_R.replace("out(list(wa", "invisible(list(wa") + """
suppressMessages(library(geepack))
d$w <- wa$ipw.weights
m <- geeglm(Y ~ cumA + L3 + t0 + I(t0^2), family = binomial, data = d, id = id, weights = w,
            corstr = "independence", control = geese.control(epsilon = 1e-14, maxit = 100))
s <- summary(m)$coefficients
out(list(coef = unname(s[, "Estimate"]), se = unname(s[, "Std.err"])))
""",
    "glucose": """
suppressMessages({library(ipw); library(geepack)})
d <- read.csv("glucose.csv")
d <- d[order(d$pid, d$wave), ]
wa <- ipwtm(exposure = walk, family = "binomial", link = "logit",
            numerator = ~ female + age + lagwalk + wave + I(wave^2),
            denominator = ~ female + age + waist + lagwalk + wave + I(wave^2),
            id = pid, timevar = wave, type = "all", data = d, control = ctl)
d$w <- wa$ipw.weights
m <- geeglm(glucose ~ walk + female + age + wave + I(wave^2), family = gaussian, data = d,
            id = pid, weights = w, corstr = "independence",
            control = geese.control(epsilon = 1e-14, maxit = 100))
s <- summary(m)$coefficients
out(list(coef = unname(s[, "Estimate"]), se = unname(s[, "Std.err"])))
""",
}


@needs_r
@pytest.mark.parametrize("case", ["haart", "basic", "glucose"])
def test_1_the_msm_estimate_and_its_robust_se_match_geeglm(case, tmp_path):
    """The weighted outcome model and its unit-clustered sandwich (``fit_msm``) against
    ``geeglm(..., weights, id, corstr = "independence")`` on R's own weights, to 1e-6 for every
    coefficient and its robust SE: a pooled logistic MSM in current (HAART) and in cumulative
    exposure (``basicdata_nocomp``), and a linear MSM of a repeated measure (glucose)."""
    if case == "haart":
        data = _haart(tmp_path)
        r = run_r(MSM_R[case], {"haart": data}, tmp_path / "r")
        exposure = tv.ipw_weights(data, id="patient", time="fuptime", indicator="haartind",
                                  numerator=["sex", "age", "tindex", "tindex2"],
                                  denominator=["sex", "age", "cd4.sqrt", "tindex", "tindex2"],
                                  kind="first")
        lost = tv.censoring_weights(
            data, id="patient", time="fuptime", indicator="dropout",
            numerator=["sex", "age", "haartind", "tindex", "tindex2"],
            denominator=["sex", "age", "cd4.sqrt", "haartind", "tindex", "tindex2"],
            at_risk=(data["event"] == 0).to_numpy())
        w = exposure.weights * lost.weights
        X, names = tv.design(data, ["haartind", "sex", "age", "tindex", "tindex2"])
        fit = tv.fit_msm(X, names, data["event"], w, data["patient"], "binomial")
    elif case == "basic":
        data = _basic(tmp_path)
        r = run_r(MSM_R[case], {"basic": data}, tmp_path / "r")
        w = tv.ipw_weights(data, id="id", time="t0", indicator="A",
                           numerator=["L3", "lagA", "t0", "t02"],
                           denominator=["L3", "L1", "L2", "lagA", "t0", "t02"], kind="all").weights
        X, names = tv.design(data, ["cumA", "L3", "t0", "t02"])
        fit = tv.fit_msm(X, names, data["Y"], w, data["id"], "binomial")
    else:
        data = repeated_measure_cohort()
        data = data.sort_values(["pid", "wave"], kind="mergesort").reset_index(drop=True)
        data["lagwalk"] = lagged_within(data["walk"].astype(float), data["pid"], 0.0)
        data["wave2"] = data["wave"].astype(float) ** 2
        r = run_r(MSM_R[case], {"glucose": data}, tmp_path / "r")
        w = tv.ipw_weights(data, id="pid", time="wave", indicator="walk",
                           numerator=["female", "age", "lagwalk", "wave", "wave2"],
                           denominator=["female", "age", "waist", "lagwalk", "wave", "wave2"],
                           kind="all").weights
        X, names = tv.design(data, ["walk", "female", "age", "wave", "wave2"])
        fit = tv.fit_msm(X, names, data["glucose"], w, data["pid"], "gaussian")
    np.testing.assert_allclose(fit.coef, r["coef"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(fit.se, r["se"], rtol=0, atol=1e-6)


STAGE_R = """
suppressMessages({library(ipw); library(geepack)})
d <- read.csv("cohort.csv")
d <- d[order(d$pid, d$visit), ]
d$t <- match(d$visit, sort(unique(d$visit))) - 1
d$lagdash <- ave(d$dash, d$pid, FUN = function(x) c(0, head(x, -1)))
d$cumdash <- ave(d$dash, d$pid, FUN = cumsum)
wa <- ipwtm(exposure = dash, family = "binomial", link = "logit",
            numerator = ~ female + age + lagdash + t + I(t^2),
            denominator = ~ female + age + sbp + lagdash + t + I(t^2),
            id = pid, timevar = t, type = "all", data = d, control = ctl)
risk <- d$cvd == 0
d0 <- d[risk, ]
wc <- ipwtm(exposure = lost, family = "binomial", link = "logit",
            numerator = ~ female + age + dash + t + I(t^2),
            denominator = ~ female + age + sbp + dash + t + I(t^2),
            id = pid, timevar = t, type = "cens", data = d0, control = ctl)
wcf <- rep(NA, nrow(d)); wcf[risk] <- wc$ipw.weights
wcl <- ave(wcf, d$pid, FUN = function(x) c(1, head(x, -1)))
w <- wa$ipw.weights * wcl
q <- quantile(w, c(0.01, 0.99))
wt <- w; wt[w <= q[1]] <- q[1]; wt[w > q[2]] <- q[2]
d$wt <- wt
d$unit <- as.integer(factor(d$pid))
m <- geeglm(cvd ~ cumdash + female + age + t + I(t^2), family = binomial, data = d, id = unit,
            weights = wt, corstr = "independence",
            control = geese.control(epsilon = 1e-14, maxit = 100))
s <- summary(m)$coefficients
qs <- unname(quantile(w, c(0.01, 0.25, 0.5, 0.75, 0.99)))
out(list(coef = unname(s[2, "Estimate"]), se = unname(s[2, "Std.err"]), mean = mean(w),
         sd = sd(w), min = min(w), max = max(w), q = qs, tmean = mean(wt), tmin = min(wt),
         tmax = max(wt)))
"""


@needs_r
def test_1_through_the_stage_the_weights_and_the_estimate_are_R_s(tmp_path):
    """The app's own path. The ``time_varying`` stage, run by the real graph on a cohort table
    (visits numbered 1–6, a confounder affected by prior exposure, loss to follow-up), builds its
    time index, lags, design and weights from the declared lane. Its weight summary agrees with R's
    on the same CSV to 1e-8: ``ipwtm`` exposure and loss weights, multiplied, summarized by R's
    ``mean``, ``sd``, ``min``, ``max`` and type-7 ``quantile``. Its estimate (truncated at the 1st
    and 99th percentiles) agrees with ``geeglm`` on R's truncated weights to 1e-6, on the log
    scale, with its SE."""
    frame = feedback_cohort(n=1500)
    lane = MSM_LANE.model_copy(update={"truncation": "p1_p99"})
    art = run_stage(frame, feedback_state(time_varying=lane), tmp_path, "cohort")
    r = run_r(STAGE_R, {"cohort": frame}, tmp_path / "r")
    w = art["diagnostics"]["weights"]
    for key, ref in (("mean", r["mean"]), ("sd", r["sd"]), ("min", r["min"]), ("max", r["max"])):
        assert w[key] == pytest.approx(ref, abs=1e-8), key
    for key, ref in zip(("p1", "p25", "median", "p75", "p99"), r["q"]):
        assert w[key] == pytest.approx(ref, abs=1e-8), key
    chosen = next(o for o in art["diagnostics"]["truncation"] if o["chosen"])
    assert chosen["key"] == "p1_p99"
    assert chosen["summary"]["mean"] == pytest.approx(r["tmean"], abs=1e-8)
    assert chosen["summary"]["min"] == pytest.approx(r["tmin"], abs=1e-8)
    assert chosen["summary"]["max"] == pytest.approx(r["tmax"], abs=1e-8)
    [row] = art["estimates"]["rows"]
    assert math.log(row["estimate"]) == pytest.approx(r["coef"], abs=1e-6)
    assert row["se"] == pytest.approx(r["se"], abs=1e-6)
    assert math.log(row["ci_low"]) == pytest.approx(r["coef"] - tv.Z95 * r["se"], abs=1e-6)
    assert art["estimates"]["n_units"] == frame["pid"].nunique()
    assert art["estimates"]["n_rows"] == len(frame)


# ── 2 · the parametric g-formula ─────────────────────────────────────────────

GFORMULA_R = """
suppressMessages({library(gfoRmula); library(data.table)})
d <- as.data.table(read.csv("basic.csv"))
g <- gformula(obs_data = d, id = "id", time_points = 7, time_name = "t0",
              covnames = c("L1", "L2", "A"), outcome_name = "Y", outcome_type = "survival",
              covtypes = c("binary", "normal", "binary"),
              histories = c(lagged), histvars = list(c("A", "L1", "L2")),
              covparams = list(covmodels = c(
                L1 ~ L3 + lag1_A + lag1_L1 + lag1_L2 + t0 + I(t0^2),
                L2 ~ L3 + L1 + lag1_A + lag1_L1 + lag1_L2 + t0 + I(t0^2),
                A ~ L3 + L1 + L2 + lag1_A + lag1_L1 + lag1_L2 + t0 + I(t0^2))),
              ymodel = Y ~ A + L1 + L2 + L3 + lag1_A + lag1_L1 + lag1_L2 + t0 + I(t0^2),
              intervention1.A = list(static, rep(0, 7)), intervention2.A = list(static, rep(1, 7)),
              int_descript = c("Never", "Always"), basecovs = c("L3"), nsimul = 50000,
              seed = 1234, model_fits = TRUE, show_progress = FALSE)
r <- as.data.frame(g$result)
last <- r[r$k == 6, ]
out(list(natural = last[last$Interv. == 0, "g-form risk"], never = last[last$Interv. == 1, "g-form risk"],
         always = last[last$Interv. == 2, "g-form risk"], np = r[r$Interv. == 0, "NP Risk"],
         fits = lapply(g$fits, function(f) as.list(coef(f))), rmse = g$fits$L2$rmse))
"""

BASIC_HISTORY = ("lag1_A", "lag1_L1", "lag1_L2", tv.TIME, tv.TIME2)
BASIC_SPEC = tv.GFormulaSpec(
    id="id", time="t0", exposure="A", outcome="Y",
    covariates=(tv.Covariate("L1", "binary", ("L3", *BASIC_HISTORY)),
                tv.Covariate("L2", "normal", ("L3", "L1", *BASIC_HISTORY))),
    exposure_terms=("L3", "L1", "L2", *BASIC_HISTORY),
    outcome_terms=("A", "L1", "L2", "L3", *BASIC_HISTORY), baseline=("L3",),
    lagged=("A", "L1", "L2"))


@needs_r
def test_2_the_g_formula_agrees_with_gfoRmula_within_monte_carlo_error(tmp_path):
    """``gfoRmula``'s example data (``basicdata_nocomp``: 2,500 units, 7 time points; ``L1``
    binary, ``L2`` normal, ``L3`` baseline, ``A`` switching), every model with lag-1 histories,
    time and its square, and 50,000 simulated units on each side.

    * Each model's coefficients agree with R's fits to 1e-6, and ``L2``'s residual root mean square
      with R's ``rmse``. The models are deterministic, so the only difference left is Monte Carlo.
    * The nonparametric (observed) risk at each time point agrees with R's ``NP Risk`` to 1e-12.
    * The risks by the last time point under "never", "always" and the natural course agree within
      4·√2 Monte Carlo standard errors. The two runs are independent with equal simulation counts,
      and ``sd(r_i)/√n`` bounds each one's error from above."""
    b = r_dataset("basicdata_nocomp", "gfoRmula", tmp_path)
    r = run_r(GFORMULA_R, {"basic": b}, tmp_path / "r")
    res = tv.gformula(b, BASIC_SPEC, n_sim=50_000, seed=7)
    rename = {"t0": tv.TIME, "I(t0^2)": tv.TIME2}
    fits = {"L1": res.models.covariates["L1"], "L2": res.models.covariates["L2"],
            "A": res.models.exposure, "Y": res.models.outcome}
    for name, fit in fits.items():
        mine = fit.coefficients()
        theirs = {rename.get(k, k): v[0] if isinstance(v, list) else v
                  for k, v in r["fits"][name].items()}
        assert set(mine) == set(theirs), name
        for term, value in theirs.items():
            assert mine[term] == pytest.approx(value, abs=1e-6), (name, term)
    rmse = r["rmse"][0] if isinstance(r["rmse"], list) else r["rmse"]
    assert fits["L2"].rmse == pytest.approx(rmse, abs=1e-10)
    np.testing.assert_allclose(res.observed, r["np"], rtol=0, atol=1e-12)
    for strategy in ("natural", "never", "always"):
        ref = r[strategy][0] if isinstance(r[strategy], list) else r[strategy]
        bound = 4 * math.sqrt(2) * res.mc_se[strategy]
        assert abs(res.final(strategy) - ref) < bound, (strategy, res.final(strategy), ref, bound)
    assert res.n_sim == 50_000


GTRUTH_HISTORY = ("lag1_diet", "lag1_htn", tv.TIME, tv.TIME2)
GTRUTH_SPEC = tv.GFormulaSpec(
    id="pid", time="visit", exposure="diet", outcome="event",
    covariates=(tv.Covariate("htn", "binary", ("female", *GTRUTH_HISTORY)),),
    exposure_terms=("female", "htn", *GTRUTH_HISTORY),
    outcome_terms=("diet", "htn", "female", *GTRUTH_HISTORY), baseline=("female",),
    lagged=("diet", "htn"))


def test_2_the_g_formula_recovers_an_exact_truth():
    """A cohort whose lag-1 logistic models are the true ones (``gformula_cohort``: hypertension a
    binary confounder the diet lowers and that prompts the diet; 40,000 people, 5 visits). The
    risks had everyone always, or never, followed the diet are computed exactly by enumerating
    hypertension's states at each visit (``exact_risks``: 0.1750 and 0.3109). The g-formula's, with
    100,000 simulated units, fall within 4 bootstrap SEs of them (30 resamples of whole people),
    and so does their difference."""
    data = gformula_cohort(n=40_000, seed=7)
    truth = {"always": exact_risks(5, 1), "never": exact_risks(5, 0)}
    truth["difference"] = truth["always"] - truth["never"]
    assert truth["always"] == pytest.approx(0.1750, abs=1e-4)
    assert truth["never"] == pytest.approx(0.3109, abs=1e-4)
    res = tv.gformula(data, GTRUTH_SPEC, n_sim=100_000, seed=11)
    boot = tv.gformula_bootstrap(data, GTRUTH_SPEC, reps=30, n_sim=None, seed=12)
    estimate = {"always": res.final("always"), "never": res.final("never")}
    estimate["difference"] = estimate["always"] - estimate["never"]
    for key in ("always", "never", "difference"):
        se = float(np.std(boot["draws"][key], ddof=1))
        assert abs(estimate[key] - truth[key]) < 4 * se, (key, estimate[key], truth[key], se)
    assert boot["failed"] == 0


# ── 3 · diagnostics before estimates ─────────────────────────────────────────


def _independent_weights(frame: pd.DataFrame) -> pd.DataFrame:
    """The DASH cohort's weights by another route: statsmodels' ``Logit`` for each model, the
    factors and products written out with pandas (no code from ``models/time_varying.py``)."""
    import statsmodels.api as sm

    f = frame.sort_values(["pid", "visit"], kind="mergesort").reset_index(drop=True)
    f["t"] = f["visit"] - f["visit"].min()
    f["t2"] = f["t"] ** 2
    f["lagdash"] = lagged_within(f["dash"].astype(float), f["pid"], 0.0)

    def prob(rows: pd.DataFrame, y: str, cols: list[str]) -> np.ndarray:
        X = sm.add_constant(rows[cols].astype(float), has_constant="add")
        fit = sm.Logit(rows[y].astype(float), X).fit(disp=0, method="newton", tol=1e-12,
                                                       maxiter=100)
        return fit.predict(X)

    p_num = prob(f, "dash", ["female", "age", "lagdash", "t", "t2"])
    p_den = prob(f, "dash", ["female", "age", "sbp", "lagdash", "t", "t2"])
    a = f["dash"].to_numpy()
    fa = np.where(a == 1, p_num, 1 - p_num) / np.where(a == 1, p_den, 1 - p_den)
    wa = pd.Series(fa).groupby(f["pid"]).cumprod().to_numpy()
    risk = (f["cvd"] == 0).to_numpy()
    r = f.loc[risk]
    c_num = prob(r, "lost", ["female", "age", "dash", "t", "t2"])
    c_den = prob(r, "lost", ["female", "age", "sbp", "dash", "t", "t2"])
    fc = pd.Series(np.nan, index=f.index)
    fc[risk] = (1 - c_num) / (1 - c_den)
    stayed = fc.groupby(f["pid"]).cumprod()
    wc = lagged_within(stayed, f["pid"], 1.0)
    f["w"] = wa * wc
    f["p_den"] = p_den
    return f


def _numpy_summary(w: np.ndarray) -> dict[str, float]:
    return {"n": len(w), "mean": w.mean(), "sd": w.std(ddof=1), "min": w.min(),
            "p1": np.percentile(w, 1), "p25": np.percentile(w, 25),
            "median": np.percentile(w, 50), "p75": np.percentile(w, 75),
            "p99": np.percentile(w, 99), "max": w.max()}


def test_3_the_weights_and_positivity_are_shown_before_any_estimate(tmp_path):
    """With the lane declared and its truncation not yet, the stage computes and returns the
    diagnostics and no estimate. Every number is checked against statsmodels and pandas, by
    :func:`_independent_weights`, to 1e-6:

    * the weights' distribution overall (n, mean, SD, min, percentiles, max) and at each time
      point; on this correctly specified model the stabilized mean is near 1 (within 0.05), which
      Cole & Hernán (2008) call "a necessary condition for correct model specification";
    * each truncation option's distribution: none, then the 1st/99th and the 5th/95th percentiles
      (NumPy's clip at its quantiles). Each step narrows the weights, and each option states its
      trade-off of bias against precision;
    * positivity at each time point: exposed and unexposed rows (pandas counts) and the range of
      the denominator model's fitted probability of exposure.

    Then: no estimate (``estimates`` null, the reason why), nothing a plan lock reads as an
    estimate, and the Router keeps the question open."""
    frame = feedback_cohort(n=1500)
    art = run_stage(frame, feedback_state(time_varying=MSM_LANE), tmp_path, "cohort")
    ref = _independent_weights(frame)
    w = ref["w"].to_numpy()
    diagnostics = art["diagnostics"]
    for key, value in _numpy_summary(w).items():
        assert diagnostics["weights"][key] == pytest.approx(value, rel=1e-6, abs=1e-9), key
    assert abs(diagnostics["weights"]["mean"] - 1) < 0.05
    for at in diagnostics["weights_by_time"]:
        rows = w[ref["t"].to_numpy() == at["time"]]
        assert at["n"] == len(rows)
        assert at["mean"] == pytest.approx(rows.mean(), rel=1e-6)
        assert at["max"] == pytest.approx(rows.max(), rel=1e-6)
    options = {o["key"]: o for o in diagnostics["truncation"]}
    assert list(options) == ["none", "p1_p99", "p5_p95"]
    for key, level in (("none", None), ("p1_p99", 0.01), ("p5_p95", 0.05)):
        clipped = w if level is None else np.clip(w, np.quantile(w, level),
                                                  np.quantile(w, 1 - level))
        for stat, value in _numpy_summary(clipped).items():
            assert options[key]["summary"][stat] == pytest.approx(value, rel=1e-6, abs=1e-9)
        assert not options[key]["chosen"]
    assert (options["none"]["summary"]["max"] > options["p1_p99"]["summary"]["max"]
            > options["p5_p95"]["summary"]["max"])
    assert options["none"]["summary"]["sd"] > options["p5_p95"]["summary"]["sd"]
    for key in ("p1_p99", "p5_p95"):
        assert "bias" in options[key]["sound"] and "precision" in options[key]["sound"]
    positivity = diagnostics["positivity"]
    assert [p["time"] for p in positivity] == list(range(6))
    for p in positivity:
        rows = ref.loc[ref["t"] == p["time"]]
        assert p["exposed"] == int((rows["dash"] == 1).sum())
        assert p["unexposed"] == int((rows["dash"] == 0).sum())
        assert p["p_min"] == pytest.approx(rows["p_den"].min(), rel=1e-6)
        assert p["p_max"] == pytest.approx(rows["p_den"].max(), rel=1e-6)
    assert art["estimates"] is None
    assert art["withheld"].startswith("Read the weights and positivity first")
    assert not plan_lock.shows_estimates("time_varying", art)
    state = feedback_state(time_varying=MSM_LANE)
    step = next(s for s in route(state, ALL_FRESH, {"time_varying": art}) if s.key == "time_varying")
    assert step.status == "open"


def test_3_a_time_point_without_an_exposed_row_and_a_mean_far_from_one_are_named(tmp_path):
    """Nonpositivity is named, not hidden. On a cohort where no one follows the diet at the first
    visit, the positivity row for that time point counts 0 exposed (pandas agrees), and the
    concerns say the estimate there rests on the model alone. Where the exposure follows blood
    pressure almost deterministically, the stabilized weights' mean leaves 1 by more than 0.1, and
    the concern cites Cole & Hernán's reading of it."""
    frame = feedback_cohort(n=1200, seed=5)
    frame.loc[frame["visit"] == 1, "dash"] = 0
    art = run_stage(frame, feedback_state(time_varying=MSM_LANE), tmp_path, "first")
    first = art["diagnostics"]["positivity"][0]
    assert first["exposed"] == 0 and first["unexposed"] == int((frame["visit"] == 1).sum())
    assert any("At time point 0 no row is exposed" in c for c in art["diagnostics"]["concerns"])

    def steep(slope: float, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(3)
        f = feedback_cohort(n=1200, seed=seed)
        f["dash"] = (rng.random(len(f)) < expit(slope * (f["sbp"] - 128))).astype(int)
        return f

    art = run_stage(steep(0.4, 7), feedback_state(time_varying=MSM_LANE), tmp_path, "steep")
    mean = art["diagnostics"]["weights"]["mean"]
    assert abs(mean - 1) > 0.1, mean
    assert any(f"The stabilized weights' mean is {mean:.2f}, not near 1" in c
               and "Cole & Hernán" in c for c in art["diagnostics"]["concerns"])
    # Steeper still, blood pressure separates the diet: no weight, no estimate, and the reason.
    art = run_stage(steep(0.9, 6), feedback_state(time_varying=MSM_LANE), tmp_path, "separated")
    assert art["diagnostics"] is None and art["estimates"] is None
    assert art["reason"].startswith("The weights cannot be estimated")
    assert "a positivity violation" in art["reason"]


def test_3_no_estimate_is_served_or_locks_the_plan_while_the_lane_is_open():
    """The server's hold (``estimand.served_gate``): while the time-varying question is open, an
    estimate in any estimate stage, the lane's own included, is withheld with the reason, and what
    is served locks no plan. Once the truncation is declared the question is answered, and the hold
    lifts."""
    from turbotab.core import estimand

    state = feedback_state(time_varying=MSM_LANE)
    steps = route(state, ALL_FRESH)
    gate = estimand.served_gate(state, steps)
    assert gate is not None and gate["question"] == "time_varying"
    assert "truncation" in gate["reason"]
    served = estimand.withhold("time_varying", {"estimates": {"rows": [{"estimate": 1.2}]},
                                                "diagnostics": {"weights": {}}}, gate)
    assert served["estimates"] is None and served["diagnostics"] == {"weights": {}}
    assert not plan_lock.shows_estimates("time_varying", served)
    assert plan_lock.shows_estimates("time_varying", {"estimates": {"rows": [{"estimate": 1.2}]}})
    done = feedback_state(time_varying=MSM_LANE.model_copy(update={"truncation": "none"}))
    # (FORM: the form question's card finds no continuous term to declare a form for here)
    no_forms = {"purpose": "inference", "ready": True, "needs": []}
    assert estimand.served_gate(done, route(done, ALL_FRESH, {"forms": no_forms})) is None


# ── 4 · routing ──────────────────────────────────────────────────────────────


def _step(state: ProjectState, artifacts: dict[str, Any] | None = None) -> Any:
    return next(s for s in route(state, ALL_FRESH, artifacts or {}) if s.key == "time_varying")


def test_4_the_question_is_asked_only_of_an_exposure_followed_through_time():
    """The Router asks the question under inference, when one exposure is declared and the rows
    are a unit's time points kept as rows. Each other setting is not applicable, with its reason:
    prediction, one row per unit, repeats of one measurement, combined records, an exposure family,
    and an exposure the values show fixed within every unit."""
    asked = _step(feedback_state())
    assert asked.status == "open"
    cases = {
        "Under prediction no coefficient is read as an effect": feedback_state(
            purpose="prediction", estimand=None, adjustment=None),
        "Each unit appears once": feedback_state(grain=d.GrainSpec(grain="one_row_per_unit")),
        "repeats of one measurement": feedback_state(
            repeat_kind=d.RepeatSpec(repeat_kind="repeats")),
        "Each unit's rows are combined": feedback_state(unit="unit"),
        "An exposure family": feedback_state(
            roles={"pid": "identifier", "visit": "time", "dash": "exposure", "sbp": "exposure",
                   "female": "covariate", "age": "covariate", "lost": "excluded"},
            estimand=d.EstimandSpec(family=True, measure="exposure_mean_difference"),
            adjustment=None),
    }
    for reason, state in cases.items():
        step = _step(state)
        assert step.status == "not_applicable" and reason in step.reason, (reason, step)
    fixed = {"setting": {"exposure": "dash", "exposure_varies": False}}
    step = _step(feedback_state(), {"time_varying": fixed})
    assert step.status == "not_applicable"
    assert step.reason == ("`dash` takes one value within every unit, so it does not change over "
                           "time; the standard model estimates it.")


def _ctx(state: ProjectState, **artifacts: Any) -> dict[str, Any]:
    return {"state": state, "columns": ["pid", "visit", "dash", "sbp", "female", "age", "lost",
                                        "cvd"],
            "artifact": lambda stage: artifacts.get(stage)}


def _refused(decision: Any, ctx: dict[str, Any]) -> Refusal:
    with pytest.raises(Refusal) as refused:
        d.validate(decision, ctx)
    return refused.value


def test_4_the_lane_needs_repeated_measures_a_settled_time_column_and_a_declared_order():
    """Each need is refused until met, each refusal with its way forward:

    * repeated measures (the grain), time points (the repeats answer, with the exit that answers
      it) and records kept as rows (the exit keeps them);
    * a settled time column: one only proposed by the repeats reading is asked, and the exit is the
      ledger's own ``confirm_reading``. Confirmed, the lane passes (BLUEPRINT §14.1);
    * a declared time ordering: "same time" and "unknown" are refused, because an exposure measured
      with its outcome cannot be told from a consequence of it. The exit declares the order."""
    lane = d.SetTimeVarying(**{**MSM_LANE.model_dump(), "truncation": "none"})
    assert d.validate(lane, _ctx(feedback_state())) is not None
    no_repeats = _refused(lane, _ctx(feedback_state(grain=d.GrainSpec(grain="one_row_per_unit"))))
    assert no_repeats.code == "no_repeated_measures"
    repeats = _refused(lane, _ctx(feedback_state(repeat_kind=d.RepeatSpec(repeat_kind="repeats"))))
    assert repeats.code == "not_time_points"
    assert repeats.exits[0]["decision"]["repeat_kind"] == "time_points"
    combined = _refused(lane, _ctx(feedback_state(unit="unit")))
    assert combined.code == "records_combined"
    assert combined.exits[0]["decision"] == {"kind": "set_unit", "unit": "row"}

    proposed = {"repeats": {"stated": True, "reading": "time_points", "confidence": "medium",
                            "spacing": {"column": "visit"}}}
    unnamed = feedback_state(repeat_kind=d.RepeatSpec(repeat_kind="time_points"))
    refused = _refused(lane, _ctx(unnamed, structure=proposed))
    assert refused.code == "time_column_unsettled"
    assert "`visit` is only proposed" in refused.message
    assert refused.exits[0]["decision"] == {"kind": "confirm_reading", "reading": "time_column",
                                            "column": "visit", "value": "orders"}
    confirmed = unnamed.model_copy(update={"shape_confirmations": {
        "code_or_count:age": "amount", "time_column:visit": "orders"}})
    assert d.validate(lane, _ctx(confirmed, structure=proposed)) is not None

    for ordering in ("same_time", "unknown"):
        refused = _refused(lane.model_copy(update={"ordering": ordering}), _ctx(feedback_state()))
        assert refused.code == "time_ordering"
        assert refused.exits[0]["decision"]["ordering"] == "exposure_precedes_outcome"


def test_4_standard_regression_over_a_confounder_affected_by_prior_exposure_is_block_and_record():
    """``sbp`` is answered a cause of the diet that the diet could have changed, and a cause of the
    event: a time-varying confounder affected by prior exposure. Standard regression is refused
    until attested. Its exits are the marginal structural model and the g-formula (each adjusting
    for ``sbp`` at each time point, and each passing the lane's checks) and the attestation, which
    is recorded and stated. A g-method that leaves ``sbp`` out is refused, with the exit that adds
    it. With ``sbp`` answered unchanged by the diet, standard regression passes and ranks first."""
    from turbotab.core import voice
    from turbotab.core.stages.time_varying import lane_options

    state = feedback_state()
    standard = d.SetTimeVarying(exposure="dash", method="standard",
                                ordering="exposure_precedes_outcome")
    refused = _refused(standard, _ctx(state))
    assert refused.code == "affected_confounder"
    assert "`sbp` is a cause of `dash` that earlier `dash` could have changed" in refused.message
    labels = [e["label"] for e in refused.exits]
    assert labels == ["Estimate by a marginal structural model (weights)",
                      "Estimate by the parametric g-formula",
                      "Keep standard regression; record that the estimate is biased by it"]
    msm, gform, attest = (e["decision"] for e in refused.exits)
    assert msm["method"] == "msm_iptw" and msm["confounders"] == ["sbp"]
    assert gform["method"] == "gformula" and gform["confounders"] == ["sbp"]
    d.validate(gform, _ctx(state))
    d.validate({**msm, "truncation": "p1_p99"}, _ctx(state))
    recorded = d.validate(attest, _ctx(state))
    assert voice.sentence_for(recorded, state) == (
        "`dash` changes over time and its effect on `cvd` is estimated by standard regression, as "
        "recorded, although `sbp` is a confounder that earlier `dash` changed: adjusting for it "
        "removes part of the effect, and leaving it out leaves later exposure confounded (Robins, "
        "Hernán & Brumback 2000); `dash` at each time point precedes the outcome it is paired "
        "with, as declared.")
    left_out = _refused(d.SetTimeVarying(**{**MSM_LANE.model_dump(), "confounders": [],
                                            "baseline": ["female", "age"],
                                            "truncation": "none"}), _ctx(state))
    assert left_out.code == "affected_confounder_left_out"
    assert left_out.exits[0]["decision"]["confounders"] == ["sbp"]

    unchanged = feedback_state(adjustment={
        **state.adjustment, "sbp": d.AdjustmentAnswer(exposure="dash", causes_exposure="yes",
                                                      causes_outcome="yes", after_exposure="no")})
    d.validate(standard, _ctx(unchanged))
    first = lane_options([])[0]
    assert first["key"] == "standard" and first["rung"] == "recommended"
    ranked = lane_options(["sbp"])
    assert [o["key"] for o in ranked] == ["msm_iptw", "gformula", "standard"]
    assert ranked[-1]["rung"] == "block_and_record"


def test_4_an_exposure_that_is_not_0_or_1_is_offered_no_g_method_it_cannot_run():
    """The weights and the always-or-never strategies here take an exposure of 0 or 1 (V2 scope).
    When the stage reads the exposure as continuous (409 values), every answer the lane offers
    runs. The g-methods are refused with that reason. The affected confounder's refusal offers the
    attestation and the exposure question, never a g-method exit that would itself be refused.
    Standard regression is ranked first with its block-and-record rung, and is recorded once
    attested."""
    from turbotab.core.stages.time_varying import lane_options

    state = feedback_state()
    setting = {"setting": {"exposure": "dash", "exposure_binary": False, "exposure_levels": 409,
                           "exposure_varies": True}}
    ctx = _ctx(state, time_varying=setting)
    g = _refused(d.SetTimeVarying(**{**MSM_LANE.model_dump(), "truncation": "none"}), ctx)
    assert g.code == "exposure_not_binary" and "takes 409 values" in g.message
    standard = d.SetTimeVarying(exposure="dash", method="standard",
                                ordering="exposure_precedes_outcome")
    refused = _refused(standard, ctx)
    assert refused.code == "affected_confounder"
    assert refused.message.endswith("The g-methods here need an exposure of 0 or 1 at each time "
                                    "point, and `dash` takes 409 values.")
    assert [e["label"] for e in refused.exits] == [
        "Keep standard regression; record that the estimate is biased by it",
        "Declare a yes/no exposure (the exposure question)"]
    assert d.validate(refused.exits[0]["decision"], ctx) is not None
    ranked = lane_options(["sbp"], binary=False)
    assert [(o["key"], o["rung"]) for o in ranked] == [
        ("standard", "block_and_record"), ("msm_iptw", "refused"), ("gformula", "refused")]


def test_4_under_the_surveyed_population_the_estimates_are_blocked_with_the_sample_only_exit(
        tmp_path):
    """MODELING_SEQUENCE §0 ruling 6 (wave 1's MS4): a population estimand under a survey design
    binds every family and display, and where no design-based estimator exists the result is
    blocked and recorded, its exit the sample-only attestation. The weights, the outcome model and
    the simulation here read the rows as sampled. So under the population answer each g-method
    shows its outcome-free diagnostics (the weights, positivity), computes no estimate, and offers
    the attestation, which the survey question records."""
    from turbotab.core.models.survey import SAMPLE_EXIT

    frame = feedback_cohort(n=900)
    frame["wt"] = np.random.default_rng(1).uniform(500, 5000, frame["pid"].nunique())[
        pd.factorize(frame["pid"])[0]]
    survey = d.SurveySpec(estimand="population", weight="wt", acknowledged=True)
    roles = {**feedback_state().roles, "wt": "design"}
    lane = MSM_LANE.model_copy(update={"truncation": "p1_p99"})
    art = run_stage(frame, feedback_state(time_varying=lane, survey=survey, roles=roles),
                    tmp_path, "population")
    assert art["diagnostics"]["weights"] and art["diagnostics"]["positivity"]
    assert art["estimates"] is None
    assert art["reason"].startswith("The g-methods here have no design-based estimator")
    assert art["exits"] == [{"label": SAMPLE_EXIT,
                             "decision": {"kind": "set_survey", "estimand": "sample"}}]
    g = run_stage(frame, feedback_state(time_varying=G_LANE.model_copy(update={
        "exposure": "dash", "confounders": ["sbp"], "baseline": ["female", "age"]}),
        survey=survey, roles=roles), tmp_path, "population_g")
    assert g["diagnostics"]["positivity"] and g["estimates"] is None and g["exits"] == art["exits"]


def test_4_confounding_affected_by_prior_exposure_biases_standard_regression_but_not_the_msm():
    """Why the leash is set where it is (Robins, Hernán & Brumback 2000). In the DASH cohort the
    diet has no effect on the event by construction, and blood pressure is both a confounder of
    the later diet and a consequence of the earlier diet, through an unmeasured vascular risk that
    also causes the event. On 20,000 people (no loss to follow-up), statsmodels' pooled logistic
    regressions put the cumulative diet's coefficient far from 0 whether they adjust for blood
    pressure (|z| > 5) or not (|z| > 10). The MSM with stabilized weights is within 3 robust SEs
    of 0."""
    import statsmodels.api as sm

    f = feedback_cohort(n=20_000, censoring=False)
    f = f.sort_values(["pid", "visit"], kind="mergesort").reset_index(drop=True)
    f["t"] = f["visit"] - 1.0
    f["t2"] = f["t"] ** 2
    f["lagdash"] = lagged_within(f["dash"].astype(float), f["pid"], 0.0)
    f["cum"] = f.groupby("pid")["dash"].cumsum().astype(float)
    groups = pd.factorize(f["pid"])[0]

    def standard(cols: list[str]) -> float:
        fit = sm.GLM(f["cvd"], sm.add_constant(f[cols]), family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": groups})
        return float(fit.params["cum"] / fit.bse["cum"])

    assert standard(["cum", "sbp", "female", "age", "t", "t2"]) > 5
    assert standard(["cum", "female", "age", "t", "t2"]) > 10
    w = tv.ipw_weights(f, id="pid", time="t", indicator="dash",
                       numerator=["female", "age", "lagdash", "t", "t2"],
                       denominator=["female", "age", "sbp", "lagdash", "t", "t2"], kind="all")
    X, names = tv.design(f, ["cum", "female", "age", "t", "t2"])
    row = tv.fit_msm(X, names, f["cvd"], w.weights, f["pid"], "binomial").row("cum")
    assert abs(row["estimate"]) < 3 * row["se"], row
    assert row["ci_low"] < 0 < row["ci_high"]


def test_4_another_exposure_re_asks_the_lane():
    """MODELING_SEQUENCE §2 ("Exposure declared … invalidates the adjustment-set answers given for
    another exposure"), for the lane: declared for ``dash``, it answers nothing once the estimand
    names ``sbp``, and the question is asked again. A lane for another exposure is refused, with
    the exit that names the declared one."""
    lane = MSM_LANE.model_copy(update={"truncation": "none"})
    state = feedback_state(time_varying=lane)
    assert _step(state).status == "answered"
    other = state.model_copy(update={"estimand": d.EstimandSpec(exposure="sbp",
                                                                measure="odds_ratio"),
                                     "roles": {**state.roles, "sbp": "exposure",
                                               "dash": "covariate"}})
    assert _step(other).status in ("open", "waiting")
    refused = _refused(d.SetTimeVarying(**{**lane.model_dump(), "exposure": "sbp",
                                           "confounders": []}), _ctx(feedback_state()))
    assert refused.code == "other_exposure"
    assert refused.exits[0]["decision"]["exposure"] == "dash"


# ── 5 · the methods sentence, verbatim ───────────────────────────────────────


MSM_METHODS = (
    "The effect of `dash` on `cvd` was estimated by a marginal structural model with stabilized "
    "inverse-probability weights (Robins, Hernán & Brumback 2000). The weights came from pooled "
    "logistic models of `dash` at each of 6 time points, given its previous value, `female`, `age` "
    "and time and its square (numerator) and also `sbp` (denominator); loss to follow-up (`lost`: "
    "1 on a unit's last time point before it was lost) was weighted by the inverse probability of "
    "having stayed through the previous time point, from the same models given current `dash` as "
    "well; the weights were truncated at the 1st and 99th percentiles (mean {mean}, from {lo} to "
    "{hi}). The outcome model was a weighted pooled logistic model of `cvd` on the number of time "
    "points exposed so far, `female`, `age` and time and its square, with a variance clustered by "
    "`pid` that treats the weights as known, so its interval is conservative (Hernán, Brumback & "
    "Robins 2000). `sbp` is a time-varying confounder affected by prior `dash`, which standard "
    "regression cannot adjust for. The weights' distribution and positivity at each time point "
    "were read before any estimate was shown. The estimate assumes no unmeasured confounding at "
    "each time point given the history, positivity, and that each time point's `dash` precedes "
    "the outcome it is paired with, as declared.")

G_METHODS = (
    "The effect of `diet` on `event` was estimated by the parametric g-formula (Robins 1986; "
    "McGrath et al. 2020). Each time-varying confounder, `htn` (logistic), was modeled given the "
    "baseline covariates (`female`), the confounders before it at that time point, every "
    "variable's previous value, time and its square; `event` at each time point was modeled by a pooled logistic model given `diet`, the "
    "confounders, the baseline covariates, the previous values, time and its square. The risks of "
    "`event` by the last of 5 time points had every unit always, and never, been exposed were "
    "simulated for 20,000 units drawn from the observed first time point (Monte Carlo error at "
    "most {mc}), with 95% intervals from 100 bootstrap resamples of units (percentiles). `htn` is "
    "a time-varying confounder affected by prior `diet`, which standard regression cannot adjust "
    "for. The natural course's simulated risk was compared with the observed risk as a check of "
    "the models. Positivity at each time point was read before any estimate was shown. The "
    "estimate assumes no unmeasured confounding at each time point given the history, positivity, "
    "and that each time point's `diet` precedes the outcome it is paired with, as declared.")


def test_5_the_methods_sentences_verbatim(tmp_path):
    """The record's sentence for each lane (``voice.sentence_for``), and the stage's methods text
    (the §13 contract's clause through ``contracts.paragraph``) for the weights and for the
    g-formula. The numbers in the latter are the weights' mean and range after truncation, and the
    larger Monte Carlo error of the two strategies, each from the artifact's own fields."""
    from turbotab.core import voice

    state = feedback_state()
    lane = d.SetTimeVarying(**{**MSM_LANE.model_dump(), "truncation": "p1_p99"})
    assert voice.sentence_for(lane, state) == (
        "`dash` changes over time (it can stop and restart), so its effect on `cvd` is estimated by "
        "a marginal structural model with stabilized inverse-probability weights: the weights "
        "model `dash` at each time point from `sbp` (time-varying), `female` and `age` (baseline) "
        "and its history, loss to follow-up (`lost`) is weighted the same way, and the weights are "
        "truncated at the 1st and 99th percentiles; `dash` at each time point precedes the outcome "
        "it is paired with, as declared.")
    open_lane = d.SetTimeVarying(**MSM_LANE.model_dump())
    assert voice.sentence_for(open_lane, state).endswith(
        "and the truncation is declared after the weights' diagnostics are read; `dash` at each "
        "time point precedes the outcome it is paired with, as declared.")
    g = d.SetTimeVarying(**G_LANE.model_dump())
    assert voice.sentence_for(g, gformula_state()) == (
        "`diet` changes over time, so its effect on `event` is estimated by the parametric "
        "g-formula: `htn` is simulated forward from each time point's history, and the risks had "
        "every unit always and never been exposed are compared, over 20,000 simulated units with "
        "100 bootstrap resamples; `diet` at each time point precedes the outcome it is paired with, "
        "as declared.")

    art = run_stage(feedback_cohort(n=1500),
                    feedback_state(time_varying=MSM_LANE.model_copy(update={"truncation": "p1_p99"})),
                    tmp_path, "msm")
    used = next(o for o in art["diagnostics"]["truncation"] if o["chosen"])["summary"]
    assert art["methods"] == MSM_METHODS.format(mean=f"{used['mean']:.2f}",
                                                lo=f"{used['min']:.2f}", hi=f"{used['max']:.2f}")
    art = run_stage(gformula_cohort(), gformula_state(time_varying=G_LANE), tmp_path, "g")
    curves = {c["strategy"]: c for c in art["estimates"]["curves"]}
    mc = max(curves["always"]["mc_se"], curves["never"]["mc_se"])
    assert art["methods"] == G_METHODS.format(mc=f"{mc:.4f}")


# ── the g-formula through the stage ──────────────────────────────────────────


def test_2_through_the_stage_the_g_formula_covers_the_truth_and_checks_its_models(tmp_path):
    """The lane run by the real graph on the exact-truth cohort (4,000 people; 20,000 simulated
    units; 100 bootstrap resamples). The 95% intervals of the risks under "always" and "never"
    and of their difference cover the enumerated truth. The natural course reproduces the observed
    risk at every time point within 0.01, the check of the models a correctly specified cohort must
    pass. The E-value reads the risk ratio, and positivity comes before the risks."""
    art = run_stage(gformula_cohort(), gformula_state(time_varying=G_LANE), tmp_path, "g")
    rows = {r["label"]: r for r in art["estimates"]["rows"]}
    truth = {"always": exact_risks(5, 1), "never": exact_risks(5, 0)}
    for label, value in (("Risk if always exposed", truth["always"]),
                         ("Risk if never exposed", truth["never"]),
                         ("Risk difference (always − never)", truth["always"] - truth["never"])):
        assert rows[label]["ci_low"] < value < rows[label]["ci_high"], (label, rows[label], value)
    curves = {c["strategy"]: c["risks"] for c in art["estimates"]["curves"]}
    assert np.max(np.abs(np.subtract(curves["natural"], curves["observed"]))) < 0.01
    ratio = rows["Risk ratio (always / never)"]
    ev = art["estimates"]["e_value"]
    assert ev["rr"] == pytest.approx(ratio["estimate"]) and ev["point"] > 1
    assert len(art["diagnostics"]["positivity"]) == 5
    assert art["estimates"]["bootstrap"] == 100 and art["estimates"]["failed_resamples"] == 0


# ── the §13 contract and its chain test ──────────────────────────────────────


def test_contract_declares_every_part_section_13_asks_for():
    """Slot, data scope, needs, routing (a question, options labeled customary and sound for each
    purpose with a rung), storyboard, sentence (its clause) and relations, in the one registry
    (``turbotab.core.contracts``, wave 1's): the decision kind and the stage it names exist, its
    sentence and every relation's ``enforced_by`` name live code, its conflict carries its exits,
    and its run order places the lane in the model slot."""
    import importlib

    from turbotab.core.contracts import CONTRACTS, contracts, options_for, run_order
    from turbotab.core.stages import build_graph
    from turbotab.core.time_varying import KEY

    assert KEY in contracts()
    c = CONTRACTS[KEY]
    assert (c.slot, c.scope) == ("model", "model")
    assert c.needs and c.question.endswith("?") and c.storyboard and c.clause is not None
    assert c.decision == "set_time_varying" and c.decision in d.SLOTS
    assert c.stage in build_graph() and "step 3" in c.place
    for target in [c.sentence, *(r.enforced_by for r in c.relations if r.enforced_by)]:
        module, name = target.split(":")
        assert callable(getattr(importlib.import_module(module), name)), target
    assert all(r.exits for r in c.relations if r.kind == "conflicts")
    for purpose in ("inference", "prediction"):
        for o in options_for(KEY, purpose):
            assert o["customary"] and o["sound"] and o["rung"]
    assert {o["rung"] for o in options_for(KEY, "prediction")} == {"refused"}
    assert run_order([KEY]) == [KEY]
    kinds = {(r.kind, r.target) for r in c.relations}
    assert ("conflicts", "standard_adjustment_for_affected_confounders") in kinds
    assert ("invalidates", "estimand") in kinds


def test_chain_every_relation_the_lane_declares_fires(tmp_path):
    """MODELING_SEQUENCE §6, the chain test, for both g-methods and for standard regression. For
    each lane, every relation the contract fires (``contracts.fired``) has its consequence
    in view where §13 says the chain shows it:

    * **implies** ``weight_diagnostics_before_estimates``: diagnostics first, no estimate until the
      truncation, then estimates with the same diagnostics;
    * **implies** ``positivity_by_time_point``: a row per time point;
    * **implies** ``intervals_by_unit``: the MSM's variance clustered by unit (its units counted),
      and the g-formula's bootstrap by unit. This is §2's "repeated units … imply cluster-aware
      intervals";
    * **implies** ``censoring_weighted``: censoring weights in the diagnostics and the sentence;
    * **implies** ``unmeasured_confounding_sensitivity``: the E-value (§0 ruling 10);
    * **invalidates** ``estimand``: another exposure re-asks the lane (§2's "exposure declared …
      invalidates");
    * **disables** ``standard_estimate_as_the_effect``: the served fit's row for the exposure is
      labeled not the effect (``estimand.annotate_fit``);
    * **enables** ``risks_under_always_and_never``: the g-formula's four rows;
    * **conflicts** ``standard_adjustment_for_affected_confounders``: block and record, then the
      recorded note on the fit.

    The artifact's ``relations`` are the fired relations' sentences, in the contract's words."""
    from turbotab.core import estimand
    from turbotab.core.contracts import fired
    from turbotab.core.time_varying import CONTRACT, KEY

    targets = {r.target for r in CONTRACT.relations}
    frame = feedback_cohort(n=1200)

    # the weights: before, then after the truncation
    before = run_stage(frame, feedback_state(time_varying=MSM_LANE), tmp_path, "before")
    assert before["diagnostics"]["weights"] and before["estimates"] is None
    lane = MSM_LANE.model_copy(update={"truncation": "p1_p99"})
    after = run_stage(frame, feedback_state(time_varying=lane), tmp_path, "after")
    said = {f.relation.target: f.says for f in fired({KEY: "msm_iptw"}, "inference",
                                                     consequences=sorted(targets))}
    assert set(said) == {"weight_diagnostics_before_estimates", "positivity_by_time_point",
                         "intervals_by_unit", "censoring_weighted",
                         "unmeasured_confounding_sensitivity", "estimand",
                         "standard_estimate_as_the_effect"}
    assert after["diagnostics"]["weights"] == before["diagnostics"]["weights"]
    assert len(after["diagnostics"]["positivity"]) == 6
    assert after["estimates"]["n_units"] == frame["pid"].nunique()
    assert after["diagnostics"]["censoring_weights"] is not None and "loss to follow-up" in after["methods"]
    assert after["estimates"]["e_value"]["point"] >= 1
    for target in ("weight_diagnostics_before_estimates", "positivity_by_time_point",
                   "intervals_by_unit", "censoring_weighted", "unmeasured_confounding_sensitivity",
                   "estimand"):
        assert said[target] in after["relations"], target
    state = feedback_state(time_varying=lane)
    fit = estimand.annotate_fit({"models": [], "task": "binary"}, state)
    assert fit["estimand"]["time_varying"] == (
        "The estimate of `dash` is the time-varying lane's (a marginal structural model with "
        "stabilized inverse-probability weights). This model cannot adjust for `sbp`, a "
        "confounder affected by prior `dash`, so its row for `dash` is not the effect.")
    moved = state.model_copy(update={"estimand": d.EstimandSpec(exposure="sbp",
                                                                measure="odds_ratio"),
                                     "roles": {**state.roles, "sbp": "exposure",
                                               "dash": "covariate"}})
    assert _step(state).status == "answered" and _step(moved).status != "answered"

    # the g-formula
    g = run_stage(gformula_cohort(n=2000), gformula_state(
        time_varying=G_LANE.model_copy(update={"simulations": 5_000})), tmp_path, "g")
    said = {f.relation.target: f.says for f in fired({KEY: "gformula"}, "inference",
                                                     consequences=sorted(targets))}
    assert "risks_under_always_and_never" in said and "weight_diagnostics_before_estimates" not in said
    assert [r["measure"] for r in g["estimates"]["rows"]] == ["risk", "risk", "risk_difference",
                                                              "risk_ratio"]
    assert said["risks_under_always_and_never"] in g["relations"]
    assert said["intervals_by_unit"] in g["relations"] and g["estimates"]["bootstrap"] == 100

    # standard regression: the conflict, then the record
    said = {f.relation.target: f.says for f in fired({KEY: "standard"}, "inference",
                                                     consequences=sorted(targets))}
    assert set(said) == {"standard_adjustment_for_affected_confounders", "estimand"}
    standard = d.SetTimeVarying(exposure="dash", method="standard",
                                ordering="exposure_precedes_outcome")
    assert _refused(standard, _ctx(feedback_state())).code == "affected_confounder"
    kept = feedback_state(time_varying=d.TimeVaryingSpec(**{**standard.model_dump(exclude={"kind"}),
                                                            "acknowledged": True}))
    art = run_stage(frame, kept, tmp_path, "standard")
    assert said["standard_adjustment_for_affected_confounders"] in art["relations"]
    assert art["estimates"] is None
    fit = estimand.annotate_fit({"models": [], "task": "binary"}, kept)
    assert fit["estimand"]["time_varying"] == (
        "Recorded: standard regression over `sbp`, a confounder affected by prior `dash`; this row "
        "for `dash` is biased by it.")


# ── sensitivity to unmeasured confounding, against R ─────────────────────────

EVALUE_R = """
suppressMessages(library(EValue))
row <- function(m) unname(m["E-values", ])
out(list(rr = row(evalues.RR(est = 0.65, lo = 0.56, hi = 0.76)),
         rr_cross = row(evalues.RR(est = 1.3, lo = 0.9, hi = 1.8)),
         or_common = row(evalues.OR(est = 1.8, lo = 1.2, hi = 2.6, rare = FALSE)),
         or_rare = row(evalues.OR(est = 0.7, lo = 0.55, hi = 0.88, rare = TRUE)),
         md = row(evalues.MD(est = 0.4, se = 0.1))))
"""


@needs_r
def test_e_values_match_EValue(tmp_path):
    """The E-values the lane reports (VanderWeele & Ding 2017) against R ``EValue`` to 1e-10: a
    risk ratio below 1 and one whose interval crosses 1, an odds ratio read as a risk ratio for a
    rare and for a common outcome, and a standardized mean difference."""
    r = run_r(EVALUE_R, {}, tmp_path / "r")

    def ci(values: list[Any]) -> float:
        return next(float(v) for v in values[1:] if v is not None and v != "NA")

    cases = {"rr": tv.e_value(0.65, 0.56, 0.76), "rr_cross": tv.e_value(1.3, 0.9, 1.8),
             "or_common": tv.e_value_or(1.8, 1.2, 2.6, rare=False),
             "or_rare": tv.e_value_or(0.7, 0.55, 0.88, rare=True),
             "md": tv.e_value_md(0.4, 0.1)}
    for key, mine in cases.items():
        assert mine["point"] == pytest.approx(float(r[key][0]), abs=1e-10), key
        assert mine["ci"] == pytest.approx(ci(r[key]), abs=1e-10), key
