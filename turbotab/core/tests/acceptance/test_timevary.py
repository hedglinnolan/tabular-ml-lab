"""TIMEVARY · A time-varying exposure by g-methods (V2_DEFINITION_OF_DONE §2, the causal row).

The package's acceptance, numbered as its spec:

1. A marginal structural model with stabilized inverse-probability-of-treatment weights (and
   censoring weights) for a time-varying exposure on long data. The weights agree with R
   ``ipw::ipwtm`` to 1e-8, and the MSM estimate and its CR0 robust SE with R ``geepack::geeglm``
   (independence working correlation) to 1e-6. This holds on ``ipw``'s own ``haartdat`` (an
   exposure that, once started, stays, and loss to follow-up), on ``gfoRmula``'s
   ``basicdata_nocomp`` (an exposure that switches), on a repeated continuous outcome, and through
   the app's own stage on a cohort table. The stage reports the small-sample interval
   MODELING_SEQUENCE §2 asks of repeated units: CR2 with Bell–McCaffrey degrees of freedom (checked
   against every matrix written out), and none below the unit floor.
2. The parametric g-formula. On ``gfoRmula``'s example data, the risks under "always", "never" and
   the natural course agree with R ``gfoRmula`` within Monte Carlo error, with **50,000 simulated
   units on each side**, and every fitted model's coefficients agree to 1e-6. On a simulated cohort
   whose truth is computed exactly by enumeration, the risks fall within four bootstrap standard
   errors of it. What the run will take is measured before it; bootstrap resamples that cannot fit
   the models are counted, and above 1% no interval is reported.
3. Diagnostics before estimates: the weight distribution (stabilized mean near 1), the truncation
   options with their stated trade-off, and positivity at each time point; for the g-formula,
   positivity and what the simulation will take. These are shown first, and the truncation (or the
   simulation's size) is refused until this lane's diagnostics on the current data have been shown.
   The declaration carries their key, and no estimate is computed unless the diagnostics computed
   now have that key. A model that cannot be fit is named, and the diagnostics stay.
4. Routing. The lane needs repeated measures with a settled time column (numbers, labels in a
   declared order, or dates) and a declared time ordering. Time-varying confounders affected by
   prior exposure require g-methods; standard regression adjustment for them is block and record,
   with an exit to this lane. Standard regression of a concurrently measured exposure is block and
   record too. A simulation with a known null shows why. Loss to follow-up without an indicator is
   stated as an assumption.
5. The methods sentence, asserted verbatim.

Also: the §13 contract with the chain test that every relation it declares fires, and the E-values
against R ``EValue`` and by hand (a pooled logistic model's ratio read as a hazard ratio, its
rarity the cumulative risk by the end of follow-up).

Every expected value comes from an independent path: R (``timevary_fixtures.run_r``; skipped
without ``Rscript``), statsmodels or NumPy written out here (``references.cr2_by_definition``), an
exact enumeration, or a simulation's known truth. The app's values come through its own path:
``models/time_varying.py``, the ``time_varying`` stage run by the real graph, the Router, the
validators and the voice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import t as student_t

from turbotab.core import decisions as d
from turbotab.core import plan_lock
from turbotab.core.decisions import ProjectState, Refusal
from turbotab.core.interview import route
from turbotab.core.models import time_varying as tv
from turbotab.core.tests.acceptance.references import cr2_by_definition
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
FEEDBACK_COLUMNS = ["pid", "visit", "dash", "sbp", "female", "age", "lost", "cvd"]


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


# The g-formula's lane as first declared (its simulation's size comes after its diagnostics).
G_LANE = d.TimeVaryingSpec(exposure="diet", method="gformula", ordering="exposure_precedes_outcome",
                           confounders=["htn"], baseline=["female"])
G_SIZE = {"simulations": 20_000, "bootstrap": 100}
G_COLUMNS = ["pid", "visit", "diet", "htn", "female", "event"]


def _ctx(state: ProjectState, columns: list[str] | None = None, **artifacts: Any) -> dict[str, Any]:
    return {"state": state, "columns": columns or FEEDBACK_COLUMNS,
            "artifact": lambda stage: artifacts.get(stage)}


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


@dataclass
class Staged:
    """A lane as a user meets it: ``first`` the artifact of the lane declared without what follows
    its diagnostics; ``second`` the artifact once that declaration is validated against ``first``
    (the server's stamp) and recorded; ``state`` the state then; ``seconds`` the second run's time."""

    first: dict[str, Any]
    second: dict[str, Any]
    state: ProjectState
    seconds: float


def staged(frame: pd.DataFrame, state: ProjectState, folder: Path, name: str,
           columns: list[str] | None = None, **declared: Any) -> Staged:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.csv"
    frame.to_csv(path, index=False)
    graph = GraphRun(path, folder / name)
    try:
        first = graph.public(graph.run(state, upto=["time_varying"]))["time_varying"]
        lane = d.SetTimeVarying(**{**state.time_varying.model_dump(), **declared})
        stamped = d.validate(lane, _ctx(state, columns or list(frame.columns), time_varying=first))
        done = state.model_copy(update={"time_varying": d.TimeVaryingSpec(
            **stamped.model_dump(exclude={"kind"}))})
        second = graph.public(graph.run(done, upto=["time_varying"]))["time_varying"]
        return Staged(first, second, done, graph.seconds["time_varying"])
    finally:
        graph.close()


def lagged_within(values: pd.Series, units: pd.Series, first: float) -> np.ndarray:
    """Each row's value at its unit's previous row (``first`` at the unit's first row)."""
    return values.groupby(units, sort=False).shift(1).fillna(first).to_numpy(float)


def _refused(decision: Any, ctx: dict[str, Any]) -> Refusal:
    with pytest.raises(Refusal) as refused:
        d.validate(decision, ctx)
    return refused.value


def _step(state: ProjectState, artifacts: dict[str, Any] | None = None) -> Any:
    return next(s for s in route(state, ALL_FRESH, artifacts or {}) if s.key == "time_varying")


def _hr_e_value_by_hand(hr: float, lo: float | None, hi: float | None,
                        rare: bool) -> tuple[float, float | None]:
    """VanderWeele & Ding (2017), written out: a hazard ratio of a common outcome as
    ``(1 − 0.5^√HR)/(1 − 0.5^√(1/HR))``; the E-value ``RR + √(RR(RR − 1))`` of the ratio away from
    1; the confidence limit nearer 1 the same, or 1 when the interval holds 1."""
    def rr(x: float) -> float:
        return x if rare else (1 - 0.5 ** math.sqrt(x)) / (1 - 0.5 ** math.sqrt(1 / x))

    def e(r: float) -> float:
        r = r if r >= 1 else 1 / r
        return r + math.sqrt(r * (r - 1))

    point = e(rr(hr))
    if lo is None or hi is None:
        return point, None
    near = rr(lo) if hr > 1 else rr(hi)
    holds = (near <= 1) if hr > 1 else (near >= 1)
    return point, 1.0 if holds else e(near)


def _km_risk(frame: pd.DataFrame, time: str, y: str) -> float:
    """The cumulative risk by the last time point, by pandas: the share of each time point's rows
    with the event, ``1 − Π(1 − h_t)``."""
    h = frame.groupby(time)[y].mean().sort_index().to_numpy(float)
    return float(1 - np.prod(1 - h))


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
    """The weighted outcome model and its unit-clustered sandwich (``fit_msm``, ``variance =
    "CR0"``) against ``geeglm(..., weights, id, corstr = "independence")`` on R's own weights, to
    1e-6 for every coefficient and its robust SE: a pooled logistic MSM in current (HAART) and in
    cumulative exposure (``basicdata_nocomp``), and a linear MSM of a repeated measure (glucose)."""
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
        fit = tv.fit_msm(X, names, data["event"], w, data["patient"], "binomial", variance="CR0")
    elif case == "basic":
        data = _basic(tmp_path)
        r = run_r(MSM_R[case], {"basic": data}, tmp_path / "r")
        w = tv.ipw_weights(data, id="id", time="t0", indicator="A",
                           numerator=["L3", "lagA", "t0", "t02"],
                           denominator=["L3", "L1", "L2", "lagA", "t0", "t02"], kind="all").weights
        X, names = tv.design(data, ["cumA", "L3", "t0", "t02"])
        fit = tv.fit_msm(X, names, data["Y"], w, data["id"], "binomial", variance="CR0")
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
        fit = tv.fit_msm(X, names, data["glucose"], w, data["pid"], "gaussian", variance="CR0")
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
out(list(coef = unname(s[2, "Estimate"]), mean = mean(w), sd = sd(w), min = min(w), max = max(w),
         q = qs, tmean = mean(wt), tmin = min(wt), tmax = max(wt), X = unname(model.matrix(m)),
         mu = unname(as.vector(fitted(m))), wt = d$wt, y = d$cvd, unit = d$unit))
"""


@needs_r
def test_1_through_the_stage_the_weights_and_the_estimate_are_R_s(tmp_path):
    """The app's own path. The ``time_varying`` stage, run by the real graph on a cohort table
    (visits numbered 1–6, a confounder affected by prior exposure, loss to follow-up), builds its
    time index, lags, design and weights from the declared lane. Its weight summary agrees with R's
    on the same CSV to 1e-8: ``ipwtm`` exposure and loss weights, multiplied, summarized by R's
    ``mean``, ``sd``, ``min``, ``max`` and type-7 ``quantile``. Its estimate (truncated at the 1st
    and 99th percentiles) agrees with ``geeglm`` on R's truncated weights to 1e-6, on the log
    scale. Its interval is CR2 with Bell–McCaffrey df: from R's design, fitted values and weights,
    the working linear model ``W^½X``, ``√w(y − μ)/√v`` is written out and
    ``references.cr2_by_definition`` gives the SE and df (to 1e-6), and the interval is
    ``exp(β ± t_df·SE)``. The E-value, from that ratio and interval, is the hazard ratio's, worked
    by hand with the cumulative risk pandas computes."""
    frame = feedback_cohort(n=300)
    run = staged(frame, feedback_state(time_varying=MSM_LANE), tmp_path, "cohort",
                 truncation="p1_p99")
    art = run.second
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
    X, mu, wt = np.asarray(r["X"], float), np.asarray(r["mu"], float), np.asarray(r["wt"], float)
    y, unit = np.asarray(r["y"], float), np.asarray(r["unit"])
    v = mu * (1 - mu)
    V, df = cr2_by_definition(X * np.sqrt(wt * v)[:, None], np.sqrt(wt) * (y - mu) / np.sqrt(v),
                              unit)
    se, nu = math.sqrt(V[1, 1]), float(df[1])
    q = float(student_t.ppf(0.975, nu))
    [row] = art["estimates"]["rows"]
    assert math.log(row["estimate"]) == pytest.approx(r["coef"], abs=1e-6)
    assert row["se"] == pytest.approx(se, abs=1e-6)
    assert row["df"] == pytest.approx(nu, rel=1e-6)
    assert math.log(row["ci_low"]) == pytest.approx(r["coef"] - q * se, abs=1e-6)
    assert math.log(row["ci_high"]) == pytest.approx(r["coef"] + q * se, abs=1e-6)
    assert art["estimates"]["n_units"] == frame["pid"].nunique()
    assert art["estimates"]["n_rows"] == len(frame)
    risk = _km_risk(frame, "visit", "cvd")
    point, limit = _hr_e_value_by_hand(row["estimate"], row["ci_low"], row["ci_high"],
                                       rare=risk < 0.15)
    assert art["estimates"]["e_value"]["point"] == pytest.approx(point, abs=1e-10)
    assert art["estimates"]["e_value"]["ci"] == pytest.approx(limit, abs=1e-10)


def test_1_a_few_units_widen_the_interval_by_their_df_and_below_the_floor_none_is_reported(
        tmp_path, monkeypatch):
    """MODELING_SEQUENCE §2: repeated units imply a small-sample sandwich, with refusal below a
    floor. On 60 people (8 events) the stage's interval is CR2 on t with Bell–McCaffrey df: the
    weights by statsmodels (``_independent_weights``), the weighted logistic fit by statsmodels'
    GLM, and the CR2 SE and df by every matrix written out agree with the stage's to 1e-6. The df
    are far below 60, so the interval is wider than CR0 on a normal reference (the clustered
    sandwich written out here, no correction) would make it. With the unit floor raised above the cohort
    (the floor is the seal's, ``inference.min_clusters``), no interval, SE or p-value is reported,
    the estimate stands alone with its reason, and the E-value is the estimate's only."""
    import statsmodels.api as sm

    frame = feedback_cohort(n=60, seed=3)
    run = staged(frame, feedback_state(time_varying=MSM_LANE), tmp_path, "few",
                 truncation="none")
    ref = _independent_weights(frame)
    ref["cum"] = ref.groupby("pid")["dash"].cumsum().astype(float)
    X = sm.add_constant(ref[["cum", "female", "age", "t", "t2"]].astype(float))
    fit = sm.GLM(ref["cvd"].astype(float), X, family=sm.families.Binomial(),
                 var_weights=ref["w"]).fit(tol=1e-14, maxiter=100)
    mu, w, y = fit.fittedvalues.to_numpy(), ref["w"].to_numpy(), ref["cvd"].to_numpy(float)
    v = mu * (1 - mu)
    codes = pd.factorize(ref["pid"])[0]
    Xw, ew = X.to_numpy() * np.sqrt(w * v)[:, None], np.sqrt(w) * (y - mu) / np.sqrt(v)
    V, df = cr2_by_definition(Xw, ew, codes)
    [row] = run.second["estimates"]["rows"]
    assert math.log(row["estimate"]) == pytest.approx(fit.params["cum"], abs=1e-6)
    assert row["se"] == pytest.approx(math.sqrt(V[1, 1]), abs=1e-6)
    assert row["df"] == pytest.approx(df[1], rel=1e-6) and row["df"] < 60
    bread = np.linalg.inv(Xw.T @ Xw)
    by_unit = np.zeros((codes.max() + 1, Xw.shape[1]))
    np.add.at(by_unit, codes, Xw * ew[:, None])
    cr0 = bread @ by_unit.T @ by_unit @ bread
    normal = 1.959963984540054 * math.sqrt(cr0[1, 1])
    assert math.log(row["ci_high"] / row["ci_low"]) / 2 > normal

    monkeypatch.setattr("turbotab.core.models.inference.min_clusters", lambda: 100)
    run = staged(frame, feedback_state(time_varying=MSM_LANE), tmp_path, "floor",
                 truncation="none")
    floor = "`pid` has 60 units, fewer than the 100 a cluster-robust interval needs"
    assert any(c.startswith(floor) for c in run.first["diagnostics"]["concerns"])
    [row] = run.second["estimates"]["rows"]
    assert row["estimate"] > 0
    assert all(row[k] is None for k in ("ci_low", "ci_high", "se", "df", "p"))
    assert run.second["estimates"]["concerns"] == [f"{floor}: no interval or p-value is reported."]
    assert run.second["estimates"]["e_value"]["ci"] is None
    assert f"and no interval: {floor}." in run.second["methods"]


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


def test_2_through_the_stage_the_g_formula_covers_the_truth_and_says_first_what_it_will_take(
        tmp_path):
    """The lane run by the real graph on the exact-truth cohort (4,000 people), declared as a user
    declares it. First, positivity at each time point and what the simulation will take at the
    default size (10,000 simulated units, 500 resamples), measured on these rows, and no risk. Then
    20,000 simulated units and 100 resamples, declared after them: the 95% intervals of the risks
    under "always" and "never" and of their difference cover the enumerated truth; the natural
    course reproduces the observed risk at every time point within 0.01, the check of the models a
    correctly specified cohort must pass; the E-value reads the risk ratio. The time measured
    beforehand, rescaled to the declared size by its own parts, is within a factor of 3 of what the
    run took (V2 gate 5: an estimate shown first)."""
    run = staged(gformula_cohort(), gformula_state(time_varying=G_LANE), tmp_path, "g",
                 columns=G_COLUMNS, **G_SIZE)
    cost = run.first["diagnostics"]["cost"]
    assert run.first["estimates"] is None and len(run.first["diagnostics"]["positivity"]) == 5
    assert (cost["simulations"], cost["bootstrap"]) == (10_000, 500)
    assert cost["seconds"] == pytest.approx(cost["fit_seconds"] + 3 * 10_000 * cost["unit_seconds"]
                                            + 500 * cost["resample_seconds"])
    declared = (cost["fit_seconds"] + 3 * G_SIZE["simulations"] * cost["unit_seconds"]
                + G_SIZE["bootstrap"] * cost["resample_seconds"])
    assert 1 / 3 < run.seconds / declared < 3, (run.seconds, declared)
    art = run.second
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


def test_2_resamples_that_cannot_fit_are_counted_and_above_one_percent_withhold_the_interval(
        tmp_path, monkeypatch):
    """On 60 people (12 events), more than 1 in 100 bootstrap resamples cannot fit the g-formula's
    models (the module's own count on its resamples shows this cohort's do fail): the resamples
    that fail are the ones whose drawn units hold too few events, so an interval from the rest
    would be too narrow (the verifier saw 41 of 100 drop silently, the interval from the survivors).
    Above 1% failed, no interval is reported: every row's limits are empty, the count is named in
    the estimates' concerns and in the methods sentence, and the E-value is the estimate's alone.
    At or below 1% the interval is reported with the count named; when every resample fails, the
    stage still answers (no lookup of a missing interval)."""
    from turbotab.core.stages.time_varying import bootstrap_verdict

    frame = gformula_cohort(n=60, seed=3)
    expected = tv.gformula_bootstrap(frame, GTRUTH_SPEC, reps=100, n_sim=None, seed=1)["failed"]
    assert expected > 1  # this cohort's resamples do fail, by the module's own count
    run = staged(frame, gformula_state(time_varying=G_LANE), tmp_path, "few", columns=G_COLUMNS,
                 simulations=1_000, bootstrap=100)
    est = run.second["estimates"]
    failed = est["failed_resamples"]
    assert failed > 1 and est["bootstrap"] == 100
    assert all(r["ci_low"] is None and r["ci_high"] is None for r in est["rows"])
    assert est["concerns"][0].startswith(f"{failed:,} of 100 bootstrap resamples could not fit the "
                                         f"models")
    assert est["concerns"][0].endswith("No interval is reported; a model with fewer terms, or more "
                                       "units, is the way to one.")
    assert (f"; no interval is reported, because {failed:,} of 100 bootstrap resamples of units "
            f"could not fit the models") in run.second["methods"]
    assert est["e_value"]["ci"] is None and est["e_value"]["point"] >= 1
    assert bootstrap_verdict(100, 0) == (True, None)
    assert bootstrap_verdict(100, 1)[0] and "from the other 99" in bootstrap_verdict(100, 1)[1]
    assert not bootstrap_verdict(100, 2)[0] and not bootstrap_verdict(100, 100)[0]

    def none_fit(data: Any, spec: Any, *, reps: int, **kw: Any) -> dict[str, Any]:
        return {"intervals": {}, "reps": reps, "failed": reps, "draws": {}}

    monkeypatch.setattr(tv, "gformula_bootstrap", none_fit)
    art = run_stage(frame, run.state, tmp_path, "none")
    assert art["estimates"]["failed_resamples"] == 100
    assert all(r["ci_low"] is None for r in art["estimates"]["rows"])


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
    estimate, the key of what the diagnostics were computed for, and the Router keeps the question
    open."""
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
    assert art["diagnosed"]["key"] and art["diagnosed"]["confounders"] == ["sbp"]
    step = _step(feedback_state(time_varying=MSM_LANE), {"time_varying": art})
    assert step.status == "open"


def test_3_a_truncation_or_a_simulation_size_before_its_diagnostics_is_refused(tmp_path):
    """The verifier's two flows, closed at the decision. A marginal structural model posted with
    its truncation in the first decision is refused (``diagnostics_first``) while no artifact holds
    this lane's weights; its exit is the lane without the truncation, which the validators pass. So
    is one whose served diagnostics were for other columns, or another method. Once this lane's
    diagnostics are served, the truncation passes and carries their key: the server's, whatever a
    client sends. The g-formula's simulation size is refused the same way, and standard regression,
    which declares nothing after diagnostics, carries no key. The affected-confounder refusal's
    g-method exits declare no truncation or size, so they lead to the diagnostics first."""
    frame = feedback_cohort(n=900)
    state = feedback_state()
    msm = d.SetTimeVarying(**{**MSM_LANE.model_dump(), "truncation": "p1_p99"})
    refused = _refused(msm, _ctx(state))
    assert refused.code == "diagnostics_first"
    assert refused.message.startswith("The truncation is declared after the weights' distribution")
    exit_ = refused.exits[0]["decision"]
    assert exit_["truncation"] is None and exit_["method"] == "msm_iptw"
    assert d.validate(exit_, _ctx(state)).truncation is None

    shown = run_stage(frame, feedback_state(time_varying=MSM_LANE), tmp_path, "msm")
    other = run_stage(frame, feedback_state(time_varying=MSM_LANE.model_copy(
        update={"baseline": ["female"], "confounders": ["sbp", "age"]})), tmp_path, "other")
    assert _refused(msm, _ctx(state, time_varying=other)).code == "diagnostics_first"
    forged = msm.model_copy(update={"diagnostics_seen": "a key the client made up"})
    stamped = d.validate(forged, _ctx(state, time_varying=shown))
    assert stamped.diagnostics_seen == shown["diagnosed"]["key"] != "a key the client made up"
    open_again = d.validate(msm.model_copy(update={"truncation": None,
                                                   "diagnostics_seen": "x"}), _ctx(state))
    assert open_again.diagnostics_seen is None

    g = d.SetTimeVarying(exposure="dash", method="gformula", ordering="exposure_precedes_outcome",
                         confounders=["sbp"], baseline=["female", "age"], simulations=5_000)
    refused = _refused(g, _ctx(state, time_varying=shown))  # the weights' diagnostics, not its own
    assert refused.code == "diagnostics_first"
    assert refused.exits[0]["decision"]["simulations"] is None
    standard = d.SetTimeVarying(exposure="dash", method="standard",
                                ordering="exposure_precedes_outcome", acknowledged=True)
    assert d.validate(standard, _ctx(state)).diagnostics_seen is None
    exits = _refused(standard.model_copy(update={"acknowledged": False}), _ctx(state)).exits
    for e in exits[:2]:
        assert e["decision"]["truncation"] is None and e["decision"]["simulations"] is None
        d.validate(e["decision"], _ctx(state))


def test_3_the_g_formula_shows_positivity_and_its_time_before_any_risk(tmp_path):
    """The verifier's first flow, at the stage: the g-formula declared without its size returns
    positivity at each time point (pandas counts the exposed and unexposed rows) and what the
    simulation will take, and no risk; the Router keeps the question open and the plan is not
    locked. Declared afterwards, the risks come with the same positivity, and the methods sentence
    says the size was declared after them."""
    frame = gformula_cohort(n=2000)
    run = staged(frame, gformula_state(time_varying=G_LANE), tmp_path, "g", columns=G_COLUMNS,
                 simulations=5_000, bootstrap=100)
    first = run.first
    assert first["estimates"] is None and first["diagnostics"]["cost"]["seconds"] > 0
    assert first["withheld"] == ("Read positivity and what the simulation will take first: no risk "
                                 "is simulated until its size is declared.")
    assert not plan_lock.shows_estimates("time_varying", first)
    by_visit = frame.groupby("visit")["diet"]
    for p in first["diagnostics"]["positivity"]:
        assert p["exposed"] == int(by_visit.sum().iloc[p["time"]])
        assert p["unexposed"] == int((by_visit.size() - by_visit.sum()).iloc[p["time"]])
    assert _step(gformula_state(time_varying=G_LANE), {"time_varying": first}).status == "open"
    assert run.second["diagnostics"]["positivity"] == first["diagnostics"]["positivity"]
    assert run.second["estimates"]["rows"]
    assert ("The simulation's size was declared after positivity at each time point and what the "
            "simulation would take were read, and no risk was simulated before it.") in \
        run.second["methods"]
    assert _step(run.state, {"time_varying": run.second}).status == "answered"


def test_3_data_changed_after_the_declaration_withholds_the_estimate_and_asks_again(tmp_path):
    """"For the current data": the truncation was declared after the weights on one table were
    read; the table then changes (100 people leave it). The stage computes the new weights, finds
    their key is not the one the truncation carries, withholds the estimate and offers the exit that
    declares it again. The Router, reading that artifact, asks the question again. Declared again
    against the new diagnostics, the estimate comes."""
    frame = feedback_cohort(n=1200)
    run = staged(frame, feedback_state(time_varying=MSM_LANE), tmp_path, "a", truncation="none")
    assert run.second["estimates"]["rows"]
    fewer = frame[~frame["pid"].isin(frame["pid"].unique()[:100])]
    moved = run_stage(fewer, run.state, tmp_path, "b")
    assert moved["estimates"] is None
    assert moved["diagnosed"]["key"] != run.state.time_varying.diagnostics_seen
    assert moved["withheld"].startswith("These weights are not the ones read when the truncation "
                                        "was declared")
    assert moved["exits"][0]["label"] == "Keep the weights not truncated, having read these"
    assert _step(run.state, {"time_varying": moved}).status == "open"
    again = d.validate(moved["exits"][0]["decision"], _ctx(run.state, time_varying=moved))
    assert again.diagnostics_seen == moved["diagnosed"]["key"]
    final = run_stage(fewer, run.state.model_copy(update={"time_varying": d.TimeVaryingSpec(
        **again.model_dump(exclude={"kind"}))}), tmp_path, "c")
    assert final["estimates"]["rows"] and final["estimates"]["n_units"] == 1100


def test_3_through_the_server_no_estimate_comes_before_its_diagnostics(tmp_path):
    """The verifier's two flows, through the real HTTP API on a fresh project (the DASH cohort,
    600 people, visits kept as rows, its adjustment answers the fixture's truth):

    * the g-formula posted first is recorded, and its artifact holds positivity at each time point
      and what the simulation will take, and no estimate; the question stays open;
    * a marginal structural model posted with its truncation in the first decision is refused 409
      ``diagnostics_first``, and nothing is estimated; its exit (the lane without the truncation)
      is recorded and serves the weights and positivity, with their key;
    * the truncation posted then is recorded with that key (the server's stamp) and the sentence
      saying it came after the diagnostics; the estimate follows, with its CR2 df, and the methods
      sentence says the truncation was declared after the diagnostics were read."""
    import time

    from turbotab.core.tests.acceptance.server_drive import local_server, open_project, settle_forms
    from turbotab.core.tests.truths import Truth

    path = tmp_path / "cohort.csv"
    feedback_cohort(n=600).to_csv(path, index=False)
    truth = Truth({"adjust:sbp": "yes,yes,yes", "adjust:female": "yes,yes,no",
                   "adjust:age": "yes,yes,no", "code_or_count:female": "code",
                   "code_or_count:age": "amount", "cluster:pid": "yes"},
                  fixture="the DASH cohort")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "cvd"})
        drive.answer("event", {"kind": "set_event", "column": "cvd", "level": "1"})
        drive.reach("purpose")
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.answer("grain", {"kind": "set_grain", "grain": "repeated", "id_column": "pid"})
        drive.answer("repeat_kind", {"kind": "set_repeat_kind", "repeat_kind": "time_points",
                                     "time_column": "visit"})
        drive.answer("unit", {"kind": "set_unit", "unit": "row"})
        drive.answer("temporal", {"kind": "set_temporal", "temporal": False})
        drive.reach("roles")
        drive.decide_roles(dict(feedback_state().roles))
        drive.exposure = "dash"
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.answer_plan("dash")
        assert drive.reach("time_varying")["status"] == "open"
        # The card is read before the lane is declared, as a person reads it: a declaration naming
        # no covariate takes the card's proposal (the confounders `sbp` among them), which exists
        # only once the stage has read the data. Posted sooner, on a loaded machine, the decision
        # met no proposal and was refused for leaving `sbp` out (wave 2a repairs integration).
        drive.artifact("time_varying")

        g = drive.post({"kind": "set_time_varying", "exposure": "dash", "method": "gformula",
                        "ordering": "exposure_precedes_outcome"})
        assert g.status_code == 200, g.text[:600]
        art = drive.artifact("time_varying")
        assert art["estimates"] is None and len(art["diagnostics"]["positivity"]) == 6
        assert art["diagnostics"]["cost"]["seconds"] > 0
        assert drive.reach("time_varying")["status"] == "open"

        first = drive.post({"kind": "set_time_varying", "exposure": "dash", "method": "msm_iptw",
                            "ordering": "exposure_precedes_outcome", "truncation": "p1_p99"})
        assert first.status_code == 409, first.text[:600]
        error = first.json()["error"]
        assert error["code"] == "diagnostics_first"
        assert drive.view()["state"]["time_varying"]["method"] == "gformula"  # nothing recorded
        drive.decide(error["exits"][0]["decision"])
        shown = drive.artifact("time_varying")
        assert shown["estimates"] is None and shown["diagnostics"]["weights"]
        key = shown["diagnosed"]["key"]

        declared = drive.post({**error["exits"][0]["decision"], "truncation": "p1_p99",
                               "diagnostics_seen": "made up by a client"})
        assert declared.status_code == 200, declared.text[:600]
        record = drive.view()["decisions"][-1]
        assert record["decision"]["diagnostics_seen"] == key
        assert ("the weights are truncated at the 1st and 99th percentiles, as declared after their "
                "diagnostics were read") in record["sentence"]
        # FORM (wave 2b): the form question follows the lane's, and every estimate waits for it.
        settle_forms(drive)
        end = time.monotonic() + 240
        while True:
            art = drive.artifact("time_varying")
            if art.get("estimates") or time.monotonic() > end:
                break
            time.sleep(0.1)
    [row] = art["estimates"]["rows"]
    assert row["df"] > 0 and row["ci_low"] < row["estimate"] < row["ci_high"]
    assert ("The truncation was declared after the weights' distribution and positivity at each "
            "time point were read, and no estimate was computed before it.") in art["methods"]


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
    assert art["reason"].startswith("The weights cannot be estimated: the denominator model of "
                                    "`dash`")
    assert art["reason"].endswith("a positivity violation).")


def test_3_a_model_that_cannot_be_fit_is_named_and_the_diagnostics_stay(tmp_path):
    """Wrong reason, closed. An outcome with 2 events: the marginal structural model's weights and
    positivity are computed and kept, and the weighted outcome model's failure is named as the
    outcome model's, with its event count, never as a positivity violation. The g-formula's
    positivity is kept too, and its failure names the outcome model with the events pandas
    counts."""
    frame = feedback_cohort(n=900)
    frame["cvd"] = 0
    first_rows = frame.groupby("pid").head(1).index[:2]
    frame.loc[first_rows, "cvd"] = 1
    frame = frame[~(frame["pid"].isin(frame.loc[first_rows, "pid"]) & (frame["visit"] > 1))]
    run = staged(frame, feedback_state(time_varying=MSM_LANE), tmp_path, "msm", truncation="none")
    art = run.second
    assert len(art["diagnostics"]["positivity"]) == 6 and art["diagnostics"]["weights"]
    assert art["estimates"] is None
    assert art["reason"].startswith("The weighted outcome model cannot be fit:")
    assert f"(`cvd` has 2 events in {len(frame):,} rows)" in art["reason"]
    assert "positivity" not in art["reason"]

    g = frame.rename(columns={"dash": "diet", "sbp": "htn_raw", "cvd": "event"})
    g["htn"] = (g["htn_raw"] > 130).astype(int)
    first = run_stage(g[["pid", "visit", "diet", "htn", "female", "event"]],
                      gformula_state(time_varying=G_LANE), tmp_path, "g")
    assert len(first["diagnostics"]["positivity"]) == 6 and first["estimates"] is None
    assert first["reason"].startswith("The g-formula's models cannot be fit: the outcome model:")
    assert f"(2 events in {len(g):,} rows for" in first["reason"]
    assert "positivity" not in first["reason"]


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
        "A family of study factors": feedback_state(
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


def test_4_the_lane_needs_repeated_measures_a_settled_time_column_and_a_declared_order():
    """Each need is refused until met, each refusal with its way forward:

    * repeated measures (the grain), time points (the repeats answer, with the exit that answers
      it) and records kept as rows (the exit keeps them);
    * a settled time column: one only proposed by the repeats reading is asked, and the exit is the
      ledger's own ``confirm_reading``. Confirmed, the lane passes (BLUEPRINT §14.1);
    * for a g-method, a declared time ordering: "same time" and "unknown" are refused, because an
      exposure measured with its outcome cannot be told from a consequence of it. The exits declare
      the order, or keep standard regression recorded as concurrent (which passes)."""
    lane = d.SetTimeVarying(**MSM_LANE.model_dump())
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

    unaffected = feedback_state(adjustment={
        **feedback_state().adjustment, "sbp": d.AdjustmentAnswer(
            exposure="dash", causes_exposure="yes", causes_outcome="yes", after_exposure="no")})
    for ordering in ("same_time", "unknown"):
        refused = _refused(lane.model_copy(update={"ordering": ordering}), _ctx(feedback_state()))
        assert refused.code == "time_ordering"
        assert "g-methods need each time point's value of what you study to precede" in refused.message
        assert refused.exits[0]["decision"]["ordering"] == "exposure_precedes_outcome"
        standard = refused.exits[1]["decision"]
        assert standard["method"] == "standard" and standard["ordering_acknowledged"]
        d.validate(standard, _ctx(unaffected))


def test_4_standard_regression_of_a_concurrent_exposure_is_block_and_record(tmp_path):
    """The verifier's "too tight", fixed to the nearest honest version. Diet and a biomarker
    measured at the same visit are a common design, and standard regression can describe how they
    go together. So for standard regression "same time" (or "unknown") is block and record: refused
    until attested, with the order and the attestation as exits; recorded, its sentence says so,
    the fit's row is labeled open to reverse causation, and the stage fires the conflict. With a
    confounder affected by prior exposure too, each attestation is its own, and a g-method exit
    declares the order only in words that say so."""
    from turbotab.core import estimand, voice
    from turbotab.core.contracts import fired
    from turbotab.core.time_varying import KEY

    unaffected = feedback_state(adjustment={
        **feedback_state().adjustment, "sbp": d.AdjustmentAnswer(
            exposure="dash", causes_exposure="yes", causes_outcome="yes", after_exposure="no")})
    concurrent = d.SetTimeVarying(exposure="dash", method="standard", ordering="same_time")
    refused = _refused(concurrent, _ctx(unaffected))
    assert refused.code == "time_ordering"
    assert "reverse causation" in refused.message
    assert [e["label"] for e in refused.exits] == [
        "What you study precedes the outcome it is paired with",
        "Keep standard regression; record that what you study is measured with its outcome"]
    recorded = d.validate(refused.exits[1]["decision"], _ctx(unaffected))
    assert recorded.ordering == "same_time" and recorded.ordering_acknowledged
    assert voice.sentence_for(recorded, unaffected) == (
        "`dash` changes over time; no time-varying confounder is affected by earlier `dash`, so "
        "standard regression estimates its effect on `cvd`; `dash` is measured at the same time as "
        "the outcome it is paired with, so its estimate cannot be told from a consequence of the "
        "outcome (reverse causation), as recorded.")
    kept = unaffected.model_copy(update={"time_varying": d.TimeVaryingSpec(
        **recorded.model_dump(exclude={"kind"}))})
    fit = estimand.annotate_fit({"models": [], "task": "binary"}, kept)
    assert fit["estimand"]["time_varying"] == (
        "Recorded: `dash` is measured at the same time as the outcome it is paired with, so this "
        "row for `dash` cannot be told from a consequence of the outcome (reverse causation).")
    art = run_stage(feedback_cohort(n=600), kept, tmp_path, "concurrent")
    said = {f.relation.target: f.says for f in fired({KEY: "standard"}, "inference",
                                                     consequences=["concurrent_exposure_and_outcome"])}
    assert said["concurrent_exposure_and_outcome"] in art["relations"]

    both = _refused(recorded, _ctx(feedback_state()))
    assert both.code == "affected_confounder"
    labels = [e["label"] for e in both.exits]
    assert labels[0] == ("Declare that what you study precedes its outcome; estimate by a marginal "
                         "structural model (weights)")
    assert both.exits[0]["decision"]["ordering"] == "exposure_precedes_outcome"
    assert both.exits[2]["decision"]["ordering"] == "same_time"
    d.validate(both.exits[0]["decision"], _ctx(feedback_state()))
    d.validate(both.exits[2]["decision"], _ctx(feedback_state()))


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
    d.validate(msm, _ctx(state))
    recorded = d.validate(attest, _ctx(state))
    assert voice.sentence_for(recorded, state) == (
        "`dash` changes over time and its effect on `cvd` is estimated by standard regression, as "
        "recorded, although `sbp` is a confounder that earlier `dash` changed: adjusting for it "
        "removes part of the effect, and leaving it out leaves later exposure confounded (Robins, "
        "Hernán & Brumback 2000); `dash` at each time point precedes the outcome it is paired "
        "with, as declared.")
    left_out = _refused(d.SetTimeVarying(**{**MSM_LANE.model_dump(), "confounders": [],
                                            "baseline": ["female", "age"]}), _ctx(state))
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


def test_4_loss_to_follow_up_without_an_indicator_is_stated_and_the_values_propose_one(tmp_path):
    """No loss-to-follow-up indicator declared on a cohort that loses people. The setting counts
    the units whose rows end before the last visit without the event (pandas agrees), and reads
    ``lost`` (excluded; 1 only on a unit's last row, never with the event) as the one indicator the
    values show, a proposal only. The concern and the methods sentence state the assumption the
    estimate then rests on, and the relation fires. The affected-confounder refusal's g-method
    exits carry ``lost``, and say so in their labels."""
    from turbotab.core.contracts import fired
    from turbotab.core.time_varying import KEY

    frame = feedback_cohort(n=1200)
    lane = MSM_LANE.model_copy(update={"censoring": None})
    run = staged(frame, feedback_state(time_varying=lane), tmp_path, "lost", truncation="none")
    last = frame.groupby("pid")["visit"].transform("max") == frame["visit"]
    early = int((last & (frame["visit"] < 6) & (frame["cvd"] == 0)).sum())
    setting = run.first["setting"]
    assert setting["ending_early"] == early > 0
    assert setting["censoring_candidates"] == ["lost"] and setting["proposal"]["censoring"] == "lost"
    said = (f"{early:,} units' rows end before the last time point without the event, and no "
            f"loss-to-follow-up indicator is declared: the estimate assumes their loss is "
            f"unrelated to the outcome. `lost` reads as one: 1 only on a unit's last row, never "
            f"with the event.")
    assert said in run.first["diagnostics"]["concerns"]
    assert (f"No loss-to-follow-up indicator was declared: {early:,} units' rows end before the last "
            f"time point without the event, and the estimate assumes their loss is unrelated to the "
            f"outcome.") in run.second["methods"]
    relation = {f.relation.target: f.says for f in fired(
        {KEY: "msm_iptw"}, "inference", consequences=["loss_assumed_independent"])}
    assert relation["loss_assumed_independent"] in run.second["relations"]

    standard = d.SetTimeVarying(exposure="dash", method="standard",
                                ordering="exposure_precedes_outcome")
    exits = _refused(standard, _ctx(feedback_state(), time_varying=run.first)).exits
    assert exits[0]["label"] == ("Estimate by a marginal structural model (weights), weighting "
                                 "loss to follow-up by `lost`")
    assert exits[0]["decision"]["censoring"] == exits[1]["decision"]["censoring"] == "lost"
    assert exits[2]["decision"]["censoring"] is None


def test_4_a_time_column_of_dates_orders_the_rows(tmp_path):
    """The verifier's dead end. Visit dates written as ISO text are read by the ingest as dates,
    and the working table hands them on as Timestamps; the time column named with the repeats answer
    orders each unit's rows by date. The stage's weights then equal statsmodels' computed on the
    visit numbers the dates stand for (``_independent_weights``), to 1e-6, with one time point per
    date. Text dates are read by the app's own reader, and dates that read two ways (month or day
    first) are sent to the date-reading repair, not to the labels' order."""
    from turbotab.core.stages.time_varying import _time_points

    frame = feedback_cohort(n=900)
    dated = frame.assign(visit_date=(pd.Timestamp("2019-03-01") + pd.to_timedelta(
        (frame["visit"] - 1) * 365, unit="D")).dt.strftime("%Y-%m-%d")).drop(columns=["visit"])
    state = feedback_state(
        time_varying=MSM_LANE,
        repeat_kind=d.RepeatSpec(repeat_kind="time_points", time_column="visit_date"),
        roles={**feedback_state().roles, "visit_date": "time"})
    state.roles.pop("visit")
    art = run_stage(dated, state, tmp_path, "dates")
    assert art["setting"]["time_kind"] == "dates"
    assert art["setting"]["time_points"] == dated["visit_date"].nunique() == 6
    w = _independent_weights(frame)["w"].to_numpy()
    for key, value in _numpy_summary(w).items():
        assert art["diagnostics"]["weights"][key] == pytest.approx(value, rel=1e-6, abs=1e-9), key

    stamps = pd.Series([pd.Timestamp("2020-01-02"), pd.Timestamp("2019-05-01")], dtype=object)
    assert _time_points(stamps, ["2019-05-01"])[0].tolist() == [1, 0]
    points, kind = _time_points(pd.Series(["14 Feb 2021", "02 Jan 2020", "14 Feb 2021"],
                                          dtype=object), None)
    assert kind == "dates" and points.tolist() == [1, 0, 1]
    assert _time_points(pd.Series(["3/1/2019", "4/2/2019"], dtype=object), None) == (None,
                                                                                    "ambiguous")


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
    g = _refused(d.SetTimeVarying(**MSM_LANE.model_dump()), ctx)
    assert g.code == "exposure_not_binary" and "takes 409 values" in g.message
    standard = d.SetTimeVarying(exposure="dash", method="standard",
                                ordering="exposure_precedes_outcome")
    refused = _refused(standard, ctx)
    assert refused.code == "affected_confounder"
    assert refused.message.endswith("The g-methods here need a study factor of 0 or 1 at each time "
                                    "point, and `dash` takes 409 values.")
    assert [e["label"] for e in refused.exits] == [
        "Keep standard regression; record that the estimate is biased by it",
        "Declare a yes/no study factor (the question on what you study)"]
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
    shows its diagnostics (the weights, positivity), computes no estimate once declared, and offers
    the attestation, which the survey question records."""
    from turbotab.core.models.survey import SAMPLE_EXIT

    frame = feedback_cohort(n=900)
    frame["wt"] = np.random.default_rng(1).uniform(500, 5000, frame["pid"].nunique())[
        pd.factorize(frame["pid"])[0]]
    survey = d.SurveySpec(estimand="population", weight="wt", acknowledged=True)
    roles = {**feedback_state().roles, "wt": "design"}
    run = staged(frame, feedback_state(time_varying=MSM_LANE, survey=survey, roles=roles),
                 tmp_path, "population", truncation="p1_p99")
    art = run.second
    assert art["diagnostics"]["weights"] and art["diagnostics"]["positivity"]
    assert art["estimates"] is None
    assert art["reason"].startswith("The g-methods here have no design-based estimator")
    assert art["exits"] == [{"label": SAMPLE_EXIT,
                             "decision": {"kind": "set_survey", "estimand": "sample"}}]
    g_lane = G_LANE.model_copy(update={"exposure": "dash", "confounders": ["sbp"],
                                       "baseline": ["female", "age"]})
    g = staged(frame, feedback_state(time_varying=g_lane, survey=survey, roles=roles), tmp_path,
               "population_g", simulations=1_000, bootstrap=100).second
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
    refused = _refused(d.SetTimeVarying(**{**MSM_LANE.model_dump(), "exposure": "sbp",
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
    "points exposed so far, `female`, `age` and time and its square, with a CR2 variance clustered "
    "by `pid` and Bell–McCaffrey degrees of freedom (Bell & McCaffrey 2002) that treats the "
    "weights as known, so its interval is conservative (Hernán, Brumback & Robins 2000). `sbp` is "
    "a time-varying confounder affected by prior `dash`, which standard regression cannot adjust "
    "for. The truncation was declared after the weights' distribution and positivity at each time "
    "point were read, and no estimate was computed before it. The estimate assumes no unmeasured "
    "confounding at each time point given the history, positivity, and that each time point's "
    "`dash` precedes the outcome it is paired with, as declared.")

G_METHODS = (
    "The effect of `diet` on `event` was estimated by the parametric g-formula (Robins 1986; "
    "McGrath et al. 2020). Each time-varying confounder, `htn` (logistic), was modeled given the "
    "baseline covariates (`female`), the confounders before it at that time point, every "
    "variable's previous value, time and its square; `event` at each time point was modeled by a "
    "pooled logistic model given `diet`, the confounders, the baseline covariates, the previous "
    "values, time and its square. The risks of `event` by the last of 5 time points had every unit "
    "always, and never, been exposed were simulated for 20,000 units drawn from the observed first "
    "time point (Monte Carlo error at most {mc}), with 95% intervals from 100 bootstrap resamples "
    "of units (percentiles). `htn` is a time-varying confounder affected by prior `diet`, which "
    "standard regression cannot adjust for. The natural course's simulated risk was compared with "
    "the observed risk as a check of the models. The simulation's size was declared after "
    "positivity at each time point and what the simulation would take were read, and no risk was "
    "simulated before it. The estimate assumes no unmeasured confounding at each time point given "
    "the history, positivity, and that each time point's `diet` precedes the outcome it is paired "
    "with, as declared.")


def test_5_the_methods_sentences_verbatim(tmp_path):
    """The record's sentence for each lane (``voice.sentence_for``), and the stage's methods text
    (the §13 contract's clause through ``contracts.paragraph``) for the weights and for the
    g-formula. The numbers in the latter are the weights' mean and range after truncation, and the
    larger Monte Carlo error of the two strategies, each from the artifact's own fields. Before the
    truncation is declared the sentence says it comes after the diagnostics, not that it did."""
    from turbotab.core import voice

    state = feedback_state()
    lane = d.SetTimeVarying(**{**MSM_LANE.model_dump(), "truncation": "p1_p99"})
    assert voice.sentence_for(lane, state) == (
        "`dash` changes over time (it can stop and restart), so its effect on `cvd` is estimated by "
        "a marginal structural model with stabilized inverse-probability weights: the weights "
        "model `dash` at each time point from `sbp` (time-varying), `female` and `age` (baseline) "
        "and its history, loss to follow-up (`lost`) is weighted the same way, and the weights are "
        "truncated at the 1st and 99th percentiles, as declared after their diagnostics were read; "
        "`dash` at each time point precedes the outcome it is paired with, as declared.")
    open_lane = d.SetTimeVarying(**MSM_LANE.model_dump())
    assert voice.sentence_for(open_lane, state).endswith(
        "and the truncation is declared after the weights' diagnostics are read; `dash` at each "
        "time point precedes the outcome it is paired with, as declared.")
    g = d.SetTimeVarying(**{**G_LANE.model_dump(), **G_SIZE})
    assert voice.sentence_for(g, gformula_state()) == (
        "`diet` changes over time, so its effect on `event` is estimated by the parametric "
        "g-formula: `htn` is simulated forward from each time point's history, and the risks had "
        "every unit always and never been exposed are compared, over 20,000 simulated units with "
        "100 bootstrap resamples, a size declared after positivity and the simulation's time were "
        "read; `diet` at each time point precedes the outcome it is paired with, as declared.")
    assert voice.sentence_for(d.SetTimeVarying(**G_LANE.model_dump()), gformula_state()) == (
        "`diet` changes over time, so its effect on `event` is estimated by the parametric "
        "g-formula: `htn` is simulated forward from each time point's history, and the risks had "
        "every unit always and never been exposed are compared; the simulation's size is declared "
        "after positivity and the time the simulation will take are read; `diet` at each time "
        "point precedes the outcome it is paired with, as declared.")

    run = staged(feedback_cohort(n=1500), feedback_state(time_varying=MSM_LANE), tmp_path, "msm",
                 truncation="p1_p99")
    used = next(o for o in run.second["diagnostics"]["truncation"] if o["chosen"])["summary"]
    assert run.second["methods"] == MSM_METHODS.format(
        mean=f"{used['mean']:.2f}", lo=f"{used['min']:.2f}", hi=f"{used['max']:.2f}")
    assert ("The truncation is declared after the weights' distribution and positivity at each "
            "time point are read; no estimate is computed before it.") in run.first["methods"]
    g_run = staged(gformula_cohort(), gformula_state(time_varying=G_LANE), tmp_path, "g",
                   columns=G_COLUMNS, **G_SIZE)
    curves = {c["strategy"]: c for c in g_run.second["estimates"]["curves"]}
    mc = max(curves["always"]["mc_se"], curves["never"]["mc_se"])
    assert g_run.second["methods"] == G_METHODS.format(mc=f"{mc:.4f}")


# ── the §13 contract and its chain test ──────────────────────────────────────


def test_contract_declares_every_part_section_13_asks_for():
    """Slot, data scope, needs, routing (a question, options labeled customary and sound for each
    purpose with a rung), storyboard, sentence (its clause) and relations, in the one registry
    (``turbotab.core.contracts``, wave 1's): the decision kind and the stage it names exist, its
    sentence and every relation's ``enforced_by`` name live code, its conflicts carry their exits,
    and its run order places the lane in the model slot. The scope note does not claim the
    diagnostics read no outcome (under an event, the rows at risk of loss are the outcome's)."""
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
    assert ("conflicts", "concurrent_exposure_and_outcome") in kinds
    assert ("invalidates", "estimand") in kinds
    assert "read no outcome" not in c.scope_note
    assert "risk of loss to follow-up" in c.scope_note


def test_chain_every_relation_the_lane_declares_fires(tmp_path):
    """MODELING_SEQUENCE §6, the chain test, for both g-methods and for standard regression. For
    each lane, every relation the contract fires (``contracts.fired``) has its consequence
    in view where §13 says the chain shows it:

    * **implies** ``weight_diagnostics_before_estimates``: diagnostics first, no estimate until the
      truncation, then estimates with the same diagnostics;
    * **implies** ``positivity_and_time_before_estimates``: the g-formula's positivity and time
      first, no risk until its size;
    * **implies** ``positivity_by_time_point``: a row per time point;
    * **implies** ``intervals_by_unit``: the MSM's CR2 variance clustered by unit (its units counted,
      its df reported), and the g-formula's bootstrap by unit. This is §2's "repeated units …
      imply cluster-aware intervals";
    * **implies** ``censoring_weighted``: censoring weights in the diagnostics and the sentence;
    * **implies** ``loss_assumed_independent``: fired in
      ``test_4_loss_to_follow_up_without_an_indicator_is_stated_and_the_values_propose_one``;
    * **implies** ``unmeasured_confounding_sensitivity``: the E-value (§0 ruling 10);
    * **invalidates** ``estimand``: another exposure re-asks the lane (§2's "exposure declared …
      invalidates");
    * **disables** ``standard_estimate_as_the_effect``: the served fit's row for the exposure is
      labeled not the effect (``estimand.annotate_fit``);
    * **enables** ``risks_under_always_and_never``: the g-formula's four rows;
    * **conflicts** ``standard_adjustment_for_affected_confounders``: block and record, then the
      recorded note on the fit;
    * **conflicts** ``concurrent_exposure_and_outcome``: fired in
      ``test_4_standard_regression_of_a_concurrent_exposure_is_block_and_record``.

    The artifact's ``relations`` are the fired relations' sentences, in the contract's words."""
    from turbotab.core import estimand
    from turbotab.core.contracts import fired
    from turbotab.core.time_varying import CONTRACT, KEY

    targets = {r.target for r in CONTRACT.relations}
    frame = feedback_cohort(n=1200)

    # the weights: before, then after the truncation
    run = staged(frame, feedback_state(time_varying=MSM_LANE), tmp_path, "msm",
                 truncation="p1_p99")
    before, after = run.first, run.second
    assert before["diagnostics"]["weights"] and before["estimates"] is None
    said = {f.relation.target: f.says for f in fired({KEY: "msm_iptw"}, "inference",
                                                     consequences=sorted(targets))}
    assert set(said) == {"weight_diagnostics_before_estimates", "positivity_by_time_point",
                         "intervals_by_unit", "censoring_weighted", "loss_assumed_independent",
                         "unmeasured_confounding_sensitivity", "estimand",
                         "standard_estimate_as_the_effect"}
    assert after["diagnostics"]["weights"] == before["diagnostics"]["weights"]
    assert len(after["diagnostics"]["positivity"]) == 6
    assert after["estimates"]["n_units"] == frame["pid"].nunique()
    assert after["estimates"]["rows"][0]["df"] > 0
    assert after["diagnostics"]["censoring_weights"] is not None and "loss to follow-up" in after["methods"]
    assert after["estimates"]["e_value"]["point"] >= 1
    for target in ("weight_diagnostics_before_estimates", "positivity_by_time_point",
                   "intervals_by_unit", "censoring_weighted", "unmeasured_confounding_sensitivity",
                   "estimand"):
        assert said[target] in after["relations"], target
    state = run.state
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
    g_run = staged(gformula_cohort(n=2000), gformula_state(time_varying=G_LANE), tmp_path, "g",
                   columns=G_COLUMNS, simulations=5_000, bootstrap=100)
    g = g_run.second
    said = {f.relation.target: f.says for f in fired({KEY: "gformula"}, "inference",
                                                     consequences=sorted(targets))}
    assert "risks_under_always_and_never" in said and "weight_diagnostics_before_estimates" not in said
    assert said["positivity_and_time_before_estimates"] in g_run.first["relations"]
    assert [r["measure"] for r in g["estimates"]["rows"]] == ["risk", "risk", "risk_difference",
                                                              "risk_ratio"]
    assert said["risks_under_always_and_never"] in g["relations"]
    assert said["intervals_by_unit"] in g["relations"] and g["estimates"]["bootstrap"] == 100

    # standard regression: the conflict, then the record
    said = {f.relation.target: f.says for f in fired({KEY: "standard"}, "inference",
                                                     consequences=sorted(targets))}
    assert set(said) == {"standard_adjustment_for_affected_confounders",
                         "concurrent_exposure_and_outcome", "estimand"}
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


# ── sensitivity to unmeasured confounding, against R and by hand ─────────────

EVALUE_R = """
suppressMessages(library(EValue))
row <- function(m) unname(m["E-values", ])
out(list(rr = row(evalues.RR(est = 0.65, lo = 0.56, hi = 0.76)),
         rr_cross = row(evalues.RR(est = 1.3, lo = 0.9, hi = 1.8)),
         or_common = row(evalues.OR(est = 1.8, lo = 1.2, hi = 2.6, rare = FALSE)),
         or_rare = row(evalues.OR(est = 0.7, lo = 0.55, hi = 0.88, rare = TRUE)),
         hr_common = row(evalues.HR(est = 1.6, lo = 1.25, hi = 2.05, rare = FALSE)),
         hr_rare = row(evalues.HR(est = 0.7, lo = 0.55, hi = 0.9, rare = TRUE)),
         md = row(evalues.MD(est = 0.4, se = 0.1))))
"""


@needs_r
def test_e_values_match_EValue(tmp_path):
    """The E-values the lane reports (VanderWeele & Ding 2017) against R ``EValue`` to 1e-10: a
    risk ratio below 1 and one whose interval crosses 1, an odds ratio read as a risk ratio for a
    rare and for a common outcome, a hazard ratio for a common and for a rare outcome, and a
    standardized mean difference."""
    r = run_r(EVALUE_R, {}, tmp_path / "r")

    def ci(values: list[Any]) -> float:
        return next(float(v) for v in values[1:] if v is not None and v != "NA")

    cases = {"rr": tv.e_value(0.65, 0.56, 0.76), "rr_cross": tv.e_value(1.3, 0.9, 1.8),
             "or_common": tv.e_value_or(1.8, 1.2, 2.6, rare=False),
             "or_rare": tv.e_value_or(0.7, 0.55, 0.88, rare=True),
             "hr_common": tv.e_value_hr(1.6, 1.25, 2.05, rare=False),
             "hr_rare": tv.e_value_hr(0.7, 0.55, 0.9, rare=True),
             "md": tv.e_value_md(0.4, 0.1)}
    for key, mine in cases.items():
        assert mine["point"] == pytest.approx(float(r[key][0]), abs=1e-10), key
        assert mine["ci"] == pytest.approx(ci(r[key]), abs=1e-10), key


def test_e_value_reads_the_pooled_logistic_ratio_as_a_hazard_ratio_by_cumulative_risk(tmp_path):
    """The verifier's case: an event on 6–7% of rows but in about a third of people by the last
    visit. The pooled logistic model of an event at each time point is a discrete-time hazard
    model, so its ratio is read as a hazard ratio, and VanderWeele & Ding's rule for a hazard ratio
    judges rarity by the outcome at the end of follow-up: here common (the cumulative risk pandas
    computes, above 15%), though below 15% of rows. The E-value is then
    ``(1 − 0.5^√HR)/(1 − 0.5^√(1/HR))``'s, worked by hand, and ``reads`` says why."""
    frame = feedback_cohort(n=1500, c0=-3.0)
    risk = _km_risk(frame, "visit", "cvd")
    assert frame["cvd"].mean() < 0.15 <= risk
    run = staged(frame, feedback_state(time_varying=MSM_LANE), tmp_path, "common",
                 truncation="none")
    [row] = run.second["estimates"]["rows"]
    ev = run.second["estimates"]["e_value"]
    point, limit = _hr_e_value_by_hand(row["estimate"], row["ci_low"], row["ci_high"], rare=False)
    assert ev["point"] == pytest.approx(point, abs=1e-10)
    assert ev["ci"] == pytest.approx(limit, abs=1e-10)
    assert ev["reads"] == (f"the pooled logistic odds ratio as a hazard ratio of an outcome common "
                           f"by the end of follow-up ({risk:.1%} cumulative risk), as "
                           f"(1 − 0.5^√HR)/(1 − 0.5^√(1/HR))")
