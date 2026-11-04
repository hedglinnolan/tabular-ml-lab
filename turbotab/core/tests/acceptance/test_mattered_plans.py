"""C1 (PP2): the curve of which decisions mattered over the plan's own pipeline, against R.

The ``mattered`` stage fits each specification whose plan holds a pipeline step as that plan,
through the path the secondary and sensitivity stages take (``sensitivity.fit_on_rows``), instead
of refusing the curve. The fixture is ``test_mattered``'s: the ESTIMAND cohort with blanks in
``activity`` (adjusted for) and ``bmi`` (left out), two exclusion rules, eight specifications.

* **A single fill and a spline** (truth a): the fill done here by hand in numpy, each blank the
  median of its column on the specification's own rows (``SimpleImputer``'s rule, which the plan's
  fill uses), then R's ``lm(glucose ~ fiber + sex + rms::rcs(age, 4) + smoking + activity)`` on
  those rows, the knots placed by ``rms`` on them, with ``sandwich::vcovHC(type = "HC3")`` on
  t(n − p). Every point matches R to 1e-8.
* **Multiple imputation** (truth b): each specification's own copies, as the engine drew them,
  each fitted here by statsmodels' least squares with HC3 on a matrix written out by hand, and
  pooled by ``mice::pool.scalar`` (mice 3.19.0) with ``dfcom = n − p``: the estimate, the interval
  on t with Barnard–Rubin degrees of freedom, the fraction of missing information, and the Monte
  Carlo error √(B/m) (White, Royston & Wood 2011). A specification with no blank imputes nothing:
  B = 0, and it is the single fit.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core import decisions as d
from turbotab.core import mattered as M
from turbotab.core.decisions import ExposureFormSpec
from turbotab.core.tests import test_mattered as T
from turbotab.core.tests.acceptance import estimand_fixtures as ef
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r

FILL = d.MissingSpec(strategy="impute", acknowledged=True, reason="a test of the curve's fill")
SPLINE = {"age": ExposureFormSpec(form="spline", knots=4, scale="raw")}


def r_formula(covariates: list[str]) -> str:
    terms = ["fiber", *("rcs(age, 4)" if c == "age" else c for c in covariates)]
    return "glucose ~ " + " + ".join(terms)


@needs_r
def test_under_a_fill_and_a_spline_each_specification_is_r_s_lm_with_rcs_and_hc3(tmp_path):
    """Truth (a): every specification is fitted as its plan, the fill and the spline repeated on
    its own rows, and each point is R's to 1e-8. On the base the curve was refused whole."""
    run = T.Run(T.cohort(), tmp_path / "run", T.state(missing=FILL, exposure_forms=SPLINE))
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    curve = art["curve"]
    got = {s["key"]: s for s in curve["specs"]}
    want = T.departures()
    assert set(got) == set(want) and curve["not_estimated"] == []
    assert curve["through"] == ["impute", "form"]
    frames: dict[str, pd.DataFrame] = {}
    formulas: list[str] = []
    order = list(want)
    for i, key in enumerate(order):
        rules, covariates = want[key]
        part = run.table[T.rows_of(run.table, rules=rules, columns=["glucose"])].copy()
        for c in ["fiber", *covariates]:  # the fill, by hand: the median on these rows
            if c != "sex" and part[c].isna().any():
                part[c] = part[c].fillna(float(np.nanmedian(part[c].to_numpy(dtype=float))))
        frames[f"s{i}"] = part[["glucose", "fiber", *covariates]]
        formulas.append(r_formula(covariates))
    specs = ", ".join(f'"{f}"' for f in formulas)
    script = f"""
suppressPackageStartupMessages({{library(rms); library(sandwich)}})
formulas <- c({specs})
res <- list()
for (i in seq_along(formulas)) {{
  d <- read.csv(get(paste0("s", i - 1, "_csv")))
  if ("sex" %in% names(d)) d$sex <- factor(d$sex, levels = c("female", "male"))
  fit <- lm(as.formula(formulas[i]), data = d)
  se <- sqrt(vcovHC(fit, type = "HC3")["fiber", "fiber"])
  q <- qt(0.975, df.residual(fit))
  b <- unname(coef(fit)["fiber"])
  res[[i]] <- list(estimate = b, low = b - q * se, high = b + q * se, se = se,
                   df = df.residual(fit), n = nrow(d))
}}
out(res)
"""
    r = run_r(script, frames, tmp_path / "r")
    for key, ref in zip(order, r):
        s = got[key]
        assert s["n"] == ref["n"] and s["df"] == pytest.approx(ref["df"], abs=0), key
        for ours, theirs in (("estimate", "estimate"), ("low", "low"), ("high", "high"),
                             ("se", "se")):
            assert s[ours] == pytest.approx(ref[theirs], rel=1e-8, abs=1e-12), (key, ours)
        assert s["fmi"] is None and s["mc_se"] is None  # a single fill pools nothing
    primary = next(s for s in curve["specs"] if s["primary"])
    reported = run.reported()
    assert primary["estimate"] == pytest.approx(reported["estimate"], rel=1e-12)
    assert "the fill of missing values and a declared form other than a straight line" \
        in art["methods"]
    assert "Rubin" not in art["methods"]


class Spy:
    """Each specification ``sensitivity.fit_on_rows`` fits for the curve: its rows, its inputs and
    the copies its missing-values answer drew."""

    def __init__(self, monkeypatch: Any):
        from turbotab.core.stages import sensitivity

        self.calls: list[dict[str, Any]] = []
        real = sensitivity.fit_on_rows

        def spy(state, family, pipeline, frame, inputs, y, task, unit_columns, **kw):
            missing = kw.get("missing")
            imputations = getattr(missing, "imputations", None)
            self.calls.append({"rows": frame.index.to_numpy(), "inputs": list(inputs),
                               "copies": None if imputations is None else
                               [f.copy() for f in imputations.frames], "y": np.asarray(y)})
            return real(state, family, pipeline, frame, inputs, y, task, unit_columns, **kw)

        monkeypatch.setattr(sensitivity, "fit_on_rows", spy)


@needs_r
def test_under_multiple_imputation_each_specification_pools_its_own_copies_as_mice_does(
        tmp_path, monkeypatch):
    """Truth (b): each specification with a blank on its rows draws its own copies and is pooled
    by Rubin's rules; fitted here copy by copy by statsmodels and pooled by ``mice::pool.scalar``
    with ``dfcom = n − p``, its estimate, interval, degrees of freedom, fraction of missing
    information and Monte Carlo error match. The specification that drops `activity` reads no
    blank: nothing is imputed (B = 0) and it is statsmodels' single fit, with no fraction of missing
    information. On the base the curve was refused whole."""
    spy = Spy(monkeypatch)
    run = T.Run(T.cohort(), tmp_path / "run",
                T.state(missing=d.MissingSpec(strategy="multiple_imputation")))
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    curve = art["curve"]
    got = {s["key"]: s for s in curve["specs"]}
    want = T.departures()
    assert set(got) == set(want) and curve["not_estimated"] == []
    positions = {key: np.flatnonzero(T.rows_of(run.table, rules=rules, columns=["glucose"]))
                 for key, (rules, _) in want.items()}
    pooled: list[dict[str, Any]] = []
    single: list[str] = []
    for key, (rules, covariates) in want.items():
        call = next((c for c in spy.calls if set(c["inputs"]) == {"fiber", *covariates}
                     and np.array_equal(np.sort(c["rows"]), positions[key])), None)
        if not run.table.loc[positions[key], ["fiber", *covariates]].isna().to_numpy().any():
            single.append(key)
            continue
        assert call is not None and call["copies"], key
        y = run.table.loc[positions[key], "glucose"].to_numpy(dtype=float)
        for j, X_k in enumerate(call["copies"]):
            X = ef.design_matrix(X_k.loc[run.table.index[positions[key]]], ["fiber", *covariates])
            fit = sm.OLS(y, X).fit(cov_type="HC3", use_t=True)
            pooled.append({"spec": key, "copy": j, "q": float(fit.params["fiber"]),
                           "u": float(fit.bse["fiber"]) ** 2, "n": len(y), "k": X.shape[1]})
    assert single == [k for k in want if "adjusted:activity=dropped" in k]
    script = """
suppressPackageStartupMessages(library(mice))
d <- read.csv(copies_csv)
res <- list()
for (s in unique(d$spec)) {
  e <- d[d$spec == s, ]
  p <- pool.scalar(e$q, e$u, n = e$n[1], k = e$k[1])
  q <- qt(0.975, p$df)
  res[[s]] <- list(estimate = p$qbar, low = p$qbar - q * sqrt(p$t), high = p$qbar + q * sqrt(p$t),
                   se = sqrt(p$t), df = p$df, fmi = p$fmi, mc_se = sqrt(p$b / p$m), m = p$m)
}
out(res)
"""
    r = run_r(script, {"copies": pd.DataFrame(pooled)}, tmp_path / "r")
    assert set(r) == set(want) - set(single)
    for key, ref in r.items():
        s = got[key]
        assert s["n"] == len(positions[key]), key
        for field in ("estimate", "low", "high", "se", "df", "fmi", "mc_se"):
            assert s[field] == pytest.approx(ref[field], rel=1e-8, abs=1e-12), (key, field)
    for key in single:  # no blank: the single fit, B = 0
        rules, covariates = want[key]
        est, low, high, n = T.statsmodels_fit(run.table, outcome="glucose", rules=rules,
                                              covariates=covariates)
        s = got[key]
        assert s["estimate"] == pytest.approx(est, rel=1e-9) and s["n"] == n
        assert s["low"] == pytest.approx(low, rel=1e-9) and s["high"] == pytest.approx(high,
                                                                                    rel=1e-9)
        assert s["fmi"] is None and s["mc_se"] is None
    primary = next(s for s in curve["specs"] if s["primary"])
    assert primary["estimate"] == pytest.approx(run.reported()["estimate"], rel=1e-12)
    assert primary["fmi"] == pytest.approx(run.reported()["fmi"], rel=1e-12)
    assert curve["through"] == ["impute"]
    assert "(multiple imputation)" in art["methods"] and "Rubin's rules" in art["methods"]
    assert math.isfinite(primary["df"]) and primary["df"] < primary["n"]


POOL_LOGIT = """
suppressPackageStartupMessages(library(mice))
d <- read.csv(copies_csv)
res <- list()
for (s in unique(d$spec)) {
  e <- d[d$spec == s, ]
  p <- pool.scalar(e$q, e$u)  # n = Inf: the Wald fit has no finite complete-data df
  q <- qt(0.975, p$df)
  res[[s]] <- list(estimate = p$qbar, low = p$qbar - q * sqrt(p$t), high = p$qbar + q * sqrt(p$t),
                   df = p$df, fmi = p$fmi, mc_se = sqrt(p$b / p$m))
}
out(res)
"""


@needs_r
def test_a_yes_no_outcome_s_specifications_pool_their_own_logistic_copies_as_mice_does(
        tmp_path, monkeypatch):
    """Truth (b) for a yes/no outcome: each specification with a blank draws its own copies; each
    copy fitted here by statsmodels' logistic regression (Wald, the log odds of `dm` being `yes`)
    and pooled by ``mice::pool.scalar`` with an infinite complete-data df (Rubin's ν), as the
    engine pools a Wald table. Its estimate, interval, df, fraction of missing information and
    Monte Carlo error match to 1e-6 (two logistic solvers, each converged to its own tolerance)."""
    spy = Spy(monkeypatch)
    run = T.Run(T.cohort(), tmp_path / "run",
                T.state(target="dm", task="binary", measure="odds_ratio", event="yes",
                        missing=d.MissingSpec(strategy="multiple_imputation")))
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    got = {s["key"]: s for s in art["curve"]["specs"]}
    want = T.departures()
    assert set(got) == set(want)
    pooled: list[dict[str, Any]] = []
    for key, (rules, covariates) in want.items():
        rows = np.flatnonzero(T.rows_of(run.table, rules=rules, columns=["dm"]))
        if not run.table.loc[rows, ["fiber", *covariates]].isna().to_numpy().any():
            continue
        call = next(c for c in spy.calls if set(c["inputs"]) == {"fiber", *covariates}
                    and np.array_equal(np.sort(c["rows"]), rows))
        y = (run.table.loc[rows, "dm"] == "yes").to_numpy(dtype=float)
        for j, X_k in enumerate(call["copies"]):
            X = ef.design_matrix(X_k.loc[run.table.index[rows]], ["fiber", *covariates])
            fit = sm.Logit(y, X).fit(disp=0, tol=1e-12, maxiter=200)
            pooled.append({"spec": key, "copy": j, "q": float(fit.params["fiber"]),
                           "u": float(fit.bse["fiber"]) ** 2})
    r = run_r(POOL_LOGIT, {"copies": pd.DataFrame(pooled)}, tmp_path / "r")
    assert len(r) == len(want) - 1  # the specification that drops `activity` imputes nothing
    for key, ref in r.items():
        for field in ("estimate", "low", "high", "df", "fmi", "mc_se"):
            assert got[key][field] == pytest.approx(ref[field], rel=1e-6), (key, field)
    assert "log of the reported odds ratio" in art["says"]
