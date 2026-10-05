"""FORM · 2 · a restricted cubic spline with k by a declared rule, knots at Harrell's percentiles,
its basis and its two Wald tests against R ``rms``, and D1 under multiple imputation
(MODELING_SEQUENCE §1 row 5).

The independent references:

* **The rule** is Harrell's text (RMS §2.4.6, quoted in ``methods/exposure_form.py``) on the
  effective sample size of RMS §4.4 (n; the smaller of the events and non-events; the events of a
  time to event; n − Σnᵢ³/n² for an ordinal outcome), counted here by hand.
* **Knots and basis**: R's ``Hmisc::rcspline.eval`` (the function ``rms::rcs`` calls; it lives in
  Hmisc and is not exported by rms) with ``nk = k, knots.only = TRUE`` for the knots, and the design
  matrix ``rms::ols`` builds from ``rcs(x, knots)``, to 1e-10.
* **The tests**: ``rms::anova`` on ``rms::lrm`` (a yes/no outcome, the model's information) and on
  ``rms::ols`` with its covariance replaced by HC3 written out in R (``f$var``; ``anova.rms`` reads
  ``vcov(f)``), the "x" row (the test of association) and its " Nonlinear" row, to 1e-6.
* **Under multiple imputation**, the engine's own completed copies refit in R by ``lm`` on the basis
  ``rcspline.eval`` makes at the knots the engine fixed, HC3 written out, pooled by
  ``mitml::testConstraints(method = "D1")``, to 1e-6.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.methods import exposure_form as ef
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r

ROLES = {"pid": "identifier", "x": "exposure", "z": "covariate"}


# ── k by the declared rule ───────────────────────────────────────────────────


def test_2_k_is_harrells_rule_on_the_effective_sample_size():
    assert [ef.knots_by_rule(n) for n in (5, 29, 29.9, 30, 64, 99, 99.9, 100, 5000, None)] == \
        [3, 3, 3, 4, 4, 4, 4, 5, 5, 4]
    rng = np.random.default_rng(1)
    y = rng.normal(size=137)
    assert ef.effective_n("regression", y) == 137
    yes = np.where(rng.random(400) < 0.12, "yes", "no")
    events = int((yes == "yes").sum())
    assert ef.effective_n("binary", yes, "yes") == min(events, 400 - events)
    assert ef.effective_n("time_to_event", yes, "yes") == events
    levels = rng.choice(["low", "mid", "high"], 300, p=[0.2, 0.5, 0.3])
    counts = np.array([np.sum(levels == v) for v in ("low", "mid", "high")], dtype=float)
    assert ef.effective_n("ordinal", levels) == pytest.approx(300 - (counts ** 3).sum() / 300 ** 2,
                                                              rel=1e-12)


@pytest.mark.parametrize("n_eff, k", [(25.0, 3), (64.0, 4), (250.0, 5)])
def test_2_a_spline_declared_without_k_records_the_rule_and_its_sentence_states_it(n_eff, k):
    st = mf.state(roles=dict(ROLES), role_confirmations=dict(ROLES), target="y",
                  task="regression", purpose="inference",
                  estimand=d.EstimandSpec(exposure="x", measure="mean_difference"))
    ctx = {"state": st, "artifact": lambda s: {"n_effective": n_eff} if s == "forms" else None}
    done = d.validate({"kind": "set_exposure_form", "column": "x", "form": "spline"}, ctx)
    assert (done.knots, done.knots_rule, done.n_effective) == (k, "harrell", n_eff)
    assert (done.scale, done.unit) == ("raw", "unit of `x`")
    from turbotab.core.voice import sentence_for

    pct = {3: "10, 50 and 90", 4: "5, 35, 65 and 95", 5: "5, 27.5, 50, 72.5 and 95"}[k]
    assert sentence_for(done, st) == (
        f"`x` entered the models as a restricted cubic spline with `{k}` knots, k = {k} by "
        f"Harrell's rule (3 knots below an effective sample size of 30, 5 from 100, else 4; here "
        f"{n_eff:,.0f}, the number of analyzed rows), at the {pct} percentiles of its values in "
        f"the rows each model was fit on (Harrell's placement); the test of association is the "
        f"Wald test that every term is zero, and nonlinearity was tested by a Wald test that its "
        f"nonlinear terms are zero, a non-significant result never refitting a straight line; "
        f"quintiles were reported beside it, their boundaries and reference stated, with the p "
        f"for linear trend (customary) across quintile medians.")


# ── knots and basis against R ────────────────────────────────────────────────


def _samples() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(22)
    zeros = np.where(rng.random(400) < 0.2, 0.0, rng.gamma(2.0, 3.0, 400))  # a tied lowest value
    return {"uniform": rng.uniform(0, 50, 500), "skewed": rng.lognormal(3, 0.6, 700),
            "small": rng.normal(20, 4, 60), "tied": zeros}


@needs_r
def test_2_knots_and_basis_agree_with_r_rcspline_eval_and_rms_to_1e_10(tmp_path):
    """Every sample, k = 3, 4 and 5: the knots ``rcspline.eval`` places, and the basis it makes
    (and the one ``rms::ols`` builds from ``rcs``), against the engine's to 1e-10."""
    samples = _samples()
    frame = pd.concat([pd.DataFrame({"sample": name, "x": x}) for name, x in samples.items()],
                      ignore_index=True)
    found = run_r("""
suppressMessages({library(rms); library(Hmisc)})
d <- read.csv(frame_csv)
res <- list()
for (s in unique(d$sample)) for (k in 3:5) {
  x <- d$x[d$sample == s]
  kn <- Hmisc::rcspline.eval(x, nk = k, knots.only = TRUE)
  b <- Hmisc::rcspline.eval(x, knots = kn, inclx = TRUE)
  y <- x + rnorm(length(x))
  f <- ols(y ~ rcs(x, kn), x = TRUE)
  res[[paste(s, k)]] <- list(knots = kn, basis = unname(as.matrix(b)),
                             design = unname(as.matrix(f$x)))
}
out(res)
""", {"frame": frame}, tmp_path)
    for name, x in samples.items():
        for k in (3, 4, 5):
            ref = found[f"{name} {k}"]
            knots, _ = ef.rcs_knots(x, k)
            assert np.allclose(knots, ref["knots"], rtol=0, atol=1e-10), (name, k)
            step = ef.ExposureForms({"x": {"form": "spline", "knots": k}}).fit(pd.DataFrame({"x": x}))
            ours = step.transform(pd.DataFrame({"x": x})).to_numpy(dtype=float)
            assert ours.shape == np.asarray(ref["basis"]).shape
            assert np.allclose(ours, ref["basis"], rtol=0, atol=1e-10), (name, k)
            assert np.allclose(ours, ref["design"], rtol=0, atol=1e-10), (name, k)


# ── the two Wald tests against rms::anova ────────────────────────────────────


def _cohort(n: int = 700, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = rng.gamma(3.0, 2.0, n) + 0.4 * z
    eta = -0.5 + 0.35 * np.log(x + 1) + 0.5 * z
    return pd.DataFrame({"pid": np.arange(n), "x": x.round(6), "z": z.round(6),
                         "y": (eta + rng.normal(0, 1, n)).round(6),
                         "b": (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)})


def _fit(frame: pd.DataFrame, folder: Path, *, target: str, task: str, k: int,
         missing: Any = "complete_case", event: str | None = None) -> Any:
    from turbotab.core.stages.modeling import design_stage, fit_stage

    roles = dict(ROLES)
    st = mf.state(roles=roles, role_confirmations=dict(roles), target=target, task=task,
                  event=event, purpose="inference", models=["linear"], lens=["clinical"],
                  missing=missing, split=d.SplitSpec(holdout=0.0, seed=0, folds=5),
                  shape_confirmations={"code_or_count:b": "amount"},
                  exposure_forms={"x": d.ExposureFormSpec(form="spline", knots=k)})
    paths = mf.ingest_frame(frame.drop(columns=[c for c in ("y", "b") if c != target]), folder)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
    info = mf.target_info(task, target)
    design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
    return fit_stage(mf.context(st, {"design": design, "split": split, "target_info": info},
                                paths))


ANOVA = """
suppressMessages({library(rms); library(Hmisc)})
d <- read.csv(frame_csv)
kn <- Hmisc::rcspline.eval(d$x, nk = %(k)d, knots.only = TRUE)
dd <- datadist(d); options(datadist = "dd")
%(fit)s
a <- anova(f)
out(list(knots = kn, overall = unname(a["x", ]), nonlinear = unname(a[" Nonlinear", ]),
         columns = colnames(a)))
"""


@needs_r
@pytest.mark.parametrize("k", [3, 4, 5])
def test_2_a_yes_no_outcomes_overall_and_nonlinearity_tests_agree_with_rms_anova(k, tmp_path):
    frame = _cohort()
    fit = _fit(frame, tmp_path / "e", target="b", task="binary", k=k, event="1")
    tests = {t["test"]: t for t in fit.data["models"][0]["exposure_tests"] if t["column"] == "x"}
    found = run_r(ANOVA % {"k": k, "fit": "f <- lrm(b ~ rcs(x, kn) + z, data = d, tol = 1e-12)"},
                  {"frame": frame}, tmp_path / "r")
    assert found["columns"] == ["Chi-Square", "d.f.", "P"]
    assert np.allclose(tests["overall"]["knots"], found["knots"], rtol=0, atol=1e-10)
    for kind in ("overall", "nonlinear"):
        chi2, df, p = found[kind]
        ours = tests[kind]
        assert ours["distribution"] == "chi2" and ours["df_num"] == int(df)
        assert ours["statistic"] == pytest.approx(chi2, rel=1e-6)
        assert ours["p"] == pytest.approx(p, rel=1e-6, abs=1e-12)
    assert tests["overall"]["caption"].startswith("The test of association: Wald test that all ")


@needs_r
@pytest.mark.parametrize("k", [3, 5])
def test_2_a_numeric_outcomes_tests_agree_with_rms_anova_on_hc3(k, tmp_path):
    """``ols``'s covariance replaced by HC3 written out in R, so ``anova.rms`` tests on the same
    covariance the engine's table uses; its F rows (F, d.f., P) against the engine's."""
    frame = _cohort(seed=6)
    fit = _fit(frame, tmp_path / "e", target="y", task="regression", k=k)
    tests = {t["test"]: t for t in fit.data["models"][0]["exposure_tests"] if t["column"] == "x"}
    hc3 = """
f <- ols(y ~ rcs(x, kn) + z, data = d, x = TRUE, y = TRUE)
X <- cbind(1, f$x); e <- f$y - X %*% coef(f); B <- solve(crossprod(X))
h <- rowSums((X %*% B) * X)
f$var <- B %*% crossprod(X * as.vector(e / (1 - h))) %*% B
dimnames(f$var) <- list(names(coef(f)), names(coef(f)))
"""
    found = run_r(ANOVA % {"k": k, "fit": hc3}, {"frame": frame}, tmp_path / "r")
    assert found["columns"][-2:] == ["F", "P"]
    for kind in ("overall", "nonlinear"):
        df, _, _, F, p = found[kind]
        ours = tests[kind]
        assert ours["distribution"] == "F" and ours["df_num"] == int(df)
        assert ours["df_den"] == len(frame) - (k - 1) - 2
        assert ours["statistic"] == pytest.approx(F, rel=1e-6)
        assert ours["p"] == pytest.approx(p, rel=1e-6, abs=1e-12)


# ── under multiple imputation: D1 ────────────────────────────────────────────


@needs_r
def test_2_under_multiple_imputation_the_tests_are_pooled_by_d1_as_mitml_pools_them(tmp_path):
    frame = _cohort(n=500, seed=8)
    rng = np.random.default_rng(80)
    frame.loc[rng.random(len(frame)) < 0.25, "z"] = np.nan
    frame.loc[rng.random(len(frame)) < 0.15, "x"] = np.nan
    fit = _fit(frame, tmp_path / "e", target="y", task="regression", k=4,
               missing=d.MissingSpec(strategy="multiple_imputation"))
    tests = {t["test"]: t for t in fit.data["models"][0]["exposure_tests"] if t["column"] == "x"}
    imputations = fit.objects["imputations"]
    knots = imputations["plan"]["forms"]["x"]["knots"]
    copies = pd.concat([f[["x", "z"]].assign(y=frame["y"].to_numpy(), copy=k + 1)
                        for k, f in enumerate(imputations["frames"])], ignore_index=True)
    found = run_r(f"""
suppressMessages({{library(mitml); library(Hmisc)}})
d <- read.csv(copies_csv)
Q <- NULL; U <- list()
for (k in sort(unique(d$copy))) {{
  dk <- d[d$copy == k, ]
  b <- rcspline.eval(dk$x, knots = c({", ".join(repr(float(v)) for v in knots)}), inclx = TRUE)
  dk$s0 <- b[, 1]; dk$s1 <- b[, 2]; dk$s2 <- b[, 3]
  fit <- lm(y ~ s0 + s1 + s2 + z, data = dk)
  X <- model.matrix(fit); e <- resid(fit); h <- hatvalues(fit); B <- solve(crossprod(X))
  Q <- cbind(Q, coef(fit)); U[[k]] <- B %*% crossprod(X * (e / (1 - h))) %*% B
}}
rownames(Q) <- gsub("[()]", "", rownames(Q))
U <- array(unlist(U), c(nrow(Q), nrow(Q), ncol(Q)), dimnames = list(rownames(Q), rownames(Q), NULL))
res <- lapply(list(overall = c("s0", "s1", "s2"), nonlinear = c("s1", "s2")), function(cs) {{
  t <- testConstraints(qhat = Q, uhat = U, constraints = cs, method = "D1")$test
  c(t[1, "F.value"], t[1, "df1"], t[1, "df2"], t[1, "P(>F)"])
}})
out(res)
""", {"copies": copies}, tmp_path / "r")
    for kind in ("overall", "nonlinear"):
        F, df1, df2, p = found[kind]
        ours = tests[kind]
        assert ours["df_num"] == int(df1)
        assert ours["statistic"] == pytest.approx(F, rel=1e-6)
        assert ours["df_den"] == pytest.approx(df2, rel=1e-6)
        assert ours["p"] == pytest.approx(p, rel=1e-6, abs=1e-12)
        assert ours["caption"].startswith("Pooled over") and "D1" in ours["caption"]
