"""The causal lane: DoubleML, TMLE and the post-double-selection lasso on the shortest leash
(V2 definition of done §2, "causal inference"; MODELING_SEQUENCE §0 ruling 1 rung (d), rulings 6
and 10; BLUEPRINT §11.3 and §13).

The package's acceptance items, in its order:

1. **DoubleML.** The partially linear model (a numeric exposure) and the interactive model (a
   yes/no exposure; the ATE and the ATT) with cross-fitting: with deterministic learners and the
   same folds the estimates and standard errors agree with R's DoubleML 1.0.2 to 1e-6; with random
   learners (the cross-validated lasso, here and as R's ``cv_glmnet``) they agree within Monte Carlo
   tolerance over repeated sample splits, each side aggregated by DoubleML's median. The partially
   linear model's required sensitivity analysis is the robustness value of its final
   least-squares step, against sensemakr on R's own fits, on a numeric and a yes/no outcome.
2. **TMLE** for a yes/no exposure and a numeric or yes/no outcome agrees with R's tmle 2.1.1 given
   the same glm Q and g models (``Qform``, ``gform``, ``cvQinit = FALSE``) to 1e-6: the average
   effect and its influence-curve standard error, and for a yes/no outcome the marginal risk and
   odds ratios. The interactive model's risk ratio (average effect and effect among the exposed)
   and its E-value agree with R (glm, survey's delta method, EValue, DoubleML); a level with no
   events has no ratio and no E-value, said with the reason.
3. **Post-double selection** agrees with hdm 0.3.2's algorithm transcribed by hand in NumPy at the
   same plug-in penalty to 1e-8 (and with hdm itself where R has it), selects the same columns, is
   ranked first, and asked, when the candidates are many relative to n, and there its interval
   covers a simulation's truth (1,000 data sets at 100 rows with 120 and 300 candidates).
4. **The assumptions before any estimate**, with their diagnostics, and a positivity violation
   block and record with the trimming exit; the trimmed estimate agrees with R's tmle on the
   overlap population.
5. **A survey design**: the weights enter every nuisance fit and the estimating equation, against
   R's survey package (``svyglm`` and ``svymean`` over the same cross-fitted nuisance models, and
   tmle's ``obsWeights``), and recover a population effect the unweighted estimate misses;
   clusters are the same computation with the cluster as the PSU.
6. **The shortest leash and the chain**: never under prediction; only after the exposure, its
   effect and the adjustment set; every §2 relation of the lane's contracts fires; sensitivity to
   unmeasured confounding is required (the marked call site); the methods sentences are asserted
   verbatim, and they say the block-and-record choices they rest on (a positivity violation kept
   on the record, a sample-only estimate under the population answer), not the Record alone.
7. **A survey design and multiple imputation in the chain** (block and record, with their exits).
8. **A yes/no outcome in the chain**: each estimator's required sensitivity, and no ratio where a
   level has no events.
9. **Each estimand reads its own positivity** (the effect among the exposed is not refused for
   rows near 0), a numeric exposure kept on the record says so, and too few clusters withhold the
   lane with exits that lead forward.

Every expected value comes from outside the engine: R (DoubleML, tmle, survey, sensemakr, EValue,
hdm) run as a subprocess on a CSV written here, a NumPy coordinate-descent lasso written here,
statsmodels, and the simulations' own truths. The R tests skip cleanly where ``Rscript`` is not
installed (and the hdm comparison where hdm is not).
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from turbotab.core import causal
from turbotab.core.models import causal as est

RSCRIPT = shutil.which("Rscript")
needs_r = pytest.mark.skipif(RSCRIPT is None, reason="R (Rscript) is not installed")
ALL4 = list(causal.ASSUMPTIONS)
R_HEAD = """suppressMessages({library(data.table); library(jsonlite)})
options(warn = -1)
"""


def run_r(tmp_path: Path, script: str, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Write ``frames`` as CSVs next to ``script``, run it with Rscript, and read the JSON it
    prints. The paths are passed as ``<name>_csv`` variables."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    lines = [R_HEAD]
    for name, frame in frames.items():
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        lines.append(f'{name}_csv <- "{path}"')
    (tmp_path / "ref.R").write_text("\n".join(lines) + "\n" + script, encoding="utf-8")
    done = subprocess.run([RSCRIPT, str(tmp_path / "ref.R")], capture_output=True, text=True,
                          timeout=900)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def folds_frame(splits: list[list[tuple[np.ndarray, np.ndarray]]], n: int) -> pd.DataFrame:
    """Each repetition's test fold of every row (1-based), as R reads it."""
    out = np.zeros((n, len(splits)), dtype=int)
    for r, rep in enumerate(splits):
        for k, (_, test) in enumerate(rep):
            out[test, r] = k + 1
    return pd.DataFrame(out, columns=[f"r{r + 1}" for r in range(len(splits))])


R_SPLITS = """folds <- fread(folds_csv)
K <- max(folds[[1]])
smpls <- lapply(seq_len(ncol(folds)), function(r) { f <- folds[[r]]
  list(train_ids = lapply(1:K, function(k) which(f != k)),
       test_ids = lapply(1:K, function(k) which(f == k))) })
"""


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulated(n: int, p: int, seed: int) -> dict[str, np.ndarray]:
    """A numeric exposure with a constant effect of 0.8, and a yes/no exposure with a constant
    effect of 1.5 on a numeric outcome and of 0.7 on the log odds of a yes/no one; the outcome
    curves in a covariate, so a linear outcome model is wrong and the nuisance learners matter."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    d = 0.6 * X[:, 0] - 0.4 * X[:, 1] + rng.normal(size=n)
    y = 0.8 * d + X[:, 0] + 0.5 * X[:, 2] ** 2 + rng.normal(size=n)
    a = (rng.uniform(size=n) < _expit(0.5 * X[:, 0] - 0.7 * X[:, 1])).astype(float)
    ya = 1.5 * a + X[:, 0] + 0.5 * X[:, 2] + rng.normal(size=n)
    yb = (rng.uniform(size=n) < _expit(-0.3 + 0.7 * a + 0.5 * X[:, 0])).astype(float)
    return {"X": X, "d": d, "y": y, "a": a, "ya": ya, "yb": yb}


def frame_of(data: dict[str, np.ndarray]) -> pd.DataFrame:
    X = data["X"]
    out = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    for k in ("d", "y", "a", "ya", "yb"):
        out[k] = data[k]
    return out


# ── 1 · DoubleML ─────────────────────────────────────────────────────────────


@needs_r
def test_1a_dml_agrees_with_r_doubleml_with_deterministic_learners_and_the_same_folds(tmp_path):
    """Reference: R DoubleML 1.0.2 (``DoubleMLPLR``, ``DoubleMLIRM`` with ``score = "ATE"`` and
    ``"ATTE"``) with mlr3's ``regr.lm`` and ``classif.log_reg``, given the app's three repetitions
    of five folds through ``set_sample_splitting``, ``dml2``, the propensity truncated at the same
    level. Tolerance 1e-6 on every estimate and standard error, and on each split's estimate."""
    n = 600
    data = simulated(n, 4, seed=7)
    splits = est.sample_splits(n, 5, 3, seed=11)
    bound = est.tmle_gbound(n)
    ref = run_r(tmp_path, R_SPLITS + f"""
suppressMessages({{library(DoubleML); library(mlr3); library(mlr3learners)}})
lgr::get_logger("mlr3")$set_threshold("warn")
df <- fread(data_csv); xs <- paste0("x", 0:3)
fit <- function(obj) {{ obj$set_sample_splitting(smpls); obj$fit(); obj }}
plr <- fit(DoubleMLPLR$new(double_ml_data_from_data_frame(df[, c(xs, "y", "d"), with = FALSE],
  y_col = "y", d_cols = "d", x_cols = xs), ml_l = lrn("regr.lm"), ml_m = lrn("regr.lm"),
  draw_sample_splitting = FALSE))
irm_data <- double_ml_data_from_data_frame(df[, c(xs, "ya", "a"), with = FALSE], y_col = "ya",
  d_cols = "a", x_cols = xs)
irm <- lapply(c("ATE", "ATTE"), function(sc) fit(DoubleMLIRM$new(irm_data, ml_g = lrn("regr.lm"),
  ml_m = lrn("classif.log_reg"), score = sc, trimming_threshold = {bound!r},
  draw_sample_splitting = FALSE)))
bin <- fit(DoubleMLIRM$new(double_ml_data_from_data_frame(df[, c(xs, "yb", "a"), with = FALSE],
  y_col = "yb", d_cols = "a", x_cols = xs), ml_g = lrn("classif.log_reg"),
  ml_m = lrn("classif.log_reg"), score = "ATE", trimming_threshold = {bound!r},
  draw_sample_splitting = FALSE))
out <- function(o) list(coef = unname(o$coef), se = unname(o$se), all = as.numeric(o$all_coef),
  all_se = as.numeric(o$all_se))
cat(toJSON(list(plr = out(plr), ate = out(irm[[1]]), att = out(irm[[2]]), bin = out(bin)),
  digits = NA, auto_unbox = TRUE))
""", {"data": frame_of(data), "folds": folds_frame(splits, n)})
    X = data["X"]
    found = {
        "plr": est.dml_plr(data["y"], data["d"], X, learner="linear", splits=splits),
        "ate": est.dml_irm(data["ya"], data["a"], X, learner="linear", splits=splits, bound=bound),
        "att": est.dml_irm(data["ya"], data["a"], X, learner="linear", splits=splits, bound=bound,
                           score="ATT"),
        "bin": est.dml_irm(data["yb"], data["a"], X, learner="linear", splits=splits, bound=bound,
                           outcome_binary=True),
    }
    for key, e in found.items():
        r = ref[key]
        assert e.estimate == pytest.approx(r["coef"], abs=1e-6), key
        assert e.se == pytest.approx(r["se"], abs=1e-6), key
        assert [s["estimate"] for s in e.repetitions] == pytest.approx(r["all"], abs=1e-6), key
        assert [s["se"] for s in e.repetitions] == pytest.approx(r["all_se"], abs=1e-6), key
    # and each covers its simulation's truth
    assert abs(found["plr"].estimate - 0.8) < 3 * found["plr"].se
    assert abs(found["ate"].estimate - 1.5) < 3 * found["ate"].se


@needs_r
def test_1b_dml_with_random_learners_agrees_within_monte_carlo_tolerance(tmp_path):
    """Reference: R DoubleML with ``regr.cv_glmnet`` and ``classif.cv_glmnet`` (``lambda.min``),
    its own ten random sample splits, against the app's cross-validated lasso learners on ten
    splits of its own; both aggregated by DoubleML's median. The tolerance is Monte Carlo: four
    standard errors of the difference of two medians of ten split estimates, each
    ``1.2533 sd/√S`` (the median's asymptotic efficiency), for the estimate and for the standard
    error. Both estimates also cover the simulation's truth."""
    n, p, S = 800, 10, 10
    rng = np.random.default_rng(3)
    X = rng.normal(size=(n, p))
    d = 0.6 * X[:, 0] - 0.4 * X[:, 1] + 0.3 * X[:, 2] + rng.normal(size=n)
    y = 0.5 * d + X[:, 0] + 0.5 * X[:, 3] + rng.normal(size=n)
    a = (rng.uniform(size=n) < _expit(0.5 * X[:, 0] - 0.7 * X[:, 1])).astype(float)
    ya = 1.0 * a + X[:, 0] + 0.5 * X[:, 3] + rng.normal(size=n)
    bound = est.tmle_gbound(n)
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    frame["d"], frame["y"], frame["a"], frame["ya"] = d, y, a, ya
    ref = run_r(tmp_path, f"""
suppressMessages({{library(DoubleML); library(mlr3); library(mlr3learners)}})
lgr::get_logger("mlr3")$set_threshold("warn")
set.seed(20261005)
df <- fread(data_csv); xs <- paste0("x", 0:{p - 1})
plr <- DoubleMLPLR$new(double_ml_data_from_data_frame(df[, c(xs, "y", "d"), with = FALSE],
  y_col = "y", d_cols = "d", x_cols = xs), ml_l = lrn("regr.cv_glmnet", s = "lambda.min"),
  ml_m = lrn("regr.cv_glmnet", s = "lambda.min"), n_folds = 5, n_rep = {S})
plr$fit()
irm <- DoubleMLIRM$new(double_ml_data_from_data_frame(df[, c(xs, "ya", "a"), with = FALSE],
  y_col = "ya", d_cols = "a", x_cols = xs), ml_g = lrn("regr.cv_glmnet", s = "lambda.min"),
  ml_m = lrn("classif.cv_glmnet", s = "lambda.min"), n_folds = 5, n_rep = {S},
  trimming_threshold = {bound!r})
irm$fit()
out <- function(o) list(coef = unname(o$coef), se = unname(o$se), all = as.numeric(o$all_coef),
  all_se = as.numeric(o$all_se))
cat(toJSON(list(plr = out(plr), irm = out(irm)), digits = NA, auto_unbox = TRUE))
""", {"data": frame})
    splits = est.sample_splits(n, 5, S, seed=5)
    found = {"plr": est.dml_plr(y, d, X, learner="lasso", splits=splits, seed=5),
             "irm": est.dml_irm(ya, a, X, learner="lasso", splits=splits, bound=bound, seed=5)}
    truth = {"plr": 0.5, "irm": 1.0}

    def mcse(values: list[float]) -> float:
        v = np.asarray(values, dtype=float)
        return 1.2533 * float(v.std(ddof=1)) / math.sqrt(len(v))

    for key, e in found.items():
        r = ref[key]
        mine = [s["estimate"] for s in e.repetitions]
        mine_se = [s["se"] for s in e.repetitions]
        tol = 4 * math.hypot(mcse(mine), mcse(r["all"]))
        assert abs(e.estimate - r["coef"]) <= tol, (key, e.estimate, r["coef"], tol)
        tol_se = 4 * math.hypot(mcse(mine_se), mcse(r["all_se"]))
        assert abs(e.se - r["se"]) <= tol_se, (key, e.se, r["se"], tol_se)
        assert abs(e.estimate - truth[key]) < 3 * e.se and abs(r["coef"] - truth[key]) < 3 * r["se"]


@needs_r
def test_1c_the_partially_linear_models_robustness_value_is_its_final_least_squares_steps(tmp_path):
    """Sensitivity to unmeasured confounding, required in the lane (ruling 10), for the partially
    linear model on a numeric and on a yes/no outcome. Its estimate is one least-squares
    coefficient: the outcome's residual regressed on the exposure's, no intercept. Reference: R
    alone, on the app's folds: ``lm`` nuisance fits per fold, ``lm(u ~ 0 + v)`` per split, and
    sensemakr's ``robustness_value`` (q = 1; α = 1 and 0.05) of that fit, the median over the three
    splits; tolerance 1e-8. On the yes/no outcome there is no risk ratio, so the robustness value
    is the analysis computed and the E-value's absence is said; the methods text names it."""
    n = 600
    data = simulated(n, 4, seed=8)
    splits = est.sample_splits(n, 5, 3, seed=4)
    ref = run_r(tmp_path, R_SPLITS + """
suppressMessages(library(sensemakr))
df <- as.data.frame(fread(data_csv)); n <- nrow(df)
one <- function(y, r) { f <- folds[[r]]; u <- v <- rep(NA, n)
  for (k in 1:K) { tr <- f != k; te <- f == k
    u[te] <- df[[y]][te] - predict(lm(as.formula(paste(y, "~ x0 + x1 + x2 + x3")), data = df[tr, ]),
                                   newdata = df[te, ])
    v[te] <- df$d[te] - predict(lm(d ~ x0 + x1 + x2 + x3, data = df[tr, ]), newdata = df[te, ]) }
  fit <- lm(u ~ 0 + v)
  c(robustness_value(fit, "v", q = 1, alpha = 1), robustness_value(fit, "v", q = 1, alpha = 0.05)) }
out <- lapply(c("y", "yb"), function(y) { m <- sapply(seq_len(ncol(folds)), function(r) one(y, r))
  c(median(m[1, ]), median(m[2, ])) })
cat(toJSON(list(numeric = out[[1]], binary = out[[2]]), digits = NA))
""", {"data": frame_of(data), "folds": folds_frame(splits, n)})
    for key, y, binary in (("numeric", data["y"], False), ("binary", data["yb"], True)):
        fit = est.dml_plr(y, data["d"], data["X"], learner="linear", splits=splits,
                          outcome_binary=binary)
        row = {"measure": "risk_difference" if binary else "mean_difference",
               "estimate": fit.estimate, "se": fit.se, "ci_low": fit.ci[0], "ci_high": fit.ci[1]}
        found = causal.sensitivity_for([row], method="dml_plr", exposure="d", outcome="y",
                                       outcome_sd=None if binary else float(np.std(y, ddof=1)),
                                       final_stage=fit.extra["final_stage"])
        robust = found["robustness"]
        assert (robust["rv"], robust["rv_alpha"]) == pytest.approx(ref[key], abs=1e-8), key
        assert found["computed"] is True
        assert found["methods"] == (["robustness_value"] if binary
                                    else ["robustness_value", "e_value"])
        if binary:
            assert found["e_value"] is None and found["not_computed"] == (
                "No E-value: a risk difference with no risk ratio beside it carries no risks to "
                "form one from.")
        assert causal.sensitivity_sentence(found) == (PLR_RV_ONLY if binary
                                                      else PLR_RV_AND_E_VALUE)


# ── 2 · TMLE ─────────────────────────────────────────────────────────────────


@needs_r
def test_2_tmle_agrees_with_r_tmle_given_the_same_glm_models(tmp_path):
    """Reference: R tmle 2.1.1, ``tmle(Y, A, W, Qform = "Y~A+x0+x1+x2+x3", gform =
    "A~x0+x1+x2+x3", cvQinit = FALSE)`` for a numeric outcome (gaussian) and a yes/no one
    (binomial), with its default truncation 5/(√n ln n). Tolerance 1e-6 on the average effect, its
    influence-curve standard error, and the marginal risk and odds ratios with their log-scale
    standard errors."""
    n = 600
    data = simulated(n, 4, seed=7)
    ref = run_r(tmp_path, """
suppressMessages(library(tmle))
df <- as.data.frame(fread(data_csv)); W <- df[, paste0("x", 0:3)]
f <- function(y, family) tmle(Y = y, A = df$a, W = W, Qform = "Y~A+x0+x1+x2+x3",
  gform = "A~x0+x1+x2+x3", cvQinit = FALSE, family = family)$estimates
g <- f(df$ya, "gaussian"); b <- f(df$yb, "binomial")
cat(toJSON(list(gaussian = c(g$ATE$psi, sqrt(g$ATE$var.psi)),
  binomial = c(b$ATE$psi, sqrt(b$ATE$var.psi), b$RR$psi, sqrt(b$RR$var.log.psi), b$OR$psi,
               sqrt(b$OR$var.log.psi))), digits = NA))
""", {"data": frame_of(data)})
    g = est.tmle(data["ya"], data["a"], data["X"], learner="linear", family="gaussian")
    assert [g.estimate, g.se] == pytest.approx(ref["gaussian"], abs=1e-6)
    b = est.tmle(data["yb"], data["a"], data["X"], learner="linear", family="binomial")
    rr, oddsr = b.extra["risk_ratio"], b.extra["odds_ratio"]
    assert [b.estimate, b.se, rr["estimate"], rr["se_log"], oddsr["estimate"],
            oddsr["se_log"]] == pytest.approx(ref["binomial"], abs=1e-6)
    assert abs(g.estimate - 1.5) < 3 * g.se  # the simulation's constant effect


@needs_r
def test_2b_the_interactive_models_risk_ratio_and_its_e_value(tmp_path):
    """The interactive model on a yes/no outcome carries its required sensitivity analysis: the
    E-value of a risk ratio formed from its own doubly robust means, for the average effect
    (μ₁/μ₀ over everyone) and for the effect among the exposed (their observed risk over their
    doubly robust risk unexposed). Reference: R alone, on the app's three splits of five folds:
    ``glm`` nuisance fits per fold (the propensity truncated at the same bound), the doubly robust
    means, the log ratio's standard error by survey's ``svycontrast`` (symbolic delta method) of
    ``svymean(~φ₁ + φ₀)`` over independent rows, rescaled by √((n − 1)/n) to DoubleML's
    ``mean(ψ²)/n`` convention, DoubleML's median aggregation written out, and ``EValue::evalues.RR``
    of the result; tolerance 1e-6 (1e-8 relative on the E-values). R's DoubleML on the same folds
    confirms the means: μ₁ − μ₀ is its ATE and ATTE per split (1e-8). The simulation's true
    marginal risk ratio (its own definition, by a million draws) lies within three standard
    errors."""
    n = 600
    data = simulated(n, 4, seed=7)
    splits = est.sample_splits(n, 5, 3, seed=11)
    bound = est.tmle_gbound(n)
    ref = run_r(tmp_path, R_SPLITS + f"""
suppressMessages({{library(survey); library(EValue); library(DoubleML); library(mlr3);
                   library(mlr3learners)}})
lgr::get_logger("mlr3")$set_threshold("warn")
df <- as.data.frame(fread(data_csv)); n <- nrow(df); b <- {bound!r}
form <- function(y) as.formula(paste(y, "~ x0 + x1 + x2 + x3"))
split <- function(r, score) {{ f <- folds[[r]]; m <- g0 <- g1 <- rep(NA, n)
  for (k in 1:K) {{ tr <- f != k; te <- f == k; d1 <- df[tr, ]
    m[te] <- predict(glm(form("a"), binomial, data = d1), df[te, ], type = "response")
    g0[te] <- predict(glm(form("yb"), binomial, data = d1[d1$a == 0, ]), df[te, ], type = "response")
    g1[te] <- predict(glm(form("yb"), binomial, data = d1[d1$a == 1, ]), df[te, ], type = "response") }}
  m <- pmin(pmax(m, b), 1 - b); a <- df$a; y <- df$yb
  if (score == "ATE") {{ phi1 <- g1 + a * (y - g1) / m; phi0 <- g0 + (1 - a) * (y - g0) / (1 - m) }}
  else {{ p <- ave(a, f); phi1 <- a * y / p; phi0 <- (a * g0 + m * (1 - a) * (y - g0) / (1 - m)) / p }}
  s <- svymean(~phi1 + phi0, svydesign(ids = ~1, data = data.frame(phi1 = phi1, phi0 = phi0)))
  lr <- svycontrast(s, quote(log(phi1 / phi0)))
  j <- if (score == "ATE") rep(1, n) else a / p
  c(coef(lr)[[1]], SE(lr)[[1]] * sqrt((n - 1) / n), (sum(phi1) - sum(phi0)) / sum(j)) }}
agg <- function(t, s) {{ th <- median(t); list(th = th, se = sqrt(median(n * s^2 + (t - th)^2) / n)) }}
dml <- function(score) {{ o <- DoubleMLIRM$new(double_ml_data_from_data_frame(
    df[, c("x0", "x1", "x2", "x3", "yb", "a")], y_col = "yb", d_cols = "a",
    x_cols = c("x0", "x1", "x2", "x3")), ml_g = lrn("classif.log_reg"),
    ml_m = lrn("classif.log_reg"), score = score, trimming_threshold = b,
    draw_sample_splitting = FALSE)
  o$set_sample_splitting(smpls); o$fit(); as.numeric(o$all_coef) }}
out <- lapply(c("ATE", "ATT"), function(sc) {{
  m <- sapply(seq_len(ncol(folds)), function(r) split(r, sc)); g <- agg(m[1, ], m[2, ])
  z <- qnorm(0.975); rr <- exp(g$th); lo <- exp(g$th - z * g$se); hi <- exp(g$th + z * g$se)
  e <- evalues.RR(rr, lo, hi)
  list(rr = rr, se = g$se, lo = lo, hi = hi, e_point = e["E-values", "point"],
       e_limit = if (rr > 1) e["E-values", "lower"] else e["E-values", "upper"],
       diff = m[3, ], dml = dml(if (sc == "ATE") "ATE" else "ATTE")) }})
cat(toJSON(list(ate = out[[1]], att = out[[2]]), digits = NA, auto_unbox = TRUE))
""", {"data": frame_of(data), "folds": folds_frame(splits, n)})
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=1_000_000)
    true_rr = float(np.mean(_expit(-0.3 + 0.7 + 0.5 * x0)) / np.mean(_expit(-0.3 + 0.5 * x0)))
    for key, score, words in (("ate", "ATE", "the marginal risk ratio"),
                              ("att", "ATT", "the risk ratio among the exposed")):
        fit = est.dml_irm(data["yb"], data["a"], data["X"], learner="linear", splits=splits,
                          bound=bound, score=score, outcome_binary=True)
        r = ref[key]
        assert [s["estimate"] for s in fit.repetitions] == pytest.approx(r["dml"], abs=1e-8)
        assert r["diff"] == pytest.approx(r["dml"], abs=1e-8)  # R's means are DoubleML's
        rr = fit.extra["risk_ratio"]
        assert rr["estimate"] == pytest.approx(r["rr"], abs=1e-6)
        assert rr["se_log"] == pytest.approx(r["se"], abs=1e-6)
        assert list(rr["ci"]) == pytest.approx([r["lo"], r["hi"]], abs=1e-6)
        rows = [{"measure": "risk_difference", "estimate": fit.estimate, "se": fit.se},
                {"measure": "risk_ratio", "estimate": rr["estimate"], "ci_low": rr["ci"][0],
                 "ci_high": rr["ci"][1]}]
        found = causal.sensitivity_for(rows, method="dml_irm", exposure="a", outcome="yb",
                                       outcome_sd=None,
                                       population="exposed" if score == "ATT" else "all")
        assert found["computed"] is True and found["methods"] == ["e_value"]
        assert found["e_value"]["point"] == pytest.approx(r["e_point"], rel=1e-8)
        assert found["e_value"]["limit"] == pytest.approx(r["e_limit"], rel=1e-8)
        assert f"E-value for {words}: " in found["reading"]
        assert found["not_computed"] == (
            "No robustness value: it is defined for one least-squares coefficient (Cinelli & "
            "Hazlett 2020, J R Stat Soc B 82:39), and double/debiased machine learning in the "
            "interactive model fits its nuisance models by cross-fitted learners.")
    ate = est.dml_irm(data["yb"], data["a"], data["X"], learner="linear", splits=splits,
                      bound=bound, outcome_binary=True).extra["risk_ratio"]
    assert abs(math.log(ate["estimate"]) - math.log(true_rr)) < 3 * ate["se_log"]


def test_2c_a_level_with_no_events_has_no_ratio_and_no_e_value():
    """A yes/no outcome with no event among the exposed: their risk is zero, so a ratio of the two
    risks is 0 or undefined, and an E-value of it is meaningless (the verifier saw a marginal risk
    ratio of 6.9e-12 and an E-value of 2.9e11). Reference: the definitions. TMLE reports neither
    the risk nor the odds ratio, the interactive model no risk ratio, each with the reason; the
    risk difference stands; the required sensitivity analysis says it could not be computed and
    why. With every exposed row an event, the odds ratio alone is refused."""
    data = simulated(600, 4, seed=7)
    a, X = data["a"], data["X"]
    none = np.where(a == 1, 0.0, data["yb"])
    reason = ("no exposed row has the event, so the exposed risk is zero and a ratio of the two "
              "risks is 0 or undefined")
    tm = est.tmle(none, a, X, learner="linear", family="binomial")
    assert "risk_ratio" not in tm.extra and "odds_ratio" not in tm.extra
    assert tm.extra["ratio_refused"] == reason and tm.extra["odds_refused"] == reason
    assert -1 < tm.estimate < 0 and math.isfinite(tm.se)
    splits = est.sample_splits(600, 5, 1, seed=3)
    irm = est.dml_irm(none, a, X, learner="linear", splits=splits, outcome_binary=True)
    assert "risk_ratio" not in irm.extra and irm.extra["ratio_refused"] == reason
    found = causal.sensitivity_for([{"measure": "risk_difference", "estimate": tm.estimate,
                                     "se": tm.se}], method="tmle", exposure="a", outcome="yb",
                                   outcome_sd=None, ratio_refused=reason)
    assert found["computed"] is False and found["e_value"] is None
    assert found["not_computed"] == (
        "No robustness value: it is defined for one least-squares coefficient (Cinelli & Hazlett "
        "2020, J R Stat Soc B 82:39), and targeted maximum likelihood averages its targeted "
        "outcome model's predictions, not one coefficient; no E-value: " + reason + ".")
    every = np.where(a == 1, 1.0, data["yb"])
    tm = est.tmle(every, a, X, learner="linear", family="binomial")
    assert "risk_ratio" in tm.extra and "odds_ratio" not in tm.extra
    assert tm.extra["odds_refused"] == ("every exposed row has the event, so the exposed odds are "
                                        "infinite and no odds ratio is defined")


# ── 3 · post-double selection ────────────────────────────────────────────────


def _cd_lasso(X: np.ndarray, y: np.ndarray, pen: np.ndarray) -> np.ndarray:
    """``argmin (1/n)‖y − Xb‖² + (1/n) Σ pen_j |b_j|`` by cyclic coordinate descent, by hand."""
    n, p = X.shape
    b = np.zeros(p)
    r = y.copy()
    sq = (X ** 2).sum(axis=0)
    for _ in range(200_000):
        biggest = 0.0
        for j in range(p):
            if sq[j] == 0:
                continue
            old = b[j]
            z = X[:, j] @ r + sq[j] * old
            new = np.sign(z) * max(abs(z) - pen[j] / 2.0, 0.0) / sq[j]
            if new != old:
                r -= X[:, j] * (new - old)
                b[j] = new
                biggest = max(biggest, abs(new - old))
        if biggest < 1e-14:
            break
    return b


def _rlasso_by_hand(X: np.ndarray, y: np.ndarray) -> tuple[list[int], float]:
    """Belloni et al. 2012's plug-in lasso (Algorithm A.1), transcribed from R's hdm 0.3.2
    ``rlasso.default`` with its defaults (``post = TRUE``, ``homoscedastic = FALSE``,
    ``X.dependent.lambda = FALSE``, c = 1.1, γ = 0.1/ln n, ``numIter = 15``, ``tol = 1e-5``),
    line by line:

    * ``startingval <- init_values(x, y)$residuals``: ``lm(y ~ X[, index])`` on the five columns
      with the largest ``abs(cor(y, X))`` (``order(corr, decreasing = TRUE)``: NA last, ties in
      column order);
    * ``Ups0 <- 1/sqrt(n) * sqrt(t(t(startingval^2) %*% (x^2)))``, ``lambda <- lambda0 * Ups0``;
    * ``s0 <- sqrt(var(y))``; then up to 15 times: the lasso at ``lambda/2`` on the first pass
      (``if (mm == 1 && post)``), ``lambda`` after; an empty selection returns at once;
      ``e1 <- y - x1 %*% coef(lm(y ~ -1 + x1))``, ``s1 <- sqrt(var(e1))``,
      ``Ups1 <- 1/sqrt(n) * sqrt(t(t(e1^2) %*% (x^2)))``; stop when ``abs(s0 - s1) < tol``.

    The lasso is :func:`_cd_lasso` (``‖y − Xb‖² + Σ λ_j|b_j|``, hdm's ``LassoShooting.fit``
    objective, solved here to 1e-14 rather than hdm's 1e-5)."""
    n, p = X.shape
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    lam = 2 * 1.1 * math.sqrt(n) * float(norm.ppf(1 - (0.1 / math.log(n)) / (2 * p)))
    corr = []
    for j in range(p):
        denom = math.sqrt(float(Xc[:, j] @ Xc[:, j]) * float(yc @ yc))
        corr.append(abs(float(Xc[:, j] @ yc)) / denom if denom > 0 else -1.0)
    top = sorted(range(p), key=lambda j: -corr[j])[:5]  # sorted is stable: ties in column order
    A = np.column_stack([np.ones(n), Xc[:, top]])
    e = yc - A @ np.linalg.lstsq(A, yc, rcond=None)[0]
    ups = np.sqrt((Xc ** 2 * e[:, None] ** 2).mean(axis=0))
    s0 = float(np.std(yc, ddof=1))
    support: list[int] = []
    for mm in range(1, 16):
        b = _cd_lasso(Xc, yc, (lam / 2 if mm == 1 else lam) * ups)
        support = [int(j) for j in np.flatnonzero(np.abs(b) > 0)]
        if not support:
            break
        S = Xc[:, support]
        e = yc - S @ np.linalg.solve(S.T @ S, S.T @ yc)
        ups = np.sqrt((Xc ** 2 * e[:, None] ** 2).mean(axis=0))
        s1 = float(np.std(e, ddof=1))
        if abs(s0 - s1) < 1e-5:
            break
        s0 = s1
    return support, lam


def _hc3_by_hand(A: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    bread = np.linalg.inv(A.T @ A)
    beta = bread @ A.T @ y
    e = y - A @ beta
    h = np.einsum("ij,jk,ik->i", A, bread, A)
    V = bread @ (A.T * (e / (1 - h)) ** 2) @ A @ bread
    return float(beta[1]), math.sqrt(float(V[1, 1]))


def _toeplitz_design(rng: np.random.Generator, n: int, p: int,
                     truth: float = -0.6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Belloni, Chernozhukov & Hansen 2014's simulation shape: candidates with Toeplitz
    correlation 0.5^|j − k|; two confounders (x0, x3), an exposure-only cause (x6) and an
    outcome-only one (x8); the exposure's effect ``truth``."""
    L = np.linalg.cholesky(0.5 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p))))
    X = rng.normal(size=(n, p)) @ L.T
    d = 0.8 * X[:, 0] - 0.6 * X[:, 3] + 0.4 * X[:, 6] + rng.normal(size=n)
    y = truth * d + 0.9 * X[:, 0] + 0.7 * X[:, 3] - 0.5 * X[:, 8] + rng.normal(size=n)
    return X, d, y


def test_3_post_double_selection_agrees_with_a_numpy_hand_implementation_and_ranks_high():
    """Reference: the plug-in lasso and the double selection written out above in NumPy (a
    coordinate-descent lasso, not the app's scikit-learn solver; hdm 0.3.2's ``rlasso`` transcribed
    from its R source, not from the app), at the same penalty: the same selections, and the
    exposure's coefficient and HC3 standard error to 1e-8, on 300 rows with 120 candidates and on
    90 rows with 400 (a constant column and ten yes/no columns among them). 120 candidates for
    300 rows are many relative to n (Harrell's one per ten), so the lane ranks the method first, as
    sound, and the Router's stated lane names it first and says why; with 20 candidates it does
    not rank first."""
    rng = np.random.default_rng(1)
    n, p = 300, 120
    cov = 0.5 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.normal(size=(n, p)) @ np.linalg.cholesky(cov).T
    beta_d, beta_y = np.zeros(p), np.zeros(p)
    beta_d[[0, 3, 7]] = [1.0, -0.8, 0.6]
    beta_y[[0, 5, 9]] = [1.2, 0.7, -0.9]
    d = X @ beta_d + rng.normal(size=n)
    y = 0.5 * d + X @ beta_y + rng.normal(size=n) * (1 + 0.5 * np.abs(X[:, 1]))
    wide = _toeplitz_design(np.random.default_rng(12), 90, 400)
    wide[0][:, 5] = 1.0
    wide[0][:, 10:20] = (wide[0][:, 10:20] > 0).astype(float)
    for XX, dd, yy, truth in ((X, d, y, 0.5), (*wide, -0.6)):
        nn = len(yy)
        found = est.pds_lasso(yy, dd, XX)
        on_y, lam = _rlasso_by_hand(XX, yy)
        on_d, _ = _rlasso_by_hand(XX, dd)
        union = sorted(set(on_y) | set(on_d))
        theta, se = _hc3_by_hand(np.column_stack([np.ones(nn), dd, XX[:, union]]), yy)
        assert found.extra["lambda"] == pytest.approx(lam, rel=1e-12)
        assert found.extra["selected_outcome"] == on_y and found.extra["selected_exposure"] == on_d
        assert found.extra["selected"] == union and union
        assert found.estimate == pytest.approx(theta, abs=1e-8)
        assert found.se == pytest.approx(se, abs=1e-8)
        assert abs(found.estimate - truth) < 3 * found.se  # covers the simulation's truth
    # Ranked first, and asked, when the candidates are many relative to n (rung (d) ranks high).
    many = causal.many_candidates(n, p)
    assert many
    ranked = causal.options("continuous", "regression", many, False)
    assert ranked[0].key == "pds_lasso" and ranked[0].sound.verdict == "sound"
    few = causal.options("continuous", "regression", causal.many_candidates(n, 20), False)
    assert few[0].key == "dml_plr"
    from turbotab.core.decisions import ProjectState
    from turbotab.core.interview import route

    state = ProjectState(
        lens=["clinical"], target="y", task="regression", purpose="inference",
        grain={"grain": "one_row_per_unit"}, roles={"d": "exposure", "x0": "covariate"},
        exclusions=[], missing="complete_case", split={"holdout": 0.0},
        estimand={"exposure": "d", "measure": "mean_difference"},
        adjustment={"x0": {"causes_exposure": "yes", "causes_outcome": "yes",
                           "after_exposure": "no", "exposure": "d"}},
        clusters={"column": None, "acknowledged": True})
    stages = {s: {"status": "fresh"} for s in ("ingest", "oriented", "profile", "findings",
                                               "structure", "working", "target_info", "roles",
                                               "proposals", "cohort", "split", "causal_design")}
    card = {"purpose": "inference", "offered": True, "exposure": "d", "many": many,
            "stated": causal.stated_reason(ranked[0].key, many, p, n)}
    step = {s.key: s for s in route(state, stages, {"causal_design": card})}["causal"]
    assert step.status == "skipped" and step.reason == (
        "the primary model estimates the declared effect; with 120 candidate terms for a limiting "
        "sample size of 300 (more than one per 10), post-double-selection lasso ranks first and is "
        "one step away.")


def test_3b_post_double_selection_covers_the_truth_where_it_ranks_first():
    """Reference: the simulation's truth. Where the lane ranks post-double selection first, as
    "sound: selecting by both lassos keeps the intervals valid" (more candidates than one per 10
    rows), its 95% interval must cover the true effect. 1,000 data sets each of 100 rows with 300
    candidates and with 120 (:func:`_toeplitz_design`): the interval covers the truth in at least
    90% of them (nominal 95%; the theory is asymptotic, and at these sizes R's hdm covers about
    93%), the strongest confounder is selected in at least 95%, and no data set selects nothing at
    more than a 1% rate. A plug-in lasso started from ``y − ȳ`` at the full penalty covered 49% to
    75% here and kept that confounder in 47%."""
    for p in (300, 120):
        n = 100
        assert causal.options("continuous", "regression", causal.many_candidates(n, p),
                              False)[0].key == "pds_lasso"
        rng = np.random.default_rng(1)
        covered, kept, empty = [], [], []
        for _ in range(1000):
            X, d, y = _toeplitz_design(rng, n, p)
            found = est.pds_lasso(y, d, X)
            covered.append(abs(found.estimate - (-0.6)) < float(norm.ppf(0.975)) * found.se)
            kept.append(0 in found.extra["selected"])
            empty.append(not found.extra["selected"])
        assert np.mean(covered) >= 0.90, (p, np.mean(covered))
        assert np.mean(kept) >= 0.95, (p, np.mean(kept))
        assert np.mean(empty) <= 0.01, (p, np.mean(empty))


def _r_has(package: str) -> bool:
    done = subprocess.run([RSCRIPT, "-e", f'cat(requireNamespace("{package}", quietly = TRUE))'],
                          capture_output=True, text=True, timeout=120)
    return done.stdout.strip().endswith("TRUE")


@needs_r
def test_3c_post_double_selection_agrees_with_r_hdm(tmp_path):
    """Reference: R's hdm 0.3.2, the plug-in lasso's authors' package, ``rlassoEffect(x, y, d,
    method = "double selection")`` with its defaults: the same selections and the same coefficient
    (to 1e-6) on 24 data sets, 90 to 300 rows with 60 to 400 candidates (a constant column and
    yes/no columns among the widest). hdm is listed with the R references
    (``turbotab/server/requirements-dev-R.txt``) but was added after some machines were set up,
    so this skips unless R finds it (``R_LIBS`` may point to a library that holds it). The
    standard errors differ by design: hdm's are HC1-type, the app's HC3."""
    if not _r_has("hdm"):
        pytest.skip("R's hdm is not installed (set R_LIBS to a library that has it)")
    rng = np.random.default_rng(31)
    frames, found = {}, []
    for k, (n, p) in enumerate([(100, 300), (100, 120), (150, 200), (300, 60), (90, 400),
                                (200, 150)] * 4):
        X, d, y = _toeplitz_design(rng, n, p)
        if p == 400:
            X[:, 5] = 1.0
            X[:, 10:20] = (X[:, 10:20] > 0).astype(float)
        frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
        frame["d"], frame["y"] = d, y
        frames[f"d{k}"] = frame
        found.append(est.pds_lasso(y, d, X))
    ref = run_r(tmp_path, f"""
suppressMessages(library(hdm))
out <- list()
for (k in 0:{len(found) - 1}) {{ s <- as.data.frame(fread(get(sprintf("d%d_csv", k))))
  X <- as.matrix(s[, grepl("^x", names(s))])
  fit <- rlassoEffect(X, s$y, s$d, method = "double selection")
  out[[k + 1]] <- list(alpha = unname(fit$alpha), sel = which(fit$selection.index) - 1) }}
cat(toJSON(out, digits = NA))
""", frames)
    for e, r in zip(found, ref):
        assert e.extra["selected"] == sorted(int(v) for v in r["sel"])
        assert e.estimate == pytest.approx(r["alpha"][0], abs=1e-6)


# ── 4 · the assumptions, positivity, and trimming: helpers for the chain ─────


def test_4a_the_overlap_diagnostics_are_the_textbook_quantities():
    """Reference: the definitions, by hand: Kish's effective sample size ``(Σw)²/Σw²`` of each
    arm's inverse-probability weights, the share of rows beyond the truncation bound, the share of
    weights above ``1/b``, and each arm's propensity histogram on [0, 1]. The violation line is the
    lane's stated 1% of rows."""
    rng = np.random.default_rng(4)
    n = 2000
    g = np.clip(rng.beta(0.6, 0.6, size=n), 1e-4, 1 - 1e-4)
    a = (rng.uniform(size=n) < g).astype(float)
    b = est.tmle_gbound(n)
    found = est.overlap(g, a, b)
    w = np.where(a == 1, 1 / g, 1 / (1 - g))
    for arm, ess in ((1, found.ess_exposed), (0, found.ess_unexposed)):
        ww = w[a == arm]
        assert ess == pytest.approx(ww.sum() ** 2 / (ww ** 2).sum(), rel=1e-12)
    outside = (g < b) | (g > 1 - b)
    assert found.n_outside == int(outside.sum())
    assert found.share_outside == pytest.approx(outside.mean(), rel=1e-12)
    assert found.extreme_weight_share == pytest.approx((w > 1 / b).mean(), rel=1e-12)
    edges = np.linspace(0, 1, 21)
    assert found.exposed == pytest.approx(np.histogram(g[a == 1], edges)[0].tolist())
    assert found.violated is bool(outside.mean() >= 0.01)
    assert est.trim_rows(g, 0.1).sum() == int(((g >= 0.1) & (g <= 0.9)).sum())


def test_4c_a_survey_weighted_share_is_named_as_weighted():
    """Under survey weights the share beyond the bound is the weights' share, not the rows': the
    reason names it so beside the row count. Reference: the definitions, by hand (the count of
    rows beyond the bound; Σ w over them / Σ w); and the effect among the exposed's reading counts
    only propensities above 1 − b."""
    from types import SimpleNamespace

    from turbotab.core.stages.causal import _positivity_reason

    rng = np.random.default_rng(6)
    n = 1500
    g = np.clip(rng.beta(0.5, 2.0, size=n), 1e-5, 1 - 1e-5)
    a = (rng.uniform(size=n) < g).astype(float)
    w = np.exp(rng.normal(size=n))
    b = est.tmle_gbound(n)
    prep = SimpleNamespace(y=np.zeros(n), exposure="x")
    for target, outside in (("ATE", (g < b) | (g > 1 - b)), ("ATT", g > 1 - b)):
        found = est.overlap(g, a, b, w, target=target)
        share = float(np.sum(w * outside) / np.sum(w))
        assert found.n_outside == int(outside.sum())
        head = (f"{int(outside.sum()):,} of {n:,} rows ({share:.1%} of the survey-weighted total, "
                f"at or above the lane's line of 1%)")
        reason = _positivity_reason(prep, found, target=target, weighted=True)
        assert head in reason
        assert reason.startswith("For the effect among the exposed, ") == (target == "ATT")


# ── 5 · a survey design ──────────────────────────────────────────────────────


@needs_r
def test_5_survey_weights_enter_every_nuisance_fit_and_the_estimating_equation(tmp_path):
    """Reference: R's survey package over the same folds. The partially linear model is
    ``svyglm(ũ ~ 0 + ṽ)`` on residuals from weighted ``lm`` fits per fold; the interactive model is
    ``svymean`` of the doubly robust pseudo-outcome from weighted ``lm`` and ``glm`` fits per fold;
    TMLE is ``tmle(…, obsWeights = w)`` with ``svymean`` of its influence curve; the interactive
    model's risk ratio on a yes/no outcome is survey's ``svycontrast`` (the delta method) of
    ``log(φ₁/φ₀)`` over ``svymean(~φ₁ + φ₀)`` of its doubly robust terms. Each design is
    ``svydesign(ids = ~psu, strata = ~stratum, weights = ~w, nest = TRUE)``; a cluster-only design
    is ``svydesign(ids = ~psu)`` without weights or strata. Tolerance 1e-6 on every estimate and
    standard error. A simulation then shows the weights matter: under sampling that depends on an
    effect modifier, the weighted estimate covers the population effect and the unweighted one
    does not."""
    rng = np.random.default_rng(21)
    H, per = 15, 40
    n = H * 2 * per
    stratum = np.repeat(np.arange(H), 2 * per)
    psu = np.repeat(np.arange(H * 2), per)
    shared = rng.normal(scale=0.5, size=H * 2)[psu]
    X = rng.normal(size=(n, 3)) + 0.3 * shared[:, None]
    w = np.exp(rng.normal(scale=0.5, size=n)) * (1 + stratum % 3)
    d = 0.5 * X[:, 0] + rng.normal(size=n)
    y = 0.7 * d + X[:, 0] - 0.3 * X[:, 1] + shared + rng.normal(size=n)
    a = (rng.uniform(size=n) < _expit(0.4 * X[:, 0] - 0.6 * X[:, 1])).astype(float)
    ya = 1.2 * a + X[:, 0] + shared + rng.normal(size=n)
    # its own generator, so the population simulation below draws what it always drew
    yb = (np.random.default_rng(22).uniform(size=n)
          < _expit(-0.4 + 0.6 * a + 0.4 * X[:, 0] + shared)).astype(float)
    splits = est.sample_splits(n, 5, 1, seed=3, groups=psu)
    assert all(set(psu[train]).isdisjoint(psu[test]) for train, test in splits[0])  # whole PSUs
    bound = est.tmle_gbound(n)
    frame = pd.DataFrame(X, columns=["x0", "x1", "x2"])
    frame["d"], frame["y"], frame["a"], frame["ya"], frame["yb"] = d, y, a, ya, yb
    frame["w"], frame["stratum"], frame["psu"] = w, stratum, psu
    frame["fold"] = folds_frame(splits, n)["r1"]
    ref = run_r(tmp_path, f"""
suppressMessages({{library(survey); library(tmle)}})
df <- as.data.frame(fread(data_csv)); n <- nrow(df)
cross <- function(weighted) {{
  wt <- if (weighted) df$w else rep(1, n)
  lh <- mh <- g0 <- g1 <- m <- rep(NA, n)
  for (k in 1:5) {{ tr <- df$fold != k; te <- df$fold == k; d1 <- df[tr, ]; d1$wt <- wt[tr]
    lh[te] <- predict(lm(y ~ x0 + x1 + x2, data = d1, weights = wt), newdata = df[te, ])
    mh[te] <- predict(lm(d ~ x0 + x1 + x2, data = d1, weights = wt), newdata = df[te, ])
    m[te] <- predict(glm(a ~ x0 + x1 + x2, data = d1, weights = wt, family = quasibinomial),
                     newdata = df[te, ], type = "response")
    g0[te] <- predict(lm(ya ~ x0 + x1 + x2, data = d1[d1$a == 0, ], weights = wt),
                      newdata = df[te, ])
    g1[te] <- predict(lm(ya ~ x0 + x1 + x2, data = d1[d1$a == 1, ], weights = wt),
                      newdata = df[te, ]) }}
  m <- pmin(pmax(m, {bound!r}), 1 - {bound!r})
  data.frame(ures = df$y - lh, vres = df$d - mh,
             phi = g1 - g0 + df$a * (df$ya - g1) / m - (1 - df$a) * (df$ya - g0) / (1 - m),
             psu = df$psu, stratum = df$stratum, w = df$w) }}
s <- cross(TRUE)
des <- svydesign(ids = ~psu, strata = ~stratum, weights = ~w, nest = TRUE, data = s)
plr <- svyglm(ures ~ 0 + vres, design = des); irm <- svymean(~phi, des)
t1 <- tmle(Y = df$ya, A = df$a, W = df[, c("x0", "x1", "x2")], Qform = "Y~A+x0+x1+x2",
  gform = "A~x0+x1+x2", cvQinit = FALSE, family = "gaussian", obsWeights = df$w)
s$ic <- t1$estimates$IC$IC.ATE / (df$w / sum(df$w) * n)
tm <- svymean(~ic, svydesign(ids = ~psu, strata = ~stratum, weights = ~w, nest = TRUE, data = s))
u <- cross(FALSE)
cl <- svyglm(ures ~ 0 + vres, design = svydesign(ids = ~psu, data = u))
g0 <- g1 <- m <- rep(NA, n)  # a yes/no outcome: the doubly robust means, weighted glm per fold
for (k in 1:5) {{ tr <- df$fold != k; te <- df$fold == k; d1 <- df[tr, ]
  m[te] <- predict(glm(a ~ x0 + x1 + x2, data = d1, weights = w, family = quasibinomial),
                   newdata = df[te, ], type = "response")
  g0[te] <- predict(glm(yb ~ x0 + x1 + x2, data = d1[d1$a == 0, ], weights = w,
                        family = quasibinomial), newdata = df[te, ], type = "response")
  g1[te] <- predict(glm(yb ~ x0 + x1 + x2, data = d1[d1$a == 1, ], weights = w,
                        family = quasibinomial), newdata = df[te, ], type = "response") }}
m <- pmin(pmax(m, {bound!r}), 1 - {bound!r})
sb <- data.frame(phi1 = g1 + df$a * (df$yb - g1) / m, phi0 = g0 + (1 - df$a) * (df$yb - g0) / (1 - m),
                 psu = df$psu, stratum = df$stratum, w = df$w)
lr <- svycontrast(svymean(~phi1 + phi0, svydesign(ids = ~psu, strata = ~stratum, weights = ~w,
                                                   nest = TRUE, data = sb)), quote(log(phi1 / phi0)))
cat(toJSON(list(plr = c(coef(plr)[[1]], SE(plr)[[1]]), irm = c(coef(irm)[[1]], SE(irm)[[1]]),
  tmle = c(t1$estimates$ATE$psi, SE(tm)[[1]]), clusters = c(coef(cl)[[1]], SE(cl)[[1]]),
  log_rr = c(coef(lr)[[1]], SE(lr)[[1]])), digits = NA))
""", {"data": frame})
    design = est.Design(weight=w, stratum=stratum, psu=psu)
    plr = est.dml_plr(y, d, X, learner="linear", splits=splits, design=design)
    assert [plr.estimate, plr.se] == pytest.approx(ref["plr"], abs=1e-6)
    irm = est.dml_irm(ya, a, X, learner="linear", splits=splits, design=design, bound=bound)
    assert [irm.estimate, irm.se] == pytest.approx(ref["irm"], abs=1e-6)
    ratio = est.dml_irm(yb, a, X, learner="linear", splits=splits, design=design, bound=bound,
                        outcome_binary=True).extra["risk_ratio"]
    assert [math.log(ratio["estimate"]), ratio["se_log"]] == pytest.approx(ref["log_rr"], abs=1e-6)
    tm = est.tmle(ya, a, X, learner="linear", design=design)
    assert [tm.estimate, tm.se] == pytest.approx(ref["tmle"], abs=1e-6)
    clustered = est.dml_plr(y, d, X, learner="linear", splits=splits, design=est.Design(psu=psu))
    assert [clustered.estimate, clustered.se] == pytest.approx(ref["clusters"], abs=1e-6)
    assert plr.df == 2 * H - H and clustered.df == 2 * H - 1  # PSUs minus strata

    # The weights matter: a population whose effect differs by an effect modifier the sampling
    # over-represents. Truth: the population average effect, by its own definition.
    pop = 200_000
    m = rng.uniform(size=pop)
    z = rng.normal(size=pop)
    effect = 1.0 + 4.0 * m  # the effect grows with m
    exposed = (rng.uniform(size=pop) < _expit(0.5 * z)).astype(float)
    outcome = effect * exposed + z + m + rng.normal(size=pop)
    truth = float(effect.mean())  # 3.0
    pick = rng.uniform(size=pop) < 0.006 * (0.2 + 1.8 * (1 - m))  # low m over-sampled
    weight = 1 / (0.006 * (0.2 + 1.8 * (1 - m[pick])))
    W = np.column_stack([z[pick], m[pick]])
    sample = est.tmle(outcome[pick], exposed[pick], W, learner="linear")
    weighted = est.tmle(outcome[pick], exposed[pick], W, learner="linear",
                        design=est.Design(weight=weight))
    assert abs(weighted.estimate - truth) < 3 * weighted.se
    assert abs(sample.estimate - truth) > 3 * sample.se


# ── 4 and 6 · the chain, through the real server ─────────────────────────────

CHAIN_ROLES = {"person_id": "identifier", "age": "covariate", "smoker": "covariate",
               "income": "covariate", "heavy_user": "exposure", "fiber": "exposure",
               "ldl": "covariate"}
ADJUSTED_FOR_HEAVY = ["age", "smoker", "income", "fiber"]
ADJUSTED_FOR_FIBER = ["age", "smoker", "income", "heavy_user"]
CLOSING = (" It rests on the declared assumptions of no unmeasured confounding given the adjustment "
           "set, positivity, consistency and time ordering.")
# Sensitivity to unmeasured confounding (ruling 10), by ESTIMAND's functions: the E-value for every
# numeric outcome's difference, and post-double selection's robustness value before it.
E_VALUE_ONLY = (" Sensitivity to unmeasured confounding is reported by the E-value for the estimate "
                "and for the confidence limit nearer the null, never as a pass or a fail.")
RV_AND_E_VALUE = (" Sensitivity to unmeasured confounding is reported by the Cinelli–Hazlett "
                  "robustness value (each selected covariate a named benchmark) and by the E-value "
                  "for the estimate and for the confidence limit nearer the null, never as a pass "
                  "or a fail.")
# The partially linear model's estimate is one least-squares coefficient of the outcome's residual
# on the exposure's: its robustness value is that fit's, ranked first.
PLR_RV = (" Sensitivity to unmeasured confounding is reported by the Cinelli–Hazlett robustness "
          "value of the final least-squares step, the outcome's residual on the exposure's (the "
          "form the omitted-variable bound of Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis "
          "2022, NBER w30302 takes in the partially linear model), the median over the sample "
          "splits")
PLR_RV_AND_E_VALUE = (PLR_RV + " and by the E-value for the estimate and for the confidence limit "
                               "nearer the null, never as a pass or a fail.")
PLR_RV_ONLY = PLR_RV + ", never as a pass or a fail."


def e_value_by_hand(estimate: float, se: float, sd: float) -> tuple[float, float]:
    """VanderWeele & Ding 2017 for a difference in means, written out: RR ≈ exp(0.91 d), d the
    standardized difference, its interval exp(0.91 d ± 1.78 se_d); the E-value RR + √(RR(RR − 1))
    on the side of the null the estimate is on, and 1 for a limit across it."""
    d, sd_d = estimate / sd, se / sd
    rr, lo, hi = math.exp(0.91 * d), math.exp(0.91 * d - 1.78 * sd_d), math.exp(0.91 * d + 1.78 * sd_d)

    def e(r: float) -> float:
        r = r if r >= 1 else 1 / r
        return r + math.sqrt(r * (r - 1))

    limit = (1.0 if lo < 1 else e(lo)) if rr > 1 else (1.0 if hi > 1 else e(hi))
    return e(rr), limit


def rv_by_hand(t: float, dof: float) -> tuple[float, float]:
    """Cinelli & Hazlett 2020 from a least-squares coefficient's classical t and residual degrees
    of freedom: f = |t|/√dof, RV = ½(√(f⁴ + 4f²) − f²), and RV at α = 0.05 with f − f*,
    f* = |t*_{dof−1}|/√(dof − 1) (0 when that is not positive)."""
    from scipy import stats

    f = abs(t) / math.sqrt(dof)
    fa = f - abs(stats.t.ppf(0.025, dof - 1)) / math.sqrt(dof - 1)
    return (0.5 * (math.sqrt(f ** 4 + 4 * f ** 2) - f ** 2),
            0.5 * (math.sqrt(fa ** 4 + 4 * fa ** 2) - fa ** 2) if fa > 0 else 0.0)


def robustness_by_hand(A: np.ndarray, y: np.ndarray, j: int) -> tuple[float, float]:
    """Cinelli & Hazlett 2020 for column ``j`` of the least-squares design ``A``
    (:func:`rv_by_hand` of its classical t)."""
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ beta
    dof = len(y) - A.shape[1]
    se = math.sqrt(float(e @ e) / dof * np.linalg.inv(A.T @ A)[j, j])
    return rv_by_hand(float(beta[j] / se), dof)


def plr_by_hand(y: np.ndarray, d: np.ndarray, X: np.ndarray,
                splits: list[list[tuple[np.ndarray, np.ndarray]]]) -> dict[str, float]:
    """The partially linear model with least-squares nuisances on the given folds, by hand: per
    split the residuals ``u = y − ℓ̂``, ``v = d − m̂`` from fits on the other folds, θ = v·u/v·v,
    and the no-intercept regression's classical t on n − 1 df; the median θ, the median robustness
    values (:func:`rv_by_hand`), and the exposure's cross-fitted R² on the first split."""
    n = len(y)
    thetas, rvs, rvas, r2 = [], [], [], None
    for rep in splits:
        u, v = np.empty(n), np.empty(n)
        for train, test in rep:
            A = np.column_stack([np.ones(len(train)), X[train]])
            At = np.column_stack([np.ones(len(test)), X[test]])
            u[test] = y[test] - At @ np.linalg.lstsq(A, y[train], rcond=None)[0]
            v[test] = d[test] - At @ np.linalg.lstsq(A, d[train], rcond=None)[0]
        theta = float(v @ u / (v @ v))
        e = u - theta * v
        t = theta / math.sqrt(float(e @ e) / (n - 1) / float(v @ v))
        rv, rva = rv_by_hand(t, n - 1)
        thetas.append(theta)
        rvs.append(rv)
        rvas.append(rva)
        if r2 is None:
            r2 = 1.0 - float(v @ v) / float(((d - d.mean()) ** 2).sum())
    return {"theta": float(np.median(thetas)), "rv": float(np.median(rvs)),
            "rv_alpha": float(np.median(rvas)), "r2": float(r2)}


def logit_by_hand(y: np.ndarray, X: np.ndarray, at: np.ndarray) -> np.ndarray:
    """statsmodels' maximum-likelihood logistic regression of ``y`` on ``X`` (an intercept added),
    its probabilities at ``at``: an independent solver, not the app's Newton steps."""
    import statsmodels.api as sm

    fit = sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=0, tol=1e-12, maxiter=200)
    return fit.predict(sm.add_constant(at, has_constant="add"))


def irm_risk_ratio_by_hand(y: np.ndarray, a: np.ndarray, X: np.ndarray,
                           splits: list[list[tuple[np.ndarray, np.ndarray]]], bound: float,
                           score: str = "ATE") -> dict[str, float]:
    """The interactive model's risk difference and risk ratio on a yes/no outcome, by hand with
    statsmodels' logistic fits on the given folds: the doubly robust means per split (for the ATT,
    the exposed's observed risk and their doubly robust risk unexposed, p the test fold's exposed
    share), the log ratio's delta-method standard error from the same terms, and DoubleML's median
    aggregation; the interval on the normal scale."""
    n = len(y)
    diffs, logs, ses = [], [], []
    for rep in splits:
        m, g0, g1 = np.empty(n), np.empty(n), np.empty(n)
        p = np.empty(n)
        for train, test in rep:
            m[test] = logit_by_hand(a[train], X[train], X[test])
            g0[test] = logit_by_hand(y[train][a[train] == 0], X[train][a[train] == 0], X[test])
            g1[test] = logit_by_hand(y[train][a[train] == 1], X[train][a[train] == 1], X[test])
            p[test] = a[test].mean()
        m = np.clip(m, bound, 1 - bound)
        if score == "ATE":
            phi1 = g1 + a * (y - g1) / m
            phi0 = g0 + (1 - a) * (y - g0) / (1 - m)
            j = np.ones(n)
        else:
            phi1 = a * y / p
            phi0 = (a * g0 + m * (1 - a) * (y - g0) / (1 - m)) / p
            j = a / p
        mu1, mu0 = phi1.sum() / j.sum(), phi0.sum() / j.sum()
        diffs.append(mu1 - mu0)
        logs.append(math.log(mu1 / mu0))
        # log μ₁ − log μ₀, each mean a ratio of sums: its gradient applied to the rows' terms
        u = phi1 / phi1.sum() - phi0 / phi0.sum()
        ses.append(math.sqrt(float(u @ u)))
    t, s = np.asarray(logs), np.asarray(ses)
    lr = float(np.median(t))
    se = math.sqrt(float(np.median(n * s ** 2 + (t - lr) ** 2)) / n)
    z = float(norm.ppf(0.975))
    return {"rd": float(np.median(diffs)), "rr": math.exp(lr), "se_log": se,
            "lo": math.exp(lr - z * se), "hi": math.exp(lr + z * se)}


def e_value_of_ratio(rr: float, lo: float, hi: float) -> tuple[float, float]:
    """VanderWeele & Ding 2017's E-value of a risk ratio, RR + √(RR(RR − 1)) (of 1/RR below 1),
    and of the confidence limit nearer the null (1 when the interval includes it)."""
    def e(r: float) -> float:
        r = r if r >= 1 else 1 / r
        return r + math.sqrt(r * (r - 1))

    limit = (1.0 if lo < 1 else e(lo)) if rr > 1 else (1.0 if hi > 1 else e(hi))
    return e(rr), limit


def cohort(path: Path, n: int = 1500, seed: int = 11) -> pd.DataFrame:
    """A cohort with a yes/no exposure (``heavy_user``) that age all but decides (poor overlap),
    a numeric one (``fiber``), three confounders, and a mediator of both (``ldl``). The truth: the
    total effect of ``heavy_user`` is −3 − 0.1·5 = −3.5 and of ``fiber`` −0.5 − 0.1·0.8 = −0.58."""
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 10, n).round(1)
    smoker = (rng.uniform(size=n) < 0.3).astype(int)
    income = rng.normal(5, 2, n).round(2)
    heavy = (rng.uniform(size=n) < _expit(-0.3 + 0.22 * (age - 50) - 0.5 * smoker
                                          + 0.2 * (income - 5))).astype(int)
    fiber = (15 + 0.6 * (income - 5) - 2 * smoker + 0.05 * (age - 50)
             + rng.normal(0, 4, n)).round(1)
    ldl = (120 - 5 * heavy - 0.8 * fiber + rng.normal(0, 12, n)).round(1)
    sbp = (120 + 0.4 * (age - 50) + 5 * smoker - 1.0 * (income - 5) - 3 * heavy - 0.5 * fiber
           + 0.1 * ldl + rng.normal(0, 8, n)).round(1)
    frame = pd.DataFrame({"person_id": np.arange(1, n + 1), "age": age, "smoker": smoker,
                          "income": income, "heavy_user": heavy, "fiber": fiber, "ldl": ldl,
                          "sbp": sbp})
    frame.to_csv(path, index=False)
    return frame


def cohort_truth() -> Any:
    from turbotab.core.tests.truths import Truth

    # The author's causal knowledge: the confounders cause both exposures and the outcome; each
    # exposure is, for the other's effect, a cause of the outcome only; ldl comes after both.
    return Truth({"adjust:age": "yes,yes,no", "adjust:smoker": "yes,yes,no",
                  "adjust:income": "yes,yes,no", "adjust:fiber": "no,yes,no",
                  "adjust:heavy_user": "no,yes,no", "adjust:ldl": "no,yes,yes",
                  "code_or_count:smoker": "code", "code_or_count:heavy_user": "code",
                  "cluster:person_id": "no"}, fixture="the causal-lane cohort")


def ticked(items: list[str]) -> str:
    shown = [f"`{c}`" for c in items]
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


def _open_cohort(tmp_path: Path, client: Any, purpose: str = "inference") -> Any:
    from turbotab.core.tests.acceptance.server_drive import open_project

    path = tmp_path / "cohort.csv"
    cohort(path)
    drive = open_project(client, path, cohort_truth())
    drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": "sbp"})
    drive.reach("purpose")
    drive.decide({"kind": "set_purpose", "purpose": purpose})
    return drive


def _to_the_plan(drive: Any, exposure: str) -> None:
    drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
    drive.reach("roles")
    drive.decide_roles(CHAIN_ROLES)
    drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
    drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
    drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
    drive.exposure = exposure


def _causal_artifact(drive: Any, check: Any, timeout: float = 240.0) -> dict[str, Any]:
    """The causal stage's artifact once ``check(artifact)`` holds (a recomputation may still be
    under way when the decision returns)."""
    import time

    end = time.monotonic() + timeout
    while True:
        art = drive.artifact("causal")
        if check(art):
            return art
        assert time.monotonic() < end, art
        time.sleep(0.1)


def test_4b_assumptions_come_before_any_estimate_and_positivity_is_block_and_record(tmp_path):
    """On a cohort where age all but decides who is a heavy user: the four assumptions are refused
    until declared, each shown with its diagnostic, and no estimate exists before; positivity is
    then refused with the trimming exit and the record exit; trimmed, the estimate is the overlap
    population's. References: statsmodels' logistic regression for the propensities the trim reads
    (the estimator's own model, fit independently); R's tmle on the trimmed rows with the same glm
    models (1e-6) where R is installed; the methods sentence, verbatim."""
    import statsmodels.api as sm

    from turbotab.core.tests.acceptance.server_drive import local_server

    with local_server(tmp_path / "home") as client:
        drive = _open_cohort(tmp_path, client)
        _to_the_plan(drive, "heavy_user")
        drive.answer_plan("heavy_user")
        # Stated while its card computes (it never holds the models question), then stated from it.
        assert drive.reach("causal")["status"] == "skipped"
        design = drive.artifact("causal_design")
        step = next(s for s in drive.view()["interview"] if s["key"] == "causal")
        assert design["offered"] and design["exposure_kind"] == "binary"
        assert [o["key"] for o in design["options"]] == ["tmle", "dml_irm", "dml_plr",
                                                         "pds_lasso", "none"]
        assert step["status"] == "skipped" and step["reason"] == design["stated"]
        assert [a["key"] for a in design["assumptions"]] == ALL4
        positivity = next(a for a in design["assumptions"] if a["key"] == "positivity")
        assert positivity["status"] == "violated" and design["positivity"]["violated"]
        overlap = design["overlap"]
        assert overlap["share_outside"] >= 0.01 and overlap["ess_exposed"] < overlap["n_exposed"]
        assert sum(overlap["exposed"]) == overlap["n_exposed"]  # every exposed row is drawn
        # The effect among the exposed reads positivity on its own terms (a propensity above
        # 1 − b): here it is violated too, so it is refused with its own reason, and the trimming
        # exit says it changes that estimand to the average effect in the overlap population.
        att = drive.post({"kind": "set_causal", "exposure": "heavy_user", "method": "dml_irm",
                          "learner": "linear", "population": "exposed", "assumptions": ALL4})
        assert att.status_code == 409, att.text
        att_error = att.json()["error"]
        assert att_error["code"] == "positivity"
        assert att_error["message"] == design["positivity_att"]["reason"]
        trim_att, record_att = att_error["exits"]
        assert trim_att["label"] == ("Trim to propensities in [0.1, 0.9] and estimate the average "
                                     "effect there (the overlap population), not the effect among "
                                     "the exposed")
        assert trim_att["decision"]["population"] == "all" and trim_att["decision"]["trim"] == 0.1
        assert record_att["decision"]["population"] == "exposed"
        assert record_att["decision"]["acknowledged"]
        body = {"kind": "set_causal", "exposure": "heavy_user", "method": "tmle",
                "learner": "linear"}
        first = drive.post(body)
        assert first.status_code == 409, first.text
        error = first.json()["error"]
        assert error["code"] == "assumptions_first"
        for a in design["assumptions"]:  # each one shown with its diagnostic, before any estimate
            assert f"{a['label']}: {a['statement']} {a['diagnostic']}" in error["message"]
        assert drive.view()["stages"]["causal"]["status"] == "blocked"  # nothing was estimated
        declared = drive.post(error["exits"][0]["decision"])
        assert declared.status_code == 409, declared.text
        error = declared.json()["error"]
        assert error["code"] == "positivity" and error["message"] == design["positivity"]["reason"]
        trim, record = error["exits"]
        assert trim["decision"]["trim"] == causal.TRIM_AT and record["decision"]["acknowledged"]
        assert drive.view()["stages"]["causal"]["status"] == "blocked"
        drive.decide(trim["decision"])
        # Recorded, and still no estimate: the primary model is declared first (the whole plan
        # before any estimate is shown).
        assert drive.view()["stages"]["causal"]["status"] == "blocked"
        drive.decide({"kind": "select_models", "models": ["linear"]})
        art = _causal_artifact(drive, lambda a: bool(a.get("estimates")))
        state = drive.view()["state"]

    order = list(art)
    assert order.index("assumptions") < order.index("overlap") < order.index("estimates")
    assert art["declared"] == ALL4 and art["withheld"] is None
    # The card's two readings, by an independent cross-fitted logistic regression on its folds.
    raw = pd.read_csv(tmp_path / "cohort.csv")
    W = raw[ADJUSTED_FOR_HEAVY].to_numpy(dtype=float)
    a = raw["heavy_user"].to_numpy(dtype=float)
    n = len(raw)
    g_card = np.empty(n)
    for train, test in est.sample_splits(n, 5, 1, seed=0)[0]:
        g_card[test] = logit_by_hand(a[train], W[train], W[test])
    b = 5 / math.sqrt(n) / math.log(n)
    assert overlap["share_outside"] == pytest.approx(np.mean((g_card < b) | (g_card > 1 - b)))
    n_att = int((g_card > 1 - b).sum())
    assert design["positivity_att"]["reason"].startswith(
        f"For the effect among the exposed, {n_att:,} of {n:,} rows ({n_att / n:.1%}, at or above "
        f"the lane's line of 1%) have a propensity above {1 - b:.4f}, the bound the estimator caps "
        f"at")
    # The trim, by an independent fit of the estimator's propensity model on every row.
    g = sm.Logit(raw["heavy_user"].to_numpy(), sm.add_constant(W)).fit(disp=0, tol=1e-12)
    g_hat = g.predict(sm.add_constant(W))
    keep = (g_hat >= 0.1) & (g_hat <= 0.9)
    kept, trimmed = int(keep.sum()), int((~keep).sum())
    assert (art["n"], art["n_trimmed"]) == (kept, trimmed)
    b = 5 / math.sqrt(kept) / math.log(kept)
    assert art["methods"] == (
        f"The total effect of `heavy_user` on `sbp` was also estimated by targeted maximum "
        f"likelihood (van der Laan & Rubin 2006, Int J Biostat 2(1); as R's tmle 2.1.1 computes "
        f"it), as the average effect in the overlap population: the difference in the mean outcome "
        f"between `heavy_user` = `1` and the other level; main-terms linear and logistic regression "
        f"modeled the outcome and the propensity given {ticked(ADJUSTED_FOR_HEAVY)}, each fit on "
        f"every analyzed row, with the propensity truncated below at {b:.4f} for each level "
        f"(Gruber et al. 2022, Am J Epidemiol 191:1640), and the interval is read off the "
        f"influence curve, on {kept:,} complete rows. The {trimmed:,} rows with a propensity "
        f"outside [0.1, 0.9] were trimmed (Crump et al. 2009, Biometrika 96:187), so the estimate "
        f"is the effect among the {kept:,} rows where both exposure levels are plausible."
        + CLOSING + E_VALUE_ONLY)
    assert state["causal"]["trim"] == causal.TRIM_AT
    [estimate] = art["estimates"]
    assert estimate["label"] == "Average effect" and estimate["measure"] == "mean_difference"
    assert abs(estimate["estimate"] - (-3.5)) < 3 * estimate["se"]  # the cohort's truth
    if RSCRIPT is not None:
        ref = run_r(tmp_path / "r", """
suppressMessages(library(tmle))
df <- as.data.frame(fread(data_csv))
t <- tmle(Y = df$sbp, A = df$heavy_user, W = df[, c("age", "smoker", "income", "fiber")],
  Qform = "Y~A+age+smoker+income+fiber", gform = "A~age+smoker+income+fiber", cvQinit = FALSE,
  family = "gaussian")$estimates$ATE
cat(toJSON(c(t$psi, sqrt(t$var.psi)), digits = NA))
""", {"data": raw.loc[keep]})
        assert [estimate["estimate"], estimate["se"]] == pytest.approx(ref, abs=1e-6)


def test_6_the_shortest_leash_and_every_relation_of_the_lane_fires(tmp_path):
    """The chain (BLUEPRINT §13; MODELING_SEQUENCE §2 and §4), on the same cohort:

    * never under prediction (refused, the exit is the purpose, and the exit is taken);
    * only after the plan: the question waits behind the exposure and the adjustment set;
    * rung (d): the learners and the lassos see only what the answers adjust for, never the
      mediator they leave out;
    * the first causal estimate shown locks the plan, with the causal answer in it;
    * sensitivity to unmeasured confounding is required beside every estimate and computed by
      ESTIMAND's functions: the E-value of the standardized difference for each estimator, the
      robustness value before it for post-double selection (each against a computation by hand),
      and the methods text says which;
    * a new exposure re-asks the causal answer, and its estimate is withheld until then;
    * the partially linear model and post-double selection, each against a computation by hand
      on the same folds or penalty, with their methods sentences verbatim.
    """
    from turbotab.core.tests.acceptance.server_drive import local_server

    with local_server(tmp_path / "home") as client:
        drive = _open_cohort(tmp_path, client, purpose="prediction")
        refused = drive.post({"kind": "set_causal", "exposure": "heavy_user", "method": "tmle",
                              "assumptions": ALL4})
        assert refused.status_code == 409
        error = refused.json()["error"]
        assert error["code"] == "not_inference"
        steps = {s["key"]: s for s in drive.view()["interview"]}
        assert steps["causal"]["status"] == "not_applicable"
        drive.decide(error["exits"][0]["decision"])  # the exit is taken: inference
        _to_the_plan(drive, "heavy_user")
        held = drive.post({"kind": "set_causal", "exposure": "heavy_user", "method": "tmle",
                           "assumptions": ALL4})
        assert held.status_code == 409 and held.json()["error"]["code"] in ("not_yet",
                                                                            "no_estimand")
        drive.answer_plan("heavy_user")
        design = drive.artifact("causal_design")
        assert "ldl" not in design["candidates"] and design["offered"]  # rung (d)
        drive.decide({"kind": "set_causal", "exposure": "heavy_user", "method": "tmle",
                      "learner": "linear", "assumptions": ALL4, "acknowledged": True})
        assert drive.view()["state"]["plan_locked"] is None  # nothing shown yet
        drive.decide({"kind": "select_models", "models": ["linear"]})
        first = _causal_artifact(drive, lambda a: bool(a.get("estimates")))
        view = drive.view()
        assert view["state"]["plan_locked"] is True
        lock = next(r for r in view["decisions"] if r["decision"]["kind"] == "lock_plan")
        assert lock["decision"]["plan"]["causal"]["method"] == "tmle"
        sensitivity = first["sensitivity"]
        assert sensitivity["required"] is True and sensitivity["computed"] is True
        assert sensitivity["estimate"] == first["estimates"][0]["estimate"]
        assert sensitivity["methods"] == ["e_value"] and sensitivity["robustness"] is None
        assert sensitivity["not_computed"].startswith("No robustness value: it is defined for one "
                                                      "least-squares coefficient")
        first_sd = float(pd.read_csv(tmp_path / "cohort.csv")["sbp"].std(ddof=1))
        assert first["n"] == len(pd.read_csv(tmp_path / "cohort.csv"))
        point, limit = e_value_by_hand(first["estimates"][0]["estimate"],
                                       first["estimates"][0]["se"], first_sd)
        assert sensitivity["e_value"]["point"] == pytest.approx(point, rel=1e-10)
        assert sensitivity["e_value"]["limit"] == pytest.approx(limit, rel=1e-10)
        kept_on_record = next(r for r in view["decisions"]
                              if r["decision"]["kind"] == "set_causal")["sentence"]

        # A new exposure re-asks the causal answer (MODELING_SEQUENCE §2, "invalidates").
        drive.decide({"kind": "set_estimand", "exposure": "fiber", "effect": "total",
                      "measure": "mean_difference"})
        steps = {s["key"]: s for s in drive.view()["interview"]}
        assert steps["causal"]["status"] != "answered"
        withheld = client.get(f"/api/projects/{drive.pid}/stages/causal").json()["artifact"]
        assert withheld["estimates"] == [] and withheld["withheld"]
        drive.answer_plan("fiber")
        reasked = _causal_artifact(drive, lambda a: a.get("exposure") == "heavy_user"
                                   and "re-asked" in str(a.get("withheld")))
        assert reasked["estimates"] == []
        assert {s["key"]: s for s in drive.view()["interview"]}["causal"]["status"] == "skipped"
        design = drive.artifact("causal_design")
        assert design["exposure"] == "fiber" and design["exposure_kind"] == "continuous"
        assert design["variation"]["violated"] is False and "ldl" not in design["candidates"]
        assert [o["key"] for o in design["options"]][:2] == ["dml_plr", "pds_lasso"]

        drive.decide({"kind": "set_causal", "exposure": "fiber", "method": "dml_plr",
                      "learner": "linear", "repetitions": 3, "assumptions": ALL4})
        plr = _causal_artifact(drive, lambda a: a.get("method") == "dml_plr"
                               and bool(a.get("estimates")))
        drive.decide({"kind": "set_causal", "exposure": "fiber", "method": "pds_lasso",
                      "assumptions": ALL4})
        pds = _causal_artifact(drive, lambda a: a.get("method") == "pds_lasso"
                               and bool(a.get("estimates")))
        drive.decide({"kind": "set_causal", "exposure": "fiber", "method": "none"})
        none = _causal_artifact(drive, lambda a: a.get("method") == "none")
        records = [r for r in drive.view()["decisions"] if r["decision"]["kind"] == "set_causal"]

    raw = pd.read_csv(tmp_path / "cohort.csv")
    n = len(raw)
    # Every row was kept on the record although positivity is practically violated: the methods
    # sentence says so with the estimator's own reading (main-terms logistic regression on every
    # row, fit here by statsmodels), not only the Record.
    W = raw[ADJUSTED_FOR_HEAVY].to_numpy(dtype=float)
    g = logit_by_hand(raw["heavy_user"].to_numpy(dtype=float), W, W)
    b = 5 / math.sqrt(n) / math.log(n)
    outside = int(((g < b) | (g > 1 - b)).sum())
    assert first["methods"].endswith(
        " It rests on the declared assumptions of no unmeasured confounding given the adjustment "
        "set, consistency and time ordering. Positivity is practically violated: "
        f"{outside:,} of {n:,} rows ({outside / n:.1%}) have a propensity outside [{b:.4f}, "
        f"{1 - b:.4f}]; every row was kept, as recorded, so the estimate extrapolates where one "
        "level of `heavy_user` is all but impossible given the covariates, a stated limitation."
        + E_VALUE_ONLY)
    assert kept_on_record.endswith("; every row is kept although positivity is practically "
                                   "violated, a stated limitation.")
    y = raw["sbp"].to_numpy(dtype=float)
    d = raw["fiber"].to_numpy(dtype=float)
    X = raw[ADJUSTED_FOR_FIBER].to_numpy(dtype=float)
    # The partially linear model, by hand on the same folds (the folds are its input), with the
    # robustness value of its final least-squares step.
    hand = plr_by_hand(y, d, X, est.sample_splits(n, 5, 3, seed=0))
    [row] = plr["estimates"]
    assert row["estimate"] == pytest.approx(hand["theta"], abs=1e-8)
    assert abs(row["estimate"] - (-0.58)) < 3 * row["se"]  # the cohort's truth
    assert plr["methods"] == (
        "The total effect of `fiber` on `sbp` was also estimated by double/debiased machine "
        "learning in the partially linear model (Chernozhukov et al. 2018, Econom J 21:C1; as R's "
        "DoubleML 1.0.2 computes it), as the difference in the mean outcome per unit of `fiber`: "
        f"main-terms linear and logistic regression predicted the outcome and the exposure from "
        f"{ticked(ADJUSTED_FOR_FIBER)}, each fit on the other 4 of 5 folds, and the outcome's "
        f"residual was regressed on the exposure's; the estimate is the median over 3 random "
        f"sample splits, on {n:,} complete rows." + CLOSING + PLR_RV_AND_E_VALUE)
    point, limit = e_value_by_hand(row["estimate"], row["se"], float(np.std(y, ddof=1)))
    assert (plr["sensitivity"]["e_value"]["point"], plr["sensitivity"]["e_value"]["limit"]) == \
        pytest.approx((point, limit), rel=1e-10)
    assert plr["sensitivity"]["methods"] == ["robustness_value", "e_value"]
    assert (plr["sensitivity"]["robustness"]["rv"], plr["sensitivity"]["robustness"]["rv_alpha"]) \
        == pytest.approx((hand["rv"], hand["rv_alpha"]), abs=1e-8)
    # Post-double selection, by hand at the same penalty, over the declared candidates only.
    on_y, _ = _rlasso_by_hand(X, y)
    on_d, _ = _rlasso_by_hand(X, d)
    union = sorted(set(on_y) | set(on_d))
    theta, se = _hc3_by_hand(np.column_stack([np.ones(n), d, X[:, union]]), y)
    [row] = pds["estimates"]
    assert row["estimate"] == pytest.approx(theta, abs=1e-8)
    assert row["se"] == pytest.approx(se, abs=1e-8)

    def names(idx: list[int]) -> list[str]:
        return [ADJUSTED_FOR_FIBER[i] for i in sorted(idx)]

    def listed(idx: list[int]) -> str:
        return ticked(names(idx)) if idx else "none"

    assert pds["selected"]["outcome"] == names(on_y)
    assert pds["selected"]["exposure"] == names(on_d)
    assert "ldl" not in pds["selected"]["union"]
    assert pds["methods"] == (
        "The total effect of `fiber` on `sbp` was also estimated by post-double-selection lasso "
        "(Belloni, Chernozhukov & Hansen 2014, Rev Econ Stud 81:608), as the difference in the "
        f"mean outcome per unit of `fiber`: plug-in lassos over the 4 declared model terms of "
        f"{ticked(ADJUSTED_FOR_FIBER)} selected {listed(on_y)} for the outcome and "
        f"{listed(on_d)} for the exposure, and the outcome was regressed on `fiber` and "
        f"{ticked(names(union)) if union else 'no covariate'}, with HC3 standard errors, on "
        f"{n:,} complete rows." + CLOSING + RV_AND_E_VALUE)
    rv, rv_alpha = robustness_by_hand(np.column_stack([np.ones(n), d, X[:, union]]), y, 1)
    found = pds["sensitivity"]
    assert found["methods"] == ["robustness_value", "e_value"] and found["not_computed"] is None
    assert (found["robustness"]["rv"], found["robustness"]["rv_alpha"]) == pytest.approx(
        (rv, rv_alpha), rel=1e-8)
    assert {b["covariate"] for b in found["robustness"]["benchmarks"]} == set(names(union))
    assert found["reading"].startswith("An unmeasured confounder would need a partial R² of "
                                       f"{rv:.1%} with both `fiber` and `sbp`")
    assert none["estimates"] == [] and none["withheld"] is None
    # recorded after the estimates were seen, and its sentence leads with it (decisions.disclose)
    assert records[-1]["sentence"] == ("After the estimates were seen, no causal machine-learning "
                                       "estimate was set beside the primary model for the effect "
                                       "of `fiber`.")
    # Every set_causal after the first estimate is marked as made after the estimates were seen.
    assert [r["after_estimates"] for r in records[-3:]] == [True, True, True]


def survey_cohort(path: Path, seed: int = 5) -> pd.DataFrame:
    """A surveyed cohort: 15 strata of two PSUs of 40, informative weights, a numeric exposure
    whose effect differs by an effect modifier the weights correct for, and 6% of incomes blank."""
    rng = np.random.default_rng(seed)
    H, per = 15, 40
    n = H * 2 * per
    stratum = np.repeat(np.arange(1, H + 1), 2 * per)
    psu = np.tile(np.repeat([1, 2], per), H)
    shared = rng.normal(scale=0.5, size=H * 2)[np.repeat(np.arange(H * 2), per)]
    age = (rng.normal(50, 10, n) + 3 * shared).round(1)
    income = (rng.normal(5, 2, n) + shared).round(2)
    fiber = (15 + 0.6 * (income - 5) + 0.05 * (age - 50) + rng.normal(0, 4, n)).round(1)
    weight = (np.exp(rng.normal(scale=0.4, size=n)) * (1 + stratum % 3) * 1000).round(1)
    sbp = (120 + 0.4 * (age - 50) - 1.0 * (income - 5) - (0.3 + 0.1 * (stratum % 3)) * fiber
           + 2 * shared + rng.normal(0, 8, n)).round(1)
    frame = pd.DataFrame({"SEQN": np.arange(1, n + 1), "age": age, "income": income,
                          "fiber": fiber, "sbp": sbp, "WTMEC2YR": weight, "SDMVSTRA": stratum,
                          "SDMVPSU": psu})
    frame.loc[rng.uniform(size=n) < 0.06, "income"] = np.nan
    frame.to_csv(path, index=False)
    return frame


@needs_r
def test_7_a_survey_design_and_multiple_imputation_in_the_chain(tmp_path):
    """The survey and missing-data relations (MODELING_SEQUENCE §2), through the real server:

    * under the surveyed-population answer the post-double-selection lasso is blocked and recorded,
      its exits the weighted partially linear model and the sample-only record;
    * the multiple-imputation answer over incomplete rows is blocked and recorded too (the lane is
      not pooled over imputations in v2), its exit the complete rows with their assumption stated;
    * the weighted estimate: the weights in every fold's least-squares fits and in the estimating
      equation, PSUs kept whole in the folds, the variance linearized over the design with the
      incomplete rows a domain. Reference: R's survey package, ``svyglm(ũ ~ 0 + ṽ)`` over
      ``subset(design, complete)`` on residuals from weighted ``lm`` fits on the same folds, each
      split aggregated as DoubleML does (written out here); 1e-6. The methods sentence, verbatim.
    """
    from turbotab.core.tests.acceptance.server_drive import local_server, open_project
    from turbotab.core.tests.truths import Truth

    path = tmp_path / "surveyed.csv"
    raw = survey_cohort(path)
    truth = Truth({"adjust:age": "yes,yes,no", "adjust:income": "yes,yes,no",
                   "cluster:SEQN": "no"}, fixture="the surveyed cohort")
    roles = {"SEQN": "identifier", "age": "covariate", "income": "covariate", "fiber": "exposure",
             "WTMEC2YR": "design", "SDMVSTRA": "design", "SDMVPSU": "design"}
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "sbp"})
        drive.reach("purpose")
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        drive.reach("roles")
        drive.decide_roles(roles)
        drive.answer("survey", {"kind": "set_survey", "estimand": "population",
                                "weight": "WTMEC2YR", "strata": "SDMVSTRA", "psu": "SDMVPSU"})
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "multiple_imputation"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.exposure = "fiber"
        drive.answer_plan("fiber")
        design = drive.artifact("causal_design")
        assert design["survey"] == {"population": True, "weight": "WTMEC2YR"}
        assert design["missing"]["blocked"] and design["missing"]["answer"] == "multiple_imputation"
        assert next(o for o in design["options"] if o["key"] == "pds_lasso")["sound"]["verdict"] \
            == "unsound"
        pds = drive.post({"kind": "set_causal", "exposure": "fiber", "method": "pds_lasso",
                          "learner": "linear", "repetitions": 3, "assumptions": ALL4})
        assert pds.status_code == 409
        error = pds.json()["error"]
        assert error["code"] == "survey_pds"
        weighted, sample = error["exits"]
        assert weighted["decision"]["method"] == "dml_plr" and sample["decision"]["sample_only"]
        incomplete = drive.post(weighted["decision"])
        assert incomplete.status_code == 409
        error = incomplete.json()["error"]
        assert error["code"] == "incomplete_rows" and error["message"] == design["missing"]["reason"]
        drive.decide(error["exits"][0]["decision"])
        drive.decide({"kind": "select_models", "models": ["linear"]})
        art = _causal_artifact(drive, lambda a: bool(a.get("estimates")))
        # The other exit: post-double selection for these participants only, recorded as
        # unweighted (its complete rows too, under the multiple-imputation answer).
        drive.decide({**sample["decision"], "complete_rows": True})
        unweighted = _causal_artifact(drive, lambda a: a.get("method") == "pds_lasso"
                                      and bool(a.get("estimates")))
        sample_record = [r for r in drive.view()["decisions"]
                         if r["decision"]["kind"] == "set_causal"][-1]["sentence"]

    complete = raw[["age", "income", "fiber", "sbp"]].notna().all(axis=1).to_numpy()
    assert design["missing"]["n_complete"] == int(complete.sum()) == art["n"]
    rows = raw.loc[complete]
    n = len(rows)
    groups = pd.factorize(pd.MultiIndex.from_arrays([
        pd.factorize(raw["SDMVSTRA"], sort=True)[0], raw["SDMVPSU"]]))[0][complete]
    splits = est.sample_splits(n, 5, 3, seed=0, groups=groups)
    folds = folds_frame(splits, n)
    data = raw.copy()
    for r in range(3):
        data[f"r{r + 1}"] = 0
        data.loc[complete, f"r{r + 1}"] = folds[f"r{r + 1}"].to_numpy()
    data["complete"] = complete.astype(int)
    ref = run_r(tmp_path / "r", """
suppressMessages({library(survey); library(sensemakr)})
df <- as.data.frame(fread(data_csv)); keep <- df$complete == 1
out <- list()
for (r in 1:3) {
  f <- df[[paste0("r", r)]]; u <- v <- rep(0, nrow(df))
  for (k in 1:5) { tr <- keep & f != k; te <- keep & f == k
    u[te] <- df$sbp[te] - predict(lm(sbp ~ age + income, data = df[tr, ], weights = WTMEC2YR),
                                  newdata = df[te, ])
    v[te] <- df$fiber[te] - predict(lm(fiber ~ age + income, data = df[tr, ], weights = WTMEC2YR),
                                    newdata = df[te, ]) }
  df$u <- u; df$v <- v
  des <- svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~WTMEC2YR, nest = TRUE, data = df)
  fit <- svyglm(u ~ 0 + v, design = subset(des, keep))
  last <- lm(u ~ 0 + v, data = df[keep, ], weights = WTMEC2YR)  # the final least-squares step
  out[[r]] <- c(coef(fit)[[1]], SE(fit)[[1]], robustness_value(last, "v", q = 1, alpha = 1),
                robustness_value(last, "v", q = 1, alpha = 0.05)) }
cat(toJSON(list(coef = sapply(out, `[`, 1), se = sapply(out, `[`, 2),
  rv = median(sapply(out, `[`, 3)), rva = median(sapply(out, `[`, 4))), digits = NA,
  auto_unbox = TRUE))
""", {"data": data})
    coefs, ses = np.asarray(ref["coef"]), np.asarray(ref["se"])
    theta = float(np.median(coefs))  # DoubleML's aggregation, written out
    se = math.sqrt(float(np.median(n * ses ** 2 + (coefs - theta) ** 2)) / n)
    [row] = art["estimates"]
    assert row["estimate"] == pytest.approx(theta, abs=1e-6)
    assert row["se"] == pytest.approx(se, abs=1e-6)
    assert row["df"] == 30 - 15  # the domain's PSUs minus its strata
    assert art["methods"] == (
        "The total effect of `fiber` on `sbp` was also estimated by double/debiased machine "
        "learning in the partially linear model (Chernozhukov et al. 2018, Econom J 21:C1; as R's "
        "DoubleML 1.0.2 computes it), as the difference in the mean outcome per unit of `fiber`: "
        "main-terms linear and logistic regression predicted the outcome and the exposure from "
        "`age` and `income`, each fit on the other 4 of 5 folds, and the outcome's residual was "
        f"regressed on the exposure's; the estimate is the median over 3 random sample splits, on "
        f"{n:,} complete rows. The survey weight `WTMEC2YR` entered every nuisance fit and the "
        f"estimating equation, the folds kept each PSU's rows together, and the variance is "
        f"linearized over the design's strata and PSUs." + CLOSING + PLR_RV_AND_E_VALUE)
    robust = art["sensitivity"]["robustness"]
    assert (robust["rv"], robust["rv_alpha"]) == pytest.approx((ref["rv"], ref["rva"]), abs=1e-8)

    # Sample-only post-double selection: by hand at the same penalty on the complete rows,
    # unweighted; the methods sentence says it is unweighted and for these participants only.
    X = rows[["age", "income"]].to_numpy(dtype=float)
    y, d = rows["sbp"].to_numpy(dtype=float), rows["fiber"].to_numpy(dtype=float)
    on_y, _ = _rlasso_by_hand(X, y)
    on_d, _ = _rlasso_by_hand(X, d)
    union = sorted(set(on_y) | set(on_d))
    theta, se = _hc3_by_hand(np.column_stack([np.ones(n), d, X[:, union]]), y)
    [row] = unweighted["estimates"]
    assert (row["estimate"], row["se"]) == pytest.approx((theta, se), abs=1e-8)

    def listed(idx: list[int]) -> str:
        return ticked([["age", "income"][i] for i in sorted(idx)]) if idx else "none"

    assert unweighted["methods"] == (
        "The total effect of `fiber` on `sbp` was also estimated by post-double-selection lasso "
        "(Belloni, Chernozhukov & Hansen 2014, Rev Econ Stud 81:608), as the difference in the "
        "mean outcome per unit of `fiber`: plug-in lassos over the 2 declared model terms of `age` "
        f"and `income` selected {listed(on_y)} for the outcome and {listed(on_d)} for the "
        f"exposure, and the outcome was regressed on `fiber` and "
        f"{listed(union) if union else 'no covariate'}, with HC3 standard errors, on {n:,} "
        f"complete rows. As recorded, the survey weights were not used: the plug-in lasso's "
        f"penalty assumes unweighted rows, so the estimate is unweighted and describes these "
        f"participants only, not the surveyed population." + CLOSING + RV_AND_E_VALUE)
    assert sample_record.endswith("; unweighted, for these participants only, as recorded.")


# ── 8 · a yes/no outcome, through the real server ────────────────────────────


def _start(client: Any, path: Path, truth: Any, target: str, *, event: str | None = None) -> Any:
    from turbotab.core.tests.acceptance.server_drive import open_project

    drive = open_project(client, path, truth)
    drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": target})
    if event is not None:
        drive.answer("event", {"kind": "set_event", "column": target, "level": event})
    drive.reach("purpose")
    drive.decide({"kind": "set_purpose", "purpose": "inference"})
    return drive


def _plan(drive: Any, roles: dict[str, str], exposure: str,
          clusters: str | None = None) -> None:
    drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
    drive.reach("roles")
    drive.decide_roles(roles)
    drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
    drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
    drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
    if clusters:
        drive.answer("clusters", {"kind": "set_clusters", "column": clusters,
                                  "adjust": "cluster_only"})
    drive.exposure = exposure
    drive.answer_plan(exposure)


def _causal_records(drive: Any) -> list[dict[str, Any]]:
    return [r for r in drive.view()["decisions"] if r["decision"]["kind"] == "set_causal"]


def outcome_cohort(path: Path, n: int = 1500, seed: int = 17) -> pd.DataFrame:
    """A yes/no outcome (``event``) with a yes/no exposure (``statin``, good overlap), a numeric
    one (``fiber``) and three confounders."""
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 10, n).round(1)
    smoker = (rng.uniform(size=n) < 0.3).astype(int)
    income = rng.normal(5, 2, n).round(2)
    statin = (rng.uniform(size=n) < _expit(-0.5 + 0.05 * (age - 50) + 0.4 * smoker
                                           - 0.1 * (income - 5))).astype(int)
    fiber = (15 + 0.6 * (income - 5) - 2 * smoker + 0.05 * (age - 50)
             + rng.normal(0, 4, n)).round(1)
    event = (rng.uniform(size=n) < _expit(-1.2 + 0.04 * (age - 50) + 0.5 * smoker
                                          - 0.1 * (income - 5) - 0.5 * statin
                                          - 0.05 * (fiber - 15))).astype(int)
    frame = pd.DataFrame({"person_id": np.arange(1, n + 1), "age": age, "smoker": smoker,
                          "income": income, "statin": statin, "fiber": fiber, "event": event})
    frame.to_csv(path, index=False)
    return frame


def zero_event_cohort(path: Path, n: int = 1200, seed: int = 23) -> pd.DataFrame:
    """A yes/no outcome with no event among the exposed (``trt``)."""
    rng = np.random.default_rng(seed)
    age = rng.normal(55, 10, n).round(1)
    smoker = (rng.uniform(size=n) < 0.3).astype(int)
    trt = (rng.uniform(size=n) < _expit(-1.0 + 0.02 * (age - 55))).astype(int)
    event = (rng.uniform(size=n) < _expit(-2.5 + 0.04 * (age - 55) + 0.6 * smoker)).astype(int)
    event[trt == 1] = 0
    frame = pd.DataFrame({"pid": np.arange(1, n + 1), "age": age, "smoker": smoker, "trt": trt,
                          "event": event})
    frame.to_csv(path, index=False)
    return frame


def test_8_a_yes_no_outcome_carries_its_required_sensitivity_and_refuses_undefined_ratios(tmp_path):
    """The causal lane on a yes/no outcome (ruling 10, sensitivity required), through the real
    server:

    * the interactive model's average effect and its effect among the exposed each carry a risk
      ratio from the same doubly robust means, and its E-value: references, the same computation
      by hand with statsmodels' logistic fits on the same folds (1e-6) and VanderWeele & Ding's
      formula (1e-8); the methods sentences verbatim, the effect among the exposed predicting the
      outcome among the unexposed only;
    * the partially linear model on the same outcome carries the robustness value of its final
      least-squares step (by hand, 1e-8), and says why it has no E-value;
    * with no event among the exposed, TMLE reports the risk difference alone: no risk or odds
      ratio, no E-value, and the methods sentence says why, verbatim."""
    from turbotab.core.tests.acceptance.server_drive import local_server
    from turbotab.core.tests.truths import Truth

    path = tmp_path / "outcome.csv"
    outcome_cohort(path)
    truth = Truth({"adjust:age": "yes,yes,no", "adjust:smoker": "yes,yes,no",
                   "adjust:income": "yes,yes,no", "adjust:fiber": "no,yes,no",
                   "adjust:statin": "no,yes,no", "code_or_count:smoker": "code",
                   "code_or_count:statin": "code", "code_or_count:event": "code",
                   "cluster:person_id": "no", "measure:statin": "risk_difference",
                   "measure:fiber": "risk_difference"}, fixture="the yes/no-outcome cohort")
    roles = {"person_id": "identifier", "age": "covariate", "smoker": "covariate",
             "income": "covariate", "statin": "exposure", "fiber": "exposure"}
    zero_path = tmp_path / "zero.csv"
    zero_event_cohort(zero_path)
    zero_truth = Truth({"adjust:age": "yes,yes,no", "adjust:smoker": "yes,yes,no",
                        "code_or_count:smoker": "code", "code_or_count:trt": "code",
                        "code_or_count:event": "code", "cluster:pid": "no",
                        "measure:trt": "risk_difference"}, fixture="the zero-event cohort")
    with local_server(tmp_path / "home") as client:
        drive = _start(client, path, truth, "event", event="1")
        _plan(drive, roles, "statin")
        lane = {"kind": "set_causal", "exposure": "statin", "method": "dml_irm",
                "learner": "linear", "repetitions": 2, "assumptions": ALL4}
        drive.decide(lane)
        drive.decide({"kind": "select_models", "models": ["linear"]})
        ate = _causal_artifact(drive, lambda a: a.get("population") == "all"
                               and bool(a.get("estimates")))
        drive.decide({**lane, "population": "exposed"})
        att = _causal_artifact(drive, lambda a: a.get("population") == "exposed"
                               and bool(a.get("estimates")))
        drive.decide({"kind": "set_estimand", "exposure": "fiber", "effect": "total",
                      "measure": "risk_difference"})
        drive.answer_plan("fiber")
        drive.decide({**lane, "exposure": "fiber", "method": "dml_plr"})
        plr = _causal_artifact(drive, lambda a: a.get("method") == "dml_plr"
                               and bool(a.get("estimates")))

        zero = _start(client, zero_path, zero_truth, "event", event="1")
        _plan(zero, {"pid": "identifier", "age": "covariate", "smoker": "covariate",
                     "trt": "exposure"}, "trt")
        zero.decide({"kind": "set_causal", "exposure": "trt", "method": "tmle", "learner": "linear",
                     "assumptions": ALL4})
        zero.decide({"kind": "select_models", "models": ["linear"]})
        none = _causal_artifact(zero, lambda a: bool(a.get("estimates")))

    raw = pd.read_csv(path)
    n = len(raw)
    adjusted = ["age", "smoker", "income", "fiber"]
    y = raw["event"].to_numpy(dtype=float)
    a = raw["statin"].to_numpy(dtype=float)
    X = raw[adjusted].to_numpy(dtype=float)
    b = 5 / math.sqrt(n) / math.log(n)
    splits = est.sample_splits(n, 5, 2, seed=0)
    for art, score, whom, label, words, predicted in (
            (ate, "ATE", "the average effect over everyone", "Marginal risk ratio",
             "the marginal risk ratio", "the outcome at each exposure level"),
            (att, "ATT", "the effect among the exposed (`statin` = `1`)", "Risk ratio among the "
             "exposed", "the risk ratio among the exposed", "the outcome among the unexposed")):
        hand = irm_risk_ratio_by_hand(y, a, X, splits, b, score)
        diff, ratio = art["estimates"]
        assert diff["measure"] == "risk_difference"
        assert diff["estimate"] == pytest.approx(hand["rd"], abs=1e-6)
        assert ratio["label"] == label and ratio["measure"] == "risk_ratio"
        assert (ratio["estimate"], ratio["se"], ratio["ci_low"], ratio["ci_high"]) == \
            pytest.approx((hand["rr"], hand["se_log"], hand["lo"], hand["hi"]), abs=1e-6)
        found = art["sensitivity"]
        assert found["computed"] is True and found["methods"] == ["e_value"]
        point, limit = e_value_of_ratio(ratio["estimate"], ratio["ci_low"], ratio["ci_high"])
        assert (found["e_value"]["point"], found["e_value"]["limit"]) == pytest.approx(
            (point, limit), rel=1e-8)
        assert f"E-value for {words}: {point:.2f}" in found["reading"]
        assert art["methods"] == (
            "The total effect of `statin` on `event` was also estimated by double/debiased "
            "machine learning in the interactive model (Chernozhukov et al. 2018, Econom J "
            f"21:C1; as R's DoubleML 1.0.2 computes it), as {whom}: the difference in risk "
            f"between `statin` = `1` and the other level; main-terms linear and logistic "
            f"regression predicted {predicted} and the propensity from {ticked(adjusted)}, each "
            f"fit on the other 4 of 5 folds, with propensities bounded to [{b:.4f}, "
            f"{1 - b:.4f}]; the estimate is the median over 2 random sample splits, on {n:,} "
            f"complete rows." + CLOSING + E_VALUE_ONLY)

    # The partially linear model on the yes/no outcome: the robustness value of its final step.
    adjusted = ["age", "smoker", "income", "statin"]
    hand = plr_by_hand(y, raw["fiber"].to_numpy(dtype=float), raw[adjusted].to_numpy(dtype=float),
                       splits)
    [row] = plr["estimates"]
    assert row["measure"] == "risk_difference"
    assert row["estimate"] == pytest.approx(hand["theta"], abs=1e-8)
    found = plr["sensitivity"]
    assert found["computed"] is True and found["methods"] == ["robustness_value"]
    assert (found["robustness"]["rv"], found["robustness"]["rv_alpha"]) == pytest.approx(
        (hand["rv"], hand["rv_alpha"]), abs=1e-8)
    assert found["not_computed"] == ("No E-value: a risk difference with no risk ratio beside it "
                                     "carries no risks to form one from.")
    assert plr["methods"] == (
        "The total effect of `fiber` on `event` was also estimated by double/debiased machine "
        "learning in the partially linear model (Chernozhukov et al. 2018, Econom J 21:C1; as R's "
        "DoubleML 1.0.2 computes it), as the difference in risk per unit of `fiber`: main-terms "
        f"linear and logistic regression predicted the outcome and the exposure from "
        f"{ticked(adjusted)}, each fit on the other 4 of 5 folds, and the outcome's residual was "
        f"regressed on the exposure's; the estimate is the median over 2 random sample splits, on "
        f"{n:,} complete rows." + CLOSING + PLR_RV_ONLY)

    # No event among the exposed: the risk difference alone, and why.
    raw = pd.read_csv(zero_path)
    m = len(raw)
    assert raw.loc[raw["trt"] == 1, "event"].sum() == 0
    [row] = none["estimates"]
    assert row["measure"] == "risk_difference" and row["estimate"] < 0
    reason = ("no exposed row has the event, so the exposed risk is zero and a ratio of the two "
              "risks is 0 or undefined")
    bz = 5 / math.sqrt(m) / math.log(m)
    found = none["sensitivity"]
    assert found["computed"] is False and found["e_value"] is None
    assert none["methods"] == (
        "The total effect of `trt` on `event` was also estimated by targeted maximum likelihood "
        "(van der Laan & Rubin 2006, Int J Biostat 2(1); as R's tmle 2.1.1 computes it), as the "
        "average effect over everyone: the difference in risk between `trt` = `1` and the other "
        "level; main-terms linear and logistic regression modeled the outcome and the propensity "
        f"given `age` and `smoker`, each fit on every analyzed row, with the propensity truncated "
        f"below at {bz:.4f} for each level (Gruber et al. 2022, Am J Epidemiol 191:1640), and the "
        f"interval is read off the influence curve, on {m:,} complete rows. No marginal risk "
        f"ratio or marginal odds ratio is reported: {reason}." + CLOSING
        + " Sensitivity to unmeasured confounding, required in the causal lane, could not be "
        "computed for this estimate: no robustness value: it is defined for one least-squares "
        "coefficient (Cinelli & Hazlett 2020, J R Stat Soc B 82:39), and targeted maximum "
        "likelihood averages its targeted outcome model's predictions, not one coefficient; no "
        f"E-value: {reason}.")


# ── 9 · the estimand's own positivity, a record kept, too few clusters ───────


def att_cohort(path: Path, n: int = 1500, seed: int = 404) -> pd.DataFrame:
    """A yes/no exposure (``trt``) most rows are all but certain not to have, and none all but
    certain to have: the average effect's positivity is violated, the effect among the exposed's
    is not. A numeric exposure (``dose``) the covariates nearly determine."""
    rng = np.random.default_rng(seed)
    age = rng.normal(55, 10, n).round(1)
    bmi = rng.normal(28, 5, n).round(1)
    trt = (rng.uniform(size=n) < _expit(-3.8 + 0.24 * (age - 55)
                                        + 0.08 * (bmi - 28))).astype(int)
    dose = (2 * (age - 55) + 0.5 * (bmi - 28) + rng.normal(0, 4, n)).round(2)
    y = (10 - 3 * trt - 0.2 * dose + 0.2 * (age - 55) + 0.3 * (bmi - 28)
         + rng.normal(0, 4, n)).round(2)
    frame = pd.DataFrame({"pid": np.arange(1, n + 1), "age": age, "bmi": bmi, "trt": trt,
                          "dose": dose, "y": y})
    frame.to_csv(path, index=False)
    return frame


def sites_cohort(path: Path, n_sites: int = 5, n: int = 700, seed: int = 9) -> pd.DataFrame:
    """Rows from a few sites, each with its own shift in the outcome."""
    rng = np.random.default_rng(seed)
    site = np.sort(rng.integers(1, n_sites + 1, n))
    shift = rng.normal(0, 3, n_sites + 1)[site]
    age = rng.normal(50, 12, n).round(1)
    female = (rng.uniform(size=n) < 0.5).astype(int)
    sodium = (3.2 + 0.01 * (age - 50) - 0.3 * female + 0.1 * shift
              + rng.normal(0, 0.8, n)).round(2)
    sbp = (125 + 2.5 * sodium + 0.4 * (age - 50) - 4 * female + shift
           + rng.normal(0, 9, n)).round(1)
    frame = pd.DataFrame({"pid": np.arange(1, n + 1), "site": site, "age": age, "female": female,
                          "sodium": sodium, "sbp": sbp})
    frame.to_csv(path, index=False)
    return frame


def test_9_each_estimand_reads_its_own_positivity_and_every_exit_leads_forward(tmp_path):
    """The leash's block-and-record rows, through the real server:

    * the effect among the exposed reads positivity on its own terms (a propensity above 1 − b):
      on a cohort whose propensities pile up near 0 the average effect is refused with its reason
      and the effect among the exposed is estimated, its Record and methods sentence claiming no
      violation (the card's two readings against an independent cross-fitted logistic regression
      on its folds);
    * a numeric exposure the covariates nearly determine is refused with the record exit only;
      taken, the methods sentence says positivity is practically violated, with the estimator's
      own R² (by hand on its folds), instead of listing positivity among the assumptions;
    * too few clusters withhold the lane with the lane's own ways forward: keep the primary model
      only (taken: the lane recorded as such, nothing withheld) or combine each unit's rows."""
    from turbotab.core.models.inference import min_clusters
    from turbotab.core.tests.acceptance.server_drive import local_server
    from turbotab.core.tests.truths import Truth

    path = tmp_path / "att.csv"
    att_cohort(path)
    truth = Truth({"adjust:age": "yes,yes,no", "adjust:bmi": "yes,yes,no",
                   "adjust:dose": "no,yes,no", "adjust:trt": "no,yes,no",
                   "code_or_count:trt": "code", "cluster:pid": "no"}, fixture="the ATT cohort")
    sites = tmp_path / "sites.csv"
    sites_cohort(sites)
    site_truth = Truth({"adjust:age": "yes,yes,no", "adjust:female": "yes,yes,no",
                        "code_or_count:female": "code", "code_or_count:site": "code",
                        "cluster:pid": "no", "cluster:site": "yes", "role:site": "cluster"},
                       fixture="the five-site cohort")
    with local_server(tmp_path / "home") as client:
        drive = _start(client, path, truth, "y")
        _plan(drive, {"pid": "identifier", "age": "covariate", "bmi": "covariate",
                      "trt": "exposure", "dose": "exposure"}, "trt")
        design = drive.artifact("causal_design")
        assert design["positivity"]["violated"]
        assert design["positivity_att"] == {"violated": False, "reason": None}
        lane = {"kind": "set_causal", "exposure": "trt", "method": "dml_irm", "learner": "linear",
                "repetitions": 2, "assumptions": ALL4}
        refused = drive.post(lane)
        assert refused.status_code == 409
        assert refused.json()["error"]["code"] == "positivity"
        assert refused.json()["error"]["message"] == design["positivity"]["reason"]
        accepted = drive.post({**lane, "population": "exposed"})
        assert accepted.status_code == 200, accepted.text
        drive.decide({"kind": "select_models", "models": ["linear"]})
        att = _causal_artifact(drive, lambda a: bool(a.get("estimates")))
        att_record = _causal_records(drive)[-1]["sentence"]

        drive.decide({"kind": "set_estimand", "exposure": "dose", "effect": "total",
                      "measure": "mean_difference"})
        drive.answer_plan("dose")
        dose_design = drive.artifact("causal_design")
        assert dose_design["exposure"] == "dose" and dose_design["variation"]["violated"]
        refused = drive.post({**lane, "exposure": "dose", "method": "dml_plr"})
        assert refused.status_code == 409
        error = refused.json()["error"]
        assert error["code"] == "positivity"
        [record] = error["exits"]  # a numeric exposure has no propensity to trim on
        assert record["decision"]["acknowledged"]
        drive.decide(record["decision"])
        dose = _causal_artifact(drive, lambda a: a.get("exposure") == "dose"
                                and bool(a.get("estimates")))

        few = _start(client, sites, site_truth, "sbp")
        _plan(few, {"pid": "identifier", "site": "cluster", "age": "covariate",
                    "female": "covariate", "sodium": "exposure"}, "sodium", clusters="site")
        few.decide({"kind": "set_causal", "exposure": "sodium", "method": "dml_plr",
                    "learner": "linear", "assumptions": ALL4})
        few.decide({"kind": "select_models", "models": ["linear"]})
        held = _causal_artifact(few, lambda a: bool(a.get("withheld")))
        keep, combine = held["exits"]
        assert keep["label"] == "Keep the primary model only"
        assert keep["decision"]["method"] == "none" and keep["decision"]["exposure"] == "sodium"
        assert combine == {"label": "Combine each `site`'s rows into one (the unit question, "
                                    "before the seal)", "decision": None}
        few.decide(keep["decision"])
        kept = _causal_artifact(few, lambda a: a.get("method") == "none")
        kept_record = _causal_records(few)[-1]["sentence"]

    raw = pd.read_csv(path)
    n = len(raw)
    b = 5 / math.sqrt(n) / math.log(n)
    # The card's readings: the average effect's and the effect among the exposed's.
    W = raw[["age", "bmi", "dose"]].to_numpy(dtype=float)
    t = raw["trt"].to_numpy(dtype=float)
    g = np.empty(n)
    for train, test in est.sample_splits(n, 5, 1, seed=0)[0]:
        g[test] = logit_by_hand(t[train], W[train], W[test])
    assert design["overlap"]["share_outside"] == pytest.approx(np.mean((g < b) | (g > 1 - b)))
    assert np.mean((g < b) | (g > 1 - b)) >= 0.01 > np.mean(g > 1 - b)
    [row] = att["estimates"]
    assert row["label"] == "Effect among the exposed" and att["population"] == "exposed"
    assert abs(row["estimate"] - (-3.0)) < 3 * row["se"]  # the cohort's constant effect
    assert att["methods"].endswith(CLOSING + E_VALUE_ONLY)
    assert "practically violated" not in att_record and "practically violated" not in att["methods"]

    # The numeric exposure kept on the record: the estimator's own R², by hand on its folds.
    adjusted = ["age", "bmi", "trt"]
    hand = plr_by_hand(raw["y"].to_numpy(dtype=float), raw["dose"].to_numpy(dtype=float),
                       raw[adjusted].to_numpy(dtype=float), est.sample_splits(n, 5, 2, seed=0))
    assert hand["r2"] >= 0.9
    assert dose["variation"]["r2"] == pytest.approx(hand["r2"], abs=1e-9)
    assert dose["estimates"][0]["estimate"] == pytest.approx(hand["theta"], abs=1e-8)
    assert dose["methods"] == (
        "The total effect of `dose` on `y` was also estimated by double/debiased machine learning "
        "in the partially linear model (Chernozhukov et al. 2018, Econom J 21:C1; as R's DoubleML "
        "1.0.2 computes it), as the difference in the mean outcome per unit of `dose`: main-terms "
        f"linear and logistic regression predicted the outcome and the exposure from "
        f"{ticked(adjusted)}, each fit on the other 4 of 5 folds, and the outcome's residual was "
        f"regressed on the exposure's; the estimate is the median over 2 random sample splits, on "
        f"{n:,} complete rows. It rests on the declared assumptions of no unmeasured confounding "
        f"given the adjustment set, consistency and time ordering. Positivity is practically "
        f"violated: the covariates explain {hand['r2']:.0%} of `dose`'s variance; every row was "
        f"kept, as recorded, so the estimate extrapolates where `dose` barely varies given the "
        f"covariates, a stated limitation." + PLR_RV_AND_E_VALUE)

    assert held["withheld"].startswith(f"`site` has 5 units, fewer than the {min_clusters()} "
                                       f"TurboTab requires for cluster-robust intervals")
    assert held["estimates"] == []
    assert kept["withheld"] is None and kept["estimates"] == []
    assert kept_record == ("No causal machine-learning estimate was set beside the primary model "
                           "for the effect of `sodium`.")
