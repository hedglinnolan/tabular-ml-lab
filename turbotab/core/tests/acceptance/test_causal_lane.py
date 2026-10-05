"""The causal lane: DoubleML, TMLE and the post-double-selection lasso on the shortest leash
(V2 definition of done §2, "causal inference"; MODELING_SEQUENCE §0 ruling 1 rung (d), rulings 6
and 10; BLUEPRINT §11.3 and §13).

The package's acceptance items, in its order:

1. **DoubleML.** The partially linear model (a numeric exposure) and the interactive model (a
   yes/no exposure; the ATE and the ATT) with cross-fitting: with deterministic learners and the
   same folds the estimates and standard errors agree with R's DoubleML 1.0.2 to 1e-6; with random
   learners (the cross-validated lasso, here and as R's ``cv_glmnet``) they agree within Monte Carlo
   tolerance over repeated sample splits, each side aggregated by DoubleML's median.
2. **TMLE** for a yes/no exposure and a numeric or yes/no outcome agrees with R's tmle 2.1.1 given
   the same glm Q and g models (``Qform``, ``gform``, ``cvQinit = FALSE``) to 1e-6: the average
   effect and its influence-curve standard error, and for a yes/no outcome the marginal risk and
   odds ratios.
3. **Post-double selection** agrees with a NumPy hand implementation at the same plug-in penalty to
   1e-8, selects the same columns, and is ranked first, and asked, when the candidates are many
   relative to n.
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
   verbatim.

Every expected value comes from outside the engine: R (DoubleML, tmle, survey) run as a subprocess
on a CSV written here, a NumPy coordinate-descent lasso written here, statsmodels, and the
simulations' own truths. The R tests skip cleanly where ``Rscript`` is not installed.
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
    """Belloni et al. 2012's plug-in lasso (Algorithm A.1; hdm's ``rlasso`` defaults: c = 1.1,
    γ = 0.1/ln n, loadings from the post-lasso residuals, 15 fits or a move below 1e-5)."""
    n, p = X.shape
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    lam = 2 * 1.1 * math.sqrt(n) * float(norm.ppf(1 - (0.1 / math.log(n)) / (2 * p)))
    ups = np.sqrt((Xc ** 2 * yc[:, None] ** 2).mean(axis=0))
    support: list[int] = []
    for _ in range(15):
        b = _cd_lasso(Xc, yc, lam * ups)
        support = [int(j) for j in np.flatnonzero(np.abs(b) > 0)]
        if support:
            S = Xc[:, support]
            e = yc - S @ np.linalg.solve(S.T @ S, S.T @ yc)
        else:
            e = yc
        new = np.sqrt((Xc ** 2 * e[:, None] ** 2).mean(axis=0))
        moved = math.sqrt(float(((new - ups) ** 2).sum()))
        ups = new
        if moved < 1e-5:
            break
    return support, lam


def _hc3_by_hand(A: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    bread = np.linalg.inv(A.T @ A)
    beta = bread @ A.T @ y
    e = y - A @ beta
    h = np.einsum("ij,jk,ik->i", A, bread, A)
    V = bread @ (A.T * (e / (1 - h)) ** 2) @ A @ bread
    return float(beta[1]), math.sqrt(float(V[1, 1]))


def test_3_post_double_selection_agrees_with_a_numpy_hand_implementation_and_ranks_high():
    """Reference: the plug-in lasso and the double selection written out above in NumPy (a
    coordinate-descent lasso, not the app's scikit-learn solver), at the same penalty: the same
    selections, and the exposure's coefficient and HC3 standard error to 1e-8. 120 candidates for
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
    found = est.pds_lasso(y, d, X)
    on_y, lam = _rlasso_by_hand(X, y)
    on_d, _ = _rlasso_by_hand(X, d)
    union = sorted(set(on_y) | set(on_d))
    theta, se = _hc3_by_hand(np.column_stack([np.ones(n), d, X[:, union]]), y)
    assert found.extra["lambda"] == pytest.approx(lam, rel=1e-12)
    assert found.extra["selected_outcome"] == on_y and found.extra["selected_exposure"] == on_d
    assert found.extra["selected"] == union
    assert found.estimate == pytest.approx(theta, abs=1e-8)
    assert found.se == pytest.approx(se, abs=1e-8)
    assert abs(found.estimate - 0.5) < 3 * found.se  # covers the simulation's truth
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


# ── 5 · a survey design ──────────────────────────────────────────────────────


@needs_r
def test_5_survey_weights_enter_every_nuisance_fit_and_the_estimating_equation(tmp_path):
    """Reference: R's survey package over the same folds. The partially linear model is
    ``svyglm(ũ ~ 0 + ṽ)`` on residuals from weighted ``lm`` fits per fold; the interactive model is
    ``svymean`` of the doubly robust pseudo-outcome from weighted ``lm`` and ``glm`` fits per fold;
    TMLE is ``tmle(…, obsWeights = w)`` with ``svymean`` of its influence curve. Each design is
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
    splits = est.sample_splits(n, 5, 1, seed=3, groups=psu)
    assert all(set(psu[train]).isdisjoint(psu[test]) for train, test in splits[0])  # whole PSUs
    bound = est.tmle_gbound(n)
    frame = pd.DataFrame(X, columns=["x0", "x1", "x2"])
    frame["d"], frame["y"], frame["a"], frame["ya"] = d, y, a, ya
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
cat(toJSON(list(plr = c(coef(plr)[[1]], SE(plr)[[1]]), irm = c(coef(irm)[[1]], SE(irm)[[1]]),
  tmle = c(t1$estimates$ATE$psi, SE(tm)[[1]]), clusters = c(coef(cl)[[1]], SE(cl)[[1]])),
  digits = NA))
""", {"data": frame})
    design = est.Design(weight=w, stratum=stratum, psu=psu)
    plr = est.dml_plr(y, d, X, learner="linear", splits=splits, design=design)
    assert [plr.estimate, plr.se] == pytest.approx(ref["plr"], abs=1e-6)
    irm = est.dml_irm(ya, a, X, learner="linear", splits=splits, design=design, bound=bound)
    assert [irm.estimate, irm.se] == pytest.approx(ref["irm"], abs=1e-6)
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
# LEASH (MODELING_SEQUENCE §0 ruling 14): under the surveyed-population answer the difference is
# standardized by the population's design-weighted SD, and the clause says so.
E_VALUE_POPULATION = (" Sensitivity to unmeasured confounding is reported by the E-value for the "
                      "estimate and for the confidence limit nearer the null, the difference "
                      "standardized by the outcome's design-weighted standard deviation in the "
                      "surveyed population, never as a pass or a fail.")
RV_AND_E_VALUE = (" Sensitivity to unmeasured confounding is reported by the Cinelli–Hazlett "
                  "robustness value (each selected covariate a named benchmark) and by the E-value "
                  "for the estimate and for the confidence limit nearer the null, never as a pass "
                  "or a fail.")


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


def robustness_by_hand(A: np.ndarray, y: np.ndarray, j: int) -> tuple[float, float]:
    """Cinelli & Hazlett 2020 for column ``j`` of the least-squares design ``A``: the classical t,
    f = |t|/√dof, RV = ½(√(f⁴ + 4f²) − f²), and RV at α = 0.05 with f − f*, f* = |t*_{dof−1}|/√(dof − 1)."""
    from scipy import stats

    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ beta
    dof = len(y) - A.shape[1]
    se = math.sqrt(float(e @ e) / dof * np.linalg.inv(A.T @ A)[j, j])
    f = abs(beta[j] / se) / math.sqrt(dof)
    fa = f - abs(stats.t.ppf(0.025, dof - 1)) / math.sqrt(dof - 1)
    return (0.5 * (math.sqrt(f ** 4 + 4 * f ** 2) - f ** 2),
            0.5 * (math.sqrt(fa ** 4 + 4 * fa ** 2) - fa ** 2) if fa > 0 else 0.0)


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
    # The trim, by an independent fit of the estimator's propensity model on every row.
    raw = pd.read_csv(tmp_path / "cohort.csv")
    W = raw[ADJUSTED_FOR_HEAVY].to_numpy(dtype=float)
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
        assert first["methods"].endswith(CLOSING + E_VALUE_ONLY)

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
    y = raw["sbp"].to_numpy(dtype=float)
    d = raw["fiber"].to_numpy(dtype=float)
    X = raw[ADJUSTED_FOR_FIBER].to_numpy(dtype=float)
    # The partially linear model, by hand on the same folds (the folds are its input).
    thetas = []
    for rep in est.sample_splits(n, 5, 3, seed=0):
        u, v = np.empty(n), np.empty(n)
        for train, test in rep:
            A = np.column_stack([np.ones(len(train)), X[train]])
            At = np.column_stack([np.ones(len(test)), X[test]])
            u[test] = y[test] - At @ np.linalg.lstsq(A, y[train], rcond=None)[0]
            v[test] = d[test] - At @ np.linalg.lstsq(A, d[train], rcond=None)[0]
        thetas.append(float(v @ u / (v @ v)))
    [row] = plr["estimates"]
    assert row["estimate"] == pytest.approx(float(np.median(thetas)), abs=1e-8)
    assert abs(row["estimate"] - (-0.58)) < 3 * row["se"]  # the cohort's truth
    assert plr["methods"] == (
        "The total effect of `fiber` on `sbp` was also estimated by double/debiased machine "
        "learning in the partially linear model (Chernozhukov et al. 2018, Econom J 21:C1; as R's "
        "DoubleML 1.0.2 computes it), as the difference in the mean outcome per unit of `fiber`: "
        f"main-terms linear and logistic regression predicted the outcome and the exposure from "
        f"{ticked(ADJUSTED_FOR_FIBER)}, each fit on the other 4 of 5 folds, and the outcome's "
        f"residual was regressed on the exposure's; the estimate is the median over 3 random "
        f"sample splits, on {n:,} complete rows." + CLOSING + E_VALUE_ONLY)
    point, limit = e_value_by_hand(row["estimate"], row["se"], float(np.std(y, ddof=1)))
    assert (plr["sensitivity"]["e_value"]["point"], plr["sensitivity"]["e_value"]["limit"]) == \
        pytest.approx((point, limit), rel=1e-10)
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
suppressMessages(library(survey))
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
  out[[r]] <- c(coef(fit)[[1]], SE(fit)[[1]]) }
cat(toJSON(list(coef = sapply(out, `[`, 1), se = sapply(out, `[`, 2)), digits = NA))
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
        f"linearized over the design's strata and PSUs." + CLOSING + E_VALUE_POPULATION)
