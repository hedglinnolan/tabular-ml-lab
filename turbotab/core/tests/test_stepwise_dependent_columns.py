"""Backward elimination by AIC on a design whose columns are dependent: a category coded by every
level beside the intercept, as a tree family's pipeline codes it (the NHANES glucose sweep's
``set_selection:stepwise`` under Predict, 2026-10-26, ended the fit with ``LinAlgError: SVD did not
converge in Linear Least Squares``).

The cause is in the selection's least squares, not in a NaN or inf: every value of the failing fold
(7,418 rows, 26 categories coded by every level, 108 columns, rank 83 with the intercept) is finite,
and numpy's ``lstsq`` (LAPACK's divide-and-conquer SVD) fails to converge on one of its submatrices,
on every call. Where it converged the AIC was wrong for such a design too: it counted every column
as a parameter, one per category more than the rows identify, and the logistic fit had no maximum
on the dependent columns, so every term was kept. As R's ``lm`` and ``glm`` fit (``extractAIC``
for ``MASS::stepAIC``), the fit is now on the identified columns (a QR factorization with column
pivoting, an aliased column's coefficient NA there) and edf is their number, the rank.

Truth by derivation: a category coded by every level and the same category coded by all levels but
the first span the same columns with the intercept, so the residuals, the likelihood's maximum and
the rank, and hence every AIC and the kept terms, are the same. The AIC's value is statsmodels'
(``OLS``'s residual sum of squares, ``Logit``'s log-likelihood) on the reference coding, with edf
its number of coefficients. With R, ``MASS::stepAIC`` on the categories as factors keeps the terms
the every-level coding keeps.

The last section holds the inference sensitivity analysis (backward elimination by Wald tests) on
dependent and nearly dependent columns, as R's ``summary.lm``, ``summary.glm`` and
``mice::pool`` read them.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import scipy.linalg
import statsmodels.api as sm

from turbotab.core.models import variable_selection as VS
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r

N = 300


def _rows(seed: int, binary: bool) -> tuple[pd.DataFrame, np.ndarray]:
    """Two numeric columns (the first with an effect), a 3-level category with a small effect and a
    4-level one with none."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N, 2))
    level = rng.integers(0, 3, N)
    other = rng.integers(0, 4, N)
    lp = 0.8 * x[:, 0] + np.array([0.0, 0.25, -0.2])[level]
    y = ((rng.random(N) < 1 / (1 + np.exp(-lp))).astype(float) if binary
         else lp + rng.normal(size=N))
    return pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "level": level, "other": other}), y


TERMS = ["x1", "x2", "level", "other"]


def _coded(frame: pd.DataFrame, every: bool) -> tuple[np.ndarray, list[list[int]]]:
    """The standardized columns and each term's positions in them: each category by every level, or
    by all levels but the first."""
    parts, groups, at = [], [], 0
    for term in TERMS:
        if term in ("level", "other"):
            block = pd.get_dummies(frame[term], dtype=float).to_numpy()
            block = block if every else block[:, 1:]
        else:
            block = frame[[term]].to_numpy(float)
        parts.append(block)
        groups.append(list(range(at, at + block.shape[1])))
        at += block.shape[1]
    Z, usable = VS.standardized(pd.DataFrame(np.hstack(parts)))
    assert usable.all()
    return Z, groups


# On the base, the every-level coding kept other terms than the reference coding for these: the
# regression's extra column per category made dropping it cheaper; the logistic fit had no maximum
# on the dependent columns, so nothing was dropped.
SEEDS = {"regression": (2, 5, 11), "binary": (0, 1, 2)}
CASES = [(task, seed) for task, seeds in SEEDS.items() for seed in seeds]


@pytest.mark.parametrize(("task", "seed"), CASES)
def test_a_category_coded_by_every_level_is_selected_as_one_coded_by_all_but_one(task, seed):
    """Each term kept or dropped alike under either coding."""
    frame, y = _rows(seed, task == "binary")
    Ze, ge = _coded(frame, True)
    Zr, gr = _coded(frame, False)
    assert VS.backward_aic(Ze, y, task, ge).tolist() == VS.backward_aic(Zr, y, task, gr).tolist()


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_the_aic_of_a_dependent_coding_is_the_reference_codings(task):
    """The full model's AIC under each coding, against statsmodels on the reference coding:
    n·ln(RSS/n) + 2·edf, or −2·log-likelihood + 2·edf, edf the reference coding's coefficients
    (the intercept, two numeric columns, 2 + 3 indicators: 8), the every-level coding's rank."""
    frame, y = _rows(SEEDS[task][0], task == "binary")
    Ze, _ = _coded(frame, True)
    Zr, _ = _coded(frame, False)
    ones = np.ones((N, 1))
    X = np.hstack([ones, Zr])
    assert X.shape[1] == 8
    if task == "binary":
        expected = -2.0 * sm.Logit(y, X).fit(disp=0).llf + 2.0 * 8
    else:
        expected = N * math.log(float(sm.OLS(y, X).fit().ssr) / N) + 2.0 * 8
    assert VS.aic(np.hstack([ones, Ze]), y, task) == pytest.approx(expected, rel=1e-9)
    assert VS.aic(X, y, task) == pytest.approx(expected, rel=1e-9)


def test_the_least_squares_does_not_rest_on_numpys_lstsq(monkeypatch):
    """Where LAPACK's divide-and-conquer SVD fails to converge on a dependent design (as on the
    NHANES fold, whose rows are not the repository's to ship), the selection still completes: here
    numpy's ``lstsq`` is made to fail as it failed there, and the kept terms are those of the
    reference coding. On the base the selection raised the ``LinAlgError``."""
    frame, y = _rows(SEEDS["regression"][0], False)
    Zr, gr = _coded(frame, False)
    reference = VS.backward_aic(Zr, y, "regression", gr)

    def no_convergence(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge in Linear Least Squares")

    monkeypatch.setattr(np.linalg, "lstsq", no_convergence)
    Ze, ge = _coded(frame, True)
    assert VS.backward_aic(Ze, y, "regression", ge).tolist() == reference.tolist()


def test_independent_columns_are_all_identified():
    """The rule's silent side: with no dependent column every column is identified, and least
    squares is statsmodels' (its estimates and residual sum of squares)."""
    frame, y = _rows(4, False)
    Zr, _ = _coded(frame, False)
    X = np.hstack([np.ones((N, 1)), Zr])
    assert VS.identified(X).tolist() == list(range(8))
    beta, rss, _, rank = VS._ols(X, y)
    ols = sm.OLS(y, X).fit()
    np.testing.assert_allclose(beta, ols.params, rtol=1e-9)
    assert rss == pytest.approx(float(ols.ssr), rel=1e-12) and rank == 8
    Ze, _ = _coded(frame, True)
    assert len(VS.identified(np.hstack([np.ones((N, 1)), Ze]))) == 8


@needs_r
@pytest.mark.parametrize("task", ["regression", "binary"])
def test_the_every_level_coding_keeps_what_mass_stepaic_keeps(task, tmp_path):
    """``MASS::stepAIC(direction = "backward")`` on the categories as factors (R's own coding, all
    levels but the first) keeps the terms the every-level coding keeps here."""
    seed = SEEDS[task][0]
    frame, y = _rows(seed, task == "binary")
    family = "binomial" if task == "binary" else "gaussian"
    r = run_r(f"""
        suppressPackageStartupMessages(library(MASS))
        d <- read.csv(s_csv)
        d$level <- factor(d$level); d$other <- factor(d$other)
        full <- glm(y ~ x1 + x2 + level + other, family = {family}, data = d)
        kept <- attr(terms(stepAIC(full, direction = "backward", trace = 0)), "term.labels")
        out(list(kept = kept))
    """, {"s": frame.assign(y=y)}, tmp_path)
    Ze, ge = _coded(frame, True)
    kept = VS.backward_aic(Ze, y, task, ge)
    assert sorted(t for t, k in zip(TERMS, kept) if k) == sorted(r["kept"])


# ── no SVD on the stepwise path ───────────────────────────────────────────────
# The NHANES glucose sweep met "SVD did not converge" again (2026-10-31), past the fix above: on a
# 7,458-row fold (108 columns, rank 83 with the intercept), after 30 drops, numpy's pinv of XᵀX for
# a trial of 59 columns of rank 44, every time. The AIC never read that (XᵀX)⁻¹; it is now the
# identified columns' R⁻¹R⁻ᵀ from the QR factor, so no SVD is taken anywhere on the path. Here every
# SVD numpy and scipy offer is made to fail as LAPACK's failed there, after each truth is computed.

SVD_SITES = ((np.linalg, "svd"), (np.linalg, "pinv"), (np.linalg, "lstsq"),
             (scipy.linalg, "svd"), (scipy.linalg, "pinv"), (scipy.linalg, "lstsq"))


def _no_svd(monkeypatch) -> None:
    def no_convergence(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    for module, name in SVD_SITES:
        monkeypatch.setattr(module, name, no_convergence)


def _reference(frame: pd.DataFrame, terms: list[str]) -> np.ndarray:
    """The intercept and ``terms`` in R's default coding: each category by all levels but the
    first."""
    parts = [np.ones((len(frame), 1))]
    for term in terms:
        block = (pd.get_dummies(frame[term], dtype=float).to_numpy()[:, 1:]
                 if term in ("level", "other") else frame[[term]].to_numpy(float))
        parts.append(block)
    return np.hstack(parts)


def _statsmodels_aic(frame: pd.DataFrame, y: np.ndarray, terms: list[str], task: str) -> float:
    X = _reference(frame, terms)
    if task == "binary":
        return -2.0 * sm.Logit(y, X).fit(disp=0).llf + 2.0 * X.shape[1]
    return N * math.log(float(sm.OLS(y, X).fit().ssr) / N) + 2.0 * X.shape[1]


def _statsmodels_backward(frame: pd.DataFrame, y: np.ndarray, task: str) -> tuple[list[str], float]:
    """``MASS::stepAIC(direction = "backward")``'s rule, each AIC statsmodels' on the reference
    coding: drop the term whose removal lowers AIC most, while that lowers it by more than 10⁻⁷.
    (The kept terms, their model's AIC.)"""
    kept = list(TERMS)
    current = _statsmodels_aic(frame, y, kept, task)
    while kept:
        trials = {t: _statsmodels_aic(frame, y, [k for k in kept if k != t], task) for t in kept}
        term = min(trials, key=trials.get)
        if trials[term] >= current - 1e-7:
            break
        kept.remove(term)
        current = trials[term]
    return kept, current


def _close(actual, expected) -> None:
    """Equal to 10⁻¹⁰ of the expected values' scale."""
    expected = np.asarray(expected, dtype=float)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10 * np.abs(expected).max())


@pytest.mark.parametrize(("task", "seed"), CASES)
def test_backward_elimination_takes_no_svd(task, seed, monkeypatch):
    """With every SVD failing, backward elimination on the every-level coding completes and keeps
    the terms statsmodels' AICs keep under ``stepAIC``'s rule, with their AIC. On the base the
    regression's raised the ``LinAlgError`` (numpy's pinv in the least squares); the logistic fit
    took no SVD there either."""
    frame, y = _rows(seed, task == "binary")
    expected_terms, expected_aic = _statsmodels_backward(frame, y, task)
    Ze, ge = _coded(frame, True)
    _no_svd(monkeypatch)
    kept = VS.backward_aic(Ze, y, task, ge)
    assert [t for t, k in zip(TERMS, kept) if k] == expected_terms
    columns = [c for g, k in zip(ge, kept) if k for c in g]
    final = VS.aic(np.hstack([np.ones((N, 1)), Ze[:, columns]]), y, task)
    assert final == pytest.approx(expected_aic, rel=1e-9)


@needs_r
@pytest.mark.parametrize("seed", SEEDS["regression"])
def test_backward_elimination_without_svd_is_mass_stepaics(seed, tmp_path, monkeypatch):
    """``MASS::stepAIC(direction = "backward")`` on ``lm`` with the categories as factors: its kept
    terms, and its final model's ``extractAIC``, are the every-level coding's with every SVD
    failing."""
    frame, y = _rows(seed, False)
    r = run_r("""
        suppressPackageStartupMessages(library(MASS))
        d <- read.csv(s_csv)
        d$level <- factor(d$level); d$other <- factor(d$other)
        fit <- stepAIC(lm(y ~ x1 + x2 + level + other, data = d), direction = "backward",
                       trace = 0)
        out(list(kept = attr(terms(fit), "term.labels"), aic = extractAIC(fit)[2]))
    """, {"s": frame.assign(y=y)}, tmp_path)
    Ze, ge = _coded(frame, True)
    _no_svd(monkeypatch)
    kept = VS.backward_aic(Ze, y, "regression", ge)
    assert sorted(t for t, k in zip(TERMS, kept) if k) == sorted(r["kept"])
    columns = [c for g, k in zip(ge, kept) if k for c in g]
    final = VS.aic(np.hstack([np.ones((N, 1)), Ze[:, columns]]), y, "regression")
    assert final == pytest.approx(float(r["aic"]), rel=1e-9)


@pytest.mark.parametrize("svd", ["available", "failing"])
def test_least_squares_on_independent_columns_is_statsmodels(svd, monkeypatch):
    """The rule's silent side: on a full-rank design (the reference coding with the intercept)
    every value the least squares' callers read is statsmodels' ``OLS``'s to 10⁻¹⁰: the
    coefficients, the residual sum of squares, (XᵀX)⁻¹ (``normalized_cov_params``, whose slope
    entry the univariable t test reads) and the rank; and each univariable p-value, a column's t
    test and a term's F test. The same with every SVD failing, which on the base raised."""
    frame, y = _rows(4, False)
    Zr, gr = _coded(frame, False)
    ones = np.ones((N, 1))
    X = np.hstack([ones, Zr])
    ols = sm.OLS(y, X).fit()
    singles = [float(sm.OLS(y, np.hstack([ones, Zr[:, [j]]])).fit().pvalues[1])
               for j in range(Zr.shape[1])]
    terms = [float(sm.OLS(y, np.hstack([ones, Zr[:, g]])).fit().f_pvalue) for g in gr]
    if svd == "failing":
        _no_svd(monkeypatch)
    beta, rss, inv, rank = VS._ols(X, y)
    _close(beta, ols.params)
    _close(rss, ols.ssr)
    _close(inv, ols.normalized_cov_params)
    assert rank == 8
    np.testing.assert_allclose(VS.univariable_p(Zr, y, "regression"), singles, rtol=1e-10)
    np.testing.assert_allclose(VS.univariable_p(Zr, y, "regression", gr), terms, rtol=1e-10)


def _spanned(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The intercept, two numeric columns and, last, a column they span, x₃ = (x₁ − x₂)/10."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N, 2))
    y = 1.0 + x @ np.array([0.5, -0.3]) + rng.normal(size=N)
    return np.column_stack([np.ones(N), x, (x[:, 0] - x[:, 1]) / 10.0]), y


@pytest.mark.parametrize("svd", ["available", "failing"])
def test_least_squares_on_a_spanned_column_reads_the_identified_columns(svd, monkeypatch):
    """The rule's firing side, with SVDs available or failing: where the last column is spanned by
    the others, the coefficients and (XᵀX)⁻¹ on the identified columns are statsmodels' on the
    design without it, the residual sum of squares its; the aliased column's coefficient is zero
    (the residuals' contract; NA in R) and its row and column of (XᵀX)⁻¹ NaN (``vcov`` in R reports
    NA): the rows identify no variance for it. On the base (XᵀX)⁻¹ was the pseudo-inverse of the
    whole, another matrix on the identified columns, and finite where nothing is identified."""
    X, y = _spanned(7)
    reduced = sm.OLS(y, X[:, :3]).fit()
    if svd == "failing":
        _no_svd(monkeypatch)
    beta, rss, inv, rank = VS._ols(X, y)
    assert rank == 3 and VS.identified(X).tolist() == [0, 1, 2]
    _close(beta[:3], reduced.params)
    assert beta[3] == 0.0
    _close(rss, reduced.ssr)
    _close(inv[:3, :3], reduced.normalized_cov_params)
    assert np.isnan(inv[3]).all() and np.isnan(inv[:, 3]).all()


@pytest.mark.parametrize("svd", ["available", "failing"])
def test_least_squares_on_a_category_coded_by_every_level_reads_the_identified_columns(
        svd, monkeypatch):
    """The same for each category coded by every level beside the intercept, as the elastic net's
    pipeline codes it: 10 columns of rank 8 (the intercept, two numeric columns, 2 + 3 levels), one
    level of each category aliased; on the identified columns (XᵀX)⁻¹ is statsmodels' on those
    columns, and NaN in the aliased ones' rows and columns."""
    frame, y = _rows(4, False)
    Ze, ge = _coded(frame, True)
    X = np.hstack([np.ones((N, 1)), Ze])
    kept = VS.identified(X)
    aliased = np.setdiff1d(np.arange(X.shape[1]), kept)
    assert len(kept) == 8
    # one level of each category (``ge`` holds positions in Ze, one before X's)
    assert sorted(len(set((aliased - 1).tolist()) & set(g)) for g in ge) == [0, 0, 1, 1]
    reduced = sm.OLS(y, X[:, kept]).fit()
    if svd == "failing":
        _no_svd(monkeypatch)
    beta, rss, inv, rank = VS._ols(X, y)
    assert rank == 8
    _close(beta[kept], reduced.params)
    assert (beta[aliased] == 0.0).all()
    _close(rss, reduced.ssr)
    _close(inv[np.ix_(kept, kept)], reduced.normalized_cov_params)
    assert np.isnan(inv[aliased]).all() and np.isnan(inv[:, aliased]).all()


@needs_r
def test_a_spanned_columns_variance_is_rs_lm(tmp_path, monkeypatch):
    """R's ``lm`` on the spanned design: x₃'s coefficient NA, ``vcov`` NA in its row and column, and
    ``summary``'s unscaled covariance of the others the least squares' (XᵀX)⁻¹ there, with every
    SVD failing."""
    X, y = _spanned(7)
    frame = pd.DataFrame({"x1": X[:, 1], "x2": X[:, 2], "x3": X[:, 3], "y": y})
    r = run_r("""
        d <- read.csv(s_csv)
        fit <- lm(y ~ x1 + x2 + x3, data = d)
        v <- vcov(fit)
        out(list(aliased = names(which(is.na(coef(fit)))),
                 na_rows = unname(which(apply(is.na(v), 1, all))) - 1,
                 na_columns = unname(which(apply(is.na(v), 2, all))) - 1,
                 unscaled = unname(summary(fit)$cov.unscaled)))
    """, {"s": frame}, tmp_path)
    assert r["aliased"] == "x3" and r["na_rows"] == 3 and r["na_columns"] == 3
    _no_svd(monkeypatch)
    _, _, inv, _ = VS._ols(X, y)
    _close(inv[:3, :3], np.asarray(r["unscaled"], dtype=float))
    assert np.isnan(inv[3]).all() and np.isnan(inv[:, 3]).all()



# ── the inference sensitivity analysis, as R's summary.lm and summary.glm ─────
# Backward elimination by pooled Wald tests (``backward_wald``, the sensitivity analysis under
# inference) reads each copy's fit as R's ``summary.lm`` and ``summary.glm`` report it: the columns
# R's rule finds aliased (of a dependent set the later column, the exposure's columns first) are
# set aside untested, their coefficients NA there; the rest are tested from the QR factor, σ̂² =
# RSS/(n − rank) and df n − rank, or by a logistic regression on them with its z test; a copy whose
# fit fails refuses the analysis. On the base (σ̂²·pinv(XᵀX), k every column) an aliased column's
# partner left first at p = 1, a covariate with a real effect, and every logistic p was NaN, so
# nothing left and the analysis read as having removed no covariate.
#
# Truths: statsmodels on the columns ``lm`` keeps (``lm`` with an NA coefficient spans the same
# columns, with the same rank, as ``lm`` without it); for a nearly spanned column, statsmodels on a
# well-conditioned basis of the same span, mapped back; R, step by step (``needs_r``).

ALPHA = VS.INFERENCE_ALPHA


def _sensitivity_rows(seed: int, partner: str, n: int = 200) -> tuple[pd.DataFrame, np.ndarray]:
    """An exposure ``e`` (forced), ``x`` with an effect, its partner ``x2``, ``u`` with none and
    ``w`` with a small one. ``x2`` is x + 10⁻⁶·noise (full rank, X's condition number about
    2·10⁶), 2x + 1, exactly or with 10⁻⁹·noise (aliased at :data:`VS.ALIASED`), or noise of its
    own (``independent``)."""
    rng = np.random.default_rng(seed)
    e, x, u, w, noise = rng.normal(size=(5, n))
    x2 = {"near": x + 1e-6 * noise, "aliased": 2 * x + 1,
          "aliased_noise": 2 * x + 1 + 1e-9 * noise, "independent": noise}[partner]
    y = 1.0 + 0.5 * e + 0.3 * x + 0.14 * w + rng.normal(size=n)
    return pd.DataFrame({"e": e, "x": x, "x2": x2, "u": u, "w": w}), y


def _binary_outcome(frame: pd.DataFrame, seed: int) -> np.ndarray:
    """A yes/no outcome with log-odds 0.5e + 0.6x."""
    rng = np.random.default_rng(seed)
    lp = (0.5 * frame["e"] + 0.6 * frame["x"]).to_numpy()
    return (rng.random(len(frame)) < 1 / (1 + np.exp(-lp))).astype(float)


def _summary_p(frame: pd.DataFrame, y: np.ndarray, columns: list[str],
               task: str = "regression") -> dict[str, float]:
    """Each unforced column's p-value as ``summary.lm`` or ``summary.glm`` reports it, by
    statsmodels on the intercept and ``columns`` (none aliased): ``OLS``'s t test on n − k df, or
    ``Logit``'s Wald z test (Newton's method to 10⁻¹²)."""
    X = np.column_stack([np.ones(len(frame)), frame[columns].to_numpy(dtype=float)])
    if task == "binary":
        fit = sm.Logit(y, X).fit(disp=0, method="newton", tol=1e-12, maxiter=100)
    else:
        fit = sm.OLS(y, X).fit()
    return {c: float(fit.pvalues[j]) for j, c in enumerate(columns, start=1) if c != "e"}


def _reparametrized_p(frame: pd.DataFrame, y: np.ndarray, columns: list[str], a: float, b: float,
                      delta: float, task: str = "regression") -> dict[str, float]:
    """The least squares' t tests (or a logistic regression's z tests) where x₂ = a + b·x + δ·v
    with δ tiny, from statsmodels on a well-conditioned basis of the same span, v = (x₂ − a −
    b·x)/δ in x₂'s place: its coefficients γ and their covariance mapped back (β_x₂ = γ_v/δ, β_x =
    γ_x − b·γ_v/δ, the intercept's γ₀ − a·γ_v/δ). Where x or x₂ is out, statsmodels on the
    columns."""
    from scipy import stats

    if not ("x" in columns and "x2" in columns):
        return _summary_p(frame, y, columns, task)
    W = frame[columns].assign(x2=(frame["x2"] - a - b * frame["x"]) / delta)
    X = np.column_stack([np.ones(len(W)), W.to_numpy(dtype=float)])
    if task == "binary":
        fit = sm.Logit(y, X).fit(disp=0, method="newton", tol=1e-12, maxiter=100)
    else:
        fit = sm.OLS(y, X).fit()
    i, j = 1 + columns.index("x"), 1 + columns.index("x2")
    back = np.eye(X.shape[1])
    back[j, j], back[i, j], back[0, j] = 1 / delta, -b / delta, -a / delta
    beta = back @ fit.params
    se = np.sqrt(np.diag(back @ fit.cov_params() @ back.T))
    df = len(y) - X.shape[1]
    tail = stats.norm.sf if task == "binary" else (lambda z: stats.t.sf(z, df))
    return {c: float(2 * tail(abs(beta[k] / se[k])))
            for k, c in enumerate(columns, start=1) if c != "e"}


def _elimination(p_of, columns: list[str], aliased_of=lambda remaining: []
                 ) -> tuple[list[dict], list[list[str]], list[str]]:
    """The elimination from ``columns``: at each step the columns ``aliased_of`` names are set
    aside, the rest tested by ``p_of``, and the largest p above α leaves. (Each step's p-values,
    each step's set-aside columns, the columns kept.)"""
    path, aside, remaining = [], [], list(columns)
    while True:
        out = aliased_of(remaining)
        tested = p_of([c for c in remaining if c not in out])
        path.append(tested)
        aside.append(out)
        worst = max(tested, key=lambda c: (tested[c], c)) if tested else None
        if worst is None or tested[worst] <= ALPHA:
            return path, aside, remaining
        remaining.remove(worst)


def _x2_spanned(remaining: list[str]) -> list[str]:
    """x₂ = 2x + 1 is aliased, the later of the two, while x is in."""
    return ["x2"] if {"x", "x2"} <= set(remaining) else []


def _steps_equal(got: dict, want: tuple[list[dict], list[list[str]], list[str]], rel: float) -> None:
    """``backward_wald``'s path and kept columns are ``want``'s (:func:`_elimination`), each p to
    ``rel`` of itself."""
    path, aside, kept = want
    assert got.get("refused") is None
    assert got["kept"] == kept
    assert len(got["path"]) == len(path)
    for step, tested, out in zip(got["path"], path, aside):
        assert step.get("aliased", []) == out, step["step"]
        assert set(step["tested"]) == set(tested), step["step"]
        for term, p in tested.items():
            assert step["tested"][term] == pytest.approx(p, rel=rel), (step["step"], term)


def _wald(task: str, frame: pd.DataFrame, y: np.ndarray, forced=("e",)) -> dict:
    return VS.backward_wald(task, [frame], [y], {c: [c] for c in frame.columns}, list(forced))


def test_identified_aliases_the_later_column_of_a_dependent_set():
    """R's rule (``lm``'s limited pivoting): of x and 2x + 1 beside the intercept, whichever comes
    later is aliased; of a category coded by every level beside the intercept, the last level. On
    the base a QR with column pivoting kept the longer column, 2x + 1, wherever it stood."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=N)
    ones = np.ones(N)
    assert VS.identified(np.column_stack([ones, x, 2 * x + 1])).tolist() == [0, 1]
    assert VS.identified(np.column_stack([ones, 2 * x + 1, x])).tolist() == [0, 1]
    levels = np.eye(3)[rng.integers(0, 3, N)]
    assert VS.identified(np.column_stack([ones, levels, x])).tolist() == [0, 1, 2, 4]


@needs_r
def test_identified_is_lms_rule_in_either_order(tmp_path):
    """R's ``lm``: the NA coefficient is the later of x and 2x + 1, in either order, and of a
    category's every-level indicators beside the intercept, the last."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=N)
    levels = np.eye(3)[rng.integers(0, 3, N)]
    frame = pd.DataFrame({"x": x, "z": 2 * x + 1, "l1": levels[:, 0], "l2": levels[:, 1],
                          "l3": levels[:, 2], "y": x + rng.normal(size=N)})
    r = run_r("""
        d <- read.csv(s_csv)
        na <- function(f) I(names(which(is.na(coef(lm(f, data = d))))))
        out(list(xz = na(y ~ x + z), zx = na(y ~ z + x), levels = na(y ~ l1 + l2 + l3 + x)))
    """, {"s": frame}, tmp_path)
    ones = np.ones((N, 1))
    for key, names in (("xz", ["x", "z"]), ("zx", ["z", "x"]), ("levels", ["l1", "l2", "l3", "x"])):
        kept = VS.identified(np.hstack([ones, frame[names].to_numpy()])).tolist()
        assert [n for j, n in enumerate(names, start=1) if j not in kept] == r[key], key


@pytest.mark.parametrize("partner", ["aliased", "aliased_noise"])
def test_the_sensitivity_analysis_sets_an_aliased_column_aside_untested(partner):
    """x₂ = 2x + 1, exactly or to 10⁻⁹: R's rule aliases x₂, the later, so it is set aside
    untested at every step x is in (an NA coefficient in ``summary.lm``) and kept, and x, u and w
    are tested as statsmodels tests them on the design without x₂, to 10⁻¹⁰. On the base x left
    first at p = 1 (a zero coefficient beside a finite pinv variance), a covariate with a real
    effect."""
    frame, y = _sensitivity_rows(3, partner)
    got = _wald("regression", frame, y)
    want = _elimination(lambda cols: _summary_p(frame, y, cols), list(frame.columns), _x2_spanned)
    _steps_equal(got, want, rel=1e-10)
    assert "x2" in got["kept"] and got["untested"] == ["x2"]


@pytest.mark.parametrize("seed", [3, 8])
def test_the_sensitivity_analysis_on_a_nearly_spanned_column_is_least_squares(seed):
    """Full rank, x₂ = x + 10⁻⁶·noise (X's condition number about 2·10⁶): every step's p-values are
    the least squares' t tests to 10⁻⁸ of themselves, from statsmodels on a well-conditioned basis
    of the same span (:func:`_reparametrized_p`), and the same terms leave. On the base
    σ̂²·pinv(XᵀX) put x's p off by 1·10⁻⁵ (seed 3) and 5·10⁻⁵ (seed 8) of itself."""
    frame, y = _sensitivity_rows(seed, "near")
    got = _wald("regression", frame, y)
    want = _elimination(lambda cols: _reparametrized_p(frame, y, cols, 0.0, 1.0, 1e-6),
                        list(frame.columns))
    _steps_equal(got, want, rel=1e-8)


@pytest.mark.parametrize("seed", [3, 8])
def test_the_binary_sensitivity_analysis_on_a_nearly_spanned_column_is_glms(seed):
    """Full rank, x₂ = x + 10⁻⁶·noise under a yes/no outcome: the maximum-likelihood fit is finite,
    x's and x₂'s coefficients near ±10⁵ a unit and a standard deviation, and every step's z tests
    are statsmodels' ``Logit`` on a well-conditioned basis of the same span, mapped back
    (:func:`_reparametrized_p`), to 10⁻⁸ of themselves, the same terms leaving. Before this repair,
    as with the columns centered and scaled, those coefficients passed
    ``performance.logistic_fit``'s bound and the analysis was refused at step 1."""
    frame, _ = _sensitivity_rows(seed, "near")
    y = _binary_outcome(frame, 5)
    got = _wald("binary", frame, y)
    want = _elimination(lambda cols: _reparametrized_p(frame, y, cols, 0.0, 1.0, 1e-6, "binary"),
                        list(frame.columns))
    _steps_equal(got, want, rel=1e-8)


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_the_sensitivity_analysis_on_a_well_conditioned_design_is_statsmodels(task):
    """The rule's silent side: with no column aliased or nearly spanned nothing is set aside, and
    every step is statsmodels': the t tests to 10⁻¹⁰, or a logistic regression's z tests (as
    ``summary.glm`` reads them) to 10⁻⁹. The regression's p-values were the base's; the logistic
    one's were not: the base read each z statistic against Student's t on n − k df, x's p 0.000162
    where ``glm``'s is 0.000119."""
    frame, y = _sensitivity_rows(3, "independent")
    if task == "binary":
        y = _binary_outcome(frame, 5)
    got = _wald(task, frame, y)
    want = _elimination(lambda cols: _summary_p(frame, y, cols, task), list(frame.columns))
    _steps_equal(got, want, rel=1e-10 if task == "regression" else 1e-9)
    assert got["untested"] == []


def test_the_binary_sensitivity_analysis_sets_an_aliased_column_aside():
    """x₂ = 2x + 1 under a yes/no outcome: the logistic regression on the identified columns, as
    ``glm`` fits it, x₂ set aside untested; each step's z tests are statsmodels' ``Logit``'s on the
    design without x₂, to 10⁻⁹. On the base the logistic fit on every column had a singular
    information, so every p was NaN and nothing left."""
    frame, _ = _sensitivity_rows(3, "aliased")
    y = _binary_outcome(frame, 5)
    got = _wald("binary", frame, y)
    want = _elimination(lambda cols: _summary_p(frame, y, cols, "binary"), list(frame.columns),
                        _x2_spanned)
    _steps_equal(got, want, rel=1e-9)
    assert got["untested"] == ["x2"]


@pytest.mark.parametrize("at", [0, 1])
def test_the_exposures_columns_come_first_so_its_partner_is_set_aside(at):
    """c = 2e + 1 before the exposure e in the model matrix, or after it: the exposure's columns
    come first in R's rule, so c is the aliased one wherever it stands, set aside untested at every
    step, and x, u and w are tested as statsmodels tests them on the design without c. On the base
    the pivoted QR aliased the exposure instead, and c was tested at p ≈ 5·10⁻²¹ with its effect."""
    frame, y = _sensitivity_rows(3, "independent")
    frame = frame.drop(columns="x2")
    frame.insert(at, "c", 2 * frame["e"] + 1)
    got = _wald("regression", frame, y)
    want = _elimination(lambda cols: _summary_p(frame, y, cols), list(frame.columns),
                        lambda remaining: ["c"])
    _steps_equal(got, want, rel=1e-10)


@pytest.mark.parametrize("task", ["regression", "binary"])
@pytest.mark.parametrize("share", [0.5, 1.0, 2.0])
def test_a_column_at_the_alias_tolerance_is_set_aside_or_tested_right(share, task):
    """x₂ = 2x + 1 + c·r, r orthogonal to the intercept, e and x, so x₂'s part outside the columns
    before it is ``share`` times :data:`VS.ALIASED` of its length. At 1 float64 can tip R's rule
    either way, so what is pinned is that the analysis is right either way (CLAUDE.md): x₂ set
    aside and the rest statsmodels' on the design without it, to 10⁻¹⁰; or x₂ tested and every p
    the least squares', from a well-conditioned basis of the same span (r in x₂'s place, mapped
    back), to 10⁻⁶ (X's condition number near 10⁷ leaves float64 least squares about 10⁻⁹). At
    half the tolerance x₂ is set aside, at twice it is tested. The same under a yes/no outcome,
    statsmodels' ``Logit`` in place of ``OLS``. On the base, at half the tolerance x was tested at
    p = 1 and left; at and above it x's p was 3% off (0.089 and 0.084 for 0.086); under a yes/no
    outcome, before this repair, the fit on x₂'s near copy of 2x + 1 was refused."""
    frame, y = _sensitivity_rows(3, "independent")
    Q, _ = np.linalg.qr(np.column_stack([np.ones(len(frame)), frame[["e", "x"]].to_numpy()]))
    r = np.random.default_rng(9).normal(size=len(frame))
    for _ in range(2):
        r = r - Q @ (Q.T @ r)
    line = 2 * frame["x"].to_numpy() + 1
    c = share * VS.ALIASED * float(np.linalg.norm(line) / np.linalg.norm(r))
    frame["x2"] = line + c * r
    if task == "binary":
        y = _binary_outcome(frame, 5)
    got = _wald(task, frame, y)
    aside = got["path"][0].get("aliased") == ["x2"]
    if aside:
        want = _elimination(lambda cols: _summary_p(frame, y, cols, task), list(frame.columns),
                            _x2_spanned)
    else:
        want = _elimination(lambda cols: _reparametrized_p(frame, y, cols, 1.0, 2.0, c, task),
                            list(frame.columns))
    _steps_equal(got, want, rel=(1e-10 if task == "regression" else 1e-9) if aside else 1e-6)
    if share != 1.0:
        assert aside == (share < 1.0)


def _units_rows(covariate: str, seed: int = 4, n: int = 300) -> tuple[pd.DataFrame, np.ndarray]:
    """A yes/no outcome on the exposure ``e`` (0.5 log-odds a unit), noise ``w``, and a covariate
    whose coefficients in its own units are large though the fit is finite: ``year``, a calendar
    year ~ U(1990, 2020) at 0.1 log-odds a year, so the intercept is near −200; or ``share``, a
    fraction ~ U(0, 0.02) at 150 log-odds a unit (0.87 a standard deviation)."""
    rng = np.random.default_rng(seed)
    e, w = rng.normal(size=(2, n))
    if covariate == "year":
        c = rng.uniform(1990, 2020, n)
        lp = -0.5 + 0.1 * (c - 2005) + 0.5 * e
    else:
        c = rng.uniform(0, 0.02, n)
        lp = 150 * (c - 0.01) + 0.5 * e
    y = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(float)
    return pd.DataFrame({"e": e, covariate: c, "w": w}), y


@pytest.mark.parametrize("covariate", ["year", "share"])
def test_a_logistic_fit_large_in_its_columns_units_is_tested(covariate):
    """A calendar year (the intercept near −200 a unit) or a fraction-scale covariate (its slope
    above 150 a unit): the maximum-likelihood fit is finite (statsmodels' ``Logit`` by Newton's
    method converges, a coefficient past 50 in the columns' units), and each step's z tests are
    ``Logit``'s on the columns as they are, to 10⁻⁹, the same terms leaving. On the base every p
    was NaN and the analysis said it removed no covariate; before this repair it was refused at
    step 1 as a fit that did not reach a finite maximum (``performance.logistic_fit``'s bound on
    a coefficient, 50, read in the columns' units)."""
    frame, y = _units_rows(covariate)
    X = np.column_stack([np.ones(len(y)), frame.to_numpy()])
    fit = sm.Logit(y, X).fit(disp=0, method="newton", tol=1e-12, maxiter=100)
    assert fit.mle_retvals["converged"] and np.max(np.abs(fit.params)) > 50
    got = _wald("binary", frame, y)
    want = _elimination(lambda cols: _summary_p(frame, y, cols, "binary"), list(frame.columns))
    _steps_equal(got, want, rel=1e-9)
    assert covariate in got["kept"]


def _separating(y: np.ndarray, seed: int) -> np.ndarray:
    """A covariate positive exactly where the outcome is an event."""
    return np.where(y == 1, 1.0, -1.0) * np.random.default_rng(seed).uniform(0.5, 1.5, len(y))


def test_a_logistic_fit_that_fails_refuses_the_analysis():
    """A covariate s that separates the outcome (an event exactly where s > 0): the logistic
    likelihood has no maximum, so the analysis is refused at step 1 in words, with no step, removal
    or kept set reported. Without s it runs. On the base every p was NaN, nothing left, and the
    analysis read as having removed no covariate."""
    frame, _ = _sensitivity_rows(3, "independent")
    y = _binary_outcome(frame, 5)
    got = _wald("binary", frame.assign(s=_separating(y, 2)), y)
    assert got["refused"] == {
        "step": 1, "reason": "the logistic regression on the complete rows was not fit (Newton's "
                             "method did not converge, or a coefficient grew past the fit's "
                             "bound, as when the covariates separate an outcome level)"}
    assert got["path"] == [] and got["kept"] is None
    assert _wald("binary", frame, y)["refused"] is None


def test_one_imputed_copys_failed_fit_refuses_the_pooled_analysis():
    """Three copies whose s is noise, but in the second it separates the outcome: the refusal names
    that copy. With every copy's s noise, the pooled analysis runs."""
    frame, _ = _sensitivity_rows(3, "independent")
    y = _binary_outcome(frame, 5)
    noise = np.random.default_rng(4).normal(size=(3, len(y)))
    terms = {c: [c] for c in [*frame.columns, "s"]}
    copies = [frame.assign(s=noise[k]) for k in range(3)]
    assert VS.backward_wald("binary", copies, [y] * 3, terms, ["e"])["refused"] is None
    copies[1] = frame.assign(s=_separating(y, 2))
    got = VS.backward_wald("binary", copies, [y] * 3, terms, ["e"])
    assert got["refused"] == {
        "step": 1, "reason": "the logistic regression on imputed copy 2 of 3 was not fit (Newton's "
                             "method did not converge, or a coefficient grew past the fit's "
                             "bound, as when the covariates separate an outcome level)"}
    assert got["kept"] is None


@pytest.mark.parametrize("rows", [6, 7])
def test_least_squares_with_no_residual_df_refuses_the_analysis(rows):
    """Six coefficients (the intercept, e, x, x₂, u, w) on six rows leave no residual variance, so
    no t test is defined and the analysis is refused; on seven rows (one residual df) it runs, each
    step statsmodels'. On the base six rows raised ``ZeroDivisionError`` (RSS/(n − k))."""
    frame, y = _sensitivity_rows(3, "independent", n=rows)
    got = _wald("regression", frame, y)
    if rows == 6:
        assert got["refused"] == {"step": 1, "reason": "the least squares on the complete rows "
                                                       "leaves no residual variance (6 rows, 6 "
                                                       "coefficients identified)"}
        assert got["kept"] is None
        return
    _steps_equal(got, _elimination(lambda cols: _summary_p(frame, y, cols), list(frame.columns)),
                 rel=1e-9)


def test_copies_that_alias_different_columns_refuse_the_pooled_analysis():
    """x₂ = 2x + 1 in one copy and 2x + 1 + noise in the other: ``lm`` aliases x₂ in the first only,
    so the copies' fits do not estimate the same coefficients and no pooled test of them is
    defined; the analysis is refused, naming x₂. Where both copies alias it, it runs."""
    exact, y = _sensitivity_rows(3, "aliased")
    noise = np.random.default_rng(6).normal(size=(2, len(y)))
    terms = {c: [c] for c in exact.columns}
    both = [exact.assign(u=exact["u"] + 0.1 * noise[0]), exact]
    ran = VS.backward_wald("regression", both, [y, y], terms, ["e"])
    assert ran["refused"] is None and ran["path"][0]["aliased"] == ["x2"]
    one = [exact.assign(x2=exact["x2"] + 0.1 * noise[1]), exact]
    got = VS.backward_wald("regression", one, [y, y], terms, ["e"])
    assert got["kept"] is None
    assert got["refused"] == {
        "step": 1, "reason": "`x2` is spanned by the other columns in some imputed copies and not "
                             "in others, so the copies' fits do not estimate the same "
                             "coefficients and no pooled test is defined"}


def test_a_pooled_test_with_no_p_value_refuses_the_analysis(monkeypatch):
    """Where a pooled test returns no p-value (here D1 is made to report the within-copy covariance
    singular, as ``pooled_wald`` does), the analysis is refused, naming the term; on the base its
    NaN left the term in, untested and unsaid. Unpatched, the same copies run."""
    from turbotab.core.methods import imputation

    frame, y = _sensitivity_rows(3, "independent")
    noise = np.random.default_rng(7).normal(size=len(y))
    copies = [frame, frame.assign(u=frame["u"] + 0.1 * noise)]
    terms = {"e": ["e"], "xu": ["x", "u"], "x2": ["x2"], "w": ["w"]}
    assert VS.backward_wald("regression", copies, [y, y], terms, ["e"])["refused"] is None
    monkeypatch.setattr(imputation, "pooled_wald", lambda *args, **kwargs: None)
    got = VS.backward_wald("regression", copies, [y, y], terms, ["e"])
    assert got["refused"] == {"step": 1, "reason": "the pooled Wald test of `xu` has no p-value"}
    assert got["kept"] is None


def test_the_sentence_says_what_the_analysis_did():
    """The evaluation stage's sentence: as before where every term was tested; naming a term kept
    untested, and why, where one was; and, where the analysis was refused, that it stopped, at
    which step, and why, with no removal claimed."""
    from turbotab.core.stages.evaluation import sensitivity_sentence
    from turbotab.core.voice import listing

    head = (f"As a labeled sensitivity analysis (not the reported model), backward elimination at "
            f"α = {ALPHA} with Wald tests on the complete rows (Wood, White & Royston 2008), the "
            f"exposure kept, ")
    tail = "; the declared model is the reported one."
    frame, y = _sensitivity_rows(3, "independent")
    found = _wald("regression", frame, y)
    dropped = [s["dropped"] for s in found["path"] if s["dropped"]]
    assert sensitivity_sentence(found) == (
        f"{head}removed {listing(dropped) if dropped else 'no covariate'}{tail}")
    frame, y = _sensitivity_rows(3, "aliased")
    found = _wald("regression", frame, y)
    dropped = [s["dropped"] for s in found["path"] if s["dropped"]]
    assert found["untested"] == ["x2"] and dropped
    assert VS.ALIASED == 1e-7
    assert sensitivity_sentence(found) == (
        f"{head}removed {listing(dropped)}; `x2` was kept untested: the other columns span it to "
        f"within 10⁻⁷ of its length, the share at which R's lm sets a column aside (its "
        f"coefficient NA){tail}")
    found["untested"] = ["x2", "z"]
    assert sensitivity_sentence(found).endswith(
        "; `x2` and `z` were kept untested: the other columns span each to within 10⁻⁷ of its "
        f"length, the share at which R's lm sets a column aside (its coefficient NA){tail}")
    pooled = VS.backward_wald("regression", [frame, frame.assign(u=frame["u"] * 1.01)], [y, y],
                              {c: [c] for c in frame.columns}, ["e"])
    assert pooled["untested"] == ["x2"]
    assert sensitivity_sentence(pooled).startswith(
        "As a labeled sensitivity analysis (not the reported model), backward elimination at "
        f"α = {ALPHA} with Wald tests pooled across the imputed copies by Rubin's rules (Wood, "
        "White & Royston 2008), the exposure kept, removed ")
    yb = _binary_outcome(frame, 5)
    refused = _wald("binary", frame.drop(columns="x2").assign(s=_separating(yb, 2)), yb)
    assert sensitivity_sentence(refused) == (
        f"{head}stopped at step 1 and reports no result: the logistic regression on the complete "
        f"rows was not fit (Newton's method did not converge, or a coefficient grew past the fit's "
        f"bound, as when the covariates separate an outcome level){tail}")


def test_the_sentence_names_lms_rule_where_glm_would_keep_the_column():
    """x₂ = 2x + 1 + 10⁻⁹·noise under a yes/no outcome: its part outside the intercept, e and x
    (numpy's least squares) is about 4·10⁻¹⁰ of its length, below lm's 10⁻⁷ (:data:`VS.ALIASED`),
    so it is set aside, but above the 10⁻¹¹ at which ``glm.fit`` aliases by default
    (``min(1e-7, epsilon/1000)``), where R's ``glm`` keeps it (its p 0.31). The sentence says the
    rule the code applies, lm's, and not that R reports the coefficient NA, as it said before this
    repair."""
    from turbotab.core.stages.evaluation import sensitivity_sentence

    frame, _ = _sensitivity_rows(3, "aliased_noise")
    y = _binary_outcome(frame, 5)
    before = np.column_stack([np.ones(len(y)), frame[["e", "x"]].to_numpy()])
    x2 = frame["x2"].to_numpy()
    rest = x2 - before @ np.linalg.lstsq(before, x2, rcond=None)[0]
    share = float(np.linalg.norm(rest) / np.linalg.norm(x2))
    assert 1e-11 < share < VS.ALIASED == 1e-7
    found = _wald("binary", frame, y)
    assert found["refused"] is None and found["untested"] == ["x2"]
    said = sensitivity_sentence(found)
    assert "R reports" not in said
    assert said.endswith("; `x2` was kept untested: the other columns span it to within 10⁻⁷ of "
                         "its length, the share at which R's lm sets a column aside (its "
                         "coefficient NA); the declared model is the reported one.")
    from turbotab.core.stages.evaluation import _power_of_ten

    assert [_power_of_ten(x) for x in (1e-7, 2.5e-7, 1e-11, 1e3)] == [
        "10⁻⁷", "2.5 × 10⁻⁷", "10⁻¹¹", "10³"]


def _r_steps(task: str, frame: pd.DataFrame, y: np.ndarray, tmp_path) -> None:
    """R's elimination by ``summary.lm``'s t tests or ``summary.glm``'s z tests over ``frame``'s
    columns, e forced, the largest p above α leaving (``which.max`` passes over an aliased term's
    NA), is ``backward_wald``'s: each step's p-values to 10⁻⁸ of themselves, the terms R reports NA
    set aside, and the terms kept. ``glm`` runs to ``epsilon = 1e-10`` and once more from its own
    estimates: ``summary.glm`` reads the information at the weights of the iteration before the
    last, so restarted at the estimates it reads it there. (``glm.fit`` aliases at
    ``min(1e-7, epsilon/1000)`` of each weighted column, 10⁻¹³ here, which an exact dependence such
    as 2x + 1 still meets.)"""
    r = run_r(f"""
        d <- read.csv(s_csv)
        terms <- setdiff(names(d), "y"); forced <- "e"; path <- list(); na <- list()
        logistic <- function(f) {{
          tight <- glm.control(epsilon = 1e-10, maxit = 100)
          b <- coef(glm(f, family = binomial, data = d, control = tight)); b[is.na(b)] <- 0
          glm(f, family = binomial, data = d, start = b, control = tight)
        }}
        repeat {{
          f <- as.formula(paste("y ~", paste(terms, collapse = " + ")))
          fit <- if ("{task}" == "binary") logistic(f) else lm(f, data = d)
          s <- summary(fit)$coefficients
          tested <- setdiff(terms, forced)
          p <- setNames(rep(NA_real_, length(tested)), tested)
          hit <- intersect(tested, rownames(s)); p[hit] <- s[hit, 4]
          path[[length(path) + 1]] <- as.list(p[!is.na(p)])
          na[[length(na) + 1]] <- I(names(which(is.na(coef(fit)))))
          worst <- names(which.max(p))
          if (length(worst) == 0 || p[[worst]] <= {ALPHA}) break
          terms <- setdiff(terms, worst)
        }}
        out(list(path = path, na = na, kept = I(terms)))
    """, {"s": frame.assign(y=y)}, tmp_path)
    got = _wald(task, frame, y)
    assert got.get("refused") is None
    assert got["kept"] == r["kept"]
    assert len(got["path"]) == len(r["path"])
    for step, want, na in zip(got["path"], r["path"], r["na"]):
        assert step.get("aliased", []) == na, step["step"]
        assert set(step["tested"]) == set(want or {}), step["step"]
        for term, p in (want or {}).items():
            assert step["tested"][term] == pytest.approx(p, rel=1e-8), (step["step"], term)


@needs_r
@pytest.mark.parametrize(("task", "partner"), [("regression", "aliased"), ("regression", "near"),
                                               ("binary", "aliased"), ("binary", "near"),
                                               ("binary", "independent")])
def test_the_sensitivity_analysis_is_rs_summary_step_by_step(task, partner, tmp_path):
    """R's elimination step by step (:func:`_r_steps`) on x₂ = 2x + 1 or x + 10⁻⁶·noise, or with
    every column independent."""
    frame, y = _sensitivity_rows(3, partner)
    if task == "binary":
        y = _binary_outcome(frame, 5)
    _r_steps(task, frame, y, tmp_path)


@needs_r
@pytest.mark.parametrize("covariate", ["year", "share"])
def test_a_logistic_fit_large_in_its_columns_units_is_rs_glm_step_by_step(covariate, tmp_path):
    """A calendar year or a fraction-scale covariate (:func:`_units_rows`), the coefficients past
    50 in the columns' own units: R's ``glm`` elimination step by step (:func:`_r_steps`), each p to
    10⁻⁸ of itself. Before this repair the analysis was refused at step 1."""
    frame, y = _units_rows(covariate)
    _r_steps("binary", frame, y, tmp_path)
