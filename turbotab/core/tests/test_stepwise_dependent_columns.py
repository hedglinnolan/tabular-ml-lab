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
