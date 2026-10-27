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
