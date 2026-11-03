"""The univariable screen (``set_selection:univariable``, ``variable_selection.univariable_p``) on a
term whose columns the rows do not all identify: a category coded by every level beside the
intercept, as a penalized family's pipeline codes it. On the base the F test counted every column
of such a term in its degrees of freedom (2 for a 2-level category, where the rows identify 1),
and the logistic regression had no maximum on the dependent columns, so a yes/no outcome's p was
1 and the term was never kept (INBOX:1018). The test now has the rank's degrees of freedom, as R's
``lm`` and ``glm`` fit: rank − 1 and n − rank, the logistic fit on the identified columns.

Truth from statsmodels on the reference coding of the same category (``C(g)``, all levels but the
first): its columns and the intercept span the every-level coding's, so the residuals, the
likelihood's maximum and the rank are the same. statsmodels on the every-level coding itself is no
truth: ``OLS`` there reports df_model 2 for a 2-level category. The bound test reads
``performance.logistic_fit``'s coefficient bound against statsmodels' ``Logit`` either side of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm

from turbotab.core.models import performance as perf
from turbotab.core.models import variable_selection as VS

N = 300
TASKS = ["regression", "binary"]


def _rows(seed: int, task: str, levels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A category of ``levels`` levels with a small effect, a numeric column with none, and the
    outcome."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, levels, N)
    x = rng.normal(size=N)
    if task == "binary":
        lp = np.linspace(-0.25, 0.25, levels)[g]
        y = (rng.random(N) < 1 / (1 + np.exp(-lp))).astype(float)
    else:
        y = np.linspace(-0.12, 0.12, levels)[g] + rng.normal(size=N)
    return g, x, y


def _every_level(g: np.ndarray, levels: int) -> np.ndarray:
    return np.column_stack([(g == k).astype(float) for k in range(levels)])


def _null_loglik(y: np.ndarray) -> float:
    """The intercept-only logistic maximum, at the event share ȳ: n·(ȳ ln ȳ + (1 − ȳ) ln(1 − ȳ)).
    ``Logit``'s llnull refits that model to a looser tolerance (1.5·10⁻⁸ off it here)."""
    m = float(y.mean())
    return len(y) * (m * np.log(m) + (1 - m) * np.log(1 - m))


def _category_p(g: np.ndarray, y: np.ndarray, task: str) -> float:
    """statsmodels on the reference coding: ``anova_lm``'s F test against the intercept alone, or
    the likelihood-ratio test from ``Logit``'s llf, against the null's maximum, on its df_model."""
    d = pd.DataFrame({"y": y, "g": g})
    if task == "binary":
        fit = smf.logit("y ~ C(g)", d).fit(disp=0, method="newton", tol=1e-12, maxiter=100)
        return float(stats.chi2.sf(2.0 * (fit.llf - _null_loglik(y)), fit.df_model))
    return float(anova_lm(smf.ols("y ~ 1", d).fit(), smf.ols("y ~ C(g)", d).fit())["Pr(>F)"].iloc[1])


def _column_p(x: np.ndarray, y: np.ndarray, task: str) -> float:
    """statsmodels' slope p-value: ``OLS``'s t test or ``Logit``'s Wald z."""
    X = np.column_stack([np.ones(len(y)), x])
    if task == "binary":
        return float(sm.Logit(y, X).fit(disp=0, method="newton", tol=1e-12, maxiter=100).pvalues[1])
    return float(sm.OLS(y, X).fit().pvalues[1])


@pytest.mark.parametrize("coding", ["standardized", "raw"])
@pytest.mark.parametrize("levels", [2, 3, 5])
@pytest.mark.parametrize("task", TASKS)
def test_a_category_coded_by_every_level_is_tested_on_its_rank(task, levels, coding):
    """The term's p-value is statsmodels' on the same category, whether its indicators are
    standardized (as the selection step reads them) or not. On the base the regression's was the
    F test on ``levels`` and n − levels − 1 df (2 levels: 0.035 against 0.0097), and the yes/no
    outcome's 1 where the logistic fit found no maximum on the dependent columns, or the χ² on
    ``levels`` df where it reached one (2 standardized levels: 0.080 against 0.025)."""
    g, _, y = _rows(levels, task, levels)
    D = _every_level(g, levels)
    Z = VS.standardized(pd.DataFrame(D))[0] if coding == "standardized" else D
    got = VS.univariable_p(Z, y, task, [list(range(levels))])[0]
    assert got == pytest.approx(_category_p(g, y, task), rel=1e-8)


@pytest.mark.parametrize("task", TASKS)
def test_every_level_and_reference_codings_give_one_p_value(task):
    """By derivation, a category of three levels gives the same p-value coded by every level as
    coded by all but the first (two columns either way, so the test is the F test or the
    likelihood-ratio test under both): the span with the intercept is the same."""
    g, _, y = _rows(3, task, 3)
    D = _every_level(g, 3)
    every = VS.univariable_p(VS.standardized(pd.DataFrame(D))[0], y, task, [[0, 1, 2]])[0]
    reference = VS.univariable_p(VS.standardized(pd.DataFrame(D[:, 1:]))[0], y, task, [[0, 1]])[0]
    assert every == pytest.approx(reference, rel=1e-8)


@pytest.mark.parametrize("task", TASKS)
def test_a_term_of_independent_columns_is_tested_as_before(task):
    """The silent side: where every column of a term is identified the rank is the column count
    plus one, so a single column's t or Wald z test, and a reference-coded category's F or
    likelihood-ratio test on its column count, are statsmodels' as on the base."""
    g, x, y = _rows(5, task, 4)
    D = _every_level(g, 4)[:, 1:]
    Z = VS.standardized(pd.DataFrame(np.column_stack([x, D])))[0]
    got = VS.univariable_p(Z, y, task, [[0], [1, 2, 3]])
    assert got == pytest.approx([_column_p(x, y, task), _category_p(g, y, task)], rel=1e-8)


@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("task", TASKS)
def test_a_term_that_adds_nothing_to_the_intercept_has_no_test(task, width):
    """A term whose columns are all zero (rank 1 with the intercept: R's ``anova`` and ``summary``
    give NA) has no test and is never kept, its p 1. On the base the regression's single zero
    column had p 0 (its standard error NaN read as no spread), so it was kept."""
    _, _, y = _rows(0, task, 2)
    got = VS.univariable_p(np.zeros((N, width)), y, task, [list(range(width))])[0]
    assert got == 1.0


@pytest.mark.parametrize(("task", "levels", "seed"), [("regression", 2, 11), ("binary", 3, 0)])
def test_the_univariable_screen_keeps_a_category_by_its_own_test(task, levels, seed):
    """The selection step keeps each term whose statsmodels p-value is below 0.25: here the
    category, coded by every level, and not the numeric column. On the base it kept neither
    (regression: the F test's p 0.295 on 2 and 297 df, statsmodels' 0.118; yes/no: p 1, statsmodels'
    0.044), so the fit there would have had no predictor."""
    g, x, y = _rows(seed, task, levels)
    X = pd.DataFrame({"x": x, **{f"g_{k}": col for k, col in enumerate(_every_level(g, levels).T)}})
    p = {"x": _column_p(x, y, task), "g": _category_p(g, y, task)}
    expected = [t for t in ("x", "g") if p[t] < VS.UNIVARIABLE_P]
    assert expected == ["g"]
    assert VS.Selector("univariable", task, inputs=["x", "g"]).fit(X, y).selected_ == expected


@pytest.mark.parametrize("share", [0.8, 1.25])
def test_the_logistic_fit_refuses_a_coefficient_past_its_bound(share):
    """``performance.logistic_fit`` gives statsmodels' ``Logit`` maximum where its largest
    coefficient is below :data:`performance.COEFFICIENT_BOUND` and None past it. The slope is set
    by scaling the column: the maximum's slope scales inversely with it, the intercept unchanged."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=400)
    y = (rng.random(400) < 1 / (1 + np.exp(-(0.3 + x)))).astype(float)
    slope = float(sm.Logit(y, np.column_stack([np.ones(400), x])).fit(disp=0).params[1])
    X = np.column_stack([np.ones(400), x * slope / (share * perf.COEFFICIENT_BOUND)])
    want = sm.Logit(y, X).fit(disp=0, method="newton", tol=1e-12, maxiter=100)
    assert np.max(np.abs(want.params)) == pytest.approx(share * perf.COEFFICIENT_BOUND, rel=1e-6)
    got = perf.logistic_fit(X, y)
    if share > 1:
        assert got is None
    else:
        np.testing.assert_allclose(got[0], want.params, rtol=1e-8)
        np.testing.assert_allclose(got[1], want.cov_params(), rtol=1e-6)
