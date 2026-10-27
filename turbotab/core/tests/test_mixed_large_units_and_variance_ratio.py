"""The mixed model on large units, and where REML's ratio of the variances lies past the search.

**Large units.** The REML profile in γ = σ²_u/σ²_e is searched on a grid up to γ = 10⁸. At a γ
the fit factored ``K = QᵀH⁻¹Q``, which it formed as ``I − Σ_g c_g b_g b_gᵀ``; K's smallest
eigenvalue is 1/(1 + γ m_g), some 5 × 10⁻¹³ at the grid's top on units of 50,000 rows, and the
difference of numbers near 1 that formed it left that to rounding: ``LinAlgError: Matrix is not
positive definite`` ended an ordinary fit (4 units of 50,000 rows, 20 constructions of 20). K is
now the Gram matrix of a triangle factored from the within-unit deviations and each unit's sums
(``repeated._at``), so it is positive definite to working precision at every γ. Truth: statsmodels
``MixedLM`` (REML) for the fixed effects; for γ̂ and the standard errors, the REML profile written
here from each unit's own cross-products of its deviations from its means, unit by unit, so no
difference of numbers near 1 forms it (lme4 2.0.6 with lmerTest 3.2.1 on R 4.6.1, offline, stops
on this flat profile at a γ̂ larger by 1.7 × 10⁻⁴ of itself, where the profile is 4 × 10⁻⁸
higher); and for the intercept's Satterthwaite df, a derivation: on units this large its variance
is σ²_u/G, whose estimate rests on G − 1 = 3 df.

**Past the search.** Where the outcome barely varies within a unit beyond what the predictors
explain, REML's γ̂ can lie past 10⁸; the fit stopped at 10⁸ and said nothing, and the table served
standard errors half the right size (50 units of 4 rows, within-unit noise of SD 10⁻⁵: 0.066
where the right one is 0.132). The fit now asks the profile's derivative at the grid's top, in
closed form (``repeated._slope``); where it still falls the table is refused in words
(``repeated.beyond_words``). Truth, by derivation: on balanced units with only unit-level columns
REML splits into the within-unit mean square s²_w (N − G df) and the unit means' least-squares
residual variance s²_b (G − p df), so γ̂ = (s²_b − s²_w/m)/s²_w, ∞ where s²_w = 0; the fixed
effects and their standard errors are the unit means' least squares (statsmodels OLS) at every
interior γ̂, and the Satterthwaite df are G − p. Where s²_w = 0 the profile's slope in log γ tends
to G − p_u − (N − P), p_u the design's unit-level directions.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.optimize import minimize_scalar

from turbotab.core.models.inference import Clusters
from turbotab.core.models.repeated import (GAMMA_GRID, RandomInterceptRegressor, _profile, _slope,
                                           _sums, fit_random_intercept, mixed_table)

# ── large units ──────────────────────────────────────────────────────────────


def _large() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4 units of 50,000 rows: y = 1 + 0.5x + a standard normal per unit + a standard normal."""
    G, m = 4, 50_000
    rng = np.random.default_rng(0)
    codes = np.repeat(np.arange(G), m)
    x = rng.normal(size=G * m)
    y = 1 + 0.5 * x + rng.normal(size=G)[codes] + rng.normal(size=G * m)
    return np.column_stack([np.ones(G * m), x]), y, codes


def _per_unit_reml(X: np.ndarray, y: np.ndarray, codes: np.ndarray) -> tuple[float, np.ndarray]:
    """γ̂ and the standard errors from the REML profile written from each unit's cross-products of
    its deviations from its means plus its sums: ``[X y]ᵀH⁻¹[X y] = Σ_g (D_gᵀD_g + s_g s_gᵀ /
    (m_g(1 + m_g γ)))``, minimized by Brent's method on [0.1, 10]."""
    N, P = X.shape
    blocks = []
    for g in np.unique(codes):
        Z = np.column_stack([X, y])[codes == g]
        D = Z - Z.mean(axis=0)
        blocks.append((D.T @ D, Z.sum(axis=0), len(Z)))

    def parts(gamma: float) -> tuple[float, float, np.ndarray]:
        M = sum(W + np.outer(s, s) / (m * (1 + m * gamma)) for W, s, m in blocks)
        A, b = M[:P, :P], M[:P, P]
        rss = M[P, P] - b @ np.linalg.solve(A, b)
        log_h = sum(math.log1p(m * gamma) for _, _, m in blocks)
        return (N - P) * math.log(rss / (N - P)) + log_h + np.linalg.slogdet(A)[1], rss, A

    found = minimize_scalar(lambda z: parts(math.exp(z))[0], bounds=(math.log(0.1), math.log(10)),
                            method="bounded", options={"xatol": 1e-12})
    gamma = math.exp(found.x)
    _, rss, A = parts(gamma)
    return gamma, np.sqrt(np.diag(rss / (N - P) * np.linalg.inv(A)))


def test_four_units_of_50000_rows_fit_at_the_reml_estimate():
    """The regressor fits, at statsmodels' REML fixed effects; on the base it raised
    ``LinAlgError: Matrix is not positive definite`` at the grid's top."""
    X, y, codes = _large()
    model = RandomInterceptRegressor(units=pd.Series(codes)).fit(pd.DataFrame({"x": X[:, 1]}), y)
    reml = sm.MixedLM(y, X, groups=codes).fit(reml=True)
    np.testing.assert_allclose([model.intercept_, *model.coef_], reml.fe_params, rtol=1e-6)
    gamma, _ = _per_unit_reml(X, y, codes)
    assert model.sigma2_u_ / model.sigma2_e_ == pytest.approx(gamma, rel=1e-6)


def test_the_table_on_four_units_of_50000_rows_is_reml_and_satterthwaite():
    """The table is served: γ̂ and the standard errors are the per-unit profile's, the intercept's
    df G − 1 = 3 to the 1/(m γ̂) its rows add, and the row-level `x`'s near N − G − 1. On the base
    the table ended in the same ``LinAlgError``."""
    X, y, codes = _large()
    clusters = Clusters(column="unit", codes=codes, n_clusters=4, n_missing=0)
    table = mixed_table(pd.DataFrame({"x": X[:, 1]}), y, clusters)
    assert table.info["refused"] is None
    fit = fit_random_intercept(X, y, codes)
    gamma, se = _per_unit_reml(X, y, codes)
    assert fit.sigma2_u / fit.sigma2_e == pytest.approx(gamma, rel=1e-6)
    np.testing.assert_allclose([r["se"] for r in table.rows], se, rtol=1e-6)
    intercept, x = table.rows
    assert intercept["df"] == pytest.approx(3.0, rel=1e-4)
    assert x["df"] == pytest.approx(len(y) - 4 - 1, rel=1e-6)


def test_ten_units_of_20000_rows_constant_within_units_fit_as_the_unit_means():
    """An outcome constant within each of 10 units of 20,000 rows, a unit-level `w`: REML's γ̂ is
    infinite (s²_w = 0), and the regressor's fixed effects are the unit means' least squares, as
    they are at every γ on balanced units with unit-level columns. On the base the fit raised
    ``LinAlgError: Matrix is not positive definite``."""
    G, m = 10, 20_000
    rng = np.random.default_rng(2)
    codes = np.repeat(np.arange(G), m)
    w = rng.normal(size=G)
    means = 2 + 0.5 * w + rng.normal(size=G)
    model = RandomInterceptRegressor(units=pd.Series(codes)).fit(pd.DataFrame({"w": w[codes]}),
                                                                 means[codes])
    ols = sm.OLS(means, sm.add_constant(w)).fit()
    np.testing.assert_allclose([model.intercept_, *model.coef_], ols.params, rtol=1e-9)


# ── past the search ──────────────────────────────────────────────────────────

G, M = 50, 4
N = G * M


def _balanced(noise: float, seed: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """50 units of 4 rows, a unit-level `w`, an outcome that is a unit's level plus within-unit
    noise of SD ``noise``."""
    rng = np.random.default_rng(seed)
    codes = np.repeat(np.arange(G), M)
    w = rng.normal(size=G)
    y = (2 + 0.5 * w + rng.normal(size=G))[codes] + noise * rng.normal(size=N)
    return w, y, codes


def _derived(w: np.ndarray, y: np.ndarray, codes: np.ndarray) -> tuple[float, Any]:
    """REML's γ̂ on balanced units with unit-level columns, and the unit means' least squares."""
    means = pd.Series(y).groupby(codes).mean().to_numpy()
    ols = sm.OLS(means, sm.add_constant(w)).fit()
    within = float(((y - means[codes]) ** 2).sum() / (N - G))
    return ((ols.scale - within / M) / within if within > 0 else math.inf), ols


def _clusters(codes: np.ndarray) -> Clusters:
    return Clusters(column="pid", codes=codes, n_clusters=int(codes.max()) + 1, n_missing=0)


BEYOND = ("The random-intercept model's REML likelihood by `pid` on these {G} units is still "
          "rising where the between-unit variance is 100,000,000 times the within-unit variance, "
          "the most the fit searches: within a `pid` unit the outcome varies so little beyond what "
          "the predictors explain that REML puts the within-unit SD below 0.0001 times the "
          "between-unit SD. The fit does not estimate variances that far apart, so no interval or "
          "p-value is reported.")

# within-unit noise: the derived γ̂ is 7.8 × 10³, 7.8 × 10⁵, 4.0 × 10⁷ (served), and
# 2.2 × 10⁸, 7.8 × 10⁹ (the case the base served at SE 0.066), ∞ (refused)
NOISE = (1e-2, 1e-3, 1.4e-4, 6e-5, 1e-5, 0.0)


@pytest.mark.parametrize("noise", NOISE)
def test_the_table_is_right_below_the_top_and_refused_past_it(noise):
    """Below the grid's top γ̂ is the derived one and the table is the unit means' least squares,
    on G − p = 48 df; past it the table is refused, its words naming the search's top, where on the
    base it served the standard errors at γ = 10⁸ (half the right ones) or refused only the df."""
    w, y, codes = _balanced(noise)
    gamma, ols = _derived(w, y, codes)
    table = mixed_table(pd.DataFrame({"w": w[codes]}), y, _clusters(codes))
    if gamma < GAMMA_GRID[-1] / 2:
        assert table.info["refused"] is None
        fit = fit_random_intercept(np.column_stack([np.ones(N), w[codes]]), y, codes)
        assert fit.sigma2_u / fit.sigma2_e == pytest.approx(gamma, rel=1e-6)
        np.testing.assert_allclose([r["estimate"] for r in table.rows], ols.params, rtol=1e-9)
        np.testing.assert_allclose([r["se"] for r in table.rows], ols.bse, rtol=1e-6)
        np.testing.assert_allclose([r["df"] for r in table.rows], G - 2, rtol=1e-6)
    else:
        assert gamma > 2 * GAMMA_GRID[-1]
        assert table.info["refused"] == BEYOND.format(G=G)
        assert all(r["se"] is None and r["ci_low"] is None for r in table.rows)
        assert [e["label"] for e in table.info["exits"]] == [
            "Combine each `pid`'s rows into one (the unit question, before the seal)",
            "Use the linear model"]


def test_a_row_level_column_beside_an_outcome_constant_within_units_is_refused_past_the_top():
    """30 units of 4 rows, an outcome constant within each, a unit-level `w` and a row-level `x`:
    REML's γ̂ is infinite, and the table is refused in the search's words; on the base it was
    refused in words about `x`'s Satterthwaite df."""
    rng = np.random.default_rng(1)
    units = 30
    codes = np.repeat(np.arange(units), M)
    w = rng.normal(size=units)
    y = (2 + 0.5 * w + rng.normal(size=units))[codes]
    matrix = pd.DataFrame({"w": w[codes], "x": rng.normal(size=units * M)})
    table = mixed_table(matrix, y, _clusters(codes))
    assert table.info["refused"] == BEYOND.format(G=units)


def test_the_slope_at_the_top_tends_to_the_derived_limit():
    """An outcome constant within units (s²_w = 0, so the within-unit RSS is rounding): the slope
    in log γ at 10⁸ is G − p_u − (N − P) = 50 − 2 − 198 = −150 to the 1/(m γ) the units add."""
    w, y, codes = _balanced(0.0)
    s = _sums(np.column_stack([np.ones(N), w[codes]]), y, codes)
    assert _slope(GAMMA_GRID[-1], s) == pytest.approx(G - 2 - (N - 2), rel=1e-6)


@pytest.mark.parametrize("gamma", [1e-2, 1.0, 1e3])
def test_the_slope_is_the_dense_profiles_derivative(gamma):
    """The closed form against central differences (±10⁻⁵ in log γ) of the REML profile written
    out with dense matrices (``test_no_standard_error_refusal._dense_profile``), on units with a
    row-level and a unit-level column."""
    from turbotab.core.tests.test_no_standard_error_refusal import _dense_profile

    rng = np.random.default_rng(6)
    codes = np.repeat(np.arange(12), 5)
    X = np.column_stack([np.ones(60), rng.normal(size=60), rng.normal(size=12)[codes]])
    y = X @ [1.0, 0.5, 0.3] + rng.normal(size=12)[codes] + rng.normal(size=60)
    h = 1e-5
    numeric = (_dense_profile(X, y, codes, gamma * math.exp(h))
               - _dense_profile(X, y, codes, gamma * math.exp(-h))) / (2 * h)
    assert _slope(gamma, _sums(X, y, codes)) == pytest.approx(numeric, rel=1e-5, abs=1e-6)


def test_the_regressor_fits_past_the_top_at_the_unit_means():
    """The regressor is not refused past the top: on balanced units with a unit-level column its
    fixed effects are the unit means' least squares at every γ, so the GLS fit at the top has
    REML's fixed effects, and its prediction is served (an outcome constant within units, γ̂
    infinite)."""
    w, y, codes = _balanced(0.0)
    _, ols = _derived(w, y, codes)
    model = RandomInterceptRegressor(units=pd.Series(codes)).fit(pd.DataFrame({"w": w[codes]}), y)
    np.testing.assert_allclose([model.intercept_, *model.coef_], ols.params, rtol=1e-9)
    assert fit_random_intercept(np.column_stack([np.ones(N), w[codes]]), y, codes).beyond
    assert not fit_random_intercept(np.column_stack([np.ones(N), w[codes]]),
                                    _balanced(1e-2)[1], codes).beyond


def test_the_profile_near_the_top_is_the_dense_ones():
    """Near the grid's top (within-unit noise of SD 1.4 × 10⁻⁴, γ̂ 4 × 10⁷) the profile computed
    here is the dense one's, so the search's top is evaluated as written."""
    from turbotab.core.tests.test_no_standard_error_refusal import _dense_profile

    w, y, codes = _balanced(1.4e-4)
    X = np.column_stack([np.ones(N), w[codes]])
    s = _sums(X, y, codes)
    for gamma in (1e6, GAMMA_GRID[-1]):
        ours = _profile(gamma, s) - _profile(1.0, s)
        dense = _dense_profile(X, y, codes, gamma) - _dense_profile(X, y, codes, 1.0)
        assert ours == pytest.approx(dense, rel=1e-6)
