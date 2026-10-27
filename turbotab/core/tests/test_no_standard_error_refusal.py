"""An estimate is never served beside a missing interval (CI on Linux, 2026-10-23: the mixed model
on 5 units of 60 rows served `fiber`'s estimate with no interval, and nothing refused it).

Two parts. **At its source**, the mixed model's REML boundary: where the REML profile in
γ = σ²_u/σ²_e is lowest toward γ = 0, the fit compared the profile at 0 with a bounded search's
point a few 10⁻¹³ from it, two numbers that differ by less than their rounding. Where the point
came out lower, γ̂ was that point, the fit was not at the boundary, and the Satterthwaite df were
taken from a Hessian in log σ²_u that is flat there (of that case's quantities, the one whose
rounding can leave a coefficient no df, as `fiber` was left on CI). The fit now asks the profile's
derivative at γ = 0, in closed form (``repeated._slope_at_zero``); where it is not negative the
boundary is the estimate, as lme4 finds it. **The guard**: where a fit's covariance gives a
coefficient no finite, positive standard error, or a t interval no finite, positive degrees of
freedom, the table is refused in words that say what could not be computed
(``inference.no_interval``), not served with the estimate and no interval. It runs in the
least-squares (HC3), cluster-robust (CR2), logistic, multinomial, proportional-odds, Cox, GEE,
mixed-model and design-based tables; it is tested here on the four whose construction below is
exact on every platform (HC3, CR2, the mixed model and the design-based table), and on the rest,
from covariances constructed directly, in ``test_no_standard_error_guards.py``.

Truth for the boundary: statsmodels OLS, which a random-intercept REML fit at σ̂²_u = 0 equals
(the estimates, the standard errors, and N − P degrees of freedom); offline, lme4 2.0.6 with
lmerTest 3.2.1 on the same four constructions gave θ̂ = 0 (``isSingular`` TRUE), the fixed effects
and standard errors of ``lm``, and Satterthwaite df 96, 116, 132 and 202, each N − P (R is not on
CI). The slope's closed form is checked against central differences of the REML profile written
out with dense matrices. Truth for the guard: an outcome with no variation (y ≡ 0) leaves every
residual exactly zero, so the HC3, CR2 and design-based covariances are exactly zero, and so is
the REML residual variance, and with it the mixed model's covariance; IEEE arithmetic gives these
alike on every platform. An ordinary outcome leaves each table unrefused.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core.models.inference import Clusters, inference_table
from turbotab.core.models.repeated import _sums, fit_random_intercept, mixed_table


def _no_unit_effect(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """G units of m rows, 1–3 standard-normal columns, and an outcome with no unit effect at all
    (100 + the first column + noise of SD 5): REML's between-unit variance is at its boundary."""
    rng = np.random.default_rng(seed)
    G = int(rng.integers(3, 8))
    m = int(rng.integers(5, 40))
    N = G * m
    codes = np.repeat(np.arange(G), m)
    X = np.column_stack([np.ones(N), rng.normal(size=(N, int(rng.integers(1, 4))))])
    y = 100 + X[:, 1] + rng.normal(scale=0.5, size=N) * 10
    return X, y, codes


# The constructions where the base's comparison came out on the wrong side (γ̂ ≈ 10⁻¹³) on macOS;
# their profile's slope at γ = 0 is 51 to 103, far from its rounding.
BOUNDARY_SEEDS = (2, 16, 43, 100)


def _dense_profile(X: np.ndarray, y: np.ndarray, codes: np.ndarray, gamma: float) -> float:
    """−2 × the REML log-likelihood with σ²_e profiled out (no constant), written out with the
    dense ``H = I + γ Σ_g 1_g 1_gᵀ``: (N − P) log(rᵀH⁻¹r / (N − P)) + log|H| + log|XᵀH⁻¹X|."""
    N, P = X.shape
    H = np.eye(N) + gamma * (codes[:, None] == codes[None, :])
    Hi = np.linalg.inv(H)
    A = X.T @ Hi @ X
    beta = np.linalg.solve(A, X.T @ Hi @ y)
    r = y - X @ beta
    return ((N - P) * np.log(float(r @ Hi @ r) / (N - P)) + np.linalg.slogdet(H)[1]
            + np.linalg.slogdet(A)[1])


@pytest.mark.parametrize("seed", BOUNDARY_SEEDS)
def test_with_no_unit_effect_the_reml_fit_is_at_its_boundary_and_is_least_squares(seed):
    """Where the profile rises from γ = 0 the estimate is the boundary: σ̂²_u = 0, the estimates
    and standard errors are least squares' and the df N − P, as lme4 and lmerTest give them. On the
    base these four came out at γ̂ of about 10⁻¹³, off the boundary, with df not N − P."""
    X, y, codes = _no_unit_effect(seed)
    fit = fit_random_intercept(X, y, codes)
    assert fit.boundary and fit.sigma2_u == 0.0
    ols = sm.OLS(y, X).fit()
    np.testing.assert_allclose(fit.beta, ols.params, rtol=1e-10)
    np.testing.assert_allclose(np.sqrt(np.diag(fit.cov)), ols.bse, rtol=1e-8)
    np.testing.assert_allclose(fit.df, len(y) - X.shape[1], rtol=1e-8)


def _with_unit_effect() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The first construction with a unit effect of SD 3 added (against 5 within units)."""
    X, y, codes = _no_unit_effect(BOUNDARY_SEEDS[0])
    unit = np.random.default_rng(7).normal(scale=3.0, size=int(codes.max()) + 1)
    return X, y + unit[codes], codes


CASES = {**{f"no unit effect, seed {s}": (lambda s=s: _no_unit_effect(s)) for s in BOUNDARY_SEEDS},
         "a unit effect": _with_unit_effect}


@pytest.mark.parametrize("case", list(CASES))
def test_the_profiles_slope_at_zero_is_its_derivative(case):
    """The closed form against central differences (±10⁻⁶) of the dense profile at γ = 0, which
    is defined on both sides (H stays positive definite while 1 + m γ > 0): positive on the four
    constructions above, negative with a unit effect, where the fit then searches inside."""
    from turbotab.core.models.repeated import _slope_at_zero

    X, y, codes = CASES[case]()
    h = 1e-6
    numeric = (_dense_profile(X, y, codes, h) - _dense_profile(X, y, codes, -h)) / (2 * h)
    slope = _slope_at_zero(_sums(X, y, codes))
    assert slope == pytest.approx(numeric, rel=1e-4)
    assert (slope > 0) == (case != "a unit effect")


def test_with_a_unit_effect_the_fit_is_inside_and_not_least_squares():
    """The boundary's silent side: a real between-unit variance is estimated, off the boundary,
    at the minimum of the dense profile (found here by Brent's method over log γ)."""
    from scipy.optimize import minimize_scalar

    X, y, codes = _with_unit_effect()
    fit = fit_random_intercept(X, y, codes)
    assert not fit.boundary and fit.sigma2_u > 0
    dense = minimize_scalar(lambda z: _dense_profile(X, y, codes, float(np.exp(z))),
                            bounds=(-10.0, 5.0), method="bounded", options={"xatol": 1e-9})
    assert fit.sigma2_u / fit.sigma2_e == pytest.approx(float(np.exp(dense.x)), rel=1e-4)


# ── the guard ────────────────────────────────────────────────────────────────

G, M = 12, 5
N = G * M


def _rows() -> tuple[pd.DataFrame, np.ndarray, Clusters]:
    rng = np.random.default_rng(3)
    codes = np.repeat(np.arange(G), M)
    matrix = pd.DataFrame({"x": rng.normal(size=N), "z": rng.normal(size=N)})
    return matrix, codes, Clusters(column="pid", codes=codes, n_clusters=G, n_missing=0)


def _ordinary(matrix: pd.DataFrame, codes: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(4)
    return (1 + matrix["x"].to_numpy() + rng.normal(size=G)[codes] + rng.normal(size=N))


TABLES = {
    "least squares": lambda matrix, y, clusters: inference_table("regression", matrix, y, None),
    "cluster-robust": lambda matrix, y, clusters: inference_table("regression", matrix, y, None,
                                                                  clusters),
    "mixed": lambda matrix, y, clusters: mixed_table(matrix, y, clusters),
    "design-based": lambda matrix, y, clusters: _survey(matrix, y),
}


def _survey(matrix: pd.DataFrame, y: np.ndarray) -> object:
    """The linear family's design-based table over 2 strata of 6 PSUs (the 12 units), weights 1
    to 3."""
    from turbotab.core.models.survey import build_design, survey_table

    frame = matrix.copy()
    frame["stratum"] = np.repeat(np.arange(G) % 2, M)
    frame["psu"] = np.repeat(np.arange(G), M)
    weight = 1.0 + np.arange(N) % 3
    design = build_design(frame, weight, weight_column="w", strata_column="stratum",
                          psu_column="psu")
    return survey_table("regression", matrix, y, None, design)
TERMS = "the intercept, `x` and `z`"
WORDS = {
    "least squares": (f"The least-squares fit's HC3 covariance gives {TERMS} no finite, positive "
                      f"standard error, so no interval or p-value is reported."),
    "cluster-robust": (f"The cluster-robust (CR2) covariance by `pid` gives {TERMS} no finite, "
                       f"positive standard error, so no interval or p-value is reported."),
    "mixed": (f"The random-intercept model's REML fit by `pid` on these {G} units gives {TERMS} no "
              f"finite, positive standard error, so no interval or p-value is reported."),
    "design-based": (f"The design-based (Taylor-linearized) covariance gives {TERMS} no finite, "
                     f"positive standard error, so no interval or p-value is reported."),
}


@pytest.mark.parametrize("table", list(TABLES))
def test_a_fit_that_gives_no_standard_error_is_refused_in_words(table):
    """y ≡ 0: every residual is exactly zero. The HC3, CR2 and design-based covariances are
    exactly zero, and so is the mixed model's residual variance, and with it its covariance (where
    it was held at a floor of 10⁻³⁰⁰, which left each coefficient a standard error near 10⁻¹⁵²
    and, by underflow, no Satterthwaite df). The table is refused in words, keeping each estimate
    (zero) with no interval; on the base it was served unrefused, each estimate beside no
    interval."""
    matrix, codes, clusters = _rows()
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        result = TABLES[table](matrix, np.zeros(N), clusters)
    assert result.info["refused"] == WORDS[table]
    assert result.info["covariance"] == "none"
    assert result.info["caption"] == f"No intervals: {WORDS[table]}"
    assert [r["feature"] for r in result.rows] == ["(intercept)", "x", "z"]
    for row in result.rows:
        assert row["estimate"] == 0.0
        assert row["ci_low"] is None and row["ci_high"] is None and row["p"] is None


@pytest.mark.parametrize("table", list(TABLES))
def test_an_ordinary_fit_is_not_refused(table):
    """The guard's silent side: an outcome with a unit effect and noise gives every coefficient a
    standard error and an interval that holds its estimate."""
    matrix, codes, clusters = _rows()
    result = TABLES[table](matrix, _ordinary(matrix, codes), clusters)
    assert result.info["refused"] is None
    for row in result.rows:
        assert row["se"] > 0 and row["ci_low"] < row["estimate"] < row["ci_high"]


def test_the_words_name_each_coefficient_and_what_it_lacks():
    """Each coefficient with no finite, positive standard error is named; of the rest, each with
    no finite, positive degrees of freedom; past six names, the count of the others. Where every
    one has both, there are no words."""
    from turbotab.core.models.inference import no_interval

    names = ["(intercept)", "a", "b", "c"]
    assert no_interval(names, [0.1, 0.2, 0.3, 0.4], [9, 9, 9, 9], fit="The fit") is None
    assert no_interval(names, [0.1, 0.2, 0.3, 0.4], fit="The fit") is None
    assert no_interval(names, [0.1, np.nan, 0.0, 0.4], [9, 9, 9, -1.0], fit="The fit",
                       df_kind="Satterthwaite") == (
        "The fit gives `a` and `b` no finite, positive standard error, and `c` a standard error "
        "but no finite, positive Satterthwaite degrees of freedom, so no interval or p-value is "
        "reported.")
    assert no_interval(names, [np.inf, 0.2, 0.3, 0.4], [9, np.nan, 9, 9], fit="The fit") == (
        "The fit gives the intercept no finite, positive standard error, and `a` a standard "
        "error but no finite, positive degrees of freedom, so no interval or p-value is "
        "reported.")
    many = [f"x{j}" for j in range(9)]
    assert no_interval(many, np.zeros(9), fit="The fit") == (
        "The fit gives `x0`, `x1`, `x2`, `x3`, `x4`, `x5` and 3 more coefficients no finite, "
        "positive standard error, so no interval or p-value is reported.")
