"""The mixed model's Satterthwaite degrees of freedom near the REML boundary, where the between-unit
variance is small but not zero (γ̂ = σ̂²_u/σ̂²_e of 10⁻⁴ to 10⁻³).

The df rest on the observed REML information in (σ²_u, σ²_e). The engine took it by central
differences of the log-likelihood at a fixed step of 10⁻⁴ in log σ²; near the boundary the
curvature in log σ²_u is of the size of the rounding, so the df depended on the rounding: on the
construction below with γ̂ = 7.3 × 10⁻⁵ the base gave a unit-level coefficient 27.2 df where
lmerTest gives 7.9, a t interval too narrow, and with γ̂ = 7.3 × 10⁻⁴ a one-ulp change in every
outcome moved the df by 9%. The information is now computed in closed form
(``repeated.satterthwaite_df``).

Truth: lmerTest 3.2.1 on lme4 2.0.6 (R 4.6.1), offline, on these rows written with 17 significant
digits: ``lmer(y ~ x + w + (1 | g), REML = TRUE)``, nloptwrap with ``xtol_abs = 1e-14,
ftol_abs = 1e-16, xtol_rel = 1e-14, ftol_rel = 1e-16`` (bobyqa with ``rhoend = 1e-14`` agrees to
4 × 10⁻⁵); R is not on CI. On CI, the information written out with dense matrices from its
definition, ``I_kl = yᵀPV_kPV_lPy − ½ tr(PV_kPV_l)`` with ``P = V⁻¹ − V⁻¹X(XᵀV⁻¹X)⁻¹XᵀV⁻¹``
(Harville 1977), which is itself checked against central differences of the dense REML
log-likelihood at an interior fit.
"""
from __future__ import annotations

import numpy as np
import pytest

from turbotab.core.models.repeated import fit_random_intercept


def _unit_level(effect: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """10 units of 20 rows: a row-level covariate `x`, a unit-level one `w` (one value per unit),
    noise of SD 2, and a unit effect ``effect`` times a centered standard normal per unit. At
    ``effect`` 0.26 and 0.27 REML's γ̂ is 7.3 × 10⁻⁵ and 7.3 × 10⁻⁴; at 1 it is 0.18, inside."""
    rng = np.random.default_rng(11)
    G, m = 10, 20
    codes = np.repeat(np.arange(G), m)
    w = rng.normal(size=G)[codes]
    x = rng.normal(size=G * m)
    unit = rng.normal(size=G)
    y = 5 + 0.5 * x + 0.3 * w + 2 * rng.normal(size=G * m) + effect * (unit - unit.mean())[codes]
    return np.column_stack([np.ones(G * m), x, w]), y, codes


# effect: (lme4's γ̂, lmerTest's df for the intercept, `x` and `w`)
LMERTEST = {0.26: (7.319674e-05, (7.91061, 196.91921, 7.93635)),
            0.27: (7.312078e-04, (7.91338, 196.94016, 7.93883)),
            1.0: (1.761197e-01, (8.00053, 192.43876, 8.00671))}


@pytest.mark.parametrize("effect", list(LMERTEST))
def test_the_df_are_lmertests_near_the_boundary_and_inside(effect):
    """Near the boundary (0.26, 0.27) and inside it (1, whose df were right before and still are),
    γ̂ is lme4's and the df lmerTest's, to its numerical Hessian's accuracy (2 × 10⁻⁴). On the base
    the unit-level coefficient `w` had 27.2 and 8.36 df near the boundary, against 7.94."""
    X, y, codes = _unit_level(effect)
    fit = fit_random_intercept(X, y, codes)
    gamma, df = LMERTEST[effect]
    assert not fit.boundary
    assert fit.sigma2_u / fit.sigma2_e == pytest.approx(gamma, rel=1e-3)
    np.testing.assert_allclose(fit.df, df, rtol=2e-4)


def _dense(X: np.ndarray, y: np.ndarray, codes: np.ndarray, s2u: float, s2e: float
           ) -> tuple[np.ndarray, np.ndarray]:
    """The Satterthwaite df at (σ²_u, σ²_e) with every matrix written out, and the observed
    information they rest on: ``ν_j = 2 C_jj² / (∇C_jjᵀ I⁻¹ ∇C_jj)``, ``∂C/∂θ_k =
    C XᵀV⁻¹V_kV⁻¹X C``, ``V_u = ZZᵀ``, ``V_e = I``."""
    ZZ = (codes[:, None] == codes[None, :]).astype(float)
    V = s2u * ZZ + s2e * np.eye(len(y))
    Vi = np.linalg.inv(V)
    C = np.linalg.inv(X.T @ Vi @ X)
    P = Vi - Vi @ X @ C @ X.T @ Vi
    Py = P @ y
    Vk = (ZZ, np.eye(len(y)))
    info = np.array([[Py @ Vk[k] @ P @ Vk[l] @ Py - 0.5 * np.trace(P @ Vk[k] @ P @ Vk[l])
                      for l in range(2)] for k in range(2)])
    grad = np.stack([np.diag(C @ X.T @ Vi @ Vk[k] @ Vi @ X @ C) for k in range(2)], axis=1)
    df = 2 * np.diag(C) ** 2 / np.einsum("ja,ab,jb->j", grad, np.linalg.inv(info), grad)
    return df, info


def _dense_loglik(X: np.ndarray, y: np.ndarray, codes: np.ndarray, s2u: float, s2e: float) -> float:
    """The REML log-likelihood ``−½[log|V| + log|XᵀV⁻¹X| + rᵀV⁻¹r]``, written out."""
    V = s2u * (codes[:, None] == codes[None, :]) + s2e * np.eye(len(y))
    Vi = np.linalg.inv(V)
    A = X.T @ Vi @ X
    r = y - X @ np.linalg.solve(A, X.T @ Vi @ y)
    return -0.5 * (np.linalg.slogdet(V)[1] + np.linalg.slogdet(A)[1] + float(r @ Vi @ r))


@pytest.mark.parametrize("effect", list(LMERTEST))
def test_the_df_are_the_observed_information_written_out(effect):
    """At the engine's estimate, its df are the dense definition's (to 10⁻⁹). On the base, at
    0.26, they were 27.2 for `w` against 7.94."""
    X, y, codes = _unit_level(effect)
    fit = fit_random_intercept(X, y, codes)
    df, _ = _dense(X, y, codes, fit.sigma2_u, fit.sigma2_e)
    np.testing.assert_allclose(fit.df, df, rtol=1e-9)


def test_the_dense_information_is_the_negative_hessian():
    """The reference checked: at the interior fit (effect 1) the dense information is the negative
    Hessian of the dense REML log-likelihood, by central differences of 10⁻³ of each variance
    (to 10⁻⁴)."""
    X, y, codes = _unit_level(1.0)
    fit = fit_random_intercept(X, y, codes)
    theta = np.array([fit.sigma2_u, fit.sigma2_e])
    _, info = _dense(X, y, codes, *theta)
    h = 1e-3 * theta
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            e_i, e_j = np.eye(2)[i] * h[i], np.eye(2)[j] * h[j]
            H[i, j] = (_dense_loglik(X, y, codes, *(theta + e_i + e_j))
                       - _dense_loglik(X, y, codes, *(theta + e_i - e_j))
                       - _dense_loglik(X, y, codes, *(theta - e_i + e_j))
                       + _dense_loglik(X, y, codes, *(theta - e_i - e_j))) / (4 * h[i] * h[j])
    np.testing.assert_allclose(info, -H, rtol=1e-4)


@pytest.mark.parametrize("effect", list(LMERTEST))
def test_a_one_ulp_change_in_the_outcome_leaves_the_df_where_they_were(effect):
    """The df do not rest on the rounding: every outcome moved up by one ulp moves no df by
    10⁻⁵ of itself (γ̂ moves within the search's tolerance). On the base the change moved the df
    at 0.27 by 9% (8.36 to 7.70 for `w`) and at 0.26 by 3 × 10⁻⁵."""
    X, y, codes = _unit_level(effect)
    before = fit_random_intercept(X, y, codes).df
    after = fit_random_intercept(X, np.nextafter(y, np.inf), codes).df
    np.testing.assert_allclose(after, before, rtol=1e-5)


def test_a_unit_code_no_row_has_changes_nothing():
    """REML is a sum over the units that have rows: numbering the 10 units 0, 2, …, 18 (the odd
    codes have no row, as a subset of the rows can leave them) gives the same estimates and the
    same df, to rounding."""
    X, y, codes = _unit_level(0.27)
    fit = fit_random_intercept(X, y, codes)
    gapped = fit_random_intercept(X, y, 2 * codes)
    np.testing.assert_allclose(gapped.beta, fit.beta, rtol=1e-12)
    np.testing.assert_allclose(gapped.df, fit.df, rtol=1e-12)
