"""Independent reference computations for the acceptance tests.

Each function computes a published quantity straight from its definition, by a different route
from the engine's (``turbotab/core/models/inference.py``): explicit n × n matrices where the
engine uses low-rank identities, closed forms where the literature gives one, and a generic
optimizer where the engine runs its own Newton iterations. None of them imports the engine.
"""
from __future__ import annotations

import numpy as np


def cr2_by_definition(X: np.ndarray, e: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CR2 covariance and Bell–McCaffrey df, with every matrix written out (small n only).

    Bell & McCaffrey (2002): ``A_g = (I − H_gg)^(−½)`` (a pseudo-inverse square root when
    singular), ``V = (XᵀX)⁻¹ Σ_g X_gᵀ A_g e_g e_gᵀ A_g X_g (XᵀX)⁻¹``. Satterthwaite df for
    coefficient j under the working model of independent, equal-variance rows (Pustejovsky &
    Tipton 2018, eq. 11): ``ν = tr(Ω)² / tr(Ω²)``, ``Ω = UᵀU``, ``u_g = (I − H) P_g``, ``P_g`` the
    n-vector holding ``A_g X_g (XᵀX)⁻¹ e_j`` on cluster g's rows and zero elsewhere.
    """
    n, P = X.shape
    M = np.linalg.pinv(X.T @ X)
    H = X @ M @ X.T
    eye = np.eye(n)
    adjust = {}
    meat = np.zeros((P, P))
    for g in np.unique(codes):
        idx = np.flatnonzero(codes == g)
        lam, Z = np.linalg.eigh(eye[np.ix_(idx, idx)] - H[np.ix_(idx, idx)])
        inv_root = np.where(lam > 1e-10, 1.0 / np.sqrt(np.clip(lam, 1e-10, None)), 0.0)
        A = Z @ np.diag(inv_root) @ Z.T
        adjust[g] = (idx, A)
        s = X[idx].T @ A @ e[idx]
        meat += np.outer(s, s)
    V = M @ meat @ M
    df = np.empty(P)
    residual_maker = eye - H
    for j in range(P):
        U = np.empty((n, len(adjust)))
        for k, (idx, A) in enumerate(adjust.values()):
            Pg = np.zeros(n)
            Pg[idx] = A @ X[idx] @ M[:, j]
            U[:, k] = residual_maker @ Pg
        omega = U.T @ U
        df[j] = np.trace(omega) ** 2 / np.trace(omega @ omega)
    return V, df


def bm_df_binary(g0: int, g1: int) -> float:
    """Imbens & Kolesár (2016, *Rev Econ Stat* 98:701), the Bell–McCaffrey df for a binary
    regressor with ``g0`` and ``g1`` (cluster) observations under homoskedasticity:
    ``K = (N0 + N1)² (N0 − 1)(N1 − 1) / (N1² (N1 − 1) + N0² (N0 − 1))``."""
    return (g0 + g1) ** 2 * (g0 - 1) * (g1 - 1) / (g1 ** 2 * (g1 - 1) + g0 ** 2 * (g0 - 1))


def hc3_by_definition(X: np.ndarray, e: np.ndarray) -> np.ndarray:
    """MacKinnon & White (1985) HC3: ``(XᵀX)⁻¹ Xᵀ diag(e_i² / (1 − h_i)²) X (XᵀX)⁻¹``."""
    M = np.linalg.inv(X.T @ X)
    h = np.einsum("ij,jk,ik->i", X, M, X)
    return M @ (X.T * (e / (1 - h)) ** 2) @ X @ M


def koenker_bp(resid: np.ndarray, X: np.ndarray) -> float:
    """Koenker's studentized Breusch–Pagan statistic: n R² of the squared residuals on X, and its
    χ² p-value with (columns − 1) degrees of freedom."""
    from scipy import stats

    u = resid ** 2
    fitted = X @ np.linalg.lstsq(X, u, rcond=None)[0]
    r2 = 1 - np.sum((u - fitted) ** 2) / np.sum((u - u.mean()) ** 2)
    return float(stats.chi2.sf(len(u) * r2, X.shape[1] - 1))


def penalized_loglik(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Firth (1993): the log-likelihood plus half the log-determinant of the Fisher information."""
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    _, logdet = np.linalg.slogdet(X.T @ (X * (mu * (1 - mu))[:, None]))
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)) + 0.5 * logdet)


def firth_by_optimizer(X: np.ndarray, y: np.ndarray, *, fixed: dict[int, float] | None = None,
                       start: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """The penalized-likelihood maximum by a general-purpose quasi-Newton optimizer (numerical
    gradients), optionally with some coefficients held fixed (for a profile)."""
    from scipy.optimize import minimize

    P = X.shape[1]
    fixed = fixed or {}
    free = [j for j in range(P) if j not in fixed]
    base = np.zeros(P) if start is None else np.array(start, dtype=float)
    for j, v in fixed.items():
        base[j] = v

    def full(z: np.ndarray) -> np.ndarray:
        b = base.copy()
        b[free] = z
        return b

    best = minimize(lambda z: -penalized_loglik(full(z), X, y), base[free], method="BFGS",
                    options={"gtol": 1e-9, "maxiter": 10_000})
    best = minimize(lambda z: -penalized_loglik(full(z), X, y), best.x, method="Nelder-Mead",
                    options={"xatol": 1e-11, "fatol": 1e-13, "maxiter": 40_000, "maxfev": 80_000})
    return full(best.x), -float(best.fun)


def haldane_log_odds_ratio(x: np.ndarray, y: np.ndarray) -> float:
    """The log odds ratio of a 2 × 2 table with ½ added to every cell (Haldane 1956), which is
    Firth's estimate for a single binary covariate with an intercept (Firth 1993 §3)."""
    a = np.sum((x == 1) & (y == 1)) + 0.5
    b = np.sum((x == 1) & (y == 0)) + 0.5
    c = np.sum((x == 0) & (y == 1)) + 0.5
    d = np.sum((x == 0) & (y == 0)) + 0.5
    return float(np.log(a * d / (b * c)))
