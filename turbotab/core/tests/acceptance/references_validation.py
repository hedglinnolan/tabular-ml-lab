"""Independent reference computations for the WP9 acceptance tests (validation and performance).

Each function computes a published quantity straight from its definition, by a different route
from the engine's (``turbotab/core/models/performance.py``, ``validation.py``): explicit pairwise
comparison matrices where the engine uses midranks, a loop over clusters where it sums influence
values, and a local regression written out point by point where it calls a library smoother. None
of them imports the engine.
"""
from __future__ import annotations

import math

import numpy as np


def psi(x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
    """DeLong et al. (1988)'s kernel: 1 when the positive outscores the negative, ½ on a tie."""
    return np.where(x > y, 1.0, np.where(x == y, 0.5, 0.0))


def delong_by_definition(positive: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """AUC and its variance, DeLong, DeLong & Clarke-Pearson (Biometrics 1988;44:837).

    With m positives X and n negatives Y: ``θ = (1/mn) Σ_i Σ_j ψ(X_i, Y_j)``;
    ``V10(X_i) = (1/n) Σ_j ψ(X_i, Y_j)``, ``V01(Y_j) = (1/m) Σ_i ψ(X_i, Y_j)``;
    ``S10 = Σ (V10 − θ)² / (m − 1)``, ``S01 = Σ (V01 − θ)² / (n − 1)``; ``Var θ = S10/m + S01/n``.
    The m × n comparison matrix is written out (small samples only).
    """
    pos = np.asarray(positive, dtype=bool)
    s = np.asarray(score, dtype=float)
    X, Y = s[pos], s[~pos]
    m, n = len(X), len(Y)
    K = psi(X[:, None], Y[None, :])
    theta = float(K.mean())
    v10, v01 = K.mean(axis=1), K.mean(axis=0)
    s10 = float(((v10 - theta) ** 2).sum() / (m - 1))
    s01 = float(((v01 - theta) ** 2).sum() / (n - 1))
    return theta, s10 / m + s01 / n


def obuchowski_clustered(positive: np.ndarray, score: np.ndarray, cluster: np.ndarray) -> tuple[float, float]:
    """AUC and its variance with clustered rows, Obuchowski (Biometrics 1997;53:567).

    For cluster i with m_i positives and n_i negatives (M and N in all): ``V10_i· = Σ_j V10(X_ij)``,
    ``V01_i· = Σ_k V01(Y_ik)``; with I clusters,
    ``S10 = I/((I − 1)M) Σ_i (V10_i· − m_i θ)²``, ``S01 = I/((I − 1)N) Σ_i (V01_i· − n_i θ)²``,
    ``S11 = I/(I − 1) Σ_i (V10_i· − m_i θ)(V01_i· − n_i θ)``;
    ``Var θ = S10/M + S01/N + 2 S11/(MN)``. Written as a loop over clusters.
    """
    pos = np.asarray(positive, dtype=bool)
    s = np.asarray(score, dtype=float)
    c = np.asarray(cluster)
    X, Y = s[pos], s[~pos]
    cx, cy = c[pos], c[~pos]
    M, N = len(X), len(Y)
    K = psi(X[:, None], Y[None, :])
    theta = float(K.mean())
    v10, v01 = K.mean(axis=1), K.mean(axis=0)
    labels = list(dict.fromkeys(c.tolist()))
    I = len(labels)
    s10 = s01 = s11 = 0.0
    for g in labels:
        a = float(v10[cx == g].sum() - (cx == g).sum() * theta)
        b = float(v01[cy == g].sum() - (cy == g).sum() * theta)
        s10 += a * a
        s01 += b * b
        s11 += a * b
    s10 *= I / ((I - 1) * M)
    s01 *= I / ((I - 1) * N)
    s11 *= I / (I - 1)
    return theta, s10 / M + s01 / N + 2 * s11 / (M * N)


def lowess_by_definition(x: np.ndarray, y: np.ndarray, at: np.ndarray, frac: float = 2 / 3) -> np.ndarray:
    """Cleveland's (JASA 1979;74:829) locally weighted linear fit, no robustness iterations,
    evaluated at each point of ``at`` (themselves values of ``x``).

    The neighborhood is the ``r = round(frac · n)`` nearest x (R's ``lowess``: ``ns = max(min(
    round(f·n), n), 2)``); ``h`` is the distance to the r-th nearest; the weights are tricube,
    ``(1 − (d/h)³)³`` for d < h; the fit is weighted least squares of y on (1, x). Every point is
    fitted exactly (R's default ``delta = 0.01·range`` instead interpolates between points closer
    than delta; the engine follows R, so the two differ by that interpolation only).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    r = max(min(int(round(frac * n)), n), 2)
    out = np.empty(len(at))
    for k, x0 in enumerate(np.asarray(at, dtype=float)):
        d = np.abs(x - x0)
        h = np.sort(d)[r - 1]
        if h <= 0:
            out[k] = float(y[d == 0].mean())
            continue
        u = d / h
        w = np.where(u < 1, (1 - u ** 3) ** 3, 0.0)
        sw = w.sum()
        xm = (w * x).sum() / sw
        ym = (w * y).sum() / sw
        sxx = (w * (x - xm) ** 2).sum()
        if sxx <= 1e-12 * max(1.0, (w * x * x).sum()):
            out[k] = ym
        else:
            slope = (w * (x - xm) * (y - ym)).sum() / sxx
            out[k] = ym + slope * (x0 - xm)
    return out


def dersimonian_laird_prediction_interval(estimates: np.ndarray, variances: np.ndarray,
                                          level: float = 0.95) -> tuple[float, float, float, float]:
    """(μ, τ², low, high): the DerSimonian–Laird mean and the prediction interval for a new
    cluster, Higgins, Thompson & Spiegelhalter (JRSS A 2009;172:137):
    ``μ ± t_{k−2} √(τ² + SE(μ)²)``. Written from the moment estimator directly."""
    from scipy import stats

    th = np.asarray(estimates, dtype=float)
    v = np.asarray(variances, dtype=float)
    k = len(th)
    w = 1 / v
    fixed = (w * th).sum() / w.sum()
    q = (w * (th - fixed) ** 2).sum()
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    mu = (ws * th).sum() / ws.sum()
    se = math.sqrt(1 / ws.sum())
    half = stats.t.ppf(0.5 + level / 2, k - 2) * math.sqrt(tau2 + se ** 2)
    return float(mu), float(tau2), float(mu - half), float(mu + half)
