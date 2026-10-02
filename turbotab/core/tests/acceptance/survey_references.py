"""Independent reference computations for the WP10 survey-design acceptance tests.

Each function computes a design-based quantity straight from its published definition, by a
different route from the engine's (``turbotab/core/models/survey.py``): the estimate by a
general-purpose optimizer (or the normal equations written out), the scores and information by
explicit per-row formulas, and the variance of the score total by Python loops over the strata and
the PSUs, in the form R's ``survey`` package documents for ``svyrecvar``/``svyglm`` (R is not
installed here). None of them imports the engine.

The variance of a total, Stata [SVY] *Variance estimation* equation (1) with no FPC:
``V̂(Ŷ) = Σ_h n_h/(n_h − 1) Σ_i (y_hi − ȳ_h)²``, ``y_hi`` the weighted total of PSU i in stratum h
and ``ȳ_h`` the stratum's mean PSU total. A regression's variance is the sandwich
``D V̂{Ĝ(β)} D′``, ``Ĝ(β) = Σ_j w_j d_j`` the weighted scores and ``D`` the inverse of the
weighted information ("Linearized/robust variance estimation", same entry).

A stratum with one PSU, R ``options(survey.lonely.psu = "adjust")``: "center the stratum at the
population mean rather than the stratum mean"; Stata ``singleunit(centered)``: "strata with one
sampling unit are centered at the population mean instead of the stratum mean. The quotient
n_h/(n_h − 1) in the variance formula is also taken to be 1 if n_h = 1." The population mean here
is the mean of all PSU totals.

Degrees of freedom: NHANES Analytic Guidelines 2011–2016 §3.2.3.2, "subtracting the number of strata
from the number of PSUs. If an analysis is performed on a subgroup of cases, the degrees of freedom
should be based on the number of strata and PSUs containing the observations of interest."
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np


def design_variance_by_definition(scores: np.ndarray, strata: Sequence[Any], psu: Sequence[Any],
                                  lonely: str = "adjust") -> np.ndarray:
    """``Σ_h c_h Σ_i (z_hi − m_h)(z_hi − m_h)′`` by explicit loops: PSU totals keyed by
    ``(stratum, psu)``, ``m_h`` the stratum mean, ``c_h = n_h/(n_h − 1)``; a single-PSU stratum
    is centered at the mean of all PSU totals with ``c_h = 1`` (``lonely="adjust"``) or left out
    (``lonely="remove"``, R's "ignore that PSU for variance computation")."""
    scores = np.asarray(scores, dtype=float)
    P = scores.shape[1]
    totals: dict[Any, dict[Any, np.ndarray]] = defaultdict(dict)
    for row, (h, i) in enumerate(zip(strata, psu)):
        stratum = totals[h]
        if i not in stratum:
            stratum[i] = np.zeros(P)
        stratum[i] = stratum[i] + scores[row]
    every = [t for stratum in totals.values() for t in stratum.values()]
    population_mean = sum(every) / len(every)
    meat = np.zeros((P, P))
    for stratum in totals.values():
        units = list(stratum.values())
        n_h = len(units)
        if n_h == 1:
            if lonely == "remove":
                continue
            d = units[0] - population_mean
            meat += np.outer(d, d)
            continue
        mean = sum(units) / n_h
        for t in units:
            meat += n_h / (n_h - 1) * np.outer(t - mean, t - mean)
    return meat


def design_df_by_definition(strata: Sequence[Any], psu: Sequence[Any], domain: Sequence[bool]) -> int:
    """PSUs minus strata, counting only those holding a domain row (the NCHS rule)."""
    psus = {(h, i) for h, i, d in zip(strata, psu, domain) if d}
    return len(psus) - len({h for h, _ in psus})


def wls_by_definition(X: np.ndarray, y: np.ndarray, w: np.ndarray, strata: Sequence[Any],
                      psu: Sequence[Any], domain: np.ndarray, lonely: str = "adjust"
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Survey-weighted least squares over the domain: ``β = (XᵀWX)⁻¹XᵀWy`` with W the diagonal of
    the weights (zero outside the domain), written out; scores ``w_i x_i (y_i − x_iᵀβ)``."""
    wd = np.where(domain, w, 0.0)
    W = np.diag(wd)
    A = X.T @ W @ X
    beta = np.linalg.solve(A, X.T @ W @ y)
    scores = X * (wd * (y - X @ beta))[:, None]
    meat = design_variance_by_definition(scores, strata, psu, lonely)
    Ainv = np.linalg.inv(A)
    return beta, Ainv @ meat @ Ainv


def logistic_by_definition(X: np.ndarray, y: np.ndarray, w: np.ndarray, strata: Sequence[Any],
                           psu: Sequence[Any], domain: np.ndarray, lonely: str = "adjust"
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Survey-weighted logistic regression: the weighted log-likelihood maximized by BFGS then
    Nelder–Mead (no Newton step of the engine's), scores ``w_i x_i (y_i − μ_i)``, information
    ``Σ w_i μ_i(1 − μ_i) x_i x_iᵀ``."""
    from scipy.optimize import minimize

    wd = np.where(domain, w, 0.0)
    scale = wd.sum() / domain.sum()
    wn = wd / scale

    def negloglik(b: np.ndarray) -> float:
        eta = X @ b
        return -float(np.sum(wn * (y * eta - np.logaddexp(0.0, eta))))

    def gradient(b: np.ndarray) -> np.ndarray:
        mu = 1.0 / (1.0 + np.exp(-(X @ b)))
        return -(X.T @ (wn * (y - mu)))

    best = minimize(negloglik, np.zeros(X.shape[1]), jac=gradient, method="BFGS",
                    options={"gtol": 1e-12, "maxiter": 10_000})
    beta = best.x
    mu = 1.0 / (1.0 + np.exp(-(X @ beta)))
    scores = X * (wd * (y - mu))[:, None]
    info = (X * (wd * mu * (1 - mu))[:, None]).T @ X
    meat = design_variance_by_definition(scores, strata, psu, lonely)
    inv = np.linalg.inv(info)
    return beta, inv @ meat @ inv


def multinomial_by_definition(X: np.ndarray, codes: np.ndarray, K: int, w: np.ndarray,
                              strata: Sequence[Any], psu: Sequence[Any], domain: np.ndarray,
                              lonely: str = "adjust") -> tuple[np.ndarray, np.ndarray]:
    """Survey-weighted multinomial logit against class 0, coefficients class by class: the
    weighted log-likelihood maximized by BFGS; scores ``w_i (y_i − π_i) ⊗ x_i``, information
    ``Σ w_i (diag π_i − π_i π_iᵀ) ⊗ x_i x_iᵀ``, both by explicit loops over rows."""
    from scipy.optimize import minimize

    n, P = X.shape
    q = K - 1
    wd = np.where(domain, w, 0.0)
    wn = wd / (wd.sum() / domain.sum())
    Y = (codes[:, None] == np.arange(1, K)[None, :]).astype(float)

    def unpack(theta: np.ndarray) -> np.ndarray:
        return theta.reshape(q, P).T  # (P, q), class by class in theta

    def probs(theta: np.ndarray) -> np.ndarray:
        eta = np.column_stack([np.zeros(n), X @ unpack(theta)])
        eta -= eta.max(axis=1, keepdims=True)
        e = np.exp(eta)
        return e / e.sum(axis=1, keepdims=True)

    def negloglik(theta: np.ndarray) -> float:
        eta = np.column_stack([np.zeros(n), X @ unpack(theta)])
        return -float(np.sum(wn * (np.sum(Y * eta[:, 1:], axis=1) - np.logaddexp.reduce(eta, axis=1))))

    def gradient(theta: np.ndarray) -> np.ndarray:
        pi = probs(theta)[:, 1:]
        return -np.concatenate([X.T @ (wn * (Y[:, a] - pi[:, a])) for a in range(q)])

    best = minimize(negloglik, np.zeros(q * P), jac=gradient, method="BFGS",
                    options={"gtol": 1e-11, "maxiter": 50_000})
    theta = best.x
    pi = probs(theta)[:, 1:]
    scores = np.zeros((n, q * P))
    info = np.zeros((q * P, q * P))
    for i in range(n):
        if wd[i] == 0:
            continue
        resid = Y[i] - pi[i]
        scores[i] = wd[i] * np.kron(resid, X[i])
        W = np.diag(pi[i]) - np.outer(pi[i], pi[i])
        info += wd[i] * np.kron(W, np.outer(X[i], X[i]))
    meat = design_variance_by_definition(scores, strata, psu, lonely)
    inv = np.linalg.inv(info)
    return theta, inv @ meat @ inv


__all__ = [
    "design_df_by_definition", "design_variance_by_definition", "logistic_by_definition",
    "multinomial_by_definition", "wls_by_definition",
]
