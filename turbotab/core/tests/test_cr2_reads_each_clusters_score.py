"""CR2's meat reads each cluster's score, never a row of Q times a residual divided by the root of
its weight, so the clustered logistic and multinomial tables do not depend on the rows' order
(T5dV-1).

Beside an event row whose fitted probability is tiny but not zero (its linear predictor −58 to
−76), the logistic working design's row is ``√w·x`` with ``√w`` 10⁻¹³ to 10⁻¹⁷ of the others',
and its Pearson residual ``(y − μ)/√w`` is 10¹³ to 10¹⁷. Householder's QR leaves rounding of the
longest row's scale in that row of Q, and the meat ``Σ_g (A_g Q_g)ᵀ e_g`` multiplied it by the
residual: with the row first in the data the CR2 standard errors were served up to 3% wrong (`x`'s
0.41218 where it is 0.42472), and with the same row 1,000 rows later they were right. Since T5d
the table is served there (its distance from the maximum read from the score), where the base
refused it as stopped short of its maximum. Now each cluster's term is taken by the push-through
identity, ``W_gᵀ A_g e_g = W_gᵀ e_g + Σ_k (f_k/λ_k) w_k w_kᵀ B W_gᵀ e_g`` over the eigenpairs of
``H_gg`` (``w_k = W_gᵀ u_k``), from the cluster's score ``Σ x_i (y_i − μ_i)``, which divides by no
weight.

Truth: the reference never divides by a weight either. The information ``XᵀWX`` (``w = σ(η)σ(−η)``
from η, by scipy) is factored by numpy's Cholesky, ``RᵀR``; each cluster's term is
``Rᵀ (I − G_g)^(−½) R⁻ᵀ X_gᵀ(y_g − μ_g)`` with ``G_g = R⁻ᵀ X_gᵀ W_g X_g R⁻¹`` (P × P, by numpy's
eigendecomposition), and ``V = B (Σ_g t_g t_gᵀ) B``, ``B = (XᵀWX)⁻¹``: by
``Mᵀ (I − MMᵀ)^(−½) = (I − MᵀM)^(−½) Mᵀ`` for ``M = W_g^½ X_g R⁻¹`` it is Bell & McCaffrey's CR2,
``B Σ_g (W_g^½X_g)ᵀ A_g e_g e_gᵀ A_g (W_g^½X_g) B``. The estimates are statsmodels'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.special import expit, softmax

from turbotab.core.models import inference as I
from turbotab.core.models.inference import inference_table


def _event_beside(at: float, where: int) -> tuple[pd.DataFrame, np.ndarray]:
    """3,000 rows of a logistic model with `x`'s slope 4 (`z`'s 0.3), the first row moved to `x` =
    ``at`` with the event, then placed at position ``where``: nothing separates, and that row's
    fitted probability at the maximum is tiny but not zero (its linear predictor about 3.75·at)."""
    rng = np.random.default_rng(1)
    n = 3000
    x, z = rng.normal(size=(2, n))
    y = (rng.random(n) < expit(4 * x + 0.3 * z)).astype(int)
    x[0], y[0] = at, 1
    order = np.arange(n) if where == 0 else np.r_[1:where + 1, 0, where + 1:n]
    return pd.DataFrame({"x": x[order], "z": z[order]}), y[order]


def _information(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``Σ_i W_i ⊗ x_i x_iᵀ``, class-major, for ``w`` the rows' weight blocks (n, q, q), or
    ``XᵀWX`` for ``w`` one weight a row (n,)."""
    if w.ndim == 1:
        return X.T @ (X * w[:, None])
    q, p = w.shape[1], X.shape[1]
    info = np.zeros((q * p, q * p))
    for a in range(q):
        for b in range(q):
            info[a * p:(a + 1) * p, b * p:(b + 1) * p] = X.T @ (X * w[:, a, b][:, None])
    return info


def _push_through(X: np.ndarray, w: np.ndarray, score_rows: np.ndarray,
                  codes: np.ndarray) -> np.ndarray:
    """The CR2 covariance from the information (:func:`_information`), each row's score (n × P)
    and the clusters, by the push-through identity: no division by a weight anywhere."""
    P = score_rows.shape[1]
    R = np.linalg.cholesky(_information(X, w)).T
    Rinv = np.linalg.inv(R)
    meat = np.zeros((P, P))
    for g in np.unique(codes):
        at = codes == g
        G = Rinv.T @ _information(X[at], w[at]) @ Rinv
        lam, V = np.linalg.eigh(np.eye(P) - (G + G.T) / 2)
        t = R.T @ (V @ np.diag(1 / np.sqrt(lam)) @ V.T) @ (Rinv.T @ score_rows[at].sum(axis=0))
        meat += np.outer(t, t)
    B = Rinv @ Rinv.T
    return B @ meat @ B


def _clusters(n: int, size: int) -> I.Clusters:
    codes = np.arange(n) // size
    return I.Clusters(column="pid", codes=codes, n_clusters=int(codes.max()) + 1, n_missing=0)


@pytest.mark.parametrize("size", [30, 3])
@pytest.mark.parametrize("where", [0, 1000])
@pytest.mark.parametrize("at", [-20.0, -19.0, -18.0, -17.0, -16.0])
def test_the_clustered_logistic_table_beside_a_tiny_probability_event_is_cr2_in_any_row_order(
        at, where, size):
    """With the event row first or 1,000 rows later, clustered in runs of ``size`` rows, the CR2
    standard errors are the push-through reference's to 10⁻⁹ (relative), the estimates
    statsmodels'. On the base the table was refused as stopped short of its maximum; since T5d it
    was served, and with the row first, wrong by 7·10⁻⁵ (at −16) to 3% (at −20)."""
    matrix, y = _event_beside(at, where)
    X = sm.add_constant(matrix).to_numpy()
    ml = sm.Logit(y, X).fit(disp=0, maxiter=100)
    assert ml.mle_retvals["converged"]
    beta = np.asarray(ml.params)
    eta = X @ beta
    row = where
    assert -76.5 < eta[row] < -58.0 and y[row] == 1  # the band, by construction
    clusters = _clusters(len(y), size)
    w = expit(eta) * expit(-eta)
    V = _push_through(X, w, X * (y - expit(eta))[:, None], clusters.codes)
    table = inference_table("binary", matrix, y, [0, 1], clusters, study=["x"])
    assert table.info["refused"] is None and table.info["covariance"] == "CR2"
    for j, r in enumerate(table.rows):
        assert r["estimate"] == pytest.approx(beta[j], rel=1e-6, abs=1e-9), r["feature"]
        assert r["se"] == pytest.approx(np.sqrt(V[j, j]), rel=1e-9), (r["feature"], at, where)


def test_where_no_weight_is_small_the_clustered_logistic_table_is_unchanged_cr2():
    """The silent side: on rows whose fitted probabilities are all between 10⁻⁴ and 1 − 10⁻⁴, the
    CR2 standard errors are the reference's to 10⁻¹⁰, and Bell–McCaffrey's df are those the
    dense ``n_g × n_g`` form gives (the meat's push-through leaves the df as they were)."""
    rng = np.random.default_rng(3)
    n = 600
    x, z = rng.normal(size=(2, n))
    y = (rng.random(n) < expit(0.8 * x - 0.5 * z)).astype(int)
    matrix = pd.DataFrame({"x": x, "z": z})
    X = sm.add_constant(matrix).to_numpy()
    beta = np.asarray(sm.Logit(y, X).fit(disp=0).params)
    eta = X @ beta
    assert np.all(np.abs(eta) < 9.0)
    clusters = _clusters(n, 6)
    w = expit(eta) * expit(-eta)
    V = _push_through(X, w, X * (y - expit(eta))[:, None], clusters.codes)
    table = inference_table("binary", matrix, y, [0, 1], clusters, study=["x"])
    assert table.info["covariance"] == "CR2"
    # Bell & McCaffrey's df, written out densely on the working design (no residual involved).
    Wd = X * np.sqrt(w)[:, None]
    Bd = np.linalg.inv(Wd.T @ Wd)
    M = np.eye(n) - Wd @ Bd @ Wd.T
    for j, r in enumerate(table.rows):
        assert r["se"] == pytest.approx(np.sqrt(V[j, j]), rel=1e-10), r["feature"]
        p = []
        for g in np.unique(clusters.codes):
            at = clusters.codes == g
            lam, E = np.linalg.eigh(np.eye(int(at.sum())) - Wd[at] @ Bd @ Wd[at].T)
            u = np.zeros(n)
            u[at] = (E @ np.diag(1 / np.sqrt(lam)) @ E.T) @ Wd[at] @ Bd[:, j]
            p.append(M @ u)
        Om = np.array([[a @ b for b in p] for a in p])
        assert r["df"] == pytest.approx(np.trace(Om) ** 2 / np.trace(Om @ Om), rel=1e-8)


def _multinomial_beside(at: float, where: int) -> tuple[pd.DataFrame, np.ndarray]:
    """2,400 rows of a three-class multinomial logit in `x` and `z`, one row moved to `x` = ``at``
    in the class whose fitted probability falls fastest there, placed at ``where``."""
    rng = np.random.default_rng(6)
    n = 2400
    x, z = rng.normal(size=(2, n))
    eta = np.column_stack([np.zeros(n), 0.2 + 3.0 * x + 0.3 * z, -0.1 + 1.5 * x - 0.4 * z])
    probs = softmax(eta, axis=1)
    y = np.array([int(rng.choice(3, p=p)) for p in probs])
    x[0], y[0] = at, 1
    order = np.arange(n) if where == 0 else np.r_[1:where + 1, 0, where + 1:n]
    return pd.DataFrame({"x": x[order], "z": z[order]}), y[order]


@pytest.mark.parametrize("where", [0, 900])
@pytest.mark.parametrize("at", [-14.0, -12.0])
def test_the_clustered_multinomial_table_beside_a_tiny_probability_class_is_cr2_in_any_order(
        at, where):
    """The same for the multinomial logit, whose working rows ``W_i^½ (I ⊗ x_iᵀ)`` and residuals
    ``W_i^(−½)(y_i − π_i)`` hold the row's tiny class probability: its CR2 standard errors, class
    by class, are the push-through reference's (each row's score ``(y_i − π_i) ⊗ x_i``, the
    information ``Σ W_i ⊗ x_i x_iᵀ``) to 10⁻⁹ in either order."""
    matrix, y = _multinomial_beside(at, where)
    X = sm.add_constant(matrix).to_numpy()
    ml = sm.MNLogit(y, X).fit(disp=0, maxiter=200)
    assert ml.mle_retvals["converged"]
    B = np.asarray(ml.params)  # (P, 2)
    eta = np.column_stack([np.zeros(len(y)), X @ B])
    probs = softmax(eta, axis=1)
    assert probs[where, 1] < 1e-12 and y[where] == 1
    pi = probs[:, 1:]
    W = np.einsum("ia,ab->iab", pi, np.eye(2)) - pi[:, :, None] * pi[:, None, :]
    onehot = (y[:, None] == np.arange(1, 3)[None, :]).astype(float)
    score_rows = np.einsum("ia,ij->iaj", onehot - pi, X).reshape(len(y), -1)  # class-major
    clusters = _clusters(len(y), 8)
    V = _push_through(X, W, score_rows, clusters.codes)
    table = inference_table("multiclass", matrix, y, [0, 1, 2], clusters, study=["x"])
    assert table.info["refused"] is None and table.info["covariance"] == "CR2"
    se = [r["se"] for r in table.rows]
    assert se == pytest.approx(list(np.sqrt(np.diag(V))), rel=1e-9), (at, where)
