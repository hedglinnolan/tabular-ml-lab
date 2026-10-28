"""An iterative fit's at-maximum distance, ``√(sᵀI⁻¹s)``, is taken from its score and the QR's
triangle, ``|R⁻ᵀs|``, never from a residual divided by a weight that can round to zero (V5c2-2; the
methods lead's ruling of 2026-10-28).

Where a row's fitted probability rounds to exactly 0 or 1 its weight ``μ(1 − μ)`` is zero, yet
Householder's QR leaves that row of Q at rounding (about 10⁻¹⁷ where it is among the first rows),
not at zero. Firth's distance divided that row's modified-score residual by ``√max(w, 10⁻³⁰⁰)``:
on ``xs = sign(y)·(10⁻⁴ + |d| − min|d|)`` beside `x1` and `x4` on 120 rows, a fit that had
converged to the maximum (32 steps) was refused as stopped short of it by "up to" a 101-digit
number of standard errors, with no estimate. The logistic and multinomial distances divided the
Pearson residual by the same clipped root: a converged logistic maximum-likelihood fit with one
outlying row (`x` = 12 where the slope is about 4, no event, its fitted probability 1.0) was
refused the same way. Rows whose weight is zero carry no information, and the score needs no
division: ``X·ᵀ(y − μ + h(½ − μ))`` for Firth's, ``Xᵀ(y − μ)`` for the logistic, each class's
``Xᵀ(y_k − π_k)`` for the multinomial. A distance too large to print reads as a number of bounded
length, interpolated from the value compared.

Truth: Firth's maximum by scipy's BFGS on the penalized log-likelihood written out with numpy (a
dense inverse of the information, ``expit``), from several starts; the logistic maximum by
statsmodels; each distance ``√(sᵀI⁻¹s)`` written out with numpy from the score and the information.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from turbotab.core.models import inference as I
from turbotab.core.models.inference import (INDEPENDENT, SHORT_OF_MAXIMUM, at_maximum,
                                            inference_table)
from turbotab.core.tests.test_firth_serves_its_estimates_alone_under_separation import (
    _design, _reference, words)

COLUMNS = ["x1", "xs", "x4"]


def _graded(seed: int, gap: float, spread: float, *, least: bool = False) -> pd.DataFrame:
    """120 rows whose `xs` separates the outcome completely, ``xs = sign(y)·(gap + spread·|d|)``
    (with ``least``, ``gap + |d| − min|d|``): far from the cut, a row's fitted probability rounds
    to exactly 0 or 1 at the penalized maximum."""
    rng = np.random.default_rng(100 + seed)
    n = 120
    x1, x4, d = rng.normal(size=(3, n))
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.5 * x1 - 0.4 * x4)))).astype(int)
    a = np.abs(d)
    xs = np.where(y == 1, 1.0, -1.0) * (gap + (a - a.min() if least else spread * a))
    return pd.DataFrame({"y": y, "x1": x1, "xs": xs, "x4": x4})


GRADED = {"the verifier's, gap 10⁻⁴ from the least": dict(seed=0, gap=1e-4, spread=1.0, least=True),
          "gap 0.05, spread 10": dict(seed=8, gap=0.05, spread=10.0),
          "gap 0.01, spread 3": dict(seed=8, gap=0.01, spread=3.0)}


def _firth_distance_by_hand(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    mu = expit(X @ beta)
    w = mu * (1 - mu)
    info = X.T @ (X * w[:, None])
    h = w * np.einsum("ij,jk,ik->i", X, np.linalg.inv(info), X)
    s = X.T @ (y - mu + h * (0.5 - mu))
    return math.sqrt(float(s @ np.linalg.solve(info, s)))


@pytest.mark.parametrize("design", list(GRADED))
def test_a_converged_firth_fit_whose_weights_round_to_zero_is_served_at_its_maximum(design):
    """Each of these separated designs has rows whose fitted probability rounds to exactly 0 or 1
    at the maximum; Firth's fit converges there, and the table serves its estimates, within 10⁻⁶
    of a standard error of the independent maximum, refusing only the profile (the data
    separate). On the base each was refused as stopped short of its maximum, with no estimate."""
    rows = _graded(**GRADED[design])
    X, y = _design(rows[COLUMNS]), rows["y"].to_numpy(dtype=float)
    beta, _, se = _reference(X, y)
    eta = X @ beta
    assert np.any(0.5 * (1.0 + np.tanh(0.5 * eta)) * (0.5 * (1.0 - np.tanh(0.5 * eta))) == 0.0)
    fit = I.firth_fit(X, y)
    assert fit.converged
    assert I.firth_distance(X, y, fit.beta) <= SHORT_OF_MAXIMUM
    table = inference_table("binary", rows[COLUMNS], rows["y"].to_numpy(), [0, 1], INDEPENDENT,
                            study=["x1"])
    assert table.info["refused"] == words(COLUMNS)
    for j, row in enumerate(table.rows):
        assert abs(row["estimate"] - beta[j]) <= 1e-6 * se[j], (design, row["feature"])


def test_firths_distance_is_the_modified_score_in_the_informations_metric_both_ways():
    """Where the fit stopped early (5 steps) the distance is far above the share and is
    ``√(sᵀI⁻¹s)`` written out with numpy to 10⁻⁶; at the maximum it is under the share. Both on a
    design with no weight at zero (`xs` 0.3 or more from the cut)."""
    rows = _graded(3, 0.3, 1.0)
    X, y = _design(rows[COLUMNS]), rows["y"].to_numpy(dtype=float)
    early = I.firth_fit(X, y, max_iter=5)
    assert not early.converged
    distance = I.firth_distance(X, y, early.beta)
    assert distance > 100 * SHORT_OF_MAXIMUM
    assert distance == pytest.approx(_firth_distance_by_hand(X, y, early.beta), rel=1e-6)
    assert I.firth_distance(X, y, I.firth_fit(X, y).beta) <= SHORT_OF_MAXIMUM


def test_a_firth_fit_stopped_short_of_its_maximum_is_still_refused(monkeypatch):
    """The guard still fires on the verifier's design: a fit stopped after 5 steps that claims
    convergence is refused as stopped short of its maximum, with no estimate, in words that state
    the distance the table compares (on its centered, scaled columns) as a number of bounded
    length."""
    rows = _graded(**GRADED["the verifier's, gap 10⁻⁴ from the least"])
    real = I.firth_fit

    def early(X: np.ndarray, y: np.ndarray, **kw: object) -> I.FirthFit:
        fit = real(X, y, max_iter=5)
        seen["distance"] = I.firth_distance(X, y, fit.beta)
        return I.FirthFit(beta=fit.beta, loglik=fit.loglik, cov=fit.cov,
                          iterations=fit.iterations, converged=True)

    seen: dict[str, float] = {}
    monkeypatch.setattr(I, "firth_fit", early)
    table = inference_table("binary", rows[COLUMNS], rows["y"].to_numpy(), [0, 1], INDEPENDENT,
                            study=["x1"])
    assert seen["distance"] > 1.0
    assert table.info["refused"] == at_maximum("Firth's penalized fit", seen["distance"])
    assert table.info["refused"].startswith("Firth's penalized fit stopped short of its maximum")
    assert all(r["estimate"] is None for r in table.rows)


def test_a_distance_too_large_to_print_reads_as_a_bounded_number():
    """The verifier's base distance, 2.89·10¹⁰¹, reads as "2.9 × 10¹⁰¹", never as its 102 digits;
    below a million the number is written out, below a thousand to two figures."""
    assert at_maximum("Firth's penalized fit", 2.886585886899571e101) == (
        "Firth's penalized fit stopped short of its maximum: where it stopped, one more Newton step "
        "would still move its estimates by up to 2.9 × 10¹⁰¹ of a standard error, and a table is "
        f"served only within 1/{round(1 / SHORT_OF_MAXIMUM):,} of one, so no estimate, interval or "
        "p-value is reported.")
    assert "by up to 3.4 × 10⁶ of" in at_maximum("the fit", 3.4e6)
    assert "by up to 999,999 of" in at_maximum("the fit", 999_999.4)
    assert "by up to 1,234 of" in at_maximum("the fit", 1234.2)
    assert "by up to 0.063 of" in at_maximum("the fit", 0.063)


# ── the logistic and multinomial distances ───────────────────────────────────


def _outlier(slope: float = 4.0, at: float = 12.0) -> tuple[pd.DataFrame, np.ndarray]:
    """3,000 rows of a logistic model with `x`'s slope ``slope``, the first row moved to `x` =
    ``at`` with no event: nothing separates, and at the maximum that row's fitted probability is
    1.0 to working precision (its linear predictor above 37)."""
    rng = np.random.default_rng(1)
    n = 3000
    x, z = rng.normal(size=(2, n))
    y = (rng.random(n) < 1 / (1 + np.exp(-(slope * x + 0.3 * z)))).astype(int)
    x[0], y[0] = at, 0
    return pd.DataFrame({"x": x, "z": z}), y


def test_a_logistic_maximum_with_a_row_whose_weight_rounds_to_zero_is_served():
    """statsmodels converges with the outlying row's fitted probability at 1.0; the table serves
    its estimates and standard errors (statsmodels' own, to 10⁻⁶), where the base refused it as
    stopped short of its maximum."""
    import statsmodels.api as sm

    matrix, y = _outlier()
    ml = sm.Logit(y, sm.add_constant(matrix)).fit(disp=0)
    assert ml.mle_retvals["converged"] and ml.predict()[0] == 1.0
    assert I.separated_columns(sm.add_constant(matrix).to_numpy(), y) == []
    table = inference_table("binary", matrix, y, [0, 1], INDEPENDENT, study=["x"])
    assert table.info["refused"] is None
    for row, b, s in zip(table.rows, ml.params, ml.bse):
        assert row["estimate"] == pytest.approx(b, abs=1e-6 * s), row["feature"]
        assert row["se"] == pytest.approx(s, rel=1e-6), row["feature"]


def _cr2_by_hand(X: np.ndarray, y: np.ndarray, beta: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """The logistic CR2 standard errors written out densely (Bell & McCaffrey 2002): ``W = √w·X``
    and Pearson residuals ``(y − μ)/√w`` with ``w = σ(η)σ(−η)`` taken from η, so no weight rounds
    to zero, ``A_g = (I − H_gg)^(−½)`` by an eigendecomposition of each cluster's ``n_g × n_g``
    matrix, and ``V = B (Σ_g W_gᵀ A_g e_g e_gᵀ A_g W_g) B`` with ``B = (WᵀW)⁻¹``."""
    eta = X @ beta
    w = expit(eta) * expit(-eta)
    W = X * np.sqrt(w)[:, None]
    e = (y - expit(eta)) / np.sqrt(w)
    B = np.linalg.inv(W.T @ W)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(codes):
        at = codes == g
        lam, V = np.linalg.eigh(np.eye(int(at.sum())) - W[at] @ B @ W[at].T)
        s = W[at].T @ (V @ np.diag(1 / np.sqrt(lam)) @ V.T) @ e[at]
        meat += np.outer(s, s)
    return np.sqrt(np.diag(B @ meat @ B))


@pytest.mark.parametrize("size, where", [(30, 0), (3, 0), (30, 1000), (2, 1001)])
def test_the_clustered_logistic_table_beside_that_row_serves_cr2_as_written_out(size, where):
    """With the rows clustered (``size`` to a cluster) and the outlying row at position
    ``where``, the CR2 standard errors are those written out densely to 10⁻⁶ (the estimates
    statsmodels'). Where the row came first its weight's zero kept rounding in its row of Q, and
    the base refused the table as stopped short of its maximum; served with the distance read
    from the score but Q not taken row by row, its standard errors were 10¹²⁷ and more. Where it
    came later the base served it, as now."""
    import statsmodels.api as sm

    matrix, y = _outlier()
    order = np.arange(len(y)) if where == 0 else np.r_[1:where + 1, 0, where + 1:len(y)]
    matrix, y = matrix.iloc[order].reset_index(drop=True), y[order]
    codes = np.arange(len(y)) // size
    clusters = I.Clusters(column="pid", codes=codes, n_clusters=int(codes.max()) + 1, n_missing=0)
    X = sm.add_constant(matrix).to_numpy()
    beta = sm.Logit(y, X).fit(disp=0).params
    table = inference_table("binary", matrix, y, [0, 1], clusters, study=["x"])
    assert table.info["refused"] is None and table.info["covariance"] == "CR2"
    for row, b, s in zip(table.rows, beta, _cr2_by_hand(X, y.astype(float), beta, codes)):
        assert row["estimate"] == pytest.approx(b, rel=1e-6, abs=1e-9), row["feature"]
        assert row["se"] == pytest.approx(s, rel=1e-6), row["feature"]


def test_the_logistic_distance_is_the_score_in_the_informations_metric_both_ways():
    """On the outlier's design: at statsmodels' maximum ``newton_distance`` (on the score) is under
    the share; moved off it, it is ``√(sᵀI⁻¹s)`` written out with numpy to 10⁻⁶."""
    import statsmodels.api as sm

    matrix, y = _outlier()
    X = sm.add_constant(matrix).to_numpy()
    beta = sm.Logit(y, X).fit(disp=0).params

    def distance(b: np.ndarray) -> tuple[float, float]:
        mu = 1 / (1 + np.exp(-(X @ b)))
        Xw, _ = I.logistic_working(X, y, mu)
        s = X.T @ (y - mu)
        info = X.T @ (X * (mu * (1 - mu))[:, None])
        return I.newton_distance(Xw, s), math.sqrt(float(s @ np.linalg.solve(info, s)))

    at, _ = distance(beta)
    assert at <= SHORT_OF_MAXIMUM
    moved, by_hand = distance(beta + np.array([0.02, -0.03, 0.01]))
    assert moved > 100 * SHORT_OF_MAXIMUM and moved == pytest.approx(by_hand, rel=1e-6)


def test_the_multinomial_distance_reads_each_classs_score_both_ways():
    """A multinomial working design with one row whose reference class's probability is 10⁻³⁰⁴
    (so its weight's smallest eigenvalue is clipped): the distance from the classes' scores is
    ``√(sᵀI⁻¹s)`` written out with numpy (the information ``Σ_i W_i ⊗ x_i x_iᵀ``, class-major)
    to 10⁻⁶, both at probabilities that are a maximum's (the score zero, the distance zero) and
    away from one."""
    rng = np.random.default_rng(4)
    n, K = 200, 3
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    eta = np.column_stack([np.zeros(n), X @ np.array([0.2, 0.5, -0.4]),
                           X @ np.array([-0.1, 0.3, 0.6])])
    eta[0] = [0.0, 700.0, 699.0]
    probs = np.exp(eta - eta.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    codes = np.array([int(rng.choice(K, p=p)) for p in probs])
    codes[0] = 1
    q, P = K - 1, X.shape[1]
    onehot = (codes[:, None] == np.arange(1, K)[None, :]).astype(float)
    s = ((onehot - probs[:, 1:]).T @ X).ravel()  # class-major, as the working design orders them
    info = np.zeros((q * P, q * P))
    for a in range(q):
        for b in range(q):
            cell = probs[:, a + 1] * ((a == b) - probs[:, b + 1])
            info[a * P:(a + 1) * P, b * P:(b + 1) * P] = X.T @ (X * cell[:, None])
    Xw, _ = I.multinomial_working(X, codes, probs)
    assert I.newton_distance(Xw, s) == pytest.approx(math.sqrt(float(s @ np.linalg.solve(info, s))),
                                                     rel=1e-6)
    assert I.newton_distance(Xw, np.zeros_like(s)) == 0.0
