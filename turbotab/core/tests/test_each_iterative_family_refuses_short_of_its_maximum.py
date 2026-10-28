"""Each iterative table's at-maximum guard fires (T5aV-3; ruling 2 of 2026-10-27): a fitter that
reports convergence 0.05 per coefficient off its maximum is refused with no estimate, in words
that interpolate its distance ``√(sᵀI⁻¹s)``, here written out with numpy from the score and the
information at the estimate it reported (the logistic, multinomial, proportional-odds, Firth,
survey multinomial in both forms and survey proportional-odds tables; the survey logistic's is
``test_iterative_tables_served_at_their_maximum``). The same table from the real fitter is
served (the silent side). With the distances returning 0, each of these tests fails; of the
tests at ac3d66da only the survey logistic's did.

The distance is a property of the fitted values and the columns' span, so it is written out on
the columns as they stand (``1, x1, d, x4``), from the fitted values at the estimate the fitter
reported on its own centered, scaled columns (the logistic, multinomial and Firth fits'); the
proportional-odds fits' and the survey multinomial's on the fitter's own columns, the survey
forms' weights scaled to sum to the rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.special import expit

from turbotab.core.models import inference as I
from turbotab.core.models import survey as S
from turbotab.core.models.inference import INDEPENDENT, Clusters, at_maximum, inference_table
from turbotab.core.models.ordinal import (
    ProportionalOddsRegression,
    ordinal_table,
    survey_ordinal_table,
)
from turbotab.core.models.survey import survey_table
from turbotab.core.tests.test_iterative_tables_served_at_their_maximum import G, N, _data

SHIFT = 0.05


def _matrix():
    x1, d, x4, *_ = _data()
    return pd.DataFrame({"x1": x1, "x2": d, "x4": x4}, index=pd.RangeIndex(N))


def _raw(*extra):
    x1, d, x4, *_ = _data()
    return np.column_stack([np.ones(N), x1, d, x4, *extra])


def _refused_at(table, fit: str, distance: float):
    assert distance > I.SHORT_OF_MAXIMUM
    assert table.info["refused"] == at_maximum(fit, distance), table.info["refused"]
    assert f"by up to {distance:.2g} of a standard error" in table.info["refused"]
    assert all(r["estimate"] is None and r["p"] is None for r in table.rows)


def _logistic_distance(X, y, mu):
    score = X.T @ (y - mu)
    info = X.T @ (X * (mu * (1 - mu))[:, None])
    return float(np.sqrt(score @ np.linalg.solve(info, score)))


def _multinomial_parts(X, codes, pi, w):
    """The weighted multinomial logit's score and information against class 0, class by class,
    at class probabilities ``pi`` (n × K)."""
    K = pi.shape[1]
    Y = (codes[:, None] == np.arange(K)[None, :]).astype(float)
    score = np.concatenate([X.T @ (w * (Y[:, a] - pi[:, a])) for a in range(1, K)])
    info = np.block([[X.T @ (X * (w * pi[:, a] * ((a == b) - pi[:, b]))[:, None])
                      for b in range(1, K)] for a in range(1, K)])
    return score, info


def _distance(score, info, w=None):
    scaled = 1.0 if w is None else len(w) / float(np.sum(w))
    return float(np.sqrt(score @ np.linalg.solve(info, score) * scaled))


@pytest.mark.parametrize("clustered", [False, True], ids=["model", "CR2"])
def test_a_logistic_fit_short_of_its_maximum_is_refused(monkeypatch, clustered):
    _, _, _, codes, yb, *_ = _data()
    clusters = Clusters(column="pid", codes=codes, n_clusters=G, n_missing=0) if clustered else \
        INDEPENDENT
    assert inference_table("binary", _matrix(), yb, [0, 1], clusters,
                           study=["x4"]).info["refused"] is None
    seen = {}

    class Short(sm.Logit):
        def fit(self, *args, **kwargs):
            result = super().fit(*args, **kwargs)
            result._results.params = result._results.params + SHIFT
            seen["mu"] = expit(np.asarray(self.exog) @ result._results.params)
            return result

    monkeypatch.setattr(sm, "Logit", Short)
    table = inference_table("binary", _matrix(), yb, [0, 1], clusters, study=["x4"])
    _refused_at(table, "the logistic model's maximum-likelihood fit",
                _logistic_distance(_raw(), yb.astype(float), seen["mu"]))


def test_a_multinomial_fit_short_of_its_maximum_is_refused(monkeypatch):
    *_, yo, _, _, _ = _data()
    assert inference_table("multiclass", _matrix(), yo, [0, 1, 2], INDEPENDENT,
                           study=["x4"]).info["refused"] is None
    seen = {}

    class Short(sm.MNLogit):
        def fit(self, *args, **kwargs):
            result = super().fit(*args, **kwargs)
            result._results.params = result._results.params + SHIFT
            eta = np.column_stack([np.zeros(N), np.asarray(self.exog) @ result._results.params])
            seen["pi"] = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
            return result

    monkeypatch.setattr(sm, "MNLogit", Short)
    table = inference_table("multiclass", _matrix(), yo, [0, 1, 2], INDEPENDENT, study=["x4"])
    score, info = _multinomial_parts(_raw(), yo.astype(np.int64), seen["pi"], np.ones(N))
    _refused_at(table, "the multinomial model's maximum-likelihood fit", _distance(score, info))


def _ordinal_score(params, X, codes, w, q):
    """The weighted cumulative-logit score over (β, θ), ``P(y ≤ j) = σ(θ_j − xᵀβ)``."""
    P = X.shape[1]
    beta, theta = params[:P], params[P:]
    eta = X @ beta
    cuts = np.concatenate([[-np.inf], theta, [np.inf]])
    Fa, Fb = expit(cuts[codes + 1] - eta), expit(cuts[codes] - eta)
    fa, fb = Fa * (1 - Fa), Fb * (1 - Fb)
    p = Fa - Fb
    score = np.zeros(P + q)
    score[:P] = -X.T @ (w * (fa - fb) / p)
    np.add.at(score, P + codes[codes < q], (w * fa / p)[codes < q])
    np.add.at(score, P + codes[codes > 0] - 1, -(w * fb / p)[codes > 0])
    return score


def _ordinal_distance(params, X, codes, w, scaled):
    """``√(sᵀ(−H)⁻¹s)``, the observed information by central differences of the score."""
    q = len(params) - X.shape[1]
    score = _ordinal_score(params, X, codes, w, q)
    h = 1e-6
    H = np.column_stack([(_ordinal_score(params + h * e, X, codes, w, q)
                          - _ordinal_score(params - h * e, X, codes, w, q)) / (2 * h)
                         for e in np.eye(len(params))])
    H = (H + H.T) / 2
    return _distance(score, -H, w if scaled else None)


def _short_ordinal(monkeypatch):
    seen = {}
    real = ProportionalOddsRegression.fit

    def short(self, X, y, sample_weight=None):
        real(self, X, y, sample_weight=sample_weight)
        self.coef_ = self.coef_ + SHIFT
        seen.update(X=np.asarray(X, dtype=float),
                    codes=np.searchsorted(self.classes_, np.asarray(y)).astype(np.int64),
                    w=np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight),
                    params=np.concatenate([self.coef_, self.thresholds_]))
        return self

    monkeypatch.setattr(ProportionalOddsRegression, "fit", short)
    return seen


@pytest.mark.parametrize("clustered", [False, True], ids=["model", "CR1"])
def test_a_proportional_odds_fit_short_of_its_maximum_is_refused(monkeypatch, clustered):
    _, _, _, codes, _, yo, *_ = _data()
    clusters = Clusters(column="pid", codes=codes, n_clusters=G, n_missing=0) if clustered else \
        INDEPENDENT
    assert ordinal_table(_matrix(), yo, [0, 1, 2], clusters, study=["x4"]).info["refused"] is None
    seen = _short_ordinal(monkeypatch)
    table = ordinal_table(_matrix(), yo, [0, 1, 2], clusters, study=["x4"])
    _refused_at(table, "the proportional-odds fit",
                _ordinal_distance(seen["params"], seen["X"], seen["codes"], seen["w"], False))


def test_a_firth_fit_short_of_its_maximum_is_refused(monkeypatch):
    *_, yf, f = _data()
    matrix = _matrix().assign(f=f)
    served = inference_table("binary", matrix, yf, [0, 1], INDEPENDENT, study=["x4"])
    # The silent side: at its maximum the table serves its estimates, refusing only the profile
    # (the data separate, T5d).
    assert served.info["estimator"] == I.FIRTH
    assert served.info["refused"].startswith("`f` separates the outcome: a cut on it")
    assert all(r["estimate"] is not None for r in served.rows)
    seen = {}
    real = I.firth_fit

    def short(X, y, **kwargs):
        fit = real(X, y, **kwargs)
        beta = fit.beta + SHIFT
        seen["mu"] = expit(np.asarray(X) @ beta)
        return I.FirthFit(beta=beta, loglik=fit.loglik, cov=fit.cov, iterations=fit.iterations,
                          converged=True)

    monkeypatch.setattr(I, "firth_fit", short)
    table = inference_table("binary", matrix, yf, [0, 1], INDEPENDENT, study=["x4"])
    X, mu, y = _raw(f), seen["mu"], yf.astype(float)
    info = X.T @ (X * (mu * (1 - mu))[:, None])
    h = mu * (1 - mu) * np.einsum("ij,jk,ik->i", X, np.linalg.inv(info), X)
    score = X.T @ (y - mu + h * (0.5 - mu))  # the modified score
    _refused_at(table, "Firth's penalized fit", _distance(score, info))
    assert table.concerns[0].startswith("`f` separates the outcome")


@pytest.mark.parametrize("task", ["multiclass", "ordinal"])
def test_a_survey_multinomial_fit_short_of_its_maximum_is_refused(monkeypatch, task):
    *_, yo, design, _, _ = _data()
    assert survey_table(task, _matrix(), yo, [0, 1, 2], design,
                        study=["x4"]).info["refused"] is None
    seen = {}
    real = S.weighted_multinomial

    def short(X, codes, K, w):
        fit = real(X, codes, K, w)
        estimate = fit.estimate + SHIFT
        P = X.shape[1]
        eta = np.column_stack([np.zeros(len(X)), X @ estimate.reshape(K - 1, P).T])
        pi = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
        Y = (codes[:, None] == np.arange(1, K)[None, :]).astype(float)
        rows = np.concatenate([X * (w * (Y[:, a] - pi[:, a + 1]))[:, None] for a in range(K - 1)],
                              axis=1)
        score, info = _multinomial_parts(X, codes, pi, w)
        assert np.allclose(rows.sum(axis=0), score)
        seen["distance"] = _distance(score, info, w)
        return S.WeightedFit(estimate, rows, info, True)

    monkeypatch.setattr(S, "weighted_multinomial", short)
    table = survey_table(task, _matrix(), yo, [0, 1, 2], design, study=["x4"])
    _refused_at(table, "the survey-weighted multinomial fit", seen["distance"])


def test_a_survey_proportional_odds_fit_short_of_its_maximum_is_refused(monkeypatch):
    *_, yo, design, _, _ = _data()
    assert survey_ordinal_table(_matrix(), yo, [0, 1, 2], design,
                                study=["x4"]).info["refused"] is None
    seen = _short_ordinal(monkeypatch)
    table = survey_ordinal_table(_matrix(), yo, [0, 1, 2], design, study=["x4"])
    _refused_at(table, "the survey-weighted proportional-odds fit",
                _ordinal_distance(seen["params"], seen["X"], seen["codes"], seen["w"], True))
