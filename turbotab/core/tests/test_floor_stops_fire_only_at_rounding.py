"""A fit stops at rounding's floor only where rounding is the cause (T5aV-1, T5aV-2).

Firth's steps are the Fisher information's, not the penalized log-likelihood's Hessian, so near
its maximum each closes the distance only by a share, and the decrement ``Uᵀ I⁻¹ U`` understates
the distance left by about ``1/(1 − rate)``. At ac3d66da a Firth fit also stopped where the
decrement fell by less than half in a step at most ``SHORT_OF_MAXIMUM`` squared, read as
rounding's floor; on a completely separated design that is the rate itself, and the table served
`x1` at 0.289998 against brglm2's 0.289514 (5.9·10⁻⁴ of a standard error). It now converges only
at its tolerance.

The survey fits' steps are Newton's on the likelihood's own Hessian, so their decrement's root is
the distance in standard errors to second order. Their floor fires where the decrement, under the
share squared, stopped falling (``survey._at_floor``): beside ``x1 + ε·d`` rounding stalls it
there, and without the floor the survey logistic table is refused as unconverged at whichever ε
rounding stalls on the platform, or served right. Where it still
falls, however slowly (each step cut to a quarter, here), the fit keeps stepping to its maximum;
at ac3d66da the floor fired at a fall of less than half a step and stopped 5·10⁻⁵ of a standard
error short.

Truth: brglm2 1.1.0 (R 4.6.1), ``glm(y ~ x1 + xs + x4, family = binomial, method = "brglmFit",
type = "AS_mean", control = brglmControl(epsilon = 1e-14, maxit = 100000))`` on the rows
:func:`_separated` builds (written with ``%.17g``); statsmodels' weighted GLM and a numpy Newton
iteration for the weighted multinomial logit, on ``(1, x1, d, x4)``, whose `x4` coefficient is the
near design's (``x1`` and ``x1 + ε·d`` span what ``x1`` and ``d`` do).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core.models import inference as I
from turbotab.core.models import survey as S
from turbotab.core.models.inference import (
    INDEPENDENT,
    SHORT_OF_MAXIMUM,
    at_maximum,
    inference_table,
)
from turbotab.core.models.survey import survey_table
from turbotab.core.tests.test_iterative_tables_served_at_their_maximum import N, _data
from turbotab.core.tests.test_firth_profile_under_complete_separation import WORDS

# brglm2's AS_mean fit on _separated(seed): (estimates, standard errors) of the intercept, x1, xs
# and x4.
BRGLM2 = {
    3: ([0.154936214838797, 0.141188553760976, 8.0033428652726, -0.443419478255955],
        [0.731629280012707, 0.792621271575366, 1.63020083579402, 0.73868400556734]),
    4: ([0.0459557258328506, 0.289514183007937, 8.36707989047361, -0.033489470094443],
        [0.734116112377647, 0.813842791578534, 1.65146800598651, 0.738775024500505]),
    5: ([-0.280481136211365, 0.282673158070255, 8.89645750649193, 0.0350559582313864],
        [0.758795200172838, 0.708844155425905, 1.76154128784312, 0.662617768963531]),
}


def _separated(seed: int) -> pd.DataFrame:
    """300 rows whose `xs` separates the outcome completely: ``xs = sign(y)·(0.3 + |d|)``."""
    rng = np.random.default_rng(100 + seed)
    x1, x4, d = rng.normal(size=(3, 300))
    y = (rng.random(300) < 1 / (1 + np.exp(-(0.5 * x1 - 0.4 * x4)))).astype(int)
    xs = np.where(y == 1, 1.0, -1.0) * (0.3 + np.abs(d))
    return pd.DataFrame({"y": y, "x1": x1, "xs": xs, "x4": x4})


@pytest.mark.parametrize("seed", sorted(BRGLM2))
def test_a_completely_separated_firth_table_serves_brglm2s_maximum(seed):
    """Each coefficient within 10⁻⁷ of a standard error of brglm2's (the share claimed is 10⁻⁴),
    and the fit's standard error to 10⁻⁷: the decrement falls to 0.52 to 0.72 of itself a step
    here, and at ac3d66da the floor stop served these tables 2.5·10⁻⁴ to 5.9·10⁻⁴ of a standard
    error off. The separation is complete, so the table reports the estimates alone, no profile
    interval or p-value (``inference.complete_separation_words``, T5aV2-1), and the standard error
    is read off the fit, whose information the table no longer reports."""
    rows = _separated(seed)
    table = inference_table("binary", rows[["x1", "xs", "x4"]], rows["y"].to_numpy(), [0, 1],
                            INDEPENDENT, study=["x1"])
    assert table.info["estimator"] == I.FIRTH
    assert table.info["refused"] == WORDS
    want, se = BRGLM2[seed]
    served = {r["feature"]: r for r in table.rows}
    for j, name in enumerate(["(intercept)", "x1", "xs", "x4"]):
        assert abs(served[name]["estimate"] - want[j]) <= 1e-7 * se[j], (name, served[name])
        assert served[name]["ci_low"] is None and served[name]["p"] is None, name
    assert abs(served["x1"]["estimate"] - want[1]) <= SHORT_OF_MAXIMUM * se[1]
    X = np.column_stack([np.ones(300), rows[["x1", "xs", "x4"]].to_numpy()])
    fit = I.firth_fit(X, rows["y"].to_numpy(dtype=float))
    np.testing.assert_allclose(np.sqrt(np.diag(fit.cov)), se, rtol=1e-7)


def test_a_firth_fit_converges_only_at_its_tolerance():
    """On the fit itself: converged where its decrement ``Uᵀ I⁻¹ U`` (the modified score and the
    information written out with numpy) is under ``tol``² = 10⁻¹⁶, not where it first fell under
    the share squared (10⁻⁸), as the floor stop called it at ac3d66da."""
    rows = _separated(4)
    X = np.column_stack([np.ones(300), rows[["x1", "xs", "x4"]].to_numpy()])
    y = rows["y"].to_numpy(dtype=float)
    fit = I.firth_fit(X, y)
    assert fit.converged
    mu = 1 / (1 + np.exp(-(X @ fit.beta)))
    info = X.T @ (X * (mu * (1 - mu))[:, None])
    h = (mu * (1 - mu)) * np.einsum("ij,jk,ik->i", X, np.linalg.inv(info), X)
    score = X.T @ (y - mu + h * (0.5 - mu))
    assert float(score @ np.linalg.solve(info, score)) < 1e-15


@pytest.mark.parametrize("eps", [3e-7, 2e-7, 1.05e-7])
def test_a_survey_fit_rounding_stalls_is_served_at_its_maximum(eps):
    """The floor fires: beside ``x1 + ε·d`` the survey logistic table is served, `x4` within
    10⁻⁶ of a standard error of statsmodels' weighted maximum on ``(1, x1, d, x4)``."""
    import statsmodels.api as sm

    x1, d, x4, _, yb, _, design, *_ = _data()
    want = sm.GLM(yb, np.column_stack([np.ones(N), x1, d, x4]), family=sm.families.Binomial(),
                  var_weights=design.weight).fit(tol=1e-14, maxiter=200).params[3]
    matrix = pd.DataFrame({"x1": x1, "x2": x1 + eps * d, "x4": x4}, index=pd.RangeIndex(N))
    table = survey_table("binary", matrix, yb, [0, 1], design, study=["x4"])
    assert table.info["refused"] is None, table.info["refused"]
    row = next(r for r in table.rows if r["feature"] == "x4")
    assert abs(row["estimate"] - want) <= 1e-6 * row["se"]


def test_without_the_floor_a_survey_fit_is_right_or_refused_as_unconverged(monkeypatch):
    """With the floor switched off, at every ε of the band the survey logistic table is either
    refused as unconverged in its words or served within ``SHORT_OF_MAXIMUM`` of a standard error
    of the weighted maximum: never served wrong. Which ε rounding stalls is the platform's: on the
    machine this was written on, 3·10⁻⁷, 2·10⁻⁷ and 1.05·10⁻⁷, where the decrement stalled at
    1.8·10⁻¹⁷, 5.3·10⁻¹⁸ and 1.7·10⁻¹⁷, 1.5 to 5 times its convergence tolerance (about
    3.6·10⁻¹⁸), so another BLAS or LAPACK can converge at each of them and no count of refusals is
    pinned (T5aV2-2). That rounding does stall it, where the floor fires, is the fires-side test's
    (:func:`test_a_survey_fit_rounding_stalls_is_served_at_its_maximum`)."""
    import statsmodels.api as sm

    x1, d, x4, _, yb, _, design, *_ = _data()
    want = sm.GLM(yb, np.column_stack([np.ones(N), x1, d, x4]), family=sm.families.Binomial(),
                  var_weights=design.weight).fit(tol=1e-14, maxiter=200).params[3]
    monkeypatch.setattr(S, "_at_floor", lambda *args: False)
    words = at_maximum("the survey-weighted logistic fit", 0.0, converged=False,
                       steps=S._NEWTON_MAX)
    for eps in (3e-7, 2e-7, 1.5e-7, 1.3e-7, 1.1e-7, 1.05e-7):
        matrix = pd.DataFrame({"x1": x1, "x2": x1 + eps * d, "x4": x4}, index=pd.RangeIndex(N))
        table = survey_table("binary", matrix, yb, [0, 1], design, study=["x4"])
        if table.info["refused"] is not None:
            assert table.info["refused"] == words, eps
            continue
        row = next(r for r in table.rows if r["feature"] == "x4")
        assert abs(row["estimate"] - want) <= SHORT_OF_MAXIMUM * row["se"], eps


def _quarter_steps(monkeypatch):
    """Each survey Newton step cut to a quarter, its decrement kept: the distance then falls to
    3/4 of itself a step, the decrement to 0.56, slowly but without stalling."""
    real = S._newton

    def quarter(working, score):
        step, decrement = real(working, score)
        return 0.25 * step, decrement

    monkeypatch.setattr(S, "_newton", quarter)


def _weighted_multinomial_maximum(X, codes, K, w):
    """The weighted multinomial logit's maximum against class 0, by Newton's method written out
    here, to a decrement of 10⁻²⁴; coefficients class by class."""
    n, P = X.shape
    q = K - 1
    Y = (codes[:, None] == np.arange(1, K)[None, :]).astype(float)
    beta = np.zeros(q * P)
    for _ in range(100):
        eta = np.column_stack([np.zeros(n), X @ beta.reshape(q, P).T])
        pi = np.exp(eta - eta.max(axis=1, keepdims=True))
        pi = (pi / pi.sum(axis=1, keepdims=True))[:, 1:]
        score = np.concatenate([X.T @ (w * (Y[:, a] - pi[:, a])) for a in range(q)])
        info = np.block([[X.T @ (X * (w * pi[:, a] * ((a == b) - pi[:, b]))[:, None])
                          for b in range(q)] for a in range(q)])
        step = np.linalg.solve(info, score)
        beta = beta + step
        if score @ step < 1e-24:
            break
    return beta, np.sqrt(np.diag(np.linalg.inv(info)))


def test_a_survey_fit_converging_slowly_keeps_stepping_to_its_maximum(monkeypatch):
    """The floor stays silent where the decrement still falls: with each step cut to a quarter,
    the survey logistic and multinomial tables are served at their maxima, every coefficient
    within 10⁻⁸ of its design-based standard error (statsmodels' weighted GLM; the multinomial's
    by Newton's method written out here). At ac3d66da the floor fired once the decrement fell by
    less than half a step under the share squared and served the logistic 5·10⁻⁵ of one off."""
    import statsmodels.api as sm

    x1, d, x4, _, yb, yo, design, *_ = _data()
    X = np.column_stack([np.ones(N), x1, d, x4])
    matrix = pd.DataFrame({"x1": x1, "x2": d, "x4": x4}, index=pd.RangeIndex(N))
    logistic = sm.GLM(yb, X, family=sm.families.Binomial(),
                      var_weights=design.weight).fit(tol=1e-14, maxiter=200).params
    multinomial, _ = _weighted_multinomial_maximum(X, yo.astype(np.int64), 3, design.weight)
    _quarter_steps(monkeypatch)
    table = survey_table("binary", matrix, yb, [0, 1], design, study=["x4"])
    assert table.info["refused"] is None, table.info["refused"]
    for j, row in enumerate(table.rows):
        assert abs(row["estimate"] - logistic[j]) <= 1e-8 * row["se"], row["feature"]
    table = survey_table("multiclass", matrix, yo, [0, 1, 2], design, study=["x4"])
    assert table.info["refused"] is None, table.info["refused"]
    assert len(table.rows) == len(multinomial)
    for j, row in enumerate(table.rows):
        assert abs(row["estimate"] - multinomial[j]) <= 1e-8 * row["se"], row["feature"]


def test_the_floor_reads_a_decrement_that_stopped_falling():
    """Fires where the decrement, scaled to weights summing to the rows, is at most the share
    squared and no less than the least reached; silent where it still falls, however slowly, or
    is above the share squared, or is only under it before the weights' scaling."""
    ones = np.ones(100)
    share = SHORT_OF_MAXIMUM ** 2
    assert S._at_floor(2.5e-16, 1.4e-16, ones)
    assert S._at_floor(share, share, ones)
    assert not S._at_floor(0.99 * share, share, ones)  # still falling, by 1% a step
    assert not S._at_floor(1.4e-16, 2.5e-16, ones)
    assert not S._at_floor(2 * share, share, ones)  # stalled above the share squared
    assert S._at_floor(5 * share, 4 * share, 10 * ones)  # weights summing to ten times the rows
    assert not S._at_floor(5 * share, 4 * share, ones)
