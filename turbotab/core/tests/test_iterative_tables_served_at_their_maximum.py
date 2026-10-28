"""An iterative fit's table (logistic, multinomial, proportional odds, Firth, and their survey forms)
is served only where its maximum is certified: the fitter converged, and the Newton step from its
estimate, on the centered, scaled design, moves no coefficient by more than
``SHORT_OF_MAXIMUM`` of its standard error; else it is refused in typed words that say so, with
the distance interpolated (T1V3-2, T1V3-3; the methods lead's ruling 2 of 2026-10-27). A Firth
profile that cannot bracket a bound, whose fits do not converge, or that runs past its steps is a
typed refusal, never an untyped ``ValueError``, and no Firth table takes minutes.

On the base, beside ``x2 = x1 + ε·d`` with ε from 1.05·10⁻⁷ to 1.3·10⁻⁷ (just above R's tolerance,
an information of condition number 10¹⁴), the survey multinomial table (and its ordered-levels
form) took its Newton steps from a least-squares solve that dropped the near pair's direction,
declared convergence far from the maximum, and served the study factor's log-odds off by up to
three times, with no concern; the Firth table beside the same near copy and a separating column
raised ``ValueError: f(a) and f(b) must have different signs`` at ε = 10⁻⁴ to 10⁻⁶ and served
`v`'s penalized log-odds as −5·10⁻⁶ (against −0.353) at 3·10⁻⁷.

Truth: the columns ``x1`` and ``x1 + ε·d`` span what ``x1`` and ``d`` do, so the study factor's
row is an exact invariant: the table on ``(x1, d, x4)``, well conditioned, is the reference; for
the distance itself, ``√(sᵀI⁻¹s)`` written out with numpy from the score and the information.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import pytest

from turbotab.core.models import inference as I
from turbotab.core.models import survey as S
from turbotab.core.models.inference import (
    INDEPENDENT,
    SHORT_OF_MAXIMUM,
    Clusters,
    FirthProfileFailed,
    at_maximum,
    inference_table,
    newton_distance,
)
from turbotab.core.models.ordinal import ProportionalOddsRegression, ordinal_table, survey_ordinal_table
from turbotab.core.models.survey import SurveyDesign, survey_table

N, G = 360, 60
BAND = (1e-4, 1e-5, 1e-6, 3e-7, 2e-7, 1.5e-7, 1.3e-7, 1.1e-7, 1.05e-7)


def _data():
    rng = np.random.default_rng(7)
    codes = np.repeat(np.arange(G), N // G)
    x1, x2, x4 = rng.normal(size=(3, N))
    d = rng.normal(size=N)
    eta = 0.4 * x1 - 0.3 * x4 + 0.25 * x2
    yb = (rng.random(N) < 1 / (1 + np.exp(-eta))).astype(int)
    yo = np.digitize(eta + rng.logistic(size=N), [-0.7, 0.7])
    design = SurveyDesign(row_ids=np.arange(N), weight=rng.uniform(1, 3, N),
                          stratum=np.repeat(np.arange(12), N // 12).astype(np.int64),
                          psu=np.repeat(np.arange(36), N // 36).astype(np.int64),
                          weight_column="wt", strata_column="st", psu_column="psu")
    yf = yb.copy()
    separating = np.arange(N) % 9 == 0
    yf[separating] = 1
    return x1, d, x4, codes, yb, yo, design, yf, separating.astype(float)


def _families():
    x1, d, x4, codes, yb, yo, design, yf, f = _data()
    cl = Clusters(column="pid", codes=codes, n_clusters=G, n_missing=0)
    s = {"study": ["x4"]}
    return {
        "logistic": lambda m: inference_table("binary", m, yb, [0, 1], INDEPENDENT, **s),
        "logistic (CR2)": lambda m: inference_table("binary", m, yb, [0, 1], cl, **s),
        "multinomial": lambda m: inference_table("multiclass", m, yo, [0, 1, 2], INDEPENDENT, **s),
        "proportional odds": lambda m: ordinal_table(m, yo, [0, 1, 2], INDEPENDENT, **s),
        "proportional odds (CR1)": lambda m: ordinal_table(m, yo, [0, 1, 2], cl, **s),
        "Firth": lambda m: inference_table("binary", m.assign(f=f), yf, [0, 1], INDEPENDENT, **s),
        "survey logistic": lambda m: survey_table("binary", m, yb, [0, 1], design, **s),
        "survey multinomial": lambda m: survey_table("multiclass", m, yo, [0, 1, 2], design, **s),
        "survey multinomial (ordered levels)": lambda m: survey_table("ordinal", m, yo, [0, 1, 2],
                                                                      design, **s),
        "survey proportional odds": lambda m: survey_ordinal_table(m, yo, [0, 1, 2], design, **s),
    }


@pytest.mark.parametrize("family", list(_families()))
def test_across_the_band_each_iterative_table_is_right_or_refused(family):
    """At each ε from 10⁻⁴ to 1.05·10⁻⁷ the table on ``(x1, x1 + ε·d, x4)`` is refused in words
    (no estimate, no exception) or its `x4` rows, the study factor's, are the reference's within
    the certified share of a standard error, their standard errors to 10⁻⁴; within 60 s for every
    ε (the verifier timed one Firth table at 215 s on the base). On the base the survey
    multinomial tables served `x4 [1]` at −0.1200 against −0.1177 at 1.1·10⁻⁷, and the Firth
    table raised the solver's ``ValueError`` at 10⁻⁴."""
    build = _families()[family]
    x1, d, x4, *_ = _data()
    index = pd.RangeIndex(N)
    reference = build(pd.DataFrame({"x1": x1, "x2": d, "x4": x4}, index=index))
    assert reference.info["refused"] is None, reference.info["refused"]
    want = {r["feature"]: r for r in reference.rows}
    started = time.monotonic()
    for eps in BAND:
        table = build(pd.DataFrame({"x1": x1, "x2": x1 + eps * d, "x4": x4}, index=index))
        if table.info["refused"] is not None:
            assert isinstance(table.info["refused"], str) and table.info["refused"]
            assert all(r.get("ci_low") is None and r.get("p") is None for r in table.rows)
            continue
        assert table.unidentified == []
        served = [r for r in table.rows if r["feature"].split(" [")[0] == "x4"]
        assert served, eps
        for row in served:
            ref = want[row["feature"]]
            assert abs(row["estimate"] - ref["estimate"]) <= SHORT_OF_MAXIMUM * ref["se"] + 1e-9, (
                eps, row["feature"], row["estimate"], ref["estimate"])
            assert row["se"] == pytest.approx(ref["se"], rel=1e-4), (eps, row["feature"])
    assert time.monotonic() - started < 60


def test_the_distance_is_the_score_in_the_informations_metric():
    """``newton_distance`` is ``√(sᵀI⁻¹s)`` written out with numpy from the logistic score and
    information (on a well-conditioned design, to 10⁻⁹), zero to rounding at the maximum, and
    for a small move ``δ`` from it about ``√(δᵀIδ)`` (second order)."""
    import statsmodels.api as sm

    x1, d, x4, _, yb, *_ = _data()
    X = np.column_stack([np.ones(N), x1, d, x4])
    mle = sm.Logit(yb, X).fit(disp=0).params

    def parts(beta):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        score = X.T @ (yb - mu)
        info = X.T @ (X * (mu * (1 - mu))[:, None])
        return mu, score, info

    mu, score, info = parts(mle)
    Xw, ew = I.logistic_working(X, yb, mu)
    assert newton_distance(Xw, ew) < 1e-6
    delta = np.array([0.01, -0.02, 0.015, 0.005])
    mu, score, info = parts(mle + delta)
    Xw, ew = I.logistic_working(X, yb, mu)
    distance = newton_distance(Xw, ew)
    assert distance == pytest.approx(math.sqrt(score @ np.linalg.solve(info, score)), rel=1e-9)
    assert distance == pytest.approx(math.sqrt(delta @ info @ delta), rel=5e-2)


def test_the_maximum_words_fire_above_the_share_and_stay_silent_at_it():
    assert at_maximum("the logistic fit", SHORT_OF_MAXIMUM) is None
    assert at_maximum("the logistic fit", 0.0) is None
    assert at_maximum("the logistic fit", 0.063) == (
        "The logistic fit stopped short of its maximum: where it stopped, one more Newton step "
        "would still move its estimates by up to 0.063 of a standard error, and a table is served "
        f"only within 1/{round(1 / SHORT_OF_MAXIMUM):,} of one, so no estimate, interval or "
        "p-value is reported.")
    assert "the information there is singular" in at_maximum("the logistic fit", math.inf)
    assert at_maximum("the proportional-odds fit", 0.0, converged=False, steps=100) == (
        "The proportional-odds fit did not converge in 100 Newton steps, so no estimate, interval "
        "or p-value is reported.")


def test_a_survey_fit_that_claims_convergence_short_of_its_maximum_is_refused(monkeypatch):
    """The guard fires where the fitter says it converged but is not at its maximum: a weighted
    logistic fit moved off its maximum by a known step and flagged converged is refused, in words
    that interpolate the distance written out here (``√(sᵀI⁻¹s)`` on the weights scaled to sum to
    the rows), with no estimate; the real fit is served (the silent side)."""
    x1, d, x4, _, yb, _, design, *_ = _data()
    matrix = pd.DataFrame({"x1": x1, "x2": d, "x4": x4}, index=pd.RangeIndex(N))
    served = survey_table("binary", matrix, yb, [0, 1], design, study=["x4"])
    assert served.info["refused"] is None
    real = S.weighted_logistic
    seen = {}

    def short(X, y, w):
        fit = real(X, y, w)
        beta = fit.estimate + 0.05
        mu = 1 / (1 + np.exp(-(X @ beta)))
        scores = X * (w * (y - mu))[:, None]
        info = X.T @ (X * (w * mu * (1 - mu))[:, None])
        s = scores.sum(axis=0)
        seen["distance"] = math.sqrt(float(s @ np.linalg.solve(info, s)) * len(w) / w.sum())
        return S.WeightedFit(beta, scores, info, True)

    monkeypatch.setattr(S, "weighted_logistic", short)
    refused = survey_table("binary", matrix, yb, [0, 1], design, study=["x4"])
    assert refused.info["refused"] == at_maximum("the survey-weighted logistic fit",
                                                 seen["distance"])
    assert seen["distance"] > SHORT_OF_MAXIMUM
    assert f"by up to {seen['distance']:.2g} of a standard error" in refused.info["refused"]
    assert all(r["estimate"] is None for r in refused.rows)


def test_a_proportional_odds_fit_that_does_not_converge_is_refused_not_served_with_care(
        monkeypatch):
    """Both ways: the ordinary table is served; capped at one Newton step it is refused in the
    words that say so, where the base served it beside "treat these numbers with care"."""
    x1, d, x4, _, _, yo, *_ = _data()
    matrix = pd.DataFrame({"x1": x1, "x2": d, "x4": x4})
    assert ordinal_table(matrix, yo, [0, 1, 2], INDEPENDENT, study=["x4"]).info["refused"] is None
    original = ProportionalOddsRegression.__init__

    def one_step(self, max_iter=100, tol=1e-10):
        original(self, max_iter=1, tol=tol)

    monkeypatch.setattr(ProportionalOddsRegression, "__init__", one_step)
    table = ordinal_table(matrix, yo, [0, 1, 2], INDEPENDENT, study=["x4"])
    assert table.info["refused"] == ("The proportional-odds fit did not converge, so no estimate, "
                                     "interval or p-value is reported.")
    assert not any("with care" in c for c in table.concerns)


def _firth_fit():
    x1, d, x4, *_, yf, f = _data()
    X = np.column_stack([np.ones(N), x1, d, x4, f])
    return X, yf.astype(float), I.firth_fit(X, yf.astype(float))


def test_a_firth_profile_that_cannot_bracket_its_bound_is_typed_and_strict_only():
    """A fit whose penalized log-likelihood is overstated by 10 (above every profile's) cannot
    bracket a bound: strict, the profile raises ``FirthProfileFailed`` in words that name the
    coefficient; unstrict (the methods' callers), the solver's ``ValueError`` as before."""
    X, y, fit = _firth_fit()
    assert fit.converged
    inflated = I.FirthFit(beta=fit.beta, loglik=fit.loglik + 10.0, cov=fit.cov,
                          iterations=fit.iterations, converged=True)
    with pytest.raises(FirthProfileFailed, match=r"Firth's profile of `x4` could not be computed: "
                                                 r"its penalized likelihood ratio did not cross"):
        I.firth_profile(X, y, inflated, 3, strict=True, name="x4")
    with pytest.raises(ValueError) as caught:
        I.firth_profile(X, y, inflated, 3)
    assert not isinstance(caught.value, FirthProfileFailed)
    low, high, p = I.firth_profile(X, y, fit, 3, strict=True, name="x4")
    assert low < fit.beta[3] < high and 0 < p < 1


def test_a_firth_profile_past_its_steps_is_typed(monkeypatch):
    X, y, fit = _firth_fit()
    monkeypatch.setattr(I, "FIRTH_PROFILE_STEPS", 3)
    with pytest.raises(FirthProfileFailed, match="could not be computed within 3 penalized Newton"):
        I.firth_profile(X, y, fit, 3, strict=True, name="x4")


def test_a_firth_table_whose_profile_fails_is_refused_in_its_words(monkeypatch):
    """The table refuses in the profile's own words, keeping the estimates and the separation
    concern; the ordinary Firth table is served with profile intervals (the silent side)."""
    x1, d, x4, *_, yf, f = _data()
    matrix = pd.DataFrame({"x1": x1, "x2": d, "x4": x4, "f": f})
    table = inference_table("binary", matrix, yf, [0, 1], INDEPENDENT, study=["x4"])
    assert table.info["refused"] is None and table.info["estimator"] == I.FIRTH
    assert all(r["ci_low"] is not None for r in table.rows if r["feature"] != "f")
    words = "Firth's profile of `x4` could not be computed (a test's words)."

    def failing(*args, **kwargs):
        raise FirthProfileFailed(words)

    monkeypatch.setattr(I, "firth_profile", failing)
    refused = inference_table("binary", matrix, yf, [0, 1], INDEPENDENT, study=["x4"])
    assert refused.info["refused"] == words
    assert refused.concerns[0].startswith("`f` separates the outcome")
