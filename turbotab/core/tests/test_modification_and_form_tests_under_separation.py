"""Through the stages, the modification's Firth family and an exposure's form tests take no Firth
profile quantity where the data separate, and are served where nothing separates (V5c2-1; the
methods lead's ruling of 2026-10-28).

Where a covariate `s` has the event on every one of its rows, the logistic model is Firth's, and
held at a value its penalized likelihood can have more than one maximum: before T5d the
modification served each odds ratio with a profile interval and a penalized likelihood-ratio
heterogeneity test, and a spline's form tests were penalized likelihood-ratio tests (under
multiple imputation pooled by D2). Now each of the modification's quantities is the penalized
estimate alone and no heterogeneity test is reported (the family names it as not computed), and
no test of the spline's form is reported, in words that name `s` and say why. Without the
separation (the same rows, `s`'s outcomes drawn as the others') the modification is the
maximum-likelihood fit's, each quantity with its Wald interval and the Wald heterogeneity test,
and the spline's tests are Wald tests, each as statsmodels computes it.

Truth: separated or not by construction, checked by scipy's linear program
(``test_firth_serves_its_estimates_alone_under_separation._cone``); Firth's maximum by scipy's
BFGS on the penalized log-likelihood written out with numpy, from several starts; the
maximum-likelihood fits and their covariances by statsmodels, on designs built here with pandas
(the spline's basis Harrell's restricted cubic spline written out, at the knots the fit states).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from turbotab.core import decisions as d
from turbotab.core.methods import interaction as ix
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.acceptance import estimand_fixtures as est_f
from turbotab.core.tests.test_firth_serves_its_estimates_alone_under_separation import (
    _cone, _reference, words)

Z = float(stats.norm.ppf(0.975))


# ── the modification ─────────────────────────────────────────────────────────


def _cohort(separate: bool) -> pd.DataFrame:
    """700 rows of the estimand fixtures' cohort, `s` on about 7% of them; with ``separate``
    every row with `s` has the event."""
    frame = est_f.cohort(n=700, seed=23)
    rng = np.random.default_rng(230)
    frame["s"] = (rng.random(len(frame)) < 0.07).astype(float)
    if separate:
        frame.loc[frame["s"] == 1, "dm"] = "yes"
    return frame


def _modification(frame: pd.DataFrame, folder: Path) -> dict[str, Any]:
    from turbotab.core.stages.modeling import design_stage

    roles = {**est_f.ROLES, "s": "covariate"}
    st = est_f.state(target="dm", task="binary", event="yes", measure="odds_ratio", roles=roles,
                     answers={**est_f.ANSWERS, "s": est_f.CONFOUNDER},
                     amounts={**est_f.AMOUNTS, "code_or_count:s": "amount"},
                     modifications={"sex": d.ModificationSpec(kind="effect_modification",
                                                              exposure="fiber")})
    paths = mf.ingest_frame(frame, folder)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
    info = mf.target_info(st.task, st.target)
    design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
    return ix.modification_stage(mf.context(st, {"design": design, "split": split,
                                                 "target_info": info}, paths)).data


def _design_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """[1, fiber, male, fiber·male, age, smoking, activity, s] (female the reference), and the
    event: the model the modification fits, written out with pandas."""
    male = (frame["sex"] == "male").to_numpy(float)
    fiber = frame["fiber"].to_numpy(float)
    X = np.column_stack([np.ones(len(frame)), fiber, male, fiber * male,
                         frame[["age", "smoking", "activity", "s"]].to_numpy(float)])
    return X, (frame["dm"] == "yes").to_numpy(float)


def _quartiles(frame: pd.DataFrame) -> tuple[float, float]:
    a0, a1 = np.quantile(frame["fiber"].to_numpy(float), [0.25, 0.75], method="linear")
    return float(a0), float(a1)


@pytest.fixture(scope="module")
def modifications(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = {}
    for separate in (True, False):
        frame = _cohort(separate)
        out[separate] = {"frame": frame, "out": _modification(
            frame, tmp_path_factory.mktemp(f"em_{separate}"))}
    return out


def test_where_the_data_separate_the_modification_reports_penalized_estimates_alone(
        modifications):
    """Each effect within a sex, each combination, the ratio of odds ratios and the RERI are
    Firth's penalized estimates (the independent maximum's to 10⁻⁵ relative), with no interval or
    p-value; no heterogeneity test is reported, the family names `sex`'s as not computed, and the
    concerns say why in words that name `s`. On the base each carried a profile interval and the
    heterogeneity test was a penalized likelihood-ratio test."""
    frame, out = modifications[True]["frame"], modifications[True]["out"]
    X, y = _design_matrix(frame)
    assert _cone(X, y) == [7]  # `s` alone, by construction
    beta, _, _ = _reference(X, y)
    (result,) = out["modifications"]
    (fit,) = result["families"]
    assert fit["penalized"] and fit["heterogeneity"] is None
    a0, a1 = _quartiles(frame)
    delta = a1 - a0
    L10 = delta * beta[1]
    L01 = beta[2] + beta[3] * a0
    L11 = L10 + beta[2] + beta[3] * a1
    effects = {q["stratum"]: q for q in fit["effects"]}
    assert effects["female"]["estimate"] == pytest.approx(math.exp(delta * beta[1]), rel=1e-5)
    assert effects["male"]["estimate"] == pytest.approx(math.exp(delta * (beta[1] + beta[3])),
                                                        rel=1e-5)
    (mult,) = fit["multiplicative"]
    assert mult["estimate"] == pytest.approx(math.exp(L11 - L10 - L01), rel=1e-5)
    (reri,) = fit["reri"]
    assert reri["estimate"] == pytest.approx(math.exp(L11) - math.exp(L10) - math.exp(L01) + 1,
                                             rel=1e-5)
    for q in [*fit["effects"], *fit["joint"], *fit["multiplicative"], *fit["reri"]]:
        assert q["estimate"] is not None and (q["ci_low"], q["ci_high"], q["p"]) == (
            None, None, None), q
    assert words(["s"]) in fit["concerns"]
    assert out["family_count"]["not_computed"] == ["sex"]
    assert "heterogeneity test is the" not in result["sentence"]
    assert result["sentence"].endswith("; no heterogeneity test is reported.")


def test_where_nothing_separates_the_modification_is_the_maximum_likelihood_fits(modifications):
    """The silent side: the same rows with `s`'s outcomes drawn as the others' are not separated,
    the family is not penalized, and each effect's odds ratio and Wald interval, and the Wald
    heterogeneity test of the product term, are statsmodels' (to 10⁻⁶)."""
    frame, out = modifications[False]["frame"], modifications[False]["out"]
    X, y = _design_matrix(frame)
    assert _cone(X, y) == []
    ml = sm.Logit(y, X).fit(disp=0)
    b, V = np.asarray(ml.params), np.asarray(ml.cov_params())
    (result,) = out["modifications"]
    (fit,) = result["families"]
    assert not fit["penalized"]
    a0, a1 = _quartiles(frame)
    delta = a1 - a0
    effects = {q["stratum"]: q for q in fit["effects"]}
    for stratum, c in (("female", np.r_[0, delta, 0, 0, 0, 0, 0, 0]),
                       ("male", np.r_[0, delta, 0, delta, 0, 0, 0, 0])):
        est, se = float(c @ b), math.sqrt(float(c @ V @ c))
        q = effects[stratum]
        assert q["estimate"] == pytest.approx(math.exp(est), rel=1e-6), stratum
        assert q["ci_low"] == pytest.approx(math.exp(est - Z * se), rel=1e-6), stratum
        assert q["ci_high"] == pytest.approx(math.exp(est + Z * se), rel=1e-6), stratum
    het = fit["heterogeneity"]
    statistic = b[3] ** 2 / V[3, 3]
    assert het["statistic"] == pytest.approx(statistic, rel=1e-6)
    assert het["p"] == pytest.approx(float(stats.chi2.sf(statistic, 1)), rel=1e-6)
    assert out["family_count"]["not_computed"] == []
    assert not any("separate" in c for c in fit["concerns"])


# ── an exposure's form tests ─────────────────────────────────────────────────


def _spline_cohort(separate: bool, n: int = 420, seed: int = 12) -> pd.DataFrame:
    """A yes/no outcome beside a curved ``x`` and a ``z``; with ``separate`` every row with ``s``
    = 1 has the event."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = rng.gamma(3.0, 2.0, n) + 0.4 * z
    s = (rng.random(n) < 0.08).astype(int)
    eta = -0.6 + 0.35 * np.log(x + 1) + 0.5 * z
    drawn = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    b = np.where(s == 1, 1, drawn) if separate else drawn
    return pd.DataFrame({"pid": np.arange(n), "x": x.round(6), "z": z.round(6), "s": s, "b": b})


def _spline_fit(frame: pd.DataFrame, folder: Path, missing: Any = "complete_case") -> dict:
    from turbotab.core.stages.modeling import design_stage, fit_stage

    roles = {"pid": "identifier", "x": "exposure", "z": "covariate", "s": "covariate"}
    st = mf.state(roles=roles, role_confirmations=dict(roles), target="b", task="binary",
                  event="1", purpose="inference", models=["linear"], lens=["clinical"],
                  split=d.SplitSpec(holdout=0.0, seed=0, folds=5), missing=missing,
                  shape_confirmations={"code_or_count:b": "amount", "code_or_count:s": "amount"},
                  exposure_forms={"x": d.ExposureFormSpec(form="spline", knots=3)})
    paths = mf.ingest_frame(frame, folder)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
    info = mf.target_info("binary", "b")
    design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": info}, paths))
    return fit.data["models"][0]


def _rcs(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell's restricted cubic spline basis: x and, for each inner knot j, ``(x − t_j)₊³ −
    (x − t_{k−1})₊³ (t_k − t_j)/(t_k − t_{k−1}) + (x − t_k)₊³ (t_{k−1} − t_j)/(t_k − t_{k−1})``."""
    t = np.asarray(knots, dtype=float)
    k = len(t)
    cube = lambda v: np.clip(v, 0, None) ** 3  # noqa: E731
    cols = [x] + [cube(x - t[j]) - cube(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                  + cube(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])
                  for j in range(k - 2)]
    return np.column_stack(cols)


def test_where_the_data_separate_no_test_of_the_splines_form_is_reported(tmp_path):
    """Complete cases: the table serves its penalized estimates alone, refused in words that name
    `s`, and no test of `x`'s form is reported (an unpooled separated table is refused before any
    test). Under multiple imputation (`z` blank on some rows, 20 copies, each separated): no
    pooled test of `x`'s form is reported and the words say why, naming `s`; on the base each
    copy's penalized likelihood-ratio tests were pooled by D2."""
    frame = _spline_cohort(True)
    model = _spline_fit(frame, tmp_path / "cc")
    assert model["inference"]["refused"] == words(["s"])
    assert [t for t in model["exposure_tests"] if t["column"] == "x"] == []
    blank = frame.copy()
    blank.loc[np.random.default_rng(4).random(len(frame)) < 0.15, "z"] = np.nan
    pooled = _spline_fit(blank, tmp_path / "mi",
                         missing={"strategy": "multiple_imputation", "m": 20})
    assert [t for t in pooled["exposure_tests"] if t["column"] == "x"] == []
    assert words(["s"], held="coefficients held at zero",
                 unreported="penalized likelihood-ratio test of `x`'s form") in pooled["concerns"]


def test_where_nothing_separates_the_splines_tests_are_wald_tests_as_statsmodels_computes(
        tmp_path):
    """The silent side: on the same rows without the separation, the test of association and the
    nonlinearity test are the Wald tests ``bᵀV⁻¹b`` on χ² of statsmodels' fit on Harrell's basis
    at the knots the fit states (to 10⁻⁶), and no concern speaks of separation."""
    frame = _spline_cohort(False)
    model = _spline_fit(frame, tmp_path / "plain")
    assert model["inference"]["refused"] is None and model["inference"]["separated"] == []
    tests = {t["test"]: t for t in model["exposure_tests"] if t["column"] == "x"}
    knots = np.asarray(tests["overall"]["knots"], dtype=float)
    X = np.column_stack([np.ones(len(frame)), _rcs(frame["x"].to_numpy(float), knots),
                         frame[["z", "s"]].to_numpy(float)])
    assert _cone(X, frame["b"].to_numpy(float)) == []
    ml = sm.Logit(frame["b"].to_numpy(float), X).fit(disp=0)
    b, V = np.asarray(ml.params), np.asarray(ml.cov_params())
    for kind, idx in (("overall", [1, 2]), ("nonlinear", [2])):
        W = float(b[idx] @ np.linalg.solve(V[np.ix_(idx, idx)], b[idx]))
        ours = tests[kind]
        assert (ours["distribution"], ours["df_num"]) == ("chi2", len(idx))
        assert ours["statistic"] == pytest.approx(W, rel=1e-6), kind
        assert ours["p"] == pytest.approx(float(stats.chi2.sf(W, len(idx))), rel=1e-6), kind
    assert not any("separate" in c for c in model.get("concerns") or [])
