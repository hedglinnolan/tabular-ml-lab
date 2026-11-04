"""Batch 1, Estimate end to end (the owner's rulings of 2026-11-04 on the vision audit's V1): under
Estimate the declared exposure's proposed form is a straight line, and whichever shape the
researcher keeps, the other is set in the plan as its check and reported beside the estimate
(calm/BEATS_MODELS_RESULTS M4: "Whichever you choose, the other shape is set in your plan, and
Results reports it beside your estimate").

* **The proposal** (``exposure_form.options`` and ``proposal``): under inference the exposure's
  straight line leads, recommended, the restricted cubic spline and every other form offered
  beside it; a confounder's spline still leads (residual confounding), and Predict's ranking is
  unchanged. Truth: the ruling, by construction.
* **The check of a straight line** (the effects stage, ``stages/effects.py``): the primary model
  refitted with the exposure's restricted cubic spline in its place, k by Harrell's rule, its
  Wald test that the nonlinear terms are zero, and its curve. Truth: statsmodels on a constructed
  table with a known bend and on one with none (least squares with HC3 and its F test; logistic
  regression and its χ² test), the knots at Harrell's percentiles written out with NumPy, the
  basis written out from Harrell's formula; under multiple imputation Li, Raghunathan & Rubin's
  D1 written out over statsmodels' fit of each completed copy.
* **The check of a curve**: the straight line's one coefficient beside the spline, statsmodels'
  straight-line fit.
* **Never in its place**: the estimate stays the kept shape's whatever the check shows; no check
  where no form was declared, or for another form.
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
from turbotab.core.methods import exposure_form as ef
from turbotab.core.tests.acceptance import estimand_fixtures as est_f

COVARIATES = ["age", "sex", "smoking", "activity"]  # Model 2: the answers' confounders, precision
FIVE = [0.05, 0.275, 0.50, 0.725, 0.95]  # Harrell's percentiles for 5 knots (RMS §2.4.6)
LINE = d.ExposureFormSpec(form="linear", scale="raw")


def bent(n: int = 600, seed: int = 21) -> pd.DataFrame:
    """The ESTIMAND cohort with a known bend in `fiber`: 0.25 (fiber − 20)² added to `glucose`."""
    frame = est_f.cohort(n=n, seed=seed)
    frame["glucose"] = frame["glucose"] + 0.25 * (frame["fiber"] - 20.0) ** 2
    return frame


def straight(n: int = 600, seed: int = 21) -> pd.DataFrame:
    """The ESTIMAND cohort as generated: `glucose` is linear in `fiber` (no bend)."""
    return est_f.cohort(n=n, seed=seed)


def harrell(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell's restricted cubic spline's k − 2 nonlinear terms (RMS 2nd ed., §2.4.5, eq. 2.25,
    ``norm = 2``), written out from the formula."""
    t = np.asarray(knots, dtype=float)
    k = len(t)

    def cube(u: np.ndarray) -> np.ndarray:
        return np.clip(u, 0.0, None) ** 3

    out = [cube(x - t[j]) - cube(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
           + cube(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]) for j in range(k - 2)]
    return np.column_stack(out) / (t[k - 1] - t[0]) ** 2


def with_spline(frame: pd.DataFrame, knots: np.ndarray) -> pd.DataFrame:
    X = est_f.design_matrix(frame, ["fiber", *COVARIATES])
    for j, column in enumerate(harrell(frame["fiber"].to_numpy(dtype=float), knots).T):
        X[f"rcs{j}"] = column
    return X


def spline_names(k: int) -> list[str]:
    return [f"rcs{j}" for j in range(k - 2)]


def check_of(effects: dict[str, Any], family: int = 0) -> dict[str, Any] | None:
    return effects["families"][family].get("check")


def model_two(effects: dict[str, Any]) -> dict[str, Any]:
    return est_f.sequence(effects)["model_2"]


def numeric_state(**slots: Any) -> d.ProjectState:
    return est_f.state(target="glucose", task="regression", measure="mean_difference", **slots)


def run(frame: pd.DataFrame, folder: Path, st: d.ProjectState) -> dict[str, Any]:
    return est_f.run(frame, folder, st)


# ── the proposal ─────────────────────────────────────────────────────────────


def _info(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {c: {"dtype": "numeric" if pd.api.types.is_numeric_dtype(frame[c]) else "text",
                "n_unique": int(frame[c].nunique())} for c in frame.columns}


def test_under_estimate_the_exposure_is_proposed_as_a_straight_line_and_a_confounder_as_a_spline():
    """Both ways. The declared exposure: the straight line is the proposal and leads, recommended;
    the spline stays offered, next. The adjusted continuous confounder: the spline still leads,
    k by Harrell's rule (600 analyzed rows: 5 knots). On the base the exposure's proposal was the
    spline."""
    frame = straight()
    st = numeric_state()
    card = ef.forms_card(st, frame, _info(frame), frame["glucose"], "regression")
    needs = {n["column"]: n for n in card["needs"]}
    fiber, age = needs["fiber"], needs["age"]
    assert fiber["role"] == "exposure" and age["role"] == "confounder"
    assert fiber["proposal"] == {"form": "linear"}
    assert [(o["value"], o["rung"]) for o in fiber["options"]] == [
        ("linear", "recommended"), ("spline", "available"), ("quintiles", "rank_lower"),
        ("categories", "rank_lower"), ("optimal", "block_and_record")]
    assert age["proposal"] == {"form": "spline", "knots": 5, "knots_rule": "harrell"}
    assert [(o["value"], o["rung"]) for o in age["options"]] == [
        ("spline", "recommended"), ("linear", "available"), ("categories", "rank_lower"),
        ("quintiles", "rank_lower"), ("optimal", "block_and_record")]
    assert card["answer"] == {"kind": "set_forms", "forms": {
        "fiber": {"form": "linear"},
        "age": {"form": "spline", "knots": 5, "knots_rule": "harrell"}}}
    # The one-tap answer is recorded as each form's own declaration would be.
    recorded = d.validate(card["answer"], {"state": st})
    assert recorded.forms["fiber"].form == "linear" and recorded.forms["age"].form == "spline"


def test_with_a_mass_at_zero_the_straight_line_still_leads_and_non_consumers_apart_follow():
    """The exposure's straight line leads even with a mass at zero; non-consumers apart (and the
    consumers-only domain) are offered next, available, ahead of the spline."""
    found = ef.options("inference", "exposure", mass_at_zero=True)
    assert [(o["value"], o["rung"]) for o in found] == [
        ("linear", "recommended"), ("zero_spline", "available"), ("consumers_only", "available"),
        ("spline", "available"), ("quintiles", "rank_lower"), ("categories", "rank_lower"),
        ("optimal", "block_and_record")]
    assert ef.proposal("exposure", "inference", 5, mass_at_zero=True) == {"form": "linear"}


def test_predicts_ranking_and_a_confounders_are_unchanged():
    """Silent where it should be: under prediction the spline still leads for every role (it nests
    the line), and a confounder's ranking under inference is the one it was."""
    for role in ("exposure", "confounder"):
        found = ef.options("prediction", role)
        assert found[0]["value"] == "spline" and found[0]["rung"] == "recommended"
        assert [o["rung"] for o in found[1:]] == ["available", "rank_lower", "rank_lower",
                                                  "rank_lower"]
    assert [o["value"] for o in ef.options("prediction", "exposure")] == [
        "spline", "linear", "quintiles", "categories", "optimal"]
    assert [o["value"] for o in ef.options("prediction", "exposure", mass_at_zero=True)] == [
        "zero_spline", "spline", "linear", "quintiles", "categories", "optimal"]
    assert ef.proposal("confounder", "inference", 4, mass_at_zero=False) == {
        "form": "spline", "knots": 4, "knots_rule": "harrell"}
    assert ef.proposal("exposure", "prediction", 4, mass_at_zero=False) == {
        "form": "spline", "knots": 4, "knots_rule": "harrell"}


# ── the check of a straight line ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def bend_run(tmp_path_factory) -> dict[str, Any]:
    frame = bent()
    return {"frame": frame, **run(frame, tmp_path_factory.mktemp("bend"),
                                  numeric_state(exposure_forms={"fiber": LINE}))}


@pytest.fixture(scope="module")
def line_run(tmp_path_factory) -> dict[str, Any]:
    frame = straight()
    return {"frame": frame, **run(frame, tmp_path_factory.mktemp("line"),
                                  numeric_state(exposure_forms={"fiber": LINE}))}


def _statsmodels_check(frame: pd.DataFrame) -> dict[str, Any]:
    """statsmodels' least squares with HC3 of the primary model with Harrell's 5-knot spline of
    `fiber` in place of its straight line, and its F test that the 3 nonlinear terms are zero; and
    the straight line's own fit."""
    x = frame["fiber"].to_numpy(dtype=float)
    knots = np.quantile(x, FIVE, method="linear")
    y = frame["glucose"].astype(float)
    curve = sm.OLS(y, with_spline(frame, knots)).fit(cov_type="HC3", use_t=True)
    R = np.zeros((3, len(curve.params)))
    for j, name in enumerate(spline_names(5)):
        R[j, list(curve.params.index).index(name)] = 1.0
    wald = curve.wald_test(R, use_f=True, scalar=True)
    line = sm.OLS(y, est_f.design_matrix(frame, ["fiber", *COVARIATES])).fit(cov_type="HC3",
                                                                             use_t=True)
    return {"knots": knots, "curve": curve, "F": float(wald.fvalue), "df_num": int(wald.df_num),
            "df_den": float(wald.df_denom), "p": float(wald.pvalue), "line": line}


def _assert_check_is_statsmodels(check: dict[str, Any], truth: dict[str, Any]) -> None:
    assert check["refused"] is None, check["refused"]
    assert (check["kept"], check["shape"], check["column"]) == ("linear", "spline", "fiber")
    np.testing.assert_allclose(check["knots"], truth["knots"], rtol=1e-12)
    test = check["test"]
    assert (test["distribution"], test["df_num"]) == ("F", truth["df_num"])
    assert test["df_den"] == pytest.approx(truth["df_den"], rel=1e-12)
    assert test["statistic"] == pytest.approx(truth["F"], rel=1e-8)
    assert test["p"] == pytest.approx(truth["p"], rel=1e-6, abs=1e-300)
    # The spline's terms, as statsmodels fits them.
    curve = truth["curve"]
    terms = {t["feature"]: t for t in check["terms"]}
    assert set(terms) == {"fiber", "fiber'", "fiber''", "fiber'''"}
    for name, theirs in zip(["fiber", "fiber'", "fiber''", "fiber'''"], ["fiber", *spline_names(5)]):
        assert terms[name]["estimate"] == pytest.approx(float(curve.params[theirs]), rel=1e-8)
        assert terms[name]["se"] == pytest.approx(float(curve.bse[theirs]), rel=1e-8)
    # Its curve: the difference from the lowest knot, at 50 values from the lowest knot to the
    # highest, and the straight line's beside it; written out from statsmodels' coefficients.
    knots = truth["knots"]
    grid = np.linspace(knots[0], knots[-1], 50)
    basis = harrell(grid, knots) - harrell(np.array([knots[0]]), knots)
    b = [float(curve.params[n]) for n in spline_names(5)]
    want = float(curve.params["fiber"]) * (grid - knots[0]) + basis @ np.array(b)
    slope = float(truth["line"].params["fiber"])
    assert check["reference"] == pytest.approx(float(knots[0]), rel=1e-12)
    assert [p["x"] for p in check["points"]] == pytest.approx(list(grid), rel=1e-12)
    assert [p["curve"] for p in check["points"]] == pytest.approx(list(want), rel=1e-7,
                                                                   abs=1e-9)
    assert [p["line"] for p in check["points"]] == pytest.approx(
        list(slope * (grid - knots[0])), rel=1e-7, abs=1e-9)


def test_with_a_bend_the_plans_check_reports_it_as_statsmodels_tests_it(bend_run):
    """Truth: statsmodels. On a table with a known bend the check's F test of the spline's 3
    nonlinear terms is statsmodels', and it reports the bend (p far below 0.001); the estimate is
    still the straight line's, statsmodels' straight-line fit (the check never replaces it). On
    the base no check was served."""
    truth = _statsmodels_check(bend_run["frame"])
    check = check_of(bend_run["effects"])
    assert check is not None
    _assert_check_is_statsmodels(check, truth)
    assert check["test"]["p"] < 1e-6
    primary = next(r for r in model_two(bend_run["effects"])["effects"] if r["feature"] == "fiber")
    assert primary["estimate"] == pytest.approx(float(truth["line"].params["fiber"]), rel=1e-8)
    low, high = truth["line"].conf_int(alpha=0.05).loc["fiber"]
    assert (primary["ci_low"], primary["ci_high"]) == pytest.approx((float(low), float(high)),
                                                                    rel=1e-8)
    assert check["caption"].startswith(
        "The plan's check of the straight line: the primary model refitted with a restricted "
        "cubic spline of `fiber` in its place, k = 5 by Harrell's rule")
    assert check["caption"].endswith("The estimate stays the straight line's: the check never "
                                     "replaces it.")


def test_on_a_straight_relation_the_check_reports_no_bend(line_run):
    """The other way: on a table whose truth is a line the same check is statsmodels', and its
    test does not report a bend (p above 0.05 here); the estimate is the straight line's."""
    truth = _statsmodels_check(line_run["frame"])
    check = check_of(line_run["effects"])
    assert check is not None
    _assert_check_is_statsmodels(check, truth)
    assert check["test"]["p"] > 0.05
    primary = next(r for r in model_two(line_run["effects"])["effects"] if r["feature"] == "fiber")
    assert primary["estimate"] == pytest.approx(float(truth["line"].params["fiber"]), rel=1e-8)


def test_the_methods_state_the_check_in_one_sentence(bend_run):
    """The methods text says what the plan set, in one sentence, its k and knots interpolated
    from what the check used."""
    check = check_of(bend_run["effects"])
    knots = ", ".join(f"{v:.4g}" for v in check["knots"][:-1]) + f" and {check['knots'][-1]:.4g}"
    sentence = (
        "As the plan's check of the straight line, the primary model was also fit with a "
        "restricted cubic spline of `fiber` in its place (k = 5 by Harrell's rule (3 knots below "
        "an effective sample size of 30, 5 from 100, else 4; here 600, the number of analyzed "
        f"rows), knots of fiber at {knots} (the 5th, 27.5th, 50th, 72.5th and 95th percentiles)), "
        "and its Wald test that the nonlinear terms are zero is reported beside the estimate, "
        "never replacing it.")
    assert check["methods"] == sentence
    assert sentence in bend_run["effects"]["methods"]


def test_a_yes_no_outcome_checks_the_line_by_the_logistic_models_chi_square(tmp_path):
    """The check on a logistic model: statsmodels' maximum-likelihood fit with the spline in
    place of the line, and its Wald χ² on the model's information."""
    frame = est_f.cohort(n=800, seed=5)
    x = frame["fiber"].to_numpy(dtype=float)
    st = est_f.state(target="dm", task="binary", measure="odds_ratio", event="yes",
                     exposure_forms={"fiber": LINE})
    out = run(frame, tmp_path, st)
    check = check_of(out["effects"])
    assert check is not None and check["refused"] is None, check
    # 800 rows, about 30% events: the effective sample size is the smaller of the two, ≥ 100.
    knots = np.quantile(x, FIVE, method="linear")
    np.testing.assert_allclose(check["knots"], knots, rtol=1e-12)
    fit = sm.Logit((frame["dm"] == "yes").astype(float), with_spline(frame, knots)).fit(
        disp=0, method="newton", tol=1e-12, maxiter=200)
    R = np.zeros((3, len(fit.params)))
    for j, name in enumerate(spline_names(5)):
        R[j, list(fit.params.index).index(name)] = 1.0
    wald = fit.wald_test(R, use_f=False, scalar=True)
    test = check["test"]
    assert (test["distribution"], test["df_num"], test["df_den"]) == ("chi2", 3, None)
    assert test["statistic"] == pytest.approx(float(wald.statistic), rel=1e-5)
    assert test["p"] == pytest.approx(float(wald.pvalue), rel=1e-4)


# ── the check of a curve ─────────────────────────────────────────────────────


def test_a_spline_kept_has_the_straight_lines_one_coefficient_beside_it(tmp_path):
    """The other shape the other way: a spline declared on `fiber` is checked by the primary
    model refitted with `fiber` as a straight line, its one coefficient statsmodels' straight-line
    fit with HC3; the estimate is the curve's terms."""
    frame = bent()
    spline = d.ExposureFormSpec(form="spline", knots=5, scale="raw")
    out = run(frame, tmp_path, numeric_state(exposure_forms={"fiber": spline}))
    check = check_of(out["effects"])
    assert check is not None and check["refused"] is None, check
    assert (check["kept"], check["shape"], check["column"]) == ("spline", "linear", "fiber")
    assert check["test"] is None and check["points"] == [] and check["knots"] is None
    line = sm.OLS(frame["glucose"].astype(float),
                  est_f.design_matrix(frame, ["fiber", *COVARIATES])).fit(cov_type="HC3",
                                                                          use_t=True)
    low, high = line.conf_int(alpha=0.05).loc["fiber"]
    got = check["line"]
    assert got["feature"] == "fiber"
    assert (got["estimate"], got["ci_low"], got["ci_high"]) == pytest.approx(
        (float(line.params["fiber"]), float(low), float(high)), rel=1e-8)
    assert check["caption"].startswith("The plan's check of the curve: the primary model refitted "
                                       "with `fiber` as a straight line, its one coefficient ")
    shown = {r["feature"] for r in model_two(out["effects"])["effects"]}
    assert shown == {"fiber", "fiber'", "fiber''", "fiber'''"}  # the curve stays the estimate
    assert check["methods"] in out["effects"]["methods"]


# ── silent where no shape is declared ────────────────────────────────────────


def test_no_check_without_a_declared_line_or_spline(tmp_path):
    """Silent where it should be: with no form declared for the exposure (a plan made outside the
    form question) or with quintiles declared, no check is set, and the methods say none."""
    frame = straight(n=400, seed=3)
    for name, forms in (("none", None), ("quintiles", {"fiber": d.ExposureFormSpec(
            form="quintiles", scale="raw")})):
        st = numeric_state(**({"exposure_forms": forms} if forms else {}))
        out = run(frame, tmp_path / name, st)
        assert check_of(out["effects"]) is None, name
        assert "plan's check" not in out["effects"]["methods"], name


# ── under multiple imputation ────────────────────────────────────────────────


@pytest.mark.slow
def test_under_multiple_imputation_the_check_is_pooled_by_d1_on_knots_fixed_on_the_observed_values(
        tmp_path):
    """Truth: statsmodels' least squares with HC3 in each completed copy (the copies the fit draws,
    the data each model is fit on), the knots at Harrell's percentiles of the observed `fiber`
    values written out with NumPy, fixed across copies, and D1 written out (Li, Raghunathan &
    Rubin 1991: r₁ = (1 + 1/m) tr(B Ū⁻¹)/k, D₁ = q̄ᵀ Ū⁻¹ q̄ / (k (1 + r₁)), ν = 4 + (t − 4)(1 +
    (1 − 2/t)/r₁)², t = k (m − 1))."""
    from turbotab.core.stages.modeling import design_stage, fit_stage
    from turbotab.core.tests import modeling_fixtures as mf

    frame = bent(n=500, seed=8)
    rng = np.random.default_rng(80)
    frame.loc[rng.random(len(frame)) < 0.15, "fiber"] = np.nan
    frame.loc[rng.random(len(frame)) < 0.15, "activity"] = np.nan
    st = numeric_state(exposure_forms={"fiber": LINE},
                       missing=d.MissingSpec(strategy="multiple_imputation"))
    out = run(frame, tmp_path / "e", st)
    # The completed copies, as the fit draws them (the effects stage pools the fit's own copies).
    paths = mf.ingest_frame(frame, tmp_path / "f")
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
    info = mf.target_info("regression", "glucose")
    design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
    fitted = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": info},
                                  paths))
    copies = fitted.objects["imputations"]["frames"]
    m = len(copies)
    assert m >= 20
    observed = frame["fiber"].dropna().to_numpy(dtype=float)
    knots = np.quantile(observed, FIVE, method="linear")
    Q, U, slopes = [], [], []
    y = frame["glucose"].astype(float).to_numpy()
    for copy in copies:
        part = copy.assign(glucose=y)
        fit = sm.OLS(part["glucose"], with_spline(part, knots)).fit(cov_type="HC3")
        names = spline_names(5)
        Q.append(fit.params[names].to_numpy())
        U.append(fit.cov_params().loc[names, names].to_numpy())
        slopes.append(float(sm.OLS(part["glucose"], est_f.design_matrix(
            part, ["fiber", *COVARIATES])).fit().params["fiber"]))
    Q, U = np.asarray(Q), np.asarray(U)
    k = Q.shape[1]
    qbar, ubar = Q.mean(axis=0), U.mean(axis=0)
    B = np.cov(Q, rowvar=False, ddof=1)
    r1 = (1 + 1 / m) * float(np.trace(B @ np.linalg.inv(ubar))) / k
    D1 = float(qbar @ np.linalg.inv(ubar) @ qbar) / (k * (1 + r1))
    t = k * (m - 1)
    nu = 4 + (t - 4) * (1 + (1 - 2 / t) / r1) ** 2
    check = check_of(out["effects"])
    assert check is not None and check["refused"] is None, check
    np.testing.assert_allclose(check["knots"], knots, rtol=1e-12)
    test = check["test"]
    assert (test["distribution"], test["df_num"]) == ("F", 3)
    assert test["statistic"] == pytest.approx(D1, rel=1e-6)
    assert test["df_den"] == pytest.approx(nu, rel=1e-6)
    assert test["p"] == pytest.approx(float(stats.f.sf(D1, k, nu)), rel=1e-6, abs=1e-300)
    assert f"(pooled over {m} imputations by Li, Raghunathan & Rubin's D1)" in check["caption"]
    assert "The knots are placed once on the observed values" in check["caption"]
    # The primary is the straight line's Rubin estimate (the copies are the effects stage's own).
    primary = next(r for r in model_two(out["effects"])["effects"] if r["feature"] == "fiber")
    assert primary["estimate"] == pytest.approx(float(np.mean(slopes)), rel=1e-8)
    assert math.isfinite(check["points"][-1]["curve"])
