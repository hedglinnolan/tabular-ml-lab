"""WP12 acceptance test 6: univariate regression calibration of energy-adjusted intakes
(docs/turbotab-next/audit/AUDIT_REPORT.md §5, WP12 · Methods a reviewer expects).

    6. "Univariate regression calibration of energy-adjusted intakes when two or more recalls
       exist. Source check: Freedman et al. 2011."

It supplies the family IN-22 points to ("prioritize univariate regression calibration of
energy-adjusted intakes when repeats exist").

**Source check.** Freedman LS, Schatzkin A, Midthune D, Kipnis V. Dealing with dietary measurement
error in nutritional cohort studies. *J Natl Cancer Inst* 2011;103:1086–1092 (PMC full text, read
2026-10-02):

* the recommendation: "performing statistical adjustment of relative risks, based on such
  validation data, if they exist, using univariate (only for energy-adjusted intakes such as
  densities or residuals) or multivariate regression calibration";
* its scope: "Note, however, that our recommendation refers to energy-adjusted intake variables used
  in the density or residual models. Univariate adjustment for the unadjusted intakes used in the
  standard and partition models is inappropriate because the attenuation factor for the nutrient
  would be too small; the multivariate adjustment is recommended in this case.";
* how: "the attenuation and contamination factors should be estimated from the validation study
  after adjustment for the (exactly measured) confounders included in the disease model. For
  example, for the univariate method, the attenuation factor is estimated as the linear slope of
  the reference instrument value on the FFQ value in a multiple regression that also includes the
  confounders", and "The univariate measurement error adjustment is simple, requiring only division
  of the unadjusted relative risk estimate by the 24-hour recall–based attenuation factor for that
  variable";
* the test: "For a single mismeasured exposure in the disease model, the usual statistical test of
  the null hypothesis (no exposure effect) remains theoretically valid even though the estimated
  relative risk is attenuated."

Here the recalls are both the instrument and their own replicates (the reliability design of
Carroll, Ruppert, Stefanski & Crainiceanu 2006, *Measurement Error in Nonlinear Models*, §4.4), not
an FFQ against a recall reference: each person's mean of k recalls is the exposure, and the spread
of their days estimates its error.

**References, each independent of the engine.**

* statsmodels ``MixedLM`` (REML) on the recalls with the model's covariates as fixed effects: its
  variance components give λ = σ²_b / (σ²_b + σ²_e / k) for k balanced recalls, and statsmodels
  OLS gives the uncorrected coefficient; Freedman's univariate correction is that coefficient
  divided by λ. With balanced recalls the calibration is exactly this; the tests hold it to
  10⁻⁶, the REML optimizer's own precision.
* Freedman's own recipe in the replicate design: the slope of one recall on the other with the
  confounders in the regression is the single-day attenuation factor λ₁; the mean of two recalls
  has λ₂ = 2λ₁ / (1 + λ₁) (Spearman–Brown). A different estimator of the same quantity: agreement
  to 0.002 at n = 20,000.
* The data-generating slope: by simulation the calibrated estimate is unbiased where the mean of
  recalls is attenuated by half, with balanced and unbalanced recalls, and its bootstrap interval
  covers the truth.
* Through the real server, on a long table of two recalls per person combined by the mean with the
  residual energy adjustment: the residual slope by statsmodels OLS on the person means, each day
  adjusted with it, MixedLM REML on the days, and statsmodels OLS on the people, all from the CSV.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from turbotab.core.methods.calibration import (CalibrationRefused, Replicates, logistic_fit,
                                               regression_calibration)
from turbotab.core.tests.acceptance.references import hc3_by_definition
from turbotab.core.tests.acceptance.server_drive import local_server, open_project


def _people(rng: np.random.Generator, n: int, k: np.ndarray, *, beta: float = 0.5,
            error_sd: float = 2.8, binary: bool = False):
    """True intake X given age and sex (var 4), recalls W = X + U (sd ``error_sd``), outcome on X."""
    age = rng.normal(50, 10, n)
    sex = rng.integers(0, 2, n).astype(float)
    x = 10 + 0.05 * age + 1.0 * sex + rng.normal(0, 2, n)
    person = np.repeat(np.arange(n), k)
    w = x[person] + rng.normal(0, error_sd, person.size)
    if binary:
        eta = -1 + beta * (x - 11) + 0.02 * (age - 50) - 0.3 * sex
        y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    else:
        y = 2 + beta * x + 0.03 * age - 0.4 * sex + rng.normal(0, 2, n)
    return age, sex, w, person, y


def _reml(w: np.ndarray, person: np.ndarray, age: np.ndarray, sex: np.ndarray) -> tuple[float, float]:
    """(σ²_b, σ²_e) of ``w ~ age + sex + (1 | person)`` by statsmodels MixedLM, REML."""
    long = pd.DataFrame({"w": w, "age": age[person], "sex": sex[person], "g": person})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm("w ~ age + sex", long, groups=long["g"]).fit(reml=True, method="bfgs",
                                                                     gtol=1e-12)
    return float(m.cov_re.iloc[0, 0]), float(m.scale)


# ── 6a · the attenuation factor and the correction ───────────────────────────


@pytest.mark.parametrize("k", [2, 3])
def test_6_balanced_recalls_give_the_reml_attenuation_and_freedmans_division(k):
    """With k recalls each, λ is the REML variance components' σ²_b / (σ²_b + σ²_e / k) and the
    calibrated coefficient is the uncorrected OLS coefficient divided by λ ("division of the
    unadjusted relative risk estimate by the … attenuation factor"), both to 10⁻⁶ (statsmodels'
    REML optimum is found numerically; the engine's moment estimates are closed-form)."""
    rng = np.random.default_rng(10 + k)
    n = 1200
    age, sex, w, person, y = _people(rng, n, np.full(n, k))
    rep = Replicates.of(w, person, n)
    w_bar = pd.Series(w).groupby(person).mean().to_numpy()
    X = np.column_stack([w_bar, age, sex])
    result = regression_calibration(rep, X, 0, y, n_boot=0)

    s2b, s2e = _reml(w, person, age, sex)
    lam = s2b / (s2b + s2e / k)
    naive = float(sm.OLS(y, sm.add_constant(X)).fit().params[1])
    assert result.attenuation == pytest.approx(lam, rel=1e-6)
    assert result.naive == pytest.approx(naive, rel=1e-10)
    assert result.estimate == pytest.approx(naive / lam, rel=1e-6)
    assert result.sigma2_u == pytest.approx(s2e, rel=1e-6)
    assert result.sigma2_xz == pytest.approx(s2b, rel=1e-6)
    assert result.recalls == {k: n} and result.n_repeat == n


def test_6_the_attenuation_factor_is_freedmans_slope_on_the_other_recall():
    """Freedman's univariate recipe, in the replicate design: the slope of one recall on the other
    "in a multiple regression that also includes the confounders" estimates the single-day λ₁ (the
    average of the two orderings is taken); the two-recall mean's is 2λ₁ / (1 + λ₁). At n = 20,000
    it agrees with the engine's λ to 0.002 on each of three datasets (the observed gaps are about
    10⁻⁴), and both are near the generating λ = 4 / (4 + 2.8² / 2) = 0.505."""
    for seed in range(3):
        rng = np.random.default_rng(seed)
        n = 20_000
        age, sex, w, person, y = _people(rng, n, np.full(n, 2))
        rep = Replicates.of(w, person, n)
        days = w.reshape(n, 2)
        result = regression_calibration(rep, np.column_stack([days.mean(1), age, sex]), 0, y, n_boot=0)
        Z = np.column_stack([age, sex])
        slopes = [sm.OLS(days[:, b], sm.add_constant(np.column_stack([days[:, a], Z]))).fit().params[1]
                  for a, b in ((0, 1), (1, 0))]
        lam1 = float(np.mean(slopes))
        lam2 = 2 * lam1 / (1 + lam1)
        assert abs(result.attenuation - lam2) < 0.002, (seed, result.attenuation, lam2)
        assert abs(result.attenuation - 4 / (4 + 2.8 ** 2 / 2)) < 0.02


@pytest.mark.parametrize("design", ["balanced", "unbalanced"])
def test_6_by_simulation_the_calibrated_slope_is_unbiased_and_its_interval_covers(design):
    """True slope 0.5; the mean of recalls is attenuated to about 0.25 (λ ≈ 0.5). Over 200 datasets
    of 500 people (two recalls each, or one to three), the calibrated estimate averages within 0.02
    of 0.5 (its Monte Carlo standard error is about 0.006) and the 95% bootstrap interval (100 refits
    over people) covers 0.5 in at least 90% of them (observed 0.945 and 0.970; the binomial Monte
    Carlo error at 0.95 is 0.015). About 8 seconds."""
    rng = np.random.default_rng(2026)
    beta, reps = 0.5, 200
    naive, calibrated, covered = [], [], 0
    for r in range(reps):
        n = 500
        k = np.full(n, 2) if design == "balanced" else rng.choice([1, 2, 3], n, p=[0.3, 0.5, 0.2])
        age, sex, w, person, y = _people(rng, n, k, beta=beta)
        rep = Replicates.of(w, person, n)
        X = np.column_stack([rep.means(), age, sex])
        result = regression_calibration(rep, X, 0, y, n_boot=100, seed=r)
        naive.append(result.naive)
        calibrated.append(result.estimate)
        covered += result.ci_low <= beta <= result.ci_high
    print(f"\n[{design}] naive {np.mean(naive):.3f} | calibrated {np.mean(calibrated):.3f} "
          f"(MC se {np.std(calibrated) / np.sqrt(reps):.4f}) | coverage {covered / reps:.3f}")
    assert np.mean(naive) < 0.3  # the attenuation the calibration has to undo
    assert abs(np.mean(calibrated) - beta) < 0.02
    assert covered / reps >= 0.90


def test_6_for_a_logistic_model_the_calibration_is_the_usual_approximation():
    """For a logistic outcome model, substituting E[X | W̄, Z] is an approximation (Carroll et al.
    §4.2), close when the effect is moderate. Log-odds slope 0.4 per unit of true intake, two recalls,
    λ ≈ 0.5: over 200 datasets of 1,500 people the mean of recalls gives about half the slope, the
    calibrated estimate within 10% of it (observed 0.374, 6.5% low)."""
    rng = np.random.default_rng(7)
    beta = 0.4
    naive, calibrated = [], []
    for _ in range(200):
        n = 1500
        age, sex, w, person, y = _people(rng, n, np.full(n, 2), beta=beta, binary=True)
        rep = Replicates.of(w, person, n)
        X = np.column_stack([rep.means(), age, sex])
        result = regression_calibration(rep, X, 0, y, fit=logistic_fit, n_boot=0)
        naive.append(result.naive)
        calibrated.append(result.estimate)
    assert np.mean(naive) / beta < 0.6
    assert abs(np.mean(calibrated) / beta - 1) < 0.10


def test_6_without_two_recalls_from_anyone_there_is_nothing_to_calibrate():
    """One recall each: no day-to-day variance can be estimated, and the method says so."""
    rng = np.random.default_rng(1)
    n = 300
    age, sex, w, person, y = _people(rng, n, np.ones(n, dtype=int))
    rep = Replicates.of(w, person, n)
    with pytest.raises(CalibrationRefused, match="two or more recalls"):
        regression_calibration(rep, np.column_stack([rep.means(), age, sex]), 0, y, n_boot=0)


# ── 6b · through the server ──────────────────────────────────────────────────


TRUE_SLOPE = 0.8  # ldl per g of energy-adjusted protein (usual intake)


def recall_table(seed: int = 5, n: int = 800) -> pd.DataFrame:
    """Two 24-hour recalls per person (long format): energy and protein vary from day to day around
    each person's usual intake; LDL depends on usual energy-adjusted protein, age and sex."""
    rng = np.random.default_rng(seed)
    age = rng.integers(25, 75, n).astype(float)
    sex = rng.choice(["F", "M"], n)
    energy = rng.normal(2100, 350, n) + 250 * (sex == "M")
    protein = 0.035 * energy + 6 * rng.standard_normal(n) + 0.05 * (age - 50)
    ldl = (100 + TRUE_SLOPE * (protein - 0.035 * (energy - 2100)) + 0.2 * age + 4 * (sex == "M")
           + rng.normal(0, 8, n))
    rows = []
    for i in range(n):
        for _ in range(2):
            e = energy[i] + rng.normal(0, 500)
            rows.append({"participant_id": f"P{i:04d}", "age": age[i], "sex": sex[i],
                         "energy_kcal": round(e, 1),
                         "protein_g": round(protein[i] * e / energy[i] + rng.normal(0, 12), 2),
                         "ldl": round(ldl[i], 1)})
    return pd.DataFrame(rows)


ROLES = {"participant_id": "identifier", "age": "covariate", "sex": "covariate",
         "energy_kcal": "energy", "protein_g": "exposure"}
RESIDUAL = {"kind": "set_energy_adjustment", "method": "residual", "energy_column": "energy_kcal",
            "nutrients": ["protein_g"]}


def up_to_the_models(d, purpose: str) -> None:
    d.decide({"kind": "set_lens", "lenses": ["dietary"]})
    d.reach("target")
    d.decide({"kind": "set_target", "column": "ldl"})
    d.answer("task", {"kind": "set_task", "column": "ldl", "task": "regression"})
    d.reach("purpose")
    d.decide({"kind": "set_purpose", "purpose": purpose})
    d.reach("grain")
    d.decide({"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"})
    d.answer("repeat_kind", {"kind": "set_repeat_kind", "repeat_kind": "repeats"})
    d.answer("unit", {"kind": "set_unit", "unit": "unit"})
    d.answer("aggregation", {"kind": "set_aggregation", "method": "mean"})
    d.answer("temporal", {"kind": "set_temporal", "temporal": False})
    d.reach("roles")
    d.decide({"kind": "set_roles", "roles": ROLES})
    d.reach("exclusions")
    d.decide({"kind": "set_exclusions", "rules": []})
    d.reach("missing")
    d.decide({"kind": "set_missing", "strategy": "complete_case"})
    d.reach("split")
    d.decide({"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
    d.answer("energy_adjustment", RESIDUAL)
    d.reach("models")
    d.decide({"kind": "select_models", "models": ["linear"]})


@pytest.fixture(scope="module")
def recalls(tmp_path_factory) -> tuple[pd.DataFrame, Path]:
    frame = recall_table()
    path = tmp_path_factory.mktemp("wp12c_recalls") / "recalls.csv"
    frame.to_csv(path, index=False)
    return frame, path


@pytest.fixture(scope="module")
def runs(recalls, tmp_path_factory) -> dict:
    """One inference project: the calibration under the residual method, then the same project's
    answer changed to the standard model, then to "no correction"."""
    _, path = recalls
    out = {}
    with local_server(tmp_path_factory.mktemp("wp12c_calibration_home")) as client:
        d = open_project(client, path)
        up_to_the_models(d, "inference")
        d.decide({"kind": "set_measurement_error", "method": "regression_calibration", "n_boot": 200})
        out["residual"] = d.artifact("calibration")
        out["fit"] = d.artifact("fit")
        d.decide({"kind": "set_energy_adjustment", "method": "standard", "energy_column": "energy_kcal",
                  "nutrients": ["protein_g"]})
        out["standard"] = d.artifact("calibration")
        d.decide({"kind": "set_measurement_error", "method": "none"})
        out["none"] = d.artifact("calibration")
    return out


def reference(frame: pd.DataFrame, energy_in_model: bool) -> dict:
    """Everything from the CSV, with statsmodels and pandas: the residual adjustment on person
    means, each day adjusted with the same slope, REML components of the days, OLS on the people."""
    people = frame.groupby("participant_id", sort=True).agg(
        protein=("protein_g", "mean"), energy=("energy_kcal", "mean"), age=("age", "first"),
        sex=("sex", "first"), ldl=("ldl", "first"), k=("protein_g", "size"))
    slope = float(sm.OLS(people["protein"], sm.add_constant(people["energy"])).fit().params["energy"])
    center = float(people["energy"].mean())
    people["adj"] = people["protein"] - slope * (people["energy"] - center)
    days = frame.assign(adj=frame["protein_g"] - slope * (frame["energy_kcal"] - center))
    codes = pd.Categorical(days["participant_id"], categories=people.index).codes
    male = (people["sex"] == "M").astype(float).to_numpy()
    covariates = ["age", "male"] + (["energy"] if energy_in_model else [])
    people["male"] = male
    long = pd.DataFrame({"w": days["adj"].to_numpy(), "g": codes,
                         **{c: people[c].to_numpy()[codes] for c in covariates}})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm("w ~ " + " + ".join(covariates), long, groups=long["g"]).fit(
            reml=True, method="bfgs", gtol=1e-12)
    s2b, s2e = float(m.cov_re.iloc[0, 0]), float(m.scale)
    lam = s2b / (s2b + s2e / 2)
    X = np.column_stack([np.ones(len(people)), people["adj"], *[people[c] for c in covariates]])
    y = people["ldl"].to_numpy(dtype=float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    se = np.sqrt(np.diag(hc3_by_definition(X, y - X @ beta)))
    t = stats.t.ppf(0.975, len(y) - X.shape[1])
    return {"lam": lam, "naive": beta[1], "naive_ci": (beta[1] - t * se[1], beta[1] + t * se[1]),
            "p": 2 * stats.t.sf(abs(beta[1] / se[1]), len(y) - X.shape[1]), "s2b": s2b, "s2e": s2e,
            "n": len(people)}


def test_6_the_server_calibrates_the_residual_protein_as_an_independent_refit_does(recalls, runs):
    """Two recalls per person combined by the mean, protein energy-adjusted by the residual method,
    purpose inference: the calibration artifact's λ, within- and between-person variances and the
    corrected coefficient match the statsmodels reference from the CSV to 10⁻⁶; the uncorrected
    coefficient, its HC3 interval and its p-value (Freedman: the test that stays valid) match the
    independent least squares on every eligible person; the attenuation the correction removes is
    large on this fixture (λ ≈ 0.33), and the corrected interval covers the generating slope 0.8
    where the uncorrected one does not."""
    frame, _ = recalls
    cal, fit = runs["residual"], runs["fit"]
    assert cal["applies"] and cal["purpose"] == "inference" and cal["rows"] == "all eligible rows"
    features = {c["feature"] for c in fit["models"][0]["coefficients"]}
    ref = reference(frame, energy_in_model="energy_kcal" in features)
    assert cal["n_persons"] == ref["n"] and cal["recalls"] == {"2": ref["n"]}
    (exposure,) = cal["exposures"]
    assert (exposure["source"], exposure["operation"]) == ("protein_g", "residual")
    assert exposure["refused"] is None
    assert exposure["attenuation"] == pytest.approx(ref["lam"], rel=1e-6)
    assert exposure["within_variance"] == pytest.approx(ref["s2e"], rel=1e-6)
    assert exposure["between_variance"] == pytest.approx(ref["s2b"], rel=1e-6)
    assert exposure["naive"] == pytest.approx(ref["naive"], rel=1e-8)
    # The fit stage's own table now estimates from every eligible row too (WP8, merged after this
    # package), so the coefficient the calibration corrects is the one the fit reports.
    reported = next(c for c in fit["models"][0]["coefficients"] if c["feature"] == exposure["feature"])
    assert reported["estimate"] == pytest.approx(exposure["naive"], rel=1e-9)
    assert exposure["estimate"] == pytest.approx(ref["naive"] / ref["lam"], rel=1e-6)
    assert exposure["naive_ci_low"] == pytest.approx(ref["naive_ci"][0], rel=1e-7)
    assert exposure["naive_ci_high"] == pytest.approx(ref["naive_ci"][1], rel=1e-7)
    assert exposure["p"] == pytest.approx(ref["p"], rel=1e-6)
    assert exposure["n_boot"] == exposure["n_boot_ok"] == 200
    assert exposure["ci_low"] < exposure["estimate"] < exposure["ci_high"]
    assert 0.25 < ref["lam"] < 0.4
    assert not exposure["naive_ci_low"] <= TRUE_SLOPE <= exposure["naive_ci_high"]
    assert exposure["ci_low"] <= TRUE_SLOPE <= exposure["ci_high"]
    # The methods sentence names the method, its sources, the recall days and λ.
    methods = cal["methods"]
    assert "univariate regression calibration" in methods and "Freedman et al. 2011" in methods
    assert "2 recalls each" in methods and f"λ = {exposure['attenuation']:.2f}" in methods
    assert any("remains theoretically valid" in a for a in cal["assumptions"])


def test_6_under_the_standard_model_univariate_calibration_is_refused_with_freedmans_reason(runs):
    """Freedman: univariate adjustment "for the unadjusted intakes used in the standard and partition
    models is inappropriate"; the stage says so rather than calibrate."""
    cal = runs["standard"]
    assert not cal["applies"] and not cal["exposures"]
    assert "Univariate adjustment for the unadjusted intakes used in the standard and partition " \
           "models is inappropriate" in cal["reason"]


def test_6_no_correction_still_states_the_recall_days(runs):
    """Answered "no correction": the methods sentence still says how many recalls each person's
    mean averages (IN-22's limitation line: the number of recall days, and no calibration)."""
    cal = runs["none"]
    assert cal["method"] == "none" and not cal["applies"]
    assert cal["methods"] == ("Energy-adjusted exposures were the mean of each participant's "
                              "recalls (2 recalls each, 800 participants) and were not corrected "
                              "for day-to-day error.")


def test_6_under_prediction_calibration_is_refused_with_exits(recalls, tmp_path_factory):
    """The leash is right for prediction (audit IN-22): the model is used on recalls measured the
    same way, so the answer is refused, with "keep the mean uncorrected" as the way forward."""
    _, path = recalls
    with local_server(tmp_path_factory.mktemp("wp12c_prediction_home")) as client:
        d = open_project(client, path)
        d.decide({"kind": "set_lens", "lenses": ["dietary"]})
        d.reach("target")
        d.decide({"kind": "set_target", "column": "ldl"})
        d.answer("task", {"kind": "set_task", "column": "ldl", "task": "regression"})
        d.reach("purpose")
        d.decide({"kind": "set_purpose", "purpose": "prediction"})
        r = d.post({"kind": "set_measurement_error", "method": "regression_calibration"})
        assert r.status_code == 409, r.text
        error = r.json()["error"]
        assert error["code"] == "not_for_prediction"
        assert error["exits"][0]["decision"] == {"kind": "set_measurement_error", "method": "none",
                                                 "exposures": [], "n_boot": 200}
        assert d.post({"kind": "set_measurement_error", "method": "none"}).status_code == 200


def test_6_rows_not_combined_from_recalls_are_not_calibrated(recalls, tmp_path_factory):
    """Each recall kept as its own row (unit = row): there is no person-level mean to correct, and
    the stage says what to change."""
    _, path = recalls
    with local_server(tmp_path_factory.mktemp("wp12c_rows_home")) as client:
        d = open_project(client, path)
        d.decide({"kind": "set_lens", "lenses": ["dietary"]})
        d.reach("target")
        d.decide({"kind": "set_target", "column": "ldl"})
        d.answer("task", {"kind": "set_task", "column": "ldl", "task": "regression"})
        d.reach("purpose")
        d.decide({"kind": "set_purpose", "purpose": "inference"})
        d.reach("grain")
        d.decide({"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"})
        d.answer("repeat_kind", {"kind": "set_repeat_kind", "repeat_kind": "repeats"})
        d.answer("unit", {"kind": "set_unit", "unit": "row"})
        d.answer("temporal", {"kind": "set_temporal", "temporal": False})
        d.reach("roles")
        d.decide({"kind": "set_roles", "roles": ROLES})
        d.reach("exclusions")
        d.decide({"kind": "set_exclusions", "rules": []})
        d.reach("missing")
        d.decide({"kind": "set_missing", "strategy": "complete_case"})
        d.reach("split")
        d.decide({"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
        d.answer("energy_adjustment", RESIDUAL)
        d.reach("models")
        d.decide({"kind": "select_models", "models": ["linear"]})
        d.decide({"kind": "set_measurement_error", "method": "regression_calibration"})
        cal = d.artifact("calibration")
    assert not cal["applies"]
    assert "not the mean of a person's repeated recalls" in cal["reason"]
