"""WP7 · Missing data by purpose (docs/turbotab-next/audit/AUDIT_REPORT.md §5; closes ME-01 and
ME-08, and the minor E14, B23, F14). BLUEPRINT §12 ruling 4: "Inference uses multiple imputation with
the outcome (and energy) in the imputation model, pooled by Rubin's rules (Moons 2006; Sterne 2009;
Sisk 2023). Prediction imputes in-fold without the outcome, so the fitted pipeline can be deployed."

The package's seven acceptance tests, in its order:

1. **Inference.** "Confounder 35–44% missing at random given the exposure: multiple imputation
   (chained equations with the outcome and energy, m ≥ 20, Rubin's rules) gives |bias| < 0.01 and
   coverage ≥ 0.93 (reference today: median fill bias +38–75%, coverage 0.00; MI −0.002, coverage
   0.92). Complete cases stay available with their assumption stated. Median fill and indicators are
   blocked and recorded for the inference table."
2. **Prediction.** "In-fold, outcome-free imputation is unchanged and its CV results reproduce today's
   to 10⁻⁹."
3. **Energy-aware fill.** "With 20% of protein missing, imputed rows' energy-adjusted values correlate
   with energy like observed rows (reference today: −1.000 against +0.78)."
4. **Methods sentence.** "It names the method ('median for numbers, most frequent for categories',
   or 'multiple imputation, m = 20'). Source check: STROBE-nut nut-13."
5. **One rule.** "ROADMAP §07, the M2_CONTRACT Tier A test and the MISSING drawer state the same
   purpose-conditional rule, citing Sisk et al. 2023 and Moons 2006 (via Harrell)."
6. **Below detection.** "When left-censoring fires, median fill is refused unless overridden with a
   reason; half-minimum and a censoring-aware option are offered; on the log-scale censoring
   simulation the censoring-aware estimate stays within 5% of 0.5 at 60% censoring (reference:
   detected-only 0.497). Zeros are recoded as non-detects only after the user says so."
7. **Row loss.** "Under inference, losing more than 10% of rows to complete cases raises a concern
   with a comparison of kept and dropped rows (CLINICAL_SURVEY_PACK's own threshold)."

**Independent references.** The simulations' truth is the data-generating coefficient (0.5), an
analytic reference. Rubin's rules are checked against the closed forms written out here from van
Buuren, *Flexible Imputation of Missing Data*, 2nd ed. (§2.3, eq. 2.16–2.32; R's ``mice::pool``,
which R is not installed to run), applied to statsmodels fits of the app's own completed tables;
the multivariate D1 against §5.3.2's formula. Today's prediction results are the output of the fit
stage at commit 514336c (``wp7_prediction_reference.json``), and the linear family's cross-validated
scores are rebuilt with scikit-learn alone. Correlations and lines are NumPy's.

**Source checks** (read for this package):

* Sisk R, Sperrin M, Peek N, van Smeden M, Martin GP. *Stat Methods Med Res* 2023;32(8):1461–1477,
  abstract: "With complete data available at deployment, our findings were in line with existing
  recommendations; that the outcome should be used to impute development data when using multiple
  imputation and omitted under regression imputation. When missingness is allowed at deployment,
  omitting the outcome from the imputation model at the development was preferred."
* Moons KG, Donders RA, Stijnen T, Harrell FE. *J Clin Epidemiol* 2006;59:1092, abstract: "MI
  without outcome yielded very biased--underestimated--coefficients … For all types of missing
  values, imputation of missing predictor values using the outcome is preferred over imputation
  without outcome and is no self-fulfilling prophecy." Harrell, *Regression Modeling Strategies*
  (hbiostat.org/rmsc/missing, §3.8): "Note that multiple imputation can and should use the response
  variable for imputing predictors", citing Moons et al. 2006.
* STROBE-nut (Lachat et al. 2016, *PLoS Med* 13:e1002036), item nut-13: "Report the number of
  individuals excluded based on missing, incomplete, or implausible dietary/nutritional data"; its
  explanation asks authors to "describe the number of missing values, cut-offs for implausible data
  leading to exclusion, characteristics of those excluded, and any method used to handle missing
  values."
* Groenwold RH et al. *CMAJ* 2012;184:1265: a complete-case association "is unbiased only if
  missingness is conditionally independent of the outcome"; "the missing-indicator method will
  almost always give biased results" in nonrandomized studies.
* Lubin JH et al. *Environ Health Perspect* 2004;112:1691: "Truncated data methods (e.g., Tobit
  regression) and multiple imputation offer two unbiased approaches for analyzing measurement data
  with detection limits."
* CLINICAL_SURVEY_PACK, anti-pattern 1: "Flag when listwise deletion drops >10% of rows."

Monte Carlo error is stated beside each simulated bound.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.decisions import (EnergyAdjustment, MissingSpec, ProjectState, Refusal,
                                     SetMissing, validate)
from turbotab.core.methods.imputation import pooled_wald
from turbotab.core.methods.missing import RULE, impute_for_inference, row_loss, row_loss_concern
from turbotab.core.models import get_family
from turbotab.core.models.inference import INDEPENDENT, Outcome
from turbotab.core.models.inner_cv import fit_pipeline
from turbotab.core.models.pipeline import (build_pipeline, design_spec, model_predictors,
                                           shared_steps, transformer)
from turbotab.core.stages.modeling import _inference_table, pooled_table
from turbotab.core.tests.acceptance.server_drive import local_server, open_project

REPO = Path(__file__).resolve().parents[4]
LINEAR = get_family("linear")
BETA = 0.5


# ── 1 · inference: multiple imputation with the outcome ──────────────────────


def confounder_table(rng: np.random.Generator, n: int = 1000) -> pd.DataFrame:
    """The audit's fixture (``repro.tar.gz``: ``E/simD_missing.py``): y = 0.5·x + z + e with the
    confounder z = 0.7-correlated with the exposure x, and z blank with probability
    expit(−0.6 + 1.5·x/sd(x)), missing at random given the exposure (about 40% blank; the audit's
    intercept −1.0 gave 35%)."""
    z = rng.normal(size=n)
    x = 0.7 * z + rng.normal(size=n)
    y = BETA * x + z + rng.normal(size=n)
    d = (x - x.mean()) / x.std()
    p = 1 / (1 + np.exp(-(-0.6 + 1.5 * d)))
    return pd.DataFrame({"x": x, "z": np.where(rng.random(n) < p, np.nan, z), "y": y})


def _state(purpose: str, missing: MissingSpec) -> ProjectState:
    return ProjectState(target="y", task="regression", purpose=purpose,
                        roles={"x": "exposure", "z": "covariate"}, missing=missing)


def app_row(frame: pd.DataFrame, missing: MissingSpec, seed: int = 0, feature: str = "x") -> dict:
    """The inference table's row for ``feature``, made as the fit stage makes it: the design's
    pipeline, then ``pooled_table`` over ``impute_for_inference`` under multiple imputation, or the
    family's table on the pipeline refit on every analyzed row (complete cases: on the complete
    ones)."""
    st = _state("inference", missing)
    X, y = frame[["x", "z"]], frame["y"].to_numpy()
    if missing.strategy == "complete_case":
        keep = X.notna().all(axis=1).to_numpy()
        X, y = X[keep], y[keep]
    spec = design_spec(st, X, ["x", "z"])
    pipeline = build_pipeline(spec, LINEAR, "regression", "inference", len(X), 2)
    if missing.strategy == "multiple_imputation":
        imputations = impute_for_inference(spec, X, y, "regression", seed=seed)
        _, rows, _, _ = pooled_table(
            LINEAR, pipeline, imputations, y, task="regression", clusters=INDEPENDENT,
            outcome=Outcome(name="y"), survey=None, design=None, spec=spec, rows="all",
            fit=lambda model, X_k: fit_pipeline(model, X_k, y))
    else:
        fitted = fit_pipeline(pipeline, X, y)
        rows = _inference_table(LINEAR, fitted, X, y, task="regression", clusters=INDEPENDENT,
                                outcome=Outcome(name="y"), rows="all").rows
    return next(r for r in rows if r["feature"] == feature)


def test_1_multiple_imputation_is_unbiased_and_its_intervals_cover():
    """400 datasets of 1,000 rows, the confounder about 40% missing at random given the exposure.
    Multiple imputation (m = 20, the outcome in the imputation model): |bias| < 0.01 and 95% coverage
    ≥ 0.93. Monte Carlo error: bias ±0.002 (one SE), coverage ±0.011. Beside it, on the same data,
    the median fill the inference table now blocks (kept here with its recorded attestation) and
    complete cases, whose assumption holds here because the blanks depend on the exposure, not the
    outcome. Measured when written: 39.5% blank; multiple imputation bias +0.0013, coverage 0.9525;
    median fill +0.297 (+59%), coverage 0.000 (the audit's +38–75%, 0.00); complete cases −0.0004,
    coverage 0.955."""
    rng = np.random.default_rng(20261002)
    found = {"mi": [], "median": [], "complete": []}
    shares = []
    for rep in range(400):
        frame = confounder_table(rng)
        shares.append(float(frame["z"].isna().mean()))
        for key, missing in (("mi", MissingSpec(strategy="multiple_imputation")),
                             ("median", MissingSpec(strategy="impute", acknowledged=True)),
                             ("complete", MissingSpec(strategy="complete_case"))):
            row = app_row(frame, missing, seed=rep)
            found[key].append((row["estimate"], row["ci_low"] <= BETA <= row["ci_high"]))
    assert 0.35 <= np.mean(shares) <= 0.44
    est = np.asarray([e for e, _ in found["mi"]])
    cover = np.mean([c for _, c in found["mi"]])
    mc = est.std(ddof=1) / np.sqrt(len(est))
    assert abs(est.mean() - BETA) < 0.01, (est.mean(), mc)
    assert cover >= 0.93, cover
    # The reference today: a median fill is biased upward and its intervals miss the truth.
    median = np.asarray([e for e, _ in found["median"]])
    assert median.mean() - BETA > 0.15 and np.mean([c for _, c in found["median"]]) < 0.05
    complete = np.asarray([e for e, _ in found["complete"]])
    assert abs(complete.mean() - BETA) < 0.01


def test_1_multiple_imputation_with_the_outcome_is_refused_under_prediction():
    """Ruling 4's other half: under prediction the imputation must not read the outcome (a held-out
    row's outcome would fill its own predictors; a new row has none), so multiple imputation is
    refused with the in-fold fill and complete cases as exits; indicators beside it are refused
    under inference."""
    for purpose in ("prediction", None):
        state = ProjectState(target="y", roles={"x": "exposure", "z": "covariate"}, purpose=purpose)
        with pytest.raises(Refusal) as refused:
            validate(SetMissing(strategy="multiple_imputation"), {"state": state})
        assert refused.value.code == "imputation_with_the_outcome"
        assert [e["decision"]["strategy"] for e in refused.value.exits if e["decision"]] == [
            "impute", "complete_case"]
        validate(SetMissing(strategy="impute", indicators=True), {"state": state})
    state = ProjectState(target="y", roles={"x": "exposure", "z": "covariate"}, purpose="inference")
    with pytest.raises(Refusal) as refused:
        validate(SetMissing(strategy="multiple_imputation", indicators=True), {"state": state})
    assert refused.value.code == "indicators_with_imputation"


def test_1_rubins_rules_and_d1_match_the_formulas():
    """The pooled row equals Rubin's rules written out (van Buuren 2018, eq. 2.16–2.32) over
    statsmodels' own HC3 fits of the app's completed tables: Q̄, T = Ū + (1 + 1/m)B, Barnard &
    Rubin's ν with ν_com = n − 3, the t interval, the p-value and the fraction of missing
    information γ. D1 (§5.3.2) for x and z together equals ``pooled_wald``."""
    frame = confounder_table(np.random.default_rng(7))
    y = frame["y"].to_numpy()
    st = _state("inference", MissingSpec(strategy="multiple_imputation"))
    spec = design_spec(st, frame[["x", "z"]], ["x", "z"])
    imputations = impute_for_inference(spec, frame[["x", "z"]], y, "regression", seed=3)
    row = app_row(frame, MissingSpec(strategy="multiple_imputation"), seed=3)
    m = imputations.m
    # MS2 (MODELING_SEQUENCE §1 step 6): m ≥ max(20, the percentage of rows with an imputed value).
    assert m == max(20, int(np.ceil(100 * frame["z"].isna().mean())))
    assert imputations.imputed == {"z": int(frame["z"].isna().sum())}
    Q, U, V = [], [], []
    for f in imputations.frames:
        fit = sm.OLS(y, sm.add_constant(f[["x", "z"]])).fit(cov_type="HC3")
        Q.append(fit.params.to_numpy())
        V.append(fit.cov_params().to_numpy())
        U.append(fit.bse["x"] ** 2)
    Q = np.asarray(Q)
    q = Q[:, 1]
    qbar, ubar, b = q.mean(), np.mean(U), q.var(ddof=1)
    t_var = ubar + (1 + 1 / m) * b
    lam = (1 + 1 / m) * b / t_var
    nu_com = len(y) - 3
    nu_old = (m - 1) / lam ** 2
    nu_obs = (nu_com + 1) / (nu_com + 3) * nu_com * (1 - lam)
    nu = nu_old * nu_obs / (nu_old + nu_obs)
    r = (1 + 1 / m) * b / ubar
    gamma = (r + 2 / (nu + 3)) / (1 + r)
    half = stats.t.ppf(0.975, nu) * np.sqrt(t_var)
    assert row["estimate"] == pytest.approx(qbar, abs=1e-10)
    assert row["se"] == pytest.approx(np.sqrt(t_var), rel=1e-9)
    assert row["df"] == pytest.approx(nu, rel=1e-9)
    assert (row["ci_low"], row["ci_high"]) == (pytest.approx(qbar - half, abs=1e-9),
                                               pytest.approx(qbar + half, abs=1e-9))
    assert row["p"] == pytest.approx(2 * stats.t.sf(abs(qbar) / np.sqrt(t_var), nu), rel=1e-7)
    assert row["fmi"] == pytest.approx(gamma, rel=1e-9)
    # D1: x and z together, against the formula.
    Qk = Q[:, 1:]
    Uk = np.asarray([v[1:, 1:] for v in V])
    k = 2
    ubar_m = Uk.mean(axis=0)
    B = np.cov(Qk, rowvar=False, ddof=1)
    r1 = (1 + 1 / m) * np.trace(B @ np.linalg.inv(ubar_m)) / k
    d1 = Qk.mean(axis=0) @ np.linalg.inv(ubar_m) @ Qk.mean(axis=0) / (k * (1 + r1))
    t = k * (m - 1)
    nu1 = 4 + (t - 4) * (1 + (1 - 2 / t) / r1) ** 2
    got = pooled_wald(Qk, Uk)
    assert got["statistic"] == pytest.approx(d1, rel=1e-10)
    assert got["df_den"] == pytest.approx(nu1, rel=1e-10)
    assert got["p"] == pytest.approx(stats.f.sf(d1, k, nu1), rel=1e-8)


# The server runs: an NHANES-shaped dietary table with an incomplete confounder and protein.


ROLES = {"participant_id": "identifier", "age": "covariate", "sex": "covariate",
         "fiber_g": "exposure", "protein_g": "exposure", "kcal": "energy"}


def diet_table(seed: int = 7, n: int = 1200, age_blank: bool = True,
               protein_share: float = 0.2) -> pd.DataFrame:
    """ldl = 130 − 0.5·fiber + 0.4·(age − 50) + …; age (a confounder of fiber) blank at random given
    fiber, about 40%, and protein blank completely at random (``protein_share``)."""
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 10, n)
    fiber = 20 + 0.3 * (age - 50) + rng.normal(0, 5, n)
    kcal = rng.normal(2100, 400, n)
    protein = (0.16 + rng.normal(0, 0.03, n)) * kcal / 4
    sex = rng.choice(["female", "male"], n)
    ldl = (130 - 0.5 * fiber + 0.4 * (age - 50) + 0.01 * protein + 0.002 * kcal
           + 3 * (sex == "male") + rng.normal(0, 10, n))
    zf = (fiber - fiber.mean()) / fiber.std()
    if age_blank:
        age = np.where(rng.random(n) < 1 / (1 + np.exp(-(-0.6 + 1.5 * zf))), np.nan, age)
    protein = np.where(rng.random(n) < protein_share, np.nan, protein)
    return pd.DataFrame({"participant_id": [f"P{i:04d}" for i in range(n)], "age": age.round(1),
                         "sex": sex, "fiber_g": fiber.round(2), "protein_g": protein.round(2),
                         "kcal": kcal.round(1), "ldl": ldl.round(2)})


# WP17: the generator's causal truth (``diet_table``): age sets fiber and LDL (a confounder); sex
# and protein set LDL only.
TRUTH = {"adjust:age": "yes,yes,no", "adjust:sex": "no,yes,no", "adjust:protein_g": "no,yes,no"}


def drive_to_missing(client, path: Path, purpose: str):
    from turbotab.core.tests.truths import Truth

    d = open_project(client, path, Truth(TRUTH, fixture="diet_table"))
    d.exposure = "fiber_g"
    d.decide({"kind": "set_lens", "lenses": ["dietary"]})
    d.reach("target")
    d.decide({"kind": "set_target", "column": "ldl"})
    d.answer("task", {"kind": "set_task", "column": "ldl", "task": "regression"})
    d.reach("purpose")
    d.decide({"kind": "set_purpose", "purpose": purpose})
    d.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit",
                       "id_column": "participant_id"})
    d.reach("roles")
    d.decide_roles(ROLES)
    d.reach("exclusions")
    d.decide({"kind": "set_exclusions", "rules": []})
    d.reach("missing")
    return d


def finish(d) -> dict:
    d.reach("split")
    d.decide({"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
    d.answer("energy_adjustment", {"kind": "set_energy_adjustment", "method": "standard",
                                   "energy_column": "kcal", "nutrients": ["fiber_g", "protein_g"]})
    d.reach("models")
    d.decide({"kind": "select_models", "models": ["linear"]})
    return d.artifact("fit")["models"][0]


def missing_sentence(d) -> str:
    said = [x["sentence"] for x in d.view()["decisions"] if x["decision"]["kind"] == "set_missing"]
    return said[-1].replace("`", "")


@pytest.fixture(scope="module")
def inference_runs(tmp_path_factory):
    """Through the real server, under inference: the refusals the missing-values answer meets, a
    single fill kept with its attestation, then multiple imputation; and complete cases."""
    home = tmp_path_factory.mktemp("wp7_home")
    path = tmp_path_factory.mktemp("wp7_table") / "diet_missing.csv"
    table = diet_table()
    table.to_csv(path, index=False)
    out: dict = {"table": table}
    with local_server(home) as client:
        d = drive_to_missing(client, path, "inference")
        posted = {key: d.post(body) for key, body in (
            ("single", {"kind": "set_missing", "strategy": "impute"}),
            ("indicators", {"kind": "set_missing", "strategy": "impute", "indicators": True}),
            ("level", {"kind": "set_missing", "strategy": "complete_case",
                       "categorical": "missing_category"}))}
        out["status"] = {key: r.status_code for key, r in posted.items()}
        out["refused"] = {key: r.json() for key, r in posted.items()}
        d.decide({"kind": "set_missing", "strategy": "impute", "acknowledged": True})
        out["recorded_sentence"] = missing_sentence(d)
        out["recorded"] = finish(d)
        d.decide({"kind": "set_missing", "strategy": "multiple_imputation"})
        out["mi_sentence"] = missing_sentence(d)
        out["mi"] = d.artifact("fit")["models"][0]
        d.decide({"kind": "set_missing", "strategy": "complete_case"})
        out["cc_sentence"] = missing_sentence(d)
        out["cc"] = d.artifact("fit")["models"][0]
        out["cohort"] = d.artifact("cohort")
        # A single fill answered under prediction, then the purpose changed to inference: the
        # table is held, not served (the validator answered under the old purpose).
        p = drive_to_missing(client, path, "prediction")
        p.decide({"kind": "set_missing", "strategy": "impute"})
        finish(p)
        p.decide({"kind": "set_purpose", "purpose": "inference"})
        # WP17: under inference no estimate is served before the exposure, its effect and the
        # adjustment set are answered; answered, the table shows what the missing values hold.
        from turbotab.core.tests.acceptance.server_drive import answer_plan

        assert p.artifact("fit")["withheld"]
        answer_plan(p, "fiber_g")
        out["flipped"] = p.artifact("fit")["models"][0]
    return out


def test_1_median_fill_and_indicators_are_blocked_and_recorded(inference_runs):
    """Under inference a single fill, a missing indicator and blanks as a level are refused (409)
    with their exits, multiple imputation first; the attestation exit records the answer, its
    sentence carries the limitation, and its table says so. A single fill answered under prediction
    holds the inference table once the purpose becomes inference."""
    refused = inference_runs["refused"]
    assert inference_runs["status"] == {"single": 409, "indicators": 409, "level": 409}
    single = refused["single"]["error"]
    assert single["code"] == "single_fill_under_inference"
    assert [e["decision"]["strategy"] for e in single["exits"]] == [
        "multiple_imputation", "complete_case", "impute"]
    assert single["exits"][-1]["decision"]["acknowledged"] is True
    for key in ("indicators", "level"):
        error = refused[key]["error"]
        assert error["code"] == "indicator_under_inference"
        assert error["exits"][0]["decision"]["strategy"] == "multiple_imputation"
        assert "Groenwold" in error["message"]
    assert "recorded limitation" in inference_runs["recorded_sentence"]
    recorded = inference_runs["recorded"]
    assert recorded["inference"]["missing"]["method"] == "single_fill"
    assert recorded["inference"]["missing"]["recorded"] is True
    assert any("recorded limitation" in c for c in recorded["concerns"])
    flipped = inference_runs["flipped"]["inference"]
    assert flipped["refused"] and "single fill is blocked" in flipped["refused"]
    assert inference_runs["flipped"]["coefficients"] == []
    assert flipped["exits"][0]["decision"]["strategy"] == "multiple_imputation"


def test_1_the_fit_stage_pools_multiple_imputations_with_the_outcome_and_energy(inference_runs):
    """On the server the coefficient table is pooled over m = 20 chained-equation imputations whose
    model holds the outcome and total energy; every analyzed row is kept; each blank was imputed;
    the fiber coefficient's interval covers its true −0.5."""
    table, mi = inference_runs["table"], inference_runs["mi"]
    info = mi["inference"]["missing"]
    # MS2: m ≥ max(20, the percentage of rows with an imputed value), pandas' count of them.
    incomplete = table[["age", "protein_g"]].isna().any(axis=1).mean()
    assert info["method"] == "multiple_imputation"
    assert info["m"] == max(20, int(np.ceil(100 * incomplete)))
    assert info["outcome_in_model"] and "the outcome" in info["variables"]
    assert info["energy"] == "kcal" and "kcal" in info["variables"]
    assert info["imputed"] == {"age": int(table["age"].isna().sum()),
                               "protein_g": int(table["protein_g"].isna().sum())}
    assert mi["inference"]["n_rows"] == len(table) == mi["coefficients_n"]
    assert "Rubin's rules" in mi["inference"]["caption"]
    said = inference_runs["mi_sentence"]
    assert "multiple imputation" in said and "m = 20" in said and "(kcal)" in said
    fiber = next(c for c in mi["coefficients"] if c["feature"] == "fiber_g")
    assert fiber["ci_low"] <= -0.5 <= fiber["ci_high"]
    assert fiber["df"] is not None and 0 < fiber["fmi"] < 1


def test_1_complete_cases_stay_available_with_their_assumption_stated(inference_runs):
    """Complete cases are accepted under inference; the table, and the methods sentence, state the
    assumption (Groenwold et al. 2012: unbiased "only if missingness is conditionally independent
    of the outcome")."""
    cc = inference_runs["cc"]
    info = cc["inference"]["missing"]
    assert info["method"] == "complete_case"
    assert "does not depend on the outcome, given the predictors" in info["assumption"]
    assert "does not depend on the outcome, given the predictors" in inference_runs["cc_sentence"]
    n_complete = int(inference_runs["table"][["age", "protein_g"]].notna().all(axis=1).sum())
    assert cc["coefficients_n"] == n_complete


# ── 2 · prediction: unchanged ────────────────────────────────────────────────


REFERENCE = Path(__file__).with_name("wp7_prediction_reference.json")


def prediction_fixture() -> pd.DataFrame:
    """The NHANES-shaped table (``modeling_fixtures.nhanes_like``, seed 707) with blanks in
    covariates only — bmi, waist, hdl, gender, meds_hbp, age — none in a nutrient, so the
    energy-aware fill (test 3) does not apply and nothing about the single fill may change."""
    from turbotab.core.tests import modeling_fixtures as mf

    frame = mf.nhanes_like(400, seed=707)
    rng = np.random.default_rng(7070)
    frame["meds_hbp"] = frame["meds_hbp"].astype(object)
    for col, share in (("bmi", 0.10), ("waist", 0.08), ("hdl", 0.06), ("gender", 0.04),
                       ("meds_hbp", 0.25), ("age", 0.05)):
        frame.loc[rng.random(len(frame)) < share, col] = np.nan
    return frame


PREDICTION_CONFIGS = {
    "impute": dict(missing=MissingSpec(strategy="impute"), energy_adjustment=None,
                   models=["linear", "elastic_net", "boosted_trees"]),
    "impute_indicators_levels": dict(
        missing=MissingSpec(strategy="impute", indicators=True, categorical="missing_category"),
        energy_adjustment=None, models=["linear", "elastic_net"]),
    "impute_residual": dict(
        missing=MissingSpec(strategy="impute", indicators=True),
        energy_adjustment=EnergyAdjustment(method="residual", energy_column="kcal",
                                           nutrients=["protein", "carb", "fat_total"]),
        models=["linear", "elastic_net"]),
}


def _numbers(a, b, path=()):
    """Every number of ``a`` against ``b`` (the same shape): the largest absolute difference."""
    if isinstance(a, dict):
        assert set(a) == set(b), (path, set(a) ^ set(b))
        return max([_numbers(a[k], b[k], (*path, k)) for k in a] or [0.0])
    if isinstance(a, list):
        assert len(a) == len(b), path
        return max([_numbers(u, v, (*path, i)) for i, (u, v) in enumerate(zip(a, b))] or [0.0])
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        if isinstance(a, float) and np.isnan(a):
            assert b is None or np.isnan(b), path
            return 0.0
        return abs(float(a) - float(b))
    assert a == b, (path, a, b)
    return 0.0


@pytest.fixture(scope="module")
def prediction_run(tmp_path_factory):
    from turbotab.core.stages.modeling import design_stage, fit_stage
    from turbotab.core.tests import modeling_fixtures as mf

    frame = prediction_fixture()
    paths = mf.ingest_frame(frame, tmp_path_factory.mktemp("wp7_prediction"))
    split = mf.split_bundle(np.arange(len(frame)), seed=707)
    ti = mf.target_info("regression")
    out = {}
    for name, slots in PREDICTION_CONFIGS.items():
        st = mf.state(**slots)
        design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
        fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
        out[name] = {m["family"]: {"cv": m["cv"], "coefficients": [
            [r["feature"], r["estimate"]] for r in (m["coefficients"] or [])]}
            for m in fit.data["models"]}
        sealed = fit.frames.get("sealed_scores")
        if sealed is not None:
            out[name]["__sealed__"] = json.loads(sealed.to_json(orient="records"))
    return frame, paths, split, out


def test_2_prediction_results_reproduce_todays_to_1e_9(prediction_run):
    """Every cross-validated score (estimate, fold values, standard error, interval), every
    training-fit coefficient and every sealed held-out score of linear, elastic net and boosted
    trees, under a plain fill, indicators with blanks as a level, and the residual energy method,
    equal the fit stage's output at commit 514336c to 10⁻⁹."""
    reference = json.loads(REFERENCE.read_text())["configs"]
    _, _, _, now = prediction_run
    assert set(now) == set(reference)
    # MS6 added the MSE (the regression primary) to every regression fit: it is new beside the
    # scores the reference holds, which must not move.
    added = {"mse"}
    trimmed = {}
    for name, families in now.items():
        trimmed[name] = {}
        for family, got in families.items():
            if family == "__sealed__":
                trimmed[name][family] = [r for r in got if r["metric"] not in added]
                continue
            assert set(got["cv"]) - set(reference[name][family]["cv"]) == added
            trimmed[name][family] = {**got, "cv": {k: v for k, v in got["cv"].items()
                                                   if k not in added}}
    worst = max(_numbers(reference[name], trimmed[name], (name,)) for name in reference)
    assert worst <= 1e-9, worst


def test_2_the_linear_scores_match_scikit_learn_alone(prediction_run):
    """The linear family's cross-validated R², RMSE and MAE under the plain fill, rebuilt with
    scikit-learn only: median for numbers and most frequent for the category, fit on each training
    fold without the outcome, one-hot with the first level dropped, least squares; pooled over the
    out-of-fold rows against each fold's training mean (``metrics.py``'s definition)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    from turbotab.core.tests import modeling_fixtures as mf

    _, paths, split, now = prediction_run
    stored = pd.read_parquet(paths["data"]).set_index("__row_id")
    predictors = [c for c, r in mf.NHANES_ROLES.items() if r in ("exposure", "covariate", "energy")]
    X = stored[predictors].copy()
    for c in X.columns:  # yes/no answers as 0/1 numbers, a blank as NaN (SimpleImputer's marker)
        if c == "gender":
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
        elif not pd.api.types.is_float_dtype(X[c]):
            X[c] = X[c].map(lambda v: np.nan if pd.isna(v) else float(v)).astype(float)
    y = stored["glucose"].to_numpy(dtype=float)
    a = split.frames["assignment"]
    train = a[a["partition"] == "train"]
    numeric = [c for c in predictors if c != "gender"]
    sse = sst = sae = n = 0.0
    for k in sorted(train["fold"].unique()):
        fit_ids = train.loc[train["fold"] != k, "row_id"].to_numpy()
        score_ids = train.loc[train["fold"] == k, "row_id"].to_numpy()
        model = Pipeline([("prep", ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                  OneHotEncoder(drop="first", handle_unknown="ignore")), ["gender"]),
        ])), ("ols", LinearRegression())])
        model.fit(X.loc[fit_ids], y[fit_ids])
        e = y[score_ids] - model.predict(X.loc[score_ids])
        sse += float((e ** 2).sum())
        sst += float(((y[score_ids] - y[fit_ids].mean()) ** 2).sum())
        sae += float(np.abs(e).sum())
        n += len(score_ids)
    cv = now["impute"]["linear"]["cv"]
    assert cv["r2"]["estimate"] == pytest.approx(1 - sse / sst, abs=1e-9)
    assert cv["rmse"]["estimate"] == pytest.approx(np.sqrt(sse / n), abs=1e-9)
    assert cv["mae"]["estimate"] == pytest.approx(sae / n, abs=1e-9)


# ── 3 · energy-aware fill ───────────────────────────────────────────────────


def protein_table() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """The audit's fixture (``D-skeptic/s11_impute.py``, seed 21): 20,000 rows, protein about 16%
    of energy, 20% of it blank completely at random."""
    rng = np.random.default_rng(21)
    n = 20000
    E = rng.normal(2100, 500, n)
    age = rng.normal(50, 12, n)
    protein = (0.16 + rng.normal(0, .03, n)) * E / 4
    y = 0.03 * protein + 0.002 * E + 0.1 * age + rng.normal(0, 5, n)
    blank = rng.random(n) < 0.2
    frame = pd.DataFrame({"protein_g": np.where(blank, np.nan, protein), "energy_kcal": E, "age": age})
    return frame, y, blank, protein


PROTEIN_ROLES = {"protein_g": "exposure", "energy_kcal": "energy", "age": "covariate"}
RESIDUAL = EnergyAdjustment(method="residual", energy_column="energy_kcal", nutrients=["protein_g"])


def test_3_imputed_rows_keep_the_nutrient_energy_relation():
    """With 20% of protein blank and the residual method, imputed rows' energy-adjusted protein
    correlates with energy as observed rows' does, not −1.

    * Under inference (multiple imputation, the analyses the table pools): imputed rows' protein
      correlates with energy within 0.02 of the observed rows' (+0.78), and their energy-adjusted
      protein within 0.02 of the observed rows' (0), averaged over the 20 imputations.
    * Under prediction a deployable fill is deterministic, so no correlation can match a spread: the
      fill is each imputed row's point on the observed rows' least-squares line of protein on energy
      (NumPy's ``polyfit``, to 10⁻⁹), so its energy-adjusted value carries no energy at all: it is the
      observed rows' mean adjusted value, at a spread 10⁻⁹ of theirs, where a median fill's correlates
      −1.000 with energy (the audit's reference, recomputed here with NumPy)."""
    frame, y, blank, protein = protein_table()
    E = frame["energy_kcal"].to_numpy()
    r_observed = np.corrcoef(protein[~blank], E[~blank])[0, 1]
    assert r_observed == pytest.approx(0.78, abs=0.01)
    median = np.where(blank, np.nanmedian(frame["protein_g"]), protein)
    slope, intercept = np.polyfit(E, median, 1)
    assert np.corrcoef((median - intercept - slope * E)[blank], E[blank])[0, 1] < -0.9999

    st = ProjectState(target="y", roles=PROTEIN_ROLES, purpose="inference", energy_adjustment=RESIDUAL,
                      missing=MissingSpec(strategy="multiple_imputation"))
    spec = design_spec(st, frame, model_predictors(st))
    imputations = impute_for_inference(spec, frame[spec.inputs], y, "regression", seed=0)
    raw, adjusted, observed = [], [], []
    for f in imputations.frames:
        M = transformer(shared_steps(spec)).fit_transform(f[spec.inputs])
        raw.append(np.corrcoef(f.loc[blank, "protein_g"], E[blank])[0, 1])
        adjusted.append(np.corrcoef(M.loc[blank, "protein_g_adj"], E[blank])[0, 1])
        observed.append(np.corrcoef(M.loc[~blank, "protein_g_adj"], E[~blank])[0, 1])
    assert np.mean(raw) == pytest.approx(r_observed, abs=0.02)
    assert np.mean(adjusted) == pytest.approx(np.mean(observed), abs=0.02)

    st = st.model_copy(update={"purpose": "prediction", "missing": MissingSpec(strategy="impute")})
    spec = design_spec(st, frame, model_predictors(st))
    assert spec.energy_fill == {"energy": "energy_kcal", "nutrients": ["protein_g"]}
    steps = transformer(shared_steps(spec)).fit(frame[spec.inputs])
    filled = steps.named_steps["impute"].transform(frame[spec.inputs])
    slope, intercept = np.polyfit(E[~blank], protein[~blank], 1)
    np.testing.assert_allclose(filled.loc[blank, "protein_g"], intercept + slope * E[blank],
                               rtol=1e-9)
    M = steps.transform(frame[spec.inputs])
    imputed, kept = M.loc[blank, "protein_g_adj"], M.loc[~blank, "protein_g_adj"]
    assert imputed.std() < 1e-9 * kept.std()
    assert imputed.mean() == pytest.approx(kept.mean(), rel=1e-9)
    assert abs(np.corrcoef(kept, E[~blank])[0, 1]) < 1e-9


def test_3_the_median_stays_for_columns_energy_does_not_carry():
    """Only energy-bearing nutrients are filled from energy: with no energy column, or for a
    covariate, the fill is SimpleImputer's median exactly."""
    from sklearn.impute import SimpleImputer

    frame, _, _, _ = protein_table()
    frame = frame.assign(age=np.where(np.arange(len(frame)) % 7 == 0, np.nan, frame["age"]))
    st = ProjectState(target="y", roles=PROTEIN_ROLES, purpose="prediction", energy_adjustment=RESIDUAL,
                      missing=MissingSpec(strategy="impute"))
    spec = design_spec(st, frame, model_predictors(st))
    step = transformer(shared_steps(spec)).fit(frame[spec.inputs]).named_steps["impute"]
    out = step.transform(frame[spec.inputs])
    median = SimpleImputer(strategy="median").fit(frame[["age"]]).statistics_[0]
    assert (out.loc[frame["age"].isna(), "age"] == median).all()


# ── 4 · the methods sentence names the method ───────────────────────────────


def _said(decision: SetMissing, state: ProjectState) -> str:
    from turbotab.core.voice import sentence_for

    return sentence_for(decision, state).replace("`", "")


def test_4_the_methods_sentence_names_the_method():
    """STROBE-nut nut-13 asks for "any method used to handle missing values": the sentence names the
    single fill's rules (and the energy-aware nutrient fill when it applies), multiple imputation
    with m and what its imputation model held, complete cases' assumption under inference, and how
    values below detection were filled."""
    roles = {"fiber_g": "exposure", "protein_g": "exposure", "kcal": "energy", "age": "covariate"}
    prediction = ProjectState(target="ldl", roles=roles, purpose="prediction")
    inference = prediction.model_copy(update={"purpose": "inference"})
    single = _said(SetMissing(strategy="impute"), prediction)
    assert "median for numbers" in single and "most frequent value for categories" in single
    assert "without the outcome" in single and "line on total energy (kcal)" in single
    mi = _said(SetMissing(strategy="multiple_imputation"), inference)
    assert "multiple imputation" in mi and "m = 20" in mi
    assert "with the outcome and total energy (kcal) in the imputation model" in mi
    assert "Rubin's rules" in mi
    cc = _said(SetMissing(strategy="complete_case"), inference)
    assert "complete-case analysis" in cc and "does not depend on the outcome" in cc
    lod = _said(SetMissing(strategy="impute", below_detection="half_minimum",
                           censored_columns=["mz_0121"]), prediction)
    assert "below the detection limit in mz_0121 were set to half the column's smallest" in lod


# ── 5 · one rule ─────────────────────────────────────────────────────────────


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("> ", " ")).strip()


def test_5_roadmap_contract_and_drawer_state_one_purpose_conditional_rule():
    """ROADMAP §07, M2_CONTRACT §4 (its Tier A test) and the MISSING drawer quote one rule, by
    purpose, citing Sisk et al. 2023 and Moons et al. 2006 via Harrell; the purpose-blind sentences
    the audit quoted ("never place the outcome in the imputation model, which is a blocker in any
    configuration"; "The outcome belongs in the imputation model") are gone."""
    from turbotab.core.teaching.content import MISSING

    assert "Sisk et al. 2023" in RULE and "Moons et al. 2006, via Harrell" in RULE
    assert "Under inference" in RULE and "Under prediction" in RULE
    roadmap = _flat((REPO / "docs/turbotab/ROADMAP.md").read_text())
    section = roadmap[roadmap.index("### 07 · Missingness"):roadmap.index("### 08 ·")]
    contract = _flat((REPO / "docs/turbotab-next/M2_CONTRACT.md").read_text())
    drawer = " ".join(s["body"] for s in MISSING["drawer"]["sections"])
    for text in (section, contract, drawer):
        assert _flat(RULE) in text
    assert "never place the outcome in the imputation model" not in roadmap
    assert "the outcome is never in the imputation model" not in contract
    assert all(s["heading"] != "The outcome belongs in the imputation model"
               for s in MISSING["drawer"]["sections"])
    # The Tier A tests the contract names exist: prediction's (outcome-free, in-fold) and this one.
    tier_a = (REPO / "turbotab/core/tests/test_repairs.py").read_text()
    assert "def test_in_fold_imputation_never_sees_held_out_rows_or_the_outcome" in tier_a
    assert "def test_the_outcome_cannot_move_the_imputer" in tier_a
    assert "acceptance/test_wp7_missing_data.py" in contract


# ── 6 · values below detection ──────────────────────────────────────────────


SAMPLES = REPO / "turbotab" / "sample_data"


@pytest.fixture(scope="module")
def untargeted():
    """``metabolomics_untargeted.csv`` and its findings under the metabolomics lens, spoken as the
    findings stage speaks them (``stages.findings.speak_for``)."""
    from turbotab import packs
    from turbotab.core.stages.findings import speak_for

    frame = pd.read_csv(SAMPLES / "metabolomics_untargeted.csv")
    spoken = speak_for(frame, ["metabolomics"], "responder", [], packs.findings(frame, ["metabolomics"]))
    return frame, [f for _, f in spoken]


def test_6_median_fill_of_non_detections_is_refused_unless_given_a_reason(untargeted):
    """The left-censoring finding names every censored column (not the eight its card shows); a
    fill that treats their blanks as any other blank is refused with half-minimum and a
    censoring-aware fill as exits, under either purpose; complete cases are accepted. Under
    prediction a reason keeps the fill (ranked lower); under inference it does not (MODELING_SEQUENCE
    §4, MS7: "MAR imputation of detection-limit blanks … refuse, censoring-aware instead"), and the
    refusal offers no "give your reason" exit."""
    frame, findings = untargeted
    found = next(f for f in findings if f["id"] == "pack::metabolomics::left_censored")
    assert len(found["censored_columns"]) > len(found["affected_columns"]) == 8
    mz = [c for c in frame.columns if c.startswith("mz_")]
    roles = {c: "exposure" for c in mz}
    for purpose in ("prediction", "inference"):
        state = ProjectState(target="responder", roles=roles, purpose=purpose)
        ctx = {"state": state, "columns": list(frame.columns),
               "artifact": lambda name: {"findings": findings} if name == "findings" else None}
        strategy = "impute" if purpose == "prediction" else "multiple_imputation"
        with pytest.raises(Refusal) as refused:
            validate(SetMissing(strategy=strategy), ctx)
        assert refused.value.code == "median_below_detection"
        offered = [e["decision"]["below_detection"] for e in refused.value.exits if e["decision"]]
        assert offered == ["censoring_aware", "half_minimum"]
        assert refused.value.exits[0]["decision"]["censored_columns"] == [
            c for c in found["censored_columns"] if c in roles]
        if purpose == "prediction":
            assert refused.value.exits[-1]["decision"] is None  # keep it: the answer must say why
            validate(SetMissing(strategy=strategy, reason="the blanks are failed injections"), ctx)
        else:
            assert all(e["decision"] is not None for e in refused.value.exits)
            with pytest.raises(Refusal):
                validate(SetMissing(strategy=strategy, reason="the blanks are failed injections"), ctx)
        validate(SetMissing(strategy="complete_case"), ctx)
        validate(refused.value.exits[0]["decision"], ctx)


def censored_table(rng: np.random.Generator, censored: float = 0.6, n: int = 300) -> pd.DataFrame:
    """The audit's log-scale simulation (``F/sim_lod3.py``): log-intensity L ~ N(10, 1), y = 0.5(L −
    10) + e, and L blank below its ``censored`` quantile."""
    L = rng.normal(10, 1, n)
    y = 0.5 * (L - 10) + rng.normal(0, 1, n)
    lod = np.quantile(L, censored)
    return pd.DataFrame({"m": np.where(L >= lod, L, np.nan), "y": y})


def _lod_estimate(frame: pd.DataFrame, purpose: str, strategy: str, method: str | None,
                  seed: int) -> dict:
    st = ProjectState(target="y", task="regression", purpose=purpose, roles={"m": "exposure"},
                      missing=MissingSpec(strategy=strategy, below_detection=method,
                                          censored_columns=["m"], reason=None if method else "x"))
    X, y = frame[["m"]], frame["y"].to_numpy()
    spec = design_spec(st, X, ["m"])
    pipeline = build_pipeline(spec, LINEAR, "regression", purpose, len(X), 1)
    if strategy == "multiple_imputation":
        imputations = impute_for_inference(spec, X, y, "regression", seed=seed)
        _, rows, _, _ = pooled_table(
            LINEAR, pipeline, imputations, y, task="regression", clusters=INDEPENDENT,
            outcome=Outcome(name="y"), survey=None, design=None, spec=spec, rows="all",
            fit=lambda model, X_k: fit_pipeline(model, X_k, y))
        return next(r for r in rows if r["feature"] == "m")
    fitted = fit_pipeline(pipeline, X, y)
    return next(r for r in LINEAR.coefficients(fitted, X, y, task="regression", purpose=purpose)
                if r["feature"] == "m")


def test_6_the_censoring_aware_estimate_stays_within_5_percent_at_60_percent_censored():
    """200 datasets of 300 rows, 60% of the log-intensity below its detection limit. The
    censoring-aware fill, under inference (a censored-normal draw within the multiple imputation,
    given the outcome) and under prediction (in-fold, outcome-free: each non-detect at its expected
    value below the limit), averages within 5% of 0.5 (0.475–0.525); Monte Carlo error ±0.005.
    Detected-only (complete cases) is the audit's reference (0.497). Half the minimum and the median
    are reported beside them: both far off (here 0.14 and 0.67)."""
    rng = np.random.default_rng(11)
    found = {k: [] for k in ("inference", "cover", "prediction", "half", "median", "detected")}
    for rep in range(200):
        frame = censored_table(rng)
        row = _lod_estimate(frame, "inference", "multiple_imputation", "censoring_aware", rep)
        found["inference"].append(row["estimate"])
        found["cover"].append(row["ci_low"] <= BETA <= row["ci_high"])
        found["prediction"].append(
            _lod_estimate(frame, "prediction", "impute", "censoring_aware", rep)["estimate"])
        found["half"].append(_lod_estimate(frame, "prediction", "impute", "half_minimum", rep)["estimate"])
        found["median"].append(_lod_estimate(frame, "prediction", "impute", None, rep)["estimate"])
        detected = frame.dropna()
        found["detected"].append(np.polyfit(detected["m"], detected["y"], 1)[0])
    means = {k: float(np.mean(v)) for k, v in found.items()}
    assert abs(means["inference"] - BETA) <= 0.05 * BETA, means
    assert abs(means["prediction"] - BETA) <= 0.05 * BETA, means
    assert means["cover"] >= 0.92, means
    assert abs(means["detected"] - BETA) <= 0.05 * BETA, means
    assert means["half"] < 0.3 and means["median"] > 0.6, means


def test_6_zeros_become_non_detections_only_when_the_user_says_so(tmp_path):
    """MZmine writes zeros for non-detections; the zeros finding offers that reading as a repair and
    assumes nothing. Until it is applied the zeros stay zeros and no column is read as censored;
    applied, every zero in the intensity block becomes blank and those columns are censored."""
    from turbotab import packs
    from turbotab.core.datastore import ingest
    from turbotab.core.decisions import FindingDisposition
    from turbotab.core.methods.missing import censored_columns
    from turbotab.core.repairs import OfferContext, column_expressions, evaluate, offer
    from turbotab.core.stages.findings import speak_for

    path = SAMPLES / "metabolomics_mzmine_zeros.csv"
    frame = pd.read_csv(path)
    raw = [f for f in packs.findings(frame, ["metabolomics"])
           if f["id"] == "pack::metabolomics::zeros_or_missing"]
    assert raw, "the zeros finding fires on the MZmine export"
    spoken = speak_for(frame, ["metabolomics"], "responder", [], raw)
    finding = next(f for r, f in spoken if f["id"] == "pack::metabolomics::zeros_or_missing")
    options = offer(finding, raw[0], OfferContext(frame=frame, target="responder"))
    assert [o.key for o in options] == ["nondetect"]
    option = options[0]
    zeros = sorted(c for c in frame.columns if c.startswith("mz_") and (frame[c] == 0).any())
    assert sorted(option.decision.params["columns"]) == zeros
    before = ProjectState(target="responder")
    assert censored_columns(None, before) == []
    assert column_expressions(None, before) == {}
    applied = before.model_copy(update={"findings": {finding["id"]: FindingDisposition(
        action="applied", option="nondetect", params=option.decision.params)}})
    assert sorted(censored_columns(None, applied)) == zeros
    parquet = tmp_path / "raw.parquet"
    ingest(path, parquet)
    values = evaluate(parquet, zeros, column_expressions(None, applied))
    assert not (values[zeros] == 0).any().any()
    assert int(values[zeros].isna().sum().sum()) == int((frame[zeros] == 0).sum().sum() + frame[zeros].isna().sum().sum())


def test_6_counts_an_assay_lens_reads_as_counts_offer_no_treat_as_missing():
    """Audit F14: a sentinel finding reframed by the genomics lens says its values are counts, so
    it offers no "Treat as missing" repair."""
    from turbotab.core.repairs import OfferContext, offer

    frame = pd.DataFrame({"gene_0017": [7, 7, 3, 0, 12, 7, 7]})
    finding = {"id": "sentinel_missing__gene_0017", "lens": "genomics", "affected_columns": ["gene_0017"]}
    raw = {"params": {"column": "gene_0017", "values": [7]}}
    assert offer(finding, raw, OfferContext(frame=frame)) == []
    assert offer({**finding, "lens": None}, raw, OfferContext(frame=frame))


# ── 7 · what complete cases cost ─────────────────────────────────────────────


def test_7_more_than_10_percent_lost_to_complete_cases_raises_a_concern(inference_runs, tmp_path_factory):
    """Under inference, complete cases dropping 53% of the rows raise a concern naming the share,
    the pack's 10% threshold and the kept-against-dropped comparison (the outcome first); its
    numbers equal NumPy's on the same rows. At 5% lost, or under prediction, there is no concern."""
    table = inference_runs["table"]
    cc = inference_runs["cc"]
    concern = next((c for c in cc["concerns"] if c.startswith("Complete cases drop")), None)
    complete = table[["age", "protein_g"]].notna().all(axis=1)
    assert concern is not None, cc["concerns"]
    assert f"{int((~complete).sum()):,} of {len(table):,} rows" in concern and "10%" in concern
    loss = inference_runs["cohort"]["complete_case_loss"]
    kept, dropped = table.loc[complete, "ldl"], table.loc[~complete, "ldl"]
    smd = (dropped.mean() - kept.mean()) / np.sqrt((kept.var() + dropped.var()) / 2)
    assert loss["outcome"]["column"] == "ldl"
    assert loss["outcome"]["kept"] == pytest.approx(kept.mean(), rel=1e-9)
    assert loss["outcome"]["dropped"] == pytest.approx(dropped.mean(), rel=1e-9)
    assert loss["outcome"]["smd"] == pytest.approx(smd, rel=1e-9)
    assert "`ldl` mean" in concern

    small = diet_table(seed=8, age_blank=False, protein_share=0.05)
    folder = tmp_path_factory.mktemp("wp7_more")
    small_path, big_path = folder / "diet_small.csv", folder / "diet_missing.csv"
    small.to_csv(small_path, index=False)
    table.to_csv(big_path, index=False)
    with local_server(tmp_path_factory.mktemp("wp7_more_home")) as client:
        for path, purpose in ((small_path, "inference"), (big_path, "prediction")):
            d = drive_to_missing(client, path, purpose)
            d.decide({"kind": "set_missing", "strategy": "complete_case"})
            model = finish(d)
            assert not any(c.startswith("Complete cases drop") for c in model["concerns"]), purpose
            if purpose == "inference":
                assert model["inference"]["missing"]["n_dropped"] == int(small["protein_g"].isna().sum())


def test_7_the_threshold_is_the_packs_10_percent():
    """``row_loss_concern`` speaks above 10% and not at or below it; ``row_loss``'s standardized
    difference is the pooled-SD one written out with NumPy."""
    rng = np.random.default_rng(3)
    kept = pd.DataFrame({"y": rng.normal(0, 1, 900), "a": rng.normal(0, 1, 900)})
    dropped = pd.DataFrame({"y": rng.normal(0.5, 1, 100), "a": rng.normal(1, 1, 100)})
    loss = row_loss(kept, dropped, "y", ["a"])
    assert loss["share"] == 0.1 and row_loss_concern(loss) is None
    a_k, a_d = kept["a"], dropped["a"]
    assert loss["columns"][0]["smd"] == pytest.approx(
        (a_d.mean() - a_k.mean()) / np.sqrt((a_k.var() + a_d.var()) / 2), rel=1e-12)
    loss = row_loss(kept.iloc[:850], dropped, "y", ["a"])
    assert loss["share"] > 0.1 and row_loss_concern(loss).startswith("Complete cases drop 100 of 950")
