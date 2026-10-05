"""MS8 · Scales: scoring, reliability (ω; α labeled customary) and the correction for measurement
error (docs/turbotab-next/MODELING_SEQUENCE.md §0 ruling 8, §2, §4, §5 MS8, §6 chain 4).

    (1) Scale scoring (sum or mean, reverse-coded items), with ω-total for a unidimensional scale and
        ω-hierarchical for a multidimensional one, agreeing with R psych::omega to 1e-3; α shown only
        labeled "customary", agreeing with psych::alpha to 1e-6.
    (2) Disattenuation by α or ω refused for formative indices, with exits: a test–retest ICC from a
        repeat administration, or a calibration substudy.
    (3) For a reflective scale, the corrected coefficient by regression calibration conditional on
        the model's covariates (not β/ω), labeled as omitting transient error, with bootstrap CIs
        that re-estimate the reliability in each replicate; reported beside the uncorrected estimate
        as a declared secondary analysis; agreeing with an independent NumPy implementation and R's
        mecor.
    (4) Under inference with missing items, item-level multiple imputation before scoring.
    (5) Chain 4 end to end, with its methods sentence.

**Sources** (quoted in ``turbotab/core/methods/scales.py`` and the review record
``docs/turbotab-next/audit/modeling-sequence-review.json``): McNeish 2018 (α vs ω); Keogh, Shaw &
Gustafson 2020, STRATOS §3.1.2 (the attenuation factor conditional on the covariates); Boe et al.
2023 (every outcome-model confounder in the calibration equation); Schmidt, Le & Ilies 2003
(transient error); Reedy et al. 2018 (the HEI-2015's multidimensionality); Eekhout et al. 2014
(item-level MI).

**References, each independent of the code under test:** R's ``psych::omega``, ``psych::alpha`` and
``psych::ICC`` and ``mecor::mecor`` (run as a subprocess on CSVs written here; the tests that need R
skip without it), pandas for keyed items and scores, NumPy by hand for the calibration (the Schur
complement of the covariance matrix), statsmodels for least squares and the proportional-odds
model, and simulations with a known truth (continuous congeneric items, where the true score and
its coefficient are known exactly).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core import decisions as d
from turbotab.core.methods import scales as S
from turbotab.core.tests.acceptance.scales_fixtures import (DQ_ITEMS, DQ_RETEST, PSS_ITEMS,
                                                            PSS_REVERSE, calibrate_by_hand, keyed,
                                                            linear_scale_table, needs_r,
                                                            omega_script, run_r,
                                                            survey_scale_table)
from turbotab.core.tests.acceptance.server_drive import (Truth, answer_wp17, local_server,
                                                         open_project)

SAT = [f"sat_{j}" for j in range(1, 9)]
SAT_REVERSE = ["sat_3", "sat_6"]
REVIEWERS = ("Reliability was estimated as the test–retest ICC from a repeat administration (or a "
             "calibration substudy); the corrected coefficient was obtained by regression "
             "calibration including all model covariates, with bootstrap CIs re-estimating the "
             "reliability; uncorrected and corrected estimates are both reported.")


def congeneric(rng: np.random.Generator, n: int, loadings, uniques, z_effect=0.0):
    """Continuous congeneric items x_j = λ_j F + e_j (no rounding), F correlated with a covariate
    z; the true score is T = (Σλ) F and the summed score W = T + Σe, classical error of variance
    Σψ_j. Returns (items, z, F)."""
    z = rng.standard_normal(n)
    F = z_effect * z + math.sqrt(1 - z_effect ** 2) * rng.standard_normal(n)
    X = np.column_stack([lam * F + math.sqrt(psi) * rng.standard_normal(n)
                         for lam, psi in zip(loadings, uniques)])
    return X, z, F


def bifactor_items(rng: np.random.Generator, n: int, groups, general: float, group: float):
    """Items with a general factor and ``len(groups)`` group factors (a multidimensional scale)."""
    G = rng.standard_normal(n)
    cols = []
    for size in groups:
        Sg = rng.standard_normal(n)
        for _ in range(size):
            cols.append(general * G + group * Sg
                        + math.sqrt(1 - general ** 2 - group ** 2) * rng.standard_normal(n))
    return np.column_stack(cols)


# ── (1) scoring, ω and α ─────────────────────────────────────────────────────


def test_1_scores_turn_reversed_items_over_on_the_response_scale_then_sum_or_average():
    """The score step reads only each row's own answers: a reverse-coded item becomes low + high − x
    on the instrument's response scale, then the items are summed (or averaged); a row with a blank
    item has no score until the item is filled. Reference: pandas by hand."""
    frame = linear_scale_table(n=60, missing=0.1)
    for scoring in ("sum", "mean"):
        step = S.ScaleScorer([{"name": "sat_score", "items": SAT, "reverse": SAT_REVERSE,
                               "low": 1, "high": 5, "scoring": scoring}])
        X = frame[["age", *SAT, "bmi"]]
        out = step.fit(X).transform(X)
        by_hand = keyed(frame, SAT, SAT_REVERSE, 1, 5)
        expected = by_hand.sum(axis=1, skipna=False)
        if scoring == "mean":
            expected = expected / len(SAT)
        assert list(out.columns) == ["age", "sat_score", "bmi"]
        np.testing.assert_allclose(out["sat_score"].to_numpy(), expected.to_numpy(), rtol=0,
                                   atol=1e-12, equal_nan=True)
        assert out["sat_score"].isna().sum() == frame[SAT].isna().any(axis=1).sum() > 0
    (entry,) = [e for e in step.lineage() if e["output"] == "sat_score"]
    assert entry["inputs"] == SAT
    assert entry["formula"].startswith("sat_score = (sat_1 + sat_2 + (1 + 5 − sat_3) + ")


@needs_r
@pytest.mark.parametrize("seed", [1, 2])
def test_1_omega_total_of_a_unidimensional_scale_agrees_with_psych(seed, tmp_path):
    """One factor (Likert answers, items 3 and 6 reverse-coded): ω-total on the standardized items
    is psych::omega(nfactors = 1)'s omega.tot, the score's own ω-total is 1 − Σ s²u² / ΣC computed
    in R from psych's Schmid–Leiman table, both to 1e-3 (observed: about 1e-6), and α is
    psych::alpha's raw_alpha to 1e-6."""
    frame = linear_scale_table(seed=seed, n=900)
    items = keyed(frame, SAT, SAT_REVERSE, 1, 5)
    items.to_csv(tmp_path / "items.csv", index=False)
    ref = run_r(omega_script("items.csv", 1), tmp_path)
    got = S.omega(items.to_numpy(), 1)
    print(f"\nω_t {got.omega_total_standardized:.6f} vs psych {ref['omega_tot']:.6f}; "
          f"score ω_t {got.omega_total:.6f} vs {ref['score_omega_tot']:.6f}; "
          f"α {got.alpha:.8f} vs {ref['alpha']:.8f}")
    assert abs(got.omega_total_standardized - ref["omega_tot"]) < 1e-3
    assert abs(got.omega_total - ref["score_omega_tot"]) < 1e-3
    assert abs(got.alpha - ref["alpha"]) < 1e-6
    assert got.coefficient("unidimensional") == ("omega_total", got.omega_total)


@needs_r
@pytest.mark.parametrize("groups,nf", [([4, 4, 4], 3), ([3, 4, 3, 4], 4), ([4, 4], 2)])
def test_1_omega_hierarchical_of_a_multidimensional_scale_agrees_with_psych(groups, nf, tmp_path):
    """A general factor and ``nf`` group factors: ω-hierarchical (minres, varimax, oblimin by
    gradient projection, Schmid–Leiman) is psych::omega(nfactors = nf)'s omega_h and the score's own
    ω_h is (Σ s g)² / ΣC from psych's general loadings, to 1e-3; every general loading agrees with
    psych's to 1e-4. With two group factors the general loadings are set equal (psych's default)
    and the result says so."""
    rng = np.random.default_rng(10 + nf)
    X = bifactor_items(rng, 1200, groups, 0.55, 0.45) * rng.uniform(0.8, 1.3, sum(groups))
    pd.DataFrame(X, columns=[f"i{j}" for j in range(X.shape[1])]).to_csv(tmp_path / "items.csv",
                                                                           index=False)
    ref = run_r(omega_script("items.csv", nf), tmp_path)
    got = S.omega(X, nf)
    print(f"\nω_h {got.omega_h_standardized:.6f} vs psych {ref['omega_h']:.6f}; score ω_h "
          f"{got.omega_h:.6f} vs {ref['score_omega_h']:.6f}")
    assert abs(got.omega_h_standardized - ref["omega_h"]) < 1e-3
    assert abs(got.omega_h - ref["score_omega_h"]) < 1e-3
    assert abs(got.omega_total_standardized - ref["omega_tot"]) < 1e-3
    assert np.max(np.abs(got.general - np.asarray(ref["g"]))) < 1e-4
    assert got.coefficient("multidimensional") == ("omega_hierarchical", got.omega_h)
    assert (got.caution is not None) == (nf == 2)


def test_1_omega_refuses_what_it_cannot_identify():
    rng = np.random.default_rng(0)
    with pytest.raises(S.ScaleRefused, match="at least 3 items"):
        S.omega(rng.standard_normal((100, 2)), 1)
    with pytest.raises(S.ScaleRefused, match="at least 9 items"):
        S.omega(rng.standard_normal((100, 8)), 3)


# ── (2) formative indices: disattenuation by α or ω refused ──────────────────


def _state(purpose: str = "inference", **extra) -> d.ProjectState:
    roles = {c: "covariate" for c in [*PSS_ITEMS, *DQ_ITEMS, "age"]}
    return d.ProjectState(purpose=purpose, target="wellbeing", roles=roles, **extra)


def _scales(dq: dict | None = None, pss: dict | None = None) -> dict:
    return {"kind": "set_scales", "scales": [
        {"name": "pss_score", "items": PSS_ITEMS, "reverse": PSS_REVERSE, "low": 0, "high": 4,
         "kind": "reflective", **(pss or {})},
        {"name": "dq_score", "items": DQ_ITEMS, "low": 0, "high": 10, "kind": "formative",
         **(dq or {})}]}


def test_2_disattenuating_a_formative_index_by_its_internal_consistency_is_refused_with_exits():
    """MODELING_SEQUENCE §4: "Disattenuation by α/ω of a formative index | — | refuse; test–retest
    ICC or a calibration substudy". The refusal names why (Reedy et al. 2018's multidimensional
    HEI-2015) and carries three exits: the test–retest ICC of the repeat administration it finds in
    the table (``dq_1_t2`` …), a calibration substudy, and the uncorrected estimate."""
    ctx = {"state": _state(), "columns": [*PSS_ITEMS, *DQ_ITEMS, *DQ_RETEST, "age", "wellbeing"]}
    with pytest.raises(d.Refusal) as refused:
        d.validate(_scales(dq={"correction": "regression_calibration"}), ctx)
    error = refused.value
    assert error.code == "formative_disattenuation"
    assert "formative index" in error.message and "at least four dimensions" in error.message
    retest, substudy, uncorrected = error.exits
    dq = retest["decision"]["scales"][1]
    assert (dq["reliability"], dq["retest"]) == ("test_retest", DQ_RETEST)
    assert substudy["decision"] is None and "calibration substudy" in substudy["label"]
    assert uncorrected["decision"]["scales"][1]["correction"] == "none"
    # Each exit with a decision passes the leash; the substudy exit, once its reference is named.
    d.validate(retest["decision"], ctx)
    d.validate(uncorrected["decision"], ctx)
    d.validate(_scales(dq={"correction": "regression_calibration",
                           "reliability": "calibration_substudy", "reference": "dq_1_t2"}), ctx)
    # A reflective scale may be corrected from ω (labeled as omitting transient error, below).
    d.validate(_scales(pss={"correction": "regression_calibration"}), ctx)


def test_2_under_prediction_every_correction_is_refused_with_the_uncorrected_exit():
    """Under prediction the deployed model sees the same error-prone score (STRATOS §1), so any
    correction is refused, the exit keeping the scores uncorrected."""
    ctx = {"state": _state("prediction")}
    with pytest.raises(d.Refusal) as refused:
        d.validate(_scales(pss={"correction": "regression_calibration"}), ctx)
    assert refused.value.code == "not_for_prediction"
    exit_ = refused.value.exits[0]["decision"]
    assert all(s["correction"] == "none" for s in exit_["scales"])
    d.validate(exit_, ctx)


# ── (3) the correction: regression calibration given the covariates ──────────


def _lin(seed: int = 11, n: int = 1200) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    frame = linear_scale_table(seed=seed, n=n)
    items = keyed(frame, SAT, SAT_REVERSE, 1, 5)
    W = items.sum(axis=1).to_numpy()
    Z = frame[["age", "bmi"]].to_numpy(dtype=float)
    return frame, items.to_numpy(), W, Z


@needs_r
def test_3_the_correction_is_numpys_and_mecors_calibration_given_the_covariates(tmp_path):
    """The score's error variance from ω (σ²_U = (1 − ω)·var W, ω the score's own, computed in R
    from psych::omega's solution); then (a) NumPy by hand — λ from the Schur complement of the
    (n − 1) covariance matrix of (W, age, bmi), the calibrated score, statsmodels OLS — and (b) R's
    mecor with ``MeasErrorRandom(W, variance = σ²_U)`` give the corrected coefficient the app does,
    to 1e-6 relative (the two ω's agree to about 1e-6); for least squares it is exactly the
    uncorrected coefficient over λ."""
    frame, items, W, Z = _lin()
    y = frame["sbp"].to_numpy()
    pd.DataFrame(items, columns=SAT).to_csv(tmp_path / "items.csv", index=False)
    pd.DataFrame({"sbp": y, "W": W, "age": Z[:, 0], "bmi": Z[:, 1]}).to_csv(tmp_path / "lin.csv",
                                                                            index=False)
    extra = """
suppressMessages(library(mecor))
dd <- read.csv("lin.csv")
s2u <- (1 - out$score_omega_tot) * var(dd$W)
m <- mecor(sbp ~ MeasErrorRandom(substitute = W, variance = s2u) + age + bmi, data = dd,
           method = "standard")
out$mecor <- unname(m$corfit$coef["cor_W"]); out$naive <- unname(m$uncorfit$coef["W"]); out$s2u <- s2u
"""
    ref = run_r(omega_script("items.csv", 1, extra), tmp_path)
    source = S.Source("internal_consistency",
                      reliability=S.internal_consistency(items, "sum", "unidimensional", 1))
    got = S.correct(W, Z, y, 0, source, n_boot=0)
    X_hat, lam = calibrate_by_hand(W, Z, ref["s2u"])
    by_hand = sm.OLS(y, sm.add_constant(np.column_stack([X_hat, Z]))).fit().params[1]
    print(f"\ncorrected {got.estimate:.6f} | by hand {by_hand:.6f} | mecor {ref['mecor']:.6f} | "
          f"naive {got.naive:.6f} | λ {got.attenuation:.4f}")
    assert got.naive == pytest.approx(ref["naive"], rel=1e-10)
    assert got.estimate == pytest.approx(by_hand, rel=1e-6)
    assert got.estimate == pytest.approx(ref["mecor"], rel=1e-6)
    assert got.attenuation == pytest.approx(lam, rel=1e-6)
    assert got.estimate == pytest.approx(got.naive / got.attenuation, rel=1e-12)
    assert got.reliability == pytest.approx(ref["score_omega_tot"], abs=1e-5)


def test_3_conditional_calibration_is_unbiased_where_dividing_by_omega_under_corrects():
    """STRATOS §3.1.2: with covariates the slope is attenuated by the reliability given them. A
    simulation with a known truth: continuous congeneric items (Σλ = 4, Σψ = 4.2, so ω ≈ 0.79), a
    true score correlated 0.6 with the covariate, the outcome 1.5 per unit of the true score. Over
    40 datasets of 4,000, the calibrated coefficient averages within 1% of 1.5, while β/ω (the
    Spearman correction) stays more than 10% short."""
    rng = np.random.default_rng(31)
    lam, psi = [0.9, 0.8, 0.7, 0.8, 0.8], [0.7, 0.9, 1.0, 0.8, 0.8]
    total = sum(lam)
    calibrated, spearman = [], []
    for _ in range(40):
        X, z, F = congeneric(rng, 4000, lam, psi, z_effect=0.6)
        y = 1.5 * total * F + 2.0 * z + rng.normal(0, 3, len(z))
        W = X.sum(axis=1)
        source = S.Source("internal_consistency",
                          reliability=S.internal_consistency(X, "sum", "unidimensional", 1))
        got = S.correct(W, z[:, None], y, 0, source, n_boot=0)
        calibrated.append(got.estimate)
        spearman.append(got.naive / got.reliability)
    print(f"\ncalibrated {np.mean(calibrated):.4f} | β/ω {np.mean(spearman):.4f} | truth 1.5")
    assert abs(np.mean(calibrated) / 1.5 - 1) < 0.01
    assert np.mean(spearman) / 1.5 < 0.9


@needs_r
def test_3_each_bootstrap_replicate_re_estimates_omega_as_psych_does(tmp_path):
    """The interval's replicates, drawn as documented (``numpy.random.default_rng(seed)``, one
    ``integers(0, n, n)`` per replicate), re-estimate ω from the replicate's own items: R refits
    psych::omega on each replicate's rows, and NumPy by hand recalibrates with that ω. The app's 40
    bootstrap estimates and their standard error agree to 1e-5."""
    frame, items, W, Z = _lin(n=500)
    y = frame["sbp"].to_numpy()
    n, B = len(W), 40
    rng = np.random.default_rng(7)
    draws = np.vstack([rng.integers(0, n, n) for _ in range(B)])
    pd.DataFrame(items, columns=SAT).to_csv(tmp_path / "items.csv", index=False)
    pd.DataFrame(draws + 1).to_csv(tmp_path / "draws.csv", index=False)
    script = """
suppressMessages({library(psych); library(jsonlite)})
x <- read.csv("items.csv"); dr <- as.matrix(read.csv("draws.csv"))
om <- sapply(seq_len(nrow(dr)), function(b) {
  xb <- x[dr[b, ], ]
  o <- suppressWarnings(suppressMessages(omega(xb, nfactors = 1, plot = FALSE, flip = FALSE)))
  s <- apply(xb, 2, sd); 1 - sum(s^2 * o$schmid$sl[, "u2"]) / sum(cov(xb))
})
writeLines(toJSON(list(omega = om), digits = NA), "out.json")
"""
    ref = run_r(script, tmp_path)
    by_hand = []
    for rows, om in zip(draws, ref["omega"]):
        Wb, Zb, yb = W[rows], Z[rows], y[rows]
        X_hat, _ = calibrate_by_hand(Wb, Zb, (1 - om) * np.var(Wb, ddof=1))
        by_hand.append(sm.OLS(yb, sm.add_constant(np.column_stack([X_hat, Zb]))).fit().params[1])
    source = S.Source("internal_consistency",
                      reliability=S.internal_consistency(items, "sum", "unidimensional", 1))
    got = S.correct(W, Z, y, 0, source, draws=draws)
    assert got.n_boot == got.n_boot_ok == B
    np.testing.assert_allclose(got.boot, by_hand, rtol=1e-5)
    assert got.se == pytest.approx(np.std(by_hand, ddof=1), rel=1e-5)
    assert got.ci_low == pytest.approx(got.estimate - S.Z_95 * got.se, rel=1e-12)


def test_3_by_simulation_the_bootstrap_interval_covers_the_true_coefficient():
    """Known truth (continuous congeneric items, ω ≈ 0.79, the true score correlated 0.5 with the
    covariate, 1.5 per unit of true score). Over 150 datasets of 300, the 95% interval from 100
    replicates that each re-estimate ω covers 1.5 in at least 90% of them (the binomial Monte Carlo
    error at 0.95 is 0.018); the uncorrected interval covers it in fewer than half."""
    rng = np.random.default_rng(2026)
    lam, psi = [0.9, 0.8, 0.7, 0.8, 0.8], [0.7, 0.9, 1.0, 0.8, 0.8]
    total = sum(lam)
    covered = naive_covered = 0
    reps = 150
    for r in range(reps):
        X, z, F = congeneric(rng, 300, lam, psi, z_effect=0.5)
        y = 1.5 * total * F + 2.0 * z + rng.normal(0, 2, len(z))
        W = X.sum(axis=1)
        source = S.Source("internal_consistency",
                          reliability=S.internal_consistency(X, "sum", "unidimensional", 1))
        got = S.correct(W, z[:, None], y, 0, source, n_boot=100, seed=r)
        covered += got.ci_low <= 1.5 <= got.ci_high
        fit = sm.OLS(y, sm.add_constant(np.column_stack([W, z]))).fit()
        lo, hi = fit.conf_int()[1]
        naive_covered += lo <= 1.5 <= hi
    print(f"\ncoverage corrected {covered / reps:.3f} | uncorrected {naive_covered / reps:.3f}")
    assert covered / reps >= 0.90
    assert naive_covered / reps < 0.5


@needs_r
def test_3_a_repeat_administrations_icc_and_correction_agree_with_psych_and_mecor(tmp_path):
    """Test–retest: ICC(3,1) is psych::ICC's ICC3 on the two administrations to 1e-9; the error
    variance is their two-way residual mean square, var(W₁ − W₂)/2 by hand; and the corrected
    coefficient is mecor's MeasErrorRandom with that variance (rows without a repeat are calibrated
    too), to 1e-8."""
    rng = np.random.default_rng(5)
    n = 900
    age = rng.uniform(20, 80, n)
    T = 20 + 4 * rng.standard_normal(n) + 0.05 * (age - 50)
    W1 = T + rng.normal(0, 2.5, n)
    W2 = np.where(rng.random(n) < 0.4, T + 0.6 + rng.normal(0, 2.5, n), np.nan)
    y = 3 + 0.8 * T + 0.1 * age + rng.normal(0, 3, n)
    pd.DataFrame({"y": y, "W1": W1, "W2": W2, "age": age}).to_csv(tmp_path / "rt.csv", index=False)
    both = np.isfinite(W2)
    ms_e = np.var(W1[both] - W2[both], ddof=1) / 2
    script = f"""
suppressMessages({{library(psych); library(mecor); library(jsonlite)}})
dd <- read.csv("rt.csv")
icc <- suppressWarnings(suppressMessages(ICC(dd[!is.na(dd$W2), c("W1", "W2")], lmer = FALSE)))
m <- mecor(y ~ MeasErrorRandom(substitute = W1, variance = {float(ms_e)!r}) + age, data = dd,
           method = "standard")
writeLines(toJSON(list(icc3 = icc$results["Single_fixed_raters", "ICC"],
                       mecor = unname(m$corfit$coef["cor_W1"])), digits = NA, auto_unbox = TRUE),
           "out.json")
"""
    ref = run_r(script, tmp_path)
    found = S.retest_reliability(W1, W2)
    assert found.icc == pytest.approx(ref["icc3"], abs=1e-9)
    assert found.error_variance == pytest.approx(ms_e, rel=1e-12)
    assert found.n == int(both.sum())
    got = S.correct(W1, age[:, None], y, 0, S.Source("test_retest",
                                                     reliability=S.retest_source(W1, W2)),
                    n_boot=0)
    assert got.estimate == pytest.approx(ref["mecor"], rel=1e-8)
    assert got.reliability == pytest.approx(ref["icc3"], abs=1e-9)


@needs_r
def test_3_a_calibration_substudy_is_mecors_internal_validation(tmp_path):
    """A reference measure on 35% of the rows: the calibration regresses it on the score and the
    covariates there, every row is calibrated, and the corrected coefficient is mecor's
    ``MeasError(W, reference = X)`` (method "standard") to 1e-8."""
    frame, items, W, Z = _lin()
    y = frame["sbp"].to_numpy()
    ref_col = frame["sat_ref"].to_numpy()
    pd.DataFrame({"sbp": y, "W": W, "age": Z[:, 0], "bmi": Z[:, 1], "ref": ref_col}).to_csv(
        tmp_path / "sub.csv", index=False)
    script = """
suppressMessages({library(mecor); library(jsonlite)})
dd <- read.csv("sub.csv")
m <- mecor(sbp ~ MeasError(substitute = W, reference = ref) + age + bmi, data = dd,
           method = "standard")
writeLines(toJSON(list(mecor = unname(m$corfit$coef["ref"])), digits = NA, auto_unbox = TRUE),
           "out.json")
"""
    ref = run_r(script, tmp_path)
    got = S.correct(W, Z, y, 0, S.Source("calibration_substudy", reference=ref_col), n_boot=0)
    assert got.estimate == pytest.approx(ref["mecor"], rel=1e-8)


# ── through the server: a linear outcome, complete and with missing items ────


ROLES_LIN = {"pid": "identifier", "age": "covariate", "bmi": "covariate", "sat_ref": "excluded",
             **{c: "covariate" for c in SAT}}
SAT_SCALE = {"name": "sat_score", "items": SAT, "reverse": SAT_REVERSE, "low": 1, "high": 5,
             "kind": "reflective", "correction": "regression_calibration", "n_boot": 100}


# WP17 asks the adjustment set column by column, before the scales answer (which is not yet one of
# the Router's questions): every predictor here, the scale's items among them, causes both the
# exposure the card leads with and the outcome, so all of them stay in the model as SCALES built it.
CONFOUNDS = "yes,yes,no"


def lin_truth() -> Truth:
    return Truth({"code_or_count:age": "amount", "code_or_count:bmi": "amount",
                  **{f"adjust:{c}": CONFOUNDS for c in ("age", "bmi", *SAT)}},
                 fixture="linear_scale_table")


def drive_linear(client, path: Path, missing: str, extra: list[dict] = (),
                 purpose: str = "inference") -> dict:
    d_ = open_project(client, path, lin_truth())
    d_.decide({"kind": "set_lens", "lenses": ["survey"]})
    d_.reach("target")
    d_.decide({"kind": "set_target", "column": "sbp"})
    d_.answer("task", {"kind": "set_task", "column": "sbp", "task": "regression"})
    d_.reach("purpose")
    d_.decide({"kind": "set_purpose", "purpose": purpose})
    d_.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
    d_.reach("roles")
    d_.decide_roles(ROLES_LIN)
    d_.answer("exclusions", {"kind": "set_exclusions", "rules": []})
    d_.answer("missing", {"kind": "set_missing", "strategy": missing})
    d_.answer("split", {"kind": "set_split", "holdout": 0.0 if purpose == "inference" else 0.2,
                        "seed": 0})
    scale = SAT_SCALE if purpose == "inference" else {**SAT_SCALE, "correction": "none"}
    d_.decide({"kind": "set_scales", "scales": [scale]})
    d_.reach("models")
    d_.decide({"kind": "select_models", "models": ["linear"]})
    out = {"scales": d_.artifact("scales"), "fit": d_.artifact("fit"),
           "design": d_.artifact("design")}
    if purpose == "prediction":
        out["sealed"] = d_.sealed()
    for i, body in enumerate(extra):
        d_.decide(body)
        out[f"extra{i}"] = d_.artifact("scales")
    return out


@pytest.fixture(scope="module")
def linear_runs(tmp_path_factory) -> dict:
    home = tmp_path_factory.mktemp("ms8_linear")
    complete = linear_scale_table()
    complete.to_csv(home / "complete.csv", index=False)
    holes = linear_scale_table(missing=0.15)
    holes.to_csv(home / "holes.csv", index=False)
    substudy = {**SAT_SCALE, "reliability": "calibration_substudy", "reference": "sat_ref"}
    with local_server(home / "srv") as client:
        runs = {"complete": drive_linear(client, home / "complete.csv", "complete_case",
                                         [{"kind": "set_scales", "scales": [substudy]}]),
                "mi": drive_linear(client, home / "holes.csv", "multiple_imputation"),
                "prediction": drive_linear(client, home / "holes.csv", "impute",
                                           purpose="prediction")}
    runs["frames"] = {"complete": complete, "holes": holes}
    return runs


@needs_r
def test_3_through_the_server_the_correction_is_mecors_beside_the_uncorrected_table(linear_runs,
                                                                                    tmp_path):
    """The real server, complete rows, a least-squares outcome: the score's ω is psych's (the
    score's own, from psych's solution), its corrected coefficient mecor's MeasErrorRandom on the
    CSV, both to 1e-5; the uncorrected coefficient beside it is the fit's own table row; the result
    is labeled a declared secondary analysis omitting transient error, its calibration covers
    every other column of the model's matrix, and its interval comes from 100 replicates."""
    frame = linear_runs["frames"]["complete"]
    items = keyed(frame, SAT, SAT_REVERSE, 1, 5)
    items.to_csv(tmp_path / "items.csv", index=False)
    frame.assign(W=items.sum(axis=1))[["sbp", "W", "age", "bmi"]].to_csv(tmp_path / "lin.csv",
                                                                          index=False)
    extra = """
suppressMessages(library(mecor))
dd <- read.csv("lin.csv")
s2u <- (1 - out$score_omega_tot) * var(dd$W)
m <- mecor(sbp ~ MeasErrorRandom(substitute = W, variance = s2u) + age + bmi, data = dd)
out$mecor <- unname(m$corfit$coef["cor_W"])
"""
    ref = run_r(omega_script("items.csv", 1, extra), tmp_path)
    (scale,) = linear_runs["complete"]["scales"]["scales"]
    rel, corr = scale["reliability"], scale["correction"]
    assert (rel["coefficient"], rel["label"]) == ("omega_total", "ω-total")
    assert rel["value"] == pytest.approx(ref["score_omega_tot"], abs=1e-5)
    assert rel["omega_total_standardized"] == pytest.approx(ref["omega_tot"], abs=1e-5)
    assert rel["alpha"] == pytest.approx(ref["alpha"], abs=1e-6)
    assert "customary" in rel["alpha_label"]
    assert corr["estimate"] == pytest.approx(ref["mecor"], rel=1e-5)
    row = next(c for c in linear_runs["complete"]["fit"]["models"][0]["coefficients"]
               if c["feature"] == "sat_score")
    assert corr["naive"] == pytest.approx(row["estimate"], rel=1e-10)
    assert corr["naive_ci_low"] == pytest.approx(row["ci_low"], rel=1e-10)
    assert corr["p"] == pytest.approx(row["p"], rel=1e-8)
    assert sorted(corr["covariates"]) == ["age", "bmi"]
    assert corr["n_boot"] == corr["n_boot_ok"] == 100 and corr["copies"] == 1
    assert corr["ci_low"] < corr["estimate"] < corr["ci_high"]
    assert any(l.startswith("A declared secondary analysis") for l in corr["labels"])
    assert any("omits transient error" in l for l in corr["labels"])
    assert not any(c["feature"] in SAT for c in
                   linear_runs["complete"]["fit"]["models"][0]["coefficients"])


@needs_r
def test_3_through_the_server_the_calibration_substudy_exit_runs(linear_runs, tmp_path):
    """The calibration-substudy answer, posted on the same project, recomputes the scales stage:
    its corrected coefficient is mecor's internal-validation calibration on the CSV to 1e-6, and its
    reliability is the calibration slope given the covariates."""
    frame = linear_runs["frames"]["complete"]
    W = keyed(frame, SAT, SAT_REVERSE, 1, 5).sum(axis=1)
    frame.assign(W=W)[["sbp", "W", "age", "bmi", "sat_ref"]].to_csv(tmp_path / "sub.csv",
                                                                     index=False)
    ref = run_r("""
suppressMessages({library(mecor); library(jsonlite)})
dd <- read.csv("sub.csv")
m <- mecor(sbp ~ MeasError(substitute = W, reference = sat_ref) + age + bmi, data = dd)
writeLines(toJSON(list(mecor = unname(m$corfit$coef["sat_ref"])), digits = NA,
                  auto_unbox = TRUE), "out.json")
""", tmp_path)
    (scale,) = linear_runs["complete"]["extra0"]["scales"]
    assert scale["reliability"]["source"] == "calibration_substudy"
    assert scale["reliability"]["n"] == int(frame["sat_ref"].notna().sum())
    assert scale["correction"]["estimate"] == pytest.approx(ref["mecor"], rel=1e-6)
    assert scale["methods"].startswith(
        "`sat_score` was the sum of its 8 items, `sat_3` and `sat_6` reverse-coded on the 1–5 "
        "response scale. Reliability was estimated from a calibration substudy against `sat_ref` (")


@needs_r
def test_1_under_prediction_the_score_is_formed_in_fold_and_its_reliability_is_descriptive(
        linear_runs, tmp_path):
    """Prediction, a holdout of 20%, blanks filled in each training fold: the score is a pipeline
    step after the fill (so a blank item is filled before the score is formed, and the held-out
    rows are scored by the fitted pipeline as a new row would be); the reliability describes the
    training rows only (psych::omega on their complete answers, to 1e-5), labeled as no modeling
    choice; nothing is corrected."""
    run = linear_runs["prediction"]
    frame = linear_runs["frames"]["holes"]
    (scale,) = run["scales"]["scales"]
    assert run["scales"]["purpose"] == "prediction" and run["scales"]["rows"] == "training rows"
    assert [s["key"] for s in run["design"]["models"][0]["steps"]][:2] == ["impute", "score"]
    assert run["fit"]["models"][0]["cv"]["r2"]["estimate"] > 0.1
    sealed = run["sealed"]
    assert len(sealed) == round(0.2 * len(frame))
    training = keyed(frame.drop(index=sorted(sealed)), SAT, SAT_REVERSE, 1, 5).dropna()
    training.to_csv(tmp_path / "items.csv", index=False)
    ref = run_r(omega_script("items.csv", 1), tmp_path)
    assert scale["reliability"]["n"] == len(training)
    assert scale["reliability"]["value"] == pytest.approx(ref["score_omega_tot"], abs=1e-5)
    assert scale["correction"] is None and scale["not_corrected"] is None
    assert scale["methods"].endswith("Its coefficient was not corrected for measurement error.")


# ── (4) item-level multiple imputation before scoring ────────────────────────


def test_4_under_multiple_imputation_items_are_imputed_and_then_scored_in_each_copy(linear_runs):
    """Missing answers (15% an item, more often for older participants) under multiple imputation:
    the imputation model holds the items and the outcome, never the score (item level, Eekhout et
    al. 2014); the score is formed in each completed copy. Independent check: the imputation API
    (``methods.missing.impute_for_inference``, owned by the MI package) run on the design's own
    spec gives the copies; in each, pandas sums the keyed imputed items and statsmodels fits OLS;
    Rubin's rules by hand over the m copies give the coefficient the server's fit table and the
    scales stage both report, to 1e-8. Imputing the total score instead gives a different
    number. m follows MS2's rule (the MI package's): at least 20 and at least the percentage of
    rows with any blank, counted here by pandas."""
    from turbotab.core.methods.missing import impute_for_inference
    from turbotab.core.models.pipeline import design_spec

    frame = linear_runs["frames"]["holes"]
    run = linear_runs["mi"]
    (scale,) = run["scales"]["scales"]
    missing = run["fit"]["models"][0]["inference"]["missing"]
    assert "sat_score" not in missing["variables"] and set(SAT) <= set(missing["variables"])
    assert "the outcome" in missing["variables"]
    incomplete = frame[["sbp", "age", "bmi", *SAT]].isna().any(axis=1).mean()
    m = max(20, math.ceil(100 * incomplete - 1e-9))
    assert m > 20  # 15% blank in each of 8 items leaves most rows incomplete
    assert scale["imputation"]["m"] == m
    assert scale["imputation"]["imputed"] == {c: int(frame[c].isna().sum()) for c in SAT}
    assert [s["key"] for s in run["design"]["models"][0]["steps"]][:2] == ["impute", "score"]

    state = d.ProjectState(purpose="inference", target="sbp", roles=dict(ROLES_LIN),
                           missing=d.MissingSpec(strategy="multiple_imputation"),
                           scales=[d.ScaleSpec(**SAT_SCALE)])
    X = frame[["age", "bmi", *SAT]]
    spec = design_spec(state, X, ["age", "bmi", *SAT])
    y = frame["sbp"].to_numpy(dtype=float)
    imputations = impute_for_inference(spec, X[spec.inputs], y, "regression", seed=0)
    q, u = [], []
    for copy in imputations.frames:
        W = keyed(copy, SAT, SAT_REVERSE, 1, 5).sum(axis=1)
        fit = sm.OLS(y, sm.add_constant(np.column_stack([W, copy["age"], copy["bmi"]]))).fit()
        q.append(fit.params[1])
        u.append(fit.get_robustcov_results("HC3").bse[1] ** 2)  # the table's HC3, per copy
    pooled = float(np.mean(q))
    table = next(c for c in run["fit"]["models"][0]["coefficients"] if c["feature"] == "sat_score")
    assert table["estimate"] == pytest.approx(pooled, rel=1e-8)
    assert scale["correction"]["naive"] == pytest.approx(pooled, rel=1e-8)
    assert len(q) == m
    total = float(np.mean(u)) + (1 + 1 / m) * float(np.var(q, ddof=1))
    assert table["se"] == pytest.approx(math.sqrt(total), rel=1e-6)
    assert scale["correction"]["copies"] == m
    assert len(scale["reliability"]["across_copies"]) == m
    assert scale["reliability"]["value"] == pytest.approx(
        np.mean(scale["reliability"]["across_copies"]), rel=1e-12)
    # The score-level alternative: impute the total, then fit.
    totals = frame.assign(total=keyed(frame, SAT, SAT_REVERSE, 1, 5).sum(axis=1, skipna=False))
    state_t = state.model_copy(update={"scales": None,
                                       "roles": {"age": "covariate", "bmi": "covariate",
                                                 "total": "covariate"}})
    Xt = totals[["age", "bmi", "total"]]
    spec_t = design_spec(state_t, Xt, ["age", "bmi", "total"])
    total_level = np.mean([sm.OLS(y, sm.add_constant(c[["total", "age", "bmi"]].to_numpy()))
                           .fit().params[1]
                           for c in impute_for_inference(spec_t, Xt, y, "regression",
                                                         seed=0).frames])
    assert abs(total_level - pooled) > 1e-3
    assert scale["methods"].endswith(
        f"Missing items were multiply imputed before scoring (item level, m = {m}), and the score, "
        "its reliability and the correction were estimated in each completed copy and pooled.")
    assert "(100 replicates in each completed copy)" in scale["methods"]


def test_4_by_simulation_item_level_imputation_and_bootstrap_cover_the_truth():
    """Known truth (continuous congeneric items, true coefficient 1.5 per unit of true score),
    item answers missing at random given the covariate (about 20% an item). Item-level multiple
    imputation by the imputation API (``methods.imputation.chained_equations``, m = 5), the score
    and its ω formed in each copy, the correction's bootstrap (60 replicates) in each copy, Rubin's
    rules over the copies: over 80 datasets of 300 the interval covers 1.5 in at least 88% of them
    (the binomial Monte Carlo error at 0.95 is 0.024)."""
    from turbotab.core.methods.imputation import chained_equations, pool_scalar

    rng = np.random.default_rng(77)
    lam, psi = [0.9, 0.8, 0.7, 0.8, 0.8], [0.7, 0.9, 1.0, 0.8, 0.8]
    total = sum(lam)
    covered, reps = 0, 80
    for r in range(reps):
        X, z, F = congeneric(rng, 300, lam, psi, z_effect=0.5)
        y = 1.5 * total * F + 2.0 * z + rng.normal(0, 2, len(z))
        p = 1 / (1 + np.exp(-(-1.5 + 0.8 * z)))
        holes = np.where(rng.random(X.shape) < p[:, None], np.nan, X)
        data = pd.DataFrame(holes, columns=[f"x{j}" for j in range(5)]).assign(z=z, y=y)
        copies = chained_equations(data, impute=[f"x{j}" for j in range(5)], m=5, seed=r).frames
        est, var = [], []
        for c in copies:
            M = c[[f"x{j}" for j in range(5)]].to_numpy()
            source = S.Source("internal_consistency",
                              reliability=S.internal_consistency(M, "sum", "unidimensional", 1))
            got = S.correct(M.sum(axis=1), z[:, None], y, 0, source, n_boot=60, seed=r)
            est.append(got.estimate)
            var.append(got.se ** 2)
        pooled = pool_scalar(est, var)
        covered += pooled.ci_low <= 1.5 <= pooled.ci_high
    print(f"\ncoverage {covered / reps:.3f}")
    assert covered / reps >= 0.88


# ── the leash: codes in the items ────────────────────────────────────────────


def test_2_a_code_among_the_answers_is_refused_until_it_is_recoded(tmp_path):
    """§4 and the contract's "conflicts" relation: an item holding a value outside the declared
    response scale (a 9, "refused", in a 0–4 item) is refused, never scored on a guess; the
    exits are the sentinel finding's own repair and a wider response scale. Once the repair is
    applied (the 9s recoded to missing, row-local), the same answer is recorded."""
    frame = survey_scale_table(n=400)
    rows = frame.index[:25]
    frame.loc[rows, "pss_2"] = 9
    frame.to_csv(tmp_path / "codes.csv", index=False)
    roles = {"participant_id": "identifier", "age": "covariate", "sex": "covariate",
             **{c: "covariate" for c in PSS_ITEMS + DQ_ITEMS}, **{c: "excluded" for c in DQ_RETEST}}
    with local_server(tmp_path / "srv") as client:
        drive = open_project(client, tmp_path / "codes.csv", Truth(fixture="codes"))
        drive.decide({"kind": "set_lens", "lenses": ["survey"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "wellbeing"})
        drive.answer("task", {"kind": "set_task", "column": "wellbeing", "task": "ordinal"})
        drive.reach("purpose")
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        drive.reach("roles")
        drive.decide_roles(roles)
        drive.artifact("findings")
        body = {"kind": "set_scales", "scales": [
            {"name": "pss_score", "items": PSS_ITEMS, "reverse": PSS_REVERSE, "low": 0,
             "high": 4, "kind": "reflective"}]}
        r = drive.post(body)
        assert r.status_code == 409, r.text
        error = r.json()["error"]
        assert error["code"] == "answers_outside_the_scale"
        assert "`pss_2` 0–9" in error["message"]
        repair, wider = error["exits"]
        assert repair["decision"]["kind"] == "apply_repair"
        assert wider["decision"]["scales"][0]["high"] == 9
        drive.decide(repair["decision"])
        drive.artifact("working")
        r = drive.post(body)
        assert r.status_code == 200, r.text


# ── (5) chain 4: a survey scale ──────────────────────────────────────────────

CHAIN_ROLES = {"participant_id": "identifier", "age": "covariate", "sex": "covariate",
               **{c: "covariate" for c in PSS_ITEMS + DQ_ITEMS},
               **{c: "excluded" for c in DQ_RETEST}}
CHAIN_SCALES = [
    {"name": "pss_score", "items": PSS_ITEMS, "reverse": PSS_REVERSE, "low": 0, "high": 4,
     "kind": "reflective", "correction": "regression_calibration", "n_boot": 50,
     "instrument": "PSS-10"},
    {"name": "dq_score", "items": DQ_ITEMS, "low": 0, "high": 10, "kind": "formative",
     "correction": "regression_calibration", "n_boot": 50},
]


@pytest.fixture(scope="module")
def chain4(tmp_path_factory) -> dict:
    """MODELING_SEQUENCE §6 chain 4 through the real server, purpose inference, an ordered outcome:
    a reflective scale corrected from ω; a formative diet score whose disattenuation by ω is refused
    and corrected instead from the test–retest ICC the refusal's exit asks for; a form recorded on
    an item re-asked; then the adjustment set changed."""
    home = tmp_path_factory.mktemp("ms8_chain4")
    frame = survey_scale_table()
    frame.to_csv(home / "chain4.csv", index=False)
    out: dict = {"frame": frame}
    with local_server(home / "srv") as client:
        drive = open_project(client, home / "chain4.csv",
                             Truth({"code_or_count:age": "amount",
                                    **{f"adjust:{c}": CONFOUNDS
                                       for c in ("age", "sex", *PSS_ITEMS, *DQ_ITEMS)}},
                                   fixture="survey_scale_table"))
        drive.decide({"kind": "set_lens", "lenses": ["survey"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "wellbeing"})
        drive.answer("task", {"kind": "set_task", "column": "wellbeing", "task": "ordinal"})
        drive.reach("purpose")
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        drive.reach("roles")
        drive.decide_roles(CHAIN_ROLES)
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0})
        drive.decide({"kind": "set_exposure_form", "column": "pss_3", "form": "spline"})
        r = drive.post({"kind": "set_scales", "scales": CHAIN_SCALES})
        out["formative"] = (r.status_code, r.json())
        retest = r.json()["error"]["exits"][0]["decision"]
        r = drive.post(retest)
        out["item_form"] = (r.status_code, r.json())
        drive.decide(r.json()["error"]["exits"][0]["decision"])
        drive.decide(retest)
        out["record"] = drive.view()["decisions"][-1]
        r = drive.post({"kind": "set_exposure_form", "column": "pss_3", "form": "spline"})
        out["scored_item"] = (r.status_code, r.json())
        drive.reach("models")
        drive.decide({"kind": "select_models", "models": ["proportional_odds"]})
        out["scales"] = drive.artifact("scales")
        out["fit"] = drive.artifact("fit")
        out["design"] = drive.artifact("design")
        drive.decide_roles({**CHAIN_ROLES, "age": "excluded"})
        # WP17: the exposure the card led with (age) left, so the plan's questions reopen, and the
        # corrected coefficient, an estimate, waits for them (served withheld until answered).
        out["withheld"] = drive.artifact("scales")
        for key in ("estimand", "adjustment"):
            answer_wp17(drive, key)
        out["without_age"] = drive.artifact("scales")
    return out


def _chain_references(frame: pd.DataFrame, tmp_path: Path) -> dict:
    """ω and α of the stress scale (psych::omega and psych::alpha on the keyed items), the diet
    score's ICC(3,1) (psych::ICC), its error variance var(W₁ − W₂)/2, by hand."""
    pss = keyed(frame, PSS_ITEMS, PSS_REVERSE, 0, 4)
    pss.to_csv(tmp_path / "items.csv", index=False)
    W1 = frame[DQ_ITEMS].sum(axis=1)
    W2 = frame[DQ_RETEST].sum(axis=1, skipna=False)
    pd.DataFrame({"W1": W1, "W2": W2}).to_csv(tmp_path / "rt.csv", index=False)
    extra = """
rt <- read.csv("rt.csv"); rt <- rt[!is.na(rt$W2), ]
icc <- suppressWarnings(suppressMessages(ICC(rt, lmer = FALSE)))
out$icc3 <- icc$results["Single_fixed_raters", "ICC"]; out$n_both <- nrow(rt)
"""
    ref = run_r(omega_script("items.csv", 1, extra), tmp_path)
    both = W2.notna().to_numpy()
    ref["dq_error"] = float(np.var((W1 - W2).to_numpy()[both], ddof=1) / 2)
    ref["pss"], ref["dq"] = pss.sum(axis=1).to_numpy(), W1.to_numpy(dtype=float)
    return ref


def _ordered(y: np.ndarray, X: np.ndarray) -> float:
    """The proportional-odds slope of the first column, by statsmodels (MASS::polr's model)."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    fit = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False, maxiter=5000,
                                                gtol=1e-7)
    assert fit.mle_retvals["converged"]
    return float(fit.params[0])


@needs_r
def test_5_chain_4_runs_end_to_end_with_its_relations_and_methods_sentences(chain4, tmp_path):
    """Chain 4 (MODELING_SEQUENCE §6): "A reflective scale: ω, conditional correction by RC, both
    estimates reported. A formative diet score: disattenuation refused, with test–retest ICC asked
    instead." Asserted, in the order the engine runs it (§1.1):

    * the formative score's disattenuation by ω is refused, its first exit the test–retest ICC of
      the repeat administration the table holds;
    * that answer is held while an item carries a form (§2: a domain transform invalidates the
      form, re-asked: its exit sets the item straight), then recorded; a form on a scored item is
      refused afterwards;
    * the scores replace their items in the matrix (the design's step and lineage; the fit's
      table), the items read as answers without a code-or-amount ask;
    * each reliability is R's (ω and α by psych::omega and psych::alpha, the ICC by psych::ICC),
      each correction the proportional-odds refit of NumPy's calibration by hand with every other
      column of the matrix as a covariate (statsmodels), labeled approximate; the uncorrected
      estimate beside it is the fit's table row;
    * the methods sentences, verbatim; the diet score's, with its numbers set aside, is the
      reviewers' required sentence;
    * changing the adjustment set (age leaves) recomputes the calibration without it.
    """
    frame = chain4["frame"]
    ref = _chain_references(frame, tmp_path)
    status, body = chain4["formative"]
    assert status == 409 and body["error"]["code"] == "formative_disattenuation"
    labels = [e["label"] for e in body["error"]["exits"]]
    assert labels[0].startswith("Use the test–retest ICC of the repeat administration")
    assert "calibration substudy" in labels[1] and "uncorrected" in labels[2]
    status, body = chain4["item_form"]
    assert status == 409 and body["error"]["code"] == "item_has_a_form"
    assert body["error"]["exits"][0]["decision"] == {"kind": "set_exposure_form",
                                                     "column": "pss_3", "form": "linear",
                                                     "knots": None}
    status, body = chain4["scored_item"]
    assert status == 409 and body["error"]["code"] == "scored_item"
    assert chain4["record"]["sentence"] == (
        "`pss_score` is the sum of the PSS-10's 10 items, `pss_4`, `pss_5`, `pss_7` and `pss_8` "
        "reverse-coded on the 0–4 response scale, a reflective scale entering the models as an "
        "exposure; its coefficient is corrected by regression calibration from its internal "
        "consistency (ω), as a secondary analysis beside the uncorrected one; `dq_score` is the "
        "sum of its 6 components, a formative index entering the models as an exposure; its "
        "coefficient is corrected by regression calibration from the test–retest ICC of its repeat "
        "administration, as a secondary analysis beside the uncorrected one.")

    steps = [s["key"] for s in chain4["design"]["models"][0]["steps"]]
    assert steps.index("score") < steps.index("model")
    ops = {(l["target"], l["operation"]) for l in chain4["design"]["lineage"]["links"]}
    assert ("adj:pss_score", "scale score (sum)") in ops
    table = {c["feature"]: c for c in chain4["fit"]["models"][0]["coefficients"]}
    assert {"pss_score", "dq_score"} <= set(table) and not set(PSS_ITEMS + DQ_ITEMS) & set(table)

    pss, dq = chain4["scales"]["scales"]
    y = frame["wellbeing"].to_numpy() - 1
    male = (frame["sex"] == "M").to_numpy(dtype=float)
    age = frame["age"].to_numpy(dtype=float)
    # the stress scale: ω, α, and the calibration given sex, age and the diet score
    assert pss["reliability"]["value"] == pytest.approx(ref["score_omega_tot"], abs=1e-5)
    assert pss["reliability"]["alpha"] == pytest.approx(ref["alpha"], abs=1e-6)
    Z = np.column_stack([male, age, ref["dq"]])
    s2u = (1 - ref["score_omega_tot"]) * np.var(ref["pss"], ddof=1)
    X_hat, lam = calibrate_by_hand(ref["pss"], Z, s2u)
    assert pss["correction"]["attenuation"] == pytest.approx(lam, rel=1e-5)
    assert pss["correction"]["estimate"] == pytest.approx(_ordered(y, np.column_stack([X_hat, Z])),
                                                          rel=1e-4)
    assert pss["correction"]["naive"] == pytest.approx(table["pss_score"]["estimate"], rel=1e-9)
    assert sorted(pss["correction"]["covariates"]) == ["age", "dq_score", "sex_M"]
    # the diet score: the ICC, the error variance, and the calibration given sex, age and stress
    assert dq["reliability"]["value"] == pytest.approx(ref["icc3"], abs=1e-9)
    assert dq["reliability"]["n"] == ref["n_both"]
    assert dq["correction"]["error_variance"] == pytest.approx(ref["dq_error"], rel=1e-10)
    Z = np.column_stack([male, age, ref["pss"]])
    X_hat, lam = calibrate_by_hand(ref["dq"], Z, ref["dq_error"])
    assert dq["correction"]["estimate"] == pytest.approx(_ordered(y, np.column_stack([X_hat, Z])),
                                                         rel=1e-4)
    assert dq["correction"]["naive"] == pytest.approx(table["dq_score"]["estimate"], rel=1e-9)
    for s in (pss, dq):
        corr = s["correction"]
        assert corr["scale"] == "odds_ratio" and corr["ratio"] == pytest.approx(
            math.exp(corr["estimate"]))
        assert corr["n_boot"] == corr["n_boot_ok"] == 50
        assert any("approximation" in l for l in corr["labels"])
        assert any("each on its own" in c for c in s["concerns"])
    assert dq["correction"]["estimate"] > dq["correction"]["naive"] > 0  # the ICC's attenuation

    approx = " In a proportional-odds model the calibrated score is an approximation (Carroll et al. 2006)."
    assert pss["methods"] == (
        "`pss_score` was the sum of the PSS-10's 10 items, `pss_4`, `pss_5`, `pss_7` and `pss_8` "
        "reverse-coded on the 0–4 response scale. Its reliability was ω-total = "
        f"{ref['score_omega_tot']:.2f} for the score (McDonald's ω; minimum-residual factor "
        "analysis of the item correlations, one factor; Cronbach's α = "
        f"{ref['alpha']:.2f}, reported as customary). The corrected coefficient was obtained by "
        "regression calibration including all model covariates, with bootstrap CIs re-estimating "
        "the reliability (50 replicates); uncorrected and corrected estimates are both reported. "
        "A reliability from internal consistency omits transient error, so the correction "
        "under-corrects." + approx)
    assert dq["methods"] == (
        "`dq_score` was the sum of its 6 components. Reliability was estimated as the test–retest "
        f"ICC from a repeat administration (ICC(3,1) = {ref['icc3']:.2f}, {ref['n_both']:,} "
        "participants with both); the corrected coefficient was obtained by regression calibration "
        "including all model covariates, with bootstrap CIs re-estimating the reliability (50 "
        "replicates); uncorrected and corrected estimates are both reported." + approx)
    bare = (dq["methods"].replace(f" (ICC(3,1) = {ref['icc3']:.2f}, {ref['n_both']:,} participants "
                                  f"with both)", "").replace(" (50 replicates)", ""))
    assert REVIEWERS.replace(" (or a calibration substudy)", "") in bare
    assert chain4["scales"]["methods"] == f"{pss['methods']} {dq['methods']}"

    # §2: a change to the adjustment set recomputes the calibration with the new covariates; WP17:
    # while the questions it reopened were open, the corrections were withheld.
    held = chain4["withheld"]
    assert held["withheld"].startswith("No estimate is shown until")
    assert all(s["correction"] is None for s in held["scales"])
    pss_b, dq_b = chain4["without_age"]["scales"]
    assert sorted(pss_b["correction"]["covariates"]) == ["dq_score", "sex_M"]
    Z = np.column_stack([male, ref["dq"]])
    X_hat, _ = calibrate_by_hand(ref["pss"], Z, s2u)
    assert pss_b["correction"]["estimate"] == pytest.approx(
        _ordered(y, np.column_stack([X_hat, Z])), rel=1e-4)


def test_5_the_items_are_answers_by_the_scales_answer_and_a_codes_answer_is_refused():
    """§14.3, every confirmation is honored: the scales answer reads its items as answers, so the
    fit asks no code-or-amount question for them; an item the user confirmed as codes is a clash the
    fit refuses, never a confirmation it ignores."""
    from turbotab.core.readings import Unsettled, predictors_or_ask

    roles = {c: "covariate" for c in PSS_ITEMS}
    info = {c: {"dtype": "integer", "n_unique": 5} for c in PSS_ITEMS}
    scales = [d.ScaleSpec(**{k: v for k, v in CHAIN_SCALES[0].items()})]
    state = d.ProjectState(purpose="inference", target="y", roles=roles, scales=scales)
    assert predictors_or_ask(state, info) == PSS_ITEMS
    with pytest.raises(Unsettled):
        predictors_or_ask(state.model_copy(update={"scales": None}), info)
    coded = state.model_copy(update={"shape_confirmations": {"code_or_count:pss_2": "code"}})
    with pytest.raises(Unsettled, match="the scales answer sums it as answers"):
        predictors_or_ask(coded, info)


def test_5_the_contract_declares_every_part_and_each_relation_names_live_code():
    """BLUEPRINT §13: the scales method enters through its contract — slot, data scope, needs,
    routing (its question, its place, its options with both labels and a rung per purpose),
    storyboard, sentence and relations — and every relation names a function that exists."""
    import importlib

    from turbotab.core.contracts import CONTRACTS
    from turbotab.core.scales import CONTRACT

    assert CONTRACTS["scales"] is CONTRACT and CONTRACT.decision == "set_scales"
    assert CONTRACT.slot == "in_fold" and CONTRACT.scope == "training_fold"
    assert CONTRACT.parts["scoring by the instrument's key (reverse coding; the sum or the mean)"] \
        == "row_local"
    assert CONTRACT.storyboard and CONTRACT.needs and "Step 4" in CONTRACT.place
    for target in [CONTRACT.sentence, *(r.enforced_by for r in CONTRACT.relations)]:
        module, name = target.split(":")
        assert callable(getattr(importlib.import_module(module), name)), target
    kinds = {(r.kind, r.target) for r in CONTRACT.relations}
    assert ("conflicts", "disattenuation of a formative index by α or ω") in kinds
    assert ("implies", "item-level multiple imputation before scoring") in kinds
    assert ("invalidates", "a functional form recorded on an item") in kinds
    inference = {o["key"]: o for o in CONTRACT.options_for("inference")}
    prediction = {o["key"]: o for o in CONTRACT.options_for("prediction")}
    assert inference["omega"]["rung"] == "recommended" and inference["alpha"]["rung"] == "rank_lower"
    assert inference["rc_internal_consistency_formative"]["rung"] == "refused"
    assert inference["spearman"]["rung"] == "refused"
    assert all(prediction[k]["rung"] == "refused" for k in prediction if k.startswith("rc_"))
    assert all(r.rung == "refused" and r.exits for r in CONTRACT.relations if r.kind == "conflicts")
    assert all(o["customary"] and o["sound"] for o in inference.values())
    assert CONTRACT.options_for("inference")[0]["rung"] == "recommended"


def test_5_every_methods_sentence_says_what_was_and_was_not_done():
    """The sentence for each answer the scales question takes, uncorrected included: finished,
    free of machinery, never claiming a reliability or a correction that was not computed."""
    from turbotab.core import voice
    from turbotab.core.scales import FORMATIVE, PREDICTION, methods_sentence

    pss = d.ScaleSpec(**{**CHAIN_SCALES[0], "correction": "none"})
    dq = d.ScaleSpec(**{**CHAIN_SCALES[1], "correction": "none"})
    dq_rt = d.ScaleSpec(**{**CHAIN_SCALES[1], "correction": "none", "reliability": "test_retest",
                           "retest": DQ_RETEST})
    omega_rel = {"source": "internal_consistency", "coefficient": "omega_total", "value": 0.871,
                 "alpha": 0.862, "factors": 1}
    cases = {
        "reflective, uncorrected": (pss, {"reliability": omega_rel}),
        "reflective, prediction": (pss, {"reliability": omega_rel, "not_corrected": PREDICTION}),
        "formative, no reliability": (dq, {"reliability": {"reason": FORMATIVE}}),
        "formative, test–retest only": (dq_rt, {"reliability": {
            "source": "test_retest", "coefficient": "test_retest_icc", "value": 0.64, "n": 300}}),
    }
    said = {k: methods_sentence(spec, result) for k, (spec, result) in cases.items()}
    for text in said.values():
        assert text.endswith(".") and "None" not in text and not voice.machinery(text)
        assert "regression calibration" not in text
    assert "ω-total = 0.87 for the score" in said["reflective, uncorrected"]
    assert said["reflective, prediction"].endswith(PREDICTION)
    assert "ω" not in said["formative, no reliability"]
    assert "α" not in said["formative, no reliability"]
    assert "ICC(3,1) = 0.64 (300 participants with both administrations)" in \
        said["formative, test–retest only"]
