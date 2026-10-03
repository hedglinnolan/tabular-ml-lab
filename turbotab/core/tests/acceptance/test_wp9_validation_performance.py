"""WP9 · Validation and performance reporting: the acceptance tests of AUDIT_REPORT.md §5 (closes
ME-10 and ME-11, and the minors E15 and E16).

Every measured value runs through the code the fit stage runs (``models/performance.py``,
``models/validation.py``, ``metrics.cross_validate``, the real split and fit stages); every
reference comes from somewhere else: a published reference output (pROC's), another library
(statsmodels, scikit-learn, scipy), or a definition written out in
``references_validation.py``. Simulations are seeded, so each run gives the same numbers; the
Monte Carlo error of each is stated beside its bound.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from turbotab.core import decisions as d
from turbotab.core import voice
from turbotab.core.decisions import GrainSpec, ProjectState, SplitSpec
from turbotab.core.models import get_family
from turbotab.core.models import performance as perf
from turbotab.core.models.inner_cv import fit_pipeline
from turbotab.core.models.metrics import PRIMARY, cross_validate, fold_pairs
from turbotab.core.models.pipeline import DesignSpec, build_pipeline
from turbotab.core.models.validation import RESAMPLE_BELOW, validation_plan
from turbotab.core.stages.modeling import design_stage, fit_stage
from turbotab.core.stages.rows import cohort_stage, draw_split, split_stage
from turbotab.core.stages.seal import seal_plan_stage
from turbotab.core.stages.target import target_info_stage
from turbotab.core.tests.acceptance import references_validation as ref
from turbotab.core.tests.acceptance.asah import asah
from turbotab.core.tests.stage_harness import Ingested


def _pipeline(family: str, task: str, columns: list[str], n: int):
    spec = DesignSpec(predictors=columns, inputs=columns, categorical=[], numeric=columns,
                      energy=None, impute=False)
    return build_pipeline(spec, get_family(family), task, "prediction", n, len(columns))


def _folds(n: int, seed: int, y=None) -> np.ndarray:
    """The app's own folds for n rows, cross-validation only (``draw_split``)."""
    frame, _ = draw_split(np.arange(n), holdout=0.0, seed=seed, folds=5, y=y)
    return frame.sort_values("row_id")["fold"].to_numpy()


def _cv(task, family, X, y, folds):
    pipe = _pipeline(family, task, list(X.columns), len(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cross_validate(task, lambda: clone(pipe), X, y, fold_pairs(folds),
                              fit=lambda m, Xf, yf, rows: fit_pipeline(m, Xf, yf),
                              keep_predictions=True)


def _logistic_table(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Five standard-normal predictors and an outcome drawn from a known logistic model, so the
    true risks are perfectly calibrated and a correctly specified logistic fit is too."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    risk = 1 / (1 + np.exp(-(-1.0 + X.to_numpy() @ np.array([0.8, -0.6, 0.5, 0.3, 0.0]))))
    return X, (rng.random(n) < risk).astype(int), risk


def _stages(table: Ingested, state: ProjectState):
    info = table.run(target_info_stage, state)
    cohort = table.run(cohort_stage, state, {"target_info": info})
    split = table.run(split_stage, state, {"cohort": cohort, "target_info": info})
    design = table.run(design_stage, state, {"split": split, "target_info": info})
    fit = table.run(fit_stage, state, {"design": design, "split": split, "target_info": info})
    return info, split, design, fit


def _opened(fit) -> dict:
    """The fit as a client sees it once the seal is opened (the held-out detail is sealed)."""
    from turbotab.core.seal import (SEALED_DETAIL, SEALED_SCORES, details_by_family,
                                    scores_by_family, serve_fit)

    assert all(m["holdout_detail"] is None for m in fit.data["models"])  # never in the public data
    return serve_fit(fit.data, opened=True,
                     scores=lambda: scores_by_family(fit.frames[SEALED_SCORES]),
                     details=lambda: details_by_family(fit.frames[SEALED_DETAIL]))


# ── 1 · calibration ──────────────────────────────────────────────────────────


def test_1a_calibration_intercept_and_slope_equal_statsmodels():
    """Calibration-in-the-large and the calibration slope (Van Calster et al., J Clin Epidemiol
    2016;74:167) are logistic regressions of the outcome on the linear predictor lp = logit(p):
    ``logit P(y) = a + lp`` (offset) and ``logit P(y) = a + b·lp``. Reference: statsmodels'
    ``GLM(Binomial)`` with ``offset``, and its cluster-robust covariance (``use_correction=False``,
    times the G/(G − 1) factor the engine states). For a numeric outcome the intercept is the mean
    of y − ŷ with the one-sample standard error, and the slope is OLS of y on ŷ (statsmodels).
    """
    import statsmodels.api as sm

    X, y, risk = _logistic_table(800, seed=1)
    p = np.clip(risk * 0.8 + 0.05, 1e-6, 1 - 1e-6)  # deliberately miscalibrated risks
    lp = np.log(p / (1 - p))
    cal = perf.calibration("binary", y, np.column_stack([1 - p, p]), classes=[0, 1])
    a = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    b = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    assert cal.intercept.estimate == pytest.approx(a.params[0], abs=1e-8)
    assert cal.intercept.se == pytest.approx(a.bse[0], rel=1e-6)
    assert cal.slope.estimate == pytest.approx(b.params[1], abs=1e-8)
    assert cal.slope.se == pytest.approx(b.bse[1], rel=1e-6)
    assert (cal.slope.ci_low, cal.slope.ci_high) == pytest.approx(tuple(b.conf_int()[1]), rel=1e-6)

    # Clustered rows: the sandwich over clusters.
    cluster = np.repeat(np.arange(160), 5)
    clustered = perf.calibration("binary", y, np.column_stack([1 - p, p]), classes=[0, 1],
                                 groups=cluster)
    g = 160
    bc = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": cluster, "use_correction": False})
    assert clustered.slope.se == pytest.approx(bc.bse[1] * math.sqrt(g / (g - 1)), rel=1e-6)
    assert clustered.slope.method == "cluster-robust Wald"

    # A numeric outcome.
    rng = np.random.default_rng(2)
    yhat = rng.normal(10, 2, 500)
    yy = 1.0 + 0.8 * yhat + rng.normal(0, 1.5, 500)
    reg = perf.calibration("regression", yy, yhat)
    ols = sm.OLS(yy, sm.add_constant(yhat)).fit()
    one = sm.OLS(yy - yhat, np.ones(500)).fit()
    assert reg.slope.estimate == pytest.approx(ols.params[1], abs=1e-10)
    assert reg.slope.se == pytest.approx(ols.bse[1], rel=1e-8)
    assert reg.intercept.estimate == pytest.approx(one.params[0], abs=1e-10)
    assert reg.intercept.se == pytest.approx(one.bse[0], rel=1e-8)


def test_1b_the_smoothed_curve_is_lowess_with_no_robustness_iterations():
    """The calibration curve is Cleveland's lowess (span 2/3, ``iter = 0``), as rms ``val.prob``
    draws it. Reference: the local regression written out point by point
    (``references_validation.lowess_by_definition``), every point fitted exactly. The engine uses
    R's default ``delta`` (1% of the range), which interpolates between close points, so the two
    agree to 0.002 on the probability scale, not to rounding; the drawn curve is clipped to
    [0, 1]. Eavg (the integrated calibration
    index, Austin & Steyerberg, Stat Med 2019;38:4051) is the mean |p − smooth(p)| and matches the
    reference to the same tolerance."""
    X, y, risk = _logistic_table(1000, seed=3)
    p = np.clip(risk ** 1.3, 1e-6, 1 - 1e-6)  # bent: the curve must follow it
    cal = perf.calibration("binary", y, np.column_stack([1 - p, p]), classes=[0, 1])
    xs = np.array([pt.x for pt in cal.curve])
    got = np.array([pt.y for pt in cal.curve])
    # A drawn curve is a probability, so it is clipped to [0, 1]; a local line can overshoot.
    want = np.clip(ref.lowess_by_definition(p, y.astype(float), xs), 0.0, 1.0)
    assert len(cal.curve) >= 30
    assert np.max(np.abs(got - want)) < 0.002, np.max(np.abs(got - want))
    every = ref.lowess_by_definition(p, y.astype(float), p)
    assert cal.eavg == pytest.approx(float(np.mean(np.abs(p - every))), abs=0.002)


def test_1c_a_well_calibrated_logistic_model_has_slope_1_and_boosted_trees_are_flagged():
    """20 datasets of 2,000 rows from a known logistic model (true risks perfectly calibrated),
    5-fold cross-validation with the app's own folds and pipelines.

    Logistic regression (correctly specified): the out-of-fold calibration slope averages within
    1 ± 0.1 (it estimates 1 minus the small overfitting of 6 parameters on 1,600 rows: about 0.98),
    its 95% interval covers 1 in at least 17 of 20 datasets (expected 19; P(≤ 16) ≈ 2% for 95%
    intervals), and no dataset is flagged. Boosted trees (scikit-learn's histogram gradient
    boosting, the shelf's defaults) on the same data: flagged as miscalibrated in every dataset —
    slope far below 1, predictions too extreme. Control: the slope of the data-generating risks
    themselves averages within 1 ± 0.04 over the same datasets (MC SE ≈ 0.012), so the bound
    measures the fit, not the data; the true risks are never flagged either (a flag needs the whole
    interval beyond the tolerance, ``performance.py``; the rule it replaced fired on true risks in
    10% of samples of this size).
    """
    slopes, covered, flagged_linear, flagged_trees, tree_slopes, truths = [], 0, 0, 0, [], []
    for r in range(20):
        X, y, risk = _logistic_table(2000, seed=100 + r)
        truth = perf.calibration("binary", y, np.column_stack([1 - risk, risk]), classes=[0, 1])
        truths.append(truth.slope.estimate)
        assert not truth.flagged
        folds = _folds(2000, seed=r, y=y)
        for family in ("linear", "boosted_trees"):
            result = _cv("binary", family, X, y, folds)
            rows, yo, po = result.out_of_fold(0)
            assert (rows == np.arange(2000)).all()
            cal = perf.calibration("binary", yo, po, classes=[0, 1])
            if family == "linear":
                slopes.append(cal.slope.estimate)
                covered += cal.slope.ci_low <= 1 <= cal.slope.ci_high
                flagged_linear += cal.flagged
            else:
                tree_slopes.append(cal.slope.estimate)
                flagged_trees += cal.flagged
                assert cal.flagged and "too extreme" in cal.concern, cal.concern
    assert abs(np.mean(truths) - 1) < 0.04, np.mean(truths)
    mean = float(np.mean(slopes))
    assert abs(mean - 1) <= 0.1, f"mean slope {mean:.3f} (MC SE {np.std(slopes) / math.sqrt(20):.3f})"
    assert covered >= 17 and flagged_linear == 0
    assert flagged_trees == 20 and max(tree_slopes) < 0.7, tree_slopes


def test_1d_the_fit_reports_out_of_fold_and_held_out_calibration(tmp_path):
    """The real stages on a 2,000-row table drawn from a known logistic model, with a 20% holdout:
    every family carries its out-of-fold calibration (intercept, slope, curve) in the public
    artifact; the held-out intervals and calibration stay sealed with the held-out scores and are
    served once the seal is opened; each equals ``performance.calibration`` on the refit model's
    held-out predictions, recomputed here from the parquet with the refit pipeline. The boosted
    trees' flagged calibration is one of its concerns, worded with its slope."""
    X, y, _ = _logistic_table(2000, seed=7)
    frame = X.assign(id=np.arange(2000), event=np.where(y == 1, "yes", "no"))
    source = tmp_path / "logistic.csv"
    frame.to_csv(source, index=False)
    table = Ingested(source, tmp_path / "ingested")
    st = ProjectState(lens=["clinical"], target="event", task="binary", event="yes",
                      purpose="prediction", roles={"id": "identifier", **{c: "covariate" for c in X}},
                      missing="complete_case", grain=GrainSpec(grain="one_row_per_unit", id_column="id"),
                      split=SplitSpec(holdout=0.2, seed=4, folds=5),
                      models=["linear", "boosted_trees"])
    _, split, design, fit = _stages(table, st)
    models = {m["family"]: m for m in fit.data["models"]}
    for m in models.values():
        cal = m["calibration"]
        assert cal is not None and len(cal["curve"]) >= 30
        assert cal["n"] == fit.data["n_train"]
    assert abs(models["linear"]["calibration"]["slope"]["estimate"] - 1) < 0.1
    assert not models["linear"]["calibration"]["flagged"]
    trees = models["boosted_trees"]
    assert trees["calibration"]["flagged"]
    assert any("calibration slope" in c and "too extreme" in c for c in trees["concerns"]), trees["concerns"]

    served = {m["family"]: m for m in _opened(fit)["models"]}
    a = split.frames["assignment"]
    hold = a.loc[a["partition"] == "holdout", "row_id"].to_numpy()
    raw = pd.read_parquet(table.parquet).set_index("__row_id").loc[hold]
    yh = (frame.loc[hold, "event"] == "yes").astype(int).to_numpy()  # the CSV's own rows
    assert 50 < yh.sum() < len(yh) - 50
    for family, m in served.items():
        detail = m["holdout_detail"]
        refit = fit.objects["fitted"][family]
        proba = refit.predict_proba(raw[design.objects["spec"]["inputs"]])
        want = perf.calibration("binary", yh, proba, classes=[0, 1], where="on the held-out rows")
        assert detail["calibration"]["slope"]["estimate"] == pytest.approx(want.slope.estimate, abs=1e-9)
        assert detail["calibration"]["intercept"]["estimate"] == pytest.approx(want.intercept.estimate, abs=1e-9)
        auc, var = ref.delong_by_definition(yh == 1, proba[:, 1])
        assert detail["intervals"]["auc"]["estimate"] == pytest.approx(auc, abs=1e-12)
        assert detail["intervals"]["auc"]["se"] == pytest.approx(math.sqrt(var), rel=1e-9)
        assert m["holdout"]["auc"] == pytest.approx(auc, abs=1e-12)


# ── 2 · intervals ────────────────────────────────────────────────────────────


def test_2a_the_holdout_auc_interval_is_delongs_and_matches_proc():
    """Published reference: pROC on its own ``aSAH`` data (``asah.py``), ``roc(aSAH$outcome,
    aSAH$s100b, ci = TRUE)`` prints "Area under the curve: 0.7314" and "95% CI: 0.6301-0.8326
    (DeLong)" (pROC's documented example, reproduced in Y. Ayue, *R clinical model*, ch. 21,
    ayueme.github.io/R_clinical_model/roc-binominal.html). The engine's interval matches to the four
    decimals pROC prints (well inside 10⁻³). Second reference: DeLong's variance written from its
    definition with the m × n comparison matrix (``references_validation.delong_by_definition``),
    on every aSAH marker and on simulated scores with heavy ties, to 10⁻¹²; and with clustered rows,
    Obuchowski's (1997) estimator written as a loop over clusters, to 10⁻¹².
    """
    data = asah()
    poor = (data["outcome"] == "Poor").to_numpy()
    s100b = perf.auc_interval(poor, data["s100b"].to_numpy(float))
    assert s100b.method == "DeLong"
    assert round(s100b.estimate, 4) == 0.7314
    assert (round(s100b.ci_low, 4), round(s100b.ci_high, 4)) == (0.6301, 0.8326)
    for marker in ("s100b", "ndka", "wfns", "age"):
        got = perf.auc_interval(poor, data[marker].to_numpy(float))
        auc, var = ref.delong_by_definition(poor, data[marker].to_numpy(float))
        assert got.estimate == pytest.approx(auc, abs=1e-12)
        assert got.se == pytest.approx(math.sqrt(var), rel=1e-10)
        assert got.estimate == pytest.approx(roc_auc_score(poor, data[marker]), abs=1e-12)

    rng = np.random.default_rng(5)
    score = rng.integers(0, 6, 400).astype(float)  # six values: most pairs tie
    positive = rng.random(400) < 0.2 + 0.1 * score
    got = perf.auc_interval(positive, score)
    auc, var = ref.delong_by_definition(positive, score)
    assert got.estimate == pytest.approx(auc, abs=1e-12) and got.se == pytest.approx(math.sqrt(var), rel=1e-10)
    cluster = np.repeat(np.arange(80), 5)
    clustered = perf.auc_interval(positive, score, groups=cluster, unit="person")
    auc_c, var_c = ref.obuchowski_clustered(positive, score, cluster)
    assert clustered.estimate == pytest.approx(auc_c, abs=1e-12)
    assert clustered.se == pytest.approx(math.sqrt(var_c), rel=1e-10)
    assert clustered.method == "DeLong, clustered by person"


def test_2b_cross_validated_scores_carry_ledells_standard_error():
    """Formula: the SE of a 5-fold AUC is LeDell, Petersen & van der Laan's (Electron J Stat
    2015;9:1583) ``√(Σ_v Var_v)/V`` with each fold's DeLong variance, recomputed here from
    scikit-learn's per-fold AUC and the definition-based DeLong variance; the pooled R²'s SE is the
    delta method, recomputed with the covariance matrix of (e², d²) from ``numpy.cov`` and the
    gradient (−1/B, A/B²) (Hawinkel et al. 2024) — a different route from the engine's
    influence-value sums.

    Simulation (the bound): 300 datasets of 1,000 rows from a known logistic model, 5-fold. The
    target is LeDell's: the mean of the five fold models' true AUCs, each measured on 50,000 fresh
    rows. The 95% interval covers it in 92–99% of datasets (MC SE ≈ 0.01; this setup measures
    about 0.97), and the mean SE is within 15% of the SD of (estimate − target). LeDell et al.'s
    Table 1, in their own simulation: "For a relatively small sample size (e.g. n = 1, 000), the
    coverage probability of the confidence intervals are slightly lower (92–93%) than specified
    (95%). However, when n ≥ 5, 000, we have coverage between 94–95%."
    """
    X, y, _ = _logistic_table(600, seed=11)
    folds = _folds(600, seed=11, y=y)
    result = _cv("binary", "linear", X, y, folds)
    summary = result.summary("binary")
    variances = []
    for k in range(5):
        f = result.predictions[k]
        assert f.y.tolist() == y[folds == k].tolist()
        variances.append(ref.delong_by_definition(f.y == 1, f.prediction[:, 1])[1])
        assert result.per_fold[k]["auc"] == pytest.approx(roc_auc_score(f.y, f.prediction[:, 1]), abs=1e-12)
    assert summary["auc"]["se"] == pytest.approx(math.sqrt(sum(variances)) / 5, rel=1e-9)
    z = 1.959963984540054
    assert summary["auc"]["ci_low"] == pytest.approx(summary["auc"]["estimate"] - z * summary["auc"]["se"], abs=1e-12)

    rng = np.random.default_rng(12)
    Xr = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    yr = Xr.to_numpy() @ np.array([1.0, 0.5, 0.0, 0.0]) + rng.normal(size=300)
    fr = _folds(300, seed=12)
    reg = _cv("regression", "linear", Xr, yr, fr)
    e2 = np.concatenate([(f.y - f.prediction) ** 2 for f in reg.predictions])
    d2 = np.concatenate([(f.y - f.reference) ** 2 for f in reg.predictions])
    A, B = e2.mean(), d2.mean()
    grad = np.array([-1 / B, A / B ** 2])
    var = grad @ np.cov(np.vstack([e2, d2]), ddof=1) @ grad / len(e2)
    s = reg.summary("regression")["r2"]
    assert s["estimate"] == pytest.approx(1 - A / B, abs=1e-12)
    assert s["se"] == pytest.approx(math.sqrt(var), rel=1e-9)

    big_X, big_y, _ = _logistic_table(50_000, seed=999)
    cover, gaps, ses = 0, [], []
    for r in range(300):
        Xs, ys, _ = _logistic_table(1000, seed=2000 + r)
        fitted = []
        pipe = _pipeline("linear", "binary", list(Xs.columns), 1000)

        def fit(m, Xf, yf, rows, _fitted=fitted):
            model = fit_pipeline(m, Xf, yf)
            _fitted.append(model)
            return model

        res = cross_validate("binary", lambda: clone(pipe), Xs, ys, fold_pairs(_folds(1000, r, ys)),
                             fit=fit, keep_predictions=True)
        auc = res.summary("binary")["auc"]
        target = float(np.mean([roc_auc_score(big_y, m.predict_proba(big_X)[:, 1]) for m in fitted]))
        cover += auc["ci_low"] <= target <= auc["ci_high"]
        gaps.append(auc["estimate"] - target)
        ses.append(auc["se"])
    rate = cover / 300
    assert 0.92 <= rate <= 0.99, f"coverage {rate:.3f} (MC SE {math.sqrt(rate * (1 - rate) / 300):.3f})"
    assert 0.85 <= np.mean(ses) / np.std(gaps) <= 1.15, (np.mean(ses), np.std(gaps))


def test_2c_a_held_out_r2_standard_error_matches_its_sampling_spread():
    """One fitted model (fixed), scored on 2,000 independent held-out samples of 200 rows each:
    the mean of the engine's R² standard errors is within 10% of the SD of the held-out R² across
    samples (the delta-method SE's target), and the 95% interval covers the model's R² on 200,000
    fresh rows in 93–97% of samples (MC SE ≈ 0.005)."""
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(21)
    beta = np.array([0.6, 0.3, 0.0])

    def draw(n):
        Xd = rng.normal(size=(n, 3))
        return Xd, Xd @ beta + rng.normal(size=n)

    X0, y0 = draw(300)
    model = LinearRegression().fit(X0, y0)
    ref_mean = float(y0.mean())
    Xb, yb = draw(200_000)
    target = 1 - np.mean((yb - model.predict(Xb)) ** 2) / np.mean((yb - ref_mean) ** 2)
    r2s, ses, cover = [], [], 0
    for _ in range(2000):
        Xh, yh = draw(200)
        got = perf.score_intervals("regression", yh, model.predict(Xh), reference=ref_mean)["r2"]
        r2s.append(got.estimate)
        ses.append(got.se)
        cover += got.ci_low <= target <= got.ci_high
    assert abs(np.mean(ses) / np.std(r2s) - 1) < 0.10, (np.mean(ses), np.std(r2s))
    assert 0.93 <= cover / 2000 <= 0.97, cover / 2000


# ── 3 · resampling ───────────────────────────────────────────────────────────


def test_3a_bootstrap_and_repeated_kfold_are_offered_and_lead_for_prediction_below_20000():
    """Under prediction, below the stated n (Harrell's 20,000), bootstrap optimism correction
    ranks first and repeated k-fold second, and cross-validation alone leads the holdout options
    with the holdout's tension in one line; at or above it, one k-fold run leads and
    internal–external validation follows; under inference one k-fold run leads. Every option stays
    on offer, and the split decision records each.

    Source check — Harrell, "Split-Sample Model Validation" (fharrell.com/post/split-val, 23
    January 2017): "Data splitting is an unstable method for validating models or classifiers,
    especially when the number of subjects is less than about 20,000 (fewer if signal:noise ratio
    is high)." Steyerberg et al., J Clin Epidemiol 2001;54:774 (PubMed abstract): "Internal
    validity could best be estimated with bootstrapping, which provided stable estimates with low
    bias. We conclude that split-sample validation is inefficient, and recommend bootstrapping for
    estimation of internal validity of a predictive logistic regression model."
    """
    assert RESAMPLE_BELOW == 20_000
    small = validation_plan("prediction", 2_000)
    assert [o.validation for o in small.options] == ["bootstrap", "repeated_kfold", "kfold",
                                                      "internal_external"]
    assert small.resampling_first and "lockbox against analyst overfitting" in small.holdout_note
    large = validation_plan("prediction", 25_000)
    assert [o.validation for o in large.options][:2] == ["kfold", "internal_external"]
    assert not large.resampling_first and large.holdout_note is None
    assert validation_plan("inference", 2_000).options[0].validation == "kfold"
    timed = validation_plan("prediction", 2_000, time_ordered=True)
    assert timed.options[0].validation == "kfold" and timed.options[-1].validation == "bootstrap"
    assert {o.validation for o in small.options} == set(d.Validation.__args__)

    for kind, extra in (("bootstrap", {"n_boot": 200}), ("repeated_kfold", {"repeats": 10}),
                        ("internal_external", {"cluster": "site"}), ("kfold", {})):
        assert d.SetSplit(holdout=0.0, validation=kind, **extra).validation == kind
    with pytest.raises(ValueError):
        d.SetSplit(holdout=0.0, validation="internal_external")  # which column?
    with pytest.raises(ValueError):
        d.SetSplit(holdout=0.0, validation="kfold", cluster="site")


def test_3b_the_seal_plan_orders_the_split_question_by_purpose_and_size(tmp_path):
    """The real seal-plan stage on a 1,000-row table, where a 20% holdout clears the floor of 100
    rows: under prediction the plan's validation options lead with the bootstrap, and
    "Cross-validation only" leads the holdout options (it trailed them before WP9, ME-11) with
    the reason and the holdout's tension.

    Under inference "no holdout" leads too, with one k-fold run first among the validations.
    (Rewritten in the repair round: this test once pinned the 20% holdout first under inference,
    the opposite of BLUEPRINT §12 ruling 3, "a holdout is a prediction concept", and of ME-12's
    recommendation, "lead the split with 'no holdout'". Source check, Shmueli 2010, Statistical
    Science 25:289, as the audit quotes it: "In explanatory modeling, data partitioning is less
    common because of the reduction in statistical power.") The holdout stays on offer."""
    rng = np.random.default_rng(8)
    X = pd.DataFrame(rng.normal(size=(1000, 3)), columns=["x0", "x1", "x2"])
    frame = X.assign(id=np.arange(1000), y=X["x0"] + rng.normal(size=1000))
    source = tmp_path / "t.csv"
    frame.to_csv(source, index=False)
    table = Ingested(source, tmp_path / "i")

    def plan(purpose):
        st = ProjectState(lens=["clinical"], target="y", task="regression", purpose=purpose,
                          roles={"id": "identifier", **{c: "covariate" for c in X}},
                          grain=GrainSpec(grain="one_row_per_unit", id_column="id"))
        info = table.run(target_info_stage, st)
        cohort = table.run(cohort_stage, st, {"target_info": info})
        return table.run(seal_plan_stage, st, {"cohort": cohort, "target_info": info})

    pred = plan("prediction")
    assert pred["validation"]["options"][0]["validation"] == "bootstrap"
    assert pred["cv_first"] and pred["options"][0]["holdout"] == 0
    assert "Harrell" in pred["reason"] and "lockbox" in pred["reason"]
    inf = plan("inference")
    assert inf["validation"]["options"][0]["validation"] == "kfold"
    assert not inf["validation"]["resampling_first"]
    assert "Harrell" not in inf["reason"]
    assert inf["cv_first"] and inf["options"][0]["holdout"] == 0
    assert {o["holdout"] for o in inf["options"]} == {o["holdout"] for o in pred["options"]}
    assert inf["reason"].startswith("Under inference every analyzed row estimates the coefficients")


def test_3c_repeated_kfold_averages_its_repeats_each_scored_as_scikit_learn_scores_it(tmp_path):
    """The real split and fit stages with ``validation = repeated_kfold`` (3 × 5): the split draws
    three stratified fold columns (``fold``, ``fold_r1``, ``fold_r2``), each a partition of the
    training rows into five folds with the outcome's classes in proportion; the fit's CV AUC is the
    mean over repeats of each repeat's mean fold AUC, recomputed with statsmodels ``Logit``
    (maximum likelihood) and scikit-learn's ``roc_auc_score`` on each column; its SE is the root
    mean square of the repeats' LeDell SEs; and the comparison with the class prior uses all 15
    paired folds, Bouckaert & Frank's corrected repeated k-fold t with 14 df."""
    import statsmodels.api as sm

    X, y, _ = _logistic_table(500, seed=9)
    frame = X.assign(id=np.arange(500), event=np.where(y == 1, "yes", "no"))
    source = tmp_path / "t.csv"
    frame.to_csv(source, index=False)
    table = Ingested(source, tmp_path / "i")
    st = ProjectState(lens=["clinical"], target="event", task="binary", event="yes", purpose="prediction",
                      roles={"id": "identifier", **{c: "covariate" for c in X}}, missing="complete_case",
                      grain=GrainSpec(grain="one_row_per_unit", id_column="id"), models=["linear"],
                      split=SplitSpec(holdout=0.0, seed=2, folds=5, validation="repeated_kfold", repeats=3))
    _, split, _, fit = _stages(table, st)
    a = split.frames["assignment"].sort_values("row_id")
    assert split.data["repeats"] == 3 and fit.data["repeats"] == 3
    assert fit.data["validation"] == "repeated_kfold"
    columns = ["fold", "fold_r1", "fold_r2"]
    assert len({tuple(a[c]) for c in columns}) == 3  # three different draws
    yy = (frame.loc[a["row_id"], "event"] == "yes").to_numpy().astype(int)
    Xs = frame.loc[a["row_id"], list(X.columns)].to_numpy()
    repeat_means, repeat_vars = [], []
    for c in columns:
        f = a[c].to_numpy()
        assert sorted(np.unique(f)) == [0, 1, 2, 3, 4]
        shares = [yy[f == k].mean() for k in range(5)]
        assert max(shares) - min(shares) < 0.03  # stratified
        aucs, variances = [], []
        for k in range(5):
            beta = sm.Logit(yy[f != k], sm.add_constant(Xs[f != k])).fit(
                disp=0, method="newton", tol=1e-12, maxiter=200).params
            p = 1 / (1 + np.exp(-(sm.add_constant(Xs[f == k]) @ beta)))
            aucs.append(roc_auc_score(yy[f == k], p))
            variances.append(ref.delong_by_definition(yy[f == k] == 1, p)[1])
        repeat_means.append(np.mean(aucs))
        repeat_vars.append(sum(variances) / 25)
    cv = fit.data["models"][0]["cv"]["auc"]
    assert cv["repeats"] == 3 and len(cv["folds"]) == 15
    assert cv["estimate"] == pytest.approx(np.mean(repeat_means), abs=1e-8)
    assert cv["repeat_sd"] == pytest.approx(np.std(repeat_means, ddof=1), abs=1e-8)
    assert cv["se"] == pytest.approx(math.sqrt(np.mean(repeat_vars)), rel=1e-5)
    assert fit.data["models"][0]["versus_baseline"]["df"] == 14
    assert "drawn `3` times" in split.data["note"]


def test_3d_the_optimism_corrected_auc_matches_an_independent_bootstrap_on_asah(tmp_path):
    """Harrell's bootstrap optimism correction (Harrell, Lee & Mark, Stat Med 1996;15:361;
    Steyerberg 2001), the whole pipeline refit on each of B = 200 resamples, through the real
    split and fit stages on pROC's ``aSAH`` data (113 patients; logistic regression of a poor
    outcome on age, gender, WFNS grade, S100B and NDKA).

    Reference: an independent implementation — statsmodels ``Logit`` (maximum likelihood, raw
    predictors and a gender dummy: no standardization, no scikit-learn model) and scikit-learn's
    ``roc_auc_score``, in a loop written here, replaying the resamples the engine documents
    (``numpy.random.default_rng(seed).integers(0, n, n)`` per resample, in turn). The apparent AUC,
    the mean optimism and the corrected AUC match to 10⁻⁶ (the two optimizers' convergence)."""
    import statsmodels.api as sm

    data = asah()
    source = tmp_path / "asah.csv"
    data.assign(id=np.arange(len(data))).to_csv(source, index=False)
    table = Ingested(source, tmp_path / "i")
    predictors = ["age", "gender", "wfns", "s100b", "ndka"]
    st = ProjectState(lens=["clinical"], target="outcome", task="binary", event="Poor", purpose="prediction",
                      roles={"id": "identifier", "gos6": "excluded", **{c: "covariate" for c in predictors}},
                      missing="complete_case", grain=GrainSpec(grain="one_row_per_unit", id_column="id"),
                      models=["linear"],
                      split=SplitSpec(holdout=0.0, seed=17, folds=5, validation="bootstrap", n_boot=200),
                      # The readings ledger (BLUEPRINT §14.1): WFNS grade 1–5 is a code or an amount,
                      # and the reference below enters it as one slope, so that is recorded.
                      shape_confirmations={"code_or_count:wfns": "amount"})
    _, split, _, fit = _stages(table, st)
    optimism = fit.data["models"][0]["optimism"]
    assert optimism["n_boot"] == 200 and optimism["n_ok"] == 200 and optimism["resampled"] == "rows"

    rows = split.frames["assignment"].sort_values("row_id")["row_id"].to_numpy()
    d_ = data.loc[rows]
    Xm = sm.add_constant(pd.DataFrame({"age": d_["age"], "male": (d_["gender"] == "Male").astype(float),
                                       "wfns": d_["wfns"], "s100b": d_["s100b"], "ndka": d_["ndka"]}).to_numpy(float))
    yv = (d_["outcome"] == "Poor").to_numpy().astype(int)

    def auc_of(beta, Xe, ye):
        return roc_auc_score(ye, Xe @ beta)

    full = sm.Logit(yv, Xm).fit(disp=0, method="newton", tol=1e-12, maxiter=200).params
    apparent = auc_of(full, Xm, yv)
    rng = np.random.default_rng(17)
    gaps = []
    for _ in range(200):
        draw = rng.integers(0, len(yv), len(yv))
        beta = sm.Logit(yv[draw], Xm[draw]).fit(disp=0, method="newton", tol=1e-12, maxiter=200).params
        gaps.append(auc_of(beta, Xm[draw], yv[draw]) - auc_of(beta, Xm, yv))
    got = optimism["estimates"]["auc"]
    assert got["apparent"] == pytest.approx(apparent, abs=1e-6)
    assert got["optimism"] == pytest.approx(np.mean(gaps), abs=1e-6)
    assert got["corrected"] == pytest.approx(apparent - np.mean(gaps), abs=1e-6)
    assert 0 < got["optimism"] < 0.1  # the corrected AUC sits below the apparent one
    # Calibration is corrected the same way: a logistic fit's apparent slope is exactly 1.
    assert optimism["estimates"]["calibration_slope"]["apparent"] == pytest.approx(1.0, abs=1e-6)
    assert optimism["estimates"]["calibration_slope"]["corrected"] < 1
    text = voice.sentence_for(d.SetSplit(holdout=0.0, seed=17, folds=5, validation="bootstrap", n_boot=200),
                              st, {})
    assert "Harrell's bootstrap (`200` resamples, the whole pipeline refit on each)" in text


def _harrell(make, X: np.ndarray, y: np.ndarray, B: int, rng: np.random.Generator) -> tuple[float, float]:
    """Harrell's bootstrap optimism correction written out with scikit-learn (apparent AUC, then
    the mean of each resample's own AUC less its AUC on the original rows): (apparent, corrected)."""
    full = make().fit(X, y)
    apparent = roc_auc_score(y, full.predict_proba(X)[:, 1])
    gaps = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        m = make().fit(X[i], y[i])
        gaps.append(roc_auc_score(y[i], m.predict_proba(X[i])[:, 1])
                    - roc_auc_score(y, m.predict_proba(X)[:, 1]))
    return apparent, apparent - float(np.mean(gaps))


def test_3e_the_bootstrap_is_not_applied_to_a_near_interpolating_learner(tmp_path):
    """Repair round (verifier's regression): Harrell's bootstrap was applied to every family, and
    for boosted trees served a corrected AUC of 0.912 (apparent 0.9999) against a 5-fold CV of
    0.686.

    **Independent replication** (scikit-learn alone, as the verifier ran it): on the
    ``_logistic_table`` model, n = 1,600, boosted trees' Harrell-corrected AUC (B = 30) sits more
    than 0.1 above its AUC on 20,000 fresh rows, while logistic regression's is within 0.02 of its
    own. Monte Carlo: B = 30 resamples put an SE of about 0.002 on the mean optimism, far inside
    both bounds. Source check — Coley et al., BMC Med Res Methodol 2023;23:33 (PMC9890785):
    "While previous literature demonstrated the validity of bootstrap optimism correction for
    parametric models in small samples, this approach did not accurately validate performance of a
    rare-event prediction model estimated with random forests in a large clinical dataset."

    **The app**: boosted trees declare ``bootstrap_optimism = False``; through the real split and
    fit stages with ``validation = bootstrap``, logistic regression gets its corrected AUC and
    boosted trees get none, with the reason as a concern; the plan's bootstrap option carries its
    caution, and the split sentence says which families it was applied to."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(5)
    X, y, _ = _logistic_table(1_600, seed=21)
    fresh, y_fresh, _ = _logistic_table(20_000, seed=22)
    for make, gap in ((lambda: HistGradientBoostingClassifier(random_state=0), "large"),
                      (lambda: LogisticRegression(C=np.inf, max_iter=5_000), "small")):
        apparent, corrected = _harrell(make, X.to_numpy(), y, 30, rng)
        truth = roc_auc_score(y_fresh, make().fit(X.to_numpy(), y).predict_proba(fresh.to_numpy())[:, 1])
        if gap == "large":
            assert apparent > 0.97 and corrected - truth > 0.1, (apparent, corrected, truth)
        else:
            assert abs(corrected - truth) < 0.02, (apparent, corrected, truth)

    assert get_family("boosted_trees").bootstrap_optimism is False
    assert get_family("linear").bootstrap_optimism is True
    frame = X.iloc[:600].assign(id=np.arange(600), event=np.where(y[:600] == 1, "case", "control"))
    source = tmp_path / "t.csv"
    frame.to_csv(source, index=False)
    table = Ingested(source, tmp_path / "i")
    st = ProjectState(lens=["clinical"], target="event", task="binary", event="case",
                      purpose="prediction", roles={"id": "identifier", **{c: "covariate" for c in X}},
                      missing="complete_case", grain=GrainSpec(grain="one_row_per_unit", id_column="id"),
                      models=["linear", "boosted_trees"],
                      split=SplitSpec(holdout=0.0, seed=3, folds=5, validation="bootstrap", n_boot=20))
    _, _, _, fit = _stages(table, st)
    linear, trees = fit.data["models"]
    assert linear["optimism"]["n_ok"] == 20 and linear["optimism"]["estimates"]["auc"]["corrected"]
    assert trees["optimism"]["estimates"] == {} and trees["optimism"]["refused"]
    assert "Coley et al. 2023" in trees["optimism"]["refused"]
    assert trees["optimism"]["refused"] in trees["concerns"]
    assert trees["cv"]["auc"]["estimate"] is not None  # its cross-validated score stands
    bootstrap = validation_plan("prediction", 600).options[0]
    assert bootstrap.validation == "bootstrap" and "boosted trees" in bootstrap.caution
    said = voice.sentence_for(d.SetSplit(holdout=0.0, seed=3, folds=5, validation="bootstrap",
                                         n_boot=20), st, {})
    assert "except for a family that nearly memorizes its rows (boosted trees)" in said


# ── 4 · the banner and the primary metric ────────────────────────────────────


def test_4_the_ranking_is_named_and_multiclass_ranks_on_log_loss(tmp_path):
    """The primary metric for a multiclass outcome is log loss, a proper scoring rule (Gneiting &
    Raftery, JASA 2007;102:359); macro-F1, the earlier primary, is not. The fit names how families
    are ranked, for the banner to say: "highest AUC" (discrimination only, so never "best"),
    "highest R²", "lowest log loss". Against the class prior, lower log loss is a gain: on data
    with a real signal the verdict is "better" with a positive gain, recomputed here as the prior's
    CV log loss minus the model's (scikit-learn ``log_loss`` per fold)."""
    from sklearn.metrics import log_loss

    # The ordered outcome WP12a added ranks on its concordance, C; these three are WP9's.
    assert {k: PRIMARY[k] for k in ("regression", "binary", "multiclass")} == {
        "regression": "r2", "binary": "auc", "multiclass": "log_loss"}
    rng = np.random.default_rng(31)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    logits = np.column_stack([np.zeros(n), 1.2 * X["a"], -1.0 * X["b"]])
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(["low", "mid", "top"], p=p_) for p_ in proba])
    frame = X.assign(id=np.arange(n), band=y)
    source = tmp_path / "m.csv"
    frame.to_csv(source, index=False)
    table = Ingested(source, tmp_path / "i")
    roles = {"id": "identifier", "a": "covariate", "b": "covariate", "c": "covariate"}
    st = ProjectState(lens=["clinical"], target="band", task="multiclass", purpose="prediction", roles=roles,
                      missing="complete_case", grain=GrainSpec(grain="one_row_per_unit", id_column="id"),
                      split=SplitSpec(holdout=0.0, seed=1, folds=5), models=["linear"])
    _, split, _, fit = _stages(table, st)
    assert fit.data["primary_metric"] == "log_loss" and fit.data["ranking"] == "lowest log loss"
    model = fit.data["models"][0]
    a = split.frames["assignment"].sort_values("row_id")
    prior = []
    for k in range(5):
        train_y = y[a["row_id"].to_numpy()][a["fold"].to_numpy() != k]
        test_y = y[a["row_id"].to_numpy()][a["fold"].to_numpy() == k]
        shares = [np.mean(train_y == c) for c in ["low", "mid", "top"]]
        prior.append(log_loss(test_y, np.tile(shares, (len(test_y), 1)), labels=["low", "mid", "top"]))
    assert model["baseline"]["value"] == pytest.approx(np.mean(prior), abs=1e-9)
    gain = np.mean(prior) - model["cv"]["log_loss"]["estimate"]
    assert model["versus_baseline"]["gain"] == pytest.approx(gain, abs=1e-9) and gain > 0
    assert model["versus_baseline"]["verdict"] == "better"
    assert model["cv"]["log_loss"]["se"] > 0
    from turbotab.core.stages.modeling import ranking_phrase

    assert ranking_phrase("auc") == "highest AUC" and ranking_phrase("r2") == "highest R²"


# ── 5 · claims ───────────────────────────────────────────────────────────────


def test_5_the_spread_is_computed_on_the_users_rows_not_claimed(tmp_path):
    """The unsourced SETTLED sentence "Below about 50 rows, a single 5-fold estimate has a
    standard error large enough that a 0.05 AUC difference is noise" (2–4× too low, audit ME-11)
    is gone from every teaching entry. In its place the fit reports, on the user's own rows, each
    family's standard error (``precision``) and every pair's paired interval of the difference
    (``comparisons``), the latter recomputed here from the per-fold scores with Nadeau & Bengio's
    corrected variance and scipy's t quantile.

    Source check — Steyerberg, "Validation in prediction research: the waste by data splitting",
    J Clin Epidemiol 2018;103:131–133 (PubMed 30063954 abstract): "In large samples, interest
    should shift to assessment of heterogeneity in model performance across settings. In small
    samples, cross-validation and bootstrapping are more efficient approaches. In conclusion,
    random data splitting should be abolished for validation of prediction models."
    """
    from scipy import stats

    from turbotab.core import teaching

    for entry in teaching.entries():
        for sec in entry.drawer.sections if entry.drawer else []:
            assert "about 50 rows" not in sec.body
    split_entry = next(e for e in teaching.entries() if e.key == "split")
    assert any("standard error" in s.body for s in split_entry.drawer.sections)

    X, y, _ = _logistic_table(400, seed=41)
    frame = X.assign(id=np.arange(400), event=np.where(y == 1, "yes", "no"))
    source = tmp_path / "t.csv"
    frame.to_csv(source, index=False)
    table = Ingested(source, tmp_path / "i")
    st = ProjectState(lens=["clinical"], target="event", task="binary", event="yes", purpose="prediction",
                      roles={"id": "identifier", **{c: "covariate" for c in X}}, missing="complete_case",
                      grain=GrainSpec(grain="one_row_per_unit", id_column="id"),
                      split=SplitSpec(holdout=0.0, seed=3, folds=5), models=["linear", "boosted_trees"])
    _, _, _, fit = _stages(table, st)
    models = {m["family"]: m for m in fit.data["models"]}
    for m in models.values():
        assert f"{m['cv']['auc']['se']:.3f} for {m['label']}" in fit.data["precision"]
    (cmp,) = fit.data["comparisons"]
    diffs = (np.array(models[cmp["a"]]["cv"]["auc"]["folds"])
             - np.array(models[cmp["b"]]["cv"]["auc"]["folds"]))
    se = math.sqrt((1 / 5 + 1 / 4) * diffs.var(ddof=1))  # test share 1/(K − 1) for 5 equal folds
    half = stats.t.ppf(0.975, 4) * se
    gain = models[cmp["a"]]["cv"]["auc"]["estimate"] - models[cmp["b"]]["cv"]["auc"]["estimate"]
    assert cmp["difference"] == pytest.approx(gain, abs=1e-12)
    assert (cmp["ci_low"], cmp["ci_high"]) == pytest.approx((gain - half, gain + half), rel=0.02)
    assert "corrected for shared training rows" in cmp["sentence"]


# ── 6 · internal–external validation ────────────────────────────────────────


def test_6_internal_external_validation_reports_each_cluster_and_the_spread(tmp_path):
    """Internal–external validation (Collins et al., BMJ 2024;384:e074819; Steyerberg 2018:
    "assessment of heterogeneity in model performance across settings") through the real split
    and fit stages: 8 sites, each held out in turn and scored by a model fit on the other seven.

    References: each site's R² (against the other sites' mean) from scikit-learn's
    ``LinearRegression`` fit on the other sites, to 10⁻⁹; each site's SE from the delta method
    written with ``numpy.cov``; the random-effects summary and τ² from statsmodels'
    ``combine_effects(method_re="dl")``; the prediction interval for a new site from Higgins,
    Thompson & Spiegelhalter (2009), written out (``references_validation``)."""
    from sklearn.linear_model import LinearRegression
    from statsmodels.stats.meta_analysis import combine_effects

    rng = np.random.default_rng(51)
    sites = np.repeat([f"site_{i}" for i in range(8)], 80)
    shift = dict(zip(np.unique(sites), rng.normal(0, 0.6, 8)))
    slope = dict(zip(np.unique(sites), rng.normal(1.0, 0.4, 8)))
    X = pd.DataFrame(rng.normal(size=(640, 3)), columns=["x0", "x1", "x2"])
    yv = (np.array([slope[s] for s in sites]) * X["x0"] + 0.5 * X["x1"]
          + np.array([shift[s] for s in sites]) + rng.normal(size=640))
    frame = X.assign(id=np.arange(640), site=sites, y=yv)
    source = tmp_path / "sites.csv"
    frame.to_csv(source, index=False)
    table = Ingested(source, tmp_path / "i")
    st = ProjectState(lens=["clinical"], target="y", task="regression", purpose="prediction",
                      roles={"id": "identifier", "site": "design", "x0": "covariate", "x1": "covariate",
                             "x2": "covariate"},
                      missing="complete_case", grain=GrainSpec(grain="one_row_per_unit", id_column="id"),
                      split=SplitSpec(holdout=0.0, seed=1, folds=5, validation="internal_external",
                                      cluster="site"),
                      models=["linear"])
    _, split, _, fit = _stages(table, st)
    assert split.data["folds"] == 8 and split.data["fold_labels"] == [f"site_{i}" for i in range(8)]
    assert "one per level of `site`" in split.data["note"]
    iecv = fit.data["models"][0]["internal_external"]
    assert iecv["cluster"] == "site" and iecv["metric"] == "r2" and len(iecv["clusters"]) == 8
    estimates, variances = [], []
    for row in iecv["clusters"]:
        held = sites == row["cluster"]
        model = LinearRegression().fit(X[~held], yv[~held])
        e2 = (yv[held] - model.predict(X[held])) ** 2
        d2 = (yv[held] - yv[~held].mean()) ** 2
        A, B = e2.mean(), d2.mean()
        assert row["n"] == 80
        assert row["primary"]["estimate"] == pytest.approx(1 - A / B, abs=1e-9)
        grad = np.array([-1 / B, A / B ** 2])
        se = math.sqrt(grad @ np.cov(np.vstack([e2, d2]), ddof=1) @ grad / 80)
        assert row["primary"]["se"] == pytest.approx(se, rel=1e-6)
        estimates.append(1 - A / B)
        variances.append(se ** 2)
    meta = combine_effects(np.array(estimates), np.array(variances), method_re="dl")
    pooled = iecv["pooled"]
    assert pooled["k"] == 8 and pooled["scale"] == "identity"
    assert pooled["estimate"] == pytest.approx(meta.mean_effect_re, abs=1e-9)
    assert pooled["tau"] == pytest.approx(math.sqrt(max(meta.tau2, 0.0)), abs=1e-9)
    assert pooled["tau"] > 0  # the sites differ, by construction
    _, _, lo, hi = ref.dersimonian_laird_prediction_interval(np.array(estimates), np.array(variances))
    assert (pooled["pi_low"], pooled["pi_high"]) == pytest.approx((lo, hi), abs=1e-9)
    low, high = min(estimates), max(estimates)
    assert f"ranged from {low:.3f}".replace("-", "−") in iecv["spread"]
    assert f"to {high:.3f}".replace("-", "−") in iecv["spread"]
    assert "95% prediction interval" in iecv["spread"]


def test_6b_the_auc_is_pooled_across_clusters_on_the_logit_scale():
    """Snell et al. (Stat Methods Med Res 2018;27:3505) recommend pooling the C-statistic on the
    logit scale. Reference: statsmodels ``combine_effects`` on logit(AUC) with variance
    se²/(AUC(1 − AUC))², back-transformed."""
    from statsmodels.stats.meta_analysis import combine_effects

    aucs = np.array([0.71, 0.78, 0.66, 0.83, 0.74, 0.69])
    ses = np.array([0.03, 0.04, 0.05, 0.03, 0.02, 0.04])
    got = perf.random_effects("auc", aucs, ses, scale="logit")
    th = np.log(aucs / (1 - aucs))
    v = (ses / (aucs * (1 - aucs))) ** 2
    meta = combine_effects(th, v, method_re="dl")
    expit = lambda t: 1 / (1 + math.exp(-t))  # noqa: E731
    assert got.estimate == pytest.approx(expit(meta.mean_effect_re), abs=1e-10)
    se_re = math.sqrt(meta.var_eff_w_re)
    assert got.ci_low == pytest.approx(expit(meta.mean_effect_re - 1.959963984540054 * se_re), abs=1e-9)
    assert got.tau == pytest.approx(math.sqrt(meta.tau2), abs=1e-10)


# ── E15 · the resampling tension is shown ────────────────────────────────────


def test_e15_a_binary_fit_states_that_nothing_was_resampled_and_why():
    """BLUEPRINT north star 5 names "SMOTE versus calibration-preserving weighting" as a tension
    to show. Source check — van den Goorbergh et al., JAMIA 2022;29:1525 (PubMed 35686364
    abstract): "The use of random undersampling, random oversampling, or SMOTE yielded poorly
    calibrated models: the probability to belong to the minority class was strongly
    overestimated. These methods did not result in higher areas under the ROC curve"."""
    from turbotab.core.stages.modeling import imbalance_sentence

    text = imbalance_sentence("binary", np.array([1] * 12 + [0] * 88), "case")
    assert text.startswith("The event, `case`, is 12% of the training rows.")
    assert "SMOTE" in text and "van den Goorbergh" in text
    assert imbalance_sentence("regression", np.arange(5.0), None) is None
