"""EXPLORE 4 · Intended use → the decision curve, subgroups, model updating, internal–external CV and
the imbalance correction (MODELING_SEQUENCE §1 rows 2 and 11, prediction; TRIPOD+AI 12e, 12f, 13, 14,
15, 23a, 24).

Every expected value comes from an independent path: R's ``dcurves`` (net benefit), ``pROC``
(DeLong's interval), ``stats::glm`` and ``lm`` (the shrunk model's intercept), statsmodels' logistic
maximum likelihood (each left-out cluster's refit), the definitions written out by hand
(``explore_references.py``), or a simulation with a known truth (the imbalance correction's
calibration). R is a reference only: those tests skip cleanly where ``Rscript`` is not installed.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core import decisions as d
from turbotab.core.decisions import Refusal
from turbotab.core.models import decision_curve as DC
from turbotab.core.tests.acceptance import explore_references as ref
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r


def _risk_data(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lp = rng.normal(-0.8, 1.2, n)
    risk = 1 / (1 + np.exp(-lp))
    event = (rng.random(n) < 1 / (1 + np.exp(-(lp + rng.normal(0, 0.5, n))))).astype(int)
    return event, risk


# ── net benefit against R dcurves ────────────────────────────────────────────


@needs_r
def test_4_net_benefit_agrees_with_r_dcurves_to_1e_8(tmp_path):
    """Vickers & Elkin (2006): NB = TP/n − FP/n · t/(1 − t), a row positive when its risk is at
    least t; treat all: π − (1 − π)·t/(1 − t). Reference: R ``dcurves::dca`` 0.5 (its
    ``.calculate_test_consequences`` counts ``risk >= threshold``) on the same rows and thresholds,
    and the definition by hand."""
    event, risk = _risk_data()
    thresholds = DC.grid(0.05, 0.50)
    got = DC.net_benefit(event, risk, thresholds)
    r = run_r("""
        library(dcurves)
        d <- read.csv(dc_csv)
        res <- dca(event ~ risk, data = d, thresholds = seq(0.05, 0.50, by = 0.01))$dca
        out(list(t = res$threshold[res$variable == "risk"],
                 nb = res$net_benefit[res$variable == "risk"],
                 all = res$net_benefit[res$variable == "all"]))
    """, {"dc": pd.DataFrame({"event": event, "risk": risk})}, tmp_path)
    assert np.allclose(r["t"], thresholds, atol=1e-12)
    assert np.max(np.abs(got["model"] - np.asarray(r["nb"]))) < 1e-8
    assert np.max(np.abs(got["all"] - np.asarray(r["all"]))) < 1e-8
    hand = [ref.net_benefit(event, risk, t) for t in thresholds]
    assert np.max(np.abs(got["model"] - hand)) < 1e-12
    rows = DC.decision_curve(event, {"model": risk}, thresholds)
    assert rows[0]["treat_none"] == 0.0 and abs(rows[3]["models"]["model"] - hand[3]) < 1e-12


# ── the threshold, declared or chosen in-fold ────────────────────────────────


def _folds(seed: int = 1, k: int = 5):
    event, risk = _risk_data(500, seed)
    rng = np.random.default_rng(seed + 10)
    fold = rng.permutation(np.arange(len(event)) % k)
    # each fold's model's predictions: its own training rows (apparent) and the held-out fold
    noise = np.random.default_rng(seed + 20).normal(0, 0.02, (k, len(event)))
    out = []
    for f in range(k):
        train, test = fold != f, fold == f
        r = np.clip(risk + noise[f], 1e-6, 1 - 1e-6)
        out.append({"train_event": event[train], "train_risk": r[train],
                    "test_event": event[test], "test_risk": r[test]})
    return out


def test_4_the_threshold_is_chosen_on_each_training_fold_by_youden_and_scored_on_the_held_out_fold():
    """TRIPOD+AI 15: "how the thresholds were identified". With no declared threshold, Youden's J
    picks it within the declared range on each outer fold's own training rows; sensitivity,
    specificity and net benefit are that threshold's on the held-out fold, pooled by rows.
    Reference: the definition by hand (``explore_references.youden``); the in-fold property: changing
    a fold's held-out outcomes never moves its threshold."""
    folds = _folds()
    grid = DC.grid(0.05, 0.50)
    got = DC.threshold_in_fold(folds, grid)
    for f, row in zip(folds, got["folds"]):
        t = ref.youden(f["train_event"], f["train_risk"], grid)
        assert row["threshold"] == t
        assert abs(row["net_benefit"] - ref.net_benefit(f["test_event"], f["test_risk"], t)) < 1e-12
        pos = f["test_risk"] >= t
        assert abs(row["sensitivity"] - pos[f["test_event"] == 1].mean()) < 1e-12
        assert abs(row["specificity"] - (~pos[f["test_event"] == 0]).mean()) < 1e-12
    n = np.asarray([len(f["test_event"]) for f in folds], dtype=float)
    assert abs(got["net_benefit"] - float(np.dot(n, [r["net_benefit"] for r in got["folds"]]) / n.sum())) < 1e-12
    flipped = [dict(f, test_event=1 - f["test_event"]) for f in folds]
    assert DC.threshold_in_fold(flipped, grid)["thresholds"] == got["thresholds"]
    declared = DC.threshold_in_fold(folds, grid, declared=0.2)
    assert declared["declared"] and set(declared["thresholds"]) == {0.2}


# ── subgroup performance ─────────────────────────────────────────────────────


@needs_r
def test_4_subgroup_performance_carries_intervals_that_match_proc_and_a_hand_mean(tmp_path):
    """TRIPOD+AI 23a: performance with confidence intervals within each named group (sex's levels;
    a continuous column's thirds by its type-7 quantiles, read as nothing but its values). AUC with
    DeLong's interval against R ``pROC::ci.auc(method = "delong")``; log loss as the mean of the
    rows' losses with the influence-function (here, the plain) standard error, by hand."""
    rng = np.random.default_rng(3)
    n = 900
    event, risk = _risk_data(n, 4)
    sex = rng.integers(1, 3, n)
    age = rng.uniform(20, 80, n)
    P = np.column_stack([1 - risk, risk])
    found = DC.subgroup_performance("binary", event, P, {"sex": sex, "age": age}, classes=[0, 1],
                                    metrics=["log_loss", "auc"])
    by = {(c["column"], g["group"]): g for c in found for g in c["groups"]}
    assert [c["grouped_by"] for c in found] == ["levels", "thirds"]
    cuts = np.quantile(age, [1 / 3, 2 / 3])
    thirds = np.searchsorted(cuts, age, side="left")
    frame = pd.DataFrame({"event": event, "risk": risk, "sex": sex, "third": thirds})
    r = run_r("""
        suppressPackageStartupMessages(library(pROC))
        d <- read.csv(g_csv)
        ci <- function(rows) { x <- ci.auc(roc(d$event[rows], d$risk[rows], quiet = TRUE,
                                                 direction = "<", levels = c(0, 1)),
                                             method = "delong"); as.numeric(x) }
        out(list(s1 = ci(d$sex == 1), s2 = ci(d$sex == 2), a0 = ci(d$third == 0),
                 a2 = ci(d$third == 2)))
    """, {"g": frame}, tmp_path)
    for key, (column, mask) in {"s1": ("sex", sex == 1), "s2": ("sex", sex == 2)}.items():
        group = by[(column, str(int(key[1])))]
        auc = group["scores"]["auc"]
        assert abs(auc["estimate"] - r[key][1]) < 1e-8
        assert abs(auc["ci_low"] - r[key][0]) < 1e-8 and abs(auc["ci_high"] - r[key][2]) < 1e-8
        losses = -(event[mask] * np.log(risk[mask]) + (1 - event[mask]) * np.log(1 - risk[mask]))
        ll = group["scores"]["log_loss"]
        assert abs(ll["estimate"] - losses.mean()) < 1e-12
        assert abs(ll["se"] - losses.std(ddof=1) / math.sqrt(mask.sum())) < 1e-9
    lowest = by[("age", next(g for (c, g) in by if c == "age" and g.startswith("≤")))]
    assert abs(lowest["scores"]["auc"]["estimate"] - r["a0"][1]) < 1e-8
    assert lowest["n"] == int((thirds == 0).sum())


def test_4_a_group_with_too_few_events_is_named_not_scored():
    event = np.array([1] * 3 + [0] * 40 + [1] * 30 + [0] * 30)
    risk = np.linspace(0.05, 0.9, len(event))
    group = np.array(["a"] * 43 + ["b"] * 60)
    found = DC.subgroup_performance("binary", event, np.column_stack([1 - risk, risk]),
                                    {"g": group}, classes=[0, 1], metrics=["auc"])
    a = found[0]["groups"][0]
    assert a["group"] == "a" and a["scores"] == {} and "Too few rows of one class" in a["note"]


# ── model updating: shrinkage by the calibration slope ──────────────────────


@needs_r
@pytest.mark.parametrize("task", ["binary", "regression"])
def test_4_shrinkage_by_the_calibration_slope_re_estimates_the_intercept_as_r_does(task, tmp_path):
    """TRIPOD+AI 12f; Collins et al. (BMJ 2024): "the value of optimism corrected calibration slope
    can be used to adjust the model from any overfitting by applying it as shrinkage factor to the
    original regression coefficients". Each coefficient × s, the intercept re-estimated with the
    shrunk linear predictor as an offset: R ``glm(y ~ 1 + offset(s·lp), binomial)`` or ``lm``."""
    rng = np.random.default_rng(5)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    beta = np.array([0.8, -0.5, 0.3])
    if task == "binary":
        y = (rng.random(n) < 1 / (1 + np.exp(-(-0.4 + X.to_numpy() @ beta)))).astype(float)
    else:
        y = 2.0 + X.to_numpy() @ beta + rng.normal(0, 1, n)
    s = 0.87
    got = DC.shrinkage(task, X, y, beta * 1.1, s)
    lp = X.to_numpy() @ (beta * 1.1)
    family = "binomial" if task == "binary" else "gaussian"
    r = run_r(f"""
        d <- read.csv(s_csv)
        fit <- glm(y ~ 1 + offset({s} * lp), family = {family}, data = d,
                   control = glm.control(epsilon = 1e-14, maxit = 100))
        out(list(a = unname(coef(fit)[1])))
    """, {"s": pd.DataFrame({"y": y, "lp": lp})}, tmp_path)
    assert abs(got["intercept"] - r["a"]) < 1e-8
    assert [c["shrunk"] for c in got["coefficients"]] == pytest.approx(list(s * beta * 1.1), abs=1e-15)


def test_4_the_slope_offered_is_the_optimism_corrected_one_when_the_bootstrap_ran():
    model = {"optimism": {"estimates": {"calibration_slope": {"corrected": 0.91}}},
             "calibration": {"slope": {"estimate": 0.95}}}
    assert DC.slope_of(model) == (0.91, "the optimism-corrected calibration slope (Harrell's bootstrap)")
    assert DC.slope_of({"calibration": {"slope": {"estimate": 0.95}}}) == (
        0.95, "the out-of-fold calibration slope (cross-validation)")


# ── the imbalance correction: in-fold, then recalibrated ─────────────────────


def test_4_an_imbalance_correction_runs_in_fold_and_is_followed_by_recalibration():
    """TRIPOD+AI 13; van den Goorbergh et al. (JAMIA 2022;29:1525): "random undersampling, random
    oversampling, or SMOTE yielded poorly calibrated models: the probability to belong to the
    minority class was strongly overestimated". Simulation with a known truth (a logistic model at
    an 8% event rate): balanced weights without recalibration overestimate the risk on fresh rows;
    after the in-fold logistic recalibration, calibration-in-the-large is near 0 and the slope near
    1. The recalibration is learned on inner cross-validated predictions of the fitting rows only."""
    from sklearn.linear_model import LogisticRegression

    from turbotab.core.methods.levers import ImbalanceCorrected

    rng = np.random.default_rng(7)

    def draw(n):
        X = rng.normal(size=(n, 3))
        p = 1 / (1 + np.exp(-(-3.0 + X @ np.array([0.9, -0.6, 0.4]))))
        return X, (rng.random(n) < p).astype(int), p

    X, y, _ = draw(4000)
    Xn, yn, pn = draw(40000)
    base = LogisticRegression(C=np.inf, solver="newton-cholesky", max_iter=1000)
    for method in ("weights", "undersample", "oversample"):
        raw = ImbalanceCorrected(base, method, recalibrate=False, seed=1).fit(X, y)
        fixed = ImbalanceCorrected(base, method, recalibrate=True, seed=1).fit(X, y)
        p_raw = raw.predict_proba(Xn)[:, 1]
        p_fix = fixed.predict_proba(Xn)[:, 1]
        assert p_raw.mean() > 2.5 * yn.mean(), method  # strongly overestimated
        assert abs(p_fix.mean() - yn.mean()) < 0.01, method
        fit = sm.Logit(yn, sm.add_constant(np.log(p_fix / (1 - p_fix)))).fit(disp=0)
        # Undersampling fits on about 640 of the 4,000 rows, so its slope varies most.
        assert abs(fit.params[1] - 1.0) < (0.2 if method == "undersample" else 0.1), method
    # In-fold: the recalibration reads only the rows it is fitted on.
    a = ImbalanceCorrected(base, "weights", seed=1).fit(X[:3000], y[:3000])
    Xb = X.copy()
    Xb[3000:] *= 5.0
    b = ImbalanceCorrected(base, "weights", seed=1).fit(Xb[:3000], y[:3000])
    assert a.calibration_ == b.calibration_


def test_4_the_imbalance_correction_wraps_each_fold_model_and_only_for_a_yes_no_outcome():
    from turbotab.core.methods.levers import ImbalanceCorrected, wrap_model
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression()
    assert isinstance(wrap_model(m, {"imbalance": "weights"}, "binary"), ImbalanceCorrected)
    assert wrap_model(m, {"imbalance": "weights"}, "regression") is m
    assert wrap_model(m, {"imbalance": "none"}, "binary") is m
    state = d.ProjectState(purpose="prediction", task="regression", target="y")
    with pytest.raises(Refusal) as caught:
        d.validate(d.SetLevers(imbalance="weights"), {"state": state})
    assert caught.value.code == "imbalance_task"


# ── the intended-use question and its leash ──────────────────────────────────


def test_4_intended_use_is_a_prediction_question_and_decision_support_a_yes_no_one():
    infer = d.ProjectState(purpose="inference", task="binary", target="y")
    with pytest.raises(Refusal) as caught:
        d.validate(d.SetIntendedUse(use="decision_support"), {"state": infer})
    assert caught.value.code == "not_prediction"
    assert caught.value.exits[0]["decision"]["purpose"] == "prediction"
    reg = d.ProjectState(purpose="prediction", task="regression", target="y")
    with pytest.raises(Refusal) as caught:
        d.validate(d.SetIntendedUse(use="decision_support"), {"state": reg})
    assert caught.value.code == "decision_support_task"
    assert caught.value.exits[0]["decision"]["use"] == "risk_estimation"
    with pytest.raises(Refusal) as caught:
        d.validate(d.SetIntendedUse(use="decision_support", threshold_low=0.4, threshold_high=0.1),
                   {"state": d.ProjectState(purpose="prediction", task="binary", target="y")})
    assert caught.value.code == "threshold_range"


def test_4_the_intended_use_sentences_verbatim():
    from turbotab.core import voice

    assert voice.sentence_for(d.SetIntendedUse(use="decision_support", threshold_low=0.05,
                                               threshold_high=0.3, subgroups=["sex", "age"])) == (
        "The model was declared for decision support: net benefit was assessed by decision curve "
        "analysis over threshold probabilities from `0.05` to `0.3` (Vickers & Elkin 2006), and the "
        "decision threshold was chosen within each training fold by Youden's J (customary) and "
        "scored on the fold its model never saw; performance was reported with 95% intervals within "
        "the groups of `sex` and `age` (TRIPOD+AI 23a); no fairness method beyond that report was "
        "applied (TRIPOD+AI 14).")
    assert voice.sentence_for(d.SetIntendedUse(use="decision_support", threshold=0.15)) == (
        "The model was declared for decision support: net benefit was assessed by decision curve "
        "analysis over threshold probabilities from `0.05` to `0.5` (Vickers & Elkin 2006), and the "
        "decision threshold `0.15` was declared from the decision's harms and benefits before any "
        "score was seen; no subgroup was named for performance; no fairness method beyond that "
        "report was applied (TRIPOD+AI 14).")
    assert voice.sentence_for(d.SetUpdating(method="shrinkage")) == (
        "The regression's coefficients were shrunk by the calibration slope as model updating, and "
        "its intercept re-estimated with the shrunk linear predictor as an offset (TRIPOD+AI 12f; "
        "Steyerberg 2019).")
