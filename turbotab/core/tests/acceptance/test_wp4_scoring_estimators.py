"""WP4 · Scoring estimators: the acceptance tests of AUDIT_REPORT.md §5 (closes MA-09, MA-10, MA-11,
and the minors A15, A16, A17, A19/E11, A20).

Every measured value runs through the code the fit stage runs (``metrics.cross_validate``,
``baseline.compare``, ``inner_cv.fit_pipeline``, ``stages.rows.draw_split``, the real stages);
every reference comes from somewhere else: an analytic formula, numpy or scipy, or scikit-learn
used plainly. Simulations are seeded, so each run gives the same numbers; the Monte Carlo error of
each is stated beside its bound.
"""
from __future__ import annotations

import math
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from turbotab.core import decisions as d
from turbotab.core import seal, voice
from turbotab.core.decisions import GrainSpec, ProjectState, RepeatSpec, SplitSpec, TemporalSpec
from turbotab.core.models import get_family
from turbotab.core.models.baseline import compare
from turbotab.core.models.inner_cv import fit_pipeline
from turbotab.core.models.metrics import cross_validate, fold_pairs, score
from turbotab.core.models.pipeline import DesignSpec, build_pipeline
from turbotab.core.stages.modeling import _baseline_cv, design_stage, fit_stage
from turbotab.core.stages.rows import cohort_stage, draw_split, split_stage
from turbotab.core.stages.seal import seal_plan_stage
from turbotab.core.stages.target import target_info_stage
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.stage_harness import SAMPLES, Ingested
from turbotab.core.tests.truths import FIXTURE_TRUTHS


def _pipeline(family: str, task: str, columns: list[str], n: int):
    spec = DesignSpec(predictors=columns, inputs=columns, categorical=[], numeric=columns,
                      energy=None, impute=False)
    return build_pipeline(spec, get_family(family), task, "prediction", n, len(columns))


def _random_folds(n: int, seed: int, y=None) -> np.ndarray:
    """The app's own folds for n rows, cross-validation only (``draw_split``)."""
    frame, _ = draw_split(np.arange(n), holdout=0.0, seed=seed, folds=5, y=y)
    return frame.sort_values("row_id")["fold"].to_numpy()


def _ols_target(m: int, p: int, sigma2: float, var_y: float) -> float:
    """1 − E[MSE]/Var(Y) for OLS with an intercept and p Gaussian predictors fit on m rows.

    E[(y₀ − ŷ₀)²] = σ²·(1 + 1/m)·(m − 2)/(m − p − 2): with S = Σ(xᵢ − x̄)(xᵢ − x̄)′ ~ Wishart(m − 1,
    Σ), E[S⁻¹] = Σ⁻¹/(m − p − 2), so E[(x₀ − x̄)′S⁻¹(x₀ − x̄)] = p(1 + 1/m)/(m − p − 2).
    """
    return 1 - sigma2 * (1 + 1 / m) * (m - 2) / (m - p - 2) / var_y


# ── 1 · pooled R² ────────────────────────────────────────────────────────────


def test_1_pooled_cv_r2_matches_the_out_of_sample_target_at_n_100():
    """OLS, p = 5, population R² 0.20, 5-fold, 500 replicates, n = 100: the reported CV R²
    averages within 0.02 of the target (the R² on new rows of a model fit on 80 rows).

    Reference: analytic, ``_ols_target(80, 5, σ² = 1, Var Y = 1.25)`` = 0.1345 (the audit's
    simulated 0.13–0.14). Today's estimator, per-fold R² against each test fold's own mean and
    averaged, is recomputed here with plain scikit-learn on the same folds: about 0.05.

    Source check — Hawinkel, Waegeman & Maere, "The out-of-sample R²: estimation and inference",
    Am Stat 2024 (arXiv:2302.05131): "The pooling R² estimator, which separately estimates the
    squared error losses of the null and prediction models and only then combines them into a
    final estimate R², is unbiased. Hence this pooling estimator should be preferred to averaging
    estimators that calculate R² values in every cross-validation fold separately and then average
    over the folds, which suffer from bias." And: "The averaging R² with test MST estimator is
    very variable and dramatically downward biased for smaller sample sizes, and even at a sample
    size of 100 some of the bias persists."
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    n, p, reps = 100, 5, 500
    rng = np.random.default_rng(20261002)
    beta = np.full(p, math.sqrt(0.25 / p))  # Var(xβ) = 0.25, noise 1: population R² = 0.2
    cols = [f"x{i}" for i in range(p)]
    pipe = _pipeline("linear", "regression", cols, n)
    reported, averaged = [], []
    for r in range(reps):
        X = pd.DataFrame(rng.normal(size=(n, p)), columns=cols)
        y = X.to_numpy() @ beta + rng.normal(size=n)
        folds = _random_folds(n, seed=r)
        result = cross_validate("regression", lambda: clone(pipe), X, y, fold_pairs(folds),
                                fit=lambda m, Xf, yf, rows: fit_pipeline(m, Xf, yf))
        reported.append(result.summary("regression")["r2"]["estimate"])
        per_fold = []
        for k in range(5):  # today's estimator, independently: sklearn's r2_score per test fold
            fit_rows, test_rows = folds != k, folds == k
            model = LinearRegression().fit(X[fit_rows], y[fit_rows])
            per_fold.append(r2_score(y[test_rows], model.predict(X[test_rows])))
        averaged.append(np.mean(per_fold))
    target = _ols_target(80, p, 1.0, 1.25)
    mean, mc = float(np.mean(reported)), float(np.std(reported) / math.sqrt(reps))
    assert target == pytest.approx(0.1345, abs=5e-4)
    assert abs(mean - target) <= 0.02, f"reported CV R² {mean:.4f} (MC SE {mc:.4f}) vs target {target:.4f}"
    # The fixture is the one the audit found biased: today's estimator misses by more than 0.05.
    assert np.mean(averaged) < target - 0.05, f"per-fold average {np.mean(averaged):.4f}"


def test_1b_the_holdout_r2_is_measured_against_the_training_mean():
    """Held-out R² uses the training rows' mean: 1 − Σ(y − ŷ)²/Σ(y − ȳ_train)².

    Part 1 (simulation): 240 training rows, 60 held out, 2,000 replicates of OLS with p = 5 and a
    population R² of 0.2. The app's held-out score (``metrics.score`` with the training mean, as
    ``fit_stage`` calls it) averages within 0.01 of the analytic target ``_ols_target(240, …)`` =
    0.1794; scored against the held-out rows' own mean (scikit-learn's ``r2_score``) it reads
    about 0.155 (the audit's 0.155–0.157 against 0.175).

    Part 2 (the stage): the sealed held-out R² of a real fit equals the numpy formula.
    """
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(7)
    p, reps = 5, 2000
    beta = np.full(p, math.sqrt(0.25 / p))
    cols = [f"x{i}" for i in range(p)]
    pipe = _pipeline("linear", "regression", cols, 240)
    app, own_mean = [], []
    for _ in range(reps):
        X = pd.DataFrame(rng.normal(size=(300, p)), columns=cols)
        y = X.to_numpy() @ beta + rng.normal(size=300)
        model = fit_pipeline(clone(pipe), X.iloc[:240], y[:240])
        app.append(score("regression", model, X.iloc[240:], y[240:], reference=float(y[:240].mean()))["r2"])
        own_mean.append(r2_score(y[240:], model.predict(X.iloc[240:])))
    target = _ols_target(240, p, 1.0, 1.25)
    mean, mc = float(np.mean(app)), float(np.std(app) / math.sqrt(reps))
    assert abs(mean - target) <= 0.01, f"held-out R² {mean:.4f} (MC SE {mc:.4f}) vs {target:.4f}"
    assert np.mean(own_mean) < target - 0.015, f"own-mean R² {np.mean(own_mean):.4f}"


def test_1c_the_stage_reports_the_pooled_r2_and_scores_the_holdout_against_the_training_mean(tmp_path):
    from turbotab.core.seal import SEALED_SCORES, scores_by_family

    frame = mf.nhanes_like(300, seed=12)
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)), seed=3)
    st = mf.state(energy_adjustment=mf.energy("none"), models=["linear"])
    ti = mf.target_info("regression")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    model = fit.data["models"][0]
    assert model["cv"]["r2"]["estimator"] == "pooled"
    assert "pool every out-of-fold prediction" in fit.data["cv_definition"]
    a = split.frames["assignment"]
    train_ids = a.loc[a["partition"] == "train", "row_id"].to_numpy()
    hold_ids = a.loc[a["partition"] == "holdout", "row_id"].to_numpy()
    inputs = design.objects["spec"]["inputs"]
    X_hold = frame.loc[hold_ids, inputs].astype({c: float for c in inputs if frame[c].dtype == bool})
    pred = fit.objects["fitted"]["linear"].predict(X_hold)
    y_hold, y_train = frame.loc[hold_ids, "glucose"].to_numpy(), frame.loc[train_ids, "glucose"].to_numpy()
    expected = 1 - ((y_hold - pred) ** 2).sum() / ((y_hold - y_train.mean()) ** 2).sum()
    sealed = scores_by_family(fit.frames[SEALED_SCORES])["linear"]
    assert sealed["r2"] == pytest.approx(expected, abs=1e-12)
    # The methods sentence defines it.
    text = voice.sentence_for(d.SetSplit(holdout=0.2, seed=3, folds=5),
                              ProjectState(target="glucose", task="regression"), {})
    assert "R² was measured against the training rows' mean and pooled over every out-of-fold" in text


# ── 2 · better than the baseline ─────────────────────────────────────────────


def _null_binary_rate(n: int, reps: int, seed: int, signal: float = 0.0, repeats: int = 1):
    """Share of datasets the app calls "better" than the class prior, and the same datasets under
    today's rule (gain > max(0.01, SD/√K)), recomputed independently from the per-fold AUCs.

    ``repeats`` > 1 pairs the folds of the app's comparison substrate (MS6: the split's folds and
    the further draws ``folds.comparison_folds`` makes), as the fit stage's verdict does."""
    from scipy import stats

    from turbotab.core.models.folds import comparison_folds
    from turbotab.core.models.metrics import repeated_pairs

    p = 10
    rng = np.random.default_rng(seed)
    cols = [f"x{i}" for i in range(p)]
    pipe = _pipeline("linear", "binary", cols, n)
    better = old = 0
    for r in range(reps):
        X = pd.DataFrame(rng.normal(size=(n, p)), columns=cols)
        logit = -0.85 + signal * X["x0"].to_numpy()  # prevalence about 0.3
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        columns = [_random_folds(n, seed=r, y=y)]
        if repeats > 1:
            columns, _, _ = comparison_folds(columns, validation="kfold", scheme="random", n=n,
                                             strata=y, folds=5, seed=r, repeats=repeats)
        pairs, repeat_of = repeated_pairs(columns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = cross_validate("binary", lambda: clone(pipe), X, y, pairs,
                                   fit=lambda m, Xf, yf, rows: fit_pipeline(m, Xf, yf),
                                   repeat_of=repeat_of)
            base = _baseline_cv("binary", X, y, pairs, repeat_of)
        _, versus = compare("binary", model, base)
        better += versus.verdict == "better"
        aucs = np.array([a["auc"] - b["auc"] for a, b in zip(model.per_fold, base.per_fold)])
        old += aucs.mean() > max(0.01, aucs.std(ddof=1) / math.sqrt(len(aucs)))  # the AUC rule
        # MS6: the verdict reads the strictly proper primary, log loss (better when lower): a gain
        # is the prior's log loss less the model's, fold by fold.
        assert versus.metric == "log_loss"
        gains = np.array([b["log_loss"] - a["log_loss"] for a, b in zip(model.per_fold, base.per_fold)])
        # The interval, recomputed: Nadeau & Bengio's corrected variance over the rK paired folds
        # (Bouckaert & Frank's repeated form), t with rK − 1 df.
        rk = len(gains)
        sizes = [(int((c != k).sum()), int((c == k).sum())) for c in columns for k in range(5)]
        share = np.mean([s / f for f, s in sizes])
        se = math.sqrt((1 / rk + share) * gains.var(ddof=1))
        half = stats.t.ppf(0.975, rk - 1) * se
        assert versus.gain == pytest.approx(gains.mean(), abs=1e-12)  # log loss: the fold mean
        assert (versus.ci_low, versus.ci_high) == pytest.approx((gains.mean() - half, gains.mean() + half), abs=1e-12)
    return better / reps, old / reps


@pytest.mark.parametrize("n", [150, 400])
def test_2_better_than_baseline_holds_its_size_on_null_binary_data(n):
    """300 binary datasets with no signal (10 predictors, prevalence 0.3): the app says "better"
    on at most 7% (today 25–31%), and every gain carries its 95% interval.

    Reference: the bound is the nominal 5% level plus Monte Carlo slack; the verdict rule clears
    zero with the two-sided 95% interval (a one-sided 2.5% test), so its expected rate is lower.
    Today's rule is recomputed on the same per-fold AUCs to show the fixture reproduces the defect.
    Source check — Bengio & Grandvalet, JMLR 2004;5:1089–1105: "naive estimators … grossly
    underestimate variance"; the correction is Nadeau & Bengio, Machine Learning 2003;52:239–281.
    """
    rate, old = _null_binary_rate(n, 300, seed=n)
    mc = math.sqrt(0.05 * 0.95 / 300)
    assert rate <= 0.07, f"'better' on {rate:.3f} of null datasets (MC SE at 5%: {mc:.3f})"
    assert old >= 0.15, f"today's rule fired on {old:.3f}: the fixture no longer shows the defect"


def test_2c_the_size_holds_on_the_comparison_substrate():
    """MS6: the fit stage reads the verdict over the comparison substrate's 10 × 5 folds, so its
    size is checked there too: 100 null binary datasets (as above, n = 150), "better" on at most
    7% (Monte Carlo SE at 5%: 0.022). Every interval is recomputed with rK − 1 = 49 df."""
    rate, _ = _null_binary_rate(150, 100, seed=7, repeats=10)
    assert rate <= 0.07, f"'better' on {rate:.3f} of null datasets over the substrate"


def test_2b_a_real_signal_is_still_called_better():
    """Power, so the size above is not bought by never saying "better": log-odds 1.0 per SD of
    one predictor (population AUC 0.74) at n = 400 is called better in at least 80% of 60
    datasets.

    MS6: the verdict reads log loss, a strictly proper score, paired over the comparison
    substrate's 10 × 5 folds as the fit stage pairs them (49 df). Measured: 0.88 (53 of 60; Monte
    Carlo SE 0.04). The interval binds, not the 0.01 minimum gain: its half-width averages 0.048
    nats against a mean gain of 0.066, because Nadeau & Bengio's n₂/n₁ term (0.25) dominates the
    corrected variance, so more repeats barely narrow it. On AUC, against a prior whose AUC is 0.5
    in every fold, the same datasets were called better in 98%; on one 5-fold partition (4 df) the
    log-loss verdict calls only 27% better, which is why the substrate exists (Bouckaert & Frank
    2004)."""
    rate, _ = _null_binary_rate(400, 60, seed=11, signal=1.0, repeats=10)
    assert rate >= 0.8, f"'better' on only {rate:.2f} of datasets with a real signal"


# ── 3 · time-ordered folds ───────────────────────────────────────────────────

CLINICAL_ROLES = {"subject_id": "identifier", "visit": "time", "visit_date": "time", "age": "covariate",
                  "sbp": "covariate", "glucose": "covariate"}


def _clinical_state(**slots) -> ProjectState:
    base = dict(target="progressed", task="binary", purpose="prediction", roles=dict(CLINICAL_ROLES),
                missing="complete_case", split=SplitSpec(holdout=0.2, seed=1, folds=5),
                grain=GrainSpec(grain="repeated", id_column="subject_id"),
                repeat_kind=RepeatSpec(repeat_kind="time_points", time_column="visit_date"),
                unit="row", temporal=TemporalSpec(temporal=True, time_column="visit_date"),
                models=["linear", "elastic_net"],
                # clinical_longitudinal.csv's truth (BLUEPRINT §14.3: whole numbers are asked)
                shape_confirmations={k: v for k, v in FIXTURE_TRUTHS["clinical_longitudinal.csv"].items()
                                     if k.startswith("code_or_count:")})
    base.update(slots)
    return ProjectState(**base)


@pytest.fixture(scope="module")
def clinical(tmp_path_factory):
    return Ingested(SAMPLES / "clinical_longitudinal.csv", tmp_path_factory.mktemp("clinical_wp4"))


@pytest.fixture(scope="module")
def last_visit():
    """Each subject's last visit date, read from the CSV with pandas (independent of the engine)."""
    raw = pd.read_csv(SAMPLES / "clinical_longitudinal.csv")
    return raw.assign(when=pd.to_datetime(raw["visit_date"])).groupby("subject_id")["when"].max()


def _stages(table: Ingested, state: ProjectState, fit: bool = False):
    info = table.run(target_info_stage, state)
    cohort = table.run(cohort_stage, state, {"target_info": info})
    split = table.run(split_stage, state, {"cohort": cohort, "target_info": info})
    if not fit:
        return info, cohort, split, None
    design = table.run(design_stage, state, {"split": split, "target_info": info})
    return info, cohort, split, table.run(fit_stage, state, {"design": design, "split": split,
                                                             "target_info": info})


def test_3a_with_temporal_yes_the_folds_forward_chain_by_whole_unit(clinical, last_visit, monkeypatch):
    """Fold time ranges are ordered: every subject in fold j has its last visit no later than every
    subject in fold j + 1; no subject spans folds; and the fit scores fold j with models fit on
    folds 0 … j − 1 only (spied on the logistic fits). Reference: last visits from the raw CSV."""
    from sklearn.linear_model import LogisticRegression

    seen = []
    original = LogisticRegression.fit

    def spy(self, X, y, *a, **k):
        seen.append(np.asarray(X.index))
        return original(self, X, y, *a, **k)

    monkeypatch.setattr(LogisticRegression, "fit", spy)
    _, _, split, fit = _stages(clinical, _clinical_state(models=["linear"]), fit=True)
    assert split.data["fold_scheme"] == "time_ordered" and split.data["time_ordered_folds"]
    assert fit.data["fold_scheme"] == "time_ordered"
    # MS6: forward-chaining folds are the same in every repeat, so the comparisons rest on one run
    # of them, and the fit says so.
    from turbotab.core.models.folds import TIME_ORDERED_ONCE

    assert fit.data["comparison"]["repeats"] == 1 and fit.data["comparison"]["note"] == TIME_ORDERED_ONCE
    a = split.frames["assignment"]
    subjects = clinical.frame(["subject_id"]).loc[a["row_id"].to_numpy(), "subject_id"].to_numpy()
    t = pd.DataFrame({"subject": subjects, "part": a["partition"].to_numpy(), "fold": a["fold"].to_numpy()})
    t = t[t["part"] == "train"]
    assert t.groupby("subject")["fold"].nunique().max() == 1  # whole units
    t["last"] = t["subject"].map(last_visit)
    ranges = t.groupby("fold")["last"].agg(["min", "max"]).sort_index()
    assert len(ranges) == split.data["folds"] + 1  # k scored folds over k + 1 blocks
    assert (ranges["max"].to_numpy()[:-1] <= ranges["min"].to_numpy()[1:]).all(), ranges
    # Five scored folds, then the refit on every training row: fit j learns from blocks before j.
    row_fold = dict(zip(a["row_id"].to_numpy(), a["fold"].to_numpy()))
    row_subject = dict(zip(a["row_id"].to_numpy(), subjects))
    assert len(seen) >= split.data["folds"] + 1  # every scored fold, then the refit
    fold_fits = seen[:split.data["folds"]]
    for j, rows in enumerate(fold_fits, start=1):
        assert {row_fold[r] for r in rows} == set(range(j))
        learned = max(last_visit[row_subject[r]] for r in rows)
        scored = ranges.loc[j, "min"]
        assert learned <= scored
    assert len(fit.data["models"][0]["cv"]["auc"]["folds"]) == split.data["folds"]


def test_3b_elastic_nets_inner_cv_receives_the_same_splitter(clinical, last_visit, monkeypatch):
    """Every elastic-net fit (each outer fold and the refit) tunes its penalty on inner splits that
    forward-chain by whole subject: no subject on both sides, and every subject it learns from has
    its last visit no later than any subject it is scored on."""
    from turbotab.core.models.elastic_net import PooledLogisticRegressionCV

    seen = []
    original = PooledLogisticRegressionCV.fit

    def spy(self, X, y, *a, **k):
        seen.append((np.asarray(X.index), self.cv))
        return original(self, X, y, *a, **k)

    monkeypatch.setattr(PooledLogisticRegressionCV, "fit", spy)
    _, _, split, _ = _stages(clinical, _clinical_state(models=["elastic_net"]), fit=True)
    subjects = clinical.frame(["subject_id"])["subject_id"]
    assert len(seen) == split.data["folds"] + 1
    for row_ids, cv in seen:
        assert isinstance(cv, list) and len(cv) >= 2
        who = subjects.loc[row_ids].to_numpy()
        for train, test in cv:
            assert not set(who[train]) & set(who[test])
            assert max(last_visit[s] for s in who[train]) <= min(last_visit[s] for s in who[test])


def test_3c_the_methods_sentence_says_time_ordered_folds(clinical):
    for temporal, expected in ((True, True), (False, False)):
        state = _clinical_state(temporal=TemporalSpec(temporal=temporal,
                                                      time_column="visit_date" if temporal else None))
        info = clinical.run(target_info_stage, state)
        cohort = clinical.run(cohort_stage, state, {"target_info": info})
        plan = clinical.run(seal_plan_stage, state, {"cohort": cohort, "target_info": info})
        assert plan["time_ordered_folds"] is expected
        text = voice.sentence_for(d.SetSplit(holdout=0.2, seed=1, folds=5), state, {"seal_plan": plan})
        assert ("time-ordered folds" in text) is expected, text
        assert ("`5`-fold cross-validation" in text) is not expected, text


def test_3d_fold_stratification_is_decided_independently_no_event_free_folds():
    """The 4%-event fixture: 375 rows, one per unit, 4% events, the latest 20% held out by time.

    Time-ordered folds cannot be stratified, so their cuts move until every block (the first
    training block too) holds an event: no event-free fold in any of 200 datasets. Reference: on
    the same training rows, plain shuffled 5-fold (what a chronological holdout used to leave the
    folds with), computed with scikit-learn, leaves a fold without events in about a third of them.
    A holdout drawn elsewhere (``held``) with random folds no longer switches their
    stratification off either: the folds are stratified and every one holds an event.
    """
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(4)
    reps, n = 200, 375
    event_free_kfold = 0
    for r in range(reps):
        times = rng.uniform(0, 10, n)
        y = np.where(rng.random(n) < 0.04, "event", "none")
        held, _ = seal.chronological_holdout(times, None, 0.2, r, "year", dated=False)
        order = seal.time_order(times, None, r)
        frame, info = draw_split(np.arange(n), holdout=0.2, seed=r, folds=5, y=y, held=held, order=order)
        frame = frame.sort_values("row_id")
        train = frame["partition"].eq("train").to_numpy()
        events = pd.Series(y[train] == "event").groupby(frame.loc[train, "fold"].to_numpy()).sum()
        assert info["fold_scheme"] == "time_ordered"
        assert (events > 0).all(), (r, events.to_dict(), info["notes"])
        # time order holds inside the training rows
        last = pd.Series(times[train]).groupby(frame.loc[train, "fold"].to_numpy()).agg(["min", "max"])
        assert (last["max"].to_numpy()[:-1] <= last["min"].to_numpy()[1:]).all()
        yt = y[train]
        if any((yt[te] == "event").sum() == 0 for _, te in KFold(5, shuffle=True, random_state=r).split(yt)):
            event_free_kfold += 1
        # random folds beside a holdout drawn elsewhere: stratified on their own
        rand, rinfo = draw_split(np.arange(n), holdout=0.2, seed=r, folds=5, y=y, held=held)
        rand = rand.sort_values("row_id")
        rt = rand["partition"].eq("train").to_numpy()
        per = pd.Series(y[rt] == "event").groupby(rand.loc[rt, "fold"].to_numpy()).sum()
        assert rinfo["folds_stratified"]
        if (y[rt] == "event").sum() >= rinfo["folds"]:  # fewer events than folds: none can
            assert (per > 0).all(), (r, per.to_dict())
    assert event_free_kfold / reps >= 0.2, f"plain folds were event-free in only {event_free_kfold}/{reps}"


def test_3e_time_ordered_folds_score_every_fold_on_the_rare_event_fixture():
    """The fit no longer fails on a rare event: every scored fold's AUC is defined."""
    rng = np.random.default_rng(9)
    n, p = 375, 4
    cols = [f"x{i}" for i in range(p)]
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=cols)
    y = (rng.random(n) < 0.04).astype(int)
    times = rng.uniform(0, 10, n)
    held, _ = seal.chronological_holdout(times, None, 0.2, 0, "year", dated=False)
    frame, info = draw_split(np.arange(n), holdout=0.2, seed=0, folds=5, y=y.astype(str), held=held,
                             order=seal.time_order(times, None, 0))
    frame = frame.sort_values("row_id")
    train = frame["partition"].eq("train").to_numpy()
    pairs = fold_pairs(frame.loc[train, "fold"].to_numpy(), "time_ordered")
    result = cross_validate("binary", lambda: clone(_pipeline("linear", "binary", cols, n)),
                            X[train], y[train], pairs, fit=lambda m, Xf, yf, rows: fit_pipeline(m, Xf, yf))
    assert len(result.per_fold) == info["folds"] and all(math.isfinite(f["auc"]) for f in result.per_fold)


# ── 4 · order independence ───────────────────────────────────────────────────


def test_4_the_same_rows_in_any_order_choose_the_same_elastic_net_penalty(tmp_path):
    """The audit's fixture: 1,000 rows, 20 predictors, y = 0.3·(x0 + … + x4) + noise, written once
    in random order and once sorted by the outcome, each run through the real design and fit
    stages (cross-validation only, so the refit sees every row). The elastic net chooses the same
    penalty and mix from both files (the audit: 0.0299 against 0.0004).

    Reference: scikit-learn's ``ElasticNetCV(cv=5)`` (unshuffled KFold, today's inner CV) on the
    same two orders, which chooses penalties more than ten-fold apart.
    """
    from sklearn.linear_model import ElasticNetCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from turbotab.core.models.elastic_net import L1_RATIOS

    rng = np.random.default_rng(0)
    n, p = 1000, 20
    cols = [f"x{i}" for i in range(p)]
    data = pd.DataFrame(rng.normal(size=(n, p)), columns=cols)
    data["y"] = data[cols[:5]].sum(axis=1) * 0.3 + rng.normal(size=n)
    orders = {"random": data, "sorted": data.sort_values("y").reset_index(drop=True)}
    chosen, today = {}, {}
    for name, frame in orders.items():
        paths = mf.ingest_frame(frame, tmp_path / name)
        st = ProjectState(target="y", task="regression", purpose="prediction",
                          roles={c: "covariate" for c in cols}, missing="complete_case",
                          split=SplitSpec(holdout=0.0, seed=0, folds=5), models=["elastic_net"])
        split = mf.split_bundle(np.arange(n), holdout=0.0, seed=0)
        ti = mf.target_info("regression", "y")
        design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
        fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
        model = fit.objects["fitted"]["elastic_net"][-1]
        chosen[name] = (float(model.alpha_), float(model.l1_ratio_))
        plain = make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=list(L1_RATIOS), cv=5, max_iter=5000))
        today[name] = float(plain.fit(frame[cols], frame["y"])[-1].alpha_)
    assert chosen["random"][0] == pytest.approx(chosen["sorted"][0], rel=1e-9), chosen
    assert chosen["random"][1] == chosen["sorted"][1], chosen
    assert max(today.values()) / min(today.values()) > 10, today  # the fixture still shows A15


# ── 5 · the seal plan's precision ────────────────────────────────────────────


@pytest.mark.parametrize("n", [100, 500])
def test_5_the_plans_holdout_r2_precision_is_within_20_percent_of_the_simulated_sd(n):
    """For an assumed R² of 0.2, the seal plan's stated precision of a held-out R² on n rows is
    within 20% of the SD of that R² over 4,000 simulated holdouts (today it overstates 2–8×).

    Reference: simulation in numpy. A model whose predictions are the true signal (population R²
    0.2) is scored on n fresh rows against the mean of 4n training rows, as the app scores a
    holdout. The stated half-width is read from the plan's own text and converted to an SE with
    the 1.96 it names.
    """
    rng = np.random.default_rng(n)
    reps, rho = 4000, 0.2
    values = np.empty(reps)
    for r in range(reps):
        f_train = rng.normal(size=4 * n) * math.sqrt(rho)
        y_train = f_train + rng.normal(size=4 * n) * math.sqrt(1 - rho)
        f = rng.normal(size=n) * math.sqrt(rho)
        y = f + rng.normal(size=n) * math.sqrt(1 - rho)
        values[r] = 1 - ((y - f) ** 2).sum() / ((y - y_train.mean()) ** 2).sum()
    sd = float(values.std(ddof=1))
    text, _ = seal.measure("regression", n)
    stated = float(re.search(r"±(\d+\.\d+)", text).group(1)) / seal.Z95
    exact = seal.r2_se(n)
    today = math.sqrt(2 / n)  # the formula it replaces: (1 − R²)·√(2/n) at R² = 0
    assert abs(exact - sd) / sd <= 0.20, f"SE {exact:.4f} vs simulated SD {sd:.4f}"
    assert abs(stated - sd) / sd <= 0.20 + 0.005 / seal.Z95 / sd, f"stated {text!r} vs SD {sd:.4f}"
    assert today / sd > 1.8, f"the old formula overstated only {today / sd:.2f}×"
    assert "R² of 0.2" in seal.PRECISION_NOTE


# ── minors: A16 (early stopping by unit), A20 (fit-time scaling) ──────────────


def test_a16_boosted_trees_stop_early_on_whole_units_the_latest_when_time_ordered(monkeypatch):
    """Above 10,000 rows boosted trees hold rows aside to stop early. Those rows are whole units
    (never a person on both sides), the latest units when the folds follow time, and the same
    units whatever order the rows arrive in. Reference: scikit-learn's own draw (a 10% split of
    positions, as HistGradientBoosting makes it) puts most held-aside people on both sides."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    seen = []
    original = HistGradientBoostingRegressor.fit

    def spy(self, X, y, *a, **k):
        seen.append((np.asarray(X.index), np.asarray(k["X_val"].index) if k.get("X_val") is not None else None))
        return original(self, X, y, *a, **k)

    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", spy)
    rng = np.random.default_rng(1)
    people, per = 3000, 4
    person = np.repeat(np.arange(people), per)
    X = pd.DataFrame(rng.normal(size=(people * per, 3)), columns=["a", "b", "c"])
    X.index = pd.Index(np.arange(len(X)) + 10_000, name="row_id")
    y = X["a"].to_numpy() + rng.normal(size=len(X))
    # Boosted trees are tuned (RT-5a), so they are built through a plan: its standard settings
    # alone (mode "standard"), for a plan whose outer training fold holds these 12,000 rows, so
    # scikit-learn's own rule (above 10,000 of the plan's rows) has them stop early.
    from dataclasses import replace

    from turbotab.core.models.tuning import make_plan

    trees = get_family("boosted_trees")
    plan = make_plan(trees, task="regression", loss="mse", n_plan=people, plan_rows=len(X),
                     unit="units", split_seed=0, mode="standard")
    assert plan.standard_stops and len(plan.candidates) == 1
    spec = DesignSpec(predictors=["a", "b", "c"], inputs=["a", "b", "c"], categorical=[],
                      numeric=["a", "b", "c"], energy=None, impute=False,
                      plans={trees.key: plan.to_dict()})
    pipe = build_pipeline(spec, trees, "regression", "prediction", len(X), 3)
    fitted = fit_pipeline(clone(pipe), X, y, groups=person)
    train_rows, val_rows = seen[-1]
    who = dict(zip(X.index, person))
    assert val_rows is not None and fitted[-1].do_early_stopping_
    assert not {who[r] for r in train_rows} & {who[r] for r in val_rows}
    assert len({who[r] for r in val_rows}) == pytest.approx(0.1 * people, abs=1)
    # the same rows shuffled: the same people held aside
    shuffle = rng.permutation(len(X))
    fit_pipeline(clone(pipe), X.iloc[shuffle], y[shuffle], groups=person[shuffle])
    assert {who[r] for r in seen[-1][1]} == {who[r] for r in val_rows}
    # time-ordered: the latest tenth of the people
    rank = rng.permutation(people).astype(float)
    fit_pipeline(clone(pipe), X, y, groups=person, order=rank[person])
    latest = {who[r] for r in seen[-1][1]}
    assert latest == set(np.flatnonzero(rank >= people - len(latest)).tolist())
    # reference: a split of positions, as scikit-learn draws it, splits people
    _, sk_val = train_test_split(np.arange(len(X)), test_size=0.1, random_state=0)
    split_people = set(person[sk_val]) & set(np.delete(person, sk_val))
    assert len(split_people) / len(set(person[sk_val])) > 0.5


def test_a20_least_squares_fit_time_scales_with_n_p_min_n_p():
    """Least squares and Newton-Cholesky factor the p × p cross-product, so their estimate scales
    as n·p·min(n, p); cell-pass families as n·p. Time-ordered folds fit k/2 + 1 whole tables.
    Each family's growth is its declared cost model (MODEL_FAMILY_CONTRACT C6): the linear family
    builds least squares and Newton-Cholesky logistic regression for every task.
    Reference: the arithmetic written out here."""
    from turbotab.core.models import get_family
    from turbotab.core.models.cost import fit_cost, full_fits

    big, small = (20_000, 2_000), (200, 1_000)
    linear = get_family("linear")
    for task in linear.tasks:
        built = linear.build(task, "prediction", 100, 2)
        assert type(built).__name__ == "LinearRegression" or built.solver == "newton-cholesky"
    ratio = fit_cost(linear, *big) / fit_cost(linear, *small)
    assert ratio == pytest.approx((20_000 * 2_000 * 2_000) / (200 * 1_000 * 200))
    for cells in (get_family("elastic_net"), get_family("boosted_trees")):
        assert fit_cost(cells, *big) / fit_cost(cells, *small) == pytest.approx((20_000 * 2_000) / (200 * 1_000))
    assert full_fits(5) == 5
    assert full_fits(5, "time_ordered") == pytest.approx(sum(k / 6 for k in range(1, 6)) + 1)
