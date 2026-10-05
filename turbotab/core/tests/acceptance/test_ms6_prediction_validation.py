"""MS6 · Prediction validation: the acceptance tests of MODELING_SEQUENCE §0 ruling 4, §1 rows 9, 11
and 12 (prediction), §4's prediction leash rows and §5 MS6.

Every measured value runs through the code the fit stage runs (the real split, design and fit
stages, or the functions they call); every expected value comes from an independent path: R
(``rms::validate``, ``survival``, ``stats::glm``, and the authors' own nested cross-validation code
from Bates, Hastie & Tibshirani's ``nestedcv`` package, MIT-licensed, given the same folds), a
definition written out in ``references_ms6.py``, statsmodels, or a simulation with a known truth.
R is a reference only: the tests that need it skip cleanly where ``Rscript`` is not installed.
"""
from __future__ import annotations

import math
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from turbotab.core import decisions as d
from turbotab.core import voice
from turbotab.core.decisions import (FollowUpSpec, GrainSpec, ProjectState, Refusal, SplitSpec,
                                     validate)
from turbotab.core.models import get_family
from turbotab.core.models import performance as perf
from turbotab.core.models.folds import (COMPARISON_REPEATS, comparison_folds, draw_units,
                                        nested_folds, unit_rows)
from turbotab.core.models.inner_cv import fit_pipeline, inner_splits
from turbotab.core.models.metrics import (HEADLINE, PRIMARY, cross_validate, fold_pairs,
                                          survival_baseline, tension)
from turbotab.core.models.pipeline import DesignSpec, build_pipeline
from turbotab.core.models.selection import OutOfFold, selection_optimism
from turbotab.core.models.validation import (PROCEDURE, TOO_NARROW, VALIDATION_CONTRACTS,
                                             corrected_t, nested_cv_fits, nested_cv_interval,
                                             optimism_bootstrap, relation_ids, validation_plan)
from turbotab.core.stages.modeling import design_stage, fit_stage, shelf_stage
from turbotab.core.stages.rows import cohort_stage, draw_split, split_stage
from turbotab.core.stages.target import target_info_stage
from turbotab.core.tests.acceptance import references_ms6 as ref
from turbotab.core.tests.stage_harness import Ingested

needs_r = pytest.mark.skipif(ref.RSCRIPT is None, reason="Rscript is not installed")


# ── fixtures ─────────────────────────────────────────────────────────────────


def _binary(n: int, seed: int, *, p: int = 5, per_unit: int = 1) -> pd.DataFrame:
    """A logistic truth on p standard-normal predictors (the last ones noise), ``per_unit`` rows
    for each ``pid`` sharing a unit effect, the outcome spelled yes/no."""
    rng = np.random.default_rng(seed)
    units = np.repeat(np.arange(n // per_unit), per_unit)[:n]
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[: min(3, p)] = [0.9, -0.6, 0.4][: min(3, p)]
    effect = rng.normal(0, 0.6, n // per_unit + 1)[units] if per_unit > 1 else 0.0
    risk = 1 / (1 + np.exp(-(-0.5 + X @ beta + effect)))
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(1, p + 1)])
    frame["pid"] = units
    frame["y"] = np.where(rng.random(n) < risk, "yes", "no")
    return frame


def _state(frame: pd.DataFrame, *, task: str, target: str, models: list[str], split: SplitSpec,
           repeated: bool = False, purpose: str = "prediction", **extra) -> ProjectState:
    predictors = [c for c in frame.columns if c.startswith("x")]
    roles = {"pid": "identifier", **{c: "covariate" for c in predictors}}
    roles.update(extra.pop("roles", {}))
    grain = (GrainSpec(grain="repeated", id_column="pid") if repeated
             else GrainSpec(grain="one_row_per_unit", id_column="pid"))
    more = {"repeat_kind": {"repeat_kind": "repeats"}} if repeated else {}
    return ProjectState(lens=["clinical"], target=target, task=task, purpose=purpose, roles=roles,
                        missing="complete_case", grain=grain, split=split, models=models,
                        **more, **extra)


def _stages(frame: pd.DataFrame, state: ProjectState, folder):
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "table.csv"
    frame.to_csv(source, index=False)
    table = Ingested(source, folder / "i")
    info = table.run(target_info_stage, state)
    cohort = table.run(cohort_stage, state, {"target_info": info})
    split = table.run(split_stage, state, {"cohort": cohort, "target_info": info})
    design = table.run(design_stage, state, {"split": split, "target_info": info})
    fit = table.run(fit_stage, state, {"design": design, "split": split, "target_info": info})
    return table, info, cohort, split, design, fit


def _assignment(split) -> pd.DataFrame:
    return split.frames["assignment"].sort_values("row_id").reset_index(drop=True)


def _pipeline(family: str, task: str, columns: list[str], n: int):
    spec = DesignSpec(predictors=columns, inputs=columns, categorical=[], numeric=columns,
                      energy=None, impute=False)
    return build_pipeline(spec, get_family(family), task, "prediction", n, len(columns))


def _logit_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray(sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(
        disp=0, method="newton", tol=1e-12, maxiter=200).params)


def _logit_proba(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    p = 1 / (1 + np.exp(-(sm.add_constant(X, has_constant="add") @ beta)))
    return np.column_stack([1 - p, p])


# ── 1 · a strictly proper primary; AUC and C the customary headline ──────────


def test_1_the_primary_is_strictly_proper_and_auc_or_c_is_reported_as_the_customary_headline(tmp_path):
    """MODELING_SEQUENCE ruling 4; Van Calster et al. (STRATOS TG6, arXiv:2412.10288): "AUROC,
    AUPRC, and pAUROC are not strictly proper"; "Because Brier and R-squared variants … measure
    overall performance and are strictly proper, they are useful to compare different models on
    the same dataset". Each task's primary is a mean of a proper per-row loss; AUC or C is always
    reported, labeled the customary headline, with the one-line tension (north star 5).

    Measured: the fit stage on a binary table (linear and elastic net, no holdout). Reference: each
    fold's log loss refit independently (statsmodels' Newton logistic MLE to 10⁻¹² on the split's
    own folds, the loss summed in ``references_ms6.log_loss``), to 10⁻⁴: the engine's logistic fit
    is scikit-learn's Newton–Cholesky, which stops at its default gradient tolerance of 10⁻⁴."""
    assert PRIMARY == {"regression": "mse", "binary": "log_loss", "multiclass": "log_loss",
                       "ordinal": "rps", "time_to_event": "brier_t"}
    assert HEADLINE == {"binary": "auc", "ordinal": "c_index", "time_to_event": "c_index"}
    assert tension("binary") == (
        "AUC is the customary headline: it ranks risks without asking whether they are right "
        "(semi-proper), so the models were compared, chosen and declared on log loss, a strictly "
        "proper score (Van Calster et al., STRATOS TG6).")
    assert tension("ordinal") == (
        "C-index is the customary headline: it ranks risks without asking whether they are right "
        "(semi-proper), so the models were compared, chosen and declared on the ranked probability "
        "score, a strictly proper score (Van Calster et al., STRATOS TG6).")
    assert tension("time_to_event") == (
        "C-index is the customary headline: it ranks risks without asking whether they are right "
        "(semi-proper), so the models were compared, chosen and declared on the Brier score at the "
        "horizon, a strictly proper score (Van Calster et al., STRATOS TG6).")
    assert tension("regression") is None and tension("multiclass") is None

    frame = _binary(300, 1)
    st = _state(frame, task="binary", target="y", models=["linear", "elastic_net"], event="yes",
                split=SplitSpec(holdout=0.0, seed=4, folds=5))
    *_, split, _, fit = _stages(frame, st, tmp_path)
    data = fit.data
    assert data["primary_metric"] == "log_loss" and data["ranking"] == "lowest log loss"
    assert data["headline_metric"] == "auc" and data["headline_label"] == "customary headline"
    assert data["tension"] == tension("binary")
    # Compared, chosen and declared on the primary; the headline reported for every family.
    assert all(c["metric"] == "log_loss" for c in data["comparisons"])
    assert data["selection"]["metric"] == "log_loss" and data["result"]["metric"] == "log_loss"
    assert set(data["selection"]["extras"]) == {"auc"}
    for m in data["models"]:
        assert m["versus_baseline"]["metric"] == "log_loss"
        assert m["cv"]["auc"]["estimate"] is not None
    # Each fold's log loss, refit independently on the split's folds.
    a = _assignment(split)
    rows = frame.iloc[a["row_id"].to_numpy()]
    X, yy = rows[[f"x{i}" for i in range(1, 6)]].to_numpy(), (rows["y"] == "yes").to_numpy().astype(int)
    folds = a["fold"].to_numpy()
    linear = next(m for m in data["models"] if m["family"] == "linear")
    for k in range(5):
        beta = _logit_fit(X[folds != k], yy[folds != k])
        expected = ref.log_loss(yy[folds == k], _logit_proba(beta, X[folds == k]), [0, 1])
        assert linear["cv"]["log_loss"]["folds"][k] == pytest.approx(expected, abs=1e-4)


def test_1_a_miscalibrated_family_with_the_same_ranking_is_never_chosen():
    """Why the primary must be strictly proper: an overconfident family whose probabilities rank the
    rows exactly as a calibrated one's (a monotone transform: logit × 3) has the same AUC, so AUC
    cannot choose between them; log loss chooses the calibrated one in every resample.

    Measured: BBC-CV's choice (``selection_optimism``) on fixed out-of-fold predictions. Reference:
    the two families' log losses and AUCs written out (``references_ms6``), and the truth: the
    calibrated family's probabilities are the true risks."""
    rng = np.random.default_rng(5)
    n = 400
    lp = rng.normal(0, 1, n)
    truth = 1 / (1 + np.exp(-lp))
    y = (rng.random(n) < truth).astype(int)
    sharp = 1 / (1 + np.exp(-3 * lp))
    preds = {"calibrated": np.column_stack([1 - truth, truth]),
             "overconfident": np.column_stack([1 - sharp, sharp])}
    assert ref.auc(y == 1, truth) == pytest.approx(ref.auc(y == 1, sharp), abs=1e-12)
    assert ref.log_loss(y, preds["calibrated"], [0, 1]) < ref.log_loss(y, preds["overconfident"], [0, 1])
    oof = _fixed_oof("binary", y, preds)
    results = {k: _summary_stub(ref.log_loss(y, v, [0, 1])) for k, v in preds.items()}
    found = selection_optimism("binary", "log_loss", results, oof, replicates=300, seed=2)
    assert found["best"] == "calibrated" and found["wins"] == {"calibrated": 300, "overconfident": 0}


class _summary_stub:
    """What ``selection_optimism`` reads of each family's cross-validation: its estimate."""

    def __init__(self, estimate: float):
        self.estimate = estimate

    def summary(self, task):
        return {"log_loss": {"estimate": self.estimate}, "auc": {"estimate": None}}


def _fixed_oof(task: str, y: np.ndarray, preds: dict[str, np.ndarray], units=None) -> OutOfFold:
    """An ``OutOfFold`` holding fixed out-of-fold predictions for every row."""
    n = len(y)
    everything = np.ones(n, dtype=bool)
    oof = OutOfFold(task, np.zeros((n, 1)), y, [(0, everything, everything)], units=units)
    for k, v in preds.items():
        oof.predictions[k] = v
    return oof


@needs_r
def test_1_the_brier_score_at_the_horizon_matches_graf_and_r_survival(tmp_path):
    """The time-to-event primary: Graf et al.'s (1999) Brier score at a horizon h, each row's risk
    by h from a Cox model and Breslow's baseline hazard on the training rows.

    Measured: the engine's Cox fit (``fit_pipeline``, which keeps the baseline hazard), its risks by
    h (``metrics.predict``) and ``performance.brier_at`` on held-aside rows. References: (a) Graf's
    formula and Breslow's baseline written out with loops (``references_ms6``) on the engine's own
    linear predictors, to 10⁻¹²; (b) R: ``coxph`` and ``survfit(…, ctype = 1)`` give each row's
    risk by h, and Graf's formula is computed in R from ``survfit``'s Kaplan–Meier of the
    censoring, to 10⁻⁶ (two Cox solvers); (c) the same R weights reproduce ``survival::brier``'s
    ratio estimator (which divides by Σw instead of n), to 10⁻¹⁰, so the R weights are the
    package's."""
    rng = np.random.default_rng(23)
    n = 400
    X = rng.normal(size=(n, 2))
    t_event = rng.exponential(np.exp(-(X @ np.array([0.7, -0.4]))) * 4)
    t_cens = rng.uniform(0.5, 9, n)
    time = np.minimum(t_event, t_cens)
    event = t_event <= t_cens
    horizon = 2.5
    assert not np.any(np.isclose(time, horizon))
    train = np.arange(n) < 250
    frame = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "time": time, "event": event.astype(int)})
    y = perf_outcome(event, time)
    pipe = _pipeline("cox", "time_to_event", ["x1", "x2"], int(train.sum()))
    fitted = fit_pipeline(pipe, frame.loc[train, ["x1", "x2"]], y[train])
    from turbotab.core.models.metrics import predict

    both = predict("time_to_event", fitted, frame.loc[~train, ["x1", "x2"]], horizon=horizon)
    lp_train = np.asarray(fitted.predict(frame.loc[train, ["x1", "x2"]]))
    by_hand = ref.breslow_risk(time[train], event[train], lp_train, both[:, 0], horizon)
    assert np.max(np.abs(both[:, 1] - by_hand)) < 1e-12
    brier = perf.brier_at(y[~train], both[:, 1], horizon)
    assert brier == pytest.approx(ref.graf_brier(time[~train], event[~train], both[:, 1], horizon),
                                  abs=1e-12)
    script = f'''
suppressMessages({{library(survival); library(jsonlite)}})
tr <- read.csv("train.csv"); te <- read.csv("test.csv"); h <- {horizon}
fit <- coxph(Surv(time, event) ~ x1 + x2, data = tr)
sf <- survfit(fit, newdata = te, ctype = 1)
risk <- 1 - summary(sf, times = h, extend = TRUE)$surv[1, ]
dtime <- te$time; dstat <- te$event
mindiff <- min(diff(sort(unique(dtime))))
ctime <- dtime + ifelse(dstat == 0, mindiff / 2, 0)
c0 <- survfit0(survfit(Surv(ctime, 1 - dstat) ~ 1))
w <- ifelse(ctime < h & dstat == 0, 0, 1 / c0$surv[findInterval(pmin(ctime, h), c0$time)])
b <- ifelse(ctime > h, risk^2, (dstat - risk)^2)
pkg <- brier(fit, times = h, newdata = te)$brier
write(toJSON(list(graf = sum(w * b) / nrow(te), ratio = sum(w * b) / sum(w), package = pkg,
                  risk = risk), digits = I(17), auto_unbox = TRUE), "out.json")
'''
    r = ref.run_r(script, {"train": frame[train], "test": frame[~train]})
    assert r["ratio"] == pytest.approx(r["package"], abs=1e-10)
    assert np.max(np.abs(np.asarray(r["risk"]) - both[:, 1])) < 1e-6
    assert brier == pytest.approx(r["graf"], abs=1e-6)


def test_1_with_delayed_entry_the_time_to_event_primary_falls_back_to_c_and_says_so(tmp_path):
    """With delayed entry Graf's censoring weights would need the truncation distribution too,
    which is not built: the Brier score at the horizon is not computed (never a number that
    ignores the truncation), the comparison falls back to Harrell's C, and the fit says the choice
    rests on a semi-proper score. Reference: the rule as stated."""
    rng = np.random.default_rng(3)
    n = 240
    X = rng.normal(size=(n, 2))
    entry = rng.uniform(0, 1, n)
    t_event = entry + rng.exponential(np.exp(-(X @ np.array([0.6, -0.4]))) * 4)
    t_cens = entry + rng.uniform(0.5, 8, n)
    frame = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "pid": np.arange(n), "entry": entry,
                          "time": np.minimum(t_event, t_cens),
                          "dead": (t_event <= t_cens).astype(int)})
    assert np.isnan(perf.brier_at(perf_outcome(frame["dead"].to_numpy() == 1,
                                               frame["time"].to_numpy(), entry),
                                  np.full(n, 0.3), 2.0))
    st = _state(frame, task="time_to_event", target="dead", models=["cox"], event="1",
                split=SplitSpec(holdout=0.0, seed=2, folds=5),
                follow_up=FollowUpSpec(time_column="time", entry_column="entry",
                                       prediction_horizon=2.0))
    *_, fit = _stages(frame, st, tmp_path)
    data = fit.data
    assert data["primary_metric"] == "c_index" and data["ranking"] == "highest C-index"
    assert data["tension"] == (
        "Harrell's C is the only score here: with delayed entry the Brier score at the horizon is "
        "not computed (its censoring weights would need the truncation distribution), so the "
        "models were compared, chosen and declared on a semi-proper score.")
    model = data["models"][0]
    assert model["cv"]["brier_t"]["estimate"] is None or math.isnan(model["cv"]["brier_t"]["estimate"])
    assert model["versus_baseline"]["metric"] == "c_index"
    assert data["result"]["metric"] == "c_index"


def perf_outcome(event: np.ndarray, time: np.ndarray, entry=None) -> np.ndarray:
    from turbotab.core.models.survival import survival_outcome

    return survival_outcome(event.astype(int), time, entry)


# ── 2 · repeated k-fold is the comparison substrate; the corrected repeated t ─


def test_2_the_corrected_repeated_kfold_t_matches_a_numpy_hand_computation():
    """Bouckaert & Frank (PAKDD 2004), Nadeau & Bengio (2003): over r repeats of K folds,
    ``t = mean(x) / √((1/(rK) + n₂/n₁)·σ̂²)`` on ``rK − 1`` df. Measured: ``corrected_t`` on
    random fold scores; reference: the formula as loops (``references_ms6``), to 10⁻¹⁰."""
    rng = np.random.default_rng(8)
    for r, K in ((10, 5), (10, 10), (3, 5)):
        a, b = rng.normal(0.6, 0.05, r * K), rng.normal(0.62, 0.05, r * K)
        n_train, n_test = rng.integers(200, 240, r * K), rng.integers(50, 60, r * K)
        share = float(np.mean(n_test / n_train))
        for lower in (True, False):
            got = corrected_t(a, b, test_share=share, repeats=r, higher_is_better=not lower)
            t, df, se = ref.corrected_repeated_t(a, b, n_train, n_test, lower_is_better=lower)
            assert got.df == df == r * K - 1
            assert abs(got.t - t) < 1e-10 and abs(got.se - se) < 1e-10


def test_2_families_are_compared_on_repeated_kfold_whatever_validation_leads(tmp_path):
    """MODELING_SEQUENCE ruling 4: "Families are compared on repeated k-fold." The review: "Whatever
    validation supplies the headline, run repeated k-fold (≥ 10 × K) as the comparison substrate.
    Paired differences (corrected repeated k-fold t, rK − 1 df), the versus-baseline verdict and
    BBC-CV all run on it."

    Measured: the fit stage under bootstrap validation (the headline is optimism-corrected), two
    families, no holdout. References: the substrate's first repeat is the split's own folds; every
    fold's log loss refit independently (statsmodels; 10⁻⁴, scikit-learn's solver tolerance); the
    t statistic, its SE and df from
    the formula written out (``references_ms6``) on the artifact's fold scores and the folds' row
    counts, to 10⁻¹⁰."""
    frame = _binary(250, 3)
    st = _state(frame, task="binary", target="y", models=["linear", "elastic_net"], event="yes",
                split=SplitSpec(holdout=0.0, seed=6, folds=5, validation="bootstrap", n_boot=20))
    *_, split, _, fit = _stages(frame, st, tmp_path)
    data = fit.data
    assert data["comparison"] == {
        "folds": 5, "repeats": 10, "shared": 1,
        "method": "the corrected repeated k-fold t (Nadeau & Bengio 2003; Bouckaert & Frank 2004)",
        "note": None}
    a = _assignment(split)
    rows = frame.iloc[a["row_id"].to_numpy()]
    X = rows[[f"x{i}" for i in range(1, 6)]].to_numpy()
    yy = (rows["y"] == "yes").to_numpy().astype(int)
    columns, shared, _ = comparison_folds([a["fold"].to_numpy()], validation="bootstrap",
                                          scheme="random", n=len(yy), strata=yy, folds=5, seed=6)
    assert shared == 1 and len(columns) == COMPARISON_REPEATS
    assert np.array_equal(columns[0], a["fold"].to_numpy())
    models = {m["family"]: m for m in data["models"]}
    for m in models.values():
        assert len(m["compared_on"]["folds"]) == 50 and m["compared_on"]["repeats"] == 10
        assert m["versus_baseline"]["df"] == 49
        assert m["optimism"]["n_boot"] == 20  # the headline stays the bootstrap's
    n_train, n_test, expected = [], [], []
    for column in columns:
        for k in range(5):
            beta = _logit_fit(X[column != k], yy[column != k])
            expected.append(ref.log_loss(yy[column == k], _logit_proba(beta, X[column == k]), [0, 1]))
            n_train.append(int((column != k).sum()))
            n_test.append(int((column == k).sum()))
    assert np.max(np.abs(np.asarray(models["linear"]["compared_on"]["folds"]) - expected)) < 1e-4
    cmp = data["comparisons"][0]
    assert (cmp["a"], cmp["b"], cmp["df"], cmp["folds"], cmp["repeats"]) == ("linear", "elastic_net", 49, 50, 10)
    t, df, se = ref.corrected_repeated_t(models["linear"]["compared_on"]["folds"],
                                         models["elastic_net"]["compared_on"]["folds"],
                                         n_train, n_test, lower_is_better=True)
    assert abs(cmp["t"] - t) < 1e-10 and abs(cmp["se"] - se) < 1e-10 and df == 49
    assert data["comparisons_note"] == (
        "Each pairwise interval is descriptive: none is corrected for how many pairs there are, "
        "and the choice among the families is corrected by bootstrap bias-corrected "
        "cross-validation instead.")
    lo, hi = cmp["ci_low"], cmp["ci_high"]
    ahead = "not distinguishable on these rows" if lo <= 0 <= hi else (
        f"{'Linear model' if cmp['difference'] >= 0 else 'Elastic net'} is ahead on these rows")
    assert cmp["sentence"] == (
        f"Linear model against Elastic net: log loss differs by {abs(cmp['difference']):.3f} (95% "
        f"interval of the difference {_m(lo)} to {_m(hi)}; corrected repeated k-fold t "
        f"{_m(cmp['t'], 2)} on 49 df, 10 × 5 folds): {ahead}.")


def _m(x: float, places: int = 3) -> str:
    return f"{x:.{places}f}".replace("-", "−")


@pytest.mark.parametrize("validation,extra,repeats,shared", [
    ("kfold", {}, 10, 1),
    ("repeated_kfold", {"repeats": 3}, 10, 3),
    ("repeated_kfold", {"repeats": 12}, 12, 12),
    ("internal_external", {"cluster": "site"}, 10, 0),
])
def test_2_the_substrate_is_at_least_10_by_k_under_every_validation(tmp_path, validation, extra,
                                                                    repeats, shared):
    """The substrate's size under each validation answer: 10 × K at least, the split's repeats
    first (they are its own); internal–external validation's folds are sites, so the substrate is
    drawn beside them. Reference: the rule as stated, counted on the artifact."""
    frame = _binary(160, 9, p=3)
    frame["site"] = np.repeat([f"s{i}" for i in range(4)], 40)
    st = _state(frame, task="binary", target="y", models=["linear"], event="yes",
                roles={"site": "design"},
                split=SplitSpec(holdout=0.0, seed=1, folds=5, validation=validation, **extra))
    *_, fit = _stages(frame, st, tmp_path)
    comparison = fit.data["comparison"]
    assert (comparison["repeats"], comparison["shared"], comparison["folds"]) == (repeats, shared, 5)
    model = fit.data["models"][0]
    assert len(model["compared_on"]["folds"]) == 5 * repeats
    assert model["versus_baseline"]["df"] == 5 * repeats - 1


# ── 3 · BBC-CV corrects the choice; with no holdout it is the result ─────────


@pytest.mark.parametrize("units", [False, True], ids=["rows", "units"])
def test_3_bbc_cv_matches_an_independent_implementation_on_fixed_predictions(units):
    """Tsamardinos, Greasidou & Borboudakis (Mach Learn 2018;107:1895), algorithm 5: resample the
    out-of-fold predictions, choose the best configuration on the resample, score it on the rows
    left out. Measured: ``selection_optimism`` on three families' fixed out-of-fold probabilities,
    resampled by row and by unit. Reference: the algorithm written with loops
    (``references_ms6.bbc_cv``), the same seed and draws: the corrected log loss, its percentile
    interval, every family's wins and the AUC corrected for the same choices, to 10⁻¹⁰."""
    rng = np.random.default_rng(31)
    n = 180
    lp = rng.normal(0, 1, n)
    y = np.where(rng.random(n) < 1 / (1 + np.exp(-lp)), "yes", "no")
    preds = {}
    for k, (noise, scale) in {"a": (0.8, 1.0), "b": (1.0, 0.7), "c": (0.9, 1.4)}.items():
        q = 1 / (1 + np.exp(-scale * (lp + rng.normal(0, noise, n))))
        preds[k] = np.column_stack([1 - q, q])
    groups = np.repeat(np.arange(n // 3), 3).astype(str) if units else None
    oof = _fixed_oof("binary", y, preds, units=groups)
    results = {k: _summary_stub(ref.log_loss(y, v, ["no", "yes"])) for k, v in preds.items()}
    got = selection_optimism("binary", "log_loss", results, oof, replicates=500, seed=13,
                             extras=["auc"])
    want = ref.bbc_cv(y, preds, ["no", "yes"], metric="log_loss", B=500, seed=13, units=groups,
                      extra="auc")
    assert got["replicates"] == want["replicates"] and got["wins"] == want["wins"]
    assert abs(got["corrected"] - want["corrected"]) < 1e-10
    assert abs(got["corrected_low"] - want["low"]) < 1e-10
    assert abs(got["corrected_high"] - want["high"]) < 1e-10
    extra = got["extras"]["auc"]
    assert abs(extra["corrected"] - want["extra"]["corrected"]) < 1e-10
    assert abs(extra["corrected_low"] - want["extra"]["low"]) < 1e-10
    assert got["by_unit"] is units


def test_3_with_no_holdout_the_declared_result_is_the_selection_corrected_estimate(tmp_path):
    """MODELING_SEQUENCE §1 row 12 (b): "No holdout: the selection-corrected estimate; only a family
    declared before any score was seen may report its own corrected score." Measured: the fit
    stage with three families and with one. Reference: the selection artifact's own BBC-CV numbers
    (checked against an independent implementation above) and the one family's cross-validated
    estimate; the sentences as written here."""
    frame = _binary(240, 12)
    st = _state(frame, task="binary", target="y", models=["linear", "elastic_net", "boosted_trees"],
                event="yes", split=SplitSpec(holdout=0.0, seed=2, folds=5))
    *_, fit = _stages(frame, st, tmp_path / "three")
    data = fit.data
    sel, result = data["selection"], data["result"]
    assert result["basis"] == "selection_corrected" and result["family"] == sel["best"]
    assert result["estimate"] == sel["corrected"]
    labels = {m["family"]: m["label"] for m in data["models"]}
    assert result["sentence"] == (
        f"{labels[sel['best']]} was chosen among 3 families on cross-validation with no rows held "
        f"out, so the result is the selection-corrected estimate, not its own score: "
        f"selection-corrected log loss {_m(sel['corrected'])} (95% interval "
        f"{_m(sel['corrected_low'])} to {_m(sel['corrected_high'])}) by bootstrap bias-corrected "
        f"cross-validation over 3 families (Tsamardinos et al. 2018; {sel['replicates']:,} "
        f"resamples): {PROCEDURE}.")
    assert {c["relation"] for c in data["chain"]} >= {"choice_among_families_bbc",
                                                      "no_holdout_declares_corrected"}
    st1 = st.model_copy(update={"models": ["linear"]})
    *_, fit1 = _stages(frame, st1, tmp_path / "one")
    one = fit1.data
    entry = one["models"][0]["cv"]["log_loss"]
    assert one["selection"] is None and one["result"]["basis"] == "own_score"
    assert one["result"]["sentence"] == (
        f"Linear model was the only family fitted, declared before any score was seen, so its own "
        f"score is the result: cross-validated log loss {_m(entry['estimate'])} (95% interval "
        f"{_m(entry['ci_low'])} to {_m(entry['ci_high'])}) by 5-fold cross-validation: {PROCEDURE}.")


def _record(seq: int, decision) -> d.DecisionRecord:
    return d.DecisionRecord(id=f"{seq:032x}", seq=seq, at=datetime(2026, 10, 5, tzinfo=timezone.utc),
                            decision=decision)


def test_3_the_winners_own_score_as_the_result_is_refused_with_its_exit():
    """MODELING_SEQUENCE §4: "Winner's own corrected score as 'the result' (no holdout) — refuse;
    report the selection-corrected estimate". Dropping the compared families after their scores
    were seen would leave the winner's own score as the result, so it is refused, with the exit
    that keeps them; with held-out rows, under inference, or after a new seal it is not."""
    families = ["linear", "elastic_net", "boosted_trees"]
    records = [_record(1, d.SetTarget(column="y")), _record(2, d.SetSplit(holdout=0.0, seed=1)),
               _record(3, d.SelectModels(models=families))]
    shown = {"models": [{"family": k, "cv": {"log_loss": {}}} for k in families]}
    state = ProjectState(target="y", task="binary", purpose="prediction", models=families,
                         split=SplitSpec(holdout=0.0, seed=1))
    ctx = {"state": state, "records": records, "shown": lambda stage: shown, "task": "binary"}
    with pytest.raises(Refusal) as refused:
        validate(d.SelectModels(models=["boosted_trees"]), ctx)
    assert refused.value.code == "compared_families_stay"
    assert str(refused.value.message) == (
        "These families' scores were compared on these rows with none held out: `linear`, "
        "`elastic_net`. Dropping them would make the remaining family's own score the result, "
        "which flatters the choice (Tsamardinos et al. 2018). Keep them: the result is then the "
        "selection-corrected estimate, and the best family is still the one deployed.")
    exit_ = refused.value.exits[0]
    assert exit_["label"] == "Keep the compared families"
    assert exit_["decision"]["models"] == ["boosted_trees", "linear", "elastic_net"]
    validate(exit_["decision"], ctx)  # the exit is accepted
    validate(d.SelectModels(models=[*families, "featurewise"]), ctx)  # adding is never refused
    held = state.model_copy(update={"split": SplitSpec(holdout=0.2, seed=1)})
    validate(d.SelectModels(models=["boosted_trees"]), {**ctx, "state": held})
    inference = state.model_copy(update={"purpose": "inference"})
    validate(d.SelectModels(models=["boosted_trees"]), {**ctx, "state": inference})
    resealed = [*records, _record(4, d.SetSplit(holdout=0.0, seed=2))]
    validate(d.SelectModels(models=["boosted_trees"]), {**ctx, "records": resealed})


# ── 4 · bootstrap optimism: B ≥ 500 with its compute shown first; rms::validate ─


def test_4_the_bootstrap_defaults_to_500_and_its_compute_is_estimated_before_it_runs(tmp_path, monkeypatch):
    """Collins et al. (BMJ 2024): "we generally recommend at least 500 bootstraps." Measured: the
    defaults; the seal plan's option; the shelf's measured estimate, which multiplies one timed fit
    by every refit the fit stage will make (captured from the call), before the fit runs; and the
    concern a smaller B states. Reference: the rule's numbers, counted."""
    assert d.OPTIMISM_BOOT == 500
    assert SplitSpec(holdout=0.0).n_boot == 500 and d.SetSplit(holdout=0.0).n_boot == 500
    plan = validation_plan("prediction", 400)
    boot = next(o for o in plan.options if o.validation == "bootstrap")
    assert boot.cost == "about 500 refits of each family, besides 50 for the comparisons' 10 × 5-fold repeats"
    assert next(o for o in plan.options if o.validation == "kfold").cost == (
        "50 refits of each family: the score's 5-fold run is the first of the comparisons' 10 × "
        "5-fold repeats")
    frame = _binary(200, 4)
    st = _state(frame, task="binary", target="y", models=["linear"], event="yes",
                split=SplitSpec(holdout=0.0, seed=1, folds=5, validation="bootstrap"))
    folder = tmp_path / "shelf"
    folder.mkdir()
    frame.to_csv(folder / "t.csv", index=False)
    table = Ingested(folder / "t.csv", folder / "i")
    info = table.run(target_info_stage, st)
    cohort = table.run(cohort_stage, st, {"target_info": info})
    split = table.run(split_stage, st, {"cohort": cohort, "target_info": info})
    seen = {}
    import turbotab.core.models.cost as cost

    real = cost.estimate_fits

    def spy(store, state, task, train_ids, families, folds, **kw):
        seen["folds"] = folds
        return real(store, state, task, train_ids, families, folds, **kw)

    monkeypatch.setattr(cost, "estimate_fits", spy)
    shelf = table.run(shelf_stage, st, {"cohort": cohort, "target_info": info, "split": split})
    assert seen["folds"] == 5 * 10 + 500
    linear = next(f for f in shelf["families"] if f["key"] == "linear")
    assert linear["estimate_seconds"] is not None and linear["estimate"]
    said = voice.sentence_for(d.SetSplit(holdout=0.0, seed=1, folds=5, validation="bootstrap"), st, {})
    assert "Harrell's bootstrap (`500` resamples, the whole pipeline refit on each)" in said
    st200 = st.model_copy(update={"split": SplitSpec(holdout=0.0, seed=1, folds=5,
                                                     validation="bootstrap", n_boot=40)})
    *_, fit = _stages(frame, st200, tmp_path / "fit")
    assert ("40 bootstrap resamples: Collins et al. (2024) recommend at least 500; 200 is the "
            "fewest Steyerberg found estimates the optimism with minor sampling variability."
            in fit.data["models"][0]["concerns"])
    assert "bootstrap_resamples" in {c["relation"] for c in fit.data["chain"]}


@needs_r
def test_4_optimism_corrected_brier_and_dxy_match_rms_validate():
    """Harrell's bootstrap optimism correction of a logistic model against R's ``rms::validate``
    (``lrm``, B = 500). Both are Monte Carlo estimates with their own random draws, so they agree
    within 3 standard errors of their difference: √2 × the engine's SD of the per-resample
    optimism over √500 (the two have the same resampling distribution). The apparent values are
    deterministic and agree to the MLE solvers' tolerance."""
    rng = np.random.default_rng(11)
    n = 250
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(1, 6)])
    risk = 1 / (1 + np.exp(-(-0.8 + X.to_numpy() @ np.array([0.9, -0.6, 0.4, 0.2, 0.0]))))
    y = (rng.random(n) < risk).astype(int)
    pipe = _pipeline("linear", "binary", list(X.columns), n)
    opt = optimism_bootstrap("binary", lambda: clone(pipe),
                             lambda m, Xb, yb, u: fit_pipeline(m, Xb, yb), X, y, n_boot=500, seed=3)
    assert opt.n_boot == 500 and opt.n_ok == 500 and opt.refused is None
    script = '''
suppressMessages({library(rms); library(jsonlite)})
d <- read.csv("data.csv")
set.seed(7)
f <- lrm(y ~ x1 + x2 + x3 + x4 + x5, data = d, x = TRUE, y = TRUE)
v <- rms::validate(f, B = 500)
out <- list(dxy = v["Dxy", "index.corrected"], brier = v["B", "index.corrected"],
            dxy_orig = v["Dxy", "index.orig"], brier_orig = v["B", "index.orig"], n = v["Dxy", "n"])
write(toJSON(lapply(out, unname), digits = I(17), auto_unbox = TRUE), "out.json")
'''
    r = ref.run_r(script, {"data": X.assign(y=y)})
    assert r["n"] == 500
    for key in ("dxy", "brier"):
        e = opt.estimates[key]
        assert e.apparent == pytest.approx(r[f"{key}_orig"], abs=1e-8)
        tolerance = 3 * math.sqrt(2) * e.optimism_sd / math.sqrt(opt.n_ok)
        assert abs(e.corrected - r[key]) < tolerance, (key, e.corrected, r[key], tolerance)


# ── 5 · calibration at a horizon, and by level; else "not assessed" ──────────


@needs_r
def test_5_calibration_at_the_horizon_matches_r_survival():
    """McLernon et al. (Ann Intern Med 2023): time-to-event predictions evaluated "for the event
    occurring by the end of a fixed time horizon". Measured: ``horizon_calibration`` (observed
    1 − KM(h) with Greenwood's SE, O/E, the calibration slope, the risk groups). Reference: R's
    ``survfit`` (Kaplan–Meier and its Greenwood SE, by group) and ``coxph`` of the outcome on
    log(−log(1 − risk)), to 10⁻⁸ (Kaplan–Meier) and 10⁻⁶ (two Cox solvers)."""
    rng = np.random.default_rng(41)
    n = 600
    lp = rng.normal(0, 0.8, n)
    t_event = rng.exponential(np.exp(-lp) * 3)
    t_cens = rng.uniform(0.5, 8, n)
    time, event = np.minimum(t_event, t_cens), t_event <= t_cens
    horizon = 2.0
    risk = 1 - np.exp(-(horizon / 3) * np.exp(1.3 * lp))  # deliberately too extreme
    y = perf_outcome(event, time)
    cal = perf.horizon_calibration(y, risk, horizon, groups=5)
    order = np.argsort(risk, kind="stable")
    group = np.empty(n, dtype=int)
    for g, part in enumerate(np.array_split(order, 5)):
        group[part] = g
    frame = pd.DataFrame({"time": time, "event": event.astype(int),
                          "cll": np.log(-np.log(1 - risk)), "grp": group})
    script = f'''
suppressMessages({{library(survival); library(jsonlite)}})
d <- read.csv("data.csv"); h <- {horizon}
s <- summary(survfit(Surv(time, event) ~ 1, data = d), times = h)
f <- coxph(Surv(time, event) ~ cll, data = d)
g <- summary(survfit(Surv(time, event) ~ grp, data = d), times = h)
write(toJSON(list(surv = s$surv, se = s$std.err, slope = unname(coef(f)),
                  slope_se = unname(sqrt(diag(vcov(f)))), groups = 1 - g$surv),
             digits = I(17), auto_unbox = TRUE), "out.json")
'''
    r = ref.run_r(script, {"data": frame})
    assert cal.observed == pytest.approx(1 - r["surv"], abs=1e-8)
    assert cal.expected == pytest.approx(risk.mean(), abs=1e-12)
    assert cal.ratio.estimate == pytest.approx((1 - r["surv"]) / risk.mean(), abs=1e-8)
    assert cal.ratio.se == pytest.approx(r["se"] / (1 - r["surv"]), abs=1e-8)
    assert cal.slope.estimate == pytest.approx(r["slope"], abs=1e-6)
    assert cal.slope.se == pytest.approx(r["slope_se"], abs=1e-6)
    assert [gr.observed for gr in cal.groups] == pytest.approx(r["groups"], abs=1e-8)
    assert cal.slope.estimate < 1 and cal.flagged  # the risks are too extreme, and it says so
    assert cal.concern.startswith("Risks by 2 are too extreme out of fold: calibration slope ")


@needs_r
@pytest.mark.parametrize("task", ["ordinal", "multiclass"])
def test_5_ordinal_and_multiclass_calibration_by_level_match_r_glm(task):
    """Calibration by level: at each ordinal cut-point (P(at or above the level)) or for each
    class (P(the class)), the calibration intercept (``glm(hit ~ offset(qlogis(p)))``) and slope
    (``glm(hit ~ qlogis(p))``), as for a binary outcome. Reference: R's ``glm`` per level, iterated
    to a deviance change of 10⁻¹⁴, to 10⁻⁶."""
    rng = np.random.default_rng(17)
    n = 500
    raw = rng.dirichlet([2, 3, 2], n)
    lp = rng.normal(0, 1, n)
    codes = np.clip(np.digitize(lp + rng.logistic(0, 0.6, n), [-0.6, 0.6]), 0, 2)
    proba = 0.6 * raw + 0.4 * np.eye(3)[codes] * 0.5 + 0.2 / 3  # informative, imperfect
    proba = proba / proba.sum(axis=1, keepdims=True)
    found = perf.level_calibration(task, codes, proba, [0, 1, 2], names=["low", "mid", "high"])
    if task == "ordinal":
        hits = [(codes >= k).astype(int) for k in (1, 2)]
        ps = [proba[:, k:].sum(axis=1) for k in (1, 2)]
        assert [f.level for f in found] == ["mid", "high"]
        assert all(f.kind == "at_or_above" for f in found)
    else:
        hits = [(codes == k).astype(int) for k in range(3)]
        ps = [proba[:, k] for k in range(3)]
        assert [f.level for f in found] == ["low", "mid", "high"]
    frame = pd.DataFrame({**{f"h{i}": h for i, h in enumerate(hits)},
                          **{f"p{i}": p for i, p in enumerate(ps)}})
    script = f'''
suppressMessages(library(jsonlite))
d <- read.csv("data.csv"); out <- list()
for (i in 0:{len(hits) - 1}) {{
  h <- d[[paste0("h", i)]]; p <- d[[paste0("p", i)]]
  tight <- glm.control(epsilon = 1e-14, maxit = 100)
  a <- glm(h ~ 1 + offset(qlogis(p)), family = binomial, control = tight)
  b <- glm(h ~ qlogis(p), family = binomial, control = tight)
  out[[i + 1]] <- list(a = unname(coef(a)[1]), a_se = unname(sqrt(vcov(a)[1, 1])),
                       b = unname(coef(b)[2]), b_se = unname(sqrt(vcov(b)[2, 2])))
}}
write(toJSON(out, digits = I(17), auto_unbox = TRUE), "out.json")
'''
    r = ref.run_r(script, {"data": frame})
    for got, want in zip(found, r):
        assert got.calibration.intercept.estimate == pytest.approx(want["a"], abs=1e-6)
        assert got.calibration.intercept.se == pytest.approx(want["a_se"], abs=1e-6)
        assert got.calibration.slope.estimate == pytest.approx(want["b"], abs=1e-6)
        assert got.calibration.slope.se == pytest.approx(want["b_se"], abs=1e-6)


def test_5_every_task_reports_calibration_or_says_it_was_not_assessed(tmp_path):
    """The record never inherits the word: each family carries its calibration for its task (binary
    and numeric: intercept and slope; ordinal and multiclass: by level; time to event: at the
    horizon) or ``calibration_note`` saying "Calibration not assessed" and why. A declared horizon
    is used and said; a horizon before any event cannot be assessed."""
    rng = np.random.default_rng(29)
    n = 200
    X = rng.normal(size=(n, 2))
    lp = X @ np.array([0.8, -0.5])
    frame = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "pid": np.arange(n)})
    frame["mc"] = np.array(["red", "green", "blue"])[np.clip(np.digitize(lp + rng.normal(size=n), [-0.5, 0.5]), 0, 2)]
    frame["ord"] = np.array(["low", "mid", "high"])[np.clip(np.digitize(lp + rng.logistic(size=n), [-0.7, 0.7]), 0, 2)]
    t_event, t_cens = rng.exponential(np.exp(-lp) * 4), rng.uniform(0.5, 8, n)
    frame["time"], frame["dead"] = np.minimum(t_event, t_cens), (t_event <= t_cens).astype(int)
    split = SplitSpec(holdout=0.0, seed=3, folds=5)
    st = _state(frame, task="multiclass", target="mc", models=["linear"], split=split)
    *_, fit = _stages(frame[["pid", "x1", "x2", "mc"]], st, tmp_path / "mc")
    m = fit.data["models"][0]
    assert m["calibration_note"] is None and len(m["calibration_levels"]) == 3
    st = _state(frame, task="ordinal", target="ord", models=["proportional_odds"], split=split,
                outcome_order=["low", "mid", "high"])
    *_, fit = _stages(frame[["pid", "x1", "x2", "ord"]], st, tmp_path / "ord")
    m = fit.data["models"][0]
    assert m["calibration_note"] is None
    assert [c["level"] for c in m["calibration_levels"]] == ["mid", "high"]
    tte = frame[["pid", "x1", "x2", "time", "dead"]]
    st = _state(frame, task="time_to_event", target="dead", models=["cox"], split=split, event="1",
                follow_up=FollowUpSpec(time_column="time", prediction_horizon=2.0))
    *_, fit = _stages(tte, st, tmp_path / "tte")
    m = fit.data["models"][0]
    assert fit.data["horizon"] == 2.0
    assert fit.data["horizon_note"] == ("Scored and calibrated by `time` = `2`, the declared "
                                       "prediction horizon.")
    assert m["calibration_horizon"]["horizon"] == 2.0 and m["calibration_note"] is None
    said = voice.sentence_for(d.SetFollowUp(column="dead", time_column="time",
                                            prediction_horizon=2.0),
                              st, {})
    assert said == ("`dead` was analyzed as a time to event, each row followed until `time`, at the "
                    "event or when follow-up ended without it; predicted risks were scored and "
                    "calibrated by `time` = `2`, the declared prediction horizon.")
    early = float(np.min(frame["time"])) / 2
    st = st.model_copy(update={"follow_up": FollowUpSpec(time_column="time",
                                                         prediction_horizon=early)})
    *_, fit = _stages(tte, st, tmp_path / "early")
    m = fit.data["models"][0]
    assert m["calibration_horizon"] is None
    assert m["calibration_note"] == ("Calibration not assessed: fewer than 10 scored rows, or no "
                                     "event by the horizon.")
    assert "calibration_not_assessed" in {c["relation"] for c in fit.data["chain"]}


# ── 6 · repeated units: grouped folds, a grouped holdout, a bootstrap by unit ─


def test_6_no_unit_spans_training_and_assessment_rows_in_any_resample():
    """MODELING_SEQUENCE §2: repeated units *imply*, under prediction, grouped folds and a grouped
    holdout, and a bootstrap by unit. A property over 25 random tables (unit sizes 1–6, some unit
    labels missing): no unit has rows on both sides of the holdout, of any fold of any repeat of
    the comparison substrate, of any inner split, of any nested cross-validation fold; every
    bootstrap resample takes a unit's rows together (each copy whole), and its inner splits keep
    the copies together; BBC-CV's drawn and left-out rows share no unit. Observed through the
    engine's own calls (the bootstrap's fit and BBC-CV's scoring are intercepted)."""
    master = np.random.default_rng(2026)
    for trial in range(25):
        rng = np.random.default_rng(int(master.integers(1 << 30)))
        sizes = rng.integers(1, 7, int(rng.integers(30, 60)))
        labels = np.repeat(np.arange(len(sizes)), sizes).astype(object)
        n = len(labels)
        missing = rng.random(n) < 0.03
        labels[missing] = None
        y = rng.integers(0, 2, n)
        unit = np.asarray([f"m{i}" if v is None else str(v) for i, v in enumerate(labels)])

        def whole(train: np.ndarray, test: np.ndarray) -> None:
            assert not set(unit[train]) & set(unit[test]), trial

        frame, _ = draw_split(np.arange(n), holdout=0.25, seed=trial, folds=5, y=y.astype(str),
                              groups=labels, grouped_by="pid")
        frame = frame.sort_values("row_id")
        held = (frame["partition"] == "holdout").to_numpy()
        whole(np.flatnonzero(~held), np.flatnonzero(held))
        train = np.flatnonzero(~held)
        columns, _, _ = comparison_folds([frame.loc[~held, "fold"].to_numpy()],
                                         validation="kfold", scheme="random", n=len(train),
                                         strata=y[train], groups=labels[train], folds=5, seed=trial)
        for column in columns:
            for _, fit_rows, test_rows in fold_pairs(column):
                whole(train[fit_rows], train[test_rows])
        for fit_rows, test_rows in inner_splits(labels, 4, trial, y=y) or []:
            whole(fit_rows, test_rows)
        fold_ids = nested_folds(n, groups=labels, folds=5, rng=np.random.default_rng(trial))
        for f in range(1, 6):
            whole(np.flatnonzero(fold_ids != f), np.flatnonzero(fold_ids == f))
        # The bootstrap: each resample's rows, observed in its fit.
        X = pd.DataFrame({"row": np.arange(n), "x": rng.normal(size=n)})
        _, members = unit_rows(labels, n)
        seen: list[tuple[np.ndarray, np.ndarray]] = []

        def recording_fit(model, Xb, yb, units_b):
            seen.append((Xb["row"].to_numpy(), np.asarray(units_b)))
            return model.fit(Xb, yb)

        from sklearn.linear_model import LogisticRegression

        class Wrapped:
            """A logistic fit on ``x`` alone; the ``row`` column only tells the test which rows."""

            def __init__(self):
                self.m = LogisticRegression()
                self.classes_ = None

            def fit(self, Xa, ya):
                self.m.fit(Xa[["x"]].to_numpy(), ya)
                self.classes_ = self.m.classes_
                return self

            def predict_proba(self, Xa):
                return self.m.predict_proba(Xa[["x"]].to_numpy())

            def predict(self, Xa):
                return self.m.predict(Xa[["x"]].to_numpy())

        optimism_bootstrap("binary", Wrapped, recording_fit, X, y, n_boot=15, seed=trial,
                           groups=labels, final=Wrapped().fit(X, y))
        assert len(seen) == 15  # every resample was fit, and seen
        for rows, units_b in seen:
            counts = pd.Series(unit[rows]).value_counts()
            for u, c in counts.items():
                size = int((unit == u).sum())
                assert c % size == 0, (trial, u, c, size)  # whole copies of each unit
            for fit_rows, test_rows in inner_splits(units_b, 3, trial) or []:
                assert not set(units_b[fit_rows]) & set(units_b[test_rows])
        # BBC-CV: the rows each resample chooses on and scores on.
        preds = {k: np.column_stack([1 - q, q]) for k, q in
                 (("a", rng.uniform(0.2, 0.8, n)), ("b", rng.uniform(0.2, 0.8, n)))}
        oof = _fixed_oof("binary", y, preds, units=labels)
        calls: list[np.ndarray] = []
        original = OutOfFold.score

        def spying(self, metric, key, rows):
            calls.append(np.asarray(rows))
            return original(self, metric, key, rows)

        OutOfFold.score = spying
        try:
            selection_optimism("binary", "log_loss",
                               {k: _summary_stub(ref.log_loss(y, v, [0, 1])) for k, v in preds.items()},
                               oof, replicates=20, seed=trial)
        finally:
            OutOfFold.score = original
        # per resample: the two families on the drawn rows, then the chosen one on the left-out
        for i in range(0, len(calls) - 2, 3):
            drawn, left = calls[i], calls[i + 2]
            assert not set(unit[drawn]) & set(unit[left]), trial


def test_6_repeated_units_reach_every_resample_through_the_fit_stage(tmp_path):
    """The same relation through the real stages: rows repeating within ``pid``, a 20% holdout,
    bootstrap validation, two families. The split holds out whole units and says so in the
    participant flow; the bootstrap resamples units; BBC-CV resamples units; the chain names the
    relation."""
    frame = _binary(360, 21, per_unit=3)
    st = _state(frame, task="binary", target="y", models=["linear", "elastic_net"], event="yes",
                repeated=True,
                split=SplitSpec(holdout=0.2, seed=5, folds=5, validation="bootstrap", n_boot=20))
    *_, split, _, fit = _stages(frame, st, tmp_path)
    assert split.data["grouped_by"] == "pid" and "grouped by `pid`" in split.data["note"]
    a = _assignment(split)
    pid = frame["pid"].to_numpy()[a["row_id"].to_numpy()]
    held = a["partition"].to_numpy() == "holdout"
    assert not set(pid[held]) & set(pid[~held])
    data = fit.data
    assert all(m["optimism"]["resampled"] == "pid units" for m in data["models"])
    assert data["selection"]["by_unit"] is True
    links = {c["relation"]: c for c in data["chain"]}
    assert links["repeated_units_group_resampling"] == {
        "relation": "repeated_units_group_resampling", "because": "rows repeat within `pid`",
        "then": ("every fold of every repeat, the held-out rows, every bootstrap resample, the "
                 "BBC-CV resamples and the nested folds keep each unit's rows together")}


# ── 7 · the procedure's expected performance; p ≫ n and the nested-CV interval ─


def test_7_performance_sentences_describe_the_procedure_and_flag_p_much_greater_than_n(tmp_path):
    """Bates, Hastie & Tibshirani (JASA 2023): cross-validation estimates "the average prediction
    error of models fit on other unseen training sets drawn from the same population", and naive
    intervals undercover at p ≫ n. Measured: every performance sentence; at p/n = 1.5 the label
    and the offered nested cross-validation interval with its refits and measured time; then the
    offer's own decision runs it and the label is gone."""
    frame = _binary(150, 3)
    st = _state(frame, task="binary", target="y", models=["linear"], event="yes",
                split=SplitSpec(holdout=0.0, seed=1, folds=5))
    *_, fit = _stages(frame, st, tmp_path / "narrow")
    m = fit.data["models"][0]
    entry = m["cv"]["log_loss"]
    assert m["performance"] == (
        f"Cross-validated log loss {_m(entry['estimate'])} (95% interval {_m(entry['ci_low'])} to "
        f"{_m(entry['ci_high'])}) by 5-fold cross-validation: {PROCEDURE}.")
    assert PROCEDURE == "the expected performance of this modeling procedure at this sample size"
    assert fit.data["wide"] is None and fit.data["nested_offer"] is None

    rng = np.random.default_rng(77)
    n, p = 40, 60
    X = rng.normal(size=(n, p))
    wide = pd.DataFrame(X, columns=[f"x{i}" for i in range(1, p + 1)])
    wide["pid"] = np.arange(n)
    wide["y"] = X[:, 0] - 0.5 * X[:, 1] + rng.normal(size=n)
    st = _state(wide, task="regression", target="y", models=["elastic_net"],
                split=SplitSpec(holdout=0.0, seed=2, folds=5))
    *_, fit = _stages(wide, st, tmp_path / "wide")
    data = fit.data
    clause = "60 candidate predictors for 40 rows (p/n 1.5, above the 1 this app takes as p ≫ n)"
    assert data["wide"] == clause
    entry = data["models"][0]["cv"]["mse"]
    assert data["models"][0]["performance"] == (
        f"Cross-validated MSE {_m(entry['estimate'])} (95% interval {_m(entry['ci_low'])} to "
        f"{_m(entry['ci_high'])}, {TOO_NARROW}: {clause}) by 5-fold cross-validation: {PROCEDURE}.")
    assert data["result"]["narrow"] == clause and TOO_NARROW in data["result"]["sentence"]
    offer = data["nested_offer"]
    assert offer["fits"] == nested_cv_fits(5) == 50 * 15 + 10 * 5
    assert offer["label"] == ("Run the nested cross-validation interval (Bates, Hastie & "
                              "Tibshirani 2023): 800 refits of each family")
    assert offer["seconds"] is not None and offer["estimate"]
    assert "p_much_greater_n" in {c["relation"] for c in data["chain"]}
    ran = st.model_copy(update={"split": d.SplitSpec(**{k: v for k, v in offer["decision"].items()
                                                        if k != "kind"})})
    assert ran.split.nested_cv is True
    *_, fit2 = _stages(wide, ran, tmp_path / "nested")
    data2 = fit2.data
    nested = data2["models"][0]["nested_cv"]
    assert nested["refused"] is None and nested["fits"] == 800 and nested["reps"] == 50
    assert data2["nested_offer"] is None
    assert TOO_NARROW not in data2["models"][0]["performance"]
    assert data2["result"]["sentence"].endswith(
        f"its own score is the result: MSE {_m(nested['estimate'])} (95% interval "
        f"{_m(nested['ci_low'])} to {_m(nested['ci_high'])}) by nested cross-validation (50 "
        f"repetitions of 5 folds; Bates, Hastie & Tibshirani 2023): {PROCEDURE}.")


@needs_r
def test_7_the_nested_cv_interval_matches_the_authors_code_in_r():
    """Bates, Hastie & Tibshirani's nested cross-validation (their ``nestedcv`` package, R/core.R,
    MIT-licensed; adapted here only to take the folds as given instead of drawing them): the same
    folds, an ordinary least-squares fit and squared error. Measured: ``nested_cv_interval``.
    Reference: the authors' algorithm in R, to 10⁻⁸: the point estimate, its bias, the inflation
    and the interval."""
    rng = np.random.default_rng(99)
    n, K, reps, bias_reps = 62, 5, 4, 2
    X = rng.normal(size=(n, 3))
    yv = X @ np.array([1.0, -0.5, 0.0]) + rng.normal(size=n)
    draws = {"nested": [nested_folds(n, folds=K, rng=rng) for _ in range(reps)],
             "plain": [(np.arange(n) % K + 1)[rng.permutation(n)] for _ in range(bias_reps)]}
    got = nested_cv_interval("regression", "mse", LinearRegression,
                             lambda m, Xf, yf: m.fit(Xf, yf), X, yv, folds=K, reps=reps,
                             bias_reps=bias_reps, draws=draws)
    assert got.fits == reps * (K * (K - 1) // 2 + K) + bias_reps * K
    frame = pd.DataFrame(X, columns=["x1", "x2", "x3"]).assign(y=yv)
    for i, f in enumerate(draws["nested"]):
        frame[f"n{i}"] = f
    for i, f in enumerate(draws["plain"]):
        frame[f"p{i}"] = f
    script = f'''
suppressMessages(library(jsonlite))
d <- read.csv("data.csv"); K <- {K}; alpha <- 0.05
X <- as.matrix(d[, c("x1", "x2", "x3")]); Y <- d$y
fitter <- function(X, Y) lm(Y ~ X)
predictor <- function(fit, X) cbind(1, X) %*% coef(fit)
loss <- function(y_hat, y) (y_hat - y)^2
# Bates, Hastie & Tibshirani, nestedcv R/core.R nested_cv_helper (MIT), the folds given.
helper <- function(X, Y, fold_id, n_folds) {{
  ho_errors <- array(0, dim = c(n_folds, n_folds, nrow(X) %/% n_folds))
  for (f1 in 1:(n_folds - 1)) for (f2 in (f1 + 1):n_folds) {{
    test_idx <- c(which(fold_id == f1), which(fold_id == f2))
    fit <- fitter(X[-test_idx, ], Y[-test_idx])
    preds <- predictor(fit, X)
    ho_errors[f1, f2, ] <- loss(preds[fold_id == f1], Y[fold_id == f1])
    ho_errors[f2, f1, ] <- loss(preds[fold_id == f2], Y[fold_id == f2])
  }}
  out_mat <- matrix(0, n_folds, 2)
  for (f1 in 1:n_folds) {{
    test_idx <- which(fold_id == f1)
    fit <- fitter(X[-test_idx, ], Y[-test_idx])
    e_out <- loss(predictor(fit, X[test_idx, ]), Y[test_idx])
    e_bar_t <- c()
    for (f2 in 1:n_folds) {{ if (f2 == f1) next; e_bar_t <- c(e_bar_t, ho_errors[f2, f1, ]) }}
    out_mat[f1, 1] <- mean(e_bar_t) - mean(e_out)
    out_mat[f1, 2] <- var(e_out) / length(test_idx)
  }}
  all_ho_errs <- c()
  for (f1 in 1:(n_folds - 1)) for (f2 in (f1 + 1):n_folds)
    all_ho_errs <- c(all_ho_errs, ho_errors[f1, f2, ], ho_errors[f2, f1, ])
  list(pivots = out_mat, errs = all_ho_errs)
}}
naive <- function(X, Y, fold_id, n_folds) {{
  errors <- c()
  for (k in 1:n_folds) {{
    fit <- fitter(X[fold_id != k, ], Y[fold_id != k])
    errors <- c(errors, loss(predictor(fit, X[fold_id == k, ]), Y[fold_id == k]))
  }}
  mean(errors)
}}
var_pivots <- c(); ho_errs <- c()
for (i in 0:{reps - 1}) {{
  temp <- helper(X, Y, d[[paste0("n", i)]], K)
  var_pivots <- rbind(var_pivots, temp$pivots); ho_errs <- c(ho_errs, temp$errs)
}}
n_sub <- floor(length(Y) * (K - 1) / K)
infl <- sqrt(max(0, mean(var_pivots[, 1]^2 - var_pivots[, 2]))) / (sd(ho_errs) / sqrt(n_sub))
infl <- max(1, min(infl, sqrt(K)))
cv_means <- sapply(0:{bias_reps - 1}, function(i) naive(X, Y, d[[paste0("p", i)]], K))
bias <- (mean(ho_errs) - mean(cv_means)) * (1 + ((K - 2) / K)^(1.5))
est <- mean(ho_errs) - bias
half <- qnorm(1 - alpha / 2) * sd(ho_errs) / sqrt(length(Y)) * infl
write(toJSON(list(est = est, lo = est - half, hi = est + half, bias = bias, infl = infl,
                  raw = mean(ho_errs), sd = sd(ho_errs)), digits = I(17), auto_unbox = TRUE),
      "out.json")
'''
    r = ref.run_r(script, {"data": frame})
    assert got.refused is None
    for mine, theirs in ((got.estimate, r["est"]), (got.ci_low, r["lo"]), (got.ci_high, r["hi"]),
                         (got.bias, r["bias"]), (got.inflation, r["infl"]),
                         (got.raw_mean, r["raw"]), (got.sd, r["sd"])):
        assert abs(mine - theirs) < 1e-8, (mine, theirs)


# ── the method contracts and the chain (BLUEPRINT §13; MODELING_SEQUENCE §6 chain 6) ─


def test_contracts_declare_every_part_and_their_relations():
    """BLUEPRINT §13: each method enters through a contract in the one registry
    (``turbotab.core.contracts``): slot, data scope, needs, routing per purpose (its option labeled
    customary and sound for each purpose, with its rung), storyboard, sentence, relations; the
    refusal of the winner's own score names the exit the refusal itself offers."""
    import importlib

    from turbotab.core import contracts as C

    assert set(VALIDATION_CONTRACTS) == {"proper_primary", "comparison_substrate", "bbc_cv",
                                         "bootstrap_optimism", "horizon_calibration",
                                         "nested_cv_interval"}
    for key in VALIDATION_CONTRACTS:
        c = C.contract(key)
        assert c.slot == "evaluation" and c.scope == "training_fold"
        assert c.needs and c.storyboard and c.sentence and c.relations and c.sources
        for option in c.options:
            assert set(option.sound) == set(option.rung) == {"prediction", "inference"}
            assert option.customary and all(option.sound.values())
        for r in c.relations:
            module, name = r.enforced_by.split(":")
            assert callable(getattr(importlib.import_module(module), name)), r.enforced_by
    refusal = C.contract("bbc_cv").relation("no_holdout_declares_corrected")
    assert refusal.kind == "conflicts" and refusal.rung == "refused"
    assert refusal.exits == ("Keep the compared families",)
    assert relation_ids() == [
        "proper_score_primary", "families_compared_on_substrate", "repeated_units_group_resampling",
        "choice_among_families_bbc", "no_holdout_declares_corrected", "bootstrap_resamples",
        "time_to_event_horizon", "calibration_not_assessed", "p_much_greater_n"]


def test_chain_6_a_multi_site_cohort_with_repeated_visits_under_prediction(tmp_path):
    """MODELING_SEQUENCE §6 chain 6, its prediction half: "grouped folds and site heterogeneity
    under prediction". Visits repeat within people, people within sites. Asserted: the order
    (the holdout and every fold keep a person whole; the sites are scored by models fit on the
    others); every relation MS6 touches fires and is named in the chain; the participant flow
    says the rows are grouped; and the methods sentences, verbatim."""
    rng = np.random.default_rng(606)
    people = 150
    site_of = rng.integers(0, 6, people)
    visits = rng.integers(2, 4, people)
    pid = np.repeat(np.arange(people), visits)
    site = np.array([f"site_{s}" for s in site_of])[pid]
    n = len(pid)
    X = rng.normal(size=(n, 3))
    person = rng.normal(0, 0.7, people)[pid]
    shift = rng.normal(0, 0.4, 6)[site_of[pid]]
    risk = 1 / (1 + np.exp(-(-0.3 + X @ np.array([0.8, -0.5, 0.0]) + person + shift)))
    frame = pd.DataFrame(X, columns=["x1", "x2", "x3"]).assign(
        pid=pid, site=site, y=np.where(rng.random(n) < risk, "yes", "no"))
    split = SplitSpec(holdout=0.2, seed=8, folds=5, validation="internal_external", cluster="site")
    st = _state(frame, task="binary", target="y", models=["linear", "elastic_net"], event="yes",
                repeated=True, roles={"site": "design"}, split=split)
    *_, sp, _, fit = _stages(frame, st, tmp_path)
    a = _assignment(sp)
    who = pid[a["row_id"].to_numpy()]
    held = a["partition"].to_numpy() == "holdout"
    assert not set(who[held]) & set(who[~held])  # the grouped holdout
    assert "grouped by `pid` so a unit's rows stay together" in sp.data["note"]
    data = fit.data
    for m in data["models"]:
        iecv = m["internal_external"]
        assert iecv["cluster"] == "site" and iecv["metric"] == "log_loss"
        assert len(iecv["clusters"]) == len(set(site[a["row_id"].to_numpy()][~held]))
    assert data["comparison"]["repeats"] == 10 and data["comparison"]["shared"] == 0
    assert data["selection"]["by_unit"] is True
    fired = [c["relation"] for c in data["chain"]]
    assert fired == ["proper_score_primary", "families_compared_on_substrate",
                     "repeated_units_group_resampling", "choice_among_families_bbc"]
    assert data["result"]["basis"] == "holdout"
    said = voice.sentence_for(d.SetSplit(**split.model_dump()), st, {})
    assert said == (
        "A random `20%` of the rows with `y` recorded (seed `8`, stratified by `y`) was held out "
        "for one final score; performance was estimated on the rest by internal–external "
        "validation, each level of `site` held out in turn and scored by models fit on the "
        "others; models were compared with each other and with the no-predictor baseline on "
        "`5`-fold cross-validation repeated `10` times, by the corrected repeated k-fold t "
        "(Nadeau & Bengio 2003; Bouckaert & Frank 2004).")
    no_holdout = st.model_copy(update={"split": SplitSpec(holdout=0.0, seed=8)})
    assert voice.sentence_for(d.SelectModels(models=["linear", "elastic_net"]), no_holdout, {}) == (
        "Two model families were chosen: logistic regression and elastic net; with no rows held "
        "out, the "
        "choice among them is corrected by bootstrap bias-corrected cross-validation (Tsamardinos "
        "et al. 2018), and that selection-corrected estimate is the reported result, not the best "
        "family's own score.")
