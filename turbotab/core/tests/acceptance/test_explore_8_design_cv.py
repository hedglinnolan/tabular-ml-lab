"""EXPLORE 8 · Design-based cross-validation under the surveyed population (MODELING_SEQUENCE §0
ruling 13; Wieczorek, Guerin & McMahon, Stat 2022;11:e454), and no cross-validated score under
inference.

Expected values: NumPy by hand on the same folds (each fold's weighted mean loss, the baseline's
training-fold mean, least squares by ``numpy.linalg.lstsq``), R's ``survey`` (``svymean``'s
linearized standard error and ``svyglm``'s weighted calibration), and a replay of the design
bootstrap's draws. The folds themselves are checked against their definition: whole PSUs within
strata, every stratum in every fold.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.models import design_cv as D
from turbotab.core.tests.acceptance import explore_references as ref
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r
from turbotab.core.tests.graph_runner import GraphRun


def _design_table(seed: int = 0, strata: int = 12, psus: int = 2, per: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(strata):
        for j in range(psus):
            effect = rng.normal(0, 0.5)
            for _ in range(per):
                x1, x2, x3 = rng.normal(size=3)
                w = rng.uniform(500, 5000) * (1 + h % 3)
                y = 1.0 + 0.8 * x1 - 0.5 * x2 + 0.2 * x3 + 0.3 * (h % 4) + effect + rng.normal()
                rows.append({"x1": x1, "x2": x2, "x3": x3, "stratum": 100 + h, "psu": j + 1,
                             "wt": w, "y": y})
    frame = pd.DataFrame(rows)
    frame.insert(0, "pid", np.arange(len(frame)))
    return frame


# ── the folds ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("psus,asked,want", [(2, 5, 2), (6, 5, 5), (3, 2, 2)])
def test_8_folds_keep_whole_psus_within_strata_and_every_stratum_in_every_fold(psus, asked, want):
    """Wieczorek et al.: with a stratified cluster sample "each fold is formed by sampling clusters
    within each stratum". Every PSU (a label within its stratum) lies in one fold; every fold holds
    a PSU of every stratum; K is the folds asked for, at most the fewest PSUs in a stratum."""
    frame = _design_table(1, strata=8, psus=psus, per=5)
    fold, k, note = D.design_folds(frame["stratum"], frame["psu"], len(frame), asked, seed=7)
    assert k == want
    assert (note is None) == (want == asked)
    per_psu = frame.assign(fold=fold).groupby(["stratum", "psu"])["fold"].nunique()
    assert (per_psu == 1).all()
    per_fold = frame.assign(fold=fold).groupby("fold")["stratum"].nunique()
    assert len(per_fold) == k and (per_fold == frame["stratum"].nunique()).all()
    again, _, _ = D.design_folds(frame["stratum"], frame["psu"], len(frame), asked, seed=7)
    assert (again == fold).all()


# ── weighted scores, through the evaluation stage ────────────────────────────


@pytest.fixture(scope="module")
def population_run(tmp_path_factory):
    frame = _design_table()
    folder = tmp_path_factory.mktemp("design")
    frame.to_csv(folder / "t.csv", index=False)
    run = GraphRun(folder / "t.csv", folder / "p")
    roles = {"pid": "identifier", "x1": "covariate", "x2": "covariate", "x3": "covariate",
             "stratum": "design", "psu": "design", "wt": "design"}
    state = d.ProjectState(
        lens=["survey"], target="y", task="regression", purpose="prediction", roles=roles,
        missing="complete_case", grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        split=d.SplitSpec(holdout=0.0, seed=2, folds=5), models=["linear"],
        survey=d.SurveySpec(estimand="population", weight="wt", strata="stratum", psu="psu"))
    out = run.run(state, upto=["evaluation"])
    yield frame, state, out
    run.close()


def _training(frame: pd.DataFrame, out) -> pd.DataFrame:
    ids = out["fit"].objects["comparison"]["train_ids"]
    return frame.set_index(frame.index.astype(np.int64)).loc[ids]


def test_8_a_weighted_score_agrees_with_a_numpy_hand_computation_on_the_same_folds(population_run):
    """Ruling 13: "every loss, calibration and comparison is survey-weighted". Each fold of each
    repeat: least squares refit by hand (``numpy.linalg.lstsq``) on the fold's training rows, the
    held-out rows' squared errors, and their weighted mean Σwℓ/Σw (``explore_references``); the
    baseline's prediction is its training rows' mean. Agreement to 10⁻¹⁰."""
    frame, state, out = population_run
    ev = out["evaluation"].data
    block = ev["design_based"]
    assert block["label"] == "design-based cross-validation"
    assert block["folds"] == 2 and block["repeats"] == D.REPEATS
    rows = _training(frame, out)
    X = np.column_stack([np.ones(len(rows)), rows[["x1", "x2", "x3"]].to_numpy()])
    y, w = rows["y"].to_numpy(), rows["wt"].to_numpy()
    by = {f["family"]: f for f in block["families"]}
    for r in range(D.REPEATS):
        fold, k, _ = D.design_folds(rows["stratum"].to_numpy(), rows["psu"].to_numpy(), len(rows),
                                    5, seed=2 + r)
        lin, base = np.zeros(len(y)), np.zeros(len(y))
        for f in range(k):
            tr, te = fold != f, fold == f
            beta = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
            lin[te] = (y[te] - X[te] @ beta) ** 2
            base[te] = (y[te] - y[tr].mean()) ** 2
        got_lin = by["linear"]["fold_scores"][r * k:(r + 1) * k]
        got_base = by["baseline"]["fold_scores"][r * k:(r + 1) * k]
        assert np.max(np.abs(np.asarray(got_lin) - ref.weighted_fold_means(lin, w, fold))) < 1e-10
        assert np.max(np.abs(np.asarray(got_base) - ref.weighted_fold_means(base, w, fold))) < 1e-10
    assert abs(by["linear"]["estimate"] - float(np.mean(by["linear"]["fold_scores"]))) < 1e-12
    # The paired comparison against the baseline: the corrected repeated k-fold t over the weighted
    # fold scores (Nadeau & Bengio), by hand.
    dif = np.asarray(by["baseline"]["fold_scores"]) - np.asarray(by["linear"]["fold_scores"])
    mean, se = ref.corrected_t(dif, share=1.0)
    vb = by["linear"]["versus_baseline"]
    assert abs(vb["mean"] - mean) < 1e-12 and abs(vb["se"] - se) < 1e-10


@needs_r
def test_8_the_pooled_weighted_loss_carries_svymeans_linearized_error(population_run, tmp_path):
    """The first repeat's out-of-fold squared errors: their weighted mean and its design-based
    standard error against R ``survey::svymean(~loss, svydesign(ids = ~psu, strata = ~stratum,
    weights = ~wt, nest = TRUE))``, to 10⁻¹⁰; the weighted calibration slope and intercept against
    ``svyglm(y ~ yhat)`` and the weighted mean residual."""
    frame, state, out = population_run
    rows = _training(frame, out)
    X = np.column_stack([np.ones(len(rows)), rows[["x1", "x2", "x3"]].to_numpy()])
    y = rows["y"].to_numpy()
    fold, k, _ = D.design_folds(rows["stratum"].to_numpy(), rows["psu"].to_numpy(), len(rows), 5,
                                seed=2)
    yhat = np.zeros(len(y))
    for f in range(k):
        tr, te = fold != f, fold == f
        yhat[te] = X[te] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
    loss = (y - yhat) ** 2
    data = pd.DataFrame({"loss": loss, "y": y, "yhat": yhat, "wt": rows["wt"].to_numpy(),
                         "stratum": rows["stratum"].to_numpy(), "psu": rows["psu"].to_numpy()})
    r = run_r("""
        suppressPackageStartupMessages(library(survey))
        d <- read.csv(l_csv)
        des <- svydesign(ids = ~psu, strata = ~stratum, weights = ~wt, nest = TRUE, data = d)
        m <- svymean(~loss, des)
        fit <- svyglm(y ~ yhat, design = des)
        out(list(mean = unname(coef(m)), se = as.numeric(SE(m)), slope = unname(coef(fit)[2]),
                 resid = unname(coef(svymean(~I(y - yhat), des)))))
    """, {"l": data}, tmp_path)
    linear = next(f for f in out["evaluation"].data["design_based"]["families"]
                  if f["family"] == "linear")
    assert abs(linear["pooled_first"] - r["mean"]) < 1e-10
    assert abs(linear["se_first"] - r["se"]) < 1e-10
    assert abs(linear["calibration"]["slope"] - r["slope"]) < 1e-8
    assert abs(linear["calibration"]["intercept"] - r["resid"]) < 1e-10


@needs_r
def test_8_a_yes_no_outcomes_weighted_calibration_matches_svyglm(tmp_path):
    """The weighted logistic recalibration: slope from ``svyglm(event ~ logit(p), quasibinomial)``,
    calibration-in-the-large from the same with ``offset(logit(p))``, to 10⁻⁸."""
    rng = np.random.default_rng(9)
    n = 800
    lp = rng.normal(-0.5, 1.0, n)
    p = 1 / (1 + np.exp(-lp))
    event = (rng.random(n) < 1 / (1 + np.exp(-(0.2 + 0.8 * lp)))).astype(int)
    w = rng.uniform(1, 10, n)
    got = D.weighted_calibration("binary", event, np.column_stack([1 - p, p]), w, [0, 1])
    r = run_r("""
        suppressPackageStartupMessages(library(survey))
        d <- read.csv(c_csv)
        des <- svydesign(ids = ~1, weights = ~w, data = d)
        a <- svyglm(event ~ lp, design = des, family = quasibinomial(),
                    control = glm.control(epsilon = 1e-14, maxit = 100))
        b <- svyglm(event ~ 1, offset = lp, design = des, family = quasibinomial(),
                    control = glm.control(epsilon = 1e-14, maxit = 100))
        out(list(slope = unname(coef(a)[2]), intercept = unname(coef(b)[1])))
    """, {"c": pd.DataFrame({"event": event, "lp": np.log(p / (1 - p)), "w": w})}, tmp_path)
    assert abs(got["slope"] - r["slope"]) < 1e-8
    assert abs(got["intercept"] - r["intercept"]) < 1e-8


def test_8_the_choice_among_families_resamples_psus_within_strata():
    """BBC-CV with the design: each resample draws, within each stratum, as many of its PSUs as it
    holds, with replacement (``default_rng(seed).integers`` stratum by stratum, in order); the
    family with the lowest weighted in-bag loss is scored on the left-out PSUs' rows. Replayed by
    hand to 10⁻¹²."""
    rng = np.random.default_rng(4)
    frame = _design_table(2, strata=6, psus=3, per=6)
    n = len(frame)
    losses = {"a": rng.gamma(2, 1, n), "b": rng.gamma(2, 1.05, n)}
    w = frame["wt"].to_numpy()
    got = D.design_bbc(losses, w, frame["stratum"], frame["psu"], replicates=200, seed=3)
    s, p = D.codes(frame["stratum"], frame["psu"], n)
    draws = np.random.default_rng(3)
    scores = []
    for _ in range(200):
        times = np.zeros(p.max() + 1)
        for h in sorted(set(s)):
            units = np.unique(p[s == h])
            for u in units[draws.integers(0, len(units), len(units))]:
                times[u] += 1
        t = times[p]
        out = t == 0
        if not out.any():
            continue
        inbag = {k: float((v * w * t).sum() / (w * t).sum()) for k, v in losses.items()}
        best = min(inbag, key=inbag.get)
        scores.append(float((losses[best][out] * w[out]).sum() / w[out].sum()))
    assert got["replicates"] == len(scores)
    assert abs(got["corrected"] - np.mean(scores)) < 1e-12
    assert abs(got["corrected_low"] - np.percentile(scores, 2.5)) < 1e-12


def test_8_the_record_labels_design_based_cross_validation(population_run):
    _, _, out = population_run
    block = out["evaluation"].data["design_based"]
    assert block["sentence"].startswith(
        "Under the surveyed-population answer, performance is the population's: design-based "
        "cross-validation (Wieczorek, Guerin & McMahon, Stat 2022;11:e454) with 2 folds of whole "
        "PSUs within strata, drawn 10 times, every loss, calibration and comparison weighted by "
        "`wt`; ")
    assert "every fold keeps a PSU of every stratum" in block["note"]
    fit = out["fit"].data
    from turbotab.core.stages.modeling import PREDICTION_POPULATION_SCORES

    assert PREDICTION_POPULATION_SCORES in fit["models"][0]["concerns"]


def test_8_the_survey_answer_is_asked_under_prediction_now():
    """Ruling 13 makes the surveyed-population answer meaningful under prediction: it is accepted
    (it was refused before the ruling)."""
    state = d.ProjectState(purpose="prediction", target="y", task="regression")
    decision = d.validate(d.SetSurvey(estimand="population", weight="wt", strata="s", psu="p"),
                          {"state": state})
    assert decision.estimand == "population"


# ── under inference, no cross-validated score is shown ──────────────────────


def test_8_under_inference_no_cross_validated_score_is_shown(tmp_path):
    """MODELING_SEQUENCE §1 row 11 and ruling 13: under inference the declared model is reported by
    its estimates; the served fit carries no family's cross-validated score, calibration, verdict
    against the baseline, comparison, selection or declared result, and no concern quoting them;
    the evaluation stage says so."""
    from turbotab.core.stages.evaluation import NO_SCORE_UNDER_INFERENCE, withhold_scores

    frame = _design_table(3, strata=6, psus=2, per=20)
    frame.to_csv(tmp_path / "t.csv", index=False)
    run = GraphRun(tmp_path / "t.csv", tmp_path / "p")
    roles = {"pid": "identifier", "x1": "exposure", "x2": "covariate", "x3": "covariate"}
    state = d.ProjectState(
        lens=["clinical"], target="y", task="regression", purpose="inference", roles=roles,
        missing="complete_case", grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        split=d.SplitSpec(holdout=0.0, seed=2, folds=5), models=["linear", "elastic_net"])
    out = run.run(state, upto=["evaluation"])
    fit = out["fit"].data
    assert fit["models"][0]["cv"]  # the stage keeps them for the stages that read it
    served = withhold_scores(fit, state)
    for m in served["models"]:
        assert m["cv"] == {} and m["performance"] is None and m["calibration"] is None
        assert m["versus_baseline"] is None and m["compared_on"] is None
        assert not set(m["concerns"]) & set(fit["models"][0].get("score_concerns") or [])
    assert served["comparisons"] == [] and served["selection"] is None and served["result"] is None
    assert served["cv_definition"] == NO_SCORE_UNDER_INFERENCE
    assert NO_SCORE_UNDER_INFERENCE == (
        "Under inference no cross-validated score is shown: the declared model is reported by its "
        "estimates, not compared or chosen by how it predicts (MODELING_SEQUENCE §1 row 11).")
    ev = out["evaluation"].data
    assert ev["scores_shown"] is False and ev["note"] == NO_SCORE_UNDER_INFERENCE
    assert ev["benchmark"] is None and ev["design_based"] is None
    assert withhold_scores(fit, state.model_copy(update={"purpose": "prediction"})) is fit
    run.close()
