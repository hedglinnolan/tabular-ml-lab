"""RT-5d · the random forest family (RECIPES_AND_TUNING §2.2, §4.1, §4.2, §4.7, T11;
MODEL_FAMILY_CONTRACT §3.2; WAVE_C6A_PLAN §3 row RT-5d).

Every expected value comes from a path independent of ``models/forest.py``:

* **predictions:** scikit-learn's own ``RandomForestRegressor`` / ``RandomForestClassifier`` fit
  directly at the same parameters, seed and threads, and predicted single-threaded, must equal the
  family's predictions exactly; the margin is the logit of the clipped probability, computed here;
* **out of bag:** the pooled loss from scikit-learn's ``oob_prediction_`` and
  ``oob_decision_function_`` through ``sklearn.metrics``; rows no tree left out, found by hand from
  a one-tree forest;
* **SHAP:** the ``shap`` package's ``TreeExplainer`` (path-dependent), against v2's numpy TreeSHAP
  on the family's ``trees``, blanks included, to 1e-6; local accuracy against ``predict``;
* **the standard settings:** randomForest's and ranger's documentation, copied below with where
  each line comes from, each default computed from those R expressions by hand;
* **invariance:** a monotone change of one column keeps the training rows' predictions; a rotation
  of two columns changes them.

Fixtures are small: at most 400 rows and 50 trees.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.pipeline import Pipeline

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.models import explain as E
from turbotab.core.models.base import Situation, assessment, get_family
from turbotab.core.models.tuning import (
    estimator_params,
    make_plan,
    map_unit,
    pooled_loss,
    tuning_for,
)

TREES = 40
THREADS = 2

# ── the R documentation the standard settings follow (copied 2026-10-10) ─────
# randomForest 4.7-1.2, ?randomForest, Usage (default S3 method):
RANDOMFOREST_USAGE = """randomForest(x, y=NULL,  xtest=NULL, ytest=NULL, ntree=500,
             mtry=if (!is.null(y) && !is.factor(y))
             max(floor(ncol(x)/3), 1) else floor(sqrt(ncol(x))),
             weights=NULL,
             replace=TRUE, classwt=NULL, cutoff, strata,
             sampsize = if (replace) nrow(x) else ceiling(.632*nrow(x)),
             nodesize = if (!is.null(y) && !is.factor(y)) 5 else 1,"""
RANDOMFOREST_NODESIZE = ("Minimum size of terminal nodes. Setting this number larger causes "
                         "smaller trees to be grown (and thus take less time). Note that the "
                         "default values are different for classification (1) and regression (5).")
# ranger 0.18.0, ?ranger, Usage and Arguments:
RANGER_USAGE = ("num.trees = 500, mtry = NULL, ... probability = FALSE, min.node.size = NULL, ... "
                "replace = TRUE, sample.fraction = ifelse(replace, 1, 0.632),")
RANGER_MIN_NODE_SIZE = ("Minimal node size to split at. Default 1 for classification, 5 for "
                        "regression, 3 for survival, and 10 for probability.")
RANGER_PROBABILITY = "Grow a probability forest as in Malley et al. (2012)."


def _data(task: str, n: int = 300, p: int = 5, *, blanks: float = 0.0, seed: int = 0
          ) -> tuple[np.ndarray, np.ndarray]:
    """Values a float32 holds exactly (scikit-learn's trees read their inputs as float32, as the
    shap package does), with ``blanks`` of the cells blank."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n, max(p, 4))).astype(np.float32).astype(float)
    f = 2 * W[:, 0] + np.sin(W[:, 1]) + W[:, 2] * W[:, 3]
    X = W[:, :p].copy()
    y = f + rng.normal(size=n)
    if blanks:
        X[rng.random(X.shape) < blanks] = np.nan
    if task == "regression":
        return X, y
    if task == "binary":
        return X, (y > 0.3).astype(int)
    return X, np.digitize(y, [-1.0, 1.0])


def _forest(task: str, X: np.ndarray, y: np.ndarray, *, plan: Any = None, **values: Any) -> Any:
    """The family's estimator at its standard settings with ``values`` set, fit on X, y."""
    fam = get_family("random_forest")
    decl = tuning_for(fam, task)
    params = estimator_params(fam, task, {**decl.standard, **values}, n_units=len(y),
                              n_rows=len(y), y=y, Z=X, plan=plan)
    params.update(n_estimators=TREES, n_jobs=THREADS)
    model = fam.build(task, "prediction", len(y), X.shape[1])
    return model.set_params(**params).fit(X, y)


def _direct(task: str, model: Any, X: np.ndarray, y: np.ndarray) -> Any:
    """scikit-learn's own forest at the family model's parameters, fit at the same threads."""
    cls = RandomForestRegressor if task == "regression" else RandomForestClassifier
    names = cls().get_params()
    direct = cls(**{k: v for k, v in model.get_params().items() if k in names}).fit(X, y)
    return direct.set_params(n_jobs=1)  # predicted single-threaded: trees summed in order


# ── the standard settings, against the R documentation ───────────────────────


@pytest.mark.parametrize("p", [1, 2, 3, 4, 9, 10, 26])
def test_the_standard_settings_are_randomforests_and_rangers(p: int) -> None:
    """Columns per split: randomForest's ``max(floor(ncol(x)/3), 1)`` for a number and
    ``floor(sqrt(ncol(x)))`` for classes (ranger's default for every forest); 500 trees; every row
    drawn with replacement (``replace=TRUE``, ``sampsize = nrow(x)``, ``sample.fraction = 1``);
    the smallest leaf 5 for a number (randomForest's ``nodesize``) and 10 for a probability
    (ranger's ``min.node.size`` for a probability forest, which a classifier's averaged class
    shares are)."""
    assert "ntree=500" in RANDOMFOREST_USAGE and "num.trees = 500" in RANGER_USAGE
    assert "replace=TRUE" in RANDOMFOREST_USAGE and "replace = TRUE" in RANGER_USAGE
    assert "sampsize = if (replace) nrow(x)" in RANDOMFOREST_USAGE
    assert "nodesize = if (!is.null(y) && !is.factor(y)) 5" in RANDOMFOREST_USAGE
    assert "regression (5)" in RANDOMFOREST_NODESIZE and "10 for probability" in RANGER_MIN_NODE_SIZE
    fam = get_family("random_forest")
    by_r = {"regression": max(math.floor(p / 3), 1), "binary": math.floor(math.sqrt(p)),
            "multiclass": math.floor(math.sqrt(p))}
    leaf = {"regression": 5, "binary": 10, "multiclass": 10}
    for task, mtry in by_r.items():
        X, y = _data(task, n=120, p=p)
        decl = tuning_for(fam, task)
        assert dict(decl.fixed) == {"n_estimators": 500, "bootstrap": True}
        params = estimator_params(fam, task, decl.standard, n_units=len(y), n_rows=len(y), y=y,
                                  Z=X)
        assert params["n_estimators"] == 500 and params["bootstrap"] is True
        assert params["max_samples"] is None  # every row: nrow(x) draws
        assert params["min_samples_leaf"] == leaf[task], task
        built = fam.build(task, "prediction", len(y), p)
        assert built.get_params()["n_estimators"] == 500
        fitted = built.set_params(n_estimators=3).fit(X, y)
        # the columns each split tries, as the fitted trees resolved them on this matrix
        assert {t.max_features_ for t in fitted.estimators_} == {mtry}, (task, p)
        assert fitted.estimators_[0].min_samples_leaf == leaf[task]


def test_the_search_space_is_recipes_section_4_1() -> None:
    fam = get_family("random_forest")
    for task in ("regression", "binary", "multiclass", "ordinal"):
        decl = tuning_for(fam, task)
        dims = {d.name: (d.low, d.high, d.scale) for d in decl.dimensions}
        assert dims["max_features"] == (0.05, 1.0, "linear")
        assert dims["max_samples"] == (0.2, 1.0, "linear")
        assert dims["min_samples_leaf"][1:] == (0.1, "share_of_units")
        assert decl.kind == "search" and decl.out_of_bag and decl.max_drawn == 8
        assert decl.early_stopping is None and decl.space_version == "random_forest/1"


# ── predictions: exactly scikit-learn's own forest ───────────────────────────


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_predictions_equal_a_direct_scikit_learn_fit(task: str) -> None:
    X, y = _data(task, blanks=0.05)
    model = _forest(task, X, y, max_features=0.6, max_samples=0.7, min_samples_leaf=0.02)
    direct = _direct(task, model, X, y)
    X_new, _ = _data(task, n=200, blanks=0.05, seed=9)
    if task == "regression":
        assert np.array_equal(model.predict(X_new), direct.predict(X_new))
        return
    assert np.array_equal(model.predict_proba(X_new), direct.predict_proba(X_new))
    assert np.array_equal(model.predict(X_new), direct.predict(X_new))
    if task == "binary":
        # the margin: the logit of the probability, held half of one tree's vote from 0 and 1
        eps = 1.0 / (2 * TREES)
        p = np.clip(direct.predict_proba(X_new)[:, 1], eps, 1 - eps)
        assert np.allclose(model.decision_function(X_new), np.log(p / (1 - p)), rtol=0,
                           atol=1e-12)
        assert (np.abs(model.decision_function(X_new)) <= math.log((1 - eps) / eps) + 1e-12).all()


def test_predicting_is_single_threaded_whatever_the_fit_used() -> None:
    X, y = _data("binary")
    model = _forest("binary", X, y)
    assert model.n_jobs == THREADS
    first = model.predict_proba(X)
    assert model.n_jobs == THREADS and np.array_equal(model.predict_proba(X), first)


# ── out of bag ───────────────────────────────────────────────────────────────


def _oob_plan(task: str, n: int) -> Any:
    fam = get_family("random_forest")
    unit = {"regression": "units", "binary": "events"}.get(task, "rarest_class")
    return make_plan(fam, task=task, loss="mse" if task == "regression" else "log_loss",
                     n_plan=n, plan_rows=n, unit=unit, split_seed=7, out_of_bag=True,
                     threads=THREADS)


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_the_out_of_bag_loss_is_scikit_learns_out_of_bag_predictions_pooled(task: str) -> None:
    from turbotab.core.models.forest import out_of_bag_loss

    X, y = _data(task, n=400)
    plan = _oob_plan(task, 400)
    assert plan.out_of_bag and plan.inner_k == 0 and plan.chooses()
    assert len(plan.candidates) == 1 + 8 and plan.fits() == len(plan.candidates) + 1
    model = _forest(task, X, y, plan=plan)
    assert model.oob_score is True  # the plan scores out of bag, so the fit keeps its OOB rows
    direct = _direct(task, model, X, y)
    if task == "regression":
        want = mean_squared_error(y, direct.oob_prediction_)
        got = out_of_bag_loss(model, y, task=task, loss="mse")
    else:
        want = log_loss(y, direct.oob_decision_function_, labels=direct.classes_)
        got = out_of_bag_loss(model, y, task=task, loss="log_loss")
    assert got == pytest.approx(want, rel=1e-12, abs=0)


@pytest.mark.filterwarnings("ignore:Some inputs do not have OOB scores")
def test_rows_no_tree_left_out_are_not_scored() -> None:
    """One tree: the out-of-bag rows are those its bootstrap did not draw, each predicted by that
    tree alone (scikit-learn would score the others as a prediction of 0)."""
    from turbotab.core.models.forest import out_of_bag_loss

    X, y = _data("regression", n=200)
    model = _forest("regression", X, y, plan=_oob_plan("regression", 200))
    model = model.set_params(n_estimators=1).fit(X, y)
    rng = np.random.RandomState(model.estimators_[0].random_state)  # scikit-learn's own draw
    drawn = rng.randint(0, len(y), len(y))
    left_out = np.setdiff1d(np.arange(len(y)), drawn)
    tree = model.estimators_[0]
    want = float(np.mean((y[left_out] - tree.predict(X[left_out])) ** 2))
    assert 0 < len(left_out) < len(y)
    assert out_of_bag_loss(model, y, task="regression", loss="mse") == pytest.approx(want,
                                                                                      rel=1e-12)
    assert pooled_loss("regression", "mse", y[left_out], tree.predict(X[left_out]),
                       classes=None) == pytest.approx(want, rel=1e-12)


def test_settings_follow_the_plan_threads_and_out_of_bag() -> None:
    fam = get_family("random_forest")
    X, y = _data("binary", n=400)
    plan = _oob_plan("binary", 400)
    params = estimator_params(fam, "binary", plan.candidates[3].values, n_units=400, n_rows=400,
                              y=y, Z=X, plan=plan)
    assert params["n_jobs"] == THREADS and params["oob_score"] is True
    inner = make_plan(fam, task="binary", loss="log_loss", n_plan=400, plan_rows=400,
                      unit="events", split_seed=7)
    params = estimator_params(fam, "binary", inner.candidates[0].values, n_units=400,
                              n_rows=400, y=y, Z=X, plan=inner)
    assert params["n_jobs"] == 1 and params["oob_score"] is False


# ── the smallest leaf: from one row of the plan's fit to a tenth of it ───────


def test_the_smallest_leaf_runs_from_one_row_to_a_tenth_for_every_outcome() -> None:
    """RECIPES §4.1: "from 1 row to a tenth of them". For a yes/no outcome n_plan counts events, so
    a share drawn from one unit of n_plan (``map_unit``'s low end) would start at a leaf of about
    rows/events rows; the family resolves the share from one row of the plan's fit instead."""
    fam = get_family("random_forest")
    leaf = next(d for d in tuning_for(fam, "binary").dimensions if d.name == "min_samples_leaf")
    for task, n_plan, unit in (("binary", 40, "events"), ("regression", 1000, "units"),
                               ("multiclass", 120, "rarest_class")):
        plan = make_plan(fam, task=task, loss="mse" if task == "regression" else "log_loss",
                         n_plan=n_plan, plan_rows=1000, unit=unit, split_seed=1)

        def rows(u: float, n_rows: int = 1000) -> int:
            share = map_unit(leaf, u, n_plan=n_plan)
            return estimator_params(fam, task, {"min_samples_leaf": share}, n_units=n_rows,
                                    n_rows=n_rows, plan=plan)["min_samples_leaf"]

        assert rows(0.0) == 1 and rows(1.0) == 100, task  # one row; a tenth of 1,000
        # log-uniform between them: the middle is √(1 × 100) = 10 rows
        assert rows(0.5) == 10, task
        # a fit of other size keeps the share: a tenth of 500 rows
        assert rows(1.0, 500) == 50
        # a share set by hand is the share it says
        manual = make_plan(fam, task=task, loss=plan.loss, n_plan=n_plan, plan_rows=1000,
                           unit=unit, split_seed=1, manual={"min_samples_leaf": 0.03})
        assert estimator_params(fam, task, manual.candidates[0].values, n_units=1000,
                                n_rows=1000, plan=manual)["min_samples_leaf"] == 30


# ── SHAP: v2's TreeSHAP and the shap package, blanks included ────────────────


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_v2_tree_shap_on_the_forests_trees_agrees_with_shap(task: str) -> None:
    import shap

    X, y = _data(task, blanks=0.08)
    model = _forest(task, X, y, min_samples_leaf=0.01)
    fam = get_family("random_forest")
    ensemble = fam.trees(model)
    assert ensemble.scale == ("margin" if task == "regression" else "probability")
    assert len(ensemble.trees) == 1 and len(ensemble.trees[0]) == TREES
    X_ex = X[:150]
    phi, expected = E.tree_shap(ensemble, X_ex)
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    values = explainer.shap_values(X_ex, check_additivity=False)
    values = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)
    ev = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
    if task == "binary":  # the class coded 1
        values, ev = values[:, :, 1], ev[1:]
    assert np.isnan(X_ex).any()
    assert np.max(np.abs(phi[:, :, 0] - values)) < 1e-6
    assert abs(expected[0] - ev[0]) < 1e-6
    # the family's compiled TreeSHAP gives the same values
    compiled, compiled_expected = fam.tree_shap(model, X_ex)
    assert compiled.shape == (len(X_ex), X.shape[1], 1)
    assert np.max(np.abs(compiled - phi)) < 1e-6 and abs(compiled_expected[0] - expected[0]) < 1e-6


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_local_accuracy_sums_to_the_prediction(task: str) -> None:
    """expected + Σφ is the forest's prediction (a number, or the probability of the class coded
    1) for every row, on any float64 input: the compiled path reads it as the trees do."""
    X, y = _data(task, blanks=0.08)
    model = _forest(task, X, y)
    rng = np.random.default_rng(11)
    X_new = rng.normal(size=(120, X.shape[1]))
    X_new[rng.random(X_new.shape) < 0.08] = np.nan
    want = model.predict(X_new) if task == "regression" else model.predict_proba(X_new)[:, 1]
    fam = get_family("random_forest")
    phi, expected = fam.tree_shap(model, X_new)
    assert np.max(np.abs(expected[0] + phi[:, :, 0].sum(axis=1) - want)) < 1e-9
    X32 = X_new.astype(np.float32).astype(float)
    phi, expected = E.tree_shap(fam.trees(model), X32)
    assert np.max(np.abs(expected[0] + phi[:, :, 0].sum(axis=1) - want)) < 1e-9


# ── invariance (C3, C13) ─────────────────────────────────────────────────────


def test_a_monotone_change_keeps_the_training_predictions_and_a_rotation_does_not() -> None:
    """Each tree's predictions on the rows it was grown on are unchanged by a strictly increasing
    change of one column: the same rows fall on each side of every split. (A row a tree left out
    can sit between two of its rows, where the split's midpoint moves with the change.) A
    rotation of two columns changes them."""
    from sklearn.base import clone

    X, y = _data("regression", blanks=0.0)
    model = _forest("regression", X, y)
    X_mono = X.copy()
    X_mono[:, 0] = np.exp(X[:, 0] / 2)  # strictly increasing, still distinct in float32
    assert len(np.unique(X_mono[:, 0].astype(np.float32))) == len(np.unique(X[:, 0]))
    again = clone(model).fit(X_mono, y)
    for before, after, drawn in zip(model.estimators_, again.estimators_,
                                    model.estimators_samples_):
        assert np.array_equal(after.predict(X_mono[drawn]), before.predict(X[drawn]))
        assert np.array_equal(after.tree_.feature, before.tree_.feature)
    c, s = math.cos(0.6), math.sin(0.6)
    X_rot = X.copy()
    X_rot[:, 0], X_rot[:, 1] = c * X[:, 0] - s * X[:, 1], s * X[:, 0] + c * X[:, 1]
    rotated = clone(model).fit(X_rot, y)
    assert np.max(np.abs(rotated.predict(X_rot) - model.predict(X))) > 1e-3


# ── the shelf (RECIPES §2.6) ─────────────────────────────────────────────────


def test_its_assessment_is_recipes_table() -> None:
    fam = get_family("random_forest")

    def judged(rows: int, cols: int = 10, purpose: str = "prediction") -> tuple[float, str]:
        a = assessment(fam, Situation(task="regression", purpose=purpose, n_rows=rows,
                                      n_features=cols))
        return a.score, a.fit

    assert judged(2500) == (2.0, "good")
    assert judged(800) == (1.5, "good")
    assert judged(300) == (1.0, "fair")
    assert judged(800, cols=900) == (1.5, "fair")
    assert judged(2500, purpose="inference") == (1.0, "fair")  # as boosted trees


# ── the explanation's two scales, each labeled ───────────────────────────────


def test_a_yes_no_forest_is_explained_with_shap_on_its_probability_and_curves_on_the_log_odds(
) -> None:
    X, y = _data("binary", n=300, p=4)
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    model = _forest("binary", X, y)
    pipe = Pipeline([("model", model)]).set_output(transform="pandas").fit(frame, y)
    fam = E.FamilyFit(key="random_forest", label="Random forest", fitted=pipe, unfitted=pipe,
                      versus={"verdict": "better"}, score=0.8, baseline=0.5)
    s = E.Setting(task="binary", purpose="prediction", target="y", event="yes", X=frame, y=y,
                  reseeds=1, exposures=("a",))
    art = E.explain([fam], s, refit=lambda m, X_, y_, u: m.fit(X_, y_))
    out = art.families[0]
    assert out.explained and out.scale == "probability of `yes`"
    obs = out.observations
    total = obs.base + np.sum(obs.phi, axis=1) + np.asarray(obs.rest)
    want = pipe.predict_proba(frame.loc[obs.row_ids])[:, 1]
    assert np.max(np.abs(total - want)) < 1e-9
    assert np.max(np.abs(np.asarray(obs.prediction) - want)) < 1e-12
    # the curve is drawn on the log-odds of the clipped probability, and the artifact says so
    curve = art.curves[0].curves[0]
    assert curve.drawn
    grid = E.ale_grid(frame["a"])
    margin = E.ale_curve(lambda A: pipe.decision_function(A), frame, "a", grid)
    assert np.allclose([v for v in curve.values if v is not None],
                       [v for v in E.masked(margin, grid) if v is not None], atol=1e-9)
    said = [n for n in art.notes if "Random forest" in n]
    assert said and "probability of `yes`" in said[0] and "log-odds" in said[0]
