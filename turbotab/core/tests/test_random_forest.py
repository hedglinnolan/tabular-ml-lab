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


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_predicting_never_changes_the_models_threads(task: str) -> None:
    """Prediction runs on a copy that holds one thread: the model's own ``n_jobs`` reads the fit's
    threads while its trees are predicting, so threads predicting at once never race on it."""
    X, y = _data(task)
    model = _forest(task, X, y)
    first = model.estimators_[0]
    seen: list[int] = []
    method = "predict" if task == "regression" else "predict_proba"
    original = getattr(first, method)

    def watched(*args: Any, **kwargs: Any) -> Any:
        seen.append(model.get_params()["n_jobs"])
        return original(*args, **kwargs)

    setattr(first, method, watched)
    try:
        model.predict(X[:20])
    finally:
        delattr(first, method)
    assert seen == [THREADS] and model.n_jobs == THREADS


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


# ── the smallest leaf: the stored share is the share applied ────────────────


def test_the_smallest_leaf_is_its_stored_share_of_each_fits_rows_with_or_without_the_plan(
) -> None:
    """WAVE_C6A_PLAN §5.2 (ruled under §7.7): the share runs log-uniformly from one unit of the
    plan, 1/n_plan, to a tenth, is stored as a share, and becomes rows on each fit. The rows
    applied are the stored share's, whether or not the plan is passed, so the record states the
    leaf the fit used. For a yes/no outcome n_plan counts events, so the low end is one event's
    share of the rows, not one row."""
    fam = get_family("random_forest")
    leaf = next(d for d in tuning_for(fam, "binary").dimensions if d.name == "min_samples_leaf")
    for task, n_plan, plan_rows, unit in (("binary", 400, 4000, "events"),
                                          ("regression", 300, 1000, "units"),
                                          ("multiclass", 120, 1000, "rarest_class")):
        plan = make_plan(fam, task=task, loss="mse" if task == "regression" else "log_loss",
                         n_plan=n_plan, plan_rows=plan_rows, unit=unit, split_seed=1)
        for c in plan.candidates[1:]:  # the Sobol candidates: a stored share each
            share = c.values["min_samples_leaf"]
            assert 1 / n_plan <= share <= 0.1
            want = max(1, math.floor(share * plan_rows + 0.5))  # by hand
            for given in (plan, None):
                got = estimator_params(fam, task, c.values, n_units=plan_rows, n_rows=plan_rows,
                                       plan=given)["min_samples_leaf"]
                assert got == want, (task, share, given is None)
    # the verifier's cases, by hand: 0.03141 of 4,000 rows; 0.00463 of 1,000 rows
    for share, n_rows, want in ((0.03141, 4000, 126), (0.00463, 1000, 5)):
        for given in (_oob_plan("binary", 400), None):
            assert estimator_params(fam, "binary", {"min_samples_leaf": share}, n_units=n_rows,
                                    n_rows=n_rows, plan=given)["min_samples_leaf"] == want
    # the ends: one event's share (1/40 of 1,000 rows is 25 rows) and a tenth
    assert map_unit(leaf, 0.0, n_plan=40) == pytest.approx(1 / 40, rel=1e-12)
    for u, want in ((0.0, 25), (1.0, 100), (0.5, 50)):  # √(25 × 100) = 50
        share = map_unit(leaf, u, n_plan=40)
        assert estimator_params(fam, "binary", {"min_samples_leaf": share}, n_units=1000,
                                n_rows=1000)["min_samples_leaf"] == want
    # a share set by hand is the share it says
    manual = make_plan(fam, task="binary", loss="log_loss", n_plan=40, plan_rows=1000,
                       unit="events", split_seed=1, manual={"min_samples_leaf": 0.03})
    assert estimator_params(fam, "binary", manual.candidates[0].values, n_units=1000,
                            n_rows=1000, plan=manual)["min_samples_leaf"] == 30


def test_settings_read_a_whole_number_leaf_as_rows_and_refuse_a_share_out_of_range() -> None:
    """scikit-learn reads an integer ``min_samples_leaf`` as rows and a float as a share; so does
    the family, so its resolved parameters fed back in give themselves, and a share outside
    (0, 0.1] is refused rather than turning the forest into a stump."""
    fam = get_family("random_forest")
    X, y = _data("binary", n=200)
    plan = _oob_plan("binary", 200)
    for values in (plan.candidates[0].values, plan.candidates[-1].values,
                   {"min_samples_leaf": 0.05, "max_features": 0.5, "max_samples": 0.8}):
        once = estimator_params(fam, "binary", values, n_units=200, n_rows=200, y=y, Z=X,
                                plan=plan)
        assert estimator_params(fam, "binary", once, n_units=200, n_rows=200, y=y, Z=X,
                                plan=plan) == once
        assert isinstance(once["min_samples_leaf"], int) and once["min_samples_leaf"] <= 20
    for rows in (10, np.int64(10)):
        assert estimator_params(fam, "binary", {"min_samples_leaf": rows}, n_units=200,
                                n_rows=200)["min_samples_leaf"] == 10
    for bad in (0, 1.0, 0.5, 0.0, -0.01, True):
        with pytest.raises(ValueError, match="smallest leaf"):
            estimator_params(fam, "binary", {"min_samples_leaf": bad}, n_units=200, n_rows=200)


def test_its_seed_is_threaded_from_the_split_through_derive_seed() -> None:
    """RECIPES §4.7: the forest's random_state is the SHA-256 of the canonical JSON of (split seed,
    family, space version), its first four bytes big-endian, computed here by hand."""
    import hashlib
    import json

    def by_hand(split_seed: int) -> int:
        text = json.dumps([split_seed, "random_forest", "random_forest/1"], sort_keys=True,
                          separators=(",", ":"), ensure_ascii=True)
        return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")

    fam = get_family("random_forest")
    seeds = []
    for split_seed in (7, 8):
        plan = make_plan(fam, task="binary", loss="log_loss", n_plan=400, plan_rows=400,
                         unit="events", split_seed=split_seed)
        for c in plan.candidates[:3]:
            params = estimator_params(fam, "binary", c.values, n_units=400, n_rows=400, plan=plan)
            assert params["random_state"] == by_hand(split_seed)
        seeds.append(by_hand(split_seed))
    assert seeds[0] != seeds[1]
    for task in ("regression", "binary"):
        assert fam.build(task, "prediction", 300, 5).get_params()["random_state"] == by_hand(0)


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
    phi, expected = E.tree_shap(fam.trees(model), X_new)  # v2's own, on the same float64 rows
    assert np.max(np.abs(expected[0] + phi[:, :, 0].sum(axis=1) - want)) < 1e-9


def test_a_float64_threshold_routes_every_input_as_its_float32_does() -> None:
    """For thresholds of every kind, x <= t' exactly when float32(x) <= t, checked by brute force
    at each float32 near t, the midpoints between them and the float64 values beside those."""
    from turbotab.core.models.forest import float64_threshold

    rng = np.random.default_rng(3)
    a = rng.normal(size=300).astype(np.float32)
    b = np.nextafter(a, np.float32(np.inf)) + rng.integers(0, 3, 300).astype(np.float32) * \
        np.spacing(a)
    t = np.concatenate([(a.astype(float) + b.astype(float)) / 2, a.astype(float),
                        rng.normal(size=300), [0.0, -0.0, 1e-30, 3.0, -2.5]])
    t2 = float64_threshold(t)
    for ti, ui in zip(t, t2):
        f = np.float32(ti)
        near = [np.float32(f)]
        for _ in range(3):
            near = sorted({*near, np.nextafter(near[0], np.float32(-np.inf)),
                           np.nextafter(near[-1], np.float32(np.inf))})
        xs = []
        for g, h in zip(near[:-1], near[1:]):
            m = (float(g) + float(h)) / 2
            xs += [float(g), m, np.nextafter(m, -np.inf), np.nextafter(m, np.inf), ti,
                   np.nextafter(ti, np.inf), np.nextafter(ti, -np.inf)]
        for x in xs:
            assert (float(np.float32(x)) <= ti) == (x <= ui), (ti, x)


def test_v2_tree_shap_adds_up_at_a_float64_value_just_above_a_threshold() -> None:
    """A float64 value just above a split's threshold whose float32 rounding falls at or below it
    goes left in the forest; v2's numpy TreeSHAP sends it the same way, so local accuracy holds."""
    X, y = _data("regression")
    model = _forest("regression", X, y)
    rows = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        t, j = float(tree.threshold[0]), int(tree.feature[0])
        x = np.nextafter(t, np.inf)
        if float(np.float32(x)) <= t:
            row = X[0].copy()
            row[j] = x
            rows.append(row)
    assert len(rows) >= 5
    X_gap = np.vstack(rows)
    phi, expected = E.tree_shap(get_family("random_forest").trees(model), X_gap)
    assert np.max(np.abs(expected[0] + phi[:, :, 0].sum(axis=1) - model.predict(X_gap))) < 1e-9


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
    # the curve is drawn on the log-odds of the clipped probability, and the artifact says so:
    # Apley and Zhu's ALE, computed here by hand on the artifact's grid from scikit-learn's own
    # forest's probabilities, clipped at 1/(2T) and turned into log-odds here
    shown = art.curves[0]
    curve = shown.curves[0]
    assert curve.drawn and shown.method == "ale"
    direct = _direct("binary", model, X, y)
    eps = 1.0 / (2 * TREES)

    def logit(A: np.ndarray) -> np.ndarray:
        p = np.clip(direct.predict_proba(A)[:, 1], eps, 1 - eps)
        return np.log(p / (1 - p))

    z = np.asarray(shown.grid)
    K = len(z) - 1
    k = np.clip(np.searchsorted(z, X[:, 0], side="left"), 1, K)  # x in (z_{k-1}, z_k]
    up, down = X.copy(), X.copy()
    up[:, 0], down[:, 0] = z[k], z[k - 1]
    effect = logit(up) - logit(down)
    local = np.array([effect[k == m].mean() if (k == m).any() else 0.0 for m in range(1, K + 1)])
    g = np.concatenate([[0.0], np.cumsum(local)])
    g = g - g[k].mean()
    for i, v in enumerate(curve.values):
        if v is not None:
            assert v == pytest.approx(g[i], abs=1e-9)
    said = [n for n in art.notes if "Random forest" in n]
    assert said and "probability of `yes`" in said[0] and "log-odds" in said[0]
    # the methods paragraph names both scales too
    assert "random forest's SHAP values are on its predicted probability of `yes`" in art.methods
    assert "log-odds of that probability" in art.methods
