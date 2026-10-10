"""RT-5a · boosted trees tuned (RECIPES_AND_TUNING §2.2, §4.1, §4.4, §4.6, §8 T2(c) and T6;
WAVE_C6A_PLAN §2, §3).

Each check stands on a reference the family's code does not compute:

* **the declaration** against plan §2's table, written out here, and the standard settings against
  scikit-learn's own ``HistGradientBoosting*().get_params()`` (the library is the reference);
* **below an effective size of 300** the plan keeps the standard candidate alone, and the tuned fit
  is bit-equal to a direct ``HistGradientBoostingRegressor()`` / ``…Classifier()`` at scikit-learn's
  defaults on the same rows (a scikit-learn identity);
* **the leaf cap** worked by hand: a twentieth of the plan's units, never the standard's 20;
* **T2(c) pinned replay:** the plain pipeline at the recorded settings, through the plain path at the
  seed of the search's own draws (the SHA-256 of the canonical JSON of (split seed, "stopping
  sets"), recomputed here), reproduces the deployed predictions exactly at the recorded thread
  count, and equals a direct ``HistGradientBoostingRegressor`` at those parameters;
* **the estimate** (T6's arithmetic): one timed fit × the outer folds × F = (K − 1)·C + 1, with
  C, K and the center candidate worked by hand from §4.1's scales;
* **the thread count** (RECIPES §4.7): inside every fit and prediction, scikit-learn's own count of
  its OpenMP threads (``_openmp_effective_n_threads``, read by a spy on the library's class) is
  the plan's, however many threads the process allows.

Fixtures stay small (n ≤ 400), so the file runs in seconds.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.models import tuning as T
from turbotab.core.models.base import get_family
from turbotab.core.models.inner_cv import fit_pipeline
from turbotab.core.models.pipeline import make_plans
from turbotab.core.models.tuning import TunedPipeline, estimator_params, make_plan

THREADS = 1
PROCESS_THREADS = 3  # more than the plan's: the family must pin its own fits


@pytest.fixture(autouse=True)
def a_wider_process():
    """The process allows more threads than any plan here states, as an unpinned machine does
    (CI's ``OMP_NUM_THREADS=2``, a laptop's ten): the family's fits must not follow it."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=PROCESS_THREADS, user_api="openmp"):
        yield


def _at(threads: int):
    """A direct scikit-learn fit at a stated thread count (the reference's)."""
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=threads, user_api="openmp")


def _seed(*parts) -> int:
    text = json.dumps(list(parts), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def _data(task: str, n: int, seed: int = 3, blanks: float = 0.05):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    signal = 1.2 * X["a"] + np.sin(2 * X["b"]) + 0.6 * X["c"] * X["d"]
    X = X.mask(rng.random(X.shape) < blanks)  # blanks reach the trees as they are
    noise = rng.normal(size=n)
    if task == "regression":
        return X, (signal + noise).to_numpy()
    return X, np.where(signal + noise > 0.9, "yes", "no")


def _tuned(task: str, plan) -> TunedPipeline:
    return TunedPipeline([("model", get_family("boosted_trees").build(task, "prediction", 0, 0))],
                         search=plan)


# ── the declaration ──────────────────────────────────────────────────────────


def test_its_tuning_is_plan_section_2s_declaration():
    family = get_family("boosted_trees")
    decl = T.tuning_for(family, "regression")
    assert decl is not None and decl.kind == "search"
    assert {t: T.tuning_for(family, t) for t in family.tasks} == {t: decl for t in family.tasks}
    got = [(d.name, d.low, d.high, d.scale, d.active) for d in decl.dimensions]
    assert got == [
        ("learning_rate", 0.01, 0.3, "log", "always"),
        ("max_leaf_nodes", 4, 128, "log_int", "always"),
        ("min_samples_leaf", 2, 200, "log_int", "always"),
        ("l2_regularization", 1e-3, 10, "log", "always"),
        ("max_features", 0.3, 1.0, "linear", "always"),
        ("max_iter", 25, 500, "log_int", "without_early_stopping"),
    ]
    assert dict(decl.early_stopping) == {"param": "max_iter", "rounds": 1000,
                                         "patience_param": "n_iter_no_change", "patience": 20,
                                         "share": 0.1}
    assert decl.space_version == "boosted_trees/1" and decl.structural == ("loss",)
    assert decl.standard_source == "scikit-learn's defaults" and not decl.by_hand
    assert family.defaults_version == "2"
    assert "derive_seed" in family.identity.seed_policy
    # "Try both" for blanks waits for RT-3 (C6b): no plan carries options yet
    plan = make_plan(family, task="regression", loss="mse", n_plan=400, plan_rows=400,
                     unit="units", split_seed=7)
    assert plan.options == ()


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_the_standard_candidate_is_scikit_learns_own_defaults(task):
    """The standard values, resolved by ``settings`` at a plan small enough to cap a searched leaf
    below 20, are scikit-learn's own defaults: the cap never reaches the standard."""
    family = get_family("boosted_trees")
    plan = make_plan(family, task=task, loss="mse" if task == "regression" else "log_loss",
                     n_plan=100, plan_rows=100, unit="units", split_seed=7)
    [standard] = plan.candidates
    params = estimator_params(family, task, standard.values, n_units=100, n_rows=100, plan=plan)
    cls = HistGradientBoostingRegressor if task == "regression" else HistGradientBoostingClassifier
    library = cls().get_params()
    for name in ("learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization",
                 "max_features", "max_iter"):
        assert params[name] == library[name], name
    # the loss is structural: never among a candidate's parameters, the library's own as built
    assert "loss" not in params
    built = family.build(task, "prediction", 100, 4).set_params(**params)
    assert built.get_params()["loss"] == library["loss"]


def test_a_searched_leaf_is_capped_at_a_twentieth_of_the_plans_units():
    family = get_family("boosted_trees")
    plan = make_plan(family, task="regression", loss="mse", n_plan=400, plan_rows=400,
                     unit="units", split_seed=7)
    values = {**plan.candidates[0].values, "min_samples_leaf": 150}

    def leaf(v, n_units=400):
        return estimator_params(family, "regression", {**values, "min_samples_leaf": v},
                                n_units=n_units, n_rows=n_units, plan=plan)["min_samples_leaf"]

    assert leaf(150) == 400 // 20 == 20
    assert leaf(7) == 7
    # the plan's units, not the fit's own: a smaller outer fold keeps the plan's cap
    assert leaf(150, n_units=250) == 20
    # every Sobol candidate's leaf sits at or under the cap
    for c in plan.candidates[1:]:
        p = estimator_params(family, "regression", c.values, n_units=400, n_rows=400, plan=plan)
        assert 2 <= p["min_samples_leaf"] <= 20


# ── below an effective size of 300: scikit-learn's defaults, bit for bit ──────


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_below_300_the_tuned_fit_is_a_direct_default_fit(task):
    """At n = 360 (n_plan 288 units, or about 90 events), the plan the design makes keeps the
    standard candidate alone and makes one fit, and the tuned fit equals a direct
    ``HistGradientBoosting*()`` at scikit-learn's defaults on the same rows, bit for bit."""
    X, y = _data(task, 360)
    family = get_family("boosted_trees")
    plans = make_plans([family], task, y, folds=5, split_seed=11)
    plan = plans["boosted_trees"]
    assert plan.n_plan < 300 and len(plan.candidates) == 1 and plan.fits() == 1
    assert not plan.early_stopping and not plan.standard_stops
    fitted = fit_pipeline(_tuned(task, plan), X, y, seed=plan.split_seed)
    assert fitted.tuning_.n_fits == 1 and fitted.tuning_.inner_k_used == 0
    if task == "regression":
        with _at(THREADS):
            direct = HistGradientBoostingRegressor().fit(X, y)
            expected = direct.predict(X)
        assert np.array_equal(fitted.predict(X), expected)
        return
    with _at(THREADS):
        direct = HistGradientBoostingClassifier().fit(X, y)
        proba, labels = direct.predict_proba(X), direct.predict(X)
    assert np.array_equal(fitted.predict_proba(X), proba)
    assert np.array_equal(fitted.predict(X), labels)


# ── T2(c) · pinned replay ─────────────────────────────────────────────────────


def test_t2c_pinned_replay_reproduces_the_deployed_predictions_at_the_recorded_threads():
    X, y = _data("regression", 400, seed=9)
    family = get_family("boosted_trees")
    plan = make_plan(family, task="regression", loss="mse", n_plan=400, plan_rows=400,
                     unit="units", split_seed=7, threads=THREADS)
    assert len(plan.candidates) == 9 and plan.inner_k == 3 and not plan.early_stopping
    deployed = fit_pipeline(_tuned("regression", plan), X, y, seed=plan.split_seed)
    record = deployed.tuning_
    assert record.threads["plan"] == THREADS and record.chosen_params["n_threads"] == THREADS
    assert all(v is not None for v in record.losses)
    pinned = fit_pipeline(_tuned("regression", plan).at(record.chosen_params), X, y,
                          seed=_seed(plan.split_seed, "stopping sets"))
    assert type(pinned) is Pipeline
    assert np.array_equal(pinned.predict(X), deployed.predict(X))
    # the refit is a direct scikit-learn fit at the recorded parameters (no early stopping here)
    params = dict(record.chosen_params)
    assert params["early_stopping"] is False
    threads = params.pop("n_threads")
    with _at(threads):
        direct = HistGradientBoostingRegressor(**params).fit(X, y)
        expected = direct.predict(X)
    assert np.array_equal(expected, deployed.predict(X))
    # the candidates are seeded from the split: the plan's seed is SHA-256 of (seed, family, space)
    assert plan.seed == _seed(7, "boosted_trees", "boosted_trees/1")


# ── the estimate counts the search ────────────────────────────────────────────


def test_the_estimate_is_the_center_timed_once_times_the_folds_and_the_plans_fits(monkeypatch):
    from turbotab.core.decisions import ProjectState
    from turbotab.core.models import cost

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x0": rng.normal(size=120), "x1": rng.normal(size=120)})
    frame["y"] = frame["x0"] + rng.normal(size=120)

    class Store:
        def materialize(self, columns, ids):
            return frame.loc[np.asarray(ids), list(columns)]

    timed = []
    monkeypatch.setattr(cost, "time_one_fit", lambda pipeline, X, y: timed.append(pipeline) or 0.5)
    state = ProjectState(target="y", task="regression", purpose="prediction",
                         roles={"x0": "exposure", "x1": "covariate"})
    family = get_family("boosted_trees")
    plan = make_plan(family, task="regression", loss="mse", n_plan=400, plan_rows=400,
                     unit="units", split_seed=7)
    out = cost.estimate_fits(Store(), state, "regression", np.arange(120), [family], folds=5,
                             plans={family.key: plan})
    # C = 1 standard + 8 Sobol (n_plan 400 ≥ 300), K = 3: F = (3 − 1)·9 + 1 = 19
    assert plan.fits() == 19
    assert out["boosted_trees"].seconds == 0.5 * 5 * 19
    [pipeline] = timed
    params = pipeline[-1].get_params()
    center = {
        "learning_rate": math.exp((math.log(0.01) + math.log(0.3)) / 2),
        "max_leaf_nodes": math.floor(math.exp((math.log(4) + math.log(129)) / 2)),  # 22
        "min_samples_leaf": min(math.floor(math.exp((math.log(2) + math.log(201)) / 2)),
                                400 // 20),  # 20
        "l2_regularization": math.exp((math.log(1e-3) + math.log(10)) / 2),  # 0.1
        "max_features": 0.65,
        "max_iter": math.floor(math.exp((math.log(25) + math.log(501)) / 2)),  # 111
    }
    assert type(pipeline) is Pipeline and params["early_stopping"] is False
    assert {k: params[k] for k in center} == pytest.approx(center)


# ── the plan's threads (RECIPES §4.7) ────────────────────────────────────────


@pytest.mark.parametrize("threads", [1, 2])
def test_every_fit_and_prediction_runs_on_the_plans_threads(monkeypatch, threads):
    """A searched fit (n_plan 400: 19 fits) and its predictions, in a process allowing three
    OpenMP threads: scikit-learn's own count inside each fit and prediction is the plan's. Without
    the pin, histogram boosting takes every thread the process allows on each of the search's
    small fits (one tuned fit at n = 480 took 1.5 s on one thread and 152 s on ten)."""
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

    seen: dict[str, list[int]] = {"fit": [], "predict": []}
    fit, predict = HistGradientBoostingRegressor.fit, HistGradientBoostingRegressor.predict

    def spy_fit(self, *a, **k):
        seen["fit"].append(_openmp_effective_n_threads())
        return fit(self, *a, **k)

    def spy_predict(self, *a, **k):
        seen["predict"].append(_openmp_effective_n_threads())
        return predict(self, *a, **k)

    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", spy_fit)
    monkeypatch.setattr(HistGradientBoostingRegressor, "predict", spy_predict)
    assert _openmp_effective_n_threads() == PROCESS_THREADS
    X, y = _data("regression", 400, seed=9)
    family = get_family("boosted_trees")
    plan = make_plan(family, task="regression", loss="mse", n_plan=400, plan_rows=400,
                     unit="units", split_seed=7, threads=threads)
    deployed = fit_pipeline(_tuned("regression", plan), X, y, seed=plan.split_seed)
    deployed.predict(X)
    assert len(seen["fit"]) == deployed.tuning_.n_fits == 3 * 9 + 1  # K·C inner fits, one refit
    assert set(seen["fit"]) == {threads} and set(seen["predict"]) == {threads}
    assert deployed.tuning_.chosen_params["n_threads"] == threads
    # the process's own count is left as it was
    assert _openmp_effective_n_threads() == PROCESS_THREADS


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_without_a_plan_the_family_fits_and_predicts_on_one_thread(monkeypatch, task):
    """The plain path (no plan: a family built alone) runs on one thread, as XGBoost's does."""
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

    cls = HistGradientBoostingRegressor if task == "regression" else HistGradientBoostingClassifier
    seen: list[int] = []
    methods = ("fit", "predict") + (() if task == "regression" else
                                    ("predict_proba", "decision_function"))
    for name in methods:
        original = getattr(cls, name)

        def spy(self, *a, _original=original, **k):
            seen.append(_openmp_effective_n_threads())
            return _original(self, *a, **k)

        monkeypatch.setattr(cls, name, spy)
    X, y = _data(task, 200)
    model = get_family("boosted_trees").build(task, "prediction", 200, 4)
    assert isinstance(model, cls) and model.get_params()["n_threads"] == 1
    model.fit(X, y)
    for name in methods[1:]:
        getattr(model, name)(X)
    assert len(seen) >= len(methods) and set(seen) == {1}
    # a thread count is a whole number of at least one (scikit-learn's own parameter check)
    from sklearn.utils._param_validation import InvalidParameterError

    with pytest.raises(InvalidParameterError, match="n_threads"):
        model.set_params(n_threads=0).fit(X, y)
