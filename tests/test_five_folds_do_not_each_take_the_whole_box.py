"""`ml/eval.py` — a fold worker gets one compute thread, and the same numbers.

`perform_cross_validation` dispatches folds with `n_jobs=-1`, so each fold is
fitted in its own process. An estimator that then starts a full compute pool
inside that process multiplies the two: five folds times eight LightGBM threads
is forty threads over the eight cores this app targets, with a browser already
competing for the machine.

**Most of the audit's claim was already handled and the test says so.** joblib
sets `OMP_NUM_THREADS` and friends to 1 in every loky worker, and measurement
inside real fold workers put XGBoost, HistGradientBoosting, ExtraTrees and kNN
at +1 thread per fit — pinned before we arrived. Only LightGBM ran a pool
(+12), because it reads neither the environment variable nor `threadpoolctl`.
So the pin is narrow on purpose, and half of what is asserted here is what it
must NOT touch:

- scikit-learn's own `n_jobs=None` is left alone. It already means one worker,
  and setting it is not free — scikit-learn 1.9 deprecated
  `LogisticRegression.n_jobs` and warns for any value that is not None.
- RandomForest is NOT pinned. `RFWrapper` now carries `n_jobs` as a real
  constructor argument — it round-trips through `get_params`, `set_params` and
  `clone`, where it used to be a literal `-1` in the body that no route could
  reach — but the default is still every core, on the measured trade in
  `_inner_thread_overrides`'s docstring. The tests at the end of the RF block
  pin both halves: the parameter is reachable, and the forest is identical at
  any `n_jobs`, so choosing is a resource decision and never a modeling one.

**And the property that outranks every performance number here: no fitted model
moves.** Thread count is a resource axis, not a modeling one, and the last two
tests fit both ways with a fixed seed and demand bit-for-bit identical
predictions, probabilities and CV scores.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.base import clone                                    # noqa: E402
from sklearn.compose import TransformedTargetRegressor            # noqa: E402
from sklearn.ensemble import (                                    # noqa: E402
    ExtraTreesRegressor, HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.model_selection import KFold, cross_val_score        # noqa: E402
from sklearn.neighbors import KNeighborsRegressor                 # noqa: E402
from sklearn.preprocessing import StandardScaler                  # noqa: E402

from ml.eval import (                                             # noqa: E402
    _inner_thread_overrides, _pin_inner_threads, _worker_thread_pin,
    make_cv_pipeline, perform_cross_validation,
)


def _frame(n=400, p=6, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, p)
    y = X @ rng.randn(p) + rng.randn(n) * 0.25
    return X, y


def _lgbm(**kw):
    lightgbm = pytest.importorskip("lightgbm")
    return lightgbm.LGBMRegressor(n_estimators=20, random_state=42,
                                  verbosity=-1, **kw)


# ── the pin reaches the estimator, in every shape the app builds ─────────────

def test_a_third_party_pool_is_pinned_in_the_plain_composite():
    """The shape pages/06 builds for a model with no target transform."""
    composite = make_cv_pipeline(StandardScaler(), _lgbm())
    assert _inner_thread_overrides(composite) == {"est__n_jobs": 1}


def test_a_third_party_pool_is_pinned_through_the_target_transform():
    """The shape pages/06 builds when a log1p / Yeo-Johnson target transform is
    active. `est__n_jobs` does not exist here at all — a literal key would have
    no-opped for every transformed regression run, which is a common path."""
    wrapped = TransformedTargetRegressor(regressor=_lgbm(), func=np.log1p,
                                         inverse_func=np.expm1)
    composite = make_cv_pipeline(StandardScaler(), wrapped)
    assert "est__n_jobs" not in composite.get_params(deep=True)
    assert _inner_thread_overrides(composite) == {"est__regressor__n_jobs": 1}
    # And the clone actually carries it — a `TransformedTargetRegressor` holding
    # bare functions is the shape most likely to refuse `clone()`.
    pinned = _pin_inner_threads(composite)
    assert pinned.get_params(deep=True)["est__regressor__n_jobs"] == 1
    X, y = _frame(n=200, p=4)
    result = perform_cross_validation(composite, X, np.abs(y), cv_folds=3,
                                      task_type='regression')
    assert np.isfinite(result['mean'])


def test_a_bare_estimator_is_pinned_too():
    """tests/test_cv_strategies.py and scripts/smoke_check.py pass estimators
    with no Pipeline around them; the key has no prefix there."""
    assert _inner_thread_overrides(_lgbm()) == {"n_jobs": 1}


def test_the_pin_survives_the_clone_cross_val_score_does_per_fold():
    """`cross_val_score` clones the estimator once per fold. A pin that does
    not survive that reaches no worker — which is exactly why the RandomForest
    wrapper below cannot be pinned this way."""
    pinned = _pin_inner_threads(make_cv_pipeline(StandardScaler(), _lgbm()))
    assert clone(pinned).get_params(deep=True)["est__n_jobs"] == 1


def test_the_callers_estimator_is_not_pinned():
    """pages/06 keeps the un-cloned estimator for the single full-data fit that
    follows CV, and that fit is entitled to the whole machine."""
    original = _lgbm()
    composite = make_cv_pipeline(StandardScaler(), original)
    pinned = _pin_inner_threads(composite)
    assert pinned is not composite
    assert pinned.get_params(deep=True)["est__n_jobs"] == 1
    assert composite.get_params(deep=True)["est__n_jobs"] is None
    assert original.n_jobs is None


# ── and leaves alone everything joblib already handled ──────────────────────

@pytest.mark.parametrize("estimator", [
    LinearRegression(),                       # no n_jobs at all
    HistGradientBoostingRegressor(),          # no n_jobs at all — OMP is its only lever
    ExtraTreesRegressor(n_estimators=5),      # n_jobs=None, measured +1 thread in-worker
    KNeighborsRegressor(),                    # n_jobs=None
    LogisticRegression(),                     # n_jobs deprecated in scikit-learn 1.9
], ids=["linear", "histgb", "extratrees", "knn", "logreg"])
def test_a_scikit_learn_estimator_is_left_alone(estimator):
    assert _inner_thread_overrides(estimator) == {}


def test_pinning_logistic_regression_would_have_bought_a_deprecation_warning():
    """The reason the rule is 'third-party only' rather than 'every n_jobs'.

    scikit-learn 1.9's `LogisticRegression.fit` warns for any `n_jobs` that is
    not None, and requirements.txt pins `scikit-learn>=1.3.0` with no upper
    bound, so this fires on a fresh install today. A blanket pin would have
    added five FutureWarnings per CV run for a parameter that has had no effect
    since 1.8. The whole range that pin allows is in scope, so the deprecation
    is probed rather than assumed — below 1.8 there is nothing to avoid.
    """
    import sklearn

    X, y = _frame(n=120, p=4)
    y = (y > np.median(y)).astype(int)

    def _warnings_from(estimator):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(X, y)
        return [str(w.message) for w in caught if "n_jobs" in str(w.message)]

    if not _warnings_from(LogisticRegression(max_iter=200, n_jobs=1)):
        pytest.skip(f"scikit-learn {sklearn.__version__} has not deprecated "
                    "LogisticRegression.n_jobs; nothing for the pin to avoid")

    assert _warnings_from(LogisticRegression(max_iter=200)) == []
    # The pin declines to set it, so the clean case above is what CV runs.
    assert _inner_thread_overrides(LogisticRegression(max_iter=200)) == {}


def test_random_forest_reaches_cv_as_a_bare_forest_and_is_left_unpinned():
    """RandomForest is REACHABLE from the pin and deliberately not pinned.

    The object under test is the one pages/06 actually cross-validates, which
    is not the wrapper: :1729 is `_sklearn_clone(model.get_model())`, so a bare
    `RandomForestRegressor` carrying models/rf.py's `n_jobs=-1` is what goes
    into `make_cv_pipeline`. An earlier version of this test asserted against
    `RFWrapper` — an object no fold worker ever sees — and so would have stayed
    green whatever the pin did.

    What is pinned down here is the trade, because the reachability is real and
    someone will be tempted: `est__n_jobs` IS in the deep params, `set_params`
    DOES reach it, and it WOULD survive the per-fold clone. It is skipped by the
    scikit-learn rule in `_inner_thread_overrides`, on the measurement recorded
    in that docstring — pinning it ran ~18% slower for 3.6% of the memory,
    because five folds cannot saturate eight cores. Changing the rule should
    turn this test red and send the author back to that measurement.
    """
    from models.rf import RFWrapper
    from ml.eval import make_cv_pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    wrapper = RFWrapper(n_estimators=5, task_type='regression')
    cv_estimator = clone(wrapper.get_model())          # exactly pages/06:1729
    assert type(cv_estimator).__name__ == "RandomForestRegressor"
    assert cv_estimator.get_params()["n_jobs"] == -1

    composite = make_cv_pipeline(
        Pipeline([("scale", StandardScaler())]), cv_estimator)
    deep = composite.get_params(deep=True)
    assert deep["est__n_jobs"] == -1, "the key the pin would use is present"

    # Reachable in both directions the old comment called impossible ...
    reachable = clone(composite)
    reachable.set_params(est__n_jobs=1)
    assert reachable.get_params(deep=True)["est__n_jobs"] == 1
    assert clone(reachable).get_params(deep=True)["est__n_jobs"] == 1, (
        "set before cross_val_score, the value survives the per-fold clone")

    # ... and left alone anyway, on the measured trade.
    assert _inner_thread_overrides(composite) == {}
    assert _pin_inner_threads(composite) is composite


def test_the_random_forest_that_reaches_cv_is_not_the_wrapper():
    """The estimator the wrapper hands to CV is the bare forest, and the
    wrapper is judged by it.

    Kept as its own test because the original mistake was built on the
    wrapper's parameter surface. That surface now exposes `n_jobs`, so the pin
    rule looks through `get_model()` and reaches the same answer for the
    wrapper as for the forest it holds: scikit-learn, left alone.
    """
    from models.rf import RFWrapper
    wrapper = RFWrapper(n_estimators=5, task_type='regression')
    assert wrapper.get_params()["n_jobs"] == -1
    assert _inner_thread_overrides(wrapper) == {},         "a wrapper around a scikit-learn forest is judged as scikit-learn"
    assert wrapper.model.n_jobs == -1
    assert wrapper.get_model() is wrapper.model
    assert clone(wrapper.get_model()).get_params()["n_jobs"] == -1


# ── the remainder of F-08: the forest that could not be pinned ───────────────
#
# `n_jobs=-1` was a literal inside `RFWrapper.__init__`. `get_params()` is the
# `__init__` signature, so `clone()` rebuilt every copy at -1 and
# `set_params(n_jobs=1)` on the wrapper set an attribute the forest never read.
# The pin above is not taken for RandomForest on measurement, and that stays;
# what changes is that the choice is now expressible at all.

def _rf_frame(n=300, p=5, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, p)
    y = X @ rng.randn(p) + rng.randn(n) * 0.1
    return X, y


def test_the_default_forest_is_the_one_main_built():
    """No result moves: the default constructor builds exactly the forest the
    literal used to — every parameter, including the seed."""
    from models.rf import RFWrapper
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    for task, cls in (("regression", RandomForestRegressor),
                      ("classification", RandomForestClassifier)):
        built = RFWrapper(task_type=task).model
        as_before = cls(n_estimators=500, max_depth=None, min_samples_leaf=10,
                        n_jobs=-1, random_state=42)
        assert type(built) is cls
        assert built.get_params() == as_before.get_params()


def test_n_jobs_round_trips_through_get_params_set_params_and_clone():
    from models.rf import RFWrapper

    wrapper = RFWrapper(n_estimators=5, n_jobs=1)
    assert wrapper.get_params()["n_jobs"] == 1
    assert wrapper.model.n_jobs == 1, "the constructor argument reaches the forest"

    copied = clone(wrapper)
    assert copied.get_params()["n_jobs"] == 1
    assert copied.model.n_jobs == 1, "a clone rebuilds the forest with the same value"

    wrapper.set_params(n_jobs=2, n_estimators=7)
    assert wrapper.model.n_jobs == 2 and wrapper.model.n_estimators == 7,         "set_params reaches the forest, not just the wrapper's attribute"

    wrapper.set_params(task_type="classification")
    assert type(wrapper.model).__name__ == "RandomForestClassifier"
    assert wrapper.model.n_jobs == 2 and wrapper.model.n_estimators == 7,         "a rebuilt forest keeps every other parameter"


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_the_forest_is_identical_at_one_thread_and_at_every_core(task):
    """The property that makes `n_jobs` a resource axis and not a modeling one.

    Thread count changes the order trees are BUILT in, not how any tree is
    built: each tree draws its bootstrap and its feature subsets from a seed
    derived from `random_state` and its own index. So the split arrays of every
    tree compare equal, and predictions differ by at most the float
    accumulation order of the ensemble average — ml/eval.py measured 1.8e-15,
    the same magnitude two runs of the unchanged code differ by.
    """
    from models.rf import RFWrapper

    X, y = _rf_frame()
    if task == "classification":
        y = (y > np.median(y)).astype(int)

    one = RFWrapper(n_estimators=12, min_samples_leaf=3, task_type=task, n_jobs=1)
    every = RFWrapper(n_estimators=12, min_samples_leaf=3, task_type=task, n_jobs=-1)
    one.fit(X, y)
    every.fit(X, y)

    for t1, t2 in zip(one.model.estimators_, every.model.estimators_):
        assert np.array_equal(t1.tree_.feature, t2.tree_.feature)
        assert np.array_equal(t1.tree_.threshold, t2.tree_.threshold)
        assert np.array_equal(t1.tree_.value, t2.tree_.value)

    np.testing.assert_allclose(one.predict(X), every.predict(X), rtol=0, atol=1e-12)
    if task == "classification":
        np.testing.assert_allclose(one.predict_proba(X), every.predict_proba(X),
                                   rtol=0, atol=1e-12)


# ── the environment pin, including the case joblib's own guard misses ────────

def _worker_env(_):
    return os.getpid(), os.environ.get("OMP_NUM_THREADS")


def test_a_fold_worker_gets_one_thread_even_when_the_parent_set_otherwise(monkeypatch):
    """joblib's guard is `os.environ.get(var, cpu_count // n_jobs)` — it only
    computes 1 when the variable is ABSENT. A launcher line or a user shell
    that exports `OMP_NUM_THREADS=8` hands 8 to every worker instead, turning
    five folds times one thread into five times eight (measured on
    HistGradientBoosting as CV wall 10.9 s -> 29.5 s). `inner_max_num_threads`
    makes joblib set the value rather than default it.

    Note that .github/workflows/ci.yml:60 runs the suite under
    `OMP_NUM_THREADS=1` for its own reasons, so this sets the variable itself
    rather than reading whatever the harness left.
    """
    from joblib import Parallel, delayed
    monkeypatch.setenv("OMP_NUM_THREADS", "8")

    leaked = Parallel(n_jobs=2)(delayed(_worker_env)(i) for i in range(2))
    if any(pid == os.getpid() for pid, _ in leaked):
        pytest.skip("joblib fell back to a sequential backend; no worker env to pin")
    assert [v for _, v in leaked] == ["8", "8"], \
        "the latent bug should still reproduce when unguarded"

    with _worker_thread_pin():
        pinned = Parallel(n_jobs=2)(delayed(_worker_env)(i) for i in range(2))
    assert [v for _, v in pinned] == ["1", "1"]


# ── the property that outranks all of the above: no result moves ─────────────

@pytest.mark.parametrize("key", ["lgbm_reg", "lgbm_clf", "xgb_reg", "xgb_clf"])
def test_pinning_threads_does_not_move_a_single_prediction(key):
    """Fit as shipped and again pinned, same seed, and demand bit-for-bit
    equality — not closeness. Thread count is a resource axis; if any of these
    estimators disagreed with itself across it, the pin could not ship."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("xgboost")
    from ml.model_registry import get_registry

    classification = key.endswith("_clf")
    X, y = _frame(n=500, p=8)
    if classification:
        y = (y > np.median(y)).astype(int)

    spec = get_registry()[key]
    shipped = spec.factory('classification' if classification else 'regression', 42)
    pinned = _pin_inner_threads(shipped)
    assert _inner_thread_overrides(shipped), f"{key} exposes no thread knob to pin"

    shipped.fit(X, y)
    pinned.fit(X, y)
    np.testing.assert_array_equal(shipped.predict(X), pinned.predict(X))
    if hasattr(shipped, "predict_proba"):
        np.testing.assert_array_equal(shipped.predict_proba(X),
                                      pinned.predict_proba(X))


def test_cross_validation_returns_the_scores_it_returned_before_the_pin():
    """End to end through `perform_cross_validation` on the composite pages/06
    builds, against the reference the function computed on main: the same folds,
    the same estimator, no pin, run serially. Equality is exact."""
    pytest.importorskip("lightgbm")
    X, y = _frame(n=400, p=6)
    composite = make_cv_pipeline(StandardScaler(), _lgbm())

    reference = cross_val_score(
        clone(composite), X, y,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='neg_mean_squared_error', n_jobs=1,
    )
    result = perform_cross_validation(composite, X, y, cv_folds=5,
                                      task_type='regression')

    np.testing.assert_array_equal(result['scores'], -reference)
    assert result['mean'] == float(np.mean(-reference))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
