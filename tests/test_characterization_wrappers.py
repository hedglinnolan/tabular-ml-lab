"""Characterization tests for every model wrapper — against real wrapper objects.

`TRANSITION_PLAN.md` §03 names this gap precisely: *`tests/integration/conftest.py`
injects a bare sklearn `Ridge` into `trained_models`; the app stores wrapper
objects.* Downstream suites therefore pass against a state shape the app never
produces, and §03 adds the consequence — *this is how a `clone()` breakage
stayed invisible.*

`models/` also has **no coverage at all** today. It is the layer that must move
intact and it is the least protected.

So this file instantiates the real wrappers, fits them on real arrays, and pins
the `BaseModelWrapper` contract that `pages/06` and everything downstream rely
on. No bare estimators appear anywhere in it.

Run:  turbotab/.venv/Scripts/python -m pytest tests/test_characterization_wrappers.py -v
"""
import numpy as np
import pytest

pytestmark = pytest.mark.timeout(600)


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(160, 5))
    y = X @ np.array([2.0, -1.0, 0.5, 0.0, 1.5]) + rng.normal(scale=0.3, size=160)
    return X, y


@pytest.fixture(scope="module")
def classification_data():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(160, 5))
    y = (X[:, 0] + rng.normal(scale=0.4, size=160) > 0).astype(int)
    return X, y


def _regression_wrappers():
    from models.glm import GLMWrapper
    from models.huber_glm import HuberGLMWrapper
    from models.rf import RFWrapper
    from models.registry_wrappers import RegistryModelWrapper
    from sklearn.linear_model import Ridge

    out = [
        ("GLMWrapper", lambda: GLMWrapper(task_type="regression")),
        ("HuberGLMWrapper", lambda: HuberGLMWrapper()),
        ("RFWrapper", lambda: RFWrapper(n_estimators=20, task_type="regression")
         if "task_type" in RFWrapper.__init__.__code__.co_varnames
         else RFWrapper(n_estimators=20)),
        # The registry wrapper is the shape the app actually stores for most
        # models: a wrapper *around* an estimator, never the estimator itself.
        ("RegistryModelWrapper",
         lambda: RegistryModelWrapper(Ridge(alpha=1.0), name="ridge")),
    ]
    return out


@pytest.mark.parametrize("name,make", _regression_wrappers(), ids=lambda v: v if isinstance(v, str) else "")
def test_regression_wrapper_honors_the_base_contract(name, make, regression_data):
    """Every wrapper must satisfy the interface `pages/06` calls through.

    This is the shape the app stores. A test that injects a bare `Ridge`
    exercises none of it — `Ridge` has no `get_model`, no `supports_proba`, and
    a different `fit` signature.
    """
    from models.base import BaseModelWrapper

    X, y = regression_data
    w = make()
    assert isinstance(w, BaseModelWrapper), f"{name} is not a wrapper"
    assert hasattr(w, "name") and w.name, f"{name} has no name"

    fitted = w.fit(X[:120], y[:120], X[120:], y[120:])
    assert fitted is None or fitted is w or isinstance(fitted, dict), (
        f"{name}.fit returned {type(fitted)} — callers treat it as in-place")

    preds = w.predict(X[120:])
    assert preds.shape[0] == 40, f"{name} predicted the wrong number of rows"
    assert np.isfinite(preds).all(), f"{name} produced non-finite predictions"

    # `get_model()` is how the app reaches the estimator for SHAP and for
    # sensitivity analysis. It must return something, and it must not be the
    # wrapper itself.
    inner = w.get_model()
    assert inner is not None, f"{name}.get_model() returned None after fit"
    assert not isinstance(inner, BaseModelWrapper), (
        f"{name}.get_model() returned a wrapper, not an estimator")

    assert w.supports_proba() is False, f"{name} claims proba on a regression task"


def test_registry_wrapper_predicts_like_the_estimator_it_wraps(regression_data):
    """The wrapper adds an interface, not arithmetic."""
    from models.registry_wrappers import RegistryModelWrapper
    from sklearn.linear_model import Ridge

    X, y = regression_data
    bare = Ridge(alpha=1.0).fit(X[:120], y[:120])
    wrapped = RegistryModelWrapper(Ridge(alpha=1.0), name="ridge")
    wrapped.fit(X[:120], y[:120], X[120:], y[120:])

    np.testing.assert_allclose(wrapped.predict(X[120:]), bare.predict(X[120:]), rtol=1e-9)


def test_classification_wrapper_supports_proba(classification_data):
    from models.glm import GLMWrapper

    X, y = classification_data
    w = GLMWrapper(task_type="classification")
    w.fit(X[:120], y[:120], X[120:], y[120:])

    assert w.supports_proba() is True
    proba = w.predict_proba(X[120:])
    assert proba is not None and proba.shape == (40, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_an_unfitted_wrapper_does_not_answer_predict(regression_data):
    """A wrapper that predicts before it is fitted returns numbers with no
    model behind them — the failure `T0-LIVE-003` describes in the NN adapter,
    checked here for the ordinary wrappers."""
    from models.glm import GLMWrapper

    X, _ = regression_data
    w = GLMWrapper(task_type="regression")
    with pytest.raises(Exception):
        w.predict(X[:5])


# ── the neural-net sklearn adapter · T0-LIVE-003 ─────────────────────────

class _StubNN:
    """Stands in for a trained `NNWeightedHuberWrapper`.

    Records whether anything ever asked it to train, and answers `predict` with
    a constant that identifies which instance produced it — so a prediction can
    be traced back to the model that made it.
    """

    def __init__(self, answer):
        self.answer = answer
        self.fit_calls = 0

    def fit(self, *a, **k):
        self.fit_calls += 1
        return self

    def predict(self, X):
        return np.full(len(X), self.answer, dtype=float)


def test_nn_adapter_refuses_to_fit_without_a_model(regression_data):
    """The half of the adapter that is right: with no wrapper behind it, `fit`
    raises rather than marking itself fitted."""
    pytest.importorskip("torch")
    from models.nn_whuber import SklearnCompatibleNNRegressor

    X, y = regression_data
    est = SklearnCompatibleNNRegressor(wrapper_instance=None)
    with pytest.raises(ValueError):
        est.fit(X[:120], y[:120])
    assert est.is_fitted_ is False


def test_nn_sklearn_adapter_fit_does_not_train(regression_data):
    """Pins the *current* behavior, which is a live bug (`T0-LIVE-003`).

    With a wrapper attached, `fit()` sets `is_fitted_`, records
    `n_features_in_`, and returns — it never asks the wrapper to train. Real
    training is expected to have happened already in `wrapper_instance.fit()`.

    That is a legitimate adapter only while every caller knows the protocol.
    This asserts the protocol as it stands, so the extraction cannot change it
    by accident. When `T0-LIVE-003` is fixed — `fit()` raising unless an
    explicit already-trained flag is set — **invert this test rather than
    deleting it**; its failure is the signal the fix landed.
    """
    pytest.importorskip("torch")
    from models.nn_whuber import SklearnCompatibleNNRegressor

    X, y = regression_data
    stub = _StubNN(answer=7.0)
    est = SklearnCompatibleNNRegressor(wrapper_instance=stub)

    est.fit(X[:120], y[:120])

    assert est.is_fitted_ is True
    assert est.n_features_in_ == 5
    assert stub.fit_calls == 0, (
        "fit() trained the wrapper — T0-LIVE-003 may be fixed; if so, invert "
        "this test rather than deleting it")


def test_clone_and_refit_silently_answers_from_the_old_model(regression_data):
    """The path that makes `T0-LIVE-003` dangerous, demonstrated end to end.

    `get_params()` returns the wrapper instance, so `clone()` carries a copy of
    the *already-trained* model. Refitting the clone on different data marks it
    fitted without retraining, and `predict()` then answers from the original
    model — no exception, just wrong numbers. `ml/sensitivity.py` already calls
    `clone()` on wrapper objects.

    Pinned, not fixed: `models/` is frozen behavior until `T0-LIVE-003` is
    worked. This test is the tripwire that says what the fix has to change.
    """
    pytest.importorskip("torch")
    from sklearn.base import clone
    from models.nn_whuber import SklearnCompatibleNNRegressor

    X, y = regression_data
    original = SklearnCompatibleNNRegressor(wrapper_instance=_StubNN(answer=7.0))
    original.fit(X[:120], y[:120])

    fresh = clone(original)
    # Different data entirely — a refit that meant anything would change the answer.
    fresh.fit(X[120:] * 100.0, y[120:] * -1.0)

    preds = fresh.predict(X[120:])
    assert np.allclose(preds, 7.0), (
        "the clone stopped answering from the original model — T0-LIVE-003 may "
        "be fixed; invert this test rather than deleting it")
    assert fresh.is_fitted_ is True, (
        "a refitted clone reports itself fitted while holding the old model: "
        "this is the silent-wrong-numbers path T0-LIVE-003 describes")


def test_nn_wrapper_is_a_real_wrapper(regression_data):
    """`NNWeightedHuberWrapper` is what the app stores; the sklearn adapter is
    only what it hands to sklearn."""
    pytest.importorskip("torch")
    from models.base import BaseModelWrapper
    from models.nn_whuber import NNWeightedHuberWrapper

    w = NNWeightedHuberWrapper(task_type="regression") \
        if "task_type" in NNWeightedHuberWrapper.__init__.__code__.co_varnames \
        else NNWeightedHuberWrapper()
    assert isinstance(w, BaseModelWrapper)
    assert callable(w.fit) and callable(w.predict) and callable(w.get_model)


# ── the shape the app really stores ──────────────────────────────────────

def test_trained_models_shape_is_wrappers_not_estimators(regression_data):
    """The specific fiction `TRANSITION_PLAN.md` §03 calls out.

    `tests/integration/conftest.py::inject_trained_state` puts a bare `Ridge`
    into `trained_models`. The app puts wrappers there. Anything reading
    `trained_models` and calling `.get_model()` works against the app and
    crashes against the fixture — or, worse, works against both while testing
    neither.
    """
    from models.base import BaseModelWrapper
    from models.registry_wrappers import RegistryModelWrapper
    from sklearn.linear_model import Ridge

    X, y = regression_data
    realistic = {"ridge": RegistryModelWrapper(Ridge(alpha=1.0), name="ridge")}
    realistic["ridge"].fit(X[:120], y[:120], X[120:], y[120:])

    for key, model in realistic.items():
        assert isinstance(model, BaseModelWrapper), (
            f"trained_models[{key!r}] must hold a wrapper")
        assert hasattr(model, "get_model"), "a bare estimator has no get_model()"

    bare = Ridge(alpha=1.0).fit(X[:120], y[:120])
    assert not hasattr(bare, "get_model"), (
        "a bare Ridge grew get_model() — the fixture's shape and the app's have "
        "converged and this test is no longer describing a real gap")
