"""`MODELS-001` / `T0-LIVE-003` — the guard, made runnable in this environment.

**The defect.** `SklearnCompatibleNNRegressor.fit()` and its classifier twin did
not train. They set `is_fitted_ = True`, recorded `n_features_in_`, and returned
`self`. Any sklearn utility that clones and refits — `cross_val_score`,
`GridSearchCV`, `RFECV`, `CalibratedClassifierCV`, `ml/sensitivity.py` — then
scored a network that had already seen every fold's held-out rows. No exception,
no warning: uniformly excellent cross-validation numbers that get written into a
manuscript.

**The fix.** `fit()` calls `_refuse_if_not_pretrained()`, and the pre-trained
mark is deliberately *not* a constructor parameter. `sklearn.base.clone` rebuilds
an estimator from `get_params()`, so anything reachable through `__init__`
survives cloning — and surviving the clone is exactly what must not happen. Set
after construction, the mark is dropped by `clone()`, and the clone's `fit()`
raises where it used to answer from the original network.

**Why this file exists rather than the row simply being closed.** Both rows named
`tests/test_characterization_wrappers.py::test_clone_and_refit_now_raises_instead_of_answering_from_the_old_model`,
and that test opens with `pytest.importorskip("torch")`. `torch` is deliberately
not installed here — 1.1 GB inside a pinned `pandas<3` envelope — so the guard
reports `SKIPPED`, which in a `-q` run is one character. The fix may well be
correct; nothing in this environment demonstrated it, and the ledger's claim is
that a `FIXED` row has a test that fails on revert. **A test that never runs
fails on nothing.**

That is `FEATURE_PARITY.md`'s fourth silence: principle-locality fails across
*space*, an expiring guarantee across *time*, an untriggered check across
*occasion*, and this one across **environment**.

**How it runs torch-free, and why that is not a weaker test.** The ledger's own
note proposed "a stub estimator with the same `get_params`/`set_params` shape".
That would guard nothing — reverting the fix in `models/nn_whuber.py` would leave
a test of a stub perfectly green, which is the revert probe's whole objection to
topical proximity. So this file loads **the real module**, with a minimal
stand-in for `torch` installed only for the duration of the import. `nn_whuber`
needs `torch` at import time for exactly two things: `torch.Tensor` in two
function annotations, and `nn.Module` as `SimpleMLP`'s base class. Neither is
touched by the adapter under test, which never reaches the network. The class
exercised here is the one the app ships.

Probed: reverting `fit()`'s `self._refuse_if_not_pretrained()` to a bare
`pass` turns this red with `DID NOT RAISE`.
"""
from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


class _StubNN:
    """Stands in for a trained `NNWeightedHuberWrapper`.

    Records whether anything ever asked it to train, and answers `predict` with a
    constant identifying which instance produced it — so a prediction can be
    traced back to the model that made it.
    """

    def __init__(self, answer):
        self.answer = answer
        self.fit_calls = 0

    def fit(self, *a, **k):
        self.fit_calls += 1
        return self

    def predict(self, X):
        return np.full(len(X), self.answer, dtype=float)

    def predict_proba(self, X):
        p = np.full(len(X), self.answer, dtype=float)
        return np.column_stack([1.0 - p, p])


@contextlib.contextmanager
def _torch_stand_in():
    """Install the smallest `torch` that lets `nn_whuber` finish importing.

    A no-op when the real `torch` is present, so this file tests the same class
    either way rather than diverging by environment — which would reintroduce
    the problem it exists to solve.
    """
    if "torch" in sys.modules:
        yield
        return
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    nn = types.ModuleType("torch.nn")

    class _Module:                                    # SimpleMLP's base class
        def __init__(self, *a, **k):
            pass

    nn.Module = _Module
    torch.nn = nn
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    try:
        yield
    finally:
        sys.modules.pop("torch", None)
        sys.modules.pop("torch.nn", None)


def _load_nn_whuber():
    """The real `models/nn_whuber.py`, under a throwaway module name.

    A throwaway name rather than `models.nn_whuber` so that nothing else in the
    session inherits a module built against the stand-in.
    """
    name = "models._nn_whuber_under_test"
    path = REPO / "models" / "nn_whuber.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module                        # so dataclasses/typing resolve
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture(scope="module")
def nn_whuber():
    with _torch_stand_in():
        module = _load_nn_whuber()
    yield module
    sys.modules.pop("models._nn_whuber_under_test", None)


@pytest.fixture(scope="module")
def data():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(160, 5))
    y = X @ np.array([2.0, -1.0, 0.5, 0.0, 1.5]) + rng.normal(scale=0.3, size=160)
    return X, y


def _adapters(module):
    return [
        ("regressor", module.SklearnCompatibleNNRegressor),
        ("classifier", module.SklearnCompatibleNNClassifier),
    ]


@pytest.mark.parametrize("which", ["regressor", "classifier"])
def test_the_pretrained_mark_does_not_survive_a_clone(nn_whuber, data, which):
    """The path that made `MODELS-001` dangerous, asserted here without torch.

    Assertion order is load-bearing: the probe reads the *first* assertion to
    fire, so the most diagnostic one is first. If the mark ever becomes a
    constructor parameter, that is the sentence a reader needs, not a downstream
    `DID NOT RAISE`.
    """
    from sklearn.base import clone
    from sklearn.exceptions import NotFittedError

    cls = dict(_adapters(nn_whuber))[which]
    X, y = data
    if which == "classifier":
        y = (y > 0).astype(int)

    stub = _StubNN(answer=1.0)
    original = cls(wrapper_instance=stub).mark_pretrained()

    # 1 · the mark is not reachable through the constructor, which is the whole
    #     mechanism — `clone` rebuilds from `get_params()`.
    assert "_trained_externally" not in original.get_params(), (
        "the pre-trained mark is a constructor parameter again; clone() will "
        "carry it and the unsafe refit goes silent")

    # 2 · the legitimate path still works — a marked adapter over an already
    #     trained wrapper is what `get_sklearn_estimator()` produces.
    original.fit(X[:120], y[:120])
    assert original.is_fitted_ is True
    assert stub.fit_calls == 0, "the adapter trained something; it must not"

    # 3 · clone drops the mark.
    fresh = clone(original)
    assert not getattr(fresh, "_trained_externally", False), (
        "the pre-trained mark survived clone()")

    # 4 · and the clone refuses to refit rather than answering from the original
    #     network. This is the defect the two rows name.
    with pytest.raises(NotFittedError, match="does not train"):
        fresh.fit(X[120:] * 100.0, y[120:] * -1.0)
    assert fresh.is_fitted_ is False
    assert stub.fit_calls == 0


@pytest.mark.parametrize("which", ["regressor", "classifier"])
def test_an_unmarked_adapter_refuses_to_fit(nn_whuber, data, which):
    """The construction path, not just the clone path.

    `cross_val_score` reaches the defect through `clone`; a caller who builds the
    adapter directly reaches it through `__init__`. Both must refuse, and only
    `get_sklearn_estimator()` — the one legitimate construction site — marks it.
    """
    from sklearn.exceptions import NotFittedError

    cls = dict(_adapters(nn_whuber))[which]
    X, y = data
    if which == "classifier":
        y = (y > 0).astype(int)

    stub = _StubNN(answer=1.0)
    est = cls(wrapper_instance=stub)
    with pytest.raises(NotFittedError, match="does not train"):
        est.fit(X[:120], y[:120])
    assert est.is_fitted_ is False
    assert stub.fit_calls == 0
