"""RECIPES F12 (§1, §4.3): an imbalance correction around a model that stops early.

``ImbalanceCorrected`` has the early-stopping interface (``early_stopping``,
``validation_fraction``, ``fit(X, y, X_val, y_val)``), so ``inner_cv.fit_pipeline`` routes it
through F11's stopping-rows path: the stopping units are drawn first, the steps before the model
are fit on the other rows, only those rows are resampled, and the stopping rows reach every model
the wrapper fits (the deployed one and the recalibration's) untouched. The probe is F11's: a step
that reads the outcome records the rows it is fit on, and the wrapped model records the rows it
trains and stops on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from turbotab.core.methods.levers import ImbalanceCorrected, wrap_model
from turbotab.core.models.inner_cv import fit_pipeline, row_keys, validation_rows
from turbotab.core.tests.test_stopping_rows import TopByCorrelation

N_ROWS = 10_500  # above scikit-learn's 10,000-row threshold for early_stopping="auto"


def _table(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"x{i}" for i in range(6)])
    X.index = pd.Index(np.arange(n) + 70_000, name="row_id")  # row ids are not positions
    y = (X["x0"].to_numpy() + rng.normal(size=n) > 1.6).astype(int)  # about 1 in 8: yes
    return X, y


def _spied(monkeypatch) -> list[dict]:
    """Each fit of the wrapped model: the rows it trains on (resampled copies included), the rows
    it stops on, and whether it stops early."""
    seen: list[dict] = []
    original = HistGradientBoostingClassifier.fit

    def fit(self, X, y, *args, **kwargs):
        val = kwargs.get("X_val")
        seen.append({"train": list(X.index), "val": None if val is None else list(val.index),
                     "stops": self.early_stopping})
        return original(self, X, y, *args, **kwargs)

    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", fit)
    return seen


def _booster() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=15, random_state=0)  # early_stopping="auto"


@pytest.mark.parametrize("units", ["rows", "people"])
def test_no_step_and_no_resampled_copy_touches_the_stopping_units(monkeypatch, units):
    """An outcome-reading step and the oversampler at 10,500 rows: the stopping units are drawn
    first (whole units, stratified, as ``validation_rows`` draws them independently of the
    pipeline), the step is fit on the other rows, the oversampler draws its copies from those rows
    only, and every fit of the wrapped model stops on the stopping rows as they are."""
    seen = _spied(monkeypatch)
    X, y = _table(N_ROWS, seed=12)
    groups = (np.arange(N_ROWS) // 3) if units == "people" else None
    model = wrap_model(_booster(), {"imbalance": "oversample"}, "binary")
    pipeline = Pipeline([("select", TopByCorrelation(k=4)), ("model", model)]
                        ).set_output(transform="pandas")
    fitted = fit_pipeline(pipeline, X, y, groups=groups)

    keys = row_keys(X, y) if groups is None else None
    held = validation_rows(0.1, groups=groups, keys=keys, y=y, seed=0)
    stopping = set(X.index[held])
    assert 0 < len(stopping) < N_ROWS
    assert abs(y[held].mean() - y.mean()) < 0.01  # stratified by class

    step = fitted.named_steps["select"]
    assert not set(step.fit_rows_) & stopping, "the outcome-reading step saw the stopping rows"
    assert set(step.fit_rows_) == set(X.index[~held])

    assert len(seen) == 1 + 5  # the deployed fit and the recalibration's five inner fits
    for fit in seen:
        assert fit["val"] is not None, "the wrapped model drew its own stopping split"
        assert fit["stops"] is True
        assert set(fit["val"]) == stopping  # untouched: every stopping row, once each
        assert len(fit["val"]) == len(stopping)
        assert not set(fit["train"]) & stopping, "a resampled copy is a stopping row"
    deployed = seen[0]["train"]
    assert len(deployed) > len(step.fit_rows_)  # the oversampler drew copies
    assert set(deployed) == set(step.fit_rows_)

    # the recalibration's inner splits are drawn over the rows the wrapper's fit receives
    cv = fitted.named_steps["model"].cv
    positions = np.concatenate([np.concatenate([a, b]) for a, b in cv])
    assert positions.min() == 0 and positions.max() == len(step.fit_rows_) - 1


def test_the_wrapper_alone_draws_its_stopping_units_before_resampling(monkeypatch):
    """Fit outside ``fit_pipeline`` (no ``X_val``), the wrapper draws the stopping units itself,
    before the draw, and its recalibration folds cover only the other rows."""
    seen = _spied(monkeypatch)
    X, y = _table(N_ROWS, seed=4)
    fitted = ImbalanceCorrected(_booster(), "undersample", early_stopping="auto",
                                validation_fraction=0.1, seed=3).fit(X, y)
    held = validation_rows(0.1, keys=row_keys(X, y), y=y, seed=3)
    stopping = set(X.index[held])
    for fit in seen:
        assert set(fit["val"]) == stopping
        assert not set(fit["train"]) & stopping
    assert fitted.n_iter_ >= 1


def test_the_threshold_reads_the_rows_received_not_the_resampled_rows(monkeypatch):
    """6,000 rows oversampled past 10,000: the wrapped model does not stop early on a split of
    the copies (``early_stopping="auto"`` is read on the rows the wrapper receives)."""
    seen = _spied(monkeypatch)
    X, y = _table(6_000, seed=8)
    model = wrap_model(_booster(), {"imbalance": "oversample"}, "binary")
    fit_pipeline(Pipeline([("model", model)]), X, y)
    assert len(seen[0]["train"]) > 10_000
    assert all(fit["stops"] is False and fit["val"] is None for fit in seen)
